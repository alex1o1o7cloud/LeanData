import Mathlib

namespace NUMINAMATH_CALUDE_power_difference_l1115_111594

theorem power_difference (m n : ℕ) (h1 : 3^m = 8) (h2 : 3^n = 2) : 3^(m-n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l1115_111594


namespace NUMINAMATH_CALUDE_ducks_in_marsh_l1115_111519

/-- The number of ducks in a marsh, given the total number of birds and the number of geese. -/
def num_ducks (total_birds geese : ℕ) : ℕ := total_birds - geese

/-- Theorem stating that there are 37 ducks in the marsh. -/
theorem ducks_in_marsh : num_ducks 95 58 = 37 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_marsh_l1115_111519


namespace NUMINAMATH_CALUDE_g_sum_property_l1115_111556

/-- Given a function g(x) = ax^6 + bx^4 - cx^3 - cx^2 + 3, 
    if g(2) = 5, then g(2) + g(-2) = 10 -/
theorem g_sum_property (a b c : ℝ) : 
  let g := λ x : ℝ => a * x^6 + b * x^4 - c * x^3 - c * x^2 + 3
  (g 2 = 5) → (g 2 + g (-2) = 10) := by
  sorry

end NUMINAMATH_CALUDE_g_sum_property_l1115_111556


namespace NUMINAMATH_CALUDE_rosa_flower_count_l1115_111555

/-- Rosa's initial number of flowers -/
def initial_flowers : ℝ := 67.0

/-- Number of flowers Andre gave to Rosa -/
def additional_flowers : ℝ := 90.0

/-- Rosa's total number of flowers -/
def total_flowers : ℝ := initial_flowers + additional_flowers

theorem rosa_flower_count : total_flowers = 157.0 := by
  sorry

end NUMINAMATH_CALUDE_rosa_flower_count_l1115_111555


namespace NUMINAMATH_CALUDE_complex_inequality_l1115_111502

theorem complex_inequality (z₁ z₂ z₃ z₄ : ℂ) :
  Complex.abs (z₁ - z₃)^2 + Complex.abs (z₂ - z₄)^2 ≤
  Complex.abs (z₁ - z₂)^2 + Complex.abs (z₂ - z₃)^2 +
  Complex.abs (z₃ - z₄)^2 + Complex.abs (z₄ - z₁)^2 ∧
  (Complex.abs (z₁ - z₃)^2 + Complex.abs (z₂ - z₄)^2 =
   Complex.abs (z₁ - z₂)^2 + Complex.abs (z₂ - z₃)^2 +
   Complex.abs (z₃ - z₄)^2 + Complex.abs (z₄ - z₁)^2 ↔
   z₁ + z₃ = z₂ + z₄) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l1115_111502


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1115_111557

theorem min_value_of_function (x : ℝ) (h : x > 0) : (x^2 + 1) / x ≥ 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) : (x^2 + 1) / x = 2 ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1115_111557


namespace NUMINAMATH_CALUDE_video_streaming_cost_theorem_l1115_111531

/-- The total cost per person for a video streaming service after one year -/
def video_streaming_cost (subscription_cost : ℚ) (num_people : ℕ) (connection_fee : ℚ) (tax_rate : ℚ) : ℚ :=
  let monthly_subscription_per_person := subscription_cost / num_people
  let monthly_cost_before_tax := monthly_subscription_per_person + connection_fee
  let monthly_tax := monthly_cost_before_tax * tax_rate
  let monthly_total := monthly_cost_before_tax + monthly_tax
  12 * monthly_total

/-- Theorem stating the total cost per person for a specific video streaming service after one year -/
theorem video_streaming_cost_theorem :
  video_streaming_cost 14 4 2 (1/10) = 726/10 := by
  sorry

#eval video_streaming_cost 14 4 2 (1/10)

end NUMINAMATH_CALUDE_video_streaming_cost_theorem_l1115_111531


namespace NUMINAMATH_CALUDE_greatest_x_value_l1115_111579

theorem greatest_x_value (x : ℤ) (h : 2.134 * (10 : ℝ) ^ (x : ℝ) < 240000) :
  x ≤ 5 ∧ ∃ y : ℤ, y > 5 → 2.134 * (10 : ℝ) ^ (y : ℝ) ≥ 240000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1115_111579


namespace NUMINAMATH_CALUDE_last_two_digits_product_l1115_111521

def last_two_digits (n : ℤ) : ℤ × ℤ :=
  let tens := (n / 10) % 10
  let ones := n % 10
  (tens, ones)

theorem last_two_digits_product (n : ℤ) : 
  n % 4 = 0 → 
  (let (a, b) := last_two_digits n; a + b = 12) → 
  (let (a, b) := last_two_digits n; a * b = 32 ∨ a * b = 36) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l1115_111521


namespace NUMINAMATH_CALUDE_base8_52_equals_base10_42_l1115_111549

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ :=
  let ones := n % 10
  let eights := n / 10
  eights * 8 + ones

/-- The base-8 number 52 is equal to the base-10 number 42 --/
theorem base8_52_equals_base10_42 : base8ToBase10 52 = 42 := by
  sorry

end NUMINAMATH_CALUDE_base8_52_equals_base10_42_l1115_111549


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1115_111546

/-- Given positive integers A, B, C, D satisfying the conditions:
    1. A, B, C form an arithmetic sequence
    2. B, C, D form a geometric sequence
    3. C/B = 4/3
    The smallest possible value of A + B + C + D is 43. -/
theorem smallest_sum_of_sequence (A B C D : ℕ+) : 
  (∃ r : ℚ, C = A + r ∧ B = A + 2*r) →  -- A, B, C form an arithmetic sequence
  (∃ q : ℚ, C = B * q ∧ D = C * q) →   -- B, C, D form a geometric sequence
  (C : ℚ) / B = 4 / 3 →                -- The ratio of the geometric sequence
  A + B + C + D ≥ 43 ∧ 
  (∃ A' B' C' D' : ℕ+, A' + B' + C' + D' = 43 ∧ 
    (∃ r : ℚ, C' = A' + r ∧ B' = A' + 2*r) ∧
    (∃ q : ℚ, C' = B' * q ∧ D' = C' * q) ∧
    (C' : ℚ) / B' = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1115_111546


namespace NUMINAMATH_CALUDE_total_bugs_is_63_l1115_111520

/-- The number of bugs eaten by the gecko, lizard, frog, and toad -/
def total_bugs_eaten (gecko_bugs : ℕ) : ℕ :=
  let lizard_bugs := gecko_bugs / 2
  let frog_bugs := lizard_bugs * 3
  let toad_bugs := frog_bugs + frog_bugs / 2
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs

/-- Theorem stating the total number of bugs eaten is 63 -/
theorem total_bugs_is_63 : total_bugs_eaten 12 = 63 := by
  sorry

#eval total_bugs_eaten 12

end NUMINAMATH_CALUDE_total_bugs_is_63_l1115_111520


namespace NUMINAMATH_CALUDE_prime_sequence_l1115_111535

theorem prime_sequence (A : ℕ) : 
  Nat.Prime A ∧ 
  Nat.Prime (A + 14) ∧ 
  Nat.Prime (A + 18) ∧ 
  Nat.Prime (A + 32) ∧ 
  Nat.Prime (A + 36) → 
  A = 5 :=
by sorry

end NUMINAMATH_CALUDE_prime_sequence_l1115_111535


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l1115_111568

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 38 / 2

/-- The number of books per bookshelf -/
def books_per_shelf : ℕ := 2

/-- The total number of books -/
def total_books : ℕ := 38

theorem bryan_bookshelves : 
  (num_bookshelves * books_per_shelf = total_books) ∧ (num_bookshelves = 19) :=
by sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l1115_111568


namespace NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l1115_111542

/-- Represents a rectangle divided into nine squares with integer side lengths -/
structure NineSquareRectangle where
  a : ℕ
  b : ℕ

/-- The perimeter of a NineSquareRectangle -/
def perimeter (r : NineSquareRectangle) : ℕ :=
  2 * ((3 * r.a + 8 * r.a) + (2 * r.a + 12 * r.a))

/-- Theorem stating the minimum perimeter of a NineSquareRectangle is 52 -/
theorem min_perimeter_nine_square_rectangle :
  ∃ (r : NineSquareRectangle), perimeter r = 52 ∧ ∀ (s : NineSquareRectangle), perimeter s ≥ 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l1115_111542


namespace NUMINAMATH_CALUDE_ellipse_equation_l1115_111567

/-- An ellipse with foci on the x-axis, focal distance 2√6, passing through (√3, √2) -/
structure Ellipse where
  /-- Half the distance between the foci -/
  c : ℝ
  /-- Semi-major axis -/
  a : ℝ
  /-- Semi-minor axis -/
  b : ℝ
  /-- Focal distance is 2√6 -/
  h_focal_distance : c = Real.sqrt 6
  /-- a > b > 0 -/
  h_a_gt_b : a > b ∧ b > 0
  /-- c² = a² - b² -/
  h_c_squared : c^2 = a^2 - b^2
  /-- The ellipse passes through (√3, √2) -/
  h_point : 3 / a^2 + 2 / b^2 = 1

/-- The standard equation of the ellipse is x²/9 + y²/3 = 1 -/
theorem ellipse_equation (e : Ellipse) : e.a^2 = 9 ∧ e.b^2 = 3 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l1115_111567


namespace NUMINAMATH_CALUDE_total_yellow_marbles_l1115_111540

theorem total_yellow_marbles (mary joan peter : ℕ) 
  (h1 : mary = 9) 
  (h2 : joan = 3) 
  (h3 : peter = 7) : 
  mary + joan + peter = 19 := by
sorry

end NUMINAMATH_CALUDE_total_yellow_marbles_l1115_111540


namespace NUMINAMATH_CALUDE_impossible_sum_of_two_smaller_angles_l1115_111534

theorem impossible_sum_of_two_smaller_angles (α β γ : ℝ) : 
  α > 0 → β > 0 → γ > 0 → 
  α + β + γ = 180 → 
  α ≤ γ → β ≤ γ → 
  α + β ≠ 130 :=
sorry

end NUMINAMATH_CALUDE_impossible_sum_of_two_smaller_angles_l1115_111534


namespace NUMINAMATH_CALUDE_bill_donut_combinations_l1115_111561

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 2 identical objects into 4 distinct boxes --/
def bill_combinations : ℕ := distribute 2 4

/-- Theorem: Bill's donut combinations equal 10 --/
theorem bill_donut_combinations : bill_combinations = 10 := by sorry

end NUMINAMATH_CALUDE_bill_donut_combinations_l1115_111561


namespace NUMINAMATH_CALUDE_impossibility_theorem_l1115_111574

/-- Represents a pile of chips -/
structure Pile :=
  (chips : ℕ)

/-- Represents the state of all piles -/
def State := List Pile

/-- The i-th prime number -/
def ithPrime (i : ℕ) : ℕ := sorry

/-- Initial state with 2018 piles, where the i-th pile has p_i chips (p_i is the i-th prime) -/
def initialState : State := 
  List.range 2018 |>.map (fun i => Pile.mk (ithPrime (i + 1)))

/-- Splits a pile into two piles and adds one chip to one of the new piles -/
def splitPile (s : State) (i : ℕ) (j k : ℕ) : State := sorry

/-- Merges two piles and adds one chip to the resulting pile -/
def mergePiles (s : State) (i j : ℕ) : State := sorry

/-- The target state with 2018 piles, each containing 2018 chips -/
def targetState : State := 
  List.replicate 2018 (Pile.mk 2018)

/-- Predicate to check if a given state is reachable from the initial state -/
def isReachable (s : State) : Prop := sorry

theorem impossibility_theorem : ¬ isReachable targetState := by
  sorry

end NUMINAMATH_CALUDE_impossibility_theorem_l1115_111574


namespace NUMINAMATH_CALUDE_sin_lt_tan_in_first_quadrant_half_angle_in_first_or_third_quadrant_sin_not_always_four_fifths_sector_angle_is_one_radian_l1115_111528

-- Statement ①
theorem sin_lt_tan_in_first_quadrant (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  Real.sin α < Real.tan α := by sorry

-- Statement ②
theorem half_angle_in_first_or_third_quadrant (α : Real) 
  (h : Real.pi / 2 < α ∧ α < Real.pi) :
  (0 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
  (Real.pi < α / 2 ∧ α / 2 < 3 * Real.pi / 2) := by sorry

-- Statement ③ (incorrect)
theorem sin_not_always_four_fifths (k : Real) (h : k ≠ 0) :
  ∃ α, Real.cos α = 3 * k / 5 ∧ Real.sin α = 4 * k / 5 ∧ Real.sin α ≠ 4 / 5 := by sorry

-- Statement ④
theorem sector_angle_is_one_radian (perimeter radius : Real) 
  (h1 : perimeter = 6) (h2 : radius = 2) :
  (perimeter - 2 * radius) / radius = 1 := by sorry

end NUMINAMATH_CALUDE_sin_lt_tan_in_first_quadrant_half_angle_in_first_or_third_quadrant_sin_not_always_four_fifths_sector_angle_is_one_radian_l1115_111528


namespace NUMINAMATH_CALUDE_inequality_proof_l1115_111530

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄)
  (h4 : x₂ + x₃ + x₄ ≥ x₁) :
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1115_111530


namespace NUMINAMATH_CALUDE_inequality_integer_solutions_l1115_111538

theorem inequality_integer_solutions :
  {x : ℤ | 3 ≤ 5 - 2*x ∧ 5 - 2*x ≤ 9} = {-2, -1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_integer_solutions_l1115_111538


namespace NUMINAMATH_CALUDE_john_pens_difference_l1115_111585

theorem john_pens_difference (total_pens blue_pens : ℕ) 
  (h_total : total_pens = 31)
  (h_blue : blue_pens = 18)
  (h_black_twice_red : ∃ (black_pens red_pens : ℕ), 
    total_pens = blue_pens + black_pens + red_pens ∧
    blue_pens = 2 * black_pens ∧
    black_pens > red_pens) :
  ∃ (black_pens red_pens : ℕ), black_pens - red_pens = 5 := by
sorry

end NUMINAMATH_CALUDE_john_pens_difference_l1115_111585


namespace NUMINAMATH_CALUDE_initial_deposit_calculation_l1115_111515

/-- Proves that the initial deposit is $1000 given the conditions of the problem -/
theorem initial_deposit_calculation (P : ℝ) : 
  (P + 100 = 1100) →                    -- First year balance
  ((P + 100) * 1.2 = P * 1.32) →         -- Second year growth equals total growth
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_deposit_calculation_l1115_111515


namespace NUMINAMATH_CALUDE_expression_value_when_x_is_two_l1115_111541

theorem expression_value_when_x_is_two :
  let x : ℝ := 2
  (x + 2 - x) * (2 - x - 2) = -4 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_when_x_is_two_l1115_111541


namespace NUMINAMATH_CALUDE_triangle_perpendicular_theorem_l1115_111529

structure Triangle (P Q R : ℝ × ℝ) where
  pq_length : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 15
  pr_length : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 20

def foot_of_perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (S.1 - Q.1) * (R.1 - Q.1) + (S.2 - Q.2) * (R.2 - Q.2) = 0 ∧
  (P.1 - S.1) * (R.1 - Q.1) + (P.2 - S.2) * (R.2 - Q.2) = 0

def segment_ratio (Q S R : ℝ × ℝ) : Prop :=
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) / Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) = 3 / 7

theorem triangle_perpendicular_theorem (P Q R S : ℝ × ℝ) 
  (tri : Triangle P Q R) (foot : foot_of_perpendicular P Q R S) (ratio : segment_ratio Q S R) :
  Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = 13.625 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perpendicular_theorem_l1115_111529


namespace NUMINAMATH_CALUDE_kyles_money_after_snowboarding_l1115_111576

/-- Calculates Kyle's remaining money after snowboarding -/
def kyles_remaining_money (daves_money : ℕ) : ℕ :=
  let kyles_initial_money := 3 * daves_money - 12
  let snowboarding_cost := kyles_initial_money / 3
  kyles_initial_money - snowboarding_cost

/-- Proves that Kyle has $84 left after snowboarding -/
theorem kyles_money_after_snowboarding :
  kyles_remaining_money 46 = 84 := by
  sorry

#eval kyles_remaining_money 46

end NUMINAMATH_CALUDE_kyles_money_after_snowboarding_l1115_111576


namespace NUMINAMATH_CALUDE_money_distribution_l1115_111537

/-- Given three people A, B, and C with money, prove that B and C together have 350 rupees. -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 450 →  -- Total money
  a + c = 200 →      -- Money A and C have together
  c = 100 →          -- Money C has
  b + c = 350 :=     -- Money B and C have together
by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1115_111537


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l1115_111550

theorem smallest_satisfying_number : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 10 → n % k = k - 1) ∧
  (∀ (m : ℕ), m > 0 ∧ 
    (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 10 → m % k = k - 1) → m ≥ 2519) ∧
  (2519 % 10 = 9 ∧ 
   2519 % 9 = 8 ∧ 
   2519 % 8 = 7 ∧ 
   2519 % 7 = 6 ∧ 
   2519 % 6 = 5 ∧ 
   2519 % 5 = 4 ∧ 
   2519 % 4 = 3 ∧ 
   2519 % 3 = 2 ∧ 
   2519 % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l1115_111550


namespace NUMINAMATH_CALUDE_camp_gender_difference_l1115_111503

theorem camp_gender_difference (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 133 →
  girls = 50 →
  boys > girls →
  total = boys + girls →
  boys - girls = 33 := by
sorry

end NUMINAMATH_CALUDE_camp_gender_difference_l1115_111503


namespace NUMINAMATH_CALUDE_systematic_sampling_first_sample_first_sample_is_18_l1115_111584

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sampleSize : ℕ
  interval : ℕ
  firstSample : ℕ
  eighteenthSample : ℕ

/-- Theorem stating the relationship between the first and eighteenth samples in systematic sampling -/
theorem systematic_sampling_first_sample
  (s : SystematicSampling)
  (h1 : s.population = 1000)
  (h2 : s.sampleSize = 40)
  (h3 : s.interval = s.population / s.sampleSize)
  (h4 : s.eighteenthSample = 443)
  (h5 : s.eighteenthSample = s.firstSample + 17 * s.interval) :
  s.firstSample = 18 := by
  sorry

/-- Main theorem proving the first sample number in the given scenario -/
theorem first_sample_is_18
  (population : ℕ)
  (sampleSize : ℕ)
  (eighteenthSample : ℕ)
  (h1 : population = 1000)
  (h2 : sampleSize = 40)
  (h3 : eighteenthSample = 443) :
  ∃ (s : SystematicSampling),
    s.population = population ∧
    s.sampleSize = sampleSize ∧
    s.interval = population / sampleSize ∧
    s.eighteenthSample = eighteenthSample ∧
    s.firstSample = 18 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_sample_first_sample_is_18_l1115_111584


namespace NUMINAMATH_CALUDE_general_term_formula_first_term_l1115_111595

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 3 * n.val

/-- The general term of the sequence -/
def a (n : ℕ+) : ℤ := 4 * n.val - 5

/-- Theorem stating that the general term formula is correct -/
theorem general_term_formula (n : ℕ+) : a n = S n - S (n - 1) := by
  sorry

/-- Theorem stating that the formula holds for the first term -/
theorem first_term : a 1 = S 1 := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_first_term_l1115_111595


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1115_111597

/-- The circle C with equation x^2 + y^2 = 10 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 10}

/-- The point P(1, 3) -/
def P : ℝ × ℝ := (1, 3)

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A line is tangent to a circle if it intersects the circle at exactly one point -/
def IsTangentTo (l : Line) (s : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ s ∧ l.a * p.1 + l.b * p.2 + l.c = 0

theorem tangent_line_equation :
  IsTangentTo (Line.mk 1 3 (-10)) C ∧ (Line.mk 1 3 (-10)).a * P.1 + (Line.mk 1 3 (-10)).b * P.2 + (Line.mk 1 3 (-10)).c = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1115_111597


namespace NUMINAMATH_CALUDE_min_staff_members_theorem_l1115_111573

/-- Represents the seating arrangement in a school hall --/
structure SchoolHall where
  male_students : ℕ
  female_students : ℕ
  benches_3_seats : ℕ
  benches_4_seats : ℕ

/-- Calculates the minimum number of staff members required --/
def min_staff_members (hall : SchoolHall) : ℕ :=
  let total_students := hall.male_students + hall.female_students
  let total_seats := 3 * hall.benches_3_seats + 4 * hall.benches_4_seats
  max (total_students - total_seats) 0

/-- Theorem stating the minimum number of staff members required --/
theorem min_staff_members_theorem (hall : SchoolHall) : 
  hall.male_students = 29 ∧ 
  hall.female_students = 4 * hall.male_students ∧ 
  hall.benches_3_seats = 15 ∧ 
  hall.benches_4_seats = 14 →
  min_staff_members hall = 44 := by
  sorry

#eval min_staff_members {
  male_students := 29,
  female_students := 116,
  benches_3_seats := 15,
  benches_4_seats := 14
}

end NUMINAMATH_CALUDE_min_staff_members_theorem_l1115_111573


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l1115_111591

theorem smallest_undefined_value (y : ℝ) : 
  (∀ z : ℝ, z < y → (z - 3) / (6 * z^2 - 37 * z + 6) ≠ 0) ∧ 
  ((y - 3) / (6 * y^2 - 37 * y + 6) = 0) → 
  y = 1/6 := by sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l1115_111591


namespace NUMINAMATH_CALUDE_divisibility_condition_l1115_111583

theorem divisibility_condition (n : ℕ) (hn : n ≥ 1) :
  (3^(n-1) + 5^(n-1)) ∣ (3^n + 5^n) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1115_111583


namespace NUMINAMATH_CALUDE_nest_distance_building_materials_distance_l1115_111578

/-- Given two birds making round trips to collect building materials, 
    calculate the distance from the nest to the materials. -/
theorem nest_distance (num_birds : ℕ) (num_trips : ℕ) (total_distance : ℝ) : ℝ :=
  let distance_per_bird := total_distance / num_birds
  let distance_per_trip := distance_per_bird / num_trips
  distance_per_trip / 4

/-- Prove that for two birds making 10 round trips each, 
    with a total distance of 8000 miles, 
    the building materials are 100 miles from the nest. -/
theorem building_materials_distance : 
  nest_distance 2 10 8000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_nest_distance_building_materials_distance_l1115_111578


namespace NUMINAMATH_CALUDE_measuring_cups_l1115_111593

theorem measuring_cups (a : Int) (h : -1562 ≤ a ∧ a ≤ 1562) :
  ∃ (b c d e f : Int),
    (b ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (c ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (d ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (e ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (f ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (a = 625*b + 125*c + 25*d + 5*e + f) :=
by sorry

end NUMINAMATH_CALUDE_measuring_cups_l1115_111593


namespace NUMINAMATH_CALUDE_number_equation_l1115_111543

theorem number_equation (x : ℝ) : 43 + 3 * x = 58 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1115_111543


namespace NUMINAMATH_CALUDE_angle_inequality_l1115_111507

theorem angle_inequality (θ : Real) : 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 2 → 
    x^2 * Real.sin θ - x * (2 - x) + (2 - x)^2 * Real.cos θ > 0) ↔ 
  (π / 12 < θ ∧ θ < 5 * π / 12) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_l1115_111507


namespace NUMINAMATH_CALUDE_point_to_line_distance_l1115_111599

/-- The distance from a point to a line in 2D space -/
theorem point_to_line_distance
  (x₀ y₀ a b c : ℝ) (h : a^2 + b^2 ≠ 0) :
  let d := |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)
  ∀ x y, a * x + b * y + c = 0 → 
    d ≤ Real.sqrt ((x - x₀)^2 + (y - y₀)^2) :=
by sorry

end NUMINAMATH_CALUDE_point_to_line_distance_l1115_111599


namespace NUMINAMATH_CALUDE_initial_salt_concentration_l1115_111510

theorem initial_salt_concentration
  (initial_volume : ℝ)
  (water_added : ℝ)
  (final_salt_percentage : ℝ)
  (h1 : initial_volume = 56)
  (h2 : water_added = 14)
  (h3 : final_salt_percentage = 0.08)
  (h4 : initial_volume * initial_salt_percentage = (initial_volume + water_added) * final_salt_percentage) :
  initial_salt_percentage = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_initial_salt_concentration_l1115_111510


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l1115_111523

theorem greatest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 9999 → n ≥ 1000 → n % 17 = 0 → n ≤ 9996 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l1115_111523


namespace NUMINAMATH_CALUDE_sum_of_four_unit_fractions_l1115_111527

theorem sum_of_four_unit_fractions : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sum_of_four_unit_fractions_l1115_111527


namespace NUMINAMATH_CALUDE_base_r_is_10_l1115_111545

/-- Converts a number from base r to base 10 -/
def toBase10 (digits : List Nat) (r : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * r ^ i) 0

/-- The problem statement -/
theorem base_r_is_10 (r : Nat) : r > 0 → 
  toBase10 [1, 3, 5] r + toBase10 [1, 5, 4] r = toBase10 [0, 0, 1, 1] r → 
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_r_is_10_l1115_111545


namespace NUMINAMATH_CALUDE_complex_exponential_185_54_l1115_111501

theorem complex_exponential_185_54 :
  (Complex.exp (185 * Real.pi / 180 * Complex.I))^54 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_185_54_l1115_111501


namespace NUMINAMATH_CALUDE_bracelets_lost_l1115_111587

theorem bracelets_lost (initial_bracelets : ℕ) (remaining_bracelets : ℕ) 
  (h1 : initial_bracelets = 9) 
  (h2 : remaining_bracelets = 7) : 
  initial_bracelets - remaining_bracelets = 2 := by
  sorry

end NUMINAMATH_CALUDE_bracelets_lost_l1115_111587


namespace NUMINAMATH_CALUDE_weight_of_b_l1115_111554

theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 47 →
  b = 39 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l1115_111554


namespace NUMINAMATH_CALUDE_second_class_size_l1115_111506

def students_first_class : ℕ := 25
def avg_marks_first_class : ℚ := 50
def avg_marks_second_class : ℚ := 65
def avg_marks_all : ℚ := 59.23076923076923

theorem second_class_size :
  ∃ (x : ℕ), 
    (students_first_class * avg_marks_first_class + x * avg_marks_second_class) / (students_first_class + x) = avg_marks_all ∧
    x = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_class_size_l1115_111506


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_correct_l1115_111569

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

def smallest_dual_palindrome : ℕ := 33

theorem smallest_dual_palindrome_correct :
  (smallest_dual_palindrome > 10) ∧
  (is_palindrome smallest_dual_palindrome 3) ∧
  (is_palindrome smallest_dual_palindrome 5) ∧
  (∀ m : ℕ, m > 10 ∧ m < smallest_dual_palindrome →
    ¬(is_palindrome m 3 ∧ is_palindrome m 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_correct_l1115_111569


namespace NUMINAMATH_CALUDE_jose_initial_caps_l1115_111572

/-- The number of bottle caps Jose gave to Rebecca -/
def given_caps : ℕ := 2

/-- The number of bottle caps Jose has left -/
def remaining_caps : ℕ := 5

/-- The initial number of bottle caps Jose had -/
def initial_caps : ℕ := given_caps + remaining_caps

theorem jose_initial_caps : initial_caps = 7 := by
  sorry

end NUMINAMATH_CALUDE_jose_initial_caps_l1115_111572


namespace NUMINAMATH_CALUDE_smallest_integer_c_l1115_111558

theorem smallest_integer_c (x : ℕ) (h : x = 8 * 3) : 
  (∃ c : ℕ, 27 ^ c > 3 ^ x ∧ ∀ k : ℕ, 27 ^ k > 3 ^ x → c ≤ k) → 
  (∃ c : ℕ, 27 ^ c > 3 ^ x ∧ ∀ k : ℕ, 27 ^ k > 3 ^ x → c ≤ k) ∧ 
  (∀ c : ℕ, 27 ^ c > 3 ^ x ∧ (∀ k : ℕ, 27 ^ k > 3 ^ x → c ≤ k) → c = 9) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_c_l1115_111558


namespace NUMINAMATH_CALUDE_chocolate_solution_l1115_111588

def chocolate_problem (n : ℕ) (c s : ℝ) : Prop :=
  -- Condition 1: The cost price of n chocolates equals the selling price of 150 chocolates
  n * c = 150 * s ∧
  -- Condition 2: The gain percent is 10
  (s - c) / c = 0.1

theorem chocolate_solution :
  ∃ (n : ℕ) (c s : ℝ), chocolate_problem n c s ∧ n = 165 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_solution_l1115_111588


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1115_111571

theorem sufficient_but_not_necessary_condition (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m = n → m^2 = n^2) ∧ ¬(m^2 = n^2 → m = n) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1115_111571


namespace NUMINAMATH_CALUDE_account_balance_difference_l1115_111582

/-- Computes the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Computes the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- The difference between two account balances after 25 years -/
theorem account_balance_difference : 
  let jessica_balance := compound_interest 12000 0.025 50
  let mark_balance := simple_interest 15000 0.06 25
  ∃ ε > 0, abs (jessica_balance - mark_balance - 3136) < ε :=
sorry

end NUMINAMATH_CALUDE_account_balance_difference_l1115_111582


namespace NUMINAMATH_CALUDE_open_box_volume_l1115_111566

/-- The volume of an open box formed by cutting squares from corners of a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_length : ℝ) :
  sheet_length = 48 ∧ 
  sheet_width = 36 ∧ 
  cut_length = 8 →
  let box_length := sheet_length - 2 * cut_length
  let box_width := sheet_width - 2 * cut_length
  let box_height := cut_length
  box_length * box_width * box_height = 5120 := by
sorry

end NUMINAMATH_CALUDE_open_box_volume_l1115_111566


namespace NUMINAMATH_CALUDE_longer_side_length_l1115_111592

/-- A rectangular plot with fence poles -/
structure FencedPlot where
  width : ℝ
  length : ℝ
  pole_distance : ℝ
  pole_count : ℕ

/-- The perimeter of a rectangle -/
def perimeter (plot : FencedPlot) : ℝ :=
  2 * (plot.width + plot.length)

/-- The total length of fencing -/
def fencing_length (plot : FencedPlot) : ℝ :=
  (plot.pole_count - 1 : ℝ) * plot.pole_distance

theorem longer_side_length (plot : FencedPlot) 
  (h1 : plot.width = 15)
  (h2 : plot.pole_distance = 5)
  (h3 : plot.pole_count = 26)
  (h4 : plot.width < plot.length)
  (h5 : perimeter plot = fencing_length plot) :
  plot.length = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_longer_side_length_l1115_111592


namespace NUMINAMATH_CALUDE_sum_of_possible_n_values_l1115_111581

/-- Given natural numbers 15, 12, and n, where the product of any two is divisible by the third,
    the sum of all possible values of n is 260. -/
theorem sum_of_possible_n_values : ∃ (S : Finset ℕ),
  (∀ n ∈ S, n > 0 ∧ 
    (15 * 12) % n = 0 ∧ 
    (15 * n) % 12 = 0 ∧ 
    (12 * n) % 15 = 0) ∧
  (∀ n > 0, 
    (15 * 12) % n = 0 ∧ 
    (15 * n) % 12 = 0 ∧ 
    (12 * n) % 15 = 0 → n ∈ S) ∧
  S.sum id = 260 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_possible_n_values_l1115_111581


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l1115_111516

theorem factorization_x_squared_minus_2x (x : ℝ) : x^2 - 2*x = x*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l1115_111516


namespace NUMINAMATH_CALUDE_inequality_proof_l1115_111575

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1115_111575


namespace NUMINAMATH_CALUDE_collinear_points_theorem_l1115_111586

/-- Four points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three vectors are collinear -/
def are_collinear (v1 v2 v3 : Point3D) : Prop :=
  ∃ (t1 t2 : ℝ), v2.x - v1.x = t1 * (v3.x - v1.x) ∧
                 v2.y - v1.y = t1 * (v3.y - v1.y) ∧
                 v2.z - v1.z = t1 * (v3.z - v1.z) ∧
                 t2 ≠ 0 ∧ t2 ≠ 1 ∧
                 v3.x - v1.x = t2 * (v2.x - v1.x) ∧
                 v3.y - v1.y = t2 * (v2.y - v1.y) ∧
                 v3.z - v1.z = t2 * (v2.z - v1.z)

theorem collinear_points_theorem (a c d : ℝ) :
  let p1 : Point3D := ⟨2, 0, a⟩
  let p2 : Point3D := ⟨2*a, 2, 0⟩
  let p3 : Point3D := ⟨0, c, 1⟩
  let p4 : Point3D := ⟨9*d, 9*d, -d⟩
  (are_collinear p1 p2 p3 ∧ are_collinear p1 p2 p4 ∧ are_collinear p1 p3 p4) →
  d = 1/9 :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_theorem_l1115_111586


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1115_111564

/-- The equation of the tangent line to y = x^2 + x + 1/2 at (0, 1/2) is y = x + 1/2 -/
theorem tangent_line_at_origin (x : ℝ) :
  let f (x : ℝ) := x^2 + x + 1/2
  let f' (x : ℝ) := 2*x + 1
  let tangent_line (x : ℝ) := x + 1/2
  (∀ x, deriv f x = f' x) →
  (f 0 = 1/2) →
  (tangent_line 0 = 1/2) →
  (f' 0 = 1) →
  ∀ x, tangent_line x = f 0 + f' 0 * x :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1115_111564


namespace NUMINAMATH_CALUDE_fraction_relationship_l1115_111547

theorem fraction_relationship (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relationship_l1115_111547


namespace NUMINAMATH_CALUDE_no_base_for_square_202_l1115_111552

-- Define the base-b representation of 202_b
def base_b_representation (b : ℕ) : ℕ := 2 * b^2 + 2

-- Define the property of being a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

-- Theorem statement
theorem no_base_for_square_202 :
  ∀ b : ℕ, b > 2 → ¬(is_perfect_square (base_b_representation b)) := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_square_202_l1115_111552


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1115_111504

theorem max_value_quadratic (x : ℝ) : 
  ∃ (max : ℝ), max = 9 ∧ ∀ y : ℝ, y = -3 * x^2 + 9 → y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1115_111504


namespace NUMINAMATH_CALUDE_gasoline_added_l1115_111512

theorem gasoline_added (tank_capacity : ℝ) (initial_fill : ℝ) (final_fill : ℝ) : tank_capacity = 54 → initial_fill = 3/4 → final_fill = 9/10 → (final_fill - initial_fill) * tank_capacity = 8.1 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_added_l1115_111512


namespace NUMINAMATH_CALUDE_brownies_remaining_l1115_111563

/-- The number of brownies left after Tina, her husband, and guests eat some. -/
def brownies_left (total : ℕ) (tina_daily : ℕ) (tina_days : ℕ) (husband_daily : ℕ) (husband_days : ℕ) (shared : ℕ) : ℕ :=
  total - (tina_daily * tina_days + husband_daily * husband_days + shared)

/-- Theorem stating that 5 brownies are left under the given conditions. -/
theorem brownies_remaining : brownies_left 24 2 5 1 5 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_brownies_remaining_l1115_111563


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1115_111509

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1115_111509


namespace NUMINAMATH_CALUDE_sequence_transformations_l1115_111551

def Sequence (α : Type) := ℕ → α

def is_obtainable (s t : Sequence ℝ) : Prop :=
  ∃ (operations : List (Sequence ℝ → Sequence ℝ)),
    (operations.foldl (λ acc op => op acc) s) = t

theorem sequence_transformations (a b c : Sequence ℝ) :
  (∀ n, a n = n^2) ∧
  (∀ n, b n = n + Real.sqrt 2) ∧
  (∀ n, c n = (n^2000 + 1) / n) →
  (is_obtainable a (λ n => n)) ∧
  (¬ is_obtainable b (λ n => n)) ∧
  (is_obtainable c (λ n => n)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_transformations_l1115_111551


namespace NUMINAMATH_CALUDE_samuel_breaks_two_cups_per_box_l1115_111517

theorem samuel_breaks_two_cups_per_box 
  (total_boxes : ℕ) 
  (pan_boxes : ℕ) 
  (cups_per_row : ℕ) 
  (rows_per_box : ℕ) 
  (remaining_cups : ℕ) 
  (h1 : total_boxes = 26)
  (h2 : pan_boxes = 6)
  (h3 : cups_per_row = 4)
  (h4 : rows_per_box = 5)
  (h5 : remaining_cups = 180) :
  let remaining_boxes := total_boxes - pan_boxes
  let decoration_boxes := remaining_boxes / 2
  let teacup_boxes := remaining_boxes - decoration_boxes
  let cups_per_box := cups_per_row * rows_per_box
  let total_cups := teacup_boxes * cups_per_box
  let broken_cups := total_cups - remaining_cups
  2 = broken_cups / teacup_boxes :=
by sorry

end NUMINAMATH_CALUDE_samuel_breaks_two_cups_per_box_l1115_111517


namespace NUMINAMATH_CALUDE_difference_of_squares_application_l1115_111536

theorem difference_of_squares_application (a b : ℝ) :
  (1/4 * a + b) * (b - 1/4 * a) = b^2 - (1/16) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_application_l1115_111536


namespace NUMINAMATH_CALUDE_book_pages_count_l1115_111548

/-- The number of pages Frank reads per day -/
def pages_per_day : ℕ := 22

/-- The number of days it took Frank to finish the book -/
def days_to_finish : ℕ := 569

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_day * days_to_finish

theorem book_pages_count : total_pages = 12518 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1115_111548


namespace NUMINAMATH_CALUDE_octagon_quad_area_ratio_l1115_111513

/-- Regular octagon with vertices ABCDEFGH -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- Quadrilateral formed by connecting alternate vertices of the octagon -/
def alternateVerticesQuad (octagon : RegularOctagon) : Fin 4 → ℝ × ℝ :=
  fun i => octagon.vertices (2 * i)

/-- Area of a polygon given its vertices -/
def polygonArea (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

theorem octagon_quad_area_ratio 
  (octagon : RegularOctagon) 
  (n : ℝ) 
  (m : ℝ) 
  (hn : n = polygonArea octagon.vertices) 
  (hm : m = polygonArea (alternateVerticesQuad octagon)) :
  m / n = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_octagon_quad_area_ratio_l1115_111513


namespace NUMINAMATH_CALUDE_atlantic_call_rate_l1115_111518

/-- Proves that the additional charge per minute for Atlantic Call is $0.20 -/
theorem atlantic_call_rate (united_base_rate : ℚ) (united_per_minute : ℚ) 
  (atlantic_base_rate : ℚ) (minutes : ℕ) :
  united_base_rate = 7 →
  united_per_minute = 0.25 →
  atlantic_base_rate = 12 →
  minutes = 100 →
  united_base_rate + united_per_minute * minutes = 
    atlantic_base_rate + (atlantic_base_rate + united_per_minute * minutes - united_base_rate) / minutes →
  (atlantic_base_rate + united_per_minute * minutes - united_base_rate) / minutes = 0.20 := by
  sorry

#check atlantic_call_rate

end NUMINAMATH_CALUDE_atlantic_call_rate_l1115_111518


namespace NUMINAMATH_CALUDE_mark_bench_press_value_l1115_111500

/-- Dave's weight in pounds -/
def dave_weight : ℝ := 175

/-- Dave's bench press multiplier -/
def dave_multiplier : ℝ := 3

/-- Craig's bench press percentage compared to Dave -/
def craig_percentage : ℝ := 0.2

/-- Difference between Craig's and Mark's bench press in pounds -/
def mark_difference : ℝ := 50

/-- Calculate Dave's bench press weight -/
def dave_bench_press : ℝ := dave_weight * dave_multiplier

/-- Calculate Craig's bench press weight -/
def craig_bench_press : ℝ := dave_bench_press * craig_percentage

/-- Calculate Mark's bench press weight -/
def mark_bench_press : ℝ := craig_bench_press - mark_difference

theorem mark_bench_press_value : mark_bench_press = 55 := by
  sorry

end NUMINAMATH_CALUDE_mark_bench_press_value_l1115_111500


namespace NUMINAMATH_CALUDE_shooting_probability_l1115_111562

/-- The probability of scoring less than 9 in a shooting practice -/
def prob_less_than_9 (prob_10 prob_9 prob_8 : ℝ) : Prop :=
  prob_10 = 0.24 ∧ prob_9 = 0.28 ∧ prob_8 = 0.19 →
  1 - (prob_10 + prob_9) = 0.29

theorem shooting_probability : 
  ∃ (prob_10 prob_9 prob_8 : ℝ), prob_less_than_9 prob_10 prob_9 prob_8 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probability_l1115_111562


namespace NUMINAMATH_CALUDE_min_area_circle_through_intersections_l1115_111577

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + y + 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the minimum area circle
def min_area_circle (x y : ℝ) : Prop := (x + 13/5)^2 + (y - 6/5)^2 = 4/5

-- Theorem statement
theorem min_area_circle_through_intersections :
  ∀ x y : ℝ, 
  (∃ x1 y1 x2 y2 : ℝ, 
    line_l x1 y1 ∧ circle_C x1 y1 ∧
    line_l x2 y2 ∧ circle_C x2 y2 ∧
    x1 ≠ x2 ∧ y1 ≠ y2 ∧
    min_area_circle x1 y1 ∧
    min_area_circle x2 y2) →
  (∀ r : ℝ, ∀ a b : ℝ,
    ((x - a)^2 + (y - b)^2 = r^2 ∧
     (∃ x1 y1 x2 y2 : ℝ, 
       line_l x1 y1 ∧ circle_C x1 y1 ∧
       line_l x2 y2 ∧ circle_C x2 y2 ∧
       (x1 - a)^2 + (y1 - b)^2 = r^2 ∧
       (x2 - a)^2 + (y2 - b)^2 = r^2)) →
    r^2 ≥ 4/5) :=
sorry

end NUMINAMATH_CALUDE_min_area_circle_through_intersections_l1115_111577


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l1115_111526

/-- Given a point with rectangular coordinates (2, -3, 6) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, π + θ, φ) has rectangular coordinates (-2, 3, 6). -/
theorem spherical_coordinate_transformation (ρ θ φ : Real) :
  (2 : Real) = ρ * Real.sin φ * Real.cos θ ∧
  (-3 : Real) = ρ * Real.sin φ * Real.sin θ ∧
  (6 : Real) = ρ * Real.cos φ →
  (-2 : Real) = ρ * Real.sin φ * Real.cos (Real.pi + θ) ∧
  (3 : Real) = ρ * Real.sin φ * Real.sin (Real.pi + θ) ∧
  (6 : Real) = ρ * Real.cos φ := by
  sorry


end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l1115_111526


namespace NUMINAMATH_CALUDE_green_paint_amount_l1115_111505

/-- The amount of green paint needed for a treehouse project. -/
def green_paint (total white brown : ℕ) : ℕ :=
  total - (white + brown)

/-- Theorem stating that the amount of green paint is 15 ounces. -/
theorem green_paint_amount :
  green_paint 69 20 34 = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_paint_amount_l1115_111505


namespace NUMINAMATH_CALUDE_no_three_digit_even_sum_27_l1115_111553

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_three_digit_even_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_no_three_digit_even_sum_27_l1115_111553


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1115_111544

/-- Given a parabola and a moving line with common points, prove the range of t and minimum value of c -/
theorem parabola_line_intersection (t c x₁ x₂ y₁ y₂ : ℝ) : 
  (∀ x, y₁ = x^2 ∧ y₁ = (2*t - 1)*x - c) →  -- Parabola and line equations
  (∀ x, y₂ = x^2 ∧ y₂ = (2*t - 1)*x - c) →  -- Parabola and line equations
  x₁^2 + x₂^2 = t^2 + 2*t - 3 →             -- Given condition
  (2 - Real.sqrt 2 ≤ t ∧ t ≤ 2 + Real.sqrt 2 ∧ t ≠ 1/2) ∧  -- Range of t
  (c ≥ (11 - 6*Real.sqrt 2) / 4) ∧                        -- Minimum value of c
  (c = (11 - 6*Real.sqrt 2) / 4 ↔ t = 2 - Real.sqrt 2)    -- When minimum occurs
  := by sorry


end NUMINAMATH_CALUDE_parabola_line_intersection_l1115_111544


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l1115_111589

def incorrect_mean : ℝ := 120
def num_values : ℕ := 40
def original_values : List ℝ := [-50, 350, 100, 25, -80]
def incorrect_values : List ℝ := [-30, 320, 120, 60, -100]

theorem correct_mean_calculation :
  let incorrect_sum := incorrect_mean * num_values
  let difference := (List.sum original_values) - (List.sum incorrect_values)
  let correct_sum := incorrect_sum + difference
  correct_sum / num_values = 119.375 := by
sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l1115_111589


namespace NUMINAMATH_CALUDE_initial_pencils_on_desk_l1115_111511

def pencils_in_drawer : ℕ := 43
def pencils_added : ℕ := 16
def total_pencils : ℕ := 78

theorem initial_pencils_on_desk :
  total_pencils = pencils_in_drawer + pencils_added + (total_pencils - pencils_in_drawer - pencils_added) ∧
  (total_pencils - pencils_in_drawer - pencils_added) = 19 :=
by sorry

end NUMINAMATH_CALUDE_initial_pencils_on_desk_l1115_111511


namespace NUMINAMATH_CALUDE_max_sum_products_l1115_111560

theorem max_sum_products (a b c d : ℕ) : 
  a ∈ ({2, 3, 4, 5} : Set ℕ) → 
  b ∈ ({2, 3, 4, 5} : Set ℕ) → 
  c ∈ ({2, 3, 4, 5} : Set ℕ) → 
  d ∈ ({2, 3, 4, 5} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (∀ x y z w, x ∈ ({2, 3, 4, 5} : Set ℕ) → 
              y ∈ ({2, 3, 4, 5} : Set ℕ) → 
              z ∈ ({2, 3, 4, 5} : Set ℕ) → 
              w ∈ ({2, 3, 4, 5} : Set ℕ) → 
              x ≠ y → x ≠ z → x ≠ w → y ≠ z → y ≠ w → z ≠ w →
              x * y + x * z + x * w + y * z ≤ a * b + a * c + a * d + b * c) →
  a * b + a * c + a * d + b * c = 39 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_products_l1115_111560


namespace NUMINAMATH_CALUDE_square_difference_equals_one_l1115_111525

theorem square_difference_equals_one (x y : ℝ) 
  (sum_eq : x + y = 5) 
  (product_eq : x * y = 6) : 
  (x - y)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_square_difference_equals_one_l1115_111525


namespace NUMINAMATH_CALUDE_snow_leopard_arrangements_l1115_111522

/-- The number of ways to arrange n different objects --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of snow leopards --/
def total_leopards : ℕ := 8

/-- The number of leopards that can be freely arranged --/
def free_leopards : ℕ := total_leopards - 2

/-- The number of ways to arrange the shortest and tallest leopards --/
def end_arrangements : ℕ := 2

theorem snow_leopard_arrangements :
  end_arrangements * permutations free_leopards = 1440 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangements_l1115_111522


namespace NUMINAMATH_CALUDE_equation_solutions_l1115_111565

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1^2 - 5*x1 - 6 = 0 ∧ x2^2 - 5*x2 - 6 = 0 ∧ x1 = 6 ∧ x2 = -1) ∧
  (∃ y1 y2 : ℝ, (y1 + 1)*(y1 - 1) + y1*(y1 + 2) = 7 + 6*y1 ∧
                (y2 + 1)*(y2 - 1) + y2*(y2 + 2) = 7 + 6*y2 ∧
                y1 = Real.sqrt 5 + 1 ∧ y2 = 1 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1115_111565


namespace NUMINAMATH_CALUDE_car_trip_speed_l1115_111508

/-- Proves that the speed for the remaining part of the trip is 20 mph given the conditions of the problem -/
theorem car_trip_speed (x t : ℝ) (h1 : x > 0) (h2 : t > 0) : ∃ s : ℝ,
  (0.75 * x / 60 + 0.25 * x / s = t) ∧ 
  (x / t = 40) →
  s = 20 := by
  sorry


end NUMINAMATH_CALUDE_car_trip_speed_l1115_111508


namespace NUMINAMATH_CALUDE_partition_of_naturals_l1115_111590

/-- The set of natural numbers starting from 1 -/
def ℕ' : Set ℕ := {n : ℕ | n ≥ 1}

/-- The set S(x, y) for real x and y -/
def S (x y : ℝ) : Set ℕ := {s : ℕ | ∃ n : ℕ, n ∈ ℕ' ∧ s = ⌊n * x + y⌋}

/-- The main theorem -/
theorem partition_of_naturals (r : ℚ) (hr : r > 1) :
  ∃ u v : ℝ, (S r 0 ∩ S u v = ∅) ∧ (S r 0 ∪ S u v = ℕ') := by
  sorry

end NUMINAMATH_CALUDE_partition_of_naturals_l1115_111590


namespace NUMINAMATH_CALUDE_age_difference_l1115_111570

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a = c + 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1115_111570


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l1115_111559

/-- The coordinates of the foci for the hyperbola x^2 - 4y^2 = 4 are (±√5, 0) -/
theorem hyperbola_foci_coordinates :
  let h : ℝ → ℝ → Prop := λ x y => x^2 - 4*y^2 = 4
  ∃ c : ℝ, c^2 = 5 ∧ 
    (∀ x y, h x y ↔ (x/2)^2 - y^2 = 1) ∧
    (∀ x y, h x y → (x = c ∨ x = -c) ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l1115_111559


namespace NUMINAMATH_CALUDE_faye_coloring_books_l1115_111533

theorem faye_coloring_books (initial : ℝ) (given_away : ℝ) (additional_percentage : ℝ) :
  initial = 52.5 →
  given_away = 38.2 →
  additional_percentage = 25 →
  let remainder : ℝ := initial - given_away
  let additional_given : ℝ := (additional_percentage / 100) * remainder
  initial - given_away - additional_given = 10.725 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l1115_111533


namespace NUMINAMATH_CALUDE_initial_items_count_l1115_111532

/-- The number of items Adam initially put in the shopping cart -/
def initial_items : ℕ := sorry

/-- The number of items Adam deleted from the shopping cart -/
def deleted_items : ℕ := 10

/-- The number of items left in Adam's shopping cart after deletion -/
def remaining_items : ℕ := 8

/-- Theorem stating that the initial number of items is 18 -/
theorem initial_items_count : initial_items = 18 :=
  by sorry

end NUMINAMATH_CALUDE_initial_items_count_l1115_111532


namespace NUMINAMATH_CALUDE_email_sample_not_representative_l1115_111524

/-- Represents the urban population --/
def UrbanPopulation : Type := Unit

/-- Represents a person in the urban population --/
def Person : Type := Unit

/-- Represents whether a person has an email address --/
def has_email (p : Person) : Prop := sorry

/-- Represents whether a person uses the internet for news --/
def uses_internet_for_news (p : Person) : Prop := sorry

/-- Represents a sample of the population --/
def Sample := Set Person

/-- Defines what it means for a sample to be representative --/
def is_representative (s : Sample) : Prop := sorry

/-- The sample of email address owners --/
def email_sample : Sample := sorry

/-- Theorem stating that the sample of email address owners is not representative --/
theorem email_sample_not_representative :
  (∀ p : Person, has_email p → uses_internet_for_news p) →
  ¬ (is_representative email_sample) := by sorry

end NUMINAMATH_CALUDE_email_sample_not_representative_l1115_111524


namespace NUMINAMATH_CALUDE_largest_result_is_630_l1115_111580

-- Define the set of available digits
def Digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the allowed operations
inductive Operation
| Add : Operation
| Sub : Operation
| Mul : Operation
| Div : Operation

-- Define a sequence of operations
def OperationSequence := List (Operation × Nat)

-- Function to apply a sequence of operations
def applyOperations (seq : OperationSequence) : Nat :=
  sorry

-- Theorem stating that 630 is the largest possible result
theorem largest_result_is_630 :
  ∀ (seq : OperationSequence),
    (∀ n ∈ Digits, (seq.map Prod.snd).count n = 1) →
    applyOperations seq ≤ 630 :=
  sorry

end NUMINAMATH_CALUDE_largest_result_is_630_l1115_111580


namespace NUMINAMATH_CALUDE_student_927_selected_l1115_111514

/-- Represents a student number in the range [1, 1000] -/
def StudentNumber := Fin 1000

/-- The total number of students -/
def totalStudents : Nat := 1000

/-- The number of students to be sampled -/
def sampleSize : Nat := 200

/-- The sampling interval -/
def samplingInterval : Nat := totalStudents / sampleSize

/-- Predicate to check if a student number is selected in the systematic sampling -/
def isSelected (n : StudentNumber) : Prop :=
  n.val % samplingInterval = 122 % samplingInterval

/-- Theorem stating that if student 122 is selected, then student 927 is also selected -/
theorem student_927_selected :
  isSelected ⟨121, by norm_num⟩ → isSelected ⟨926, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_student_927_selected_l1115_111514


namespace NUMINAMATH_CALUDE_smallest_append_digits_for_2014_l1115_111539

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10 → k > 0 → n % k = 0

def append_digits (base n digits : ℕ) : ℕ :=
  base * (10 ^ digits) + n

theorem smallest_append_digits_for_2014 :
  (∃ n : ℕ, n < 10000 ∧ is_divisible_by_all_less_than_10 (append_digits 2014 4 n)) ∧
  (∀ d : ℕ, d < 4 → ∀ n : ℕ, n < 10^d → ¬is_divisible_by_all_less_than_10 (append_digits 2014 d n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_append_digits_for_2014_l1115_111539


namespace NUMINAMATH_CALUDE_constant_avg_speed_not_imply_uniform_motion_l1115_111596

/-- A snail's motion over a time interval -/
structure SnailMotion where
  /-- The time interval in minutes -/
  interval : ℝ
  /-- The distance traveled in meters -/
  distance : ℝ
  /-- The average speed in meters per minute -/
  avg_speed : ℝ
  /-- Condition: The average speed is constant -/
  constant_avg_speed : avg_speed = distance / interval

/-- Definition of uniform motion -/
def is_uniform_motion (motion : SnailMotion) : Prop :=
  ∀ t : ℝ, 0 ≤ t → t ≤ motion.interval →
    motion.distance * (t / motion.interval) = motion.avg_speed * t

/-- Theorem: Constant average speed does not imply uniform motion -/
theorem constant_avg_speed_not_imply_uniform_motion :
  ∃ (motion : SnailMotion), ¬(is_uniform_motion motion) :=
sorry

end NUMINAMATH_CALUDE_constant_avg_speed_not_imply_uniform_motion_l1115_111596


namespace NUMINAMATH_CALUDE_polygon_sides_l1115_111598

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 360 * 3) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1115_111598
