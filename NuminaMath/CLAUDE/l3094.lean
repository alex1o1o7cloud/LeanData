import Mathlib

namespace baking_time_with_oven_failure_l3094_309429

/-- The time taken to make caramel-apple coffee cakes on a day when the oven failed -/
theorem baking_time_with_oven_failure 
  (assembly_time : ℝ) 
  (normal_baking_time : ℝ) 
  (decoration_time : ℝ) 
  (h1 : assembly_time = 1) 
  (h2 : normal_baking_time = 1.5) 
  (h3 : decoration_time = 1) :
  assembly_time + 2 * normal_baking_time + decoration_time = 5 := by
sorry

end baking_time_with_oven_failure_l3094_309429


namespace maximum_marks_calculation_l3094_309415

theorem maximum_marks_calculation (percentage : ℝ) (scored_marks : ℝ) (maximum_marks : ℝ) :
  percentage = 90 →
  scored_marks = 405 →
  percentage / 100 * maximum_marks = scored_marks →
  maximum_marks = 450 :=
by
  sorry

end maximum_marks_calculation_l3094_309415


namespace set_equality_implies_sum_zero_l3094_309494

theorem set_equality_implies_sum_zero (x y : ℝ) : 
  ({x, y, x + y} : Set ℝ) = {0, x^2, x*y} → x + y = 0 := by
  sorry

end set_equality_implies_sum_zero_l3094_309494


namespace tangent_line_to_ellipse_l3094_309455

/-- Ellipse definition -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

/-- Line equation -/
def line_equation (x y x₀ y₀ : ℝ) : Prop :=
  x * x₀ / 16 + y * y₀ / 4 = 1

/-- Tangent line property -/
def is_tangent_line (x₀ y₀ : ℝ) : Prop :=
  is_on_ellipse x₀ y₀ →
  ∀ x y : ℝ, line_equation x y x₀ y₀ →
  (x = x₀ ∧ y = y₀) ∨ ¬(is_on_ellipse x y)

/-- Main theorem -/
theorem tangent_line_to_ellipse :
  ∀ x₀ y₀ : ℝ, is_tangent_line x₀ y₀ :=
sorry

end tangent_line_to_ellipse_l3094_309455


namespace triangle_problem_l3094_309450

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  a = 6 →
  1/2 * a * c * Real.sin B = 6 * Real.sqrt 3 →
  ((B = π/3 ∨ B = 2*π/3) ∧ (b = 2 * Real.sqrt 7 ∨ b = Real.sqrt 76)) := by
  sorry

end triangle_problem_l3094_309450


namespace smallest_sum_is_11_l3094_309467

/-- B is a digit in base 4 -/
def is_base_4_digit (B : ℕ) : Prop := B < 4

/-- b is a base greater than 5 -/
def is_base_greater_than_5 (b : ℕ) : Prop := b > 5

/-- BBB₄ = 44ᵦ -/
def equality_condition (B b : ℕ) : Prop := 21 * B = 4 * (b + 1)

/-- The smallest possible sum of B and b is 11 -/
theorem smallest_sum_is_11 :
  ∃ (B b : ℕ), is_base_4_digit B ∧ is_base_greater_than_5 b ∧ equality_condition B b ∧
  B + b = 11 ∧
  ∀ (B' b' : ℕ), is_base_4_digit B' → is_base_greater_than_5 b' → equality_condition B' b' →
  B' + b' ≥ 11 :=
sorry

end smallest_sum_is_11_l3094_309467


namespace pen_price_before_discount_l3094_309422

-- Define the problem parameters
def num_pens : ℕ := 30
def num_pencils : ℕ := 75
def total_cost : ℚ := 570
def pen_discount : ℚ := 0.1
def pencil_tax : ℚ := 0.05
def avg_pencil_price : ℚ := 2

-- Define the theorem
theorem pen_price_before_discount :
  let pencil_cost := num_pencils * avg_pencil_price
  let pencil_cost_with_tax := pencil_cost * (1 + pencil_tax)
  let pen_cost_with_discount := total_cost - pencil_cost_with_tax
  let pen_cost_before_discount := pen_cost_with_discount / (1 - pen_discount)
  let avg_pen_price := pen_cost_before_discount / num_pens
  ∃ (x : ℚ), abs (x - avg_pen_price) < 0.005 ∧ x = 15.28 :=
by sorry


end pen_price_before_discount_l3094_309422


namespace composition_ratio_l3094_309440

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 2 * x - 1

theorem composition_ratio : f (g (f 3)) / g (f (g 3)) = 79 / 37 := by
  sorry

end composition_ratio_l3094_309440


namespace largest_multiple_of_seven_less_than_negative_hundred_l3094_309483

theorem largest_multiple_of_seven_less_than_negative_hundred :
  ∀ n : ℤ, n * 7 < -100 → n * 7 ≤ -105 :=
by
  sorry

end largest_multiple_of_seven_less_than_negative_hundred_l3094_309483


namespace valid_arrangements_count_l3094_309461

def digits : List Nat := [1, 1, 5, 5]

def is_multiple_of_five (n : Nat) : Prop :=
  n % 5 = 0

def is_four_digit (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000

def count_valid_arrangements (ds : List Nat) : Nat :=
  sorry

theorem valid_arrangements_count :
  count_valid_arrangements digits = 3 := by
  sorry

end valid_arrangements_count_l3094_309461


namespace equation_solution_l3094_309498

theorem equation_solution :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 2 * x - 3 = 0 :=
by
  use 3/2
  sorry

end equation_solution_l3094_309498


namespace odd_function_property_l3094_309423

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Property of the function f as given in the problem -/
def HasProperty (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem odd_function_property (f : ℝ → ℝ) (h_odd : IsOdd f) (h_prop : HasProperty f) :
  f (-4) > f (-6) := by
  sorry

end odd_function_property_l3094_309423


namespace differentiable_functions_inequality_l3094_309496

open Set

theorem differentiable_functions_inequality 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x > deriv g x) 
  (a b x : ℝ) 
  (h_x : x ∈ Ioo a b) : 
  (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) := by
  sorry

end differentiable_functions_inequality_l3094_309496


namespace sasha_sticker_collection_l3094_309487

theorem sasha_sticker_collection (m n : ℕ) (t : ℝ) : 
  m < n →
  m > 0 →
  t > 1 →
  m * t + n = 100 →
  m + n * t = 101 →
  (n = 34 ∨ n = 66) ∧ ∀ k : ℕ, (k ≠ 34 ∧ k ≠ 66) → 
    ¬(∃ m' : ℕ, ∃ t' : ℝ, 
      m' < k ∧ 
      m' > 0 ∧ 
      t' > 1 ∧ 
      m' * t' + k = 100 ∧ 
      m' + k * t' = 101) :=
by sorry

end sasha_sticker_collection_l3094_309487


namespace ellipse_foci_distance_sum_l3094_309452

/-- An ellipse with semi-major axis 5 and semi-minor axis 4 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 25) + (p.2^2 / 16) = 1}

/-- The foci of the ellipse -/
def Foci : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: For any point on the ellipse, the sum of distances to the foci is 10 -/
theorem ellipse_foci_distance_sum (p : ℝ × ℝ) (h : p ∈ Ellipse) :
  distance p Foci.1 + distance p Foci.2 = 10 := by sorry

end ellipse_foci_distance_sum_l3094_309452


namespace line_equation_through_point_with_inclination_l3094_309420

/-- The equation of a line passing through (-2, 3) with an inclination angle of 45° is x - y + 5 = 0 -/
theorem line_equation_through_point_with_inclination (x y : ℝ) : 
  (∃ (m : ℝ), m = Real.tan (π / 4) ∧ 
    y - 3 = m * (x - (-2))) ↔ 
  x - y + 5 = 0 := by
  sorry

end line_equation_through_point_with_inclination_l3094_309420


namespace max_stamps_with_50_dollars_l3094_309466

/-- The maximum number of stamps that can be purchased with a given budget and stamp price -/
def max_stamps (budget : ℕ) (stamp_price : ℕ) : ℕ :=
  (budget / stamp_price : ℕ)

/-- Theorem: Given a stamp price of 25 cents and a budget of 5000 cents, 
    the maximum number of stamps that can be purchased is 200 -/
theorem max_stamps_with_50_dollars :
  max_stamps 5000 25 = 200 := by
  sorry

end max_stamps_with_50_dollars_l3094_309466


namespace same_even_on_all_dice_l3094_309484

/-- A standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The probability of rolling an even number on a standard die -/
def probEven : ℚ := 1/2

/-- The probability of rolling a specific number on a standard die -/
def probSpecific : ℚ := 1/6

/-- The number of dice being rolled -/
def numDice : ℕ := 4

/-- Theorem: The probability of all dice showing the same even number -/
theorem same_even_on_all_dice : 
  probEven * probSpecific^(numDice - 1) = 1/432 := by sorry

end same_even_on_all_dice_l3094_309484


namespace smallest_third_term_l3094_309408

/-- An arithmetic sequence of five positive integers with sum 80 -/
structure ArithmeticSequence where
  a : ℕ+  -- first term
  d : ℕ+  -- common difference
  sum_eq_80 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 80

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem stating that the smallest possible third term is 16 -/
theorem smallest_third_term :
  ∀ seq : ArithmeticSequence, third_term seq ≥ 16 := by
  sorry

#check smallest_third_term

end smallest_third_term_l3094_309408


namespace prime_pairs_divisibility_l3094_309435

theorem prime_pairs_divisibility (p q : ℕ) : 
  Prime p → Prime q → p ≤ q → (p * q) ∣ ((5^p - 2^p) * (7^q - 2^q)) →
  ((p = 3 ∧ q = 5) ∨ (p = 3 ∧ q = 3) ∨ (p = 5 ∧ q = 37) ∨ (p = 5 ∧ q = 83)) :=
by sorry

end prime_pairs_divisibility_l3094_309435


namespace symmetric_point_wrt_x_axis_l3094_309449

/-- Given a point P(-2, 1), prove that its symmetric point Q with respect to the x-axis has coordinates (-2, -1) -/
theorem symmetric_point_wrt_x_axis :
  let P : ℝ × ℝ := (-2, 1)
  let Q : ℝ × ℝ := (-2, -1)
  (Q.1 = P.1) ∧ (Q.2 = -P.2) := by sorry

end symmetric_point_wrt_x_axis_l3094_309449


namespace tangent_perpendicular_l3094_309499

-- Define the curve C
def C (x : ℝ) : ℝ := x^2 + x

-- Define the derivative of C
def C' (x : ℝ) : ℝ := 2*x + 1

-- Define the perpendicular line
def perp_line (a x y : ℝ) : Prop := a*x - y + 1 = 0

theorem tangent_perpendicular :
  ∀ a : ℝ, 
  (C' 1 = -1/a) →  -- The slope of the tangent at x=1 is the negative reciprocal of a
  a = -1/3 := by
sorry

end tangent_perpendicular_l3094_309499


namespace roots_product_plus_one_l3094_309434

theorem roots_product_plus_one (a b : ℝ) : 
  a^2 + 2*a - 2023 = 0 → 
  b^2 + 2*b - 2023 = 0 → 
  (a + 1) * (b + 1) = -2024 := by
sorry

end roots_product_plus_one_l3094_309434


namespace point_on_line_l3094_309486

/-- Given a line passing through points (0, 3) and (-8, 0),
    if the point (t, 7) lies on this line, then t = 32/3 -/
theorem point_on_line (t : ℚ) :
  (∀ (x y : ℚ), (y - 3) / (x - 0) = (0 - 3) / (-8 - 0) →
    (7 - 3) / (t - 0) = (0 - 3) / (-8 - 0)) →
  t = 32 / 3 := by sorry

end point_on_line_l3094_309486


namespace inscribed_circle_radius_l3094_309497

/-- The radius of an inscribed circle in a sector -/
theorem inscribed_circle_radius (R : ℝ) (h : R = 5) :
  let sector_angle : ℝ := 2 * Real.pi / 3
  let r : ℝ := R * (Real.sqrt 3 - 1) / 2
  r = (5 * Real.sqrt 3 - 5) / 2 ∧ 
  r > 0 ∧ 
  r * (Real.sqrt 3 + 1) = R := by
  sorry

end inscribed_circle_radius_l3094_309497


namespace distribute_five_balls_three_boxes_l3094_309427

/-- The number of ways to distribute indistinguishable objects into distinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : distribute 5 3 = 21 := by
  sorry

end distribute_five_balls_three_boxes_l3094_309427


namespace reflection_segment_length_C_l3094_309432

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflection_segment_length (x y : ℝ) : ℝ :=
  2 * |y|

/-- Theorem: The length of the segment from C(4, 3) to its reflection C' over the x-axis is 6 --/
theorem reflection_segment_length_C : reflection_segment_length 4 3 = 6 := by
  sorry

end reflection_segment_length_C_l3094_309432


namespace vikas_rank_among_boys_l3094_309465

/-- Represents the ranking information of students in a class -/
structure ClassRanking where
  total_students : ℕ
  vikas_overall_rank : ℕ
  tanvi_overall_rank : ℕ
  girls_between : ℕ
  vikas_boys_top_rank : ℕ
  vikas_bottom_rank : ℕ

/-- The theorem to prove Vikas's rank among boys -/
theorem vikas_rank_among_boys (c : ClassRanking) 
  (h1 : c.vikas_overall_rank = 9)
  (h2 : c.tanvi_overall_rank = 17)
  (h3 : c.girls_between = 2)
  (h4 : c.vikas_boys_top_rank = 4)
  (h5 : c.vikas_bottom_rank = 18) :
  c.vikas_boys_top_rank = 4 := by
  sorry


end vikas_rank_among_boys_l3094_309465


namespace line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l3094_309464

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)

-- State the theorem
theorem line_perp_to_plane_and_line_para_to_plane_implies_lines_perp
  (m n : Line) (α : Plane)
  (h1 : perp m α)
  (h2 : para n α) :
  perpLine m n :=
sorry

end line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l3094_309464


namespace jessica_driving_days_l3094_309419

/-- Calculates the number of days needed to meet a driving hour requirement -/
def daysToMeetRequirement (requiredHours : ℕ) (minutesPerTrip : ℕ) : ℕ :=
  let requiredMinutes := requiredHours * 60
  let minutesPerDay := minutesPerTrip * 2
  requiredMinutes / minutesPerDay

theorem jessica_driving_days :
  daysToMeetRequirement 50 20 = 75 := by
  sorry

#eval daysToMeetRequirement 50 20

end jessica_driving_days_l3094_309419


namespace power_equation_solution_l3094_309492

theorem power_equation_solution : ∃! x : ℝ, (5 : ℝ)^3 + (5 : ℝ)^3 + (5 : ℝ)^3 = (15 : ℝ)^x := by
  sorry

end power_equation_solution_l3094_309492


namespace factorization_a_squared_minus_8a_l3094_309493

theorem factorization_a_squared_minus_8a (a : ℝ) : a^2 - 8*a = a*(a - 8) := by
  sorry

end factorization_a_squared_minus_8a_l3094_309493


namespace ratio_of_11th_terms_l3094_309495

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := n * (a₁ + (n - 1) / 2 * d)

theorem ratio_of_11th_terms
  (a₁ d₁ a₂ d₂ : ℚ)
  (h : ∀ n : ℕ, sum_arithmetic_sequence a₁ d₁ n / sum_arithmetic_sequence a₂ d₂ n = (7 * n + 1) / (4 * n + 27)) :
  (arithmetic_sequence a₁ d₁ 11) / (arithmetic_sequence a₂ d₂ 11) = 4 / 3 := by
  sorry

end ratio_of_11th_terms_l3094_309495


namespace divisibility_of_binomial_difference_l3094_309463

theorem divisibility_of_binomial_difference (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (a + b)^p - a^p - b^p = k * p :=
sorry

end divisibility_of_binomial_difference_l3094_309463


namespace bisection_next_step_l3094_309403

/-- The bisection method's next step for a function with given properties -/
theorem bisection_next_step (f : ℝ → ℝ) (h1 : f 1 < 0) (h2 : f 1.5 > 0) :
  let x₀ : ℝ := (1 + 1.5) / 2
  x₀ = 1.25 := by sorry

end bisection_next_step_l3094_309403


namespace linear_functions_properties_l3094_309460

/-- Linear function y₁ -/
def y₁ (x : ℝ) : ℝ := 50 + 2 * x

/-- Linear function y₂ -/
def y₂ (x : ℝ) : ℝ := 5 * x

theorem linear_functions_properties :
  (∃ x : ℝ, y₁ x > y₂ x) ∧ 
  (∃ x : ℝ, y₁ x < y₂ x) ∧
  (∀ x dx : ℝ, y₁ (x + dx) - y₁ x = 2 * dx) ∧
  (∀ x dx : ℝ, y₂ (x + dx) - y₂ x = 5 * dx) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≥ 1 ∧ x₂ ≥ 1 ∧ y₂ x₂ ≥ 100 ∧ y₁ x₁ ≥ 100 ∧ x₂ < x₁ ∧
    ∀ x : ℝ, x ≥ 1 → y₂ x ≥ 100 → x ≥ x₂) :=
by sorry

end linear_functions_properties_l3094_309460


namespace fermat_number_prime_divisors_l3094_309417

theorem fermat_number_prime_divisors (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_divides : p ∣ 2^(2^n) + 1) : 
  ∃ k : ℕ, p = k * 2^(n + 1) + 1 := by
sorry

end fermat_number_prime_divisors_l3094_309417


namespace max_x_value_l3094_309469

/-- Represents the linear relationship between x and y --/
def linear_relation (x y : ℝ) : Prop := y = x - 5

/-- The maximum forecast value for y --/
def max_y : ℝ := 10

/-- Theorem stating the maximum value of x given the conditions --/
theorem max_x_value (h : linear_relation max_y max_x) : max_x = 15 := by
  sorry

end max_x_value_l3094_309469


namespace bacteria_growth_relation_l3094_309445

/-- Represents the amount of bacteria at a given time point -/
structure BacteriaAmount where
  amount : ℝ
  time : ℕ

/-- Represents a growth factor between two time points -/
structure GrowthFactor where
  factor : ℝ
  startTime : ℕ
  endTime : ℕ

/-- Theorem stating the relationship between initial, intermediate, and final bacteria amounts
    and their corresponding growth factors -/
theorem bacteria_growth_relation 
  (A₁ : BacteriaAmount) 
  (A₂ : BacteriaAmount) 
  (A₃ : BacteriaAmount) 
  (g : GrowthFactor) 
  (h : GrowthFactor) : 
  A₁.time = 1 →
  A₂.time = 4 →
  A₃.time = 7 →
  g.startTime = 1 →
  g.endTime = 4 →
  h.startTime = 4 →
  h.endTime = 7 →
  A₁.amount = 10 →
  A₃.amount = 12.1 →
  A₂.amount = A₁.amount * g.factor →
  A₃.amount = A₂.amount * h.factor →
  A₃.amount = A₁.amount * g.factor * h.factor :=
by sorry

end bacteria_growth_relation_l3094_309445


namespace negative_two_and_negative_half_are_reciprocals_l3094_309412

-- Define the concept of reciprocals
def are_reciprocals (a b : ℚ) : Prop := a * b = 1

-- Theorem statement
theorem negative_two_and_negative_half_are_reciprocals :
  are_reciprocals (-2) (-1/2) := by
  sorry

end negative_two_and_negative_half_are_reciprocals_l3094_309412


namespace division_problem_l3094_309414

theorem division_problem : ∃ (D : ℕ+) (N : ℤ), 
  N = 5 * D.val ∧ N % 11 = 2 ∧ D = 7 := by
  sorry

end division_problem_l3094_309414


namespace rectangle_grid_ratio_l3094_309447

/-- Given a 3x2 grid of identical rectangles with height h and width w,
    and a line segment PQ intersecting the grid as described,
    prove that h/w = 3/8 -/
theorem rectangle_grid_ratio (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) : 
  let grid_width := 3 * w
  let grid_height := 2 * h
  ∃ (X Y Z : ℝ × ℝ),
    X.1 ∈ Set.Icc 0 grid_width ∧
    X.2 ∈ Set.Icc 0 grid_height ∧
    Z.1 ∈ Set.Icc 0 grid_width ∧
    Z.2 ∈ Set.Icc 0 grid_height ∧
    Y.1 = X.1 ∧
    Y.2 = Z.2 ∧
    (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = (Y.1 - X.1)^2 + (Y.2 - X.2)^2 + (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 ∧
    (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 4 * ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) →
  h / w = 3 / 8 := by
  sorry

end rectangle_grid_ratio_l3094_309447


namespace maria_savings_l3094_309462

/-- The amount of money Maria will have left after buying sweaters and scarves -/
def money_left (sweater_price scarf_price num_sweaters num_scarves savings : ℕ) : ℕ :=
  savings - (sweater_price * num_sweaters + scarf_price * num_scarves)

/-- Theorem stating that Maria will have $200 left after her purchases -/
theorem maria_savings : money_left 30 20 6 6 500 = 200 := by
  sorry

end maria_savings_l3094_309462


namespace larger_square_construction_l3094_309491

/-- Represents a square in 2D space -/
structure Square where
  side : ℝ
  deriving Inhabited

/-- Represents the construction of a larger square from two smaller squares -/
def construct_larger_square (s1 s2 : Square) : Square :=
  sorry

/-- Theorem stating that it's possible to construct a larger square from two smaller squares
    without cutting the smaller square -/
theorem larger_square_construction (s1 s2 : Square) :
  ∃ (large : Square), 
    large.side^2 = s1.side^2 + s2.side^2 ∧
    construct_larger_square s1 s2 = large :=
  sorry

end larger_square_construction_l3094_309491


namespace abs_neg_2023_l3094_309476

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_neg_2023_l3094_309476


namespace nonagon_diagonals_l3094_309457

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by sorry

end nonagon_diagonals_l3094_309457


namespace cab_driver_income_l3094_309479

theorem cab_driver_income (income : List ℝ) (average : ℝ) : 
  income.length = 5 →
  income[0]! = 600 →
  income[1]! = 250 →
  income[2]! = 450 →
  income[4]! = 800 →
  average = (income.sum / income.length) →
  average = 500 →
  income[3]! = 400 := by
sorry

end cab_driver_income_l3094_309479


namespace infinitely_many_squares_in_ap_l3094_309474

/-- An arithmetic progression of positive integers. -/
def ArithmeticProgression (a d : ℕ) : ℕ → ℕ
  | n => a + n * d

/-- Predicate to check if a number is a perfect square. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- The main theorem to be proved. -/
theorem infinitely_many_squares_in_ap (a d : ℕ) (h : d > 0) :
  (∃ n : ℕ, IsPerfectSquare (ArithmeticProgression a d n)) →
  (∀ m : ℕ, ∃ n : ℕ, n > m ∧ IsPerfectSquare (ArithmeticProgression a d n)) :=
by sorry


end infinitely_many_squares_in_ap_l3094_309474


namespace unique_solution_condition_l3094_309407

/-- The equation (3x+7)(x-5) = -27 + kx has exactly one real solution if and only if k = -8 + 4√6 or k = -8 - 4√6 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x+7)*(x-5) = -27 + k*x) ↔ 
  (k = -8 + 4*Real.sqrt 6 ∨ k = -8 - 4*Real.sqrt 6) := by
sorry

end unique_solution_condition_l3094_309407


namespace questionnaire_responses_l3094_309490

theorem questionnaire_responses (response_rate : ℝ) (questionnaires_mailed : ℕ) 
  (h1 : response_rate = 0.8)
  (h2 : questionnaires_mailed = 375) :
  ⌊response_rate * questionnaires_mailed⌋ = 300 := by
  sorry

end questionnaire_responses_l3094_309490


namespace eight_term_sequence_sum_l3094_309472

def sequence_sum (seq : List ℤ) (i : ℕ) : ℤ :=
  if i + 2 < seq.length then
    seq[i]! + seq[i+1]! + seq[i+2]!
  else
    0

theorem eight_term_sequence_sum (P Q R S T U V W : ℤ) : 
  R = 8 →
  (∀ i, i + 2 < 8 → sequence_sum [P, Q, R, S, T, U, V, W] i = 35) →
  P + W = 27 := by
sorry

end eight_term_sequence_sum_l3094_309472


namespace intersection_P_complement_M_l3094_309448

def U : Set Int := Set.univ

def M : Set Int := {1, 2}

def P : Set Int := {-2, -1, 0, 1, 2}

theorem intersection_P_complement_M :
  P ∩ (U \ M) = {-2, -1, 0} := by sorry

end intersection_P_complement_M_l3094_309448


namespace cube_root_two_identity_l3094_309438

theorem cube_root_two_identity (s : ℝ) : s = 1 / (1 - Real.rpow 2 (1/3)) → s = -(1 + Real.rpow 2 (1/3) + Real.rpow 2 (2/3)) := by
  sorry

end cube_root_two_identity_l3094_309438


namespace cauchy_schwarz_inequality_l3094_309451

theorem cauchy_schwarz_inequality (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 := by
  sorry

end cauchy_schwarz_inequality_l3094_309451


namespace punger_pages_needed_l3094_309441

/-- The number of pages needed to hold all baseball cards -/
def pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) : ℕ :=
  (packs * cards_per_pack + cards_per_page - 1) / cards_per_page

/-- Proof that 42 pages are needed for Punger's baseball cards -/
theorem punger_pages_needed :
  pages_needed 60 7 10 = 42 := by
  sorry

end punger_pages_needed_l3094_309441


namespace properties_of_f_l3094_309456

def f (x : ℝ) := -5 * x

theorem properties_of_f :
  (∃ m b : ℝ, ∀ x, f x = m * x + b) ∧  -- f is linear
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂) ∧   -- f is decreasing
  (f 0 = 0) ∧                          -- f passes through (0,0)
  (∀ x ≠ 0, x * (f x) < 0) :=          -- f is in 2nd and 4th quadrants
by sorry

end properties_of_f_l3094_309456


namespace sqrt_negative_square_defined_unique_l3094_309442

theorem sqrt_negative_square_defined_unique : 
  ∃! a : ℝ, ∃ x : ℝ, x^2 = -(1-a)^2 := by
  sorry

end sqrt_negative_square_defined_unique_l3094_309442


namespace xyz_sum_l3094_309454

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y = 32)
  (h2 : x * z = 64)
  (h3 : y * z = 96) :
  x + y + z = 28 := by
  sorry

end xyz_sum_l3094_309454


namespace pot_stacking_l3094_309416

theorem pot_stacking (total_pots : ℕ) (vertical_stack : ℕ) (shelves : ℕ) 
  (h1 : total_pots = 60)
  (h2 : vertical_stack = 5)
  (h3 : shelves = 4) :
  (total_pots / vertical_stack) / shelves = 3 := by
  sorry

end pot_stacking_l3094_309416


namespace trig_roots_equation_l3094_309424

theorem trig_roots_equation (θ : ℝ) (a : ℝ) :
  (∀ x, x^2 - a*x + a = 0 ↔ x = Real.sin θ ∨ x = Real.cos θ) →
  Real.cos (θ - 3*Real.pi/2) + Real.sin (3*Real.pi/2 + θ) = Real.sqrt 2 - 1 := by
  sorry

end trig_roots_equation_l3094_309424


namespace special_function_sum_l3094_309406

/-- A function satisfying f(p+q) = f(p)f(q) for all p and q, and f(1) = 3 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ p q : ℝ, f (p + q) = f p * f q) ∧ (f 1 = 3)

/-- The main theorem to prove -/
theorem special_function_sum (f : ℝ → ℝ) (h : special_function f) :
  (f 1^2 + f 2) / f 1 + (f 2^2 + f 4) / f 3 + (f 3^2 + f 6) / f 5 +
  (f 4^2 + f 8) / f 7 + (f 5^2 + f 10) / f 9 = 30 := by
  sorry

end special_function_sum_l3094_309406


namespace circle_line_intersection_l3094_309405

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 15 = 0

/-- The line equation -/
def line_eq (x y m : ℝ) : Prop := (1 + 3*m)*x + (3 - 2*m)*y + 4*m - 17 = 0

/-- The theorem stating that the circle and line always intersect at two points -/
theorem circle_line_intersection :
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    (∀ m : ℝ, circle_eq p.1 p.2 ∧ line_eq p.1 p.2 m) ∧
    (∀ m : ℝ, circle_eq q.1 q.2 ∧ line_eq q.1 q.2 m) ∧
    (∀ r : ℝ × ℝ, (∀ m : ℝ, circle_eq r.1 r.2 ∧ line_eq r.1 r.2 m) → r = p ∨ r = q) :=
sorry

end circle_line_intersection_l3094_309405


namespace complement_of_A_l3094_309482

-- Define the set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- State the theorem
theorem complement_of_A : 
  {x : ℝ | ¬ (x ∈ A)} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end complement_of_A_l3094_309482


namespace company_survey_l3094_309439

/-- The number of employees who do not use social networks -/
def non_users : ℕ := 40

/-- The fraction of social network users who use VKontakte -/
def vk_users : ℚ := 3/4

/-- The fraction of social network users who use both VKontakte and Odnoklassniki -/
def both_users : ℚ := 13/20

/-- The fraction of total employees who use Odnoklassniki -/
def ok_users : ℚ := 5/6

/-- The total number of employees in the company -/
def total_employees : ℕ := 540

theorem company_survey :
  ∃ (N : ℕ),
    N = total_employees ∧
    (N - non_users : ℚ) * (vk_users + (1 - vk_users)) = N * ok_users :=
by sorry

end company_survey_l3094_309439


namespace complex_equation_solution_l3094_309430

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I)^2 * z = 3 + 2 * Complex.I →
  z = -1 + (3/2) * Complex.I :=
by sorry

end complex_equation_solution_l3094_309430


namespace unique_magnitude_of_quadratic_roots_l3094_309418

theorem unique_magnitude_of_quadratic_roots (w : ℂ) :
  w^2 - 6*w + 40 = 0 → ∃! m : ℝ, ∃ w : ℂ, w^2 - 6*w + 40 = 0 ∧ Complex.abs w = m := by
  sorry

end unique_magnitude_of_quadratic_roots_l3094_309418


namespace solution_set_equals_expected_solutions_l3094_309426

/-- The set of solutions to the system of equations -/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | 3 * (x^2 + y^2 + z^2) = 1 ∧ x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x + y + z)^3}

/-- The set of expected solutions -/
def ExpectedSolutions : Set (ℝ × ℝ × ℝ) :=
  {(1/3, 1/3, 1/3), (-1/3, -1/3, -1/3), (1/Real.sqrt 3, 0, 0), (0, 1/Real.sqrt 3, 0), (0, 0, 1/Real.sqrt 3)}

/-- Theorem stating that the solution set is equal to the expected solutions -/
theorem solution_set_equals_expected_solutions : SolutionSet = ExpectedSolutions := by
  sorry


end solution_set_equals_expected_solutions_l3094_309426


namespace hcf_problem_l3094_309481

theorem hcf_problem (a b : ℕ+) : 
  (∃ h : ℕ+, Nat.lcm a b = h * 13 * 14) →
  max a b = 322 →
  Nat.gcd a b = 7 := by
sorry

end hcf_problem_l3094_309481


namespace sine_graph_shift_l3094_309489

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (2 * x + π / 4) = 3 * Real.sin (2 * (x + π / 8)) :=
by sorry

end sine_graph_shift_l3094_309489


namespace spinner_probability_l3094_309400

/-- Represents an equilateral triangle dissected by its altitudes -/
structure DissectedTriangle where
  regions : ℕ
  shaded_regions : ℕ

/-- The probability of a spinner landing in a shaded region -/
def landing_probability (t : DissectedTriangle) : ℚ :=
  t.shaded_regions / t.regions

/-- Theorem stating the probability of landing in a shaded region -/
theorem spinner_probability (t : DissectedTriangle) 
  (h1 : t.regions = 6)
  (h2 : t.shaded_regions = 3) : 
  landing_probability t = 1/2 := by
  sorry

end spinner_probability_l3094_309400


namespace billy_video_watching_l3094_309413

def total_time : ℕ := 90
def video_watch_time : ℕ := 4
def search_time : ℕ := 3
def break_time : ℕ := 5
def trial_count : ℕ := 5
def suggestions_per_trial : ℕ := 15
def additional_categories : ℕ := 2
def suggestions_per_category : ℕ := 10

def max_videos_watched : ℕ := 13

theorem billy_video_watching :
  let total_search_time := search_time * trial_count
  let total_break_time := break_time * (trial_count - 1)
  let available_watch_time := total_time - (total_search_time + total_break_time)
  max_videos_watched = available_watch_time / video_watch_time ∧
  max_videos_watched ≤ suggestions_per_trial * trial_count +
                       suggestions_per_category * additional_categories :=
by sorry

end billy_video_watching_l3094_309413


namespace min_value_arithmetic_progression_l3094_309409

/-- Given real numbers x, y, z in [0, 4] where x^2, y^2, z^2 form an arithmetic progression 
    with common difference 2, the minimum value of |x-y|+|y-z| is 4 - 2√3 -/
theorem min_value_arithmetic_progression (x y z : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 4) 
  (h2 : 0 ≤ y ∧ y ≤ 4) 
  (h3 : 0 ≤ z ∧ z ≤ 4) 
  (h4 : y^2 - x^2 = z^2 - y^2) 
  (h5 : y^2 - x^2 = 2) : 
  ∃ (m : ℝ), m = 4 - 2 * Real.sqrt 3 ∧ 
  ∀ (x' y' z' : ℝ), 0 ≤ x' ∧ x' ≤ 4 → 0 ≤ y' ∧ y' ≤ 4 → 0 ≤ z' ∧ z' ≤ 4 → 
  y'^2 - x'^2 = z'^2 - y'^2 → y'^2 - x'^2 = 2 → 
  m ≤ |x' - y'| + |y' - z'| :=
by
  sorry

end min_value_arithmetic_progression_l3094_309409


namespace clothing_price_problem_l3094_309470

theorem clothing_price_problem (total_spent : ℕ) (num_pieces : ℕ) (tax_rate : ℚ)
  (untaxed_piece1 : ℕ) (untaxed_piece2 : ℕ) (h_total : total_spent = 610)
  (h_num : num_pieces = 7) (h_tax : tax_rate = 1/10) (h_untaxed1 : untaxed_piece1 = 49)
  (h_untaxed2 : untaxed_piece2 = 81) :
  ∃ (price : ℕ), price * 5 = (total_spent - untaxed_piece1 - untaxed_piece2) * 10 / 11 ∧
  price % 5 = 0 ∧ price = 87 := by
sorry

end clothing_price_problem_l3094_309470


namespace compute_expression_l3094_309402

theorem compute_expression : 4 * 4^3 - 16^60 / 16^57 = -3840 := by sorry

end compute_expression_l3094_309402


namespace total_cost_correct_l3094_309478

def makeup_palette_price : ℝ := 15
def lipstick_price : ℝ := 2.5
def hair_color_price : ℝ := 4

def makeup_palette_count : ℕ := 3
def lipstick_count : ℕ := 4
def hair_color_count : ℕ := 3

def makeup_palette_discount : ℝ := 0.2
def hair_color_coupon_discount : ℝ := 0.1
def reward_points_discount : ℝ := 5

def storewide_discount_threshold : ℝ := 50
def storewide_discount_rate : ℝ := 0.1

def sales_tax_threshold : ℝ := 25
def sales_tax_rate_low : ℝ := 0.05
def sales_tax_rate_high : ℝ := 0.08

def calculate_total_cost : ℝ := sorry

theorem total_cost_correct : 
  ∀ ε > 0, |calculate_total_cost - 47.41| < ε := by sorry

end total_cost_correct_l3094_309478


namespace parabola_y_comparison_l3094_309404

/-- Given a parabola y = -x² + 4x + c, prove that the y-coordinate of the point (-1, y₁) 
    is less than the y-coordinate of the point (1, y₂) on this parabola. -/
theorem parabola_y_comparison (c : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : y₁ = -(-1)^2 + 4*(-1) + c) 
  (h₂ : y₂ = -(1)^2 + 4*(1) + c) : 
  y₁ < y₂ := by
  sorry

end parabola_y_comparison_l3094_309404


namespace median_condition_implies_right_triangle_l3094_309436

/-- Given a triangle with medians m₁, m₂, and m₃, if m₁² + m₂² = 5m₃², then the triangle is right. -/
theorem median_condition_implies_right_triangle 
  (m₁ m₂ m₃ : ℝ) 
  (h_medians : ∃ (a b c : ℝ), 
    m₁^2 = (2*(b^2 + c^2) - a^2) / 4 ∧ 
    m₂^2 = (2*(a^2 + c^2) - b^2) / 4 ∧ 
    m₃^2 = (2*(a^2 + b^2) - c^2) / 4)
  (h_condition : m₁^2 + m₂^2 = 5 * m₃^2) :
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 := by
  sorry

end median_condition_implies_right_triangle_l3094_309436


namespace problem_solution_l3094_309485

theorem problem_solution : ∃ n : ℕ+, 
  (24 ∣ n) ∧ 
  (8.2 < (n : ℝ) ^ (1/3 : ℝ)) ∧ 
  ((n : ℝ) ^ (1/3 : ℝ) < 8.3) := by
  sorry

end problem_solution_l3094_309485


namespace no_consecutive_squares_equal_consecutive_fourth_powers_l3094_309401

theorem no_consecutive_squares_equal_consecutive_fourth_powers :
  ¬ ∃ (m n : ℕ), m^2 + (m+1)^2 = n^4 + (n+1)^4 := by
sorry

end no_consecutive_squares_equal_consecutive_fourth_powers_l3094_309401


namespace slope_of_line_intersecting_hyperbola_l3094_309468

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 3)

-- Define the distance from a point to the focus
def dist_to_focus (x y : ℝ) : ℝ := 2*x - 1

theorem slope_of_line_intersecting_hyperbola (k : ℝ) :
  (∃ A B : ℝ × ℝ,
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    line k A.1 A.2 ∧
    line k B.1 B.2 ∧
    A.1 > 1 ∧
    B.1 > 1 ∧
    dist_to_focus A.1 A.2 + dist_to_focus B.1 B.2 = 16) →
  k = 3 ∨ k = -3 :=
sorry

end slope_of_line_intersecting_hyperbola_l3094_309468


namespace problem_statement_l3094_309473

theorem problem_statement (x : ℝ) (h : x^2 + x = 1) :
  3*x^4 + 3*x^3 + 3*x + 1 = 4 := by
  sorry

end problem_statement_l3094_309473


namespace fifth_root_equality_l3094_309437

theorem fifth_root_equality : ∃ (x y : ℤ), (119287 - 48682 * Real.sqrt 6) ^ (1/5 : ℝ) = x + y * Real.sqrt 6 :=
by sorry

end fifth_root_equality_l3094_309437


namespace bowtie_equation_solution_l3094_309428

/-- Definition of the bow-tie operation -/
noncomputable def bowtie (p q : ℝ) : ℝ := p + Real.sqrt (q + Real.sqrt (q + Real.sqrt (q + Real.sqrt q)))

/-- Theorem: If 5 bow-tie q equals 13, then q equals 56 -/
theorem bowtie_equation_solution (q : ℝ) : bowtie 5 q = 13 → q = 56 := by
  sorry

end bowtie_equation_solution_l3094_309428


namespace grill_runtime_proof_l3094_309459

/-- Represents the burning rate of coals in a grill -/
structure BurningRate :=
  (coals : ℕ)
  (minutes : ℕ)

/-- Represents a bag of coals -/
structure CoalBag :=
  (coals : ℕ)

def grill_running_time (rate : BurningRate) (bags : ℕ) (bag : CoalBag) : ℕ :=
  (bags * bag.coals * rate.minutes) / rate.coals

theorem grill_runtime_proof (rate : BurningRate) (bags : ℕ) (bag : CoalBag)
  (h1 : rate.coals = 15)
  (h2 : rate.minutes = 20)
  (h3 : bags = 3)
  (h4 : bag.coals = 60) :
  grill_running_time rate bags bag = 240 :=
by
  sorry

#check grill_runtime_proof

end grill_runtime_proof_l3094_309459


namespace range_of_m_when_p_implies_q_l3094_309444

/-- Represents an ellipse equation with parameter m -/
def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2*m) + y^2 / (1-m) = 1 ∧ 0 < m ∧ m < 1/3

/-- Represents a hyperbola equation with parameter m -/
def is_hyperbola_with_eccentricity_between_1_and_2 (m : ℝ) : Prop :=
  ∃ x y e : ℝ, x^2 / 5 - y^2 / m = 1 ∧ 1 < e ∧ e < 2 ∧ m > 0

/-- The main theorem stating the range of m -/
theorem range_of_m_when_p_implies_q :
  (∀ m : ℝ, is_ellipse_with_foci_on_y_axis m → is_hyperbola_with_eccentricity_between_1_and_2 m) →
  ∃ m : ℝ, 1/3 ≤ m ∧ m < 15 :=
sorry

end range_of_m_when_p_implies_q_l3094_309444


namespace special_trapezoid_ratio_l3094_309453

/-- Isosceles trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- Length of the shorter base -/
  a : ℝ
  /-- Length of the longer base -/
  long_base : ℝ
  /-- Length of the altitude -/
  altitude : ℝ
  /-- Length of a diagonal -/
  diagonal : ℝ
  /-- Longer base is square of shorter base -/
  long_base_eq : long_base = a^2
  /-- Shorter base equals altitude -/
  altitude_eq : altitude = a
  /-- Diagonal equals radius of circumscribed circle -/
  diagonal_eq : diagonal = 2

/-- The ratio of shorter base to longer base in the special trapezoid is 3/16 -/
theorem special_trapezoid_ratio (t : SpecialTrapezoid) : t.a / t.long_base = 3/16 := by
  sorry

end special_trapezoid_ratio_l3094_309453


namespace quadratic_unique_solution_l3094_309425

theorem quadratic_unique_solution (b c : ℝ) : 
  (∃! x, 3 * x^2 + b * x + c = 0) →
  b + c = 15 →
  3 * c = b^2 →
  b = (-3 + 3 * Real.sqrt 21) / 2 ∧ 
  c = (33 - 3 * Real.sqrt 21) / 2 := by
sorry

end quadratic_unique_solution_l3094_309425


namespace balls_after_2023_steps_l3094_309488

/-- Converts a natural number to its base-8 representation --/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else go (m / 8) ((m % 8) :: acc)
  go n []

/-- Sums the digits in a list of natural numbers --/
def sumDigits (l : List ℕ) : ℕ :=
  l.sum

/-- The process of placing balls in boxes as described in the problem --/
def ballProcess (steps : ℕ) : ℕ :=
  sumDigits (toBase8 steps)

/-- The theorem stating that the number of balls after 2023 steps is 21 --/
theorem balls_after_2023_steps :
  ballProcess 2023 = 21 := by
  sorry

end balls_after_2023_steps_l3094_309488


namespace quadratic_equation_transform_l3094_309410

/-- Given a quadratic equation 4x^2 + 16x - 400 = 0, prove that when transformed
    into the form (x + k)^2 = t, the value of t is 104. -/
theorem quadratic_equation_transform (x k t : ℝ) : 
  (4 * x^2 + 16 * x - 400 = 0) → 
  (∃ k, ∀ x, 4 * x^2 + 16 * x - 400 = 0 ↔ (x + k)^2 = t) →
  t = 104 := by
  sorry

end quadratic_equation_transform_l3094_309410


namespace unique_items_count_l3094_309477

/-- Represents a Beatles fan's collection --/
structure BeatlesFan where
  albums : ℕ
  memorabilia : ℕ

/-- Given the information about Andrew and John's collections, prove that the number of items
    in either Andrew's or John's collection or memorabilia, but not both, is 24. --/
theorem unique_items_count (andrew john : BeatlesFan) 
  (h1 : andrew.albums = 23)
  (h2 : andrew.memorabilia = 5)
  (h3 : john.albums = andrew.albums - 12 + 8) : 
  (andrew.albums - 12) + (john.albums - (andrew.albums - 12)) + andrew.memorabilia = 24 := by
  sorry

end unique_items_count_l3094_309477


namespace tangent_slope_at_point_A_l3094_309471

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2

-- State the theorem
theorem tangent_slope_at_point_A :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  (y₀ = 8) →  -- This ensures the point (2,8) is on the curve
  (deriv f x₀ = 8) :=
by sorry

end tangent_slope_at_point_A_l3094_309471


namespace age_relationships_l3094_309411

-- Define variables for ages
variable (a b c d : ℝ)

-- Define the conditions
def condition1 : Prop := a + b = b + c + d + 18
def condition2 : Prop := a / c = 3 / 2

-- Define the theorem
theorem age_relationships 
  (h1 : condition1 a b c d) 
  (h2 : condition2 a c) : 
  c = (2/3) * a ∧ d = (1/3) * a - 18 ∧ b = 0 := by
  sorry

end age_relationships_l3094_309411


namespace fib_equals_tiling_pred_l3094_309458

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of ways to tile a 1 × n rectangle with 1 × 1 squares and 1 × 2 dominos -/
def tiling : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => tiling (n + 1) + tiling n

/-- Theorem: The n-th Fibonacci number equals the number of ways to tile a 1 × (n-1) rectangle -/
theorem fib_equals_tiling_pred (n : ℕ) : fib n = tiling (n - 1) := by
  sorry

end fib_equals_tiling_pred_l3094_309458


namespace school_travel_time_l3094_309433

/-- If a boy reaches school t minutes early when walking at 1.2 times his usual speed,
    his usual time to reach school is 6t minutes. -/
theorem school_travel_time (t : ℝ) (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) 
    (h2 : usual_time > 0) 
    (h3 : usual_speed * usual_time = 1.2 * usual_speed * (usual_time - t)) : 
  usual_time = 6 * t := by
sorry

end school_travel_time_l3094_309433


namespace parallel_line_intercepts_l3094_309446

/-- A line parallel to y = 3x - 2 passing through (5, -1) has y-intercept -16 and x-intercept 16/3 -/
theorem parallel_line_intercepts :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b y = 3 * x + (b 0 - 3 * 0)) →  -- b is parallel to y = 3x - 2
  b (-1) = 5 →  -- b passes through (5, -1)
  b 0 = -16 ∧ b (16/3) = 0 := by
sorry

end parallel_line_intercepts_l3094_309446


namespace hyperbola_eccentricity_is_sqrt_5_l3094_309480

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Predicate indicating that the focus of a hyperbola is symmetric with respect to its asymptote -/
def focus_symmetric_to_asymptote (h : Hyperbola a b) : Prop := sorry

/-- Predicate indicating that the focus of a hyperbola lies on the hyperbola -/
def focus_on_hyperbola (h : Hyperbola a b) : Prop := sorry

/-- Theorem stating that if the focus of a hyperbola is symmetric with respect to its asymptote
    and lies on the hyperbola, then its eccentricity is √5 -/
theorem hyperbola_eccentricity_is_sqrt_5 {a b : ℝ} (h : Hyperbola a b)
  (h_sym : focus_symmetric_to_asymptote h) (h_on : focus_on_hyperbola h) :
  eccentricity h = Real.sqrt 5 := by sorry

end hyperbola_eccentricity_is_sqrt_5_l3094_309480


namespace permutations_of_red_l3094_309443

def word : String := "red"

theorem permutations_of_red (w : String) (h : w = word) : 
  Nat.factorial w.length = 6 := by
  sorry

end permutations_of_red_l3094_309443


namespace book_pages_theorem_l3094_309421

/-- Calculates the total number of pages in a book given the number of chapters and pages per chapter -/
def totalPages (chapters : ℕ) (pagesPerChapter : ℕ) : ℕ :=
  chapters * pagesPerChapter

/-- Theorem stating that a book with 31 chapters, each 61 pages long, has 1891 pages in total -/
theorem book_pages_theorem :
  totalPages 31 61 = 1891 := by
  sorry

end book_pages_theorem_l3094_309421


namespace max_value_theorem_l3094_309431

theorem max_value_theorem (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 9) 
  (h2 : 3 * a + 5 * b ≤ 12) : 
  2 * a + b ≤ 39 / 11 := by
  sorry

end max_value_theorem_l3094_309431


namespace max_tickets_purchasable_l3094_309475

theorem max_tickets_purchasable (ticket_price : ℚ) (budget : ℚ) : 
  ticket_price = 15.75 → budget = 200 → 
  ∃ n : ℕ, n * ticket_price ≤ budget ∧ 
           ∀ m : ℕ, m * ticket_price ≤ budget → m ≤ n ∧
           n = 12 :=
by sorry

end max_tickets_purchasable_l3094_309475
