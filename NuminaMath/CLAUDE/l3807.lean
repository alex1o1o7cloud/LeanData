import Mathlib

namespace fourth_rectangle_area_l3807_380770

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  total_area : ℝ
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  vertical_halves : area1 + area4 = area2 + area3
  sum_of_areas : total_area = area1 + area2 + area3 + area4

/-- Theorem stating that given the areas of three rectangles in a divided rectangle,
    we can determine the area of the fourth rectangle -/
theorem fourth_rectangle_area
  (rect : DividedRectangle)
  (h1 : rect.area1 = 12)
  (h2 : rect.area2 = 27)
  (h3 : rect.area3 = 18) :
  rect.area4 = 27 := by
  sorry

#check fourth_rectangle_area

end fourth_rectangle_area_l3807_380770


namespace rug_inner_length_is_four_l3807_380708

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the rug with three regions -/
structure Rug where
  innerLength : ℝ
  innerWidth : ℝ := 2
  middleWidth : ℝ := 6
  outerWidth : ℝ := 10

/-- Calculates the areas of the three regions of the rug -/
def rugAreas (r : Rug) : Fin 3 → ℝ
  | 0 => rectangleArea ⟨r.innerLength, r.innerWidth⟩
  | 1 => rectangleArea ⟨r.innerLength + 4, r.middleWidth⟩ - rectangleArea ⟨r.innerLength, r.innerWidth⟩
  | 2 => rectangleArea ⟨r.innerLength + 8, r.outerWidth⟩ - rectangleArea ⟨r.innerLength + 4, r.middleWidth⟩

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four :
  ∀ (r : Rug), isArithmeticProgression (rugAreas r 0) (rugAreas r 1) (rugAreas r 2) →
  r.innerLength = 4 := by
  sorry

end rug_inner_length_is_four_l3807_380708


namespace c_investment_is_10500_l3807_380766

/-- Represents the investment and profit distribution in a partnership business -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates C's investment given the partnership details -/
def calculate_c_investment (p : Partnership) : ℕ :=
  p.total_profit * p.a_investment / p.a_profit_share - p.a_investment - p.b_investment

/-- Theorem stating that C's investment is 10500 given the problem conditions -/
theorem c_investment_is_10500 (p : Partnership) 
  (h1 : p.a_investment = 6300)
  (h2 : p.b_investment = 4200)
  (h3 : p.total_profit = 12500)
  (h4 : p.a_profit_share = 3750) :
  calculate_c_investment p = 10500 := by
  sorry

#eval calculate_c_investment {
  a_investment := 6300, 
  b_investment := 4200, 
  c_investment := 0,  -- This value doesn't affect the calculation
  total_profit := 12500, 
  a_profit_share := 3750
}

end c_investment_is_10500_l3807_380766


namespace trapezoid_area_coefficient_l3807_380754

-- Define the triangle
def triangle_side_1 : ℝ := 15
def triangle_side_2 : ℝ := 39
def triangle_side_3 : ℝ := 36

-- Define the area formula for the trapezoid
def trapezoid_area (γ δ ω : ℝ) : ℝ := γ * ω - δ * ω^2

-- State the theorem
theorem trapezoid_area_coefficient :
  ∃ (γ : ℝ), 
    (trapezoid_area γ (60/169) triangle_side_2 = 0) ∧
    (trapezoid_area γ (60/169) (triangle_side_2/2) = 
      (1/2) * Real.sqrt (
        (triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 *
        ((triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 - triangle_side_1) *
        ((triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 - triangle_side_2) *
        ((triangle_side_1 + triangle_side_2 + triangle_side_3) / 2 - triangle_side_3)
      )) := by
  sorry

end trapezoid_area_coefficient_l3807_380754


namespace even_periodic_function_monotonicity_l3807_380779

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def hasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def isIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem even_periodic_function_monotonicity 
  (f : ℝ → ℝ) 
  (h_even : isEven f) 
  (h_period : hasPeriod f 2) : 
  isIncreasingOn f 0 1 ↔ isDecreasingOn f 3 4 := by
  sorry

end even_periodic_function_monotonicity_l3807_380779


namespace fraction_addition_l3807_380716

theorem fraction_addition : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end fraction_addition_l3807_380716


namespace unique_b_value_l3807_380712

/-- The value of 524123 in base 81 -/
def base_81_value : ℕ := 3 + 2 * 81 + 4 * 81^2 + 1 * 81^3 + 2 * 81^4 + 5 * 81^5

/-- Theorem stating that if b is an integer between 1 and 30 (inclusive),
    and base_81_value - b is divisible by 17, then b must equal 11 -/
theorem unique_b_value (b : ℤ) (h1 : 1 ≤ b) (h2 : b ≤ 30) 
    (h3 : (base_81_value : ℤ) - b ≡ 0 [ZMOD 17]) : b = 11 := by
  sorry

end unique_b_value_l3807_380712


namespace equation_solution_system_of_equations_solution_l3807_380748

-- Problem 1
theorem equation_solution : 
  let x : ℚ := -1
  (2*x + 1) / 6 - (5*x - 1) / 8 = 7 / 12 := by sorry

-- Problem 2
theorem system_of_equations_solution :
  let x : ℚ := 4
  let y : ℚ := 3
  3*x - 2*y = 6 ∧ 2*x + 3*y = 17 := by sorry

end equation_solution_system_of_equations_solution_l3807_380748


namespace square_existence_l3807_380743

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a square
structure Square where
  side1 : Line2D
  side2 : Line2D
  side3 : Line2D
  side4 : Line2D

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if three points are collinear
def areCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

-- Theorem statement
theorem square_existence
  (A B C D : Point2D)
  (h_not_collinear : ¬(areCollinear A B C ∨ areCollinear A B D ∨ areCollinear A C D ∨ areCollinear B C D)) :
  ∃ (s : Square),
    pointOnLine A s.side1 ∧
    pointOnLine B s.side2 ∧
    pointOnLine C s.side3 ∧
    pointOnLine D s.side4 :=
sorry

end square_existence_l3807_380743


namespace divisible_by_45_sum_of_digits_l3807_380799

theorem divisible_by_45_sum_of_digits (a b : ℕ) : 
  (a < 10) →
  (b < 10) →
  (6 * 10000 + a * 1000 + 700 + 80 + b) % 45 = 0 →
  a + b = 6 := by sorry

end divisible_by_45_sum_of_digits_l3807_380799


namespace right_triangle_construction_condition_l3807_380773

/-- Given a right triangle ABC with leg AC = b and perimeter 2s, 
    prove that the construction is possible if and only if b < s -/
theorem right_triangle_construction_condition 
  (b s : ℝ) 
  (h_positive_b : 0 < b) 
  (h_positive_s : 0 < s) 
  (h_perimeter : ∃ (c : ℝ), b + c + (b^2 + c^2).sqrt = 2*s) :
  (∃ (c : ℝ), c > 0 ∧ b^2 + c^2 = ((2*s - b - c)^2)) ↔ b < s :=
by sorry

end right_triangle_construction_condition_l3807_380773


namespace storybook_pages_l3807_380752

theorem storybook_pages : (10 + 5) / (1 - 1/5 * 2) = 25 := by
  sorry

end storybook_pages_l3807_380752


namespace arithmetic_sequence_ratio_l3807_380784

/-- Given an arithmetic sequence with non-zero common difference, 
    if a_2 + a_3 = a_6, then (a_1 + a_2) / (a_3 + a_4 + a_5) = 1/3 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℚ) (d : ℚ) (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : a 2 + a 3 = a 6) : 
  (a 1 + a 2) / (a 3 + a 4 + a 5) = 1 / 3 := by
sorry

end arithmetic_sequence_ratio_l3807_380784


namespace pentagon_angle_measure_l3807_380774

theorem pentagon_angle_measure (P Q R S T : ℝ) : 
  -- Pentagon condition
  P + Q + R + S + T = 540 →
  -- Equal angles condition
  P = R ∧ P = T →
  -- Supplementary angles condition
  Q + S = 180 →
  -- Conclusion
  T = 120 := by
sorry

end pentagon_angle_measure_l3807_380774


namespace parabola_and_line_properties_l3807_380746

-- Define the parabola C: y^2 = 2px
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define point A
def point_A : ℝ × ℝ := (2, -4)

-- Define point B
def point_B : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem parabola_and_line_properties
  (p : ℝ)
  (h_p_pos : p > 0)
  (h_A_on_C : parabola p point_A.1 point_A.2) :
  -- Part 1: Equation of parabola and its directrix
  (∃ (x y : ℝ), parabola 4 x y ∧ y^2 = 8*x) ∧
  (∃ (x : ℝ), x = -2) ∧
  -- Part 2: Equations of line l
  (∃ (x y : ℝ),
    (x = 0 ∨ y = 2 ∨ x - y + 2 = 0) ∧
    (x = point_B.1 ∧ y = point_B.2) ∧
    (∃! (z : ℝ), parabola 4 x z ∧ z = y)) :=
sorry

end parabola_and_line_properties_l3807_380746


namespace C_work_duration_l3807_380707

-- Define the work rates and durations
def work_rate_A : ℚ := 1 / 30
def work_rate_B : ℚ := 1 / 30
def days_A_worked : ℕ := 10
def days_B_worked : ℕ := 10
def days_C_worked : ℕ := 10

-- Define the total work as 1 (representing 100%)
def total_work : ℚ := 1

-- Theorem to prove
theorem C_work_duration :
  let work_done_A : ℚ := work_rate_A * days_A_worked
  let work_done_B : ℚ := work_rate_B * days_B_worked
  let work_done_C : ℚ := total_work - (work_done_A + work_done_B)
  let work_rate_C : ℚ := work_done_C / days_C_worked
  (total_work / work_rate_C : ℚ) = 30 := by
  sorry

end C_work_duration_l3807_380707


namespace problem_1_l3807_380769

theorem problem_1 : (-5) + (-2) + 9 - (-8) = 10 := by
  sorry

end problem_1_l3807_380769


namespace unique_four_digit_square_l3807_380792

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits_equal (n : ℕ) : Prop :=
  (n / 1000) = ((n / 100) % 10)

def last_two_digits_equal (n : ℕ) : Prop :=
  ((n / 10) % 10) = (n % 10)

theorem unique_four_digit_square :
  ∃! n : ℕ, is_four_digit n ∧
             ∃ k : ℕ, n = k^2 ∧
             first_two_digits_equal n ∧
             last_two_digits_equal n ∧
             n = 7744 :=
by sorry

end unique_four_digit_square_l3807_380792


namespace part_one_part_two_part_three_l3807_380744

/- Define the constants -/
def total_weight : ℕ := 1000
def round_weight : ℕ := 8
def square_weight : ℕ := 18
def round_price : ℕ := 160
def square_price : ℕ := 270

/- Part 1 -/
theorem part_one (a : ℕ) : 
  round_price * a + square_price * a = 8600 → a = 20 := by sorry

/- Part 2 -/
theorem part_two (x y : ℕ) :
  round_price * x + square_price * y = 16760 ∧
  round_weight * x + square_weight * y = total_weight →
  x = 44 ∧ y = 36 := by sorry

/- Part 3 -/
theorem part_three (m n b : ℕ) :
  b > 0 →
  round_price * m + square_price * n = 16760 ∧
  round_weight * (m + b) + square_weight * n = total_weight →
  (m + b = 80 ∧ n = 20) ∨ (m + b = 116 ∧ n = 4) := by sorry

end part_one_part_two_part_three_l3807_380744


namespace salt_solution_replacement_l3807_380733

/-- Given two solutions with different salt concentrations, prove the fraction of
    the first solution replaced to achieve a specific final concentration -/
theorem salt_solution_replacement
  (initial_salt_concentration : Real)
  (second_salt_concentration : Real)
  (final_salt_concentration : Real)
  (h1 : initial_salt_concentration = 0.14)
  (h2 : second_salt_concentration = 0.22)
  (h3 : final_salt_concentration = 0.16) :
  ∃ (x : Real), 
    x = 1/4 ∧ 
    initial_salt_concentration + x * second_salt_concentration - 
      x * initial_salt_concentration = final_salt_concentration :=
by sorry

end salt_solution_replacement_l3807_380733


namespace expression_simplification_l3807_380747

theorem expression_simplification (x y z : ℝ) :
  ((x + y) - (z - y)) - ((x + z) - (y + z)) = 3 * y - z := by
  sorry

end expression_simplification_l3807_380747


namespace arithmetic_sequence_problem_l3807_380709

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 5 = 10 ∧
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 = 3) →
  a₁ = -2 ∧ d = 3 := by
  sorry

end arithmetic_sequence_problem_l3807_380709


namespace brown_mms_in_first_bag_l3807_380732

/-- The number of bags of M&M's. -/
def num_bags : ℕ := 5

/-- The number of brown M&M's in the second bag. -/
def second_bag : ℕ := 12

/-- The number of brown M&M's in the third bag. -/
def third_bag : ℕ := 8

/-- The number of brown M&M's in the fourth bag. -/
def fourth_bag : ℕ := 8

/-- The number of brown M&M's in the fifth bag. -/
def fifth_bag : ℕ := 3

/-- The average number of brown M&M's per bag. -/
def average : ℕ := 8

/-- Theorem stating the number of brown M&M's in the first bag. -/
theorem brown_mms_in_first_bag :
  ∃ (first_bag : ℕ),
    (first_bag + second_bag + third_bag + fourth_bag + fifth_bag) / num_bags = average ∧
    first_bag = 9 := by
  sorry

end brown_mms_in_first_bag_l3807_380732


namespace flower_arrangement_theorem_l3807_380761

/-- Represents a flower arrangement on a square -/
structure FlowerArrangement where
  corners : ℕ  -- number of flowers at each corner
  midpoints : ℕ  -- number of flowers at each midpoint

/-- The total number of flowers in the arrangement -/
def total_flowers (arrangement : FlowerArrangement) : ℕ :=
  4 * arrangement.corners + 4 * arrangement.midpoints

/-- The number of flowers seen on each side of the square -/
def flowers_per_side (arrangement : FlowerArrangement) : ℕ :=
  2 * arrangement.corners + arrangement.midpoints

theorem flower_arrangement_theorem :
  (∃ (arr : FlowerArrangement), 
    flowers_per_side arr = 9 ∧ 
    total_flowers arr = 36 ∧ 
    (∀ (other : FlowerArrangement), flowers_per_side other = 9 → total_flowers other ≤ 36)) ∧
  (∃ (arr : FlowerArrangement), 
    flowers_per_side arr = 12 ∧ 
    total_flowers arr = 24 ∧ 
    (∀ (other : FlowerArrangement), flowers_per_side other = 12 → total_flowers other ≥ 24)) :=
by sorry

end flower_arrangement_theorem_l3807_380761


namespace lemonade_pitchers_sum_l3807_380768

theorem lemonade_pitchers_sum : 
  let first_intermission : ℝ := 0.25
  let second_intermission : ℝ := 0.42
  let third_intermission : ℝ := 0.25
  first_intermission + second_intermission + third_intermission = 0.92 := by
sorry

end lemonade_pitchers_sum_l3807_380768


namespace maximum_value_inequality_l3807_380753

theorem maximum_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ M : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
    a^3 + b^3 + c^3 - 3*a*b*c ≥ M*(a*b^2 + b*c^2 + c*a^2 - 3*a*b*c)) →
  M ≤ 3 / Real.rpow 4 (1/3) :=
by sorry

end maximum_value_inequality_l3807_380753


namespace probability_N_16_mod_7_eq_1_l3807_380713

theorem probability_N_16_mod_7_eq_1 (N : ℕ) : 
  (∃ (k : ℕ), N = k ∧ 1 ≤ k ∧ k ≤ 2027) →
  (Nat.card {k : ℕ | 1 ≤ k ∧ k ≤ 2027 ∧ (k^16 % 7 = 1)}) / 2027 = 2 / 7 :=
by sorry

end probability_N_16_mod_7_eq_1_l3807_380713


namespace union_of_M_and_N_l3807_380726

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by sorry

end union_of_M_and_N_l3807_380726


namespace symmetric_point_coordinates_l3807_380781

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given two points, checks if they are symmetric with respect to the origin. -/
def symmetricWrtOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- The theorem stating that the point (-1, 2) is symmetric to (1, -2) with respect to the origin. -/
theorem symmetric_point_coordinates :
  let p : Point := ⟨1, -2⟩
  let q : Point := ⟨-1, 2⟩
  symmetricWrtOrigin p q := by sorry

end symmetric_point_coordinates_l3807_380781


namespace max_class_size_is_17_l3807_380706

/-- Represents a school with students and buses -/
structure School where
  total_students : ℕ
  num_buses : ℕ
  seats_per_bus : ℕ

/-- Checks if it's possible to seat all students with the given max class size -/
def can_seat_all (s : School) (max_class_size : ℕ) : Prop :=
  ∀ (class_sizes : List ℕ),
    (class_sizes.sum = s.total_students) →
    (∀ size ∈ class_sizes, size ≤ max_class_size) →
    ∃ (allocation : List (List ℕ)),
      (allocation.length ≤ s.num_buses) ∧
      (∀ bus ∈ allocation, bus.sum ≤ s.seats_per_bus) ∧
      (allocation.join.sum = s.total_students)

/-- The theorem to be proved -/
theorem max_class_size_is_17 (s : School) 
    (h1 : s.total_students = 920)
    (h2 : s.num_buses = 16)
    (h3 : s.seats_per_bus = 71) :
  (can_seat_all s 17 ∧ ¬can_seat_all s 18) := by
  sorry

end max_class_size_is_17_l3807_380706


namespace age_double_time_l3807_380717

/-- Proves that the number of years until a man's age is twice his son's age is 2,
    given that the man is currently 22 years older than his son and the son is currently 20 years old. -/
theorem age_double_time : ∃ (x : ℕ), 
  (20 + x) * 2 = (20 + 22 + x) ∧ x = 2 := by sorry

end age_double_time_l3807_380717


namespace joans_spending_l3807_380725

/-- Calculates the total spending on video games after discounts and sales tax --/
def total_spending (basketball_price : ℝ) (basketball_discount : ℝ) 
                   (racing_price : ℝ) (racing_discount : ℝ)
                   (puzzle_price : ℝ) (sales_tax : ℝ) : ℝ :=
  let basketball_discounted := basketball_price * (1 - basketball_discount)
  let racing_discounted := racing_price * (1 - racing_discount)
  let total_before_tax := basketball_discounted + racing_discounted + puzzle_price
  total_before_tax * (1 + sales_tax)

/-- Theorem stating that Joan's total spending on video games is $12.67 --/
theorem joans_spending :
  ∃ (δ : ℝ), δ > 0 ∧ δ < 0.005 ∧ 
  |total_spending 5.20 0.15 4.23 0.10 3.50 0.08 - 12.67| < δ :=
sorry

end joans_spending_l3807_380725


namespace average_children_in_families_with_children_l3807_380730

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3)
  : (total_families * total_average) / (total_families - childless_families) = 45 / 12 := by
  sorry

end average_children_in_families_with_children_l3807_380730


namespace max_product_constraint_l3807_380757

theorem max_product_constraint (a b : ℝ) (h : a + b = 5) : 
  a * b ≤ 25 / 4 ∧ (a * b = 25 / 4 ↔ a = 5 / 2 ∧ b = 5 / 2) := by
  sorry

end max_product_constraint_l3807_380757


namespace place_five_in_three_l3807_380780

/-- The number of ways to place n distinct objects into k distinct containers -/
def place_objects (n k : ℕ) : ℕ := k^n

/-- Theorem: Placing 5 distinct objects into 3 distinct containers results in 3^5 ways -/
theorem place_five_in_three : place_objects 5 3 = 3^5 := by
  sorry

end place_five_in_three_l3807_380780


namespace complex_multiplication_l3807_380734

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 - i) = 1 + 2 * i := by
  sorry

end complex_multiplication_l3807_380734


namespace final_position_of_942nd_square_l3807_380704

/-- Represents the state of a square after folding -/
structure SquareState where
  position : ℕ
  below : ℕ

/-- Calculates the new state of a square after a fold -/
def fold (state : SquareState) (stripLength : ℕ) : SquareState :=
  if state.position ≤ stripLength then
    state
  else
    { position := 2 * stripLength + 1 - state.position,
      below := stripLength - (2 * stripLength + 1 - state.position) }

/-- Performs multiple folds on a square -/
def foldMultiple (initialState : SquareState) (numFolds : ℕ) : SquareState :=
  match numFolds with
  | 0 => initialState
  | n + 1 => fold (foldMultiple initialState n) (1024 / 2^(n + 1))

theorem final_position_of_942nd_square :
  (foldMultiple { position := 942, below := 0 } 10).below = 1 := by
  sorry

end final_position_of_942nd_square_l3807_380704


namespace jerry_bacon_calories_l3807_380718

/-- Represents Jerry's breakfast -/
structure Breakfast where
  pancakes : ℕ
  pancake_calories : ℕ
  bacon_strips : ℕ
  cereal_calories : ℕ
  total_calories : ℕ

/-- Calculates the calories per strip of bacon -/
def bacon_calories_per_strip (b : Breakfast) : ℕ :=
  (b.total_calories - (b.pancakes * b.pancake_calories + b.cereal_calories)) / b.bacon_strips

/-- Theorem stating that each strip of bacon in Jerry's breakfast has 100 calories -/
theorem jerry_bacon_calories :
  let jerry_breakfast : Breakfast := {
    pancakes := 6,
    pancake_calories := 120,
    bacon_strips := 2,
    cereal_calories := 200,
    total_calories := 1120
  }
  bacon_calories_per_strip jerry_breakfast = 100 := by
  sorry

end jerry_bacon_calories_l3807_380718


namespace complex_sum_equality_l3807_380762

theorem complex_sum_equality (z : ℂ) (h : z^2 + z + 1 = 0) :
  2 * z^96 + 3 * z^97 + 4 * z^98 + 5 * z^99 + 6 * z^100 = 3 + 5 * z := by
  sorry

end complex_sum_equality_l3807_380762


namespace max_product_with_constraint_l3807_380756

theorem max_product_with_constraint (a b c : ℕ+) (h : a + 2*b + 3*c = 100) :
  a * b * c ≤ 6171 :=
sorry

end max_product_with_constraint_l3807_380756


namespace pyramid_volume_l3807_380798

/-- The volume of a pyramid with a regular hexagonal base and specific triangle areas -/
theorem pyramid_volume (base_area : ℝ) (triangle_ABG_area : ℝ) (triangle_DEG_area : ℝ)
  (h_base : base_area = 648)
  (h_ABG : triangle_ABG_area = 180)
  (h_DEG : triangle_DEG_area = 162) :
  ∃ (volume : ℝ), volume = 432 * Real.sqrt 22 := by
  sorry

#check pyramid_volume

end pyramid_volume_l3807_380798


namespace village_population_proof_l3807_380736

/-- Proves that given a 20% increase followed by a 20% decrease resulting in 9600,
    the initial population must have been 10000 -/
theorem village_population_proof (initial_population : ℝ) : 
  (initial_population * 1.2 * 0.8 = 9600) → initial_population = 10000 := by
  sorry

end village_population_proof_l3807_380736


namespace chromium_percentage_proof_l3807_380785

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_second_alloy : ℝ := 8

theorem chromium_percentage_proof (
  first_alloy_chromium_percentage : ℝ)
  (first_alloy_weight : ℝ)
  (second_alloy_weight : ℝ)
  (new_alloy_chromium_percentage : ℝ)
  (h1 : first_alloy_chromium_percentage = 15)
  (h2 : first_alloy_weight = 15)
  (h3 : second_alloy_weight = 35)
  (h4 : new_alloy_chromium_percentage = 10.1)
  : chromium_percentage_second_alloy = 8 := by
  sorry

#check chromium_percentage_proof

end chromium_percentage_proof_l3807_380785


namespace problem_solution_l3807_380764

theorem problem_solution (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -12) : 
  x^2 + 9*y^2 = 108 := by
sorry

end problem_solution_l3807_380764


namespace difference_of_squares_l3807_380740

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by sorry

end difference_of_squares_l3807_380740


namespace smallest_odd_six_digit_divisible_by_125_l3807_380700

def is_odd_digit (d : Nat) : Prop := d % 2 = 1 ∧ d < 10

def all_digits_odd (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_odd_digit d

def is_six_digit (n : Nat) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

theorem smallest_odd_six_digit_divisible_by_125 :
  ∀ n : Nat, is_six_digit n → all_digits_odd n → n % 125 = 0 →
  111375 ≤ n := by sorry

end smallest_odd_six_digit_divisible_by_125_l3807_380700


namespace chair_color_probability_l3807_380794

/-- The probability that the last two remaining chairs are of the same color -/
def same_color_probability (black_chairs brown_chairs : ℕ) : ℚ :=
  let total_chairs := black_chairs + brown_chairs
  let black_prob := (black_chairs : ℚ) / total_chairs * ((black_chairs - 1) : ℚ) / (total_chairs - 1)
  let brown_prob := (brown_chairs : ℚ) / total_chairs * ((brown_chairs - 1) : ℚ) / (total_chairs - 1)
  black_prob + brown_prob

/-- Theorem stating that the probability of the last two chairs being the same color is 43/88 -/
theorem chair_color_probability :
  same_color_probability 15 18 = 43 / 88 := by
  sorry

end chair_color_probability_l3807_380794


namespace average_rainfall_proof_l3807_380776

/-- The average rainfall for the first three days of May in a normal year -/
def average_rainfall : ℝ := 140

/-- Rainfall on the first day in cm -/
def first_day_rainfall : ℝ := 26

/-- Rainfall on the second day in cm -/
def second_day_rainfall : ℝ := 34

/-- Rainfall difference between second and third day in cm -/
def third_day_difference : ℝ := 12

/-- Difference between this year's total rainfall and average in cm -/
def rainfall_difference : ℝ := 58

theorem average_rainfall_proof :
  let third_day_rainfall := second_day_rainfall - third_day_difference
  let this_year_total := first_day_rainfall + second_day_rainfall + third_day_rainfall
  average_rainfall = this_year_total + rainfall_difference := by
  sorry

end average_rainfall_proof_l3807_380776


namespace arithmetic_expression_equality_l3807_380775

theorem arithmetic_expression_equality : 3^2 + 4 * 2 - 6 / 3 + 7 = 22 := by sorry

end arithmetic_expression_equality_l3807_380775


namespace integral_cos_quadratic_l3807_380778

theorem integral_cos_quadratic (f : ℝ → ℝ) :
  (∫ x in (0)..(2 * Real.pi), (1 - 8 * x^2) * Real.cos (4 * x)) = -2 * Real.pi :=
by sorry

end integral_cos_quadratic_l3807_380778


namespace otimes_properties_l3807_380791

def otimes (a b : ℝ) : ℝ := a * (1 - b)

theorem otimes_properties : 
  (otimes 2 (-2) = 6) ∧ 
  (¬ ∀ a b, otimes a b = otimes b a) ∧ 
  (∀ a, otimes 5 a + otimes 6 a = otimes 11 a) ∧ 
  (¬ ∀ b, otimes 3 b = 3 → b = 1) := by sorry

end otimes_properties_l3807_380791


namespace equation_solution_l3807_380765

theorem equation_solution (x : ℝ) :
  x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 1 →
  (3 * x + 2) / (x^2 + 5 * x + 6) = 3 * x / (x - 1) →
  3 * x^3 + 12 * x^2 + 19 * x + 2 = 0 :=
by sorry

end equation_solution_l3807_380765


namespace remainder_1234567_div_12_l3807_380758

theorem remainder_1234567_div_12 : 1234567 % 12 = 7 := by
  sorry

end remainder_1234567_div_12_l3807_380758


namespace gym_charges_twice_a_month_l3807_380720

/-- Represents a gym's monthly charging system -/
structure Gym where
  members : ℕ
  charge_per_payment : ℕ
  monthly_income : ℕ

/-- Calculates the number of times a gym charges its members per month -/
def charges_per_month (g : Gym) : ℕ :=
  g.monthly_income / (g.members * g.charge_per_payment)

/-- Theorem stating that for the given gym conditions, the number of charges per month is 2 -/
theorem gym_charges_twice_a_month :
  let g : Gym := { members := 300, charge_per_payment := 18, monthly_income := 10800 }
  charges_per_month g = 2 := by
  sorry

end gym_charges_twice_a_month_l3807_380720


namespace ten_machines_four_minutes_production_l3807_380790

/-- The number of bottles produced per minute by a single machine -/
def bottles_per_machine_per_minute (total_bottles : ℕ) (num_machines : ℕ) : ℕ :=
  total_bottles / num_machines

/-- The number of bottles produced per minute by a given number of machines -/
def bottles_per_minute (bottles_per_machine : ℕ) (num_machines : ℕ) : ℕ :=
  bottles_per_machine * num_machines

/-- The total number of bottles produced in a given time -/
def total_bottles (bottles_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  bottles_per_minute * minutes

/-- Theorem stating that 10 machines will produce 1800 bottles in 4 minutes -/
theorem ten_machines_four_minutes_production 
  (h : bottles_per_minute (bottles_per_machine_per_minute 270 6) 10 = 450) :
  total_bottles 450 4 = 1800 := by
  sorry

end ten_machines_four_minutes_production_l3807_380790


namespace absolute_value_and_sqrt_simplification_l3807_380714

theorem absolute_value_and_sqrt_simplification :
  |-Real.sqrt 3| + Real.sqrt 12 + Real.sqrt 3 * (Real.sqrt 3 - 3) = 3 := by
  sorry

end absolute_value_and_sqrt_simplification_l3807_380714


namespace triangle_theorem_l3807_380789

-- Define the triangle ABC
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  Triangle a b c →
  b^2 * c * Real.cos C + c^2 * b * Real.cos B = a * b^2 + a * c^2 - a^3 →
  (A = Real.pi / 3 ∧
   (b + c = 2 → ∀ a' : ℝ, Triangle a' b c → a' ≥ 1)) :=
by sorry

end triangle_theorem_l3807_380789


namespace right_isosceles_triangle_exists_l3807_380711

-- Define the set of points
def Points : Set (ℤ × ℤ) :=
  {(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)}

-- Define the color type
inductive Color
| red
| blue

-- Define what it means for three points to form a right isosceles triangle
def isRightIsosceles (p1 p2 p3 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  ((x2 - x1)^2 + (y2 - y1)^2 = (x3 - x1)^2 + (y3 - y1)^2) ∧
  ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1) = 0)

-- State the theorem
theorem right_isosceles_triangle_exists (f : ℤ × ℤ → Color) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ Points ∧ p2 ∈ Points ∧ p3 ∈ Points ∧
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
  f p1 = f p2 ∧ f p2 = f p3 ∧
  isRightIsosceles p1 p2 p3 := by
  sorry


end right_isosceles_triangle_exists_l3807_380711


namespace train_length_is_286_l3807_380742

/-- The speed of the pedestrian in meters per second -/
def pedestrian_speed : ℝ := 1

/-- The speed of the cyclist in meters per second -/
def cyclist_speed : ℝ := 3

/-- The time it takes for the train to pass the pedestrian in seconds -/
def pedestrian_passing_time : ℝ := 22

/-- The time it takes for the train to pass the cyclist in seconds -/
def cyclist_passing_time : ℝ := 26

/-- The speed of the train in meters per second -/
def train_speed : ℝ := 14

/-- The length of the train in meters -/
def train_length : ℝ := (train_speed - pedestrian_speed) * pedestrian_passing_time

theorem train_length_is_286 : train_length = 286 := by
  sorry

end train_length_is_286_l3807_380742


namespace bag_probability_l3807_380749

theorem bag_probability (n : ℕ) : 
  (6 : ℚ) / (6 + n) = 2 / 5 → n = 9 := by
sorry

end bag_probability_l3807_380749


namespace easter_egg_probability_l3807_380796

theorem easter_egg_probability : ∀ (total eggs : ℕ) (red_eggs : ℕ) (small_box : ℕ) (large_box : ℕ),
  total = 16 →
  red_eggs = 3 →
  small_box = 6 →
  large_box = 10 →
  small_box + large_box = total →
  (Nat.choose red_eggs 1 * Nat.choose (total - red_eggs) (small_box - 1) +
   Nat.choose red_eggs 2 * Nat.choose (total - red_eggs) (small_box - 2)) /
  Nat.choose total small_box = 3 / 4 := by
sorry

end easter_egg_probability_l3807_380796


namespace geometric_sequence_third_term_l3807_380722

theorem geometric_sequence_third_term :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n > 0) →  -- Sequence of positive integers
    (∃ r : ℕ, ∀ n, a (n + 1) = a n * r) →  -- Geometric sequence
    a 1 = 5 →  -- First term is 5
    a 5 = 405 →  -- Fifth term is 405
    a 3 = 45 :=  -- Third term is 45
by
  sorry

end geometric_sequence_third_term_l3807_380722


namespace pen_price_ratio_l3807_380750

theorem pen_price_ratio :
  ∀ (x y : ℕ) (b g : ℝ),
    x > 0 → y > 0 → b > 0 → g > 0 →
    (x + y) * g = 4 * (x * b + y * g) →
    (x + y) * b = (1 / 2) * (x * b + y * g) →
    g = 8 * b := by
  sorry

end pen_price_ratio_l3807_380750


namespace diagonal_length_l3807_380705

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_special_quadrilateral (q : Quadrilateral) : Prop :=
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  dist q.A q.B = 12 ∧
  dist q.B q.C = 12 ∧
  dist q.C q.D = 15 ∧
  dist q.D q.A = 15 ∧
  let angle := λ p₁ p₂ p₃ : ℝ × ℝ => Real.arccos (
    ((p₁.1 - p₂.1) * (p₃.1 - p₂.1) + (p₁.2 - p₂.2) * (p₃.2 - p₂.2)) /
    (dist p₁ p₂ * dist p₂ p₃)
  )
  angle q.A q.D q.C = 2 * Real.pi / 3

-- Theorem statement
theorem diagonal_length (q : Quadrilateral) (h : is_special_quadrilateral q) :
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  dist q.A q.C = 15 := by
  sorry

end diagonal_length_l3807_380705


namespace problem_solution_l3807_380715

theorem problem_solution (a b c : ℕ+) 
  (eq1 : a^3 + 32*b + 2*c = 2018)
  (eq2 : b^3 + 32*a + 2*c = 1115) :
  a^2 + b^2 + c^2 = 226 := by
  sorry

end problem_solution_l3807_380715


namespace max_equal_distribution_l3807_380724

theorem max_equal_distribution (bags : Nat) (eyeliners : Nat) : 
  bags = 2923 → eyeliners = 3239 → Nat.gcd bags eyeliners = 1 := by
  sorry

end max_equal_distribution_l3807_380724


namespace continuity_not_implies_differentiability_l3807_380701

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define a point in the real line
variable (x₀ : ℝ)

-- Theorem statement
theorem continuity_not_implies_differentiability :
  ∃ f : ℝ → ℝ, ∃ x₀ : ℝ, ContinuousAt f x₀ ∧ ¬DifferentiableAt ℝ f x₀ := by
  sorry

end continuity_not_implies_differentiability_l3807_380701


namespace triangle_side_length_l3807_380795

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  Real.cos A = Real.sqrt 3 / 2 →
  b < c →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  b = 2 := by
  sorry

end triangle_side_length_l3807_380795


namespace gym_income_calculation_l3807_380767

/-- Calculates the monthly income of a gym given its bi-monthly charge and number of members. -/
def gym_monthly_income (bi_monthly_charge : ℕ) (num_members : ℕ) : ℕ :=
  2 * bi_monthly_charge * num_members

/-- Proves that a gym charging $18 twice a month with 300 members makes $10,800 per month. -/
theorem gym_income_calculation :
  gym_monthly_income 18 300 = 10800 := by
  sorry

end gym_income_calculation_l3807_380767


namespace intersection_of_A_and_B_l3807_380710

def A : Set ℝ := {x : ℝ | x^2 + x = 0}
def B : Set ℝ := {x : ℝ | x^2 - x = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end intersection_of_A_and_B_l3807_380710


namespace draw_balls_count_l3807_380745

/-- The number of ways to draw 3 balls in order from a bin of 12 balls, 
    where each ball remains outside the bin after it is drawn. -/
def draw_balls : ℕ :=
  12 * 11 * 10

/-- Theorem stating that the number of ways to draw 3 balls in order 
    from a bin of 12 balls, where each ball remains outside the bin 
    after it is drawn, is equal to 1320. -/
theorem draw_balls_count : draw_balls = 1320 := by
  sorry

end draw_balls_count_l3807_380745


namespace negation_equivalence_l3807_380797

theorem negation_equivalence : 
  (¬∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ (∀ x : ℝ, -1 < x ∧ x < 2) :=
by sorry

end negation_equivalence_l3807_380797


namespace heart_ratio_equals_one_l3807_380782

-- Define the ♥ operation
def heart (n m : ℕ) : ℕ := n^3 * m^3

-- Theorem statement
theorem heart_ratio_equals_one : (heart 3 2) / (heart 2 3) = 1 := by
  sorry

end heart_ratio_equals_one_l3807_380782


namespace purple_socks_added_theorem_l3807_380731

/-- Represents the number of socks of each color -/
structure SockDrawer where
  green : Nat
  purple : Nat
  orange : Nat

/-- The initial state of the sock drawer -/
def initialDrawer : SockDrawer :=
  { green := 6, purple := 18, orange := 12 }

/-- Calculates the total number of socks in a drawer -/
def totalSocks (drawer : SockDrawer) : Nat :=
  drawer.green + drawer.purple + drawer.orange

/-- Calculates the probability of picking a purple sock -/
def purpleProbability (drawer : SockDrawer) : Rat :=
  drawer.purple / (totalSocks drawer)

/-- Adds purple socks to the drawer -/
def addPurpleSocks (drawer : SockDrawer) (n : Nat) : SockDrawer :=
  { drawer with purple := drawer.purple + n }

theorem purple_socks_added_theorem :
  ∃ n : Nat, purpleProbability (addPurpleSocks initialDrawer n) = 3/5 ∧ n = 9 := by
  sorry

end purple_socks_added_theorem_l3807_380731


namespace complex_magnitude_range_l3807_380783

theorem complex_magnitude_range (z₁ z₂ : ℂ) 
  (h₁ : (z₁ - Complex.I) * (z₂ + Complex.I) = 1)
  (h₂ : Complex.abs z₁ = Real.sqrt 2) :
  ∃ (a b : ℝ), a = 2 - Real.sqrt 2 ∧ b = 2 + Real.sqrt 2 ∧ 
  a ≤ Complex.abs z₂ ∧ Complex.abs z₂ ≤ b :=
sorry

end complex_magnitude_range_l3807_380783


namespace missy_tv_watching_l3807_380741

/-- The number of reality shows Missy watches -/
def num_reality_shows : ℕ := 5

/-- The duration of each reality show in minutes -/
def reality_show_duration : ℕ := 28

/-- The duration of the cartoon in minutes -/
def cartoon_duration : ℕ := 10

/-- The total time Missy spends watching TV in minutes -/
def total_watch_time : ℕ := 150

theorem missy_tv_watching :
  num_reality_shows * reality_show_duration + cartoon_duration = total_watch_time :=
by sorry

end missy_tv_watching_l3807_380741


namespace linear_function_value_l3807_380703

/-- Given a linear function f(x) = ax + b, if f(3) = 7 and f(5) = -1, then f(0) = 19 -/
theorem linear_function_value (a b : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x + b)
    (h_3 : f 3 = 7)
    (h_5 : f 5 = -1) : 
  f 0 = 19 := by
sorry

end linear_function_value_l3807_380703


namespace igor_sequence_uses_three_infinitely_l3807_380793

/-- Represents a sequence of natural numbers where each number is obtained
    from the previous one by adding n/p, where p is a prime divisor of n. -/
def IgorSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 1) ∧
  (∀ n, ∃ p, Nat.Prime p ∧ p ∣ a n ∧ a (n + 1) = a n + a n / p)

/-- The theorem stating that in an infinite IgorSequence,
    the prime 3 must be used as a divisor infinitely many times. -/
theorem igor_sequence_uses_three_infinitely (a : ℕ → ℕ) (h : IgorSequence a) :
  ∀ m, ∃ n > m, ∃ p, p = 3 ∧ Nat.Prime p ∧ p ∣ a n ∧ a (n + 1) = a n + a n / p :=
sorry

end igor_sequence_uses_three_infinitely_l3807_380793


namespace sum_odd_integers_less_than_100_l3807_380788

theorem sum_odd_integers_less_than_100 : 
  (Finset.filter (fun n => n % 2 = 1) (Finset.range 100)).sum id = 2500 := by
  sorry

end sum_odd_integers_less_than_100_l3807_380788


namespace store_brand_butter_price_l3807_380728

/-- The price of a single 16 oz package of store-brand butter -/
def single_package_price : ℝ := 6

/-- The price of an 8 oz package of butter -/
def eight_oz_price : ℝ := 4

/-- The normal price of a 4 oz package of butter -/
def four_oz_normal_price : ℝ := 2

/-- The discount rate for 4 oz packages -/
def discount_rate : ℝ := 0.5

/-- The lowest price for 16 oz of butter -/
def lowest_price : ℝ := 6

theorem store_brand_butter_price :
  single_package_price = lowest_price ∧
  lowest_price ≤ eight_oz_price + 2 * (four_oz_normal_price * (1 - discount_rate)) :=
by sorry

end store_brand_butter_price_l3807_380728


namespace largest_possible_b_l3807_380727

theorem largest_possible_b (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (Nat.Prime c) →
  (∀ b' : ℕ, (∃ a' c' : ℕ, 
    (a' * b' * c' = 360) ∧
    (1 < c') ∧
    (c' < b') ∧
    (b' < a') ∧
    (Nat.Prime c')) → b' ≤ b) →
  b = 12 := by
sorry

end largest_possible_b_l3807_380727


namespace product_binary1011_ternary212_eq_253_l3807_380787

/-- Converts a list of digits in a given base to its decimal representation -/
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

/-- The binary representation of 1011 -/
def binary1011 : List Nat := [1, 0, 1, 1]

/-- The base-3 representation of 212 -/
def ternary212 : List Nat := [2, 1, 2]

theorem product_binary1011_ternary212_eq_253 :
  (toDecimal binary1011 2) * (toDecimal ternary212 3) = 253 := by
  sorry

end product_binary1011_ternary212_eq_253_l3807_380787


namespace road_trip_distance_l3807_380771

theorem road_trip_distance (first_day : ℝ) (second_day : ℝ) (third_day : ℝ) : 
  first_day = 200 →
  second_day = 3/4 * first_day →
  third_day = 1/2 * (first_day + second_day) →
  first_day + second_day + third_day = 525 := by
sorry

end road_trip_distance_l3807_380771


namespace equation_solution_l3807_380751

theorem equation_solution : ∃ c : ℚ, (c - 37) / 3 = (3 * c + 7) / 8 ∧ c = -317 := by
  sorry

end equation_solution_l3807_380751


namespace factory_material_usage_extension_l3807_380760

/-- Given a factory with m tons of raw materials and an original plan to use a tons per day (a > 1),
    prove that if the factory reduces daily usage by 1 ton, it can use the materials for m / (a(a-1))
    additional days compared to the original plan. -/
theorem factory_material_usage_extension (m a : ℝ) (ha : a > 1) :
  let original_days := m / a
  let new_days := m / (a - 1)
  new_days - original_days = m / (a * (a - 1)) := by sorry

end factory_material_usage_extension_l3807_380760


namespace complex_fraction_simplification_l3807_380786

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i / (2 + i)) = ((1 : ℂ) + 2 * i) / 5 := by sorry

end complex_fraction_simplification_l3807_380786


namespace expression_equals_two_l3807_380755

theorem expression_equals_two (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (2 * x^2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 := by
  sorry

end expression_equals_two_l3807_380755


namespace largest_divisible_by_9_l3807_380721

def original_number : ℕ := 547654765476

def remove_digits (n : ℕ) (positions : List ℕ) : ℕ :=
  sorry

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem largest_divisible_by_9 :
  ∀ (positions : List ℕ),
    let result := remove_digits original_number positions
    is_divisible_by_9 result →
    result ≤ 5476547646 :=
by sorry

end largest_divisible_by_9_l3807_380721


namespace complement_of_M_l3807_380739

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - x > 0}

-- State the theorem
theorem complement_of_M : 
  Set.compl M = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end complement_of_M_l3807_380739


namespace square_plus_product_equals_square_l3807_380759

theorem square_plus_product_equals_square (x y : ℤ) :
  x^2 + x*y = y^2 ↔ x = 0 ∧ y = 0 := by sorry

end square_plus_product_equals_square_l3807_380759


namespace kevins_cards_l3807_380702

/-- Kevin's card problem -/
theorem kevins_cards (initial_cards found_cards : ℕ) 
  (h1 : initial_cards = 65)
  (h2 : found_cards = 539) :
  initial_cards + found_cards = 604 := by
  sorry

end kevins_cards_l3807_380702


namespace fifteenth_term_of_sequence_l3807_380772

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The 15th term of the specific arithmetic sequence -/
theorem fifteenth_term_of_sequence : arithmetic_sequence (-3) 4 15 = 53 := by
  sorry

end fifteenth_term_of_sequence_l3807_380772


namespace units_digit_of_2_pow_20_minus_1_l3807_380735

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_2_pow_20_minus_1 :
  unitsDigit ((2 ^ 20) - 1) = 5 := by
  sorry

end units_digit_of_2_pow_20_minus_1_l3807_380735


namespace power_mod_seventeen_l3807_380737

theorem power_mod_seventeen : 7^2023 % 17 = 16 := by
  sorry

end power_mod_seventeen_l3807_380737


namespace sphere_properties_l3807_380729

/-- For a sphere with volume 72π cubic inches, prove its surface area and diameter -/
theorem sphere_properties (V : ℝ) (h : V = 72 * Real.pi) :
  let r := (3 * V / (4 * Real.pi)) ^ (1/3)
  (4 * Real.pi * r^2 = 36 * Real.pi * 2^(2/3)) ∧
  (2 * r = 6 * 2^(1/3)) := by
sorry

end sphere_properties_l3807_380729


namespace absoluteError_2175000_absoluteError_1730000_l3807_380719

/-- Calculates the absolute error of an approximate number -/
def absoluteError (x : ℕ) : ℕ :=
  if x % 10 ≠ 0 then 1
  else if x % 100 ≠ 0 then 10
  else if x % 1000 ≠ 0 then 100
  else if x % 10000 ≠ 0 then 1000
  else 10000

/-- The absolute error of 2175000 is 1 -/
theorem absoluteError_2175000 : absoluteError 2175000 = 1 := by sorry

/-- The absolute error of 1730000 (173 * 10^4) is 10000 -/
theorem absoluteError_1730000 : absoluteError 1730000 = 10000 := by sorry

end absoluteError_2175000_absoluteError_1730000_l3807_380719


namespace rectangle_area_l3807_380738

/-- Given a rectangle with perimeter 28 cm and width 6 cm, prove its area is 48 square cm. -/
theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 28) (h2 : width = 6) :
  let length := (perimeter - 2 * width) / 2
  width * length = 48 :=
by sorry

end rectangle_area_l3807_380738


namespace tornado_distance_l3807_380723

theorem tornado_distance (car_distance lawn_chair_distance birdhouse_distance : ℝ)
  (h1 : lawn_chair_distance = 2 * car_distance)
  (h2 : birdhouse_distance = 3 * lawn_chair_distance)
  (h3 : birdhouse_distance = 1200) :
  car_distance = 200 := by
sorry

end tornado_distance_l3807_380723


namespace customers_in_other_countries_l3807_380763

theorem customers_in_other_countries 
  (total_customers : ℕ) 
  (us_customers : ℕ) 
  (h1 : total_customers = 7422) 
  (h2 : us_customers = 723) : 
  total_customers - us_customers = 6699 := by
  sorry

end customers_in_other_countries_l3807_380763


namespace red_triangles_in_colored_graph_l3807_380777

/-- A coloring of a complete graph is a function that assigns either red or blue to each edge. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- The set of vertices connected to a given vertex by red edges. -/
def RedNeighborhood (n : ℕ) (c : Coloring n) (v : Fin n) : Finset (Fin n) :=
  Finset.filter (fun u => c v u) (Finset.univ.erase v)

/-- A red triangle in a colored complete graph. -/
def RedTriangle (n : ℕ) (c : Coloring n) (v1 v2 v3 : Fin n) : Prop :=
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ c v1 v2 ∧ c v2 v3 ∧ c v1 v3

theorem red_triangles_in_colored_graph (k : ℕ) (h : k ≥ 3) :
  ∀ (c : Coloring (3*k+2)),
  (∀ v, (RedNeighborhood (3*k+2) c v).card ≥ k+2) →
  (∀ v w, ¬c v w → (RedNeighborhood (3*k+2) c v ∪ RedNeighborhood (3*k+2) c w).card ≥ 2*k+2) →
  ∃ (S : Finset (Fin (3*k+2) × Fin (3*k+2) × Fin (3*k+2))),
    S.card ≥ k+2 ∧ ∀ (t : Fin (3*k+2) × Fin (3*k+2) × Fin (3*k+2)), t ∈ S → RedTriangle (3*k+2) c t.1 t.2.1 t.2.2 :=
by sorry

end red_triangles_in_colored_graph_l3807_380777
