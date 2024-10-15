import Mathlib

namespace NUMINAMATH_CALUDE_derivative_y_l1647_164723

noncomputable def y (x : ℝ) : ℝ := (Real.cos (2 * x)) ^ ((Real.log (Real.cos (2 * x))) / 4)

theorem derivative_y (x : ℝ) (h : Real.cos (2 * x) ≠ 0) :
  deriv y x = -(Real.cos (2 * x)) ^ ((Real.log (Real.cos (2 * x))) / 4) * 
               Real.tan (2 * x) * 
               Real.log (Real.cos (2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_y_l1647_164723


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1647_164782

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1647_164782


namespace NUMINAMATH_CALUDE_bean_region_probability_l1647_164791

noncomputable def probability_bean_region : ℝ :=
  let total_area := (1 - 0) * ((Real.exp 1 + 1) - 0)
  let specific_area := ∫ x in (0)..(1), (Real.exp x + 1) - (Real.exp 1 + 1)
  specific_area / total_area

theorem bean_region_probability : probability_bean_region = 1 / (Real.exp 1 + 1) := by
  sorry

end NUMINAMATH_CALUDE_bean_region_probability_l1647_164791


namespace NUMINAMATH_CALUDE_positive_integer_divisible_by_15_with_sqrt_between_33_and_33_5_l1647_164733

theorem positive_integer_divisible_by_15_with_sqrt_between_33_and_33_5 :
  ∃ n : ℕ+, 
    (∃ k : ℕ, n = 15 * k) ∧ 
    (33 * 33 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) < (33.5 * 33.5) ∧
    (n = 1095 ∨ n = 1110) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_divisible_by_15_with_sqrt_between_33_and_33_5_l1647_164733


namespace NUMINAMATH_CALUDE_student_count_l1647_164750

theorem student_count (avg_age_students : ℝ) (teacher_age : ℕ) (new_avg_age : ℝ)
  (h1 : avg_age_students = 14)
  (h2 : teacher_age = 65)
  (h3 : new_avg_age = 15) :
  ∃ n : ℕ, n * avg_age_students + teacher_age = (n + 1) * new_avg_age ∧ n = 50 :=
by sorry

end NUMINAMATH_CALUDE_student_count_l1647_164750


namespace NUMINAMATH_CALUDE_sand_weight_l1647_164721

/-- Given the total weight of materials and the weight of gravel, 
    calculate the weight of sand -/
theorem sand_weight (total_weight gravel_weight : ℝ) 
  (h1 : total_weight = 14.02)
  (h2 : gravel_weight = 5.91) : 
  total_weight - gravel_weight = 8.11 := by
  sorry

end NUMINAMATH_CALUDE_sand_weight_l1647_164721


namespace NUMINAMATH_CALUDE_solution_difference_l1647_164777

theorem solution_difference (r s : ℝ) : 
  ((6 * r - 18) / (r^2 - 4*r - 21) = r - 3) →
  ((6 * s - 18) / (s^2 - 4*s - 21) = s - 3) →
  r ≠ s →
  r > s →
  r - s = 4 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1647_164777


namespace NUMINAMATH_CALUDE_lee_cookies_l1647_164776

/-- Given that Lee can make 24 cookies with 4 cups of flour, 
    this function calculates how many cookies he can make with any amount of flour. -/
def cookies_from_flour (flour : ℚ) : ℚ :=
  (24 / 4) * flour

/-- Theorem stating that Lee can make 36 cookies with 6 cups of flour. -/
theorem lee_cookies : cookies_from_flour 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l1647_164776


namespace NUMINAMATH_CALUDE_f_is_odd_and_piecewise_l1647_164751

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f defined piecewise -/
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x * (x + 2) else -x^2 + 2*x

theorem f_is_odd_and_piecewise :
  OddFunction f ∧ (∀ x < 0, f x = x * (x + 2)) → ∀ x > 0, f x = -x^2 + 2*x := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_and_piecewise_l1647_164751


namespace NUMINAMATH_CALUDE_new_men_average_age_l1647_164746

/-- Given a group of 12 men, where replacing two men aged 21 and 23 with two new men
    increases the average age by 1 year, prove that the average age of the two new men is 28 years. -/
theorem new_men_average_age
  (n : ℕ) -- number of men
  (old_age1 old_age2 : ℕ) -- ages of the two replaced men
  (avg_increase : ℚ) -- increase in average age
  (h1 : n = 12)
  (h2 : old_age1 = 21)
  (h3 : old_age2 = 23)
  (h4 : avg_increase = 1) :
  (old_age1 + old_age2 + n * avg_increase) / 2 = 28 :=
by sorry

end NUMINAMATH_CALUDE_new_men_average_age_l1647_164746


namespace NUMINAMATH_CALUDE_base8_4532_equals_2394_l1647_164783

-- Define a function to convert a base 8 number to base 10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- State the theorem
theorem base8_4532_equals_2394 :
  base8ToBase10 [2, 3, 5, 4] = 2394 := by
  sorry

end NUMINAMATH_CALUDE_base8_4532_equals_2394_l1647_164783


namespace NUMINAMATH_CALUDE_polynomial_coefficient_identity_l1647_164705

theorem polynomial_coefficient_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_identity_l1647_164705


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l1647_164796

theorem supplementary_angles_ratio (angle1 angle2 : ℝ) : 
  angle1 + angle2 = 180 →  -- angles are supplementary
  angle1 = 4 * angle2 →    -- angles are in ratio 4:1
  angle2 = 36 :=           -- smaller angle is 36°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l1647_164796


namespace NUMINAMATH_CALUDE_seller_loss_is_30_l1647_164701

/-- Represents the transaction between a seller and a buyer -/
structure Transaction where
  goods_value : ℕ
  payment : ℕ
  counterfeit : Bool

/-- Calculates the seller's loss given a transaction -/
def seller_loss (t : Transaction) : ℕ :=
  if t.counterfeit then
    t.payment + (t.payment - t.goods_value)
  else
    0

/-- Theorem stating that the seller's loss is 30 rubles given the specific transaction -/
theorem seller_loss_is_30 (t : Transaction) 
  (h1 : t.goods_value = 10)
  (h2 : t.payment = 25)
  (h3 : t.counterfeit = true) : 
  seller_loss t = 30 := by
  sorry

#eval seller_loss { goods_value := 10, payment := 25, counterfeit := true }

end NUMINAMATH_CALUDE_seller_loss_is_30_l1647_164701


namespace NUMINAMATH_CALUDE_intersection_characterization_l1647_164726

-- Define the sets M and N
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define the intersection of N and the complement of M
def intersection : Set ℝ := N ∩ (Set.univ \ M)

-- State the theorem
theorem intersection_characterization : intersection = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_characterization_l1647_164726


namespace NUMINAMATH_CALUDE_max_planes_15_points_l1647_164745

/-- The maximum number of planes determined by 15 points in space, where no four points are coplanar -/
def max_planes (n : ℕ) : ℕ :=
  Nat.choose n 3

/-- Theorem stating that the maximum number of planes determined by 15 points in space, 
    where no four points are coplanar, is equal to 455 -/
theorem max_planes_15_points : max_planes 15 = 455 := by
  sorry

end NUMINAMATH_CALUDE_max_planes_15_points_l1647_164745


namespace NUMINAMATH_CALUDE_dans_remaining_limes_l1647_164795

/-- Given that Dan initially had 9 limes and gave away 4 limes, prove that he now has 5 limes. -/
theorem dans_remaining_limes (initial_limes : ℕ) (given_away : ℕ) (h1 : initial_limes = 9) (h2 : given_away = 4) :
  initial_limes - given_away = 5 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_limes_l1647_164795


namespace NUMINAMATH_CALUDE_max_value_xy_8x_y_l1647_164731

theorem max_value_xy_8x_y (x y : ℝ) (h : x^2 + y^2 = 20) :
  ∃ (M : ℝ), M = 42 ∧ xy + 8*x + y ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 20 ∧ x₀*y₀ + 8*x₀ + y₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_xy_8x_y_l1647_164731


namespace NUMINAMATH_CALUDE_line_points_equation_l1647_164734

/-- Given a line and two points on it, prove an equation relating to the x-coordinate of the first point -/
theorem line_points_equation (m n : ℝ) : 
  (∀ x y, x - 5/2 * y + 1 = 0 → 
    ((x = m ∧ y = n) ∨ (x = m + 1/2 ∧ y = n + 1)) → 
      m + 1 = m - 3) := by
  sorry

end NUMINAMATH_CALUDE_line_points_equation_l1647_164734


namespace NUMINAMATH_CALUDE_heather_block_distribution_l1647_164799

/-- Given an initial number of blocks, the number of blocks shared, and the number of friends,
    calculate the number of blocks each friend receives when distributing the remaining blocks equally. -/
def blocks_per_friend (initial_blocks : ℕ) (shared_blocks : ℕ) (num_friends : ℕ) : ℕ :=
  (initial_blocks - shared_blocks) / num_friends

/-- Theorem stating that given 258 initial blocks, after sharing 129 blocks and
    distributing the remainder equally among 6 friends, each friend receives 21 blocks. -/
theorem heather_block_distribution :
  blocks_per_friend 258 129 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_heather_block_distribution_l1647_164799


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1647_164755

theorem tan_alpha_value (α : Real) (h_obtuse : π / 2 < α ∧ α < π) 
  (h_eq : (Real.sin α - 3 * Real.cos α) / (Real.cos α - Real.sin α) = Real.tan (2 * α)) :
  Real.tan α = 2 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1647_164755


namespace NUMINAMATH_CALUDE_part_one_part_two_l1647_164748

noncomputable section

-- Define the function f
def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k + 1) * a^(-x)

-- Define the function g
def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * m * f a 0 x

-- Theorem for part (1)
theorem part_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ k : ℝ, ∀ x : ℝ, f a k x = -f a k (-x)) → k = 0 :=
sorry

-- Theorem for part (2)
theorem part_two (a m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a 0 1 = 3/2) 
  (h4 : ∀ x : ℝ, x ≥ 0 → g a m x ≥ -6) 
  (h5 : ∃ x : ℝ, x ≥ 0 ∧ g a m x = -6) :
  m = 2 * Real.sqrt 2 ∨ m = -2 * Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_part_one_part_two_l1647_164748


namespace NUMINAMATH_CALUDE_tax_free_amount_correct_l1647_164788

/-- The tax-free amount for goods purchased in country B -/
def tax_free_amount : ℝ := 600

/-- The total value of goods purchased -/
def total_value : ℝ := 1720

/-- The tax rate applied to the excess amount -/
def tax_rate : ℝ := 0.07

/-- The amount of tax paid -/
def tax_paid : ℝ := 78.4

/-- Theorem stating that the tax-free amount satisfies the given conditions -/
theorem tax_free_amount_correct : 
  tax_rate * (total_value - tax_free_amount) = tax_paid := by
  sorry

end NUMINAMATH_CALUDE_tax_free_amount_correct_l1647_164788


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1647_164709

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 3*y = 1) :
  (1/x + 1/(3*y)) ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1647_164709


namespace NUMINAMATH_CALUDE_special_polyhedron_interior_segments_l1647_164739

/-- A convex polyhedron with specific face composition -/
structure SpecialPolyhedron where
  /-- The polyhedron is convex -/
  is_convex : Bool
  /-- Number of square faces -/
  num_square_faces : Nat
  /-- Number of regular hexagonal faces -/
  num_hexagonal_faces : Nat
  /-- Number of regular octagonal faces -/
  num_octagonal_faces : Nat
  /-- Property that exactly one square, one hexagon, and one octagon meet at each vertex -/
  vertex_property : Bool

/-- Calculate the number of interior segments in the special polyhedron -/
def interior_segments (p : SpecialPolyhedron) : Nat :=
  sorry

/-- Theorem stating the number of interior segments in the special polyhedron -/
theorem special_polyhedron_interior_segments 
  (p : SpecialPolyhedron) 
  (h1 : p.is_convex = true)
  (h2 : p.num_square_faces = 12)
  (h3 : p.num_hexagonal_faces = 8)
  (h4 : p.num_octagonal_faces = 6)
  (h5 : p.vertex_property = true) :
  interior_segments p = 840 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_interior_segments_l1647_164739


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1647_164719

/-- The shortest altitude of a right triangle with legs 9 and 12 is 7.2 -/
theorem shortest_altitude_right_triangle :
  let a : ℝ := 9
  let b : ℝ := 12
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let area : ℝ := (1/2) * a * b
  let h : ℝ := (2 * area) / c
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1647_164719


namespace NUMINAMATH_CALUDE_B_subset_A_l1647_164730

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x^2)}

-- Define set B
def B : Set ℝ := {x | ∃ m ∈ A, x = m^2}

-- Theorem statement
theorem B_subset_A : B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_B_subset_A_l1647_164730


namespace NUMINAMATH_CALUDE_book_has_2000_pages_l1647_164798

/-- The number of pages Juan reads per hour -/
def pages_per_hour : ℕ := 250

/-- The time it takes Juan to grab lunch (in hours) -/
def lunch_time : ℕ := 4

/-- The time it takes Juan to read the book (in hours) -/
def reading_time : ℕ := 2 * lunch_time

/-- The total number of pages in the book -/
def book_pages : ℕ := pages_per_hour * reading_time

theorem book_has_2000_pages : book_pages = 2000 := by
  sorry

end NUMINAMATH_CALUDE_book_has_2000_pages_l1647_164798


namespace NUMINAMATH_CALUDE_minimize_sum_of_number_and_square_l1647_164758

/-- The function representing the sum of a number and its square -/
def f (x : ℝ) : ℝ := x + x^2

/-- The theorem stating that -1/2 minimizes the function f -/
theorem minimize_sum_of_number_and_square :
  ∀ x : ℝ, f (-1/2) ≤ f x :=
by
  sorry

end NUMINAMATH_CALUDE_minimize_sum_of_number_and_square_l1647_164758


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1647_164729

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) - 3 * Complex.I * z = (3 : ℂ) + 5 * Complex.I * z ∧ z = Complex.I / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1647_164729


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1647_164754

theorem max_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n < 0) (h : 1/m + 1/n = 1) :
  ∃ (x : ℝ), ∀ (m' n' : ℝ), m' > 0 → n' < 0 → 1/m' + 1/n' = 1 → 4*m' + n' ≤ x ∧ 4*m + n = x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1647_164754


namespace NUMINAMATH_CALUDE_shoe_production_facts_l1647_164760

/-- The daily production cost function -/
def C (n : ℕ) : ℝ := 4000 + 50 * n

/-- The selling price per pair of shoes -/
def sellingPrice : ℝ := 90

/-- All produced shoes are sold out -/
axiom all_sold : ∀ n : ℕ, n > 0 → ∃ revenue : ℝ, revenue = sellingPrice * n

/-- The profit function -/
def P (n : ℕ) : ℝ := sellingPrice * n - C n

theorem shoe_production_facts :
  (C 1000 = 54000) ∧
  (∃ n : ℕ, C n = 48000 ∧ n = 880) ∧
  (∀ n : ℕ, P n = 40 * n - 4000) ∧
  (∃ min_n : ℕ, min_n = 100 ∧ ∀ n : ℕ, n ≥ min_n → P n ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_shoe_production_facts_l1647_164760


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1647_164792

theorem polygon_diagonals (n : ℕ+) : 
  (∃ n, n * (n - 3) / 2 = 2 ∨ n * (n - 3) / 2 = 54) ∧ 
  (∀ n, n * (n - 3) / 2 ≠ 21 ∧ n * (n - 3) / 2 ≠ 32 ∧ n * (n - 3) / 2 ≠ 63) :=
by sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1647_164792


namespace NUMINAMATH_CALUDE_total_balls_in_box_l1647_164781

theorem total_balls_in_box (yellow_balls : ℕ) (prob_yellow : ℚ) (total_balls : ℕ) : 
  yellow_balls = 6 → 
  prob_yellow = 1 / 9 → 
  prob_yellow = yellow_balls / total_balls → 
  total_balls = 54 := by sorry

end NUMINAMATH_CALUDE_total_balls_in_box_l1647_164781


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1647_164790

-- Define the sample space for two coin tosses
inductive CoinToss
  | HH  -- Two heads
  | HT  -- Head then tail
  | TH  -- Tail then head
  | TT  -- Two tails

-- Define the event "At least one head"
def atLeastOneHead (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HH ∨ outcome = CoinToss.HT ∨ outcome = CoinToss.TH

-- Define the event "Both tosses are tails"
def bothTails (outcome : CoinToss) : Prop :=
  outcome = CoinToss.TT

-- Theorem stating that "Both tosses are tails" is mutually exclusive to "At least one head"
theorem mutually_exclusive_events :
  ∀ (outcome : CoinToss), ¬(atLeastOneHead outcome ∧ bothTails outcome) :=
by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l1647_164790


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1647_164742

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0 ∧ (∀ y : ℝ, y^2 + y + m = 0 → y = x)) → m = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1647_164742


namespace NUMINAMATH_CALUDE_algebraic_identities_l1647_164735

theorem algebraic_identities :
  (∀ (a : ℝ), a ≠ 0 → 2 * a^5 + a^7 / a^2 = 3 * a^5) ∧
  (∀ (x y : ℝ), (x + y) * (x - y) + x * (2 * y - x) = 2 * x * y - y^2) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_identities_l1647_164735


namespace NUMINAMATH_CALUDE_log_equation_solution_l1647_164710

theorem log_equation_solution (b x : ℝ) 
  (hb_pos : b > 0) 
  (hb_neq_one : b ≠ 1) 
  (hx_neq_one : x ≠ 1) 
  (h_eq : (Real.log x) / (Real.log (b^3)) + (Real.log b) / (Real.log (x^3)) + (Real.log x) / (Real.log b) = 2) : 
  x = b^((6 - 2 * Real.sqrt 5) / 8) :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1647_164710


namespace NUMINAMATH_CALUDE_fast_clock_next_correct_time_l1647_164756

/-- Represents a clock that gains time uniformly --/
structure FastClock where
  /-- The rate at which the clock gains time, in minutes per day --/
  gain_rate : ℝ
  /-- The gain rate is positive but less than 60 minutes per day --/
  gain_rate_bounds : 0 < gain_rate ∧ gain_rate < 60

/-- Represents a specific date and time --/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Function to check if two DateTimes are equal --/
def DateTime.eq (dt1 dt2 : DateTime) : Prop :=
  dt1.year = dt2.year ∧ 
  dt1.month = dt2.month ∧ 
  dt1.day = dt2.day ∧ 
  dt1.hour = dt2.hour ∧ 
  dt1.minute = dt2.minute

/-- Function to calculate when the clock will next show the correct time --/
def next_correct_time (c : FastClock) (start : DateTime) (overlap : DateTime) : DateTime :=
  sorry

/-- Theorem stating when the clock will next show the correct time --/
theorem fast_clock_next_correct_time (c : FastClock) :
  let start := DateTime.mk 1982 1 1 0 0
  let overlap := DateTime.mk 1982 1 1 13 5
  let result := DateTime.mk 1984 5 13 12 0
  DateTime.eq (next_correct_time c start overlap) result := by
  sorry

end NUMINAMATH_CALUDE_fast_clock_next_correct_time_l1647_164756


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1647_164793

/-- Represents the relationship between y, x, and z -/
def relation (k : ℝ) (x y z : ℝ) : Prop :=
  7 * y = (k * z) / (2 * x)^2

theorem inverse_variation_problem (k : ℝ) :
  relation k 1 20 5 →
  relation k 8 0.625 10 :=
by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l1647_164793


namespace NUMINAMATH_CALUDE_total_clothing_pieces_l1647_164732

theorem total_clothing_pieces (shirts trousers : ℕ) 
  (h1 : shirts = 589) 
  (h2 : trousers = 345) : 
  shirts + trousers = 934 := by
  sorry

end NUMINAMATH_CALUDE_total_clothing_pieces_l1647_164732


namespace NUMINAMATH_CALUDE_kelsey_ekon_difference_l1647_164780

/-- The number of videos watched by three friends. -/
def total_videos : ℕ := 411

/-- The number of videos watched by Kelsey. -/
def kelsey_videos : ℕ := 160

/-- The number of videos watched by Uma. -/
def uma_videos : ℕ := (total_videos - kelsey_videos + 17) / 2

/-- The number of videos watched by Ekon. -/
def ekon_videos : ℕ := uma_videos - 17

/-- Theorem stating the difference in videos watched between Kelsey and Ekon. -/
theorem kelsey_ekon_difference :
  kelsey_videos - ekon_videos = 43 :=
by sorry

end NUMINAMATH_CALUDE_kelsey_ekon_difference_l1647_164780


namespace NUMINAMATH_CALUDE_log_equation_range_l1647_164767

theorem log_equation_range (a : ℝ) :
  (∃ y : ℝ, y = Real.log (5 - a) / Real.log (a - 2)) ↔ (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) :=
sorry

end NUMINAMATH_CALUDE_log_equation_range_l1647_164767


namespace NUMINAMATH_CALUDE_max_peak_consumption_l1647_164720

theorem max_peak_consumption (original_price peak_price off_peak_price total_consumption : ℝ)
  (h1 : original_price = 0.52)
  (h2 : peak_price = 0.55)
  (h3 : off_peak_price = 0.35)
  (h4 : total_consumption = 200)
  (h5 : ∀ x : ℝ, x ≥ 0 ∧ x ≤ total_consumption →
    (x * peak_price + (total_consumption - x) * off_peak_price) ≤ 0.9 * (total_consumption * original_price)) :
  ∃ max_peak : ℝ, max_peak = 118 ∧
    ∀ y : ℝ, y > max_peak →
      (y * peak_price + (total_consumption - y) * off_peak_price) > 0.9 * (total_consumption * original_price) :=
by sorry

end NUMINAMATH_CALUDE_max_peak_consumption_l1647_164720


namespace NUMINAMATH_CALUDE_cos_difference_formula_l1647_164764

theorem cos_difference_formula (a b : ℝ) 
  (h1 : Real.sin a + Real.sin b = 1)
  (h2 : Real.cos a + Real.cos b = 3/2) : 
  Real.cos (a - b) = 5/8 := by sorry

end NUMINAMATH_CALUDE_cos_difference_formula_l1647_164764


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1647_164759

/-- The hyperbola and parabola equations -/
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

def parabola (x y : ℝ) : Prop := y^2 = 8 * Real.sqrt 2 * x

/-- The right focus of the hyperbola coincides with the focus of the parabola -/
axiom focus_coincide : ∃ (x₀ y₀ : ℝ), 
  (x₀ = 2 * Real.sqrt 2 ∧ y₀ = 0) ∧
  (∀ x y b, hyperbola x y b → (x - x₀)^2 + y^2 = (2 * Real.sqrt 2)^2)

/-- The theorem stating that the asymptotes of the hyperbola are y = ±x -/
theorem hyperbola_asymptotes : 
  ∃ b, ∀ x y, hyperbola x y b → (y = x ∨ y = -x) ∨ (x^2 > y^2) := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1647_164759


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1647_164797

theorem quadratic_equation_solution : 
  let x₁ : ℝ := (1 + Real.sqrt 17) / 4
  let x₂ : ℝ := (1 - Real.sqrt 17) / 4
  ∀ x : ℝ, 2 * x^2 - x = 2 ↔ (x = x₁ ∨ x = x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1647_164797


namespace NUMINAMATH_CALUDE_correct_mass_units_l1647_164771

-- Define the mass units
inductive MassUnit
| Kilogram
| Gram

-- Define a structure to represent a mass measurement
structure Mass where
  value : ℝ
  unit : MassUnit

-- Define Xiaogang's weight
def xiaogang_weight : Mass := { value := 25, unit := MassUnit.Kilogram }

-- Define chalk's weight
def chalk_weight : Mass := { value := 15, unit := MassUnit.Gram }

-- Theorem to prove the correct units for Xiaogang and chalk
theorem correct_mass_units :
  xiaogang_weight.unit = MassUnit.Kilogram ∧
  chalk_weight.unit = MassUnit.Gram :=
by sorry

end NUMINAMATH_CALUDE_correct_mass_units_l1647_164771


namespace NUMINAMATH_CALUDE_future_age_calculation_l1647_164769

theorem future_age_calculation (nora_current_age terry_current_age : ℕ) 
  (h1 : nora_current_age = 10)
  (h2 : terry_current_age = 30) :
  ∃ (years_future : ℕ), terry_current_age + years_future = 4 * nora_current_age ∧ years_future = 10 :=
by sorry

end NUMINAMATH_CALUDE_future_age_calculation_l1647_164769


namespace NUMINAMATH_CALUDE_matchsticks_100th_stage_l1647_164773

/-- Represents the number of matchsticks in a stage of the pattern -/
def matchsticks (n : ℕ) : ℕ := 4 + (n - 1) * 4

/-- Proves that the 100th stage of the pattern contains 400 matchsticks -/
theorem matchsticks_100th_stage : matchsticks 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_matchsticks_100th_stage_l1647_164773


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l1647_164789

-- Define the set of integers
def U : Set Int := Set.univ

-- Define set A
def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}

-- Define set B
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}

-- Define the set of integers divisible by 3
def DivisibleBy3 : Set Int := {x | ∃ k : Int, x = 3 * k}

-- Theorem statement
theorem complement_of_A_union_B (x : Int) : 
  x ∈ (U \ (A ∪ B)) ↔ x ∈ DivisibleBy3 :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l1647_164789


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1647_164752

theorem cube_surface_area_equal_volume (a b c : ℝ) (h1 : a = 12) (h2 : b = 4) (h3 : c = 18) :
  let prism_volume := a * b * c
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 864 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l1647_164752


namespace NUMINAMATH_CALUDE_right_triangle_altitude_ratio_l1647_164772

/-- 
Given a right triangle ABC with legs a and b (a ≤ b) and hypotenuse c,
where the triangle formed by its altitudes is also a right triangle,
prove that the ratio of the shorter leg to the longer leg is √((√5 - 1) / 2).
-/
theorem right_triangle_altitude_ratio (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≤ b) :
  a^2 + b^2 = c^2 →
  a^2 + (a^2 * b^2) / (a^2 + b^2) = b^2 →
  a / b = Real.sqrt ((Real.sqrt 5 - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_ratio_l1647_164772


namespace NUMINAMATH_CALUDE_max_value_expression_l1647_164774

theorem max_value_expression (x y : ℝ) : 
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1647_164774


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1647_164753

/-- The area of a circle with diameter 10 centimeters is 25π square centimeters. -/
theorem circle_area_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1647_164753


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1647_164714

theorem complex_modulus_problem (z : ℂ) : z = (1 + Complex.I) / (1 - Complex.I) + 2 * Complex.I → Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1647_164714


namespace NUMINAMATH_CALUDE_prop_a_false_prop_b_true_prop_c_true_prop_d_true_propositions_bcd_true_a_false_l1647_164744

-- Proposition A (false)
theorem prop_a_false : ¬ (∀ a b : ℝ, a > b → 1 / b > 1 / a) := by sorry

-- Proposition B (true)
theorem prop_b_true : ∀ a b : ℝ, a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2 := by sorry

-- Proposition C (true)
theorem prop_c_true : ∀ a b c : ℝ, c ≠ 0 → (a*c^2 > b*c^2 → a > b) := by sorry

-- Proposition D (true)
theorem prop_d_true : ∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → 1 / (a - c) < 1 / (b - d) := by sorry

-- Combined theorem
theorem propositions_bcd_true_a_false : 
  (¬ (∀ a b : ℝ, a > b → 1 / b > 1 / a)) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ a b c : ℝ, c ≠ 0 → (a*c^2 > b*c^2 → a > b)) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → 1 / (a - c) < 1 / (b - d)) := by
  exact ⟨prop_a_false, prop_b_true, prop_c_true, prop_d_true⟩

end NUMINAMATH_CALUDE_prop_a_false_prop_b_true_prop_c_true_prop_d_true_propositions_bcd_true_a_false_l1647_164744


namespace NUMINAMATH_CALUDE_age_difference_l1647_164778

theorem age_difference (alvin_age simon_age : ℕ) (h1 : alvin_age = 30) (h2 : simon_age = 10) :
  alvin_age / 2 - simon_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1647_164778


namespace NUMINAMATH_CALUDE_bakery_theft_l1647_164702

/-- The number of breads remaining after a thief takes their share -/
def breads_after_thief (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => (breads_after_thief initial n - 1) / 2

/-- The proposition that given 5 thieves and 3 breads remaining at the end, 
    the initial number of breads was 127 -/
theorem bakery_theft (initial : ℕ) :
  breads_after_thief initial 5 = 3 → initial = 127 := by
  sorry

#check bakery_theft

end NUMINAMATH_CALUDE_bakery_theft_l1647_164702


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1647_164775

theorem cricketer_average_score (score1 score2 : ℝ) (matches1 matches2 : ℕ) 
  (h1 : score1 = 20)
  (h2 : score2 = 30)
  (h3 : matches1 = 2)
  (h4 : matches2 = 3) :
  let total_matches := matches1 + matches2
  let total_score := score1 * matches1 + score2 * matches2
  total_score / total_matches = 26 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l1647_164775


namespace NUMINAMATH_CALUDE_smallest_integer_below_sqrt5_plus_sqrt3_to_6th_l1647_164706

theorem smallest_integer_below_sqrt5_plus_sqrt3_to_6th :
  ∃ n : ℤ, n = 3322 ∧ n < (Real.sqrt 5 + Real.sqrt 3)^6 ∧ ∀ m : ℤ, m < (Real.sqrt 5 + Real.sqrt 3)^6 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_below_sqrt5_plus_sqrt3_to_6th_l1647_164706


namespace NUMINAMATH_CALUDE_expression_evaluation_l1647_164768

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 8) + 2 = -x^4 + 3*x^3 - 5*x^2 + 8*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1647_164768


namespace NUMINAMATH_CALUDE_monomial_sum_l1647_164784

/-- Given two monomials of the same type, prove their sum -/
theorem monomial_sum (m n : ℕ) : 
  (2 : ℤ) * X^m * Y^3 + (-5 : ℤ) * X^1 * Y^(n+1) = (-3 : ℤ) * X^1 * Y^3 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_l1647_164784


namespace NUMINAMATH_CALUDE_board_cut_theorem_l1647_164757

theorem board_cut_theorem (total_length shorter_length longer_length : ℝ) :
  total_length = 20 ∧
  total_length = shorter_length + longer_length ∧
  2 * shorter_length = longer_length + 4 →
  shorter_length = 8 := by
sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l1647_164757


namespace NUMINAMATH_CALUDE_pure_imaginary_equation_l1647_164766

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is nonzero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_equation (z : ℂ) (a : ℝ) 
  (h1 : IsPureImaginary z) 
  (h2 : (1 + Complex.I) * z = 1 - a * Complex.I) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_equation_l1647_164766


namespace NUMINAMATH_CALUDE_pencils_per_row_l1647_164737

/-- Given 6 pencils placed equally into 2 rows, prove that there are 3 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  total_pencils = 6 → num_rows = 2 → total_pencils = num_rows * pencils_per_row → pencils_per_row = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l1647_164737


namespace NUMINAMATH_CALUDE_equation_solution_l1647_164787

theorem equation_solution :
  let f (x : ℂ) := (x^2 + 4*x + 20) / (x^2 - 7*x + 12)
  let g (x : ℂ) := (x - 3) / (x - 1)
  ∀ x : ℂ, f x = g x ↔ x = (17 + Complex.I * Real.sqrt 543) / 26 ∨ x = (17 - Complex.I * Real.sqrt 543) / 26 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1647_164787


namespace NUMINAMATH_CALUDE_age_puzzle_l1647_164715

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 50) (h2 : 5 * (A + x) - 5 * (A - 5) = A) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l1647_164715


namespace NUMINAMATH_CALUDE_topsoil_cost_l1647_164738

/-- The cost of topsoil in euros per cubic meter -/
def cost_per_cubic_meter : ℝ := 12

/-- The volume of topsoil to be purchased in cubic meters -/
def volume : ℝ := 3

/-- The total cost of purchasing the topsoil -/
def total_cost : ℝ := cost_per_cubic_meter * volume

/-- Theorem stating that the total cost of purchasing 3 cubic meters of topsoil is 36 euros -/
theorem topsoil_cost : total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l1647_164738


namespace NUMINAMATH_CALUDE_bikes_added_per_week_l1647_164741

/-- 
Proves that the number of bikes added per week is 3, given the initial stock,
bikes sold in a month, stock after one month, and the number of weeks in a month.
-/
theorem bikes_added_per_week 
  (initial_stock : ℕ) 
  (bikes_sold : ℕ) 
  (final_stock : ℕ) 
  (weeks_in_month : ℕ) 
  (h1 : initial_stock = 51)
  (h2 : bikes_sold = 18)
  (h3 : final_stock = 45)
  (h4 : weeks_in_month = 4)
  : (final_stock - (initial_stock - bikes_sold)) / weeks_in_month = 3 := by
  sorry

end NUMINAMATH_CALUDE_bikes_added_per_week_l1647_164741


namespace NUMINAMATH_CALUDE_floor_sum_example_l1647_164770

theorem floor_sum_example : ⌊(23.8 : ℝ)⌋ + ⌊(-23.8 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l1647_164770


namespace NUMINAMATH_CALUDE_equation_solution_exists_l1647_164736

theorem equation_solution_exists : ∃ x : ℝ, 85 * x^2 + ((20 - 7) * 4)^3 / 2 - 15 * 7 = 75000 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l1647_164736


namespace NUMINAMATH_CALUDE_common_material_choices_eq_120_l1647_164765

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways two students can choose 2 materials each from 6 materials,
    such that they have exactly 1 material in common -/
def commonMaterialChoices : ℕ :=
  choose 6 1 * choose 5 2

theorem common_material_choices_eq_120 : commonMaterialChoices = 120 := by
  sorry


end NUMINAMATH_CALUDE_common_material_choices_eq_120_l1647_164765


namespace NUMINAMATH_CALUDE_farthest_point_l1647_164740

def points : List (ℝ × ℝ) := [(0, 7), (2, 3), (-4, 1), (5, -5), (7, 0)]

def distance_squared (p : ℝ × ℝ) : ℝ :=
  p.1 ^ 2 + p.2 ^ 2

theorem farthest_point :
  ∀ p ∈ points, distance_squared (5, -5) ≥ distance_squared p :=
by sorry

end NUMINAMATH_CALUDE_farthest_point_l1647_164740


namespace NUMINAMATH_CALUDE_meeting_arrangements_l1647_164749

/-- Represents the number of schools -/
def num_schools : ℕ := 4

/-- Represents the number of members per school -/
def members_per_school : ℕ := 6

/-- Represents the total number of members -/
def total_members : ℕ := num_schools * members_per_school

/-- Represents the number of representatives from the host school -/
def host_representatives : ℕ := 1

/-- Represents the number of non-host schools that send representatives -/
def non_host_schools : ℕ := 2

/-- Represents the number of representatives from each non-host school -/
def non_host_representatives : ℕ := 2

/-- Theorem stating the number of ways to arrange the meeting -/
theorem meeting_arrangements : 
  (num_schools) * (members_per_school.choose host_representatives) * 
  ((num_schools - 1).choose non_host_schools) * 
  ((members_per_school.choose non_host_representatives) ^ non_host_schools) = 16200 := by
  sorry


end NUMINAMATH_CALUDE_meeting_arrangements_l1647_164749


namespace NUMINAMATH_CALUDE_purple_chip_value_l1647_164708

def blue_value : ℕ := 1
def green_value : ℕ := 5
def red_value : ℕ := 11

def is_valid_product (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), blue_value^a * green_value^b * x^c * red_value^d = 28160

theorem purple_chip_value :
  ∀ x : ℕ,
  green_value < x →
  x < red_value →
  is_valid_product x →
  x = 7 :=
by sorry

end NUMINAMATH_CALUDE_purple_chip_value_l1647_164708


namespace NUMINAMATH_CALUDE_linear_system_solution_l1647_164704

/-- The system of linear equations ax + by = 10 -/
def linear_system (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = 10

theorem linear_system_solution :
  ∃ (a b : ℝ),
    (linear_system a b 2 4 ∧ linear_system a b 3 1) ∧
    (a = 3 ∧ b = 1) ∧
    (∀ x : ℝ, x > 10 / 3 → linear_system a b x 0 → linear_system a b x y → y < 0) :=
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1647_164704


namespace NUMINAMATH_CALUDE_table_tennis_games_l1647_164779

theorem table_tennis_games (total_games : ℕ) 
  (petya_games : ℕ) (kolya_games : ℕ) (vasya_games : ℕ) : 
  petya_games = total_games / 2 →
  kolya_games = total_games / 3 →
  vasya_games = total_games / 5 →
  petya_games + kolya_games + vasya_games ≤ total_games →
  (∃ (games_between_petya_kolya : ℕ), 
    games_between_petya_kolya ≤ 1 ∧
    petya_games + kolya_games + vasya_games + games_between_petya_kolya = total_games) →
  total_games = 30 := by
sorry

end NUMINAMATH_CALUDE_table_tennis_games_l1647_164779


namespace NUMINAMATH_CALUDE_sequence_sum_l1647_164700

theorem sequence_sum (a b c d : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧
  b - a = c - b ∧
  c * c = b * d ∧
  d - a = 20 →
  a + b + c + d = 46 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1647_164700


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1647_164717

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1647_164717


namespace NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l1647_164718

theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ) 
  (total_marks : ℕ) 
  (correct_sums : ℕ) 
  (penalty_per_incorrect : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : total_marks = 65) 
  (h3 : correct_sums = 25) 
  (h4 : penalty_per_incorrect = 2) :
  (total_marks + penalty_per_incorrect * (total_sums - correct_sums)) / correct_sums = 3 := by
sorry

end NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l1647_164718


namespace NUMINAMATH_CALUDE_abs_z_equals_sqrt_two_l1647_164711

-- Define the complex number z
def z : ℂ := 1 + 2 * Complex.I + Complex.I ^ 3

-- Theorem statement
theorem abs_z_equals_sqrt_two : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_sqrt_two_l1647_164711


namespace NUMINAMATH_CALUDE_garden_length_l1647_164703

/-- Proves that a rectangular garden with length twice its width and perimeter 900 yards has a length of 300 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- The length is twice the width
  2 * length + 2 * width = 900 →  -- The perimeter is 900 yards
  length = 300 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l1647_164703


namespace NUMINAMATH_CALUDE_machine_selling_price_l1647_164712

/-- Calculates the selling price of a machine given its costs and profit percentage -/
def selling_price (purchase_price repair_cost transportation_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transportation_charges
  let profit := (total_cost * profit_percentage) / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 22500 Rs -/
theorem machine_selling_price :
  selling_price 9000 5000 1000 50 = 22500 := by
  sorry

end NUMINAMATH_CALUDE_machine_selling_price_l1647_164712


namespace NUMINAMATH_CALUDE_quilt_patch_cost_l1647_164763

/-- The total cost of patches for a quilt with given dimensions and pricing structure -/
theorem quilt_patch_cost (quilt_length quilt_width patch_area : ℕ)
  (first_batch_size first_batch_price : ℕ) : 
  quilt_length = 16 →
  quilt_width = 20 →
  patch_area = 4 →
  first_batch_size = 10 →
  first_batch_price = 10 →
  (quilt_length * quilt_width) % patch_area = 0 →
  (first_batch_size * first_batch_price) + 
  ((quilt_length * quilt_width / patch_area - first_batch_size) * (first_batch_price / 2)) = 450 :=
by sorry

end NUMINAMATH_CALUDE_quilt_patch_cost_l1647_164763


namespace NUMINAMATH_CALUDE_total_original_cost_l1647_164727

theorem total_original_cost (x y z : ℝ) : 
  x * (1 + 0.3) = 351 →
  y * (1 + 0.25) = 275 →
  z * (1 + 0.2) = 96 →
  x + y + z = 570 := by
sorry

end NUMINAMATH_CALUDE_total_original_cost_l1647_164727


namespace NUMINAMATH_CALUDE_sara_paycheck_l1647_164716

/-- Sara's paycheck calculation --/
theorem sara_paycheck (weeks : ℕ) (hours_per_week : ℕ) (hourly_rate : ℚ) (tire_cost : ℚ) :
  weeks = 2 →
  hours_per_week = 40 →
  hourly_rate = 11.5 →
  tire_cost = 410 →
  (weeks * hours_per_week : ℚ) * hourly_rate - tire_cost = 510 :=
by sorry

end NUMINAMATH_CALUDE_sara_paycheck_l1647_164716


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1647_164743

theorem infinite_series_sum : 
  (∑' n : ℕ, (n^2 + 3*n + 2) / (n * (n + 1) * (n + 3))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1647_164743


namespace NUMINAMATH_CALUDE_mikeys_leaves_theorem_l1647_164794

/-- Given an initial number of leaves and the remaining number of leaves,
    calculate the number of leaves that blew away. -/
def leaves_blown_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that for Mikey's specific case, 
    the number of leaves blown away is 244. -/
theorem mikeys_leaves_theorem :
  leaves_blown_away 356 112 = 244 := by
  sorry

end NUMINAMATH_CALUDE_mikeys_leaves_theorem_l1647_164794


namespace NUMINAMATH_CALUDE_constant_value_l1647_164724

theorem constant_value (t : ℝ) (C : ℝ) : 
  let x := 1 - 4 * t
  let y := 2 * t + C
  (x = y → t = 0.5) → C = -2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l1647_164724


namespace NUMINAMATH_CALUDE_purple_balls_count_l1647_164762

theorem purple_balls_count (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 60 + purple)
  (h2 : white = 22)
  (h3 : green = 18)
  (h4 : yellow = 17)
  (h5 : red = 3)
  (h6 : (white + green + yellow : ℚ) / total = 95/100) : 
  purple = 0 := by
sorry

end NUMINAMATH_CALUDE_purple_balls_count_l1647_164762


namespace NUMINAMATH_CALUDE_function_max_min_sum_l1647_164761

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (Real.log x) / (Real.log a)

theorem function_max_min_sum (a : ℝ) :
  a > 0 ∧ a ≠ 1 →
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f a x ≤ max) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = max) ∧
    (∀ x ∈ Set.Icc 1 2, min ≤ f a x) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = min) ∧
    max + min = (Real.log 2) / (Real.log a) + 6) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_function_max_min_sum_l1647_164761


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l1647_164786

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 15 → 
  (40/100 : ℝ) * N = 180 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l1647_164786


namespace NUMINAMATH_CALUDE_abc_inequality_l1647_164725

theorem abc_inequality (a b c : ℝ) (h : a^2 * b * c + a * b^2 * c + a * b * c^2 = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l1647_164725


namespace NUMINAMATH_CALUDE_fraction_equality_l1647_164785

theorem fraction_equality : (2 : ℚ) / 5 - (1 : ℚ) / 7 = 1 / ((35 : ℚ) / 9) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1647_164785


namespace NUMINAMATH_CALUDE_polygon_has_five_sides_l1647_164722

/-- The set T of points (x, y) satisfying the given conditions -/
def T (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    a / 3 ≤ x ∧ x ≤ 5 * a / 2 ∧
    a / 3 ≤ y ∧ y ≤ 5 * a / 2 ∧
    x + y ≥ 3 * a / 2 ∧
    x + 2 * a ≥ 2 * y ∧
    2 * y + 2 * a ≥ 3 * x}

/-- The theorem stating that the polygon formed by T has 5 sides -/
theorem polygon_has_five_sides (a : ℝ) (ha : a > 0) :
  ∃ (vertices : Finset (ℝ × ℝ)), vertices.card = 5 ∧
  (∀ p ∈ T a, p ∈ convexHull ℝ (↑vertices : Set (ℝ × ℝ))) ∧
  (∀ v ∈ vertices, v ∈ T a) :=
sorry

end NUMINAMATH_CALUDE_polygon_has_five_sides_l1647_164722


namespace NUMINAMATH_CALUDE_lauren_revenue_l1647_164707

def commercial_revenue (per_commercial : ℚ) (num_commercials : ℕ) : ℚ :=
  per_commercial * num_commercials

def subscription_revenue (per_subscription : ℚ) (num_subscriptions : ℕ) : ℚ :=
  per_subscription * num_subscriptions

theorem lauren_revenue 
  (per_commercial : ℚ) 
  (per_subscription : ℚ) 
  (num_commercials : ℕ) 
  (num_subscriptions : ℕ) 
  (total_revenue : ℚ) :
  per_subscription = 1 →
  num_commercials = 100 →
  num_subscriptions = 27 →
  total_revenue = 77 →
  commercial_revenue per_commercial num_commercials + 
    subscription_revenue per_subscription num_subscriptions = total_revenue →
  per_commercial = 1/2 := by
sorry

end NUMINAMATH_CALUDE_lauren_revenue_l1647_164707


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1647_164713

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 13*n + 40 ≤ 0 ∧
  ∀ (m : ℤ), m^2 - 13*m + 40 ≤ 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1647_164713


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1647_164747

theorem log_sum_equals_two : Real.log 4 + 2 * Real.log 5 = 2 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1647_164747


namespace NUMINAMATH_CALUDE_mika_stickers_total_l1647_164728

/-- The total number of stickers Mika has -/
def total_stickers (initial bought birthday sister mother : ℝ) : ℝ :=
  initial + bought + birthday + sister + mother

/-- Theorem stating that Mika has 130.0 stickers in total -/
theorem mika_stickers_total :
  total_stickers 20.0 26.0 20.0 6.0 58.0 = 130.0 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_total_l1647_164728
