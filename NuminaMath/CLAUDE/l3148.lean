import Mathlib

namespace inequality_solution_count_l3148_314897

theorem inequality_solution_count : 
  (Finset.filter (fun x => (x - 2)^2 ≤ 4) (Finset.range 100)).card = 5 := by
  sorry

end inequality_solution_count_l3148_314897


namespace average_weight_increase_l3148_314839

/-- 
Proves that replacing a person weighing 65 kg with a person weighing 97 kg 
in a group of 10 people increases the average weight by 3.2 kg
-/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 10 * initial_average
  let new_total := initial_total - 65 + 97
  let new_average := new_total / 10
  new_average - initial_average = 3.2 := by
  sorry

end average_weight_increase_l3148_314839


namespace min_sum_xy_l3148_314817

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 := by
  sorry

end min_sum_xy_l3148_314817


namespace tangent_circles_m_range_l3148_314850

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 25 - m^2 = 0

-- Define the property of being externally tangent
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m

-- State the theorem
theorem tangent_circles_m_range :
  ∀ m : ℝ, externally_tangent m ↔ m ∈ Set.Ioo (-4) 0 ∪ Set.Ioo 0 4 :=
sorry

end tangent_circles_m_range_l3148_314850


namespace museum_exhibit_group_size_l3148_314851

/-- Represents the ticket sales data for a museum exhibit --/
structure TicketSales where
  regular_price : ℕ
  student_price : ℕ
  total_revenue : ℕ
  regular_to_student_ratio : ℕ
  start_time : ℕ  -- in minutes since midnight
  end_time : ℕ    -- in minutes since midnight
  interval : ℕ    -- in minutes

/-- Calculates the number of people in each group for the given ticket sales data --/
def people_per_group (sales : TicketSales) : ℕ :=
  let student_tickets := sales.total_revenue / (sales.regular_price * sales.regular_to_student_ratio + sales.student_price)
  let regular_tickets := student_tickets * sales.regular_to_student_ratio
  let total_tickets := student_tickets + regular_tickets
  let num_groups := (sales.end_time - sales.start_time) / sales.interval
  total_tickets / num_groups

/-- Theorem stating that for the given conditions, the number of people in each group is 30 --/
theorem museum_exhibit_group_size :
  let sales : TicketSales := {
    regular_price := 10,
    student_price := 5,
    total_revenue := 28350,
    regular_to_student_ratio := 3,
    start_time := 9 * 60,      -- 9:00 AM in minutes
    end_time := 17 * 60 + 55,  -- 5:55 PM in minutes
    interval := 5
  }
  people_per_group sales = 30 := by
  sorry


end museum_exhibit_group_size_l3148_314851


namespace digit_sum_difference_l3148_314875

/-- Represents a two-digit number -/
def TwoDigitNumber (tens ones : Nat) : Nat := 10 * tens + ones

theorem digit_sum_difference (A B C D : Nat) (E F : Nat) 
  (h1 : TwoDigitNumber A B + TwoDigitNumber C D = TwoDigitNumber A E)
  (h2 : TwoDigitNumber A B - TwoDigitNumber D C = TwoDigitNumber A F)
  (h3 : A < 10) (h4 : B < 10) (h5 : C < 10) (h6 : D < 10) : E = 9 := by
  sorry

end digit_sum_difference_l3148_314875


namespace sqrt_3_times_sqrt_12_l3148_314820

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l3148_314820


namespace remainder_problem_l3148_314837

theorem remainder_problem : (29 * 171997^2000) % 7 = 4 := by
  sorry

end remainder_problem_l3148_314837


namespace three_triples_l3148_314805

/-- Least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given conditions -/
def count_triples : ℕ := sorry

/-- Theorem stating that there are exactly 3 ordered triples satisfying the conditions -/
theorem three_triples : count_triples = 3 := by sorry

end three_triples_l3148_314805


namespace square_pens_area_ratio_l3148_314803

/-- Given four congruent square pens with side length s, prove that the ratio of their
    total area to the area of a single square pen formed by reusing the same amount
    of fencing is 1/4. -/
theorem square_pens_area_ratio (s : ℝ) (h : s > 0) : 
  (4 * s^2) / ((4 * s)^2) = 1 / 4 := by
sorry

end square_pens_area_ratio_l3148_314803


namespace parabola_equation_l3148_314800

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 9 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -3)

-- Define the parabola properties
structure Parabola where
  -- The coordinate axes are the axes of symmetry
  symmetry_axes : Prop
  -- The origin is the vertex
  vertex_at_origin : Prop
  -- The parabola passes through the center of the circle
  passes_through_center : Prop

-- Theorem statement
theorem parabola_equation (p : Parabola) :
  (∀ x y : ℝ, y^2 = 9*x) ∨ (∀ x y : ℝ, x^2 = -1/3*y) :=
sorry

end parabola_equation_l3148_314800


namespace modulo_problem_l3148_314810

theorem modulo_problem (n : ℕ) : 
  (215 * 789) % 75 = n ∧ 0 ≤ n ∧ n < 75 → n = 60 :=
by
  sorry

end modulo_problem_l3148_314810


namespace not_always_cylinder_l3148_314870

/-- A cylinder in 3D space -/
structure Cylinder where
  base : Set (ℝ × ℝ)  -- Base of the cylinder
  height : ℝ          -- Height of the cylinder

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ  -- Normal vector of the plane
  point : ℝ × ℝ × ℝ   -- A point on the plane

/-- Two planes are parallel if their normal vectors are parallel -/
def parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), p1.normal = k • p2.normal

/-- The result of cutting a cylinder with two parallel planes -/
def cut_cylinder (c : Cylinder) (p1 p2 : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry  -- Definition of the cut cylinder

/-- Theorem: Cutting a cylinder with two arbitrary parallel planes 
    does not always result in a cylinder -/
theorem not_always_cylinder (c : Cylinder) :
  ∃ (p1 p2 : Plane), parallel p1 p2 ∧ ¬∃ (c' : Cylinder), cut_cylinder c p1 p2 = {(x, y, z) | (x, y) ∈ c'.base ∧ 0 ≤ z ∧ z ≤ c'.height} :=
sorry


end not_always_cylinder_l3148_314870


namespace base_2016_remainder_l3148_314808

theorem base_2016_remainder (N A B C k : ℕ) : 
  (N = A * 2016^2 + B * 2016 + C) →
  (A < 2016 ∧ B < 2016 ∧ C < 2016) →
  (1 ≤ k ∧ k ≤ 2015) →
  (N - (A + B + C + k)) % 2015 = 2015 - k := by
  sorry

end base_2016_remainder_l3148_314808


namespace one_correct_statement_l3148_314830

-- Define a sequence as a function from natural numbers to real numbers
def Sequence := ℕ → ℝ

-- Statement 1: A sequence represented graphically appears as a group of isolated points
def graphical_representation (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ ε > 0, ∀ m : ℕ, m ≠ n → |s m - s n| > ε

-- Statement 2: The terms of a sequence are finite
def finite_terms (s : Sequence) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n > N → s n = 0

-- Statement 3: If a sequence is decreasing, then the sequence must be finite
def decreasing_implies_finite (s : Sequence) : Prop :=
  (∀ n : ℕ, s (n + 1) ≤ s n) → finite_terms s

-- Theorem stating that only one of the above statements is correct
theorem one_correct_statement :
  (∀ s : Sequence, graphical_representation s) ∧
  (∃ s : Sequence, ¬finite_terms s) ∧
  (∃ s : Sequence, (∀ n : ℕ, s (n + 1) ≤ s n) ∧ ¬finite_terms s) :=
sorry

end one_correct_statement_l3148_314830


namespace divisibility_equivalence_l3148_314807

theorem divisibility_equivalence (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) := by sorry

end divisibility_equivalence_l3148_314807


namespace abs_and_opposite_l3148_314892

theorem abs_and_opposite :
  (abs (-2) = 2) ∧ (-(1/2) = -1/2) := by sorry

end abs_and_opposite_l3148_314892


namespace probability_one_white_one_red_l3148_314896

theorem probability_one_white_one_red (total : ℕ) (white : ℕ) (red : ℕ) :
  total = white + red →
  total = 15 →
  white = 10 →
  red = 5 →
  (white.choose 1 * red.choose 1 : ℚ) / total.choose 2 = 10 / 21 := by
  sorry

end probability_one_white_one_red_l3148_314896


namespace area_square_on_hypotenuse_for_24cm_l3148_314844

/-- An isosceles right triangle with an inscribed square -/
structure TriangleWithSquare where
  /-- Side length of the inscribed square touching the right angle -/
  s : ℝ
  /-- The square touches the right angle vertex -/
  touches_right_angle : s > 0
  /-- The opposite side of the square is parallel to the hypotenuse -/
  parallel_to_hypotenuse : True

/-- The area of a square inscribed along the hypotenuse of the triangle -/
def area_square_on_hypotenuse (t : TriangleWithSquare) : ℝ :=
  t.s ^ 2

theorem area_square_on_hypotenuse_for_24cm (t : TriangleWithSquare) 
  (h : t.s = 24) : area_square_on_hypotenuse t = 576 := by
  sorry

end area_square_on_hypotenuse_for_24cm_l3148_314844


namespace smallest_n_divisible_by_24_and_1024_l3148_314816

theorem smallest_n_divisible_by_24_and_1024 :
  ∃ n : ℕ+, (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(1024 ∣ m^3))) ∧
  (24 ∣ n^2) ∧ (1024 ∣ n^3) ∧ n = 48 := by
  sorry

end smallest_n_divisible_by_24_and_1024_l3148_314816


namespace pen_profit_percentage_l3148_314864

/-- Calculates the profit percentage for a retailer selling pens -/
theorem pen_profit_percentage 
  (num_pens : ℕ) 
  (cost_pens : ℕ) 
  (discount_percent : ℚ) : 
  num_pens = 60 → 
  cost_pens = 36 → 
  discount_percent = 1/100 →
  (((num_pens : ℚ) * (1 - discount_percent) - cost_pens) / cost_pens) * 100 = 65 := by
  sorry

#check pen_profit_percentage

end pen_profit_percentage_l3148_314864


namespace henry_lawn_mowing_l3148_314825

/-- The number of lawns Henry was supposed to mow -/
def total_lawns : ℕ := 12

/-- The amount Henry earns per lawn -/
def earnings_per_lawn : ℕ := 5

/-- The number of lawns Henry forgot to mow -/
def forgotten_lawns : ℕ := 7

/-- The amount Henry actually earned -/
def actual_earnings : ℕ := 25

theorem henry_lawn_mowing :
  total_lawns = (actual_earnings / earnings_per_lawn) + forgotten_lawns :=
by sorry

end henry_lawn_mowing_l3148_314825


namespace megan_carrots_count_l3148_314884

/-- The total number of carrots Megan has after picking, throwing out some, and picking more. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

/-- Theorem stating that Megan's total carrots can be calculated using the given formula. -/
theorem megan_carrots_count (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ)
    (h1 : initial ≥ thrown_out) :
  total_carrots initial thrown_out picked_next_day = initial - thrown_out + picked_next_day :=
by
  sorry

#eval total_carrots 19 4 46  -- Should evaluate to 61

end megan_carrots_count_l3148_314884


namespace points_on_same_side_l3148_314819

/-- The time when two points moving on a square are first on the same side -/
def time_on_same_side (square_side : ℝ) (speed_A : ℝ) (speed_B : ℝ) : ℝ :=
  25

/-- Theorem stating that the time when the points are first on the same side is 25 seconds -/
theorem points_on_same_side (square_side : ℝ) (speed_A : ℝ) (speed_B : ℝ) 
  (h1 : square_side = 100)
  (h2 : speed_A = 5)
  (h3 : speed_B = 10) :
  time_on_same_side square_side speed_A speed_B = 25 :=
by
  sorry

#check points_on_same_side

end points_on_same_side_l3148_314819


namespace function_transformation_l3148_314815

/-- Given a function f where f(2) = 0, prove that g(x) = f(x-3)+1 passes through (5, 1) -/
theorem function_transformation (f : ℝ → ℝ) (h : f 2 = 0) :
  let g := λ x => f (x - 3) + 1
  g 5 = 1 := by
sorry

end function_transformation_l3148_314815


namespace smallest_divisible_by_15_and_24_l3148_314832

theorem smallest_divisible_by_15_and_24 : ∃ n : ℕ, (n > 0 ∧ n % 15 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 15 = 0 ∧ m % 24 = 0) → n ≤ m) ∧ n = 120 := by
  sorry

end smallest_divisible_by_15_and_24_l3148_314832


namespace cube_root_last_three_digits_l3148_314818

theorem cube_root_last_three_digits :
  ∃ (n : ℕ+) (a : ℕ+) (b : ℕ),
    n = 1000 * a + b ∧
    b < 1000 ∧
    n = a^3 ∧
    n = 32768 := by
  sorry

end cube_root_last_three_digits_l3148_314818


namespace circle_tangent_at_origin_l3148_314898

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- The equation of the circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y + c.F = 0

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A circle is tangent to the x-axis at the origin -/
def tangent_at_origin (c : Circle) : Prop :=
  c.equation origin.x origin.y ∧
  ∀ (p : Point), p.y = 0 → p = origin ∨ ¬c.equation p.x p.y

theorem circle_tangent_at_origin (c : Circle) :
  tangent_at_origin c → c.E ≠ 0 ∧ c.D = 0 ∧ c.F = 0 := by
  sorry

end circle_tangent_at_origin_l3148_314898


namespace davis_items_left_l3148_314876

/-- The number of items Miss Davis has left after distributing popsicle sticks and straws --/
def items_left (popsicle_sticks_per_group : ℕ) (straws_per_group : ℕ) (num_groups : ℕ) (total_items : ℕ) : ℕ :=
  total_items - (popsicle_sticks_per_group + straws_per_group) * num_groups

/-- Theorem stating that Miss Davis has 150 items left --/
theorem davis_items_left :
  items_left 15 20 10 500 = 150 := by
  sorry

end davis_items_left_l3148_314876


namespace book_price_increase_percentage_l3148_314806

theorem book_price_increase_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 300)
  (h2 : new_price = 480) :
  (new_price - original_price) / original_price * 100 = 60 := by
  sorry

end book_price_increase_percentage_l3148_314806


namespace largest_three_digit_product_l3148_314822

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

theorem largest_three_digit_product : ∃ (m x y : ℕ),
  100 ≤ m ∧ m < 1000 ∧
  isPrime x ∧ isPrime y ∧ isPrime (10 * x - y) ∧
  x < 10 ∧ y < 10 ∧ x ≠ y ∧
  m = x * y * (10 * x - y) ∧
  ∀ (m' x' y' : ℕ),
    100 ≤ m' ∧ m' < 1000 →
    isPrime x' ∧ isPrime y' ∧ isPrime (10 * x' - y') →
    x' < 10 ∧ y' < 10 ∧ x' ≠ y' →
    m' = x' * y' * (10 * x' - y') →
    m' ≤ m ∧
  m = 705 := by
  sorry

end largest_three_digit_product_l3148_314822


namespace johns_allowance_l3148_314888

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : A = 4.80 :=
  let arcade_spent := (3 : ℚ) / 5
  let arcade_remaining := 1 - arcade_spent
  let toy_store_spent := (1 : ℚ) / 3 * arcade_remaining
  let candy_store_remaining := arcade_remaining - toy_store_spent
  have h1 : arcade_remaining = (2 : ℚ) / 5 := by sorry
  have h2 : candy_store_remaining = (4 : ℚ) / 15 := by sorry
  have h3 : candy_store_remaining * A = 1.28 := by sorry
  sorry

#eval (4.80 : ℚ)

end johns_allowance_l3148_314888


namespace sum_with_radical_conjugate_l3148_314869

theorem sum_with_radical_conjugate : 
  let x : ℝ := 10 - Real.sqrt 2018
  let y : ℝ := 10 + Real.sqrt 2018  -- Definition of radical conjugate
  x + y = 20 := by
sorry

end sum_with_radical_conjugate_l3148_314869


namespace no_both_squares_l3148_314858

theorem no_both_squares : ¬∃ (x y : ℕ+), 
  ∃ (a b : ℕ+), (x^2 + 2*y : ℕ) = a^2 ∧ (y^2 + 2*x : ℕ) = b^2 := by
  sorry

end no_both_squares_l3148_314858


namespace curve_and_tangent_l3148_314866

noncomputable section

-- Define the curve C
def C (k : ℝ) (x y : ℝ) : Prop :=
  x^(2/3) + y^(2/3) = k^(2/3)

-- Define the line segment AB
def AB (k : ℝ) (α β : ℝ) : Prop :=
  α^2 + β^2 = k^2

-- Define the midpoint M of AB
def M (α β : ℝ) : ℝ × ℝ :=
  (α^3 / (α^2 + β^2), β^3 / (α^2 + β^2))

-- State the theorem
theorem curve_and_tangent (k : ℝ) (h : k > 0) :
  ∀ α β : ℝ, AB k α β →
  let (x, y) := M α β
  (C k x y) ∧
  (∃ t : ℝ, t * α + (1 - t) * 0 = x ∧ t * 0 + (1 - t) * β = y) :=
sorry

end

end curve_and_tangent_l3148_314866


namespace stratified_sampling_business_personnel_l3148_314895

theorem stratified_sampling_business_personnel 
  (total_employees : ℕ) 
  (business_personnel : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 160) 
  (h2 : business_personnel = 120) 
  (h3 : sample_size = 20) :
  (business_personnel * sample_size) / total_employees = 15 := by
sorry

end stratified_sampling_business_personnel_l3148_314895


namespace angle_calculations_l3148_314813

theorem angle_calculations (α : Real) (h : Real.tan α = -3/7) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/7 ∧
  2 + Real.sin α * Real.cos α - Real.cos α ^ 2 = 23/29 := by
sorry

end angle_calculations_l3148_314813


namespace meadow_diaper_earnings_l3148_314862

/-- Calculates the total money earned from selling diapers -/
def total_money (boxes : ℕ) (packs_per_box : ℕ) (diapers_per_pack : ℕ) (price_per_diaper : ℕ) : ℕ :=
  boxes * packs_per_box * diapers_per_pack * price_per_diaper

/-- Proves that Meadow's total earnings from selling diapers is $960,000 -/
theorem meadow_diaper_earnings :
  total_money 30 40 160 5 = 960000 := by
  sorry

#eval total_money 30 40 160 5

end meadow_diaper_earnings_l3148_314862


namespace simple_interest_rate_calculation_l3148_314838

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 1050) (h3 : T = 5) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 8 := by sorry

end simple_interest_rate_calculation_l3148_314838


namespace pure_imaginary_complex_number_l3148_314867

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (2 + a * Complex.I) / (1 - Complex.I)) → a = 2 := by
  sorry

end pure_imaginary_complex_number_l3148_314867


namespace pasta_preference_ratio_is_two_l3148_314863

/-- The ratio of students preferring spaghetti to those preferring manicotti -/
def pasta_preference_ratio (spaghetti_count : ℕ) (manicotti_count : ℕ) : ℚ :=
  spaghetti_count / manicotti_count

/-- The total number of students surveyed -/
def total_students : ℕ := 800

/-- The number of students who preferred spaghetti -/
def spaghetti_preference : ℕ := 320

/-- The number of students who preferred manicotti -/
def manicotti_preference : ℕ := 160

theorem pasta_preference_ratio_is_two :
  pasta_preference_ratio spaghetti_preference manicotti_preference = 2 := by
  sorry

end pasta_preference_ratio_is_two_l3148_314863


namespace arrange_algebra_and_calculus_books_l3148_314882

/-- The number of ways to arrange books on a shelf --/
def arrange_books (algebra_copies : ℕ) (calculus_copies : ℕ) : ℕ :=
  Nat.choose (algebra_copies + calculus_copies) algebra_copies

/-- Theorem: Arranging 4 algebra books and 5 calculus books yields 126 possibilities --/
theorem arrange_algebra_and_calculus_books :
  arrange_books 4 5 = 126 := by
  sorry

end arrange_algebra_and_calculus_books_l3148_314882


namespace vector_subtraction_l3148_314861

/-- Given complex numbers z1 and z2 representing vectors OA and OB respectively,
    prove that the complex number representing BA is equal to 5-5i. -/
theorem vector_subtraction (z1 z2 : ℂ) (h1 : z1 = 2 - 3*I) (h2 : z2 = -3 + 2*I) :
  z1 - z2 = 5 - 5*I := by sorry

end vector_subtraction_l3148_314861


namespace min_value_and_ellipse_l3148_314831

theorem min_value_and_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∀ x : ℝ, x > 0 → (a + b) * x - 1 ≤ x^2) :
  (∀ c d : ℝ, c > 0 → d > 0 → 1 / c + 1 / d ≥ 2) ∧
  (1 / a^2 + 1 / b^2 > 1) := by
  sorry

end min_value_and_ellipse_l3148_314831


namespace total_money_l3148_314840

/-- Given that A and C together have 200, B and C together have 350, and C has 200,
    prove that the total amount of money A, B, and C have between them is 350. -/
theorem total_money (A B C : ℕ) 
  (hAC : A + C = 200)
  (hBC : B + C = 350)
  (hC : C = 200) : 
  A + B + C = 350 := by
sorry

end total_money_l3148_314840


namespace min_value_problem_l3148_314889

theorem min_value_problem (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2*y = 1) :
  ∃ m : ℝ, m = 8/9 ∧ ∀ x' y' : ℝ, x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*y'^2 ≥ m :=
by sorry

end min_value_problem_l3148_314889


namespace vector_addition_l3148_314802

/-- Given two vectors AB and BC in ℝ², prove that AC = AB + BC -/
theorem vector_addition (AB BC : ℝ × ℝ) : 
  AB = (2, -1) → BC = (-4, 1) → AB + BC = (-2, 0) := by sorry

end vector_addition_l3148_314802


namespace cos_equality_angle_l3148_314855

theorem cos_equality_angle (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (280 * π / 180) → n = 80 := by
sorry

end cos_equality_angle_l3148_314855


namespace total_amount_after_two_years_l3148_314856

/-- Calculates the total amount returned after compound interest --/
def totalAmountAfterCompoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) (compoundInterest : ℝ) : ℝ :=
  principal + compoundInterest

/-- Theorem stating the total amount returned after two years of compound interest --/
theorem total_amount_after_two_years 
  (principal : ℝ) 
  (rate : ℝ) 
  (compoundInterest : ℝ) 
  (h1 : rate = 0.05) 
  (h2 : compoundInterest = 246) 
  (h3 : principal * ((1 + rate)^2 - 1) = compoundInterest) : 
  totalAmountAfterCompoundInterest principal rate 2 compoundInterest = 2646 := by
  sorry

#check total_amount_after_two_years

end total_amount_after_two_years_l3148_314856


namespace horner_method_correct_l3148_314899

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f_coeffs : List ℝ := [5, 4, 3, 2, 1, 0]

theorem horner_method_correct :
  horner_eval f_coeffs 3 = 1641 := by
  sorry

#eval horner_eval f_coeffs 3

end horner_method_correct_l3148_314899


namespace brick_height_calculation_l3148_314860

/-- Prove that the height of each brick is 67.5 cm, given the wall dimensions,
    brick dimensions (except height), and the number of bricks needed. -/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ)
                                 (brick_length brick_width : ℝ)
                                 (num_bricks : ℕ) :
  wall_length = 900 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 7200 →
  ∃ (brick_height : ℝ),
    brick_height = 67.5 ∧
    wall_length * wall_width * wall_height =
      num_bricks * brick_length * brick_width * brick_height :=
by
  sorry

end brick_height_calculation_l3148_314860


namespace total_animals_savanna_l3148_314801

-- Define the number of animals in Safari National Park
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10
def safari_elephants : ℕ := safari_lions / 4

-- Define the number of animals in Savanna National Park
def savanna_lions : ℕ := safari_lions * 2
def savanna_snakes : ℕ := safari_snakes * 3
def savanna_giraffes : ℕ := safari_giraffes + 20
def savanna_elephants : ℕ := safari_elephants * 5
def savanna_zebras : ℕ := (savanna_lions + savanna_snakes) / 2

-- Theorem statement
theorem total_animals_savanna : 
  savanna_lions + savanna_snakes + savanna_giraffes + savanna_elephants + savanna_zebras = 710 := by
  sorry

end total_animals_savanna_l3148_314801


namespace absolute_value_sqrt_two_plus_half_inverse_l3148_314842

theorem absolute_value_sqrt_two_plus_half_inverse :
  |1 - Real.sqrt 2| + (1/2)⁻¹ = Real.sqrt 2 + 1 := by sorry

end absolute_value_sqrt_two_plus_half_inverse_l3148_314842


namespace doubling_condition_iff_triangle_or_quadrilateral_l3148_314828

/-- The sum of interior angles of an n-sided polygon is (n-2) * 180°. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A polygon satisfies the doubling condition if the sum of angles after doubling
    the sides is an integer multiple of the original sum of angles. -/
def satisfies_doubling_condition (m : ℕ) : Prop :=
  ∃ k : ℕ, sum_interior_angles (2 * m) = k * sum_interior_angles m

/-- Theorem: A polygon satisfies the doubling condition if and only if
    it has 3 or 4 sides. -/
theorem doubling_condition_iff_triangle_or_quadrilateral (m : ℕ) :
  satisfies_doubling_condition m ↔ m = 3 ∨ m = 4 :=
sorry

end doubling_condition_iff_triangle_or_quadrilateral_l3148_314828


namespace solution_set_exponential_inequality_l3148_314885

theorem solution_set_exponential_inequality :
  ∀ x : ℝ, (2 : ℝ) ^ (x^2 - 5*x + 5) > (1/2 : ℝ) ↔ x < 2 ∨ x > 3 := by
  sorry

end solution_set_exponential_inequality_l3148_314885


namespace solution_set_correct_l3148_314891

/-- The solution set for the system of equations:
    x + y + z = 2
    (x+y)(y+z) + (y+z)(z+x) + (z+x)(x+y) = 1
    x²(y+z) + y²(z+x) + z²(x+y) = -6 -/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 3, -1), (0, -1, 3), (3, 0, -1), (3, -1, 0), (-1, 0, 3), (-1, 3, 0)}

/-- The system of equations -/
def satisfies_equations (x y z : ℝ) : Prop :=
  x + y + z = 2 ∧
  (x+y)*(y+z) + (y+z)*(z+x) + (z+x)*(x+y) = 1 ∧
  x^2*(y+z) + y^2*(z+x) + z^2*(x+y) = -6

theorem solution_set_correct :
  ∀ (x y z : ℝ), (x, y, z) ∈ solution_set ↔ satisfies_equations x y z :=
sorry

end solution_set_correct_l3148_314891


namespace max_xy_value_l3148_314854

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x/3 + y/4 = 1) :
  ∃ (M : ℝ), M = 3 ∧ xy ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀/3 + y₀/4 = 1 ∧ x₀*y₀ = M :=
sorry

end max_xy_value_l3148_314854


namespace product_repeating_third_and_nine_l3148_314812

/-- The repeating decimal 0.3̄ -/
def repeating_third : ℚ := 1/3

/-- Theorem stating that the product of 0.3̄ and 9 is 3 -/
theorem product_repeating_third_and_nine :
  repeating_third * 9 = 3 := by
  sorry

end product_repeating_third_and_nine_l3148_314812


namespace clothing_sale_theorem_l3148_314834

/-- The marked price of an item of clothing --/
def marked_price : ℝ := 300

/-- The loss per item when sold at 40% of marked price --/
def loss_at_40_percent : ℝ := 30

/-- The profit per item when sold at 70% of marked price --/
def profit_at_70_percent : ℝ := 60

/-- The maximum discount percentage that can be offered without incurring a loss --/
def max_discount_percent : ℝ := 50

theorem clothing_sale_theorem :
  (0.4 * marked_price - loss_at_40_percent = 0.7 * marked_price + profit_at_70_percent) ∧
  (max_discount_percent / 100 * marked_price = 0.4 * marked_price + loss_at_40_percent) := by
  sorry

end clothing_sale_theorem_l3148_314834


namespace average_marks_l3148_314852

theorem average_marks (avg_five : ℝ) (sixth_mark : ℝ) : 
  avg_five = 74 → sixth_mark = 80 → 
  ((avg_five * 5 + sixth_mark) / 6 : ℝ) = 75 := by
  sorry

end average_marks_l3148_314852


namespace olivia_remaining_money_l3148_314878

theorem olivia_remaining_money (initial_amount spent_amount : ℕ) 
  (h1 : initial_amount = 128)
  (h2 : spent_amount = 38) :
  initial_amount - spent_amount = 90 := by
sorry

end olivia_remaining_money_l3148_314878


namespace order_inequality_l3148_314836

theorem order_inequality (x a b : ℝ) (h1 : x < a) (h2 : a < b) (h3 : b < 0) :
  x^2 > a*x ∧ a*x > a*b ∧ a*b > a^2 := by
  sorry

end order_inequality_l3148_314836


namespace rotation_theorem_l3148_314873

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the rotation transformation
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

-- State the theorem
theorem rotation_theorem :
  ∀ x y : ℝ, y = f x ↔ (rotate90 (x, y)).2 = -(Function.invFun f) (rotate90 (x, y)).1 :=
by sorry

end rotation_theorem_l3148_314873


namespace number_difference_l3148_314848

theorem number_difference (L S : ℕ) (h1 : L > S) (h2 : L = 1650) (h3 : L = 5 * S + 5) : L - S = 1321 := by
  sorry

end number_difference_l3148_314848


namespace parabola_tangent_to_circle_l3148_314893

/-- Given a parabola and a circle, if the parabola's axis is tangent to the circle, 
    then the parameter p of the parabola equals 2. -/
theorem parabola_tangent_to_circle (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- parabola equation
  (∀ x y : ℝ, x^2 + y^2 - 8*x - 9 = 0) →  -- circle equation
  (∃ x : ℝ, x = -p/2 ∧ (x-4)^2 = 25) →  -- parabola's axis is tangent to the circle
  p = 2 :=
by sorry

end parabola_tangent_to_circle_l3148_314893


namespace solution_set_for_a_equals_one_range_of_a_for_solutions_l3148_314872

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | |x - 1| > (1/2) * (x + 1)} = {x : ℝ | x > 3 ∨ x < 1/3} := by sorry

-- Part 2
theorem range_of_a_for_solutions :
  ∀ a : ℝ, (∃ x : ℝ, f a x + |x - 2| ≤ 3) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end solution_set_for_a_equals_one_range_of_a_for_solutions_l3148_314872


namespace smallest_positive_integer_congruence_l3148_314843

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ+, (42 * x.val + 9) % 15 = 3 ∧
  ∀ y : ℕ+, (42 * y.val + 9) % 15 = 3 → x ≤ y ∧
  x = 2 := by
  sorry

end smallest_positive_integer_congruence_l3148_314843


namespace parallel_transitive_l3148_314835

-- Define the concept of straight lines
variable (Line : Type)

-- Define the parallel relationship between lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by
  sorry

end parallel_transitive_l3148_314835


namespace lcm_6_15_l3148_314890

theorem lcm_6_15 : Nat.lcm 6 15 = 30 := by
  sorry

end lcm_6_15_l3148_314890


namespace intersection_M_N_l3148_314880

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 2 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2, -1, 2} := by
  sorry

end intersection_M_N_l3148_314880


namespace prove_distance_l3148_314871

def distance_between_cities : ℝ → Prop := λ d =>
  let speed_ab : ℝ := 40
  let speed_ba : ℝ := 49.99999999999999
  let total_time : ℝ := 5 + 24 / 60
  (d / speed_ab + d / speed_ba) = total_time

theorem prove_distance : distance_between_cities 120 := by
  sorry

end prove_distance_l3148_314871


namespace four_line_corresponding_angles_l3148_314894

/-- Represents a line in a plane -/
structure Line

/-- Represents an intersection point of two lines -/
structure IntersectionPoint

/-- Represents a pair of corresponding angles -/
structure CorrespondingAnglePair

/-- A configuration of four lines intersecting pairwise -/
structure FourLineConfiguration where
  lines : Fin 4 → Line
  intersections : Fin 6 → IntersectionPoint
  no_triple_intersection : ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ∃ (p q : IntersectionPoint), p ≠ q ∧ 
    (p ∈ (Set.range intersections) ∧ q ∈ (Set.range intersections))

/-- The number of corresponding angle pairs in a four-line configuration -/
def num_corresponding_angles (config : FourLineConfiguration) : ℕ :=
  48

/-- Theorem stating that a four-line configuration has 48 pairs of corresponding angles -/
theorem four_line_corresponding_angles (config : FourLineConfiguration) :
  num_corresponding_angles config = 48 := by sorry

end four_line_corresponding_angles_l3148_314894


namespace smallest_dual_base_palindrome_is_585_dual_base_palindrome_properties_no_smaller_dual_base_palindrome_l3148_314846

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Finds the smallest number greater than 10 that is a palindrome in both base 2 and base 3 -/
def smallestDualBasePalindrome : ℕ := sorry

theorem smallest_dual_base_palindrome_is_585 :
  smallestDualBasePalindrome = 585 := by sorry

theorem dual_base_palindrome_properties (n : ℕ) :
  n = smallestDualBasePalindrome →
  n > 10 ∧ isPalindrome n 2 ∧ isPalindrome n 3 := by sorry

theorem no_smaller_dual_base_palindrome (n : ℕ) :
  10 < n ∧ n < smallestDualBasePalindrome →
  ¬(isPalindrome n 2 ∧ isPalindrome n 3) := by sorry

end smallest_dual_base_palindrome_is_585_dual_base_palindrome_properties_no_smaller_dual_base_palindrome_l3148_314846


namespace a_minus_c_equals_296_l3148_314886

theorem a_minus_c_equals_296 (A B C : ℤ) 
  (h1 : A = B - 397)
  (h2 : A = 742)
  (h3 : B = C + 693) : 
  A - C = 296 := by
  sorry

end a_minus_c_equals_296_l3148_314886


namespace seating_arrangements_count_l3148_314847

def num_chairs : ℕ := 12
def num_students : ℕ := 5
def num_professors : ℕ := 4
def available_positions : ℕ := 6

theorem seating_arrangements_count :
  (Nat.choose available_positions num_professors) * (Nat.factorial num_professors) = 360 :=
by sorry

end seating_arrangements_count_l3148_314847


namespace soccer_boys_percentage_l3148_314821

theorem soccer_boys_percentage (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) (girls_not_playing : ℕ)
  (h1 : total_students = 420)
  (h2 : boys = 320)
  (h3 : soccer_players = 250)
  (h4 : girls_not_playing = 65) :
  (boys - (total_students - boys - girls_not_playing)) / soccer_players * 100 = 86 := by
  sorry

end soccer_boys_percentage_l3148_314821


namespace second_bucket_contents_l3148_314887

def bucket_contents : List ℕ := [11, 13, 12, 16, 10]

theorem second_bucket_contents (h : ∃ x ∈ bucket_contents, x + 10 = 23) :
  (List.sum bucket_contents) - 23 = 39 := by
  sorry

end second_bucket_contents_l3148_314887


namespace tower_surface_area_l3148_314849

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def Cube.volume (c : Cube) : ℝ := c.sideLength ^ 3

/-- Calculates the surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℝ := 6 * c.sideLength ^ 2

/-- Represents the tower of cubes -/
structure CubeTower where
  cubes : List Cube
  isDecreasing : ∀ i j, i < j → (cubes.get i).volume > (cubes.get j).volume
  thirdCubeShifted : True

/-- Calculates the total surface area of the tower -/
def CubeTower.totalSurfaceArea (t : CubeTower) : ℝ :=
  let visibleFaces := [5, 5, 4.5] ++ List.replicate 5 4 ++ [5]
  List.sum (List.zipWith (λ c f => f * c.sideLength ^ 2) t.cubes visibleFaces)

/-- The theorem to be proved -/
theorem tower_surface_area (t : CubeTower) 
  (h1 : t.cubes.length = 9)
  (h2 : List.map Cube.volume t.cubes = [512, 343, 216, 125, 64, 27, 8, 1, 0.125]) :
  t.totalSurfaceArea = 948.25 := by
  sorry

end tower_surface_area_l3148_314849


namespace cycle_transactions_result_l3148_314829

/-- Calculates the final amount after three cycle transactions -/
def final_amount (initial_cost : ℝ) (loss1 gain2 gain3 : ℝ) : ℝ :=
  let selling_price1 := initial_cost * (1 - loss1)
  let selling_price2 := selling_price1 * (1 + gain2)
  selling_price2 * (1 + gain3)

/-- Theorem stating the final amount after three cycle transactions -/
theorem cycle_transactions_result :
  final_amount 1600 0.12 0.15 0.20 = 1943.04 := by
  sorry

#eval final_amount 1600 0.12 0.15 0.20

end cycle_transactions_result_l3148_314829


namespace nine_squared_minus_sqrt_nine_l3148_314877

theorem nine_squared_minus_sqrt_nine : 9^2 - Real.sqrt 9 = 78 := by
  sorry

end nine_squared_minus_sqrt_nine_l3148_314877


namespace pens_and_pencils_equation_system_l3148_314827

theorem pens_and_pencils_equation_system (x y : ℕ) : 
  (x + y = 30 ∧ x = 2 * y - 3) ↔ 
  (x + y = 30 ∧ x = 2 * y - 3 ∧ x < 2 * y) := by
  sorry

end pens_and_pencils_equation_system_l3148_314827


namespace parrots_per_cage_l3148_314826

theorem parrots_per_cage (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_birds : ℕ) :
  num_cages = 4 →
  parakeets_per_cage = 2 →
  total_birds = 40 →
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 8 :=
by
  sorry

end parrots_per_cage_l3148_314826


namespace cheetah_catches_deer_l3148_314853

/-- Proves that a cheetah catches up with a deer in 10 minutes given specific conditions -/
theorem cheetah_catches_deer (deer_speed cheetah_speed : ℝ) 
  (time_difference : ℝ) (catch_up_time : ℝ) : 
  deer_speed = 50 → 
  cheetah_speed = 60 → 
  time_difference = 2 / 60 → 
  (deer_speed * time_difference) / (cheetah_speed - deer_speed) = catch_up_time →
  catch_up_time = 1 / 6 := by
  sorry

#check cheetah_catches_deer

end cheetah_catches_deer_l3148_314853


namespace alices_number_l3148_314879

theorem alices_number (n : ℕ) 
  (h1 : n % 243 = 0)
  (h2 : n % 36 = 0)
  (h3 : 1000 < n ∧ n < 3000) :
  n = 1944 ∨ n = 2916 := by
sorry

end alices_number_l3148_314879


namespace modulus_of_complex_fraction_l3148_314845

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  Complex.abs ((1 + 3 * i) / (1 - i)) = Real.sqrt 5 := by sorry

end modulus_of_complex_fraction_l3148_314845


namespace brick_count_for_wall_l3148_314814

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

/-- Converts meters to centimeters -/
def meters_to_cm (m : ℝ) : ℝ :=
  m * 100

theorem brick_count_for_wall :
  let brick_length : ℝ := 20
  let brick_width : ℝ := 10
  let brick_height : ℝ := 7.5
  let wall_length : ℝ := 27
  let wall_width : ℝ := 2
  let wall_height : ℝ := 0.75
  let brick_volume : ℝ := volume brick_length brick_width brick_height
  let wall_volume : ℝ := volume (meters_to_cm wall_length) (meters_to_cm wall_width) (meters_to_cm wall_height)
  (wall_volume / brick_volume : ℝ) = 27000 :=
by sorry

end brick_count_for_wall_l3148_314814


namespace problem_solution_l3148_314833

def f (x : ℝ) : ℝ := 2 * x - 4
def g (x : ℝ) : ℝ := -x + 4

theorem problem_solution :
  (f 1 = -2 ∧ g 1 = 3) ∧
  (∀ x, f x * g x = -2 * x^2 + 12 * x - 16) ∧
  (Set.Icc 2 4 = {x | f x * g x = 0}) ∧
  (∀ x y, x < 3 ∧ y < 3 ∧ x < y → f x * g x < f y * g y) ∧
  (∀ x y, x > 3 ∧ y > 3 ∧ x < y → f x * g x > f y * g y) :=
by sorry

end problem_solution_l3148_314833


namespace kim_easy_round_answers_l3148_314883

/-- Represents the number of points for each round in the math contest -/
structure ContestPoints where
  easy : ℕ
  average : ℕ
  hard : ℕ

/-- Represents the number of correct answers for each round -/
structure ContestAnswers where
  easy : ℕ
  average : ℕ
  hard : ℕ

def totalPoints (points : ContestPoints) (answers : ContestAnswers) : ℕ :=
  points.easy * answers.easy + points.average * answers.average + points.hard * answers.hard

theorem kim_easy_round_answers 
  (points : ContestPoints) 
  (answers : ContestAnswers) 
  (h1 : points.easy = 2) 
  (h2 : points.average = 3) 
  (h3 : points.hard = 5)
  (h4 : answers.average = 2)
  (h5 : answers.hard = 4)
  (h6 : totalPoints points answers = 38) : 
  answers.easy = 6 := by
sorry

end kim_easy_round_answers_l3148_314883


namespace freds_dark_blue_marbles_l3148_314823

/-- Proves that the number of dark blue marbles is 6 given the conditions of Fred's marble collection. -/
theorem freds_dark_blue_marbles :
  let total_marbles : ℕ := 63
  let red_marbles : ℕ := 38
  let green_marbles : ℕ := red_marbles / 2
  let dark_blue_marbles : ℕ := total_marbles - red_marbles - green_marbles
  dark_blue_marbles = 6 := by
  sorry

end freds_dark_blue_marbles_l3148_314823


namespace cos_2a_given_tan_a_l3148_314865

theorem cos_2a_given_tan_a (a : ℝ) (h : Real.tan a = 2) : Real.cos (2 * a) = -3/5 := by
  sorry

end cos_2a_given_tan_a_l3148_314865


namespace polynomial_division_theorem_l3148_314874

theorem polynomial_division_theorem (x : ℝ) :
  8 * x^3 - 4 * x^2 + 6 * x - 15 = (x - 3) * (8 * x^2 + 20 * x + 66) + 183 := by
  sorry

end polynomial_division_theorem_l3148_314874


namespace sticker_distribution_l3148_314868

/-- The number of ways to distribute n identical objects into k groups,
    with each group containing at least one object -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The problem statement -/
theorem sticker_distribution :
  distribute 10 3 = 36 := by
  sorry

end sticker_distribution_l3148_314868


namespace rhombus_longer_diagonal_l3148_314881

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.diagonal1 * r.diagonal2) / 2

theorem rhombus_longer_diagonal (r : Rhombus) 
  (h1 : r.diagonal1 = 12)
  (h2 : r.area = 120) :
  r.diagonal2 = 20 := by
  sorry

end rhombus_longer_diagonal_l3148_314881


namespace x_squared_congruence_l3148_314804

theorem x_squared_congruence (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 20 [ZMOD 25]) : 
  x^2 ≡ 0 [ZMOD 25] := by
  sorry

end x_squared_congruence_l3148_314804


namespace probability_x_less_than_2y_is_five_sixths_l3148_314811

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The specific rectangle in the problem --/
def problemRectangle : Rectangle where
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 2
  h_x := by norm_num
  h_y := by norm_num

/-- The probability of selecting a point (x,y) from the rectangle such that x < 2y --/
def probabilityXLessThan2Y (r : Rectangle) : ℝ :=
  sorry

theorem probability_x_less_than_2y_is_five_sixths :
  probabilityXLessThan2Y problemRectangle = 5/6 := by
  sorry

end probability_x_less_than_2y_is_five_sixths_l3148_314811


namespace absolute_value_equation_range_l3148_314859

theorem absolute_value_equation_range :
  ∀ x : ℝ, (|3*x - 2| + |3*x + 1| = 3) ↔ (-1/3 ≤ x ∧ x ≤ 2/3) := by
  sorry

end absolute_value_equation_range_l3148_314859


namespace polynomial_equality_l3148_314841

/-- Given a polynomial Q(x) = Q(0) + Q(1)x + Q(3)x^2 where Q(-1) = 2, 
    prove that Q(x) = 0.6x^2 - 2x - 0.6 -/
theorem polynomial_equality (Q : ℝ → ℝ) (h1 : ∀ x, Q x = Q 0 + Q 1 * x + Q 3 * x^2)
    (h2 : Q (-1) = 2) : ∀ x, Q x = 0.6 * x^2 - 2 * x - 0.6 := by
  sorry

end polynomial_equality_l3148_314841


namespace simplify_fraction_l3148_314809

theorem simplify_fraction : (144 : ℚ) / 1008 = 1 / 7 := by sorry

end simplify_fraction_l3148_314809


namespace games_purchased_l3148_314824

theorem games_purchased (total_income : ℕ) (expense : ℕ) (game_cost : ℕ) :
  total_income = 69 →
  expense = 24 →
  game_cost = 5 →
  (total_income - expense) / game_cost = 9 :=
by sorry

end games_purchased_l3148_314824


namespace expression_simplification_l3148_314857

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (((2 * x - 1) / (x + 1) - x + 1) / ((x - 2) / (x^2 + 2*x + 1))) = -2 - Real.sqrt 2 := by
  sorry

end expression_simplification_l3148_314857
