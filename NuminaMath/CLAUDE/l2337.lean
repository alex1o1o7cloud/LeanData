import Mathlib

namespace NUMINAMATH_CALUDE_layoff_plans_count_l2337_233707

def staff_count : ℕ := 10
def layoff_count : ℕ := 4

/-- The number of ways to select 4 people out of 10 for layoff, 
    where two specific people (A and B) cannot both be kept -/
def layoff_plans : ℕ := Nat.choose (staff_count - 2) layoff_count + 
                        2 * Nat.choose (staff_count - 2) (layoff_count - 1)

theorem layoff_plans_count : layoff_plans = 182 := by
  sorry

end NUMINAMATH_CALUDE_layoff_plans_count_l2337_233707


namespace NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_prop_4_false_l2337_233761

-- Define the function f
def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

-- Proposition ①
theorem prop_1 (x : ℝ) : f x 0 0 = -f (-x) 0 0 := by sorry

-- Proposition ②
theorem prop_2 : ∃! x : ℝ, f x 0 1 = 0 := by sorry

-- Proposition ③
theorem prop_3 (x b c : ℝ) : f x b c - c = -(f (-x) b c - c) := by sorry

-- Proposition ④ (false)
theorem prop_4_false : ∃ b c : ℝ, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0 := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_prop_4_false_l2337_233761


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2337_233740

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^4 / (a^3 + a^2*b + a*b^2 + b^3) +
   b^4 / (b^3 + b^2*c + b*c^2 + c^3) +
   c^4 / (c^3 + c^2*d + c*d^2 + d^3) +
   d^4 / (d^3 + d^2*a + d*a^2 + a^3)) ≥ (a + b + c + d) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2337_233740


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2337_233742

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b : ℝ),
      a = 2 ∧ b = 5 ∧
      (a + b + b = perimeter ∨ a + a + b = perimeter) ∧
      perimeter = 12

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2337_233742


namespace NUMINAMATH_CALUDE_first_year_interest_rate_l2337_233737

/-- Given an initial amount, time period, interest rates, and final amount,
    calculate the interest rate for the first year. -/
theorem first_year_interest_rate
  (initial_amount : ℝ)
  (time_period : ℕ)
  (second_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_amount = 5000)
  (h2 : time_period = 2)
  (h3 : second_year_rate = 0.25)
  (h4 : final_amount = 7500) :
  ∃ (first_year_rate : ℝ),
    first_year_rate = 0.20 ∧
    final_amount = initial_amount * (1 + first_year_rate) * (1 + second_year_rate) :=
by sorry

end NUMINAMATH_CALUDE_first_year_interest_rate_l2337_233737


namespace NUMINAMATH_CALUDE_birds_landed_l2337_233731

/-- Given an initial number of birds on a fence and a final number of birds on the fence,
    this theorem proves that the number of birds that landed is equal to
    the difference between the final and initial numbers. -/
theorem birds_landed (initial final : ℕ) (h : initial ≤ final) :
  final - initial = final - initial :=
by sorry

end NUMINAMATH_CALUDE_birds_landed_l2337_233731


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l2337_233750

theorem smallest_absolute_value : ∀ (a b c : ℝ),
  a = 4.1 → b = 13 → c = 3 →
  |(-Real.sqrt 7)| < |a| ∧ |(-Real.sqrt 7)| < Real.sqrt b ∧ |(-Real.sqrt 7)| < |c| :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l2337_233750


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l2337_233712

theorem cos_sin_sum_equals_sqrt2_over_2 :
  Real.cos (58 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (58 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l2337_233712


namespace NUMINAMATH_CALUDE_pyramid_side_length_l2337_233708

/-- Regular triangular pyramid with specific properties -/
structure RegularPyramid where
  -- Base triangle side length
  a : ℝ
  -- Angle of inclination of face to base
  α : ℝ
  -- Height of the pyramid
  h : ℝ
  -- Condition that α is arctan(3/4)
  angle_condition : α = Real.arctan (3/4)
  -- Relation between height, side length, and angle
  height_relation : h = (a * Real.sqrt 3) / 2

/-- Polyhedron formed by intersecting prism with pyramid -/
structure Polyhedron (p : RegularPyramid) where
  -- Surface area of the polyhedron
  surface_area : ℝ
  -- Condition that surface area is 53√3
  area_condition : surface_area = 53 * Real.sqrt 3

/-- Theorem stating the side length of the base triangle -/
theorem pyramid_side_length (p : RegularPyramid) (poly : Polyhedron p) :
  p.a = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_side_length_l2337_233708


namespace NUMINAMATH_CALUDE_apple_count_l2337_233704

/-- Represents the total number of apples -/
def total_apples : ℕ := sorry

/-- Represents the price of a sweet apple in dollars -/
def sweet_price : ℚ := 1/2

/-- Represents the price of a sour apple in dollars -/
def sour_price : ℚ := 1/10

/-- Represents the proportion of sweet apples -/
def sweet_proportion : ℚ := 3/4

/-- Represents the proportion of sour apples -/
def sour_proportion : ℚ := 1/4

/-- Represents the total earnings in dollars -/
def total_earnings : ℚ := 40

theorem apple_count : 
  sweet_proportion * total_apples * sweet_price + 
  sour_proportion * total_apples * sour_price = total_earnings ∧
  total_apples = 100 := by sorry

end NUMINAMATH_CALUDE_apple_count_l2337_233704


namespace NUMINAMATH_CALUDE_additional_weight_needed_l2337_233764

/-- Calculates the additional weight needed to open the cave doors -/
theorem additional_weight_needed 
  (set1_weight : ℝ) 
  (set1_count : ℕ) 
  (set2_weight : ℝ) 
  (set2_count : ℕ) 
  (switch_weight : ℝ) 
  (total_needed : ℝ) 
  (large_rock_kg : ℝ) 
  (kg_to_lbs : ℝ) 
  (h1 : set1_weight = 60) 
  (h2 : set1_count = 3) 
  (h3 : set2_weight = 42) 
  (h4 : set2_count = 5) 
  (h5 : switch_weight = 234) 
  (h6 : total_needed = 712) 
  (h7 : large_rock_kg = 12) 
  (h8 : kg_to_lbs = 2.2) : 
  total_needed - (switch_weight + set1_weight * set1_count + set2_weight * set2_count + large_rock_kg * kg_to_lbs) = 61.6 := by
  sorry

#check additional_weight_needed

end NUMINAMATH_CALUDE_additional_weight_needed_l2337_233764


namespace NUMINAMATH_CALUDE_duck_flock_size_l2337_233734

/-- Calculates the total number of ducks in a combined flock after a given number of years -/
def combined_flock_size (initial_size : ℕ) (annual_increase : ℕ) (years : ℕ) (joining_flock : ℕ) : ℕ :=
  initial_size + annual_increase * years + joining_flock

/-- Theorem stating the combined flock size after 5 years -/
theorem duck_flock_size :
  combined_flock_size 100 10 5 150 = 300 := by
  sorry

#eval combined_flock_size 100 10 5 150

end NUMINAMATH_CALUDE_duck_flock_size_l2337_233734


namespace NUMINAMATH_CALUDE_pattern_solution_l2337_233791

theorem pattern_solution (n : ℕ+) (a b : ℕ+) :
  (∀ k : ℕ+, Real.sqrt (k + k / (k^2 - 1)) = k * Real.sqrt (k / (k^2 - 1))) →
  (Real.sqrt (8 + b / a) = 8 * Real.sqrt (b / a)) →
  a = 63 ∧ b = 8 := by sorry

end NUMINAMATH_CALUDE_pattern_solution_l2337_233791


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2337_233748

-- Define the sets
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}
def U : Set ℝ := Set.univ

-- State the theorem
theorem intersection_with_complement :
  P ∩ (U \ Q) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2337_233748


namespace NUMINAMATH_CALUDE_corrected_mean_l2337_233733

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) 
  (h1 : n = 50) 
  (h2 : original_mean = 41) 
  (h3 : incorrect_value = 23) 
  (h4 : correct_value = 48) : 
  (n : ℝ) * original_mean - incorrect_value + correct_value = n * 41.5 := by
  sorry

#check corrected_mean

end NUMINAMATH_CALUDE_corrected_mean_l2337_233733


namespace NUMINAMATH_CALUDE_michael_needs_eleven_more_l2337_233766

/-- Given Michael's current money and the total cost of items he wants to buy,
    calculate the additional money he needs. -/
def additional_money_needed (current_money total_cost : ℕ) : ℕ :=
  if total_cost > current_money then total_cost - current_money else 0

/-- Theorem stating that Michael needs $11 more to buy all items. -/
theorem michael_needs_eleven_more :
  let current_money : ℕ := 50
  let cake_cost : ℕ := 20
  let bouquet_cost : ℕ := 36
  let balloons_cost : ℕ := 5
  let total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
  additional_money_needed current_money total_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_michael_needs_eleven_more_l2337_233766


namespace NUMINAMATH_CALUDE_triangle_side_range_l2337_233741

open Real

theorem triangle_side_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- ABC is an acute triangle
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  Real.sqrt 3 * (a * cos B + b * cos A) = 2 * c * sin C ∧  -- Given equation
  b = 1 →  -- Given condition
  sqrt 3 / 2 < c ∧ c < sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l2337_233741


namespace NUMINAMATH_CALUDE_max_n_minus_sum_digits_l2337_233792

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum value of n satisfying n - S(n) = 2007 is 2019 -/
theorem max_n_minus_sum_digits : 
  ∀ n : ℕ, n - sum_of_digits n = 2007 → n ≤ 2019 ∧ ∃ m : ℕ, m - sum_of_digits m = 2007 ∧ m = 2019 :=
sorry

end NUMINAMATH_CALUDE_max_n_minus_sum_digits_l2337_233792


namespace NUMINAMATH_CALUDE_average_marks_second_class_l2337_233794

theorem average_marks_second_class 
  (students1 : ℕ) 
  (students2 : ℕ) 
  (avg1 : ℝ) 
  (avg_combined : ℝ) :
  students1 = 25 →
  students2 = 40 →
  avg1 = 50 →
  avg_combined = 59.23076923076923 →
  let total_students := students1 + students2
  let avg2 := (avg_combined * total_students - avg1 * students1) / students2
  avg2 = 65 := by sorry

end NUMINAMATH_CALUDE_average_marks_second_class_l2337_233794


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l2337_233720

/-- A trapezoid with the given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The property that the line joining the midpoints of the diagonals is half the difference of the bases -/
def midpoint_property (t : Trapezoid) : Prop :=
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

/-- The theorem to prove -/
theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 120)
  (h2 : t.midpoint_segment = 7)
  (h3 : midpoint_property t) : 
  t.shorter_base = 106 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l2337_233720


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2337_233797

theorem inequality_solution_set (θ : ℝ) (x : ℝ) :
  (|x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) ↔ (-1 ≤ x ∧ x ≤ -Real.cos (2 * θ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2337_233797


namespace NUMINAMATH_CALUDE_symmetry_wrt_origin_l2337_233759

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem symmetry_wrt_origin :
  symmetric_point (4, -1) = (-4, 1) := by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_origin_l2337_233759


namespace NUMINAMATH_CALUDE_correct_simplification_l2337_233749

theorem correct_simplification (a b : ℝ) : 5*a - (b - 1) = 5*a - b + 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_simplification_l2337_233749


namespace NUMINAMATH_CALUDE_zero_division_not_always_zero_l2337_233728

theorem zero_division_not_always_zero : ¬ (∀ a : ℝ, a ≠ 0 → 0 / a = 0) :=
sorry

end NUMINAMATH_CALUDE_zero_division_not_always_zero_l2337_233728


namespace NUMINAMATH_CALUDE_k_range_for_single_extremum_l2337_233795

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x / x + k * (Real.log x - x)

theorem k_range_for_single_extremum (k : ℝ) :
  (∀ x > 0, x ≠ 1 → (deriv (f k)) x ≠ 0) →
  (deriv (f k)) 1 = 0 →
  k ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_k_range_for_single_extremum_l2337_233795


namespace NUMINAMATH_CALUDE_cd_cost_l2337_233757

/-- Given that two identical CDs cost $24, prove that seven CDs cost $84. -/
theorem cd_cost (cost_of_two : ℕ) (h : cost_of_two = 24) : 7 * (cost_of_two / 2) = 84 := by
  sorry

end NUMINAMATH_CALUDE_cd_cost_l2337_233757


namespace NUMINAMATH_CALUDE_square_diagonal_side_perimeter_l2337_233709

theorem square_diagonal_side_perimeter :
  ∀ (d s p : ℝ),
  d = 2 * Real.sqrt 2 →  -- diagonal is 2√2 inches
  d = s * Real.sqrt 2 →  -- relation between diagonal and side in a square
  s = 2 ∧                -- side length is 2 inches
  p = 4 * s              -- perimeter is 4 times the side length
  := by sorry

end NUMINAMATH_CALUDE_square_diagonal_side_perimeter_l2337_233709


namespace NUMINAMATH_CALUDE_inequality_for_negative_reals_l2337_233776

theorem inequality_for_negative_reals (a b : ℝ) : 
  a < b → b < 0 → a + 1/b < b + 1/a := by sorry

end NUMINAMATH_CALUDE_inequality_for_negative_reals_l2337_233776


namespace NUMINAMATH_CALUDE_equation_solutions_l2337_233711

theorem equation_solutions :
  (∃ x : ℝ, (x - 1)^3 = 64 ∧ x = 5) ∧
  (∃ x : ℝ, 25 * x^2 + 3 = 12 ∧ (x = 3/5 ∨ x = -3/5)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2337_233711


namespace NUMINAMATH_CALUDE_roots_reality_l2337_233747

theorem roots_reality (p q : ℝ) (h : p^2 - 4*q > 0) :
  ∀ a : ℝ, (2*a + 3*p)^2 - 4*3*(q + a*p) > 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_reality_l2337_233747


namespace NUMINAMATH_CALUDE_circles_M_N_common_tangents_l2337_233746

/-- Circle M with equation x^2 + y^2 - 4y = 0 -/
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.2 = 0}

/-- Circle N with equation (x - 1)^2 + (y - 1)^2 = 1 -/
def circle_N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- The number of common tangents between two circles -/
def num_common_tangents (C1 C2 : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- Theorem stating that circles M and N have exactly 2 common tangents -/
theorem circles_M_N_common_tangents :
  num_common_tangents circle_M circle_N = 2 :=
sorry

end NUMINAMATH_CALUDE_circles_M_N_common_tangents_l2337_233746


namespace NUMINAMATH_CALUDE_function_properties_l2337_233726

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x * (a * x + b) + x^2 + 2 * x

theorem function_properties (a b : ℝ) :
  (f a b 0 = 1 ∧ (deriv (f a b)) 0 = 4) →
  (a = 1 ∧ b = 1) ∧
  (∀ k, (∀ x ∈ Set.Icc (-2) (-1), f 1 1 x ≥ x^2 + 2*(k+1)*x + k) ↔ 
        k ≥ (1/4) * Real.exp (-3/2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2337_233726


namespace NUMINAMATH_CALUDE_cube_root_equality_l2337_233768

theorem cube_root_equality (a b : ℝ) :
  (a ^ (1/3 : ℝ) = -(b ^ (1/3 : ℝ))) → a = -b := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equality_l2337_233768


namespace NUMINAMATH_CALUDE_opera_house_empty_seats_percentage_l2337_233769

theorem opera_house_empty_seats_percentage
  (total_rows : ℕ)
  (seats_per_row : ℕ)
  (ticket_price : ℕ)
  (earnings : ℕ)
  (h1 : total_rows = 150)
  (h2 : seats_per_row = 10)
  (h3 : ticket_price = 10)
  (h4 : earnings = 12000) :
  (((total_rows * seats_per_row) - (earnings / ticket_price)) * 100) / (total_rows * seats_per_row) = 20 := by
  sorry

#check opera_house_empty_seats_percentage

end NUMINAMATH_CALUDE_opera_house_empty_seats_percentage_l2337_233769


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_l2337_233725

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- Calculates the minimum number of gumballs needed to ensure four of the same color -/
def minGumballsForFourSameColor (machine : GumballMachine) : Nat :=
  sorry

/-- Theorem stating the minimum number of gumballs needed for the given machine -/
theorem min_gumballs_for_four_same_color 
  (machine : GumballMachine) 
  (h_red : machine.red = 10)
  (h_white : machine.white = 8)
  (h_blue : machine.blue = 9)
  (h_green : machine.green = 6) :
  minGumballsForFourSameColor machine = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_l2337_233725


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_of_P_l2337_233785

def M : Finset ℤ := {-1, 1, 2, 3, 4, 5}
def N : Finset ℤ := {1, 2, 4}
def P : Finset ℤ := M ∩ N

theorem number_of_proper_subsets_of_P : (Finset.powerset P).card - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_of_P_l2337_233785


namespace NUMINAMATH_CALUDE_intersection_point_of_linear_system_l2337_233752

theorem intersection_point_of_linear_system (b : ℝ) :
  let eq1 : ℝ → ℝ → Prop := λ x y => x + y - b = 0
  let eq2 : ℝ → ℝ → Prop := λ x y => 3 * x + y - 2 = 0
  let line1 : ℝ → ℝ → Prop := λ x y => y = -x + b
  let line2 : ℝ → ℝ → Prop := λ x y => y = -3 * x + 2
  (∃ m, eq1 (-1) m ∧ eq2 (-1) m) →
  (∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (-1, 5)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_linear_system_l2337_233752


namespace NUMINAMATH_CALUDE_johns_age_l2337_233703

/-- Given that John is 30 years younger than his dad and the sum of their ages is 80,
    prove that John is 25 years old. -/
theorem johns_age (john dad : ℕ) 
  (h1 : john = dad - 30)
  (h2 : john + dad = 80) : 
  john = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l2337_233703


namespace NUMINAMATH_CALUDE_survey_respondents_l2337_233718

theorem survey_respondents (x y : ℕ) : 
  x = 60 → -- 60 people preferred brand X
  x = 3 * y → -- The ratio of preference for X to Y is 3:1
  x + y = 80 -- Total number of respondents
  :=
by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l2337_233718


namespace NUMINAMATH_CALUDE_books_read_total_l2337_233790

theorem books_read_total (may june july : ℕ) 
  (h_may : may = 2) 
  (h_june : june = 6) 
  (h_july : july = 10) : 
  may + june + july = 18 := by
  sorry

end NUMINAMATH_CALUDE_books_read_total_l2337_233790


namespace NUMINAMATH_CALUDE_all_parameterizations_valid_l2337_233715

/-- The slope of the line -/
def m : ℝ := -3

/-- The y-intercept of the line -/
def b : ℝ := 4

/-- The line equation: y = mx + b -/
def on_line (x y : ℝ) : Prop := y = m * x + b

/-- A parameterization is valid if it satisfies the line equation for all t -/
def valid_parameterization (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, on_line (p.1 + t * v.1) (p.2 + t * v.2)

/-- Theorem: All given parameterizations are valid -/
theorem all_parameterizations_valid :
  valid_parameterization (0, 4) (1, -3) ∧
  valid_parameterization (-2/3, 0) (3, -9) ∧
  valid_parameterization (-4/3, 8) (2, -6) ∧
  valid_parameterization (-2, 10) (1/2, -1) ∧
  valid_parameterization (1, 1) (4, -12) :=
sorry

end NUMINAMATH_CALUDE_all_parameterizations_valid_l2337_233715


namespace NUMINAMATH_CALUDE_sequence_median_l2337_233724

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sequence_median : 
  let total_elements := sequence_sum 100
  let median_position := total_elements / 2
  ∃ k : ℕ, 
    k ≤ 100 ∧ 
    sequence_sum (k - 1) < median_position ∧ 
    median_position ≤ sequence_sum k ∧
    k = 71 := by sorry

end NUMINAMATH_CALUDE_sequence_median_l2337_233724


namespace NUMINAMATH_CALUDE_prob_four_of_a_kind_after_reroll_l2337_233739

/-- Represents the outcome of rolling five dice -/
structure DiceRoll where
  pairs : Nat -- Number of pairs
  fourOfAKind : Bool -- Whether there's a four-of-a-kind

/-- Represents the possible outcomes after re-rolling the fifth die -/
inductive ReRollOutcome
  | fourOfAKind : ReRollOutcome
  | nothingSpecial : ReRollOutcome

/-- The probability of getting at least four of a kind after re-rolling -/
def probFourOfAKind (initialRoll : DiceRoll) : ℚ :=
  sorry

theorem prob_four_of_a_kind_after_reroll :
  ∀ (initialRoll : DiceRoll),
    initialRoll.pairs = 2 ∧ ¬initialRoll.fourOfAKind →
    probFourOfAKind initialRoll = 1 / 3 :=
  sorry

end NUMINAMATH_CALUDE_prob_four_of_a_kind_after_reroll_l2337_233739


namespace NUMINAMATH_CALUDE_discounted_cost_six_books_l2337_233744

/-- The cost of three identical books -/
def cost_three_books : ℚ := 45

/-- The number of books in the discounted purchase -/
def num_books_discounted : ℕ := 6

/-- The discount rate applied when buying six books -/
def discount_rate : ℚ := 1 / 10

/-- The cost of six books with a 10% discount, given that three identical books cost $45 -/
theorem discounted_cost_six_books : 
  (num_books_discounted : ℚ) * (cost_three_books / 3) * (1 - discount_rate) = 81 := by
  sorry

end NUMINAMATH_CALUDE_discounted_cost_six_books_l2337_233744


namespace NUMINAMATH_CALUDE_millet_percentage_in_mix_l2337_233745

/-- Theorem: Millet percentage in a birdseed mix -/
theorem millet_percentage_in_mix
  (brand_a_millet : ℝ)
  (brand_b_millet : ℝ)
  (mix_brand_a : ℝ)
  (h1 : brand_a_millet = 0.60)
  (h2 : brand_b_millet = 0.65)
  (h3 : mix_brand_a = 0.60)
  (h4 : 0 ≤ mix_brand_a ∧ mix_brand_a ≤ 1) :
  mix_brand_a * brand_a_millet + (1 - mix_brand_a) * brand_b_millet = 0.62 := by
  sorry


end NUMINAMATH_CALUDE_millet_percentage_in_mix_l2337_233745


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l2337_233771

theorem complex_sum_theorem (a c d e f g : ℝ) : 
  let b : ℝ := 5
  let e : ℝ := -(a + c) + g
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = g + 9 * Complex.I →
  d + f = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l2337_233771


namespace NUMINAMATH_CALUDE_smallest_sum_x_y_l2337_233786

theorem smallest_sum_x_y (x y : ℕ+) 
  (h1 : (2010 : ℚ) / 2011 < (x : ℚ) / y)
  (h2 : (x : ℚ) / y < (2011 : ℚ) / 2012) :
  ∀ (a b : ℕ+), 
    ((2010 : ℚ) / 2011 < (a : ℚ) / b ∧ (a : ℚ) / b < (2011 : ℚ) / 2012) →
    (x + y : ℕ) ≤ (a + b : ℕ) ∧
    (x + y : ℕ) = 8044 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_x_y_l2337_233786


namespace NUMINAMATH_CALUDE_triathlon_problem_l2337_233743

/-- Triathlon problem -/
theorem triathlon_problem (v₁ v₂ v₃ : ℝ) 
  (h1 : 1 / v₁ + 25 / v₂ + 4 / v₃ = 5 / 4)
  (h2 : v₁ / 16 + v₂ / 49 + v₃ / 49 = 5 / 4) :
  v₃ = 14 ∧ 4 / v₃ = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_problem_l2337_233743


namespace NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l2337_233755

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 202 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 202 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l2337_233755


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2337_233719

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let area_small : ℝ := π * r₁^2  -- area of smaller circle
  let area_large : ℝ := π * r₂^2  -- area of larger circle
  let area_between : ℝ := area_large - area_small  -- area between circles
  (area_between / area_small) = 8 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2337_233719


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l2337_233770

theorem complex_product_real_imag_parts : ∃ (m n : ℝ), 
  let Z : ℂ := (1 + Complex.I) * (2 + Complex.I^607)
  m = Z.re ∧ n = Z.im ∧ m * n = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l2337_233770


namespace NUMINAMATH_CALUDE_polyhedron_sum_l2337_233730

/-- A convex polyhedron with triangular, pentagonal, and hexagonal faces. -/
structure Polyhedron where
  T : ℕ  -- Number of triangular faces
  P : ℕ  -- Number of pentagonal faces
  H : ℕ  -- Number of hexagonal faces
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges

/-- Properties of the polyhedron -/
def is_valid_polyhedron (poly : Polyhedron) : Prop :=
  -- Total number of faces is 42
  poly.T + poly.P + poly.H = 42 ∧
  -- At each vertex, 3 triangular, 2 pentagonal, and 1 hexagonal face meet
  6 * poly.V = 3 * poly.T + 2 * poly.P + poly.H ∧
  -- Edge count
  2 * poly.E = 3 * poly.T + 5 * poly.P + 6 * poly.H ∧
  -- Euler's formula
  poly.V - poly.E + (poly.T + poly.P + poly.H) = 2

/-- Theorem statement -/
theorem polyhedron_sum (poly : Polyhedron) 
  (h : is_valid_polyhedron poly) : 
  100 * poly.H + 10 * poly.P + poly.T + poly.V = 714 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_sum_l2337_233730


namespace NUMINAMATH_CALUDE_base4_divisibility_by_17_l2337_233722

def base4_to_decimal (a b c d : ℕ) : ℕ :=
  a * 4^3 + b * 4^2 + c * 4^1 + d * 4^0

def is_base4_digit (x : ℕ) : Prop :=
  x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3

theorem base4_divisibility_by_17 (x : ℕ) :
  is_base4_digit x →
  (base4_to_decimal 2 3 x 2 ∣ 17) ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_base4_divisibility_by_17_l2337_233722


namespace NUMINAMATH_CALUDE_spade_problem_l2337_233732

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 5 (spade 3 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spade_problem_l2337_233732


namespace NUMINAMATH_CALUDE_right_triangle_area_l2337_233706

theorem right_triangle_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a^2 + b^2 = 10^2 → a + b + 10 = 24 → (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2337_233706


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_five_l2337_233700

theorem smallest_four_digit_mod_five : ∃ n : ℕ,
  (n ≥ 1000) ∧                 -- four-digit number
  (n < 10000) ∧                -- four-digit number
  (n % 5 = 4) ∧                -- equivalent to 4 mod 5
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 5 = 4 → m ≥ n) ∧  -- smallest such number
  (n = 1004) := by             -- the answer is 1004
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_five_l2337_233700


namespace NUMINAMATH_CALUDE_julian_comic_frames_l2337_233705

/-- Calculates the total number of frames in Julian's comic book --/
def total_frames (total_pages : Nat) (avg_frames : Nat) (pages_305 : Nat) (pages_250 : Nat) : Nat :=
  let frames_305 := pages_305 * 305
  let frames_250 := pages_250 * 250
  let remaining_pages := total_pages - pages_305 - pages_250
  let frames_avg := remaining_pages * avg_frames
  frames_305 + frames_250 + frames_avg

/-- Proves that the total number of frames in Julian's comic book is 7040 --/
theorem julian_comic_frames :
  total_frames 25 280 10 7 = 7040 := by
  sorry

end NUMINAMATH_CALUDE_julian_comic_frames_l2337_233705


namespace NUMINAMATH_CALUDE_unit_digit_of_large_exponentiation_l2337_233721

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_of_large_exponentiation : 
  unit_digit ((23^100000 * 56^150000) / Nat.gcd 23 56) = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_large_exponentiation_l2337_233721


namespace NUMINAMATH_CALUDE_hyperbola_intersection_line_l2337_233723

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define eccentricity
def e : ℝ := 2

-- Define point M
def M : ℝ × ℝ := (1, 3)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem hyperbola_intersection_line :
  ∀ A B : ℝ × ℝ,
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 →  -- A and B are on the hyperbola
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is midpoint of AB
  line_l A.1 A.2 ∧ line_l B.1 B.2 →  -- A and B are on line l
  ∀ x y : ℝ, line_l x y ↔ y = x + 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_line_l2337_233723


namespace NUMINAMATH_CALUDE_fraction_addition_l2337_233787

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2337_233787


namespace NUMINAMATH_CALUDE_problem_2023_l2337_233714

theorem problem_2023 : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_problem_2023_l2337_233714


namespace NUMINAMATH_CALUDE_wire_length_proof_l2337_233782

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 20 →
  shorter_piece = (2 / 7) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 90 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l2337_233782


namespace NUMINAMATH_CALUDE_bakery_pies_relation_l2337_233727

/-- The number of pies Mcgee's Bakery sold -/
def mcgees_pies : ℕ := 16

/-- The number of pies Smith's Bakery sold -/
def smiths_pies : ℕ := 70

/-- The difference between Smith's pies and the multiple of Mcgee's pies -/
def difference : ℕ := 6

/-- The multiple of Mcgee's pies related to Smith's pies -/
def multiple : ℕ := 4

theorem bakery_pies_relation :
  multiple * mcgees_pies + difference = smiths_pies :=
by sorry

end NUMINAMATH_CALUDE_bakery_pies_relation_l2337_233727


namespace NUMINAMATH_CALUDE_line_parallel_from_plane_parallel_l2337_233774

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the parallelism relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_from_plane_parallel
  (a b : Line) (α β γ δ : Plane)
  (h_distinct_lines : a ≠ b)
  (h_distinct_planes : α ≠ β ∧ α ≠ γ ∧ α ≠ δ ∧ β ≠ γ ∧ β ≠ δ ∧ γ ≠ δ)
  (h_intersect_ab : intersect α β = a)
  (h_intersect_gd : intersect γ δ = b)
  (h_parallel_ag : planeParallel α γ)
  (h_parallel_bd : planeParallel β δ) :
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_from_plane_parallel_l2337_233774


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2337_233716

theorem complex_modulus_problem (z : ℂ) (h : z * (2 - Complex.I) = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2337_233716


namespace NUMINAMATH_CALUDE_not_sum_of_three_squares_l2337_233751

theorem not_sum_of_three_squares (n : ℕ) : ¬ ∃ (a b c : ℕ+), (8 * n - 1 : ℤ) = a ^ 2 + b ^ 2 + c ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_three_squares_l2337_233751


namespace NUMINAMATH_CALUDE_exists_2x2_square_after_removal_l2337_233781

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a two-cell rectangle (domino) -/
structure Domino where
  cell1 : Cell
  cell2 : Cell

/-- The grid size -/
def gridSize : Nat := 100

/-- The number of dominoes removed -/
def removedDominoes : Nat := 1950

/-- Function to check if a cell is within the grid -/
def isValidCell (c : Cell) : Prop :=
  c.row < gridSize ∧ c.col < gridSize

/-- Function to check if two cells are adjacent -/
def areAdjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col + 1 = c2.col ∨ c2.col + 1 = c1.col)) ∨
  (c1.col = c2.col ∧ (c1.row + 1 = c2.row ∨ c2.row + 1 = c1.row))

/-- Function to check if a domino is valid -/
def isValidDomino (d : Domino) : Prop :=
  isValidCell d.cell1 ∧ isValidCell d.cell2 ∧ areAdjacent d.cell1 d.cell2

/-- Theorem: After removing 1950 dominoes, there exists a 2x2 square in the remaining cells -/
theorem exists_2x2_square_after_removal 
  (removed : Finset Domino) 
  (h_removed : removed.card = removedDominoes) 
  (h_valid : ∀ d ∈ removed, isValidDomino d) :
  ∃ (c : Cell), isValidCell c ∧ 
    isValidCell { row := c.row, col := c.col + 1 } ∧ 
    isValidCell { row := c.row + 1, col := c.col } ∧ 
    isValidCell { row := c.row + 1, col := c.col + 1 } ∧
    (∀ d ∈ removed, d.cell1 ≠ c ∧ d.cell2 ≠ c) ∧
    (∀ d ∈ removed, d.cell1 ≠ { row := c.row, col := c.col + 1 } ∧ d.cell2 ≠ { row := c.row, col := c.col + 1 }) ∧
    (∀ d ∈ removed, d.cell1 ≠ { row := c.row + 1, col := c.col } ∧ d.cell2 ≠ { row := c.row + 1, col := c.col }) ∧
    (∀ d ∈ removed, d.cell1 ≠ { row := c.row + 1, col := c.col + 1 } ∧ d.cell2 ≠ { row := c.row + 1, col := c.col + 1 }) :=
by sorry

end NUMINAMATH_CALUDE_exists_2x2_square_after_removal_l2337_233781


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l2337_233767

/-- Represents a department in the unit -/
inductive Department
| A
| B
| C

/-- The number of employees in each department -/
def employeeCount (d : Department) : ℕ :=
  match d with
  | .A => 27
  | .B => 63
  | .C => 81

/-- The number of people drawn from department B -/
def drawnFromB : ℕ := 7

/-- The number of people drawn from a department in stratified sampling -/
def peopleDrawn (d : Department) : ℚ :=
  (employeeCount d : ℚ) * (drawnFromB : ℚ) / (employeeCount .B : ℚ)

/-- The total number of people drawn from all departments -/
def totalDrawn : ℚ :=
  peopleDrawn .A + peopleDrawn .B + peopleDrawn .C

theorem stratified_sampling_result :
  totalDrawn = 23 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l2337_233767


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2337_233760

/-- 
Theorem: For the quadratic equation x^2 - 6x + k = 0 to have two distinct real roots, k must be less than 9.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + k = 0 ∧ y^2 - 6*y + k = 0) → k < 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2337_233760


namespace NUMINAMATH_CALUDE_two_digit_addition_puzzle_l2337_233793

theorem two_digit_addition_puzzle :
  ∀ (A B : ℕ),
    A ≠ B →
    A < 10 →
    B < 10 →
    10 * A + B + 25 = 10 * B + 3 →
    B = 8 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_addition_puzzle_l2337_233793


namespace NUMINAMATH_CALUDE_function_extrema_l2337_233773

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - log x

theorem function_extrema (a b : ℝ) :
  (a = -1 ∧ b = 3 →
    (∃ (max min : ℝ),
      (∀ x ∈ Set.Icc (1/2) 2, f a b x ≤ max) ∧
      (∃ x ∈ Set.Icc (1/2) 2, f a b x = max) ∧
      (∀ x ∈ Set.Icc (1/2) 2, f a b x ≥ min) ∧
      (∃ x ∈ Set.Icc (1/2) 2, f a b x = min) ∧
      max = 2 ∧
      min = log 2 + 5/4)) ∧
  (a = 0 →
    (∃! b : ℝ,
      b > 0 ∧
      (∃ min : ℝ,
        (∀ x ∈ Set.Ioo 0 (exp 1), f a b x ≥ min) ∧
        (∃ x ∈ Set.Ioo 0 (exp 1), f a b x = min) ∧
        min = 3) ∧
      b = exp 2)) :=
sorry

end NUMINAMATH_CALUDE_function_extrema_l2337_233773


namespace NUMINAMATH_CALUDE_odd_heads_probability_l2337_233754

/-- The probability of getting heads for the kth coin -/
def p (k : ℕ) : ℚ := 1 / (2 * k + 1)

/-- The probability of getting an odd number of heads when tossing n biased coins -/
def odd_heads_prob (n : ℕ) : ℚ := n / (2 * n + 1)

/-- Theorem stating that the probability of getting an odd number of heads
    when tossing n biased coins is n/(2n+1), where the kth coin has
    probability 1/(2k+1) of falling heads -/
theorem odd_heads_probability (n : ℕ) :
  (∀ k, k ≤ n → p k = 1 / (2 * k + 1)) →
  odd_heads_prob n = n / (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_heads_probability_l2337_233754


namespace NUMINAMATH_CALUDE_shop_earnings_l2337_233762

theorem shop_earnings : 
  let cola_price : ℚ := 3
  let juice_price : ℚ := 3/2
  let water_price : ℚ := 1
  let cola_sold : ℕ := 15
  let juice_sold : ℕ := 12
  let water_sold : ℕ := 25
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold = 88
  := by sorry

end NUMINAMATH_CALUDE_shop_earnings_l2337_233762


namespace NUMINAMATH_CALUDE_problem_statement_l2337_233780

theorem problem_statement (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! d : ℝ, d > 0 ∧ 1 / (a + d) + 1 / (b + d) + 1 / (c + d) = 2 / d ∧
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → a * x + b * y + c * z = x * y * z →
    x + y + z ≥ (2 / d) * Real.sqrt ((a + d) * (b + d) * (c + d)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2337_233780


namespace NUMINAMATH_CALUDE_probability_sum_30_l2337_233717

/-- Represents a 20-faced die with specific numbering --/
structure Die :=
  (faces : Finset ℕ)
  (blank_face : Bool)
  (fair : Bool)
  (face_count : faces.card + (if blank_face then 1 else 0) = 20)

/-- Die 1 with faces numbered 1-18 and one blank face --/
def die1 : Die :=
  { faces := Finset.range 19 \ {0},
    blank_face := true,
    fair := true,
    face_count := sorry }

/-- Die 2 with faces numbered 1-9 and 11-20 and one blank face --/
def die2 : Die :=
  { faces := (Finset.range 21 \ {0, 10}),
    blank_face := true,
    fair := true,
    face_count := sorry }

/-- The probability of an event given the number of favorable outcomes and total outcomes --/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

/-- The number of ways to roll a sum of 30 with the given dice --/
def favorable_outcomes : ℕ := 8

/-- The total number of possible outcomes when rolling two 20-faced dice --/
def total_outcomes : ℕ := 400

/-- The main theorem: probability of rolling a sum of 30 is 1/50 --/
theorem probability_sum_30 :
  probability favorable_outcomes total_outcomes = 1 / 50 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_30_l2337_233717


namespace NUMINAMATH_CALUDE_average_problem_l2337_233788

theorem average_problem (a b c d P : ℝ) :
  (a + b + c + d) / 4 = 8 →
  (a + b + c + d + P) / 5 = P →
  P = 8 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l2337_233788


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2337_233783

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2337_233783


namespace NUMINAMATH_CALUDE_cuboid_height_from_volume_and_base_area_l2337_233735

/-- Represents the properties of a cuboid -/
structure Cuboid where
  volume : ℝ
  baseArea : ℝ
  height : ℝ

/-- Theorem stating that a cuboid with volume 144 and base area 18 has height 8 -/
theorem cuboid_height_from_volume_and_base_area :
  ∀ (c : Cuboid), c.volume = 144 → c.baseArea = 18 → c.height = 8 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_from_volume_and_base_area_l2337_233735


namespace NUMINAMATH_CALUDE_Q_R_mutually_exclusive_l2337_233713

-- Define the sample space
structure Outcome :=
  (first : Bool) -- true for black, false for white
  (second : Bool)

-- Define the probability space
def Ω : Type := Outcome

-- Define the events
def P (ω : Ω) : Prop := ω.first ∧ ω.second
def Q (ω : Ω) : Prop := ¬ω.first ∧ ¬ω.second
def R (ω : Ω) : Prop := ω.first ∨ ω.second

-- State the theorem
theorem Q_R_mutually_exclusive : ∀ (ω : Ω), ¬(Q ω ∧ R ω) := by
  sorry

end NUMINAMATH_CALUDE_Q_R_mutually_exclusive_l2337_233713


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_1_l2337_233784

theorem min_value_of_3a_plus_1 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) :
  ∃ (min_val : ℝ), min_val = -5/4 ∧ ∀ (x : ℝ), 8 * x^2 + 6 * x + 5 = 2 → 3 * x + 1 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_1_l2337_233784


namespace NUMINAMATH_CALUDE_pole_height_l2337_233779

theorem pole_height (cable_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ)
  (h1 : cable_ground_distance = 5)
  (h2 : person_distance = 4)
  (h3 : person_height = 3) :
  let pole_height := cable_ground_distance * person_height / (cable_ground_distance - person_distance)
  pole_height = 15 := by sorry

end NUMINAMATH_CALUDE_pole_height_l2337_233779


namespace NUMINAMATH_CALUDE_negative_negative_eight_properties_l2337_233729

theorem negative_negative_eight_properties :
  let x : ℤ := -8
  let y : ℤ := -(-x)
  (y = -x) ∧ 
  (y = -1 * x) ∧ 
  (y = |x|) ∧ 
  (y = 8) := by sorry

end NUMINAMATH_CALUDE_negative_negative_eight_properties_l2337_233729


namespace NUMINAMATH_CALUDE_min_modulus_m_for_real_roots_l2337_233775

/-- Given a complex number m such that the equation x^2 + mx + 1 + 2i = 0 has real roots,
    the minimum value of |m| is sqrt(2 + 2sqrt(5)). -/
theorem min_modulus_m_for_real_roots (m : ℂ) : 
  (∃ x : ℝ, x^2 + m * x + (1 : ℂ) + 2*I = 0) → 
  ∀ m' : ℂ, (∃ x : ℝ, x^2 + m' * x + (1 : ℂ) + 2*I = 0) → Complex.abs m ≤ Complex.abs m' → 
  Complex.abs m = Real.sqrt (2 + 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_min_modulus_m_for_real_roots_l2337_233775


namespace NUMINAMATH_CALUDE_basketball_game_properties_l2337_233702

/-- Represents the score of player A in a single round -/
inductive Score
  | Minus : Score  -- A loses the round
  | Zero : Score   -- Tie in the round
  | Plus : Score   -- A wins the round

/-- Represents the number of rounds played -/
inductive Rounds
  | Two : Rounds
  | Three : Rounds
  | Four : Rounds

/-- The basketball shooting game between A and B -/
structure BasketballGame where
  a_accuracy : ℝ
  b_accuracy : ℝ
  max_rounds : ℕ
  win_difference : ℕ

/-- The probability distribution of the score in a single round -/
def score_distribution (game : BasketballGame) : Score → ℝ
  | Score.Minus => game.b_accuracy * (1 - game.a_accuracy)
  | Score.Zero => game.a_accuracy * game.b_accuracy + (1 - game.a_accuracy) * (1 - game.b_accuracy)
  | Score.Plus => game.a_accuracy * (1 - game.b_accuracy)

/-- The probability of a tie in the game -/
def tie_probability (game : BasketballGame) : ℝ := sorry

/-- The probability distribution of the number of rounds played -/
def rounds_distribution (game : BasketballGame) : Rounds → ℝ
  | Rounds.Two => sorry
  | Rounds.Three => sorry
  | Rounds.Four => sorry

/-- The expected number of rounds played -/
def expected_rounds (game : BasketballGame) : ℝ := sorry

theorem basketball_game_properties (game : BasketballGame) 
  (h1 : game.a_accuracy = 0.5)
  (h2 : game.b_accuracy = 0.6)
  (h3 : game.max_rounds = 4)
  (h4 : game.win_difference = 4) :
  score_distribution game Score.Minus = 0.3 ∧ 
  score_distribution game Score.Zero = 0.5 ∧
  score_distribution game Score.Plus = 0.2 ∧
  tie_probability game = 0.2569 ∧
  rounds_distribution game Rounds.Two = 0.13 ∧
  rounds_distribution game Rounds.Three = 0.13 ∧
  rounds_distribution game Rounds.Four = 0.74 ∧
  expected_rounds game = 3.61 := by sorry

end NUMINAMATH_CALUDE_basketball_game_properties_l2337_233702


namespace NUMINAMATH_CALUDE_arithmetic_progression_quartic_l2337_233796

theorem arithmetic_progression_quartic (q : ℝ) : 
  (∃ (a d : ℝ), ∀ (x : ℝ), x^4 - 40*x^2 + q = 0 ↔ 
    (x = a - 3*d/2 ∨ x = a - d/2 ∨ x = a + d/2 ∨ x = a + 3*d/2)) → 
  q = 144 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_progression_quartic_l2337_233796


namespace NUMINAMATH_CALUDE_laptop_cost_ratio_l2337_233756

theorem laptop_cost_ratio : 
  ∀ (first_laptop_cost second_laptop_cost : ℝ),
    first_laptop_cost = 500 →
    first_laptop_cost + second_laptop_cost = 2000 →
    second_laptop_cost / first_laptop_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_laptop_cost_ratio_l2337_233756


namespace NUMINAMATH_CALUDE_cost_comparison_l2337_233753

/-- The price of a suit in yuan -/
def suit_price : ℕ := 1000

/-- The price of a tie in yuan -/
def tie_price : ℕ := 200

/-- The number of suits to be purchased -/
def num_suits : ℕ := 20

/-- The discount rate for Option 2 -/
def discount_rate : ℚ := 9/10

/-- The cost calculation for Option 1 -/
def option1_cost (x : ℕ) : ℕ := 
  num_suits * suit_price + (x - num_suits) * tie_price

/-- The cost calculation for Option 2 -/
def option2_cost (x : ℕ) : ℚ := 
  discount_rate * (num_suits * suit_price + x * tie_price)

theorem cost_comparison (x : ℕ) (h : x > num_suits) : 
  option1_cost x = 200 * x + 16000 ∧ 
  option2_cost x = 180 * x + 18000 := by
  sorry

#check cost_comparison

end NUMINAMATH_CALUDE_cost_comparison_l2337_233753


namespace NUMINAMATH_CALUDE_perfectSquareFactorsOf1800_l2337_233738

/-- The number of positive factors of 1800 that are perfect squares -/
def perfectSquareFactors : ℕ := 8

/-- 1800 as a natural number -/
def n : ℕ := 1800

/-- A function that returns the number of positive factors of a natural number that are perfect squares -/
def countPerfectSquareFactors (m : ℕ) : ℕ := sorry

theorem perfectSquareFactorsOf1800 : countPerfectSquareFactors n = perfectSquareFactors := by sorry

end NUMINAMATH_CALUDE_perfectSquareFactorsOf1800_l2337_233738


namespace NUMINAMATH_CALUDE_cookies_in_box_proof_l2337_233701

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 7

/-- The number of boxes -/
def num_boxes : ℕ := 8

/-- The number of bags -/
def num_bags : ℕ := 9

/-- The additional number of cookies in boxes compared to bags -/
def additional_cookies : ℕ := 33

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 12

theorem cookies_in_box_proof :
  num_boxes * cookies_per_box = num_bags * cookies_per_bag + additional_cookies :=
sorry

end NUMINAMATH_CALUDE_cookies_in_box_proof_l2337_233701


namespace NUMINAMATH_CALUDE_max_term_T_l2337_233777

def geometric_sequence (a₁ : ℚ) (q : ℚ) : ℕ+ → ℚ :=
  fun n => a₁ * q ^ (n.val - 1)

def sum_geometric_sequence (a₁ : ℚ) (q : ℚ) : ℕ+ → ℚ :=
  fun n => a₁ * (1 - q^n.val) / (1 - q)

def T (S : ℕ+ → ℚ) : ℕ+ → ℚ :=
  fun n => S n + 1 / (S n)

theorem max_term_T 
  (a : ℕ+ → ℚ)
  (S : ℕ+ → ℚ)
  (h₁ : a 1 = 3/2)
  (h₂ : ∀ n, S n = sum_geometric_sequence (a 1) (-1/2) n)
  (h₃ : -2*(S 2) + 4*(S 4) = 2*(S 3))
  : ∀ n, T S n ≤ 13/6 ∧ T S 1 = 13/6 :=
sorry

end NUMINAMATH_CALUDE_max_term_T_l2337_233777


namespace NUMINAMATH_CALUDE_wall_cleaning_time_l2337_233710

/-- Represents the cleaning rate in minutes per section -/
def cleaning_rate (time_spent : ℕ) (sections_cleaned : ℕ) : ℚ :=
  (time_spent : ℚ) / sections_cleaned

/-- Calculates the remaining time to clean the wall -/
def remaining_time (total_sections : ℕ) (cleaned_sections : ℕ) (rate : ℚ) : ℚ :=
  ((total_sections - cleaned_sections) : ℚ) * rate

/-- Theorem stating the remaining time to clean the wall -/
theorem wall_cleaning_time (total_sections : ℕ) (cleaned_sections : ℕ) (time_spent : ℕ) :
  total_sections = 18 ∧ cleaned_sections = 3 ∧ time_spent = 33 →
  remaining_time total_sections cleaned_sections (cleaning_rate time_spent cleaned_sections) = 165 := by
  sorry

end NUMINAMATH_CALUDE_wall_cleaning_time_l2337_233710


namespace NUMINAMATH_CALUDE_ellipse_vertices_distance_l2337_233772

/-- Given an ellipse with equation (x^2 / 45) + (y^2 / 11) = 1, 
    the distance between its vertices is 6√5 -/
theorem ellipse_vertices_distance : 
  ∀ (x y : ℝ), x^2/45 + y^2/11 = 1 → 
  ∃ (d : ℝ), d = 6 * Real.sqrt 5 ∧ d = 2 * Real.sqrt (max 45 11) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_vertices_distance_l2337_233772


namespace NUMINAMATH_CALUDE_towel_shrinkage_l2337_233799

/-- Given a rectangular towel that loses 20% of its length and has a total area
    decrease of 27.999999999999993%, the percentage decrease in breadth is 10%. -/
theorem towel_shrinkage (L B : ℝ) (L' B' : ℝ) : 
  L' = 0.8 * L →
  L' * B' = 0.72 * (L * B) →
  B' = 0.9 * B :=
by sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l2337_233799


namespace NUMINAMATH_CALUDE_school_distance_proof_l2337_233736

/-- The distance to school in miles -/
def distance_to_school : ℝ := 5

/-- The speed of walking in miles per hour for the first scenario -/
def speed1 : ℝ := 4

/-- The speed of walking in miles per hour for the second scenario -/
def speed2 : ℝ := 5

/-- The time difference in hours between arriving early and late -/
def time_difference : ℝ := 0.25

theorem school_distance_proof :
  (distance_to_school / speed1 - distance_to_school / speed2 = time_difference) ∧
  (distance_to_school = 5) := by
  sorry

end NUMINAMATH_CALUDE_school_distance_proof_l2337_233736


namespace NUMINAMATH_CALUDE_square_area_with_perimeter_40_l2337_233758

theorem square_area_with_perimeter_40 :
  ∀ s : ℝ, s > 0 → 4 * s = 40 → s * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_perimeter_40_l2337_233758


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2337_233763

theorem product_sum_theorem :
  ∀ (a b c d : ℝ),
  (∀ x : ℝ, (5 * x^2 - 3 * x + 7) * (9 - 4 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = -29 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2337_233763


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2337_233789

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2337_233789


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2337_233778

/-- A quadratic expression in x and y with a parameter k -/
def quadratic (x y : ℝ) (k : ℝ) : ℝ := 2 * x^2 - 6 * y^2 + x * y + k * x + 6

/-- Predicate to check if an expression is a product of two linear factors -/
def is_product_of_linear_factors (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c d e g : ℝ), ∀ x y, f x y = (a * x + b * y + c) * (d * x + e * y + g)

/-- Theorem stating that if the quadratic expression is factorizable, then k = 7 or k = -7 -/
theorem quadratic_factorization (k : ℝ) :
  is_product_of_linear_factors (quadratic · · k) → k = 7 ∨ k = -7 :=
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2337_233778


namespace NUMINAMATH_CALUDE_contestant_final_score_l2337_233798

/-- Calculates the final score of a contestant given their individual scores and weightings -/
def final_score (etiquette_score language_score behavior_score : ℝ)
  (etiquette_weight language_weight behavior_weight : ℝ) : ℝ :=
  etiquette_score * etiquette_weight +
  language_score * language_weight +
  behavior_score * behavior_weight

/-- Theorem stating that the contestant's final score is 89 points -/
theorem contestant_final_score :
  final_score 95 92 80 0.4 0.25 0.35 = 89 := by
  sorry

end NUMINAMATH_CALUDE_contestant_final_score_l2337_233798


namespace NUMINAMATH_CALUDE_minkowski_sum_properties_l2337_233765

/-- A convex polygon with perimeter and area -/
structure ConvexPolygon where
  perimeter : ℝ
  area : ℝ

/-- The Minkowski sum of a convex polygon and a circle -/
def minkowskiSum (K : ConvexPolygon) (r : ℝ) : Set (ℝ × ℝ) := sorry

/-- The length of the curve resulting from the Minkowski sum -/
def curveLength (K : ConvexPolygon) (r : ℝ) : ℝ := sorry

/-- The area of the figure bounded by the Minkowski sum -/
def boundedArea (K : ConvexPolygon) (r : ℝ) : ℝ := sorry

/-- Main theorem about the Minkowski sum of a convex polygon and a circle -/
theorem minkowski_sum_properties (K : ConvexPolygon) (r : ℝ) :
  (curveLength K r = K.perimeter + 2 * Real.pi * r) ∧
  (boundedArea K r = K.area + K.perimeter * r + Real.pi * r^2) := by
  sorry

end NUMINAMATH_CALUDE_minkowski_sum_properties_l2337_233765
