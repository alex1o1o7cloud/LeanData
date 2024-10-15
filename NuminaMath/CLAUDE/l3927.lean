import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l3927_392711

/-- Given a right-angled triangle with catheti of lengths a and b, and a circle
    with radius r inscribed such that it touches both catheti and has its center
    on the hypotenuse, prove that 1/a + 1/b = 1/r. -/
theorem inscribed_circle_theorem (a b r : ℝ) 
    (ha : a > 0) (hb : b > 0) (hr : r > 0)
    (h_right_triangle : ∃ c, a^2 + b^2 = c^2)
    (h_circle_inscribed : ∃ x y, x^2 + y^2 = r^2 ∧ x + y = r ∧ x < a ∧ y < b) :
    1/a + 1/b = 1/r := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l3927_392711


namespace NUMINAMATH_CALUDE_problem_solution_l3927_392702

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / (x^2 + 3)

theorem problem_solution (a : ℝ) :
  (∀ x, deriv (f a) x = (a * (x^2 + 3) - a * x * (2 * x)) / (x^2 + 3)^2) →
  deriv (f a) 1 = 1/2 →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3927_392702


namespace NUMINAMATH_CALUDE_negation_of_no_red_cards_negation_equivalent_to_some_red_cards_l3927_392726

-- Define the universe of cards
variable (U : Type)

-- Define the property of being a red card
variable (red : U → Prop)

-- Define the property of being in the deck
variable (in_deck : U → Prop)

-- Statement to be proven
theorem negation_of_no_red_cards (h : ¬∃ x, red x ∧ in_deck x) :
  ¬∀ x, red x → ¬in_deck x :=
sorry

-- Proof that the negation is equivalent to "Some red cards are in this deck"
theorem negation_equivalent_to_some_red_cards :
  (¬∀ x, red x → ¬in_deck x) ↔ (∃ x, red x ∧ in_deck x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_no_red_cards_negation_equivalent_to_some_red_cards_l3927_392726


namespace NUMINAMATH_CALUDE_inverse_g_at_negative_43_l3927_392704

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_g_at_negative_43 : g⁻¹ (-43) = -2 := by sorry

end NUMINAMATH_CALUDE_inverse_g_at_negative_43_l3927_392704


namespace NUMINAMATH_CALUDE_function_below_x_axis_iff_k_in_range_l3927_392798

/-- The function f(x) parameterized by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k^2 - k - 2) * x^2 - (k - 2) * x - 1

/-- The theorem stating the equivalence between the function being always below the x-axis and the range of k -/
theorem function_below_x_axis_iff_k_in_range :
  ∀ k : ℝ, (∀ x : ℝ, f k x < 0) ↔ k ∈ Set.Ioo (-2/5 : ℝ) 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_function_below_x_axis_iff_k_in_range_l3927_392798


namespace NUMINAMATH_CALUDE_sculpture_cost_in_inr_l3927_392737

/-- Exchange rate from British pounds to Indian rupees -/
def gbp_to_inr : ℚ := 20

/-- Exchange rate from British pounds to Namibian dollars -/
def gbp_to_nad : ℚ := 18

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 360

/-- Theorem stating the equivalent cost of the sculpture in Indian rupees -/
theorem sculpture_cost_in_inr :
  (sculpture_cost_nad / gbp_to_nad) * gbp_to_inr = 400 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_inr_l3927_392737


namespace NUMINAMATH_CALUDE_fraction_simplification_l3927_392706

theorem fraction_simplification : (1952^2 - 1940^2) / (1959^2 - 1933^2) = 6/13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3927_392706


namespace NUMINAMATH_CALUDE_furniture_making_l3927_392723

theorem furniture_making (total_wood pieces_per_table pieces_per_chair chairs_made : ℕ) 
  (h1 : total_wood = 672)
  (h2 : pieces_per_table = 12)
  (h3 : pieces_per_chair = 8)
  (h4 : chairs_made = 48) :
  (total_wood - chairs_made * pieces_per_chair) / pieces_per_table = 24 := by
  sorry

end NUMINAMATH_CALUDE_furniture_making_l3927_392723


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_29_l3927_392746

theorem closest_integer_to_sqrt_29 :
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 29| ≤ |m - Real.sqrt 29| ∧ n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_29_l3927_392746


namespace NUMINAMATH_CALUDE_quadrilateral_property_implication_l3927_392712

-- Define a quadrilateral type
structure Quadrilateral :=
  (A B C D : Point)

-- Define the three properties
def diagonals_perpendicular (q : Quadrilateral) : Prop := sorry

def inscribed_in_circle (q : Quadrilateral) : Prop := sorry

def perpendicular_through_intersection (q : Quadrilateral) : Prop := sorry

-- Main theorem
theorem quadrilateral_property_implication (q : Quadrilateral) :
  (diagonals_perpendicular q ∧ inscribed_in_circle q) ∨
  (diagonals_perpendicular q ∧ perpendicular_through_intersection q) ∨
  (inscribed_in_circle q ∧ perpendicular_through_intersection q) →
  diagonals_perpendicular q ∧ inscribed_in_circle q ∧ perpendicular_through_intersection q :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_property_implication_l3927_392712


namespace NUMINAMATH_CALUDE_soccer_substitution_ratio_l3927_392749

/-- Soccer team substitution ratio theorem -/
theorem soccer_substitution_ratio 
  (total_players : ℕ) 
  (starters : ℕ) 
  (first_half_subs : ℕ) 
  (non_players : ℕ) 
  (h1 : total_players = 24) 
  (h2 : starters = 11) 
  (h3 : first_half_subs = 2) 
  (h4 : non_players = 7) : 
  (total_players - non_players - (starters + first_half_subs)) / first_half_subs = 2 := by
sorry

end NUMINAMATH_CALUDE_soccer_substitution_ratio_l3927_392749


namespace NUMINAMATH_CALUDE_triangle_construction_solutions_l3927_392741

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Checks if a point is the foot of an altitude in a triangle -/
def isAltitudeFoot (P : Point2D) (T : Triangle) : Prop := sorry

/-- Checks if a point is the midpoint of a side in a triangle -/
def isMidpoint (P : Point2D) (A B : Point2D) : Prop := sorry

/-- Checks if a point is the midpoint of an altitude in a triangle -/
def isAltitudeMidpoint (P : Point2D) (T : Triangle) : Prop := sorry

/-- The main theorem statement -/
theorem triangle_construction_solutions 
  (A₀ B₁ C₂ : Point2D) : 
  ∃ (T₁ T₂ : Triangle), 
    T₁ ≠ T₂ ∧ 
    isAltitudeFoot A₀ T₁ ∧
    isAltitudeFoot A₀ T₂ ∧
    isMidpoint B₁ T₁.A T₁.C ∧
    isMidpoint B₁ T₂.A T₂.C ∧
    isAltitudeMidpoint C₂ T₁ ∧
    isAltitudeMidpoint C₂ T₂ :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_solutions_l3927_392741


namespace NUMINAMATH_CALUDE_second_car_speed_l3927_392715

/-- Given two cars starting from opposite ends of a 60-mile highway at the same time,
    with one car traveling at 13 mph and both cars meeting after 2 hours,
    prove that the speed of the second car is 17 mph. -/
theorem second_car_speed (highway_length : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  highway_length = 60 →
  time = 2 →
  speed1 = 13 →
  speed1 * time + speed2 * time = highway_length →
  speed2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l3927_392715


namespace NUMINAMATH_CALUDE_number_problem_l3927_392772

theorem number_problem : ∃! x : ℝ, (x / 3) + 12 = 20 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3927_392772


namespace NUMINAMATH_CALUDE_aaron_scarves_count_l3927_392748

/-- The number of scarves Aaron made -/
def aaronScarves : ℕ := 10

/-- The number of sweaters Aaron made -/
def aaronSweaters : ℕ := 5

/-- The number of sweaters Enid made -/
def enidSweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def woolPerScarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def woolPerSweater : ℕ := 4

/-- The total number of balls of wool used -/
def totalWool : ℕ := 82

theorem aaron_scarves_count : 
  woolPerScarf * aaronScarves + 
  woolPerSweater * (aaronSweaters + enidSweaters) = 
  totalWool := by sorry

end NUMINAMATH_CALUDE_aaron_scarves_count_l3927_392748


namespace NUMINAMATH_CALUDE_circle_center_distance_l3927_392739

theorem circle_center_distance (x y : ℝ) : 
  x^2 + y^2 = 6*x + 8*y + 3 →
  Real.sqrt ((10 - x)^2 + (5 - y)^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_distance_l3927_392739


namespace NUMINAMATH_CALUDE_circle_and_tangents_l3927_392716

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line y = 2x
def Line (m : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1}

-- Define the point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem circle_and_tangents 
  (C : ℝ × ℝ) -- Center of the circle
  (h1 : C ∈ Line 2) -- Center lies on y = 2x
  (h2 : (0, 0) ∈ Circle C (Real.sqrt 5)) -- Circle passes through (0,0)
  (h3 : (2, 0) ∈ Circle C (Real.sqrt 5)) -- Circle passes through (2,0)
  : 
  -- 1. The circle equation
  Circle C (Real.sqrt 5) = {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 5} ∧
  -- 2. The tangent line equations
  ∃ (k₁ k₂ : ℝ), 
    k₁ = Real.sqrt 5 / 2 ∧ 
    k₂ = -Real.sqrt 5 / 2 ∧
    (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.2 - 2 = k₁ * (p.1 + 2)} → 
      ((x, y) ∈ Circle C (Real.sqrt 5) → (x, y) = P)) ∧
    (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.2 - 2 = k₂ * (p.1 + 2)} → 
      ((x, y) ∈ Circle C (Real.sqrt 5) → (x, y) = P)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangents_l3927_392716


namespace NUMINAMATH_CALUDE_stating_sum_of_nth_group_is_cube_l3927_392744

/-- 
Given a grouping of consecutive odd numbers as follows:
1; (3,5); (7,9,11); (13, 15, 17, 19); ...
This function represents the sum of the numbers in the n-th group.
-/
def sumOfNthGroup (n : ℕ) : ℕ :=
  n^3

/-- 
Theorem stating that the sum of the numbers in the n-th group
of the described sequence is equal to n^3.
-/
theorem sum_of_nth_group_is_cube (n : ℕ) :
  sumOfNthGroup n = n^3 := by
  sorry

end NUMINAMATH_CALUDE_stating_sum_of_nth_group_is_cube_l3927_392744


namespace NUMINAMATH_CALUDE_interior_angle_non_integer_count_l3927_392738

theorem interior_angle_non_integer_count :
  ∃! (n : ℕ), 3 ≤ n ∧ n ≤ 10 ∧ ¬(∃ (k : ℕ), (180 * (n - 2)) / n = k) :=
by sorry

end NUMINAMATH_CALUDE_interior_angle_non_integer_count_l3927_392738


namespace NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l3927_392725

theorem absolute_value_and_quadratic_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x - 3| = 5 ↔ x^2 + b*x + c = 0) →
  b = -6 ∧ c = -16 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l3927_392725


namespace NUMINAMATH_CALUDE_double_acute_angle_range_l3927_392707

theorem double_acute_angle_range (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  0 < 2 * α ∧ 2 * α < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_range_l3927_392707


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l3927_392797

/-- Given two digits A and B in base d > 6, if AB_d + AA_d = 172_d, then A_d - B_d = 3_d -/
theorem digit_difference_in_base_d (d A B : ℕ) (h_d : d > 6)
  (h_digits : A < d ∧ B < d)
  (h_sum : d * B + A + d * A + A = d^2 + 7 * d + 2) :
  A - B = 3 :=
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l3927_392797


namespace NUMINAMATH_CALUDE_sin_pi_sufficient_not_necessary_l3927_392753

open Real

theorem sin_pi_sufficient_not_necessary :
  (∀ x : ℝ, x = π → sin x = 0) ∧
  (∃ x : ℝ, x ≠ π ∧ sin x = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_sufficient_not_necessary_l3927_392753


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l3927_392781

theorem smallest_integer_with_remainder_one (k : ℕ) : k = 400 ↔ 
  (k > 1) ∧ 
  (k % 19 = 1) ∧ 
  (k % 7 = 1) ∧ 
  (k % 3 = 1) ∧ 
  (∀ m : ℕ, m > 1 → m % 19 = 1 → m % 7 = 1 → m % 3 = 1 → k ≤ m) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l3927_392781


namespace NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l3927_392778

theorem tan_sum_from_sin_cos_sum (α β : Real) 
  (h1 : Real.sin α + Real.sin β = (4/5) * Real.sqrt 2)
  (h2 : Real.cos α + Real.cos β = (4/5) * Real.sqrt 3) :
  Real.tan α + Real.tan β = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l3927_392778


namespace NUMINAMATH_CALUDE_smallest_winning_number_l3927_392762

theorem smallest_winning_number : ∃ N : ℕ, N ≤ 999 ∧ 
  (∀ m : ℕ, m < N → (16 * m + 980 > 1200 ∨ 16 * m + 1050 ≤ 1200)) ∧
  16 * N + 980 ≤ 1200 ∧ 
  16 * N + 1050 > 1200 ∧
  N = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l3927_392762


namespace NUMINAMATH_CALUDE_angle_of_inclination_l3927_392710

/-- The angle of inclination of the line x + √3 y - 5 = 0 is 150° -/
theorem angle_of_inclination (x y : ℝ) : 
  x + Real.sqrt 3 * y - 5 = 0 → 
  ∃ θ : ℝ, θ = 150 * π / 180 ∧ 
    Real.tan θ = -(1 / Real.sqrt 3) ∧
    0 ≤ θ ∧ θ < π := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_l3927_392710


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3927_392743

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, a (n + 1) - a n = 1) →  -- arithmetic sequence with common difference 1
  (∀ n, S n = n * a 1 + n * (n - 1) / 2) →  -- sum formula for arithmetic sequence
  S 8 = 4 * S 4 →  -- given condition
  a 10 = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3927_392743


namespace NUMINAMATH_CALUDE_right_trapezoid_inscribed_circle_theorem_l3927_392767

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  /-- Length of the longer base -/
  a : ℝ
  /-- Length of the shorter base -/
  c : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The longer base is longer than the shorter base -/
  h1 : a > c
  /-- The bases are positive -/
  h2 : a > 0
  h3 : c > 0
  /-- The radius is positive -/
  h4 : r > 0
  /-- Relation between radius and bases -/
  h5 : r = (a * c) / (a + c)

/-- The theorem to be proved -/
theorem right_trapezoid_inscribed_circle_theorem (t : RightTrapezoidWithInscribedCircle) :
  (2 : ℝ) * t.r = 2 / ((1 / t.a) + (1 / t.c)) :=
by sorry

end NUMINAMATH_CALUDE_right_trapezoid_inscribed_circle_theorem_l3927_392767


namespace NUMINAMATH_CALUDE_positive_root_existence_l3927_392735

def f (x : ℝ) := x^5 - x - 1

theorem positive_root_existence :
  ∃ x ∈ Set.Icc 1 2, f x = 0 ∧ x > 0 :=
sorry

end NUMINAMATH_CALUDE_positive_root_existence_l3927_392735


namespace NUMINAMATH_CALUDE_car_cost_share_l3927_392713

/-- Given a car that costs $2,100 and is used for 7 days a week, with one person using it for 4 days,
    prove that the other person's share of the cost is $900. -/
theorem car_cost_share (total_cost : ℕ) (total_days : ℕ) (days_used_by_first : ℕ) :
  total_cost = 2100 →
  total_days = 7 →
  days_used_by_first = 4 →
  (total_cost * (total_days - days_used_by_first) / total_days : ℚ) = 900 := by
  sorry

#check car_cost_share

end NUMINAMATH_CALUDE_car_cost_share_l3927_392713


namespace NUMINAMATH_CALUDE_largest_product_of_digits_l3927_392701

/-- A function that returns the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is equal to the product of its digits -/
def isProductOfDigits (n : ℕ) : Prop :=
  n = productOfDigits n

/-- Theorem stating that 9 is the largest natural number equal to the product of its digits -/
theorem largest_product_of_digits : 
  ∀ n : ℕ, isProductOfDigits n → n ≤ 9 := by sorry

end NUMINAMATH_CALUDE_largest_product_of_digits_l3927_392701


namespace NUMINAMATH_CALUDE_interactive_lines_count_l3927_392771

/-- Represents a four-digit number M with specific digit placement. -/
structure FourDigitNumber where
  a : ℕ  -- Thousands place
  b : ℕ  -- Hundreds place
  c : ℕ  -- Ones place
  h1 : c ≠ 0
  h2 : a < 10 ∧ b < 10 ∧ c < 10

/-- Calculates the value of M given its digit representation. -/
def M (n : FourDigitNumber) : ℕ :=
  1000 * n.a + 100 * n.b + 10 + n.c

/-- Calculates the value of N by moving the ones digit to the front. -/
def N (n : FourDigitNumber) : ℕ :=
  1000 * n.c + 100 * n.a + 10 * n.b + 1

/-- Defines the function F(M) = (M + N) / 11. -/
def F (n : FourDigitNumber) : ℚ :=
  (M n + N n : ℚ) / 11

/-- Predicate for the interactive line condition. -/
def IsInteractiveLine (n : FourDigitNumber) : Prop :=
  n.c = n.a + n.b

/-- The main theorem stating the number of interactive lines satisfying the condition. -/
theorem interactive_lines_count :
  (∃ (S : Finset FourDigitNumber),
    S.card = 8 ∧
    (∀ n ∈ S, IsInteractiveLine n ∧ ∃ k : ℕ, F n = 6 * k) ∧
    (∀ n : FourDigitNumber, IsInteractiveLine n → (∃ k : ℕ, F n = 6 * k) → n ∈ S)) :=
  sorry


end NUMINAMATH_CALUDE_interactive_lines_count_l3927_392771


namespace NUMINAMATH_CALUDE_square_and_rectangle_area_sum_l3927_392742

/-- Given a square and a rectangle satisfying certain conditions, prove that the sum of their areas is approximately 118 square units. -/
theorem square_and_rectangle_area_sum :
  ∀ (s w : ℝ),
    s > 0 →
    w > 0 →
    s^2 + 2*w^2 = 130 →
    4*s - 2*(w + 2*w) = 20 →
    abs (s^2 + 2*w^2 - 118) < 1 :=
by
  sorry

#check square_and_rectangle_area_sum

end NUMINAMATH_CALUDE_square_and_rectangle_area_sum_l3927_392742


namespace NUMINAMATH_CALUDE_remainder_425421_div_12_l3927_392732

theorem remainder_425421_div_12 : 425421 % 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_425421_div_12_l3927_392732


namespace NUMINAMATH_CALUDE_jerry_shelf_theorem_l3927_392718

/-- Calculates the total number of action figures on Jerry's shelf -/
def total_action_figures (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of action figures is the sum of initial and added figures -/
theorem jerry_shelf_theorem (initial : ℕ) (added : ℕ) :
  total_action_figures initial added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_jerry_shelf_theorem_l3927_392718


namespace NUMINAMATH_CALUDE_equation_solutions_l3927_392747

theorem equation_solutions :
  let f : ℝ → ℝ := λ x => x * (5 * x + 2) - 6 * (5 * x + 2)
  (f 6 = 0 ∧ f (-2/5) = 0) ∧ 
  ∀ x : ℝ, f x = 0 → x = 6 ∨ x = -2/5 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3927_392747


namespace NUMINAMATH_CALUDE_bushes_needed_for_zucchinis_l3927_392727

/-- Represents the number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers of blueberries that can be traded for zucchinis -/
def containers_traded : ℕ := 6

/-- Represents the number of zucchinis received in trade for containers_traded -/
def zucchinis_received : ℕ := 3

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- Theorem stating that 12 bushes are needed to obtain 60 zucchinis -/
theorem bushes_needed_for_zucchinis : 
  (target_zucchinis * containers_traded) / (zucchinis_received * containers_per_bush) = 12 :=
sorry

end NUMINAMATH_CALUDE_bushes_needed_for_zucchinis_l3927_392727


namespace NUMINAMATH_CALUDE_garrison_provision_problem_l3927_392785

/-- Calculates the initial number of days provisions would last for a garrison --/
def initial_provision_days (initial_men : ℕ) (reinforcement_men : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  (initial_men * days_before_reinforcement + (initial_men + reinforcement_men) * days_after_reinforcement) / initial_men

theorem garrison_provision_problem :
  initial_provision_days 1850 1110 12 10 = 28 := by
  sorry

end NUMINAMATH_CALUDE_garrison_provision_problem_l3927_392785


namespace NUMINAMATH_CALUDE_divisors_of_8n_cubed_l3927_392722

theorem divisors_of_8n_cubed (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 12) :
  (Nat.divisors (8 * n^3)).card = 280 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8n_cubed_l3927_392722


namespace NUMINAMATH_CALUDE_integer_solutions_inequalities_l3927_392791

theorem integer_solutions_inequalities (x : ℤ) : 
  ((x - 2) / 2 ≤ -x / 2 + 2 ∧ 4 - 7*x < -3) ↔ (x = 2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_inequalities_l3927_392791


namespace NUMINAMATH_CALUDE_deposit_calculation_l3927_392770

theorem deposit_calculation (remaining_amount : ℝ) (deposit_percentage : ℝ) :
  remaining_amount = 1350 ∧ deposit_percentage = 0.1 →
  (remaining_amount / (1 - deposit_percentage)) * deposit_percentage = 150 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l3927_392770


namespace NUMINAMATH_CALUDE_apple_purchase_cost_l3927_392777

/-- The cost of apples in dollars per 7 pounds -/
def apple_cost : ℚ := 5

/-- The rate of apples in pounds per cost unit -/
def apple_rate : ℚ := 7

/-- The amount of apples we want to buy in pounds -/
def apple_amount : ℚ := 21

/-- Theorem: The cost of 21 pounds of apples is $15 -/
theorem apple_purchase_cost : (apple_amount / apple_rate) * apple_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_cost_l3927_392777


namespace NUMINAMATH_CALUDE_problem_statement_l3927_392788

theorem problem_statement (n : ℤ) (a : ℝ) : 
  (6 * 11 * n > 0) → (a^(2*n) = 5) → (2 * a^(6*n) - 4 = 246) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3927_392788


namespace NUMINAMATH_CALUDE_power_outage_duration_is_three_l3927_392736

/-- The duration of the power outage in hours -/
def power_outage_duration : ℝ := 3

/-- The temperature rise rate during the power outage in degrees per hour -/
def temperature_rise_rate : ℝ := 8

/-- The temperature decrease rate when the air conditioner is on in degrees per hour -/
def temperature_decrease_rate : ℝ := 4

/-- The time taken by the air conditioner to restore the temperature in hours -/
def air_conditioner_duration : ℝ := 6

/-- Theorem stating that the power outage duration is 3 hours -/
theorem power_outage_duration_is_three :
  power_outage_duration = temperature_rise_rate⁻¹ * temperature_decrease_rate * air_conditioner_duration :=
by sorry

end NUMINAMATH_CALUDE_power_outage_duration_is_three_l3927_392736


namespace NUMINAMATH_CALUDE_correct_quotient_l3927_392786

theorem correct_quotient (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 70) : D / 21 = 40 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_l3927_392786


namespace NUMINAMATH_CALUDE_log_power_sum_l3927_392703

theorem log_power_sum (a b : ℝ) (h1 : a = Real.log 64) (h2 : b = Real.log 25) :
  (8 : ℝ) ^ (a / b) + (5 : ℝ) ^ (b / a) = 89 := by
  sorry

end NUMINAMATH_CALUDE_log_power_sum_l3927_392703


namespace NUMINAMATH_CALUDE_crypto_encoding_l3927_392728

/-- Represents the encoding of digits in the cryptographic system -/
inductive Digit
| A
| B
| C
| D

/-- Converts a Digit to its corresponding base-4 value -/
def digit_to_base4 : Digit → Nat
| Digit.A => 3
| Digit.B => 1
| Digit.C => 0
| Digit.D => 2

/-- Converts a three-digit code to its base-10 value -/
def code_to_base10 (d₁ d₂ d₃ : Digit) : Nat :=
  16 * (digit_to_base4 d₁) + 4 * (digit_to_base4 d₂) + (digit_to_base4 d₃)

/-- The main theorem stating the result of the cryptographic encoding -/
theorem crypto_encoding :
  code_to_base10 Digit.B Digit.C Digit.D + 1 = code_to_base10 Digit.B Digit.D Digit.A ∧
  code_to_base10 Digit.B Digit.D Digit.A + 1 = code_to_base10 Digit.B Digit.C Digit.A →
  code_to_base10 Digit.D Digit.A Digit.C = 44 :=
by sorry

end NUMINAMATH_CALUDE_crypto_encoding_l3927_392728


namespace NUMINAMATH_CALUDE_power_of_power_l3927_392775

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3927_392775


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_80_l3927_392756

theorem last_three_digits_of_7_to_80 : 7^80 ≡ 961 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_80_l3927_392756


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l3927_392750

theorem greatest_distance_between_circle_centers 
  (circle_diameter : ℝ) 
  (rectangle_length : ℝ) 
  (rectangle_width : ℝ) 
  (h_diameter : circle_diameter = 8)
  (h_length : rectangle_length = 20)
  (h_width : rectangle_width = 16)
  (h_tangent : circle_diameter ≤ rectangle_width) :
  let circle_radius := circle_diameter / 2
  let horizontal_distance := 2 * circle_radius
  let vertical_distance := rectangle_width
  ∃ (max_distance : ℝ), 
    max_distance = (horizontal_distance^2 + vertical_distance^2).sqrt ∧
    max_distance = 8 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l3927_392750


namespace NUMINAMATH_CALUDE_triangle_vector_intersection_l3927_392768

/-- Given a triangle XYZ with points M, N, and Q satisfying specific conditions,
    prove that Q can be expressed as a linear combination of X, Y, and Z with specific coefficients. -/
theorem triangle_vector_intersection (X Y Z M N Q : ℝ × ℝ) : 
  (∃ (k : ℝ), M = k • Z + (1 - k) • Y ∧ k = 1/5) →  -- M lies on YZ extended
  (∃ (l : ℝ), N = l • X + (1 - l) • Z ∧ l = 3/5) →  -- N lies on XZ
  (∃ (s t : ℝ), Q = s • Y + (1 - s) • N ∧ Q = t • X + (1 - t) • M) →  -- Q is intersection of YN and XM
  Q = (12/23) • X + (3/23) • Y + (8/23) • Z :=
by sorry

end NUMINAMATH_CALUDE_triangle_vector_intersection_l3927_392768


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3927_392709

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3927_392709


namespace NUMINAMATH_CALUDE_triangle_properties_l3927_392758

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions and the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.c = abc.a)
  (h2 : abc.c = Real.sqrt 3)
  (h3 : Real.sin abc.B ^ 2 = 2 * Real.sin abc.A * Real.sin abc.C) :
  Real.cos abc.B = 0 ∧ (1/2 * abc.a * abc.c * Real.sin abc.B = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3927_392758


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_b_greater_than_neg_one_l3927_392730

def A : Set ℝ := {x | Real.log (x + 2) / Real.log (1/2) < 0}
def B (a b : ℝ) : Set ℝ := {x | (x - a) * (x - b) < 0}

theorem intersection_nonempty_implies_b_greater_than_neg_one :
  (∀ b : ℝ, (A ∩ B (-3) b).Nonempty) → ∀ b : ℝ, b > -1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_b_greater_than_neg_one_l3927_392730


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3927_392765

/-- Given two points M and N that are symmetric with respect to the y-axis,
    prove that the sum of their x-coordinates is zero. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (m - 1 = -(3: ℝ)) → -- M's x-coordinate is opposite to N's
  (1 : ℝ) = n - 1 →    -- M's y-coordinate equals N's
  m + n = 0 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3927_392765


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l3927_392745

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a sphere -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- The specific configuration described in the problem -/
def problemConfig : ConeConfiguration :=
  { cone1 := { baseRadius := 4, height := 10 }
  , cone2 := { baseRadius := 4, height := 10 }
  , intersectionDistance := 4
  , sphereRadius := 0  -- To be determined
  }

/-- The theorem statement -/
theorem max_sphere_radius_squared (config : ConeConfiguration) :
  config = problemConfig →
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (s : ℝ), s > 0 → 
      (∃ (c : ConeConfiguration), c.cone1 = config.cone1 ∧ 
                                  c.cone2 = config.cone2 ∧ 
                                  c.intersectionDistance = config.intersectionDistance ∧
                                  c.sphereRadius = s) →
      s^2 ≤ r^2) ∧
    r^2 = 8704 / 29 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_l3927_392745


namespace NUMINAMATH_CALUDE_sphere_volume_from_cross_section_l3927_392766

/-- Given a sphere with a circular cross-section of radius 4 and the distance
    from the sphere's center to the center of the cross-section is 3,
    prove that the volume of the sphere is (500/3)π. -/
theorem sphere_volume_from_cross_section (r : ℝ) (h : ℝ) :
  r^2 = 4^2 + 3^2 →
  (4 / 3) * π * r^3 = (500 / 3) * π := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_from_cross_section_l3927_392766


namespace NUMINAMATH_CALUDE_lizzy_money_theorem_l3927_392793

def lizzy_money_problem (mother_gift uncle_gift father_gift candy_cost : ℕ) : Prop :=
  let initial_amount := mother_gift + father_gift
  let amount_after_spending := initial_amount - candy_cost
  let final_amount := amount_after_spending + uncle_gift
  final_amount = 140

theorem lizzy_money_theorem :
  lizzy_money_problem 80 70 40 50 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_theorem_l3927_392793


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3927_392754

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 2 * x * y) :
  x + y ≥ 9 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3927_392754


namespace NUMINAMATH_CALUDE_smallest_sausage_packages_l3927_392794

theorem smallest_sausage_packages (sausage_pack : ℕ) (bun_pack : ℕ) 
  (h1 : sausage_pack = 10) (h2 : bun_pack = 15) :
  ∃ n : ℕ, n > 0 ∧ sausage_pack * n % bun_pack = 0 ∧ 
  ∀ m : ℕ, m > 0 → sausage_pack * m % bun_pack = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sausage_packages_l3927_392794


namespace NUMINAMATH_CALUDE_remainder_theorem_l3927_392721

def dividend (b x : ℝ) : ℝ := 12 * x^3 - 9 * x^2 + b * x + 8
def divisor (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

theorem remainder_theorem (b : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, dividend b x = divisor x * q x + 10) ↔ b = -31/3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3927_392721


namespace NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l3927_392720

/-- The side length of the largest inscribed square in a specific configuration -/
theorem largest_inscribed_square_side_length :
  ∃ (large_square_side : ℝ) (triangle_side : ℝ) (inscribed_square_side : ℝ),
    large_square_side = 12 ∧
    triangle_side = 4 * Real.sqrt 6 ∧
    inscribed_square_side = 6 - Real.sqrt 6 ∧
    2 * inscribed_square_side * Real.sqrt 2 + triangle_side = large_square_side * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l3927_392720


namespace NUMINAMATH_CALUDE_zero_location_l3927_392769

theorem zero_location (x y : ℝ) 
  (h1 : x^5 < y^8) 
  (h2 : y^8 < y^3) 
  (h3 : y^3 < x^6)
  (h4 : x < 0)
  (h5 : 0 < y)
  (h6 : y < 1) : 
  x^5 < 0 ∧ 0 < y^8 := by
  sorry

end NUMINAMATH_CALUDE_zero_location_l3927_392769


namespace NUMINAMATH_CALUDE_billy_video_count_l3927_392734

theorem billy_video_count 
  (suggestions_per_round : ℕ) 
  (num_rounds : ℕ) 
  (final_pick : ℕ) :
  suggestions_per_round = 15 →
  num_rounds = 5 →
  final_pick = 5 →
  suggestions_per_round * num_rounds - (suggestions_per_round - final_pick) = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_billy_video_count_l3927_392734


namespace NUMINAMATH_CALUDE_pat_candy_count_l3927_392774

/-- The number of cookies Pat has -/
def num_cookies : ℕ := 42

/-- The number of brownies Pat has -/
def num_brownies : ℕ := 21

/-- The number of people in Pat's family -/
def num_people : ℕ := 7

/-- The number of dessert pieces each person gets -/
def dessert_per_person : ℕ := 18

/-- The number of candy pieces Pat has -/
def num_candy : ℕ := num_people * dessert_per_person - (num_cookies + num_brownies)

theorem pat_candy_count : num_candy = 63 := by
  sorry

end NUMINAMATH_CALUDE_pat_candy_count_l3927_392774


namespace NUMINAMATH_CALUDE_quentavious_gum_pieces_l3927_392714

/-- Represents the types of coins --/
inductive Coin
  | Nickel
  | Dime
  | Quarter

/-- Calculates the number of gum pieces for a given coin type --/
def gumPieces (c : Coin) : ℕ :=
  match c with
  | Coin.Nickel => 2
  | Coin.Dime => 3
  | Coin.Quarter => 5

/-- Represents the initial state of coins --/
structure InitialCoins where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Represents the final state of coins --/
structure FinalCoins where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the number of gum pieces received --/
def gumReceived (initial : InitialCoins) (final : FinalCoins) : ℕ :=
  let exchanged_nickels := initial.nickels - final.nickels
  let exchanged_dimes := initial.dimes - final.dimes
  let exchanged_quarters := initial.quarters - final.quarters
  if exchanged_nickels > 0 && exchanged_dimes > 0 && exchanged_quarters > 0 then
    15
  else
    exchanged_nickels * gumPieces Coin.Nickel +
    exchanged_dimes * gumPieces Coin.Dime +
    exchanged_quarters * gumPieces Coin.Quarter

theorem quentavious_gum_pieces :
  let initial := InitialCoins.mk 5 6 4
  let final := FinalCoins.mk 2 1 0
  gumReceived initial final = 15 := by
  sorry

end NUMINAMATH_CALUDE_quentavious_gum_pieces_l3927_392714


namespace NUMINAMATH_CALUDE_final_pressure_is_three_l3927_392759

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
structure GasState where
  pressure : ℝ
  volume : ℝ
  constant : ℝ
  h : pressure * volume = constant

/-- The initial state of the hydrogen gas -/
def initial_state : GasState :=
  { pressure := 6
  , volume := 3
  , constant := 18
  , h := by sorry }

/-- The final state of the hydrogen gas after transfer -/
def final_state : GasState :=
  { pressure := 3
  , volume := 6
  , constant := 18
  , h := by sorry }

/-- Theorem stating that the final pressure is 3 kPa -/
theorem final_pressure_is_three :
  final_state.pressure = 3 :=
by sorry

end NUMINAMATH_CALUDE_final_pressure_is_three_l3927_392759


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3927_392780

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n, b (n + 1) - b n = d) →  -- arithmetic sequence
  b 4 * b 5 = 18 →
  b 3 * b 6 = -80 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3927_392780


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_five_equals_sqrt_fifteen_l3927_392779

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b :=
by sorry

theorem sqrt_three_times_sqrt_five_equals_sqrt_fifteen :
  Real.sqrt 3 * Real.sqrt 5 = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_five_equals_sqrt_fifteen_l3927_392779


namespace NUMINAMATH_CALUDE_distance_to_center_of_gravity_l3927_392761

/-- Regular hexagon with side length a -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Square cut out from the hexagon -/
structure CutOutSquare (hex : RegularHexagon) where
  diagonal : ℝ
  diagonal_eq_side : diagonal = hex.side_length

/-- Remaining plate after cutting out the square -/
structure RemainingPlate (hex : RegularHexagon) (square : CutOutSquare hex) where

/-- Center of gravity of the remaining plate -/
noncomputable def center_of_gravity (plate : RemainingPlate hex square) : ℝ × ℝ := sorry

/-- Distance from the hexagon center to the center of gravity -/
noncomputable def distance_to_center (plate : RemainingPlate hex square) : ℝ :=
  let cog := center_of_gravity plate
  Real.sqrt ((cog.1 ^ 2) + (cog.2 ^ 2))

/-- Main theorem: The distance from the hexagon center to the center of gravity of the remaining plate -/
theorem distance_to_center_of_gravity 
  (hex : RegularHexagon) 
  (square : CutOutSquare hex) 
  (plate : RemainingPlate hex square) : 
  distance_to_center plate = (3 * Real.sqrt 3 + 1) / 52 * hex.side_length := by
  sorry

end NUMINAMATH_CALUDE_distance_to_center_of_gravity_l3927_392761


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1560_l3927_392790

theorem sum_of_largest_and_smallest_prime_factors_of_1560 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 1560 ∧ largest ∣ 1560 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1560 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1560 → p ≥ smallest) ∧
    smallest + largest = 15 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1560_l3927_392790


namespace NUMINAMATH_CALUDE_power_of_product_equals_power_l3927_392752

theorem power_of_product_equals_power (n : ℕ) : 3^12 * 3^18 = 243^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_equals_power_l3927_392752


namespace NUMINAMATH_CALUDE_probability_one_boy_one_girl_l3927_392740

/-- The probability of selecting exactly one boy and one girl when randomly choosing 2 people from 2 boys and 2 girls -/
theorem probability_one_boy_one_girl (num_boys num_girls : ℕ) (h1 : num_boys = 2) (h2 : num_girls = 2) :
  let total_combinations := num_boys * num_girls + (num_boys.choose 2) + (num_girls.choose 2)
  let favorable_combinations := num_boys * num_girls
  (favorable_combinations : ℚ) / total_combinations = 2/3 := by
sorry

end NUMINAMATH_CALUDE_probability_one_boy_one_girl_l3927_392740


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_27_l3927_392784

theorem x_squared_minus_y_squared_equals_27
  (x y : ℝ)
  (h1 : y + 6 = (x - 3)^2)
  (h2 : x + 6 = (y - 3)^2)
  (h3 : x ≠ y) :
  x^2 - y^2 = 27 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_27_l3927_392784


namespace NUMINAMATH_CALUDE_vals_money_value_is_38_80_l3927_392773

/-- Calculates the total value of Val's money in USD -/
def valsMoneyValue (initialNickels : ℕ) (dimesToNickelsRatio : ℕ) (quartersToDimesRatio : ℕ) 
  (newNickelsMultiplier : ℕ) (canadianNickelRatio : ℚ) (exchangeRate : ℚ) : ℚ :=
  let initialDimes := initialNickels * dimesToNickelsRatio
  let initialQuarters := initialDimes * quartersToDimesRatio
  let newNickels := initialNickels * newNickelsMultiplier
  let canadianNickels := (newNickels : ℚ) * canadianNickelRatio
  let usNickels := (newNickels : ℚ) - canadianNickels
  let initialValue := (initialNickels : ℚ) * (5 / 100) + (initialDimes : ℚ) * (10 / 100) + (initialQuarters : ℚ) * (25 / 100)
  let newUsNickelsValue := usNickels * (5 / 100)
  let canadianNickelsValue := canadianNickels * (5 / 100) * exchangeRate
  initialValue + newUsNickelsValue + canadianNickelsValue

/-- Theorem stating that Val's money value is $38.80 given the problem conditions -/
theorem vals_money_value_is_38_80 :
  valsMoneyValue 20 3 2 2 (1/2) (4/5) = 388/10 := by
  sorry

end NUMINAMATH_CALUDE_vals_money_value_is_38_80_l3927_392773


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3927_392733

theorem solution_set_inequality (x : ℝ) : -x^2 + 2*x > 0 ↔ 0 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3927_392733


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l3927_392708

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 4

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 24

/-- The total number of guitar strings Dave needs to replace -/
def total_strings : ℕ := strings_per_night * shows_per_week * total_weeks

theorem dave_guitar_strings :
  total_strings = 576 := by sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l3927_392708


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l3927_392764

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List (Fin 4) :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

def binary : List Bool := [true, true, false, true, false, false, true, false]

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal binary) = [3, 1, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l3927_392764


namespace NUMINAMATH_CALUDE_unique_bagel_count_l3927_392799

def is_valid_purchase (bagels : ℕ) : Prop :=
  ∃ (muffins : ℕ),
    bagels + muffins = 7 ∧
    (90 * bagels + 40 * muffins) % 150 = 0

theorem unique_bagel_count : ∃! b : ℕ, is_valid_purchase b ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_bagel_count_l3927_392799


namespace NUMINAMATH_CALUDE_dry_grapes_weight_l3927_392757

-- Define the parameters
def fresh_water_content : Real := 0.90
def dried_water_content : Real := 0.20
def fresh_grapes_weight : Real := 5

-- Define the theorem
theorem dry_grapes_weight :
  let non_water_content := (1 - fresh_water_content) * fresh_grapes_weight
  let dry_grapes_weight := non_water_content / (1 - dried_water_content)
  dry_grapes_weight = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_dry_grapes_weight_l3927_392757


namespace NUMINAMATH_CALUDE_certain_number_problem_l3927_392792

theorem certain_number_problem : ∃ x : ℚ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3927_392792


namespace NUMINAMATH_CALUDE_mikail_birthday_money_l3927_392795

/-- Mikail's age tomorrow -/
def mikail_age : ℕ := 9

/-- Amount of money Mikail receives per year of age -/
def money_per_year : ℕ := 5

/-- Cost of the video game -/
def game_cost : ℕ := 80

/-- Theorem stating Mikail's situation -/
theorem mikail_birthday_money :
  (mikail_age = 3 * 3) ∧
  (mikail_age * money_per_year = 45) ∧
  (mikail_age * money_per_year < game_cost) :=
by sorry

end NUMINAMATH_CALUDE_mikail_birthday_money_l3927_392795


namespace NUMINAMATH_CALUDE_orange_basket_problem_l3927_392776

/-- 
Given:
- When 2 oranges are put in each basket, 4 oranges are left over.
- When 5 oranges are put in each basket, 1 basket is left over.

Prove that the number of baskets is 3 and the number of oranges is 10.
-/
theorem orange_basket_problem (b o : ℕ) 
  (h1 : 2 * b + 4 = o) 
  (h2 : 5 * (b - 1) = o) : 
  b = 3 ∧ o = 10 := by
  sorry


end NUMINAMATH_CALUDE_orange_basket_problem_l3927_392776


namespace NUMINAMATH_CALUDE_petya_spent_less_than_5000_l3927_392731

/-- Represents the purchase of a book -/
inductive Purchase
  | Expensive (cost : ℕ)
  | Cheap (cost : ℕ)

/-- Represents Petya's shopping process -/
structure ShoppingProcess where
  initial_money : ℕ
  purchases : List Purchase
  final_coins : ℕ

/-- Checks if a shopping process is valid according to the problem conditions -/
def is_valid_process (p : ShoppingProcess) : Prop :=
  p.initial_money % 100 = 0 ∧
  (∀ purchase ∈ p.purchases, match purchase with
    | Purchase.Expensive cost => cost ≥ 100
    | Purchase.Cheap cost => cost < 100
  ) ∧
  p.final_coins < 100 ∧
  2 * (p.initial_money - p.final_coins) = p.initial_money

/-- Calculates the total amount spent on books -/
def total_spent (p : ShoppingProcess) : ℕ :=
  p.initial_money - p.final_coins

/-- Theorem stating that Petya could not have spent at least 5000 rubles on books -/
theorem petya_spent_less_than_5000 (p : ShoppingProcess) :
  is_valid_process p → total_spent p < 5000 := by
  sorry

end NUMINAMATH_CALUDE_petya_spent_less_than_5000_l3927_392731


namespace NUMINAMATH_CALUDE_no_double_application_function_l3927_392787

theorem no_double_application_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l3927_392787


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3927_392729

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- The main theorem -/
theorem tangent_line_y_intercept (c1 c2 : Circle) (l : Line) :
  c1.center = (3, 0) →
  c1.radius = 3 →
  c2.center = (8, 0) →
  c2.radius = 2 →
  is_tangent l c1 →
  is_tangent l c2 →
  l.y_intercept = 15 * Real.sqrt 26 / 26 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3927_392729


namespace NUMINAMATH_CALUDE_not_prime_sum_product_l3927_392783

theorem not_prime_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ 0 < c ∧ 0 < b ∧ 0 < a)
  (h_order : d < c ∧ c < b ∧ b < a)
  (h_eq : a * c + b * d = (b + d + a - c) * (b + d - a + c)) :
  ¬ Nat.Prime (a * b + c * d) :=
by sorry

end NUMINAMATH_CALUDE_not_prime_sum_product_l3927_392783


namespace NUMINAMATH_CALUDE_complement_of_A_l3927_392755

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 2 > 4}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3927_392755


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l3927_392751

theorem fraction_ratio_equality : ∃ (X Y : ℚ), (X / Y) / (2 / 6) = (1 / 2) / (1 / 2) → X / Y = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l3927_392751


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l3927_392796

/-- A quadratic equation in terms of x is of the form ax^2 + bx + c = 0, where a ≠ 0 --/
def IsQuadraticEquation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation 3x^2 + 1 = 0 --/
def f (x : ℝ) : ℝ := 3 * x^2 + 1

theorem equation_is_quadratic : IsQuadraticEquation f := by
  sorry


end NUMINAMATH_CALUDE_equation_is_quadratic_l3927_392796


namespace NUMINAMATH_CALUDE_sunday_production_l3927_392789

/-- The number of toys produced on a given day of the week -/
def toysProduced (day : Nat) : Nat :=
  2500 + 25 * day

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Theorem stating that the number of toys produced on Sunday (day 6) is 2650 -/
theorem sunday_production :
  toysProduced (daysInWeek - 1) = 2650 := by
  sorry


end NUMINAMATH_CALUDE_sunday_production_l3927_392789


namespace NUMINAMATH_CALUDE_wrong_divisor_problem_l3927_392763

theorem wrong_divisor_problem (correct_divisor correct_answer student_answer : ℕ) 
  (h1 : correct_divisor = 36)
  (h2 : correct_answer = 58)
  (h3 : student_answer = 24) :
  ∃ (wrong_divisor : ℕ), 
    (correct_divisor * correct_answer) / wrong_divisor = student_answer ∧ 
    wrong_divisor = 87 := by
  sorry

end NUMINAMATH_CALUDE_wrong_divisor_problem_l3927_392763


namespace NUMINAMATH_CALUDE_football_match_problem_l3927_392705

/-- Represents a football team's match statistics -/
structure TeamStats :=
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

/-- Calculate the total matches played by a team -/
def total_matches (team : TeamStats) : ℕ :=
  team.wins + team.draws + team.losses

/-- The football match problem -/
theorem football_match_problem 
  (home : TeamStats)
  (rival : TeamStats)
  (h1 : home.wins = 3)
  (h2 : home.draws = 4)
  (h3 : home.losses = 0)
  (h4 : rival.wins = 2 * home.wins)
  (h5 : rival.draws = 4)
  (h6 : rival.losses = 0) :
  total_matches home + total_matches rival = 17 :=
sorry

end NUMINAMATH_CALUDE_football_match_problem_l3927_392705


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3927_392782

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2^x - 1 > 0) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3927_392782


namespace NUMINAMATH_CALUDE_max_b_value_l3927_392700

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + 3 * b * x

-- State the theorem
theorem max_b_value (a b : ℝ) (ha : a < 0) (hb : b > 0) 
  (hf : ∀ x ∈ Set.Icc 0 1, f a b x ∈ Set.Icc 0 1) : 
  b ≤ Real.sqrt 3 / 2 ∧ ∃ x ∈ Set.Icc 0 1, f a (Real.sqrt 3 / 2) x = 1 := by
sorry

end NUMINAMATH_CALUDE_max_b_value_l3927_392700


namespace NUMINAMATH_CALUDE_work_completion_time_l3927_392760

theorem work_completion_time (x y : ℕ) (h1 : x = 14) 
  (h2 : (5 : ℝ) * ((1 : ℝ) / x + (1 : ℝ) / y) = 0.6071428571428572) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3927_392760


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3927_392719

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) := by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3927_392719


namespace NUMINAMATH_CALUDE_shaded_area_9x7_grid_l3927_392724

/-- Represents a grid with 2x2 squares, where alternate squares are split and shaded -/
structure ShadedGrid :=
  (width : ℕ)
  (height : ℕ)
  (square_size : ℕ)

/-- Calculates the area of the shaded region in the grid -/
def shaded_area (grid : ShadedGrid) : ℕ :=
  let horizontal_squares := grid.width / grid.square_size
  let vertical_squares := grid.height / grid.square_size
  let total_squares := horizontal_squares * vertical_squares
  let shaded_triangle_area := (grid.square_size * grid.square_size) / 2
  total_squares * shaded_triangle_area

/-- Theorem: The shaded area in a 9x7 grid with 2x2 squares is 24 square units -/
theorem shaded_area_9x7_grid :
  let grid : ShadedGrid := ⟨9, 7, 2⟩
  shaded_area grid = 24 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_9x7_grid_l3927_392724


namespace NUMINAMATH_CALUDE_divisible_by_120_l3927_392717

theorem divisible_by_120 (n : ℤ) : ∃ k : ℤ, n^6 + 2*n^5 - n^2 - 2*n = 120*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l3927_392717
