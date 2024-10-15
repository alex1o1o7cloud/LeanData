import Mathlib

namespace NUMINAMATH_CALUDE_train_length_l1837_183719

/-- The length of a train given its speed, time to cross a bridge, and bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 265 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1837_183719


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l1837_183783

theorem min_sum_with_reciprocal_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 2/b = 2) : 
  a + b ≥ 3/2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l1837_183783


namespace NUMINAMATH_CALUDE_min_value_f_when_a_1_range_of_a_for_inequality_l1837_183717

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x + a|

-- Theorem for the minimum value of f when a = 1
theorem min_value_f_when_a_1 :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x 1 ≥ f x_min 1 ∧ f x_min 1 = 3/2 :=
sorry

-- Theorem for the range of a
theorem range_of_a_for_inequality (a : ℝ) :
  (a > 0 ∧ ∃ (x : ℝ), x ∈ [1, 2] ∧ f x a < 5/x + a) ↔ 0 < a ∧ a < 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_when_a_1_range_of_a_for_inequality_l1837_183717


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1837_183715

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, (a < x ∧ x < a + 2) → x > 3) ∧
  (∃ x, x > 3 ∧ ¬(a < x ∧ x < a + 2)) →
  a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1837_183715


namespace NUMINAMATH_CALUDE_garden_width_l1837_183754

/-- A rectangular garden with specific dimensions. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter_eq : width + length = 30
  length_eq : length = width + 8

/-- The width of the garden is 11 feet. -/
theorem garden_width (g : RectangularGarden) : g.width = 11 := by
  sorry

#check garden_width

end NUMINAMATH_CALUDE_garden_width_l1837_183754


namespace NUMINAMATH_CALUDE_marble_ratio_l1837_183738

def marble_problem (pink : ℕ) (orange_diff : ℕ) (total : ℕ) : Prop :=
  let orange := pink - orange_diff
  let purple := total - pink - orange
  purple = 4 * orange

theorem marble_ratio :
  marble_problem 13 9 33 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l1837_183738


namespace NUMINAMATH_CALUDE_cd_case_side_length_l1837_183752

/-- Given a square CD case with a circumference of 60 centimeters,
    prove that the length of one side is 15 centimeters. -/
theorem cd_case_side_length (circumference : ℝ) (side_length : ℝ) 
  (h1 : circumference = 60) 
  (h2 : circumference = 4 * side_length) : 
  side_length = 15 := by
  sorry

end NUMINAMATH_CALUDE_cd_case_side_length_l1837_183752


namespace NUMINAMATH_CALUDE_sequence_problem_l1837_183759

theorem sequence_problem (m : ℕ+) (a : ℕ → ℝ) 
  (h0 : a 0 = 37)
  (h1 : a 1 = 72)
  (hm : a m = 0)
  (h_rec : ∀ k : ℕ, 1 ≤ k → k < m → a (k + 1) = a (k - 1) - 3 / a k) :
  m = 889 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1837_183759


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l1837_183722

theorem number_satisfying_condition : ∃! x : ℝ, x / 3 + 12 = 20 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l1837_183722


namespace NUMINAMATH_CALUDE_larger_number_is_nine_l1837_183798

theorem larger_number_is_nine (a b : ℕ+) (h1 : a - b = 3) (h2 : a^2 + b^2 = 117) : a = 9 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_nine_l1837_183798


namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_max_area_equilateral_triangle_proof_l1837_183781

/-- The maximum area of an equilateral triangle inscribed in a 10x11 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle : ℝ :=
  let rectangle_width := 10
  let rectangle_height := 11
  let max_area := 221 * Real.sqrt 3 - 330
  max_area

/-- Proof that the maximum area of an equilateral triangle inscribed in a 10x11 rectangle is 221√3 - 330 -/
theorem max_area_equilateral_triangle_proof : 
  max_area_equilateral_triangle_in_rectangle = 221 * Real.sqrt 3 - 330 := by
  sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_max_area_equilateral_triangle_proof_l1837_183781


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l1837_183735

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- Predicate to check if a polynomial is nonnegative on [0,1] -/
def IsNonnegativeOn01 (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → P x ≥ 0

/-- Predicate to check if a polynomial is nonnegative on ℝ -/
def IsNonnegativeOnReals (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, P x ≥ 0

theorem polynomial_decomposition (P : RealPolynomial) (h : IsNonnegativeOn01 P) :
  ∃ (P₀ P₁ P₂ : RealPolynomial),
    (IsNonnegativeOnReals P₀) ∧
    (IsNonnegativeOnReals P₁) ∧
    (IsNonnegativeOnReals P₂) ∧
    (∀ x : ℝ, P x = P₀ x + x * P₁ x + (1 - x) * P₂ x) :=
  sorry

end NUMINAMATH_CALUDE_polynomial_decomposition_l1837_183735


namespace NUMINAMATH_CALUDE_geometric_series_equality_l1837_183718

/-- Given real numbers a and b satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 3/4. -/
theorem geometric_series_equality (a b : ℝ) 
  (h : (a / (2 * b)) / (1 - 1 / (2 * b)) = 6) :
  (a / (a + 2 * b)) / (1 - 1 / (a + 2 * b)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l1837_183718


namespace NUMINAMATH_CALUDE_age_cube_sum_l1837_183708

theorem age_cube_sum (j a r : ℕ) (h1 : 2 * j + 3 * a = 4 * r) 
  (h2 : j^3 + a^3 = (1/2) * r^3) (h3 : j + a + r = 50) : 
  j^3 + a^3 + r^3 = 24680 := by
  sorry

end NUMINAMATH_CALUDE_age_cube_sum_l1837_183708


namespace NUMINAMATH_CALUDE_slurpee_purchase_l1837_183742

theorem slurpee_purchase (money_given : ℕ) (slurpee_cost : ℕ) (change : ℕ) : 
  money_given = 20 ∧ slurpee_cost = 2 ∧ change = 8 → 
  (money_given - change) / slurpee_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_slurpee_purchase_l1837_183742


namespace NUMINAMATH_CALUDE_sugar_water_and_triangle_inequality_l1837_183772

theorem sugar_water_and_triangle_inequality 
  (a b m : ℝ) 
  (hab : b > a) (ha : a > 0) (hm : m > 0) 
  (A B C : ℝ) 
  (hABC : A > 0 ∧ B > 0 ∧ C > 0) 
  (hAcute : A < B + C ∧ B < C + A ∧ C < A + B) : 
  (a / b < (a + m) / (b + m)) ∧ 
  (A / (B + C) + B / (C + A) + C / (A + B) < 2) := by
  sorry

end NUMINAMATH_CALUDE_sugar_water_and_triangle_inequality_l1837_183772


namespace NUMINAMATH_CALUDE_triangle_inequality_l1837_183744

theorem triangle_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hab : a ≥ b) (hbc : b ≥ c) :
  Real.sqrt (a * (a + b - Real.sqrt (a * b))) +
  Real.sqrt (b * (a + c - Real.sqrt (a * c))) +
  Real.sqrt (c * (b + c - Real.sqrt (b * c))) ≥
  a + b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1837_183744


namespace NUMINAMATH_CALUDE_garland_arrangement_l1837_183765

theorem garland_arrangement (blue : Nat) (red : Nat) (white : Nat) :
  blue = 8 →
  red = 7 →
  white = 12 →
  (Nat.choose (blue + red) blue) * (Nat.choose (blue + red + 1) white) = 11711700 :=
by sorry

end NUMINAMATH_CALUDE_garland_arrangement_l1837_183765


namespace NUMINAMATH_CALUDE_fraction_of_male_fish_l1837_183703

theorem fraction_of_male_fish (total : ℕ) (female : ℕ) (h1 : total = 45) (h2 : female = 15) :
  (total - female : ℚ) / total = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_male_fish_l1837_183703


namespace NUMINAMATH_CALUDE_c_is_largest_l1837_183755

/-- Given that a - 1 = b + 2 = c - 3 = d + 4, prove that c is the largest among a, b, c, and d -/
theorem c_is_largest (a b c d : ℝ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) :
  c = max a (max b (max c d)) := by
  sorry

end NUMINAMATH_CALUDE_c_is_largest_l1837_183755


namespace NUMINAMATH_CALUDE_indistinguishable_ball_sequences_l1837_183766

/-- The number of different sequences when drawing indistinguishable balls -/
def number_of_sequences (total : ℕ) (white : ℕ) (black : ℕ) : ℕ :=
  Nat.choose total white

theorem indistinguishable_ball_sequences :
  number_of_sequences 13 8 5 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_indistinguishable_ball_sequences_l1837_183766


namespace NUMINAMATH_CALUDE_car_interval_duration_l1837_183732

/-- Proves that the duration of each interval is 1/7.5 hours given the conditions of the car problem -/
theorem car_interval_duration 
  (initial_speed : ℝ) 
  (speed_decrease : ℝ) 
  (fifth_interval_distance : ℝ) 
  (h1 : initial_speed = 45)
  (h2 : speed_decrease = 3)
  (h3 : fifth_interval_distance = 4.4)
  : ∃ (t : ℝ), t = 1 / 7.5 ∧ fifth_interval_distance = (initial_speed - 4 * speed_decrease) * t :=
sorry

end NUMINAMATH_CALUDE_car_interval_duration_l1837_183732


namespace NUMINAMATH_CALUDE_product_fixed_sum_squares_not_always_minimized_when_equal_l1837_183737

theorem product_fixed_sum_squares_not_always_minimized_when_equal :
  ¬ (∀ (k : ℝ), k > 0 →
    ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = k →
      ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = k →
        x^2 + y^2 ≤ a^2 + b^2 → x = y) :=
by sorry

end NUMINAMATH_CALUDE_product_fixed_sum_squares_not_always_minimized_when_equal_l1837_183737


namespace NUMINAMATH_CALUDE_equivalent_representations_l1837_183727

theorem equivalent_representations (x y z w : ℚ) : 
  x = 1 / 8 ∧ 
  y = 2 / 16 ∧ 
  z = 3 / 24 ∧ 
  w = 125 / 1000 → 
  x = y ∧ y = z ∧ z = w := by
sorry

end NUMINAMATH_CALUDE_equivalent_representations_l1837_183727


namespace NUMINAMATH_CALUDE_clown_balloons_l1837_183796

/-- The number of additional balloons blown up by the clown -/
def additional_balloons (initial final : ℕ) : ℕ := final - initial

/-- Theorem: Given the initial and final number of balloons, prove that the clown blew up 13 more balloons -/
theorem clown_balloons : additional_balloons 47 60 = 13 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l1837_183796


namespace NUMINAMATH_CALUDE_value_of_x_l1837_183700

theorem value_of_x : ∀ (w y z x : ℤ), 
  w = 50 → 
  z = w + 25 → 
  y = z + 15 → 
  x = y + 7 → 
  x = 97 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1837_183700


namespace NUMINAMATH_CALUDE_function_properties_l1837_183773

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 - x) = -f (2 + x)) 
  (h2 : ∀ x, f (x + 2) = -f x) : 
  (f 0 = 0) ∧ 
  (∀ x, f (x + 4) = f x) ∧ 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (2 + x) = -f (2 - x)) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1837_183773


namespace NUMINAMATH_CALUDE_circle_radius_l1837_183705

theorem circle_radius (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  ∃ (r : ℝ), r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1837_183705


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_equation_l1837_183721

theorem larger_solution_quadratic_equation :
  ∃ (x y : ℝ), x ≠ y ∧ 
  x^2 - 13*x + 40 = 0 ∧ 
  y^2 - 13*y + 40 = 0 ∧ 
  (∀ z : ℝ, z^2 - 13*z + 40 = 0 → z = x ∨ z = y) ∧
  max x y = 8 := by
sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_equation_l1837_183721


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l1837_183733

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
def arithmeticSum (a₁ d : ℤ) (n : ℕ) : ℤ := n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum_10 (a₁ a₂ a₆ : ℤ) (d : ℤ) :
  a₁ = -2 →
  a₂ + a₆ = 2 →
  (∀ n : ℕ, arithmeticSequence a₁ d n = a₁ + (n - 1) * d) →
  arithmeticSum a₁ d 10 = 25 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l1837_183733


namespace NUMINAMATH_CALUDE_equation_solution_unique_l1837_183750

theorem equation_solution_unique :
  ∃! (x y : ℝ), x ≥ 2 ∧ y ≥ 1 ∧
  36 * Real.sqrt (x - 2) + 4 * Real.sqrt (y - 1) = 28 - 4 * Real.sqrt (x - 2) - Real.sqrt (y - 1) ∧
  x = 5 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_unique_l1837_183750


namespace NUMINAMATH_CALUDE_square_39_relation_l1837_183770

theorem square_39_relation : (39 : ℕ)^2 = (40 : ℕ)^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_39_relation_l1837_183770


namespace NUMINAMATH_CALUDE_no_solution_exists_l1837_183740

/-- S(x) represents the sum of the digits of the natural number x -/
def S (x : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are no natural numbers x satisfying the equation -/
theorem no_solution_exists : ¬ ∃ x : ℕ, x + S x + S (S x) = 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1837_183740


namespace NUMINAMATH_CALUDE_problem_part1_l1837_183713

theorem problem_part1 : (-2)^2 + |Real.sqrt 2 - 1| - Real.sqrt 4 = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_part1_l1837_183713


namespace NUMINAMATH_CALUDE_investment_amount_l1837_183780

/-- Represents the investment scenario with changing interest rates and inflation --/
structure Investment where
  principal : ℝ
  baseRate : ℝ
  years : ℕ
  rateChangeYear2 : ℝ
  rateChangeYear4 : ℝ
  inflationRate : ℝ
  interestDifference : ℝ

/-- Calculates the total interest earned with rate changes --/
def totalInterestWithChanges (inv : Investment) : ℝ :=
  inv.principal * (5 * inv.baseRate + inv.rateChangeYear2 + inv.rateChangeYear4)

/-- Calculates the total interest earned without rate changes --/
def totalInterestWithoutChanges (inv : Investment) : ℝ :=
  inv.principal * 5 * inv.baseRate

/-- Theorem stating that the original investment amount is $30,000 --/
theorem investment_amount (inv : Investment) 
  (h1 : inv.years = 5)
  (h2 : inv.rateChangeYear2 = 0.005)
  (h3 : inv.rateChangeYear4 = 0.01)
  (h4 : inv.inflationRate = 0.01)
  (h5 : totalInterestWithChanges inv - totalInterestWithoutChanges inv = inv.interestDifference)
  (h6 : inv.interestDifference = 450) :
  inv.principal = 30000 := by
  sorry

#check investment_amount

end NUMINAMATH_CALUDE_investment_amount_l1837_183780


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l1837_183734

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  focus : ℝ
  equation : ℝ → ℝ → Prop

/-- A circle with center (a, 0) and radius r -/
structure Circle where
  center : ℝ
  radius : ℝ
  equation : ℝ → ℝ → Prop

/-- The theorem statement -/
theorem parabola_circle_tangency 
  (C : Parabola) 
  (M : Circle)
  (h1 : C.focus > 0)
  (h2 : ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ C.equation 1 y₁ ∧ C.equation 1 y₂)
  (h3 : ∀ (y₁ y₂ : ℝ), y₁ ≠ y₂ → C.equation 1 y₁ → C.equation 1 y₂ → y₁ * y₂ = -1)
  (h4 : M.center = 2)
  (h5 : M.radius = 1) :
  (C.equation = fun x y ↦ y^2 = x) ∧ 
  (M.equation = fun x y ↦ (x - 2)^2 + y^2 = 1) ∧ 
  (∀ (A₁ A₂ A₃ : ℝ × ℝ), 
    C.equation A₁.1 A₁.2 → 
    C.equation A₂.1 A₂.2 → 
    C.equation A₃.1 A₃.2 → 
    (∃ (k₁ k₂ : ℝ), 
      (∀ x y, y = k₁ * (x - A₁.1) + A₁.2 → 
        ((x - M.center)^2 + y^2 = M.radius^2 → x = M.center - M.radius ∨ x = M.center + M.radius)) ∧
      (∀ x y, y = k₂ * (x - A₁.1) + A₁.2 → 
        ((x - M.center)^2 + y^2 = M.radius^2 → x = M.center - M.radius ∨ x = M.center + M.radius))) →
    ∃ (k : ℝ), ∀ x y, y = k * (x - A₂.1) + A₂.2 → 
      ((x - M.center)^2 + y^2 = M.radius^2 → x = M.center - M.radius ∨ x = M.center + M.radius)) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l1837_183734


namespace NUMINAMATH_CALUDE_area_of_trapezoid_DBCE_l1837_183777

-- Define the triangle ABC
structure Triangle (ABC : Type) where
  AB : ℝ
  AC : ℝ
  area : ℝ

-- Define the smallest triangle
def SmallestTriangle : Triangle Unit :=
  { AB := 1, AC := 1, area := 2 }

-- Define the triangle ADE
def TriangleADE : Triangle Unit :=
  { AB := 1, AC := 1, area := 5 * SmallestTriangle.area }

-- Define the triangle ABC
def TriangleABC : Triangle Unit :=
  { AB := 1, AC := 1, area := 80 }

-- Define the trapezoid DBCE
def TrapezoidDBCE : ℝ := TriangleABC.area - TriangleADE.area

-- Theorem statement
theorem area_of_trapezoid_DBCE : TrapezoidDBCE = 70 := by
  sorry

end NUMINAMATH_CALUDE_area_of_trapezoid_DBCE_l1837_183777


namespace NUMINAMATH_CALUDE_min_value_sum_l1837_183757

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l1837_183757


namespace NUMINAMATH_CALUDE_right_triangle_set_l1837_183795

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The theorem stating that only one set of numbers forms a right triangle --/
theorem right_triangle_set :
  ¬(is_right_triangle 0.1 0.2 0.3) ∧
  ¬(is_right_triangle 1 1 2) ∧
  is_right_triangle 10 24 26 ∧
  ¬(is_right_triangle 9 16 25) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_set_l1837_183795


namespace NUMINAMATH_CALUDE_f_of_f_of_3_l1837_183788

def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 4

theorem f_of_f_of_3 : f (f 3) = 692 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_3_l1837_183788


namespace NUMINAMATH_CALUDE_chernov_has_gray_hair_l1837_183776

-- Define the three people
inductive Person : Type
| Sedov : Person
| Chernov : Person
| Ryzhov : Person

-- Define the hair colors
inductive HairColor : Type
| Gray : HairColor
| Red : HairColor
| Black : HairColor

-- Define the sports ranks
inductive SportsRank : Type
| MasterOfSports : SportsRank
| CandidateMaster : SportsRank
| FirstRank : SportsRank

-- Define the function that assigns a hair color to each person
def hairColor : Person → HairColor := sorry

-- Define the function that assigns a sports rank to each person
def sportsRank : Person → SportsRank := sorry

-- State the theorem
theorem chernov_has_gray_hair :
  -- No person's hair color matches their surname
  (hairColor Person.Sedov ≠ HairColor.Gray) ∧
  (hairColor Person.Chernov ≠ HairColor.Black) ∧
  (hairColor Person.Ryzhov ≠ HairColor.Red) ∧
  -- One person is gray-haired, one is red-haired, and one is black-haired
  (∃! p : Person, hairColor p = HairColor.Gray) ∧
  (∃! p : Person, hairColor p = HairColor.Red) ∧
  (∃! p : Person, hairColor p = HairColor.Black) ∧
  -- The black-haired person made the statement
  (∃ p : Person, hairColor p = HairColor.Black ∧ p ≠ Person.Sedov ∧ p ≠ Person.Chernov) ∧
  -- The Master of Sports confirmed the statement
  (sportsRank Person.Sedov = SportsRank.MasterOfSports) ∧
  (sportsRank Person.Chernov = SportsRank.CandidateMaster) ∧
  (sportsRank Person.Ryzhov = SportsRank.FirstRank) →
  -- Conclusion: Chernov has gray hair
  hairColor Person.Chernov = HairColor.Gray :=
by
  sorry


end NUMINAMATH_CALUDE_chernov_has_gray_hair_l1837_183776


namespace NUMINAMATH_CALUDE_pool_width_proof_l1837_183774

theorem pool_width_proof (drain_rate : ℝ) (drain_time : ℝ) (length : ℝ) (depth : ℝ) 
  (h1 : drain_rate = 60)
  (h2 : drain_time = 2000)
  (h3 : length = 150)
  (h4 : depth = 10) :
  drain_rate * drain_time / (length * depth) = 80 :=
by sorry

end NUMINAMATH_CALUDE_pool_width_proof_l1837_183774


namespace NUMINAMATH_CALUDE_intersection_distance_l1837_183746

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^3
def curve2 (x y : ℝ) : Prop := x + y^3 = 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve1 x y ∧ curve2 x y}

-- State the theorem
theorem intersection_distance :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1837_183746


namespace NUMINAMATH_CALUDE_gina_college_cost_l1837_183736

/-- Calculates the total cost of Gina's college expenses -/
def total_college_cost (num_credits : ℕ) (cost_per_credit : ℕ) (num_textbooks : ℕ) (cost_per_textbook : ℕ) (facilities_fee : ℕ) : ℕ :=
  num_credits * cost_per_credit + num_textbooks * cost_per_textbook + facilities_fee

/-- Proves that Gina's total college expenses are $7100 -/
theorem gina_college_cost :
  total_college_cost 14 450 5 120 200 = 7100 := by
  sorry

end NUMINAMATH_CALUDE_gina_college_cost_l1837_183736


namespace NUMINAMATH_CALUDE_factorial_sum_theorem_l1837_183789

def is_solution (x y : ℕ) (z : ℤ) : Prop :=
  (Nat.factorial x + Nat.factorial y = 16 * z + 2017) ∧
  z % 2 ≠ 0

theorem factorial_sum_theorem :
  ∀ x y : ℕ, ∀ z : ℤ,
    is_solution x y z →
    ((x = 1 ∧ y = 6 ∧ z = -81) ∨
     (x = 6 ∧ y = 1 ∧ z = -81) ∨
     (x = 1 ∧ y = 7 ∧ z = 189) ∨
     (x = 7 ∧ y = 1 ∧ z = 189)) :=
by
  sorry

#check factorial_sum_theorem

end NUMINAMATH_CALUDE_factorial_sum_theorem_l1837_183789


namespace NUMINAMATH_CALUDE_pastries_cakes_difference_l1837_183710

theorem pastries_cakes_difference (pastries_sold : ℕ) (cakes_sold : ℕ) 
  (h1 : pastries_sold = 154) (h2 : cakes_sold = 78) : 
  pastries_sold - cakes_sold = 76 := by
  sorry

end NUMINAMATH_CALUDE_pastries_cakes_difference_l1837_183710


namespace NUMINAMATH_CALUDE_range_of_a_l1837_183707

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * x^2 + a * x - a^2 = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- Theorem statement
theorem range_of_a (a : ℝ) : ¬(p a ∨ q a) ↔ a > 2 ∨ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1837_183707


namespace NUMINAMATH_CALUDE_sphere_radius_from_depression_l1837_183797

theorem sphere_radius_from_depression (r : ℝ) 
  (depression_depth : ℝ) (depression_diameter : ℝ) : 
  depression_depth = 8 ∧ 
  depression_diameter = 24 ∧ 
  r^2 = (r - depression_depth)^2 + (depression_diameter / 2)^2 → 
  r = 13 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_depression_l1837_183797


namespace NUMINAMATH_CALUDE_agent_encryption_possible_l1837_183731

theorem agent_encryption_possible : ∃ (m n p q : ℕ), 
  (m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0) ∧ 
  (7 / 100 : ℚ) = 1 / m + 1 / n ∧
  (13 / 100 : ℚ) = 1 / p + 1 / q :=
sorry

end NUMINAMATH_CALUDE_agent_encryption_possible_l1837_183731


namespace NUMINAMATH_CALUDE_bachuan_jiaoqing_extrema_l1837_183769

/-- Definition of a "Bachuan Jiaoqing password number" -/
def is_bachuan_jiaoqing (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 ≤ n ∧ n < 10000 ∧ b ≥ c ∧ a = b + c ∧ d = b - c

/-- Additional divisibility condition -/
def satisfies_divisibility (n : ℕ) : Prop :=
  let a := n / 1000
  let bcd := n % 1000
  (bcd - 7 * a) % 13 = 0

/-- Theorem stating the largest and smallest "Bachuan Jiaoqing password numbers" -/
theorem bachuan_jiaoqing_extrema :
  (∀ n, is_bachuan_jiaoqing n → n ≤ 9909) ∧
  (∃ n, is_bachuan_jiaoqing n ∧ satisfies_divisibility n ∧
    ∀ m, is_bachuan_jiaoqing m ∧ satisfies_divisibility m → n ≤ m) ∧
  (is_bachuan_jiaoqing 9909) ∧
  (is_bachuan_jiaoqing 5321 ∧ satisfies_divisibility 5321) := by
  sorry

end NUMINAMATH_CALUDE_bachuan_jiaoqing_extrema_l1837_183769


namespace NUMINAMATH_CALUDE_square_perimeter_l1837_183741

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 360 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 24 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l1837_183741


namespace NUMINAMATH_CALUDE_inequality_solution_l1837_183714

theorem inequality_solution (x : ℝ) : 
  let x₁ : ℝ := (-9 - Real.sqrt 21) / 2
  let x₂ : ℝ := (-9 + Real.sqrt 21) / 2
  (x - 1) / (x + 3) > (4 * x + 5) / (3 * x + 8) ↔ 
    (x > -3 ∧ x < x₁) ∨ (x > x₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1837_183714


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_2395_l1837_183767

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 8 * (x^3 - 2*x^2 + 4*x - 1)

/-- The sum of squares of coefficients of the simplified expression -/
def sum_of_squared_coefficients : ℝ := 2395

theorem sum_of_squared_coefficients_is_2395 :
  sum_of_squared_coefficients = 2395 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_2395_l1837_183767


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l1837_183761

/-- The speed of the east-bound cyclist in mph -/
def east_speed : ℝ := 18

/-- The speed of the west-bound cyclist in mph -/
def west_speed : ℝ := east_speed + 4

/-- The time traveled in hours -/
def time : ℝ := 5

/-- The total distance between the cyclists after the given time -/
def total_distance : ℝ := 200

theorem cyclist_speed_proof :
  east_speed * time + west_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l1837_183761


namespace NUMINAMATH_CALUDE_fraction_order_l1837_183753

theorem fraction_order : 
  let f1 := (4 : ℚ) / 3
  let f2 := (4 : ℚ) / 5
  let f3 := (4 : ℚ) / 6
  let f4 := (3 : ℚ) / 5
  let f5 := (6 : ℚ) / 5
  let f6 := (2 : ℚ) / 5
  (f6 < f4) ∧ (f4 < f3) ∧ (f3 < f2) ∧ (f2 < f5) ∧ (f5 < f1) := by
sorry

end NUMINAMATH_CALUDE_fraction_order_l1837_183753


namespace NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l1837_183745

theorem min_product_of_reciprocal_sum (a b : ℕ+) 
  (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (6 : ℚ)⁻¹) : 
  (∀ c d : ℕ+, (c : ℚ)⁻¹ + (3 * d : ℚ)⁻¹ = (6 : ℚ)⁻¹ → a * b ≤ c * d) ∧ a * b = 48 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l1837_183745


namespace NUMINAMATH_CALUDE_sum_of_squares_l1837_183791

theorem sum_of_squares (n : ℕ) (h1 : n > 2) 
  (h2 : ∃ m : ℕ, n^2 = (m + 1)^3 - m^3) : 
  ∃ a b : ℕ, n = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1837_183791


namespace NUMINAMATH_CALUDE_tims_soda_cans_l1837_183751

theorem tims_soda_cans (x : ℕ) : 
  x - 6 + (x - 6) / 2 = 24 → x = 22 := by sorry

end NUMINAMATH_CALUDE_tims_soda_cans_l1837_183751


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1837_183725

theorem polynomial_evaluation (f : ℝ → ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, f x = (1 - 3*x) * (1 + x)^5) →
  (∀ x, f x = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + (1/3)*a₁ + (1/3^2)*a₂ + (1/3^3)*a₃ + (1/3^4)*a₄ + (1/3^5)*a₅ + (1/3^6)*a₆ = 0 :=
by sorry


end NUMINAMATH_CALUDE_polynomial_evaluation_l1837_183725


namespace NUMINAMATH_CALUDE_simplify_fraction_l1837_183782

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 50 + Real.sqrt 32 + 3 * Real.sqrt 18) = (5 * Real.sqrt 2) / 36 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1837_183782


namespace NUMINAMATH_CALUDE_subset_intersection_implies_empty_complement_l1837_183729

theorem subset_intersection_implies_empty_complement
  (A B : Set ℝ) (h : A ⊆ A ∩ B) : A ∩ (Set.univ \ B) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_implies_empty_complement_l1837_183729


namespace NUMINAMATH_CALUDE_simple_interest_principal_l1837_183709

/-- Simple interest calculation -/
theorem simple_interest_principal 
  (interest : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : interest = 4016.25)
  (h2 : rate = 9)
  (h3 : time = 5) : 
  ∃ (principal : ℝ), principal = 8925 ∧ interest = principal * rate * time / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l1837_183709


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l1837_183749

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := -Real.log x

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂) :=
by sorry

end

end NUMINAMATH_CALUDE_f_satisfies_conditions_l1837_183749


namespace NUMINAMATH_CALUDE_condition_equivalence_l1837_183778

theorem condition_equivalence (x : ℝ) : x > 0 ↔ x + 1/x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_condition_equivalence_l1837_183778


namespace NUMINAMATH_CALUDE_function_zeros_inequality_l1837_183763

open Real

theorem function_zeros_inequality (a b c : ℝ) (x₁ x₂ : ℝ) :
  0 < a → a < 1 → b > 0 →
  let f := fun x => a * exp x - b * x - c
  f x₁ = 0 → f x₂ = 0 → x₁ > x₂ →
  exp x₁ / a + exp x₂ / (1 - a) > 4 * b / a :=
by sorry

end NUMINAMATH_CALUDE_function_zeros_inequality_l1837_183763


namespace NUMINAMATH_CALUDE_factorization1_factorization2_factorization3_l1837_183764

-- Given formulas
axiom formula1 (x a b : ℝ) : (x + a) * (x + b) = x^2 + (a + b) * x + a * b
axiom formula2 (x y : ℝ) : (x + y)^2 + 2 * (x + y) + 1 = (x + y + 1)^2

-- Theorems to prove
theorem factorization1 (x : ℝ) : x^2 + 4 * x + 3 = (x + 3) * (x + 1) := by sorry

theorem factorization2 (x y : ℝ) : (x - y)^2 - 10 * (x - y) + 25 = (x - y - 5)^2 := by sorry

theorem factorization3 (m : ℝ) : (m^2 - 2 * m) * (m^2 - 2 * m + 4) + 3 = (m^2 - 2 * m + 3) * (m - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization1_factorization2_factorization3_l1837_183764


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1837_183724

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | 56 * x^2 + a * x - a^2 < 0} = {x : ℝ | a / 8 < x ∧ x < -a / 7} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1837_183724


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l1837_183748

def number_of_arrangements (men women : ℕ) : ℕ :=
  let first_group := men.choose 1 * women.choose 2
  let remaining_men := men - 1
  let remaining_women := women - 2
  let remaining_groups := remaining_men.choose 1 * remaining_women.choose 2
  first_group * remaining_groups

theorem arrangement_count_correct :
  number_of_arrangements 4 5 = 360 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l1837_183748


namespace NUMINAMATH_CALUDE_third_term_of_x_plus_two_pow_five_l1837_183739

/-- The coefficient of the r-th term in the expansion of (a + b)^n -/
def binomial_coefficient (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n r

/-- The r-th term in the expansion of (a + b)^n -/
def binomial_term (n : ℕ) (r : ℕ) (a b : ℚ) : ℚ :=
  (binomial_coefficient n r : ℚ) * a^(n - r) * b^r

/-- The third term of (x + 2)^5 is 40x^3 -/
theorem third_term_of_x_plus_two_pow_five (x : ℚ) :
  binomial_term 5 2 x 2 = 40 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_x_plus_two_pow_five_l1837_183739


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1837_183712

theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1837_183712


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l1837_183775

theorem constant_ratio_problem (x y : ℝ) (k : ℝ) :
  (∀ x y, (4 * x - 5) / (2 * y + 20) = k) →
  (4 * 4 - 5) / (2 * 5 + 20) = k →
  (4 * 9 - 5) / (2 * (355 / 11) + 20) = k :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l1837_183775


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l1837_183704

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, n > 0 ∧
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 ∧ m < n →
    ¬(∃ q₁ q₂ q₃ q₄ q₅ : ℕ,
      Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧ Nat.Prime q₄ ∧ Nat.Prime q₅ ∧
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l1837_183704


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1837_183771

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def cow_count : ℕ := 20
def cow_cost_per_unit : ℕ := 1000
def chicken_count : ℕ := 100
def chicken_cost_per_unit : ℕ := 5
def solar_installation_hours : ℕ := 6
def solar_installation_cost_per_hour : ℕ := 100
def solar_equipment_cost : ℕ := 6000

theorem total_cost_calculation :
  land_acres * land_cost_per_acre +
  house_cost +
  cow_count * cow_cost_per_unit +
  chicken_count * chicken_cost_per_unit +
  solar_installation_hours * solar_installation_cost_per_hour +
  solar_equipment_cost = 147700 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1837_183771


namespace NUMINAMATH_CALUDE_complex_inequality_l1837_183723

theorem complex_inequality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
  Complex.abs (a - b) ≥ (1/2 : ℝ) * (Complex.abs a + Complex.abs b) * Complex.abs ((a / Complex.abs a) - (b / Complex.abs b)) ∧
  (Complex.abs (a - b) = (1/2 : ℝ) * (Complex.abs a + Complex.abs b) * Complex.abs ((a / Complex.abs a) - (b / Complex.abs b)) ↔ Complex.abs a = Complex.abs b) :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_l1837_183723


namespace NUMINAMATH_CALUDE_fourth_group_number_l1837_183726

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  sample_size : Nat
  second_group_number : Nat

/-- The number drawn from a specific group in systematic sampling -/
def number_in_group (setup : SystematicSampling) (group : Nat) : Nat :=
  setup.second_group_number + (group - 2) * (setup.total_students / setup.num_groups)

/-- Theorem stating the relationship between the numbers drawn from different groups -/
theorem fourth_group_number (setup : SystematicSampling) 
  (h1 : setup.total_students = 72)
  (h2 : setup.num_groups = 6)
  (h3 : setup.sample_size = 6)
  (h4 : setup.second_group_number = 16) :
  number_in_group setup 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_number_l1837_183726


namespace NUMINAMATH_CALUDE_bela_wins_iff_m_odd_l1837_183716

/-- The game interval --/
def GameInterval (m : ℕ) := Set.Icc (0 : ℝ) m

/-- Predicate for a valid move --/
def ValidMove (m : ℕ) (prev_moves : List ℝ) (x : ℝ) : Prop :=
  x ∈ GameInterval m ∧ ∀ y ∈ prev_moves, |x - y| > 2

/-- The game result --/
inductive GameResult
  | BelaWins
  | JennWins

/-- The game outcome based on the optimal strategy --/
def GameOutcome (m : ℕ) : GameResult :=
  if m % 2 = 1 then GameResult.BelaWins else GameResult.JennWins

/-- The main theorem --/
theorem bela_wins_iff_m_odd (m : ℕ) (h : m > 2) :
  GameOutcome m = GameResult.BelaWins ↔ m % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_bela_wins_iff_m_odd_l1837_183716


namespace NUMINAMATH_CALUDE_x_24_value_l1837_183730

theorem x_24_value (x : ℝ) (h : x + 1/x = -Real.sqrt 3) : x^24 = 390625 := by
  sorry

end NUMINAMATH_CALUDE_x_24_value_l1837_183730


namespace NUMINAMATH_CALUDE_equal_cost_guests_l1837_183786

def caesars_cost (guests : ℕ) : ℚ := 800 + 30 * guests
def venus_cost (guests : ℕ) : ℚ := 500 + 35 * guests

theorem equal_cost_guests : ∃ (x : ℕ), caesars_cost x = venus_cost x ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_guests_l1837_183786


namespace NUMINAMATH_CALUDE_max_value_of_sum_product_l1837_183779

theorem max_value_of_sum_product (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 → 
  a * b + a * c + a * d ≤ 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_product_l1837_183779


namespace NUMINAMATH_CALUDE_confectioner_pastry_count_l1837_183706

theorem confectioner_pastry_count :
  ∀ (P : ℕ),
  (P / 28 : ℚ) - (P / 49 : ℚ) = 6 →
  P = 378 :=
by
  sorry

end NUMINAMATH_CALUDE_confectioner_pastry_count_l1837_183706


namespace NUMINAMATH_CALUDE_min_teachers_for_given_counts_l1837_183756

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  english : Nat
  history : Nat
  geography : Nat

/-- Calculates the minimum number of teachers required to cover all subjects -/
def minTeachersRequired (counts : TeacherCounts) : Nat :=
  sorry

/-- Theorem stating the minimum number of teachers required for the given conditions -/
theorem min_teachers_for_given_counts :
  let counts : TeacherCounts := { english := 9, history := 7, geography := 6 }
  minTeachersRequired counts = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_teachers_for_given_counts_l1837_183756


namespace NUMINAMATH_CALUDE_inequality_solution_l1837_183758

theorem inequality_solution (x : ℝ) : 2 ≤ x / (2 * x - 4) ∧ x / (2 * x - 4) < 7 ↔ x ∈ Set.Ici 2 ∩ Set.Iio (28 / 13) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1837_183758


namespace NUMINAMATH_CALUDE_brothers_sisters_ratio_l1837_183762

theorem brothers_sisters_ratio :
  ∀ (num_brothers : ℕ),
    (num_brothers + 2) * 2 = 12 →
    num_brothers / 2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_sisters_ratio_l1837_183762


namespace NUMINAMATH_CALUDE_infinite_functions_satisfying_condition_l1837_183768

theorem infinite_functions_satisfying_condition :
  ∃ (S : Set (ℝ → ℝ)), (Set.Infinite S) ∧ 
  (∀ f ∈ S, 2 * f 3 - 10 = f 1) := by
sorry

end NUMINAMATH_CALUDE_infinite_functions_satisfying_condition_l1837_183768


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1837_183743

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + Real.cos x

theorem inequality_solution_set (m : ℝ) :
  f (2 * m) > f (m - 2) ↔ m < -2 ∨ m > 2/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1837_183743


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l1837_183702

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of ways to choose 3 vertices from 10 vertices -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with one side being a side of the decagon -/
def one_side_triangles : ℕ := decagon_vertices * 5

/-- The number of triangles with two sides being sides of the decagon -/
def two_side_triangles : ℕ := decagon_vertices

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of selecting a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l1837_183702


namespace NUMINAMATH_CALUDE_jills_total_earnings_l1837_183790

/-- Calculates Jill's earnings over three months based on specific working conditions. -/
def jills_earnings (days_per_month : ℕ) (first_month_rate : ℕ) : ℕ :=
  let second_month_rate := 2 * first_month_rate
  let first_month := days_per_month * first_month_rate
  let second_month := days_per_month * second_month_rate
  let third_month := (days_per_month / 2) * second_month_rate
  first_month + second_month + third_month

/-- Theorem stating that Jill's earnings over three months equal $1,200 -/
theorem jills_total_earnings : 
  jills_earnings 30 10 = 1200 := by
  sorry

#eval jills_earnings 30 10

end NUMINAMATH_CALUDE_jills_total_earnings_l1837_183790


namespace NUMINAMATH_CALUDE_equality_comparison_l1837_183785

theorem equality_comparison : 
  (2^3 ≠ 6) ∧ 
  (-1^2 ≠ (-1)^2) ∧ 
  (-2^3 = (-2)^3) ∧ 
  (4^2 / 9 ≠ (4/9)^2) :=
by sorry

end NUMINAMATH_CALUDE_equality_comparison_l1837_183785


namespace NUMINAMATH_CALUDE_binomial_coefficient_200_l1837_183787

theorem binomial_coefficient_200 :
  (Nat.choose 200 200 = 1) ∧ (Nat.choose 200 0 = 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_200_l1837_183787


namespace NUMINAMATH_CALUDE_camp_acquaintances_l1837_183728

/-- Represents the number of acquaintances of a child -/
def Acquaintances : Type := ℕ

/-- Represents a child in the group -/
structure Child :=
  (name : String)
  (acquaintances : Acquaintances)

/-- The fraction of one child's acquaintances who are also acquainted with another child -/
def mutualAcquaintanceFraction (a b : Child) : ℚ := sorry

/-- Petya, one of the children in the group -/
def petya : Child := ⟨"Petya", sorry⟩

/-- Vasya, one of the children in the group -/
def vasya : Child := ⟨"Vasya", sorry⟩

/-- Timofey, one of the children in the group -/
def timofey : Child := ⟨"Timofey", sorry⟩

theorem camp_acquaintances :
  (mutualAcquaintanceFraction petya vasya = 1/2) →
  (mutualAcquaintanceFraction petya timofey = 1/7) →
  (mutualAcquaintanceFraction vasya petya = 1/3) →
  (mutualAcquaintanceFraction vasya timofey = 1/6) →
  (mutualAcquaintanceFraction timofey petya = 1/5) →
  (mutualAcquaintanceFraction timofey vasya = 7/20) :=
by sorry

end NUMINAMATH_CALUDE_camp_acquaintances_l1837_183728


namespace NUMINAMATH_CALUDE_difference_of_squares_l1837_183760

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) :
  x^2 - y^2 = 160 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1837_183760


namespace NUMINAMATH_CALUDE_gcd_217_155_l1837_183794

theorem gcd_217_155 : Nat.gcd 217 155 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_217_155_l1837_183794


namespace NUMINAMATH_CALUDE_initial_potatoes_l1837_183747

theorem initial_potatoes (initial_tomatoes picked_potatoes remaining_total : ℕ) : 
  initial_tomatoes = 175 →
  picked_potatoes = 172 →
  remaining_total = 80 →
  initial_tomatoes + (initial_tomatoes + picked_potatoes - remaining_total) = 175 + 77 :=
by sorry

end NUMINAMATH_CALUDE_initial_potatoes_l1837_183747


namespace NUMINAMATH_CALUDE_tan_problem_l1837_183793

theorem tan_problem (α β : Real) 
  (h1 : Real.tan (π/4 + α) = 2) 
  (h2 : Real.tan β = 1/2) : 
  Real.tan α = 1/3 ∧ 
  (Real.sin (α + β) - 2 * Real.sin α * Real.cos β) / 
  (2 * Real.sin α * Real.sin β + Real.cos (α + β)) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_problem_l1837_183793


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1837_183720

theorem absolute_value_inequality (x : ℝ) : 
  x ≠ 2 → (|(3 * x - 2) / (x - 2)| > 3 ↔ (4/3 < x ∧ x < 2) ∨ x > 2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1837_183720


namespace NUMINAMATH_CALUDE_series_sum_equals_five_l1837_183711

theorem series_sum_equals_five (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (7 * n + 2) / k^n = 5) : k = (7 + Real.sqrt 14) / 5 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_five_l1837_183711


namespace NUMINAMATH_CALUDE_det_dilation_matrix_5_l1837_183792

/-- A 2x2 matrix representing a dilation with scale factor k centered at the origin -/
def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- Theorem: The determinant of a 2x2 dilation matrix with scale factor 5 is 25 -/
theorem det_dilation_matrix_5 :
  let E := dilation_matrix 5
  Matrix.det E = 25 := by sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_5_l1837_183792


namespace NUMINAMATH_CALUDE_least_possible_difference_l1837_183799

theorem least_possible_difference (x y z N : ℤ) (h1 : x < y) (h2 : y < z) 
  (h3 : y - x > 5) (h4 : Even x) (h5 : Odd y) (h6 : Odd z) (h7 : ∃ k : ℤ, x = 5 * k) 
  (h8 : y^2 + z^2 = N) (h9 : N > 0) : 
  (∀ w : ℤ, w ≥ 0 → z - x ≥ w + 9) ∧ (z - x = 9) :=
sorry

end NUMINAMATH_CALUDE_least_possible_difference_l1837_183799


namespace NUMINAMATH_CALUDE_stock_price_increase_l1837_183784

theorem stock_price_increase (P : ℝ) (X : ℝ) : 
  P * (1 + X / 100) * 0.75 * 1.35 = P * 1.215 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l1837_183784


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1837_183701

-- Define the surface area of the cube
def surface_area : ℝ := 1350

-- Theorem stating the relationship between surface area and volume
theorem cube_volume_from_surface_area :
  ∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧
    side_length > 0 ∧
    side_length^3 = 3375 := by
  sorry


end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1837_183701
