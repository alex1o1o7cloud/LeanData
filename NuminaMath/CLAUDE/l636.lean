import Mathlib

namespace NUMINAMATH_CALUDE_exists_concave_to_convex_map_not_exists_convex_to_concave_map_l636_63611

-- Define the plane
def Plane := ℝ × ℝ

-- Define a polygon as a list of points in the plane
def Polygon := List Plane

-- Define a simple polygon
def SimplePolygon (p : Polygon) : Prop := sorry

-- Define a convex polygon
def ConvexPolygon (p : Polygon) : Prop := sorry

-- Define a concave polygon
def ConcavePolygon (p : Polygon) : Prop := ¬ConvexPolygon p

-- State the existence of the function for part (a)
theorem exists_concave_to_convex_map :
  ∃ (f : Plane → Plane), ∀ (n : ℕ) (p : Polygon),
    n ≥ 4 →
    SimplePolygon p →
    ConcavePolygon p →
    p.length = n →
    ∃ (q : Polygon), SimplePolygon q ∧ ConvexPolygon q ∧ q = p.map f :=
sorry

-- State the non-existence of the function for part (b)
theorem not_exists_convex_to_concave_map :
  ¬∃ (f : Plane → Plane), ∀ (n : ℕ) (p : Polygon),
    n ≥ 4 →
    SimplePolygon p →
    ConvexPolygon p →
    p.length = n →
    ∃ (q : Polygon), SimplePolygon q ∧ ConcavePolygon q ∧ q = p.map f :=
sorry

end NUMINAMATH_CALUDE_exists_concave_to_convex_map_not_exists_convex_to_concave_map_l636_63611


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l636_63622

/-- Given a quadratic function f(x) = x^2 - 2ax + b where a and b are real numbers,
    and the solution set of f(x) ≤ 0 is [-1, 2], this theorem proves two statements about f. -/
theorem quadratic_function_properties (a b : ℝ) 
    (f : ℝ → ℝ) 
    (h_f : ∀ x, f x = x^2 - 2*a*x + b) 
    (h_solution_set : Set.Icc (-1 : ℝ) 2 = {x | f x ≤ 0}) : 
  (∀ x, b*x^2 - 2*a*x + 1 ≤ 0 ↔ x ≤ -1 ∨ x ≥ 1/2) ∧ 
  (b = a^2 → 
   (∀ x₁ ∈ Set.Icc 2 4, ∃ x₂ ∈ Set.Icc 2 4, f x₁ * f x₂ = 1) → 
   a = 3 + Real.sqrt 2 ∨ a = 3 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l636_63622


namespace NUMINAMATH_CALUDE_x_is_perfect_square_l636_63653

theorem x_is_perfect_square (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x > y)
  (h_div : (x^2019 + x + y^2) % (x*y) = 0) : 
  ∃ (n : ℕ), x = n^2 := by
sorry

end NUMINAMATH_CALUDE_x_is_perfect_square_l636_63653


namespace NUMINAMATH_CALUDE_euler_totient_equality_l636_63689

-- Define the Euler's totient function
def phi (n : ℕ) : ℕ := sorry

-- Define the property of being an odd number
def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

-- Theorem statement
theorem euler_totient_equality (n : ℕ) (p : ℕ) (h_p : Prime p) :
  phi n = phi (n * p) ↔ p = 2 ∧ is_odd n :=
sorry

end NUMINAMATH_CALUDE_euler_totient_equality_l636_63689


namespace NUMINAMATH_CALUDE_total_marbles_count_l636_63666

def initial_marbles : ℝ := 87.0
def received_marbles : ℝ := 8.0

theorem total_marbles_count : 
  initial_marbles + received_marbles = 95.0 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_count_l636_63666


namespace NUMINAMATH_CALUDE_parity_of_expression_l636_63683

theorem parity_of_expression (o n c : ℤ) 
  (ho : Odd o) (hc : Odd c) : Even (o^2 + n*o + c) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_expression_l636_63683


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l636_63618

theorem quadratic_equation_solution : ∃ (a b : ℝ), 
  (a^2 - 6*a + 11 = 27) ∧ 
  (b^2 - 6*b + 11 = 27) ∧ 
  (a ≥ b) ∧ 
  (3*a - 2*b = 28) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l636_63618


namespace NUMINAMATH_CALUDE_marcus_car_mileage_l636_63612

/-- Calculates the final mileage of a car after a road trip --/
def final_mileage (initial_mileage : ℕ) (tank_capacity : ℕ) (fuel_efficiency : ℕ) (refills : ℕ) : ℕ :=
  initial_mileage + tank_capacity * refills * fuel_efficiency

/-- Theorem stating the final mileage of Marcus' car after the road trip --/
theorem marcus_car_mileage :
  final_mileage 1728 20 30 2 = 2928 := by
  sorry

#eval final_mileage 1728 20 30 2

end NUMINAMATH_CALUDE_marcus_car_mileage_l636_63612


namespace NUMINAMATH_CALUDE_equation_solution_l636_63621

theorem equation_solution (y : ℚ) : 
  (8 * y^2 + 127 * y + 5) / (4 * y + 41) = 2 * y + 3 → y = 118 / 33 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l636_63621


namespace NUMINAMATH_CALUDE_square_sum_equals_34_l636_63661

theorem square_sum_equals_34 (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 4.5) : a^2 + b^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_34_l636_63661


namespace NUMINAMATH_CALUDE_yacht_distance_squared_l636_63659

theorem yacht_distance_squared (AB BC : ℝ) (angle_B : ℝ) (AC_squared : ℝ) : 
  AB = 15 → 
  BC = 25 → 
  angle_B = 150 * Real.pi / 180 →
  AC_squared = AB^2 + BC^2 - 2 * AB * BC * Real.cos angle_B →
  AC_squared = 850 - 375 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_yacht_distance_squared_l636_63659


namespace NUMINAMATH_CALUDE_slope_of_line_l636_63687

theorem slope_of_line (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : 1 / x₁ + 2 / y₁ = 0) (h₃ : 1 / x₂ + 2 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -2 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l636_63687


namespace NUMINAMATH_CALUDE_n_value_l636_63664

theorem n_value (n : ℝ) (h1 : n > 0) (h2 : Real.sqrt (4 * n^2) = 64) : n = 32 := by
  sorry

end NUMINAMATH_CALUDE_n_value_l636_63664


namespace NUMINAMATH_CALUDE_union_complement_equal_l636_63649

def U : Finset ℕ := {1,2,3,4,5,6}
def M : Finset ℕ := {1,3,4}
def N : Finset ℕ := {3,5,6}

theorem union_complement_equal : M ∪ (U \ N) = {1,2,3,4} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equal_l636_63649


namespace NUMINAMATH_CALUDE_rectangle_area_l636_63697

theorem rectangle_area (width : ℝ) (length : ℝ) (h1 : width = 4) (h2 : length = 3 * width) :
  width * length = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l636_63697


namespace NUMINAMATH_CALUDE_rectangle_area_l636_63693

theorem rectangle_area (square_area : ℝ) (rect_length rect_width : ℝ) : 
  square_area = 36 →
  4 * square_area.sqrt = 2 * (rect_length + rect_width) →
  rect_length = 3 * rect_width →
  rect_length * rect_width = 27 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l636_63693


namespace NUMINAMATH_CALUDE_jamie_coin_problem_l636_63629

/-- The number of nickels (and dimes and quarters) in Jamie's jar -/
def num_coins : ℕ := 33

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of coins in Jamie's jar in cents -/
def total_value : ℕ := 1320

theorem jamie_coin_problem :
  num_coins * nickel_value + num_coins * dime_value + num_coins * quarter_value = total_value :=
by sorry

end NUMINAMATH_CALUDE_jamie_coin_problem_l636_63629


namespace NUMINAMATH_CALUDE_cloak_change_in_silver_l636_63639

/-- Represents the price of an invisibility cloak and the change received in different scenarios --/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Calculates the exchange rate between silver and gold coins --/
def exchange_rate (t1 t2 : CloakTransaction) : ℚ :=
  (t1.silver_paid - t2.silver_paid : ℚ) / (t1.gold_change - t2.gold_change)

/-- Calculates the price of the cloak in gold coins --/
def cloak_price_gold (t : CloakTransaction) (rate : ℚ) : ℚ :=
  t.silver_paid / rate - t.gold_change

/-- Theorem stating the change received when buying a cloak with gold coins --/
theorem cloak_change_in_silver 
  (t1 t2 : CloakTransaction)
  (h1 : t1.silver_paid = 20 ∧ t1.gold_change = 4)
  (h2 : t2.silver_paid = 15 ∧ t2.gold_change = 1)
  (gold_paid : ℕ)
  (h3 : gold_paid = 14) :
  ∃ (silver_change : ℕ), silver_change = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloak_change_in_silver_l636_63639


namespace NUMINAMATH_CALUDE_cherries_theorem_l636_63606

def cherries_problem (initial_cherries : ℕ) (difference : ℕ) : ℕ :=
  initial_cherries - difference

theorem cherries_theorem (initial_cherries : ℕ) (difference : ℕ) 
  (h1 : initial_cherries = 16) (h2 : difference = 10) :
  cherries_problem initial_cherries difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_cherries_theorem_l636_63606


namespace NUMINAMATH_CALUDE_trajectory_and_circle_properties_l636_63626

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the fixed line l
def line_l (x : ℝ) : Prop := x = 4

-- Define point F as the intersection of parabola and line l
def point_F : ℝ × ℝ := (2, 4)

-- Define the condition for point P
def condition_P (P Q F : ℝ × ℝ) : Prop :=
  let PQ := (Q.1 - P.1, Q.2 - P.2)
  let PF := (F.1 - P.1, F.2 - P.2)
  (PQ.1 + Real.sqrt 2 * PF.1, PQ.2 + Real.sqrt 2 * PF.2) • (PQ.1 - Real.sqrt 2 * PF.1, PQ.2 - Real.sqrt 2 * PF.2) = 0

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 8/3

-- Define the range of |AB|
def range_AB (ab : ℝ) : Prop := 4 * Real.sqrt 6 / 3 ≤ ab ∧ ab ≤ 2 * Real.sqrt 3

theorem trajectory_and_circle_properties :
  ∀ (P : ℝ × ℝ),
  (∃ (Q : ℝ × ℝ), line_l Q.1 ∧ condition_P P Q point_F) →
  trajectory_C P.1 P.2 ∧
  (∀ (A B : ℝ × ℝ),
    (circle_O A.1 A.2 ∧ line_l A.1 ∧ trajectory_C A.1 A.2) →
    (circle_O B.1 B.2 ∧ line_l B.1 ∧ trajectory_C B.1 B.2) →
    A ≠ B →
    (let O := (0, 0); let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2);
     (A.1 * B.1 + A.2 * B.2 = 0) → range_AB AB)) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_circle_properties_l636_63626


namespace NUMINAMATH_CALUDE_alpha_beta_inequality_l636_63680

theorem alpha_beta_inequality (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) :
  -2 < α - β ∧ α - β < 0 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_inequality_l636_63680


namespace NUMINAMATH_CALUDE_trapezium_area_l636_63656

/-- The area of a trapezium with given dimensions -/
theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (a + b) * h / 2 = 285 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_area_l636_63656


namespace NUMINAMATH_CALUDE_age_solution_l636_63655

/-- The age equation as described in the problem -/
def age_equation (x : ℝ) : Prop :=
  3 * (x + 3) - 3 * (x - 3) = x

/-- Theorem stating that 18 is the solution to the age equation -/
theorem age_solution : ∃ x : ℝ, age_equation x ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_solution_l636_63655


namespace NUMINAMATH_CALUDE_expression_simplification_l636_63685

theorem expression_simplification (x y : ℚ) (hx : x = -4) (hy : y = -1/2) :
  x^2 - (x^2 - 2*x*y + 3*(x*y - 1/3*y^2)) = -7/4 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l636_63685


namespace NUMINAMATH_CALUDE_pizza_size_relation_l636_63609

theorem pizza_size_relation (r : ℝ) (h : r > 0) :
  let R := r * Real.sqrt (1 + 156 / 100)
  (R - r) / r * 100 = 60 := by sorry

end NUMINAMATH_CALUDE_pizza_size_relation_l636_63609


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l636_63613

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The equation of our specific circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

theorem circle_passes_through_points :
  ∃ (c : Circle),
    (∀ (x y : ℝ), circle_equation x y ↔ c.contains (x, y)) ∧
    c.contains (0, 0) ∧
    c.contains (4, 0) ∧
    c.contains (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l636_63613


namespace NUMINAMATH_CALUDE_buratino_betting_strategy_l636_63673

theorem buratino_betting_strategy :
  ∃ (x₁ x₂ x₃ y : ℕ+),
    x₁ + x₂ + x₃ + y = 20 ∧
    5 * x₁ + y ≥ 21 ∧
    4 * x₂ + y ≥ 21 ∧
    2 * x₃ + y ≥ 21 :=
by sorry

end NUMINAMATH_CALUDE_buratino_betting_strategy_l636_63673


namespace NUMINAMATH_CALUDE_correct_calculation_l636_63608

theorem correct_calculation (x : ℝ) : x * 7 = 126 → x / 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l636_63608


namespace NUMINAMATH_CALUDE_c_is_largest_l636_63669

-- Define the five numbers
def a : ℚ := 7.4683
def b : ℚ := 7 + 468/1000 + 3/9990  -- 7.468̅3
def c : ℚ := 7 + 46/100 + 83/9900   -- 7.46̅83
def d : ℚ := 7 + 4/10 + 683/999     -- 7.4̅683
def e : ℚ := 7 + 4683/9999          -- 7.̅4683

-- Theorem stating that c is the largest
theorem c_is_largest : c > a ∧ c > b ∧ c > d ∧ c > e := by sorry

end NUMINAMATH_CALUDE_c_is_largest_l636_63669


namespace NUMINAMATH_CALUDE_sequence_sum_l636_63688

theorem sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n : ℕ, S n = n^3) → 
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) →
  a 1 = S 1 →
  a 5 + a 6 = 152 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l636_63688


namespace NUMINAMATH_CALUDE_f_range_l636_63686

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3*x/2), Real.sin (3*x/2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))

noncomputable def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 -
  Real.sqrt (((a x).1 - (b x).1)^2 + ((a x).2 - (b x).2)^2)

theorem f_range :
  ∀ y ∈ Set.Icc (-3 : ℝ) (-1/2),
    ∃ x ∈ Set.Ico (π/6 : ℝ) (2*π/3),
      f x = y :=
sorry

end NUMINAMATH_CALUDE_f_range_l636_63686


namespace NUMINAMATH_CALUDE_mode_of_visual_acuity_l636_63604

-- Define the visual acuity values and their frequencies
def visual_acuity : List ℝ := [4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
def frequencies : List ℕ := [2, 3, 6, 9, 12, 8, 5, 3]

-- Define a function to find the mode
def mode (values : List ℝ) (freqs : List ℕ) : ℝ :=
  let pairs := List.zip values freqs
  let maxFreq := List.foldl (fun acc (_, f) => max acc f) 0 pairs
  let modes := List.filter (fun (_, f) => f == maxFreq) pairs
  (List.head! modes).1

-- Theorem: The mode of visual acuity is 4.7
theorem mode_of_visual_acuity :
  mode visual_acuity frequencies = 4.7 :=
by sorry

end NUMINAMATH_CALUDE_mode_of_visual_acuity_l636_63604


namespace NUMINAMATH_CALUDE_fraction_of_25_smaller_than_40_percent_of_60_by_4_l636_63648

theorem fraction_of_25_smaller_than_40_percent_of_60_by_4 : 
  (25 * (40 / 100 * 60 - 4)) / 25 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_25_smaller_than_40_percent_of_60_by_4_l636_63648


namespace NUMINAMATH_CALUDE_even_number_divisor_sum_l636_63657

theorem even_number_divisor_sum (n : ℕ) : 
  Even n →
  (∃ (divs : Finset ℕ), divs = {d : ℕ | d ∣ n} ∧ 
    (divs.sum (λ d => (1 : ℚ) / d) = 1620 / 1003)) →
  ∃ k : ℕ, n = 2006 * k :=
by sorry

end NUMINAMATH_CALUDE_even_number_divisor_sum_l636_63657


namespace NUMINAMATH_CALUDE_large_triangle_perimeter_l636_63684

/-- An isosceles triangle with two sides of length 12 and one side of length 14 -/
structure SmallTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : side1 = side2 ∧ side1 = 12 ∧ side3 = 14

/-- A triangle similar to the small triangle with longest side 42 -/
structure LargeTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  similar_to_small : ∃ (k : ℝ), side1 = k * 12 ∧ side2 = k * 12 ∧ side3 = k * 14
  longest_side : side3 = 42

/-- The perimeter of the large triangle is 114 -/
theorem large_triangle_perimeter (small : SmallTriangle) (large : LargeTriangle) :
  large.side1 + large.side2 + large.side3 = 114 := by
  sorry

end NUMINAMATH_CALUDE_large_triangle_perimeter_l636_63684


namespace NUMINAMATH_CALUDE_choose_one_book_result_l636_63607

/-- The number of ways to choose one book from a collection of Chinese, English, and Math books -/
def choose_one_book (chinese : ℕ) (english : ℕ) (math : ℕ) : ℕ :=
  chinese + english + math

/-- Theorem: Choosing one book from 10 Chinese, 7 English, and 5 Math books has 22 possibilities -/
theorem choose_one_book_result : choose_one_book 10 7 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_choose_one_book_result_l636_63607


namespace NUMINAMATH_CALUDE_range_of_a_l636_63620

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else (a-3)*x + 4*a

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  (0 < a ∧ a ≤ 3/4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l636_63620


namespace NUMINAMATH_CALUDE_students_on_field_trip_l636_63679

def total_budget : ℕ := 350
def bus_rental_cost : ℕ := 100
def admission_cost_per_student : ℕ := 10

theorem students_on_field_trip : 
  (total_budget - bus_rental_cost) / admission_cost_per_student = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_on_field_trip_l636_63679


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l636_63696

/-- The eccentricity of a hyperbola with equation x²/4 - y²/2 = 1 is √6/2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 6 / 2 ∧
  ∀ x y : ℝ, x^2 / 4 - y^2 / 2 = 1 → 
  e = Real.sqrt ((x / 2)^2 + (y / Real.sqrt 2)^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l636_63696


namespace NUMINAMATH_CALUDE_a_is_best_l636_63638

-- Define the structure for an athlete
structure Athlete where
  name : String
  average : ℝ
  variance : ℝ

-- Define the athletes
def athleteA : Athlete := ⟨"A", 185, 3.6⟩
def athleteB : Athlete := ⟨"B", 180, 3.6⟩
def athleteC : Athlete := ⟨"C", 185, 7.4⟩
def athleteD : Athlete := ⟨"D", 180, 8.1⟩

-- Define a function to compare athletes
def isBetterAthlete (a1 a2 : Athlete) : Prop :=
  (a1.average > a2.average) ∨ (a1.average = a2.average ∧ a1.variance < a2.variance)

-- Theorem stating that A is the best athlete
theorem a_is_best : 
  isBetterAthlete athleteA athleteB ∧ 
  isBetterAthlete athleteA athleteC ∧ 
  isBetterAthlete athleteA athleteD :=
sorry

end NUMINAMATH_CALUDE_a_is_best_l636_63638


namespace NUMINAMATH_CALUDE_intersection_of_sets_l636_63647

theorem intersection_of_sets : 
  let A : Set ℕ := {x | ∃ n, x = 2 * n}
  let B : Set ℕ := {x | ∃ n, x = 3 * n}
  let C : Set ℕ := {x | ∃ n, x = n * n}
  A ∩ B ∩ C = {x | ∃ n, x = 36 * n * n} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l636_63647


namespace NUMINAMATH_CALUDE_unknown_number_proof_l636_63646

theorem unknown_number_proof : (12^1 * 6^4) / 432 = 36 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l636_63646


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l636_63645

/-- Given two quadratic equations with real roots and specific conditions, prove that m = 4 --/
theorem quadratic_roots_relation (m n : ℝ) (x₁ x₂ y₁ y₂ : ℝ) : 
  n < 0 →
  x₁^2 + m^2*x₁ + n = 0 →
  x₂^2 + m^2*x₂ + n = 0 →
  y₁^2 + 5*m*y₁ + 7 = 0 →
  y₂^2 + 5*m*y₂ + 7 = 0 →
  x₁ - y₁ = 2 →
  x₂ - y₂ = 2 →
  m = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l636_63645


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l636_63692

theorem circular_seating_arrangement (n : ℕ) (π : Fin (2*n) → Fin (2*n)) 
  (hπ : Function.Bijective π) : 
  ∃ (i j : Fin (2*n)), i ≠ j ∧ (π i - π j) % (2*n) = (i - j) % (2*n) := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l636_63692


namespace NUMINAMATH_CALUDE_integral_sin4_cos2_l636_63698

theorem integral_sin4_cos2 (x : Real) :
  let f := fun (x : Real) => (1/16) * x - (1/64) * Real.sin (4*x) - (1/48) * Real.sin (2*x)^3
  (deriv f) x = Real.sin x^4 * Real.cos x^2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin4_cos2_l636_63698


namespace NUMINAMATH_CALUDE_multiplication_proof_l636_63610

theorem multiplication_proof :
  ∀ (a b c : ℕ),
  a = 60 + b →
  c = 14 →
  a * c = 882 ∧
  68 * 14 = 952 :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_proof_l636_63610


namespace NUMINAMATH_CALUDE_inequality_proof_l636_63616

theorem inequality_proof (x : ℝ) : 2 ≤ (3 * x^2 - 6 * x + 6) / (x^2 - x + 1) ∧ (3 * x^2 - 6 * x + 6) / (x^2 - x + 1) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l636_63616


namespace NUMINAMATH_CALUDE_exponent_multiplication_l636_63640

theorem exponent_multiplication (a : ℝ) : a^4 * a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l636_63640


namespace NUMINAMATH_CALUDE_equation_solution_l636_63660

theorem equation_solution (x : ℝ) (some_number : ℝ) 
  (h1 : x + 1 = some_number) (h2 : x = 4) : some_number = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l636_63660


namespace NUMINAMATH_CALUDE_orchid_rose_difference_l636_63630

theorem orchid_rose_difference (initial_roses initial_orchids final_roses final_orchids : ℕ) :
  initial_roses = 7 →
  initial_orchids = 12 →
  final_roses = 11 →
  final_orchids = 20 →
  final_orchids - final_roses = 9 := by
sorry

end NUMINAMATH_CALUDE_orchid_rose_difference_l636_63630


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l636_63663

theorem sufficient_but_not_necessary_condition :
  ∃ (a b : ℝ), (a > 1 ∧ b > 1 → a * b > 1) ∧
  ¬(a * b > 1 → a > 1 ∧ b > 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l636_63663


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l636_63642

/-- Given a rectangle with length to width ratio of 5:4 and diagonal d, 
    its area A can be expressed as A = (20/41)d^2 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : 
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l / w = 5 / 4 ∧ l^2 + w^2 = d^2 ∧ l * w = (20/41) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l636_63642


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l636_63601

/-- A right triangle with side lengths 9, 12, and 15 has an inradius of 3 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →  -- Given side lengths
  a^2 + b^2 = c^2 →          -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l636_63601


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l636_63600

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The number of people sitting around the circular table. -/
def numPeople : ℕ := 10

/-- The probability of getting the desired outcome (no two adjacent people standing). -/
def probability : ℚ := validArrangements numPeople / 2^numPeople

/-- Theorem stating that the probability of no two adjacent people standing
    in a circular arrangement of 10 people, each flipping a fair coin, is 123/1024. -/
theorem no_adjacent_standing_probability :
  probability = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l636_63600


namespace NUMINAMATH_CALUDE_complex_root_magnitude_l636_63605

theorem complex_root_magnitude (z : ℂ) : z^2 + 2*z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l636_63605


namespace NUMINAMATH_CALUDE_inequality_proof_l636_63627

theorem inequality_proof (m n : ℝ) (h : m > n) : 1 - 2*m < 1 - 2*n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l636_63627


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l636_63654

theorem greatest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 13*n + 36 ≤ 0 → n ≤ 9 ∧
  ∃ m : ℤ, m^2 - 13*m + 36 ≤ 0 ∧ m = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l636_63654


namespace NUMINAMATH_CALUDE_ellipse_satisfies_conditions_l636_63632

/-- An ellipse with foci on the y-axis, focal distance 4, and passing through (3,2) -/
def ellipse_equation (x y : ℝ) : Prop :=
  y^2 / 16 + x^2 / 12 = 1

/-- The focal distance of the ellipse -/
def focal_distance : ℝ := 4

/-- A point on the ellipse -/
def point_on_ellipse : ℝ × ℝ := (3, 2)

/-- Theorem stating that the ellipse equation satisfies the given conditions -/
theorem ellipse_satisfies_conditions :
  (∀ x y, ellipse_equation x y → (x = point_on_ellipse.1 ∧ y = point_on_ellipse.2)) ∧
  (∃ f₁ f₂ : ℝ, f₁ = -f₂ ∧ f₁^2 = (focal_distance/2)^2 ∧
    ∀ x y, ellipse_equation x y →
      (x^2 + (y - f₁)^2)^(1/2) + (x^2 + (y - f₂)^2)^(1/2) = 2 * (16^(1/2))) :=
sorry

end NUMINAMATH_CALUDE_ellipse_satisfies_conditions_l636_63632


namespace NUMINAMATH_CALUDE_oreilly_triple_8_49_l636_63644

/-- Definition of an O'Reilly triple -/
def is_oreilly_triple (a b x : ℕ+) : Prop :=
  (a.val : ℝ)^(1/3) + (b.val : ℝ)^(1/2) = x.val

/-- Theorem: If (8,49,x) is an O'Reilly triple, then x = 9 -/
theorem oreilly_triple_8_49 (x : ℕ+) :
  is_oreilly_triple 8 49 x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_oreilly_triple_8_49_l636_63644


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l636_63625

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l636_63625


namespace NUMINAMATH_CALUDE_greatest_savings_l636_63691

def plane_cost : ℚ := 600
def boat_cost : ℚ := 254
def helicopter_cost : ℚ := 850

def savings (cost1 cost2 : ℚ) : ℚ := max cost1 cost2 - min cost1 cost2

theorem greatest_savings :
  max (savings plane_cost boat_cost) (savings helicopter_cost boat_cost) = 596 :=
by sorry

end NUMINAMATH_CALUDE_greatest_savings_l636_63691


namespace NUMINAMATH_CALUDE_chess_tournament_games_l636_63602

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  (n.choose 2) = 45 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l636_63602


namespace NUMINAMATH_CALUDE_probability_one_black_ball_l636_63658

/-- The probability of drawing exactly one black ball when drawing two balls without replacement from a box containing 3 white balls and 2 black balls -/
theorem probability_one_black_ball (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : total_balls = 5)
  (h3 : white_balls = 3)
  (h4 : black_balls = 2) : 
  (white_balls * black_balls) / ((total_balls * (total_balls - 1)) / 2) = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_one_black_ball_l636_63658


namespace NUMINAMATH_CALUDE_complement_B_A_when_m_2_range_of_m_for_necessary_not_sufficient_l636_63623

def A : Set ℝ := {x | 4 < x ∧ x ≤ 8}
def B (m : ℝ) : Set ℝ := {x | 5 - m^2 ≤ x ∧ x ≤ 5 + m^2}

theorem complement_B_A_when_m_2 :
  {x : ℝ | 1 ≤ x ∧ x ≤ 4 ∨ 8 < x ∧ x ≤ 9} = (B 2) \ A := by sorry

theorem range_of_m_for_necessary_not_sufficient :
  {m : ℝ | A ⊆ B m ∧ A ≠ B m} = {m : ℝ | -1 < m ∧ m < 1} := by sorry

end NUMINAMATH_CALUDE_complement_B_A_when_m_2_range_of_m_for_necessary_not_sufficient_l636_63623


namespace NUMINAMATH_CALUDE_roots_are_irrational_l636_63678

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - 4*k*x + 3*k^2 - 2

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (4*k)^2 - 4*(3*k^2 - 2)

-- Theorem statement
theorem roots_are_irrational (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧ x * y = 10) →
  ∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧ ¬(∃ q : ℚ, x = ↑q) ∧ ¬(∃ q : ℚ, y = ↑q) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l636_63678


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l636_63665

theorem diophantine_equation_solutions :
  ∀ (a b : ℕ), 2017^a = b^6 - 32*b + 1 ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l636_63665


namespace NUMINAMATH_CALUDE_segment_length_in_dihedral_angle_l636_63667

/-- Given a segment AB with ends on the faces of a dihedral angle φ, where the distances from A and B
    to the edge of the angle are a and b respectively, and the distance between the projections of A
    and B on the edge is c, the length of AB is equal to √(a² + b² + c² - 2ab cos φ). -/
theorem segment_length_in_dihedral_angle (φ a b c : ℝ) (h_φ : 0 < φ ∧ φ < π) 
    (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  ∃ (AB : ℝ), AB = Real.sqrt (a^2 + b^2 + c^2 - 2 * a * b * Real.cos φ) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_in_dihedral_angle_l636_63667


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l636_63676

def U : Set ℕ := { x | (x - 1) / (5 - x) > 0 ∧ x > 0 }

def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : Set.compl A ∩ U = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l636_63676


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l636_63635

/-- Proves that given a cycle sold for Rs. 1080 with a 60% gain, the original price of the cycle was Rs. 675. -/
theorem cycle_price_calculation (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 1080)
  (h2 : gain_percent = 60) :
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 675 :=
by sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l636_63635


namespace NUMINAMATH_CALUDE_biscuit_boxes_combination_exists_l636_63617

theorem biscuit_boxes_combination_exists : ∃ (a b c d e : ℕ), 16*a + 17*b + 23*c + 39*d + 40*e = 100 := by
  sorry

end NUMINAMATH_CALUDE_biscuit_boxes_combination_exists_l636_63617


namespace NUMINAMATH_CALUDE_nail_pierces_one_shape_l636_63672

/-- Represents a shape that can be placed on a rectangular surface --/
structure Shape where
  area : ℝ
  -- Other properties of the shape could be added here

/-- Represents a rectangular box --/
structure Box where
  length : ℝ
  width : ℝ
  center : ℝ × ℝ

/-- Represents the placement of a shape on the box's bottom --/
structure Placement where
  shape : Shape
  position : ℝ × ℝ

/-- Checks if two placements completely cover the box's bottom --/
def covers (b : Box) (p1 p2 : Placement) : Prop := sorry

/-- Checks if a point is inside a placed shape --/
def pointInPlacement (point : ℝ × ℝ) (p : Placement) : Prop := sorry

/-- Main theorem: It's possible to arrange two identical shapes to cover a box's bottom
    such that the center point is in only one of the shapes --/
theorem nail_pierces_one_shape (b : Box) (s : Shape) :
  ∃ (p1 p2 : Placement),
    p1.shape = s ∧ p2.shape = s ∧
    covers b p1 p2 ∧
    (pointInPlacement b.center p1 ↔ ¬pointInPlacement b.center p2) := sorry

end NUMINAMATH_CALUDE_nail_pierces_one_shape_l636_63672


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l636_63675

/-- Three lines intersecting at a single point -/
structure ThreeIntersectingLines where
  k : ℚ
  intersect_point : ℝ × ℝ
  line1 : ∀ (x y : ℝ), x + k * y = 0 → (x, y) = intersect_point
  line2 : ∀ (x y : ℝ), 2 * x + 3 * y + 8 = 0 → (x, y) = intersect_point
  line3 : ∀ (x y : ℝ), x - y - 1 = 0 → (x, y) = intersect_point

/-- If three lines intersect at a single point, then k = -1/2 -/
theorem intersecting_lines_k_value (lines : ThreeIntersectingLines) : lines.k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l636_63675


namespace NUMINAMATH_CALUDE_secret_spread_day_l636_63637

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when 3280 students know the secret -/
theorem secret_spread_day :
  ∃ n : ℕ, secret_spread n = 3280 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_secret_spread_day_l636_63637


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l636_63650

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, -1/2 < x ∧ x < 2 ↔ f a b c x > 0) : 
  a < 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l636_63650


namespace NUMINAMATH_CALUDE_fifth_student_stickers_l636_63628

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fifth_student_stickers :
  let a₁ := 29  -- first term
  let d := 6    -- common difference
  let n := 5    -- position of the term we're looking for
  arithmetic_sequence a₁ d n = 53 := by sorry

end NUMINAMATH_CALUDE_fifth_student_stickers_l636_63628


namespace NUMINAMATH_CALUDE_mail_cost_theorem_l636_63615

def cost_per_package : ℕ := 5
def number_of_parents : ℕ := 2
def number_of_brothers : ℕ := 3
def children_per_brother : ℕ := 2

def total_relatives : ℕ := 
  number_of_parents + number_of_brothers + 
  number_of_brothers * (1 + 1 + children_per_brother)

def total_cost : ℕ := total_relatives * cost_per_package

theorem mail_cost_theorem : total_cost = 70 := by
  sorry

end NUMINAMATH_CALUDE_mail_cost_theorem_l636_63615


namespace NUMINAMATH_CALUDE_hans_room_options_l636_63636

/-- Represents a hotel with floors and rooms -/
structure Hotel where
  total_floors : ℕ
  rooms_per_floor : ℕ
  available_rooms_on_odd_floor : ℕ

/-- Calculates the number of available rooms in the hotel -/
def available_rooms (h : Hotel) : ℕ :=
  (h.total_floors / 2) * h.available_rooms_on_odd_floor

/-- The specific hotel in the problem -/
def problem_hotel : Hotel :=
  { total_floors := 20
    rooms_per_floor := 15
    available_rooms_on_odd_floor := 10 }

/-- Theorem stating that the number of available rooms in the problem hotel is 100 -/
theorem hans_room_options : available_rooms problem_hotel = 100 := by
  sorry

end NUMINAMATH_CALUDE_hans_room_options_l636_63636


namespace NUMINAMATH_CALUDE_fill_time_three_pipes_l636_63668

-- Define the tank's volume
variable (T : ℝ)

-- Define the rates at which pipes X, Y, and Z fill the tank
variable (X Y Z : ℝ)

-- Define the conditions
def condition1 : Prop := X + Y = T / 3
def condition2 : Prop := X + Z = T / 4
def condition3 : Prop := Y + Z = T / 2

-- State the theorem
theorem fill_time_three_pipes 
  (h1 : condition1 T X Y) 
  (h2 : condition2 T X Z) 
  (h3 : condition3 T Y Z) :
  1 / (X + Y + Z) = 24 / 13 := by
  sorry

end NUMINAMATH_CALUDE_fill_time_three_pipes_l636_63668


namespace NUMINAMATH_CALUDE_imaginary_complex_implies_m_conditions_l636_63694

theorem imaginary_complex_implies_m_conditions (m : ℝ) : 
  (∃ (z : ℂ), z = Complex.mk (m^2 - 3*m - 4) (m^2 - 5*m - 6) ∧ z.re = 0 ∧ z.im ≠ 0) →
  (m ≠ -1 ∧ m ≠ 6) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_complex_implies_m_conditions_l636_63694


namespace NUMINAMATH_CALUDE_max_value_theorem_l636_63674

theorem max_value_theorem (a b : ℝ) : 
  a^2 = (1 + 2*b) * (1 - 2*b) →
  ∃ (x : ℝ), x = (2*a*b)/(|a| + 2*|b|) ∧ 
             ∀ (y : ℝ), y = (2*a*b)/(|a| + 2*|b|) → y ≤ x ∧
             x = Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l636_63674


namespace NUMINAMATH_CALUDE_inverse_square_theorem_l636_63670

/-- A function representing the inverse square relationship between x and y -/
noncomputable def inverse_square_relation (k : ℝ) (y : ℝ) : ℝ := k / y^2

/-- Theorem stating that given the inverse square relationship and a known point,
    we can determine the value of x for y = 3 -/
theorem inverse_square_theorem (k : ℝ) :
  (inverse_square_relation k 4 = 0.5625) →
  (inverse_square_relation k 3 = 1) :=
by
  sorry

#check inverse_square_theorem

end NUMINAMATH_CALUDE_inverse_square_theorem_l636_63670


namespace NUMINAMATH_CALUDE_truck_toll_calculation_l636_63633

/-- Calculates the toll for a truck given the number of axles -/
def toll (x : ℕ) : ℚ :=
  0.50 + 0.50 * (x - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    number of wheels on the front axle, and number of wheels on each other axle -/
def numAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_calculation :
  let x := numAxles 18 2 4
  toll x = 2 :=
by sorry

end NUMINAMATH_CALUDE_truck_toll_calculation_l636_63633


namespace NUMINAMATH_CALUDE_B_power_99_is_identity_l636_63699

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_99_is_identity :
  B ^ 99 = (1 : Matrix (Fin 3) (Fin 3) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_99_is_identity_l636_63699


namespace NUMINAMATH_CALUDE_train_distance_l636_63651

theorem train_distance (x : ℝ) :
  (x > 0) →
  (x / 40 + 2*x / 20 = (x + 2*x) / 48) →
  (x + 2*x = 6) :=
by sorry

end NUMINAMATH_CALUDE_train_distance_l636_63651


namespace NUMINAMATH_CALUDE_cost_price_calculation_l636_63695

/-- Proves that if an article is sold for Rs. 400 with a 60% profit, its cost price is Rs. 250. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 400 →
  profit_percentage = 60 →
  selling_price = (1 + profit_percentage / 100) * 250 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l636_63695


namespace NUMINAMATH_CALUDE_solution_set_for_a_4_range_of_a_for_all_x_geq_4_l636_63624

def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

theorem solution_set_for_a_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := by sorry

theorem range_of_a_for_all_x_geq_4 :
  (∀ x : ℝ, f a x ≥ 4) → (a ≤ -3 ∨ a ≥ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_4_range_of_a_for_all_x_geq_4_l636_63624


namespace NUMINAMATH_CALUDE_lines_parallel_lines_perpendicular_l636_63671

/-- Two lines in the plane --/
structure Lines where
  a : ℝ
  l1 : ℝ → ℝ → ℝ := λ x y => a * x + 2 * y + 6
  l2 : ℝ → ℝ → ℝ := λ x y => x + (a - 1) * y + a^2 - 1

/-- The lines are parallel iff a = -1 --/
theorem lines_parallel (lines : Lines) : 
  (∃ k : ℝ, ∀ x y : ℝ, lines.l1 x y = k * lines.l2 x y) ↔ lines.a = -1 :=
sorry

/-- The lines are perpendicular iff a = 2/3 --/
theorem lines_perpendicular (lines : Lines) :
  (∀ x1 y1 x2 y2 : ℝ, 
    (lines.l1 x1 y1 = 0 ∧ lines.l1 x2 y2 = 0) → 
    (lines.l2 x1 y1 = 0 ∧ lines.l2 x2 y2 = 0) → 
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) * 
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) = 
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))^2) 
  ↔ lines.a = 2/3 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_lines_perpendicular_l636_63671


namespace NUMINAMATH_CALUDE_marks_animals_legs_count_l636_63614

theorem marks_animals_legs_count :
  let kangaroo_count : ℕ := 23
  let goat_count : ℕ := 3 * kangaroo_count
  let kangaroo_legs : ℕ := 2
  let goat_legs : ℕ := 4
  kangaroo_count * kangaroo_legs + goat_count * goat_legs = 322 := by
  sorry

end NUMINAMATH_CALUDE_marks_animals_legs_count_l636_63614


namespace NUMINAMATH_CALUDE_multiples_of_10_and_12_within_100_l636_63631

theorem multiples_of_10_and_12_within_100 : 
  ∃! n : ℕ, n ≤ 100 ∧ 10 ∣ n ∧ 12 ∣ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_multiples_of_10_and_12_within_100_l636_63631


namespace NUMINAMATH_CALUDE_door_opening_proofs_l636_63634

/-- The number of buttons on the lock -/
def num_buttons : Nat := 10

/-- The number of buttons that need to be pressed simultaneously -/
def buttons_to_press : Nat := 3

/-- Time taken for each attempt in seconds -/
def time_per_attempt : Nat := 2

/-- The total number of possible combinations -/
def total_combinations : Nat := (num_buttons.choose buttons_to_press)

/-- The maximum time needed to try all combinations in seconds -/
def max_time : Nat := total_combinations * time_per_attempt

/-- The average number of attempts needed -/
def avg_attempts : Rat := (1 + total_combinations) / 2

/-- The average time needed in seconds -/
def avg_time : Rat := avg_attempts * time_per_attempt

/-- The maximum number of attempts possible in 60 seconds -/
def max_attempts_in_minute : Nat := 60 / time_per_attempt

theorem door_opening_proofs :
  (max_time = 240) ∧
  (avg_time = 121) ∧
  (max_attempts_in_minute = 30) ∧
  ((max_attempts_in_minute - 1 : Rat) / total_combinations = 29 / 120) := by
  sorry

end NUMINAMATH_CALUDE_door_opening_proofs_l636_63634


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l636_63682

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 160 * r = b ∧ b * r = 1) → 
  b = 4 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l636_63682


namespace NUMINAMATH_CALUDE_problem_statement_l636_63619

theorem problem_statement :
  (∀ a : ℝ, Real.exp a ≥ a + 1) ∧
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l636_63619


namespace NUMINAMATH_CALUDE_hcl_moles_combined_l636_63677

/-- The number of moles of HCl combined to produce a given amount of NH4Cl -/
theorem hcl_moles_combined 
  (nh3_moles : ℝ) 
  (nh4cl_grams : ℝ) 
  (nh4cl_molar_mass : ℝ) 
  (h1 : nh3_moles = 3)
  (h2 : nh4cl_grams = 159)
  (h3 : nh4cl_molar_mass = 53.50) :
  ∃ hcl_moles : ℝ, abs (hcl_moles - (nh4cl_grams / nh4cl_molar_mass)) < 0.001 :=
by
  sorry

#check hcl_moles_combined

end NUMINAMATH_CALUDE_hcl_moles_combined_l636_63677


namespace NUMINAMATH_CALUDE_herb_leaf_difference_l636_63690

theorem herb_leaf_difference : 
  ∀ (basil sage verbena : ℕ),
  basil = 2 * sage →
  basil + sage + verbena = 29 →
  basil = 12 →
  verbena - sage = 5 := by
sorry

end NUMINAMATH_CALUDE_herb_leaf_difference_l636_63690


namespace NUMINAMATH_CALUDE_extreme_value_cubic_l636_63643

/-- Given a cubic function f(x) = x^3 + ax^2 + bx with an extreme value of -2 at x = 1,
    prove that a + 2b = -6 -/
theorem extreme_value_cubic (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x
  (f 1 = -2) ∧ (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1 ∨ f x ≤ f 1) →
  a + 2*b = -6 := by
sorry

end NUMINAMATH_CALUDE_extreme_value_cubic_l636_63643


namespace NUMINAMATH_CALUDE_book_distribution_count_l636_63603

def distribute_books (total_books : ℕ) (min_per_location : ℕ) : ℕ :=
  (total_books - 2 * min_per_location + 1)

theorem book_distribution_count :
  distribute_books 8 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_book_distribution_count_l636_63603


namespace NUMINAMATH_CALUDE_first_player_can_draw_l636_63641

/-- Represents a chess position -/
def ChessPosition : Type := Unit

/-- Represents a chess move -/
def ChessMove : Type := Unit

/-- Represents a strategy in double chess -/
def DoubleChessStrategy : Type := ChessPosition → ChessMove × ChessMove

/-- The initial chess position -/
def initialPosition : ChessPosition := sorry

/-- Applies a move to a position, returning the new position -/
def applyMove (pos : ChessPosition) (move : ChessMove) : ChessPosition := sorry

/-- Applies two consecutive moves to a position, returning the new position -/
def applyDoubleMoves (pos : ChessPosition) (moves : ChessMove × ChessMove) : ChessPosition := sorry

/-- Determines if a position is a win for the current player -/
def isWinningPosition (pos : ChessPosition) : Prop := sorry

/-- A knight move that doesn't change the position -/
def neutralKnightMove : ChessMove := sorry

/-- Theorem: The first player in double chess can always force at least a draw -/
theorem first_player_can_draw :
  ∀ (secondPlayerStrategy : DoubleChessStrategy),
  ∃ (firstPlayerStrategy : DoubleChessStrategy),
  ¬(isWinningPosition (applyDoubleMoves (applyDoubleMoves initialPosition (neutralKnightMove, neutralKnightMove)) (secondPlayerStrategy (applyDoubleMoves initialPosition (neutralKnightMove, neutralKnightMove))))) :=
sorry

end NUMINAMATH_CALUDE_first_player_can_draw_l636_63641


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l636_63652

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_first_term 
  (a : ℕ → ℚ) 
  (h_geometric : is_geometric_sequence a) 
  (h_third_term : a 2 = 8)
  (h_fifth_term : a 4 = 27 / 4) :
  a 0 = 256 / 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l636_63652


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l636_63662

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 75 / 16 ∧ B = 21 / 16 ∧
  ∀ (x : ℚ), x ≠ 12 → x ≠ -4 →
    (6 * x + 3) / (x^2 - 8*x - 48) = A / (x - 12) + B / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l636_63662


namespace NUMINAMATH_CALUDE_max_profit_price_l636_63681

/-- The cost of one item in yuan -/
def cost : ℝ := 30

/-- The number of items sold as a function of price -/
def itemsSold (x : ℝ) : ℝ := 200 - x

/-- The profit function -/
def profit (x : ℝ) : ℝ := (x - cost) * (itemsSold x)

/-- Theorem: The price that maximizes profit is 115 yuan -/
theorem max_profit_price : 
  ∃ (x : ℝ), x = 115 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_price_l636_63681
