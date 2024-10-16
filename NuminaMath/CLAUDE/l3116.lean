import Mathlib

namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3116_311665

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3116_311665


namespace NUMINAMATH_CALUDE_cube_geometry_l3116_311627

-- Define a cube
def Cube : Type := Unit

-- Define a vertex of a cube
def Vertex (c : Cube) : Type := Unit

-- Define a set of 4 vertices
def FourVertices (c : Cube) : Type := Fin 4 → Vertex c

-- Define a spatial quadrilateral
def SpatialQuadrilateral (c : Cube) (v : FourVertices c) : Prop := sorry

-- Define a tetrahedron
def Tetrahedron (c : Cube) (v : FourVertices c) : Prop := sorry

-- Define an equilateral triangle
def EquilateralTriangle (c : Cube) (v1 v2 v3 : Vertex c) : Prop := sorry

-- Define an isosceles right-angled triangle
def IsoscelesRightTriangle (c : Cube) (v1 v2 v3 : Vertex c) : Prop := sorry

-- Theorem statement
theorem cube_geometry (c : Cube) : 
  (∃ v : FourVertices c, SpatialQuadrilateral c v) ∧ 
  (∃ v : FourVertices c, Tetrahedron c v ∧ 
    (∀ face : Fin 4 → Fin 3, EquilateralTriangle c (v (face 0)) (v (face 1)) (v (face 2)))) ∧
  (∃ v : FourVertices c, Tetrahedron c v ∧ 
    (∃ face : Fin 4 → Fin 3, EquilateralTriangle c (v (face 0)) (v (face 1)) (v (face 2))) ∧
    (∃ faces : Fin 3 → (Fin 4 → Fin 3), 
      ∀ i : Fin 3, IsoscelesRightTriangle c (v ((faces i) 0)) (v ((faces i) 1)) (v ((faces i) 2)))) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_geometry_l3116_311627


namespace NUMINAMATH_CALUDE_scale_division_l3116_311637

/-- Given a scale of length 188 inches divided into 8 equal parts, 
    the length of each part is 23.5 inches. -/
theorem scale_division (total_length : ℝ) (num_parts : ℕ) 
  (h1 : total_length = 188) 
  (h2 : num_parts = 8) :
  total_length / num_parts = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l3116_311637


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3116_311605

/-- 
A quadratic equation x^2 - 2x + 2a = 0 has two equal real roots if and only if a = 1/2.
-/
theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + 2*a = 0 ∧ (∀ y : ℝ, y^2 - 2*y + 2*a = 0 → y = x)) ↔ a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3116_311605


namespace NUMINAMATH_CALUDE_abby_peeled_22_l3116_311693

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  abby_rate : ℕ
  homer_solo_time : ℕ

/-- Calculates the number of potatoes Abby peeled -/
def abby_peeled (scenario : PotatoPeeling) : ℕ :=
  let homer_solo := scenario.homer_rate * scenario.homer_solo_time
  let remaining := scenario.total_potatoes - homer_solo
  let combined_rate := scenario.homer_rate + scenario.abby_rate
  let combined_time := remaining / combined_rate
  scenario.abby_rate * combined_time

/-- The main theorem stating that Abby peeled 22 potatoes -/
theorem abby_peeled_22 (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 60)
  (h2 : scenario.homer_rate = 4)
  (h3 : scenario.abby_rate = 6)
  (h4 : scenario.homer_solo_time = 6) :
  abby_peeled scenario = 22 := by
  sorry

end NUMINAMATH_CALUDE_abby_peeled_22_l3116_311693


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l3116_311635

-- Define the conditions
variable (a : ℝ) (ha : a > 0) (hna : a ≠ 1)
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf1 : f (Real.sqrt 2) = 2)
variable (hf2 : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α)

-- State the theorem
theorem fixed_point_power_function : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l3116_311635


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3116_311607

def f (x : ℝ) : ℝ := x^2 - 6*x + 13

theorem quadratic_minimum :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3116_311607


namespace NUMINAMATH_CALUDE_mono_increasing_range_l3116_311614

/-- A function f is monotonically increasing on ℝ -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem mono_increasing_range (f : ℝ → ℝ) (h : MonoIncreasing f) :
  ∀ m : ℝ, f (2 * m - 3) > f (-m) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_mono_increasing_range_l3116_311614


namespace NUMINAMATH_CALUDE_range_of_M_M_lower_bound_l3116_311626

theorem range_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
  let M := (1/a - 1) * (1/b - 1) * (1/c - 1)
  ∀ x : ℝ, x ≥ 8 → ∃ a' b' c' : ℝ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' + b' + c' = 1 ∧
    (1/a' - 1) * (1/b' - 1) * (1/c' - 1) = x :=
by sorry

theorem M_lower_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_M_M_lower_bound_l3116_311626


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l3116_311610

theorem perfect_square_divisibility (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ (n : ℕ), x = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l3116_311610


namespace NUMINAMATH_CALUDE_fraction_equality_l3116_311679

theorem fraction_equality (A B : ℤ) : 
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ -5 ∧ x ≠ 2 → 
    (A / (x + 2) + B / (x^2 - 4*x - 5) = (x^2 + x + 7) / (x^3 + 6*x^2 - 13*x - 10))) → 
  B / A = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3116_311679


namespace NUMINAMATH_CALUDE_buses_in_parking_lot_l3116_311664

theorem buses_in_parking_lot (initial_buses additional_buses : ℕ) : 
  initial_buses = 7 → additional_buses = 6 → initial_buses + additional_buses = 13 :=
by sorry

end NUMINAMATH_CALUDE_buses_in_parking_lot_l3116_311664


namespace NUMINAMATH_CALUDE_no_rational_solution_sqrt2_equation_l3116_311606

theorem no_rational_solution_sqrt2_equation :
  ∀ (x y z t : ℚ), (x + y * Real.sqrt 2)^2 + (z + t * Real.sqrt 2)^2 ≠ 5 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_sqrt2_equation_l3116_311606


namespace NUMINAMATH_CALUDE_work_completion_time_l3116_311632

/-- Proves that given two people who can complete a task in 4 days together, 
    and one of them can complete it in 12 days alone, 
    the other person can complete the task in 24 days alone. -/
theorem work_completion_time 
  (joint_time : ℝ) 
  (person1_time : ℝ) 
  (h1 : joint_time = 4) 
  (h2 : person1_time = 12) : 
  ∃ person2_time : ℝ, 
    person2_time = 24 ∧ 
    1 / joint_time = 1 / person1_time + 1 / person2_time :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3116_311632


namespace NUMINAMATH_CALUDE_intersection_length_l3116_311602

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 5
def circle2 (x y m : ℝ) : Prop := (x - m)^2 + y^2 = 20

-- Define the intersection points
def intersectionPoints (m : ℝ) : Prop := ∃ (A B : ℝ × ℝ), 
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 A.1 A.2 m ∧ circle2 B.1 B.2 m

-- Define the perpendicular tangents condition
def perpendicularTangents (m : ℝ) (A : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle2 A.1 A.2 m ∧
  (A.1 * m = 5) -- This condition represents perpendicular tangents

-- Theorem statement
theorem intersection_length (m : ℝ) :
  intersectionPoints m →
  (∃ (A : ℝ × ℝ), perpendicularTangents m A) →
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ 
                     circle2 A.1 A.2 m ∧ circle2 B.1 B.2 m ∧
                     ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16) :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_l3116_311602


namespace NUMINAMATH_CALUDE_professional_doctors_percentage_l3116_311636

theorem professional_doctors_percentage 
  (leaders_percent : ℝ)
  (nurses_percent : ℝ)
  (h1 : leaders_percent = 4)
  (h2 : nurses_percent = 56)
  (h3 : ∃ (doctors_percent psychologists_percent : ℝ), 
    leaders_percent + nurses_percent + doctors_percent + psychologists_percent = 100) :
  ∃ (doctors_percent : ℝ), doctors_percent = 40 := by
  sorry

end NUMINAMATH_CALUDE_professional_doctors_percentage_l3116_311636


namespace NUMINAMATH_CALUDE_intersection_line_parabola_l3116_311678

/-- The line y = kx - 2 intersects the parabola y² = 8x at two points A and B,
    and the x-coordinate of the midpoint of AB is 2. Then k = 2. -/
theorem intersection_line_parabola (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    (∀ x y, (x, y) = A ∨ (x, y) = B → y = k * x - 2 ∧ y^2 = 8 * x) ∧
    (A.1 + B.1) / 2 = 2) →
  k = 2 := by
sorry


end NUMINAMATH_CALUDE_intersection_line_parabola_l3116_311678


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l3116_311684

theorem cubic_root_equation_solution :
  ∃ x : ℝ, x = 1674 / 15 ∧ (30 * x + (30 * x + 27) ^ (1/3)) ^ (1/3) = 15 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l3116_311684


namespace NUMINAMATH_CALUDE_vacation_tents_l3116_311641

/-- Given a total number of people, house capacity, and tent capacity, 
    calculate the minimum number of tents needed. -/
def tents_needed (total_people : ℕ) (house_capacity : ℕ) (tent_capacity : ℕ) : ℕ :=
  ((total_people - house_capacity + tent_capacity - 1) / tent_capacity)

/-- Theorem stating that for 13 people, a house capacity of 4, and tents that sleep 2 each, 
    the minimum number of tents needed is 5. -/
theorem vacation_tents : tents_needed 13 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vacation_tents_l3116_311641


namespace NUMINAMATH_CALUDE_iteration_convergence_l3116_311625

theorem iteration_convergence (a b : ℝ) (h : a > b) :
  ∃ k : ℕ, (2 : ℝ)^(-k : ℤ) * (a - b) < 1 / 2002 := by
  sorry

end NUMINAMATH_CALUDE_iteration_convergence_l3116_311625


namespace NUMINAMATH_CALUDE_range_of_a_l3116_311618

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ |x - 5| + |x - 3|) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3116_311618


namespace NUMINAMATH_CALUDE_card_distribution_l3116_311639

theorem card_distribution (total_cards : ℕ) (total_people : ℕ) 
  (h1 : total_cards = 100) (h2 : total_people = 15) :
  let cards_per_person := total_cards / total_people
  let remainder := total_cards % total_people
  let people_with_extra := remainder
  let people_with_fewer := total_people - people_with_extra
  cards_per_person = 6 ∧ people_with_fewer = 5 := by
  sorry

#check card_distribution

end NUMINAMATH_CALUDE_card_distribution_l3116_311639


namespace NUMINAMATH_CALUDE_divisibility_implies_one_or_seven_l3116_311662

theorem divisibility_implies_one_or_seven (a n : ℤ) 
  (ha : a ≥ 1) 
  (h1 : a ∣ n + 2) 
  (h2 : a ∣ n^2 + n + 5) : 
  a = 1 ∨ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_one_or_seven_l3116_311662


namespace NUMINAMATH_CALUDE_problem_statement_l3116_311629

theorem problem_statement (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (a^2)^(4*b) = a^(2*b) * x^(3*b) → x = a^2 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3116_311629


namespace NUMINAMATH_CALUDE_concert_attendance_l3116_311692

theorem concert_attendance (adult_price child_price total_collected : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 3)
  (h3 : total_collected = 6000)
  (h4 : ∃ (a c : ℕ), c = 3 * a ∧ adult_price * a + child_price * c = total_collected) :
  ∃ (total : ℕ), total = 1500 ∧ 
    ∃ (a c : ℕ), c = 3 * a ∧ adult_price * a + child_price * c = total_collected ∧ 
    total = a + c := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l3116_311692


namespace NUMINAMATH_CALUDE_direct_proportion_m_value_l3116_311628

-- Define the function y as a direct proportion function
def is_direct_proportion (m : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → (m - 2) * x^(m^2 - 3) = k * x

-- Theorem statement
theorem direct_proportion_m_value :
  (∃ m : ℝ, is_direct_proportion m) → (∃ m : ℝ, is_direct_proportion m ∧ m = -2) :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_m_value_l3116_311628


namespace NUMINAMATH_CALUDE_expression_evaluation_l3116_311680

/-- Given a = 4 and b = -3, prove that 2a^2 - 3b^2 + 4ab = -43 -/
theorem expression_evaluation (a b : ℤ) (ha : a = 4) (hb : b = -3) :
  2 * a^2 - 3 * b^2 + 4 * a * b = -43 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3116_311680


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3116_311619

theorem min_value_of_expression (n : ℕ) (h : 10 ≤ n ∧ n ≤ 99) : 
  3 * (300 - n) ≥ 603 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3116_311619


namespace NUMINAMATH_CALUDE_tan_105_degrees_l3116_311697

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l3116_311697


namespace NUMINAMATH_CALUDE_tamara_brownie_pans_l3116_311667

def total_revenue : ℕ := 32
def brownie_price : ℕ := 2
def pieces_per_pan : ℕ := 8

theorem tamara_brownie_pans : 
  total_revenue / (brownie_price * pieces_per_pan) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tamara_brownie_pans_l3116_311667


namespace NUMINAMATH_CALUDE_function_property_l3116_311656

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_property (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3116_311656


namespace NUMINAMATH_CALUDE_square_of_1031_l3116_311649

theorem square_of_1031 : (1031 : ℕ)^2 = 1060961 := by sorry

end NUMINAMATH_CALUDE_square_of_1031_l3116_311649


namespace NUMINAMATH_CALUDE_problem_solution_l3116_311685

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -2 ∨ |x - 30| ≤ 2)
  (h2 : a < b) : 
  a + 2*b + 3*c = 86 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3116_311685


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3116_311659

theorem inscribed_squares_ratio (a b : ℝ) : 
  (5 : ℝ) ^ 2 + 12 ^ 2 = 13 ^ 2 →  -- Pythagorean theorem for the triangle
  a * (5 + 12 - a) = 5 * 12 →  -- Condition for the first square
  b * (5 - b) = (13 - b) * (12 - b) →  -- Condition for the second square
  0 < a ∧ 0 < b →  -- Positive side lengths
  a / b = 25 / 370 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3116_311659


namespace NUMINAMATH_CALUDE_lollipop_calories_l3116_311646

/-- Calculates the calories in a giant lollipop based on given candy information --/
theorem lollipop_calories 
  (chocolate_bars : ℕ) 
  (sugar_per_bar : ℕ) 
  (total_sugar : ℕ) 
  (calories_per_gram : ℕ) 
  (h1 : chocolate_bars = 14)
  (h2 : sugar_per_bar = 10)
  (h3 : total_sugar = 177)
  (h4 : calories_per_gram = 4) :
  (total_sugar - chocolate_bars * sugar_per_bar) * calories_per_gram = 148 := by
  sorry

#check lollipop_calories

end NUMINAMATH_CALUDE_lollipop_calories_l3116_311646


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l3116_311645

theorem two_numbers_with_given_means (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0) :
  Real.sqrt (x * y) = Real.sqrt 5 ∧ (x + y) / 2 = 5 →
  (x = 5 + 2 * Real.sqrt 5 ∧ y = 5 - 2 * Real.sqrt 5) ∨
  (x = 5 - 2 * Real.sqrt 5 ∧ y = 5 + 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l3116_311645


namespace NUMINAMATH_CALUDE_tan_105_degrees_l3116_311642

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l3116_311642


namespace NUMINAMATH_CALUDE_smallest_prime_angle_in_inscribed_triangle_l3116_311653

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The theorem statement -/
theorem smallest_prime_angle_in_inscribed_triangle :
  ∀ q : ℕ,
  q > 0 →
  isPrime q →
  isPrime (2 * q) →
  isPrime (180 - 3 * q) →
  (∀ p : ℕ, p < q → p > 0 → ¬(isPrime p ∧ isPrime (2 * p) ∧ isPrime (180 - 3 * p))) →
  q = 7 := by
  sorry

#check smallest_prime_angle_in_inscribed_triangle

end NUMINAMATH_CALUDE_smallest_prime_angle_in_inscribed_triangle_l3116_311653


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3116_311631

theorem quadratic_equation_solution : 
  let x₁ : ℝ := 2 + Real.sqrt 11
  let x₂ : ℝ := 2 - Real.sqrt 11
  ∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3116_311631


namespace NUMINAMATH_CALUDE_total_cents_l3116_311615

-- Define the values in cents
def lance_cents : ℕ := 70
def margaret_cents : ℕ := 75 -- three-fourths of a dollar
def guy_cents : ℕ := 50 + 10 -- two quarters and a dime
def bill_cents : ℕ := 6 * 10 -- six dimes

-- Theorem to prove
theorem total_cents : 
  lance_cents + margaret_cents + guy_cents + bill_cents = 265 := by
  sorry

end NUMINAMATH_CALUDE_total_cents_l3116_311615


namespace NUMINAMATH_CALUDE_mean_goals_is_4_1_l3116_311652

/-- The mean number of goals scored by soccer players -/
def mean_goals (players_3 players_4 players_5 players_6 : ℕ) 
               (goals_3 goals_4 goals_5 goals_6 : ℕ) : ℚ :=
  let total_goals := players_3 * goals_3 + players_4 * goals_4 + 
                     players_5 * goals_5 + players_6 * goals_6
  let total_players := players_3 + players_4 + players_5 + players_6
  (total_goals : ℚ) / total_players

/-- Theorem stating that the mean number of goals is 4.1 -/
theorem mean_goals_is_4_1 : 
  mean_goals 4 3 1 2 3 4 5 6 = 41 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_goals_is_4_1_l3116_311652


namespace NUMINAMATH_CALUDE_cookies_to_sell_l3116_311668

theorem cookies_to_sell (total : ℕ) (grandmother : ℕ) (uncle : ℕ) (neighbor : ℕ) 
  (h1 : total = 50)
  (h2 : grandmother = 12)
  (h3 : uncle = 7)
  (h4 : neighbor = 5) :
  total - (grandmother + uncle + neighbor) = 26 := by
  sorry

end NUMINAMATH_CALUDE_cookies_to_sell_l3116_311668


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3116_311688

theorem unique_three_digit_number : ∃! (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ (π b γ : ℕ), 
    π ≠ b ∧ π ≠ γ ∧ b ≠ γ ∧
    π < 10 ∧ b < 10 ∧ γ < 10 ∧
    n = 100 * π + 10 * b + γ ∧
    n = (π + b + γ) * (π + b + γ + 1)) ∧
  n = 156 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3116_311688


namespace NUMINAMATH_CALUDE_gain_percentage_cloth_sale_l3116_311673

/-- Calculates the gain percentage given the total quantity sold and the profit quantity -/
def gainPercentage (totalQuantity : ℕ) (profitQuantity : ℕ) : ℚ :=
  (profitQuantity : ℚ) / (totalQuantity : ℚ)

/-- Theorem: The gain percentage is 1/6 when selling 60 meters of cloth and gaining the selling price of 10 meters as profit -/
theorem gain_percentage_cloth_sale : 
  gainPercentage 60 10 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_gain_percentage_cloth_sale_l3116_311673


namespace NUMINAMATH_CALUDE_f_simplification_g_definition_g_value_at_pi_over_6_l3116_311689

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * sin (π - x) * sin x - (sin x - cos x)^2

noncomputable def g (x : ℝ) : ℝ := 2 * sin x + Real.sqrt 3 - 1

theorem f_simplification (x : ℝ) : f x = 2 * sin (2*x - π/3) + Real.sqrt 3 - 1 := by sorry

theorem g_definition (x : ℝ) : g x = 2 * sin x + Real.sqrt 3 - 1 := by sorry

theorem g_value_at_pi_over_6 : g (π/6) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_f_simplification_g_definition_g_value_at_pi_over_6_l3116_311689


namespace NUMINAMATH_CALUDE_carolyn_initial_marbles_l3116_311624

/-- Represents the number of marbles Carolyn started with -/
def initial_marbles : ℕ := sorry

/-- Represents the number of items Carolyn shared with Diana -/
def shared_items : ℕ := 42

/-- Represents the number of marbles Carolyn ended with -/
def remaining_marbles : ℕ := 5

/-- Theorem stating that Carolyn started with 47 marbles -/
theorem carolyn_initial_marbles : initial_marbles = 47 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_initial_marbles_l3116_311624


namespace NUMINAMATH_CALUDE_product_sum_equals_power_l3116_311681

theorem product_sum_equals_power : 
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) * (3^32 + 1) * (3^64 + 1) + 1 = 3^128 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equals_power_l3116_311681


namespace NUMINAMATH_CALUDE_minimum_number_of_boys_l3116_311630

theorem minimum_number_of_boys (k : ℕ) (n m : ℕ) : 
  (k > 0) →  -- total number of apples is positive
  (n > 0) →  -- there is at least one boy who collected 10 apples
  (m > 0) →  -- there is at least one boy who collected 10% of apples
  (100 * n + m * k = 10 * k) →  -- equation representing total apples collected
  (n + m ≥ 6) →  -- total number of boys is at least 6
  ∀ (n' m' : ℕ), (n' > 0) → (m' > 0) → 
    (∃ (k' : ℕ), k' > 0 ∧ 100 * n' + m' * k' = 10 * k') → 
    (n' + m' ≥ 6) :=
by
  sorry

#check minimum_number_of_boys

end NUMINAMATH_CALUDE_minimum_number_of_boys_l3116_311630


namespace NUMINAMATH_CALUDE_parabola_intersection_l3116_311658

theorem parabola_intersection (a b : ℝ) (h1 : a ≠ 0) : 
  (∀ x, a * (x - b) * (x - 1) = 0 → x = 3 ∨ x = 1) ∧
  a * (3 - b) * (3 - 1) = 0 →
  ∃ x, x ≠ 3 ∧ a * (x - b) * (x - 1) = 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3116_311658


namespace NUMINAMATH_CALUDE_basketball_expected_score_l3116_311683

def expected_score (p_in : ℝ) (p_out : ℝ) (n_in : ℕ) (n_out : ℕ) (points_in : ℕ) (points_out : ℕ) : ℝ :=
  (p_in * n_in * points_in) + (p_out * n_out * points_out)

theorem basketball_expected_score :
  expected_score 0.7 0.4 10 5 2 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_expected_score_l3116_311683


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l3116_311601

theorem cylinder_volume_change (r h : ℝ) (h_positive : 0 < h) (r_positive : 0 < r) : 
  π * r^2 * h = 15 → π * (3*r)^2 * (h/2) = 67.5 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l3116_311601


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3116_311672

theorem complex_fraction_simplification :
  (7 + 18 * Complex.I) / (3 - 4 * Complex.I) = (-51 / 25 : ℝ) + (82 / 25 : ℝ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3116_311672


namespace NUMINAMATH_CALUDE_principal_calculation_l3116_311643

/-- Proves that given specific conditions, the principal amount is 1600 --/
theorem principal_calculation (rate : ℚ) (time : ℚ) (amount : ℚ) :
  rate = 5 / 100 →
  time = 12 / 5 →
  amount = 1792 →
  amount = (1600 : ℚ) * (1 + rate * time) :=
by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l3116_311643


namespace NUMINAMATH_CALUDE_two_points_l3116_311687

/-- The number of integer points satisfying the given equation and conditions -/
def num_points : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let x := p.1
    let y := p.2
    x > 0 ∧ y > 0 ∧ x > y ∧ y * (x - 1) = 2 * x + 2018
  ) (Finset.product (Finset.range 10000) (Finset.range 10000))).card

/-- Theorem stating that there are exactly two points satisfying the conditions -/
theorem two_points : num_points = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_points_l3116_311687


namespace NUMINAMATH_CALUDE_arwen_tulips_l3116_311654

/-- Proves that Arwen picked 20 tulips, given the conditions of the problem -/
theorem arwen_tulips : 
  ∀ (a e : ℕ), 
    e = 2 * a →  -- Elrond picked twice as many tulips as Arwen
    a + e = 60 →  -- They picked 60 tulips in total
    a = 20  -- Arwen picked 20 tulips
    := by sorry

end NUMINAMATH_CALUDE_arwen_tulips_l3116_311654


namespace NUMINAMATH_CALUDE_mrs_heine_dogs_l3116_311622

theorem mrs_heine_dogs (total_biscuits : ℕ) (biscuits_per_dog : ℕ) (num_dogs : ℕ) : 
  total_biscuits = 6 →
  biscuits_per_dog = 3 →
  total_biscuits = num_dogs * biscuits_per_dog →
  num_dogs = 2 := by
sorry

end NUMINAMATH_CALUDE_mrs_heine_dogs_l3116_311622


namespace NUMINAMATH_CALUDE_matching_probability_theorem_l3116_311682

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.yellow + jb.red

/-- Abe's jelly beans -/
def abe : JellyBeans :=
  { green := 2, yellow := 0, red := 3 }

/-- Bob's jelly beans -/
def bob : JellyBeans :=
  { green := 2, yellow := 2, red := 3 }

/-- Calculates the probability of matching colors -/
def matchingProbability (person1 person2 : JellyBeans) : ℚ :=
  (person1.green * person2.green + person1.red * person2.red : ℚ) /
  ((person1.total * person2.total) : ℚ)

theorem matching_probability_theorem :
  matchingProbability abe bob = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_theorem_l3116_311682


namespace NUMINAMATH_CALUDE_stratified_sampling_l3116_311686

theorem stratified_sampling (total_male : ℕ) (total_female : ℕ) (selected_male : ℕ) :
  total_male = 56 →
  total_female = 42 →
  selected_male = 8 →
  (total_male : ℚ) / total_female = 4 / 3 →
  ∃ selected_female : ℕ, 
    (selected_female : ℚ) / selected_male = total_female / total_male ∧
    selected_female = 6 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l3116_311686


namespace NUMINAMATH_CALUDE_davids_biology_marks_l3116_311647

def marks_english : ℕ := 72
def marks_mathematics : ℕ := 60
def marks_physics : ℕ := 35
def marks_chemistry : ℕ := 62
def num_subjects : ℕ := 5
def average_marks : ℚ := 62.6

theorem davids_biology_marks :
  ∃ (marks_biology : ℕ),
    (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology) / num_subjects = average_marks ∧
    marks_biology = 84 := by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l3116_311647


namespace NUMINAMATH_CALUDE_particle_position_at_2004_l3116_311657

/-- Represents the position of a particle -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Defines the movement pattern of the particle -/
def next_position (p : Position) : Position :=
  if p.x = p.y then Position.mk (p.x + 1) p.y
  else if p.x > p.y then Position.mk p.x (p.y + 1)
  else Position.mk (p.x + 1) p.y

/-- Calculates the position of the particle after n seconds -/
def position_at_time (n : ℕ) : Position :=
  match n with
  | 0 => Position.mk 0 0
  | n + 1 => next_position (position_at_time n)

/-- The main theorem stating the position of the particle after 2004 seconds -/
theorem particle_position_at_2004 :
  position_at_time 2004 = Position.mk 20 44 := by
  sorry


end NUMINAMATH_CALUDE_particle_position_at_2004_l3116_311657


namespace NUMINAMATH_CALUDE_point_outside_circle_l3116_311623

/-- Given a circle defined by x^2 + y^2 - 2ax + a^2 - a = 0, if the point (a, a) can be outside this circle, then a > 1. -/
theorem point_outside_circle (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 2*a*x + a^2 - a = 0) → 
  (∃ x y : ℝ, x^2 + y^2 - 2*a*x + a^2 - a < 0 ∧ x = a ∧ y = a) → 
  a > 1 :=
by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3116_311623


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3116_311694

theorem modular_arithmetic_problem : ((367 * 373 * 379 % 53) * 383) % 47 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3116_311694


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l3116_311609

theorem arithmetic_series_sum : ∀ (a₁ aₙ : ℕ), 
  a₁ = 5 → aₙ = 105 → 
  ∃ (n : ℕ), n > 1 ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ n → ∃ d, a₁ + (k - 1) * d = aₙ) →
  (n * (a₁ + aₙ)) / 2 = 5555 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l3116_311609


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3116_311696

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3116_311696


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3116_311612

theorem decimal_to_fraction :
  (2.75 : ℚ) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3116_311612


namespace NUMINAMATH_CALUDE_factorization_proof_l3116_311691

theorem factorization_proof (x : ℝ) : 221 * x^2 + 68 * x + 17 = 17 * (13 * x^2 + 4 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3116_311691


namespace NUMINAMATH_CALUDE_initial_charge_calculation_l3116_311669

/-- A taxi company's pricing model -/
structure TaxiPricing where
  initial_charge : ℝ  -- Charge for the first 1/5 mile
  additional_charge : ℝ  -- Charge for each additional 1/5 mile
  total_charge : ℝ  -- Total charge for a specific ride
  ride_distance : ℝ  -- Distance of the ride in miles

/-- Theorem stating the initial charge for the first 1/5 mile -/
theorem initial_charge_calculation (tp : TaxiPricing) 
  (h1 : tp.additional_charge = 0.40)
  (h2 : tp.total_charge = 18.40)
  (h3 : tp.ride_distance = 8) :
  tp.initial_charge = 2.80 := by
  sorry

end NUMINAMATH_CALUDE_initial_charge_calculation_l3116_311669


namespace NUMINAMATH_CALUDE_third_runner_distance_l3116_311617

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ → ℝ

/-- The race setup -/
structure Race where
  length : ℝ
  runner1 : Runner
  runner2 : Runner
  runner3 : Runner

/-- Properties of the race -/
def RaceProperties (race : Race) : Prop :=
  -- The race is 100 meters long
  race.length = 100 ∧
  -- Runner1 is fastest, followed by runner2, then runner3
  race.runner1.speed > race.runner2.speed ∧ race.runner2.speed > race.runner3.speed ∧
  -- All runners maintain constant speeds
  (∀ t : ℝ, race.runner1.distance t = race.runner1.speed * t) ∧
  (∀ t : ℝ, race.runner2.distance t = race.runner2.speed * t) ∧
  (∀ t : ℝ, race.runner3.distance t = race.runner3.speed * t) ∧
  -- When runner1 finishes, runner2 is 10m behind
  race.runner2.distance (race.length / race.runner1.speed) = race.length - 10 ∧
  -- When runner2 finishes, runner3 is 10m behind
  race.runner3.distance (race.length / race.runner2.speed) = race.length - 10

/-- The main theorem to prove -/
theorem third_runner_distance (race : Race) (h : RaceProperties race) :
  race.runner3.distance (race.length / race.runner1.speed) = race.length - 19 := by
  sorry

end NUMINAMATH_CALUDE_third_runner_distance_l3116_311617


namespace NUMINAMATH_CALUDE_average_problem_l3116_311620

theorem average_problem (x : ℝ) : (20 + 30 + 40 + x) / 4 = 35 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l3116_311620


namespace NUMINAMATH_CALUDE_product_of_reciprocal_minus_one_bound_l3116_311661

theorem product_of_reciprocal_minus_one_bound 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reciprocal_minus_one_bound_l3116_311661


namespace NUMINAMATH_CALUDE_shanna_garden_harvest_l3116_311613

/-- Represents Shanna's garden --/
structure Garden where
  tomato : ℕ
  eggplant : ℕ
  pepper : ℕ

/-- Calculates the number of vegetables harvested from Shanna's garden --/
def harvest_vegetables (g : Garden) : ℕ :=
  let remaining_tomato := g.tomato / 2
  let remaining_pepper := g.pepper - 1
  let remaining_eggplant := g.eggplant
  let total_remaining := remaining_tomato + remaining_pepper + remaining_eggplant
  total_remaining * 7

/-- Theorem stating the total number of vegetables harvested from Shanna's garden --/
theorem shanna_garden_harvest :
  let initial_garden : Garden := ⟨6, 2, 4⟩
  harvest_vegetables initial_garden = 56 := by
  sorry

end NUMINAMATH_CALUDE_shanna_garden_harvest_l3116_311613


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3116_311648

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15) 
  (h2 : avg_age_all = 15) 
  (h3 : group1_size = 7) 
  (h4 : avg_age_group1 = 14) 
  (h5 : group2_size = 7) 
  (h6 : avg_age_group2 = 16) : 
  ℝ := by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l3116_311648


namespace NUMINAMATH_CALUDE_all_positive_l3116_311690

theorem all_positive (a b c : ℝ) 
  (sum_pos : a + b + c > 0) 
  (sum_prod_pos : a * b + b * c + c * a > 0) 
  (prod_pos : a * b * c > 0) : 
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_all_positive_l3116_311690


namespace NUMINAMATH_CALUDE_math_competition_scores_math_competition_result_l3116_311660

theorem math_competition_scores (total_avg : ℝ) (male_ratio : ℝ) (female_score_ratio : ℝ) : ℝ :=
  let female_ratio := 1
  let male_count := male_ratio * female_ratio
  let total_count := male_count + female_ratio
  let male_avg := total_avg * total_count / (male_count + female_ratio * female_score_ratio)
  let female_avg := male_avg * female_score_ratio
  female_avg

theorem math_competition_result : 
  math_competition_scores 75 1.8 1.2 = 84 := by sorry

end NUMINAMATH_CALUDE_math_competition_scores_math_competition_result_l3116_311660


namespace NUMINAMATH_CALUDE_order_of_abc_l3116_311621

-- Define the constants
noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log (1/2)
noncomputable def c : ℝ := Real.log 2 / Real.log (1/3)

-- State the theorem
theorem order_of_abc : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l3116_311621


namespace NUMINAMATH_CALUDE_book_length_is_4556_l3116_311671

/-- Represents the properties of a book and a reader's progress. -/
structure BookReading where
  total_hours : Nat
  pages_read : Nat
  speed_increase : Nat
  extra_pages : Nat

/-- Calculates the total number of pages in the book based on the given reading information. -/
def calculate_total_pages (reading : BookReading) : Nat :=
  reading.pages_read + (reading.pages_read - reading.extra_pages)

/-- Theorem stating that given the specific reading conditions, the total number of pages in the book is 4556. -/
theorem book_length_is_4556 (reading : BookReading)
  (h1 : reading.total_hours = 5)
  (h2 : reading.pages_read = 2323)
  (h3 : reading.speed_increase = 10)
  (h4 : reading.extra_pages = 90) :
  calculate_total_pages reading = 4556 := by
  sorry

#eval calculate_total_pages { total_hours := 5, pages_read := 2323, speed_increase := 10, extra_pages := 90 }

end NUMINAMATH_CALUDE_book_length_is_4556_l3116_311671


namespace NUMINAMATH_CALUDE_apples_buyers_l3116_311695

theorem apples_buyers (men_apples : ℕ) (women_apples : ℕ) (total_apples : ℕ) :
  men_apples = 30 →
  women_apples = men_apples + 20 →
  total_apples = 210 →
  ∃ (num_men : ℕ), num_men * men_apples + 3 * women_apples = total_apples ∧ num_men = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_buyers_l3116_311695


namespace NUMINAMATH_CALUDE_gp_ratio_is_four_l3116_311663

theorem gp_ratio_is_four (x : ℝ) :
  (∀ r : ℝ, (40 + x) = (10 + x) * r ∧ (160 + x) = (40 + x) * r) →
  r = 4 :=
by sorry

end NUMINAMATH_CALUDE_gp_ratio_is_four_l3116_311663


namespace NUMINAMATH_CALUDE_value_of_k_l3116_311616

theorem value_of_k (a b k : ℝ) (h1 : 2 * a = k) (h2 : 3 * b = k) (h3 : k ≠ 1) (h4 : 2 * a + b = a * b) : k = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_k_l3116_311616


namespace NUMINAMATH_CALUDE_total_distance_QY_l3116_311633

/-- Proves that the total distance between Q and Y is 45 km --/
theorem total_distance_QY (matthew_speed johnny_speed : ℝ)
  (johnny_distance : ℝ) (time_difference : ℝ) :
  matthew_speed = 3 →
  johnny_speed = 4 →
  johnny_distance = 24 →
  time_difference = 1 →
  ∃ (total_distance : ℝ), total_distance = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_total_distance_QY_l3116_311633


namespace NUMINAMATH_CALUDE_bucket_capacity_is_seven_l3116_311675

/-- The capacity of the tank in litres -/
def tank_capacity : ℕ := 12 * 49

/-- The number of buckets needed in the second scenario -/
def buckets_second_scenario : ℕ := 84

/-- The capacity of each bucket in the second scenario -/
def bucket_capacity_second_scenario : ℚ := tank_capacity / buckets_second_scenario

theorem bucket_capacity_is_seven :
  bucket_capacity_second_scenario = 7 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_is_seven_l3116_311675


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_4_l3116_311698

theorem ceiling_neg_sqrt_64_over_4 : ⌈-Real.sqrt (64 / 4)⌉ = -4 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_4_l3116_311698


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l3116_311677

/-- Given a quadratic inequality x^2 + bx - a < 0 with solution set {x | 3 < x < 4},
    prove that the sum of coefficients a + b = -19 -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) :
  (∀ x : ℝ, x^2 + b*x - a < 0 ↔ 3 < x ∧ x < 4) →
  a + b = -19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l3116_311677


namespace NUMINAMATH_CALUDE_factorization_4x3_4x2_x_l3116_311638

theorem factorization_4x3_4x2_x (x : ℝ) : 4 * x^3 - 4 * x^2 + x = x * (2*x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_4x3_4x2_x_l3116_311638


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3116_311600

/-- Represents a rectangle with length and width --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def areaOfRectangle (r : Rectangle) : ℝ :=
  r.length * r.width

theorem equal_area_rectangles (carol_rect jordan_rect : Rectangle) :
  carol_rect.length = 5 →
  carol_rect.width = 24 →
  jordan_rect.length = 12 →
  areaOfRectangle carol_rect = areaOfRectangle jordan_rect →
  jordan_rect.width = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_equal_area_rectangles_l3116_311600


namespace NUMINAMATH_CALUDE_expression_simplification_l3116_311634

theorem expression_simplification (m n : ℝ) (hm : m ≠ 0) :
  (m^(4/3) - 27 * m^(1/3) * n) / (m^(2/3) + 3 * (m*n)^(1/3) + 9 * n^(2/3)) / (1 - 3 * (n/m)^(1/3)) - m^(2/3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3116_311634


namespace NUMINAMATH_CALUDE_min_value_xyz_one_min_value_achievable_l3116_311644

theorem min_value_xyz_one (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  3 * x^2 + 12 * x * y + 9 * y^2 + 15 * y * z + 3 * z^2 ≥ 243 / Real.rpow 4 (1/9) :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 1 ∧
  3 * x^2 + 12 * x * y + 9 * y^2 + 15 * y * z + 3 * z^2 = 243 / Real.rpow 4 (1/9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_one_min_value_achievable_l3116_311644


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3116_311604

theorem triangle_abc_properties (a b c A B C : ℝ) : 
  b = Real.sqrt 2 →
  c = 1 →
  Real.cos B = 3/4 →
  (Real.sin C = Real.sqrt 14 / 8 ∧ 
   Real.sin A * b * c / 2 = Real.sqrt 7 / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3116_311604


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l3116_311699

theorem number_subtraction_problem (x y : ℝ) : 
  (x - 5) / 7 = 7 → (x - y) / 13 = 4 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l3116_311699


namespace NUMINAMATH_CALUDE_solve_equation_l3116_311674

theorem solve_equation (c m n x : ℝ) 
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hmn : m ≠ n)
  (hc : c = 3)
  (hm2 : m = 2)
  (hn5 : n = 5)
  (heq : (x + c * m)^2 - (x + c * n)^2 = (m - n)^2) :
  x = -11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3116_311674


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_ten_l3116_311670

def A (n : ℕ) : ℕ := (List.range n).foldl (λ acc k => acc * Nat.choose (k^2) k) 1

theorem smallest_n_divisible_by_ten : 
  (∀ m < 4, ¬(10 ∣ A m)) ∧ (10 ∣ A 4) := by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_ten_l3116_311670


namespace NUMINAMATH_CALUDE_total_components_total_components_proof_l3116_311611

/-- The total number of components of types A, B, and C is 900. -/
theorem total_components : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (total_B : ℕ) (total_C : ℕ) (sample_size : ℕ) (sample_A : ℕ) (sample_C : ℕ) (total : ℕ) =>
    total_B = 300 →
    total_C = 200 →
    sample_size = 45 →
    sample_A = 20 →
    sample_C = 10 →
    total = 900

/-- Proof of the theorem -/
theorem total_components_proof :
  total_components 300 200 45 20 10 900 := by
  sorry

end NUMINAMATH_CALUDE_total_components_total_components_proof_l3116_311611


namespace NUMINAMATH_CALUDE_solve_for_x_l3116_311640

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3116_311640


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3116_311651

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ - 4 = 0) → (x₂^2 - 3*x₂ - 4 = 0) → (x₁ + x₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3116_311651


namespace NUMINAMATH_CALUDE_solve_simple_interest_l3116_311676

def simple_interest_problem (rate : ℚ) (principal : ℚ) (interest_difference : ℚ) : Prop :=
  let simple_interest := principal - interest_difference
  let years := simple_interest / (principal * rate)
  years = 5

theorem solve_simple_interest :
  simple_interest_problem (4/100) 3000 2400 := by
  sorry

end NUMINAMATH_CALUDE_solve_simple_interest_l3116_311676


namespace NUMINAMATH_CALUDE_unique_solution_in_interval_l3116_311650

theorem unique_solution_in_interval (x : ℝ) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  ((2 - Real.sin (2 * x)) * Real.sin (x + Real.pi / 4) = 1) ↔
  (x = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_in_interval_l3116_311650


namespace NUMINAMATH_CALUDE_jason_gave_keith_47_pears_l3116_311655

/-- The number of pears Jason gave to Keith -/
def pears_given_to_keith (initial_pears : ℕ) (pears_from_mike : ℕ) (pears_left : ℕ) : ℕ :=
  initial_pears + pears_from_mike - pears_left

theorem jason_gave_keith_47_pears :
  pears_given_to_keith 46 12 11 = 47 := by
  sorry

end NUMINAMATH_CALUDE_jason_gave_keith_47_pears_l3116_311655


namespace NUMINAMATH_CALUDE_remainder_problem_l3116_311603

theorem remainder_problem (N : ℤ) : 
  (∃ k : ℤ, N = 45 * k + 31) → (∃ m : ℤ, N = 15 * m + 1) :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3116_311603


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3116_311666

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 - |x| + 6 > 0

-- Define the solution set
def solution_set : Set ℝ := {x | -2 < x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3116_311666


namespace NUMINAMATH_CALUDE_count_pairs_with_harmonic_mean_5_20_l3116_311608

/-- The number of ordered pairs of positive integers with harmonic mean 5^20 -/
def count_pairs : ℕ := 20

/-- Harmonic mean of two numbers -/
def harmonic_mean (x y : ℕ) : ℚ := 2 * x * y / (x + y)

/-- Predicate for valid pairs -/
def is_valid_pair (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ x < y ∧ harmonic_mean x y = 5^20

/-- The main theorem -/
theorem count_pairs_with_harmonic_mean_5_20 :
  (∃ (S : Finset (ℕ × ℕ)), S.card = count_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ S ↔ is_valid_pair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_count_pairs_with_harmonic_mean_5_20_l3116_311608
