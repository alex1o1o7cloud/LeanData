import Mathlib

namespace NUMINAMATH_CALUDE_range_of_m_range_of_t_l2339_233955

noncomputable section

-- Define the functions f and g
def f (x t : ℝ) : ℝ := -x^2 + 2 * Real.exp 1 * x + t - 1
def g (x : ℝ) : ℝ := x + (Real.exp 1)^2 / x

-- State the theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, x > 0 ∧ g x = m) ↔ m ≥ 2 * Real.exp 1 :=
sorry

-- State the theorem for the range of t
theorem range_of_t :
  ∀ t : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ g x₁ - f x₁ t = 0 ∧ g x₂ - f x₂ t = 0)
  ↔ t > 2 * Real.exp 1 - (Real.exp 1)^2 + 1 :=
sorry

end

end NUMINAMATH_CALUDE_range_of_m_range_of_t_l2339_233955


namespace NUMINAMATH_CALUDE_min_value_expression_l2339_233949

theorem min_value_expression (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x^2 + y^2 + z^2 = 1) :
  (x*y/z + y*z/x + z*x/y) ≥ Real.sqrt 3 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 1 ∧ 
    a*b/c + b*c/a + c*a/b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2339_233949


namespace NUMINAMATH_CALUDE_gerald_remaining_pfennigs_l2339_233968

/-- Represents the number of farthings in a pfennig -/
def farthings_per_pfennig : ℕ := 6

/-- Represents the number of farthings Gerald has -/
def geralds_farthings : ℕ := 54

/-- Represents the cost of a meat pie in pfennigs -/
def meat_pie_cost : ℕ := 2

/-- Calculates the number of pfennigs Gerald will have left after buying the pie -/
def remaining_pfennigs : ℕ :=
  geralds_farthings / farthings_per_pfennig - meat_pie_cost

/-- Theorem stating that Gerald will have 7 pfennigs left after buying the pie -/
theorem gerald_remaining_pfennigs :
  remaining_pfennigs = 7 := by sorry

end NUMINAMATH_CALUDE_gerald_remaining_pfennigs_l2339_233968


namespace NUMINAMATH_CALUDE_ball_drawing_game_l2339_233935

theorem ball_drawing_game (x : ℕ) : 
  (2 : ℕ) > 0 ∧ x > 0 →
  (4 * x : ℚ) / ((x + 2) * (x + 1)) ≥ 1/5 ∧
  (4 * x : ℚ) / ((x + 2) * (x + 1)) ≤ 33/100 →
  9 ≤ x ∧ x ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_game_l2339_233935


namespace NUMINAMATH_CALUDE_odd_painted_faces_5x5x1_l2339_233903

/-- Represents a 3D grid of unit cubes -/
structure CubeGrid :=
  (length : Nat)
  (width : Nat)
  (height : Nat)

/-- Counts the number of cubes with an odd number of painted faces in a given grid -/
def countOddPaintedFaces (grid : CubeGrid) : Nat :=
  sorry

/-- The main theorem stating that a 5x5x1 grid has 9 cubes with an odd number of painted faces -/
theorem odd_painted_faces_5x5x1 :
  let grid := CubeGrid.mk 5 5 1
  countOddPaintedFaces grid = 9 := by
  sorry

end NUMINAMATH_CALUDE_odd_painted_faces_5x5x1_l2339_233903


namespace NUMINAMATH_CALUDE_circle_tangent_and_passes_through_l2339_233932

/-- The line to which the circle is tangent -/
def tangent_line (x y : ℝ) : Prop := 4 * x - 3 * y + 6 = 0

/-- The point of tangency -/
def point_A : ℝ × ℝ := (3, 6)

/-- The point through which the circle passes -/
def point_B : ℝ × ℝ := (5, 2)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x - 5)^2 + (y - 9/2)^2 = 25/4

/-- Theorem stating that the given circle equation represents the circle
    that is tangent to the line at point A and passes through point B -/
theorem circle_tangent_and_passes_through :
  (∀ x y, tangent_line x y → circle_equation x y → (x, y) = point_A) ∧
  circle_equation point_B.1 point_B.2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_and_passes_through_l2339_233932


namespace NUMINAMATH_CALUDE_convex_polygon_four_equal_areas_l2339_233907

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  -- We don't need to define the internal structure of the polygon
  -- for this theorem statement

/-- Represents a line in 2D space -/
structure Line where
  -- We don't need to define the internal structure of the line
  -- for this theorem statement

/-- Represents an area measurement -/
def Area : Type := ℝ

/-- Function to calculate the area of a region of a polygon -/
def areaOfRegion (p : ConvexPolygon) (region : Set (ℝ × ℝ)) : Area :=
  sorry -- Implementation not needed for the theorem statement

/-- Two lines are perpendicular -/
def arePerpendicular (l1 l2 : Line) : Prop :=
  sorry -- Definition not needed for the theorem statement

/-- A line divides a polygon into two regions -/
def dividePolygon (p : ConvexPolygon) (l : Line) : (Set (ℝ × ℝ)) × (Set (ℝ × ℝ)) :=
  sorry -- Implementation not needed for the theorem statement

/-- Theorem: Any convex polygon can be divided into four equal areas by two perpendicular lines -/
theorem convex_polygon_four_equal_areas (p : ConvexPolygon) :
  ∃ (l1 l2 : Line),
    arePerpendicular l1 l2 ∧
    let (r1, r2) := dividePolygon p l1
    let (r11, r12) := dividePolygon p l2
    let a1 := areaOfRegion p (r1 ∩ r11)
    let a2 := areaOfRegion p (r1 ∩ r12)
    let a3 := areaOfRegion p (r2 ∩ r11)
    let a4 := areaOfRegion p (r2 ∩ r12)
    a1 = a2 ∧ a2 = a3 ∧ a3 = a4 :=
  sorry

end NUMINAMATH_CALUDE_convex_polygon_four_equal_areas_l2339_233907


namespace NUMINAMATH_CALUDE_orange_preference_percentage_l2339_233920

/-- The color preferences survey results -/
def color_frequencies : List (String × ℕ) :=
  [("Red", 75), ("Blue", 80), ("Green", 50), ("Yellow", 45), ("Purple", 60), ("Orange", 55)]

/-- The total number of responses in the survey -/
def total_responses : ℕ := (color_frequencies.map (·.2)).sum

/-- Calculate the percentage of respondents who preferred a given color -/
def color_percentage (color : String) : ℚ :=
  match color_frequencies.find? (·.1 = color) with
  | some (_, freq) => (freq : ℚ) / (total_responses : ℚ) * 100
  | none => 0

/-- The theorem stating that the percentage who preferred orange is 15% -/
theorem orange_preference_percentage :
  ⌊color_percentage "Orange"⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_orange_preference_percentage_l2339_233920


namespace NUMINAMATH_CALUDE_rachels_apple_tree_l2339_233976

theorem rachels_apple_tree (initial : ℕ) : 
  (initial - 2 + 3 = 5) → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachels_apple_tree_l2339_233976


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2339_233996

theorem polynomial_division_theorem (x : ℝ) :
  (8*x^3 - 9*x^2 + 21*x - 47) * (x + 2) + 86 = 8*x^4 + 7*x^3 + 3*x^2 - 5*x - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2339_233996


namespace NUMINAMATH_CALUDE_right_triangle_in_circle_l2339_233975

theorem right_triangle_in_circle (diameter : ℝ) (leg1 : ℝ) (leg2 : ℝ) : 
  diameter = 10 → leg1 = 6 → leg2 * leg2 = diameter * diameter - leg1 * leg1 → leg2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_in_circle_l2339_233975


namespace NUMINAMATH_CALUDE_probability_no_shaded_correct_l2339_233956

/-- Represents a 2 by 1001 rectangle with middle squares shaded --/
structure ShadedRectangle where
  width : Nat
  height : Nat
  shaded_column : Nat

/-- The probability of choosing a rectangle without a shaded square --/
def probability_no_shaded (r : ShadedRectangle) : ℚ :=
  500 / 1001

/-- Theorem stating the probability of choosing a rectangle without a shaded square --/
theorem probability_no_shaded_correct (r : ShadedRectangle) 
  (h1 : r.width = 1001) 
  (h2 : r.height = 2) 
  (h3 : r.shaded_column = 501) : 
  probability_no_shaded r = 500 / 1001 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_shaded_correct_l2339_233956


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2339_233930

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 12) : 
  x^2 + 2*x*y + 3*y^2 ≤ 18 + 12*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2339_233930


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l2339_233989

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 20) 
  (h2 : c + a = 22) 
  (h3 : a + b = 24) : 
  Real.sqrt (2 * a * b * c * (a + b + c)) = 1287 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l2339_233989


namespace NUMINAMATH_CALUDE_fraction_inequality_l2339_233918

theorem fraction_inequality (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a : ℚ) / b < (a + 1 : ℚ) / (b + 1)) : 
  (2012 * a : ℚ) / b > 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2339_233918


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l2339_233933

theorem quadratic_equation_two_distinct_roots :
  let a : ℝ := 1
  let b : ℝ := 0
  let c : ℝ := -2
  let Δ : ℝ := b^2 - 4*a*c
  (∀ (a b c : ℝ), (b^2 - 4*a*c > 0) ↔ (∃ (x y : ℝ), x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0)) →
  ∃ (x y : ℝ), x ≠ y ∧ x^2 - 2 = 0 ∧ y^2 - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l2339_233933


namespace NUMINAMATH_CALUDE_roll_three_probability_l2339_233922

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Finset Nat)
  (fair : sides = {1, 2, 3, 4, 5, 6})

/-- The event of rolling a 3 -/
def rollThree (d : FairDie) : Finset Nat :=
  {3}

/-- The probability of an event for a fair die -/
def probability (d : FairDie) (event : Finset Nat) : Rat :=
  (event ∩ d.sides).card / d.sides.card

theorem roll_three_probability (d : FairDie) :
  probability d (rollThree d) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_roll_three_probability_l2339_233922


namespace NUMINAMATH_CALUDE_circle_equation_characterization_l2339_233986

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt_2 : radius = Real.sqrt 2
  passes_through_point : passes_through = (-2, 1)

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  ((x - c.center.1) ^ 2 + y ^ 2) = c.radius ^ 2

theorem circle_equation_characterization (c : Circle) :
  ∃ a : ℝ, (a = -1 ∨ a = -3) ∧
    ∀ x y : ℝ, circle_equation c x y ↔ ((x + a) ^ 2 + y ^ 2 = 2) :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_characterization_l2339_233986


namespace NUMINAMATH_CALUDE_harmonic_sum_equality_l2339_233998

/-- The nth harmonic number -/
def h (n : ℕ+) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

/-- The sum of harmonic numbers up to n-1 -/
def sum_h (n : ℕ+) : ℚ :=
  (Finset.range (n - 1)).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem: n + sum of h(1) to h(n-1) equals n * h(n) for n ≥ 2 -/
theorem harmonic_sum_equality (n : ℕ+) (hn : n ≥ 2) :
  (n : ℚ) + sum_h n = n * h n := by sorry

end NUMINAMATH_CALUDE_harmonic_sum_equality_l2339_233998


namespace NUMINAMATH_CALUDE_system_solution_l2339_233959

def satisfies_system (u v w : ℝ) : Prop :=
  u + v * w = 12 ∧ v + w * u = 12 ∧ w + u * v = 12

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(3, 3, 3), (-4, -4, -4), (1, 1, 11), (11, 1, 1), (1, 11, 1)}

theorem system_solution :
  {p : ℝ × ℝ × ℝ | satisfies_system p.1 p.2.1 p.2.2} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2339_233959


namespace NUMINAMATH_CALUDE_fraction_leading_zeros_l2339_233926

-- Define the fraction
def fraction : ℚ := 7 / 5000

-- Define a function to count leading zeros in a decimal representation
def countLeadingZeros (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem fraction_leading_zeros :
  countLeadingZeros fraction = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_leading_zeros_l2339_233926


namespace NUMINAMATH_CALUDE_multiply_polynomial_equality_l2339_233946

theorem multiply_polynomial_equality (x : ℝ) :
  (x^6 + 27*x^3 + 729) * (x^3 - 27) = x^12 + 27*x^9 - 19683*x^3 - 531441 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_equality_l2339_233946


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l2339_233934

/-- The shortest distance between a point on the parabola y = -x^2 + 5x + 7 
    and a point on the line y = 2x - 3 is 31√5/20 -/
theorem shortest_distance_parabola_to_line :
  let parabola := fun x : ℝ => -x^2 + 5*x + 7
  let line := fun x : ℝ => 2*x - 3
  ∃ (d : ℝ), d = (31 * Real.sqrt 5) / 20 ∧
    ∀ (x₁ x₂ : ℝ), 
      d ≤ Real.sqrt ((x₁ - x₂)^2 + (parabola x₁ - line x₂)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l2339_233934


namespace NUMINAMATH_CALUDE_sqrt_five_squared_minus_four_squared_l2339_233927

theorem sqrt_five_squared_minus_four_squared : 
  Real.sqrt (5^2 - 4^2) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_minus_four_squared_l2339_233927


namespace NUMINAMATH_CALUDE_perimeter_of_square_region_l2339_233952

/-- The perimeter of a region formed by 8 congruent squares arranged in a 2x4 rectangle,
    given that the total area of the region is 512 square centimeters. -/
theorem perimeter_of_square_region (total_area : ℝ) (num_squares : ℕ) (rows cols : ℕ) :
  total_area = 512 →
  num_squares = 8 →
  rows = 2 →
  cols = 4 →
  let square_side := Real.sqrt (total_area / num_squares)
  let perimeter := 2 * square_side * (rows + cols)
  perimeter = 128 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_square_region_l2339_233952


namespace NUMINAMATH_CALUDE_scheduled_halt_duration_l2339_233971

def average_speed : ℝ := 87
def total_distance : ℝ := 348
def scheduled_start_time : ℝ := 9
def scheduled_end_time : ℝ := 13.75  -- 1:45 PM in decimal hours

theorem scheduled_halt_duration :
  let travel_time_without_halt := total_distance / average_speed
  let scheduled_travel_time := scheduled_end_time - scheduled_start_time
  scheduled_travel_time - travel_time_without_halt = 0.75 := by sorry

end NUMINAMATH_CALUDE_scheduled_halt_duration_l2339_233971


namespace NUMINAMATH_CALUDE_expected_other_marbles_is_two_l2339_233950

/-- Represents the distribution of marble colors in Percius's collection -/
structure MarbleCollection where
  clear_percent : ℚ
  black_percent : ℚ
  other_percent : ℚ
  sum_to_one : clear_percent + black_percent + other_percent = 1

/-- Calculates the expected number of marbles of a certain color when taking a sample -/
def expected_marbles (collection : MarbleCollection) (sample_size : ℕ) (color_percent : ℚ) : ℚ :=
  color_percent * sample_size

/-- Theorem: The expected number of other-colored marbles in a sample of 5 is 2 -/
theorem expected_other_marbles_is_two (collection : MarbleCollection) 
    (h1 : collection.clear_percent = 2/5)
    (h2 : collection.black_percent = 1/5) :
    expected_marbles collection 5 collection.other_percent = 2 := by
  sorry

#eval expected_marbles ⟨2/5, 1/5, 2/5, by norm_num⟩ 5 (2/5)

end NUMINAMATH_CALUDE_expected_other_marbles_is_two_l2339_233950


namespace NUMINAMATH_CALUDE_divisibility_by_power_of_two_l2339_233939

theorem divisibility_by_power_of_two (n : ℕ) (h : n > 0) :
  ∃ x : ℤ, (2^n : ℤ) ∣ (x^2 - 17) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_power_of_two_l2339_233939


namespace NUMINAMATH_CALUDE_perfect_cubes_between_500_and_2000_l2339_233965

theorem perfect_cubes_between_500_and_2000 : 
  (Finset.filter (fun n => 500 ≤ n^3 ∧ n^3 ≤ 2000) (Finset.range 13)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cubes_between_500_and_2000_l2339_233965


namespace NUMINAMATH_CALUDE_rosie_pies_theorem_l2339_233979

/-- Represents the number of pies Rosie can make given a certain number of apples -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples * 3) / 12

theorem rosie_pies_theorem :
  pies_from_apples 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_theorem_l2339_233979


namespace NUMINAMATH_CALUDE_proposition_relation_l2339_233957

theorem proposition_relation (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) →
  (0 < a ∧ a < 1) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) → (0 < a ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relation_l2339_233957


namespace NUMINAMATH_CALUDE_inverse_of_A_l2339_233937

def A : Matrix (Fin 2) (Fin 2) ℚ := ![![5, -3], ![2, 1]]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![1/11, 3/11], ![-2/11, 5/11]]

theorem inverse_of_A : A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2339_233937


namespace NUMINAMATH_CALUDE_no_matching_roots_l2339_233978

theorem no_matching_roots : ∀ x : ℝ,
  (x^2 - 4*x + 3 = 0) → 
  ¬(∃ y : ℝ, (y = x - 1 ∧ y = x - 3)) :=
by sorry

end NUMINAMATH_CALUDE_no_matching_roots_l2339_233978


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2339_233919

theorem least_addition_for_divisibility (n : ℕ) : 
  let x := 278
  (∀ y : ℕ, y < x → ¬((1056 + y) % 23 = 0 ∧ (1056 + y) % 29 = 0)) ∧
  ((1056 + x) % 23 = 0 ∧ (1056 + x) % 29 = 0) := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2339_233919


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2339_233942

/-- The sum of roots of two quadratic equations given specific conditions -/
theorem sum_of_roots_quadratic (a b c d p q : ℝ) : a ≠ 0 →
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2020*a*x + c = 0 ∧ y^2 + 2020*a*y + c = 0) →
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + d = 0 ∧ a*y^2 + b*y + d = 0) →
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + p*x + q = 0 ∧ a*y^2 + p*y + q = 0) →
  (∃ w x y z : ℝ, a*w^2 + b*w + d = 0 ∧ a*x^2 + b*x + d = 0 ∧
                  a*y^2 + p*y + q = 0 ∧ a*z^2 + p*z + q = 0 ∧
                  w + x + y + z = 2020) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2339_233942


namespace NUMINAMATH_CALUDE_paper_sheets_count_l2339_233929

/-- Represents the dimensions of a rectangle in centimeters -/
structure Dimensions where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℝ := d.width * d.height

/-- Converts meters to centimeters -/
def meters_to_cm (m : ℝ) : ℝ := m * 100

theorem paper_sheets_count :
  let plank : Dimensions := ⟨meters_to_cm 6, meters_to_cm 4⟩
  let paper : Dimensions := ⟨60, 20⟩
  (area plank) / (area paper) = 200 := by sorry

end NUMINAMATH_CALUDE_paper_sheets_count_l2339_233929


namespace NUMINAMATH_CALUDE_bottle_production_l2339_233970

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 12 such machines will produce 2160 bottles in 4 minutes. -/
theorem bottle_production
  (machines : ℕ)
  (bottles_per_minute : ℕ)
  (h1 : machines = 6)
  (h2 : bottles_per_minute = 270)
  (time : ℕ)
  (h3 : time = 4) :
  (12 * bottles_per_minute * time) / machines = 2160 :=
sorry

end NUMINAMATH_CALUDE_bottle_production_l2339_233970


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l2339_233953

theorem quadratic_roots_problem (x₁ x₂ m : ℝ) : 
  (x₁^2 - 2*(m+1)*x₁ + m^2 - 3 = 0) →
  (x₂^2 - 2*(m+1)*x₂ + m^2 - 3 = 0) →
  (x₁^2 + x₂^2 - x₁*x₂ = 33) →
  (m = 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l2339_233953


namespace NUMINAMATH_CALUDE_equation_solutions_l2339_233964

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 9*x - 12) + 1 / (x^2 + 5*x - 14) - 1 / (x^2 - 15*x - 18) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = {2, -9, 6, -3} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2339_233964


namespace NUMINAMATH_CALUDE_smallest_common_multiple_9_6_l2339_233908

theorem smallest_common_multiple_9_6 : ∀ n : ℕ, n > 0 ∧ 9 ∣ n ∧ 6 ∣ n → n ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_9_6_l2339_233908


namespace NUMINAMATH_CALUDE_sin_equation_range_l2339_233966

theorem sin_equation_range : 
  let f : ℝ → ℝ := λ x => Real.sin x ^ 2 - 2 * Real.sin x
  ∃ (a_min a_max : ℝ), a_min = -1 ∧ a_max = 3 ∧
    (∀ a : ℝ, (∃ x : ℝ, f x = a) ↔ a_min ≤ a ∧ a ≤ a_max) :=
by sorry

end NUMINAMATH_CALUDE_sin_equation_range_l2339_233966


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2339_233961

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 8) :
  ∀ z : ℝ, z = x + 2*y → z ≥ 4 ∧ ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ x' + 2*y' + 2*x'*y' = 8 ∧ x' + 2*y' = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2339_233961


namespace NUMINAMATH_CALUDE_equal_cake_division_l2339_233910

theorem equal_cake_division (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) :
  total_cakes = 18 →
  num_children = 3 →
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
sorry

end NUMINAMATH_CALUDE_equal_cake_division_l2339_233910


namespace NUMINAMATH_CALUDE_triangle_exists_and_satisfies_inequality_l2339_233901

/-- Theorem: Existence of a triangle with sides 9, 15, and 21 satisfying the triangle inequality. -/
theorem triangle_exists_and_satisfies_inequality : ∃ (a b c : ℝ),
  a = 9 ∧ b = 15 ∧ c = 21 ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
  (∃ (x : ℝ), a = 9 ∧ b = x + 6 ∧ c = 2*x + 3 ∧ a + b + c = 45) :=
by sorry

end NUMINAMATH_CALUDE_triangle_exists_and_satisfies_inequality_l2339_233901


namespace NUMINAMATH_CALUDE_three_lines_intersection_l2339_233940

/-- A line in the plane represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of intersection points between three lines -/
def num_intersections (l1 l2 l3 : Line) : ℕ :=
  sorry

theorem three_lines_intersection :
  let l1 : Line := { a := -4, b := 6, c := 2 }
  let l2 : Line := { a := 1, b := 2, c := 2 }
  let l3 : Line := { a := -4, b := 6, c := 3 }
  num_intersections l1 l2 l3 = 2 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l2339_233940


namespace NUMINAMATH_CALUDE_right_triangle_dot_product_l2339_233911

/-- Given a right triangle ABC with ∠ABC = 90°, AB = 4, and BC = 3, 
    prove that the dot product of AC and BC is 9. -/
theorem right_triangle_dot_product (A B C : ℝ × ℝ) : 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 4^2 →  -- AB = 4
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 3^2 →  -- BC = 3
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0 →  -- ∠ABC = 90°
  ((C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2)) = 9 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_dot_product_l2339_233911


namespace NUMINAMATH_CALUDE_proposition_count_l2339_233912

theorem proposition_count : 
  (∃ (correct : Finset (Fin 6)) (h : correct.card = 5),
    (∀ i : Fin 6, i ∈ correct ↔
      (i = 0 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → |a| > |b|) ∨
      (i = 1 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → a + b < a * b) ∨
      (i = 2 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → b / a + a / b > 2) ∨
      (i = 3 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → a^2 / b < 2 * a - b) ∨
      (i = 4 ∧ ∀ a b : ℝ, b < a ∧ a < 0 → (2 * a + b) / (a + 2 * b) > a / b) ∨
      (i = 5 ∧ ∀ a b : ℝ, a + b = 1 → a^2 + b^2 ≥ 1 / 2))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_count_l2339_233912


namespace NUMINAMATH_CALUDE_no_solution_implies_a_range_l2339_233948

theorem no_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(x - 2*a > 0 ∧ 3 - 2*x > x - 6)) → a ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_range_l2339_233948


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2339_233938

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2339_233938


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l2339_233999

theorem no_infinite_sequence_exists : ¬ ∃ (k : ℕ → ℝ), 
  (∀ n : ℕ, k (n + 1) = k n - 1 / k n) ∧ 
  (∀ n : ℕ, k n * k (n + 1) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l2339_233999


namespace NUMINAMATH_CALUDE_employed_males_percentage_proof_l2339_233906

/-- The percentage of the population that is employed -/
def employed_percentage : ℝ := 72

/-- The percentage of employed people who are female -/
def female_employed_percentage : ℝ := 50

/-- The percentage of the population who are employed males -/
def employed_males_percentage : ℝ := 36

theorem employed_males_percentage_proof :
  employed_males_percentage = employed_percentage * (100 - female_employed_percentage) / 100 :=
by sorry

end NUMINAMATH_CALUDE_employed_males_percentage_proof_l2339_233906


namespace NUMINAMATH_CALUDE_not_divisible_by_169_l2339_233997

theorem not_divisible_by_169 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 7*n - 4 = 169*k := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_169_l2339_233997


namespace NUMINAMATH_CALUDE_smallest_fraction_above_five_sevenths_l2339_233931

theorem smallest_fraction_above_five_sevenths :
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →  -- a is a two-digit number
  10 ≤ b ∧ b ≤ 99 →  -- b is a two-digit number
  (5 : ℚ) / 7 < (a : ℚ) / b →  -- fraction is greater than 5/7
  (68 : ℚ) / 95 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_above_five_sevenths_l2339_233931


namespace NUMINAMATH_CALUDE_imperial_examination_middle_volume_l2339_233944

/-- The number of candidates admitted in the Middle volume given a total number of candidates and a proportion -/
def middle_volume_candidates (total : ℕ) (south north middle : ℕ) : ℕ :=
  total * middle /(south + north + middle)

/-- Theorem stating that given 100 total candidates and a proportion of 11:7:2,
    the number of candidates in the Middle volume is 10 -/
theorem imperial_examination_middle_volume :
  middle_volume_candidates 100 11 7 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_imperial_examination_middle_volume_l2339_233944


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_greater_than_one_l2339_233945

theorem intersection_nonempty_iff_a_greater_than_one (a : ℝ) :
  ({x : ℝ | x > 1} ∩ {x : ℝ | x ≤ a}).Nonempty ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_greater_than_one_l2339_233945


namespace NUMINAMATH_CALUDE_binders_for_1600_books_20_days_l2339_233921

/-- The number of binders required to bind a certain number of books in a given number of days -/
def binders_required (books : ℕ) (days : ℕ) : ℚ :=
  books / (days * (1400 / (30 * 21)))

theorem binders_for_1600_books_20_days :
  binders_required 1600 20 = 36 :=
sorry

end NUMINAMATH_CALUDE_binders_for_1600_books_20_days_l2339_233921


namespace NUMINAMATH_CALUDE_rented_cars_at_3600_max_revenue_l2339_233992

/-- Represents the rental company's car fleet and pricing model. -/
structure RentalCompany where
  total_cars : ℕ
  base_rent : ℕ
  rent_increment : ℕ
  rented_maintenance : ℕ
  unrented_maintenance : ℕ

/-- Calculates the number of rented cars given a certain rent. -/
def rented_cars (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.total_cars - (rent - company.base_rent) / company.rent_increment

/-- Calculates the monthly revenue given a certain rent. -/
def monthly_revenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  let rented := rented_cars company rent
  rent * rented - company.rented_maintenance * rented - 
    company.unrented_maintenance * (company.total_cars - rented)

/-- The rental company with the given parameters. -/
def our_company : RentalCompany := {
  total_cars := 100,
  base_rent := 3000,
  rent_increment := 50,
  rented_maintenance := 150,
  unrented_maintenance := 50
}

/-- Theorem stating the number of rented cars when rent is 3600 yuan. -/
theorem rented_cars_at_3600 : 
  rented_cars our_company 3600 = 88 := by sorry

/-- Theorem stating the rent that maximizes revenue and the maximum revenue. -/
theorem max_revenue : 
  ∃ (max_rent : ℕ), max_rent = 4050 ∧ 
  monthly_revenue our_company max_rent = 37050 ∧
  ∀ (rent : ℕ), monthly_revenue our_company rent ≤ monthly_revenue our_company max_rent := by sorry

end NUMINAMATH_CALUDE_rented_cars_at_3600_max_revenue_l2339_233992


namespace NUMINAMATH_CALUDE_watch_correction_theorem_l2339_233928

/-- Represents the time loss of a watch in minutes per day -/
def timeLossPerDay : ℚ := 13/4

/-- Represents the number of hours between 4 PM on March 21 and 12 PM on March 28 -/
def totalHours : ℕ := 7 * 24 + 20

/-- Calculates the positive correction in minutes needed for the watch -/
def positiveCorrection : ℚ :=
  (timeLossPerDay * (totalHours : ℚ)) / 24

theorem watch_correction_theorem :
  positiveCorrection = 25 + 17/96 := by sorry

end NUMINAMATH_CALUDE_watch_correction_theorem_l2339_233928


namespace NUMINAMATH_CALUDE_f_derivative_correct_l2339_233963

/-- The exponential function -/
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

/-- The function f(x) = e^(-2x) -/
noncomputable def f (x : ℝ) : ℝ := exp (-2 * x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := -2 * exp (-2 * x)

theorem f_derivative_correct :
  ∀ x : ℝ, deriv f x = f_derivative x :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_correct_l2339_233963


namespace NUMINAMATH_CALUDE_num_arrangements_l2339_233985

/-- Represents the number of volunteers --/
def num_volunteers : ℕ := 4

/-- Represents the number of communities --/
def num_communities : ℕ := 3

/-- Calculates the number of ways to arrange volunteers into communities --/
def arrange_volunteers : ℕ := sorry

/-- Theorem stating that the number of arrangements is 36 --/
theorem num_arrangements : arrange_volunteers = 36 := by sorry

end NUMINAMATH_CALUDE_num_arrangements_l2339_233985


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l2339_233925

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 5

/-- The number of gift card types -/
def gift_card_types : ℕ := 4

/-- The number of decorative bow types -/
def bow_types : ℕ := 2

/-- The total number of distinct gift-wrapping combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types * bow_types

theorem gift_wrapping_combinations :
  total_combinations = 400 :=
by sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l2339_233925


namespace NUMINAMATH_CALUDE_pink_roses_count_l2339_233982

theorem pink_roses_count (total_rows : ℕ) (roses_per_row : ℕ) 
  (red_fraction : ℚ) (white_fraction : ℚ) :
  total_rows = 10 →
  roses_per_row = 20 →
  red_fraction = 1/2 →
  white_fraction = 3/5 →
  (total_rows * roses_per_row * (1 - red_fraction) * (1 - white_fraction) : ℚ) = 40 :=
by sorry

end NUMINAMATH_CALUDE_pink_roses_count_l2339_233982


namespace NUMINAMATH_CALUDE_average_salary_l2339_233991

def salary_a : ℕ := 9000
def salary_b : ℕ := 5000
def salary_c : ℕ := 11000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

def total_salary : ℕ := salary_a + salary_b + salary_c + salary_d + salary_e
def num_people : ℕ := 5

theorem average_salary :
  (total_salary : ℚ) / num_people = 8200 := by sorry

end NUMINAMATH_CALUDE_average_salary_l2339_233991


namespace NUMINAMATH_CALUDE_deaf_students_count_l2339_233917

/-- Represents a school for deaf and blind students. -/
structure DeafBlindSchool where
  total_students : ℕ
  deaf_students : ℕ
  blind_students : ℕ
  deaf_triple_blind : deaf_students = 3 * blind_students
  total_sum : total_students = deaf_students + blind_students

/-- Theorem: In a school with 240 total students, where the number of deaf students
    is three times the number of blind students, the number of deaf students is 180. -/
theorem deaf_students_count (school : DeafBlindSchool) 
  (h_total : school.total_students = 240) : school.deaf_students = 180 := by
  sorry

end NUMINAMATH_CALUDE_deaf_students_count_l2339_233917


namespace NUMINAMATH_CALUDE_correct_mean_problem_l2339_233974

def correct_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean - wrong_value + correct_value) / n

theorem correct_mean_problem :
  correct_mean 50 41 23 48 = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_problem_l2339_233974


namespace NUMINAMATH_CALUDE_actual_height_is_191_l2339_233947

/-- Represents the height correction problem for a class of students. -/
structure HeightCorrectionProblem where
  num_students : ℕ
  initial_average : ℝ
  incorrect_height : ℝ
  actual_average : ℝ

/-- Calculates the actual height of the student with the incorrect measurement. -/
def calculate_actual_height (problem : HeightCorrectionProblem) : ℝ :=
  problem.num_students * (problem.initial_average - problem.actual_average) + problem.incorrect_height

/-- Theorem stating that the actual height of the student with the incorrect measurement is 191 cm. -/
theorem actual_height_is_191 (problem : HeightCorrectionProblem)
  (h1 : problem.num_students = 20)
  (h2 : problem.initial_average = 175)
  (h3 : problem.incorrect_height = 151)
  (h4 : problem.actual_average = 173) :
  calculate_actual_height problem = 191 := by
  sorry

end NUMINAMATH_CALUDE_actual_height_is_191_l2339_233947


namespace NUMINAMATH_CALUDE_not_right_triangle_6_7_8_l2339_233924

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- Theorem stating that 6, 7, and 8 cannot form a right triangle --/
theorem not_right_triangle_6_7_8 : ¬ is_right_triangle 6 7 8 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_6_7_8_l2339_233924


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2339_233994

/-- A quadratic function with vertex form (x + h)^2 + k -/
def quadratic_vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x + h)^2 + k

theorem quadratic_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∃ b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (f (-4) = 0 ∧ f 1 = -75) →
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2339_233994


namespace NUMINAMATH_CALUDE_no_triangle_cosine_sum_one_l2339_233915

theorem no_triangle_cosine_sum_one :
  ¬ ∃ (A B C : ℝ), 
    (0 < A ∧ A < π) ∧ 
    (0 < B ∧ B < π) ∧ 
    (0 < C ∧ C < π) ∧ 
    (A + B + C = π) ∧
    (Real.cos A + Real.cos B + Real.cos C = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_cosine_sum_one_l2339_233915


namespace NUMINAMATH_CALUDE_expression_evaluation_l2339_233967

theorem expression_evaluation : 3 - 5 * (6 - 2^3) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2339_233967


namespace NUMINAMATH_CALUDE_distance_to_sons_house_l2339_233913

/-- The distance to Jennie's son's house -/
def distance : ℝ := 200

/-- The travel time during heavy traffic (in hours) -/
def heavy_traffic_time : ℝ := 5

/-- The travel time with no traffic (in hours) -/
def no_traffic_time : ℝ := 4

/-- The difference in average speed between no traffic and heavy traffic conditions (in mph) -/
def speed_difference : ℝ := 10

/-- Theorem stating that the distance to Jennie's son's house is 200 miles -/
theorem distance_to_sons_house :
  distance = heavy_traffic_time * (distance / heavy_traffic_time) ∧
  distance = no_traffic_time * (distance / no_traffic_time) ∧
  distance / no_traffic_time = distance / heavy_traffic_time + speed_difference :=
by sorry

end NUMINAMATH_CALUDE_distance_to_sons_house_l2339_233913


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l2339_233916

/-- An equilateral hexagon with three nonadjacent 120° angles -/
structure SpecialHexagon where
  -- Side length
  s : ℝ
  -- Condition that s is positive
  s_pos : s > 0
  -- Area of the hexagon
  area : ℝ
  -- Condition that area is 12 square units
  area_eq : area = 12

/-- The perimeter of a SpecialHexagon is 24 units -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : 
  6 * h.s = 24 := by sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l2339_233916


namespace NUMINAMATH_CALUDE_share_ratio_l2339_233958

/-- Prove that the ratio of A's share to the combined share of B and C is 2:3 --/
theorem share_ratio (total a b c : ℚ) (x : ℚ) : 
  total = 200 →
  a = 80 →
  a = x * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a / (b + c) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_l2339_233958


namespace NUMINAMATH_CALUDE_find_n_l2339_233902

theorem find_n (e n : ℕ+) (h1 : Nat.lcm e n = 690) 
  (h2 : ¬ 3 ∣ n) (h3 : ¬ 2 ∣ e) : n = 230 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2339_233902


namespace NUMINAMATH_CALUDE_sum_interior_angles_is_3240_l2339_233941

/-- A regular polygon Q where each interior angle is 9 times its corresponding exterior angle -/
structure RegularPolygon where
  n : ℕ  -- number of sides
  interior_angle : ℝ  -- measure of each interior angle
  exterior_angle : ℝ  -- measure of each exterior angle
  is_regular : interior_angle = 9 * exterior_angle
  sum_exterior : n * exterior_angle = 360

/-- The sum of interior angles of a RegularPolygon -/
def sum_interior_angles (Q : RegularPolygon) : ℝ :=
  Q.n * Q.interior_angle

/-- Theorem: The sum of interior angles of a RegularPolygon is 3240° -/
theorem sum_interior_angles_is_3240 (Q : RegularPolygon) :
  sum_interior_angles Q = 3240 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_is_3240_l2339_233941


namespace NUMINAMATH_CALUDE_trap_is_feeder_l2339_233960

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An interval is a trap (cover) for a sequence if only finitely many terms lie outside it -/
def IsTrap (s : Sequence) (a b : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a ≤ s n ∧ s n ≤ b

/-- An interval is a feeder for a sequence if infinitely many terms lie inside it -/
def IsFeeder (s : Sequence) (a b : ℝ) : Prop :=
  ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a ≤ s n ∧ s n ≤ b

/-- Theorem: Every trap is a feeder -/
theorem trap_is_feeder (s : Sequence) (a b : ℝ) :
  IsTrap s a b → IsFeeder s a b := by
  sorry

end NUMINAMATH_CALUDE_trap_is_feeder_l2339_233960


namespace NUMINAMATH_CALUDE_students_taking_paper_c_l2339_233909

/-- Represents the systematic sampling setup for the school test -/
structure SchoolSampling where
  total_students : ℕ
  sample_size : ℕ
  first_selected : ℕ
  sampling_interval : ℕ

/-- Calculates the nth term in the arithmetic sequence of selected student numbers -/
def nth_selected (s : SchoolSampling) (n : ℕ) : ℕ :=
  s.first_selected + s.sampling_interval * (n - 1)

/-- Theorem stating the number of students taking test paper C -/
theorem students_taking_paper_c (s : SchoolSampling) 
  (h1 : s.total_students = 800)
  (h2 : s.sample_size = 40)
  (h3 : s.first_selected = 18)
  (h4 : s.sampling_interval = 20) :
  (Finset.filter (fun n => 561 ≤ nth_selected s n ∧ nth_selected s n ≤ 800) 
    (Finset.range s.sample_size)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_paper_c_l2339_233909


namespace NUMINAMATH_CALUDE_incorrect_locus_definition_l2339_233943

-- Define the type for points in our space
variable {X : Type*}

-- Define the locus as a set of points
variable (locus : Set X)

-- Define the condition as a predicate on points
variable (condition : X → Prop)

-- Statement to be proven incorrect
theorem incorrect_locus_definition :
  ¬(∀ x : X, condition x → x ∈ locus) ∧
  (∃ x : X, x ∈ locus ∧ condition x) →
  ¬(∀ x : X, x ∈ locus ↔ condition x) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_locus_definition_l2339_233943


namespace NUMINAMATH_CALUDE_michaels_money_ratio_l2339_233973

/-- Given the following conditions:
    - Michael has $42 initially
    - Michael gives some money to his brother
    - His brother buys $3 worth of candy
    - His brother has $35 left after buying candy
    - His brother had $17 at first
    Prove that the ratio of money Michael gave to his brother to Michael's initial money is 1:2 -/
theorem michaels_money_ratio :
  ∀ (initial_money : ℕ) (brother_initial : ℕ) (candy_cost : ℕ) (brother_final : ℕ),
    initial_money = 42 →
    brother_initial = 17 →
    candy_cost = 3 →
    brother_final = 35 →
    ∃ (money_given : ℕ),
      money_given = brother_final + candy_cost - brother_initial ∧
      2 * money_given = initial_money :=
by sorry

end NUMINAMATH_CALUDE_michaels_money_ratio_l2339_233973


namespace NUMINAMATH_CALUDE_parabola_equation_l2339_233951

/-- The equation of a parabola with focus (2, 0) and directrix x + 2 = 0 -/
theorem parabola_equation :
  ∀ (x y : ℝ),
    (∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y) →
    (∀ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y →
      (P.1 - 2)^2 + P.2^2 = (P.1 + 2)^2) ↔
    y^2 = 8*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2339_233951


namespace NUMINAMATH_CALUDE_different_color_probability_l2339_233993

/-- The probability of drawing two chips of different colors from a bag containing 
    7 blue chips, 5 yellow chips, and 4 red chips, when drawing with replacement. -/
theorem different_color_probability :
  let total_chips := 7 + 5 + 4
  let p_blue := 7 / total_chips
  let p_yellow := 5 / total_chips
  let p_red := 4 / total_chips
  let p_different := p_blue * (p_yellow + p_red) + 
                     p_yellow * (p_blue + p_red) + 
                     p_red * (p_blue + p_yellow)
  p_different = 83 / 128 :=
by sorry

end NUMINAMATH_CALUDE_different_color_probability_l2339_233993


namespace NUMINAMATH_CALUDE_complex_number_range_l2339_233990

theorem complex_number_range (m : ℝ) :
  let z : ℂ := 1 + Complex.I + m / (1 + Complex.I)
  (0 < z.re ∧ 0 < z.im) ↔ -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_range_l2339_233990


namespace NUMINAMATH_CALUDE_lori_marble_sharing_l2339_233914

theorem lori_marble_sharing :
  ∀ (total_marbles : ℕ) (share_percent : ℚ) (num_friends : ℕ),
    total_marbles = 60 →
    share_percent = 75 / 100 →
    num_friends = 5 →
    (total_marbles : ℚ) * share_percent / num_friends = 9 := by
  sorry

end NUMINAMATH_CALUDE_lori_marble_sharing_l2339_233914


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l2339_233962

/-- Given a parabola with equation y = 8x^2, its latus rectum has equation y = 1/32 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  y = 8 * x^2 → (∃ (x₀ : ℝ), y = 1/32 ∧ x₀ ≠ 0 ∧ y = 8 * x₀^2) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l2339_233962


namespace NUMINAMATH_CALUDE_shelter_animals_count_l2339_233977

/-- Calculates the total number of animals in a shelter given the initial conditions --/
def totalAnimals (initialCats : ℕ) : ℕ :=
  let adoptedCats := initialCats / 3
  let remainingCats := initialCats - adoptedCats
  let newCats := adoptedCats * 2
  let totalCats := remainingCats + newCats
  let dogs := totalCats * 2
  totalCats + dogs

/-- Theorem stating that given the initial conditions, the total number of animals is 60 --/
theorem shelter_animals_count : totalAnimals 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_shelter_animals_count_l2339_233977


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l2339_233981

/-- Represents a pentagon with side lengths and angles -/
structure Pentagon where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  de : ℝ
  ae : ℝ
  angle_bac : ℝ
  angle_abc : ℝ
  angle_bcd : ℝ
  angle_cde : ℝ
  angle_dea : ℝ

/-- Calculate the total cost of fencing a pentagon -/
def fencingCost (p : Pentagon) (costPerMeter : ℝ) : ℝ :=
  (p.ab + p.bc + p.cd + p.de + p.ae) * costPerMeter

/-- Theorem: The total cost of fencing the given irregular pentagon is Rs. 300 -/
theorem fencing_cost_theorem (p : Pentagon) (h1 : p.ab = 20)
    (h2 : p.bc = 25) (h3 : p.cd = 30) (h4 : p.de = 35) (h5 : p.ae = 40)
    (h6 : p.angle_bac = 110) (h7 : p.angle_abc = 95) (h8 : p.angle_bcd = 100)
    (h9 : p.angle_cde = 105) (h10 : p.angle_dea = 115) :
    fencingCost p 2 = 300 := by
  sorry

#check fencing_cost_theorem

end NUMINAMATH_CALUDE_fencing_cost_theorem_l2339_233981


namespace NUMINAMATH_CALUDE_train_length_l2339_233905

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : speed = 52 → time = 18 → ∃ length : ℝ, abs (length - 259.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2339_233905


namespace NUMINAMATH_CALUDE_expand_expression_l2339_233936

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2339_233936


namespace NUMINAMATH_CALUDE_subset_iff_a_in_range_l2339_233984

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 3*a^2 < 0}
def B : Set ℝ := {x | (x + 1)/(x - 2) < 0}

-- State the theorem
theorem subset_iff_a_in_range (a : ℝ) :
  A a ⊆ B ↔ -1/3 ≤ a ∧ a ≤ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_a_in_range_l2339_233984


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l2339_233900

open Real

theorem sin_2x_derivative (x : ℝ) : 
  deriv (λ x => sin (2 * x)) x = 2 * cos (2 * x) := by sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l2339_233900


namespace NUMINAMATH_CALUDE_wire_division_l2339_233972

/-- Given a wire of length 28 cm divided into quarters, prove that each quarter is 7 cm long. -/
theorem wire_division (wire_length : ℝ) (h : wire_length = 28) :
  wire_length / 4 = 7 := by
sorry

end NUMINAMATH_CALUDE_wire_division_l2339_233972


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2339_233969

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of (2,3) and (10,7) lies on the line y = mx + b
    y = m * x + b ∧ 
    x = (2 + 10) / 2 ∧ 
    y = (3 + 7) / 2 ∧
    -- The slope of the line is perpendicular to the slope of the segment connecting (2,3) and (10,7)
    m * ((7 - 3) / (10 - 2)) = -1) →
  m + b = 15 := by
sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l2339_233969


namespace NUMINAMATH_CALUDE_reciprocal_expression_l2339_233987

theorem reciprocal_expression (x y : ℝ) (h : x * y = 1) :
  (x + 1 / y) * (2 * y - 1 / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_expression_l2339_233987


namespace NUMINAMATH_CALUDE_hip_hop_class_cost_l2339_233980

/-- The cost of one hip-hop class -/
def hip_hop_cost : ℕ := sorry

/-- The cost of one ballet class -/
def ballet_cost : ℕ := 12

/-- The cost of one jazz class -/
def jazz_cost : ℕ := 8

/-- The total number of hip-hop classes per week -/
def hip_hop_classes : ℕ := 2

/-- The total number of ballet classes per week -/
def ballet_classes : ℕ := 2

/-- The total number of jazz classes per week -/
def jazz_classes : ℕ := 1

/-- The total cost of all classes per week -/
def total_cost : ℕ := 52

theorem hip_hop_class_cost :
  hip_hop_cost * hip_hop_classes + ballet_cost * ballet_classes + jazz_cost * jazz_classes = total_cost ∧
  hip_hop_cost = 10 :=
sorry

end NUMINAMATH_CALUDE_hip_hop_class_cost_l2339_233980


namespace NUMINAMATH_CALUDE_f_properties_l2339_233904

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

theorem f_properties :
  (∃ (x₀ : ℝ), IsLocalMax f x₀) ∧
  (¬ ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M) ∧
  (∀ (b : ℝ), (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = b ∧ f x₂ = b ∧ f x₃ = b) →
    (0 < b ∧ b < 6 * Real.exp (-3))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2339_233904


namespace NUMINAMATH_CALUDE_agathas_bike_frame_cost_l2339_233954

/-- Agatha's bike purchase problem -/
theorem agathas_bike_frame_cost (total : ℕ) (wheel_cost : ℕ) (remaining : ℕ) (frame_cost : ℕ) :
  total = 60 →
  wheel_cost = 25 →
  remaining = 20 →
  frame_cost = total - wheel_cost - remaining →
  frame_cost = 15 := by
sorry

end NUMINAMATH_CALUDE_agathas_bike_frame_cost_l2339_233954


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l2339_233923

/-- A function that checks if a number is a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number has its first two digits equal -/
def firstTwoDigitsEqual (n : ℕ) : Prop :=
  (n / 1000 = (n / 100) % 10)

/-- A function that checks if a number has its last two digits equal -/
def lastTwoDigitsEqual (n : ℕ) : Prop :=
  ((n / 10) % 10 = n % 10)

/-- The main theorem stating that 7744 is the only four-digit perfect square
    with equal first two digits and equal last two digits -/
theorem unique_four_digit_square :
  ∀ n : ℕ, isFourDigit n ∧ ∃ k : ℕ, n = k^2 ∧ firstTwoDigitsEqual n ∧ lastTwoDigitsEqual n
  ↔ n = 7744 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l2339_233923


namespace NUMINAMATH_CALUDE_two_angles_in_fourth_quadrant_l2339_233988

def is_fourth_quadrant (angle : Int) : Bool :=
  let normalized := angle % 360
  normalized > 270 || normalized ≤ 0

def count_fourth_quadrant (angles : List Int) : Nat :=
  (angles.filter is_fourth_quadrant).length

theorem two_angles_in_fourth_quadrant :
  count_fourth_quadrant [-20, -400, -2000, 1600] = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_angles_in_fourth_quadrant_l2339_233988


namespace NUMINAMATH_CALUDE_problem_solution_l2339_233983

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) :
  (x - 3)^4 + 81 / (x - 3)^4 = 63 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2339_233983


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2339_233995

theorem absolute_value_equality (x : ℝ) : 
  |x^2 - 8*x + 12| = x^2 - 8*x + 12 ↔ x ≤ 2 ∨ x ≥ 6 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2339_233995
