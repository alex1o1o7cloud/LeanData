import Mathlib

namespace NUMINAMATH_CALUDE_total_cost_ratio_l1498_149838

-- Define the cost of shorts
variable (x : ℝ)

-- Define the costs of other items based on the given conditions
def cost_tshirt : ℝ := x
def cost_boots : ℝ := 4 * x
def cost_shinguards : ℝ := 2 * x

-- State the theorem
theorem total_cost_ratio : 
  (x + cost_tshirt x + cost_boots x + cost_shinguards x) / x = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_ratio_l1498_149838


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l1498_149837

/-- Triangle XYZ with side lengths -/
structure Triangle where
  xy : ℝ
  yz : ℝ
  xz : ℝ

/-- Rectangle MNPQ inscribed in Triangle XYZ -/
structure InscribedRectangle where
  triangle : Triangle
  ω : ℝ  -- side length MN

/-- Area of rectangle MNPQ as a function of ω -/
def rectangleArea (rect : InscribedRectangle) : ℝ → ℝ :=
  fun ω => a * ω - b * ω^2
  where
    a : ℝ := sorry
    b : ℝ := sorry

/-- Theorem statement -/
theorem inscribed_rectangle_area_coefficient
  (t : Triangle)
  (h1 : t.xy = 15)
  (h2 : t.yz = 20)
  (h3 : t.xz = 13) :
  ∃ (rect : InscribedRectangle),
    rect.triangle = t ∧
    ∃ (a b : ℝ),
      (∀ ω, rectangleArea rect ω = a * ω - b * ω^2) ∧
      b = 9 / 25 :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l1498_149837


namespace NUMINAMATH_CALUDE_annie_aaron_visibility_time_l1498_149871

/-- The time (in minutes) Annie can see Aaron given their speeds and distances -/
theorem annie_aaron_visibility_time : 
  let annie_speed : ℝ := 10  -- Annie's speed in miles per hour
  let aaron_speed : ℝ := 6   -- Aaron's speed in miles per hour
  let initial_distance : ℝ := 1/4  -- Initial distance between Annie and Aaron in miles
  let final_distance : ℝ := 1/4   -- Final distance between Annie and Aaron in miles
  let relative_speed : ℝ := annie_speed - aaron_speed
  let time_hours : ℝ := (initial_distance + final_distance) / relative_speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 7.5
  := by sorry


end NUMINAMATH_CALUDE_annie_aaron_visibility_time_l1498_149871


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1498_149878

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 210)
  (rel_b : 5 * a = b - 11)
  (rel_c : 5 * a = c + 11) :
  a * b * c = 168504 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1498_149878


namespace NUMINAMATH_CALUDE_two_integers_sum_l1498_149868

theorem two_integers_sum (x y : ℕ+) : 
  x - y = 4 → x * y = 192 → x + y = 28 := by sorry

end NUMINAMATH_CALUDE_two_integers_sum_l1498_149868


namespace NUMINAMATH_CALUDE_power_equality_l1498_149886

theorem power_equality : (4 : ℝ) ^ 10 = 16 ^ 5 := by sorry

end NUMINAMATH_CALUDE_power_equality_l1498_149886


namespace NUMINAMATH_CALUDE_calculate_expression_l1498_149882

theorem calculate_expression : 150 * (150 - 5) - (150 * 150 - 7) = -743 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1498_149882


namespace NUMINAMATH_CALUDE_reader_group_size_l1498_149883

theorem reader_group_size (S L B : ℕ) (hS : S = 180) (hL : L = 88) (hB : B = 18) :
  S + L - B = 250 := by
  sorry

end NUMINAMATH_CALUDE_reader_group_size_l1498_149883


namespace NUMINAMATH_CALUDE_distance_implies_abs_x_l1498_149848

theorem distance_implies_abs_x (x : ℝ) :
  |((3 + x) - (3 - x))| = 8 → |x| = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_implies_abs_x_l1498_149848


namespace NUMINAMATH_CALUDE_weather_ratings_theorem_l1498_149872

-- Define the weather observation
structure WeatherObservation :=
  (morning : Bool)
  (afternoon : Bool)
  (evening : Bool)

-- Define the rating system for each child
def firstChildRating (w : WeatherObservation) : Bool :=
  ¬(w.morning ∨ w.afternoon ∨ w.evening)

def secondChildRating (w : WeatherObservation) : Bool :=
  ¬w.morning ∨ ¬w.afternoon ∨ ¬w.evening

-- Define the combined rating
def combinedRating (w : WeatherObservation) : Bool × Bool :=
  (firstChildRating w, secondChildRating w)

-- Define the set of all possible weather observations
def allWeatherObservations : Set WeatherObservation :=
  {w | w.morning = true ∨ w.morning = false ∧
       w.afternoon = true ∨ w.afternoon = false ∧
       w.evening = true ∨ w.evening = false}

-- Theorem statement
theorem weather_ratings_theorem :
  {(true, true), (true, false), (false, true), (false, false)} =
  {r | ∃ w ∈ allWeatherObservations, combinedRating w = r} :=
by sorry

end NUMINAMATH_CALUDE_weather_ratings_theorem_l1498_149872


namespace NUMINAMATH_CALUDE_inequality_solution_l1498_149825

theorem inequality_solution (a x : ℝ) : 
  a * x^2 - 2 ≥ 2 * x - a * x ↔ 
  (a = 0 ∧ x ≤ -1) ∨
  (a > 0 ∧ (x ≥ 2/a ∨ x ≤ -1)) ∨
  (-2 < a ∧ a < 0 ∧ 2/a ≤ x ∧ x ≤ -1) ∨
  (a = -2 ∧ x = -1) ∨
  (a < -2 ∧ -1 ≤ x ∧ x ≤ 2/a) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1498_149825


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1498_149897

theorem complex_magnitude_product : 
  Complex.abs ((5 * Real.sqrt 2 - 5 * Complex.I) * (2 * Real.sqrt 3 + 6 * Complex.I)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1498_149897


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1498_149827

theorem x_squared_plus_reciprocal (x : ℝ) (h : 54 = x^4 + 1/x^4) :
  x^2 + 1/x^2 = Real.sqrt 56 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1498_149827


namespace NUMINAMATH_CALUDE_salt_mixture_concentration_l1498_149892

/-- Given two salt solutions and their volumes, calculate the salt concentration of the mixture -/
theorem salt_mixture_concentration 
  (vol1 : ℝ) (conc1 : ℝ) (vol2 : ℝ) (conc2 : ℝ) 
  (h1 : vol1 = 600) 
  (h2 : conc1 = 0.03) 
  (h3 : vol2 = 400) 
  (h4 : conc2 = 0.12) 
  (h5 : vol1 + vol2 = 1000) :
  (vol1 * conc1 + vol2 * conc2) / (vol1 + vol2) = 0.066 := by
sorry

end NUMINAMATH_CALUDE_salt_mixture_concentration_l1498_149892


namespace NUMINAMATH_CALUDE_prism_volume_l1498_149846

/-- The volume of a right rectangular prism with face areas 30, 40, and 60 is 120√5 -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 40) (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1498_149846


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l1498_149887

-- Define Pascal's triangle
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

-- Theorem statement
theorem fifth_element_row_20 : pascal 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l1498_149887


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_six_l1498_149844

theorem sum_of_fractions_geq_six (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / y + y / z + z / x + x / z + z / y + y / x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_six_l1498_149844


namespace NUMINAMATH_CALUDE_problem_solution_l1498_149860

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : ((x * y) / 7)^3 = x^2) (h2 : ((x * y) / 7)^3 = y^3) : 
  x = 7 ∧ y = 7^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1498_149860


namespace NUMINAMATH_CALUDE_messages_sent_l1498_149816

theorem messages_sent (lucia_day1 : ℕ) : 
  lucia_day1 > 20 →
  let alina_day1 := lucia_day1 - 20
  let lucia_day2 := lucia_day1 / 3
  let alina_day2 := 2 * alina_day1
  let lucia_day3 := lucia_day1
  let alina_day3 := alina_day1
  lucia_day1 + alina_day1 + lucia_day2 + alina_day2 + lucia_day3 + alina_day3 = 680 →
  lucia_day1 = 120 := by
sorry

end NUMINAMATH_CALUDE_messages_sent_l1498_149816


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l1498_149804

/-- A trapezoid with a line joining the midpoints of its diagonals. -/
structure Trapezoid where
  /-- The length of the longer base of the trapezoid. -/
  longer_base : ℝ
  /-- The length of the shorter base of the trapezoid. -/
  shorter_base : ℝ
  /-- The length of the line joining the midpoints of the diagonals. -/
  midline_length : ℝ
  /-- The midline length is half the difference of the bases. -/
  midline_property : midline_length = (longer_base - shorter_base) / 2

/-- 
Given a trapezoid where the line joining the midpoints of the diagonals has length 5
and the longer base is 105, the shorter base has length 95.
-/
theorem trapezoid_shorter_base (t : Trapezoid) 
    (h1 : t.longer_base = 105)
    (h2 : t.midline_length = 5) : 
    t.shorter_base = 95 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_shorter_base_l1498_149804


namespace NUMINAMATH_CALUDE_function_inequality_implies_t_bound_l1498_149853

theorem function_inequality_implies_t_bound (t : ℝ) : 
  (∀ x : ℝ, (Real.exp (2 * x) - t) ≥ (t * Real.exp x - 1)) → 
  t ≤ 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_t_bound_l1498_149853


namespace NUMINAMATH_CALUDE_xy_sum_l1498_149885

theorem xy_sum (x y : ℤ) (h : 2*x*y + x + y = 83) : x + y = 83 ∨ x + y = -85 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_l1498_149885


namespace NUMINAMATH_CALUDE_triangle_problem_l1498_149821

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  a = 3 * Real.sqrt 2 →
  c = Real.sqrt 3 →
  Real.cos C = 2 * Real.sqrt 2 / 3 →
  b < a →
  -- Conclusion
  Real.sin A = Real.sqrt 6 / 3 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1498_149821


namespace NUMINAMATH_CALUDE_weekend_classes_count_l1498_149894

/-- The number of beginning diving classes offered on each day of the weekend -/
def weekend_classes : ℕ := 4

/-- The number of beginning diving classes offered on weekdays -/
def weekday_classes : ℕ := 2

/-- The number of people that can be accommodated in each class -/
def class_capacity : ℕ := 5

/-- The total number of people that can take classes in 3 weeks -/
def total_people : ℕ := 270

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- The number of weeks considered -/
def weeks : ℕ := 3

theorem weekend_classes_count :
  weekend_classes * class_capacity * weekend_days_per_week * weeks +
  weekday_classes * class_capacity * weekdays_per_week * weeks = total_people :=
by sorry

end NUMINAMATH_CALUDE_weekend_classes_count_l1498_149894


namespace NUMINAMATH_CALUDE_bacteria_growth_calculation_l1498_149863

/-- Given an original bacteria count and a current bacteria count, 
    calculate the increase in bacteria. -/
def bacteria_increase (original current : ℕ) : ℕ :=
  current - original

/-- Theorem stating that the increase in bacteria from 600 to 8917 is 8317. -/
theorem bacteria_growth_calculation :
  bacteria_increase 600 8917 = 8317 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_calculation_l1498_149863


namespace NUMINAMATH_CALUDE_prob_different_grades_is_four_fifths_l1498_149851

/-- Represents the number of students in each grade --/
structure GradeDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the probability of selecting two students from different grades --/
def probabilityDifferentGrades (dist : GradeDistribution) : ℚ :=
  4/5

/-- Theorem stating that the probability of selecting two students from different grades is 4/5 --/
theorem prob_different_grades_is_four_fifths (dist : GradeDistribution) 
  (h1 : dist.grade10 = 180)
  (h2 : dist.grade11 = 180)
  (h3 : dist.grade12 = 90) :
  probabilityDifferentGrades dist = 4/5 := by
  sorry

#check prob_different_grades_is_four_fifths

end NUMINAMATH_CALUDE_prob_different_grades_is_four_fifths_l1498_149851


namespace NUMINAMATH_CALUDE_tony_between_paul_and_rochelle_l1498_149877

-- Define the set of people
inductive Person : Type
  | Paul : Person
  | Quincy : Person
  | Rochelle : Person
  | Surinder : Person
  | Tony : Person

-- Define the seating arrangement as a function from Person to ℕ
def SeatingArrangement := Person → ℕ

-- Define the conditions of the seating arrangement
def ValidSeatingArrangement (s : SeatingArrangement) : Prop :=
  -- Condition 1: All seats are distinct
  (∀ p q : Person, p ≠ q → s p ≠ s q) ∧
  -- Condition 2: Seats are consecutive around a circular table
  (∀ p : Person, s p < 5) ∧
  -- Condition 3: Quincy sits between Paul and Surinder
  ((s Person.Quincy = (s Person.Paul + 1) % 5 ∧ s Person.Quincy = (s Person.Surinder + 4) % 5) ∨
   (s Person.Quincy = (s Person.Paul + 4) % 5 ∧ s Person.Quincy = (s Person.Surinder + 1) % 5)) ∧
  -- Condition 4: Tony is not beside Surinder
  (s Person.Tony ≠ (s Person.Surinder + 1) % 5 ∧ s Person.Tony ≠ (s Person.Surinder + 4) % 5)

-- Theorem: In any valid seating arrangement, Paul and Rochelle must be sitting on either side of Tony
theorem tony_between_paul_and_rochelle (s : SeatingArrangement) 
  (h : ValidSeatingArrangement s) : 
  (s Person.Tony = (s Person.Paul + 1) % 5 ∧ s Person.Tony = (s Person.Rochelle + 4) % 5) ∨
  (s Person.Tony = (s Person.Paul + 4) % 5 ∧ s Person.Tony = (s Person.Rochelle + 1) % 5) :=
sorry

end NUMINAMATH_CALUDE_tony_between_paul_and_rochelle_l1498_149877


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1498_149869

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x - a + 1 ≥ 0 ∧ 3 - 2*x > 0))) → 
  -1 < a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1498_149869


namespace NUMINAMATH_CALUDE_max_sin_a_given_condition_l1498_149855

open Real

theorem max_sin_a_given_condition (a b : ℝ) :
  cos (a + b) + sin (a - b) = cos a + cos b →
  ∃ (max_sin_a : ℝ), (∀ x, sin x ≤ max_sin_a) ∧ (max_sin_a = 1) :=
sorry

end NUMINAMATH_CALUDE_max_sin_a_given_condition_l1498_149855


namespace NUMINAMATH_CALUDE_initial_group_size_l1498_149852

/-- The number of initial persons in a group, given specific average age conditions. -/
theorem initial_group_size (initial_avg : ℝ) (new_persons : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 16 →
  new_persons = 12 →
  new_avg = 15 →
  final_avg = 15.5 →
  ∃ n : ℕ, n * initial_avg + new_persons * new_avg = (n + new_persons) * final_avg ∧ n = 12 :=
by sorry

end NUMINAMATH_CALUDE_initial_group_size_l1498_149852


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l1498_149811

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x^2 - 4 * x - 6)

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 1 ∧
    min = 1 / Real.exp 8 :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l1498_149811


namespace NUMINAMATH_CALUDE_vegetable_options_count_l1498_149889

/-- The number of cheese options available -/
def cheese_options : ℕ := 3

/-- The number of meat options available -/
def meat_options : ℕ := 4

/-- The total number of topping combinations -/
def total_combinations : ℕ := 57

/-- Calculates the number of topping combinations given the number of vegetable options -/
def calculate_combinations (veg_options : ℕ) : ℕ :=
  cheese_options * meat_options * veg_options - 
  cheese_options * (veg_options - 1) + 
  cheese_options

/-- Theorem stating that there are 5 vegetable options -/
theorem vegetable_options_count : 
  ∃ (veg_options : ℕ), veg_options = 5 ∧ calculate_combinations veg_options = total_combinations :=
sorry

end NUMINAMATH_CALUDE_vegetable_options_count_l1498_149889


namespace NUMINAMATH_CALUDE_flatrate_calculation_l1498_149823

/-- Represents the tutoring session details and pricing -/
structure TutoringSession where
  flatRate : ℕ
  perMinuteRate : ℕ
  durationMinutes : ℕ
  totalAmount : ℕ

/-- Theorem stating the flat rate for the given tutoring session -/
theorem flatrate_calculation (session : TutoringSession)
  (h1 : session.perMinuteRate = 7)
  (h2 : session.durationMinutes = 18)
  (h3 : session.totalAmount = 146)
  (h4 : session.totalAmount = session.flatRate + session.perMinuteRate * session.durationMinutes) :
  session.flatRate = 20 := by
  sorry

#check flatrate_calculation

end NUMINAMATH_CALUDE_flatrate_calculation_l1498_149823


namespace NUMINAMATH_CALUDE_center_coordinate_sum_l1498_149867

/-- Given two points that are endpoints of a diameter of a circle,
    prove that the sum of the coordinates of the center is -3. -/
theorem center_coordinate_sum (p1 p2 : ℝ × ℝ) : 
  p1 = (5, -7) → p2 = (-7, 3) → 
  (∃ (c : ℝ × ℝ), c.1 + c.2 = -3 ∧ 
    c = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_center_coordinate_sum_l1498_149867


namespace NUMINAMATH_CALUDE_fabric_C_required_is_120_l1498_149800

/-- Calculates the amount of fabric C required for pants production every week -/
def fabric_C_required (
  kingsley_pants_per_day : ℕ)
  (kingsley_work_days : ℕ)
  (fabric_C_per_pants : ℕ) : ℕ :=
  kingsley_pants_per_day * kingsley_work_days * fabric_C_per_pants

/-- Proves that the amount of fabric C required for pants production every week is 120 yards -/
theorem fabric_C_required_is_120 :
  fabric_C_required 4 6 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_fabric_C_required_is_120_l1498_149800


namespace NUMINAMATH_CALUDE_train_length_is_600_l1498_149896

/-- The length of the train in meters -/
def train_length : ℝ := 600

/-- The time it takes for the train to cross a tree, in seconds -/
def time_to_cross_tree : ℝ := 60

/-- The time it takes for the train to pass a platform, in seconds -/
def time_to_pass_platform : ℝ := 105

/-- The length of the platform, in meters -/
def platform_length : ℝ := 450

/-- Theorem stating that the train length is 600 meters -/
theorem train_length_is_600 :
  train_length = (time_to_pass_platform * platform_length) / (time_to_pass_platform - time_to_cross_tree) :=
by sorry

end NUMINAMATH_CALUDE_train_length_is_600_l1498_149896


namespace NUMINAMATH_CALUDE_locus_is_circle_l1498_149898

/-- An isosceles triangle with side length s and base b -/
structure IsoscelesTriangle where
  s : ℝ
  b : ℝ
  s_pos : 0 < s
  b_pos : 0 < b
  triangle_ineq : b < 2 * s

/-- The locus of points P such that the sum of distances from P to the vertices equals a -/
def Locus (t : IsoscelesTriangle) (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧
    Real.sqrt (x^2 + y^2) +
    Real.sqrt ((x - t.b)^2 + y^2) +
    Real.sqrt ((x - t.b/2)^2 + (y - Real.sqrt (t.s^2 - (t.b/2)^2))^2) = a}

/-- The theorem stating that the locus is a circle if and only if a > 2s + b -/
theorem locus_is_circle (t : IsoscelesTriangle) (a : ℝ) :
  (∃ (c : ℝ × ℝ) (r : ℝ), Locus t a = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}) ↔
  a > 2 * t.s + t.b := by
  sorry

end NUMINAMATH_CALUDE_locus_is_circle_l1498_149898


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1498_149810

/-- The equation of an ellipse with foci at (4,0) and (-4,0) -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + y^2) + Real.sqrt ((x + 4)^2 + y^2) = 10

/-- The simplified equation of the ellipse -/
def simplified_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

/-- Theorem stating the equivalence of the two equations -/
theorem ellipse_equation_equivalence :
  ∀ x y : ℝ, ellipse_equation x y ↔ simplified_equation x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1498_149810


namespace NUMINAMATH_CALUDE_smallest_N_with_301_l1498_149864

/-- The function that generates the concatenated string for a given N -/
def generateString (N : ℕ) : String := sorry

/-- The predicate that checks if "301" appears in a string -/
def contains301 (s : String) : Prop := sorry

/-- The theorem stating that 38 is the smallest N that satisfies the condition -/
theorem smallest_N_with_301 : 
  (∀ n < 38, ¬ contains301 (generateString n)) ∧ 
  contains301 (generateString 38) := by sorry

end NUMINAMATH_CALUDE_smallest_N_with_301_l1498_149864


namespace NUMINAMATH_CALUDE_gcd_360_150_l1498_149862

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_150_l1498_149862


namespace NUMINAMATH_CALUDE_rectangle_area_function_l1498_149854

/-- For a rectangle with area 10 and adjacent sides x and y, prove that y = 10/x --/
theorem rectangle_area_function (x y : ℝ) (h : x * y = 10) : y = 10 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_function_l1498_149854


namespace NUMINAMATH_CALUDE_binomial_12_6_l1498_149836

theorem binomial_12_6 : Nat.choose 12 6 = 924 := by sorry

end NUMINAMATH_CALUDE_binomial_12_6_l1498_149836


namespace NUMINAMATH_CALUDE_crayons_count_l1498_149839

/-- The number of rows of crayons --/
def num_rows : ℕ := 7

/-- The number of crayons in each row --/
def crayons_per_row : ℕ := 30

/-- The total number of crayons --/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem crayons_count : total_crayons = 210 := by
  sorry

end NUMINAMATH_CALUDE_crayons_count_l1498_149839


namespace NUMINAMATH_CALUDE_system_solution_l1498_149879

theorem system_solution : 
  ∃! (x y : ℚ), (2010 * x - 2011 * y = 2009) ∧ (2009 * x - 2008 * y = 2010) ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1498_149879


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1498_149819

/-- Definition of the binary operation ◇ -/
noncomputable def diamond (k : ℝ) (a b : ℝ) : ℝ :=
  k / b

/-- Theorem stating the solution to the equation -/
theorem diamond_equation_solution (k : ℝ) (h1 : k = 2) :
  ∃ x : ℝ, diamond k 2023 (diamond k 7 x) = 150 ∧ x = 150 / 2023 := by
  sorry

/-- Properties of the binary operation ◇ -/
axiom diamond_assoc (k a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond k a (diamond k b c) = k * (diamond k a b) * c

axiom diamond_self (k a : ℝ) (ha : a ≠ 0) :
  diamond k a a = k

end NUMINAMATH_CALUDE_diamond_equation_solution_l1498_149819


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1498_149874

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f 1 = 2 → f (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1498_149874


namespace NUMINAMATH_CALUDE_selling_price_is_200_l1498_149866

/-- Calculates the selling price per acre given the initial purchase details and profit --/
def selling_price_per_acre (total_acres : ℕ) (purchase_price_per_acre : ℕ) (profit : ℕ) : ℕ :=
  let total_cost := total_acres * purchase_price_per_acre
  let acres_sold := total_acres / 2
  let total_revenue := total_cost + profit
  total_revenue / acres_sold

/-- Proves that the selling price per acre is $200 given the problem conditions --/
theorem selling_price_is_200 :
  selling_price_per_acre 200 70 6000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_200_l1498_149866


namespace NUMINAMATH_CALUDE_cattle_milk_production_l1498_149802

/-- Represents the total milk production of a group of cows over a given number of days -/
def total_milk_production (num_cows : ℕ) (milk_per_cow : ℕ) (num_days : ℕ) : ℕ :=
  num_cows * milk_per_cow * num_days

/-- Proves that the total milk production of 150 cows over 12 days is 2655000 oz -/
theorem cattle_milk_production : 
  let total_cows : ℕ := 150
  let group1_cows : ℕ := 75
  let group2_cows : ℕ := 75
  let group1_milk_per_cow : ℕ := 1300
  let group2_milk_per_cow : ℕ := 1650
  let num_days : ℕ := 12
  total_milk_production group1_cows group1_milk_per_cow num_days + 
  total_milk_production group2_cows group2_milk_per_cow num_days = 2655000 :=
by
  sorry

end NUMINAMATH_CALUDE_cattle_milk_production_l1498_149802


namespace NUMINAMATH_CALUDE_wedding_ring_cost_l1498_149833

/-- Proves that the cost of the first wedding ring is $10,000 given the problem conditions --/
theorem wedding_ring_cost (first_ring_cost : ℝ) : 
  (3 * first_ring_cost - first_ring_cost / 2 = 25000) → 
  first_ring_cost = 10000 := by
  sorry

#check wedding_ring_cost

end NUMINAMATH_CALUDE_wedding_ring_cost_l1498_149833


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l1498_149828

/-- Represents the number of students in each grade --/
structure GradePopulation where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Represents the number of students to be sampled from each grade --/
structure SampleSize where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the stratified sample size for each grade --/
def stratifiedSample (pop : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPop := pop.grade10 + pop.grade11 + pop.grade12
  { grade10 := totalSample * pop.grade10 / totalPop,
    grade11 := totalSample * pop.grade11 / totalPop,
    grade12 := totalSample * pop.grade12 / totalPop }

/-- Theorem: The stratified sample for the given population and sample size is correct --/
theorem correct_stratified_sample :
  let pop := GradePopulation.mk 600 800 400
  let sample := stratifiedSample pop 18
  sample = SampleSize.mk 6 8 4 := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l1498_149828


namespace NUMINAMATH_CALUDE_A_when_half_in_A_B_values_l1498_149880

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + 1 = 0}

theorem A_when_half_in_A (a : ℝ) (h : (1/2 : ℝ) ∈ A a) : 
  A a = {-(1/4), 1/2} := by sorry

def B : Set ℝ := {a : ℝ | ∃! x, x ∈ A a}

theorem B_values : B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_when_half_in_A_B_values_l1498_149880


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1498_149826

/-- Calculates the percentage reduction in oil price given the original amount, additional amount, total cost, and reduced price. -/
theorem oil_price_reduction 
  (X : ℝ)              -- Original amount of oil in kg
  (additional : ℝ)     -- Additional amount of oil in kg
  (total_cost : ℝ)     -- Total cost in Rs
  (reduced_price : ℝ)  -- Reduced price per kg in Rs
  (h1 : additional = 5)
  (h2 : total_cost = 600)
  (h3 : reduced_price = 30)
  (h4 : X + additional = total_cost / reduced_price)
  (h5 : X = total_cost / (total_cost / X))
  : (1 - reduced_price / (total_cost / X)) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_oil_price_reduction_l1498_149826


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1498_149859

/-- A circle with two parallel tangents and a third connecting tangent -/
structure TangentCircle where
  -- Radius of the circle
  r : ℝ
  -- Length of the first parallel tangent
  ab : ℝ
  -- Length of the second parallel tangent
  cd : ℝ
  -- Length of the connecting tangent
  ef : ℝ
  -- Condition that ab and cd are parallel tangents
  h_parallel : ab < cd
  -- Condition that ef is a tangent connecting ab and cd
  h_connecting : ef > ab ∧ ef < cd

/-- The theorem stating that for the given configuration, the radius is 2.5 -/
theorem tangent_circle_radius (c : TangentCircle)
    (h_ab : c.ab = 5)
    (h_cd : c.cd = 11)
    (h_ef : c.ef = 15) :
    c.r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l1498_149859


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1498_149858

/-- The asymptotic lines of a hyperbola with equation x^2 - y^2/9 = 1 are y = ±3x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 - y^2/9 = 1) → (∃ k : ℝ, k = 3 ∨ k = -3) → (y = k*x) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1498_149858


namespace NUMINAMATH_CALUDE_system_solution_l1498_149865

theorem system_solution :
  let S := {(x, y, z, t) : ℕ × ℕ × ℕ × ℕ | 
    x + y + z + t = 5 ∧ 
    x + 2*y + 5*z + 10*t = 17}
  S = {(1, 3, 0, 1), (2, 0, 3, 0)} := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1498_149865


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l1498_149847

theorem proportion_fourth_term (x y : ℝ) : 
  (0.6 : ℝ) / x = 5 / y → x = 0.96 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l1498_149847


namespace NUMINAMATH_CALUDE_hyperbola_symmetric_points_parabola_midpoint_l1498_149893

/-- Given a hyperbola, two symmetric points on it, and their midpoint on a parabola, prove the possible values of m -/
theorem hyperbola_symmetric_points_parabola_midpoint (m : ℝ) : 
  (∃ (M N : ℝ × ℝ),
    -- M and N are on the hyperbola
    (M.1^2 - M.2^2/3 = 1) ∧ (N.1^2 - N.2^2/3 = 1) ∧
    -- M and N are symmetric about y = x + m
    (M.2 + N.2 = M.1 + N.1 + 2*m) ∧
    -- The midpoint of MN is on the parabola y^2 = 18x
    (((M.2 + N.2)/2)^2 = 18 * ((M.1 + N.1)/2))) →
  (m = 0 ∨ m = -8) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_symmetric_points_parabola_midpoint_l1498_149893


namespace NUMINAMATH_CALUDE_gina_textbooks_l1498_149890

/-- Calculates the number of textbooks Gina needs to buy given her college expenses. -/
def calculate_textbooks (credits : ℕ) (credit_cost : ℕ) (facilities_fee : ℕ) (textbook_cost : ℕ) (total_spending : ℕ) : ℕ :=
  let credit_total := credits * credit_cost
  let non_textbook_cost := credit_total + facilities_fee
  let textbook_budget := total_spending - non_textbook_cost
  textbook_budget / textbook_cost

theorem gina_textbooks :
  calculate_textbooks 14 450 200 120 7100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gina_textbooks_l1498_149890


namespace NUMINAMATH_CALUDE_tan_zero_degrees_l1498_149875

theorem tan_zero_degrees : Real.tan 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_zero_degrees_l1498_149875


namespace NUMINAMATH_CALUDE_unique_prime_factorization_and_sum_l1498_149822

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_prime_factorization_and_sum (q r s p1 p2 p3 : ℕ) : 
  (q * r * s = 2206 ∧ 
   is_prime q ∧ is_prime r ∧ is_prime s ∧ 
   q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  (p1 + p2 + p3 = q + r + s + 1 ∧ 
   is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ 
   p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) →
  ((q = 2 ∧ r = 3 ∧ s = 367) ∨ (q = 2 ∧ r = 367 ∧ s = 3) ∨ (q = 3 ∧ r = 2 ∧ s = 367) ∨ 
   (q = 3 ∧ r = 367 ∧ s = 2) ∧ (q = 367 ∧ r = 2 ∧ s = 3) ∨ (q = 367 ∧ r = 3 ∧ s = 2)) ∧
  (p1 = 2 ∧ p2 = 3 ∧ p3 = 367) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_factorization_and_sum_l1498_149822


namespace NUMINAMATH_CALUDE_sum_of_digits_of_k_l1498_149801

def k : ℕ := 10^30 - 54

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_k : sum_of_digits k = 11 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_k_l1498_149801


namespace NUMINAMATH_CALUDE_solution_existence_l1498_149850

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The system of equations has a solution -/
def has_solution (K : ℤ) : Prop :=
  ∃ (x y : ℝ), (2 * (floor x) + y = 3/2) ∧ ((floor x - x)^2 - 2 * (floor y) = K)

/-- The theorem stating the conditions for the existence of a solution -/
theorem solution_existence (K : ℤ) :
  has_solution K ↔ ∃ (M : ℤ), K = 4*M - 2 ∧ has_solution (4*M - 2) :=
sorry

end NUMINAMATH_CALUDE_solution_existence_l1498_149850


namespace NUMINAMATH_CALUDE_problem_solution_l1498_149861

def p (m : ℝ) : Prop := ∀ x, 2*x - 5 > 0 → x > m

def q (m : ℝ) : Prop := ∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ 
  ∀ x y, x^2/(m-1) + y^2/(2-m) = 1 ↔ (x/a)^2 - (y/b)^2 = 1

theorem problem_solution (m : ℝ) : 
  (p m ∧ q m → m < 1 ∨ (2 < m ∧ m ≤ 5/2)) ∧
  (¬(p m ∧ q m) ∧ (p m ∨ q m) → (1 ≤ m ∧ m ≤ 2) ∨ m > 5/2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1498_149861


namespace NUMINAMATH_CALUDE_divisibility_implies_divisibility_l1498_149870

theorem divisibility_implies_divisibility (a b m n : ℕ) 
  (ha : a > 1) (hcoprime : Nat.Coprime a b) :
  (((a^m + 1) ∣ (a^n + 1)) → (m ∣ n)) ∧
  (((a^m + b^m) ∣ (a^n + b^n)) → (m ∣ n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_divisibility_l1498_149870


namespace NUMINAMATH_CALUDE_largest_increase_2018_2019_l1498_149829

def students : Fin 6 → ℕ
  | 0 => 110  -- 2015
  | 1 => 125  -- 2016
  | 2 => 130  -- 2017
  | 3 => 140  -- 2018
  | 4 => 160  -- 2019
  | 5 => 165  -- 2020

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseYears : Fin 5 := sorry

theorem largest_increase_2018_2019 :
  largestIncreaseYears = 3 ∧
  ∀ i : Fin 5, percentageIncrease (students i) (students (i + 1)) ≤
    percentageIncrease (students 3) (students 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2018_2019_l1498_149829


namespace NUMINAMATH_CALUDE_correct_number_probability_l1498_149891

-- Define the number of options for the first three digits
def first_three_options : ℕ := 3

-- Define the number of digits used in the last five digits
def last_five_digits : ℕ := 5

-- Theorem statement
theorem correct_number_probability :
  (1 : ℚ) / (first_three_options * Nat.factorial last_five_digits) = (1 : ℚ) / 360 :=
by sorry

end NUMINAMATH_CALUDE_correct_number_probability_l1498_149891


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l1498_149815

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder structure

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l₁ l₂ : Line3D) : Prop :=
  sorry

/-- Two lines do not intersect if they have no point in common -/
def do_not_intersect (l₁ l₂ : Line3D) : Prop :=
  sorry

theorem skew_lines_sufficient_not_necessary :
  (∀ l₁ l₂ : Line3D, are_skew l₁ l₂ → do_not_intersect l₁ l₂) ∧
  (∃ l₁ l₂ : Line3D, do_not_intersect l₁ l₂ ∧ ¬are_skew l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l1498_149815


namespace NUMINAMATH_CALUDE_current_speed_l1498_149843

/-- Proves that the speed of the current is approximately 3 km/hr given the conditions -/
theorem current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) :
  rowing_speed = 6 →
  distance = 80 →
  time = 31.99744020478362 →
  ∃ (current_speed : ℝ), 
    (abs (current_speed - 3) < 0.001) ∧ 
    (distance / time = rowing_speed / 3.6 + current_speed / 3.6) := by
  sorry


end NUMINAMATH_CALUDE_current_speed_l1498_149843


namespace NUMINAMATH_CALUDE_expression_evaluation_l1498_149842

theorem expression_evaluation : (980^2 : ℚ) / (210^2 - 206^2) = 577.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1498_149842


namespace NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l1498_149814

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 20, when divided by π, is equal to 1125√7. -/
theorem cone_volume_divided_by_pi (r l : Real) (h : Real) : 
  r = 15 → l = 20 → h = 5 * Real.sqrt 7 → (1/3 * π * r^2 * h) / π = 1125 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l1498_149814


namespace NUMINAMATH_CALUDE_circle_radius_l1498_149813

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 + 36 = 6*x + 24*y) → 
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = Real.sqrt 117 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1498_149813


namespace NUMINAMATH_CALUDE_gcd_of_136_and_1275_l1498_149834

theorem gcd_of_136_and_1275 : Nat.gcd 136 1275 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_136_and_1275_l1498_149834


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_l1498_149845

/-- The prime counting function -/
def prime_counting (n : ℕ) : ℕ := sorry

/-- π(n) is non-decreasing -/
axiom prime_counting_nondecreasing : ∀ m n : ℕ, m ≤ n → prime_counting m ≤ prime_counting n

/-- The set of integers n such that π(n) divides n -/
def divisible_set : Set ℕ := {n : ℕ | prime_counting n ∣ n}

/-- There are infinitely many integers n such that π(n) divides n -/
theorem infinitely_many_divisible : Set.Infinite divisible_set := by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_l1498_149845


namespace NUMINAMATH_CALUDE_impossible_time_reduction_l1498_149817

/-- Proves that it's impossible to reduce the time per kilometer by 1 minute when starting from a speed of 60 km/h. -/
theorem impossible_time_reduction (initial_speed : ℝ) (time_reduction : ℝ) : 
  initial_speed = 60 → time_reduction = 1 → ¬ (∃ (new_speed : ℝ), new_speed > 0 ∧ (1 / new_speed) * 60 = (1 / initial_speed) * 60 - time_reduction) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_time_reduction_l1498_149817


namespace NUMINAMATH_CALUDE_bridge_toll_base_cost_l1498_149820

/-- Represents the toll calculation for a bridge -/
structure BridgeToll where
  base_cost : ℝ
  axle_cost : ℝ

/-- Calculates the toll for a given number of axles -/
def calc_toll (bt : BridgeToll) (axles : ℕ) : ℝ :=
  bt.base_cost + bt.axle_cost * (axles - 2)

/-- Represents a truck with a specific number of wheels and axles -/
structure Truck where
  total_wheels : ℕ
  front_axle_wheels : ℕ
  other_axle_wheels : ℕ

/-- Calculates the number of axles for a truck -/
def calc_axles (t : Truck) : ℕ :=
  1 + (t.total_wheels - t.front_axle_wheels) / t.other_axle_wheels

theorem bridge_toll_base_cost :
  ∃ (bt : BridgeToll),
    bt.axle_cost = 0.5 ∧
    let truck := Truck.mk 18 2 4
    let axles := calc_axles truck
    calc_toll bt axles = 5 ∧
    bt.base_cost = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_bridge_toll_base_cost_l1498_149820


namespace NUMINAMATH_CALUDE_sandwich_contest_difference_l1498_149807

theorem sandwich_contest_difference : (5 : ℚ) / 6 - (2 : ℚ) / 3 = (1 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_sandwich_contest_difference_l1498_149807


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1498_149808

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) :
  x + 2 * y ≥ 1 / 2 + Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    1 / (2 * x₀ + y₀) + 1 / (y₀ + 1) = 1 ∧
    x₀ + 2 * y₀ = 1 / 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1498_149808


namespace NUMINAMATH_CALUDE_taylor_series_expansion_of_f_l1498_149830

def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 2

def taylor_expansion (x : ℝ) : ℝ := -12 + 16*(x + 1) - 7*(x + 1)^2 + (x + 1)^3

theorem taylor_series_expansion_of_f :
  ∀ x : ℝ, f x = taylor_expansion x := by
  sorry

end NUMINAMATH_CALUDE_taylor_series_expansion_of_f_l1498_149830


namespace NUMINAMATH_CALUDE_square_minus_self_divisible_by_two_l1498_149840

theorem square_minus_self_divisible_by_two (n : ℕ) : 
  2 ∣ (n^2 - n) := by sorry

end NUMINAMATH_CALUDE_square_minus_self_divisible_by_two_l1498_149840


namespace NUMINAMATH_CALUDE_oatmeal_cookies_divisible_by_containers_l1498_149895

/-- The number of chocolate chip cookies Kiara baked -/
def chocolate_chip_cookies : ℕ := 48

/-- The number of containers Kiara wants to use -/
def num_containers : ℕ := 6

/-- The number of oatmeal cookies Kiara baked -/
def oatmeal_cookies : ℕ := sorry

/-- Theorem stating that the number of oatmeal cookies must be divisible by the number of containers -/
theorem oatmeal_cookies_divisible_by_containers :
  oatmeal_cookies % num_containers = 0 :=
sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_divisible_by_containers_l1498_149895


namespace NUMINAMATH_CALUDE_yoki_cans_collected_l1498_149824

/-- Given the conditions of the can collection problem, prove that Yoki picked up 9 cans. -/
theorem yoki_cans_collected (total_cans ladonna_cans prikya_cans avi_cans yoki_cans : ℕ) : 
  total_cans = 85 →
  ladonna_cans = 25 →
  prikya_cans = 2 * ladonna_cans - 3 →
  avi_cans = 8 / 2 →
  yoki_cans = total_cans - (ladonna_cans + prikya_cans + avi_cans) →
  yoki_cans = 9 := by
sorry

end NUMINAMATH_CALUDE_yoki_cans_collected_l1498_149824


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_800_l1498_149818

def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def isDistinctPowerSum (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.length ≥ 2) ∧
  (powers.sum = n) ∧
  (∀ p ∈ powers, isPowerOfTwo p) ∧
  powers.Nodup

theorem least_exponent_sum_for_800 :
  ∃ (powers : List ℕ),
    isDistinctPowerSum 800 powers ∧
    (∀ (other_powers : List ℕ),
      isDistinctPowerSum 800 other_powers →
      (powers.map (fun p => (Nat.log p 2))).sum ≤ (other_powers.map (fun p => (Nat.log p 2))).sum) ∧
    (powers.map (fun p => (Nat.log p 2))).sum = 22 :=
sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_800_l1498_149818


namespace NUMINAMATH_CALUDE_minimum_average_score_for_target_l1498_149812

def current_scores : List ℝ := [92, 81, 75, 65, 88]
def bonus_points : ℝ := 5
def target_increase : ℝ := 6

theorem minimum_average_score_for_target (new_test1 new_test2 : ℝ) :
  let current_avg := (current_scores.sum) / current_scores.length
  let new_avg := ((current_scores.sum + (new_test1 + bonus_points) + new_test2) / 
                  (current_scores.length + 2))
  let min_new_avg := (new_test1 + new_test2) / 2
  (new_avg = current_avg + target_increase) → min_new_avg ≥ 99 := by
  sorry

end NUMINAMATH_CALUDE_minimum_average_score_for_target_l1498_149812


namespace NUMINAMATH_CALUDE_travel_time_ratio_l1498_149849

/-- Proves that the ratio of the time taken to travel a fixed distance at a given speed
    to the time taken to travel the same distance in a given time is equal to a specific ratio. -/
theorem travel_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 360 ∧ original_time = 6 ∧ new_speed = 40 →
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_ratio_l1498_149849


namespace NUMINAMATH_CALUDE_product_w_z_l1498_149835

/-- A parallelogram with side lengths defined in terms of w and z -/
structure Parallelogram (w z : ℝ) :=
  (ef : ℝ)
  (fg : ℝ)
  (gh : ℝ)
  (he : ℝ)
  (ef_eq : ef = 50)
  (fg_eq : fg = 4 * z^2)
  (gh_eq : gh = 3 * w + 6)
  (he_eq : he = 32)
  (opposite_sides_equal : ef = gh ∧ fg = he)

/-- The product of w and z in the given parallelogram is 88√2/3 -/
theorem product_w_z (w z : ℝ) (p : Parallelogram w z) : w * z = 88 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_w_z_l1498_149835


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l1498_149888

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) : 
  initial_volume = 6 → 
  initial_percentage = 0.3 → 
  added_alcohol = 2.4 → 
  let final_volume := initial_volume + added_alcohol
  let final_alcohol := initial_volume * initial_percentage + added_alcohol
  final_alcohol / final_volume = 0.5 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l1498_149888


namespace NUMINAMATH_CALUDE_complex_equality_l1498_149806

/-- Given a complex number z = 1-ni, prove that m+ni = 2-i -/
theorem complex_equality (m n : ℝ) (z : ℂ) (h : z = 1 - n * Complex.I) :
  m + n * Complex.I = 2 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_equality_l1498_149806


namespace NUMINAMATH_CALUDE_parabola_vertex_l1498_149805

/-- The vertex of the parabola y = 3(x+1)^2 + 4 is (-1, 4) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 3 * (x + 1)^2 + 4 → (∃ h k : ℝ, h = -1 ∧ k = 4 ∧ ∀ x y : ℝ, y = 3 * (x - h)^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1498_149805


namespace NUMINAMATH_CALUDE_pairwise_ratio_sum_bound_l1498_149876

theorem pairwise_ratio_sum_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_pairwise_ratio_sum_bound_l1498_149876


namespace NUMINAMATH_CALUDE_jim_gave_eight_sets_to_brother_l1498_149857

/-- The number of trading cards in one set -/
def cards_per_set : ℕ := 13

/-- The number of sets Jim gave to his sister -/
def sets_to_sister : ℕ := 5

/-- The number of sets Jim gave to his friend -/
def sets_to_friend : ℕ := 2

/-- The total number of trading cards Jim had initially -/
def initial_cards : ℕ := 365

/-- The total number of trading cards Jim gave away -/
def total_given_away : ℕ := 195

/-- The number of sets Jim gave to his brother -/
def sets_to_brother : ℕ := (total_given_away - (sets_to_sister + sets_to_friend) * cards_per_set) / cards_per_set

theorem jim_gave_eight_sets_to_brother : sets_to_brother = 8 := by
  sorry

end NUMINAMATH_CALUDE_jim_gave_eight_sets_to_brother_l1498_149857


namespace NUMINAMATH_CALUDE_original_number_is_27_l1498_149884

theorem original_number_is_27 :
  ∃ (n : ℕ), 
    (Odd (3 * n)) ∧ 
    (∃ (k : ℕ), k > 1 ∧ (3 * n) % k = 0) ∧ 
    (4 * n = 108) ∧
    n = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_27_l1498_149884


namespace NUMINAMATH_CALUDE_total_profit_calculation_l1498_149831

theorem total_profit_calculation (x_investment y_investment z_investment : ℕ)
  (x_months y_months z_months : ℕ) (z_profit : ℕ) :
  x_investment = 36000 →
  y_investment = 42000 →
  z_investment = 48000 →
  x_months = 12 →
  y_months = 12 →
  z_months = 8 →
  z_profit = 4096 →
  (z_investment * z_months * 14080 = z_profit * (x_investment * x_months + y_investment * y_months + z_investment * z_months)) :=
by
  sorry

#check total_profit_calculation

end NUMINAMATH_CALUDE_total_profit_calculation_l1498_149831


namespace NUMINAMATH_CALUDE_task_completion_probability_l1498_149899

theorem task_completion_probability (p1 p2 p3 : ℚ) 
  (h1 : p1 = 2/3) (h2 : p2 = 3/5) (h3 : p3 = 4/7) :
  p1 * (1 - p2) * p3 = 16/105 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l1498_149899


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1498_149809

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 3 + a 4 + a 5 = 3)
    (h_a8 : a 8 = 8) : 
  a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1498_149809


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1498_149881

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 3 = 0) → (x₂^2 - 2*x₂ - 3 = 0) → (x₁ + x₂ = 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1498_149881


namespace NUMINAMATH_CALUDE_max_pairs_sum_l1498_149856

theorem max_pairs_sum (n : ℕ) (h : n = 3011) : 
  (∃ (k : ℕ) (pairs : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    pairs.length = k) ∧
  (∀ (m : ℕ) (pairs : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
    pairs.length = m →
    m ≤ k) →
  k = 1204 := by
sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l1498_149856


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l1498_149873

/-- Given two employees with a total pay of 570 and one paid 150% of the other,
    prove that the lower-paid employee receives 228. -/
theorem employee_pay_calculation (total_pay : ℝ) (ratio : ℝ) :
  total_pay = 570 →
  ratio = 1.5 →
  ∃ (low_pay : ℝ), low_pay * (1 + ratio) = total_pay ∧ low_pay = 228 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l1498_149873


namespace NUMINAMATH_CALUDE_product_and_multiple_l1498_149841

theorem product_and_multiple : ∃ x : ℕ, x = 320 * 6 ∧ x * 7 = 420 → x = 1920 := by
  sorry

end NUMINAMATH_CALUDE_product_and_multiple_l1498_149841


namespace NUMINAMATH_CALUDE_completing_square_correct_l1498_149832

-- Define the original equation
def original_equation (x : ℝ) : Prop := x^2 - 4*x - 22 = 0

-- Define the result of completing the square
def completed_square_result (x : ℝ) : Prop := (x - 2)^2 = 26

-- Theorem statement
theorem completing_square_correct :
  ∀ x : ℝ, original_equation x ↔ completed_square_result x :=
by sorry

end NUMINAMATH_CALUDE_completing_square_correct_l1498_149832


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1498_149803

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
    (x * y * z + 2 * x + 3 * y + 6 * z = x * y + 2 * x * z + 3 * y * z) →
    (x = 4 ∧ y = 3 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1498_149803
