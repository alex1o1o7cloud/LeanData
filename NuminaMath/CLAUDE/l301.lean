import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_sum_inequality_l301_30108

theorem quadratic_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≤ 3*b ∧ b ≤ 3*a)
  (hac : a ≤ 3*c ∧ c ≤ 3*a)
  (had : a ≤ 3*d ∧ d ≤ 3*a)
  (hbc : b ≤ 3*c ∧ c ≤ 3*b)
  (hbd : b ≤ 3*d ∧ d ≤ 3*b)
  (hcd : c ≤ 3*d ∧ d ≤ 3*c) :
  a^2 + b^2 + c^2 + d^2 < 2*(a*b + a*c + a*d + b*c + b*d + c*d) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_inequality_l301_30108


namespace NUMINAMATH_CALUDE_polynomial_identity_l301_30158

/-- For any real numbers a, b, and c, 
    a(b - c)^4 + b(c - a)^4 + c(a - b)^4 = (a - b)(b - c)(c - a)(a + b + c) -/
theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l301_30158


namespace NUMINAMATH_CALUDE_marias_trip_distance_l301_30168

/-- The total distance of Maria's trip -/
def total_distance : ℝ := 480

/-- Theorem stating that the total distance of Maria's trip is 480 miles -/
theorem marias_trip_distance :
  ∃ (D : ℝ),
    D / 2 + (D / 2) / 4 + 180 = D ∧
    D = total_distance :=
by sorry

end NUMINAMATH_CALUDE_marias_trip_distance_l301_30168


namespace NUMINAMATH_CALUDE_problem_statement_l301_30191

theorem problem_statement : 
  (Real.sqrt (5 + Real.sqrt 6) + Real.sqrt (5 - Real.sqrt 6)) / Real.sqrt (Real.sqrt 6 - 1) - Real.sqrt (4 - 2 * Real.sqrt 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l301_30191


namespace NUMINAMATH_CALUDE_octadecagon_diagonals_l301_30167

/-- The number of sides in an octadecagon -/
def octadecagon_sides : ℕ := 18

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in an octadecagon is 135 -/
theorem octadecagon_diagonals : 
  num_diagonals octadecagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_octadecagon_diagonals_l301_30167


namespace NUMINAMATH_CALUDE_parabola_focus_l301_30134

/-- A parabola is defined by its equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k) -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Given a parabola y = -2x^2 + 5, its focus is (0, 9/2) -/
theorem parabola_focus (p : Parabola) (f : Focus) : 
  p.a = -2 ∧ p.b = 0 ∧ p.c = 5 → f.h = 0 ∧ f.k = 9/2 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l301_30134


namespace NUMINAMATH_CALUDE_function_form_proof_l301_30154

theorem function_form_proof (f : ℝ → ℝ) (k : ℝ) 
  (h_continuous : Continuous f)
  (h_zero : f 0 = 0)
  (h_inequality : ∀ x y, f (x + y) ≥ f x + f y + k * x * y) :
  ∃ b : ℝ, ∀ x, f x = k / 2 * x^2 + b * x :=
sorry

end NUMINAMATH_CALUDE_function_form_proof_l301_30154


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l301_30105

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l301_30105


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l301_30196

theorem necessary_not_sufficient_condition (a b c d : ℝ) (h : c > d) :
  (∀ a b, (a - c > b - d) → (a > b)) ∧
  (∃ a b, (a > b) ∧ ¬(a - c > b - d)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l301_30196


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l301_30198

/-- A quadratic function is a function of the form f(x) = ax² + bx + c, where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique
  (f : ℝ → ℝ)
  (h_quad : QuadraticFunction f)
  (h_f_2 : f 2 = -1)
  (h_f_neg1 : f (-1) = -1)
  (h_max : ∃ x_max, ∀ x, f x ≤ f x_max ∧ f x_max = 8) :
  ∀ x, f x = -4 * x^2 + 4 * x + 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l301_30198


namespace NUMINAMATH_CALUDE_f_of_4_equals_22_l301_30100

/-- Given a function f(x) = 5x + 2, prove that f(4) = 22 -/
theorem f_of_4_equals_22 :
  let f : ℝ → ℝ := λ x ↦ 5 * x + 2
  f 4 = 22 := by sorry

end NUMINAMATH_CALUDE_f_of_4_equals_22_l301_30100


namespace NUMINAMATH_CALUDE_total_surface_area_circumscribed_prism_l301_30117

/-- A prism circumscribed about a sphere -/
structure CircumscribedPrism where
  -- The area of the base of the prism
  base_area : ℝ
  -- The semi-perimeter of the base of the prism
  semi_perimeter : ℝ
  -- The radius of the sphere
  sphere_radius : ℝ
  -- The base area is equal to the product of semi-perimeter and sphere radius
  base_area_eq : base_area = semi_perimeter * sphere_radius

/-- The total surface area of a circumscribed prism is 6 times its base area -/
theorem total_surface_area_circumscribed_prism (p : CircumscribedPrism) :
  ∃ (total_surface_area : ℝ), total_surface_area = 6 * p.base_area :=
by
  sorry

end NUMINAMATH_CALUDE_total_surface_area_circumscribed_prism_l301_30117


namespace NUMINAMATH_CALUDE_smallest_number_greater_than_0_4_l301_30115

theorem smallest_number_greater_than_0_4 (S : Set ℝ) : 
  S = {0.8, 1/2, 0.3, 1/3} → 
  (∃ x ∈ S, x > 0.4 ∧ ∀ y ∈ S, y > 0.4 → x ≤ y) → 
  (1/2 ∈ S ∧ 1/2 > 0.4 ∧ ∀ y ∈ S, y > 0.4 → 1/2 ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_greater_than_0_4_l301_30115


namespace NUMINAMATH_CALUDE_arc_length_for_given_angle_l301_30140

theorem arc_length_for_given_angle (r : ℝ) (α : ℝ) (h1 : r = 2) (h2 : α = π / 7) :
  r * α = 2 * π / 7 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_given_angle_l301_30140


namespace NUMINAMATH_CALUDE_product_xyz_l301_30164

theorem product_xyz (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xy : x * y = 27 * Real.rpow 3 (1/3))
  (h_xz : x * z = 45 * Real.rpow 3 (1/3))
  (h_yz : y * z = 18 * Real.rpow 3 (1/3))
  (h_x_2y : x = 2 * y) : 
  x * y * z = 108 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l301_30164


namespace NUMINAMATH_CALUDE_fraction_equality_l301_30177

theorem fraction_equality (x y : ℚ) (hx : x = 4/6) (hy : y = 8/10) :
  (6 * x^2 + 10 * y) / (60 * x * y) = 11/36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l301_30177


namespace NUMINAMATH_CALUDE_min_value_of_f_l301_30182

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) ∧ 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -1) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ -1) :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_f_l301_30182


namespace NUMINAMATH_CALUDE_right_triangle_area_l301_30150

/-- The area of a right triangle with hypotenuse 12 inches and one angle 30° is 18√3 square inches. -/
theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) :
  h = 12 →  -- hypotenuse is 12 inches
  θ = 30 * π / 180 →  -- one angle is 30°
  area = h * h * Real.sin θ * Real.cos θ / 2 →  -- area formula for right triangle
  area = 18 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l301_30150


namespace NUMINAMATH_CALUDE_sum_of_third_and_fourth_terms_l301_30101

theorem sum_of_third_and_fourth_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, S n = n^2 + n) → a 3 + a 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_third_and_fourth_terms_l301_30101


namespace NUMINAMATH_CALUDE_volume_of_extended_parallelepiped_with_caps_l301_30137

/-- The volume of a set of points that are inside or within one unit of a rectangular parallelepiped
    with semi-spherical caps on the longest side vertices. -/
theorem volume_of_extended_parallelepiped_with_caps : ℝ := by
  -- Define the dimensions of the parallelepiped
  let length : ℝ := 6
  let width : ℝ := 3
  let height : ℝ := 2

  -- Define the radius of the semi-spherical caps
  let cap_radius : ℝ := 1

  -- Define the number of semi-spherical caps
  let num_caps : ℕ := 4

  -- Calculate the volume
  have volume : ℝ := (324 + 8 * Real.pi) / 3

  sorry

#check volume_of_extended_parallelepiped_with_caps

end NUMINAMATH_CALUDE_volume_of_extended_parallelepiped_with_caps_l301_30137


namespace NUMINAMATH_CALUDE_min_ab_value_l301_30118

theorem min_ab_value (a b : ℕ+) 
  (h : (fun x y : ℝ => x^2 + y^2 - 2*a*x + a^2*(1-b)) = 0 ↔ 
       (fun x y : ℝ => x^2 + y^2 - 2*y + 1 - a^2*b) = 0) : 
  (a : ℝ) * (b : ℝ) ≥ (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_min_ab_value_l301_30118


namespace NUMINAMATH_CALUDE_minimum_yellow_balls_l301_30199

theorem minimum_yellow_balls
  (g : ℕ) -- number of green balls
  (y : ℕ) -- number of yellow balls
  (o : ℕ) -- number of orange balls
  (h1 : o ≥ g / 3)  -- orange balls at least one-third of green balls
  (h2 : o ≤ y / 4)  -- orange balls at most one-fourth of yellow balls
  (h3 : g + o ≥ 75) -- combined green and orange balls at least 75
  : y ≥ 76 := by
  sorry

#check minimum_yellow_balls

end NUMINAMATH_CALUDE_minimum_yellow_balls_l301_30199


namespace NUMINAMATH_CALUDE_expression_equals_one_l301_30179

theorem expression_equals_one :
  (π + 2023) ^ 0 + 2 * Real.sin (π / 4) - (1 / 2)⁻¹ + |Real.sqrt 2 - 2| = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l301_30179


namespace NUMINAMATH_CALUDE_triangle_area_with_given_base_height_l301_30187

/-- The area of a triangle with base 12 cm and height 15 cm is 90 cm². -/
theorem triangle_area_with_given_base_height :
  let base : ℝ := 12
  let height : ℝ := 15
  let area : ℝ := (1 / 2) * base * height
  area = 90 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_base_height_l301_30187


namespace NUMINAMATH_CALUDE_workshop_average_salary_l301_30110

/-- Given a workshop with workers, prove that the average salary is 8000 --/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 30)
  (h2 : technicians = 10)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 8000 := by
  sorry

#check workshop_average_salary

end NUMINAMATH_CALUDE_workshop_average_salary_l301_30110


namespace NUMINAMATH_CALUDE_worker_a_time_l301_30163

theorem worker_a_time (worker_b_time worker_ab_time : ℝ) 
  (hb : worker_b_time = 12)
  (hab : worker_ab_time = 4.8) : ℝ :=
  let worker_a_time := (worker_b_time * worker_ab_time) / (worker_b_time - worker_ab_time)
  8

#check worker_a_time

end NUMINAMATH_CALUDE_worker_a_time_l301_30163


namespace NUMINAMATH_CALUDE_x_value_proof_l301_30102

theorem x_value_proof (x : ℚ) 
  (eq1 : 8 * x^2 + 7 * x - 1 = 0) 
  (eq2 : 24 * x^2 + 53 * x - 7 = 0) : 
  x = 1/8 := by sorry

end NUMINAMATH_CALUDE_x_value_proof_l301_30102


namespace NUMINAMATH_CALUDE_sams_nickels_l301_30152

/-- Given Sam's initial nickels and his dad's gift of nickels, calculate Sam's total nickels -/
theorem sams_nickels (initial_nickels dad_gift_nickels : ℕ) :
  initial_nickels = 24 → dad_gift_nickels = 39 →
  initial_nickels + dad_gift_nickels = 63 := by sorry

end NUMINAMATH_CALUDE_sams_nickels_l301_30152


namespace NUMINAMATH_CALUDE_odd_function_implies_a_equals_negative_one_l301_30148

def f (a : ℝ) (x : ℝ) : ℝ := x - a - 1

theorem odd_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a (-x) = -(f a x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_equals_negative_one_l301_30148


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l301_30120

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hourlyWage : ℝ
  hoursMonWedFri : ℝ
  hoursTueThu : ℝ
  daysWithLongHours : ℕ
  daysWithShortHours : ℕ

/-- Calculates the weekly earnings based on the work schedule --/
def weeklyEarnings (schedule : WorkSchedule) : ℝ :=
  (schedule.hourlyWage * schedule.hoursMonWedFri * schedule.daysWithLongHours) +
  (schedule.hourlyWage * schedule.hoursTueThu * schedule.daysWithShortHours)

/-- Theorem stating that Sheila's weekly earnings are $288 --/
theorem sheila_weekly_earnings :
  let schedule : WorkSchedule := {
    hourlyWage := 8,
    hoursMonWedFri := 8,
    hoursTueThu := 6,
    daysWithLongHours := 3,
    daysWithShortHours := 2
  }
  weeklyEarnings schedule = 288 := by
  sorry

end NUMINAMATH_CALUDE_sheila_weekly_earnings_l301_30120


namespace NUMINAMATH_CALUDE_smallest_three_digit_geometric_sequence_l301_30169

/-- Checks if a number is a three-digit integer -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Extracts the hundreds digit of a three-digit number -/
def hundredsDigit (n : ℕ) : ℕ := n / 100

/-- Extracts the tens digit of a three-digit number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- Extracts the ones digit of a three-digit number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- Checks if the digits of a three-digit number are distinct -/
def hasDistinctDigits (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  h ≠ t ∧ t ≠ o ∧ h ≠ o

/-- Checks if the digits of a three-digit number form a geometric sequence -/
def formsGeometricSequence (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  ∃ r : ℚ, r > 1 ∧ t = h * r ∧ o = t * r

theorem smallest_three_digit_geometric_sequence :
  ∀ n : ℕ, isThreeDigit n → hasDistinctDigits n → formsGeometricSequence n → n ≥ 248 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_geometric_sequence_l301_30169


namespace NUMINAMATH_CALUDE_exp_gt_one_plus_x_when_not_zero_l301_30143

theorem exp_gt_one_plus_x_when_not_zero (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end NUMINAMATH_CALUDE_exp_gt_one_plus_x_when_not_zero_l301_30143


namespace NUMINAMATH_CALUDE_product_one_sum_greater_than_reciprocals_l301_30176

theorem product_one_sum_greater_than_reciprocals 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > 1/a + 1/b + 1/c) : 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
sorry

end NUMINAMATH_CALUDE_product_one_sum_greater_than_reciprocals_l301_30176


namespace NUMINAMATH_CALUDE_bg_length_is_two_l301_30124

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 0 ∧ B.2 = Real.sqrt 12 ∧ C.1 = 2 ∧ C.2 = 0

-- Define the square BDEC
def Square (B D E C : ℝ × ℝ) : Prop :=
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = (E.1 - D.1)^2 + (E.2 - D.2)^2 ∧
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = (C.1 - E.1)^2 + (C.2 - E.2)^2 ∧
  (C.1 - E.1)^2 + (C.2 - E.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Define the center of the square
def CenterOfSquare (F B C : ℝ × ℝ) : Prop :=
  F.1 = (B.1 + C.1) / 2 ∧ F.2 = (B.2 + C.2) / 2

-- Define the intersection point G
def Intersection (A F B C G : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, G.1 = t * (F.1 - A.1) + A.1 ∧ G.2 = t * (F.2 - A.2) + A.2 ∧
  G.1 = B.1 + (C.1 - B.1) * ((G.2 - B.2) / (C.2 - B.2))

-- Main theorem
theorem bg_length_is_two 
  (A B C D E F G : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Square B D E C) 
  (h3 : CenterOfSquare F B C) 
  (h4 : Intersection A F B C G) : 
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bg_length_is_two_l301_30124


namespace NUMINAMATH_CALUDE_margo_walking_distance_l301_30181

/-- Proves that Margo's total walking distance is 2 miles given the specified conditions -/
theorem margo_walking_distance
  (time_to_friend : ℝ)
  (time_to_return : ℝ)
  (average_speed : ℝ)
  (h1 : time_to_friend = 15)
  (h2 : time_to_return = 25)
  (h3 : average_speed = 3)
  : (time_to_friend + time_to_return) / 60 * average_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_margo_walking_distance_l301_30181


namespace NUMINAMATH_CALUDE_population_change_theorem_l301_30127

/-- Represents the population change factor for a given percentage change -/
def change_factor (percent : ℚ) : ℚ := 1 + percent / 100

/-- Calculates the net change in population over 5 years given the yearly changes -/
def net_change (year1 year2 year3 year4 year5 : ℚ) : ℚ :=
  (change_factor year1 * change_factor year2 * change_factor year3 * 
   change_factor year4 * change_factor year5 - 1) * 100

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  if q - ⌊q⌋ < 1/2 then ⌊q⌋ else ⌈q⌉

theorem population_change_theorem :
  round_to_nearest (net_change 20 10 (-30) (-20) 10) = -19 := by sorry

end NUMINAMATH_CALUDE_population_change_theorem_l301_30127


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l301_30103

theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 3) - 5
  f (3/2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l301_30103


namespace NUMINAMATH_CALUDE_dave_total_earnings_l301_30112

/-- Calculates daily earnings after tax -/
def dailyEarnings (hourlyWage : ℚ) (hoursWorked : ℚ) (unpaidBreak : ℚ) : ℚ :=
  let actualHours := hoursWorked - unpaidBreak
  let earningsBeforeTax := actualHours * hourlyWage
  let taxDeduction := earningsBeforeTax * (1 / 10)
  earningsBeforeTax - taxDeduction

/-- Represents Dave's total earnings for the week -/
def daveEarnings : ℚ :=
  dailyEarnings 6 6 (1/2) +  -- Monday
  dailyEarnings 7 2 (1/4) +  -- Tuesday
  dailyEarnings 9 3 0 +      -- Wednesday
  dailyEarnings 8 5 (1/2)    -- Thursday

theorem dave_total_earnings :
  daveEarnings = 9743 / 100 := by sorry

end NUMINAMATH_CALUDE_dave_total_earnings_l301_30112


namespace NUMINAMATH_CALUDE_sin_inequality_l301_30194

theorem sin_inequality (θ : Real) (h : 0 < θ ∧ θ < Real.pi) :
  Real.sin θ + (1/2) * Real.sin (2*θ) + (1/3) * Real.sin (3*θ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_l301_30194


namespace NUMINAMATH_CALUDE_vasims_share_l301_30125

/-- Represents the share of money for each person -/
structure Share :=
  (amount : ℕ)

/-- Represents the distribution of money among three people -/
structure Distribution :=
  (faruk : Share)
  (vasim : Share)
  (ranjith : Share)

/-- The ratio of the distribution -/
def distribution_ratio (d : Distribution) : Prop :=
  5 * d.faruk.amount = 3 * d.vasim.amount ∧
  6 * d.faruk.amount = 3 * d.ranjith.amount

/-- The difference between the largest and smallest share is 900 -/
def share_difference (d : Distribution) : Prop :=
  d.ranjith.amount - d.faruk.amount = 900

theorem vasims_share (d : Distribution) 
  (h1 : distribution_ratio d) 
  (h2 : share_difference d) : 
  d.vasim.amount = 1500 :=
sorry

end NUMINAMATH_CALUDE_vasims_share_l301_30125


namespace NUMINAMATH_CALUDE_specific_tournament_balls_used_l301_30119

/-- A tennis tournament with specific rules for ball usage -/
structure TennisTournament where
  rounds : Nat
  games_per_round : List Nat
  cans_per_game : Nat
  balls_per_can : Nat

/-- Calculate the total number of tennis balls used in a tournament -/
def total_balls_used (t : TennisTournament) : Nat :=
  (t.games_per_round.sum * t.cans_per_game * t.balls_per_can)

/-- Theorem: The total number of tennis balls used in the specific tournament is 225 -/
theorem specific_tournament_balls_used :
  let t : TennisTournament := {
    rounds := 4,
    games_per_round := [8, 4, 2, 1],
    cans_per_game := 5,
    balls_per_can := 3
  }
  total_balls_used t = 225 := by
  sorry


end NUMINAMATH_CALUDE_specific_tournament_balls_used_l301_30119


namespace NUMINAMATH_CALUDE_problem_solution_l301_30142

theorem problem_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b * c = 1) (h5 : a + 1 / c = 7) (h6 : b + 1 / a = 34) :
  c + 1 / b = 43 / 237 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l301_30142


namespace NUMINAMATH_CALUDE_buffet_meal_combinations_l301_30136

def meat_options : ℕ := 4
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 5
def vegetables_to_choose : ℕ := 3

theorem buffet_meal_combinations :
  meat_options * Nat.choose vegetable_options vegetables_to_choose * dessert_options = 200 := by
  sorry

end NUMINAMATH_CALUDE_buffet_meal_combinations_l301_30136


namespace NUMINAMATH_CALUDE_function_zero_set_empty_l301_30171

theorem function_zero_set_empty (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + 3 * f (1 - x) = x^2) :
  {x : ℝ | f x = 0} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_function_zero_set_empty_l301_30171


namespace NUMINAMATH_CALUDE_unique_solution_exists_l301_30132

theorem unique_solution_exists (x y z : ℝ) : 
  (x / 6) * 12 = 11 ∧ 
  4 * (x - y) + 5 = 11 ∧ 
  Real.sqrt z = (3 * x + y / 2) ^ 2 →
  x = 5.5 ∧ y = 4 ∧ z = 117132.0625 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l301_30132


namespace NUMINAMATH_CALUDE_total_results_l301_30111

theorem total_results (avg_all : ℝ) (avg_first_six : ℝ) (avg_last_six : ℝ) (sixth_result : ℝ) : 
  avg_all = 52 → 
  avg_first_six = 49 → 
  avg_last_six = 52 → 
  sixth_result = 34 → 
  ∃ n : ℕ, n = 11 ∧ n * avg_all = (6 * avg_first_six + 6 * avg_last_six - sixth_result) :=
by
  sorry

#check total_results

end NUMINAMATH_CALUDE_total_results_l301_30111


namespace NUMINAMATH_CALUDE_walkers_speed_l301_30122

/-- Proves that given the conditions of the problem, A's walking speed is 10 kmph -/
theorem walkers_speed (v : ℝ) : 
  (∃ t : ℝ, v * (t + 7) = 20 * t) →  -- B catches up with A
  (∃ t : ℝ, v * (t + 7) = 140) →     -- Distance traveled is 140 km
  v = 10 := by sorry

end NUMINAMATH_CALUDE_walkers_speed_l301_30122


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l301_30195

theorem arithmetic_calculations : 
  (1 - (-5) * ((-1)^2) - 4 / ((-1/2)^2) = -11) ∧ 
  ((-2)^3 * (1/8) + (2/3 - 1/2 - 1/4) / (1/12) = -2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l301_30195


namespace NUMINAMATH_CALUDE_sara_lunch_cost_l301_30161

/-- The cost of Sara's lunch given the prices of a hotdog and a salad -/
def lunch_cost (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem stating that Sara's lunch cost is $10.46 -/
theorem sara_lunch_cost :
  lunch_cost 5.36 5.10 = 10.46 := by
  sorry

end NUMINAMATH_CALUDE_sara_lunch_cost_l301_30161


namespace NUMINAMATH_CALUDE_equation_solution_l301_30144

theorem equation_solution (a b : ℝ) :
  (∀ x y : ℝ, y = a + b / x) →
  (3 = a + b / 2) →
  (-1 = a + b / (-4)) →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l301_30144


namespace NUMINAMATH_CALUDE_vector_orthogonality_l301_30162

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def b : Fin 2 → ℝ := ![1, -1]

-- Define the orthogonality condition
def orthogonal (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

-- Theorem statement
theorem vector_orthogonality (x : ℝ) :
  orthogonal (λ i => a x i - b i) b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_orthogonality_l301_30162


namespace NUMINAMATH_CALUDE_least_number_divisible_l301_30107

theorem least_number_divisible (n : ℕ) : n = 858 ↔ 
  (∀ m : ℕ, m < n → 
    ¬((m + 6) % 24 = 0 ∧ 
      (m + 6) % 32 = 0 ∧ 
      (m + 6) % 36 = 0 ∧ 
      (m + 6) % 54 = 0)) ∧
  ((n + 6) % 24 = 0 ∧ 
   (n + 6) % 32 = 0 ∧ 
   (n + 6) % 36 = 0 ∧ 
   (n + 6) % 54 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_l301_30107


namespace NUMINAMATH_CALUDE_puppy_cost_l301_30106

theorem puppy_cost (items_cost : ℝ) (discount_rate : ℝ) (total_spent : ℝ)
  (h1 : items_cost = 95)
  (h2 : discount_rate = 0.2)
  (h3 : total_spent = 96) :
  total_spent - items_cost * (1 - discount_rate) = 20 :=
by sorry

end NUMINAMATH_CALUDE_puppy_cost_l301_30106


namespace NUMINAMATH_CALUDE_sum_from_true_discount_and_simple_interest_l301_30188

theorem sum_from_true_discount_and_simple_interest 
  (S : ℝ) 
  (D I : ℝ) 
  (h1 : D = 75) 
  (h2 : I = 85) 
  (h3 : D / I = (S - D) / S) : S = 637.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_from_true_discount_and_simple_interest_l301_30188


namespace NUMINAMATH_CALUDE_cos_120_degrees_l301_30130

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l301_30130


namespace NUMINAMATH_CALUDE_walking_distance_problem_l301_30138

theorem walking_distance_problem (x t d : ℝ) 
  (h1 : d = (x + 1) * (3/4 * t))
  (h2 : d = (x - 1) * (t + 3)) :
  d = 18 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_problem_l301_30138


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l301_30185

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, k > 0 ∧ n = 45 * k) → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l301_30185


namespace NUMINAMATH_CALUDE_square_area_decrease_l301_30153

theorem square_area_decrease (initial_area : ℝ) (side_decrease_percent : ℝ) 
  (h1 : initial_area = 50)
  (h2 : side_decrease_percent = 25) : 
  let new_side := (1 - side_decrease_percent / 100) * Real.sqrt initial_area
  let new_area := new_side * Real.sqrt initial_area
  let percent_decrease := (initial_area - new_area) / initial_area * 100
  percent_decrease = 43.75 := by
sorry

end NUMINAMATH_CALUDE_square_area_decrease_l301_30153


namespace NUMINAMATH_CALUDE_odd_function_property_l301_30131

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h_odd : is_odd f) (h_sum : f (-2) + f 0 + f 3 = 2) :
  f 2 - f 3 = -2 := by sorry

end NUMINAMATH_CALUDE_odd_function_property_l301_30131


namespace NUMINAMATH_CALUDE_timePerPlayer_is_36_l301_30159

/-- Represents a sports tournament with given parameters -/
structure Tournament where
  teamSize : ℕ
  playersOnField : ℕ
  matchDuration : ℕ
  hTeamSize : teamSize = 10
  hPlayersOnField : playersOnField = 8
  hMatchDuration : matchDuration = 45
  hPlayersOnFieldLessTeamSize : playersOnField < teamSize

/-- Calculates the time each player spends on the field -/
def timePerPlayer (t : Tournament) : ℕ :=
  t.playersOnField * t.matchDuration / t.teamSize

/-- Theorem stating that each player spends 36 minutes on the field -/
theorem timePerPlayer_is_36 (t : Tournament) : timePerPlayer t = 36 := by
  sorry

end NUMINAMATH_CALUDE_timePerPlayer_is_36_l301_30159


namespace NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l301_30175

/-- The ratio of the surface area of a cube to the surface area of a regular tetrahedron
    formed by four vertices of the cube, given that the cube has side length 2. -/
theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let tetrahedron_side_length : ℝ := 2 * Real.sqrt 2
  let cube_surface_area : ℝ := 6 * cube_side_length ^ 2
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length ^ 2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l301_30175


namespace NUMINAMATH_CALUDE_candy_cost_l301_30151

theorem candy_cost (packs : ℕ) (paid : ℕ) (change : ℕ) (h1 : packs = 3) (h2 : paid = 20) (h3 : change = 11) :
  (paid - change) / packs = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l301_30151


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l301_30149

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  a 1 = 2 ∧ 
  d ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (∃ r : ℝ, r ≠ 0 ∧ a 3 = r * a 1 ∧ a 11 = r * a 3) →
  (∃ r : ℝ, r = 4 ∧ a 3 = r * a 1 ∧ a 11 = r * a 3) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l301_30149


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l301_30145

/-- Represents the repeating decimal 0.090909... -/
def a : ℚ := 1 / 11

/-- Represents the repeating decimal 0.777777... -/
def b : ℚ := 7 / 9

/-- The product of the repeating decimals 0.090909... and 0.777777... equals 7/99 -/
theorem product_of_repeating_decimals : a * b = 7 / 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l301_30145


namespace NUMINAMATH_CALUDE_rectangles_with_one_gray_count_l301_30116

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the count of different types of cells in the grid -/
structure CellCount :=
  (total_gray : ℕ)
  (interior_gray : ℕ)
  (edge_gray : ℕ)

/-- Calculates the number of rectangles containing exactly one gray cell -/
def count_rectangles_with_one_gray (g : Grid) (c : CellCount) : ℕ :=
  c.interior_gray * 4 + c.edge_gray * 8

/-- The main theorem stating the number of rectangles with one gray cell -/
theorem rectangles_with_one_gray_count 
  (g : Grid) 
  (c : CellCount) 
  (h1 : g.rows = 5) 
  (h2 : g.cols = 22) 
  (h3 : c.total_gray = 40) 
  (h4 : c.interior_gray = 36) 
  (h5 : c.edge_gray = 4) :
  count_rectangles_with_one_gray g c = 176 := by
  sorry

#check rectangles_with_one_gray_count

end NUMINAMATH_CALUDE_rectangles_with_one_gray_count_l301_30116


namespace NUMINAMATH_CALUDE_window_height_is_four_l301_30178

-- Define the room dimensions
def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

-- Define the cost of whitewashing per square foot
def cost_per_sqft : ℝ := 3

-- Define the door dimensions
def door_height : ℝ := 6
def door_width : ℝ := 3

-- Define the number of windows and their width
def num_windows : ℕ := 3
def window_width : ℝ := 3

-- Define the total cost of whitewashing
def total_cost : ℝ := 2718

-- Theorem to prove
theorem window_height_is_four :
  ∃ (h : ℝ),
    h = 4 ∧
    (2 * (room_length * room_height + room_width * room_height) -
     (door_height * door_width + num_windows * h * window_width)) * cost_per_sqft = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_window_height_is_four_l301_30178


namespace NUMINAMATH_CALUDE_prop1_prop2_prop3_prop4_l301_30113

-- Define the function f
variable (f : ℝ → ℝ)

-- Proposition 1
theorem prop1 (h : ∀ x, f (1 + 2*x) = f (1 - 2*x)) :
  ∀ x, f (2 - x) = f x :=
sorry

-- Proposition 2
theorem prop2 :
  ∀ x, f (x - 2) = f (2 - x) :=
sorry

-- Proposition 3
theorem prop3 (h1 : ∀ x, f x = f (-x)) (h2 : ∀ x, f (2 + x) = -f x) :
  ∀ x, f (4 - x) = f x :=
sorry

-- Proposition 4
theorem prop4 (h1 : ∀ x, f x = -f (-x)) (h2 : ∀ x, f x = f (-x - 2)) :
  ∀ x, f (2 - x) = f x :=
sorry

end NUMINAMATH_CALUDE_prop1_prop2_prop3_prop4_l301_30113


namespace NUMINAMATH_CALUDE_leila_order_proof_l301_30186

/-- The number of chocolate cakes Leila ordered -/
def chocolate_cakes : ℕ := 3

/-- The cost of each chocolate cake -/
def chocolate_cake_cost : ℕ := 12

/-- The number of strawberry cakes Leila ordered -/
def strawberry_cakes : ℕ := 6

/-- The cost of each strawberry cake -/
def strawberry_cake_cost : ℕ := 22

/-- The total amount Leila should pay -/
def total_amount : ℕ := 168

theorem leila_order_proof :
  chocolate_cakes * chocolate_cake_cost + 
  strawberry_cakes * strawberry_cake_cost = total_amount :=
by sorry

end NUMINAMATH_CALUDE_leila_order_proof_l301_30186


namespace NUMINAMATH_CALUDE_no_four_binomial_coeff_arithmetic_progression_l301_30114

theorem no_four_binomial_coeff_arithmetic_progression :
  ∀ n m : ℕ, n > 0 → m > 0 → m + 3 ≤ n →
  ¬∃ d : ℕ, 
    (Nat.choose n (m+1) = Nat.choose n m + d) ∧
    (Nat.choose n (m+2) = Nat.choose n (m+1) + d) ∧
    (Nat.choose n (m+3) = Nat.choose n (m+2) + d) :=
by sorry

end NUMINAMATH_CALUDE_no_four_binomial_coeff_arithmetic_progression_l301_30114


namespace NUMINAMATH_CALUDE_allen_blocks_count_l301_30165

/-- The number of blocks per color -/
def blocks_per_color : ℕ := 7

/-- The number of colors used -/
def colors_used : ℕ := 7

/-- The total number of blocks -/
def total_blocks : ℕ := blocks_per_color * colors_used

theorem allen_blocks_count : total_blocks = 49 := by
  sorry

end NUMINAMATH_CALUDE_allen_blocks_count_l301_30165


namespace NUMINAMATH_CALUDE_no_valid_pairs_l301_30157

theorem no_valid_pairs : ¬∃ (M N K : ℕ), 
  M > 0 ∧ N > 0 ∧ 
  (M : ℚ) / 5 = 5 / (N : ℚ) ∧ 
  M = 2 * K := by
  sorry

end NUMINAMATH_CALUDE_no_valid_pairs_l301_30157


namespace NUMINAMATH_CALUDE_student_number_problem_l301_30156

theorem student_number_problem (x : ℝ) : 7 * x - 150 = 130 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l301_30156


namespace NUMINAMATH_CALUDE_computation_problem_points_l301_30128

theorem computation_problem_points :
  ∀ (total_problems : ℕ) 
    (computation_problems : ℕ) 
    (word_problem_points : ℕ) 
    (total_points : ℕ),
  total_problems = 30 →
  computation_problems = 20 →
  word_problem_points = 5 →
  total_points = 110 →
  ∃ (computation_problem_points : ℕ),
    computation_problem_points * computation_problems +
    word_problem_points * (total_problems - computation_problems) = total_points ∧
    computation_problem_points = 3 :=
by sorry

end NUMINAMATH_CALUDE_computation_problem_points_l301_30128


namespace NUMINAMATH_CALUDE_irrational_among_given_numbers_l301_30193

theorem irrational_among_given_numbers : 
  (∃ (q : ℚ), (1 : ℝ) / 2 = ↑q) ∧ 
  (∃ (q : ℚ), (1 : ℝ) / 3 = ↑q) ∧ 
  (∃ (q : ℚ), Real.sqrt 4 = ↑q) ∧ 
  (∀ (q : ℚ), Real.sqrt 5 ≠ ↑q) := by
  sorry

end NUMINAMATH_CALUDE_irrational_among_given_numbers_l301_30193


namespace NUMINAMATH_CALUDE_cos_90_degrees_l301_30173

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_l301_30173


namespace NUMINAMATH_CALUDE_malfunction_time_proof_l301_30147

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  hh_valid : hours < 24
  mm_valid : minutes < 60

/-- Represents a malfunctioning clock where each digit changed by ±1 -/
def is_malfunction (original : Time) (displayed : Time) : Prop :=
  (displayed.hours / 10 = original.hours / 10 + 1 ∨ displayed.hours / 10 = original.hours / 10 - 1) ∧
  (displayed.hours % 10 = (original.hours % 10 + 1) % 10 ∨ displayed.hours % 10 = (original.hours % 10 - 1 + 10) % 10) ∧
  (displayed.minutes / 10 = original.minutes / 10 + 1 ∨ displayed.minutes / 10 = original.minutes / 10 - 1) ∧
  (displayed.minutes % 10 = (original.minutes % 10 + 1) % 10 ∨ displayed.minutes % 10 = (original.minutes % 10 - 1 + 10) % 10)

theorem malfunction_time_proof (displayed : Time) 
  (h_displayed : displayed.hours = 20 ∧ displayed.minutes = 9) :
  ∃ (original : Time), is_malfunction original displayed ∧ original.hours = 11 ∧ original.minutes = 18 := by
  sorry

end NUMINAMATH_CALUDE_malfunction_time_proof_l301_30147


namespace NUMINAMATH_CALUDE_percent_of_y_l301_30129

theorem percent_of_y (y : ℝ) (h : y > 0) : ((8 * y) / 20 + (3 * y) / 10) / y = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l301_30129


namespace NUMINAMATH_CALUDE_opposite_of_neg_six_l301_30174

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- The opposite of -6 is 6. -/
theorem opposite_of_neg_six : opposite (-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_six_l301_30174


namespace NUMINAMATH_CALUDE_complex_cube_problem_l301_30160

theorem complex_cube_problem :
  ∀ (x y : ℕ+) (c : ℤ),
    (x : ℂ) + y * Complex.I ≠ 1 + 6 * Complex.I →
    ((x : ℂ) + y * Complex.I) ^ 3 ≠ -107 + c * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_problem_l301_30160


namespace NUMINAMATH_CALUDE_problem_proof_l301_30166

theorem problem_proof (x v : ℝ) (hx : x = 2) (hv : v = 3 * x) :
  (2 * v - 5) - (2 * x - 5) = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l301_30166


namespace NUMINAMATH_CALUDE_carts_needed_is_15_l301_30183

-- Define the total volume of goods
def total_volume : ℚ := 1

-- Define the daily capacity of each vehicle type
def large_truck_capacity : ℚ := total_volume / (3 * 4)
def small_truck_capacity : ℚ := total_volume / (4 * 5)
def cart_capacity : ℚ := total_volume / (20 * 6)

-- Define the work done in the first 2 days
def work_done_2_days : ℚ := 2 * (2 * large_truck_capacity + 3 * small_truck_capacity + 7 * cart_capacity)

-- Define the remaining work
def remaining_work : ℚ := total_volume - work_done_2_days

-- Define the number of carts needed for the last 2 days
def carts_needed : ℕ := (remaining_work / (2 * cart_capacity)).ceil.toNat

-- Theorem statement
theorem carts_needed_is_15 : carts_needed = 15 := by
  sorry

end NUMINAMATH_CALUDE_carts_needed_is_15_l301_30183


namespace NUMINAMATH_CALUDE_triangle_properties_l301_30190

open Real

theorem triangle_properties (A B C a b c : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 → b > 0 → c > 0 →
  2 * a * cos C + c = 2 * b →
  a = Real.sqrt 3 →
  (1 / 2) * b * c * sin A = Real.sqrt 3 / 2 →
  A = π / 3 ∧ a + b + c = 3 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l301_30190


namespace NUMINAMATH_CALUDE_min_difference_sine_extrema_l301_30184

open Real

theorem min_difference_sine_extrema (f : ℝ → ℝ) (h : ∀ x, f x = 2 * sin x) :
  (∃ x₁ x₂, ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  (∃ x₁ x₂, ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂ ∧ |x₁ - x₂| = π) ∧
  (∀ x₁ x₂, (∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) → |x₁ - x₂| ≥ π) :=
sorry

end NUMINAMATH_CALUDE_min_difference_sine_extrema_l301_30184


namespace NUMINAMATH_CALUDE_log_sum_property_l301_30146

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_sum_property (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  ∃ (f : ℝ → ℝ) (f_inv : ℝ → ℝ),
    (∀ x > 0, f x = Real.log x / Real.log a) ∧
    (∀ x, f (f_inv x) = x) ∧
    (f_inv 2 = 9) →
    f 9 + f 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_property_l301_30146


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_tenth_squared_l301_30139

theorem decimal_equivalent_of_one_tenth_squared : (1 / 10 : ℚ) ^ 2 = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_tenth_squared_l301_30139


namespace NUMINAMATH_CALUDE_soft_drink_storage_l301_30135

theorem soft_drink_storage (initial_small : ℕ) (initial_big : ℕ) 
  (percent_big_sold : ℚ) (total_remaining : ℕ) :
  initial_small = 6000 →
  initial_big = 15000 →
  percent_big_sold = 14 / 100 →
  total_remaining = 18180 →
  ∃ (percent_small_sold : ℚ),
    percent_small_sold = 12 / 100 ∧
    (initial_small - initial_small * percent_small_sold) +
    (initial_big - initial_big * percent_big_sold) = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_soft_drink_storage_l301_30135


namespace NUMINAMATH_CALUDE_tv_discount_theorem_l301_30109

/-- Represents the price of a TV with successive discounts -/
def discounted_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the final price of a TV after successive discounts is 63% of the original price -/
theorem tv_discount_theorem :
  let original_price : ℝ := 450
  let discount1 : ℝ := 0.30
  let discount2 : ℝ := 0.10
  let final_price := discounted_price original_price discount1 discount2
  final_price / original_price = 0.63 := by
  sorry


end NUMINAMATH_CALUDE_tv_discount_theorem_l301_30109


namespace NUMINAMATH_CALUDE_basketball_team_points_l301_30197

theorem basketball_team_points (x : ℚ) (z : ℕ) : 
  (1 / 3 : ℚ) * x + (3 / 8 : ℚ) * x + 18 + z = x →
  z ≤ 27 →
  z = 21 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_points_l301_30197


namespace NUMINAMATH_CALUDE_average_star_rating_l301_30170

def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

def total_star_points : ℕ := 5 * five_star_reviews + 4 * four_star_reviews + 3 * three_star_reviews + 2 * two_star_reviews

theorem average_star_rating :
  (total_star_points : ℚ) / total_reviews = 4 := by sorry

end NUMINAMATH_CALUDE_average_star_rating_l301_30170


namespace NUMINAMATH_CALUDE_rice_containers_l301_30155

theorem rice_containers (total_weight : ℚ) (container_weight : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 29/4 →
  container_weight = 29 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce / container_weight : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rice_containers_l301_30155


namespace NUMINAMATH_CALUDE_line_slope_l301_30121

/-- Given a line with equation 3y + 2x = 6x - 9, its slope is -4/3 -/
theorem line_slope (x y : ℝ) : 3*y + 2*x = 6*x - 9 → (y - 3 = (-4/3) * (x - 0)) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l301_30121


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l301_30104

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, 
    (10000 ≤ m ∧ m < 100000) ∧ 
    (∃ x : ℕ, m = x^2) ∧ 
    (∃ y : ℕ, m = y^3) → 
    n ≤ m) ∧                  -- least such number
  n = 15625 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l301_30104


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l301_30123

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_first_six_primes_mod_seventh_prime :
  (first_six_primes.sum % seventh_prime) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l301_30123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l301_30133

theorem arithmetic_sequence_product (a : ℤ) : 
  (∃ x : ℤ, x * (x + 1) * (x + 2) * (x + 3) = 360) → 
  (a * (a + 1) * (a + 2) * (a + 3) = 360 → (a = 3 ∨ a = -6)) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l301_30133


namespace NUMINAMATH_CALUDE_cookies_in_blue_tin_l301_30172

/-- Proves that the fraction of cookies in the blue tin is 8/27 -/
theorem cookies_in_blue_tin
  (total_cookies : ℚ)
  (blue_green_fraction : ℚ)
  (red_fraction : ℚ)
  (green_fraction_of_blue_green : ℚ)
  (h1 : blue_green_fraction = 2 / 3)
  (h2 : red_fraction = 1 - blue_green_fraction)
  (h3 : green_fraction_of_blue_green = 5 / 9)
  : (blue_green_fraction * (1 - green_fraction_of_blue_green)) = 8 / 27 := by
  sorry


end NUMINAMATH_CALUDE_cookies_in_blue_tin_l301_30172


namespace NUMINAMATH_CALUDE_probability_after_removal_l301_30126

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : ℕ)
  (each : ℕ)
  (h1 : total = numbers * each)

/-- Calculates the number of ways to choose 2 cards from n cards -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the probability of selecting a pair from the deck after removing two pairs -/
def probability_of_pair (d : Deck) : ℚ :=
  let remaining := d.total - 4
  let total_choices := choose_two remaining
  let pair_choices := (d.numbers - 1) * choose_two d.each
  pair_choices / total_choices

theorem probability_after_removal (d : Deck) 
  (h2 : d.total = 60) 
  (h3 : d.numbers = 12) 
  (h4 : d.each = 5) : 
  probability_of_pair d = 11 / 154 := by
  sorry

end NUMINAMATH_CALUDE_probability_after_removal_l301_30126


namespace NUMINAMATH_CALUDE_base_five_to_decimal_l301_30189

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

/-- The base 5 number 243₅ is equal to 73 in base 10 -/
theorem base_five_to_decimal : to_decimal [2, 4, 3] 5 = 73 := by
  sorry

end NUMINAMATH_CALUDE_base_five_to_decimal_l301_30189


namespace NUMINAMATH_CALUDE_inequality_range_l301_30192

theorem inequality_range (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l301_30192


namespace NUMINAMATH_CALUDE_not_geometric_progression_l301_30180

theorem not_geometric_progression : 
  ¬∃ (a r : ℝ) (p q k : ℤ), 
    p ≠ q ∧ q ≠ k ∧ p ≠ k ∧ 
    a * r^(p-1) = 10 ∧ 
    a * r^(q-1) = 11 ∧ 
    a * r^(k-1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_not_geometric_progression_l301_30180


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l301_30141

/-- Given a parabola with equation y² = 4ax where a < 0, 
    prove that the coordinates of its focus are (a, 0) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a < 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*a*x}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (a, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l301_30141
