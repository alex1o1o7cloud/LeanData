import Mathlib

namespace spade_nested_operation_l2474_247486

/-- The spade operation defined as the absolute difference between two numbers -/
def spade (a b : ℝ) : ℝ := |a - b|

/-- Theorem stating that 3 ♠ (5 ♠ (8 ♠ 11)) = 1 -/
theorem spade_nested_operation : spade 3 (spade 5 (spade 8 11)) = 1 := by
  sorry

end spade_nested_operation_l2474_247486


namespace inverse_proportion_y_relationship_l2474_247475

/-- Proves the relationship between y-coordinates of three points on an inverse proportion function -/
theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ y₃ : ℝ),
  (y₁ = -4 / (-1)) →  -- Point A(-1, y₁) lies on y = -4/x
  (y₂ = -4 / 2) →     -- Point B(2, y₂) lies on y = -4/x
  (y₃ = -4 / 3) →     -- Point C(3, y₃) lies on y = -4/x
  (y₁ > y₃ ∧ y₃ > y₂) := by
  sorry

end inverse_proportion_y_relationship_l2474_247475


namespace final_price_after_reductions_ball_price_reduction_l2474_247454

/-- Calculates the final price of an item after two successive price reductions -/
theorem final_price_after_reductions (original_price : ℝ) 
  (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : 
  original_price * (1 - first_reduction_percent / 100) * (1 - second_reduction_percent / 100) = 8 :=
by
  sorry

/-- The specific case of the ball price reduction problem -/
theorem ball_price_reduction : 
  20 * (1 - 20 / 100) * (1 - 50 / 100) = 8 :=
by
  sorry

end final_price_after_reductions_ball_price_reduction_l2474_247454


namespace seven_digit_sum_2015_l2474_247434

theorem seven_digit_sum_2015 :
  ∃ (a b c d e f g : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧
    (1000 * a + 100 * b + 10 * c + d) + (10 * e + f) + g = 2015 :=
by sorry

end seven_digit_sum_2015_l2474_247434


namespace quadratic_equation_coefficient_l2474_247422

theorem quadratic_equation_coefficient (p q : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + p) = x^2 + q*x + 12) → q = 7 :=
by
  sorry

end quadratic_equation_coefficient_l2474_247422


namespace trigonometric_equation_solution_l2474_247404

theorem trigonometric_equation_solution (k : ℤ) :
  (∃ x : ℝ, 2 + Real.cos x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) + Real.sin (7 * x) ^ 2 = 
   Real.cos (π * k / 2022) ^ 2) ↔ 
  (∃ m : ℤ, k = 2022 * m) ∧
  (∀ x : ℝ, 2 + Real.cos x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) + Real.sin (7 * x) ^ 2 = 
   Real.cos (π * k / 2022) ^ 2 →
   ∃ n : ℤ, x = π / 4 + π * n / 2) :=
by sorry

end trigonometric_equation_solution_l2474_247404


namespace initial_puppies_count_l2474_247485

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℚ := 8.5

/-- The number of puppies Alyssa kept -/
def puppies_kept : ℚ := 12.5

/-- The total number of puppies Alyssa had initially -/
def total_puppies : ℚ := puppies_given_away + puppies_kept

theorem initial_puppies_count : total_puppies = 21 := by
  sorry

end initial_puppies_count_l2474_247485


namespace problem_solution_l2474_247407

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1) / Real.log (1/2)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 6

theorem problem_solution :
  ∀ a : ℝ,
  (∀ x : ℝ, g a x = g a (-x)) →
  (a = 0 ∧ ∀ x > 0, ∀ y > x, g a y > g a x) ∧
  (({x : ℝ | g a x < 0} = {x : ℝ | 2 < x ∧ x < 3}) →
    (∀ x > 1, g a x / (x - 1) ≥ 2 * Real.sqrt 2 - 3) ∧
    (∃ x > 1, g a x / (x - 1) = 2 * Real.sqrt 2 - 3)) ∧
  ((∀ x₁ ≥ 1, ∀ x₂ ∈ Set.Icc (-2) 4, f x₁ ≤ g a x₂) →
    -11/2 ≤ a ∧ a ≤ 2 * Real.sqrt 7) :=
by sorry

end problem_solution_l2474_247407


namespace triangle_property_l2474_247464

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c * Real.cos t.C = (t.a * Real.cos t.B + t.b * Real.cos t.A) / 2)
  (h2 : t.c = 2) :
  t.C = π / 3 ∧ 
  (∀ (t' : Triangle), t'.c = 2 → t.a + t.b + t.c ≥ t'.a + t'.b + t'.c) ∧
  t.a + t.b + t.c = 6 := by
  sorry

end triangle_property_l2474_247464


namespace trigonometric_expression_equals_one_l2474_247425

theorem trigonometric_expression_equals_one (α : Real) (h : Real.tan α = 2) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / (Real.cos (3 * π / 2 + α) + Real.cos (π - α)) = 1 := by
  sorry

end trigonometric_expression_equals_one_l2474_247425


namespace yoga_to_exercise_ratio_l2474_247445

/-- Represents the time spent on different activities in minutes -/
structure ExerciseTime where
  gym : ℕ
  bike : ℕ
  yoga : ℕ

/-- Calculates the total exercise time -/
def totalExerciseTime (t : ExerciseTime) : ℕ := t.gym + t.bike

/-- Represents the ratio of gym to bike time -/
def gymBikeRatio (t : ExerciseTime) : ℚ := t.gym / t.bike

theorem yoga_to_exercise_ratio (t : ExerciseTime) 
  (h1 : gymBikeRatio t = 2/3)
  (h2 : t.bike = 18) :
  ∃ (y : ℕ), t.yoga = y ∧ y / (totalExerciseTime t) = y / 30 := by
  sorry

end yoga_to_exercise_ratio_l2474_247445


namespace angle_between_specific_vectors_l2474_247405

/-- The angle between two 2D vectors -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

/-- Converts degrees to radians -/
def deg_to_rad (deg : ℝ) : ℝ := sorry

theorem angle_between_specific_vectors :
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (-1/2, Real.sqrt 3/2)
  angle_between a b = deg_to_rad 120
  := by sorry

end angle_between_specific_vectors_l2474_247405


namespace binomial_12_3_l2474_247429

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_l2474_247429


namespace walter_bus_time_l2474_247489

def minutes_in_hour : ℕ := 60

def walter_schedule : Prop :=
  let wake_up_time : ℕ := 6 * 60 + 15
  let bus_departure_time : ℕ := 7 * 60
  let class_duration : ℕ := 45
  let num_classes : ℕ := 8
  let lunch_duration : ℕ := 30
  let break_duration : ℕ := 15
  let additional_time : ℕ := 2 * 60
  let return_home_time : ℕ := 16 * 60 + 30
  let total_away_time : ℕ := return_home_time - bus_departure_time
  let school_activities_time : ℕ := num_classes * class_duration + lunch_duration + break_duration + additional_time
  let bus_time : ℕ := total_away_time - school_activities_time
  bus_time = 45

theorem walter_bus_time : walter_schedule := by
  sorry

end walter_bus_time_l2474_247489


namespace geometric_sequence_special_sum_l2474_247413

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_special_sum
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a3 : a 3 = Real.sqrt 2 - 1)
  (h_a5 : a 5 = Real.sqrt 2 + 1) :
  a 3 ^ 2 + 2 * a 2 * a 6 + a 3 * a 7 = 8 :=
sorry

end geometric_sequence_special_sum_l2474_247413


namespace factorization_cubic_minus_linear_l2474_247465

theorem factorization_cubic_minus_linear (a : ℝ) : a^3 - 16*a = a*(a + 4)*(a - 4) := by
  sorry

end factorization_cubic_minus_linear_l2474_247465


namespace nathan_baseball_weeks_l2474_247499

/-- Nathan's baseball playing problem -/
theorem nathan_baseball_weeks (nathan_daily_hours tobias_daily_hours : ℕ) 
  (total_hours : ℕ) (tobias_weeks : ℕ) :
  nathan_daily_hours = 3 →
  tobias_daily_hours = 5 →
  tobias_weeks = 1 →
  total_hours = 77 →
  ∃ w : ℕ, w * (7 * nathan_daily_hours) + tobias_weeks * (7 * tobias_daily_hours) = total_hours ∧ w = 2 := by
  sorry

#check nathan_baseball_weeks

end nathan_baseball_weeks_l2474_247499


namespace min_sum_of_product_l2474_247438

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 1806) :
  ∃ (x y z : ℕ+), x * y * z = 1806 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 153 :=
sorry

end min_sum_of_product_l2474_247438


namespace fraction_equality_l2474_247439

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end fraction_equality_l2474_247439


namespace exists_point_P_satisfying_condition_l2474_247431

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with edge length 10 -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A' : Point3D
  B' : Point3D
  C' : Point3D
  D' : Point3D

/-- Represents the plane intersecting the cube -/
structure IntersectingPlane where
  P : Point3D
  Q : Point3D
  R : Point3D
  S : Point3D
  T : Point3D

/-- Function to calculate distance between two points -/
def distance (p1 p2 : Point3D) : ℝ :=
  sorry

/-- Theorem stating the existence of point P satisfying the condition -/
theorem exists_point_P_satisfying_condition 
  (cube : Cube) 
  (plane : IntersectingPlane) 
  (h1 : distance cube.A plane.R / distance plane.R cube.B = 7 / 3)
  (h2 : distance cube.C plane.S / distance plane.S cube.B = 7 / 3)
  (h3 : plane.P.x = cube.D.x ∧ plane.P.y = cube.D.y)
  (h4 : plane.Q.x = cube.A.x ∧ plane.Q.y = cube.A.y)
  (h5 : plane.R.z = cube.A.z ∧ plane.R.y = cube.A.y)
  (h6 : plane.S.z = cube.B.z ∧ plane.S.x = cube.B.x)
  (h7 : plane.T.x = cube.C.x ∧ plane.T.y = cube.C.y) :
  ∃ (P : Point3D), 
    P.x = cube.D.x ∧ P.y = cube.D.y ∧ 
    cube.D.z ≤ P.z ∧ P.z ≤ cube.D'.z ∧
    2 * distance plane.Q plane.R = distance P plane.Q + distance plane.R plane.S :=
sorry

end exists_point_P_satisfying_condition_l2474_247431


namespace smallest_positive_b_l2474_247423

/-- Circle w1 defined by the equation x^2+y^2+6x-8y-23=0 -/
def w1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y - 23 = 0

/-- Circle w2 defined by the equation x^2+y^2-6x-8y+65=0 -/
def w2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 65 = 0

/-- A circle is externally tangent to w2 -/
def externally_tangent_w2 (x y r : ℝ) : Prop := 
  r + 2 = Real.sqrt ((x - 3)^2 + (y - 4)^2)

/-- A circle is internally tangent to w1 -/
def internally_tangent_w1 (x y r : ℝ) : Prop := 
  6 - r = Real.sqrt ((x + 3)^2 + (y - 4)^2)

/-- The line y = bx contains the center (x, y) of the tangent circle -/
def center_on_line (x y b : ℝ) : Prop := y = b * x

theorem smallest_positive_b : 
  ∃ (b : ℝ), b > 0 ∧ 
  (∀ (b' : ℝ), b' > 0 → 
    (∃ (x y r : ℝ), externally_tangent_w2 x y r ∧ 
                    internally_tangent_w1 x y r ∧ 
                    center_on_line x y b') 
    → b ≤ b') ∧
  b = 1 := by sorry

end smallest_positive_b_l2474_247423


namespace above_x_axis_on_line_l2474_247401

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

-- Statement for the first part of the problem
theorem above_x_axis (m : ℝ) : 
  Complex.im (z m) > 0 ↔ m < -3 ∨ m > 5 := by sorry

-- Statement for the second part of the problem
theorem on_line (m : ℝ) :
  Complex.re (z m) + Complex.im (z m) + 5 = 0 ↔ 
  m = (-3 + Real.sqrt 41) / 4 ∨ m = (-3 - Real.sqrt 41) / 4 := by sorry

end above_x_axis_on_line_l2474_247401


namespace dave_baseball_cards_l2474_247415

/-- Calculates the number of pages required to organize baseball cards in a binder -/
def pages_required (cards_per_page new_cards old_cards : ℕ) : ℕ :=
  (new_cards + old_cards + cards_per_page - 1) / cards_per_page

/-- Proves that Dave will use 2 pages to organize his baseball cards -/
theorem dave_baseball_cards : pages_required 8 3 13 = 2 := by
  sorry

end dave_baseball_cards_l2474_247415


namespace field_trip_vans_l2474_247487

theorem field_trip_vans (buses : ℝ) (people_per_van : ℝ) (people_per_bus : ℝ) 
  (extra_people_in_buses : ℝ) :
  buses = 8 →
  people_per_van = 6 →
  people_per_bus = 18 →
  extra_people_in_buses = 108 →
  ∃ (vans : ℝ), vans = 6 ∧ people_per_bus * buses - people_per_van * vans = extra_people_in_buses :=
by
  sorry

end field_trip_vans_l2474_247487


namespace sum_in_quadrant_IV_l2474_247458

/-- Given complex numbers z₁ and z₂, prove that their sum lies in Quadrant IV -/
theorem sum_in_quadrant_IV (z₁ z₂ : ℂ) : 
  z₁ = 1 - 3*I ∧ z₂ = 3 + 2*I → (z₁ + z₂).re > 0 ∧ (z₁ + z₂).im < 0 :=
by
  sorry

#check sum_in_quadrant_IV

end sum_in_quadrant_IV_l2474_247458


namespace seashells_given_to_jessica_l2474_247479

theorem seashells_given_to_jessica (original_seashells : ℕ) (seashells_left : ℕ) 
  (h1 : original_seashells = 56)
  (h2 : seashells_left = 22)
  (h3 : seashells_left < original_seashells) :
  original_seashells - seashells_left = 34 := by
  sorry

end seashells_given_to_jessica_l2474_247479


namespace range_of_S_3_l2474_247482

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the sum of the first three terms
def S_3 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3

-- Theorem statement
theorem range_of_S_3 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a2 : a 2 = 2) :
  ∀ x : ℝ, (x ∈ Set.Iic (-2) ∪ Set.Ici 6) ↔ ∃ q : ℝ, q ≠ 0 ∧ S_3 a = x :=
sorry

end range_of_S_3_l2474_247482


namespace expected_value_is_six_point_five_l2474_247442

/-- A function representing a fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of a roll of the 12-sided die -/
def expected_value : ℚ := (twelve_sided_die.sum id + twelve_sided_die.card) / (2 * twelve_sided_die.card)

/-- Theorem stating that the expected value of a roll of the 12-sided die is 6.5 -/
theorem expected_value_is_six_point_five : expected_value = 13/2 := by sorry

end expected_value_is_six_point_five_l2474_247442


namespace sum_inverse_g_eq_758_3125_l2474_247400

/-- The function g(n) that returns the integer closest to the cube root of n -/
def g (n : ℕ) : ℕ := sorry

/-- The sum of 1/g(k) from k=1 to 4096 -/
def sum_inverse_g : ℚ :=
  (Finset.range 4096).sum (fun k => 1 / (g (k + 1) : ℚ))

/-- Theorem stating that the sum equals 758.3125 -/
theorem sum_inverse_g_eq_758_3125 : sum_inverse_g = 758.3125 := by sorry

end sum_inverse_g_eq_758_3125_l2474_247400


namespace distance_between_B_and_C_l2474_247456

/-- The distance between two locations in kilometers -/
def distance_between (x y : ℝ) : ℝ := |x - y|

/-- The position of an individual after traveling for a given time -/
def position_after_time (initial_position velocity time : ℝ) : ℝ :=
  initial_position + velocity * time

/-- Arithmetic sequence of four speeds -/
structure ArithmeticSpeedSequence (v₁ v₂ v₃ v₄ : ℝ) : Prop where
  decreasing : v₁ > v₂ ∧ v₂ > v₃ ∧ v₃ > v₄
  arithmetic : ∃ d : ℝ, v₁ - v₂ = d ∧ v₂ - v₃ = d ∧ v₃ - v₄ = d

theorem distance_between_B_and_C
  (vA vB vC vD : ℝ)  -- Speeds of individuals A, B, C, and D
  (n : ℝ)            -- Time when B and C meet
  (h1 : ArithmeticSpeedSequence vA vB vC vD)
  (h2 : position_after_time 0 vB n = position_after_time 60 (-vC) n)  -- B and C meet after n hours
  (h3 : position_after_time 0 vA (2*n) = position_after_time 60 vD (2*n))  -- A catches up with D after 2n hours
  : distance_between 60 (position_after_time 60 (-vC) n) = 30 :=
sorry

end distance_between_B_and_C_l2474_247456


namespace division_problem_l2474_247497

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by
sorry

end division_problem_l2474_247497


namespace geometric_sequence_inequality_l2474_247472

/-- A geometric sequence with positive terms and common ratio greater than 1 -/
structure GeometricSequence where
  b : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, b n > 0
  h_q_gt_one : q > 1
  h_geometric : ∀ n, b (n + 1) = b n * q

/-- In a geometric sequence with positive terms and common ratio greater than 1,
    the sum of the 6th and 7th terms is less than the sum of the 4th and 9th terms -/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  seq.b 6 + seq.b 7 < seq.b 4 + seq.b 9 := by
  sorry

end geometric_sequence_inequality_l2474_247472


namespace exponential_function_max_min_sum_l2474_247495

theorem exponential_function_max_min_sum (a : ℝ) (f : ℝ → ℝ) :
  a > 1 →
  (∀ x, f x = a^x) →
  (∃ max min : ℝ, (∀ x ∈ Set.Icc 0 1, f x ≤ max) ∧
                  (∀ x ∈ Set.Icc 0 1, min ≤ f x) ∧
                  max + min = 3) →
  a = 2 := by
sorry

end exponential_function_max_min_sum_l2474_247495


namespace farm_sections_l2474_247406

theorem farm_sections (section_area : ℝ) (total_area : ℝ) (h1 : section_area = 60) (h2 : total_area = 300) :
  total_area / section_area = 5 := by
  sorry

end farm_sections_l2474_247406


namespace intersection_point_l2474_247477

/-- The line equation y = x + 3 -/
def line_equation (x y : ℝ) : Prop := y = x + 3

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- Theorem: The point (0, 3) is the intersection of the line y = x + 3 and the y-axis -/
theorem intersection_point :
  line_equation 0 3 ∧ on_y_axis 0 3 := by
  sorry

end intersection_point_l2474_247477


namespace dvds_left_l2474_247459

def debby_dvds : ℕ := 13
def sold_dvds : ℕ := 6

theorem dvds_left : debby_dvds - sold_dvds = 7 := by sorry

end dvds_left_l2474_247459


namespace popsicle_stick_difference_l2474_247420

theorem popsicle_stick_difference :
  let num_boys : ℕ := 10
  let num_girls : ℕ := 12
  let sticks_per_boy : ℕ := 15
  let sticks_per_girl : ℕ := 12
  let total_boys_sticks := num_boys * sticks_per_boy
  let total_girls_sticks := num_girls * sticks_per_girl
  total_boys_sticks - total_girls_sticks = 6 := by
  sorry

end popsicle_stick_difference_l2474_247420


namespace water_velocity_proof_l2474_247414

-- Define the relationship between force, height, and velocity
def force_relation (k : ℝ) (H : ℝ) (V : ℝ) : ℝ := k * H * V^3

-- Theorem statement
theorem water_velocity_proof :
  ∀ k : ℝ,
  -- Given conditions
  (force_relation k 1 5 = 100) →
  -- Prove that
  (force_relation k 8 10 = 6400) :=
by
  sorry

end water_velocity_proof_l2474_247414


namespace planes_perpendicular_from_lines_l2474_247473

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_from_lines
  (α β : Plane) (a b : Line)
  (h1 : perpendicular_line_plane a α)
  (h2 : perpendicular_line_plane b β)
  (h3 : perpendicular_line_line a b) :
  perpendicular_plane_plane α β :=
sorry

end planes_perpendicular_from_lines_l2474_247473


namespace cos_50_cos_20_plus_sin_50_sin_20_l2474_247412

theorem cos_50_cos_20_plus_sin_50_sin_20 :
  Real.cos (50 * π / 180) * Real.cos (20 * π / 180) + Real.sin (50 * π / 180) * Real.sin (20 * π / 180) = Real.sqrt 3 / 2 :=
by sorry

end cos_50_cos_20_plus_sin_50_sin_20_l2474_247412


namespace doughnut_machine_completion_time_l2474_247484

-- Define the start time and quarter completion time
def start_time : Nat := 7 * 60  -- 7:00 AM in minutes
def quarter_completion_time : Nat := 10 * 60  -- 10:00 AM in minutes

-- Define the maintenance break duration
def maintenance_break : Nat := 30  -- 30 minutes

-- Theorem to prove
theorem doughnut_machine_completion_time : 
  -- Given conditions
  (quarter_completion_time - start_time = 3 * 60) →  -- 3 hours to complete 1/4 of the job
  -- Conclusion
  (start_time + 4 * (quarter_completion_time - start_time) + maintenance_break = 19 * 60 + 30) :=  -- 7:30 PM in minutes
by
  sorry

end doughnut_machine_completion_time_l2474_247484


namespace parabola_line_intersection_l2474_247491

/-- A point (x, y) on the parabola y^2 = 4x that maintains equal distance from (1, 0) and the line x = -1 -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- A line passing through (-2, 0) with slope k -/
def Line (k x y : ℝ) : Prop := y = k*(x + 2)

/-- The set of k values for which the line intersects the parabola -/
def IntersectionSet : Set ℝ := {k : ℝ | ∃ x y : ℝ, Parabola x y ∧ Line k x y}

theorem parabola_line_intersection :
  IntersectionSet = {k : ℝ | k ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2)} := by sorry

#check parabola_line_intersection

end parabola_line_intersection_l2474_247491


namespace rectangle_ratio_l2474_247424

theorem rectangle_ratio (w : ℝ) (h1 : w > 0) (h2 : 2 * w + 2 * 8 = 24) :
  w / 8 = 1 / 2 := by
  sorry

end rectangle_ratio_l2474_247424


namespace total_dresses_l2474_247403

theorem total_dresses (ana_dresses : ℕ) (lisa_more_dresses : ℕ) : 
  ana_dresses = 15 → lisa_more_dresses = 18 → 
  ana_dresses + (ana_dresses + lisa_more_dresses) = 48 := by
  sorry

end total_dresses_l2474_247403


namespace marigold_fraction_l2474_247474

/-- Represents the composition of flowers in a bouquet -/
structure Bouquet where
  yellow_daisies : ℚ
  white_daisies : ℚ
  yellow_marigolds : ℚ
  white_marigolds : ℚ

/-- The conditions of the flower bouquet problem -/
def bouquet_conditions (b : Bouquet) : Prop :=
  -- Half of the yellow flowers are daisies
  b.yellow_daisies = b.yellow_marigolds ∧
  -- Two-thirds of the white flowers are marigolds
  b.white_marigolds = 2 * b.white_daisies ∧
  -- Four-sevenths of the flowers are yellow
  b.yellow_daisies + b.yellow_marigolds = (4:ℚ)/7 * (b.yellow_daisies + b.white_daisies + b.yellow_marigolds + b.white_marigolds) ∧
  -- All fractions are non-negative
  0 ≤ b.yellow_daisies ∧ 0 ≤ b.white_daisies ∧ 0 ≤ b.yellow_marigolds ∧ 0 ≤ b.white_marigolds ∧
  -- The sum of all fractions is 1
  b.yellow_daisies + b.white_daisies + b.yellow_marigolds + b.white_marigolds = 1

/-- The theorem stating that marigolds constitute 4/7 of the flowers -/
theorem marigold_fraction (b : Bouquet) (h : bouquet_conditions b) :
  b.yellow_marigolds + b.white_marigolds = (4:ℚ)/7 := by
  sorry

end marigold_fraction_l2474_247474


namespace two_problems_without_conditional_l2474_247408

/-- Represents a mathematical problem that may or may not require conditional statements in its algorithm. -/
inductive Problem
| OppositeNumber
| SquarePerimeter
| MaximumOfThree
| FunctionValue

/-- Determines if a problem requires conditional statements in its algorithm. -/
def requiresConditional (p : Problem) : Bool :=
  match p with
  | Problem.OppositeNumber => false
  | Problem.SquarePerimeter => false
  | Problem.MaximumOfThree => true
  | Problem.FunctionValue => true

/-- The list of all problems given in the question. -/
def allProblems : List Problem :=
  [Problem.OppositeNumber, Problem.SquarePerimeter, Problem.MaximumOfThree, Problem.FunctionValue]

/-- Theorem stating that the number of problems not requiring conditional statements is 2. -/
theorem two_problems_without_conditional :
  (allProblems.filter (fun p => ¬requiresConditional p)).length = 2 := by
  sorry


end two_problems_without_conditional_l2474_247408


namespace possible_values_of_a_l2474_247452

theorem possible_values_of_a (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) →
  (a = 8 ∨ a = 12) :=
by sorry

end possible_values_of_a_l2474_247452


namespace system_solution_l2474_247427

theorem system_solution (x y : ℝ) : 
  (x^2 + x*y + y^2 = 37 ∧ x^4 + x^2*y^2 + y^4 = 481) ↔ 
  ((x = -4 ∧ y = -3) ∨ (x = -3 ∧ y = -4) ∨ (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3)) :=
by sorry

end system_solution_l2474_247427


namespace average_marks_combined_classes_l2474_247451

theorem average_marks_combined_classes (n₁ n₂ : ℕ) (avg₁ avg₂ : ℝ) 
  (h₁ : n₁ = 20) (h₂ : n₂ = 50) (h₃ : avg₁ = 40) (h₄ : avg₂ = 60) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℝ) = 3800 / 70 :=
sorry

end average_marks_combined_classes_l2474_247451


namespace certain_number_problem_l2474_247448

theorem certain_number_problem : ∃ x : ℚ, (x * 30 + (12 + 8) * 3) / 5 = 1212 ∧ x = 40 := by
  sorry

end certain_number_problem_l2474_247448


namespace reflections_count_l2474_247447

/-- Number of reflections Sarah sees in tall mirrors -/
def sarah_tall : ℕ := 10

/-- Number of reflections Sarah sees in wide mirrors -/
def sarah_wide : ℕ := 5

/-- Number of reflections Sarah sees in narrow mirrors -/
def sarah_narrow : ℕ := 8

/-- Number of reflections Ellie sees in tall mirrors -/
def ellie_tall : ℕ := 6

/-- Number of reflections Ellie sees in wide mirrors -/
def ellie_wide : ℕ := 3

/-- Number of reflections Ellie sees in narrow mirrors -/
def ellie_narrow : ℕ := 4

/-- Number of times they pass through tall mirrors -/
def times_tall : ℕ := 3

/-- Number of times they pass through wide mirrors -/
def times_wide : ℕ := 5

/-- Number of times they pass through narrow mirrors -/
def times_narrow : ℕ := 4

/-- The total number of reflections seen by Sarah and Ellie -/
def total_reflections : ℕ :=
  (sarah_tall * times_tall + sarah_wide * times_wide + sarah_narrow * times_narrow) +
  (ellie_tall * times_tall + ellie_wide * times_wide + ellie_narrow * times_narrow)

theorem reflections_count : total_reflections = 136 := by
  sorry

end reflections_count_l2474_247447


namespace average_score_is_1_9_l2474_247418

/-- Represents the score distribution for a test -/
structure ScoreDistribution where
  threePoints : Rat
  twoPoints : Rat
  onePoint : Rat
  zeroPoints : Rat

/-- Calculates the average score given a score distribution and number of students -/
def averageScore (dist : ScoreDistribution) (numStudents : ℕ) : ℚ :=
  (3 * dist.threePoints + 2 * dist.twoPoints + dist.onePoint) * numStudents / 100

/-- Theorem: The average score for the given test is 1.9 -/
theorem average_score_is_1_9 :
  let dist : ScoreDistribution := {
    threePoints := 30,
    twoPoints := 40,
    onePoint := 20,
    zeroPoints := 10
  }
  averageScore dist 30 = 19/10 := by
  sorry

end average_score_is_1_9_l2474_247418


namespace second_bus_ride_time_l2474_247468

theorem second_bus_ride_time (waiting_time first_bus_time : ℕ) 
  (h1 : waiting_time = 12)
  (h2 : first_bus_time = 30)
  (h3 : ∀ x, x = (waiting_time + first_bus_time) / 2 → x = 21) :
  ∃ second_bus_time : ℕ, second_bus_time = 21 :=
by
  sorry

end second_bus_ride_time_l2474_247468


namespace fraction_of_time_at_4kmh_l2474_247488

/-- Represents the walking scenario described in the problem -/
structure WalkScenario where
  totalTime : ℝ
  timeAt2kmh : ℝ
  timeAt3kmh : ℝ
  timeAt4kmh : ℝ
  distanceAt2kmh : ℝ
  distanceAt3kmh : ℝ
  distanceAt4kmh : ℝ

/-- Theorem stating the fraction of time walked at 4 km/h -/
theorem fraction_of_time_at_4kmh (w : WalkScenario) : 
  w.timeAt2kmh = w.totalTime / 2 →
  w.distanceAt3kmh = (w.distanceAt2kmh + w.distanceAt3kmh + w.distanceAt4kmh) / 2 →
  w.distanceAt2kmh = 2 * w.timeAt2kmh →
  w.distanceAt3kmh = 3 * w.timeAt3kmh →
  w.distanceAt4kmh = 4 * w.timeAt4kmh →
  w.totalTime = w.timeAt2kmh + w.timeAt3kmh + w.timeAt4kmh →
  w.timeAt4kmh / w.totalTime = 1 / 14 := by
  sorry

end fraction_of_time_at_4kmh_l2474_247488


namespace group_size_proof_l2474_247476

theorem group_size_proof (n : ℕ) 
  (h1 : (40 - 20 : ℝ) / n = 2.5) : n = 8 := by
  sorry

#check group_size_proof

end group_size_proof_l2474_247476


namespace compute_M_l2474_247466

def M : ℕ → ℕ 
| 0 => 0
| (n + 1) => 
  let k := 4 * n + 2
  (k + 2)^2 + k^2 - 2*((k + 1)^2) - 2*((k - 1)^2) + M n

theorem compute_M : M 12 = 75 := by
  sorry

end compute_M_l2474_247466


namespace gcd_problem_l2474_247498

theorem gcd_problem (n : ℕ) : 
  30 ≤ n ∧ n ≤ 40 ∧ Nat.gcd 15 n = 5 → n = 35 ∨ n = 40 := by
  sorry

end gcd_problem_l2474_247498


namespace largest_consecutive_sum_is_eight_l2474_247421

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of n consecutive integers starting from a -/
def sum_consecutive (a n : ℕ) : ℕ := n * a + sum_first_n (n - 1)

/-- The proposition that n is the largest number of positive consecutive integers summing to 36 -/
def is_largest_consecutive_sum (n : ℕ) : Prop :=
  (∃ a : ℕ, a > 0 ∧ sum_consecutive a n = 36) ∧
  (∀ m : ℕ, m > n → ∀ a : ℕ, a > 0 → sum_consecutive a m ≠ 36)

theorem largest_consecutive_sum_is_eight :
  is_largest_consecutive_sum 8 := by
  sorry

end largest_consecutive_sum_is_eight_l2474_247421


namespace chairs_built_in_ten_days_l2474_247433

/-- Represents the time it takes to build different types of chairs -/
structure ChairTimes where
  rocking : ℕ
  dining : ℕ
  armchair : ℕ

/-- Represents the number of chairs built -/
structure ChairsBuilt where
  rocking : ℕ
  dining : ℕ
  armchair : ℕ

/-- Calculates the maximum number of chairs that can be built in a given number of days -/
def maxChairsBuilt (shiftLength : ℕ) (times : ChairTimes) (days : ℕ) : ChairsBuilt :=
  sorry

/-- Theorem stating the maximum number of chairs that can be built in 10 days -/
theorem chairs_built_in_ten_days :
  let times : ChairTimes := ⟨5, 3, 6⟩
  let result : ChairsBuilt := maxChairsBuilt 8 times 10
  result.rocking = 10 ∧ result.dining = 10 ∧ result.armchair = 0 := by
  sorry

end chairs_built_in_ten_days_l2474_247433


namespace sixtieth_pair_is_five_seven_l2474_247494

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- The sequence of integer pairs sorted by sum and then by first element -/
def sortedPairs : List IntPair := sorry

/-- The 60th element in the sortedPairs sequence -/
def sixtiethPair : IntPair := sorry

/-- Theorem stating that the 60th pair in the sequence is (5,7) -/
theorem sixtieth_pair_is_five_seven : 
  sixtiethPair = IntPair.mk 5 7 := by sorry

end sixtieth_pair_is_five_seven_l2474_247494


namespace batsman_total_matches_l2474_247461

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  initial_matches : ℕ
  initial_average : ℝ
  additional_matches : ℕ
  additional_average : ℝ
  overall_average : ℝ

/-- Calculates the total number of matches played by a batsman -/
def total_matches (performance : BatsmanPerformance) : ℕ :=
  performance.initial_matches + performance.additional_matches

/-- Theorem stating that given the specific performance, the total matches played is 30 -/
theorem batsman_total_matches (performance : BatsmanPerformance) 
  (h1 : performance.initial_matches = 20)
  (h2 : performance.initial_average = 40)
  (h3 : performance.additional_matches = 10)
  (h4 : performance.additional_average = 13)
  (h5 : performance.overall_average = 31) :
  total_matches performance = 30 := by
  sorry


end batsman_total_matches_l2474_247461


namespace sin_max_min_difference_l2474_247469

theorem sin_max_min_difference (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 9 → f x = 2 * Real.sin (π * x / 6 - π / 3)) →
  (⨆ x ∈ Set.Icc 0 9, f x) - (⨅ x ∈ Set.Icc 0 9, f x) = 2 + Real.sqrt 3 := by
  sorry

end sin_max_min_difference_l2474_247469


namespace fifth_month_sales_l2474_247480

def sales_problem (month1 month2 month3 month4 month6 : ℕ) (target_average : ℕ) : ℕ :=
  6 * target_average - (month1 + month2 + month3 + month4 + month6)

theorem fifth_month_sales :
  sales_problem 6635 6927 6855 7230 4791 6500 = 6562 := by
  sorry

end fifth_month_sales_l2474_247480


namespace race_heartbeats_l2474_247453

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  heart_rate * pace * race_distance

theorem race_heartbeats :
  total_heartbeats 140 6 30 = 25200 := by
  sorry

end race_heartbeats_l2474_247453


namespace gary_remaining_money_l2474_247440

/-- Calculates the remaining money after a purchase. -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Proves that Gary has 18 dollars left after his purchase. -/
theorem gary_remaining_money :
  remaining_money 73 55 = 18 := by
  sorry

end gary_remaining_money_l2474_247440


namespace probability_at_least_one_defective_l2474_247483

theorem probability_at_least_one_defective (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (h1 : total_bulbs = 23) (h2 : defective_bulbs = 4) :
  let non_defective := total_bulbs - defective_bulbs
  let prob_both_non_defective := (non_defective / total_bulbs) * ((non_defective - 1) / (total_bulbs - 1))
  1 - prob_both_non_defective = 164 / 506 := by
  sorry

end probability_at_least_one_defective_l2474_247483


namespace initial_population_village1_is_correct_l2474_247467

/-- The initial population of the first village -/
def initial_population_village1 : ℕ := 78000

/-- The yearly decrease in population of the first village -/
def yearly_decrease_village1 : ℕ := 1200

/-- The initial population of the second village -/
def initial_population_village2 : ℕ := 42000

/-- The yearly increase in population of the second village -/
def yearly_increase_village2 : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 18

/-- Theorem stating that the initial population of the first village is correct -/
theorem initial_population_village1_is_correct :
  initial_population_village1 - years_until_equal * yearly_decrease_village1 =
  initial_population_village2 + years_until_equal * yearly_increase_village2 :=
by sorry

end initial_population_village1_is_correct_l2474_247467


namespace inequality_equivalence_l2474_247478

theorem inequality_equivalence (x : ℝ) : 
  -2 < (x^2 - 18*x + 24) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 18*x + 24) / (x^2 - 4*x + 8) < 2 ↔ 
  -2 < x ∧ x < 10/3 :=
by sorry

end inequality_equivalence_l2474_247478


namespace composite_function_evaluation_l2474_247416

/-- Given two functions f and g, prove that f(g(f(3))) = 79 -/
theorem composite_function_evaluation :
  let f : ℝ → ℝ := λ x ↦ 2 * x + 5
  let g : ℝ → ℝ := λ x ↦ 3 * x + 4
  f (g (f 3)) = 79 := by
  sorry

end composite_function_evaluation_l2474_247416


namespace no_positive_integer_solutions_l2474_247446

theorem no_positive_integer_solutions :
  ∀ x : ℕ+, ¬(15 < -3 * (x : ℤ) + 18) := by
  sorry

end no_positive_integer_solutions_l2474_247446


namespace subtraction_result_l2474_247436

theorem subtraction_result (chosen_number : ℕ) : 
  chosen_number = 127 → (2 * chosen_number) - 152 = 102 := by
  sorry

end subtraction_result_l2474_247436


namespace max_area_rectangle_l2474_247463

/-- The maximum area of a rectangle with integer side lengths and perimeter 150 feet is 1406 square feet. -/
theorem max_area_rectangle (w h : ℕ) : 
  w + h = 75 → w * h ≤ 1406 :=
by sorry

end max_area_rectangle_l2474_247463


namespace fantasy_creatures_gala_handshakes_l2474_247411

-- Define the number of gremlins and imps
def num_gremlins : ℕ := 30
def num_imps : ℕ := 20

-- Define the number of imps each imp shakes hands with
def imp_imp_handshakes : ℕ := 5

-- Calculate the number of handshakes between gremlins
def gremlin_gremlin_handshakes : ℕ := num_gremlins * (num_gremlins - 1) / 2

-- Calculate the number of handshakes between imps
def imp_imp_total_handshakes : ℕ := num_imps * imp_imp_handshakes / 2

-- Calculate the number of handshakes between gremlins and imps
def gremlin_imp_handshakes : ℕ := num_gremlins * num_imps

-- Define the total number of handshakes
def total_handshakes : ℕ := gremlin_gremlin_handshakes + imp_imp_total_handshakes + gremlin_imp_handshakes

-- Theorem statement
theorem fantasy_creatures_gala_handshakes : total_handshakes = 1085 := by
  sorry

end fantasy_creatures_gala_handshakes_l2474_247411


namespace shortest_distance_is_eight_fifths_l2474_247481

/-- Square ABCD with side length 2 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : side_length = 2)

/-- Circular arc with center A from B to D -/
structure CircularArc (sq : Square) :=
  (center : ℝ × ℝ)
  (start_point : ℝ × ℝ)
  (end_point : ℝ × ℝ)
  (is_valid : center = sq.A ∧ start_point = sq.B ∧ end_point = sq.D)

/-- Semicircle with center at midpoint of CD, from C to D -/
structure Semicircle (sq : Square) :=
  (center : ℝ × ℝ)
  (start_point : ℝ × ℝ)
  (end_point : ℝ × ℝ)
  (is_valid : center = ((sq.C.1 + sq.D.1) / 2, (sq.C.2 + sq.D.2) / 2) ∧ 
              start_point = sq.C ∧ end_point = sq.D)

/-- Intersection point of the circular arc and semicircle -/
def intersectionPoint (sq : Square) (arc : CircularArc sq) (semi : Semicircle sq) : ℝ × ℝ := sorry

/-- Shortest distance from a point to a line segment -/
def shortestDistance (point : ℝ × ℝ) (segment_start : ℝ × ℝ) (segment_end : ℝ × ℝ) : ℝ := sorry

/-- Main theorem: The shortest distance from the intersection point to AD is 8/5 -/
theorem shortest_distance_is_eight_fifths (sq : Square) 
  (arc : CircularArc sq) (semi : Semicircle sq) :
  shortestDistance (intersectionPoint sq arc semi) sq.A sq.D = 8/5 := by sorry

end shortest_distance_is_eight_fifths_l2474_247481


namespace binomial_coefficient_divisibility_l2474_247492

theorem binomial_coefficient_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  p ∣ Nat.choose n p := by
  sorry

end binomial_coefficient_divisibility_l2474_247492


namespace reflect_F_l2474_247435

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Theorem: Reflecting point F(3, 3) over y-axis then x-axis results in F''(-3, -3) -/
theorem reflect_F : 
  let F : ℝ × ℝ := (3, 3)
  reflect_x (reflect_y F) = (-3, -3) := by
sorry

end reflect_F_l2474_247435


namespace sum_gcd_lcm_factorial_l2474_247490

theorem sum_gcd_lcm_factorial : 
  Nat.gcd 48 180 + Nat.lcm 48 180 + Nat.factorial 4 = 756 := by
  sorry

end sum_gcd_lcm_factorial_l2474_247490


namespace complement_of_A_in_U_l2474_247441

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 6, 7} := by sorry

end complement_of_A_in_U_l2474_247441


namespace spelling_bee_contestants_l2474_247471

theorem spelling_bee_contestants (initial_students : ℕ) : 
  (initial_students / 2 : ℚ) / 3 = 24 → initial_students = 144 := by
  sorry

end spelling_bee_contestants_l2474_247471


namespace problem_solution_l2474_247417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

noncomputable def g (x : ℝ) : ℝ := x - 2 * Real.log x - 1

theorem problem_solution :
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → f a x ≥ 1 + Real.log 2) ∧
  (∀ (x : ℝ), x > 0 → HasDerivAt g ((x - 2) / x) x) ∧
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ → (x₁ - x₂) / (Real.log x₁ - Real.log x₂) < 2 * x₂) := by
  sorry

end problem_solution_l2474_247417


namespace min_value_of_f_l2474_247462

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end min_value_of_f_l2474_247462


namespace union_of_sets_l2474_247444

theorem union_of_sets : 
  let A : Set ℕ := {1,2,3,4}
  let B : Set ℕ := {2,4,5}
  A ∪ B = {1,2,3,4,5} := by sorry

end union_of_sets_l2474_247444


namespace no_stop_probability_theorem_l2474_247449

/-- Represents the probability of a green light at a traffic point -/
def greenLightProbability (duration : ℕ) : ℚ := duration / 60

/-- The probability that a car doesn't stop at all three points -/
def noStopProbability (durationA durationB durationC : ℕ) : ℚ :=
  (greenLightProbability durationA) * (greenLightProbability durationB) * (greenLightProbability durationC)

theorem no_stop_probability_theorem (durationA durationB durationC : ℕ) 
  (hA : durationA = 25) (hB : durationB = 35) (hC : durationC = 45) :
  noStopProbability durationA durationB durationC = 35 / 192 := by
  sorry

end no_stop_probability_theorem_l2474_247449


namespace salary_increase_percentage_l2474_247402

/-- Given an original salary and a final salary after an increase followed by a decrease,
    calculate the initial percentage increase. -/
theorem salary_increase_percentage
  (original_salary : ℝ)
  (final_salary : ℝ)
  (decrease_percentage : ℝ)
  (h1 : original_salary = 6000)
  (h2 : final_salary = 6270)
  (h3 : decrease_percentage = 5)
  : ∃ x : ℝ,
    final_salary = original_salary * (1 + x / 100) * (1 - decrease_percentage / 100) ∧
    x = 10 := by
  sorry

end salary_increase_percentage_l2474_247402


namespace monthly_income_calculation_l2474_247496

/-- Proves that if a deposit of 5000 is 20% of a person's monthly income, then their monthly income is 25000. -/
theorem monthly_income_calculation (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 5000)
  (h2 : percentage = 20)
  (h3 : deposit = (percentage / 100) * monthly_income) :
  monthly_income = 25000 := by
  sorry

end monthly_income_calculation_l2474_247496


namespace cubic_sum_theorem_l2474_247437

theorem cubic_sum_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_eq : (p^3 + 7) / p = (q^3 + 7) / q ∧ (q^3 + 7) / q = (r^3 + 7) / r) : 
  p^3 + q^3 + r^3 = -21 := by
  sorry

end cubic_sum_theorem_l2474_247437


namespace additional_buffaloes_count_l2474_247428

/-- Represents the daily fodder consumption of one buffalo -/
def buffalo_consumption : ℚ := 1

/-- Represents the daily fodder consumption of one cow -/
def cow_consumption : ℚ := 3/4 * buffalo_consumption

/-- Represents the daily fodder consumption of one ox -/
def ox_consumption : ℚ := 3/2 * buffalo_consumption

/-- Represents the initial number of buffaloes -/
def initial_buffaloes : ℕ := 15

/-- Represents the initial number of oxen -/
def initial_oxen : ℕ := 8

/-- Represents the initial number of cows -/
def initial_cows : ℕ := 24

/-- Represents the initial duration of fodder in days -/
def initial_duration : ℕ := 24

/-- Represents the number of additional cows -/
def additional_cows : ℕ := 60

/-- Represents the new duration of fodder in days -/
def new_duration : ℕ := 9

/-- Theorem stating that the number of additional buffaloes is 30 -/
theorem additional_buffaloes_count : 
  ∃ (x : ℕ), 
    (initial_buffaloes * buffalo_consumption + 
     initial_oxen * ox_consumption + 
     initial_cows * cow_consumption) * initial_duration =
    ((initial_buffaloes + x) * buffalo_consumption + 
     initial_oxen * ox_consumption + 
     (initial_cows + additional_cows) * cow_consumption) * new_duration ∧
    x = 30 := by sorry

end additional_buffaloes_count_l2474_247428


namespace exponent_product_equality_l2474_247457

theorem exponent_product_equality : 
  (10 ^ 0.4) * (10 ^ 0.1) * (10 ^ 0.7) * (10 ^ 0.2) * (10 ^ 0.6) * (5 ^ 2) = 2500 := by
  sorry

end exponent_product_equality_l2474_247457


namespace negative_odd_number_representation_l2474_247450

theorem negative_odd_number_representation (x : ℤ) :
  (x < 0 ∧ x % 2 = 1) → ∃ n : ℕ+, x = -2 * n + 1 := by
  sorry

end negative_odd_number_representation_l2474_247450


namespace area_relationship_l2474_247410

/-- A circle circumscribed about a right triangle with sides 12, 35, and 37 -/
structure CircumscribedTriangle where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The area of the largest non-triangular region -/
  C : ℝ
  /-- The sum of the areas of the two smaller non-triangular regions -/
  A_plus_B : ℝ
  /-- The radius is half of the hypotenuse -/
  radius_eq : radius = 37 / 2
  /-- The largest non-triangular region is a semicircle -/
  C_eq : C = π * radius^2 / 2
  /-- The sum of all regions equals the circle's area -/
  area_eq : A_plus_B + 210 + C = π * radius^2

/-- The relationship between the areas of the non-triangular regions -/
theorem area_relationship (t : CircumscribedTriangle) : t.A_plus_B + 210 = t.C := by
  sorry

end area_relationship_l2474_247410


namespace product_expansion_l2474_247426

theorem product_expansion (x : ℝ) : (9*x + 2) * (4*x^2 + 3) = 36*x^3 + 8*x^2 + 27*x + 6 := by
  sorry

end product_expansion_l2474_247426


namespace constant_quantity_l2474_247419

/-- A sequence of real numbers satisfying the recurrence relation a_{n+2} = a_{n+1} + a_n -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) + a n

/-- The theorem stating that |a_n^2 - a_{n-1} a_{n+1}| is constant for n ≥ 2 -/
theorem constant_quantity (a : ℕ → ℝ) (h : RecurrenceSequence a) :
  ∃ c : ℝ, ∀ n : ℕ, n ≥ 2 → |a n ^ 2 - a (n - 1) * a (n + 1)| = c :=
sorry

end constant_quantity_l2474_247419


namespace double_yolked_eggs_l2474_247430

/-- Given a carton of eggs with some double-yolked eggs, calculate the number of double-yolked eggs. -/
theorem double_yolked_eggs (total_eggs : ℕ) (total_yolks : ℕ) (double_yolked : ℕ) : 
  total_eggs = 12 → total_yolks = 17 → double_yolked = 5 → 
  2 * double_yolked + (total_eggs - double_yolked) = total_yolks := by
  sorry

#check double_yolked_eggs

end double_yolked_eggs_l2474_247430


namespace stratified_sampling_elderly_count_l2474_247460

theorem stratified_sampling_elderly_count 
  (total_employees : ℕ) 
  (young_employees : ℕ) 
  (middle_aged_employees : ℕ) 
  (elderly_employees : ℕ) 
  (sample_size : ℕ) :
  total_employees = young_employees + middle_aged_employees + elderly_employees →
  total_employees = 550 →
  young_employees = 300 →
  middle_aged_employees = 150 →
  elderly_employees = 100 →
  sample_size = 33 →
  (elderly_employees * sample_size) / total_employees = 6 :=
by sorry

end stratified_sampling_elderly_count_l2474_247460


namespace quadratic_discriminant_l2474_247470

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- For the quadratic equation 3x^2 - 2x - 1 = 0, the discriminant equals 16 -/
theorem quadratic_discriminant : discriminant 3 (-2) (-1) = 16 := by
  sorry

end quadratic_discriminant_l2474_247470


namespace inverse_composition_l2474_247432

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the given condition
axiom inverse_relation (x : ℝ) : (f⁻¹ ∘ g) x = 4 * x - 1

-- State the theorem to be proved
theorem inverse_composition : g⁻¹ (f 5) = 3/2 := by
  sorry

end inverse_composition_l2474_247432


namespace arithmetic_geometric_ratio_l2474_247409

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    where a₁, a₃, a₉ form a geometric sequence,
    prove that (a₁ + a₃ + a₉) / (a₂ + a₄ + a₁₀) = 13/16. -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℚ) 
  (d : ℚ) 
  (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : (a 3) ^ 2 = a 1 * a 9) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 := by
  sorry

end arithmetic_geometric_ratio_l2474_247409


namespace anns_shopping_problem_l2474_247443

theorem anns_shopping_problem (total_spent : ℝ) (shorts_price : ℝ) (shorts_count : ℕ) 
  (shoes_price : ℝ) (shoes_count : ℕ) (tops_count : ℕ) :
  total_spent = 75 →
  shorts_price = 7 →
  shorts_count = 5 →
  shoes_price = 10 →
  shoes_count = 2 →
  tops_count = 4 →
  (total_spent - (shorts_price * shorts_count + shoes_price * shoes_count)) / tops_count = 5 := by
sorry

end anns_shopping_problem_l2474_247443


namespace parallel_lines_sum_l2474_247493

/-- Two parallel lines with a given distance -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  distance : ℝ
  is_parallel : m = 8
  satisfies_distance : distance = 3

/-- The sum m + n for parallel lines with the given properties is either 48 or -12 -/
theorem parallel_lines_sum (lines : ParallelLines) : lines.m + lines.n = 48 ∨ lines.m + lines.n = -12 := by
  sorry

end parallel_lines_sum_l2474_247493


namespace weekly_pig_feed_l2474_247455

def feed_per_pig_per_day : ℕ := 10
def number_of_pigs : ℕ := 2
def days_in_week : ℕ := 7

theorem weekly_pig_feed : 
  feed_per_pig_per_day * number_of_pigs * days_in_week = 140 := by
  sorry

end weekly_pig_feed_l2474_247455
