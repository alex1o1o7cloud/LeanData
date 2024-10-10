import Mathlib

namespace cricket_player_innings_l1951_195107

/-- A cricket player's innings problem -/
theorem cricket_player_innings (current_average : ℚ) (next_innings_runs : ℕ) (average_increase : ℚ) :
  current_average = 25 →
  next_innings_runs = 121 →
  average_increase = 6 →
  (∃ n : ℕ, (n * current_average + next_innings_runs) / (n + 1) = current_average + average_increase ∧ n = 15) :=
by sorry

end cricket_player_innings_l1951_195107


namespace expression_value_l1951_195180

theorem expression_value
  (a b c d e : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |e| = 1) :
  e^2 + 2023*c*d - (a + b)/20 = 2024 := by
  sorry

end expression_value_l1951_195180


namespace expected_weekly_rainfall_l1951_195161

/-- Represents the daily rainfall probabilities and amounts -/
structure DailyRainfall where
  no_rain_prob : Real
  light_rain_prob : Real
  heavy_rain_prob : Real
  light_rain_amount : Real
  heavy_rain_amount : Real

/-- Calculates the expected rainfall for a single day -/
def expected_daily_rainfall (d : DailyRainfall) : Real :=
  d.no_rain_prob * 0 + d.light_rain_prob * d.light_rain_amount + d.heavy_rain_prob * d.heavy_rain_amount

/-- Theorem: Expected total rainfall for a week -/
theorem expected_weekly_rainfall (d : DailyRainfall)
  (h1 : d.no_rain_prob = 0.2)
  (h2 : d.light_rain_prob = 0.3)
  (h3 : d.heavy_rain_prob = 0.5)
  (h4 : d.light_rain_amount = 2)
  (h5 : d.heavy_rain_amount = 8)
  (h6 : d.no_rain_prob + d.light_rain_prob + d.heavy_rain_prob = 1) :
  7 * (expected_daily_rainfall d) = 32.2 := by
  sorry

#eval 7 * (0.2 * 0 + 0.3 * 2 + 0.5 * 8)

end expected_weekly_rainfall_l1951_195161


namespace function_inequality_l1951_195188

noncomputable def f (x : ℝ) : ℝ := (Real.exp 2 * x^2 + 1) / x

noncomputable def g (x : ℝ) : ℝ := (Real.exp 2 * x^2) / Real.exp x

theorem function_inequality (k : ℝ) (hk : k > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → g x₁ / k ≤ f x₂ / (k + 1)) →
  k ≥ 4 / (2 * Real.exp 1 - 4) :=
by sorry

end function_inequality_l1951_195188


namespace cube_volume_in_pyramid_l1951_195178

/-- Represents a pyramid with a square base and isosceles right triangle lateral faces -/
structure Pyramid where
  base_side_length : ℝ
  is_square_base : base_side_length = 2
  is_isosceles_right_triangle_faces : True

/-- Represents a cube inside the pyramid -/
structure InsideCube where
  edge_length : ℝ
  vertex_at_base_center : True
  three_vertices_touch_faces : True

/-- The volume of the cube inside the pyramid is 1 -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : InsideCube) : c.edge_length ^ 3 = 1 := by
  sorry

#check cube_volume_in_pyramid

end cube_volume_in_pyramid_l1951_195178


namespace trig_identity_equivalence_l1951_195130

theorem trig_identity_equivalence (x : ℝ) :
  (2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x)) ↔
  (∃ k : ℤ, x = (π / 16) * (4 * ↑k + 1)) :=
sorry

end trig_identity_equivalence_l1951_195130


namespace function_property_l1951_195136

theorem function_property (f : ℝ → ℝ) (k : ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + k * f x * y) :
  ∃ (a b : ℝ), (a = 0 ∧ b = 4) ∧ 
    (f 2 = a ∨ f 2 = b) ∧
    (∀ c : ℝ, f 2 = c → (c = a ∨ c = b)) := by
  sorry

end function_property_l1951_195136


namespace trail_mix_composition_l1951_195196

/-- The weight of peanuts used in the trail mix -/
def peanuts : ℚ := 0.16666666666666666

/-- The weight of raisins used in the trail mix -/
def raisins : ℚ := 0.08333333333333333

/-- The total weight of the trail mix -/
def total_mix : ℚ := 0.4166666666666667

/-- The weight of chocolate chips used in the trail mix -/
def chocolate_chips : ℚ := total_mix - (peanuts + raisins)

theorem trail_mix_composition :
  chocolate_chips = 0.1666666666666667 := by
  sorry

end trail_mix_composition_l1951_195196


namespace complex_point_location_l1951_195198

theorem complex_point_location (z : ℂ) : 
  (2 + Complex.I) * z = Complex.abs (1 - 2 * Complex.I) →
  Real.sign (z.re) > 0 ∧ Real.sign (z.im) < 0 :=
by sorry

end complex_point_location_l1951_195198


namespace count_rectangles_l1951_195184

/-- The number of checkered rectangles containing exactly one gray cell -/
def num_rectangles (total_gray_cells : ℕ) (blue_cells : ℕ) (red_cells : ℕ) 
  (rectangles_per_blue : ℕ) (rectangles_per_red : ℕ) : ℕ :=
  blue_cells * rectangles_per_blue + red_cells * rectangles_per_red

/-- Theorem stating the number of checkered rectangles containing exactly one gray cell -/
theorem count_rectangles : 
  num_rectangles 40 36 4 4 8 = 176 := by
  sorry

end count_rectangles_l1951_195184


namespace quilt_square_transformation_l1951_195165

theorem quilt_square_transformation (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ (side : ℝ), side^2 = length * width ∧ side = 12 := by
sorry

end quilt_square_transformation_l1951_195165


namespace inner_triangle_perimeter_is_330_75_l1951_195186

/-- Triangle ABC with given side lengths and parallel lines forming a new triangle -/
structure TriangleWithParallelLines where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Lengths of segments formed by parallel lines
  ℓA_length : ℝ
  ℓB_length : ℝ
  ℓC_length : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  AC_positive : AC > 0
  ℓA_positive : ℓA_length > 0
  ℓB_positive : ℓB_length > 0
  ℓC_positive : ℓC_length > 0
  triangle_inequality : AB + BC > AC ∧ BC + AC > AB ∧ AC + AB > BC
  ℓA_inside : ℓA_length < BC
  ℓB_inside : ℓB_length < AC
  ℓC_inside : ℓC_length < AB

/-- The perimeter of the triangle formed by parallel lines -/
def innerTrianglePerimeter (t : TriangleWithParallelLines) : ℝ :=
  sorry

/-- Theorem stating that for the given triangle and parallel lines, the inner triangle perimeter is 330.75 -/
theorem inner_triangle_perimeter_is_330_75 
  (t : TriangleWithParallelLines) 
  (h1 : t.AB = 150) 
  (h2 : t.BC = 270) 
  (h3 : t.AC = 210) 
  (h4 : t.ℓA_length = 65) 
  (h5 : t.ℓB_length = 60) 
  (h6 : t.ℓC_length = 20) : 
  innerTrianglePerimeter t = 330.75 := by
  sorry

end inner_triangle_perimeter_is_330_75_l1951_195186


namespace part_one_part_two_l1951_195110

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f a x < 4 - 2*a ↔ -4 < x ∧ x < 4) → a = 0 := by sorry

-- Part 2
theorem part_two : 
  (∀ x : ℝ, f 1 x - f 1 (-2*x) ≤ x + 2) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, f 1 x - f 1 (-2*x) ≤ x + m) → m ≥ 2) := by sorry

end part_one_part_two_l1951_195110


namespace log_base_2_derivative_l1951_195144

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) :=
by sorry

end log_base_2_derivative_l1951_195144


namespace parallel_lines_x_intercept_l1951_195145

/-- Given two lines that are parallel, prove that the x-intercept of one line is -1. -/
theorem parallel_lines_x_intercept (m : ℝ) :
  (∀ x y, y + m * (x + 1) = 0 ↔ m * y - (2 * m + 1) * x = 1) →
  (m ≠ 0 ∧ 2 * m + 1 ≠ 0) →
  ∃ x, x + m * (x + 1) = 0 ∧ x = -1 :=
by sorry

end parallel_lines_x_intercept_l1951_195145


namespace yan_distance_ratio_l1951_195127

/-- Yan's scenario with distances and speeds -/
structure YanScenario where
  a : ℝ  -- distance from Yan to home
  b : ℝ  -- distance from Yan to mall
  w : ℝ  -- Yan's walking speed
  bike_speed : ℝ -- Yan's bicycle speed

/-- The conditions of Yan's scenario -/
def valid_scenario (s : YanScenario) : Prop :=
  s.a > 0 ∧ s.b > 0 ∧ s.w > 0 ∧
  s.bike_speed = 5 * s.w ∧
  s.b / s.w = s.a / s.w + (s.a + s.b) / s.bike_speed

/-- The theorem stating the ratio of distances -/
theorem yan_distance_ratio (s : YanScenario) (h : valid_scenario s) :
  s.a / s.b = 2 / 3 :=
sorry

end yan_distance_ratio_l1951_195127


namespace greatest_three_digit_multiple_of_17_l1951_195119

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n < 1000 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l1951_195119


namespace mangoes_quantity_l1951_195197

/-- The quantity of mangoes purchased by Harkamal -/
def mangoes_kg : ℕ := sorry

/-- The price of grapes per kg -/
def grapes_price : ℕ := 70

/-- The price of mangoes per kg -/
def mangoes_price : ℕ := 45

/-- The quantity of grapes purchased in kg -/
def grapes_kg : ℕ := 8

/-- The total amount paid -/
def total_paid : ℕ := 965

theorem mangoes_quantity :
  grapes_kg * grapes_price + mangoes_kg * mangoes_price = total_paid ∧
  mangoes_kg = 9 := by sorry

end mangoes_quantity_l1951_195197


namespace airplane_distance_difference_l1951_195172

/-- The difference in distance traveled by an airplane flying without wind for 4 hours
    and against a 20 km/h wind for 3 hours, given that the airplane's windless speed is a km/h. -/
theorem airplane_distance_difference (a : ℝ) : 
  4 * a - (3 * (a - 20)) = a + 60 := by
  sorry

end airplane_distance_difference_l1951_195172


namespace car_distance_theorem_l1951_195155

/-- Represents a car traveling between two points --/
structure Car where
  speed_forward : ℝ
  speed_backward : ℝ

/-- The problem setup --/
def problem_setup : Prop :=
  ∃ (distance : ℝ) (car_a car_b : Car),
    distance = 900 ∧
    car_a.speed_forward = 40 ∧
    car_a.speed_backward = 50 ∧
    car_b.speed_forward = 50 ∧
    car_b.speed_backward = 40

/-- The theorem to be proved --/
theorem car_distance_theorem (setup : problem_setup) :
  ∃ (total_distance : ℝ),
    total_distance = 1813900 ∧
    (∀ (distance : ℝ) (car_a car_b : Car),
      distance = 900 ∧
      car_a.speed_forward = 40 ∧
      car_a.speed_backward = 50 ∧
      car_b.speed_forward = 50 ∧
      car_b.speed_backward = 40 →
      total_distance = 
        (2016 / 2 - 1) * 2 * distance + 
        (car_a.speed_backward * distance) / (car_a.speed_backward + car_b.speed_backward)) :=
by
  sorry

end car_distance_theorem_l1951_195155


namespace two_tetrahedra_in_cube_l1951_195182

/-- A cube with edge length a -/
structure Cube (a : ℝ) where
  edge_length : a > 0

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- Represents the placement of a tetrahedron within a cube -/
def TetrahedronPlacement (a : ℝ) := Cube a → RegularTetrahedron a → Prop

/-- Two tetrahedra do not overlap -/
def NonOverlapping (a : ℝ) (t1 t2 : RegularTetrahedron a) : Prop := sorry

/-- Theorem stating that two non-overlapping regular tetrahedra can be inscribed in a cube -/
theorem two_tetrahedra_in_cube (a : ℝ) (h : a > 0) :
  ∃ (c : Cube a) (t1 t2 : RegularTetrahedron a) (p1 p2 : TetrahedronPlacement a),
    p1 c t1 ∧ p2 c t2 ∧ NonOverlapping a t1 t2 :=
  sorry

end two_tetrahedra_in_cube_l1951_195182


namespace smallest_positive_period_sin_cos_l1951_195134

/-- The smallest positive period of f(x) = sin x cos x is π -/
theorem smallest_positive_period_sin_cos (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x * Real.cos x) :
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = π :=
sorry

end smallest_positive_period_sin_cos_l1951_195134


namespace isosceles_triangle_largest_angle_l1951_195120

theorem isosceles_triangle_largest_angle (α β γ : ℝ) :
  -- The triangle is isosceles with two angles equal
  α = β →
  -- One of the equal angles is 50°
  α = 50 →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 80°
  max α (max β γ) = 80 := by
sorry

end isosceles_triangle_largest_angle_l1951_195120


namespace movies_watched_count_l1951_195131

/-- The number of movies watched in the 'crazy silly school' series -/
def movies_watched : ℕ := 21

/-- The number of books read in the 'crazy silly school' series -/
def books_read : ℕ := 7

/-- Theorem stating that the number of movies watched is 21 -/
theorem movies_watched_count : 
  movies_watched = books_read + 14 := by sorry

end movies_watched_count_l1951_195131


namespace profit_percentage_15_20_l1951_195148

/-- Represents the profit percentage when selling articles -/
def profit_percentage (sold : ℕ) (cost_equivalent : ℕ) : ℚ :=
  (cost_equivalent - sold) / sold

/-- Theorem: The profit percentage when selling 15 articles at the cost of 20 is 1/3 -/
theorem profit_percentage_15_20 : profit_percentage 15 20 = 1/3 := by
  sorry

end profit_percentage_15_20_l1951_195148


namespace collinear_vectors_n_equals_one_l1951_195116

def a (n : ℝ) : Fin 2 → ℝ := ![1, n]
def b (n : ℝ) : Fin 2 → ℝ := ![-1, n - 2]

def collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (∀ i, u i = k * v i)

theorem collinear_vectors_n_equals_one :
  ∀ n : ℝ, collinear (a n) (b n) → n = 1 := by
  sorry

end collinear_vectors_n_equals_one_l1951_195116


namespace construct_equilateral_from_given_l1951_195143

-- Define the given triangle
structure GivenTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_angles : angle1 + angle2 + angle3 = 180
  angle_values : angle1 = 40 ∧ angle2 = 70 ∧ angle3 = 70

-- Define an equilateral triangle
def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

-- Theorem statement
theorem construct_equilateral_from_given (t : GivenTriangle) :
  ∃ (a b c : ℝ), is_equilateral a b c :=
sorry


end construct_equilateral_from_given_l1951_195143


namespace number_puzzle_l1951_195156

theorem number_puzzle (x : ℝ) : 3 * (2 * x^2 + 15) - 7 = 91 → x = Real.sqrt (53 / 6) := by
  sorry

end number_puzzle_l1951_195156


namespace rectangle_max_area_l1951_195160

/-- A rectangle with whole number dimensions and perimeter 40 has a maximum area of 100 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 20 →
  l * w ≤ 100 :=
by sorry

end rectangle_max_area_l1951_195160


namespace equation_proof_l1951_195141

theorem equation_proof : (12 : ℕ)^3 * 6^4 / 432 = 5184 := by
  sorry

end equation_proof_l1951_195141


namespace magic_8_ball_theorem_l1951_195109

def magic_8_ball_probability : ℚ := 242112 / 823543

theorem magic_8_ball_theorem (n : ℕ) (k : ℕ) (p : ℚ) 
  (h1 : n = 7) 
  (h2 : k = 3) 
  (h3 : p = 3 / 7) :
  Nat.choose n k * p^k * (1 - p)^(n - k) = magic_8_ball_probability := by
  sorry

#check magic_8_ball_theorem

end magic_8_ball_theorem_l1951_195109


namespace smallest_perfect_square_divisible_by_5_and_7_l1951_195106

theorem smallest_perfect_square_divisible_by_5_and_7 : ∃ n : ℕ,
  n > 0 ∧
  (∃ m : ℕ, n = m ^ 2) ∧
  n % 5 = 0 ∧
  n % 7 = 0 ∧
  n = 1225 ∧
  (∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l ^ 2) → k % 5 = 0 → k % 7 = 0 → k ≥ 1225) :=
by
  sorry

#check smallest_perfect_square_divisible_by_5_and_7

end smallest_perfect_square_divisible_by_5_and_7_l1951_195106


namespace line_passes_through_fixed_point_l1951_195173

/-- The line equation as a function of m, x, and y -/
def line_equation (m x y : ℝ) : Prop :=
  (m - 1) * x - y + 2 * m + 1 = 0

/-- The theorem stating that the line passes through (-2, 3) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m (-2) 3 := by
sorry

end line_passes_through_fixed_point_l1951_195173


namespace students_per_group_l1951_195100

theorem students_per_group (total_students : ℕ) (num_teachers : ℕ) 
  (h1 : total_students = 850) (h2 : num_teachers = 23) :
  (total_students / num_teachers : ℕ) = 36 := by
  sorry

end students_per_group_l1951_195100


namespace eggs_used_for_crepes_l1951_195183

theorem eggs_used_for_crepes 
  (total_eggs : ℕ) 
  (eggs_left : ℕ) 
  (h1 : total_eggs = 3 * 12)
  (h2 : eggs_left = 9)
  (h3 : ∃ remaining_after_crepes : ℕ, 
    remaining_after_crepes ≤ total_eggs ∧ 
    eggs_left = remaining_after_crepes - (2 * remaining_after_crepes / 3)) :
  (total_eggs - (total_eggs - eggs_left * 3)) / total_eggs = 1 / 4 := by
  sorry

end eggs_used_for_crepes_l1951_195183


namespace fraction_power_equality_l1951_195194

theorem fraction_power_equality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = (x/y)^(y-x) := by sorry

end fraction_power_equality_l1951_195194


namespace cylinder_volume_l1951_195124

/-- The volume of a cylinder with radius 5 cm and height 8 cm is 628 cm³, given that π ≈ 3.14 -/
theorem cylinder_volume : 
  let r : ℝ := 5
  let h : ℝ := 8
  let π : ℝ := 3.14
  π * r^2 * h = 628 := by
  sorry

end cylinder_volume_l1951_195124


namespace satisfying_polynomial_iff_polynomial_form_l1951_195118

/-- A polynomial that satisfies the given equation for all real x -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - 6*x + 8) * P x = (x^2 + 2*x) * P (x - 2)

/-- The form of the polynomial that satisfies the equation -/
def PolynomialForm (P : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, P x = c * x^2 * (x^2 - 4)

/-- Theorem stating the equivalence between satisfying the equation and having the specific form -/
theorem satisfying_polynomial_iff_polynomial_form :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P ↔ PolynomialForm P :=
sorry

end satisfying_polynomial_iff_polynomial_form_l1951_195118


namespace minimize_sqrt_difference_l1951_195137

theorem minimize_sqrt_difference (p : ℕ) (h_p : Nat.Prime p) (h_odd : Odd p) :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ x ≤ y ∧
    (∀ (a b : ℕ), a > 0 → b > 0 → a ≤ b → 
      Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≤ 
      Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) ∧
    Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 0 ∧
    x = (p - 1) / 2 ∧ y = (p + 1) / 2 := by
  sorry

#check minimize_sqrt_difference

end minimize_sqrt_difference_l1951_195137


namespace max_movies_watched_l1951_195149

def movie_duration : ℕ := 90
def tuesday_watch_time : ℕ := 270

theorem max_movies_watched (wednesday_multiplier : ℕ) (h : wednesday_multiplier = 2) :
  let tuesday_movies := tuesday_watch_time / movie_duration
  let wednesday_movies := wednesday_multiplier * tuesday_movies
  tuesday_movies + wednesday_movies = 9 :=
by sorry

end max_movies_watched_l1951_195149


namespace pyramid_height_equal_volume_l1951_195175

theorem pyramid_height_equal_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 6 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 6.48 := by
sorry

end pyramid_height_equal_volume_l1951_195175


namespace peanut_bags_needed_l1951_195105

-- Define the flight duration in hours
def flight_duration : ℕ := 2

-- Define the number of peanuts per bag
def peanuts_per_bag : ℕ := 30

-- Define the interval between eating peanuts in minutes
def eating_interval : ℕ := 1

-- Theorem statement
theorem peanut_bags_needed : 
  (flight_duration * 60) / peanuts_per_bag = 4 := by
  sorry

end peanut_bags_needed_l1951_195105


namespace hyperbola_equation_from_foci_and_eccentricity_l1951_195133

/-- A hyperbola with given foci and eccentricity -/
structure Hyperbola where
  foci : ℝ × ℝ × ℝ × ℝ  -- Represents (x₁, y₁, x₂, y₂)
  eccentricity : ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 2 = 1

/-- Theorem stating that a hyperbola with given foci and eccentricity has the specified equation -/
theorem hyperbola_equation_from_foci_and_eccentricity (h : Hyperbola)
    (h_foci : h.foci = (-2, 0, 2, 0))
    (h_eccentricity : h.eccentricity = Real.sqrt 2) :
    ∀ x y, hyperbola_equation h x y :=
  sorry

end hyperbola_equation_from_foci_and_eccentricity_l1951_195133


namespace min_period_cos_x_div_3_l1951_195146

/-- The minimum positive period of y = cos(x/3) is 6π -/
theorem min_period_cos_x_div_3 : ∃ (T : ℝ), T > 0 ∧ T = 6 * Real.pi ∧
  ∀ (x : ℝ), Real.cos (x / 3) = Real.cos ((x + T) / 3) ∧
  ∀ (T' : ℝ), 0 < T' ∧ T' < T → ∃ (x : ℝ), Real.cos (x / 3) ≠ Real.cos ((x + T') / 3) := by
  sorry

end min_period_cos_x_div_3_l1951_195146


namespace interval_condition_l1951_195123

theorem interval_condition (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) :=
by sorry

end interval_condition_l1951_195123


namespace cabin_cost_l1951_195152

theorem cabin_cost (total_cost land_cost cabin_cost : ℕ) : 
  total_cost = 30000 →
  land_cost = 4 * cabin_cost →
  total_cost = land_cost + cabin_cost →
  cabin_cost = 6000 := by
  sorry

end cabin_cost_l1951_195152


namespace certain_number_problem_l1951_195104

theorem certain_number_problem (x : ℝ) (h : 0.6 * x = 0.4 * 30 + 18) : x = 50 := by
  sorry

end certain_number_problem_l1951_195104


namespace obtuse_angle_measure_l1951_195129

/-- An obtuse angle divided by a perpendicular line into two angles with a ratio of 6:1 measures 105°. -/
theorem obtuse_angle_measure (θ : ℝ) (h1 : 90 < θ) (h2 : θ < 180) : 
  ∃ (α β : ℝ), α + β = θ ∧ α / β = 6 ∧ θ = 105 := by
  sorry

end obtuse_angle_measure_l1951_195129


namespace interest_rate_calculation_l1951_195158

theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 432)
  (h3 : ∀ rate time, interest_paid = principal * rate * time / 100 → rate = time) :
  ∃ rate : ℝ, rate = 6 ∧ interest_paid = principal * rate * rate / 100 := by
sorry

end interest_rate_calculation_l1951_195158


namespace race_speed_ratio_l1951_195140

/-- Represents the speeds and distances in a race between two runners A and B -/
structure RaceParameters where
  speedA : ℝ
  speedB : ℝ
  totalDistance : ℝ
  headStart : ℝ

/-- Theorem stating that if A and B finish at the same time in a race with given parameters,
    then A's speed is 4 times B's speed -/
theorem race_speed_ratio 
  (race : RaceParameters) 
  (h1 : race.totalDistance = 100)
  (h2 : race.headStart = 75)
  (h3 : race.totalDistance / race.speedA = (race.totalDistance - race.headStart) / race.speedB) :
  race.speedA = 4 * race.speedB := by
  sorry


end race_speed_ratio_l1951_195140


namespace sum_of_coefficients_is_zero_l1951_195174

/-- Two parabolas with different vertices, where each parabola's vertex lies on the other parabola -/
structure TwoParabolas where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  p : ℝ
  q : ℝ
  h_diff_vertices : x₁ ≠ x₂
  h_vertex_on_other₁ : y₂ = p * (x₂ - x₁)^2 + y₁
  h_vertex_on_other₂ : y₁ = q * (x₁ - x₂)^2 + y₂

/-- The sum of the leading coefficients of two parabolas with the described properties is zero -/
theorem sum_of_coefficients_is_zero (tp : TwoParabolas) : tp.p + tp.q = 0 := by
  sorry

end sum_of_coefficients_is_zero_l1951_195174


namespace power_equality_l1951_195112

theorem power_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 := by
  sorry

end power_equality_l1951_195112


namespace combination_permutation_properties_l1951_195108

-- Define combination function
def C (n m : ℕ) : ℕ := 
  if m ≤ n then Nat.choose n m else 0

-- Define permutation function
def A (n m : ℕ) : ℕ := 
  if m ≤ n then Nat.factorial n / Nat.factorial (n - m) else 0

-- Theorem statement
theorem combination_permutation_properties (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (C n m = C n (n - m)) ∧
  (C (n + 1) m = C n (m - 1) + C n m) ∧
  (A n m = C n m * A m m) ∧
  (A (n + 1) (m + 1) ≠ (m + 1) * A n m) := by
  sorry

end combination_permutation_properties_l1951_195108


namespace vector_at_negative_two_l1951_195168

/-- A parameterized line in 2D space. -/
structure ParameterizedLine where
  vector : ℝ → (ℝ × ℝ)

/-- Given conditions for the parameterized line. -/
def line_conditions (L : ParameterizedLine) : Prop :=
  L.vector 1 = (2, 5) ∧ L.vector 4 = (5, -7)

/-- The theorem stating the vector at t = -2 given the conditions. -/
theorem vector_at_negative_two
  (L : ParameterizedLine)
  (h : line_conditions L) :
  L.vector (-2) = (-1, 17) := by
  sorry

end vector_at_negative_two_l1951_195168


namespace partial_fraction_decomposition_l1951_195150

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 12 ∧ x ≠ -4 →
    (6 * x + 15) / (x^2 - 8*x - 48) = (87/16) / (x - 12) + (9/16) / (x + 4) := by
  sorry

end partial_fraction_decomposition_l1951_195150


namespace range_of_a_for_full_range_l1951_195169

/-- Piecewise function f(x) defined by a real parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x - 1 else x^2 - 2 * a * x

/-- The range of f(x) is all real numbers -/
def has_full_range (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- The range of a for which f(x) has a full range is [2/3, +∞) -/
theorem range_of_a_for_full_range :
  {a : ℝ | has_full_range a} = {a : ℝ | a ≥ 2/3} :=
sorry

end range_of_a_for_full_range_l1951_195169


namespace platform_length_l1951_195101

/-- Given a train of length 600 meters that takes 54 seconds to cross a platform
    and 36 seconds to cross a signal pole, the length of the platform is 300 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 600 →
  time_platform = 54 →
  time_pole = 36 →
  ∃ platform_length : ℝ,
    platform_length = 300 ∧
    train_length / time_pole = (train_length + platform_length) / time_platform :=
by sorry

end platform_length_l1951_195101


namespace quadratic_real_roots_l1951_195159

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - x + a = 0) ↔ a ≤ 1/4 := by sorry

end quadratic_real_roots_l1951_195159


namespace distance_to_asymptotes_l1951_195121

/-- The distance from point P(0,1) to the asymptotes of the hyperbola y²/4 - x² = 1 is √5/5 -/
theorem distance_to_asymptotes (x y : ℝ) : 
  let P : ℝ × ℝ := (0, 1)
  let hyperbola := {(x, y) | y^2/4 - x^2 = 1}
  let asymptote (m : ℝ) := {(x, y) | y = m*x}
  let distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) := 
    |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)
  ∃ (m : ℝ), m^2 = 4 ∧ 
    distance_point_to_line P m (-1) 0 = Real.sqrt 5 / 5 :=
by sorry

end distance_to_asymptotes_l1951_195121


namespace first_term_of_special_arithmetic_sequence_l1951_195142

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℚ
  /-- The common difference of the sequence -/
  d : ℚ
  /-- The sum of the first 60 terms is 660 -/
  sum_first_60 : (60 : ℚ) / 2 * (2 * a + 59 * d) = 660
  /-- The sum of the next 60 terms (terms 61 to 120) is 3660 -/
  sum_next_60 : (60 : ℚ) / 2 * (2 * (a + 60 * d) + 59 * d) = 3660

/-- The first term of the arithmetic sequence with the given properties is -163/12 -/
theorem first_term_of_special_arithmetic_sequence (seq : ArithmeticSequence) : seq.a = -163/12 := by
  sorry

end first_term_of_special_arithmetic_sequence_l1951_195142


namespace draw_with_replacement_l1951_195179

/-- The number of items to choose from -/
def n : ℕ := 15

/-- The number of times we draw -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from n items -/
def num_lists (n k : ℕ) : ℕ := n^k

theorem draw_with_replacement :
  num_lists n k = 50625 := by
  sorry

end draw_with_replacement_l1951_195179


namespace remainder_of_x_120_divided_by_x2_minus_4x_plus_3_l1951_195102

theorem remainder_of_x_120_divided_by_x2_minus_4x_plus_3 :
  ∀ (x : ℝ), ∃ (Q : ℝ → ℝ),
    x^120 = (x^2 - 4*x + 3) * Q x + ((3^120 - 1)*x + (3 - 3^120)) / 2 :=
by sorry

end remainder_of_x_120_divided_by_x2_minus_4x_plus_3_l1951_195102


namespace complex_roots_equilateral_triangle_l1951_195135

theorem complex_roots_equilateral_triangle (z₁ z₂ p q : ℂ) : 
  z₁^2 + p*z₁ + q = 0 →
  z₂^2 + p*z₂ + q = 0 →
  z₂ = Complex.exp (2*Real.pi*Complex.I/3) * z₁ →
  p^2 / q = 1 := by
sorry

end complex_roots_equilateral_triangle_l1951_195135


namespace jenny_sleep_hours_l1951_195113

theorem jenny_sleep_hours (minutes_per_hour : ℕ) (total_sleep_minutes : ℕ) 
  (h1 : minutes_per_hour = 60) 
  (h2 : total_sleep_minutes = 480) : 
  total_sleep_minutes / minutes_per_hour = 8 := by
sorry

end jenny_sleep_hours_l1951_195113


namespace black_friday_sales_l1951_195132

/-- Proves that if a store sells 477 televisions three years from now, 
    and the number of televisions sold increases by 50 each year, 
    then the store sold 327 televisions this year. -/
theorem black_friday_sales (current_sales : ℕ) : 
  (current_sales + 3 * 50 = 477) → current_sales = 327 := by
  sorry

end black_friday_sales_l1951_195132


namespace line_and_circle_problem_l1951_195193

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line3 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Theorem statement
theorem line_and_circle_problem :
  ∀ (x_int y_int : ℝ),
    (line1 x_int y_int ∧ line2 x_int y_int) →  -- Intersection point condition
    (∀ (x y : ℝ), line_l x y → (x + y - 2 ≠ 0)) →  -- Perpendicularity condition
    circle_C 1 0 →  -- Circle passes through (1,0)
    (∃ (a : ℝ), a > 0 ∧ circle_C a 0) →  -- Center on positive x-axis
    (∃ (x1 y1 x2 y2 : ℝ),
      line_l x1 y1 ∧ line_l x2 y2 ∧
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = 8) →  -- Chord length condition
    (∀ (x y : ℝ), line_l x y ↔ y = x - 1) ∧
    (∀ (x y : ℝ), circle_C x y ↔ (x - 3)^2 + y^2 = 4) :=
by sorry

end line_and_circle_problem_l1951_195193


namespace max_solitar_result_l1951_195164

/-- The greatest prime divisor of a natural number -/
def greatestPrimeDivisor (n : ℕ) : ℕ := sorry

/-- The set of numbers from 1 to 16 -/
def initialSet : Finset ℕ := Finset.range 16

/-- The result of one step in the solitar game -/
def solitarStep (s : Finset ℕ) : Finset ℕ := sorry

/-- The final result of the solitar game -/
def solitarResult (s : Finset ℕ) : ℕ := sorry

/-- The maximum possible final number in the solitar game -/
theorem max_solitar_result : 
  ∃ (result : ℕ), solitarResult initialSet = result ∧ result ≤ 19 ∧ 
  ∀ (other : ℕ), solitarResult initialSet = other → other ≤ result :=
sorry

end max_solitar_result_l1951_195164


namespace square_on_circle_radius_l1951_195185

theorem square_on_circle_radius (S : ℝ) (x : ℝ) (R : ℝ) : 
  S = 256 → -- Square area is 256 cm²
  x^2 = S → -- Side length of the square
  (x - R)^2 = R^2 - (x/2)^2 → -- Pythagoras theorem application
  R = 10 := by
  sorry

end square_on_circle_radius_l1951_195185


namespace rectangle_triangle_area_ratio_l1951_195187

/-- 
Given a rectangle with length L and width W, and a triangle with one side of the rectangle as its base 
and a vertex on the opposite side of the rectangle, the ratio of the area of the rectangle to the area 
of the triangle is 2:1.
-/
theorem rectangle_triangle_area_ratio 
  (L W : ℝ) 
  (hL : L > 0) 
  (hW : W > 0) : 
  (L * W) / ((1/2) * L * W) = 2 := by
sorry

end rectangle_triangle_area_ratio_l1951_195187


namespace additional_money_needed_per_twin_l1951_195190

def initial_amount : ℝ := 50
def toilet_paper_cost : ℝ := 12
def groceries_cost : ℝ := 2 * toilet_paper_cost
def remaining_after_groceries : ℝ := initial_amount - toilet_paper_cost - groceries_cost
def boot_cost : ℝ := 3 * remaining_after_groceries
def total_boot_cost : ℝ := 2 * boot_cost

theorem additional_money_needed_per_twin : 
  (total_boot_cost - remaining_after_groceries) / 2 = 35 := by sorry

end additional_money_needed_per_twin_l1951_195190


namespace nellie_legos_l1951_195111

theorem nellie_legos (initial : ℕ) (lost : ℕ) (given_away : ℕ) :
  initial ≥ lost + given_away →
  initial - (lost + given_away) = initial - lost - given_away :=
by
  sorry

#check nellie_legos 380 57 24

end nellie_legos_l1951_195111


namespace sum_of_fractions_to_decimal_l1951_195128

theorem sum_of_fractions_to_decimal : (5 : ℚ) / 16 + (1 : ℚ) / 4 = (5625 : ℚ) / 10000 := by
  sorry

end sum_of_fractions_to_decimal_l1951_195128


namespace functional_equation_solution_l1951_195199

open Real

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_pos : ∀ x, x > 0 → f x > 0) 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_eq : ∀ x, x > 0 → deriv f (a / x) = x / f x) :
  ∃ b : ℝ, b > 0 ∧ ∀ x, x > 0 → f x = a^(1 - a/b) * x^(a/b) := by
  sorry

end functional_equation_solution_l1951_195199


namespace problem_statement_l1951_195166

theorem problem_statement (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 5 = 23 := by
  sorry

end problem_statement_l1951_195166


namespace apple_probability_l1951_195176

def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def chosen_apples : ℕ := 3

theorem apple_probability :
  (Nat.choose red_apples chosen_apples +
   Nat.choose green_apples chosen_apples +
   (Nat.choose red_apples 2 * Nat.choose green_apples 1) +
   (Nat.choose green_apples 2 * Nat.choose red_apples 1)) /
  Nat.choose total_apples chosen_apples = 7 / 15 := by
  sorry

end apple_probability_l1951_195176


namespace solve_equation_l1951_195126

theorem solve_equation (x : ℝ) : 2 * x = (26 - x) + 19 → x = 15 := by
  sorry

end solve_equation_l1951_195126


namespace work_completion_time_l1951_195117

/-- The number of days B needs to complete the entire work alone -/
def B_total_days : ℝ := 14.999999999999996

/-- The number of days A works before leaving -/
def A_partial_days : ℝ := 5

/-- The number of days B needs to complete the remaining work after A leaves -/
def B_remaining_days : ℝ := 10

/-- The number of days A needs to complete the entire work alone -/
def A_total_days : ℝ := 15

theorem work_completion_time :
  B_total_days = 14.999999999999996 →
  A_partial_days = 5 →
  B_remaining_days = 10 →
  A_total_days = 15 := by
  sorry

end work_completion_time_l1951_195117


namespace almost_every_graph_chromatic_number_l1951_195151

-- Define the random graph model
structure RandomGraph (n : ℕ) (p : ℝ) where
  -- Add necessary fields here

-- Define the chromatic number
def chromaticNumber (G : RandomGraph n p) : ℝ := sorry

-- Main theorem
theorem almost_every_graph_chromatic_number 
  (p : ℝ) (ε : ℝ) (n : ℕ) (h_p : 0 < p ∧ p < 1) (h_ε : ε > 0) :
  ∃ (G : RandomGraph n p), 
    chromaticNumber G > (Real.log (1 / (1 - p))) / (2 + ε) * (n / Real.log n) := by
  sorry

end almost_every_graph_chromatic_number_l1951_195151


namespace karens_paddling_speed_l1951_195103

/-- Karen's canoe paddling problem -/
theorem karens_paddling_speed
  (river_current : ℝ)
  (river_length : ℝ)
  (paddling_time : ℝ)
  (h1 : river_current = 4)
  (h2 : river_length = 12)
  (h3 : paddling_time = 2)
  : ∃ (still_water_speed : ℝ),
    still_water_speed = 10 ∧
    river_length = (still_water_speed - river_current) * paddling_time :=
by sorry

end karens_paddling_speed_l1951_195103


namespace candy_packing_problem_l1951_195192

theorem candy_packing_problem :
  ∃! (s : Finset ℕ),
    (∀ a ∈ s, 200 ≤ a ∧ a ≤ 250) ∧
    (∀ a ∈ s, a % 10 = 6) ∧
    (∀ a ∈ s, a % 15 = 11) ∧
    s.card = 2 :=
by sorry

end candy_packing_problem_l1951_195192


namespace curve_tangent_problem_l1951_195147

/-- The curve C is defined by the equation y = 2x³ + ax + a -/
def C (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x + a

/-- The derivative of C with respect to x -/
def C_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 + a

theorem curve_tangent_problem (a : ℝ) :
  (C a (-1) = 0) →  -- C passes through point M(-1, 0)
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    C_derivative a t₁ + C_derivative a t₂ = 0 ∧  -- |MA| = |MB| condition
    4 * t₁^3 + 6 * t₁^2 = 0 ∧                   -- Tangent line condition for t₁
    4 * t₂^3 + 6 * t₂^2 = 0) →                  -- Tangent line condition for t₂
  a = -27/4 := by
  sorry

end curve_tangent_problem_l1951_195147


namespace sum_of_reciprocals_positive_l1951_195157

theorem sum_of_reciprocals_positive 
  (a b c d : ℝ) 
  (ha : |a| > 1) 
  (hb : |b| > 1) 
  (hc : |c| > 1) 
  (hd : |d| > 1) 
  (h_sum : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) : 
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end sum_of_reciprocals_positive_l1951_195157


namespace samantha_more_heads_prob_l1951_195167

def fair_coin_prob : ℚ := 1/2
def biased_coin_prob1 : ℚ := 3/5
def biased_coin_prob2 : ℚ := 2/3

def coin_set := (fair_coin_prob, biased_coin_prob1, biased_coin_prob2)

def prob_more_heads (coins : ℚ × ℚ × ℚ) : ℚ :=
  sorry

theorem samantha_more_heads_prob :
  prob_more_heads coin_set = 436/225 :=
sorry

end samantha_more_heads_prob_l1951_195167


namespace bicycle_parking_income_l1951_195139

/-- Represents the total income from bicycle parking --/
def total_income (x : ℝ) : ℝ := -0.3 * x + 1600

/-- Theorem stating the relationship between the number of ordinary bicycles parked and the total income --/
theorem bicycle_parking_income (x : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ 2000) : 
  total_income x = 0.5 * x + 0.8 * (2000 - x) := by
  sorry

#check bicycle_parking_income

end bicycle_parking_income_l1951_195139


namespace quadratic_equation_h_value_l1951_195114

theorem quadratic_equation_h_value (h : ℝ) : 
  (∃ r s : ℝ, r^2 + 2*h*r = 3 ∧ s^2 + 2*h*s = 3 ∧ r^2 + s^2 = 10) → 
  |h| = 1 := by
sorry

end quadratic_equation_h_value_l1951_195114


namespace gcd_consecutive_b_terms_l1951_195181

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem gcd_consecutive_b_terms (n : ℕ) : Nat.gcd (b n) (b (n + 1)) = 2 := by
  sorry

end gcd_consecutive_b_terms_l1951_195181


namespace A_sufficient_not_necessary_for_B_l1951_195191

/-- Proposition A: 0 < x < 5 -/
def prop_A (x : ℝ) : Prop := 0 < x ∧ x < 5

/-- Proposition B: |x - 2| < 3 -/
def prop_B (x : ℝ) : Prop := |x - 2| < 3

theorem A_sufficient_not_necessary_for_B :
  (∀ x : ℝ, prop_A x → prop_B x) ∧
  (∃ x : ℝ, prop_B x ∧ ¬prop_A x) := by sorry

end A_sufficient_not_necessary_for_B_l1951_195191


namespace range_of_a_l1951_195162

-- Define the conditions
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (x a : ℝ) :
  (∀ x, p x → (x < -3 ∨ x > 1)) →
  (∀ x, ¬(p x) ↔ (-3 ≤ x ∧ x ≤ 1)) →
  (∀ x, ¬(q x a) ↔ x ≤ a) →
  (∀ x, ¬(p x) → ¬(q x a)) →
  (∃ x, ¬(p x) ∧ q x a) →
  a ≥ 1 := by
  sorry

end range_of_a_l1951_195162


namespace number_equals_five_l1951_195189

theorem number_equals_five (N x : ℝ) (h1 : N / (4 + 1/x) = 1) (h2 : x = 1) : N = 5 := by
  sorry

end number_equals_five_l1951_195189


namespace removed_number_value_l1951_195138

theorem removed_number_value (S : ℝ) (X : ℝ) : 
  S / 50 = 56 →
  (S - X - 55) / 48 = 56.25 →
  X = 45 := by
sorry

end removed_number_value_l1951_195138


namespace seven_twelfths_decimal_l1951_195170

theorem seven_twelfths_decimal : 7 / 12 = 0.5833333333333333 := by sorry

end seven_twelfths_decimal_l1951_195170


namespace one_meeting_l1951_195195

/-- Represents a boy moving on a circular track -/
structure Boy where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The problem setup -/
def circularTrackProblem (circumference : ℝ) (boy1 boy2 : Boy) : Prop :=
  circumference > 0 ∧
  boy1.speed = 6 ∧
  boy2.speed = 10 ∧
  boy1.direction ≠ boy2.direction

/-- The number of meetings between the two boys -/
def numberOfMeetings (circumference : ℝ) (boy1 boy2 : Boy) : ℕ := sorry

/-- The theorem stating that the boys meet exactly once -/
theorem one_meeting (circumference : ℝ) (boy1 boy2 : Boy) 
  (h : circularTrackProblem circumference boy1 boy2) : 
  numberOfMeetings circumference boy1 boy2 = 1 := by sorry

end one_meeting_l1951_195195


namespace peanuts_in_jar_l1951_195115

theorem peanuts_in_jar (initial_peanuts : ℕ) : 
  (initial_peanuts : ℚ) - (1/4 : ℚ) * initial_peanuts - 29 = 82 → 
  initial_peanuts = 148 := by
  sorry

end peanuts_in_jar_l1951_195115


namespace smallest_k_for_three_reals_l1951_195177

theorem smallest_k_for_three_reals : ∃ (k : ℝ),
  (∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
    (|x - y| ≤ k ∨ |1/x - 1/y| ≤ k) ∨
    (|y - z| ≤ k ∨ |1/y - 1/z| ≤ k) ∨
    (|x - z| ≤ k ∨ |1/x - 1/z| ≤ k)) ∧
  (∀ (k' : ℝ), k' < k →
    ∃ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
      (|x - y| > k' ∧ |1/x - 1/y| > k') ∧
      (|y - z| > k' ∧ |1/y - 1/z| > k') ∧
      (|x - z| > k' ∧ |1/x - 1/z| > k')) ∧
  k = 1.5 :=
by sorry

end smallest_k_for_three_reals_l1951_195177


namespace range_of_f_l1951_195122

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0} := by sorry

end range_of_f_l1951_195122


namespace ad_length_is_sqrt_397_l1951_195163

/-- A quadrilateral with intersecting diagonals -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (bo : dist B O = 5)
  (od : dist O D = 7)
  (ao : dist A O = 9)
  (oc : dist O C = 4)
  (ab : dist A B = 7)

/-- The length of AD in the quadrilateral -/
def ad_length (q : Quadrilateral) : ℝ := dist q.A q.D

/-- Theorem stating that AD length is √397 -/
theorem ad_length_is_sqrt_397 (q : Quadrilateral) : ad_length q = Real.sqrt 397 := by
  sorry


end ad_length_is_sqrt_397_l1951_195163


namespace fraction_power_equality_l1951_195125

theorem fraction_power_equality : (72000 ^ 5 : ℕ) / (9000 ^ 5) = 32768 := by sorry

end fraction_power_equality_l1951_195125


namespace base_conversion_1850_to_base_7_l1951_195154

theorem base_conversion_1850_to_base_7 :
  (5 * 7^3 + 2 * 7^2 + 5 * 7^1 + 2 * 7^0 : ℕ) = 1850 := by
  sorry

end base_conversion_1850_to_base_7_l1951_195154


namespace concert_ticket_price_l1951_195171

/-- Proves that the cost of each ticket is $30 given the concert conditions --/
theorem concert_ticket_price :
  ∀ (ticket_price : ℝ),
    (500 : ℝ) * ticket_price * 0.7 = (4 : ℝ) * 2625 →
    ticket_price = 30 := by
  sorry

end concert_ticket_price_l1951_195171


namespace min_value_cubic_function_l1951_195153

/-- The function f(x) = x^3 - 3x has a minimum value of -2. -/
theorem min_value_cubic_function :
  ∃ m : ℝ, m = -2 ∧ ∀ x : ℝ, x^3 - 3*x ≥ m :=
by
  sorry

end min_value_cubic_function_l1951_195153
