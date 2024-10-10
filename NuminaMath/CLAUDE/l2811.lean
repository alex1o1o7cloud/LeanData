import Mathlib

namespace expression_simplification_l2811_281182

theorem expression_simplification (m n : ℝ) 
  (hm : m = (400 : ℝ) ^ (1/4))
  (hn : n = (5 : ℝ) ^ (1/2)) :
  ((2 - n) / (n - 1) + 4 * (m - 1) / (m - 2)) / 
  (n^2 * (m - 1) / (n - 1) + m^2 * (2 - n) / (m - 2)) = 
  ((5 : ℝ) ^ (1/2)) / 5 := by sorry

end expression_simplification_l2811_281182


namespace temperature_problem_l2811_281171

/-- Given the average temperatures for two sets of three consecutive days and the temperature of the last day, prove the temperature of the first day. -/
theorem temperature_problem (T W Th F : ℝ) 
  (h1 : (T + W + Th) / 3 = 42)
  (h2 : (W + Th + F) / 3 = 44)
  (h3 : F = 43) :
  T = 37 := by
  sorry

end temperature_problem_l2811_281171


namespace min_distance_parallel_lines_l2811_281197

/-- Given two parallel lines and a point between them, prove the minimum distance from the point to a fixed point. -/
theorem min_distance_parallel_lines (x₀ y₀ : ℝ) :
  (∃ (xp yp xq yq : ℝ),
    (xp - 2*yp - 1 = 0) ∧
    (xq - 2*yq + 3 = 0) ∧
    (x₀ = (xp + xq) / 2) ∧
    (y₀ = (yp + yq) / 2) ∧
    (y₀ > -x₀ + 2)) →
  Real.sqrt ((x₀ - 4)^2 + y₀^2) ≥ Real.sqrt 5 :=
by sorry

end min_distance_parallel_lines_l2811_281197


namespace strawberry_price_proof_l2811_281101

/-- The cost of strawberries in dollars per pound -/
def strawberry_cost : ℝ := sorry

/-- The cost of cherries in dollars per pound -/
def cherry_cost : ℝ := sorry

/-- The total cost of 5 pounds of strawberries and 5 pounds of cherries -/
def total_cost : ℝ := sorry

theorem strawberry_price_proof :
  (cherry_cost = 6 * strawberry_cost) →
  (total_cost = 5 * strawberry_cost + 5 * cherry_cost) →
  (total_cost = 77) →
  (strawberry_cost = 2.2) := by
  sorry

end strawberry_price_proof_l2811_281101


namespace intersection_points_parabola_and_circle_l2811_281137

theorem intersection_points_parabola_and_circle (A : ℝ) (h : A > 0) :
  ∃! (points : Finset (ℝ × ℝ)), points.card = 4 ∧
    ∀ (x y : ℝ), (x, y) ∈ points ↔ 
      (y = A * x^2 ∧ y^2 + 5 = x^2 + 6 * y) :=
by sorry

end intersection_points_parabola_and_circle_l2811_281137


namespace units_digit_base_8_l2811_281113

theorem units_digit_base_8 (n₁ n₂ : ℕ) (h₁ : n₁ = 198) (h₂ : n₂ = 53) :
  (((n₁ - 3) * (n₂ + 7)) % 8) = 4 := by
  sorry

end units_digit_base_8_l2811_281113


namespace expand_product_l2811_281152

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := by
  sorry

end expand_product_l2811_281152


namespace parallelogram_area_specific_l2811_281184

/-- The area of a parallelogram with base b and height h. -/
def parallelogram_area (b h : ℝ) : ℝ := b * h

/-- Theorem: The area of a parallelogram with a base of 15 meters and an altitude
    that is twice the base is 450 square meters. -/
theorem parallelogram_area_specific : 
  let base : ℝ := 15
  let height : ℝ := 2 * base
  parallelogram_area base height = 450 := by
sorry

end parallelogram_area_specific_l2811_281184


namespace ball_difference_l2811_281120

def soccer_boxes : ℕ := 8
def basketball_boxes : ℕ := 5
def balls_per_box : ℕ := 12

theorem ball_difference : 
  soccer_boxes * balls_per_box - basketball_boxes * balls_per_box = 36 := by
  sorry

end ball_difference_l2811_281120


namespace instantaneous_velocity_at_2_time_to_water_surface_l2811_281185

-- Define the height function
def h (t : ℝ) : ℝ := -4.8 * t^2 + 8 * t + 10

-- Theorem for instantaneous velocity at t = 2
theorem instantaneous_velocity_at_2 : 
  (deriv h) 2 = -11.2 := by sorry

-- Theorem for time when athlete reaches water surface
theorem time_to_water_surface : 
  ∃ t : ℝ, t = 2.5 ∧ h t = 0 := by sorry

end instantaneous_velocity_at_2_time_to_water_surface_l2811_281185


namespace necessary_not_sufficient_l2811_281119

-- Define the proposition
def P (x m : ℝ) : Prop := x / 2 + 1 / (2 * x) - 3 / 2 > m

-- Define the condition
def condition (m : ℝ) : Prop := ∀ x > 0, P x m

-- Define necessary condition
def necessary (m : ℝ) : Prop := condition m → m ≤ -1/2

-- Define sufficient condition
def sufficient (m : ℝ) : Prop := m ≤ -1/2 → condition m

-- Theorem statement
theorem necessary_not_sufficient :
  (∀ m : ℝ, necessary m) ∧ (∃ m : ℝ, ¬ sufficient m) :=
sorry

end necessary_not_sufficient_l2811_281119


namespace first_dog_bones_l2811_281188

theorem first_dog_bones (total_bones : ℕ) (total_dogs : ℕ) 
  (h_total_bones : total_bones = 12)
  (h_total_dogs : total_dogs = 5)
  (first_dog : ℕ)
  (second_dog : ℕ)
  (third_dog : ℕ)
  (fourth_dog : ℕ)
  (fifth_dog : ℕ)
  (h_second_dog : second_dog = first_dog - 1)
  (h_third_dog : third_dog = 2 * second_dog)
  (h_fourth_dog : fourth_dog = 1)
  (h_fifth_dog : fifth_dog = 2 * fourth_dog)
  (h_all_bones : first_dog + second_dog + third_dog + fourth_dog + fifth_dog = total_bones) :
  first_dog = 3 := by
sorry

end first_dog_bones_l2811_281188


namespace no_rational_solution_l2811_281127

theorem no_rational_solution :
  ∀ (a b c d : ℚ), (a + b * Real.sqrt 3)^4 + (c + d * Real.sqrt 3)^4 ≠ 1 + Real.sqrt 3 := by
  sorry

end no_rational_solution_l2811_281127


namespace sufficient_not_necessary_condition_l2811_281154

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x ∈ Set.Ioo 0 3 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x ∉ Set.Ioo 0 3) := by
  sorry

end sufficient_not_necessary_condition_l2811_281154


namespace parallelogram_in_grid_l2811_281128

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a vector between two points in the grid -/
structure GridVector where
  dx : ℤ
  dy : ℤ

/-- The theorem to be proved -/
theorem parallelogram_in_grid (n : ℕ) (h : n ≥ 2) :
  ∀ (chosen : Finset GridPoint),
    chosen.card = 2 * n →
    ∃ (a b c d : GridPoint),
      a ∈ chosen ∧ b ∈ chosen ∧ c ∈ chosen ∧ d ∈ chosen ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      (GridVector.mk (b.x - a.x) (b.y - a.y) =
       GridVector.mk (d.x - c.x) (d.y - c.y)) ∧
      (GridVector.mk (c.x - a.x) (c.y - a.y) =
       GridVector.mk (d.x - b.x) (d.y - b.y)) :=
sorry

end parallelogram_in_grid_l2811_281128


namespace greatest_multiple_of_5_and_6_less_than_1000_l2811_281190

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n = 990 ∧ 
  (∀ m : ℕ, m < 1000 → m % 5 = 0 → m % 6 = 0 → m ≤ n) :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l2811_281190


namespace fraction_equality_l2811_281195

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : a / b = (a * b) / (b ^ 2) := by
  sorry

end fraction_equality_l2811_281195


namespace congruence_problem_l2811_281158

theorem congruence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n < 23 ∧ -135 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end congruence_problem_l2811_281158


namespace limit_one_minus_cos_x_over_x_squared_l2811_281141

theorem limit_one_minus_cos_x_over_x_squared :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((1 - Real.cos x) / x^2) - (1/2)| < ε := by
  sorry

end limit_one_minus_cos_x_over_x_squared_l2811_281141


namespace nine_integer_lengths_l2811_281181

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ

/-- Counts the number of distinct integer lengths of line segments
    that can be drawn from a vertex to the opposite side in a right triangle -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem stating that for a right triangle with legs 24 and 25,
    there are exactly 9 distinct integer lengths of line segments
    that can be drawn from a vertex to the hypotenuse -/
theorem nine_integer_lengths :
  let t : RightTriangle := { leg1 := 24, leg2 := 25 }
  countIntegerLengths t = 9 :=
by
  sorry

end nine_integer_lengths_l2811_281181


namespace min_value_sum_reciprocals_l2811_281199

/-- Given a line 2ax - by + 2 = 0 (where a > 0, b > 0) passing through the point (-1, 2),
    the minimum value of 1/a + 1/b is 4. -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : 2 * a * (-1) - b * 2 + 2 = 0) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x * (-1) - y * 2 + 2 = 0 → 1 / x + 1 / y ≥ 1 / a + 1 / b) ∧
  1 / a + 1 / b = 4 := by
sorry

end min_value_sum_reciprocals_l2811_281199


namespace polynomial_inequality_l2811_281143

theorem polynomial_inequality (x : ℝ) : 
  x^4 - 4*x^3 + 8*x^2 - 8*x ≤ 96 → -2 ≤ x ∧ x ≤ 4 := by
  sorry

end polynomial_inequality_l2811_281143


namespace soccer_club_girls_l2811_281104

theorem soccer_club_girls (total_members : ℕ) (attended_members : ℕ) 
  (h1 : total_members = 30)
  (h2 : attended_members = 18)
  (h3 : ∃ (boys girls : ℕ), boys + girls = total_members ∧ boys + girls / 3 = attended_members) :
  ∃ (girls : ℕ), girls = 18 ∧ ∃ (boys : ℕ), boys + girls = total_members := by
sorry

end soccer_club_girls_l2811_281104


namespace complex_number_quadrant_l2811_281176

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 + I) / (1 + 2*I) ∧ z.re > 0 ∧ z.im < 0 :=
by sorry

end complex_number_quadrant_l2811_281176


namespace intersection_of_A_and_B_l2811_281129

-- Define sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2*x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x < 3/2} := by sorry

end intersection_of_A_and_B_l2811_281129


namespace investment_interest_rate_l2811_281155

theorem investment_interest_rate 
  (principal : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 900) 
  (h2 : rate1 = 0.04) 
  (h3 : time = 7) 
  (h4 : principal * rate2 * time - principal * rate1 * time = interest_difference) 
  (h5 : interest_difference = 31.50) : 
  rate2 = 0.045 := by
sorry

end investment_interest_rate_l2811_281155


namespace equation_solution_l2811_281109

theorem equation_solution (x y : ℝ) : 
  (x^4 + 1) * (y^4 + 1) = 4 * x^2 * y^2 ↔ 
  ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) :=
by sorry

end equation_solution_l2811_281109


namespace worksheets_graded_l2811_281121

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 15 →
  problems_per_worksheet = 3 →
  problems_left = 24 →
  (total_worksheets * problems_per_worksheet - problems_left) / problems_per_worksheet = 7 := by
sorry

end worksheets_graded_l2811_281121


namespace circulation_within_period_l2811_281132

/-- Represents the average yearly circulation for magazine P from 1962 to 1970 -/
def average_circulation : ℝ := sorry

/-- Represents the circulation of magazine P in 1961 -/
def circulation_1961 : ℝ := sorry

/-- Represents the year when the circulation was 4 times the average -/
def special_year : ℕ := sorry

/-- The circulation in the special year -/
def special_circulation : ℝ := 4 * average_circulation

/-- The total circulation from 1961 to 1970 -/
def total_circulation : ℝ := circulation_1961 + 9 * average_circulation

/-- The ratio of special circulation to total circulation -/
def circulation_ratio : ℝ := 0.2857142857142857

theorem circulation_within_period : 
  (special_circulation / total_circulation = circulation_ratio) →
  (circulation_1961 = 5 * average_circulation) →
  (special_year ≥ 1961 ∧ special_year ≤ 1970) :=
by sorry

end circulation_within_period_l2811_281132


namespace ball_drawing_game_l2811_281172

/-- The probability that the last ball is white in a ball-drawing game -/
def last_ball_white_probability (p q : ℕ) : ℚ :=
  if p % 2 = 0 then 0 else 1

/-- The ball-drawing game process -/
theorem ball_drawing_game (p q : ℕ) :
  let initial_total := p + q
  let final_total := 1
  let draw_count := initial_total - final_total
  ∀ (draw_process : ℕ → ℕ × ℕ),
    (∀ i < draw_count, 
      let (w, b) := draw_process i
      let (w', b') := draw_process (i + 1)
      ((w = w' ∧ b = b' + 1) ∨ (w = w' - 1 ∧ b = b' + 1) ∨ (w = w' - 2 ∧ b = b' + 1))) →
    (draw_process 0 = (p, q)) →
    (draw_process draw_count).fst + (draw_process draw_count).snd = final_total →
    (last_ball_white_probability p q = if (draw_process draw_count).fst = 1 then 1 else 0) :=
sorry

end ball_drawing_game_l2811_281172


namespace sector_central_angle_l2811_281187

/-- Given a sector with radius 10 cm and area 100 cm², prove that the central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (α : ℝ) :
  r = 10 →
  S = 100 →
  S = (1 / 2) * α * r^2 →
  α = 2 := by
  sorry

end sector_central_angle_l2811_281187


namespace polynomial_value_l2811_281118

theorem polynomial_value (x y : ℚ) (h : x - 2 * y - 3 = -5) : 2 * y - x = 2 := by
  sorry

end polynomial_value_l2811_281118


namespace car_travel_distance_l2811_281105

theorem car_travel_distance (rate : ℚ) (time : ℚ) : 
  rate = 3 / 4 → time = 2 → rate * time * 60 = 90 := by
  sorry

end car_travel_distance_l2811_281105


namespace sum_of_cubes_equation_l2811_281153

theorem sum_of_cubes_equation (x y : ℝ) :
  x^3 + 21*x*y + y^3 = 343 → x + y = 7 ∨ x + y = -14 := by
  sorry

end sum_of_cubes_equation_l2811_281153


namespace area_of_M_figure_l2811_281183

-- Define the set of points M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ α : ℝ, (p.1 - 3 * Real.cos α)^2 + (p.2 - 3 * Real.sin α)^2 = 25}

-- Define the area of the figure formed by all points in M
noncomputable def area_of_figure : ℝ := Real.pi * ((3 + 5)^2 - (5 - 3)^2)

-- Theorem statement
theorem area_of_M_figure : area_of_figure = 60 * Real.pi := by sorry

end area_of_M_figure_l2811_281183


namespace point_outside_circle_l2811_281134

/-- A point is outside a circle if its distance from the center is greater than the radius -/
def IsOutsideCircle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2) > radius

/-- Given a circle with radius 3 and a point at distance 5 from the center,
    prove that the point is outside the circle -/
theorem point_outside_circle (O : ℝ × ℝ) (A : ℝ × ℝ) 
    (h1 : Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = 5) :
    IsOutsideCircle O 3 A := by
  sorry


end point_outside_circle_l2811_281134


namespace not_sufficient_nor_necessary_l2811_281122

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  ¬(((a > 0 ∧ b > 0) → (a * b < ((a + b) / 2)^2)) ∧
    ((a * b < ((a + b) / 2)^2) → (a > 0 ∧ b > 0))) := by
  sorry

end not_sufficient_nor_necessary_l2811_281122


namespace students_liking_sports_l2811_281150

theorem students_liking_sports (total : ℕ) (basketball cricket soccer : ℕ)
  (basketball_cricket basketball_soccer cricket_soccer : ℕ) (all_three : ℕ)
  (h1 : total = 30)
  (h2 : basketball = 12)
  (h3 : cricket = 10)
  (h4 : soccer = 8)
  (h5 : basketball_cricket = 4)
  (h6 : basketball_soccer = 3)
  (h7 : cricket_soccer = 2)
  (h8 : all_three = 1) :
  basketball + cricket + soccer - (basketball_cricket + basketball_soccer + cricket_soccer) + all_three = 22 := by
  sorry

end students_liking_sports_l2811_281150


namespace chess_game_most_likely_outcome_l2811_281196

theorem chess_game_most_likely_outcome
  (prob_A_win : ℝ)
  (prob_A_not_lose : ℝ)
  (h1 : prob_A_win = 0.3)
  (h2 : prob_A_not_lose = 0.7)
  (h3 : 0 ≤ prob_A_win ∧ prob_A_win ≤ 1)
  (h4 : 0 ≤ prob_A_not_lose ∧ prob_A_not_lose ≤ 1) :
  let prob_draw := prob_A_not_lose - prob_A_win
  let prob_B_win := 1 - prob_A_not_lose
  prob_draw > prob_A_win ∧ prob_draw > prob_B_win :=
by sorry

end chess_game_most_likely_outcome_l2811_281196


namespace parametric_to_cartesian_l2811_281194

theorem parametric_to_cartesian :
  ∀ (x y : ℝ), (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 * Real.pi ∧ x = 2 * Real.cos t ∧ y = 3 * Real.sin t) →
  x^2 / 4 + y^2 / 9 = 1 :=
by
  sorry

end parametric_to_cartesian_l2811_281194


namespace ellipse_properties_l2811_281112

/-- The ellipse C: x^2/4 + y^2/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l: y = kx, where k ≠ 0 -/
def line_l (k x y : ℝ) : Prop := y = k * x ∧ k ≠ 0

/-- M and N are intersection points of line l and ellipse C -/
def intersection_points (M N : ℝ × ℝ) (k : ℝ) : Prop :=
  ellipse_C M.1 M.2 ∧ ellipse_C N.1 N.2 ∧
  line_l k M.1 M.2 ∧ line_l k N.1 N.2

/-- F₁ and F₂ are the foci of the ellipse C -/
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-1, 0) ∧ F₂ = (1, 0)

/-- B is the top vertex of the ellipse C -/
def top_vertex (B : ℝ × ℝ) : Prop :=
  B = (0, Real.sqrt 3)

/-- The perimeter of quadrilateral MF₁NF₂ is 8 -/
def perimeter_is_8 (M N F₁ F₂ : ℝ × ℝ) : Prop :=
  dist M F₁ + dist F₁ N + dist N F₂ + dist F₂ M = 8

/-- The product of the slopes of lines BM and BN is -3/4 -/
def slope_product (M N B : ℝ × ℝ) : Prop :=
  ((M.2 - B.2) / (M.1 - B.1)) * ((N.2 - B.2) / (N.1 - B.1)) = -3/4

theorem ellipse_properties (k : ℝ) (M N F₁ F₂ B : ℝ × ℝ) :
  intersection_points M N k →
  foci F₁ F₂ →
  top_vertex B →
  perimeter_is_8 M N F₁ F₂ ∧ slope_product M N B :=
sorry

end ellipse_properties_l2811_281112


namespace complex_magnitude_proof_l2811_281139

theorem complex_magnitude_proof : Complex.abs (7/4 - 3*I) = Real.sqrt 193 / 4 := by
  sorry

end complex_magnitude_proof_l2811_281139


namespace hidden_primes_average_l2811_281174

-- Define the visible numbers on the cards
def visible_numbers : List Nat := [44, 59, 38]

-- Define a function to check if a number is prime
def is_prime (n : Nat) : Prop := Nat.Prime n

-- Define the property that the sum of numbers on each card is equal
def equal_sums (x y z : Nat) : Prop :=
  44 + x = 59 + y ∧ 59 + y = 38 + z

-- The main theorem
theorem hidden_primes_average (x y z : Nat) : 
  is_prime x ∧ is_prime y ∧ is_prime z ∧ 
  equal_sums x y z → 
  (x + y + z) / 3 = 14 :=
sorry

end hidden_primes_average_l2811_281174


namespace least_multiple_75_with_digit_product_75_l2811_281125

def is_multiple_of_75 (n : ℕ) : Prop := ∃ k : ℕ, n = 75 * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_75 n ∧ is_multiple_of_75 (digit_product n)

theorem least_multiple_75_with_digit_product_75 :
  satisfies_conditions 75375 ∧ ∀ m : ℕ, m < 75375 → ¬(satisfies_conditions m) :=
sorry

end least_multiple_75_with_digit_product_75_l2811_281125


namespace billy_homework_ratio_l2811_281100

/-- Represents the number of questions solved in each hour -/
structure QuestionsSolved where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents Billy's homework solving session -/
def BillyHomework (qs : QuestionsSolved) : Prop :=
  qs.third = 132 ∧
  qs.third = 2 * qs.second ∧
  qs.first + qs.second + qs.third = 242

theorem billy_homework_ratio (qs : QuestionsSolved) 
  (h : BillyHomework qs) : qs.third / qs.first = 3 := by
  sorry

end billy_homework_ratio_l2811_281100


namespace evaluate_expression_l2811_281193

theorem evaluate_expression : (36 - 6 * 3) / (6 / 3 * 2) = 4.5 := by
  sorry

end evaluate_expression_l2811_281193


namespace triangle_area_ratio_l2811_281136

theorem triangle_area_ratio (k : ℕ) (H : ℝ) (h : ℝ) :
  k > 0 →
  H > 0 →
  h > 0 →
  h / H = 1 / Real.sqrt k →
  (h / H)^2 = 1 / k :=
by sorry


end triangle_area_ratio_l2811_281136


namespace simplify_fraction_product_l2811_281198

theorem simplify_fraction_product : (360 / 32) * (10 / 240) * (16 / 6) = 10 := by
  sorry

end simplify_fraction_product_l2811_281198


namespace hyperbola_eccentricity_l2811_281156

/-- 
Given a hyperbola with equation x^2 + my^2 = 1, where m is a real number,
if its conjugate axis is twice the length of its transverse axis,
then its eccentricity e is equal to √5.
-/
theorem hyperbola_eccentricity (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b = 2*a) →
  (∃ e : ℝ, e = Real.sqrt 5) :=
by sorry

end hyperbola_eccentricity_l2811_281156


namespace perfect_square_4p_minus_3_l2811_281108

theorem perfect_square_4p_minus_3 (n p : ℕ) (hn : n > 1) (hp : p > 1) (p_prime : Nat.Prime p)
  (n_divides_p_minus_1 : n ∣ (p - 1)) (p_divides_n_cube_minus_1 : p ∣ (n^3 - 1)) :
  ∃ k : ℤ, (4 : ℤ) * p - 3 = k^2 := by
sorry

end perfect_square_4p_minus_3_l2811_281108


namespace mary_seth_age_ratio_l2811_281126

/-- Given that Mary is 9 years older than Seth and Seth is currently 3.5 years old,
    prove that the ratio of Mary's age to Seth's age in a year is 3:1. -/
theorem mary_seth_age_ratio :
  ∀ (seth_age mary_age seth_future_age mary_future_age : ℝ),
  seth_age = 3.5 →
  mary_age = seth_age + 9 →
  seth_future_age = seth_age + 1 →
  mary_future_age = mary_age + 1 →
  mary_future_age / seth_future_age = 3 := by
sorry

end mary_seth_age_ratio_l2811_281126


namespace intersection_and_complement_union_l2811_281177

-- Define the universe U as the real numbers
def U := ℝ

-- Define set M
def M : Set ℝ := {x | x ≥ 1}

-- Define set N
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem intersection_and_complement_union :
  (M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 5}) ∧
  ((Mᶜ ∪ Nᶜ) = {x : ℝ | x < 1 ∨ x ≥ 5}) := by
  sorry

end intersection_and_complement_union_l2811_281177


namespace survey_analysis_l2811_281170

-- Define the survey data
structure SurveyData where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  male_like : ℕ
  female_like : ℕ
  male_dislike : ℕ
  female_dislike : ℕ

-- Define the theorem
theorem survey_analysis (data : SurveyData) 
  (h1 : data.total_students = 400 + data.female_like + data.male_dislike)
  (h2 : data.male_students = 280 + data.male_dislike)
  (h3 : data.female_students = 120 + data.female_like)
  (h4 : data.male_students = (4 : ℚ) / 7 * data.total_students)
  (h5 : data.female_like = (3 : ℚ) / 5 * data.female_students)
  (h6 : data.male_like = 280)
  (h7 : data.female_dislike = 120) :
  data.female_like = 180 ∧ 
  data.male_dislike = 120 ∧ 
  ((700 : ℚ) * (280 * 120 - 180 * 120)^2 / (460 * 240 * 400 * 300) < (10828 : ℚ) / 1000) :=
sorry


end survey_analysis_l2811_281170


namespace range_of_m_l2811_281130

-- Define the equation
def equation (m x : ℝ) : Prop := (m - 1) / (x + 1) = 1

-- Define the theorem
theorem range_of_m (m x : ℝ) : 
  equation m x ∧ x < 0 → m < 2 ∧ m ≠ 1 :=
by sorry

end range_of_m_l2811_281130


namespace sqrt_14_bounds_l2811_281142

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end sqrt_14_bounds_l2811_281142


namespace integer_root_quadratic_count_l2811_281168

theorem integer_root_quadratic_count :
  ∃! (S : Finset ℝ), 
    Finset.card S = 8 ∧ 
    (∀ a ∈ S, ∃ r s : ℤ, 
      (∀ x : ℝ, x^2 + a*x + 12*a = 0 ↔ x = r ∨ x = s)) :=
sorry

end integer_root_quadratic_count_l2811_281168


namespace cake_mix_distribution_l2811_281178

theorem cake_mix_distribution (first_tray second_tray total : ℕ) : 
  first_tray = second_tray + 20 →
  first_tray + second_tray = 500 →
  second_tray = 240 := by
sorry

end cake_mix_distribution_l2811_281178


namespace problem_statement_l2811_281169

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 :=
by sorry

end problem_statement_l2811_281169


namespace genuine_purses_and_handbags_l2811_281161

theorem genuine_purses_and_handbags 
  (total_purses : ℕ) 
  (total_handbags : ℕ) 
  (fake_purses_ratio : ℚ) 
  (fake_handbags_ratio : ℚ) 
  (h1 : total_purses = 26) 
  (h2 : total_handbags = 24) 
  (h3 : fake_purses_ratio = 1/2) 
  (h4 : fake_handbags_ratio = 1/4) :
  (total_purses - total_purses * fake_purses_ratio) + 
  (total_handbags - total_handbags * fake_handbags_ratio) = 31 := by
sorry

end genuine_purses_and_handbags_l2811_281161


namespace highway_length_l2811_281147

theorem highway_length (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 14)
  (h2 : speed2 = 16)
  (h3 : time = 1.5)
  : speed1 * time + speed2 * time = 45 := by
  sorry

end highway_length_l2811_281147


namespace amy_age_2005_l2811_281162

/-- Amy's age at the end of 2000 -/
def amy_age_2000 : ℕ := sorry

/-- Amy's grandfather's age at the end of 2000 -/
def grandfather_age_2000 : ℕ := sorry

/-- The year 2000 -/
def year_2000 : ℕ := 2000

/-- The sum of Amy's and her grandfather's birth years -/
def birth_years_sum : ℕ := 3900

theorem amy_age_2005 : 
  grandfather_age_2000 = 3 * amy_age_2000 →
  year_2000 - amy_age_2000 + (year_2000 - grandfather_age_2000) = birth_years_sum →
  amy_age_2000 + 5 = 30 := by sorry

end amy_age_2005_l2811_281162


namespace similar_triangles_height_l2811_281149

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  let h_large := h_small * Real.sqrt area_ratio
  h_small = 5 →
  h_large = 15 := by
sorry

end similar_triangles_height_l2811_281149


namespace five_point_questions_count_l2811_281175

/-- Represents a test with two types of questions -/
structure Test where
  total_points : ℕ
  total_questions : ℕ
  five_point_questions : ℕ
  ten_point_questions : ℕ

/-- Checks if a test configuration is valid -/
def is_valid_test (t : Test) : Prop :=
  t.total_questions = t.five_point_questions + t.ten_point_questions ∧
  t.total_points = 5 * t.five_point_questions + 10 * t.ten_point_questions

theorem five_point_questions_count (t : Test) 
  (h1 : t.total_points = 200)
  (h2 : t.total_questions = 30)
  (h3 : is_valid_test t) :
  t.five_point_questions = 20 := by
  sorry

end five_point_questions_count_l2811_281175


namespace chord_sum_squares_l2811_281107

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 100}

def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry
def E : ℝ × ℝ := sorry

-- State the theorem
theorem chord_sum_squares (h1 : A ∈ Circle) (h2 : B ∈ Circle) (h3 : C ∈ Circle) (h4 : D ∈ Circle) (h5 : E ∈ Circle)
  (h6 : A.1 = -B.1 ∧ A.2 = -B.2) -- AB is a diameter
  (h7 : (E.1 - C.1) * (B.1 - A.1) + (E.2 - C.2) * (B.2 - A.2) = 0) -- CD intersects AB at E
  (h8 : (B.1 - E.1)^2 + (B.2 - E.2)^2 = 40) -- BE = 2√10
  (h9 : (A.1 - E.1) * (C.1 - E.1) + (A.2 - E.2) * (C.2 - E.2) = 
        Real.sqrt 3 * Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) * Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2) / 2) -- Angle AEC = 30°
  : (C.1 - E.1)^2 + (C.2 - E.2)^2 + (D.1 - E.1)^2 + (D.2 - E.2)^2 = 200 := by
  sorry

end chord_sum_squares_l2811_281107


namespace complex_equidistant_point_l2811_281180

theorem complex_equidistant_point : ∃! (z : ℂ), Complex.abs (z - 2) = Complex.abs (z + 4) ∧ Complex.abs (z - 2) = Complex.abs (z - 3*I) := by
  sorry

end complex_equidistant_point_l2811_281180


namespace hyperbola_unique_solution_l2811_281157

/-- The hyperbola equation -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (2 * m^2) - y^2 / (3 * m) = 1

/-- The focal length of the hyperbola -/
def focal_length : ℝ := 6

/-- Theorem stating that 3/2 is the only positive real solution for m -/
theorem hyperbola_unique_solution :
  ∃! m : ℝ, m > 0 ∧ 
  (∀ x y : ℝ, hyperbola_equation x y m) ∧
  (∃ c : ℝ, c^2 = 2 * m^2 + 3 * m ∧ c = focal_length / 2) :=
by
  sorry

end hyperbola_unique_solution_l2811_281157


namespace lions_in_first_group_l2811_281138

/-- The killing rate of lions in deers per minute -/
def killing_rate (lions : ℕ) (deers : ℕ) (minutes : ℕ) : ℚ :=
  (deers : ℚ) / (lions : ℚ) / (minutes : ℚ)

/-- The number of lions in the first group -/
def first_group_lions : ℕ := 10

theorem lions_in_first_group :
  (killing_rate first_group_lions 10 10 = killing_rate 100 100 10) →
  first_group_lions = 10 := by
  sorry

end lions_in_first_group_l2811_281138


namespace pitchers_needed_l2811_281124

def glasses_per_pitcher : ℝ := 4.5
def total_glasses_served : ℕ := 30

theorem pitchers_needed : 
  ∃ (n : ℕ), n * glasses_per_pitcher ≥ total_glasses_served ∧ 
  ∀ (m : ℕ), m * glasses_per_pitcher ≥ total_glasses_served → n ≤ m :=
by sorry

end pitchers_needed_l2811_281124


namespace consecutive_numbers_product_divisibility_l2811_281165

theorem consecutive_numbers_product_divisibility (n : ℕ) (hn : n > 1) :
  ∃ k : ℕ, ∀ p : ℕ,
    Prime p →
    (∀ i : ℕ, i < n → (p ∣ (k + i + 1))) ↔ p ≤ 2 * n + 1 := by
  sorry

end consecutive_numbers_product_divisibility_l2811_281165


namespace dvd_shipping_cost_percentage_l2811_281160

/-- Given Mike's DVD cost, Steve's DVD cost as twice Mike's, and Steve's total cost,
    prove that the shipping cost percentage of Steve's DVD price is 80% -/
theorem dvd_shipping_cost_percentage
  (mike_cost : ℝ)
  (steve_dvd_cost : ℝ)
  (steve_total_cost : ℝ)
  (h1 : mike_cost = 5)
  (h2 : steve_dvd_cost = 2 * mike_cost)
  (h3 : steve_total_cost = 18) :
  (steve_total_cost - steve_dvd_cost) / steve_dvd_cost * 100 = 80 := by
  sorry

end dvd_shipping_cost_percentage_l2811_281160


namespace largest_angle_of_inclination_l2811_281114

-- Define the angle of inclination for a line given its slope
noncomputable def angle_of_inclination (slope : ℝ) : ℝ :=
  Real.arctan slope * (180 / Real.pi)

-- Define the lines
def line_A : ℝ → ℝ := λ x => -x + 1
def line_B : ℝ → ℝ := λ x => x + 1
def line_C : ℝ → ℝ := λ x => 2*x + 1
def line_D : ℝ → ℝ := λ _ => 1

-- Theorem statement
theorem largest_angle_of_inclination :
  let angle_A := angle_of_inclination (-1)
  let angle_B := angle_of_inclination 1
  let angle_C := angle_of_inclination 2
  let angle_D := 90
  angle_A > angle_B ∧ angle_A > angle_C ∧ angle_A > angle_D :=
by sorry


end largest_angle_of_inclination_l2811_281114


namespace exactly_three_prime_values_l2811_281179

def polynomial (n : ℕ+) : ℤ := (n.val : ℤ)^3 - 6*(n.val : ℤ)^2 + 17*(n.val : ℤ) - 19

def is_prime_for_n (n : ℕ+) : Prop := Nat.Prime (Int.natAbs (polynomial n))

theorem exactly_three_prime_values :
  ∃! (s : Finset ℕ+), s.card = 3 ∧ ∀ n, n ∈ s ↔ is_prime_for_n n :=
sorry

end exactly_three_prime_values_l2811_281179


namespace voltage_meter_max_value_l2811_281191

/-- Represents a voltage meter with a maximum recordable value -/
structure VoltageMeter where
  max_value : ℝ
  records_nonnegative : 0 ≤ max_value

/-- Theorem: Given the conditions, the maximum recordable value is 14 volts -/
theorem voltage_meter_max_value (meter : VoltageMeter) 
  (avg_recording : ℝ) 
  (min_recording : ℝ) 
  (h1 : avg_recording = 6)
  (h2 : min_recording = 2)
  (h3 : ∃ (a b c : ℝ), 
    0 ≤ a ∧ a ≤ meter.max_value ∧
    0 ≤ b ∧ b ≤ meter.max_value ∧
    0 ≤ c ∧ c ≤ meter.max_value ∧
    (a + b + c) / 3 = avg_recording ∧
    min_recording ≤ a ∧ min_recording ≤ b ∧ min_recording ≤ c) :
  meter.max_value = 14 := by
sorry

end voltage_meter_max_value_l2811_281191


namespace volunteer_assignment_count_l2811_281159

/-- The number of ways to assign volunteers to areas --/
def assignmentCount (volunteers : ℕ) (areas : ℕ) : ℕ :=
  areas^volunteers - areas * (areas - 1)^volunteers + areas * (areas - 2)^volunteers

/-- Theorem stating that the number of ways to assign 5 volunteers to 3 areas,
    with at least one volunteer in each area, is equal to 150 --/
theorem volunteer_assignment_count :
  assignmentCount 5 3 = 150 := by
  sorry

end volunteer_assignment_count_l2811_281159


namespace jersey_t_shirt_price_difference_l2811_281144

/-- The price difference between a jersey and a t-shirt -/
def price_difference (jersey_profit t_shirt_profit : ℕ) : ℕ :=
  jersey_profit - t_shirt_profit

/-- Theorem stating that the price difference between a jersey and a t-shirt is $90 -/
theorem jersey_t_shirt_price_difference :
  price_difference 115 25 = 90 := by
  sorry

end jersey_t_shirt_price_difference_l2811_281144


namespace jake_weight_loss_l2811_281173

/-- Given that Jake and his sister together weigh 156 pounds and Jake's current weight is 108 pounds,
    this theorem proves that Jake needs to lose 12 pounds to weigh twice as much as his sister. -/
theorem jake_weight_loss (total_weight : ℕ) (jake_weight : ℕ) 
  (h1 : total_weight = 156)
  (h2 : jake_weight = 108) :
  jake_weight - (2 * (total_weight - jake_weight)) = 12 := by
  sorry

#check jake_weight_loss

end jake_weight_loss_l2811_281173


namespace liquid_film_radius_l2811_281110

/-- The radius of a circular film formed by a liquid on water -/
theorem liquid_film_radius 
  (thickness : ℝ) 
  (volume : ℝ) 
  (h1 : thickness = 0.2)
  (h2 : volume = 320) : 
  ∃ (r : ℝ), r = Real.sqrt (1600 / Real.pi) ∧ π * r^2 * thickness = volume :=
sorry

end liquid_film_radius_l2811_281110


namespace total_marbles_is_172_l2811_281117

/-- Represents the number of marbles of each color in a bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  purple : ℕ

/-- Checks if the given MarbleBag satisfies the ratio conditions -/
def satisfiesRatios (bag : MarbleBag) : Prop :=
  7 * bag.red = 4 * bag.blue ∧ 3 * bag.blue = 2 * bag.purple

/-- Theorem: Given the conditions, the total number of marbles is 172 -/
theorem total_marbles_is_172 (bag : MarbleBag) 
  (h1 : satisfiesRatios bag) 
  (h2 : bag.red = 32) : 
  bag.red + bag.blue + bag.purple = 172 := by
  sorry

#check total_marbles_is_172

end total_marbles_is_172_l2811_281117


namespace pizza_coverage_l2811_281189

theorem pizza_coverage (pizza_diameter : ℝ) (pepperoni_diameter : ℝ) (num_pepperoni : ℕ) : 
  pizza_diameter = 2 * pepperoni_diameter →
  num_pepperoni = 32 →
  (num_pepperoni * (pepperoni_diameter / 2)^2 * π) / ((pizza_diameter / 2)^2 * π) = 1 / 2 := by
  sorry

end pizza_coverage_l2811_281189


namespace minimum_a_value_l2811_281135

theorem minimum_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 1 → |x| < a) → a ≥ 1 := by
  sorry

end minimum_a_value_l2811_281135


namespace ratio_equality_l2811_281131

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (norm_abc : a^2 + b^2 + c^2 = 25)
  (norm_xyz : x^2 + y^2 + z^2 = 36)
  (dot_product : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
sorry

end ratio_equality_l2811_281131


namespace isosceles_triangle_quadratic_roots_l2811_281133

theorem isosceles_triangle_quadratic_roots (k : ℝ) : 
  (∃ (a b : ℝ), 
    (a^2 - 6*a + k = 0) ∧ 
    (b^2 - 6*b + k = 0) ∧ 
    (a = b ∨ a = 2 ∨ b = 2) ∧
    (a + b > 2) ∧ (a + 2 > b) ∧ (b + 2 > a)) → k = 9 :=
by sorry

end isosceles_triangle_quadratic_roots_l2811_281133


namespace product_of_five_consecutive_integers_not_square_l2811_281151

theorem product_of_five_consecutive_integers_not_square (n : ℤ) : 
  ∃ (m : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ≠ m ^ 2 := by
sorry

end product_of_five_consecutive_integers_not_square_l2811_281151


namespace geometry_biology_overlap_l2811_281103

theorem geometry_biology_overlap (total : ℕ) (geometry : ℕ) (biology : ℕ) 
  (h1 : total = 350) (h2 : geometry = 210) (h3 : biology = 175) :
  let max_overlap := min geometry biology
  let min_overlap := max 0 (geometry + biology - total)
  max_overlap - min_overlap = 140 := by
sorry

end geometry_biology_overlap_l2811_281103


namespace first_half_speed_l2811_281111

/-- Proves that given a trip of 8 hours, where the second half is traveled at 85 km/h, 
    and the total distance is 620 km, the speed during the first half of the trip is 70 km/h. -/
theorem first_half_speed 
  (total_time : ℝ) 
  (second_half_speed : ℝ) 
  (total_distance : ℝ) 
  (h1 : total_time = 8) 
  (h2 : second_half_speed = 85) 
  (h3 : total_distance = 620) : 
  (total_distance - (second_half_speed * (total_time / 2))) / (total_time / 2) = 70 := by
sorry

end first_half_speed_l2811_281111


namespace product_of_reals_l2811_281164

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 8) (cube_sum_eq : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end product_of_reals_l2811_281164


namespace principal_is_2000_l2811_281167

/-- Calculates the simple interest for a given principal, rate, and time. -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

/-- Proves that the principal is 2000 given the conditions of the problem. -/
theorem principal_is_2000 (rate : ℚ) (time : ℚ) (interest : ℚ) 
    (h_rate : rate = 5)
    (h_time : time = 13)
    (h_interest : interest = 1300)
    (h_simple_interest : simpleInterest principal rate time = interest) :
  principal = 2000 := by
  sorry

#check principal_is_2000

end principal_is_2000_l2811_281167


namespace equation_solution_l2811_281140

theorem equation_solution : ∃ x : ℚ, (x - 75) / 3 = (8 - 3*x) / 4 ∧ x = 324 / 13 := by
  sorry

end equation_solution_l2811_281140


namespace candy_bar_cost_l2811_281116

/-- The cost of a candy bar given initial and remaining amounts -/
theorem candy_bar_cost (initial_amount : ℚ) (remaining_amount : ℚ) 
  (h1 : initial_amount = 3)
  (h2 : remaining_amount = 2) :
  initial_amount - remaining_amount = 1 := by
sorry

end candy_bar_cost_l2811_281116


namespace roots_of_derivative_in_triangle_l2811_281163

open Complex

-- Define the polynomial f(x) = (x-a)(x-b)(x-c)
def f (x a b c : ℂ) : ℂ := (x - a) * (x - b) * (x - c)

-- Define the derivative of f
def f_derivative (x a b c : ℂ) : ℂ := 
  (x - b) * (x - c) + (x - a) * (x - c) + (x - a) * (x - b)

-- Define a triangle in the complex plane
def triangle_contains (a b c z : ℂ) : Prop :=
  ∃ (t1 t2 t3 : ℝ), t1 ≥ 0 ∧ t2 ≥ 0 ∧ t3 ≥ 0 ∧ t1 + t2 + t3 = 1 ∧
    z = t1 • a + t2 • b + t3 • c

-- Theorem statement
theorem roots_of_derivative_in_triangle (a b c : ℂ) :
  ∀ z : ℂ, f_derivative z a b c = 0 → triangle_contains a b c z :=
sorry

end roots_of_derivative_in_triangle_l2811_281163


namespace S_eq_EvenPositive_l2811_281145

/-- The set of all positive integers that can be written in the form ([x, y] + [y, z]) / [x, z] -/
def S : Set ℕ+ :=
  {n | ∃ (x y z : ℕ+), n = (Nat.lcm x y + Nat.lcm y z) / Nat.lcm x z}

/-- The set of all even positive integers -/
def EvenPositive : Set ℕ+ :=
  {n | ∃ (k : ℕ+), n = 2 * k}

/-- Theorem stating that S is equal to the set of all even positive integers -/
theorem S_eq_EvenPositive : S = EvenPositive := by
  sorry

end S_eq_EvenPositive_l2811_281145


namespace meaningful_expression_range_l2811_281186

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (2 * x - 7) + Real.sqrt (5 - x)) ↔ 3.5 ≤ x ∧ x ≤ 5 :=
by sorry

end meaningful_expression_range_l2811_281186


namespace courtyard_paving_l2811_281148

/-- Given a rectangular courtyard and rectangular bricks, calculate the number of bricks needed to pave the courtyard -/
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width : ℕ) 
  (h1 : courtyard_length = 25)
  (h2 : courtyard_width = 16)
  (h3 : brick_length = 20)
  (h4 : brick_width = 10) :
  (courtyard_length * 100) * (courtyard_width * 100) / (brick_length * brick_width) = 20000 := by
  sorry

#check courtyard_paving

end courtyard_paving_l2811_281148


namespace cube_root_64_minus_sqrt_8_squared_l2811_281102

theorem cube_root_64_minus_sqrt_8_squared : 
  (64 ^ (1/3) - Real.sqrt 8) ^ 2 = 24 - 16 * Real.sqrt 2 := by
  sorry

end cube_root_64_minus_sqrt_8_squared_l2811_281102


namespace distribute_5_4_l2811_281192

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
theorem distribute_5_4 : distribute 5 4 = 67 := by sorry

end distribute_5_4_l2811_281192


namespace fish_count_l2811_281115

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 8

/-- The number of fish Max has -/
def max_fish : ℕ := 15

/-- The total number of fish Lilly, Rosy, and Max have -/
def total_fish : ℕ := lilly_fish + rosy_fish + max_fish

theorem fish_count : total_fish = 33 := by sorry

end fish_count_l2811_281115


namespace senior_tickets_first_day_l2811_281166

/- Define the variables -/
def student_ticket_price : ℕ := 9
def first_day_student_tickets : ℕ := 3
def first_day_total : ℕ := 79
def second_day_senior_tickets : ℕ := 12
def second_day_student_tickets : ℕ := 10
def second_day_total : ℕ := 246

/- Theorem to prove -/
theorem senior_tickets_first_day :
  ∃ (senior_ticket_price : ℕ) (first_day_senior_tickets : ℕ),
    senior_ticket_price * second_day_senior_tickets + 
    student_ticket_price * second_day_student_tickets = second_day_total ∧
    senior_ticket_price * first_day_senior_tickets + 
    student_ticket_price * first_day_student_tickets = first_day_total ∧
    first_day_senior_tickets = 4 :=
by
  sorry

end senior_tickets_first_day_l2811_281166


namespace fifth_seat_is_37_l2811_281146

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  selectedSeats : Finset ℕ

/-- The seat number of the fifth selected student in the systematic sampling. -/
def fifthSelectedSeat (sampling : SystematicSampling) : ℕ :=
  37

/-- Theorem stating that given the conditions, the fifth selected seat is 37. -/
theorem fifth_seat_is_37 (sampling : SystematicSampling) 
  (h1 : sampling.totalStudents = 55)
  (h2 : sampling.sampleSize = 5)
  (h3 : sampling.selectedSeats = {4, 15, 26, 48}) :
  fifthSelectedSeat sampling = 37 := by
  sorry

#check fifth_seat_is_37

end fifth_seat_is_37_l2811_281146


namespace zoe_leo_difference_l2811_281106

-- Define variables
variable (t : ℝ) -- Leo's driving time
variable (s : ℝ) -- Leo's speed

-- Define Leo's distance
def leo_distance (t s : ℝ) : ℝ := t * s

-- Define Maria's distance
def maria_distance (t s : ℝ) : ℝ := (t + 2) * (s + 15)

-- Define Zoe's distance
def zoe_distance (t s : ℝ) : ℝ := (t + 3) * (s + 20)

-- Theorem statement
theorem zoe_leo_difference (t s : ℝ) :
  maria_distance t s = leo_distance t s + 110 →
  zoe_distance t s - leo_distance t s = 180 := by
  sorry


end zoe_leo_difference_l2811_281106


namespace excavator_transport_theorem_l2811_281123

/-- Represents the transportation problem for excavators after an earthquake. -/
structure ExcavatorTransport where
  area_a_need : ℕ := 27
  area_b_need : ℕ := 25
  province_a_donate : ℕ := 28
  province_b_donate : ℕ := 24
  cost_a_to_a : ℚ := 0.4
  cost_a_to_b : ℚ := 0.3
  cost_b_to_a : ℚ := 0.5
  cost_b_to_b : ℚ := 0.2

/-- The functional relationship between total cost y and number of excavators x
    transported from Province A to Area A. -/
def total_cost (et : ExcavatorTransport) (x : ℕ) : ℚ :=
  et.cost_a_to_a * x + et.cost_a_to_b * (et.province_a_donate - x) +
  et.cost_b_to_a * (et.area_a_need - x) + et.cost_b_to_b * (x - 3)

/-- The theorem stating the functional relationship and range of x. -/
theorem excavator_transport_theorem (et : ExcavatorTransport) :
  ∀ x : ℕ, 3 ≤ x ∧ x ≤ 27 →
    total_cost et x = -0.2 * x + 21.3 ∧
    (∀ y : ℚ, y = total_cost et x → -0.2 * x + 21.3 = y) := by
  sorry

#check excavator_transport_theorem

end excavator_transport_theorem_l2811_281123
