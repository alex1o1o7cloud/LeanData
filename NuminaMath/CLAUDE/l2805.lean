import Mathlib

namespace NUMINAMATH_CALUDE_complex_magnitude_l2805_280552

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2805_280552


namespace NUMINAMATH_CALUDE_complex_division_result_l2805_280556

theorem complex_division_result : (5 - I) / (1 - I) = 3 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l2805_280556


namespace NUMINAMATH_CALUDE_cube_volume_from_circumscribed_sphere_l2805_280597

/-- Given a cube with a circumscribed sphere of volume 32π/3, the volume of the cube is 64√3/9 -/
theorem cube_volume_from_circumscribed_sphere (V_sphere : ℝ) (V_cube : ℝ) :
  V_sphere = 32 / 3 * Real.pi → V_cube = 64 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_circumscribed_sphere_l2805_280597


namespace NUMINAMATH_CALUDE_minimum_teams_l2805_280598

theorem minimum_teams (total_players : Nat) (max_team_size : Nat) : total_players = 30 → max_team_size = 8 → ∃ (num_teams : Nat), num_teams = 5 ∧ 
  (∃ (players_per_team : Nat), 
    players_per_team ≤ max_team_size ∧ 
    total_players = num_teams * players_per_team ∧
    ∀ (x : Nat), x < num_teams → 
      total_players % x ≠ 0 ∨ (total_players / x) > max_team_size) := by
  sorry

end NUMINAMATH_CALUDE_minimum_teams_l2805_280598


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2805_280514

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < a) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2805_280514


namespace NUMINAMATH_CALUDE_incorrect_equality_l2805_280575

theorem incorrect_equality (h : (12.5 / 12.5) = (2.4 / 2.4)) :
  ¬ (25 * (0.5 / 0.5) = 4 * (0.6 / 0.6)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_equality_l2805_280575


namespace NUMINAMATH_CALUDE_pizza_counting_theorem_l2805_280524

/-- The number of available pizza toppings -/
def num_toppings : ℕ := 6

/-- Calculates the number of pizzas with exactly k toppings -/
def pizzas_with_k_toppings (k : ℕ) : ℕ := Nat.choose num_toppings k

/-- The total number of pizzas with one, two, or three toppings -/
def total_pizzas : ℕ := 
  pizzas_with_k_toppings 1 + pizzas_with_k_toppings 2 + pizzas_with_k_toppings 3

theorem pizza_counting_theorem : total_pizzas = 41 := by
  sorry

end NUMINAMATH_CALUDE_pizza_counting_theorem_l2805_280524


namespace NUMINAMATH_CALUDE_car_speed_problem_l2805_280539

/-- Proves that car R's speed is 75 mph given the problem conditions -/
theorem car_speed_problem (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  distance = 1200 →
  time_difference = 4 →
  speed_difference = 20 →
  ∃ (speed_R : ℝ),
    distance / speed_R - time_difference = distance / (speed_R + speed_difference) ∧
    speed_R = 75 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_problem_l2805_280539


namespace NUMINAMATH_CALUDE_six_times_r_of_30_l2805_280537

def r (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem six_times_r_of_30 : r (r (r (r (r (r 30))))) = 144 / 173 := by
  sorry

end NUMINAMATH_CALUDE_six_times_r_of_30_l2805_280537


namespace NUMINAMATH_CALUDE_mean_of_data_is_10_l2805_280520

def data : List ℝ := [8, 12, 10, 11, 9]

theorem mean_of_data_is_10 :
  (data.sum / data.length : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_data_is_10_l2805_280520


namespace NUMINAMATH_CALUDE_pipe_fill_time_l2805_280530

theorem pipe_fill_time (fill_time_A fill_time_all empty_time : ℝ) 
  (h1 : fill_time_A = 60)
  (h2 : fill_time_all = 50)
  (h3 : empty_time = 100.00000000000001) :
  ∃ fill_time_B : ℝ, fill_time_B = 75 ∧ 
  (1 / fill_time_A + 1 / fill_time_B - 1 / empty_time = 1 / fill_time_all) := by
  sorry

#check pipe_fill_time

end NUMINAMATH_CALUDE_pipe_fill_time_l2805_280530


namespace NUMINAMATH_CALUDE_linear_function_theorem_l2805_280555

/-- A linear function that intersects the x-axis at (-2, 0) and forms a triangle with area 8 with the coordinate axes -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  x_intercept : k * (-2) + b = 0
  triangle_area : |k * 2 * b / 2| = 8

/-- The two possible linear functions satisfying the given conditions -/
def possible_functions : Set LinearFunction :=
  { f | f.k = 4 ∧ f.b = 8 } ∪ { f | f.k = -4 ∧ f.b = -8 }

/-- Theorem stating that the only linear functions satisfying the conditions are y = 4x + 8 or y = -4x - 8 -/
theorem linear_function_theorem :
  ∀ f : LinearFunction, f ∈ possible_functions :=
by sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l2805_280555


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_l2805_280527

def n : ℕ := 1020000000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a > b ∧ b > c ∧ c > d ∧ d > e ∧ 
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    ∀ (x : ℕ), x ∣ n → x ≤ e ∨ x = d ∨ x = c ∨ x = b ∨ x = a ∨ x = n)

theorem fifth_largest_divisor : is_fifth_largest_divisor 63750000 := by
  sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_l2805_280527


namespace NUMINAMATH_CALUDE_sequence_properties_l2805_280506

def sequence_a (n : ℕ) : ℝ := sorry
def sequence_b (n : ℕ) : ℝ := sorry
def sequence_c (n : ℕ) : ℝ := sequence_a n * sequence_b n

def sum_S (n : ℕ) : ℝ := sorry
def sum_T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, sequence_a n = (sum_S n + 2) / 2) ∧
  (sequence_b 1 = 1) ∧
  (∀ n : ℕ, sequence_b n - sequence_b (n + 1) + 2 = 0) →
  (sequence_a 1 = 2) ∧
  (sequence_a 2 = 4) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2^n) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_b n = 2*n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → sum_T n = (2*n - 3) * 2^(n+1) + 6) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2805_280506


namespace NUMINAMATH_CALUDE_average_speed_uphill_downhill_l2805_280501

/-- Theorem: Average speed of a car traveling uphill and downhill -/
theorem average_speed_uphill_downhill 
  (uphill_speed : ℝ) 
  (downhill_speed : ℝ) 
  (uphill_distance : ℝ) 
  (downhill_distance : ℝ) 
  (h1 : uphill_speed = 30) 
  (h2 : downhill_speed = 40) 
  (h3 : uphill_distance = 100) 
  (h4 : downhill_distance = 50) : 
  (uphill_distance + downhill_distance) / 
  (uphill_distance / uphill_speed + downhill_distance / downhill_speed) = 1800 / 55 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_uphill_downhill_l2805_280501


namespace NUMINAMATH_CALUDE_divisible_by_nine_sequence_l2805_280590

theorem divisible_by_nine_sequence (start : ℕ) (h1 : start ≥ 32) (h2 : start % 9 = 0) : 
  let sequence := List.range 7
  let last_number := start + 9 * 6
  last_number = 90 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_nine_sequence_l2805_280590


namespace NUMINAMATH_CALUDE_sin_alpha_cos_beta_value_l2805_280540

theorem sin_alpha_cos_beta_value (α β : Real) 
  (h1 : Real.sin (α + β) = 1/2) 
  (h2 : Real.sin (α - β) = 1/4) : 
  Real.sin α * Real.cos β = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_cos_beta_value_l2805_280540


namespace NUMINAMATH_CALUDE_cannot_determine_unique_order_l2805_280515

/-- Represents a query about the relative ordering of 3 weights -/
structure Query where
  a : Fin 5
  b : Fin 5
  c : Fin 5
  h₁ : a ≠ b
  h₂ : b ≠ c
  h₃ : a ≠ c

/-- Represents a permutation of 5 weights -/
def Permutation := Fin 5 → Fin 5

/-- Checks if a permutation is consistent with a query -/
def consistentWithQuery (p : Permutation) (q : Query) : Prop :=
  p q.a < p q.b ∧ p q.b < p q.c

/-- Checks if a permutation is consistent with all queries in a list -/
def consistentWithAllQueries (p : Permutation) (qs : List Query) : Prop :=
  ∀ q ∈ qs, consistentWithQuery p q

theorem cannot_determine_unique_order :
  ∀ (qs : List Query),
    qs.length = 9 →
    ∃ (p₁ p₂ : Permutation),
      p₁ ≠ p₂ ∧
      consistentWithAllQueries p₁ qs ∧
      consistentWithAllQueries p₂ qs :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_unique_order_l2805_280515


namespace NUMINAMATH_CALUDE_g_composition_15_l2805_280521

def g (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_composition_15 : g (g (g (g 15))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_15_l2805_280521


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l2805_280596

theorem difference_of_squares_example : (527 : ℤ) * 527 - 526 * 528 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l2805_280596


namespace NUMINAMATH_CALUDE_triangle_theorem_l2805_280560

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sinA : ℝ
  sinB : ℝ
  sinC : ℝ
  cosA : ℝ
  cosB : ℝ
  cosC : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 * t.b ∧ 
  t.sinC = 3/4 ∧ 
  t.b^2 + t.b * t.c = 2 * t.a^2

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.sinB = 3/8 ∧ t.cosB = (3 * Real.sqrt 6) / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2805_280560


namespace NUMINAMATH_CALUDE_lcm_problem_l2805_280532

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 11) (h2 : a * b = 1991) :
  Nat.lcm a b = 181 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2805_280532


namespace NUMINAMATH_CALUDE_product_and_closest_value_l2805_280578

def calculate_product : ℝ := 2.5 * (53.6 - 0.4)

def options : List ℝ := [120, 130, 133, 140, 150]

theorem product_and_closest_value :
  calculate_product = 133 ∧
  ∀ x ∈ options, |calculate_product - 133| ≤ |calculate_product - x| :=
by sorry

end NUMINAMATH_CALUDE_product_and_closest_value_l2805_280578


namespace NUMINAMATH_CALUDE_sum_twenty_from_negative_nine_l2805_280580

/-- The sum of n consecutive integers starting from a given first term -/
def sumConsecutiveIntegers (n : ℕ) (first : ℤ) : ℤ :=
  n * (2 * first + n - 1) / 2

/-- Theorem: The sum of 20 consecutive integers starting from -9 is 10 -/
theorem sum_twenty_from_negative_nine :
  sumConsecutiveIntegers 20 (-9) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_twenty_from_negative_nine_l2805_280580


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2805_280577

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence : 
  let a₁ := 8
  let a₂ := 5
  let a₃ := 2
  let d := a₂ - a₁
  arithmeticSequence a₁ d 30 = -79 := by
sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2805_280577


namespace NUMINAMATH_CALUDE_ellipse_chord_ratio_theorem_l2805_280585

noncomputable section

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: For any ellipse satisfying the given conditions, 
    the ratio of the square of the chord length passing through the origin 
    to the chord length passing through the left focus is always 4, 
    when the slope angles of these chords sum to π -/
theorem ellipse_chord_ratio_theorem (e : Ellipse) 
    (h_focus : e.eccentricity * e.a = 1)
    (h_b_mean : e.b^2 = 3 * e.eccentricity * e.a)
    (α β : ℝ)
    (h_angle_sum : α + β = π)
    (A B D E : Point)
    (h_AB_on_ellipse : A.x^2 / e.a^2 + A.y^2 / e.b^2 = 1 ∧ 
                       B.x^2 / e.a^2 + B.y^2 / e.b^2 = 1)
    (h_DE_on_ellipse : D.x^2 / e.a^2 + D.y^2 / e.b^2 = 1 ∧ 
                       E.x^2 / e.a^2 + E.y^2 / e.b^2 = 1)
    (h_AB_through_origin : ∃ (k : ℝ), A.y = k * A.x ∧ B.y = k * B.x)
    (h_DE_through_focus : ∃ (m : ℝ), D.y = m * (D.x + 1) ∧ E.y = m * (E.x + 1))
    (h_AB_slope : ∃ (k : ℝ), k = Real.tan α)
    (h_DE_slope : ∃ (m : ℝ), m = Real.tan β) :
    (distance A B)^2 / (distance D E) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_ratio_theorem_l2805_280585


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2805_280522

theorem fixed_point_on_line (a b : ℝ) (h : a + 2 * b = 1) :
  a * (1/2) + 3 * (-1/6) + b = 0 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2805_280522


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2805_280533

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 15 → n = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2805_280533


namespace NUMINAMATH_CALUDE_circle_properties_l2805_280544

/-- A circle with center on the y-axis, radius 1, passing through (1, 2) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

theorem circle_properties :
  ∃ (b : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ x^2 + (y - b)^2 = 1) ∧
    (circle_equation 1 2) ∧
    (∀ x y : ℝ, circle_equation x y → x^2 + y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l2805_280544


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l2805_280587

theorem similar_triangles_shortest_side 
  (a b c : ℝ) -- sides of the first triangle
  (d e f : ℝ) -- sides of the second triangle
  (h1 : a^2 + b^2 = c^2) -- first triangle is right-angled
  (h2 : d^2 + e^2 = f^2) -- second triangle is right-angled
  (h3 : a = 24) -- first condition on first triangle
  (h4 : b = 32) -- second condition on first triangle
  (h5 : f = 80) -- condition on second triangle's hypotenuse
  (h6 : a / d = b / e) -- triangles are similar
  (h7 : b / e = c / f) -- triangles are similar
  : d = 48 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l2805_280587


namespace NUMINAMATH_CALUDE_max_value_product_l2805_280510

theorem max_value_product (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a^2 - b*c) * (b^2 - c*a) * (c^2 - a*b) ≤ 1/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l2805_280510


namespace NUMINAMATH_CALUDE_puppies_calculation_l2805_280595

/-- The number of puppies Alyssa initially had -/
def initial_puppies : ℕ := 12

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has now -/
def remaining_puppies : ℕ := initial_puppies - puppies_given_away

theorem puppies_calculation : remaining_puppies = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_calculation_l2805_280595


namespace NUMINAMATH_CALUDE_fourth_number_proof_l2805_280564

theorem fourth_number_proof (sum : ℝ) (a b c : ℝ) (h1 : sum = 221.2357) 
  (h2 : a = 217) (h3 : b = 2.017) (h4 : c = 0.217) : 
  sum - (a + b + c) = 2.0017 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l2805_280564


namespace NUMINAMATH_CALUDE_agency_a_cheaper_l2805_280541

/-- Represents a travel agency with a pricing function -/
structure TravelAgency where
  price : ℕ → ℝ

/-- The initial price per person -/
def initialPrice : ℝ := 200

/-- Travel Agency A with 25% discount for all -/
def agencyA : TravelAgency :=
  { price := λ x => initialPrice * 0.75 * x }

/-- Travel Agency B with one free and 20% discount for the rest -/
def agencyB : TravelAgency :=
  { price := λ x => initialPrice * 0.8 * (x - 1) }

/-- Theorem stating when Agency A is cheaper than Agency B -/
theorem agency_a_cheaper (x : ℕ) :
  x > 16 → agencyA.price x < agencyB.price x :=
sorry

end NUMINAMATH_CALUDE_agency_a_cheaper_l2805_280541


namespace NUMINAMATH_CALUDE_f_zero_range_l2805_280572

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - (Real.log x) / x + a

theorem f_zero_range (a : ℝ) :
  (∃ x > 0, f x a = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_f_zero_range_l2805_280572


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2805_280517

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 3 + a 5 = 21 →
  a 2 * a 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2805_280517


namespace NUMINAMATH_CALUDE_second_tract_length_l2805_280511

/-- Given two rectangular tracts of land with specified dimensions and combined area,
    prove that the length of the second tract is 250 meters. -/
theorem second_tract_length
  (tract1_length : ℝ)
  (tract1_width : ℝ)
  (tract2_width : ℝ)
  (combined_area : ℝ)
  (h1 : tract1_length = 300)
  (h2 : tract1_width = 500)
  (h3 : tract2_width = 630)
  (h4 : combined_area = 307500)
  : ∃ tract2_length : ℝ,
    tract2_length = 250 ∧
    tract1_length * tract1_width + tract2_length * tract2_width = combined_area :=
by
  sorry

end NUMINAMATH_CALUDE_second_tract_length_l2805_280511


namespace NUMINAMATH_CALUDE_route_length_l2805_280567

/-- Proves that given a round trip with total time of 1 hour, average speed of 8 miles/hour,
    and return speed of 20 miles/hour along the same path, the length of the one-way route is 4 miles. -/
theorem route_length (total_time : ℝ) (avg_speed : ℝ) (return_speed : ℝ) (route_length : ℝ) : 
  total_time = 1 →
  avg_speed = 8 →
  return_speed = 20 →
  route_length * 2 = avg_speed * total_time →
  route_length / return_speed + route_length / (route_length * 2 / total_time - return_speed) = total_time →
  route_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_route_length_l2805_280567


namespace NUMINAMATH_CALUDE_german_students_count_l2805_280546

theorem german_students_count (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) :
  total = 79 →
  french = 41 →
  both = 9 →
  neither = 25 →
  ∃ german : ℕ, german = 22 ∧ 
    total = french + german - both + neither :=
by sorry

end NUMINAMATH_CALUDE_german_students_count_l2805_280546


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_quadratic_equation_l2805_280557

/-- A triangle with sides a, b, and c is isosceles if the quadratic equation
    (c-b)x^2 + 2(b-a)x + (a-b) = 0 has two equal real roots. -/
theorem triangle_isosceles_from_quadratic_equation (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_eq : ∃ x : ℝ, (c - b) * x^2 + 2*(b - a)*x + (a - b) = 0 ∧ 
    ∀ y : ℝ, (c - b) * y^2 + 2*(b - a)*y + (a - b) = 0 → y = x) :
  (a = b ∧ c ≠ b) ∨ (a = c ∧ b ≠ c) ∨ (b = c ∧ a ≠ b) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_from_quadratic_equation_l2805_280557


namespace NUMINAMATH_CALUDE_angle_pcq_is_45_deg_l2805_280592

/-- Given a unit square ABCD with points P on AB and Q on AD forming
    triangle APQ with perimeter 2, angle PCQ is 45 degrees. -/
theorem angle_pcq_is_45_deg (A B C D P Q : ℝ × ℝ) : 
  -- Square ABCD is a unit square
  A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1) →
  -- P is on AB
  ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ P = (a, 0) →
  -- Q is on AD
  ∃ b : ℝ, 0 ≤ b ∧ b ≤ 1 ∧ Q = (0, b) →
  -- Perimeter of APQ is 2
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) +
  Real.sqrt ((A.1 - Q.1)^2 + (A.2 - Q.2)^2) = 2 →
  -- Angle PCQ is 45 degrees
  (Real.arctan ((C.2 - P.2) / (C.1 - P.1)) -
   Real.arctan ((C.1 - Q.1) / (C.2 - Q.2))) * (180 / Real.pi) = 45 := by
  sorry


end NUMINAMATH_CALUDE_angle_pcq_is_45_deg_l2805_280592


namespace NUMINAMATH_CALUDE_band_earnings_theorem_l2805_280574

/-- Represents a band with its earnings and gig information -/
structure Band where
  members : ℕ
  totalEarnings : ℕ
  gigs : ℕ

/-- Calculates the earnings per member per gig for a given band -/
def earningsPerMemberPerGig (b : Band) : ℚ :=
  (b.totalEarnings : ℚ) / (b.members : ℚ) / (b.gigs : ℚ)

/-- Theorem: For a band with 4 members that earned $400 after 5 gigs, 
    each member earns $20 per gig -/
theorem band_earnings_theorem (b : Band) 
    (h1 : b.members = 4) 
    (h2 : b.totalEarnings = 400) 
    (h3 : b.gigs = 5) : 
  earningsPerMemberPerGig b = 20 := by
  sorry


end NUMINAMATH_CALUDE_band_earnings_theorem_l2805_280574


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2805_280566

theorem solve_exponential_equation :
  ∃ x : ℤ, (2^x : ℝ) - (2^(x-2) : ℝ) = 3 * (2^10 : ℝ) ∧ x = 12 :=
by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2805_280566


namespace NUMINAMATH_CALUDE_complex_point_in_first_quadrant_l2805_280565

theorem complex_point_in_first_quadrant : 
  let z : ℂ := (1 - 2*I)^3 / I
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_point_in_first_quadrant_l2805_280565


namespace NUMINAMATH_CALUDE_m_range_theorem_l2805_280579

open Set

noncomputable def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

noncomputable def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

def p_set : Set ℝ := {x | P x}
def q_set (m : ℝ) : Set ℝ := {x | Q x m}

theorem m_range_theorem :
  ∀ m : ℝ, (0 < m ∧ m ≤ 3) ↔ 
    (m > 0 ∧ q_set m ⊂ p_set ∧ q_set m ≠ p_set) :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2805_280579


namespace NUMINAMATH_CALUDE_ceiling_sqrt_count_l2805_280507

theorem ceiling_sqrt_count (x : ℤ) : (∃ (count : ℕ), count = 39 ∧ 
  (∀ y : ℤ, ⌈Real.sqrt (y : ℝ)⌉ = 20 ↔ 362 ≤ y ∧ y ≤ 400) ∧
  count = (Finset.range 39).card) :=
sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_count_l2805_280507


namespace NUMINAMATH_CALUDE_orange_cost_l2805_280591

/-- Given the cost of 3 dozen oranges, calculate the cost of 5 dozen oranges at the same rate -/
theorem orange_cost (cost_3_dozen : ℝ) (h : cost_3_dozen = 28.80) :
  let cost_per_dozen := cost_3_dozen / 3
  let cost_5_dozen := 5 * cost_per_dozen
  cost_5_dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_l2805_280591


namespace NUMINAMATH_CALUDE_linear_system_solution_quadratic_system_result_l2805_280526

-- Define the system of linear equations
def linear_system (x y : ℝ) : Prop :=
  3 * x - 2 * y = 5 ∧ 9 * x - 4 * y = 19

-- Define the system of quadratic equations
def quadratic_system (x y : ℝ) : Prop :=
  3 * x^2 - 2 * x * y + 12 * y^2 = 47 ∧ 2 * x^2 + x * y + 8 * y^2 = 36

-- Theorem for the linear system
theorem linear_system_solution :
  ∃ x y : ℝ, linear_system x y ∧ x = 3 ∧ y = 2 :=
sorry

-- Theorem for the quadratic system
theorem quadratic_system_result :
  ∀ x y : ℝ, quadratic_system x y → x^2 + 4 * y^2 = 17 :=
sorry

end NUMINAMATH_CALUDE_linear_system_solution_quadratic_system_result_l2805_280526


namespace NUMINAMATH_CALUDE_green_apples_count_l2805_280542

/-- Given a basket with red and green apples, prove the number of green apples. -/
theorem green_apples_count (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 9 → red = 7 → green = total - red → green = 2 := by sorry

end NUMINAMATH_CALUDE_green_apples_count_l2805_280542


namespace NUMINAMATH_CALUDE_constant_sum_property_l2805_280583

/-- Represents a triangle with numbers at its vertices -/
structure NumberedTriangle where
  a : ℝ  -- Number at vertex A
  b : ℝ  -- Number at vertex B
  c : ℝ  -- Number at vertex C

/-- The sum of a vertex number and the opposite side sum is constant -/
theorem constant_sum_property (t : NumberedTriangle) :
  t.a + (t.b + t.c) = t.b + (t.c + t.a) ∧
  t.b + (t.c + t.a) = t.c + (t.a + t.b) ∧
  t.c + (t.a + t.b) = t.a + t.b + t.c :=
sorry

end NUMINAMATH_CALUDE_constant_sum_property_l2805_280583


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2805_280568

theorem complex_fraction_simplification :
  let z : ℂ := (3 - 2*I) / (1 + 5*I)
  z = -7/26 - 17/26*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2805_280568


namespace NUMINAMATH_CALUDE_mountain_climb_fraction_l2805_280570

theorem mountain_climb_fraction (mountain_height : ℕ) (num_trips : ℕ) (total_distance : ℕ)
  (h1 : mountain_height = 40000)
  (h2 : num_trips = 10)
  (h3 : total_distance = 600000) :
  (total_distance / (2 * num_trips)) / mountain_height = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_mountain_climb_fraction_l2805_280570


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2805_280529

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h5 : a 5 = 15) :
  a 3 + a 4 + a 7 + a 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2805_280529


namespace NUMINAMATH_CALUDE_license_plate_count_l2805_280534

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 4

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- The number of license plate combinations with the given conditions -/
def license_plate_combinations : ℕ :=
  alphabet_size *  -- Choose the repeated letter
  (alphabet_size - 1).choose 2 *  -- Choose the other two distinct letters
  letter_positions.choose 2 *  -- Arrange the repeated letters
  2 *  -- Arrange the remaining two letters
  digit_count *  -- Choose the digit to repeat
  digit_positions.choose 2 *  -- Choose positions for the repeated digit
  (digit_count - 1)  -- Choose the second, different digit

/-- Theorem stating the number of possible license plate combinations -/
theorem license_plate_count : license_plate_combinations = 4212000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2805_280534


namespace NUMINAMATH_CALUDE_evaluate_expression_l2805_280551

theorem evaluate_expression (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  4 * x^y - 5 * y^x = -4 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2805_280551


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l2805_280509

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem max_sum_arithmetic_sequence
  (a₁ : ℚ)
  (h1 : a₁ = 13)
  (h2 : sum_arithmetic_sequence a₁ d 3 = sum_arithmetic_sequence a₁ d 11) :
  ∃ (n : ℕ), ∀ (m : ℕ), sum_arithmetic_sequence a₁ d n ≥ sum_arithmetic_sequence a₁ d m ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l2805_280509


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l2805_280508

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 24 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l2805_280508


namespace NUMINAMATH_CALUDE_expand_expression_l2805_280513

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 - 2 * y) = 36 * x + 48 - 24 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2805_280513


namespace NUMINAMATH_CALUDE_factor_expression_l2805_280582

theorem factor_expression (x : ℝ) : 
  (4 * x^4 + 128 * x^3 - 9) - (-6 * x^4 + 2 * x^3 - 9) = 2 * x^3 * (5 * x + 63) := by
sorry

end NUMINAMATH_CALUDE_factor_expression_l2805_280582


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l2805_280543

theorem thirty_percent_less_than_eighty (x : ℝ) : x + x/2 = 80 * (1 - 0.3) → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l2805_280543


namespace NUMINAMATH_CALUDE_triangle_side_length_bound_l2805_280589

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where the area S = √3/4 * (a² + c² - b²) and b = √3,
    prove that (√3 - 1)a + 2c is bounded by (3 - √3, 2√6]. -/
theorem triangle_side_length_bound (a c : ℝ) (h_positive : a > 0 ∧ c > 0) :
  let b := Real.sqrt 3
  let S := Real.sqrt 3 / 4 * (a^2 + c^2 - b^2)
  3 - Real.sqrt 3 < (Real.sqrt 3 - 1) * a + 2 * c ∧
  (Real.sqrt 3 - 1) * a + 2 * c ≤ 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_bound_l2805_280589


namespace NUMINAMATH_CALUDE_empty_lorry_weight_l2805_280503

/-- The weight of an empty lorry given the following conditions:
  * The lorry is loaded with 20 bags of apples.
  * Each bag of apples weighs 60 pounds.
  * The weight of the loaded lorry is 1700 pounds.
-/
theorem empty_lorry_weight : ℕ := by
  sorry

#check empty_lorry_weight

end NUMINAMATH_CALUDE_empty_lorry_weight_l2805_280503


namespace NUMINAMATH_CALUDE_max_value_constraint_l2805_280554

theorem max_value_constraint (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) :
  (8*a + 5*b + 15*c) ≤ Real.sqrt 115 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2805_280554


namespace NUMINAMATH_CALUDE_sum_of_squares_mod_13_l2805_280581

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_mod_13 : sum_of_squares 15 % 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_mod_13_l2805_280581


namespace NUMINAMATH_CALUDE_unique_configuration_l2805_280548

/-- A configuration of n points in the plane with associated real numbers -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  r : Fin n → ℝ

/-- The area of a triangle given by three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Predicate for non-collinearity of three points -/
def non_collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- The main theorem: only n = 4 satisfies the conditions -/
theorem unique_configuration :
  ∀ n : ℕ, n > 3 →
  (∃ (config : PointConfiguration n),
    (∀ i j k : Fin n, i < j → j < k →
      non_collinear (config.points i) (config.points j) (config.points k)) ∧
    (∀ i j k : Fin n, i < j → j < k →
      triangle_area (config.points i) (config.points j) (config.points k) =
        config.r i + config.r j + config.r k)) →
  n = 4 := by sorry

end NUMINAMATH_CALUDE_unique_configuration_l2805_280548


namespace NUMINAMATH_CALUDE_octal_567_equals_decimal_375_l2805_280516

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let hundreds := octal / 100
  let tens := (octal / 10) % 10
  let ones := octal % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The octal number 567 is equal to the decimal number 375 --/
theorem octal_567_equals_decimal_375 : octal_to_decimal 567 = 375 := by
  sorry

end NUMINAMATH_CALUDE_octal_567_equals_decimal_375_l2805_280516


namespace NUMINAMATH_CALUDE_card_sum_problem_l2805_280584

theorem card_sum_problem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_problem_l2805_280584


namespace NUMINAMATH_CALUDE_dorothy_doughnuts_l2805_280504

/-- Represents the problem of calculating the number of doughnuts Dorothy made. -/
theorem dorothy_doughnuts (ingredient_cost : ℕ) (selling_price : ℕ) (profit : ℕ) 
  (h1 : ingredient_cost = 53)
  (h2 : selling_price = 3)
  (h3 : profit = 22) :
  ∃ (num_doughnuts : ℕ), 
    selling_price * num_doughnuts = ingredient_cost + profit ∧ 
    num_doughnuts = 25 := by
  sorry


end NUMINAMATH_CALUDE_dorothy_doughnuts_l2805_280504


namespace NUMINAMATH_CALUDE_invalid_votes_count_l2805_280535

/-- Proves that the number of invalid votes is 100 in an election with given conditions -/
theorem invalid_votes_count (total_votes : ℕ) (valid_votes : ℕ) (loser_percentage : ℚ) (vote_difference : ℕ) : 
  total_votes = 12600 →
  loser_percentage = 30/100 →
  vote_difference = 5000 →
  valid_votes = vote_difference / (1/2 - loser_percentage) →
  total_votes - valid_votes = 100 := by
  sorry

#check invalid_votes_count

end NUMINAMATH_CALUDE_invalid_votes_count_l2805_280535


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2805_280558

/-- The quadratic equation x^2 - (2m+1)x + m^2 + m = 0 -/
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (2*m+1)*x + m^2 + m

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (2*m+1)^2 - 4*(m^2 + m)

/-- The sum of roots of the quadratic equation -/
def sum_of_roots (m : ℝ) : ℝ := 2*m + 1

/-- The product of roots of the quadratic equation -/
def product_of_roots (m : ℝ) : ℝ := m^2 + m

theorem quadratic_equation_properties (m : ℝ) :
  (discriminant m = 1) ∧
  (∃ a b : ℝ, quadratic_equation m a = 0 ∧ quadratic_equation m b = 0 ∧
    (2*a + b) * (a + 2*b) = 20 → (m = -2 ∨ m = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2805_280558


namespace NUMINAMATH_CALUDE_parabola_vertex_l2805_280573

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by the equation y^2 - 8y + 4x = 12 -/
def Parabola := {p : Point | p.y^2 - 8*p.y + 4*p.x = 12}

/-- The vertex of a parabola -/
def vertex : Point := ⟨7, 4⟩

/-- Theorem stating that the vertex of the parabola is (7, 4) -/
theorem parabola_vertex : vertex ∈ Parabola ∧ ∀ p ∈ Parabola, p.x ≥ vertex.x := by
  sorry

#check parabola_vertex

end NUMINAMATH_CALUDE_parabola_vertex_l2805_280573


namespace NUMINAMATH_CALUDE_impossibleToMakeAllEqual_l2805_280559

/-- Represents the possible values in a cell of the table -/
inductive CellValue
  | Zero
  | One
  deriving Repr

/-- Represents a 4x4 table of cell values -/
def Table := Fin 4 → Fin 4 → CellValue

/-- Represents the initial state of the table -/
def initialTable : Table := fun i j =>
  if i = 0 ∧ j = 1 then CellValue.One else CellValue.Zero

/-- Represents the allowed operations on the table -/
inductive Operation
  | AddToRow (row : Fin 4)
  | AddToColumn (col : Fin 4)
  | AddToDiagonal (startRow startCol : Fin 4)

/-- Applies an operation to a table -/
def applyOperation (t : Table) (op : Operation) : Table :=
  sorry

/-- Checks if all values in the table are equal -/
def allEqual (t : Table) : Prop :=
  ∀ i j k l, t i j = t k l

/-- The main theorem stating that it's impossible to make all numbers equal -/
theorem impossibleToMakeAllEqual :
  ¬∃ (ops : List Operation), allEqual (ops.foldl applyOperation initialTable) :=
sorry

end NUMINAMATH_CALUDE_impossibleToMakeAllEqual_l2805_280559


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2805_280538

theorem quadratic_real_roots (k : ℝ) : 
  k > 0 → ∃ x : ℝ, x^2 - x - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2805_280538


namespace NUMINAMATH_CALUDE_sqrt_three_minus_pi_squared_l2805_280528

theorem sqrt_three_minus_pi_squared : Real.sqrt ((3 - Real.pi) ^ 2) = Real.pi - 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_pi_squared_l2805_280528


namespace NUMINAMATH_CALUDE_bathroom_extension_l2805_280512

/-- Represents the dimensions and area of a rectangular bathroom -/
structure Bathroom where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Calculates the new area of a bathroom after extension -/
def extended_area (b : Bathroom) (extension : ℝ) : ℝ :=
  (b.width + 2 * extension) * (b.length + 2 * extension)

/-- Theorem: Given a bathroom with area 96 sq ft and width 8 ft, 
    extending it by 2 ft on each side results in an area of 140 sq ft -/
theorem bathroom_extension :
  ∀ (b : Bathroom),
    b.area = 96 ∧ b.width = 8 →
    extended_area b 2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_extension_l2805_280512


namespace NUMINAMATH_CALUDE_min_value_f_min_value_f_attained_max_value_g_max_value_g_attained_l2805_280523

-- Part Ⅰ
theorem min_value_f (x : ℝ) (hx : x > 0) : 12/x + 3*x ≥ 12 := by
  sorry

theorem min_value_f_attained : ∃ x : ℝ, x > 0 ∧ 12/x + 3*x = 12 := by
  sorry

-- Part Ⅱ
theorem max_value_g (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) : x*(1 - 3*x) ≤ 1/12 := by
  sorry

theorem max_value_g_attained : ∃ x : ℝ, x > 0 ∧ x < 1/3 ∧ x*(1 - 3*x) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_min_value_f_attained_max_value_g_max_value_g_attained_l2805_280523


namespace NUMINAMATH_CALUDE_ship_capacity_and_tax_calculation_l2805_280594

/-- Represents the types of cargo --/
inductive CargoType
  | Steel
  | Timber
  | Electronics
  | Textiles

/-- Represents a cargo load with its type and weight --/
structure CargoLoad :=
  (type : CargoType)
  (weight : Nat)

/-- Calculates the total weight of a list of cargo loads --/
def totalWeight (loads : List CargoLoad) : Nat :=
  loads.foldl (fun acc load => acc + load.weight) 0

/-- Calculates the import tax for a single cargo load --/
def importTax (load : CargoLoad) : Nat :=
  match load.type with
  | CargoType.Steel => load.weight * 50
  | CargoType.Timber => load.weight * 75
  | CargoType.Electronics => load.weight * 100
  | CargoType.Textiles => load.weight * 40

/-- Calculates the total import tax for a list of cargo loads --/
def totalImportTax (loads : List CargoLoad) : Nat :=
  loads.foldl (fun acc load => acc + importTax load) 0

/-- The main theorem to prove --/
theorem ship_capacity_and_tax_calculation 
  (maxCapacity : Nat)
  (initialCargo : List CargoLoad)
  (additionalCargo : List CargoLoad) :
  maxCapacity = 20000 →
  initialCargo = [
    ⟨CargoType.Steel, 3428⟩,
    ⟨CargoType.Timber, 1244⟩,
    ⟨CargoType.Electronics, 1301⟩
  ] →
  additionalCargo = [
    ⟨CargoType.Steel, 3057⟩,
    ⟨CargoType.Textiles, 2364⟩,
    ⟨CargoType.Timber, 1517⟩,
    ⟨CargoType.Electronics, 1785⟩
  ] →
  totalWeight (initialCargo ++ additionalCargo) ≤ maxCapacity ∧
  totalImportTax (initialCargo ++ additionalCargo) = 934485 :=
by sorry


end NUMINAMATH_CALUDE_ship_capacity_and_tax_calculation_l2805_280594


namespace NUMINAMATH_CALUDE_kristine_has_more_cd_difference_l2805_280562

/-- The number of CDs Dawn has -/
def dawn_cds : ℕ := 10

/-- The total number of CDs Kristine and Dawn have together -/
def total_cds : ℕ := 27

/-- Kristine's CDs -/
def kristine_cds : ℕ := total_cds - dawn_cds

/-- The statement that Kristine has more CDs than Dawn -/
theorem kristine_has_more : kristine_cds > dawn_cds := by sorry

/-- The main theorem: Kristine has 7 more CDs than Dawn -/
theorem cd_difference : kristine_cds - dawn_cds = 7 := by sorry

end NUMINAMATH_CALUDE_kristine_has_more_cd_difference_l2805_280562


namespace NUMINAMATH_CALUDE_solution_set_l2805_280588

theorem solution_set : ∀ x y : ℝ,
  (3/20 + |x - 15/40| < 7/20 ∧ y = 2*x + 1) ↔ 
  (7/20 < x ∧ x < 2/5 ∧ 17/10 ≤ y ∧ y ≤ 11/5) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l2805_280588


namespace NUMINAMATH_CALUDE_toy_box_problem_l2805_280550

/-- The time taken to put all toys in the box -/
def time_to_fill_box (total_toys : ℕ) (toys_in_per_cycle : ℕ) (toys_out_per_cycle : ℕ) (cycle_time : ℕ) : ℕ := 
  sorry

/-- The problem statement -/
theorem toy_box_problem :
  let total_toys : ℕ := 50
  let toys_in_per_cycle : ℕ := 5
  let toys_out_per_cycle : ℕ := 3
  let cycle_time_seconds : ℕ := 45
  let minutes_per_hour : ℕ := 60
  time_to_fill_box total_toys toys_in_per_cycle toys_out_per_cycle cycle_time_seconds = 18 * minutes_per_hour :=
by sorry

end NUMINAMATH_CALUDE_toy_box_problem_l2805_280550


namespace NUMINAMATH_CALUDE_tommys_coin_collection_l2805_280561

theorem tommys_coin_collection (nickels dimes quarters pennies : ℕ) : 
  nickels = 100 →
  nickels = 2 * dimes →
  quarters = 4 →
  pennies = 10 * quarters →
  dimes - pennies = 10 := by
  sorry

end NUMINAMATH_CALUDE_tommys_coin_collection_l2805_280561


namespace NUMINAMATH_CALUDE_range_of_x_in_negative_sqrt_l2805_280576

theorem range_of_x_in_negative_sqrt (x : ℝ) :
  (3 * x + 5 ≥ 0) ↔ (x ≥ -5/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_in_negative_sqrt_l2805_280576


namespace NUMINAMATH_CALUDE_correct_operation_l2805_280519

theorem correct_operation (x y : ℝ) : 4 * x^3 * y^2 - (-2)^2 * x^3 * y^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2805_280519


namespace NUMINAMATH_CALUDE_smallest_k_correct_l2805_280553

/-- The smallest integer k for which kx^2 - 4x - 4 = 0 has two distinct real roots -/
def smallest_k : ℤ := 1

/-- Quadratic equation ax^2 + bx + c = 0 has two distinct real roots iff b^2 - 4ac > 0 -/
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c > 0

theorem smallest_k_correct :
  (∀ k : ℤ, k < smallest_k → ¬(has_two_distinct_real_roots k (-4) (-4))) ∧
  has_two_distinct_real_roots smallest_k (-4) (-4) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_correct_l2805_280553


namespace NUMINAMATH_CALUDE_expression_factorization_l2805_280599

theorem expression_factorization (x : ℝ) : 
  4*x*(x-5) + 5*(x-5) + 6*x*(x-2) = (4*x+5)*(x-5) + 6*x*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2805_280599


namespace NUMINAMATH_CALUDE_minuend_is_zero_l2805_280563

theorem minuend_is_zero (x y : ℝ) (h : x - y = -y) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_minuend_is_zero_l2805_280563


namespace NUMINAMATH_CALUDE_girls_percentage_less_than_boys_l2805_280505

theorem girls_percentage_less_than_boys (boys girls : ℝ) 
  (h : boys = girls * 1.25) : 
  (boys - girls) / boys = 0.2 := by
sorry

end NUMINAMATH_CALUDE_girls_percentage_less_than_boys_l2805_280505


namespace NUMINAMATH_CALUDE_regression_line_equation_l2805_280586

/-- Given a regression line with slope 1.23 and a point (4, 5) on the line,
    prove that the equation of the line is y = 1.23x + 0.08 -/
theorem regression_line_equation (x y : ℝ) :
  let slope : ℝ := 1.23
  let point : ℝ × ℝ := (4, 5)
  (y - point.2 = slope * (x - point.1)) → (y = slope * x + 0.08) :=
by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l2805_280586


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2805_280547

theorem no_integer_solutions : ¬∃ (a b : ℤ), a^3 + 3*a^2 + 2*a = 125*b^3 + 75*b^2 + 15*b + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2805_280547


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2805_280502

theorem lcm_factor_proof (A B : ℕ+) (Y : ℕ+) : 
  Nat.gcd A B = 63 →
  Nat.lcm A B = 63 * 11 * Y →
  A = 1071 →
  Y = 17 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l2805_280502


namespace NUMINAMATH_CALUDE_combined_rate_is_90_l2805_280525

/-- Represents the fish fillet production scenario -/
structure FishFilletProduction where
  totalRequired : ℕ
  deadline : ℕ
  firstTeamProduction : ℕ
  secondTeamProduction : ℕ
  thirdTeamRate : ℕ

/-- Calculates the combined production rate of the third and fourth teams -/
def combinedRate (p : FishFilletProduction) : ℕ :=
  let remainingPieces := p.totalRequired - (p.firstTeamProduction + p.secondTeamProduction)
  let thirdTeamProduction := p.thirdTeamRate * p.deadline
  let fourthTeamProduction := remainingPieces - thirdTeamProduction
  p.thirdTeamRate + (fourthTeamProduction / p.deadline)

/-- Theorem stating that the combined production rate is 90 pieces per hour -/
theorem combined_rate_is_90 (p : FishFilletProduction)
    (h1 : p.totalRequired = 500)
    (h2 : p.deadline = 2)
    (h3 : p.firstTeamProduction = 189)
    (h4 : p.secondTeamProduction = 131)
    (h5 : p.thirdTeamRate = 45) :
    combinedRate p = 90 := by
  sorry

#eval combinedRate {
  totalRequired := 500,
  deadline := 2,
  firstTeamProduction := 189,
  secondTeamProduction := 131,
  thirdTeamRate := 45
}

end NUMINAMATH_CALUDE_combined_rate_is_90_l2805_280525


namespace NUMINAMATH_CALUDE_a_spending_percentage_l2805_280531

def total_salary : ℝ := 4000
def a_salary : ℝ := 3000
def b_spending_percentage : ℝ := 0.85

theorem a_spending_percentage :
  ∃ (a_spending : ℝ),
    a_spending = 0.95 ∧
    a_salary * (1 - a_spending) = (total_salary - a_salary) * (1 - b_spending_percentage) :=
by sorry

end NUMINAMATH_CALUDE_a_spending_percentage_l2805_280531


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2805_280500

theorem trigonometric_identities :
  (Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4) ∧
  (Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4) ∧
  (Real.sin (18 * π / 180) = (-1 + Real.sqrt 5) / 4) ∧
  (Real.cos (18 * π / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2805_280500


namespace NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l2805_280549

theorem sunglasses_and_caps_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_cap_and_sunglasses : ℚ) :
  total_sunglasses = 60 →
  total_caps = 40 →
  prob_cap_and_sunglasses = 2/5 →
  (prob_cap_and_sunglasses * total_caps) / total_sunglasses = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l2805_280549


namespace NUMINAMATH_CALUDE_no_valid_cube_labeling_l2805_280593

/-- A cube vertex labeling is a function from vertex indices to odd numbers -/
def CubeLabeling := Fin 8 → Nat

/-- Predicate to check if two numbers are adjacent on a cube -/
def adjacent (i j : Fin 8) : Prop :=
  (i.val + j.val) % 2 = 1 ∧ i ≠ j

/-- Predicate to check if a labeling satisfies the problem conditions -/
def validLabeling (f : CubeLabeling) : Prop :=
  (∀ i, 1 ≤ f i ∧ f i ≤ 600 ∧ f i % 2 = 1) ∧
  (∀ i j, adjacent i j → ∃ d > 1, d ∣ f i ∧ d ∣ f j) ∧
  (∀ i j, ¬adjacent i j → ∀ d > 1, ¬(d ∣ f i ∧ d ∣ f j)) ∧
  (∀ i j, i ≠ j → f i ≠ f j)

theorem no_valid_cube_labeling : ¬∃ f : CubeLabeling, validLabeling f :=
sorry

end NUMINAMATH_CALUDE_no_valid_cube_labeling_l2805_280593


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2805_280569

/-- Given Sam's current age s and Tim's current age t, where:
    1. s - 4 = 4(t - 4)
    2. s - 10 = 5(t - 10)
    Prove that the number of years x until their age ratio is 3:1 is 8. -/
theorem age_ratio_problem (s t : ℕ) 
  (h1 : s - 4 = 4 * (t - 4)) 
  (h2 : s - 10 = 5 * (t - 10)) : 
  ∃ x : ℕ, x = 8 ∧ (s + x) / (t + x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2805_280569


namespace NUMINAMATH_CALUDE_max_gcd_sum_1023_l2805_280545

theorem max_gcd_sum_1023 :
  ∃ (c d : ℕ+), c + d = 1023 ∧
  ∀ (x y : ℕ+), x + y = 1023 → Nat.gcd x y ≤ Nat.gcd c d ∧
  Nat.gcd c d = 341 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1023_l2805_280545


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2805_280571

theorem geometric_sequence_problem (a b c d e : ℕ) : 
  (2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100) →
  Nat.gcd a e = 1 →
  (∃ (r : ℚ), b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4) →
  c = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2805_280571


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2805_280536

theorem solve_exponential_equation (y : ℝ) :
  (5 : ℝ) ^ (3 * y) = Real.sqrt 125 → y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2805_280536


namespace NUMINAMATH_CALUDE_power_mod_eleven_l2805_280518

theorem power_mod_eleven : 5^303 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l2805_280518
