import Mathlib

namespace jakes_test_scores_l3497_349725

theorem jakes_test_scores (average : ℝ) (first_test : ℝ) (second_test : ℝ) (third_test : ℝ) :
  average = 75 →
  first_test = 80 →
  second_test = 90 →
  (first_test + second_test + third_test + third_test) / 4 = average →
  third_test = 65 := by
sorry

end jakes_test_scores_l3497_349725


namespace polynomial_roots_product_l3497_349789

theorem polynomial_roots_product (p q r s : ℝ) : 
  let Q : ℝ → ℝ := λ x => x^4 + p*x^3 + q*x^2 + r*x + s
  (Q (Real.cos (π/8)) = 0) ∧ 
  (Q (Real.cos (3*π/8)) = 0) ∧ 
  (Q (Real.cos (5*π/8)) = 0) ∧ 
  (Q (Real.cos (7*π/8)) = 0) →
  p * q * r * s = 0 := by
sorry

end polynomial_roots_product_l3497_349789


namespace symmetrical_parabola_directrix_l3497_349728

/-- Given a parabola y = 2x², prove that the equation of the directrix of the parabola
    symmetrical to it with respect to the line y = x is x = -1/8 -/
theorem symmetrical_parabola_directrix (x y : ℝ) :
  (y = 2 * x^2) →  -- Original parabola
  ∃ (x₀ : ℝ), 
    (∀ (x' y' : ℝ), y'^2 = (1/2) * x' ↔ (y = x ∧ x' = y ∧ y' = x)) →  -- Symmetry condition
    (x₀ = -1/8 ∧ ∀ (x' y' : ℝ), y'^2 = (1/2) * x' → |x' - x₀| = (1/4)) :=  -- Directrix equation
sorry

end symmetrical_parabola_directrix_l3497_349728


namespace exists_a_for_even_f_l3497_349778

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem exists_a_for_even_f : ∃ a : ℝ, ∀ x : ℝ, f a x = f a (-x) := by
  sorry

end exists_a_for_even_f_l3497_349778


namespace unique_solution_for_equation_l3497_349747

theorem unique_solution_for_equation (x y : ℝ) :
  (x - 14)^2 + (y - 15)^2 + (x - y)^2 = 1/3 ↔ x = 14 + 1/3 ∧ y = 14 + 2/3 := by
  sorry

end unique_solution_for_equation_l3497_349747


namespace triangle_theorem_l3497_349730

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * Real.cos (2 * t.A) + 4 * Real.cos (t.B + t.C) + 3 = 0 ∧
  t.a = Real.sqrt 3 ∧
  t.b + t.c = 3

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.A = π / 3 ∧ ((t.b = 2 ∧ t.c = 1) ∨ (t.b = 1 ∧ t.c = 2)) := by
  sorry

end triangle_theorem_l3497_349730


namespace fractional_equation_solution_l3497_349791

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end fractional_equation_solution_l3497_349791


namespace ellipse_hyperbola_eccentricity_l3497_349772

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A conic section (ellipse or hyperbola) -/
structure Conic where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  isEllipse : Bool

/-- The eccentricity of a conic section -/
def eccentricity (c : Conic) : ℝ :=
  sorry

/-- The foci of a conic section -/
def foci (c : Conic) : (Point × Point) :=
  sorry

/-- The angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- The intersection points of two conic sections -/
def intersection (c1 c2 : Conic) : Set Point :=
  sorry

theorem ellipse_hyperbola_eccentricity 
  (C₁ : Conic) (C₂ : Conic) (F₁ F₂ P : Point) :
  C₁.isEllipse = true →
  C₂.isEllipse = false →
  foci C₁ = (F₁, F₂) →
  foci C₂ = (F₁, F₂) →
  P ∈ intersection C₁ C₂ →
  P.x > 0 ∧ P.y > 0 →
  eccentricity C₁ * eccentricity C₂ = 1 →
  angle F₁ P F₂ = π / 3 →
  eccentricity C₁ = Real.sqrt 3 / 3 :=
sorry

end ellipse_hyperbola_eccentricity_l3497_349772


namespace ellipse_a_plus_k_equals_nine_l3497_349771

/-- An ellipse with given properties -/
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  point : ℝ × ℝ
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The ellipse satisfies the given conditions -/
def satisfies_conditions (e : Ellipse) : Prop :=
  e.foci1 = (1, 5) ∧
  e.foci2 = (1, 1) ∧
  e.point = (7, 3) ∧
  e.a > 0 ∧
  e.b > 0 ∧
  (e.point.1 - e.h)^2 / e.a^2 + (e.point.2 - e.k)^2 / e.b^2 = 1

/-- The theorem stating that a + k equals 9 for the given ellipse -/
theorem ellipse_a_plus_k_equals_nine (e : Ellipse) 
  (h : satisfies_conditions e) : e.a + e.k = 9 := by
  sorry

end ellipse_a_plus_k_equals_nine_l3497_349771


namespace apple_price_proof_l3497_349745

def grocery_problem (total_spent milk_price cereal_price banana_price cookie_multiplier
                     milk_qty cereal_qty banana_qty cookie_qty apple_qty : ℚ) : Prop :=
  let cereal_total := cereal_price * cereal_qty
  let banana_total := banana_price * banana_qty
  let cookie_price := milk_price * cookie_multiplier
  let cookie_total := cookie_price * cookie_qty
  let known_items_total := milk_price * milk_qty + cereal_total + banana_total + cookie_total
  let apple_total := total_spent - known_items_total
  let apple_price := apple_total / apple_qty
  apple_price = 0.5

theorem apple_price_proof :
  grocery_problem 25 3 3.5 0.25 2 1 2 4 2 4 := by
  sorry

end apple_price_proof_l3497_349745


namespace alyssa_kittens_l3497_349729

/-- The number of kittens Alyssa initially had -/
def initial_kittens : ℕ := 8

/-- The number of kittens Alyssa gave away -/
def kittens_given_away : ℕ := 4

/-- The number of kittens Alyssa now has -/
def remaining_kittens : ℕ := initial_kittens - kittens_given_away

theorem alyssa_kittens : remaining_kittens = 4 := by
  sorry

end alyssa_kittens_l3497_349729


namespace line_intercepts_sum_l3497_349706

/-- Proves that for the line 2x - 3y - 6k = 0, if the sum of its x-intercept and y-intercept is 1, then k = 1 -/
theorem line_intercepts_sum (k : ℝ) : 
  (∃ x y : ℝ, 2*x - 3*y - 6*k = 0 ∧ 
   (2*(3*k) - 3*0 - 6*k = 0) ∧ 
   (2*0 - 3*(-2*k) - 6*k = 0) ∧ 
   3*k + (-2*k) = 1) → 
  k = 1 := by
  sorry

end line_intercepts_sum_l3497_349706


namespace point_not_on_graph_l3497_349724

theorem point_not_on_graph : ¬ ∃ (y : ℝ), y = (-2 - 1) / (-2 + 2) ∧ y = 1 := by sorry

end point_not_on_graph_l3497_349724


namespace perimeter_difference_l3497_349707

-- Define the perimeter of the first figure
def perimeter_figure1 : ℕ :=
  -- Outer rectangle perimeter
  2 * (5 + 2) +
  -- Middle vertical rectangle contribution
  2 * 3 +
  -- Inner vertical rectangle contribution
  2 * 2

-- Define the perimeter of the second figure
def perimeter_figure2 : ℕ :=
  -- Outer rectangle perimeter
  2 * (5 + 3) +
  -- Vertical lines contribution
  5 * 2

-- Theorem statement
theorem perimeter_difference : perimeter_figure2 - perimeter_figure1 = 2 := by
  sorry

end perimeter_difference_l3497_349707


namespace distribution_count_theorem_l3497_349743

/-- Represents a boat with its capacity -/
structure Boat where
  capacity : Nat

/-- Represents the distribution of people on boats -/
structure Distribution where
  adults : Nat
  children : Nat

/-- Checks if a distribution is valid (i.e., has an adult if there's a child) -/
def is_valid_distribution (d : Distribution) : Bool :=
  d.children > 0 → d.adults > 0

/-- Counts the number of valid ways to distribute people on boats -/
def count_valid_distributions (boats : List Boat) (total_adults total_children : Nat) : Nat :=
  sorry -- The actual implementation would go here

/-- The main theorem to prove -/
theorem distribution_count_theorem :
  let boats := [Boat.mk 3, Boat.mk 2, Boat.mk 1]
  count_valid_distributions boats 3 2 = 33 := by
  sorry

#check distribution_count_theorem

end distribution_count_theorem_l3497_349743


namespace third_sample_is_43_l3497_349787

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) (n : ℕ) : ℕ :=
  start + (n - 1) * (total / sample_size)

/-- Theorem for the specific problem -/
theorem third_sample_is_43 
  (total : ℕ) (sample_size : ℕ) (start : ℕ) 
  (h1 : total = 900) 
  (h2 : sample_size = 50) 
  (h3 : start = 7) :
  systematic_sample total sample_size start 3 = 43 := by
  sorry

#eval systematic_sample 900 50 7 3

end third_sample_is_43_l3497_349787


namespace tom_total_games_l3497_349754

/-- The number of hockey games Tom attended over two years -/
def total_games (games_this_year games_last_year : ℕ) : ℕ :=
  games_this_year + games_last_year

/-- Theorem: Tom attended 13 hockey games in total over two years -/
theorem tom_total_games :
  total_games 4 9 = 13 := by
  sorry

end tom_total_games_l3497_349754


namespace composite_form_l3497_349777

theorem composite_form (x : ℤ) (m n : ℕ) (hm : m > 0) (hn : n ≥ 0) :
  x^(4*m) + 2^(4*n + 2) = (x^(2*m) + 2^(2*n + 1) + 2^(n + 1) * x^m) * ((x^m - 2^n)^2 + 2^(2*n)) :=
by sorry

end composite_form_l3497_349777


namespace intersection_points_l3497_349703

-- Define the functions f and g
def f (t x : ℝ) : ℝ := t * x^2 - x + 1
def g (t x : ℝ) : ℝ := 2 * t * x - 1

-- Define the discriminant function
def discriminant (t : ℝ) : ℝ := (2 * t - 1)^2

-- Theorem statement
theorem intersection_points (t : ℝ) :
  (∃ x : ℝ, f t x = g t x) ∧
  (∀ x y : ℝ, f t x = g t x ∧ f t y = g t y → x = y ∨ (∃ z : ℝ, f t z = g t z ∧ z ≠ x ∧ z ≠ y)) :=
by sorry

end intersection_points_l3497_349703


namespace quadratic_single_solution_l3497_349792

theorem quadratic_single_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x : ℝ, b * x^2 + 16 * x + 5 = 0) →
  (∃ x : ℝ, b * x^2 + 16 * x + 5 = 0 ∧ x = -5/8) :=
by sorry

end quadratic_single_solution_l3497_349792


namespace four_inequalities_l3497_349784

theorem four_inequalities :
  (∃ (x : ℝ), x = Real.sqrt (2 * Real.sqrt (2 * Real.sqrt (2 * Real.sqrt 2))) ∧ x < 2) ∧
  (∃ (y : ℝ), y = Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))) ∧ y < 2) ∧
  (∃ (z : ℝ), z = Real.sqrt (3 * Real.sqrt (3 * Real.sqrt (3 * Real.sqrt 3))) ∧ z < 3) ∧
  (∃ (w : ℝ), w = Real.sqrt (3 + Real.sqrt (3 + Real.sqrt (3 + Real.sqrt 3))) ∧ w < 3) :=
by sorry

end four_inequalities_l3497_349784


namespace hyperbola_transverse_axis_range_l3497_349744

theorem hyperbola_transverse_axis_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ y : ℝ, y^2 / a^2 - (2*y)^2 / b^2 = 1) →
  b^2 = 1 - a^2 →
  0 < 2*a ∧ 2*a < 2*Real.sqrt 5 / 5 := by
sorry

end hyperbola_transverse_axis_range_l3497_349744


namespace complement_union_equals_set_l3497_349700

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_union_equals_set : (U \ (M ∪ N)) = {1, 6} := by sorry

end complement_union_equals_set_l3497_349700


namespace distance_to_karasuk_proof_l3497_349727

/-- The distance from Novosibirsk to Karasuk -/
def distance_to_karasuk : ℝ := 140

/-- The speed of the bus -/
def bus_speed : ℝ := 1

/-- The speed of the car -/
def car_speed : ℝ := 2 * bus_speed

/-- The initial distance the bus traveled before the car started -/
def initial_bus_distance : ℝ := 70

/-- The distance the bus traveled after Karasuk -/
def bus_distance_after_karasuk : ℝ := 20

/-- The distance the car traveled after Karasuk -/
def car_distance_after_karasuk : ℝ := 40

theorem distance_to_karasuk_proof :
  distance_to_karasuk = initial_bus_distance + 
    (car_distance_after_karasuk * bus_speed / car_speed) :=
by sorry

end distance_to_karasuk_proof_l3497_349727


namespace school_population_proof_l3497_349701

theorem school_population_proof (x : ℝ) (h1 : 162 = (x / 100) * (0.5 * x)) : x = 180 := by
  sorry

end school_population_proof_l3497_349701


namespace pace_ratio_l3497_349702

/-- The ratio of a man's pace on a day he was late to his usual pace -/
theorem pace_ratio (usual_time : ℝ) (late_time : ℝ) (h1 : usual_time = 2) 
  (h2 : late_time = usual_time + 1/3) : 
  (usual_time / late_time) = 6/7 := by
  sorry

end pace_ratio_l3497_349702


namespace cube_root_problem_l3497_349748

theorem cube_root_problem (a : ℝ) : 
  (27 : ℝ) ^ (1/3) = a + 3 → (a + 4).sqrt = 2 := by
  sorry

end cube_root_problem_l3497_349748


namespace range_of_a_l3497_349731

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x - 5) / (x - 3) ≥ 2
def q (x a : ℝ) : Prop := x^2 - a*x ≤ x - a

-- Define the condition that not p implies not q
def not_p_implies_not_q (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(p x) → ¬(q x a)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, not_p_implies_not_q a) → a ∈ Set.Icc 1 3 := by
  sorry

end range_of_a_l3497_349731


namespace min_cubes_for_specific_block_l3497_349721

/-- The minimum number of cubes needed to create a hollow block -/
def min_cubes_hollow_block (length width depth : ℕ) : ℕ :=
  let total_cubes := length * width * depth
  let hollow_length := length - 2
  let hollow_width := width - 2
  let hollow_depth := depth - 2
  let hollow_cubes := hollow_length * hollow_width * hollow_depth
  total_cubes - hollow_cubes

/-- Theorem stating the minimum number of cubes needed for the specific block -/
theorem min_cubes_for_specific_block :
  min_cubes_hollow_block 4 10 7 = 200 := by
  sorry

#eval min_cubes_hollow_block 4 10 7

end min_cubes_for_specific_block_l3497_349721


namespace square_sum_equality_l3497_349795

theorem square_sum_equality (p q r a b c : ℝ) 
  (h1 : p + q + r = 1) 
  (h2 : 1/p + 1/q + 1/r = 0) : 
  a^2 + b^2 + c^2 = (p*a + q*b + r*c)^2 + (q*a + r*b + p*c)^2 + (r*a + p*b + q*c)^2 := by
  sorry

end square_sum_equality_l3497_349795


namespace ball_cost_l3497_349759

/-- Given that 3 balls cost $4.62, prove that each ball costs $1.54. -/
theorem ball_cost (total_cost : ℝ) (num_balls : ℕ) (cost_per_ball : ℝ) 
  (h1 : total_cost = 4.62)
  (h2 : num_balls = 3)
  (h3 : cost_per_ball = total_cost / num_balls) : 
  cost_per_ball = 1.54 := by
  sorry

end ball_cost_l3497_349759


namespace alice_coin_difference_l3497_349769

/-- Proves that given the conditions of Alice's coin collection, she has 3 more 10-cent coins than 25-cent coins -/
theorem alice_coin_difference :
  ∀ (n d q : ℕ),
  n + d + q = 30 →
  5 * n + 10 * d + 25 * q = 435 →
  d = n + 6 →
  q = 10 →
  d - q = 3 := by
sorry

end alice_coin_difference_l3497_349769


namespace solve_linear_system_l3497_349739

/-- Given a system of linear equations:
     a + b = c
     b + c = 7
     c - a = 2
    Prove that b = 2 -/
theorem solve_linear_system (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 7) 
  (eq3 : c - a = 2) : 
  b = 2 := by
  sorry

end solve_linear_system_l3497_349739


namespace binomial_expansion_theorem_l3497_349736

theorem binomial_expansion_theorem (y b : ℚ) (m : ℕ) : 
  (Nat.choose m 4 : ℚ) * y^(m-4) * b^4 = 210 →
  (Nat.choose m 5 : ℚ) * y^(m-5) * b^5 = 462 →
  (Nat.choose m 6 : ℚ) * y^(m-6) * b^6 = 792 →
  m = 7 := by sorry

end binomial_expansion_theorem_l3497_349736


namespace stratified_sampling_seniors_l3497_349749

/-- Represents the number of students in each grade -/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  seniors : ℕ

/-- Calculates the total number of students -/
def totalStudents (pop : GradePopulation) : ℕ :=
  pop.freshmen + pop.sophomores + pop.seniors

/-- Calculates the number of students to sample from each grade -/
def stratifiedSample (pop : GradePopulation) (sampleSize : ℕ) : GradePopulation :=
  let total := totalStudents pop
  let factor := sampleSize / total
  { freshmen := pop.freshmen * factor,
    sophomores := pop.sophomores * factor,
    seniors := pop.seniors * factor }

theorem stratified_sampling_seniors
  (pop : GradePopulation)
  (h1 : pop.freshmen = 520)
  (h2 : pop.sophomores = 500)
  (h3 : pop.seniors = 580)
  (h4 : totalStudents pop = 1600)
  (sampleSize : ℕ)
  (h5 : sampleSize = 80) :
  (stratifiedSample pop sampleSize).seniors = 29 := by
  sorry

end stratified_sampling_seniors_l3497_349749


namespace zeros_properties_l3497_349738

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 3

theorem zeros_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : f a x₁ = 0) 
  (h₂ : f a x₂ = 0) 
  (h₃ : x₁ < x₂) : 
  (0 < a ∧ a < Real.exp 2) ∧ x₁ + x₂ > 2 * a := by
  sorry

end zeros_properties_l3497_349738


namespace lesser_fraction_proof_l3497_349705

theorem lesser_fraction_proof (x y : ℚ) : 
  x + y = 11/12 → x * y = 1/6 → min x y = 1/4 := by
  sorry

end lesser_fraction_proof_l3497_349705


namespace fourth_rectangle_area_determined_l3497_349733

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the division of a large rectangle into four smaller rectangles -/
structure DividedRectangle where
  large : Rectangle
  efgh : Rectangle
  efij : Rectangle
  ijkl : Rectangle
  ghkl : Rectangle
  h_division : 
    large.length = efgh.length + ijkl.length ∧ 
    large.width = efgh.width + efij.width

/-- Theorem stating that the area of the fourth rectangle (GHKL) is uniquely determined -/
theorem fourth_rectangle_area_determined (dr : DividedRectangle) : 
  ∃! a : ℝ, a = dr.ghkl.area ∧ 
    dr.large.area = dr.efgh.area + dr.efij.area + dr.ijkl.area + a :=
sorry

end fourth_rectangle_area_determined_l3497_349733


namespace triangle_abc_property_l3497_349750

theorem triangle_abc_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  b * Real.sin B - a * Real.sin A = c →
  -- Additional conditions
  c = Real.sqrt 3 →
  C = π / 3 →
  -- Conclusions
  B - A = π / 2 ∧
  (1 / 2 : Real) * a * c * Real.sin B = Real.sqrt 3 / 4 := by
  sorry

end triangle_abc_property_l3497_349750


namespace multiples_of_seven_square_l3497_349775

theorem multiples_of_seven_square (a b : ℕ) : 
  (∀ k : ℕ, k ≤ a → (7 * k < 50)) ∧ 
  (∀ k : ℕ, k > a → (7 * k ≥ 50)) ∧
  (∀ k : ℕ, k ≤ b → (k * 7 < 50 ∧ k > 0)) ∧
  (∀ k : ℕ, k > b → (k * 7 ≥ 50 ∨ k ≤ 0)) →
  (a + b)^2 = 196 := by
sorry

end multiples_of_seven_square_l3497_349775


namespace right_triangle_from_medians_l3497_349780

theorem right_triangle_from_medians (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    x^2 = (16 * b^2 - 4 * a^2) / 15 ∧
    y^2 = (16 * a^2 - 4 * b^2) / 15 ∧
    x^2 + y^2 = a^2 + b^2 := by
  sorry

end right_triangle_from_medians_l3497_349780


namespace annie_original_seat_l3497_349741

-- Define the type for seats
inductive Seat
| one
| two
| three
| four
| five

-- Define the type for friends
inductive Friend
| Annie
| Beth
| Cass
| Dana
| Ella

-- Define the function type for seating arrangement
def SeatingArrangement := Seat → Friend

-- Define the movement function type
def Movement := SeatingArrangement → SeatingArrangement

-- Define the specific movements
def bethMove : Movement := sorry
def cassDanaSwap : Movement := sorry
def ellaMove : Movement := sorry

-- Define the property of Ella ending in an end seat
def ellaInEndSeat (arrangement : SeatingArrangement) : Prop := sorry

-- Define the theorem
theorem annie_original_seat (initial : SeatingArrangement) :
  (∃ (final : SeatingArrangement),
    final = ellaMove (cassDanaSwap (bethMove initial)) ∧
    ellaInEndSeat final) →
  initial Seat.one = Friend.Annie := by sorry

end annie_original_seat_l3497_349741


namespace savings_percentage_l3497_349740

theorem savings_percentage (income : ℝ) (savings_rate : ℝ) : 
  savings_rate = 0.35 →
  (2 : ℝ) * (income * (1 - savings_rate)) = 
    income * (1 - savings_rate) + income * (1 - 2 * savings_rate) →
  savings_rate = 0.35 := by
  sorry

end savings_percentage_l3497_349740


namespace locus_is_equidistant_l3497_349711

/-- The locus of points equidistant from the x-axis and point F(0, 2) -/
def locus_equation (x y : ℝ) : Prop :=
  y = x^2 / 4 + 1

/-- A point is equidistant from the x-axis and F(0, 2) -/
def is_equidistant (x y : ℝ) : Prop :=
  abs y = Real.sqrt (x^2 + (y - 2)^2)

/-- Theorem: The locus equation represents points equidistant from x-axis and F(0, 2) -/
theorem locus_is_equidistant :
  ∀ x y : ℝ, locus_equation x y ↔ is_equidistant x y :=
by sorry

end locus_is_equidistant_l3497_349711


namespace vector_b_magnitude_l3497_349753

def a : ℝ × ℝ := (-2, -1)

theorem vector_b_magnitude (b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10)
  (h2 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5) : 
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end vector_b_magnitude_l3497_349753


namespace yellow_balls_count_l3497_349768

theorem yellow_balls_count (total : ℕ) (red yellow green : ℕ) : 
  total = 68 →
  2 * red = yellow →
  3 * green = 4 * yellow →
  red + yellow + green = total →
  yellow = 24 := by
sorry

end yellow_balls_count_l3497_349768


namespace cubic_root_sum_l3497_349770

theorem cubic_root_sum (a b c d : ℝ) (ha : a ≠ 0) 
  (h1 : a * 5^3 + b * 5^2 + c * 5 + d = 0)
  (h2 : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -19 := by
  sorry

end cubic_root_sum_l3497_349770


namespace quadratic_equations_integer_solutions_l3497_349756

theorem quadratic_equations_integer_solutions 
  (b c : ℤ) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h1 : ∃ x y : ℤ, x ≠ y ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0)
  (h2 : ∃ u v : ℤ, u ≠ v ∧ u^2 + b*u - c = 0 ∧ v^2 + b*v - c = 0) :
  (∃ p q : ℕ+, p ≠ q ∧ 2*b^2 = p^2 + q^2) ∧
  (∃ r s : ℕ+, r ≠ s ∧ b^2 = r^2 + s^2) := by
sorry

end quadratic_equations_integer_solutions_l3497_349756


namespace reading_assignment_solution_l3497_349719

/-- Represents the reading assignment for Mrs. Reed's English class -/
structure ReadingAssignment where
  total_pages : ℕ
  alice_speed : ℕ  -- seconds per page
  bob_speed : ℕ    -- seconds per page
  chandra_speed : ℕ -- seconds per page
  x : ℕ  -- last page Alice reads
  y : ℕ  -- last page Chandra reads

/-- Checks if the reading assignment satisfies the given conditions -/
def is_valid_assignment (r : ReadingAssignment) : Prop :=
  r.total_pages = 910 ∧
  r.alice_speed = 30 ∧
  r.bob_speed = 60 ∧
  r.chandra_speed = 45 ∧
  r.x < r.y ∧
  r.y < r.total_pages ∧
  r.alice_speed * r.x = r.chandra_speed * (r.y - r.x) ∧
  r.chandra_speed * (r.y - r.x) = r.bob_speed * (r.total_pages - r.y)

/-- Theorem stating the unique solution for the reading assignment -/
theorem reading_assignment_solution (r : ReadingAssignment) :
  is_valid_assignment r → r.x = 420 ∧ r.y = 700 := by
  sorry


end reading_assignment_solution_l3497_349719


namespace food_distribution_proof_l3497_349761

/-- The initial number of men in the group -/
def initial_men : ℕ := 760

/-- The number of additional men who join after 2 days -/
def additional_men : ℕ := 190

/-- The initial number of days the food would last -/
def initial_days : ℕ := 22

/-- The number of days that pass before additional men join -/
def days_before_addition : ℕ := 2

/-- The number of days the food lasts after additional men join -/
def remaining_days : ℕ := 16

theorem food_distribution_proof :
  initial_men * initial_days = 
  (initial_men * days_before_addition) + 
  ((initial_men + additional_men) * remaining_days) := by
  sorry

end food_distribution_proof_l3497_349761


namespace calculate_expression_l3497_349712

theorem calculate_expression : -1^4 - (1 - 0.4) * (1/3) * (2 - 3^2) = 0.4 := by
  sorry

end calculate_expression_l3497_349712


namespace inequality_proof_l3497_349751

theorem inequality_proof (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c + b + a = 0) : b * c > a * c := by
  sorry

end inequality_proof_l3497_349751


namespace james_new_friends_l3497_349786

def number_of_new_friends (initial_friends lost_friends final_friends : ℕ) : ℕ :=
  final_friends - (initial_friends - lost_friends)

theorem james_new_friends :
  number_of_new_friends 20 2 19 = 1 := by
  sorry

end james_new_friends_l3497_349786


namespace a_gt_abs_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3497_349790

theorem a_gt_abs_b_sufficient_not_necessary_for_a_sq_gt_b_sq :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) := by
  sorry

end a_gt_abs_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3497_349790


namespace correct_calculation_l3497_349765

theorem correct_calculation : -5 * (-4) * (-2) * (-2) = 80 := by
  sorry

end correct_calculation_l3497_349765


namespace gcd_of_squares_sum_l3497_349794

theorem gcd_of_squares_sum : Nat.gcd (125^2 + 235^2 + 349^2) (124^2 + 234^2 + 350^2) = 1 := by
  sorry

end gcd_of_squares_sum_l3497_349794


namespace expression_simplification_and_evaluation_l3497_349781

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 1 / (Real.sqrt 2 - 1)) :
  (1 - 4 / (x + 3)) / ((x^2 - 2*x + 1) / (2*x + 6)) = Real.sqrt 2 := by
  sorry

end expression_simplification_and_evaluation_l3497_349781


namespace arithmetic_associativity_l3497_349783

theorem arithmetic_associativity (a b c : ℚ) : 
  ((a + b) + c = a + (b + c)) ∧
  ((a - b) - c ≠ a - (b - c)) ∧
  ((a * b) * c = a * (b * c)) ∧
  (a / b / c ≠ a / (b / c)) := by
  sorry

#check arithmetic_associativity

end arithmetic_associativity_l3497_349783


namespace center_is_five_l3497_349752

-- Define the grid type
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define the property of consecutive numbers sharing an edge
def ConsecutiveShareEdge (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, g i j = g k l + 1 →
    ((i = k ∧ j.val + 1 = l.val) ∨ (i = k ∧ j.val = l.val + 1) ∨
     (i.val + 1 = k.val ∧ j = l) ∨ (i.val = k.val + 1 ∧ j = l))

-- Define the sum of corner numbers
def CornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

-- Define the sum of numbers along one diagonal
def DiagonalSum (g : Grid) : Nat :=
  g 0 0 + g 1 1 + g 2 2

-- Theorem statement
theorem center_is_five (g : Grid) 
  (grid_nums : ∀ i j, g i j ∈ Finset.range 9)
  (consecutive_edge : ConsecutiveShareEdge g)
  (corner_sum : CornerSum g = 20)
  (diagonal_sum : DiagonalSum g = 15) :
  g 1 1 = 5 := by
  sorry

end center_is_five_l3497_349752


namespace factors_of_M_l3497_349713

/-- The number of natural-number factors of M, where M = 2^5 · 3^4 · 5^3 · 7^3 · 11^2 -/
def num_factors (M : ℕ) : ℕ :=
  (5 + 1) * (4 + 1) * (3 + 1) * (3 + 1) * (2 + 1)

/-- Theorem stating that the number of natural-number factors of M is 1440 -/
theorem factors_of_M :
  let M : ℕ := 2^5 * 3^4 * 5^3 * 7^3 * 11^2
  num_factors M = 1440 := by sorry

end factors_of_M_l3497_349713


namespace probability_a_speaks_truth_l3497_349764

theorem probability_a_speaks_truth 
  (prob_b : ℝ) 
  (prob_both : ℝ) 
  (h1 : prob_b = 0.60)
  (h2 : prob_both = 0.33)
  (h3 : prob_both = prob_a * prob_b)
  : prob_a = 0.55 :=
by sorry

end probability_a_speaks_truth_l3497_349764


namespace bucket_water_difference_l3497_349735

/-- Given two buckets with initial volumes and a water transfer between them,
    prove the resulting volume difference. -/
theorem bucket_water_difference 
  (large_initial small_initial transfer : ℕ)
  (h1 : large_initial = 7)
  (h2 : small_initial = 5)
  (h3 : transfer = 2)
  : large_initial + transfer - (small_initial - transfer) = 6 := by
  sorry

end bucket_water_difference_l3497_349735


namespace range_of_z_l3497_349798

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  ∃ (z : ℝ), z = x + y ∧ -Real.sqrt 6 ≤ z ∧ z ≤ Real.sqrt 6 :=
by sorry

end range_of_z_l3497_349798


namespace eighty_sixth_word_ends_with_E_l3497_349773

-- Define the set of letters
inductive Letter : Type
| A | H | S | M | E

-- Define a permutation as a list of letters
def Permutation := List Letter

-- Define the dictionary order for permutations
def dict_order (p1 p2 : Permutation) : Prop := sorry

-- Define a function to get the nth permutation in dictionary order
def nth_permutation (n : Nat) : Permutation := sorry

-- Define a function to get the last letter of a permutation
def last_letter (p : Permutation) : Letter := sorry

-- State the theorem
theorem eighty_sixth_word_ends_with_E : 
  last_letter (nth_permutation 86) = Letter.E := by sorry

end eighty_sixth_word_ends_with_E_l3497_349773


namespace planted_fraction_of_field_l3497_349763

theorem planted_fraction_of_field (a b c x : ℝ) (h1 : a = 5) (h2 : b = 12) 
  (h3 : c^2 = a^2 + b^2) (h4 : x^2 * (a - x) * (b - x) = 3 * c * x^2) : 
  (a * b / 2 - x^2) / (a * b / 2) = 1461 / 1470 := by
  sorry

end planted_fraction_of_field_l3497_349763


namespace decimal_to_fraction_l3497_349717

theorem decimal_to_fraction : 
  (2.35 : ℚ) = 47 / 20 := by sorry

end decimal_to_fraction_l3497_349717


namespace complex_fraction_equals_i_l3497_349797

theorem complex_fraction_equals_i : (1 + Complex.I * Real.sqrt 3) / (Real.sqrt 3 - Complex.I) = Complex.I := by
  sorry

end complex_fraction_equals_i_l3497_349797


namespace largest_four_digit_divisible_by_88_and_prime_l3497_349722

theorem largest_four_digit_divisible_by_88_and_prime : ∃ (p : ℕ), 
  p.Prime ∧ 
  p > 100 ∧ 
  9944 % 88 = 0 ∧ 
  9944 % p = 0 ∧ 
  ∀ (n : ℕ), n > 9944 → n < 10000 → ¬(n % 88 = 0 ∧ ∃ (q : ℕ), q.Prime ∧ q > 100 ∧ n % q = 0) :=
by sorry

end largest_four_digit_divisible_by_88_and_prime_l3497_349722


namespace fraction_expansion_invariance_l3497_349715

theorem fraction_expansion_invariance (m n : ℝ) (h : m ≠ n) :
  (2 * (3 * m)) / ((3 * m) - (3 * n)) = (2 * m) / (m - n) := by
  sorry

end fraction_expansion_invariance_l3497_349715


namespace pie_not_crust_percentage_l3497_349796

/-- Given a pie weighing 200 grams with 50 grams of crust, 
    the percentage of the pie that is not crust is 75%. -/
theorem pie_not_crust_percentage 
  (total_weight : ℝ) 
  (crust_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : crust_weight = 50) : 
  (total_weight - crust_weight) / total_weight * 100 = 75 := by
sorry

end pie_not_crust_percentage_l3497_349796


namespace quadratic_inequality_solution_set_l3497_349779

theorem quadratic_inequality_solution_set (a : ℝ) (ha : a > 0) :
  {x : ℝ | x^2 - 4*a*x - 5*a^2 < 0} = {x : ℝ | -a < x ∧ x < 5*a} := by
sorry

end quadratic_inequality_solution_set_l3497_349779


namespace no_real_solutions_for_composition_l3497_349718

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Theorem: If f(x) = ax^2 + bx + c is a quadratic function and f(x) = x has no real solutions,
    then f(f(x)) = x also has no real solutions -/
theorem no_real_solutions_for_composition
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c x ≠ x) :
  ∀ x : ℝ, quadratic_function a b c (quadratic_function a b c x) ≠ x :=
by
  sorry

end no_real_solutions_for_composition_l3497_349718


namespace arithmetic_sum_specific_l3497_349760

def arithmetic_sum (a₁ l d : ℤ) : ℤ :=
  let n := (l - a₁) / d + 1
  n * (a₁ + l) / 2

theorem arithmetic_sum_specific : arithmetic_sum (-45) 1 2 = -528 := by
  sorry

end arithmetic_sum_specific_l3497_349760


namespace problem_solution_l3497_349734

theorem problem_solution (a : ℝ) (h : a = 2 / (3 - Real.sqrt 7)) :
  -2 * a^2 + 12 * a + 3 = 7 := by
  sorry

end problem_solution_l3497_349734


namespace sawyer_cut_difference_l3497_349762

/-- Represents a sawyer with their stick length and number of sections sawed -/
structure Sawyer where
  stickLength : Nat
  sectionsSawed : Nat

/-- Calculates the number of cuts made by a sawyer -/
def calculateCuts (s : Sawyer) : Nat :=
  (s.stickLength / 2 - 1) * (s.sectionsSawed / (s.stickLength / 2))

theorem sawyer_cut_difference (a b c : Sawyer)
  (h1 : a.stickLength = 8 ∧ b.stickLength = 10 ∧ c.stickLength = 6)
  (h2 : a.sectionsSawed = 24 ∧ b.sectionsSawed = 25 ∧ c.sectionsSawed = 27) :
  (max (max (calculateCuts a) (calculateCuts b)) (calculateCuts c) -
   min (min (calculateCuts a) (calculateCuts b)) (calculateCuts c)) = 2 := by
  sorry

end sawyer_cut_difference_l3497_349762


namespace chord_length_squared_l3497_349785

/-- Two circles with radii 10 and 7, centers 15 units apart, intersecting at P.
    A line through P creates equal chords QP and PR. -/
structure IntersectingCircles where
  r₁ : ℝ
  r₂ : ℝ
  center_distance : ℝ
  chord_length : ℝ
  h₁ : r₁ = 10
  h₂ : r₂ = 7
  h₃ : center_distance = 15
  h₄ : chord_length > 0

/-- The square of the length of chord QP in the given configuration is 289. -/
theorem chord_length_squared (c : IntersectingCircles) : c.chord_length ^ 2 = 289 := by
  sorry

end chord_length_squared_l3497_349785


namespace cos_15_cos_30_minus_sin_15_sin_150_l3497_349726

theorem cos_15_cos_30_minus_sin_15_sin_150 :
  Real.cos (15 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end cos_15_cos_30_minus_sin_15_sin_150_l3497_349726


namespace sin_cos_cube_sum_l3497_349774

theorem sin_cos_cube_sum (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/2) :
  Real.sin θ ^ 3 + Real.cos θ ^ 3 = 11/16 := by
  sorry

end sin_cos_cube_sum_l3497_349774


namespace bags_collection_l3497_349720

/-- Calculates the total number of bags collected over three days -/
def totalBags (initial : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  initial + day2 + day3

/-- Theorem stating that the total number of bags is 20 given the specific conditions -/
theorem bags_collection :
  totalBags 10 3 7 = 20 := by
  sorry

end bags_collection_l3497_349720


namespace spelling_homework_time_l3497_349708

theorem spelling_homework_time (total_time math_time reading_time : ℕ) 
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : reading_time = 27) :
  total_time - math_time - reading_time = 18 := by
  sorry

end spelling_homework_time_l3497_349708


namespace problem_solution_l3497_349714

theorem problem_solution (x : ℝ) (h : x + Real.sqrt (x^2 - 1) + 1 / (x + Real.sqrt (x^2 - 1)) = 12) :
  x^3 + Real.sqrt (x^6 - 1) + 1 / (x^3 + Real.sqrt (x^6 - 1)) = 432 := by
  sorry

end problem_solution_l3497_349714


namespace only_234_and_468_satisfy_l3497_349716

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def satisfiesCondition (n : Nat) : Prop :=
  n < 10000 ∧ n = 26 * sumOfDigits n

theorem only_234_and_468_satisfy :
  ∀ n : Nat, satisfiesCondition n ↔ n = 234 ∨ n = 468 := by
  sorry

end only_234_and_468_satisfy_l3497_349716


namespace number_of_bowls_l3497_349723

theorem number_of_bowls (n : ℕ) 
  (h1 : n > 0)  -- There is at least one bowl
  (h2 : 12 ≤ n)  -- There are at least 12 bowls to add grapes to
  (h3 : (96 : ℝ) / n = 6)  -- The average increase is 6
  : n = 16 := by
sorry

end number_of_bowls_l3497_349723


namespace triangles_from_parallel_lines_l3497_349758

/-- The number of triangles formed by points on two parallel lines -/
theorem triangles_from_parallel_lines (n m : ℕ) (hn : n = 6) (hm : m = 8) :
  n.choose 2 * m + n * m.choose 2 = 288 := by
  sorry

end triangles_from_parallel_lines_l3497_349758


namespace hyperbola_real_axis_length_l3497_349704

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points A and B
def intersectionPoints (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), hyperbola a b x₁ y₁ ∧ parabola x₁ y₁ ∧
                       hyperbola a b x₂ y₂ ∧ parabola x₂ y₂ ∧
                       (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the common focus F
def commonFocus (a b : ℝ) : Prop :=
  ∃ (xf yf : ℝ), (xf = a ∧ yf = 0) ∧ (xf = 1 ∧ yf = 0)

-- Define that line AB passes through F
def lineABThroughF (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ xf yf : ℝ),
    hyperbola a b x₁ y₁ ∧ parabola x₁ y₁ ∧
    hyperbola a b x₂ y₂ ∧ parabola x₂ y₂ ∧
    commonFocus a b ∧
    (y₂ - y₁) * (xf - x₁) = (yf - y₁) * (x₂ - x₁)

-- Theorem statement
theorem hyperbola_real_axis_length
  (a b : ℝ)
  (h_intersect : intersectionPoints a b)
  (h_focus : commonFocus a b)
  (h_line : lineABThroughF a b) :
  2 * a = 2 * Real.sqrt 2 - 2 :=
sorry

end hyperbola_real_axis_length_l3497_349704


namespace tan_product_seventh_pi_l3497_349737

theorem tan_product_seventh_pi : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end tan_product_seventh_pi_l3497_349737


namespace milk_consumption_l3497_349710

theorem milk_consumption (initial_milk : ℚ) (rachel_fraction : ℚ) (monica_fraction : ℚ) : 
  initial_milk = 3/4 →
  rachel_fraction = 1/2 →
  monica_fraction = 1/3 →
  let rachel_consumption := rachel_fraction * initial_milk
  let remaining_milk := initial_milk - rachel_consumption
  let monica_consumption := monica_fraction * remaining_milk
  rachel_consumption + monica_consumption = 1/2 := by
sorry

end milk_consumption_l3497_349710


namespace xiaoGang_weight_not_80_grams_l3497_349755

-- Define a person
structure Person where
  name : String
  weight : Float  -- weight in kilograms

-- Define Xiao Gang
def xiaoGang : Person := { name := "Xiao Gang", weight := 80 }

-- Theorem to prove
theorem xiaoGang_weight_not_80_grams : 
  xiaoGang.weight ≠ 0.08 := by sorry

end xiaoGang_weight_not_80_grams_l3497_349755


namespace banana_bread_ratio_l3497_349776

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves made on Monday -/
def loaves_monday : ℕ := 3

/-- The total number of bananas used for both days -/
def total_bananas : ℕ := 36

/-- The number of loaves made on Tuesday -/
def loaves_tuesday : ℕ := (total_bananas - loaves_monday * bananas_per_loaf) / bananas_per_loaf

theorem banana_bread_ratio :
  loaves_tuesday / loaves_monday = 2 := by sorry

end banana_bread_ratio_l3497_349776


namespace circle_intersection_range_l3497_349757

-- Define the circles
def circle1 (x y m : ℝ) : Prop := x^2 + y^2 = m
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y - 11 = 0

-- Define the intersection of the circles
def circles_intersect (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle1 x y m ∧ circle2 x y

-- Theorem statement
theorem circle_intersection_range (m : ℝ) :
  circles_intersect m ↔ 1 < m ∧ m < 121 :=
sorry

end circle_intersection_range_l3497_349757


namespace zeros_of_f_l3497_349766

def f (x : ℝ) := -x^2 + 5*x - 6

theorem zeros_of_f :
  ∃ (a b : ℝ), (∀ x, f x = 0 ↔ x = a ∨ x = b) ∧ a = 2 ∧ b = 3 := by
  sorry

end zeros_of_f_l3497_349766


namespace billiard_path_equals_diagonals_l3497_349746

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a point on the rectangle's perimeter -/
structure PerimeterPoint where
  x : ℝ
  y : ℝ

/-- Calculates the length of the billiard ball's path -/
def billiardPathLength (rect : Rectangle) (start : PerimeterPoint) : ℝ :=
  sorry

/-- Calculates the sum of the diagonals of the rectangle -/
def sumOfDiagonals (rect : Rectangle) : ℝ :=
  sorry

/-- Theorem: The billiard path length equals the sum of the rectangle's diagonals -/
theorem billiard_path_equals_diagonals (rect : Rectangle) (start : PerimeterPoint) :
  billiardPathLength rect start = sumOfDiagonals rect :=
  sorry

end billiard_path_equals_diagonals_l3497_349746


namespace ellipse_properties_l3497_349767

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  sum_focal_distances : ℝ → ℝ → ℝ
  eccentricity : ℝ
  focal_sum_eq : ∀ x y, x^2/a^2 + y^2/b^2 = 1 → sum_focal_distances x y = 2 * Real.sqrt 3
  ecc_eq : eccentricity = Real.sqrt 3 / 3

/-- Point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2/E.a^2 + y^2/E.b^2 = 1

/-- Theorem about the standard form of the ellipse and slope product -/
theorem ellipse_properties (E : Ellipse) :
  (E.a^2 = 3 ∧ E.b^2 = 2) ∧
  ∀ (P : PointOnEllipse E) (Q : PointOnEllipse E),
    P.x = 3 →
    (Q.x - 1) * (P.y - 0) + (Q.y - 0) * (P.x - 1) = 0 →
    (Q.y / Q.x) * ((Q.y - P.y) / (Q.x - P.x)) = -2/3 :=
by sorry

end ellipse_properties_l3497_349767


namespace consumption_increase_l3497_349709

theorem consumption_increase (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_tax := 0.65 * original_tax
  let new_revenue := 0.7475 * (original_tax * original_consumption)
  ∃ (new_consumption : ℝ), 
    new_revenue = new_tax * new_consumption ∧ 
    new_consumption = 1.15 * original_consumption :=
by
  sorry

end consumption_increase_l3497_349709


namespace monotonicity_intervals_min_value_l3497_349732

-- Define the function f
def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (∀ x < -1, (f' x) < 0) ∧
  (∀ x > 3, (f' x) < 0) ∧
  (∀ x ∈ Set.Ioo (-1) 3, (f' x) > 0) :=
sorry

-- Theorem for minimum value
theorem min_value (a : ℝ) :
  (∃ x ∈ Set.Icc (-2) 2, f x a = 20) →
  (∃ y ∈ Set.Icc (-2) 2, f y a = -7 ∧ ∀ z ∈ Set.Icc (-2) 2, f z a ≥ -7) :=
sorry

end monotonicity_intervals_min_value_l3497_349732


namespace mosaic_tile_size_l3497_349782

theorem mosaic_tile_size (height width : ℝ) (num_tiles : ℕ) (tile_side : ℝ) : 
  height = 10 → width = 15 → num_tiles = 21600 → 
  (height * width * 144) / num_tiles = tile_side^2 → tile_side = 1 := by
sorry

end mosaic_tile_size_l3497_349782


namespace farm_rent_calculation_l3497_349742

-- Define the constants
def rent_per_acre_per_month : ℝ := 60
def plot_length : ℝ := 360
def plot_width : ℝ := 1210
def square_feet_per_acre : ℝ := 43560

-- Define the theorem
theorem farm_rent_calculation :
  let plot_area : ℝ := plot_length * plot_width
  let acres : ℝ := plot_area / square_feet_per_acre
  let monthly_rent : ℝ := rent_per_acre_per_month * acres
  monthly_rent = 600 := by sorry

end farm_rent_calculation_l3497_349742


namespace volume_of_T_l3497_349799

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the volume of T -/
theorem volume_of_T : volume T = 32 * Real.sqrt 3 / 9 := by sorry

end volume_of_T_l3497_349799


namespace books_printed_count_l3497_349788

def pages_per_book : ℕ := 600
def pages_per_sheet : ℕ := 8  -- 4 pages per side, double-sided
def sheets_used : ℕ := 150

theorem books_printed_count :
  (sheets_used * pages_per_sheet) / pages_per_book = 2 :=
by sorry

end books_printed_count_l3497_349788


namespace quadratic_solution_difference_l3497_349793

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧ 
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧ 
  (x₁ ≠ x₂) ∧ 
  (|x₁ - x₂| = 14) := by
  sorry

end quadratic_solution_difference_l3497_349793
