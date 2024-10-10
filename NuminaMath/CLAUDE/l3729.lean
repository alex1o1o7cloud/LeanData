import Mathlib

namespace sum_of_roots_l3729_372968

theorem sum_of_roots (k d x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ →
  (4 * x₁^2 - k * x₁ = d) →
  (4 * x₂^2 - k * x₂ = d) →
  x₁ + x₂ = k / 4 := by
sorry

end sum_of_roots_l3729_372968


namespace candy_cost_problem_l3729_372954

/-- The cost per pound of the first type of candy -/
def first_candy_cost : ℝ := sorry

/-- The weight of the first type of candy in pounds -/
def first_candy_weight : ℝ := 10

/-- The weight of the second type of candy in pounds -/
def second_candy_weight : ℝ := 20

/-- The cost per pound of the second type of candy -/
def second_candy_cost : ℝ := 5

/-- The cost per pound of the mixture -/
def mixture_cost : ℝ := 6

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := 30

theorem candy_cost_problem :
  first_candy_cost * first_candy_weight + 
  second_candy_cost * second_candy_weight = 
  mixture_cost * total_weight ∧
  first_candy_cost = 8 := by sorry

end candy_cost_problem_l3729_372954


namespace chemical_mixture_percentage_l3729_372945

/-- Given two solutions x and y with different compositions of chemicals a and b,
    and a mixture of these solutions, prove that the percentage of chemical a
    in the mixture is 12%. -/
theorem chemical_mixture_percentage : 
  let x_percent_a : ℝ := 10  -- Percentage of chemical a in solution x
  let x_percent_b : ℝ := 90  -- Percentage of chemical b in solution x
  let y_percent_a : ℝ := 20  -- Percentage of chemical a in solution y
  let y_percent_b : ℝ := 80  -- Percentage of chemical b in solution y
  let mixture_percent_x : ℝ := 80  -- Percentage of solution x in the mixture
  let mixture_percent_y : ℝ := 20  -- Percentage of solution y in the mixture

  -- Ensure percentages add up to 100%
  x_percent_a + x_percent_b = 100 →
  y_percent_a + y_percent_b = 100 →
  mixture_percent_x + mixture_percent_y = 100 →

  -- Calculate the percentage of chemical a in the mixture
  (mixture_percent_x * x_percent_a + mixture_percent_y * y_percent_a) / 100 = 12 :=
by
  sorry


end chemical_mixture_percentage_l3729_372945


namespace quadratic_root_difference_condition_l3729_372925

/-- For a quadratic equation x^2 + px + q = 0, 
    the condition for the difference of its roots to be 'a' is a^2 - p^2 = -4q -/
theorem quadratic_root_difference_condition 
  (p q a : ℝ) 
  (hq : ∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ x₁ ≠ x₂) :
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ x₁ - x₂ = a) ↔ 
  a^2 - p^2 = -4*q :=
by sorry


end quadratic_root_difference_condition_l3729_372925


namespace base_eight_unique_for_729_l3729_372996

/-- Represents a number in base b with digits d₃d₂d₁d₀ --/
def BaseRepresentation (b : ℕ) (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * b^3 + d₂ * b^2 + d₁ * b + d₀

/-- Checks if a number is in XYXY format --/
def IsXYXY (d₃ d₂ d₁ d₀ : ℕ) : Prop :=
  d₃ = d₁ ∧ d₂ = d₀ ∧ d₃ ≠ d₂

theorem base_eight_unique_for_729 :
  ∃! b : ℕ, 6 ≤ b ∧ b ≤ 9 ∧
    ∃ X Y : ℕ, X ≠ Y ∧
      BaseRepresentation b X Y X Y = 729 ∧
      IsXYXY X Y X Y :=
by sorry

end base_eight_unique_for_729_l3729_372996


namespace midpoint_coordinate_product_l3729_372909

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (4, -3) and (-8, 7) is equal to -4. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -3
  let x2 : ℝ := -8
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -4 := by
  sorry

end midpoint_coordinate_product_l3729_372909


namespace mutual_acquaintance_exists_l3729_372921

/-- Represents a diplomatic reception with a fixed number of participants. -/
structure DiplomaticReception where
  participants : Nat
  heardOf : Nat → Nat → Prop
  heardOfCount : Nat → Nat

/-- The minimum number of people each participant has heard of that guarantees mutual acquaintance. -/
def minHeardOfCount : Nat := 50

/-- Theorem stating that if each participant has heard of at least 50 others,
    there must be a pair who have heard of each other. -/
theorem mutual_acquaintance_exists (reception : DiplomaticReception)
    (h1 : reception.participants = 99)
    (h2 : ∀ i, i < reception.participants → reception.heardOfCount i ≥ minHeardOfCount)
    (h3 : ∀ i j, i < reception.participants → j < reception.participants → 
         reception.heardOf i j → reception.heardOfCount i > 0) :
    ∃ i j, i < reception.participants ∧ j < reception.participants ∧ 
    i ≠ j ∧ reception.heardOf i j ∧ reception.heardOf j i := by
  sorry

end mutual_acquaintance_exists_l3729_372921


namespace prob_two_red_is_two_fifths_l3729_372992

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def num_white_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- The number of balls drawn from the bag -/
def num_drawn : ℕ := 2

/-- The probability of drawing two red balls -/
def prob_two_red : ℚ := (num_red_balls.choose num_drawn : ℚ) / (total_balls.choose num_drawn)

theorem prob_two_red_is_two_fifths : prob_two_red = 2 / 5 := by
  sorry

end prob_two_red_is_two_fifths_l3729_372992


namespace kamari_toys_l3729_372913

theorem kamari_toys (kamari_toys : ℕ) (anais_toys : ℕ) :
  anais_toys = kamari_toys + 30 →
  kamari_toys + anais_toys = 160 →
  kamari_toys = 65 := by
sorry

end kamari_toys_l3729_372913


namespace parallelogram_area_l3729_372928

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area :
  let base : ℝ := 20
  let height : ℝ := 4
  let area : ℝ := base * height
  area = 80 := by sorry

end parallelogram_area_l3729_372928


namespace cubic_equation_product_l3729_372949

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2015 ∧ y₁^3 - 3*x₁^2*y₁ = 2014)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2015 ∧ y₂^3 - 3*x₂^2*y₂ = 2014)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2015 ∧ y₃^3 - 3*x₃^2*y₃ = 2014) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -4/1007 := by
  sorry

end cubic_equation_product_l3729_372949


namespace cornbread_pieces_l3729_372917

-- Define the dimensions of the pan
def pan_length : ℕ := 20
def pan_width : ℕ := 18

-- Define the dimensions of each piece of cornbread
def piece_length : ℕ := 2
def piece_width : ℕ := 2

-- Theorem to prove
theorem cornbread_pieces :
  (pan_length * pan_width) / (piece_length * piece_width) = 90 := by
  sorry


end cornbread_pieces_l3729_372917


namespace transformed_point_sum_l3729_372971

/-- Given a function g : ℝ → ℝ such that g(8) = 5, 
    prove that (8/3, 14/9) is on the graph of 3y = g(3x)/3 + 3 
    and that the sum of its coordinates is 38/9 -/
theorem transformed_point_sum (g : ℝ → ℝ) (h : g 8 = 5) : 
  let f : ℝ → ℝ := λ x => (g (3 * x) / 3 + 3) / 3
  f (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
sorry

end transformed_point_sum_l3729_372971


namespace final_savings_after_expense_increase_l3729_372963

/-- Calculates the final savings after expense increase -/
def finalSavings (salary : ℝ) (initialSavingsRate : ℝ) (expenseIncreaseRate : ℝ) : ℝ :=
  let initialExpenses := salary * (1 - initialSavingsRate)
  let newExpenses := initialExpenses * (1 + expenseIncreaseRate)
  salary - newExpenses

/-- Theorem stating that given the problem conditions, the final savings is 250 -/
theorem final_savings_after_expense_increase :
  finalSavings 6250 0.2 0.2 = 250 := by
  sorry

end final_savings_after_expense_increase_l3729_372963


namespace locus_of_centers_is_hyperbola_l3729_372943

/-- A circle with center (x, y) and radius R that touches the diameter of circle k -/
structure TouchingCircle where
  x : ℝ
  y : ℝ
  R : ℝ
  touches_diameter : (-r : ℝ) ≤ x ∧ x ≤ r
  non_negative_y : y ≥ 0
  tangent_to_diameter : R = y

/-- The locus of centers of circles touching the diameter of k and with closest point at distance R from k -/
def locus_of_centers (r : ℝ) (c : TouchingCircle) : Prop :=
  (c.y - 2*r/3)^2 / (r/3)^2 - c.x^2 / (r/Real.sqrt 3)^2 = 1

theorem locus_of_centers_is_hyperbola (r : ℝ) (h : r > 0) :
  ∀ c : TouchingCircle, locus_of_centers r c ↔ 
    c.R = 2 * c.y ∧ 
    Real.sqrt (c.x^2 + c.y^2) = r - 2 * c.y ∧
    r ≥ 3 * c.y :=
  sorry

end locus_of_centers_is_hyperbola_l3729_372943


namespace rectangle_area_l3729_372904

/-- Theorem: Area of a rectangle with length-to-width ratio 4:3 and diagonal d -/
theorem rectangle_area (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 4 / 3 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = 12 / 25 * d ^ 2 := by
  sorry

#check rectangle_area

end rectangle_area_l3729_372904


namespace cobbler_working_hours_l3729_372947

/-- Represents the number of pairs of shoes a cobbler can mend in an hour -/
def shoes_per_hour : ℕ := 3

/-- Represents the number of hours the cobbler works on Friday -/
def friday_hours : ℕ := 3

/-- Represents the total number of pairs of shoes the cobbler can mend in a week -/
def total_shoes_per_week : ℕ := 105

/-- Represents the number of working days from Monday to Thursday -/
def working_days : ℕ := 4

theorem cobbler_working_hours :
  ∃ (h : ℕ), h * working_days * shoes_per_hour + friday_hours * shoes_per_hour = total_shoes_per_week ∧ h = 8 := by
  sorry

end cobbler_working_hours_l3729_372947


namespace trigonometric_simplification_l3729_372910

theorem trigonometric_simplification (α : ℝ) :
  (1 + Real.cos α + Real.cos (2 * α) + Real.cos (3 * α)) /
  (Real.cos α + 2 * (Real.cos α)^2 - 1) = 2 * Real.cos α :=
by sorry

end trigonometric_simplification_l3729_372910


namespace simplify_expression_l3729_372932

theorem simplify_expression (b : ℝ) : 3*b*(3*b^3 - 2*b) - 4*b^2 = 9*b^4 - 10*b^2 := by
  sorry

end simplify_expression_l3729_372932


namespace at_least_one_quadratic_has_two_roots_l3729_372915

theorem at_least_one_quadratic_has_two_roots (p q₁ q₂ : ℝ) (h : p = q₁ + q₂ + 1) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + x₁ + q₁ = 0 ∧ x₂^2 + x₂ + q₁ = 0) ∨
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + p*y₁ + q₂ = 0 ∧ y₂^2 + p*y₂ + q₂ = 0) := by
  sorry

end at_least_one_quadratic_has_two_roots_l3729_372915


namespace max_value_expression_l3729_372978

theorem max_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  (1 / (a^2 - 4*a + 9)) + (1 / (b^2 - 4*b + 9)) + (1 / (c^2 - 4*c + 9)) ≤ 7/18 :=
sorry

end max_value_expression_l3729_372978


namespace water_evaporation_rate_l3729_372933

/-- Given a glass with 10 ounces of water, with 6% evaporating over 20 days,
    the amount of water evaporating each day is 0.03 ounces. -/
theorem water_evaporation_rate (initial_water : ℝ) (days : ℕ) (evaporation_percentage : ℝ) :
  initial_water = 10 →
  days = 20 →
  evaporation_percentage = 0.06 →
  (initial_water * evaporation_percentage) / days = 0.03 :=
by sorry

end water_evaporation_rate_l3729_372933


namespace ellipse_trajectory_l3729_372974

-- Define the focal points
def F1 : ℝ × ℝ := (3, 0)
def F2 : ℝ × ℝ := (-3, 0)

-- Define the distance sum constant
def distanceSum : ℝ := 10

-- Define the equation of the ellipse
def isOnEllipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 25) + (P.2^2 / 16) = 1

-- Theorem statement
theorem ellipse_trajectory :
  ∀ P : ℝ × ℝ, 
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
  Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = distanceSum →
  isOnEllipse P :=
sorry

end ellipse_trajectory_l3729_372974


namespace total_students_suggestion_l3729_372991

theorem total_students_suggestion (mashed_potatoes bacon tomatoes : ℕ) 
  (h1 : mashed_potatoes = 324)
  (h2 : bacon = 374)
  (h3 : tomatoes = 128) :
  mashed_potatoes + bacon + tomatoes = 826 := by
  sorry

end total_students_suggestion_l3729_372991


namespace tan_theta_value_l3729_372962

theorem tan_theta_value (θ : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2)
  (h2 : (Real.sin θ + Real.cos θ)^2 + Real.sqrt 3 * Real.cos (2 * θ) = 3) :
  Real.tan θ = 2 - Real.sqrt 3 :=
by sorry

end tan_theta_value_l3729_372962


namespace miser_knight_theorem_l3729_372970

theorem miser_knight_theorem (N : ℕ) (h2 : ∀ (a b : ℕ), a + b = 2 → N % a = 0 ∧ N % b = 0)
  (h3 : ∀ (a b c : ℕ), a + b + c = 3 → N % a = 0 ∧ N % b = 0 ∧ N % c = 0)
  (h4 : ∀ (a b c d : ℕ), a + b + c + d = 4 → N % a = 0 ∧ N % b = 0 ∧ N % c = 0 ∧ N % d = 0)
  (h5 : ∀ (a b c d e : ℕ), a + b + c + d + e = 5 → N % a = 0 ∧ N % b = 0 ∧ N % c = 0 ∧ N % d = 0 ∧ N % e = 0) :
  N % 6 = 0 := by
sorry

end miser_knight_theorem_l3729_372970


namespace cube_volume_from_surface_area_l3729_372918

theorem cube_volume_from_surface_area (surface_area : ℝ) (h : surface_area = 600) :
  let side_length := Real.sqrt (surface_area / 6)
  side_length ^ 3 = 1000 := by
  sorry

end cube_volume_from_surface_area_l3729_372918


namespace negation_of_universal_proposition_l3729_372929

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end negation_of_universal_proposition_l3729_372929


namespace eulers_formula_l3729_372908

/-- A connected planar graph -/
structure PlanarGraph where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  is_connected : Prop
  is_planar : Prop

/-- Euler's formula for connected planar graphs -/
theorem eulers_formula (G : PlanarGraph) : G.V - G.E + G.F = 2 := by
  sorry

end eulers_formula_l3729_372908


namespace quadratic_inequality_solution_range_l3729_372907

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a - 1)*x + 1 > 0) → a ∈ Set.Ioo (-1 : ℝ) 3 := by
  sorry

end quadratic_inequality_solution_range_l3729_372907


namespace max_third_altitude_l3729_372935

/-- A triangle with two known altitudes and one unknown integer altitude -/
structure TriangleWithAltitudes where
  /-- The length of the first known altitude -/
  h₁ : ℝ
  /-- The length of the second known altitude -/
  h₂ : ℝ
  /-- The length of the unknown altitude (assumed to be an integer) -/
  h₃ : ℤ
  /-- Condition that h₁ and h₂ are 3 and 9 (in either order) -/
  known_altitudes : (h₁ = 3 ∧ h₂ = 9) ∨ (h₁ = 9 ∧ h₂ = 3)

/-- The theorem stating that the maximum possible integer length for h₃ is 4 -/
theorem max_third_altitude (t : TriangleWithAltitudes) : t.h₃ ≤ 4 := by
  sorry

end max_third_altitude_l3729_372935


namespace constant_sum_through_P_l3729_372976

/-- The function f(x) = x³ + 3x² + x -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + x

/-- The point P on the graph of f -/
def P : ℝ × ℝ := (-1, f (-1))

theorem constant_sum_through_P :
  ∃ (y : ℝ), ∀ (x₁ x₂ : ℝ),
    x₁ ≠ -1 → x₂ ≠ -1 →
    (x₂ - (-1)) * (f x₁ - f (-1)) = (x₁ - (-1)) * (f x₂ - f (-1)) →
    f x₁ + f x₂ = y :=
  sorry

end constant_sum_through_P_l3729_372976


namespace quadratic_negative_value_condition_l3729_372997

/-- Given a quadratic function f(x) = x^2 + mx + 1, 
    this theorem states that there exists a positive x₀ such that f(x₀) < 0 
    if and only if m < -2 -/
theorem quadratic_negative_value_condition (m : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + m*x₀ + 1 < 0) ↔ m < -2 :=
by sorry

end quadratic_negative_value_condition_l3729_372997


namespace conic_section_eccentricity_l3729_372989

/-- Given that real numbers 4, m, 9 form a geometric sequence,
    prove that the eccentricity of the conic section x^2/m + y^2 = 1
    is either √30/6 or √7 -/
theorem conic_section_eccentricity (m : ℝ) :
  (4 * m = m * 9) →
  let e := if m > 0
           then Real.sqrt (1 - m / 6) / Real.sqrt (m / 6)
           else Real.sqrt (1 + 6 / m) / 1
  (e = Real.sqrt 30 / 6 ∨ e = Real.sqrt 7) :=
by sorry

end conic_section_eccentricity_l3729_372989


namespace intersection_distance_sum_l3729_372964

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) - 4*ρ*(Real.sin θ) + 7 = 0

/-- Line C₂ in polar coordinates -/
def C₂ (θ : ℝ) : Prop :=
  Real.tan θ = Real.sqrt 3

/-- Theorem stating the sum of reciprocals of distances to intersection points -/
theorem intersection_distance_sum :
  ∀ ρ₁ ρ₂ θ : ℝ,
  C₁ ρ₁ θ → C₁ ρ₂ θ → C₂ θ →
  1 / ρ₁ + 1 / ρ₂ = (2 + 2 * Real.sqrt 3) / 7 := by
  sorry

end intersection_distance_sum_l3729_372964


namespace quadratic_no_real_roots_l3729_372995

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 - 2*x + 3 ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l3729_372995


namespace sqrt_x_minus_5_range_l3729_372993

theorem sqrt_x_minus_5_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by sorry

end sqrt_x_minus_5_range_l3729_372993


namespace lower_right_is_one_l3729_372911

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Checks if a number appears exactly once in each row -/
def unique_in_rows (g : Grid) : Prop :=
  ∀ i n, (∃! j, g i j = n) ∧ (1 ≤ n ∧ n ≤ 5)

/-- Checks if a number appears exactly once in each column -/
def unique_in_columns (g : Grid) : Prop :=
  ∀ j n, (∃! i, g i j = n) ∧ (1 ≤ n ∧ n ≤ 5)

/-- Initial grid configuration -/
def initial_grid : Grid :=
  fun i j =>
    if i = 0 ∧ j = 0 then 1
    else if i = 0 ∧ j = 2 then 2
    else if i = 1 ∧ j = 0 then 2
    else if i = 1 ∧ j = 1 then 4
    else if i = 2 ∧ j = 3 then 5
    else if i = 3 ∧ j = 1 then 5
    else 0  -- placeholder for empty cells

/-- The main theorem -/
theorem lower_right_is_one :
  ∀ g : Grid,
    (∀ i j, initial_grid i j ≠ 0 → g i j = initial_grid i j) →
    unique_in_rows g →
    unique_in_columns g →
    g 4 4 = 1 :=
by sorry

end lower_right_is_one_l3729_372911


namespace hat_number_problem_l3729_372966

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem hat_number_problem (alice_number bob_number : ℕ) : 
  alice_number ∈ Finset.range 50 →
  bob_number ∈ Finset.range 50 →
  alice_number ≠ bob_number →
  alice_number ≠ 1 →
  bob_number < alice_number →
  is_prime bob_number →
  bob_number < 10 →
  is_perfect_square (100 * bob_number + alice_number) →
  alice_number = 24 ∧ bob_number = 3 :=
by sorry

end hat_number_problem_l3729_372966


namespace complex_equation_solution_l3729_372936

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + 3 * Complex.I = (b + Complex.I) * Complex.I → a = -1 ∧ b = 3 :=
by sorry

end complex_equation_solution_l3729_372936


namespace flower_bed_fraction_is_4_75_l3729_372984

/-- Represents the dimensions of a rectangular yard -/
structure YardDimensions where
  length : ℝ
  width : ℝ

/-- Represents the parallel sides of the rectangular remainder -/
structure RemainderSides where
  side1 : ℝ
  side2 : ℝ

/-- Calculates the fraction of the yard occupied by flower beds -/
def flowerBedFraction (yard : YardDimensions) (remainder : RemainderSides) : ℚ :=
  sorry

/-- Theorem statement -/
theorem flower_bed_fraction_is_4_75 
  (yard : YardDimensions)
  (remainder : RemainderSides)
  (h1 : yard.length = 30)
  (h2 : yard.width = 10)
  (h3 : remainder.side1 = 30)
  (h4 : remainder.side2 = 22) :
  flowerBedFraction yard remainder = 4/75 := by
  sorry

end flower_bed_fraction_is_4_75_l3729_372984


namespace original_number_proof_l3729_372926

theorem original_number_proof :
  ∀ (original_number : ℤ),
    original_number + 3377 = 13200 →
    original_number = 9823 := by
  sorry

end original_number_proof_l3729_372926


namespace y_in_terms_of_x_l3729_372977

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : 
  y = x / (x - 1) := by
sorry

end y_in_terms_of_x_l3729_372977


namespace solution_set_l3729_372950

/-- A function f : ℝ → ℝ satisfying certain properties -/
axiom f : ℝ → ℝ

/-- The derivative of f -/
axiom f' : ℝ → ℝ

/-- f(x-1) is an odd function -/
axiom f_odd : ∀ x, f ((-x) - 1) = -f (x - 1)

/-- For x < -1, (x+1)[f(x) + (x+1)f'(x)] < 0 -/
axiom f_property : ∀ x, x < -1 → (x + 1) * (f x + (x + 1) * f' x) < 0

/-- The solution set for xf(x-1) > f(0) is (-1, 1) -/
theorem solution_set : 
  {x : ℝ | x * f (x - 1) > f 0} = Set.Ioo (-1) 1 := by sorry

end solution_set_l3729_372950


namespace min_value_sum_fractions_equality_condition_l3729_372951

theorem min_value_sum_fractions (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a ≥ 12 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a = 12 ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end min_value_sum_fractions_equality_condition_l3729_372951


namespace largest_integer_b_for_all_real_domain_l3729_372923

theorem largest_integer_b_for_all_real_domain : 
  ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 12 ≠ 0) → b ≤ 6 :=
by
  sorry

end largest_integer_b_for_all_real_domain_l3729_372923


namespace johns_car_trade_in_value_l3729_372912

/-- Calculates the trade-in value of John's car based on his Uber earnings, initial car purchase price, and profit. -/
def trade_in_value (uber_earnings profit initial_car_price : ℕ) : ℕ :=
  initial_car_price - (uber_earnings - profit)

/-- Theorem stating that John's car trade-in value is $6,000 given the provided conditions. -/
theorem johns_car_trade_in_value :
  trade_in_value 30000 18000 18000 = 6000 := by
  sorry

end johns_car_trade_in_value_l3729_372912


namespace range_of_c_l3729_372946

-- Define the triangular pyramid
structure TriangularPyramid where
  -- Base edges
  base_edge1 : ℝ
  base_edge2 : ℝ
  base_edge3 : ℝ
  -- Side edges opposite to base edges
  side_edge1 : ℝ
  side_edge2 : ℝ
  side_edge3 : ℝ

-- Define the specific triangular pyramid from the problem
def specificPyramid (c : ℝ) : TriangularPyramid :=
  { base_edge1 := 1
  , base_edge2 := 1
  , base_edge3 := c
  , side_edge1 := 1
  , side_edge2 := c
  , side_edge3 := c }

-- Theorem stating the range of c
theorem range_of_c :
  ∀ c : ℝ, (∃ p : TriangularPyramid, p = specificPyramid c) →
  (Real.sqrt 5 - 1) / 2 < c ∧ c < (Real.sqrt 5 + 1) / 2 :=
by sorry

end range_of_c_l3729_372946


namespace factor_and_divisor_relations_l3729_372944

theorem factor_and_divisor_relations : 
  (∃ n : ℤ, 45 = 5 * n) ∧ 
  (209 % 19 = 0 ∧ 95 % 19 = 0) ∧ 
  (∃ m : ℤ, 180 = 9 * m) := by
sorry


end factor_and_divisor_relations_l3729_372944


namespace customers_remaining_l3729_372948

theorem customers_remaining (initial : ℕ) (difference : ℕ) (final : ℕ) : 
  initial = 19 → difference = 15 → final = initial - difference → final = 4 := by
  sorry

end customers_remaining_l3729_372948


namespace profit_is_24000_l3729_372973

def initial_value : ℝ := 150000
def depreciation_rate : ℝ := 0.22
def selling_price : ℝ := 115260
def years : ℕ := 2

def value_after_years (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 - rate) ^ years

def profit (initial : ℝ) (rate : ℝ) (years : ℕ) (selling_price : ℝ) : ℝ :=
  selling_price - value_after_years initial rate years

theorem profit_is_24000 :
  profit initial_value depreciation_rate years selling_price = 24000 := by
  sorry

end profit_is_24000_l3729_372973


namespace solution_to_system_of_equations_l3729_372961

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ), 
    (x^2 * y - x * y^2 - 3*x + 3*y + 1 = 0) ∧
    (x^3 * y - x * y^3 - 3*x^2 + 3*y^2 + 3 = 0) ∧
    (x = 2 ∧ y = 1) := by
  sorry

end solution_to_system_of_equations_l3729_372961


namespace combination_sequence_implies_value_l3729_372916

theorem combination_sequence_implies_value (n : ℕ) : 
  (2 * (n.choose 5) = (n.choose 4) + (n.choose 6)) → 
  (n.choose 12) = 91 := by sorry

end combination_sequence_implies_value_l3729_372916


namespace ratio_evaluation_l3729_372955

theorem ratio_evaluation : (2^121 * 3^123) / 6^122 = 3/2 := by
  sorry

end ratio_evaluation_l3729_372955


namespace smallest_consecutive_non_primes_l3729_372980

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_non_primes (start : ℕ) : Prop :=
  ∀ i : ℕ, i < 5 → ¬(is_prime (start + i))

theorem smallest_consecutive_non_primes :
  ∃ (n : ℕ), n > 90 ∧ n < 96 ∧ consecutive_non_primes n ∧
  ∀ m : ℕ, m > 90 ∧ m < 96 ∧ consecutive_non_primes m → n ≤ m :=
by sorry

end smallest_consecutive_non_primes_l3729_372980


namespace intercept_sum_lower_bound_l3729_372920

/-- A line passing through (1,3) intersecting positive x and y axes -/
structure InterceptLine where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  passes_through_P : 1 / a + 3 / b = 1

/-- The sum of intercepts is at least 4 + 2√3 -/
theorem intercept_sum_lower_bound (l : InterceptLine) : l.a + l.b ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

#check intercept_sum_lower_bound

end intercept_sum_lower_bound_l3729_372920


namespace abes_age_problem_l3729_372939

/-- Abe's age problem -/
theorem abes_age_problem (present_age : ℕ) (x : ℕ) 
  (h1 : present_age = 28)
  (h2 : present_age + (present_age - x) = 35) :
  present_age + x = 49 := by
  sorry

end abes_age_problem_l3729_372939


namespace car_speed_ratio_l3729_372914

theorem car_speed_ratio : 
  ∀ (speed_A speed_B : ℝ),
    speed_B = 50 →
    speed_A * 6 + speed_B * 2 = 1000 →
    speed_A / speed_B = 3 := by
  sorry

end car_speed_ratio_l3729_372914


namespace root_inequality_l3729_372965

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) := exp x + x - 2
def g (x : ℝ) := log x + x - 2

-- State the theorem
theorem root_inequality (a b : ℝ) (ha : f a = 0) (hb : g b = 0) : f a < f 1 ∧ f 1 < f b := by
  sorry

end

end root_inequality_l3729_372965


namespace trees_survived_vs_died_l3729_372988

theorem trees_survived_vs_died (initial_trees dead_trees : ℕ) : 
  initial_trees = 11 → dead_trees = 2 → 
  (initial_trees - dead_trees) - dead_trees = 7 := by
  sorry

end trees_survived_vs_died_l3729_372988


namespace circular_seating_arrangements_l3729_372999

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem circular_seating_arrangements :
  let total_people : ℕ := 8
  let seats : ℕ := 7
  let reserved_seats : ℕ := 1
  let remaining_people : ℕ := total_people - reserved_seats
  let people_to_arrange : ℕ := seats - reserved_seats
  
  (choose remaining_people people_to_arrange * factorial people_to_arrange) / seats = 720 := by
sorry

end circular_seating_arrangements_l3729_372999


namespace original_celery_cost_l3729_372937

def original_order : ℝ := 25
def new_tomatoes : ℝ := 2.20
def old_tomatoes : ℝ := 0.99
def new_lettuce : ℝ := 1.75
def old_lettuce : ℝ := 1.00
def new_celery : ℝ := 2.00
def delivery_tip : ℝ := 8.00
def new_total : ℝ := 35

theorem original_celery_cost :
  ∃ (old_celery : ℝ),
    old_celery = 0.04 ∧
    original_order = old_tomatoes + old_lettuce + old_celery ∧
    new_total = new_tomatoes + new_lettuce + new_celery + delivery_tip :=
by sorry

end original_celery_cost_l3729_372937


namespace y_intercept_of_line_a_l3729_372900

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- The given line y = 2x + 4 -/
def given_line : Line :=
  { slope := 2, point := (0, 4) }

/-- Line a, which is parallel to the given line and passes through (2, 5) -/
def line_a : Line :=
  { slope := given_line.slope, point := (2, 5) }

theorem y_intercept_of_line_a :
  y_intercept line_a = 1 := by
  sorry

end y_intercept_of_line_a_l3729_372900


namespace lawn_length_l3729_372941

/-- Given a rectangular lawn with specified conditions, prove its length is 70 meters -/
theorem lawn_length (width : ℝ) (road_width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) : 
  width = 60 → 
  road_width = 10 → 
  total_cost = 3600 → 
  cost_per_sqm = 3 → 
  ∃ (length : ℝ), 
    (road_width * length + road_width * (width - road_width)) * cost_per_sqm = total_cost ∧ 
    length = 70 := by
  sorry

end lawn_length_l3729_372941


namespace cone_base_radius_l3729_372902

/-- Given a cone whose lateral surface is formed by a sector with radius 6cm and central angle 120°,
    the radius of the base of the cone is 2cm. -/
theorem cone_base_radius (r : ℝ) : r > 0 → 2 * π * r = 120 * π * 6 / 180 → r = 2 := by
  sorry

end cone_base_radius_l3729_372902


namespace wendy_first_day_miles_l3729_372985

theorem wendy_first_day_miles (total_miles second_day_miles third_day_miles : ℕ) 
  (h1 : total_miles = 493)
  (h2 : second_day_miles = 223)
  (h3 : third_day_miles = 145) :
  total_miles - (second_day_miles + third_day_miles) = 125 := by
  sorry

end wendy_first_day_miles_l3729_372985


namespace a_share_is_3690_l3729_372998

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem stating that A's share of the profit is 3690 given the specified investments and total profit. -/
theorem a_share_is_3690 :
  calculate_share_of_profit 6300 4200 10500 12300 = 3690 := by
  sorry

end a_share_is_3690_l3729_372998


namespace max_guaranteed_sum_l3729_372905

/-- A strategy for placing signs in the game -/
def Strategy := Fin 20 → Bool

/-- The set of numbers used in the game -/
def GameNumbers : Finset ℕ := Finset.range 20

/-- Calculate the sum given a strategy -/
def calculateSum (s : Strategy) : ℤ :=
  (Finset.sum GameNumbers fun i => if s i then (i + 1) else -(i + 1))

/-- Player B's objective is to maximize the absolute value of the sum -/
def playerBObjective (s : Strategy) : ℕ := Int.natAbs (calculateSum s)

/-- The theorem stating the maximum value Player B can guarantee -/
theorem max_guaranteed_sum :
  ∃ (s : Strategy), ∀ (t : Strategy), playerBObjective s ≥ 30 :=
sorry

end max_guaranteed_sum_l3729_372905


namespace boat_speed_in_still_water_l3729_372979

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 189)
  (h3 : downstream_time = 7)
  : ∃ (boat_speed : ℝ), boat_speed = 22 ∧ 
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end boat_speed_in_still_water_l3729_372979


namespace range_of_a_l3729_372956

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (Set.compl A ∪ B a = U) ↔ a < 1 :=
by sorry

end range_of_a_l3729_372956


namespace largest_expression_l3729_372930

theorem largest_expression : 
  let a := 1 - 2 + 3 + 4
  let b := 1 + 2 - 3 + 4
  let c := 1 + 2 + 3 - 4
  let d := 1 + 2 - 3 - 4
  let e := 1 - 2 - 3 + 4
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) := by
  sorry

#eval (1 - 2 + 3 + 4)
#eval (1 + 2 - 3 + 4)
#eval (1 + 2 + 3 - 4)
#eval (1 + 2 - 3 - 4)
#eval (1 - 2 - 3 + 4)

end largest_expression_l3729_372930


namespace difference_second_third_bus_l3729_372924

/-- The number of buses hired for the school trip -/
def num_buses : ℕ := 4

/-- The number of people on the first bus -/
def first_bus : ℕ := 12

/-- The number of people on the second bus -/
def second_bus : ℕ := 2 * first_bus

/-- The number of people on the fourth bus -/
def fourth_bus : ℕ := first_bus + 9

/-- The total number of people going to the museum -/
def total_people : ℕ := 75

/-- The number of people on the third bus -/
def third_bus : ℕ := total_people - (first_bus + second_bus + fourth_bus)

theorem difference_second_third_bus : second_bus - third_bus = 6 := by
  sorry

end difference_second_third_bus_l3729_372924


namespace cos_105_cos_45_plus_sin_105_sin_45_l3729_372967

theorem cos_105_cos_45_plus_sin_105_sin_45 :
  Real.cos (105 * π / 180) * Real.cos (45 * π / 180) +
  Real.sin (105 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end cos_105_cos_45_plus_sin_105_sin_45_l3729_372967


namespace basketball_team_starters_l3729_372981

theorem basketball_team_starters : Nat.choose 16 8 = 12870 := by
  sorry

end basketball_team_starters_l3729_372981


namespace light_bulb_probability_l3729_372942

/-- The number of screw-in light bulbs in the box -/
def screwIn : ℕ := 3

/-- The number of bayonet light bulbs in the box -/
def bayonet : ℕ := 5

/-- The total number of light bulbs in the box -/
def totalBulbs : ℕ := screwIn + bayonet

/-- The number of draws -/
def draws : ℕ := 5

/-- The probability of drawing all screw-in light bulbs by the 5th draw -/
def probability : ℚ := 3 / 28

theorem light_bulb_probability :
  probability = (Nat.choose screwIn (screwIn - 1) * Nat.choose bayonet (draws - screwIn) * Nat.factorial (draws - 1)) /
                (Nat.choose totalBulbs draws * Nat.factorial draws) :=
sorry

end light_bulb_probability_l3729_372942


namespace flashlight_problem_l3729_372938

/-- Represents the minimum number of attempts needed to guarantee a flashlight lights up -/
def min_attempts (total_batteries : ℕ) (good_batteries : ℕ) : ℕ :=
  if total_batteries = 2 * good_batteries - 1 
  then good_batteries + 1
  else if total_batteries = 2 * good_batteries 
  then good_batteries + 3
  else 0  -- undefined for other cases

/-- Theorem for the flashlight problem -/
theorem flashlight_problem (n : ℕ) (h : n > 2) :
  (min_attempts (2 * n + 1) (n + 1) = n + 2) ∧
  (min_attempts (2 * n) n = n + 3) := by
  sorry

#check flashlight_problem

end flashlight_problem_l3729_372938


namespace twenty_five_percent_less_than_eighty_l3729_372982

theorem twenty_five_percent_less_than_eighty (x : ℝ) : x = 40 ↔ 80 - 0.25 * 80 = x + 0.5 * x := by
  sorry

end twenty_five_percent_less_than_eighty_l3729_372982


namespace triangle_side_range_l3729_372990

theorem triangle_side_range (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- acute triangle
  A + B + C = π ∧ -- sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧ -- positive sides
  (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C ∧ -- given equation
  a = Real.sqrt 3 -- given value of a
  →
  5 < b^2 + c^2 ∧ b^2 + c^2 ≤ 6 := by sorry

end triangle_side_range_l3729_372990


namespace equivalence_of_statements_l3729_372986

-- Define the propositions
variable (P Q : Prop)

-- Define the original implication
def original_statement : Prop := P → Q

-- Define the contrapositive
def contrapositive : Prop := ¬Q → ¬P

-- Define the disjunction form
def disjunction_form : Prop := ¬P ∨ Q

-- Theorem stating the equivalence of the three forms
theorem equivalence_of_statements :
  (original_statement P Q) ↔ (contrapositive P Q) ∧ (disjunction_form P Q) :=
sorry

end equivalence_of_statements_l3729_372986


namespace teacher_assignment_schemes_l3729_372922

theorem teacher_assignment_schemes (male_teachers : Nat) (female_teachers : Nat) : 
  male_teachers = 5 → 
  female_teachers = 4 → 
  (Nat.factorial 9 / Nat.factorial 6) - 
  (Nat.factorial 5 / Nat.factorial 2 + Nat.factorial 4 / Nat.factorial 1) = 420 := by
  sorry

end teacher_assignment_schemes_l3729_372922


namespace distance_to_line_rational_l3729_372953

/-- The distance from any lattice point to the line 3x - 4y + 4 = 0 is rational -/
theorem distance_to_line_rational (a b : ℤ) : ∃ (q : ℚ), q = |4 * b - 3 * a - 4| / 5 := by
  sorry

end distance_to_line_rational_l3729_372953


namespace initial_order_size_l3729_372987

/-- The number of cogs produced per hour in the initial phase -/
def initial_rate : ℕ := 36

/-- The number of cogs produced per hour in the second phase -/
def second_rate : ℕ := 60

/-- The number of additional cogs produced in the second phase -/
def additional_cogs : ℕ := 60

/-- The overall average output in cogs per hour -/
def average_output : ℝ := 45

/-- The theorem stating that the initial order was for 60 cogs -/
theorem initial_order_size :
  ∃ x : ℕ, 
    (x + additional_cogs) / (x / initial_rate + 1 : ℝ) = average_output →
    x = 60 := by
  sorry

end initial_order_size_l3729_372987


namespace infinite_geometric_series_sum_l3729_372931

/-- The sum of the infinite geometric series 1 - (1/4) + (1/4)^2 - (1/4)^3 + ... -/
def infiniteGeometricSeriesSum : ℚ := 4/5

/-- The first term of the series -/
def a : ℚ := 1

/-- The common ratio of the series -/
def r : ℚ := -1/4

/-- Theorem: The sum of the infinite geometric series 1 - (1/4) + (1/4)^2 - (1/4)^3 + ... is 4/5 -/
theorem infinite_geometric_series_sum :
  infiniteGeometricSeriesSum = a / (1 - r) :=
by sorry

end infinite_geometric_series_sum_l3729_372931


namespace simplify_fraction_l3729_372940

theorem simplify_fraction (x : ℝ) (hx : x ≠ 0) : (x + 1) / x - 1 / x = 1 := by
  sorry

end simplify_fraction_l3729_372940


namespace local_max_implies_neg_local_min_l3729_372959

open Function Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define x₀ as a non-zero real number
variable (x₀ : ℝ)
variable (hx₀ : x₀ ≠ 0)

-- Define that x₀ is a local maximum point of f
def IsLocalMaxAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀

-- Define local minimum
def IsLocalMinAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x₀ ≤ f x

-- State the theorem
theorem local_max_implies_neg_local_min
  (h : IsLocalMaxAt f x₀) :
  IsLocalMinAt (fun x ↦ -f (-x)) (-x₀) :=
sorry

end local_max_implies_neg_local_min_l3729_372959


namespace polynomial_division_theorem_l3729_372972

theorem polynomial_division_theorem (x : ℝ) :
  x^5 + 8 = (x + 2) * (x^4 - 2*x^3 + 4*x^2 - 8*x + 16) + (-24) := by
  sorry

end polynomial_division_theorem_l3729_372972


namespace iodine131_electrons_l3729_372975

structure Atom where
  atomicMass : ℕ
  protonNumber : ℕ

def numberOfNeutrons (a : Atom) : ℕ := a.atomicMass - a.protonNumber

def numberOfElectrons (a : Atom) : ℕ := a.protonNumber

def iodine131 : Atom := ⟨131, 53⟩

theorem iodine131_electrons : numberOfElectrons iodine131 = 53 := by
  sorry

end iodine131_electrons_l3729_372975


namespace least_number_to_add_l3729_372927

theorem least_number_to_add (n : ℕ) : 
  (∀ m : ℕ, m < 234 → ¬(5 ∣ (2496 + m) ∧ 7 ∣ (2496 + m) ∧ 13 ∣ (2496 + m))) ∧ 
  (5 ∣ (2496 + 234) ∧ 7 ∣ (2496 + 234) ∧ 13 ∣ (2496 + 234)) := by
  sorry

end least_number_to_add_l3729_372927


namespace flour_needed_l3729_372994

theorem flour_needed (total : ℝ) (added : ℝ) (needed : ℝ) :
  total = 8.5 ∧ added = 2.25 ∧ needed = total - added → needed = 6.25 := by
  sorry

end flour_needed_l3729_372994


namespace girls_in_study_group_l3729_372903

theorem girls_in_study_group (n : ℕ) :
  (Nat.choose 6 2 - Nat.choose (6 - n) 2 = 12) →
  n = 3 :=
by sorry

end girls_in_study_group_l3729_372903


namespace worker_number_40th_segment_l3729_372934

/-- Calculates the individual number of a worker in systematic sampling -/
def systematicSamplingNumber (totalStaff : ℕ) (segments : ℕ) (startNumber : ℕ) (segmentIndex : ℕ) : ℕ :=
  startNumber + (segmentIndex - 1) * (totalStaff / segments)

/-- Proves that the individual number of the worker from the 40th segment is 394 -/
theorem worker_number_40th_segment :
  systematicSamplingNumber 620 62 4 40 = 394 := by
  sorry

#eval systematicSamplingNumber 620 62 4 40

end worker_number_40th_segment_l3729_372934


namespace runner_picture_probability_l3729_372960

/-- Rachel's lap time in seconds -/
def rachel_lap_time : ℕ := 100

/-- Robert's lap time in seconds -/
def robert_lap_time : ℕ := 70

/-- Duration of the observation period in seconds -/
def observation_period : ℕ := 60

/-- Fraction of the track captured in the picture -/
def picture_fraction : ℚ := 1/5

/-- Time when the picture is taken (in seconds after start) -/
def picture_time : ℕ := 720  -- 12 minutes

theorem runner_picture_probability :
  let rachel_position := picture_time % rachel_lap_time
  let robert_position := robert_lap_time - (picture_time % robert_lap_time)
  let rachel_in_picture := rachel_position ≤ (rachel_lap_time * picture_fraction / 2) ∨
                           rachel_position ≥ rachel_lap_time - (rachel_lap_time * picture_fraction / 2)
  let robert_in_picture := robert_position ≤ (robert_lap_time * picture_fraction / 2) ∨
                           robert_position ≥ robert_lap_time - (robert_lap_time * picture_fraction / 2)
  (∃ t : ℕ, t ≥ picture_time ∧ t < picture_time + observation_period ∧
            rachel_in_picture ∧ robert_in_picture) →
  (1 : ℚ) / 16 = ↑(Nat.card {t : ℕ | t ≥ picture_time ∧ t < picture_time + observation_period ∧
                              rachel_in_picture ∧ robert_in_picture}) / observation_period :=
by sorry

end runner_picture_probability_l3729_372960


namespace triangle_mn_length_l3729_372957

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let AC := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let BC := Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)
  AB = 5 ∧ AC = 4 ∧ BC = 6

-- Define the angle bisector and point X
def angleBisector (t : Triangle) : ℝ × ℝ → Prop := sorry

-- Define points M and N
def pointM (t : Triangle) : ℝ × ℝ := sorry
def pointN (t : Triangle) : ℝ × ℝ := sorry

-- Define parallel lines
def isParallel (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

theorem triangle_mn_length (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : ∃ X, angleBisector t X ∧ X.1 ∈ Set.Icc t.A.1 t.B.1 ∧ X.2 = t.A.2)
  (h3 : isParallel (X, pointM t) (t.A, t.C))
  (h4 : isParallel (X, pointN t) (t.B, t.C)) :
  let MN := Real.sqrt ((pointM t).1 - (pointN t).1)^2 + ((pointM t).2 - (pointN t).2)^2
  MN = 3 * Real.sqrt 14 / 5 := by
  sorry

end triangle_mn_length_l3729_372957


namespace gcd_equals_2023_l3729_372901

theorem gcd_equals_2023 (a b c : ℕ+) 
  (h : Nat.gcd a b + Nat.gcd a c + Nat.gcd b c = b + c + 2023) : 
  Nat.gcd b c = 2023 := by
  sorry

end gcd_equals_2023_l3729_372901


namespace broken_tree_height_l3729_372969

/-- 
Given a tree that broke and fell across a road, this theorem proves that 
if the breadth of the road is 12 m and the tree broke at a height of 16 m, 
then the original height of the tree is 36 m.
-/
theorem broken_tree_height 
  (breadth : ℝ) 
  (broken_height : ℝ) 
  (h_breadth : breadth = 12) 
  (h_broken : broken_height = 16) : 
  ∃ (original_height : ℝ), original_height = 36 := by
  sorry

end broken_tree_height_l3729_372969


namespace vector_on_line_l3729_372983

/-- Given distinct vectors a and b in a vector space V over ℝ,
    prove that the vector (1/2)a + (1/2)b lies on the line passing through a and b. -/
theorem vector_on_line {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
  (a b : V) (h : a ≠ b) :
  ∃ t : ℝ, (1/2 : ℝ) • a + (1/2 : ℝ) • b = a + t • (b - a) :=
sorry

end vector_on_line_l3729_372983


namespace two_consecutive_count_l3729_372952

/-- Represents the number of balls in the box -/
def n : ℕ := 5

/-- Represents the number of people drawing balls -/
def k : ℕ := 3

/-- Counts the number of ways to draw balls with exactly two consecutive numbers -/
def count_two_consecutive (n k : ℕ) : ℕ :=
  sorry

theorem two_consecutive_count :
  count_two_consecutive n k = 36 := by
  sorry

end two_consecutive_count_l3729_372952


namespace four_digit_int_problem_l3729_372906

/-- Represents a four-digit positive integer -/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : a > 0
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

/-- Converts a FourDigitInt to a natural number -/
def FourDigitInt.toNat (n : FourDigitInt) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

theorem four_digit_int_problem (n : FourDigitInt) 
  (h1 : n.a + n.b + n.c + n.d = 18)
  (h2 : n.b + n.c = 11)
  (h3 : n.a - n.d = 3)
  (h4 : n.toNat % 9 = 0) :
  n.toNat = 5472 := by
  sorry

end four_digit_int_problem_l3729_372906


namespace min_value_quadratic_form_l3729_372958

theorem min_value_quadratic_form (x y : ℤ) (h : (x, y) ≠ (0, 0)) :
  |5 * x^2 + 11 * x * y - 5 * y^2| ≥ 5 := by
  sorry

end min_value_quadratic_form_l3729_372958


namespace regular_decagon_exterior_angle_regular_decagon_exterior_angle_is_36_l3729_372919

/-- The exterior angle of a regular decagon is 36 degrees. -/
theorem regular_decagon_exterior_angle : ℝ :=
  let n : ℕ := 10  -- number of sides in a decagon
  let interior_angle_sum : ℝ := 180 * (n - 2)
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- Proof that the exterior angle of a regular decagon is 36 degrees. -/
theorem regular_decagon_exterior_angle_is_36 : 
  regular_decagon_exterior_angle = 36 := by
  sorry

end regular_decagon_exterior_angle_regular_decagon_exterior_angle_is_36_l3729_372919
