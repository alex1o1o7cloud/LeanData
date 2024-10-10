import Mathlib

namespace truck_travel_distance_l3300_330070

/-- Given a truck that travels 300 kilometers on 5 liters of diesel,
    prove that it can travel 420 kilometers on 7 liters of diesel. -/
theorem truck_travel_distance (initial_distance : ℝ) (initial_fuel : ℝ) (new_fuel : ℝ) :
  initial_distance = 300 ∧ initial_fuel = 5 ∧ new_fuel = 7 →
  (initial_distance / initial_fuel) * new_fuel = 420 := by
  sorry

end truck_travel_distance_l3300_330070


namespace magazine_cost_l3300_330082

theorem magazine_cost (total_books : ℕ) (book_cost : ℕ) (total_magazines : ℕ) (total_spent : ℕ) :
  total_books = 10 →
  book_cost = 15 →
  total_magazines = 10 →
  total_spent = 170 →
  ∃ (magazine_cost : ℕ), magazine_cost = 2 ∧ total_spent = total_books * book_cost + total_magazines * magazine_cost :=
by
  sorry

end magazine_cost_l3300_330082


namespace condition_analysis_l3300_330080

theorem condition_analysis (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * b^2 > 0 → a > b) ∧
  (∃ a b : ℝ, a > b ∧ (a - b) * b^2 ≤ 0) := by
  sorry

end condition_analysis_l3300_330080


namespace lines_are_parallel_l3300_330043

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem lines_are_parallel : 
  let line1 : Line := ⟨3, 1, 1⟩
  let line2 : Line := ⟨6, 2, 1⟩
  parallel line1 line2 := by
  sorry

end lines_are_parallel_l3300_330043


namespace smallest_n_for_integer_roots_l3300_330074

def n : ℕ := 2^5 * 3^5 * 5^4 * 7^6

theorem smallest_n_for_integer_roots :
  (∃ (a b c : ℕ), (5 * n = a^5) ∧ (6 * n = b^6) ∧ (7 * n = c^7)) ∧
  (∀ m : ℕ, m < n → ¬(∃ (x y z : ℕ), (5 * m = x^5) ∧ (6 * m = y^6) ∧ (7 * m = z^7))) :=
by sorry

end smallest_n_for_integer_roots_l3300_330074


namespace tessellation_theorem_l3300_330075

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  interiorAngle : ℝ

/-- Checks if two regular polygons can tessellate -/
def canTessellate (p1 p2 : RegularPolygon) : Prop :=
  ∃ (n1 n2 : ℕ), n1 * p1.interiorAngle + n2 * p2.interiorAngle = 360

theorem tessellation_theorem :
  let triangle : RegularPolygon := ⟨3, 60⟩
  let square : RegularPolygon := ⟨4, 90⟩
  let hexagon : RegularPolygon := ⟨6, 120⟩
  let octagon : RegularPolygon := ⟨8, 135⟩

  (canTessellate triangle square) ∧
  (canTessellate triangle hexagon) ∧
  (canTessellate octagon square) ∧
  ¬(canTessellate hexagon square) :=
by sorry

end tessellation_theorem_l3300_330075


namespace second_question_percentage_l3300_330000

theorem second_question_percentage
  (first_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : first_correct = 75)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 65)
  : ∃ (second_correct : ℝ), second_correct = 70 :=
by
  sorry

end second_question_percentage_l3300_330000


namespace monotonicity_of_g_no_solutions_for_equation_l3300_330093

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a
def g (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) / (x + 1)

theorem monotonicity_of_g :
  a = 1 →
  (∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 - 1 → g a x₁ < g a x₂) ∧
  (∀ x₁ x₂, Real.exp 1 - 1 < x₁ ∧ x₁ < x₂ → g a x₁ > g a x₂) :=
sorry

theorem no_solutions_for_equation :
  0 < a → a < 2/3 → ∀ x, f a x ≠ (x + 1) * g a x :=
sorry

end monotonicity_of_g_no_solutions_for_equation_l3300_330093


namespace twelve_factorial_base_nine_zeroes_l3300_330050

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 12! ends with 2 zeroes when written in base 9 -/
theorem twelve_factorial_base_nine_zeroes : trailingZeroes 12 9 = 2 := by sorry

end twelve_factorial_base_nine_zeroes_l3300_330050


namespace max_value_complex_expression_l3300_330086

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs w = 1 → Complex.abs (w^3 - 3*w - 2) ≤ max_val :=
sorry

end max_value_complex_expression_l3300_330086


namespace betty_herb_garden_l3300_330011

/-- The number of basil plants in Betty's herb garden -/
def basil : ℕ := 5

/-- The number of oregano plants in Betty's herb garden -/
def oregano : ℕ := 2 * basil + 2

/-- The total number of plants in Betty's herb garden -/
def total_plants : ℕ := basil + oregano

theorem betty_herb_garden :
  total_plants = 17 := by sorry

end betty_herb_garden_l3300_330011


namespace ding_xiaole_jogging_distances_l3300_330048

/-- Represents the jogging distances for 4 days -/
structure JoggingData :=
  (days : Nat)
  (max_daily : ℝ)
  (min_daily : ℝ)

/-- Calculates the maximum total distance for the given jogging data -/
def max_total_distance (data : JoggingData) : ℝ :=
  data.max_daily * (data.days - 1) + data.min_daily

/-- Calculates the minimum total distance for the given jogging data -/
def min_total_distance (data : JoggingData) : ℝ :=
  data.min_daily * (data.days - 1) + data.max_daily

/-- Theorem stating the maximum and minimum total distances for Ding Xiaole's jogging -/
theorem ding_xiaole_jogging_distances :
  let data : JoggingData := ⟨4, 3.3, 2.4⟩
  max_total_distance data = 12.3 ∧ min_total_distance data = 10.5 := by
  sorry

end ding_xiaole_jogging_distances_l3300_330048


namespace empty_intersection_l3300_330041

def S : Set ℚ := {x : ℚ | x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1}

def f (x : ℚ) : ℚ := x - 1/x

def f_iter (n : ℕ) : Set ℚ → Set ℚ :=
  match n with
  | 0 => id
  | n + 1 => f_iter n ∘ (λ s => f '' s)

theorem empty_intersection :
  (⋂ n : ℕ, f_iter n S) = ∅ := by sorry

end empty_intersection_l3300_330041


namespace product_from_lcm_hcf_l3300_330020

theorem product_from_lcm_hcf (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750) 
  (h_hcf : Nat.gcd a b = 25) : 
  a * b = 18750 := by
  sorry

end product_from_lcm_hcf_l3300_330020


namespace tank_filling_l3300_330044

/-- Proves that adding 4 gallons to a 32-gallon tank that is 3/4 full results in the tank being 7/8 full -/
theorem tank_filling (tank_capacity : ℚ) (initial_fraction : ℚ) (added_amount : ℚ) : 
  tank_capacity = 32 →
  initial_fraction = 3 / 4 →
  added_amount = 4 →
  (initial_fraction * tank_capacity + added_amount) / tank_capacity = 7 / 8 := by
  sorry

end tank_filling_l3300_330044


namespace diamond_commutative_l3300_330034

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^2 - b^2

-- Theorem statement
theorem diamond_commutative : ∀ x y : ℝ, diamond x y = diamond y x := by
  sorry

end diamond_commutative_l3300_330034


namespace simplify_square_roots_l3300_330029

theorem simplify_square_roots : 
  Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^4) = 225 * Real.sqrt 5 := by
  sorry

end simplify_square_roots_l3300_330029


namespace point_coordinates_on_horizontal_line_l3300_330065

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line parallel to the x-axis -/
structure HorizontalLine where
  y : ℝ

def Point.liesOn (p : Point) (l : HorizontalLine) : Prop :=
  p.y = l.y

theorem point_coordinates_on_horizontal_line 
  (m : ℝ)
  (P : Point)
  (A : Point)
  (l : HorizontalLine)
  (h1 : P = ⟨2*m + 4, m - 1⟩)
  (h2 : A = ⟨2, -4⟩)
  (h3 : l.y = A.y)
  (h4 : P.liesOn l) :
  P = ⟨-2, -4⟩ :=
sorry

end point_coordinates_on_horizontal_line_l3300_330065


namespace zeros_in_quotient_l3300_330030

/-- S_k represents the k-length sequence of twos in its decimal presentation -/
def S (k : ℕ) : ℕ := (2 * (10^k - 1)) / 9

/-- The quotient of S_30 divided by S_5 -/
def Q : ℕ := S 30 / S 5

/-- The number of zeros in the decimal representation of Q -/
def num_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_quotient : num_zeros Q = 20 := by sorry

end zeros_in_quotient_l3300_330030


namespace complement_determines_set_l3300_330071

def U : Set ℕ := {0, 1, 2, 3}

theorem complement_determines_set (M : Set ℕ) (h : Set.compl M = {2}) : M = {0, 1, 3} := by
  sorry

end complement_determines_set_l3300_330071


namespace quadratic_equation_real_roots_l3300_330019

theorem quadratic_equation_real_roots (m n : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁^2 - (m + n)*x₁ + m*n = 0 ∧ x₂^2 - (m + n)*x₂ + m*n = 0 := by
  sorry

end quadratic_equation_real_roots_l3300_330019


namespace circle_area_difference_l3300_330076

theorem circle_area_difference (r₁ r₂ d : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 15) (h₃ : d = 8) 
  (h₄ : d = r₁ + r₂) : π * r₂^2 - π * r₁^2 = 200 * π := by
  sorry

end circle_area_difference_l3300_330076


namespace min_shift_for_symmetry_l3300_330005

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

theorem min_shift_for_symmetry :
  let g (φ : ℝ) (x : ℝ) := f (x - φ)
  ∃ (φ : ℝ), φ > 0 ∧
    (∀ x, g φ (π/6 + x) = g φ (π/6 - x)) ∧
    (∀ ψ, ψ > 0 ∧ (∀ x, g ψ (π/6 + x) = g ψ (π/6 - x)) → ψ ≥ φ) ∧
    φ = 5*π/12 :=
sorry

end min_shift_for_symmetry_l3300_330005


namespace rationalize_denominator_l3300_330072

theorem rationalize_denominator :
  ∃ (A B C : ℕ) (D : ℕ+),
    (1 : ℝ) / (Real.rpow 5 (1/3) - Real.rpow 4 (1/3)) =
    (Real.rpow A (1/3) + Real.rpow B (1/3) + Real.rpow C (1/3)) / D ∧
    A = 25 ∧ B = 20 ∧ C = 16 ∧ D = 1 := by
  sorry

end rationalize_denominator_l3300_330072


namespace plan_C_not_more_expensive_l3300_330078

/-- Represents the number of days required for Team A to complete the project alone -/
def x : ℕ := sorry

/-- Cost per day for Team A -/
def cost_A : ℕ := 10000

/-- Cost per day for Team B -/
def cost_B : ℕ := 6000

/-- Number of days both teams work together in Plan C -/
def days_together : ℕ := 3

/-- Extra days required for Team B to complete the project alone -/
def extra_days_B : ℕ := 4

/-- Equation representing the work done in Plan C -/
axiom plan_C_equation : (days_together : ℝ) / x + x / (x + extra_days_B) = 1

/-- Cost of Plan A -/
def cost_plan_A : ℕ := x * cost_A

/-- Cost of Plan C -/
def cost_plan_C : ℕ := days_together * (cost_A + cost_B) + (x - days_together) * cost_B

/-- Theorem stating that Plan C is not more expensive than Plan A -/
theorem plan_C_not_more_expensive : cost_plan_C ≤ cost_plan_A := by sorry

end plan_C_not_more_expensive_l3300_330078


namespace tetrahedron_intersection_theorem_l3300_330079

/-- Represents a tetrahedron with an inscribed sphere -/
structure TetrahedronWithSphere where
  volume : ℝ
  surface_area : ℝ
  inscribed_sphere_radius : ℝ

/-- Represents a plane intersecting three edges of a tetrahedron -/
structure IntersectingPlane where
  passes_through_center : Bool

/-- Represents the parts of the tetrahedron created by the intersecting plane -/
structure TetrahedronParts where
  volume_ratio : ℝ
  surface_area_ratio : ℝ

/-- The main theorem statement -/
theorem tetrahedron_intersection_theorem 
  (t : TetrahedronWithSphere) 
  (p : IntersectingPlane) 
  (parts : TetrahedronParts) : 
  (parts.volume_ratio = parts.surface_area_ratio) ↔ p.passes_through_center := by
  sorry

end tetrahedron_intersection_theorem_l3300_330079


namespace average_sale_is_7500_l3300_330026

def monthly_sales : List ℕ := [7435, 7920, 7855, 8230, 7560, 6000]

def total_sales : ℕ := monthly_sales.sum

def num_months : ℕ := monthly_sales.length

def average_sale : ℚ := (total_sales : ℚ) / (num_months : ℚ)

theorem average_sale_is_7500 : average_sale = 7500 := by
  sorry

end average_sale_is_7500_l3300_330026


namespace nancy_balloon_count_l3300_330057

/-- Given that Mary has 7 balloons and Nancy has 4 times as many balloons as Mary,
    prove that Nancy has 28 balloons. -/
theorem nancy_balloon_count :
  ∀ (mary_balloons nancy_balloons : ℕ),
    mary_balloons = 7 →
    nancy_balloons = 4 * mary_balloons →
    nancy_balloons = 28 :=
by
  sorry

end nancy_balloon_count_l3300_330057


namespace student_grade_problem_l3300_330032

theorem student_grade_problem (grade2 grade3 overall : ℚ) :
  grade2 = 80 →
  grade3 = 75 →
  overall = 75 →
  ∃ grade1 : ℚ, (grade1 + grade2 + grade3) / 3 = overall ∧ grade1 = 70 :=
by sorry

end student_grade_problem_l3300_330032


namespace train_speed_l3300_330061

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 200) (h2 : time = 20) :
  length / time = 10 := by
  sorry

end train_speed_l3300_330061


namespace printing_presses_l3300_330028

theorem printing_presses (papers : ℕ) (initial_time hours : ℝ) (known_presses : ℕ) :
  papers > 0 →
  initial_time > 0 →
  hours > 0 →
  known_presses > 0 →
  (papers : ℝ) / (initial_time * (papers / (hours * known_presses : ℝ))) = 40 :=
by
  sorry

#check printing_presses 500000 9 12 30

end printing_presses_l3300_330028


namespace polar_equation_is_circle_l3300_330066

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 1 / (1 - Real.sin θ)

-- Define the Cartesian equation derived from the polar equation
def cartesian_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 0

-- Theorem stating that the equation represents a circle (point)
theorem polar_equation_is_circle :
  ∃! (x y : ℝ), cartesian_equation x y ∧ x = 0 ∧ y = 1 :=
sorry

end polar_equation_is_circle_l3300_330066


namespace part_one_part_two_l3300_330091

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 3

-- Part 1
theorem part_one (x : ℝ) :
  (p x 1) ∧ (q x) → 2 ≤ x ∧ x < 3 := by sorry

-- Part 2
theorem part_two :
  (∀ x a : ℝ, (¬(p x a) → ¬(q x)) ∧ ¬(q x → ¬(p x a))) →
  ∃ a : ℝ, 1 < a ∧ a < 2 := by sorry

end part_one_part_two_l3300_330091


namespace alternating_ball_probability_l3300_330021

-- Define the number of balls of each color
def white_balls : ℕ := 5
def black_balls : ℕ := 5
def red_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := white_balls + black_balls + red_balls

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the function to calculate the number of successful arrangements
def successful_arrangements : ℕ := 
  (binomial (total_balls) red_balls) * (binomial (white_balls + black_balls) white_balls)

-- Define the function to calculate the total number of arrangements
def total_arrangements : ℕ := 
  Nat.factorial total_balls / (Nat.factorial white_balls * Nat.factorial black_balls * Nat.factorial red_balls)

-- State the theorem
theorem alternating_ball_probability : 
  (successful_arrangements : ℚ) / total_arrangements = 123 / 205 := by sorry

end alternating_ball_probability_l3300_330021


namespace fraction_of_books_sold_l3300_330062

theorem fraction_of_books_sold (total_revenue : ℕ) (remaining_books : ℕ) (price_per_book : ℕ) :
  total_revenue = 288 →
  remaining_books = 36 →
  price_per_book = 4 →
  (total_revenue / price_per_book : ℚ) / ((total_revenue / price_per_book) + remaining_books) = 2/3 :=
by sorry

end fraction_of_books_sold_l3300_330062


namespace city_distance_proof_l3300_330024

/-- Given a map distance between two cities and a map scale, calculates the actual distance between the cities. -/
def actualDistance (mapDistance : ℝ) (mapScale : ℝ) : ℝ :=
  mapDistance * mapScale

/-- Theorem stating that for a map distance of 120 cm and a scale of 1 cm : 20 km, the actual distance is 2400 km. -/
theorem city_distance_proof :
  let mapDistance : ℝ := 120
  let mapScale : ℝ := 20
  actualDistance mapDistance mapScale = 2400 := by
  sorry

#eval actualDistance 120 20

end city_distance_proof_l3300_330024


namespace scissors_count_l3300_330098

/-- The total number of scissors after adding more to an initial amount -/
def total_scissors (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 54 initial scissors and 22 added scissors, the total is 76 -/
theorem scissors_count : total_scissors 54 22 = 76 := by
  sorry

end scissors_count_l3300_330098


namespace min_value_expression_l3300_330035

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 ∧ y > 1 ∧ x + y = 2 → 4/x + 1/(y-1) ≥ 4/a + 1/(b-1)) →
  4/a + 1/(b-1) = 9 :=
by sorry

end min_value_expression_l3300_330035


namespace naClOConcentrationDecreases_l3300_330031

-- Define the disinfectant solution
structure DisinfectantSolution :=
  (volume : ℝ)
  (naClOConcentration : ℝ)
  (density : ℝ)

-- Define the properties of the initial solution
def initialSolution : DisinfectantSolution :=
  { volume := 480,
    naClOConcentration := 0.25,
    density := 1.19 }

-- Define the property that NaClO absorbs H₂O and CO₂ from air and degrades
axiom naClODegrades : ∀ (t : ℝ), t > 0 → ∃ (δ : ℝ), δ > 0 ∧ δ < initialSolution.naClOConcentration

-- Theorem stating that NaClO concentration decreases over time
theorem naClOConcentrationDecreases :
  ∀ (t : ℝ), t > 0 →
  ∃ (s : DisinfectantSolution),
    s.volume = initialSolution.volume ∧
    s.density = initialSolution.density ∧
    s.naClOConcentration < initialSolution.naClOConcentration :=
sorry

end naClOConcentrationDecreases_l3300_330031


namespace triangle_area_approx_036_l3300_330084

-- Define the slopes and intersection point
def slope1 : ℚ := 3/4
def slope2 : ℚ := 1/3
def intersection : ℚ × ℚ := (3, 3)

-- Define the lines
def line1 (x : ℚ) : ℚ := slope1 * (x - intersection.1) + intersection.2
def line2 (x : ℚ) : ℚ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℚ) : Prop := x + y = 8

-- Define the function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Theorem statement
theorem triangle_area_approx_036 :
  ∃ (p1 p2 p3 : ℚ × ℚ),
    p1 = intersection ∧
    line1 p2.1 = p2.2 ∧
    line2 p3.1 = p3.2 ∧
    line3 p2.1 p2.2 ∧
    line3 p3.1 p3.2 ∧
    abs (triangleArea p1 p2 p3 - 0.36) < 0.01 := by
  sorry

end triangle_area_approx_036_l3300_330084


namespace toys_produced_daily_l3300_330014

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 8000

/-- The number of working days per week -/
def working_days : ℕ := 4

/-- The number of toys produced each day -/
def toys_per_day : ℕ := toys_per_week / working_days

/-- Theorem stating that the number of toys produced each day is 2000 -/
theorem toys_produced_daily :
  toys_per_day = 2000 :=
by sorry

end toys_produced_daily_l3300_330014


namespace east_high_sports_percentage_l3300_330006

/-- The percentage of students who play sports at East High School -/
def percentage_sports (total_students : ℕ) (soccer_players : ℕ) (soccer_percentage : ℚ) : ℚ :=
  (soccer_players : ℚ) / (soccer_percentage * (total_students : ℚ))

theorem east_high_sports_percentage :
  percentage_sports 400 26 (25 / 200) = 13 / 25 :=
by sorry

end east_high_sports_percentage_l3300_330006


namespace car_distance_theorem_l3300_330073

/-- Calculates the distance between two cars on a main road -/
def distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - car1_distance - car2_distance

theorem car_distance_theorem (initial_distance car1_distance car2_distance : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : car1_distance = 50)
  (h3 : car2_distance = 62) :
  distance_between_cars initial_distance car1_distance car2_distance = 38 := by
  sorry

#eval distance_between_cars 150 50 62

end car_distance_theorem_l3300_330073


namespace rectangle_perimeter_from_squares_l3300_330040

theorem rectangle_perimeter_from_squares (side_length : ℝ) : 
  side_length = 3 → 
  ∃ (perimeter₁ perimeter₂ : ℝ), 
    (perimeter₁ = 24 ∧ perimeter₂ = 30) ∧ 
    (∀ (p : ℝ), p ≠ perimeter₁ ∧ p ≠ perimeter₂ → 
      ¬∃ (length width : ℝ), 
        (length * width = 4 * side_length^2) ∧ 
        (2 * (length + width) = p)) :=
by sorry

end rectangle_perimeter_from_squares_l3300_330040


namespace chess_and_go_pricing_and_max_purchase_l3300_330090

/-- The unit price of a Chinese chess set -/
def chinese_chess_price : ℝ := 25

/-- The unit price of a Go set -/
def go_price : ℝ := 30

/-- The total number of sets to be purchased -/
def total_sets : ℕ := 120

/-- The maximum total cost -/
def max_total_cost : ℝ := 3500

theorem chess_and_go_pricing_and_max_purchase :
  (2 * chinese_chess_price + go_price = 80) ∧
  (4 * chinese_chess_price + 3 * go_price = 190) ∧
  (∀ m : ℕ, m ≤ total_sets → 
    chinese_chess_price * (total_sets - m) + go_price * m ≤ max_total_cost →
    m ≤ 100) ∧
  (∃ m : ℕ, m = 100 ∧ 
    chinese_chess_price * (total_sets - m) + go_price * m ≤ max_total_cost) :=
by sorry

end chess_and_go_pricing_and_max_purchase_l3300_330090


namespace polynomial_derivative_sum_l3300_330087

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 20 := by
  sorry

end polynomial_derivative_sum_l3300_330087


namespace martha_problems_l3300_330039

theorem martha_problems (total : ℕ) (angela_unique : ℕ) : total = 20 → angela_unique = 9 → ∃ martha : ℕ,
  martha + (4 * martha - 2) + ((4 * martha - 2) / 2) + angela_unique = total ∧ martha = 2 := by
  sorry

end martha_problems_l3300_330039


namespace product_sum_theorem_l3300_330033

theorem product_sum_theorem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by
  sorry

end product_sum_theorem_l3300_330033


namespace inequality_proof_l3300_330045

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end inequality_proof_l3300_330045


namespace number_of_terms_S_9891_1989_l3300_330056

/-- Elementary symmetric expression -/
def S (k : ℕ) (n : ℕ) : ℕ := Nat.choose k n

/-- The number of terms in S_{9891}(1989) -/
theorem number_of_terms_S_9891_1989 : S 9891 1989 = Nat.choose 9891 1989 := by
  sorry

end number_of_terms_S_9891_1989_l3300_330056


namespace product_of_non_shared_sides_squared_l3300_330068

/-- Represents a right triangle with given area and side lengths -/
structure RightTriangle where
  area : ℝ
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area_eq : area = (side1 * side2) / 2
  pythagoras : side1^2 + side2^2 = hypotenuse^2

/-- Theorem about the product of non-shared sides of two specific right triangles -/
theorem product_of_non_shared_sides_squared
  (T₁ T₂ : RightTriangle)
  (h₁ : T₁.area = 3)
  (h₂ : T₂.area = 4)
  (h₃ : T₁.side1 = T₂.side1)  -- Shared side
  (h₄ : T₁.side2 = T₂.side2)  -- Shared side
  (h₅ : T₁.side1 = T₁.side2)  -- 45°-45°-90° triangle condition
  : (T₁.hypotenuse * T₂.hypotenuse)^2 = 64 := by
  sorry

end product_of_non_shared_sides_squared_l3300_330068


namespace average_problem_l3300_330022

theorem average_problem (x y : ℝ) : 
  ((100 + 200300 + x) / 3 = 250) → 
  ((300 + 150100 + x + y) / 4 = 200) → 
  y = -4250 := by
sorry

end average_problem_l3300_330022


namespace pyramid_volume_l3300_330010

theorem pyramid_volume (h : ℝ) (h_parallel : ℝ) (cross_section_area : ℝ) :
  h = 8 →
  h_parallel = 3 →
  cross_section_area = 4 →
  (1/3 : ℝ) * (cross_section_area * (h / h_parallel)^2) * h = 2048/27 :=
by sorry

end pyramid_volume_l3300_330010


namespace mrs_hilt_walking_distance_l3300_330036

/-- The total distance walked to and from a water fountain -/
def total_distance (distance_to_fountain : ℕ) (num_trips : ℕ) : ℕ :=
  2 * distance_to_fountain * num_trips

/-- Theorem: Mrs. Hilt walks 240 feet given the problem conditions -/
theorem mrs_hilt_walking_distance :
  total_distance 30 4 = 240 := by
  sorry

end mrs_hilt_walking_distance_l3300_330036


namespace max_value_of_f_l3300_330023

def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ 
  (∀ x, x ∈ Set.Icc 0 1 → f x ≤ f c) ∧
  f c = 1 := by
  sorry

end max_value_of_f_l3300_330023


namespace sqrt_real_implies_x_geq_8_l3300_330051

theorem sqrt_real_implies_x_geq_8 (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 8) → x ≥ 8 := by
  sorry

end sqrt_real_implies_x_geq_8_l3300_330051


namespace arithmetic_mean_problem_l3300_330002

theorem arithmetic_mean_problem (x y : ℝ) : 
  ((x + y) + (y + 30) + (3 * x) + (y - 10) + (2 * x + y + 20)) / 5 = 50 → 
  x = 21 ∧ y = 21 := by
sorry

end arithmetic_mean_problem_l3300_330002


namespace polynomial_solution_set_l3300_330096

theorem polynomial_solution_set : ∃ (S : Set ℂ), 
  S = {z : ℂ | z^4 + 2*z^3 + 2*z^2 + 2*z + 1 = 0} ∧ 
  S = {-1, Complex.I, -Complex.I} := by
  sorry

end polynomial_solution_set_l3300_330096


namespace maria_coin_count_l3300_330038

theorem maria_coin_count (num_stacks : ℕ) (coins_per_stack : ℕ) : 
  num_stacks = 5 → coins_per_stack = 3 → num_stacks * coins_per_stack = 15 := by
  sorry

end maria_coin_count_l3300_330038


namespace john_half_decks_l3300_330063

/-- The number of cards in a full deck -/
def full_deck : ℕ := 52

/-- The number of full decks John has -/
def num_full_decks : ℕ := 3

/-- The number of cards John threw away -/
def discarded_cards : ℕ := 34

/-- The number of cards John has after discarding -/
def remaining_cards : ℕ := 200

/-- Calculates the number of half-full decks John found -/
def num_half_decks : ℕ :=
  (remaining_cards + discarded_cards - num_full_decks * full_deck) / (full_deck / 2)

theorem john_half_decks :
  num_half_decks = 3 := by sorry

end john_half_decks_l3300_330063


namespace milk_fraction_after_pours_l3300_330037

/-- Represents the contents of a cup -/
structure CupContents where
  tea : ℚ
  milk : ℚ

/-- Represents the state of both cups -/
structure TwoCapState where
  cup1 : CupContents
  cup2 : CupContents

/-- Initial state of the cups -/
def initial_state : TwoCapState :=
  { cup1 := { tea := 6, milk := 0 },
    cup2 := { tea := 0, milk := 6 } }

/-- Pour one-third of tea from cup1 to cup2 -/
def pour_tea (state : TwoCapState) : TwoCapState := sorry

/-- Pour half of the mixture from cup2 to cup1 -/
def pour_mixture (state : TwoCapState) : TwoCapState := sorry

/-- Calculate the fraction of milk in a cup -/
def milk_fraction (cup : CupContents) : ℚ := sorry

/-- The main theorem to prove -/
theorem milk_fraction_after_pours :
  let state1 := pour_tea initial_state
  let state2 := pour_mixture state1
  milk_fraction state2.cup1 = 3/8 := by sorry

end milk_fraction_after_pours_l3300_330037


namespace chord_length_of_concentric_circles_l3300_330085

/-- Given two concentric circles with the following properties:
  - The area of the ring between the circles is 50π/3 square inches
  - The diameter of the larger circle is 10 inches
  This theorem proves that the length of a chord of the larger circle 
  that is tangent to the smaller circle is 10√6/3 inches. -/
theorem chord_length_of_concentric_circles (a b : ℝ) : 
  a = 5 →  -- Radius of larger circle
  π * a^2 - π * b^2 = (50/3) * π →  -- Area of ring
  ∃ c : ℝ, c = (10 * Real.sqrt 6) / 3 ∧ 
    c^2 = 4 * (a^2 - b^2) :=  -- Length of chord tangent to smaller circle
by
  sorry

#check chord_length_of_concentric_circles

end chord_length_of_concentric_circles_l3300_330085


namespace expression_equality_l3300_330016

theorem expression_equality : 
  |1 - Real.sqrt 2| - 2 * Real.cos (45 * π / 180) + (1 / 2)⁻¹ = 1 := by
  sorry

end expression_equality_l3300_330016


namespace expected_value_of_x_l3300_330047

/-- Represents the contingency table data -/
structure ContingencyTable where
  boys_a : ℕ
  boys_b : ℕ
  girls_a : ℕ
  girls_b : ℕ

/-- Represents the distribution of X -/
structure Distribution where
  p0 : ℚ
  p1 : ℚ
  p2 : ℚ
  p3 : ℚ

/-- Main theorem statement -/
theorem expected_value_of_x (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ) 
  (table : ContingencyTable) (dist : Distribution) : 
  total_students = 450 →
  total_boys = 250 →
  total_girls = 200 →
  table.boys_a + table.boys_b = total_boys →
  table.girls_a + table.girls_b = total_girls →
  table.boys_b = 150 →
  table.girls_a = 50 →
  dist.p0 = 1/6 →
  dist.p1 = 1/2 →
  dist.p2 = 3/10 →
  dist.p3 = 1/30 →
  0 * dist.p0 + 1 * dist.p1 + 2 * dist.p2 + 3 * dist.p3 = 6/5 := by
  sorry


end expected_value_of_x_l3300_330047


namespace reciprocal_sum_one_l3300_330053

theorem reciprocal_sum_one (x y z : ℕ+) : 
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 1 ↔ 
  ((x = 2 ∧ y = 4 ∧ z = 4) ∨ 
   (x = 2 ∧ y = 3 ∧ z = 6) ∨ 
   (x = 3 ∧ y = 3 ∧ z = 3)) :=
sorry

end reciprocal_sum_one_l3300_330053


namespace f_2_value_l3300_330042

/-- An odd function -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- An even function -/
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

/-- The main theorem -/
theorem f_2_value (f g : ℝ → ℝ) (a : ℝ) :
  odd_function f →
  even_function g →
  (∀ x, f x + g x = a^x - a^(-x) + 2) →
  a > 0 →
  a ≠ 1 →
  g 2 = a →
  f 2 = 15/4 := by
  sorry


end f_2_value_l3300_330042


namespace ellipse_standard_equation_l3300_330004

/-- Given an ellipse with specific properties, prove its standard equation -/
theorem ellipse_standard_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : c / a = Real.sqrt 3 / 3) 
  (h4 : 2 * b^2 / a = 4 * Real.sqrt 3 / 3) 
  (h5 : a^2 = b^2 + c^2) :
  ∃ (x y : ℝ), x^2 / 3 + y^2 / 2 = 1 ∧ 
    x^2 / a^2 + y^2 / b^2 = 1 := by sorry

end ellipse_standard_equation_l3300_330004


namespace horizontal_shift_shift_left_3_units_l3300_330009

-- Define the original function
def f (x : ℝ) : ℝ := 2 * x - 3

-- Define the transformed function
def g (x : ℝ) : ℝ := f (2 * x + 3)

-- Theorem stating the horizontal shift
theorem horizontal_shift :
  ∀ x : ℝ, g x = f (x + 3) :=
by
  sorry

-- Theorem stating that the shift is 3 units to the left
theorem shift_left_3_units :
  ∀ x : ℝ, g x = f (x + 3) ∧ (x + 3) - x = 3 :=
by
  sorry

end horizontal_shift_shift_left_3_units_l3300_330009


namespace amy_homework_time_l3300_330054

/-- Calculates the total time needed to complete homework with breaks -/
def total_homework_time (math_problems : ℕ) (spelling_problems : ℕ) 
  (math_rate : ℕ) (spelling_rate : ℕ) (break_duration : ℚ) : ℚ :=
  let work_hours : ℚ := (math_problems / math_rate + spelling_problems / spelling_rate : ℚ)
  let break_hours : ℚ := (work_hours.floor - 1) * break_duration
  work_hours + break_hours

/-- Theorem: Amy will take 11 hours to finish her homework -/
theorem amy_homework_time : 
  total_homework_time 18 6 3 2 (1/4) = 11 := by sorry

end amy_homework_time_l3300_330054


namespace distance_is_sqrt_6_l3300_330058

def A : ℝ × ℝ × ℝ := (1, -1, -1)
def P : ℝ × ℝ × ℝ := (1, 1, 1)
def direction_vector : ℝ × ℝ × ℝ := (1, 0, -1)

def distance_point_to_line (P : ℝ × ℝ × ℝ) (A : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_sqrt_6 :
  distance_point_to_line P A direction_vector = Real.sqrt 6 := by
  sorry

end distance_is_sqrt_6_l3300_330058


namespace composition_of_transformations_l3300_330067

-- Define the transformations
def f (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def g (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 - p.2)

-- State the theorem
theorem composition_of_transformations :
  f (g (-1, 2)) = (1, -3) := by sorry

end composition_of_transformations_l3300_330067


namespace contemporary_probability_is_five_ninths_l3300_330046

/-- Represents a scientist with a birth year and lifespan -/
structure Scientist where
  birth_year : ℝ
  lifespan : ℝ

/-- The total time span in years -/
def total_span : ℝ := 600

/-- The probability that two scientists were contemporaries -/
noncomputable def contemporary_probability (s1 s2 : Scientist) : ℝ :=
  let overlap_area := (total_span - (s1.lifespan + s2.lifespan)) ^ 2
  (total_span ^ 2 - overlap_area) / (total_span ^ 2)

/-- The main theorem stating the probability of two scientists being contemporaries -/
theorem contemporary_probability_is_five_ninths :
  ∃ (s1 s2 : Scientist),
    s1.lifespan = 110 ∧
    s2.lifespan = 90 ∧
    s1.birth_year ≥ 0 ∧
    s1.birth_year ≤ total_span ∧
    s2.birth_year ≥ 0 ∧
    s2.birth_year ≤ total_span ∧
    contemporary_probability s1 s2 = 5 / 9 :=
by
  sorry

end contemporary_probability_is_five_ninths_l3300_330046


namespace sqrt_two_between_one_and_two_l3300_330081

theorem sqrt_two_between_one_and_two :
  1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2 := by
  sorry

end sqrt_two_between_one_and_two_l3300_330081


namespace invalid_votes_percentage_l3300_330027

theorem invalid_votes_percentage (total_votes : ℕ) (winning_percentage : ℚ) (losing_votes : ℕ) :
  total_votes = 7500 →
  winning_percentage = 55 / 100 →
  losing_votes = 2700 →
  (total_votes - (losing_votes / (1 - winning_percentage))) / total_votes = 1 / 5 := by
  sorry

end invalid_votes_percentage_l3300_330027


namespace rotation_coordinates_l3300_330003

/-- 
Given a point (x, y) in a Cartesian coordinate plane and a rotation by angle α around the origin,
the coordinates of the rotated point are (x cos α - y sin α, x sin α + y cos α).
-/
theorem rotation_coordinates (x y α : ℝ) : 
  let original_point := (x, y)
  let rotated_point := (x * Real.cos α - y * Real.sin α, x * Real.sin α + y * Real.cos α)
  ∃ (R φ : ℝ), 
    x = R * Real.cos φ ∧ 
    y = R * Real.sin φ ∧ 
    rotated_point = (R * Real.cos (φ + α), R * Real.sin (φ + α)) :=
by sorry

end rotation_coordinates_l3300_330003


namespace total_candy_is_54_l3300_330092

/-- The number of students in the group -/
def num_students : ℕ := 9

/-- The number of chocolate pieces given to each student -/
def chocolate_per_student : ℕ := 2

/-- The number of hard candy pieces given to each student -/
def hard_candy_per_student : ℕ := 3

/-- The number of gummy candy pieces given to each student -/
def gummy_per_student : ℕ := 1

/-- The total number of candy pieces given away -/
def total_candy : ℕ := num_students * (chocolate_per_student + hard_candy_per_student + gummy_per_student)

theorem total_candy_is_54 : total_candy = 54 := by
  sorry

end total_candy_is_54_l3300_330092


namespace video_recorder_markup_percentage_l3300_330052

/-- Proves that the markup percentage is 20% given the problem conditions -/
theorem video_recorder_markup_percentage
  (wholesale_cost : ℝ)
  (employee_price : ℝ)
  (employee_discount : ℝ)
  (markup_percentage : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : employee_price = 192)
  (h3 : employee_discount = 0.20)
  (h4 : employee_price = (1 - employee_discount) * (wholesale_cost * (1 + markup_percentage / 100)))
  : markup_percentage = 20 := by
  sorry

end video_recorder_markup_percentage_l3300_330052


namespace inequalities_always_hold_l3300_330095

theorem inequalities_always_hold :
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → c / a < c / b) ∧
  (∀ a b : ℝ, (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2) ∧
  (∀ a b : ℝ, a + b ≤ Real.sqrt (2 * (a^2 + b^2))) :=
by sorry

end inequalities_always_hold_l3300_330095


namespace four_digit_permutations_eq_six_l3300_330064

/-- The number of different positive, four-digit integers that can be formed using the digits 3, 3, 8, and 8 -/
def four_digit_permutations : ℕ :=
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive, four-digit integers
    that can be formed using the digits 3, 3, 8, and 8 is equal to 6 -/
theorem four_digit_permutations_eq_six :
  four_digit_permutations = 6 := by
  sorry

#eval four_digit_permutations

end four_digit_permutations_eq_six_l3300_330064


namespace prob_spade_then_king_value_l3300_330060

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of kings in a standard deck -/
def NumKings : ℕ := 4

/-- Probability of drawing a spade as the first card and a king as the second card -/
def prob_spade_then_king : ℚ :=
  (NumSpades / StandardDeck) * (NumKings / (StandardDeck - 1))

theorem prob_spade_then_king_value :
  prob_spade_then_king = 17 / 884 := by
  sorry

end prob_spade_then_king_value_l3300_330060


namespace inequality_holds_iff_x_in_range_l3300_330089

theorem inequality_holds_iff_x_in_range :
  ∀ x : ℝ, (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) ↔ 
  (x < -1 ∨ x > 3) := by
  sorry

end inequality_holds_iff_x_in_range_l3300_330089


namespace regular_pentagon_diagonal_intersection_angle_l3300_330025

/-- A regular pentagon is a polygon with 5 equal sides and 5 equal angles. -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- The diagonals of a pentagon are line segments connecting non-adjacent vertices. -/
def diagonal (p : RegularPentagon) (i j : Fin 5) : sorry := sorry

/-- The intersection point of two diagonals in a pentagon. -/
def intersectionPoint (p : RegularPentagon) (d1 d2 : sorry) : ℝ × ℝ := sorry

/-- The angle between two line segments at their intersection point. -/
def angleBetween (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem regular_pentagon_diagonal_intersection_angle (p : RegularPentagon) :
  let s := intersectionPoint p (diagonal p 0 2) (diagonal p 1 3)
  angleBetween (p.vertices 2) s (p.vertices 3) = 72 := by sorry

end regular_pentagon_diagonal_intersection_angle_l3300_330025


namespace smallest_integer_satisfying_inequality_ten_satisfies_inequality_ten_is_smallest_satisfying_integer_l3300_330077

theorem smallest_integer_satisfying_inequality :
  ∀ n : ℤ, n^2 - 14*n + 45 > 0 → n ≥ 10 :=
by sorry

theorem ten_satisfies_inequality :
  10^2 - 14*10 + 45 > 0 :=
by sorry

theorem ten_is_smallest_satisfying_integer :
  ∀ n : ℤ, n < 10 → n^2 - 14*n + 45 ≤ 0 :=
by sorry

end smallest_integer_satisfying_inequality_ten_satisfies_inequality_ten_is_smallest_satisfying_integer_l3300_330077


namespace intersection_sum_zero_l3300_330094

theorem intersection_sum_zero (α β : ℝ) : 
  (∃ x₀ : ℝ, 
    (x₀ / (Real.sin α + Real.sin β) + (-x₀) / (Real.sin α + Real.cos β) = 1) ∧
    (x₀ / (Real.cos α + Real.sin β) + (-x₀) / (Real.cos α + Real.cos β) = 1)) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
sorry

end intersection_sum_zero_l3300_330094


namespace conference_handshakes_result_l3300_330008

/-- The number of handshakes at a conference with specified conditions -/
def conference_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let total_possible_handshakes := total_people * (total_people - 1)
  let handshakes_not_occurring := num_companies * reps_per_company * (reps_per_company - 1)
  (total_possible_handshakes - handshakes_not_occurring) / 2

/-- Theorem stating the number of handshakes for the given conference conditions -/
theorem conference_handshakes_result :
  conference_handshakes 3 5 = 75 := by
  sorry


end conference_handshakes_result_l3300_330008


namespace expression_simplification_l3300_330099

theorem expression_simplification (x y : ℝ) :
  (1/2) * x - 2 * (x - (1/3) * y^2) + (-3/2 * x + (1/3) * y^2) = -3 * x + y^2 := by
  sorry

end expression_simplification_l3300_330099


namespace parallelogram_area_l3300_330007

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 inches and 20 inches is 100√3 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin ((180 - θ) * π / 180) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l3300_330007


namespace angle_bisector_sum_l3300_330059

/-- A triangle with vertices A(2,3), B(-4,1), and C(5,-6) -/
structure Triangle where
  A : ℝ × ℝ := (2, 3)
  B : ℝ × ℝ := (-4, 1)
  C : ℝ × ℝ := (5, -6)

/-- The equation of an angle bisector in the form 3x + by + c = 0 -/
structure AngleBisectorEquation where
  b : ℝ
  c : ℝ

/-- The angle bisector of ∠A in the given triangle -/
def angleBisectorA (t : Triangle) : AngleBisectorEquation :=
  sorry

theorem angle_bisector_sum (t : Triangle) :
  let bisector := angleBisectorA t
  bisector.b + bisector.c = -2 := by sorry

end angle_bisector_sum_l3300_330059


namespace range_of_a_l3300_330097

theorem range_of_a (x : ℝ) (h1 : x > 0) (h2 : 2^x * (x - a) < 1) : a > -1 :=
sorry

end range_of_a_l3300_330097


namespace x_value_l3300_330015

theorem x_value : ∃ x : ℤ, 9823 + x = 13200 ∧ x = 3377 := by
  sorry

end x_value_l3300_330015


namespace perfect_fit_R_squared_eq_one_l3300_330012

/-- A structure representing a set of observations in a linear regression model. -/
structure LinearRegressionData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  a : ℝ
  b : ℝ
  e : Fin n → ℝ

/-- The coefficient of determination (R-squared) for a linear regression model. -/
def R_squared (data : LinearRegressionData) : ℝ := sorry

/-- Theorem stating that if all error terms are zero, then R-squared equals 1. -/
theorem perfect_fit_R_squared_eq_one (data : LinearRegressionData) 
  (h1 : ∀ i, data.y i = data.b * data.x i + data.a + data.e i)
  (h2 : ∀ i, data.e i = 0) :
  R_squared data = 1 := by sorry

end perfect_fit_R_squared_eq_one_l3300_330012


namespace expected_sixes_two_dice_l3300_330001

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 2

/-- The expected number of 6's when rolling two standard dice -/
def expected_sixes : ℚ := 1 / 3

/-- Theorem: The expected number of 6's when rolling two standard dice is 1/3 -/
theorem expected_sixes_two_dice : 
  expected_sixes = num_dice * prob_six := by sorry

end expected_sixes_two_dice_l3300_330001


namespace probability_sum_four_two_dice_l3300_330018

theorem probability_sum_four_two_dice : 
  let dice_count : ℕ := 2
  let faces_per_die : ℕ := 6
  let target_sum : ℕ := 4
  let total_outcomes : ℕ := faces_per_die ^ dice_count
  let favorable_outcomes : ℕ := 3
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by
  sorry

end probability_sum_four_two_dice_l3300_330018


namespace systematic_sampling_third_selection_l3300_330017

theorem systematic_sampling_third_selection
  (total_students : ℕ)
  (selected_students : ℕ)
  (first_selection : ℕ)
  (h1 : total_students = 100)
  (h2 : selected_students = 10)
  (h3 : first_selection = 3)
  : (first_selection + 2 * (total_students / selected_students)) % 100 = 23 := by
  sorry

end systematic_sampling_third_selection_l3300_330017


namespace gcd_of_225_and_135_l3300_330049

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end gcd_of_225_and_135_l3300_330049


namespace lindas_savings_l3300_330069

theorem lindas_savings (savings : ℚ) : 
  (7/13 : ℚ) * savings + (3/13 : ℚ) * savings + 180 = savings ∧ 
  (3/13 : ℚ) * savings = 2 * 180 → 
  savings = 1560 := by sorry

end lindas_savings_l3300_330069


namespace nine_times_eleven_and_two_fifths_l3300_330088

theorem nine_times_eleven_and_two_fifths (x : ℝ) : 
  9 * (11 + 2/5) = 102 + 3/5 := by
  sorry

end nine_times_eleven_and_two_fifths_l3300_330088


namespace science_fair_students_l3300_330083

theorem science_fair_students (know_it_all : ℕ) (karen : ℕ) (novel_corona : ℕ) (total : ℕ) :
  know_it_all = 50 →
  karen = 3 * know_it_all / 5 →
  total = 240 →
  total = know_it_all + karen + novel_corona →
  novel_corona = 160 := by
sorry

end science_fair_students_l3300_330083


namespace marikas_mothers_age_l3300_330013

/-- Given:
  - Marika was 10 years old in 2006
  - On Marika's 10th birthday, her mother's age was five times Marika's age

  Prove that the year when Marika's mother's age will be twice Marika's age is 2036
-/
theorem marikas_mothers_age (marika_birth_year : ℕ) (mothers_birth_year : ℕ) : 
  marika_birth_year = 1996 →
  mothers_birth_year = 1956 →
  ∃ (future_year : ℕ), future_year = 2036 ∧ 
    (future_year - mothers_birth_year) = 2 * (future_year - marika_birth_year) := by
  sorry

end marikas_mothers_age_l3300_330013


namespace locus_of_fourth_vertex_l3300_330055

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle with center and radius -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Checks if a point lies on a circle -/
def lies_on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Represents a rectangle by its vertices -/
structure Rectangle :=
  (A B C D : Point)

theorem locus_of_fourth_vertex 
  (O : Point) (r R : ℝ) (hr : 0 < r) (hR : r < R)
  (c1 : Circle) (c2 : Circle) (rect : Rectangle)
  (hc1 : c1 = ⟨O, r⟩) (hc2 : c2 = ⟨O, R⟩)
  (hA : lies_on_circle rect.A c2 ∨ lies_on_circle rect.A c1)
  (hB : lies_on_circle rect.B c2 ∨ lies_on_circle rect.B c1)
  (hD : lies_on_circle rect.D c2 ∨ lies_on_circle rect.D c1) :
  lies_on_circle rect.C c1 ∨ lies_on_circle rect.C c2 ∨
  (lies_on_circle rect.C c1 ∧ 
   (rect.C.x - O.x)^2 + (rect.C.y - O.y)^2 + 
   (rect.B.x - O.x)^2 + (rect.B.y - O.y)^2 = 2 * R^2) :=
sorry

end locus_of_fourth_vertex_l3300_330055
