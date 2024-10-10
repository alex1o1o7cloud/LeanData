import Mathlib

namespace pentagonal_prism_sum_l2945_294596

/-- Represents a pentagonal prism -/
structure PentagonalPrism where
  /-- Number of faces of the pentagonal prism -/
  faces : Nat
  /-- Number of edges of the pentagonal prism -/
  edges : Nat
  /-- Number of vertices of the pentagonal prism -/
  vertices : Nat
  /-- The number of faces is 7 (2 pentagonal + 5 rectangular) -/
  faces_eq : faces = 7
  /-- The number of edges is 15 (5 + 5 + 5) -/
  edges_eq : edges = 15
  /-- The number of vertices is 10 (5 + 5) -/
  vertices_eq : vertices = 10

/-- The sum of faces, edges, and vertices of a pentagonal prism is 32 -/
theorem pentagonal_prism_sum (p : PentagonalPrism) :
  p.faces + p.edges + p.vertices = 32 := by
  sorry

end pentagonal_prism_sum_l2945_294596


namespace extension_point_coordinates_l2945_294522

/-- Given points A and B, and a point C on the extension of AB such that BC = 2/3 * AB,
    prove that the coordinates of C are (53/3, 17/3). -/
theorem extension_point_coordinates (A B C : ℝ × ℝ) : 
  A = (1, -1) →
  B = (11, 3) →
  C - B = 2/3 • (B - A) →
  C = (53/3, 17/3) := by
sorry

end extension_point_coordinates_l2945_294522


namespace hypotenuse_squared_length_l2945_294529

/-- The ellipse in which the triangle is inscribed -/
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The right triangle inscribed in the ellipse -/
structure InscribedRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A = (0, 1)
  h_B_on_x_axis : B.2 = 0
  h_C_on_x_axis : C.2 = 0
  h_A_on_ellipse : ellipse A.1 A.2
  h_B_on_ellipse : ellipse B.1 B.2
  h_C_on_ellipse : ellipse C.1 C.2
  h_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

theorem hypotenuse_squared_length 
  (t : InscribedRightTriangle) : (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 = 10 := by
  sorry

end hypotenuse_squared_length_l2945_294529


namespace sum_of_three_numbers_l2945_294510

theorem sum_of_three_numbers : 731 + 672 + 586 = 1989 := by
  sorry

end sum_of_three_numbers_l2945_294510


namespace coconut_trips_proof_l2945_294542

def coconut_problem (total_coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : ℕ :=
  (total_coconuts + barbie_capacity + bruno_capacity - 1) / (barbie_capacity + bruno_capacity)

theorem coconut_trips_proof :
  coconut_problem 144 4 8 = 12 := by
  sorry

end coconut_trips_proof_l2945_294542


namespace basketball_team_sales_l2945_294573

/-- The number of cupcakes sold by the basketball team -/
def num_cupcakes : ℕ := 50

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The number of cookies sold -/
def num_cookies : ℕ := 40

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 1/2

/-- The number of basketballs bought -/
def num_basketballs : ℕ := 2

/-- The price of each basketball in dollars -/
def basketball_price : ℚ := 40

/-- The number of energy drinks bought -/
def num_energy_drinks : ℕ := 20

/-- The price of each energy drink in dollars -/
def energy_drink_price : ℚ := 2

theorem basketball_team_sales :
  (num_cupcakes * cupcake_price + num_cookies * cookie_price : ℚ) =
  (num_basketballs * basketball_price + num_energy_drinks * energy_drink_price : ℚ) :=
by sorry

end basketball_team_sales_l2945_294573


namespace vector_angle_obtuse_m_values_l2945_294577

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -3)

def angle_obtuse (x y : ℝ × ℝ) : Prop :=
  let dot_product := x.1 * y.1 + x.2 * y.2
  let magnitude_x := Real.sqrt (x.1^2 + x.2^2)
  let magnitude_y := Real.sqrt (y.1^2 + y.2^2)
  dot_product < 0 ∧ dot_product ≠ -magnitude_x * magnitude_y

theorem vector_angle_obtuse_m_values :
  ∀ m : ℝ, angle_obtuse a (b m) → m = -4 ∨ m = 7/4 :=
sorry

end vector_angle_obtuse_m_values_l2945_294577


namespace horses_equal_to_four_oxen_l2945_294579

/-- The cost of animals in Rupees --/
structure AnimalCosts where
  camel : ℝ
  horse : ℝ
  ox : ℝ
  elephant : ℝ

/-- The conditions of the problem --/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 170000 ∧
  costs.camel = 4184.615384615385

/-- The theorem to prove --/
theorem horses_equal_to_four_oxen (costs : AnimalCosts) 
  (h : problem_conditions costs) : 
  costs.horse = 4 * costs.ox := by
  sorry

#check horses_equal_to_four_oxen

end horses_equal_to_four_oxen_l2945_294579


namespace helga_shoe_shopping_l2945_294536

/-- The number of pairs of shoes Helga tried on at the first store -/
def first_store : ℕ := 7

/-- The number of pairs of shoes Helga tried on at the second store -/
def second_store : ℕ := first_store + 2

/-- The number of pairs of shoes Helga tried on at the third store -/
def third_store : ℕ := 0

/-- The total number of pairs of shoes Helga tried on at the first three stores -/
def first_three_stores : ℕ := first_store + second_store + third_store

/-- The number of pairs of shoes Helga tried on at the fourth store -/
def fourth_store : ℕ := 2 * first_three_stores

/-- The total number of pairs of shoes Helga tried on -/
def total_shoes : ℕ := first_three_stores + fourth_store

theorem helga_shoe_shopping : total_shoes = 48 := by
  sorry

end helga_shoe_shopping_l2945_294536


namespace number_of_true_statements_number_of_true_statements_is_correct_l2945_294531

/-- A quadratic equation x^2 + x - m = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

/-- The number of true propositions among the statement and its variants -/
theorem number_of_true_statements : ℕ :=
  let s1 := ∀ m : ℝ, m > 0 → has_real_roots m
  let s2 := ∀ m : ℝ, has_real_roots m → m > 0
  let s3 := ∀ m : ℝ, m ≤ 0 → ¬has_real_roots m
  let s4 := ∀ m : ℝ, ¬has_real_roots m → m ≤ 0
  2

theorem number_of_true_statements_is_correct :
  number_of_true_statements = 2 :=
by sorry

end number_of_true_statements_number_of_true_statements_is_correct_l2945_294531


namespace marble_remainder_l2945_294526

theorem marble_remainder (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) : 
  (r + p) % 8 = 4 := by
sorry

end marble_remainder_l2945_294526


namespace inequality_proof_l2945_294515

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end inequality_proof_l2945_294515


namespace combination_equality_l2945_294571

theorem combination_equality (x : ℕ) : 
  (Nat.choose 18 x = Nat.choose 18 (3*x - 6)) → (x = 3 ∨ x = 6) :=
by sorry

end combination_equality_l2945_294571


namespace equation_has_two_distinct_roots_l2945_294554

theorem equation_has_two_distinct_roots (a b : ℝ) (h : a ≠ b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁ + a) * (x₁ + b) = 2 * x₁ + a + b ∧
  (x₂ + a) * (x₂ + b) = 2 * x₂ + a + b :=
by sorry

end equation_has_two_distinct_roots_l2945_294554


namespace decimal_sum_to_fraction_l2945_294547

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 3125 := by
  sorry

end decimal_sum_to_fraction_l2945_294547


namespace double_force_quadruple_power_l2945_294502

/-- Represents the scenario of tugboats pushing a barge -/
structure TugboatScenario where
  k : ℝ  -- Constant of proportionality for water resistance
  F : ℝ  -- Initial force applied by one tugboat
  v : ℝ  -- Initial speed of the barge

/-- Calculates the power expended given force and velocity -/
def power (force velocity : ℝ) : ℝ := force * velocity

/-- Theorem stating that doubling the force quadruples the power when water resistance is proportional to speed -/
theorem double_force_quadruple_power (scenario : TugboatScenario) :
  let initial_power := power scenario.F scenario.v
  let final_power := power (2 * scenario.F) ((2 * scenario.F) / scenario.k)
  final_power = 4 * initial_power := by
  sorry


end double_force_quadruple_power_l2945_294502


namespace flu_infection_equation_l2945_294503

theorem flu_infection_equation (x : ℝ) : 
  (∃ (initial_infected : ℕ) (rounds : ℕ) (total_infected : ℕ),
    initial_infected = 1 ∧ 
    rounds = 2 ∧ 
    total_infected = 64 ∧ 
    (∀ r : ℕ, r ≤ rounds → 
      (initial_infected * (1 + x)^r = initial_infected * (total_infected / initial_infected)^(r/rounds))))
  → (1 + x)^2 = 64 :=
by sorry

end flu_infection_equation_l2945_294503


namespace sum_of_special_sequence_l2945_294543

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = 5

def geometric_subsequence (a : ℕ → ℚ) : Prop :=
  (a 2)^2 = a 1 * a 5

def sum_of_first_six (a : ℕ → ℚ) : ℚ :=
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6)

theorem sum_of_special_sequence :
  ∀ a : ℕ → ℚ,
  arithmetic_sequence a →
  geometric_subsequence a →
  sum_of_first_six a = 90 :=
sorry

end sum_of_special_sequence_l2945_294543


namespace sqrt_equation_solution_l2945_294591

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (9 + 3 * z) = 12 :=
by
  sorry

end sqrt_equation_solution_l2945_294591


namespace special_function_at_five_l2945_294586

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = f x * f y) ∧ 
  (f 0 ≠ 0) ∧ 
  (f 1 = 2)

/-- Theorem stating that f(5) = 0 for any function satisfying the special properties -/
theorem special_function_at_five {f : ℝ → ℝ} (hf : special_function f) : f 5 = 0 := by
  sorry

end special_function_at_five_l2945_294586


namespace shark_sighting_relationship_l2945_294523

/-- The relationship between shark sightings in Cape May and Daytona Beach --/
theorem shark_sighting_relationship (total_sightings cape_may_sightings : ℕ) 
  (h1 : total_sightings = 40)
  (h2 : cape_may_sightings = 24)
  (h3 : ∃ R : ℕ, cape_may_sightings = R - 8) :
  ∃ R : ℕ, R = 32 ∧ cape_may_sightings = R - 8 := by
  sorry

end shark_sighting_relationship_l2945_294523


namespace equidistant_circles_count_l2945_294574

-- Define a point in a 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle in a 2D plane
structure Circle2D where
  center : Point2D
  radius : ℝ

-- Function to check if a circle is equidistant from a set of points
def isEquidistant (c : Circle2D) (points : List Point2D) : Prop :=
  ∀ p ∈ points, (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Theorem statement
theorem equidistant_circles_count 
  (A B C D : Point2D) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  ∃! (circles : List Circle2D), 
    circles.length = 7 ∧ 
    ∀ c ∈ circles, isEquidistant c [A, B, C, D] :=
sorry

end equidistant_circles_count_l2945_294574


namespace elberta_amount_l2945_294500

/-- The amount of money Granny Smith has -/
def granny_smith : ℚ := 75

/-- The amount of money Anjou has -/
def anjou : ℚ := granny_smith / 4

/-- The amount of money Elberta has -/
def elberta : ℚ := anjou + 3

/-- Theorem stating that Elberta has $21.75 -/
theorem elberta_amount : elberta = 21.75 := by
  sorry

end elberta_amount_l2945_294500


namespace inequality_and_equality_condition_l2945_294534

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + c^2*b^2 ≥ 15/16 ∧ 
  (a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + c^2*b^2 = 15/16 ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end inequality_and_equality_condition_l2945_294534


namespace fraction_problem_l2945_294578

theorem fraction_problem (N : ℝ) (f : ℝ) :
  N = 24 →
  N * f - 10 = 0.25 * N →
  f = 2 / 3 := by
sorry

end fraction_problem_l2945_294578


namespace overhead_cost_calculation_l2945_294513

/-- The overhead cost for Steve's circus production -/
def overhead_cost : ℕ := sorry

/-- The production cost per performance -/
def production_cost_per_performance : ℕ := 7000

/-- The revenue from a sold-out performance -/
def revenue_per_performance : ℕ := 16000

/-- The number of sold-out performances needed to break even -/
def break_even_performances : ℕ := 9

/-- Theorem stating that the overhead cost is $81,000 -/
theorem overhead_cost_calculation :
  overhead_cost = 81000 :=
by
  sorry

end overhead_cost_calculation_l2945_294513


namespace remainder_after_adding_5000_l2945_294545

theorem remainder_after_adding_5000 (n : ℤ) (h : n % 6 = 4) : (n + 5000) % 6 = 0 := by
  sorry

end remainder_after_adding_5000_l2945_294545


namespace internet_service_fee_l2945_294532

/-- Internet service billing problem -/
theorem internet_service_fee (feb_bill mar_bill : ℝ) (usage_ratio : ℝ) 
  (hfeb : feb_bill = 18.60)
  (hmar : mar_bill = 30.90)
  (husage : usage_ratio = 3) : 
  ∃ (fixed_fee hourly_rate : ℝ),
    fixed_fee + hourly_rate = feb_bill ∧
    fixed_fee + usage_ratio * hourly_rate = mar_bill ∧
    fixed_fee = 12.45 := by
  sorry

end internet_service_fee_l2945_294532


namespace daily_rental_cost_is_30_l2945_294557

/-- Represents a car rental with a daily rate and a per-mile rate. -/
structure CarRental where
  dailyRate : ℝ
  perMileRate : ℝ

/-- Calculates the total cost of renting a car for one day and driving a given distance. -/
def totalCost (rental : CarRental) (distance : ℝ) : ℝ :=
  rental.dailyRate + rental.perMileRate * distance

/-- Theorem: Given the specified conditions, the daily rental cost is 30 dollars. -/
theorem daily_rental_cost_is_30 (rental : CarRental)
    (h1 : rental.perMileRate = 0.18)
    (h2 : totalCost rental 250.0 = 75) :
    rental.dailyRate = 30 := by
  sorry

end daily_rental_cost_is_30_l2945_294557


namespace area_equality_l2945_294518

-- Define the types for points and quadrilaterals
variable (Point : Type) [AddCommGroup Point] [Module ℝ Point]
variable (Quadrilateral : Type)

-- Define the necessary functions
variable (is_cyclic : Quadrilateral → Prop)
variable (midpoint : Point → Point → Point)
variable (orthocenter : Point → Point → Point → Point)
variable (area : Quadrilateral → ℝ)

-- Define the theorem
theorem area_equality 
  (A B C D E F G H W X Y Z : Point)
  (quad_ABCD quad_WXYZ : Quadrilateral) :
  is_cyclic quad_ABCD →
  E = midpoint A B →
  F = midpoint B C →
  G = midpoint C D →
  H = midpoint D A →
  W = orthocenter A H E →
  X = orthocenter B E F →
  Y = orthocenter C F G →
  Z = orthocenter D G H →
  area quad_ABCD = area quad_WXYZ :=
by sorry

end area_equality_l2945_294518


namespace arcsin_double_angle_l2945_294570

theorem arcsin_double_angle (x : ℝ) (θ : ℝ) 
  (h1 : x ∈ Set.Icc (-1) 1) 
  (h2 : Real.arcsin x = θ) 
  (h3 : θ ∈ Set.Icc (-Real.pi/2) (-Real.pi/4)) :
  Real.arcsin (2 * x * Real.sqrt (1 - x^2)) = -(Real.pi + 2*θ) := by
  sorry

end arcsin_double_angle_l2945_294570


namespace parabola_properties_l2945_294528

/-- Parabola represented by y = x^2 + bx - 2 -/
structure Parabola where
  b : ℝ

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about a specific parabola and its properties -/
theorem parabola_properties (p : Parabola) 
  (h1 : p.b = 4) -- Derived from the condition that the parabola passes through (1, 3)
  (A : Point) 
  (hA : A.x = 0 ∧ A.y = -2) -- A is the y-axis intersection point
  (B : Point) 
  (hB : B.x = -2 ∧ B.y = -6) -- B is the vertex of the parabola
  (k : ℝ) 
  (hk : k^2 + p.b * k - 2 = 0) -- k is the x-coordinate of x-axis intersection
  : 
  (1/2 * |A.y| * |B.x| = 2) ∧ 
  ((4*k^4 + 3*k^2 + 12*k - 6) / (k^8 + 2*k^6 + k^5 - 2*k^3 + 8*k^2 + 16) = 1/107) := by
  sorry

end parabola_properties_l2945_294528


namespace distance_range_l2945_294505

/-- Given three points A, B, and C in a metric space, if the distance between A and B is 8,
    and the distance between A and C is 5, then the distance between B and C is between 3 and 13. -/
theorem distance_range (X : Type*) [MetricSpace X] (A B C : X)
  (h1 : dist A B = 8) (h2 : dist A C = 5) :
  3 ≤ dist B C ∧ dist B C ≤ 13 := by
  sorry

end distance_range_l2945_294505


namespace shop_owner_gain_l2945_294549

/-- Calculates the overall percentage gain for a shop owner based on purchase and sale data -/
def overall_percentage_gain (
  notebook_purchase_qty : ℕ) (notebook_purchase_price : ℚ)
  (notebook_sale_qty : ℕ) (notebook_sale_price : ℚ)
  (pen_purchase_qty : ℕ) (pen_purchase_price : ℚ)
  (pen_sale_qty : ℕ) (pen_sale_price : ℚ)
  (bowl_purchase_qty : ℕ) (bowl_purchase_price : ℚ)
  (bowl_sale_qty : ℕ) (bowl_sale_price : ℚ) : ℚ :=
  let total_cost := notebook_purchase_qty * notebook_purchase_price +
                    pen_purchase_qty * pen_purchase_price +
                    bowl_purchase_qty * bowl_purchase_price
  let total_sale := notebook_sale_qty * notebook_sale_price +
                    pen_sale_qty * pen_sale_price +
                    bowl_sale_qty * bowl_sale_price
  let gain := total_sale - total_cost
  (gain / total_cost) * 100

/-- The overall percentage gain for the shop owner is approximately 16.01% -/
theorem shop_owner_gain :
  let gain := overall_percentage_gain 150 25 140 30 90 15 80 20 114 13 108 17
  ∃ ε > 0, |gain - 16.01| < ε :=
by sorry

end shop_owner_gain_l2945_294549


namespace heart_calculation_l2945_294555

-- Define the ♥ operation
def heart (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heart_calculation : heart 3 (heart 4 5) = -72 := by
  sorry

end heart_calculation_l2945_294555


namespace intersection_of_A_and_B_l2945_294530

def set_A : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def set_B : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt p.1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {(0, 0), (1, 1)} := by sorry

end intersection_of_A_and_B_l2945_294530


namespace line_through_points_l2945_294559

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def on_line (a b x : V) : Prop := ∃ t : ℝ, x = a + t • (b - a)

theorem line_through_points (a b : V) (h : a ≠ b) :
  ∃ k m : ℝ, m = 5/8 ∧ on_line a b (k • a + m • b) → k = 3/8 := by
  sorry

end line_through_points_l2945_294559


namespace tower_of_hanoi_l2945_294546

/-- The minimal number of moves required to transfer n disks
    from one rod to another in the Tower of Hanoi game. -/
def minMoves : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 * minMoves n + 1

/-- Theorem stating that the minimal number of moves for n disks
    in the Tower of Hanoi game is 2^n - 1. -/
theorem tower_of_hanoi (n : ℕ) : minMoves n = 2^n - 1 := by
  sorry

#eval minMoves 3  -- Expected output: 7
#eval minMoves 4  -- Expected output: 15

end tower_of_hanoi_l2945_294546


namespace pizza_eaters_l2945_294588

theorem pizza_eaters (total_slices : ℕ) (slices_left : ℕ) (slices_per_person : ℕ) : 
  total_slices = 16 →
  slices_left = 4 →
  slices_per_person = 2 →
  (total_slices - slices_left) / slices_per_person = 6 :=
by sorry

end pizza_eaters_l2945_294588


namespace beth_cookie_price_l2945_294589

/-- Represents a cookie baker --/
structure Baker where
  name : String
  cookieShape : String
  cookieCount : ℕ
  cookieArea : ℝ
  cookiePrice : ℝ

/-- Given conditions of the problem --/
def alexBaker : Baker := {
  name := "Alex"
  cookieShape := "rectangle"
  cookieCount := 10
  cookieArea := 20
  cookiePrice := 0.50
}

def bethBaker : Baker := {
  name := "Beth"
  cookieShape := "circle"
  cookieCount := 16
  cookieArea := 12.5
  cookiePrice := 0  -- To be calculated
}

/-- The total dough used by each baker --/
def totalDough (b : Baker) : ℝ := b.cookieCount * b.cookieArea

/-- The total earnings of a baker --/
def totalEarnings (b : Baker) : ℝ := b.cookieCount * b.cookiePrice * 100  -- in cents

/-- The main theorem to prove --/
theorem beth_cookie_price :
  totalDough alexBaker = totalDough bethBaker →
  totalEarnings alexBaker = bethBaker.cookieCount * 31.25 := by
  sorry


end beth_cookie_price_l2945_294589


namespace percentage_seven_plus_years_l2945_294565

/-- Represents the number of employees in each employment duration range --/
structure EmployeeDistribution :=
  (less_than_1_year : ℕ)
  (one_to_two_years : ℕ)
  (two_to_three_years : ℕ)
  (three_to_four_years : ℕ)
  (four_to_five_years : ℕ)
  (five_to_six_years : ℕ)
  (six_to_seven_years : ℕ)
  (seven_to_eight_years : ℕ)
  (eight_to_nine_years : ℕ)
  (nine_to_ten_years : ℕ)
  (ten_plus_years : ℕ)

/-- Calculates the total number of employees --/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1_year + d.one_to_two_years + d.two_to_three_years + d.three_to_four_years +
  d.four_to_five_years + d.five_to_six_years + d.six_to_seven_years + d.seven_to_eight_years +
  d.eight_to_nine_years + d.nine_to_ten_years + d.ten_plus_years

/-- Calculates the number of employees employed for 7 years or more --/
def employees_seven_plus_years (d : EmployeeDistribution) : ℕ :=
  d.seven_to_eight_years + d.eight_to_nine_years + d.nine_to_ten_years + d.ten_plus_years

/-- Theorem stating that the percentage of employees employed for 7 years or more is 21.43% --/
theorem percentage_seven_plus_years (d : EmployeeDistribution) 
  (h : d = {
    less_than_1_year := 4,
    one_to_two_years := 6,
    two_to_three_years := 5,
    three_to_four_years := 2,
    four_to_five_years := 3,
    five_to_six_years := 1,
    six_to_seven_years := 1,
    seven_to_eight_years := 2,
    eight_to_nine_years := 2,
    nine_to_ten_years := 1,
    ten_plus_years := 1
  }) :
  (employees_seven_plus_years d : ℚ) / (total_employees d : ℚ) * 100 = 21.43 := by
  sorry

end percentage_seven_plus_years_l2945_294565


namespace smallest_integer_l2945_294593

theorem smallest_integer (a b : ℕ) (ha : a = 75) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 45) :
  ∃ (m : ℕ), m ≥ b ∧ m = 135 ∧ Nat.lcm a m / Nat.gcd a m = 45 :=
sorry

end smallest_integer_l2945_294593


namespace all_sums_representable_l2945_294535

/-- Available coin denominations -/
def Coins : Set ℕ := {1, 2, 5, 10}

/-- A function that checks if a sum can be represented by both even and odd number of coins -/
def canRepresentEvenAndOdd (S : ℕ) : Prop :=
  ∃ (even_coins odd_coins : List ℕ),
    (∀ c ∈ even_coins, c ∈ Coins) ∧
    (∀ c ∈ odd_coins, c ∈ Coins) ∧
    (even_coins.sum = S) ∧
    (odd_coins.sum = S) ∧
    (even_coins.length % 2 = 0) ∧
    (odd_coins.length % 2 = 1)

/-- Theorem stating that any sum greater than 1 can be represented by both even and odd number of coins -/
theorem all_sums_representable (S : ℕ) (h : S > 1) :
  canRepresentEvenAndOdd S := by
  sorry

end all_sums_representable_l2945_294535


namespace dvd_rental_cost_l2945_294590

def rental_problem (num_dvds : ℕ) (original_price : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : Prop :=
  let discounted_price := original_price * (1 - discount_rate)
  let total_with_tax := discounted_price * (1 + tax_rate)
  let cost_per_dvd := total_with_tax / num_dvds
  ∃ (rounded_cost : ℚ), 
    rounded_cost = (cost_per_dvd * 100).floor / 100 ∧ 
    rounded_cost = 116 / 100

theorem dvd_rental_cost : 
  rental_problem 4 (480 / 100) (10 / 100) (7 / 100) :=
sorry

end dvd_rental_cost_l2945_294590


namespace line_tangent_to_curve_l2945_294520

/-- The line equation: kx - y + 1 = 0 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 = 0

/-- The curve equation: y² = 4x -/
def curve_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- The tangency condition: the discriminant of the resulting quadratic equation is zero -/
def tangency_condition (k : ℝ) : Prop :=
  (4 * k - 8)^2 - 16 * k^2 = 0

theorem line_tangent_to_curve (k : ℝ) :
  (∀ x y : ℝ, line_equation k x y ∧ curve_equation x y → tangency_condition k) →
  k = 1 := by
  sorry

end line_tangent_to_curve_l2945_294520


namespace inequality_system_solution_l2945_294550

theorem inequality_system_solution (p : ℝ) :
  (19 * p < 10 ∧ p > 1/2) ↔ (1/2 < p ∧ p < 10/19) := by
sorry

end inequality_system_solution_l2945_294550


namespace area_ratio_l2945_294563

-- Define the side lengths of the squares
def side_length_A (x : ℝ) : ℝ := x
def side_length_B (x : ℝ) : ℝ := 3 * x
def side_length_C (x : ℝ) : ℝ := 2 * x

-- Define the areas of the squares
def area_A (x : ℝ) : ℝ := (side_length_A x) ^ 2
def area_B (x : ℝ) : ℝ := (side_length_B x) ^ 2
def area_C (x : ℝ) : ℝ := (side_length_C x) ^ 2

-- Theorem stating the ratio of areas
theorem area_ratio (x : ℝ) (h : x > 0) : 
  area_A x / (area_B x + area_C x) = 1 / 13 := by
  sorry

end area_ratio_l2945_294563


namespace book_sale_profit_l2945_294560

theorem book_sale_profit (total_cost selling_price_1 cost_1 : ℚ) : 
  total_cost = 500 →
  cost_1 = 291.67 →
  selling_price_1 = cost_1 * (1 - 15/100) →
  selling_price_1 = (total_cost - cost_1) * (1 + 19/100) →
  True := by sorry

end book_sale_profit_l2945_294560


namespace ball_travel_distance_l2945_294558

/-- The distance traveled by the center of a ball rolling along a track of semicircular arcs -/
theorem ball_travel_distance (ball_diameter : ℝ) (R₁ R₂ R₃ : ℝ) : 
  ball_diameter = 6 → R₁ = 120 → R₂ = 70 → R₃ = 90 → 
  (R₁ - ball_diameter / 2) * π + (R₂ - ball_diameter / 2) * π + (R₃ - ball_diameter / 2) * π = 271 * π :=
by sorry

end ball_travel_distance_l2945_294558


namespace smallest_positive_congruence_l2945_294583

theorem smallest_positive_congruence :
  ∃ (n : ℕ), n > 0 ∧ n < 13 ∧ -1234 ≡ n [ZMOD 13] ∧
  ∀ (m : ℕ), m > 0 ∧ m < 13 ∧ -1234 ≡ m [ZMOD 13] → n ≤ m :=
by sorry

end smallest_positive_congruence_l2945_294583


namespace simplify_expression_1_simplify_expression_2_l2945_294572

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 
  (-1 + 3*x) * (-3*x - 1) = 1 - 9*x^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) : 
  (x + 1)^2 - (1 - 3*x) * (1 + 3*x) = 10*x^2 + 2*x := by sorry

end simplify_expression_1_simplify_expression_2_l2945_294572


namespace yogurt_calories_l2945_294594

def calories_per_ounce_yogurt (strawberries : ℕ) (yogurt_ounces : ℕ) (calories_per_strawberry : ℕ) (total_calories : ℕ) : ℕ :=
  (total_calories - strawberries * calories_per_strawberry) / yogurt_ounces

theorem yogurt_calories (strawberries : ℕ) (yogurt_ounces : ℕ) (calories_per_strawberry : ℕ) (total_calories : ℕ)
  (h1 : strawberries = 12)
  (h2 : yogurt_ounces = 6)
  (h3 : calories_per_strawberry = 4)
  (h4 : total_calories = 150) :
  calories_per_ounce_yogurt strawberries yogurt_ounces calories_per_strawberry total_calories = 17 := by
  sorry

end yogurt_calories_l2945_294594


namespace z2_magnitude_range_l2945_294566

theorem z2_magnitude_range (z₁ z₂ : ℂ) 
  (h1 : (z₁ - Complex.I) * (z₂ + Complex.I) = 1)
  (h2 : Complex.abs z₁ = Real.sqrt 2) :
  2 - Real.sqrt 2 ≤ Complex.abs z₂ ∧ Complex.abs z₂ ≤ 2 + Real.sqrt 2 := by
sorry

end z2_magnitude_range_l2945_294566


namespace factorization_problems_l2945_294595

theorem factorization_problems :
  (∀ a : ℝ, 4 * a^2 - 9 = (2*a + 3) * (2*a - 3)) ∧
  (∀ x y : ℝ, 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2) := by
  sorry

end factorization_problems_l2945_294595


namespace tv_selection_combinations_l2945_294551

def type_a_count : ℕ := 4
def type_b_count : ℕ := 5
def total_selection : ℕ := 3

theorem tv_selection_combinations : 
  (Nat.choose type_a_count 1 * Nat.choose type_b_count 2) + 
  (Nat.choose type_a_count 2 * Nat.choose type_b_count 1) = 70 := by
  sorry

end tv_selection_combinations_l2945_294551


namespace units_digit_27_pow_23_l2945_294581

def units_digit (n : ℕ) : ℕ := n % 10

def units_digit_power (base : ℕ) (exp : ℕ) : ℕ :=
  units_digit ((units_digit base)^exp)

theorem units_digit_27_pow_23 :
  units_digit (27^23) = 3 :=
by
  sorry

end units_digit_27_pow_23_l2945_294581


namespace circle_equation_proof_l2945_294539

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: The equation (x - 3)² + (y + 1)² = 25 represents the circle
    with center (3, -1) passing through the point (7, -4) -/
theorem circle_equation_proof (x y : ℝ) : 
  let center : Point := ⟨3, -1⟩
  let point : Point := ⟨7, -4⟩
  (x - center.x)^2 + (y - center.y)^2 = squaredDistance center point := by
  sorry

end circle_equation_proof_l2945_294539


namespace alyssa_games_this_year_l2945_294533

/-- The number of soccer games Alyssa attended over three years -/
def total_games : ℕ := 39

/-- The number of games Alyssa attended last year -/
def last_year_games : ℕ := 13

/-- The number of games Alyssa plans to attend next year -/
def next_year_games : ℕ := 15

/-- The number of games Alyssa attended this year -/
def this_year_games : ℕ := total_games - last_year_games - next_year_games

theorem alyssa_games_this_year :
  this_year_games = 11 := by sorry

end alyssa_games_this_year_l2945_294533


namespace circle_properties_l2945_294582

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x + 18*y + 98 = -y^2 - 6*x

-- Define the center and radius
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- State the theorem
theorem circle_properties :
  ∃ a b r : ℝ,
    is_center_radius a b r ∧
    a = -7 ∧
    b = -9 ∧
    r = 4 * Real.sqrt 2 ∧
    a + b + r = -16 + 4 * Real.sqrt 2 := by
  sorry

end circle_properties_l2945_294582


namespace simplify_and_evaluate_l2945_294575

theorem simplify_and_evaluate (a : ℝ) : 
  (a - 3)^2 - (a - 1) * (a + 1) + 2 * (a + 3) = 4 ↔ a = 3 := by
  sorry

end simplify_and_evaluate_l2945_294575


namespace kevin_initial_cards_l2945_294541

/-- The number of cards Kevin lost -/
def lost_cards : ℝ := 7.0

/-- The number of cards Kevin has after losing some -/
def remaining_cards : ℕ := 40

/-- The initial number of cards Kevin found -/
def initial_cards : ℝ := remaining_cards + lost_cards

theorem kevin_initial_cards : initial_cards = 47.0 := by
  sorry

end kevin_initial_cards_l2945_294541


namespace inequality_system_solution_l2945_294511

theorem inequality_system_solution :
  let S : Set ℝ := {x | 2 * x + 1 > 0 ∧ (x + 1) / 3 > x - 1}
  S = {x | -1/2 < x ∧ x < 2} := by
sorry

end inequality_system_solution_l2945_294511


namespace pauls_allowance_l2945_294561

/-- Paul's savings in dollars -/
def savings : ℕ := 3

/-- Cost of one toy in dollars -/
def toy_cost : ℕ := 5

/-- Number of toys Paul wants to buy -/
def num_toys : ℕ := 2

/-- Paul's allowance in dollars -/
def allowance : ℕ := 7

theorem pauls_allowance :
  savings + allowance = num_toys * toy_cost :=
sorry

end pauls_allowance_l2945_294561


namespace solution_property_l2945_294519

theorem solution_property (m n : ℝ) (hm : m ≠ 0) 
  (h : m^2 + n*m - m = 0) : m + n = 1 := by
  sorry

end solution_property_l2945_294519


namespace set_intersection_complement_empty_l2945_294544

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x | ∃ a : ℕ, x = a - 1}

theorem set_intersection_complement_empty : A ∩ (Set.univ \ B) = ∅ := by
  sorry

end set_intersection_complement_empty_l2945_294544


namespace equidistant_after_1_min_equidistant_after_5_min_speed_ratio_l2945_294598

-- Define the speeds of A and B
def v_A : ℝ := sorry
def v_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := -800

-- Define the equidistant condition after 1 minute
theorem equidistant_after_1_min : v_A = |initial_B_position + v_B| := sorry

-- Define the equidistant condition after 5 minutes
theorem equidistant_after_5_min : 5 * v_A = |initial_B_position + 5 * v_B| := sorry

-- Theorem to prove
theorem speed_ratio : v_A / v_B = 1 / 9 := sorry

end equidistant_after_1_min_equidistant_after_5_min_speed_ratio_l2945_294598


namespace a_squared_greater_than_b_squared_l2945_294538

theorem a_squared_greater_than_b_squared (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a = Real.log (1 + b) - Real.log (1 - b)) : a^2 > b^2 := by
  sorry

end a_squared_greater_than_b_squared_l2945_294538


namespace apples_remaining_l2945_294568

theorem apples_remaining (total : ℕ) (eaten : ℕ) (h1 : total = 15) (h2 : eaten = 7) :
  total - eaten = 8 := by
  sorry

end apples_remaining_l2945_294568


namespace governor_addresses_ratio_l2945_294576

theorem governor_addresses_ratio (S : ℕ) : 
  S + S / 2 + (S + 10) = 40 → S / (S / 2) = 2 := by
  sorry

end governor_addresses_ratio_l2945_294576


namespace sugar_solution_problem_l2945_294584

/-- Calculates the final sugar percentage when replacing part of a solution --/
def finalSugarPercentage (initialPercent : ℝ) (replacementPercent : ℝ) (replacementFraction : ℝ) : ℝ :=
  (initialPercent * (1 - replacementFraction) + replacementPercent * replacementFraction) * 100

/-- Theorem stating the final sugar percentage for the given problem --/
theorem sugar_solution_problem :
  finalSugarPercentage 0.1 0.42 0.25 = 18 := by
  sorry

#eval finalSugarPercentage 0.1 0.42 0.25

end sugar_solution_problem_l2945_294584


namespace bobby_candy_problem_l2945_294512

theorem bobby_candy_problem (x : ℕ) :
  x + 17 = 43 → x = 26 := by
  sorry

end bobby_candy_problem_l2945_294512


namespace polynomial_integer_solution_l2945_294585

theorem polynomial_integer_solution (p : ℤ → ℤ) 
  (h_integer_coeff : ∀ x y : ℤ, x - y ∣ p x - p y)
  (h_p_15 : p 15 = 6)
  (h_p_22 : p 22 = 1196)
  (h_p_35 : p 35 = 26) :
  ∃ n : ℤ, p n = n + 82 ∧ n = 28 := by
  sorry

end polynomial_integer_solution_l2945_294585


namespace flower_theorem_l2945_294597

def flower_problem (alissa_flowers melissa_flowers flowers_left : ℕ) : Prop :=
  alissa_flowers + melissa_flowers - flowers_left = 18

theorem flower_theorem :
  flower_problem 16 16 14 := by
  sorry

end flower_theorem_l2945_294597


namespace bank_line_time_l2945_294508

/-- Given a constant speed calculated from moving 20 meters in 40 minutes,
    prove that the time required to move an additional 100 meters is 200 minutes. -/
theorem bank_line_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ)
    (h1 : initial_distance = 20)
    (h2 : initial_time = 40)
    (h3 : additional_distance = 100) :
    (additional_distance / (initial_distance / initial_time)) = 200 := by
  sorry

end bank_line_time_l2945_294508


namespace john_and_alice_money_l2945_294599

theorem john_and_alice_money : 5/8 + 7/20 = 0.975 := by
  sorry

end john_and_alice_money_l2945_294599


namespace lampshade_container_volume_l2945_294506

/-- The volume of the smallest cylindrical container that can fit a conical lampshade -/
theorem lampshade_container_volume
  (h : ℝ) -- height of the lampshade
  (d : ℝ) -- diameter of the lampshade base
  (h_pos : h > 0)
  (d_pos : d > 0)
  (h_val : h = 15)
  (d_val : d = 8) :
  let r := d / 2 -- radius of the container
  let v := π * r^2 * h -- volume of the container
  v = 240 * π :=
sorry

end lampshade_container_volume_l2945_294506


namespace journey_duration_l2945_294507

/-- Given a journey with two parts, prove that the duration of the first part is 3 hours. -/
theorem journey_duration (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 240)
  (h2 : total_time = 5)
  (h3 : speed1 = 40)
  (h4 : speed2 = 60)
  (h5 : ∃ (distance1 : ℝ), 
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time) :
  ∃ (duration1 : ℝ), duration1 = 3 ∧ duration1 * speed1 + (total_time - duration1) * speed2 = total_distance :=
by sorry

end journey_duration_l2945_294507


namespace wall_height_proof_l2945_294553

/-- Given a wall and bricks with specified dimensions, proves the height of the wall. -/
theorem wall_height_proof (wall_length : Real) (wall_width : Real) (num_bricks : Nat)
  (brick_length : Real) (brick_width : Real) (brick_height : Real)
  (h_wall_length : wall_length = 8)
  (h_wall_width : wall_width = 6)
  (h_num_bricks : num_bricks = 1600)
  (h_brick_length : brick_length = 1)
  (h_brick_width : brick_width = 0.1125)
  (h_brick_height : brick_height = 0.06) :
  ∃ (wall_height : Real),
    wall_height = 0.225 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by
  sorry


end wall_height_proof_l2945_294553


namespace face_value_is_75_l2945_294587

/-- Given banker's discount (BD) and true discount (TD), calculate the face value (FV) -/
def calculate_face_value (BD TD : ℚ) : ℚ :=
  (TD^2) / (BD - TD)

/-- Theorem stating that given BD = 18 and TD = 15, the face value is 75 -/
theorem face_value_is_75 :
  calculate_face_value 18 15 = 75 := by
  sorry

#eval calculate_face_value 18 15

end face_value_is_75_l2945_294587


namespace symmetric_points_sum_l2945_294556

/-- Given two points A(a,3) and B(4,b) that are symmetric with respect to the y-axis,
    prove that a + b = -1 -/
theorem symmetric_points_sum (a b : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (a, 3) ∧ B = (4, b) ∧ 
    (A.1 = -B.1 ∧ A.2 = B.2)) → a + b = -1 := by
  sorry

end symmetric_points_sum_l2945_294556


namespace range_of_a_l2945_294517

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 1 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → a^2 - 3*a - x + 1 ≤ 0

-- Define the theorem
theorem range_of_a :
  ∃ a : ℝ, (¬(p a ∧ q a) ∧ ¬(¬(q a))) ∧ a ∈ Set.Icc 1 2 ∧ a ≠ 2 ∧
  (∀ b : ℝ, (¬(p b ∧ q b) ∧ ¬(¬(q b))) → b ∈ Set.Icc 1 2 ∧ b < 2) :=
sorry

end range_of_a_l2945_294517


namespace jenna_reading_goal_l2945_294537

/-- Calculates the number of pages Jenna needs to read per day to meet her reading goal --/
theorem jenna_reading_goal (total_pages : ℕ) (total_days : ℕ) (busy_days : ℕ) (special_day_pages : ℕ) :
  total_pages = 600 →
  total_days = 30 →
  busy_days = 4 →
  special_day_pages = 100 →
  (total_pages - special_day_pages) / (total_days - busy_days - 1) = 20 := by
  sorry

#check jenna_reading_goal

end jenna_reading_goal_l2945_294537


namespace rental_miles_driven_l2945_294580

/-- Given rental information, calculate the number of miles driven -/
theorem rental_miles_driven (rental_fee : ℝ) (charge_per_mile : ℝ) (total_paid : ℝ) : 
  rental_fee = 20.99 →
  charge_per_mile = 0.25 →
  total_paid = 95.74 →
  (total_paid - rental_fee) / charge_per_mile = 299 :=
by
  sorry

end rental_miles_driven_l2945_294580


namespace sector_angle_l2945_294514

theorem sector_angle (r : ℝ) (l : ℝ) (α : ℝ) :
  r = 1 →
  l = 2 →
  l = α * r →
  α = 2 := by sorry

end sector_angle_l2945_294514


namespace geometric_arithmetic_sequence_l2945_294548

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  q ≠ 1 →  -- q is not equal to 1
  2 * a 3 = a 1 + a 2 →  -- arithmetic sequence condition
  q = -1/2 := by
sorry

end geometric_arithmetic_sequence_l2945_294548


namespace chosen_number_l2945_294516

theorem chosen_number (x : ℝ) : (x / 4) - 175 = 10 → x = 740 := by
  sorry

end chosen_number_l2945_294516


namespace card_sorting_moves_l2945_294569

theorem card_sorting_moves (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) : ℕ := by
  let card_label (i : ℕ) : ℕ := (i + k - 1) % n + 1
  let min_moves := n - Nat.gcd n k
  sorry

#check card_sorting_moves

end card_sorting_moves_l2945_294569


namespace rectangle_area_l2945_294592

theorem rectangle_area (y : ℝ) (h : y > 0) : ∃ w l : ℝ,
  w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = y^2 ∧ w * l = (3 * y^2) / 10 := by
  sorry

end rectangle_area_l2945_294592


namespace sin_double_angle_for_line_l2945_294525

/-- Given a line with equation 2x-4y+5=0 and angle of inclination α, prove that sin2α = 4/5 -/
theorem sin_double_angle_for_line (x y : ℝ) (α : ℝ) 
  (h : 2 * x - 4 * y + 5 = 0) 
  (h_incline : α = Real.arctan (1 / 2)) : 
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_double_angle_for_line_l2945_294525


namespace min_value_squared_sum_l2945_294501

theorem min_value_squared_sum (a b c : ℝ) (h : a + 2*b + 3*c = 6) :
  ∃ m : ℝ, m = 12 ∧ ∀ x y z : ℝ, x + 2*y + 3*z = 6 → x^2 + 4*y^2 + 9*z^2 ≥ m :=
by sorry

end min_value_squared_sum_l2945_294501


namespace trig_expression_simplification_l2945_294567

theorem trig_expression_simplification :
  (Real.tan (40 * π / 180) + Real.tan (50 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (30 * π / 180) =
  2 * (Real.cos (60 * π / 180) * Real.cos (70 * π / 180) + Real.sin (50 * π / 180) * Real.cos (40 * π / 180) * Real.cos (50 * π / 180)) /
  (Real.sqrt 3 * Real.cos (40 * π / 180) * Real.cos (50 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) := by
  sorry

end trig_expression_simplification_l2945_294567


namespace bowls_sold_calculation_l2945_294552

def total_bowls : ℕ := 114
def cost_per_bowl : ℚ := 13
def sell_price_per_bowl : ℚ := 17
def percentage_gain : ℚ := 23.88663967611336

theorem bowls_sold_calculation :
  ∃ (x : ℕ), 
    x ≤ total_bowls ∧ 
    (x : ℚ) * sell_price_per_bowl = 
      (total_bowls : ℚ) * cost_per_bowl * (1 + percentage_gain / 100) ∧
    x = 108 := by
  sorry

end bowls_sold_calculation_l2945_294552


namespace radical_simplification_l2945_294540

theorem radical_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (75 * x) * Real.sqrt (2 * x) * Real.sqrt (14 * x) = 10 * x * Real.sqrt (21 * x) := by
  sorry

end radical_simplification_l2945_294540


namespace triangle_side_length_l2945_294527

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2, b = 3, and angle C is twice angle A, then the length of side c is √10. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a = 2 ∧ 
  b = 3 ∧ 
  C = 2 * A ∧  -- Angle C is twice angle A
  a / Real.sin A = b / Real.sin B ∧  -- Sine theorem
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C  -- Cosine theorem
  → c = Real.sqrt 10 := by
sorry

end triangle_side_length_l2945_294527


namespace star_properties_l2945_294509

-- Define the new operation "*"
def star (a b : ℚ) : ℚ := (2 + a) / b

-- Theorem statement
theorem star_properties :
  (star 4 (-3) = -2) ∧ (star 8 (star 4 3) = 5) := by
  sorry

end star_properties_l2945_294509


namespace smallest_divisible_by_18_and_24_l2945_294504

theorem smallest_divisible_by_18_and_24 : ∃ n : ℕ, n > 0 ∧ n % 18 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 18 = 0 → m % 24 = 0 → n ≤ m :=
by sorry

end smallest_divisible_by_18_and_24_l2945_294504


namespace actual_distance_traveled_l2945_294524

/-- Given a person's walking speeds and additional distance covered at higher speed,
    prove the actual distance traveled. -/
theorem actual_distance_traveled
  (original_speed : ℝ)
  (faster_speed : ℝ)
  (additional_distance : ℝ)
  (h1 : original_speed = 10)
  (h2 : faster_speed = 15)
  (h3 : additional_distance = 20)
  (h4 : faster_speed * (additional_distance / (faster_speed - original_speed)) =
        original_speed * (additional_distance / (faster_speed - original_speed)) + additional_distance) :
  original_speed * (additional_distance / (faster_speed - original_speed)) = 40 := by
sorry

end actual_distance_traveled_l2945_294524


namespace polar_curve_arc_length_l2945_294564

noncomputable def arcLength (ρ : Real → Real) (a b : Real) : Real :=
  ∫ x in a..b, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

theorem polar_curve_arc_length :
  let ρ : Real → Real := λ φ ↦ 8 * Real.cos φ
  arcLength ρ 0 (Real.pi / 4) = 2 * Real.pi := by
  sorry

end polar_curve_arc_length_l2945_294564


namespace power_of_four_l2945_294521

theorem power_of_four (n : ℕ) : 
  (2 * n + 7 + 2 = 31) → n = 11 := by
  sorry

end power_of_four_l2945_294521


namespace paulson_income_increase_paulson_income_increase_percentage_proof_l2945_294562

/-- Paulson's financial situation --/
structure PaulsonFinances where
  income : ℝ
  expenditure_ratio : ℝ
  income_increase_ratio : ℝ
  expenditure_increase_ratio : ℝ
  savings_increase_ratio : ℝ

/-- Theorem stating the relationship between Paulson's financial changes --/
theorem paulson_income_increase
  (p : PaulsonFinances)
  (h1 : p.expenditure_ratio = 0.75)
  (h2 : p.expenditure_increase_ratio = 0.1)
  (h3 : p.savings_increase_ratio = 0.4999999999999996)
  : p.income_increase_ratio = 0.2 := by
  sorry

/-- The main result: Paulson's income increase percentage --/
def paulson_income_increase_percentage : ℝ := 20

/-- Theorem proving the income increase percentage --/
theorem paulson_income_increase_percentage_proof
  (p : PaulsonFinances)
  (h1 : p.expenditure_ratio = 0.75)
  (h2 : p.expenditure_increase_ratio = 0.1)
  (h3 : p.savings_increase_ratio = 0.4999999999999996)
  : paulson_income_increase_percentage = 100 * p.income_increase_ratio := by
  sorry

end paulson_income_increase_paulson_income_increase_percentage_proof_l2945_294562
