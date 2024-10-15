import Mathlib

namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_l827_82780

/-- The number of coin tosses -/
def n : ℕ := 10

/-- The probability of no two heads appearing consecutively in n coin tosses -/
def prob_no_consecutive_heads (n : ℕ) : ℚ :=
  if n ≤ 1 then 1 else sorry

theorem prob_no_consecutive_heads_10 : 
  prob_no_consecutive_heads n = 9/64 := by sorry

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_l827_82780


namespace NUMINAMATH_CALUDE_shells_added_l827_82755

theorem shells_added (initial_amount final_amount : ℕ) 
  (h1 : initial_amount = 5)
  (h2 : final_amount = 17) :
  final_amount - initial_amount = 12 := by
  sorry

end NUMINAMATH_CALUDE_shells_added_l827_82755


namespace NUMINAMATH_CALUDE_anya_wins_l827_82784

/-- Represents the possible choices in rock-paper-scissors -/
inductive Choice
| Rock
| Paper
| Scissors

/-- Defines the outcome of a game -/
inductive Outcome
| Win
| Lose

/-- Determines the outcome of a game given two choices -/
def gameOutcome (player1 player2 : Choice) : Outcome :=
  match player1, player2 with
  | Choice.Rock, Choice.Scissors => Outcome.Win
  | Choice.Scissors, Choice.Paper => Outcome.Win
  | Choice.Paper, Choice.Rock => Outcome.Win
  | _, _ => Outcome.Lose

theorem anya_wins (
  total_rounds : Nat)
  (anya_rock anya_scissors anya_paper : Nat)
  (borya_rock borya_scissors borya_paper : Nat)
  (h_total : total_rounds = 25)
  (h_anya_rock : anya_rock = 12)
  (h_anya_scissors : anya_scissors = 6)
  (h_anya_paper : anya_paper = 7)
  (h_borya_rock : borya_rock = 13)
  (h_borya_scissors : borya_scissors = 9)
  (h_borya_paper : borya_paper = 3)
  (h_no_draws : anya_rock + anya_scissors + anya_paper = borya_rock + borya_scissors + borya_paper)
  (h_total_choices : anya_rock + anya_scissors + anya_paper = total_rounds) :
  ∃ (anya_wins : Nat), anya_wins = 19 ∧
    anya_wins ≤ min anya_rock borya_scissors +
               min anya_scissors borya_paper +
               min anya_paper borya_rock :=
by sorry


end NUMINAMATH_CALUDE_anya_wins_l827_82784


namespace NUMINAMATH_CALUDE_math_homework_pages_l827_82748

theorem math_homework_pages (total_pages reading_pages : ℕ) 
  (h1 : total_pages = 7)
  (h2 : reading_pages = 2) :
  total_pages - reading_pages = 5 := by
  sorry

end NUMINAMATH_CALUDE_math_homework_pages_l827_82748


namespace NUMINAMATH_CALUDE_jim_car_efficiency_l827_82704

/-- Calculates the fuel efficiency of a car given its tank capacity, remaining fuel ratio, and trip distance. -/
def fuel_efficiency (tank_capacity : ℚ) (remaining_ratio : ℚ) (trip_distance : ℚ) : ℚ :=
  trip_distance / (tank_capacity * (1 - remaining_ratio))

/-- Theorem stating that under the given conditions, the fuel efficiency is 5 miles per gallon. -/
theorem jim_car_efficiency :
  let tank_capacity : ℚ := 12
  let remaining_ratio : ℚ := 2/3
  let trip_distance : ℚ := 20
  fuel_efficiency tank_capacity remaining_ratio trip_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_jim_car_efficiency_l827_82704


namespace NUMINAMATH_CALUDE_smaller_root_comparison_l827_82779

theorem smaller_root_comparison (a a' b b' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-b / a < -b' / a') ↔ (b / a > b' / a') :=
sorry

end NUMINAMATH_CALUDE_smaller_root_comparison_l827_82779


namespace NUMINAMATH_CALUDE_diameter_height_ratio_l827_82737

/-- A cylinder whose lateral surface unfolds into a square -/
structure SquareUnfoldCylinder where
  diameter : ℝ
  height : ℝ
  square_unfold : height = π * diameter

theorem diameter_height_ratio (c : SquareUnfoldCylinder) :
  c.diameter / c.height = 1 / π := by
  sorry

end NUMINAMATH_CALUDE_diameter_height_ratio_l827_82737


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l827_82787

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 26 * x + k = 0) ↔ k = 5 := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l827_82787


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l827_82781

/-- A line is tangent to a curve if it intersects the curve at exactly one point and has the same slope as the curve at that point. -/
def is_tangent (f g : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = g x₀ ∧ (deriv f) x₀ = (deriv g) x₀

/-- The problem statement -/
theorem tangent_line_to_exponential_curve (a : ℝ) :
  is_tangent (fun x => x - 3) (fun x => Real.exp (x + a)) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l827_82781


namespace NUMINAMATH_CALUDE_equation_solutions_l827_82785

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 25 ↔ x = 7 ∨ x = -3) ∧
  (∀ x : ℝ, (x - 5)^2 = 2*(5 - x) ↔ x = 5 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l827_82785


namespace NUMINAMATH_CALUDE_min_value_sum_l827_82716

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  x + y ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_l827_82716


namespace NUMINAMATH_CALUDE_max_distance_Z₁Z₂_l827_82789

-- Define complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the conditions
def condition_z₁ : Prop := Complex.abs z₁ ≤ 2
def condition_z₂ : Prop := z₂ = Complex.mk 3 (-4)

-- Define the vector from Z₁ to Z₂
def vector_Z₁Z₂ : ℂ := z₂ - z₁

-- Theorem statement
theorem max_distance_Z₁Z₂ (hz₁ : condition_z₁ z₁) (hz₂ : condition_z₂ z₂) :
  ∃ (max_dist : ℝ), max_dist = 7 ∧ ∀ (z₁' : ℂ), condition_z₁ z₁' → Complex.abs (vector_Z₁Z₂ z₁' z₂) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_Z₁Z₂_l827_82789


namespace NUMINAMATH_CALUDE_a_6_value_l827_82757

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a_6_value
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_roots : a 4 * a 8 = 9 ∧ a 4 + a 8 = -11) :
  a 6 = -3 := by
sorry

end NUMINAMATH_CALUDE_a_6_value_l827_82757


namespace NUMINAMATH_CALUDE_cubic_polynomial_relation_l827_82783

/-- Given a cubic polynomial f and another cubic polynomial g satisfying certain conditions, 
    prove that g(4) = 105. -/
theorem cubic_polynomial_relation (f g : ℝ → ℝ) : 
  (∀ x, f x = x^3 - 2*x^2 + x + 1) →
  (∃ A r s t : ℝ, ∀ x, g x = A * (x - r^2) * (x - s^2) * (x - t^2)) →
  g 0 = -1 →
  (∀ x, f x = 0 ↔ g (x^2) = 0) →
  g 4 = 105 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_relation_l827_82783


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l827_82736

/-- Represents the cost and capacity of buses -/
structure BusType where
  cost : ℕ
  capacity : ℕ

/-- Represents the problem setup -/
structure BusRentalProblem where
  busA : BusType
  busB : BusType
  totalPeople : ℕ
  totalBuses : ℕ
  costOneEach : ℕ
  costTwoAThreeB : ℕ

/-- Calculates the total cost for a given number of each bus type -/
def totalCost (problem : BusRentalProblem) (numA : ℕ) : ℕ :=
  numA * problem.busA.cost + (problem.totalBuses - numA) * problem.busB.cost

/-- Calculates the total capacity for a given number of each bus type -/
def totalCapacity (problem : BusRentalProblem) (numA : ℕ) : ℕ :=
  numA * problem.busA.capacity + (problem.totalBuses - numA) * problem.busB.capacity

/-- The main theorem to prove -/
theorem minimum_cost_theorem (problem : BusRentalProblem) 
  (h1 : problem.busA.cost + problem.busB.cost = problem.costOneEach)
  (h2 : 2 * problem.busA.cost + 3 * problem.busB.cost = problem.costTwoAThreeB)
  (h3 : problem.busA.capacity = 15)
  (h4 : problem.busB.capacity = 25)
  (h5 : problem.totalPeople = 170)
  (h6 : problem.totalBuses = 8)
  (h7 : problem.costOneEach = 500)
  (h8 : problem.costTwoAThreeB = 1300) :
  ∃ (numA : ℕ), 
    numA ≤ problem.totalBuses ∧ 
    totalCapacity problem numA ≥ problem.totalPeople ∧
    totalCost problem numA = 2100 ∧
    ∀ (k : ℕ), k ≤ problem.totalBuses → 
      totalCapacity problem k ≥ problem.totalPeople → 
      totalCost problem k ≥ 2100 := by
  sorry


end NUMINAMATH_CALUDE_minimum_cost_theorem_l827_82736


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l827_82768

/-- The time it takes for the first pipe to fill the cistern -/
def t1 : ℝ := 10

/-- The time it takes for the third pipe to empty the cistern -/
def t3 : ℝ := 25

/-- The time it takes to fill the cistern when all pipes are opened simultaneously -/
def t_all : ℝ := 6.976744186046512

/-- The time it takes for the second pipe to fill the cistern -/
def t2 : ℝ := 11.994

theorem second_pipe_fill_time :
  ∃ (t2 : ℝ), t2 > 0 ∧ (1 / t1 + 1 / t2 - 1 / t3 = 1 / t_all) :=
sorry

end NUMINAMATH_CALUDE_second_pipe_fill_time_l827_82768


namespace NUMINAMATH_CALUDE_difference_of_squares_form_l827_82715

theorem difference_of_squares_form (x y : ℝ) :
  ∃ a b : ℝ, (-x + y) * (x + y) = -(a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_form_l827_82715


namespace NUMINAMATH_CALUDE_min_value_ab_l827_82714

/-- Given b > 0 and two perpendicular lines, prove the minimum value of ab is 2 -/
theorem min_value_ab (b : ℝ) (a : ℝ) (h1 : b > 0) 
  (h2 : ∀ x y : ℝ, (b^2 + 1) * x + a * y + 2 = 0 ↔ x - b^2 * y - 1 = 0) : 
  (∀ a' b' : ℝ, b' > 0 ∧ (∀ x y : ℝ, (b'^2 + 1) * x + a' * y + 2 = 0 ↔ x - b'^2 * y - 1 = 0) → a' * b' ≥ 2) ∧ 
  (∃ a₀ b₀ : ℝ, b₀ > 0 ∧ (∀ x y : ℝ, (b₀^2 + 1) * x + a₀ * y + 2 = 0 ↔ x - b₀^2 * y - 1 = 0) ∧ a₀ * b₀ = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_ab_l827_82714


namespace NUMINAMATH_CALUDE_product_maximized_at_11_l827_82772

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ

/-- Calculates the nth term of a geometric sequence -/
def nthTerm (gs : GeometricSequence) (n : ℕ) : ℝ :=
  gs.a₁ * gs.q ^ (n - 1)

/-- Calculates the product of the first n terms of a geometric sequence -/
def productFirstNTerms (gs : GeometricSequence) (n : ℕ) : ℝ :=
  (gs.a₁ ^ n) * (gs.q ^ (n * (n - 1) / 2))

/-- Theorem: The product of the first n terms is maximized when n = 11 for the given sequence -/
theorem product_maximized_at_11 (gs : GeometricSequence) 
    (h1 : gs.a₁ = 1536) (h2 : gs.q = -1/2) :
    ∀ k : ℕ, k ≠ 11 → productFirstNTerms gs 11 ≥ productFirstNTerms gs k := by
  sorry

end NUMINAMATH_CALUDE_product_maximized_at_11_l827_82772


namespace NUMINAMATH_CALUDE_expression_value_l827_82750

def opposite_numbers (a b : ℝ) : Prop := a = -b ∧ a ≠ 0 ∧ b ≠ 0

def reciprocals (c d : ℝ) : Prop := c * d = 1

def distance_from_one (m : ℝ) : Prop := |m - 1| = 2

theorem expression_value (a b c d m : ℝ) 
  (h1 : opposite_numbers a b) 
  (h2 : reciprocals c d) 
  (h3 : distance_from_one m) : 
  (a + b) * (c / d) + m * c * d + (b / a) = 2 ∨ 
  (a + b) * (c / d) + m * c * d + (b / a) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l827_82750


namespace NUMINAMATH_CALUDE_cameron_paper_count_l827_82773

theorem cameron_paper_count (initial_papers : ℕ) : 
  (initial_papers : ℚ) * (60 : ℚ) / 100 = 240 → initial_papers = 400 := by
  sorry

end NUMINAMATH_CALUDE_cameron_paper_count_l827_82773


namespace NUMINAMATH_CALUDE_trail_mix_nuts_l827_82747

theorem trail_mix_nuts (walnuts almonds : ℚ) 
  (h1 : walnuts = 0.25)
  (h2 : almonds = 0.25) : 
  walnuts + almonds = 0.50 := by
sorry

end NUMINAMATH_CALUDE_trail_mix_nuts_l827_82747


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l827_82709

theorem smallest_fraction_between (p q : ℕ+) : 
  (5 : ℚ) / 9 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (3 : ℚ) / 5 ∧ 
  (∀ (p' q' : ℕ+), (5 : ℚ) / 9 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (3 : ℚ) / 5 → q ≤ q') →
  q - p = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l827_82709


namespace NUMINAMATH_CALUDE_neglart_students_count_l827_82732

/-- Represents the number of toes on a Hoopit's hand -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of hands a Hoopit has -/
def hoopit_hands : ℕ := 4

/-- Represents the number of toes on a Neglart's hand -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands a Neglart has -/
def neglart_hands : ℕ := 5

/-- Represents the number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating that the number of Neglart students on the bus is 8 -/
theorem neglart_students_count : ∃ (n : ℕ), 
  n * (neglart_toes_per_hand * neglart_hands) + 
  hoopit_students * (hoopit_toes_per_hand * hoopit_hands) = total_toes ∧ 
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_neglart_students_count_l827_82732


namespace NUMINAMATH_CALUDE_bottle_capacity_proof_l827_82728

theorem bottle_capacity_proof (num_boxes : ℕ) (bottles_per_box : ℕ) (fill_ratio : ℚ) (total_volume : ℚ) :
  num_boxes = 10 →
  bottles_per_box = 50 →
  fill_ratio = 3/4 →
  total_volume = 4500 →
  (num_boxes * bottles_per_box * fill_ratio * (12 : ℚ) = total_volume) := by
  sorry

end NUMINAMATH_CALUDE_bottle_capacity_proof_l827_82728


namespace NUMINAMATH_CALUDE_planes_parallel_condition_l827_82733

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- Define the intersection operation for lines
variable (intersect : Line → Line → Set Point)

-- Define the specific lines and planes
variable (m n l₁ l₂ : Line) (α β : Plane) (M : Point)

-- State the theorem
theorem planes_parallel_condition 
  (h1 : subset m α)
  (h2 : subset n α)
  (h3 : subset l₁ β)
  (h4 : subset l₂ β)
  (h5 : intersect l₁ l₂ = {M})
  (h6 : parallel m l₁)
  (h7 : parallel n l₂) :
  planeParallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_condition_l827_82733


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l827_82761

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_4 : a 4 = -8)
  (h_8 : a 8 = 2) :
  a 12 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l827_82761


namespace NUMINAMATH_CALUDE_wedding_catering_ratio_l827_82720

/-- Represents the catering problem for Jenny's wedding --/
def CateringProblem (total_guests : ℕ) (steak_cost chicken_cost : ℚ) (total_budget : ℚ) : Prop :=
  ∃ (steak_guests chicken_guests : ℕ),
    steak_guests + chicken_guests = total_guests ∧
    steak_cost * steak_guests + chicken_cost * chicken_guests = total_budget ∧
    steak_guests = 3 * chicken_guests

/-- Theorem stating that the given conditions result in a 3:1 ratio of steak to chicken guests --/
theorem wedding_catering_ratio :
  CateringProblem 80 25 18 1860 :=
by
  sorry

end NUMINAMATH_CALUDE_wedding_catering_ratio_l827_82720


namespace NUMINAMATH_CALUDE_tylers_meal_combinations_l827_82719

theorem tylers_meal_combinations (meat_types : ℕ) (vegetable_types : ℕ) (dessert_types : ℕ) 
  (h1 : meat_types = 4)
  (h2 : vegetable_types = 5)
  (h3 : dessert_types = 5) :
  meat_types * (vegetable_types.choose 3) * (dessert_types.choose 2) = 400 := by
  sorry

end NUMINAMATH_CALUDE_tylers_meal_combinations_l827_82719


namespace NUMINAMATH_CALUDE_max_sum_of_counts_l827_82753

/-- Represents a sign in the table -/
inductive Sign
| Plus
| Minus

/-- Represents the table -/
def Table := Fin 30 → Fin 30 → Option Sign

/-- Count of pluses in the table -/
def count_pluses (t : Table) : ℕ := sorry

/-- Count of minuses in the table -/
def count_minuses (t : Table) : ℕ := sorry

/-- Check if a row has at most 17 signs -/
def row_valid (t : Table) (row : Fin 30) : Prop := sorry

/-- Check if a column has at most 17 signs -/
def col_valid (t : Table) (col : Fin 30) : Prop := sorry

/-- Calculate the sum of counts -/
def sum_of_counts (t : Table) : ℕ := sorry

/-- Main theorem -/
theorem max_sum_of_counts :
  ∀ (t : Table),
    count_pluses t = 162 →
    count_minuses t = 144 →
    (∀ row, row_valid t row) →
    (∀ col, col_valid t col) →
    sum_of_counts t ≤ 2592 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_counts_l827_82753


namespace NUMINAMATH_CALUDE_angle_trig_values_l827_82740

def l₁ (x y : ℝ) : Prop := x - y = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y - 3 = 0

def intersection_point (P : ℝ × ℝ) : Prop :=
  l₁ P.1 P.2 ∧ l₂ P.1 P.2

theorem angle_trig_values (α : ℝ) (P : ℝ × ℝ) :
  intersection_point P →
  Real.sin α = Real.sqrt 2 / 2 ∧
  Real.cos α = Real.sqrt 2 / 2 ∧
  Real.tan α = 1 :=
by sorry

end NUMINAMATH_CALUDE_angle_trig_values_l827_82740


namespace NUMINAMATH_CALUDE_inverse_89_mod_91_l827_82790

theorem inverse_89_mod_91 : ∃ x : ℕ, x < 91 ∧ (89 * x) % 91 = 1 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_inverse_89_mod_91_l827_82790


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l827_82796

/-- A geometric sequence with positive terms and common ratio 2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l827_82796


namespace NUMINAMATH_CALUDE_visited_neither_country_l827_82730

theorem visited_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) : 
  total = 60 → iceland = 35 → norway = 23 → both = 31 → 
  total - (iceland + norway - both) = 33 := by
sorry

end NUMINAMATH_CALUDE_visited_neither_country_l827_82730


namespace NUMINAMATH_CALUDE_object_is_cylinder_l827_82791

-- Define the possible shapes
inductive Shape
  | Rectangle
  | Cylinder
  | Cuboid
  | Cone

-- Define the types of views
inductive View
  | Rectangular
  | Circular

-- Define the object's properties
structure Object where
  frontView : View
  topView : View
  sideView : View

-- Theorem statement
theorem object_is_cylinder (obj : Object)
  (h1 : obj.frontView = View.Rectangular)
  (h2 : obj.sideView = View.Rectangular)
  (h3 : obj.topView = View.Circular) :
  Shape.Cylinder = 
    match obj.frontView, obj.topView, obj.sideView with
    | View.Rectangular, View.Circular, View.Rectangular => Shape.Cylinder
    | _, _, _ => Shape.Rectangle  -- default case, won't be reached
  := by sorry

end NUMINAMATH_CALUDE_object_is_cylinder_l827_82791


namespace NUMINAMATH_CALUDE_dillon_luca_sum_difference_l827_82738

def dillon_list := List.range 40

def replace_three_with_two (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

def luca_list := dillon_list.map replace_three_with_two

theorem dillon_luca_sum_difference :
  (dillon_list.sum - luca_list.sum) = 104 := by
  sorry

end NUMINAMATH_CALUDE_dillon_luca_sum_difference_l827_82738


namespace NUMINAMATH_CALUDE_sqrt_seven_squared_minus_four_l827_82739

theorem sqrt_seven_squared_minus_four : (Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_squared_minus_four_l827_82739


namespace NUMINAMATH_CALUDE_johnny_red_pencils_l827_82710

/-- The number of red pencils Johnny bought given the conditions of the problem -/
def total_red_pencils (total_packs : ℕ) (regular_red_per_pack : ℕ) (extra_red_packs : ℕ) (extra_red_per_pack : ℕ) : ℕ :=
  total_packs * regular_red_per_pack + extra_red_packs * extra_red_per_pack

/-- Theorem stating that Johnny bought 21 red pencils -/
theorem johnny_red_pencils :
  total_red_pencils 15 1 3 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_johnny_red_pencils_l827_82710


namespace NUMINAMATH_CALUDE_divisibility_by_72_l827_82769

theorem divisibility_by_72 (n : ℕ) : 
  ∃ d : ℕ, d < 10 ∧ 32235717 * 10 + d = n * 72 :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_72_l827_82769


namespace NUMINAMATH_CALUDE_complex_equation_solution_l827_82775

theorem complex_equation_solution (z : ℂ) (h : Complex.I * z = 2 - 4 * Complex.I) : 
  z = -4 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l827_82775


namespace NUMINAMATH_CALUDE_twelfth_number_with_digit_sum_12_l827_82792

/-- A function that returns the sum of the digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ+) : ℕ+ := sorry

/-- Theorem stating that the 12th number with digit sum 12 is 165 -/
theorem twelfth_number_with_digit_sum_12 : 
  nth_number_with_digit_sum_12 12 = 165 := by sorry

end NUMINAMATH_CALUDE_twelfth_number_with_digit_sum_12_l827_82792


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_union_equals_A_iff_m_in_range_l827_82731

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Theorem for part 1
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x | 2 ≤ x ∧ x ≤ 5}) ∧
  ((Aᶜ ∪ B 3) = {x | x < -2 ∨ 2 ≤ x}) :=
sorry

-- Theorem for part 2
theorem union_equals_A_iff_m_in_range :
  ∀ m : ℝ, (A ∪ B m = A) ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_union_equals_A_iff_m_in_range_l827_82731


namespace NUMINAMATH_CALUDE_pages_left_after_eleven_days_l827_82782

/-- Represents the number of pages left unread after reading for a given number of days -/
def pages_left (total_pages : ℕ) (pages_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pages - pages_per_day * days

/-- Theorem stating that reading 15 pages a day for 11 days from a 250-page book leaves 85 pages unread -/
theorem pages_left_after_eleven_days :
  pages_left 250 15 11 = 85 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_after_eleven_days_l827_82782


namespace NUMINAMATH_CALUDE_distribute_five_balls_to_three_children_l827_82706

/-- The number of ways to distribute n identical balls to k children,
    with each child receiving at least one ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 6 ways to distribute 5 identical balls to 3 children,
    with each child receiving at least one ball -/
theorem distribute_five_balls_to_three_children :
  distribute_balls 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_to_three_children_l827_82706


namespace NUMINAMATH_CALUDE_eight_round_game_probability_l827_82793

/-- Represents the probability of a specific outcome in an 8-round game -/
def game_probability (p1 p2 p3 : ℝ) (n1 n2 n3 : ℕ) : ℝ :=
  (p1^n1 * p2^n2 * p3^n3) * (Nat.choose 8 n1 * Nat.choose (8 - n1) n2)

theorem eight_round_game_probability :
  let p1 := (1 : ℝ) / 2
  let p2 := (1 : ℝ) / 3
  let p3 := (1 : ℝ) / 6
  game_probability p1 p2 p3 4 3 1 = 35 / 324 := by
  sorry

end NUMINAMATH_CALUDE_eight_round_game_probability_l827_82793


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l827_82763

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 7071 : ℤ) ≡ 3540 [ZMOD 15] ∧
  ∀ (y : ℕ), y > 0 → (y + 7071 : ℤ) ≡ 3540 [ZMOD 15] → x ≤ y ∧
  x = 9 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l827_82763


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_500_l827_82754

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_with_odd_factors_under_500 :
  ∃ n : ℕ, n < 500 ∧ has_odd_number_of_factors n ∧
  ∀ m : ℕ, m < 500 ∧ has_odd_number_of_factors m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_500_l827_82754


namespace NUMINAMATH_CALUDE_minimize_f_minimum_l827_82758

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  |7*x - 3*a + 8| + |5*x + 4*a - 6| + |x - a - 8| - 24

/-- Theorem stating that 82/43 is the value of a that minimizes the minimum value of f(x) -/
theorem minimize_f_minimum (a : ℝ) :
  (∀ x, f (82/43) x ≤ f a x) ∧ (∃ x, f (82/43) x < f a x) ∨ a = 82/43 := by
  sorry

#check minimize_f_minimum

end NUMINAMATH_CALUDE_minimize_f_minimum_l827_82758


namespace NUMINAMATH_CALUDE_quadruplet_babies_l827_82725

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1500)
  (h_triplets : ∃ (b c : ℕ), b = 3 * c)
  (h_twins : ∃ (a b : ℕ), a = 5 * b)
  (h_sum : ∃ (a b c : ℕ), 2 * a + 3 * b + 4 * c = total_babies) :
  ∃ (c : ℕ), 4 * c = 136 ∧ c * 4 ≤ total_babies := by
sorry

#eval 136

end NUMINAMATH_CALUDE_quadruplet_babies_l827_82725


namespace NUMINAMATH_CALUDE_line_x_intercept_l827_82766

/-- A line passing through two points (-3, 3) and (2, 10) has x-intercept -36/7 -/
theorem line_x_intercept : 
  let p₁ : ℝ × ℝ := (-3, 3)
  let p₂ : ℝ × ℝ := (2, 10)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  (0 - b) / m = -36/7 :=
by sorry

end NUMINAMATH_CALUDE_line_x_intercept_l827_82766


namespace NUMINAMATH_CALUDE_compact_connected_preserving_implies_continuous_l827_82702

/-- A function that maps compact sets to compact sets and connected sets to connected sets -/
def CompactConnectedPreserving (n m : ℕ) :=
  {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin m) |
    (∀ S : Set (EuclideanSpace ℝ (Fin n)), IsCompact S → IsCompact (f '' S)) ∧
    (∀ S : Set (EuclideanSpace ℝ (Fin n)), IsConnected S → IsConnected (f '' S))}

/-- Theorem: A function preserving compactness and connectedness is continuous -/
theorem compact_connected_preserving_implies_continuous
  {n m : ℕ} (f : CompactConnectedPreserving n m) :
  Continuous (f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin m)) :=
by sorry

end NUMINAMATH_CALUDE_compact_connected_preserving_implies_continuous_l827_82702


namespace NUMINAMATH_CALUDE_min_value_in_region_l827_82767

-- Define the region
def enclosed_region (x y : ℝ) : Prop :=
  abs x ≤ y ∧ y ≤ 2

-- Define the function to minimize
def f (x y : ℝ) : ℝ := 2 * x - y

-- Theorem statement
theorem min_value_in_region :
  ∃ (min : ℝ), min = -6 ∧
  ∀ (x y : ℝ), enclosed_region x y → f x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_in_region_l827_82767


namespace NUMINAMATH_CALUDE_concert_ticket_purchase_daria_concert_money_l827_82712

/-- Calculates the additional money needed to purchase concert tickets --/
theorem concert_ticket_purchase (num_tickets : ℕ) (original_price : ℚ) 
  (discount_percent : ℚ) (gift_card : ℚ) (current_money : ℚ) : ℚ :=
  let discounted_price := original_price * (1 - discount_percent / 100)
  let total_cost := num_tickets * discounted_price
  let after_gift_card := total_cost - gift_card
  after_gift_card - current_money

/-- Proves that Daria needs to earn $85 more for the concert tickets --/
theorem daria_concert_money : 
  concert_ticket_purchase 4 90 10 50 189 = 85 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_purchase_daria_concert_money_l827_82712


namespace NUMINAMATH_CALUDE_magnitude_of_5_minus_12i_l827_82752

theorem magnitude_of_5_minus_12i : Complex.abs (5 - 12 * Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_5_minus_12i_l827_82752


namespace NUMINAMATH_CALUDE_books_checked_out_wednesday_l827_82743

theorem books_checked_out_wednesday (initial_books : ℕ) (thursday_returned : ℕ) 
  (thursday_checked_out : ℕ) (friday_returned : ℕ) (final_books : ℕ) :
  initial_books = 98 →
  thursday_returned = 23 →
  thursday_checked_out = 5 →
  friday_returned = 7 →
  final_books = 80 →
  ∃ (wednesday_checked_out : ℕ),
    wednesday_checked_out = 43 ∧
    final_books = initial_books - wednesday_checked_out + thursday_returned - 
      thursday_checked_out + friday_returned :=
by
  sorry

end NUMINAMATH_CALUDE_books_checked_out_wednesday_l827_82743


namespace NUMINAMATH_CALUDE_last_tree_distance_l827_82705

/-- The distance between the last pair of trees in a yard with a specific planting pattern -/
theorem last_tree_distance (yard_length : ℕ) (num_trees : ℕ) (first_distance : ℕ) (increment : ℕ) :
  yard_length = 1200 →
  num_trees = 117 →
  first_distance = 5 →
  increment = 2 →
  (num_trees - 1) * (2 * first_distance + (num_trees - 2) * increment) ≤ 2 * yard_length →
  first_distance + (num_trees - 2) * increment = 235 :=
by sorry

end NUMINAMATH_CALUDE_last_tree_distance_l827_82705


namespace NUMINAMATH_CALUDE_stuffed_animal_sales_difference_stuffed_animal_sales_difference_proof_l827_82724

theorem stuffed_animal_sales_difference : ℕ → ℕ → ℕ → Prop :=
  fun thor jake quincy =>
    (jake = thor + 10) →
    (quincy = thor * 10) →
    (quincy = 200) →
    (quincy - jake = 170)

-- The proof would go here, but we're skipping it as requested
theorem stuffed_animal_sales_difference_proof :
  ∃ (thor jake quincy : ℕ), stuffed_animal_sales_difference thor jake quincy :=
sorry

end NUMINAMATH_CALUDE_stuffed_animal_sales_difference_stuffed_animal_sales_difference_proof_l827_82724


namespace NUMINAMATH_CALUDE_quiz_score_theorem_l827_82749

def quiz_scores : List ℕ := [91, 94, 88, 90, 101]
def target_mean : ℕ := 95
def num_quizzes : ℕ := 6
def required_score : ℕ := 106

theorem quiz_score_theorem :
  (List.sum quiz_scores + required_score) / num_quizzes = target_mean := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_theorem_l827_82749


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l827_82711

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * x + 1 > 4 * x - 6) → x ≤ 6 ∧ (3 * 6 + 1 > 4 * 6 - 6) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l827_82711


namespace NUMINAMATH_CALUDE_original_number_not_800_l827_82744

theorem original_number_not_800 : ¬(∃ x : ℝ, x * 10 = x + 720 ∧ x = 800) := by
  sorry

end NUMINAMATH_CALUDE_original_number_not_800_l827_82744


namespace NUMINAMATH_CALUDE_alcohol_concentration_reduction_specific_alcohol_reduction_l827_82788

/-- Calculates the percentage reduction in alcohol concentration when water is added to a solution. -/
theorem alcohol_concentration_reduction 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (water_added : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  let final_concentration := initial_alcohol / final_volume
  let reduction_percentage := (initial_concentration - final_concentration) / initial_concentration * 100
  by
    -- Proof goes here
    sorry

/-- The specific problem statement -/
theorem specific_alcohol_reduction : 
  alcohol_concentration_reduction 15 0.20 25 = 62.5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_alcohol_concentration_reduction_specific_alcohol_reduction_l827_82788


namespace NUMINAMATH_CALUDE_frisbee_game_probability_l827_82713

/-- The probability that Alice has the frisbee after three turns in the frisbee game. -/
theorem frisbee_game_probability : 
  let alice_toss_prob : ℚ := 2/3
  let alice_keep_prob : ℚ := 1/3
  let bob_toss_prob : ℚ := 1/4
  let bob_keep_prob : ℚ := 3/4
  let alice_has_frisbee_after_three_turns : ℚ := 
    alice_toss_prob * bob_keep_prob * bob_keep_prob +
    alice_keep_prob * alice_keep_prob
  alice_has_frisbee_after_three_turns = 35/72 :=
by sorry

end NUMINAMATH_CALUDE_frisbee_game_probability_l827_82713


namespace NUMINAMATH_CALUDE_gym_income_is_10800_l827_82794

/-- A gym charges its members twice a month and has a fixed number of members. -/
structure Gym where
  charge_per_half_month : ℕ
  charges_per_month : ℕ
  num_members : ℕ

/-- Calculate the monthly income of the gym -/
def monthly_income (g : Gym) : ℕ :=
  g.charge_per_half_month * g.charges_per_month * g.num_members

/-- Theorem stating that the gym's monthly income is $10,800 -/
theorem gym_income_is_10800 (g : Gym) 
  (h1 : g.charge_per_half_month = 18)
  (h2 : g.charges_per_month = 2)
  (h3 : g.num_members = 300) : 
  monthly_income g = 10800 := by
  sorry

end NUMINAMATH_CALUDE_gym_income_is_10800_l827_82794


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l827_82703

/-- Given a line with equation y = 3x + 1, this theorem states that its symmetric line
    with respect to the y-axis has the equation y = -3x + 1 -/
theorem symmetric_line_wrt_y_axis :
  ∀ (x y : ℝ), (∃ (m n : ℝ), n = 3 * m + 1 ∧ x + m = 0 ∧ y = n) →
  y = -3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l827_82703


namespace NUMINAMATH_CALUDE_base6_45_equals_decimal_29_l827_82741

-- Define a function to convert a base-6 number to decimal
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

-- Theorem statement
theorem base6_45_equals_decimal_29 :
  base6ToDecimal [5, 4] = 29 := by
  sorry

end NUMINAMATH_CALUDE_base6_45_equals_decimal_29_l827_82741


namespace NUMINAMATH_CALUDE_specific_arrangement_eq_3456_l827_82734

/-- The number of ways to arrange players from different teams in a row -/
def arrange_players (num_teams : ℕ) (team_sizes : List ℕ) : ℕ :=
  (Nat.factorial num_teams) * (team_sizes.map Nat.factorial).prod

/-- The specific arrangement for the given problem -/
def specific_arrangement : ℕ :=
  arrange_players 4 [3, 2, 3, 2]

/-- Theorem stating that the specific arrangement equals 3456 -/
theorem specific_arrangement_eq_3456 : specific_arrangement = 3456 := by
  sorry

end NUMINAMATH_CALUDE_specific_arrangement_eq_3456_l827_82734


namespace NUMINAMATH_CALUDE_quadratic_complex_conjugate_roots_l827_82797

theorem quadratic_complex_conjugate_roots (a b : ℝ) : 
  (∃ x y : ℝ, (Complex.I * x + y) ^ 2 + (6 + Complex.I * a) * (Complex.I * x + y) + (15 + Complex.I * b) = 0 ∧
               (Complex.I * (-x) + y) ^ 2 + (6 + Complex.I * a) * (Complex.I * (-x) + y) + (15 + Complex.I * b) = 0) →
  a = 0 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_complex_conjugate_roots_l827_82797


namespace NUMINAMATH_CALUDE_sales_volume_estimate_l827_82723

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := -10 * x + 200

-- State the theorem
theorem sales_volume_estimate :
  ∃ ε > 0, |regression_equation 10 - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_sales_volume_estimate_l827_82723


namespace NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l827_82751

theorem polar_to_rectangular_transformation (x y : ℝ) (h : x = 12 ∧ y = 5) :
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r^2 * Real.cos (3 * θ), r^2 * Real.sin (3 * θ)) = (-494004 / 2197, 4441555 / 2197) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l827_82751


namespace NUMINAMATH_CALUDE_florist_roses_l827_82799

theorem florist_roses (initial : ℕ) : 
  (initial - 3 + 34 = 36) → initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l827_82799


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l827_82759

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l827_82759


namespace NUMINAMATH_CALUDE_stephens_number_l827_82707

theorem stephens_number : ∃! n : ℕ, 
  9000 ≤ n ∧ n ≤ 15000 ∧ 
  n % 216 = 0 ∧ 
  n % 55 = 0 ∧ 
  n = 11880 := by sorry

end NUMINAMATH_CALUDE_stephens_number_l827_82707


namespace NUMINAMATH_CALUDE_rectangular_field_width_l827_82798

theorem rectangular_field_width (length width : ℝ) : 
  length = 24 ∧ length = 2 * width - 3 → width = 13.5 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l827_82798


namespace NUMINAMATH_CALUDE_square_difference_of_product_and_sum_l827_82765

theorem square_difference_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_product_and_sum_l827_82765


namespace NUMINAMATH_CALUDE_valid_set_iff_ge_four_l827_82735

/-- A set of positive integers satisfying the given conditions -/
def ValidSet (n : ℕ) (S : Finset ℕ) : Prop :=
  (S.card = n) ∧
  (∀ x ∈ S, x > 0 ∧ x < 2^(n-1)) ∧
  (∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ ∅ → B ≠ ∅ → A ≠ B →
    (A.sum id ≠ B.sum id))

/-- The main theorem stating the existence of a valid set if and only if n ≥ 4 -/
theorem valid_set_iff_ge_four (n : ℕ) :
  (∃ S : Finset ℕ, ValidSet n S) ↔ n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_set_iff_ge_four_l827_82735


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l827_82746

/-- Given two vectors a and b in ℝ³, where a = (-2, 1, 5) and b = (6, m, -15),
    if a and b are parallel, then m = -3. -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 3 → ℝ := ![(-2 : ℝ), 1, 5]
  let b : Fin 3 → ℝ := ![6, m, -15]
  (∃ (t : ℝ), b = fun i => t * a i) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l827_82746


namespace NUMINAMATH_CALUDE_brand_a_millet_percentage_l827_82777

/-- Represents the composition of a birdseed brand -/
structure BirdseedBrand where
  millet : ℝ
  sunflower : ℝ
  composition_sum : millet + sunflower = 100

/-- Represents a mix of two birdseed brands -/
structure BirdseedMix where
  brand_a : BirdseedBrand
  brand_b : BirdseedBrand
  proportion_a : ℝ
  proportion_b : ℝ
  proportions_sum : proportion_a + proportion_b = 100
  sunflower_percent : ℝ
  sunflower_balance : proportion_a / 100 * brand_a.sunflower + proportion_b / 100 * brand_b.sunflower = sunflower_percent

/-- Theorem stating that Brand A has 40% millet given the problem conditions -/
theorem brand_a_millet_percentage 
  (brand_a : BirdseedBrand)
  (brand_b : BirdseedBrand)
  (mix : BirdseedMix)
  (ha : brand_a.sunflower = 60)
  (hb1 : brand_b.millet = 65)
  (hb2 : brand_b.sunflower = 35)
  (hm1 : mix.sunflower_percent = 50)
  (hm2 : mix.proportion_a = 60)
  (hm3 : mix.brand_a = brand_a)
  (hm4 : mix.brand_b = brand_b) :
  brand_a.millet = 40 :=
sorry

end NUMINAMATH_CALUDE_brand_a_millet_percentage_l827_82777


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l827_82700

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define the sum of the first n terms of b_n
def T (n : ℕ) : ℚ := n / (2 * n + 1)

theorem arithmetic_sequence_problem :
  S 9 = 81 ∧ a 3 + a 5 = 14 →
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∀ n : ℕ, T n = n / (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l827_82700


namespace NUMINAMATH_CALUDE_f_one_lower_bound_l827_82727

/-- Given a quadratic function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞),
    prove that f(1) ≥ 25. -/
theorem f_one_lower_bound
  (f : ℝ → ℝ)
  (m : ℝ)
  (h1 : ∀ x, f x = 4 * x^2 - m * x + 5)
  (h2 : ∀ x y, x ≥ -2 → y ≥ -2 → x < y → f x < f y) :
  f 1 ≥ 25 := by
sorry

end NUMINAMATH_CALUDE_f_one_lower_bound_l827_82727


namespace NUMINAMATH_CALUDE_adult_ticket_price_is_five_l827_82771

/-- Represents the ticket sales for a baseball game at a community center -/
structure TicketSales where
  total_tickets : ℕ
  adult_tickets : ℕ
  child_ticket_price : ℕ
  total_revenue : ℕ

/-- The price of an adult ticket given the ticket sales information -/
def adult_ticket_price (sales : TicketSales) : ℕ :=
  (sales.total_revenue - (sales.total_tickets - sales.adult_tickets) * sales.child_ticket_price) / sales.adult_tickets

/-- Theorem stating that the adult ticket price is $5 given the specific sales information -/
theorem adult_ticket_price_is_five :
  let sales : TicketSales := {
    total_tickets := 85,
    adult_tickets := 35,
    child_ticket_price := 2,
    total_revenue := 275
  }
  adult_ticket_price sales = 5 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_is_five_l827_82771


namespace NUMINAMATH_CALUDE_sum_of_squares_minus_fourth_power_l827_82760

theorem sum_of_squares_minus_fourth_power (a b : ℕ+) : 
  a^2 - b^4 = 2009 → a + b = 47 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_minus_fourth_power_l827_82760


namespace NUMINAMATH_CALUDE_old_bridge_traffic_l827_82778

/-- Represents the number of vehicles passing through the old bridge every month -/
def old_bridge_monthly_traffic : ℕ := sorry

/-- Represents the number of vehicles passing through the new bridge every month -/
def new_bridge_monthly_traffic : ℕ := sorry

/-- The new bridge has twice the capacity of the old one -/
axiom new_bridge_capacity : new_bridge_monthly_traffic = 2 * old_bridge_monthly_traffic

/-- The number of vehicles passing through the new bridge increased by 60% compared to the old bridge -/
axiom traffic_increase : new_bridge_monthly_traffic = old_bridge_monthly_traffic + (60 * old_bridge_monthly_traffic) / 100

/-- The total number of vehicles passing through both bridges in a year is 62,400 -/
axiom total_yearly_traffic : 12 * (old_bridge_monthly_traffic + new_bridge_monthly_traffic) = 62400

theorem old_bridge_traffic : old_bridge_monthly_traffic = 2000 :=
sorry

end NUMINAMATH_CALUDE_old_bridge_traffic_l827_82778


namespace NUMINAMATH_CALUDE_dragon_unicorn_equivalence_l827_82776

theorem dragon_unicorn_equivalence (R U : Prop) :
  (R → U) ↔ ((¬U → ¬R) ∧ (¬R ∨ U)) :=
sorry

end NUMINAMATH_CALUDE_dragon_unicorn_equivalence_l827_82776


namespace NUMINAMATH_CALUDE_cos_sum_seventh_roots_l827_82762

theorem cos_sum_seventh_roots : 
  Real.cos (2 * π / 7) + Real.cos (4 * π / 7) + Real.cos (6 * π / 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_seventh_roots_l827_82762


namespace NUMINAMATH_CALUDE_max_value_AMC_l827_82726

theorem max_value_AMC (A M C : ℕ) (h : A + M + C = 12) :
  A * M * C + A * M + M * C + C * A ≤ 112 :=
by sorry

end NUMINAMATH_CALUDE_max_value_AMC_l827_82726


namespace NUMINAMATH_CALUDE_square_difference_theorem_l827_82718

theorem square_difference_theorem (x y : ℚ) 
  (sum_eq : x + y = 8/15) 
  (diff_eq : x - y = 1/105) : 
  x^2 - y^2 = 8/1575 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l827_82718


namespace NUMINAMATH_CALUDE_f_monotonicity_and_max_root_difference_l827_82756

noncomputable def f (x : ℝ) : ℝ := 4 * x - x^4

theorem f_monotonicity_and_max_root_difference :
  (∀ x y, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x > f y) ∧
  (∀ a x₁ x₂, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ 1 < x₂ → x₂ - 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_max_root_difference_l827_82756


namespace NUMINAMATH_CALUDE_x_pow_zero_eq_one_f_eq_S_l827_82795

-- Define the functions
def f (x : ℝ) := x^2
def S (t : ℝ) := t^2

-- Theorem statements
theorem x_pow_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem f_eq_S : ∀ x : ℝ, f x = S x := by sorry

end NUMINAMATH_CALUDE_x_pow_zero_eq_one_f_eq_S_l827_82795


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_5_l827_82701

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + a

-- State the theorem
theorem max_value_implies_a_equals_5 :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, f a x ≤ 5) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 5) → a = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_5_l827_82701


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l827_82721

theorem solution_set_equivalence (x : ℝ) :
  (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l827_82721


namespace NUMINAMATH_CALUDE_parallel_segment_ratio_sum_l827_82722

/-- Represents a triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- Represents the parallel line segments drawn from a point P inside the triangle -/
structure ParallelSegments where
  a' : ℝ
  b' : ℝ
  c' : ℝ
  ha' : a' > 0
  hb' : b' > 0
  hc' : c' > 0

/-- Theorem: For any triangle and any point P inside it, the sum of ratios of 
    parallel segments to corresponding sides is always 1 -/
theorem parallel_segment_ratio_sum (t : Triangle) (p : ParallelSegments) :
  p.a' / t.a + p.b' / t.b + p.c' / t.c = 1 := by sorry

end NUMINAMATH_CALUDE_parallel_segment_ratio_sum_l827_82722


namespace NUMINAMATH_CALUDE_egg_pack_size_l827_82774

/-- The number of rotten eggs in the pack -/
def rotten_eggs : ℕ := 3

/-- The probability of choosing 2 rotten eggs -/
def prob_two_rotten : ℚ := 47619047619047615 / 10000000000000000

/-- The total number of eggs in the pack -/
def total_eggs : ℕ := 36

/-- Theorem stating that given the number of rotten eggs and the probability of choosing 2 rotten eggs, 
    the total number of eggs in the pack is 36 -/
theorem egg_pack_size :
  (rotten_eggs : ℚ) / total_eggs * (rotten_eggs - 1 : ℚ) / (total_eggs - 1) = prob_two_rotten :=
sorry

end NUMINAMATH_CALUDE_egg_pack_size_l827_82774


namespace NUMINAMATH_CALUDE_old_man_coins_l827_82764

theorem old_man_coins (x y : ℕ) (h1 : x ≠ y) (h2 : x^2 - y^2 = 49 * (x - y)) : x + y = 49 := by
  sorry

end NUMINAMATH_CALUDE_old_man_coins_l827_82764


namespace NUMINAMATH_CALUDE_angle_problem_l827_82742

theorem angle_problem (angle1 angle2 angle3 angle4 : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = angle4 →
  angle1 + 50 + 60 = 180 →
  angle4 = 35 := by
sorry

end NUMINAMATH_CALUDE_angle_problem_l827_82742


namespace NUMINAMATH_CALUDE_m_range_l827_82708

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Theorem statement
theorem m_range (m : ℝ) : p m ∧ q m → m ∈ Set.Ioo (-2 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l827_82708


namespace NUMINAMATH_CALUDE_transformed_graph_point_l827_82729

theorem transformed_graph_point (f : ℝ → ℝ) (h : f 12 = 5) :
  ∃ (x y : ℝ), 1.5 * y = (f (3 * x) + 3) / 3 ∧ x = 4 ∧ y = 16 / 9 ∧ x + y = 52 / 9 := by
  sorry

end NUMINAMATH_CALUDE_transformed_graph_point_l827_82729


namespace NUMINAMATH_CALUDE_factorization_equalities_l827_82786

theorem factorization_equalities (x y : ℝ) : 
  (2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2)) ∧ 
  (x^3 - 5 * x^2 + 6 * x = x * (x - 3) * (x - 2)) ∧ 
  (4 * x^4 * y^2 - 5 * x^2 * y^2 - 9 * y^2 = y^2 * (2 * x + 3) * (2 * x - 3) * (x^2 + 1)) ∧ 
  (3 * x^2 - 10 * x * y + 3 * y^2 = (3 * x - y) * (x - 3 * y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equalities_l827_82786


namespace NUMINAMATH_CALUDE_statement_true_for_lines_statement_true_for_planes_statement_true_cases_l827_82745

-- Define a type for geometric objects (lines or planes)
inductive GeometricObject
| Line
| Plane

-- Define a parallel relation
def parallel (a b : GeometricObject) : Prop := sorry

-- Define the statement we want to prove
def statement (x y z : GeometricObject) : Prop :=
  (parallel x z ∧ parallel y z) ∧ ¬(parallel x y)

-- Theorem for the case when all objects are lines
theorem statement_true_for_lines :
  ∃ (x y z : GeometricObject), 
    x = GeometricObject.Line ∧ 
    y = GeometricObject.Line ∧ 
    z = GeometricObject.Line ∧ 
    statement x y z := by sorry

-- Theorem for the case when all objects are planes
theorem statement_true_for_planes :
  ∃ (x y z : GeometricObject), 
    x = GeometricObject.Plane ∧ 
    y = GeometricObject.Plane ∧ 
    z = GeometricObject.Plane ∧ 
    statement x y z := by sorry

-- Main theorem combining both cases
theorem statement_true_cases :
  (∃ (x y z : GeometricObject), 
    x = GeometricObject.Line ∧ 
    y = GeometricObject.Line ∧ 
    z = GeometricObject.Line ∧ 
    statement x y z) ∧
  (∃ (x y z : GeometricObject), 
    x = GeometricObject.Plane ∧ 
    y = GeometricObject.Plane ∧ 
    z = GeometricObject.Plane ∧ 
    statement x y z) := by sorry

end NUMINAMATH_CALUDE_statement_true_for_lines_statement_true_for_planes_statement_true_cases_l827_82745


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l827_82770

/-- Given a rhombus with one diagonal of length 30 meters and an area of 600 square meters,
    prove that the length of the other diagonal is 40 meters. -/
theorem rhombus_other_diagonal (d₁ : ℝ) (d₂ : ℝ) (area : ℝ) 
    (h₁ : d₁ = 30)
    (h₂ : area = 600)
    (h₃ : area = d₁ * d₂ / 2) : 
  d₂ = 40 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l827_82770


namespace NUMINAMATH_CALUDE_carpenters_for_chairs_l827_82717

/-- Represents the number of carpenters needed to make a certain number of chairs in a given number of days. -/
def carpenters_needed (initial_carpenters : ℕ) (initial_chairs : ℕ) (target_chairs : ℕ) : ℕ :=
  (initial_carpenters * target_chairs + initial_chairs - 1) / initial_chairs

/-- Proves that 12 carpenters are needed to make 75 chairs in 10 days, given that 8 carpenters can make 50 chairs in 10 days. -/
theorem carpenters_for_chairs : carpenters_needed 8 50 75 = 12 := by
  sorry

end NUMINAMATH_CALUDE_carpenters_for_chairs_l827_82717
