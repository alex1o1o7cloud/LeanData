import Mathlib

namespace fraction_well_defined_l1014_101454

theorem fraction_well_defined (x : ℝ) (h : x ≠ 2) : 2 * x - 4 ≠ 0 := by
  sorry

#check fraction_well_defined

end fraction_well_defined_l1014_101454


namespace unit_rectangle_coverage_l1014_101464

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle parallel to the axes -/
structure Rectangle where
  left : ℝ
  bottom : ℝ
  width : ℝ
  height : ℝ

/-- The theorem stating that 1821 points can be arranged to cover all unit-area rectangles in a 15x15 square -/
theorem unit_rectangle_coverage : ∃ (points : Finset Point),
  (points.card = 1821) ∧ 
  (∀ p : Point, p ∈ points → p.x ≥ 0 ∧ p.x ≤ 15 ∧ p.y ≥ 0 ∧ p.y ≤ 15) ∧
  (∀ r : Rectangle, 
    r.left ≥ 0 ∧ r.left + r.width ≤ 15 ∧ 
    r.bottom ≥ 0 ∧ r.bottom + r.height ≤ 15 ∧
    r.width * r.height = 1 →
    ∃ p : Point, p ∈ points ∧ 
      p.x ≥ r.left ∧ p.x ≤ r.left + r.width ∧
      p.y ≥ r.bottom ∧ p.y ≤ r.bottom + r.height) := by
  sorry


end unit_rectangle_coverage_l1014_101464


namespace transformed_graph_point_l1014_101410

theorem transformed_graph_point (f : ℝ → ℝ) (h : f 12 = 5) :
  ∃ (x y : ℝ), 1.5 * y = (f (3 * x) + 3) / 3 ∧ x = 4 ∧ y = 16 / 9 ∧ x + y = 52 / 9 := by
  sorry

end transformed_graph_point_l1014_101410


namespace set_operations_and_subset_l1014_101475

-- Define the sets A, B, and M
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}
def M (k : ℝ) : Set ℝ := {x | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- Theorem statement
theorem set_operations_and_subset :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) ∧
  (∀ k, M k ⊆ A ↔ k < -5/2 ∨ k > 1) := by sorry

end set_operations_and_subset_l1014_101475


namespace pizza_flour_calculation_l1014_101470

theorem pizza_flour_calculation (bases : ℕ) (total_flour : ℚ) : 
  bases = 15 → total_flour = 8 → (total_flour / bases : ℚ) = 8/15 := by
  sorry

end pizza_flour_calculation_l1014_101470


namespace tangent_line_equation_l1014_101443

/-- The equation of the tangent line to the curve y = x^3 - 3x^2 + 1 at the point (1, -1) is y = -3x + 2 -/
theorem tangent_line_equation (x y : ℝ) : 
  y = x^3 - 3*x^2 + 1 → -- curve equation
  (1 : ℝ)^3 - 3*(1 : ℝ)^2 + 1 = -1 → -- point (1, -1) satisfies the curve equation
  ∃ (m b : ℝ), 
    (∀ t, y = m*t + b → (t - 1)*(3*(1 : ℝ)^2 - 6*(1 : ℝ)) = y + 1) ∧ -- point-slope form of tangent line
    m = -3 ∧ b = 2 -- coefficients of the tangent line equation
  := by sorry

end tangent_line_equation_l1014_101443


namespace equation_solution_l1014_101493

theorem equation_solution : ∃! x : ℝ, (27 : ℝ) ^ (x - 2) / (9 : ℝ) ^ (x - 1) = (81 : ℝ) ^ (3 * x - 1) := by
  sorry

end equation_solution_l1014_101493


namespace chocolate_bar_problem_l1014_101453

/-- The number of chocolate bars Min bought -/
def min_bars : ℕ := 67

/-- The initial number of chocolate bars in the store -/
def initial_bars : ℕ := 376

/-- The number of chocolate bars Max bought -/
def max_bars : ℕ := min_bars + 41

/-- The number of chocolate bars remaining in the store after purchases -/
def remaining_bars : ℕ := initial_bars - min_bars - max_bars

theorem chocolate_bar_problem :
  min_bars = 67 ∧
  initial_bars = 376 ∧
  max_bars = min_bars + 41 ∧
  remaining_bars = 3 * min_bars :=
sorry

end chocolate_bar_problem_l1014_101453


namespace octahedron_triangle_count_l1014_101450

/-- The number of vertices in a regular octahedron -/
def octahedron_vertices : ℕ := 6

/-- The number of distinct triangles that can be formed by connecting three different vertices of a regular octahedron -/
def octahedron_triangles : ℕ := Nat.choose octahedron_vertices 3

theorem octahedron_triangle_count : octahedron_triangles = 20 := by
  sorry

end octahedron_triangle_count_l1014_101450


namespace quadratic_equation_solutions_l1014_101429

theorem quadratic_equation_solutions : {x : ℝ | x^2 = x} = {0, 1} := by sorry

end quadratic_equation_solutions_l1014_101429


namespace line_condition_perpendicular_to_x_axis_equal_intercepts_l1014_101451

-- Define the equation coefficients as functions of m
def a (m : ℝ) := m^2 - 2*m - 3
def b (m : ℝ) := 2*m^2 + m - 1
def c (m : ℝ) := 5 - 2*m

-- Theorem 1: Condition for the equation to represent a line
theorem line_condition (m : ℝ) : 
  (a m = 0 ∧ b m = 0) ↔ m = -1 :=
sorry

-- Theorem 2: Condition for the line to be perpendicular to x-axis
theorem perpendicular_to_x_axis (m : ℝ) :
  (a m ≠ 0 ∧ b m = 0) ↔ (m^2 - 2*m - 3 ≠ 0 ∧ 2*m^2 + m - 1 = 0) :=
sorry

-- Theorem 3: Condition for equal intercepts on both axes
theorem equal_intercepts (m : ℝ) :
  (m ≠ 5/2 → (2*m - 5)/(m^2 - 2*m - 3) = (2*m - 5)/(2*m^2 + m - 1)) ↔ m = 5/2 :=
sorry

end line_condition_perpendicular_to_x_axis_equal_intercepts_l1014_101451


namespace grape_juice_theorem_l1014_101461

/-- Represents a fruit drink composition -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ

/-- Calculates the amount of grape juice in the drink -/
def grape_juice_amount (drink : FruitDrink) : ℝ :=
  drink.total - (drink.orange_percent * drink.total + drink.watermelon_percent * drink.total)

/-- Theorem: The amount of grape juice in the specified drink is 70 ounces -/
theorem grape_juice_theorem (drink : FruitDrink) 
    (h1 : drink.total = 200)
    (h2 : drink.orange_percent = 0.25)
    (h3 : drink.watermelon_percent = 0.40) : 
  grape_juice_amount drink = 70 := by
  sorry

#eval grape_juice_amount { total := 200, orange_percent := 0.25, watermelon_percent := 0.40 }

end grape_juice_theorem_l1014_101461


namespace expression_value_l1014_101412

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

end expression_value_l1014_101412


namespace monotonic_cubic_function_parameter_range_l1014_101415

/-- Given that f(x) = -x^3 + 2ax^2 - x - 3 is a monotonic function on ℝ, 
    prove that a ∈ [-√3/2, √3/2] -/
theorem monotonic_cubic_function_parameter_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + 2*a*x^2 - x - 3)) →
  a ∈ Set.Icc (-Real.sqrt 3 / 2) (Real.sqrt 3 / 2) :=
sorry

end monotonic_cubic_function_parameter_range_l1014_101415


namespace hyperbola_eccentricity_range_l1014_101419

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  m : ℝ
  equation : (x : ℝ) → (y : ℝ) → Prop :=
    fun x y => x^2 / (m + 1) - y^2 / (3 - m) = 1

/-- Theorem statement for the hyperbola eccentricity problem -/
theorem hyperbola_eccentricity_range 
  (C : Hyperbola) 
  (F : Point) 
  (k : ℝ) 
  (A B P Q : Point) 
  (h1 : F.x < 0) -- F is the left focus
  (h2 : k ≥ Real.sqrt 3) -- Line slope condition
  (h3 : C.equation A.x A.y ∧ C.equation B.x B.y) -- A and B are on the hyperbola
  (h4 : P.x = (A.x + F.x) / 2 ∧ P.y = (A.y + F.y) / 2) -- P is midpoint of AF
  (h5 : Q.x = (B.x + F.x) / 2 ∧ Q.y = (B.y + F.y) / 2) -- Q is midpoint of BF
  (h6 : (P.y - 0) * (Q.y - 0) = -(P.x - 0) * (Q.x - 0)) -- OP ⊥ OQ
  : ∃ (e : ℝ), e ≥ Real.sqrt 3 + 1 ∧ 
    ∀ (e' : ℝ), e' ≥ Real.sqrt 3 + 1 → 
    ∃ (C' : Hyperbola), C'.m = C.m ∧ 
    (∃ (F' A' B' P' Q' : Point) (k' : ℝ), 
      F'.x < 0 ∧ 
      k' ≥ Real.sqrt 3 ∧
      C'.equation A'.x A'.y ∧ C'.equation B'.x B'.y ∧
      P'.x = (A'.x + F'.x) / 2 ∧ P'.y = (A'.y + F'.y) / 2 ∧
      Q'.x = (B'.x + F'.x) / 2 ∧ Q'.y = (B'.y + F'.y) / 2 ∧
      (P'.y - 0) * (Q'.y - 0) = -(P'.x - 0) * (Q'.x - 0) ∧
      e' = C'.m) := by sorry

end hyperbola_eccentricity_range_l1014_101419


namespace sum_of_digits_0_to_2012_l1014_101414

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers in a range -/
def sumOfDigitsInRange (start finish : ℕ) : ℕ :=
  (List.range (finish - start + 1)).map (fun i => sumOfDigits (start + i))
    |> List.sum

/-- The sum of the digits of all numbers from 0 to 2012 is 28077 -/
theorem sum_of_digits_0_to_2012 :
    sumOfDigitsInRange 0 2012 = 28077 := by sorry

end sum_of_digits_0_to_2012_l1014_101414


namespace smallest_integer_quadratic_inequality_l1014_101444

theorem smallest_integer_quadratic_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 36 ≤ 0 → n ≤ m) ∧ n^2 - 13*n + 36 ≤ 0 ∧ n = 4 :=
by sorry

end smallest_integer_quadratic_inequality_l1014_101444


namespace modulus_of_complex_number_l1014_101496

theorem modulus_of_complex_number (z : ℂ) (h : z = Complex.I * (2 - Complex.I)) : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_complex_number_l1014_101496


namespace factors_of_96_with_square_sum_208_l1014_101489

theorem factors_of_96_with_square_sum_208 : 
  ∃ (a b : ℕ+), (a * b = 96) ∧ (a^2 + b^2 = 208) := by sorry

end factors_of_96_with_square_sum_208_l1014_101489


namespace anya_wins_l1014_101406

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


end anya_wins_l1014_101406


namespace legs_exceed_twice_heads_by_30_l1014_101438

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ := sorry

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 15

/-- Calculates the total number of legs in the group -/
def total_legs : ℕ := 2 * num_ducks + 4 * num_cows

/-- Calculates the total number of heads in the group -/
def total_heads : ℕ := num_ducks + num_cows

/-- Theorem stating that the number of legs exceeds twice the number of heads by 30 -/
theorem legs_exceed_twice_heads_by_30 : total_legs = 2 * total_heads + 30 := by
  sorry

end legs_exceed_twice_heads_by_30_l1014_101438


namespace ellipse_properties_l1014_101494

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with slope 1 -/
structure Line where
  c : ℝ

/-- Theorem about ellipse properties -/
theorem ellipse_properties (E : Ellipse) (F₁ F₂ A B P : Point) (l : Line) :
  -- Line l passes through F₁ and has slope 1
  F₁.x = -E.a.sqrt^2 - E.b^2 ∧ F₁.y = 0 ∧ l.c = F₁.x →
  -- A and B are intersection points of l and E
  (A.x^2 / E.a^2 + A.y^2 / E.b^2 = 1) ∧ (B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1) ∧
  A.x = A.y - l.c ∧ B.x = B.y - l.c →
  -- |AF₂|, |AB|, |BF₂| form arithmetic sequence
  2 * ((A.x - B.x)^2 + (A.y - B.y)^2) = 
    ((A.x - F₂.x)^2 + (A.y - F₂.y)^2) + ((B.x - F₂.x)^2 + (B.y - F₂.y)^2) →
  -- P(0, -1) satisfies |PA| = |PB|
  P.x = 0 ∧ P.y = -1 ∧
  (P.x - A.x)^2 + (P.y - A.y)^2 = (P.x - B.x)^2 + (P.y - B.y)^2 →
  -- Eccentricity is √2/2 and equation is x^2/18 + y^2/9 = 1
  (E.a^2 - E.b^2) / E.a^2 = 1/2 ∧ E.a^2 = 18 ∧ E.b^2 = 9 := by
sorry

end ellipse_properties_l1014_101494


namespace division_remainder_l1014_101492

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 271 →
  divisor = 30 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
sorry

end division_remainder_l1014_101492


namespace sophia_bus_time_l1014_101457

def sophia_schedule : Prop :=
  let leave_home : Nat := 8 * 60 + 15  -- 8:15 AM in minutes
  let catch_bus : Nat := 8 * 60 + 45   -- 8:45 AM in minutes
  let class_duration : Nat := 55
  let num_classes : Nat := 5
  let lunch_break : Nat := 45
  let club_activities : Nat := 3 * 60  -- 3 hours in minutes
  let arrive_home : Nat := 17 * 60 + 30  -- 5:30 PM in minutes

  let total_away_time : Nat := arrive_home - leave_home
  let school_activities_time : Nat := num_classes * class_duration + lunch_break + club_activities
  let bus_time : Nat := total_away_time - school_activities_time

  bus_time = 25

theorem sophia_bus_time : sophia_schedule := by
  sorry

end sophia_bus_time_l1014_101457


namespace mens_wages_l1014_101446

/-- Proves that given the conditions in the problem, the total wages for 9 men is Rs. 72 -/
theorem mens_wages (total_earnings : ℕ) (num_men num_boys : ℕ) (W : ℕ) :
  total_earnings = 216 →
  num_men = 9 →
  num_boys = 7 →
  num_men * W = num_men * num_boys →
  (3 * num_men : ℕ) * (total_earnings / (3 * num_men)) = 72 :=
by sorry

end mens_wages_l1014_101446


namespace quadratic_equal_roots_l1014_101471

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - (k + 1) * x + 2 * k = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - (k + 1) * y + 2 * k = 0 → y = x) ↔ 
  k = 11 + 10 * Real.sqrt 6 ∨ k = 11 - 10 * Real.sqrt 6 :=
sorry

end quadratic_equal_roots_l1014_101471


namespace multiply_63_57_l1014_101487

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l1014_101487


namespace circle_equation_l1014_101441

-- Define the circle C
def circle_C (a r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = r^2}

-- Define the line 2x - y = 0
def line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 = 0}

theorem circle_equation : 
  ∃ (a r : ℝ), 
    a > 0 ∧ 
    (0, Real.sqrt 5) ∈ circle_C a r ∧ 
    (abs (2 * a) / Real.sqrt 5 = 4 * Real.sqrt 5 / 5) ∧
    circle_C a r = circle_C 2 3 :=
  sorry

end circle_equation_l1014_101441


namespace not_all_numbers_representable_l1014_101459

theorem not_all_numbers_representable :
  ∃ k : ℕ, k % 6 = 0 ∧ k > 1000 ∧
  ∀ m n : ℕ, k ≠ n * (n + 1) * (n + 2) * (n + 3) * (n + 4) - m * (m + 1) * (m + 2) :=
by sorry

end not_all_numbers_representable_l1014_101459


namespace complex_fraction_squared_l1014_101455

theorem complex_fraction_squared (i : ℂ) : i * i = -1 → ((1 - i) / (1 + i))^2 = -1 := by
  sorry

end complex_fraction_squared_l1014_101455


namespace a_6_value_l1014_101408

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a_6_value
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_roots : a 4 * a 8 = 9 ∧ a 4 + a 8 = -11) :
  a 6 = -3 := by
sorry

end a_6_value_l1014_101408


namespace power_equation_solution_l1014_101491

theorem power_equation_solution (m : ℕ) : 8^2 = 4^2 * 2^m → m = 2 := by
  sorry

end power_equation_solution_l1014_101491


namespace dante_balloon_sharing_l1014_101495

theorem dante_balloon_sharing :
  ∀ (num_friends : ℕ),
    num_friends > 0 →
    250 / num_friends - 11 = 39 →
    num_friends = 5 := by
  sorry

end dante_balloon_sharing_l1014_101495


namespace angle_equality_l1014_101434

-- Define angles A, B, and C
variable (A B C : ℝ)

-- Define the conditions
axiom angle_sum_1 : A + B = 180
axiom angle_sum_2 : B + C = 180

-- State the theorem
theorem angle_equality : A = C := by
  sorry

end angle_equality_l1014_101434


namespace v2_equals_5_l1014_101422

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Definition of V₂ in Horner's method -/
def V₂ (a₅ a₄ a₃ a₂ a₁ a₀ : ℝ) (x : ℝ) : ℝ :=
  (a₅ * x + a₄) * x - a₃

/-- Theorem: V₂ equals 5 for the given polynomial when x = 2 -/
theorem v2_equals_5 :
  let f : ℝ → ℝ := fun x => 2 * x^5 - 3 * x^3 + 2 * x^2 - x + 5
  V₂ 2 0 (-3) 2 (-1) 5 2 = 5 := by
  sorry

#eval V₂ 2 0 (-3) 2 (-1) 5 2

end v2_equals_5_l1014_101422


namespace dillon_luca_sum_difference_l1014_101404

def dillon_list := List.range 40

def replace_three_with_two (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

def luca_list := dillon_list.map replace_three_with_two

theorem dillon_luca_sum_difference :
  (dillon_list.sum - luca_list.sum) = 104 := by
  sorry

end dillon_luca_sum_difference_l1014_101404


namespace hcf_of_12_and_15_l1014_101480

theorem hcf_of_12_and_15 : 
  ∀ (hcf lcm : ℕ), 
    lcm = 60 → 
    12 * 15 = lcm * hcf → 
    hcf = 3 :=
by
  sorry

end hcf_of_12_and_15_l1014_101480


namespace no_roots_geq_two_l1014_101432

theorem no_roots_geq_two : ∀ x : ℝ, x ≥ 2 → 4 * x^3 - 5 * x^2 - 6 * x + 3 > 0 := by
  sorry

end no_roots_geq_two_l1014_101432


namespace halfway_fraction_l1014_101449

theorem halfway_fraction : (3 : ℚ) / 4 + ((5 : ℚ) / 6 - (3 : ℚ) / 4) / 2 = (19 : ℚ) / 24 := by
  sorry

end halfway_fraction_l1014_101449


namespace ten_two_zero_one_composite_l1014_101479

theorem ten_two_zero_one_composite (n : ℕ) (h : n > 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 2*n^2 + 1 = a * b :=
sorry

end ten_two_zero_one_composite_l1014_101479


namespace smallest_max_sum_l1014_101469

theorem smallest_max_sum (p q r s t : ℕ+) 
  (sum_condition : p + q + r + s + t = 4020) : 
  (∃ (N : ℕ), 
    N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ 
    (∀ (M : ℕ), M = max (p + q) (max (q + r) (max (r + s) (s + t))) → N ≤ M) ∧
    N = 1005) := by
  sorry

end smallest_max_sum_l1014_101469


namespace instant_noodle_price_reduction_l1014_101431

theorem instant_noodle_price_reduction 
  (original_weight : ℝ) 
  (original_price : ℝ) 
  (weight_increase_percentage : ℝ) 
  (h1 : weight_increase_percentage = 0.25) 
  (h2 : original_weight > 0) 
  (h3 : original_price > 0) : 
  let new_weight := original_weight * (1 + weight_increase_percentage)
  let original_price_per_unit := original_price / original_weight
  let new_price_per_unit := original_price / new_weight
  (original_price_per_unit - new_price_per_unit) / original_price_per_unit = 0.2
  := by sorry

end instant_noodle_price_reduction_l1014_101431


namespace kanul_original_amount_l1014_101483

theorem kanul_original_amount (raw_materials machinery marketing : ℕ) 
  (h1 : raw_materials = 35000)
  (h2 : machinery = 40000)
  (h3 : marketing = 15000)
  (h4 : (raw_materials + machinery + marketing : ℚ) = 0.25 * 360000) :
  360000 = 360000 := by sorry

end kanul_original_amount_l1014_101483


namespace base_with_final_digit_two_l1014_101417

theorem base_with_final_digit_two : 
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 20 ∧ 625 % b = 2 := by
  sorry

end base_with_final_digit_two_l1014_101417


namespace sqrt_domain_sqrt_nonneg_sqrt_undefined_neg_l1014_101436

-- Define the square root function for non-negative real numbers
noncomputable def sqrt (a : ℝ) : ℝ := Real.sqrt a

-- Theorem stating that the domain of the square root function is non-negative real numbers
theorem sqrt_domain (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a → a ≥ 0 := by
  sorry

-- Theorem stating that the square root of a non-negative number is non-negative
theorem sqrt_nonneg (a : ℝ) (h : a ≥ 0) : sqrt a ≥ 0 := by
  sorry

-- Theorem stating that the square root function is undefined for negative numbers
theorem sqrt_undefined_neg (a : ℝ) : a < 0 → ¬∃ (x : ℝ), x ^ 2 = a := by
  sorry

end sqrt_domain_sqrt_nonneg_sqrt_undefined_neg_l1014_101436


namespace target_probability_l1014_101481

/-- The probability of hitting a target once -/
def p : ℝ := 0.6

/-- The number of shots -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots -/
def prob_at_least_two (p : ℝ) (n : ℕ) : ℝ :=
  3 * p^2 * (1 - p) + p^3

theorem target_probability :
  prob_at_least_two p n = 0.648 :=
sorry

end target_probability_l1014_101481


namespace quadratic_rational_solution_l1014_101458

theorem quadratic_rational_solution (a b : ℕ+) :
  (∃ x : ℚ, x^2 + (a + b : ℚ)^2 * x + 4 * (a : ℚ) * (b : ℚ) = 1) ↔ a = b :=
by sorry

end quadratic_rational_solution_l1014_101458


namespace trajectory_of_point_M_l1014_101448

/-- The trajectory of point M satisfying the given distance conditions -/
theorem trajectory_of_point_M (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), (x₀ - 0)^2 + (y₀ - 4)^2 = (abs (y₀ + 5) - 1)^2 → x₀^2 = 16 * y₀) →
  x^2 + (y - 4)^2 = (abs (y + 5) - 1)^2 →
  x^2 = 16 * y := by
  sorry


end trajectory_of_point_M_l1014_101448


namespace sufficient_not_necessary_condition_l1014_101435

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, (a - b) * a^2 < 0 → a < b) ∧
  ¬(∀ a b, a < b → (a - b) * a^2 < 0) :=
by sorry

end sufficient_not_necessary_condition_l1014_101435


namespace cubic_polynomial_relation_l1014_101405

/-- Given a cubic polynomial f and another cubic polynomial g satisfying certain conditions, 
    prove that g(4) = 105. -/
theorem cubic_polynomial_relation (f g : ℝ → ℝ) : 
  (∀ x, f x = x^3 - 2*x^2 + x + 1) →
  (∃ A r s t : ℝ, ∀ x, g x = A * (x - r^2) * (x - s^2) * (x - t^2)) →
  g 0 = -1 →
  (∀ x, f x = 0 ↔ g (x^2) = 0) →
  g 4 = 105 :=
by sorry

end cubic_polynomial_relation_l1014_101405


namespace arithmetic_sequence_12th_term_l1014_101400

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

end arithmetic_sequence_12th_term_l1014_101400


namespace polar_to_rectangular_transformation_l1014_101413

theorem polar_to_rectangular_transformation (x y : ℝ) (h : x = 12 ∧ y = 5) :
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r^2 * Real.cos (3 * θ), r^2 * Real.sin (3 * θ)) = (-494004 / 2197, 4441555 / 2197) := by
  sorry

end polar_to_rectangular_transformation_l1014_101413


namespace alice_exam_score_l1014_101467

theorem alice_exam_score (exam1 exam2 exam3 : ℕ) 
  (h1 : exam1 = 85) (h2 : exam2 = 76) (h3 : exam3 = 83)
  (h4 : ∀ exam, exam ≤ 100) : 
  ∃ (exam4 exam5 : ℕ), 
    exam4 ≤ 100 ∧ exam5 ≤ 100 ∧ 
    (exam1 + exam2 + exam3 + exam4 + exam5) / 5 = 80 ∧
    (exam4 = 56 ∨ exam5 = 56) ∧
    ∀ (x : ℕ), x < 56 → 
      ¬∃ (y : ℕ), y ≤ 100 ∧ (exam1 + exam2 + exam3 + x + y) / 5 = 80 :=
by sorry

end alice_exam_score_l1014_101467


namespace octal_2011_equals_base5_13113_l1014_101497

-- Define a function to convert from octal to decimal
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldr (fun (i, digit) acc => acc + digit * (8 ^ i)) 0

-- Define a function to convert from decimal to base-5
def decimal_to_base5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

-- Theorem statement
theorem octal_2011_equals_base5_13113 :
  decimal_to_base5 (octal_to_decimal [1, 1, 0, 2]) = [3, 1, 1, 3, 1] := by
  sorry

end octal_2011_equals_base5_13113_l1014_101497


namespace mixed_number_less_than_decimal_l1014_101425

theorem mixed_number_less_than_decimal : -1 - (3 / 5 : ℚ) < -1.5 := by sorry

end mixed_number_less_than_decimal_l1014_101425


namespace f_monotonicity_and_max_root_difference_l1014_101407

noncomputable def f (x : ℝ) : ℝ := 4 * x - x^4

theorem f_monotonicity_and_max_root_difference :
  (∀ x y, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x > f y) ∧
  (∀ a x₁ x₂, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ 1 < x₂ → x₂ - 1 ≤ 0) :=
sorry

end f_monotonicity_and_max_root_difference_l1014_101407


namespace minimize_f_minimum_l1014_101409

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  |7*x - 3*a + 8| + |5*x + 4*a - 6| + |x - a - 8| - 24

/-- Theorem stating that 82/43 is the value of a that minimizes the minimum value of f(x) -/
theorem minimize_f_minimum (a : ℝ) :
  (∀ x, f (82/43) x ≤ f a x) ∧ (∃ x, f (82/43) x < f a x) ∨ a = 82/43 := by
  sorry

#check minimize_f_minimum

end minimize_f_minimum_l1014_101409


namespace unreachable_one_if_not_div_three_l1014_101482

/-- The operation of adding 3 repeatedly until divisible by 5, then dividing by 5 -/
def operation (n : ℕ) : ℕ :=
  let m := n + 3 * (5 - n % 5) % 5
  m / 5

/-- Predicate to check if a number can reach 1 through repeated applications of the operation -/
def can_reach_one (n : ℕ) : Prop :=
  ∃ k : ℕ, (operation^[k] n) = 1

/-- Theorem stating that numbers not divisible by 3 cannot reach 1 through the given operations -/
theorem unreachable_one_if_not_div_three (n : ℕ) (h : ¬ 3 ∣ n) : ¬ can_reach_one n :=
sorry

end unreachable_one_if_not_div_three_l1014_101482


namespace simple_interest_difference_l1014_101424

/-- Simple interest calculation and comparison --/
theorem simple_interest_difference (principal rate time : ℕ) : 
  principal = 3000 → 
  rate = 4 → 
  time = 5 → 
  principal - (principal * rate * time) / 100 = 2400 := by
  sorry

end simple_interest_difference_l1014_101424


namespace special_number_exists_l1014_101460

/-- Number of digits in a natural number -/
def number_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For every natural number a, there exists a natural number b and a non-negative integer k
    such that a * 10^k + b = a * (b * 10^(number_of_digits a) + a) -/
theorem special_number_exists (a : ℕ) : ∃ (b : ℕ) (k : ℕ), 
  a * 10^k + b = a * (b * 10^(number_of_digits a) + a) := by sorry

end special_number_exists_l1014_101460


namespace sum_of_reciprocal_relations_l1014_101442

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1 / x + 1 / y = 5) 
  (h4 : 1 / x - 1 / y = -9) : 
  x + y = -5/14 := by
  sorry

end sum_of_reciprocal_relations_l1014_101442


namespace divisibility_by_72_l1014_101401

theorem divisibility_by_72 (n : ℕ) : 
  ∃ d : ℕ, d < 10 ∧ 32235717 * 10 + d = n * 72 :=
sorry

end divisibility_by_72_l1014_101401


namespace lcm_5_7_10_21_l1014_101440

theorem lcm_5_7_10_21 : Nat.lcm 5 (Nat.lcm 7 (Nat.lcm 10 21)) = 210 := by
  sorry

end lcm_5_7_10_21_l1014_101440


namespace min_sum_distances_l1014_101465

-- Define a rectangle in 2D space
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_rectangle : sorry -- Condition ensuring ABCD forms a rectangle

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the center of a rectangle
def center (r : Rectangle) : ℝ × ℝ := sorry

-- Define the sum of distances from a point to the corners
def sum_distances (r : Rectangle) (p : ℝ × ℝ) : ℝ :=
  distance p r.A + distance p r.B + distance p r.C + distance p r.D

-- Theorem statement
theorem min_sum_distances (r : Rectangle) :
  ∀ p : ℝ × ℝ, sum_distances r (center r) ≤ sum_distances r p :=
sorry

end min_sum_distances_l1014_101465


namespace radius_greater_than_distance_to_center_l1014_101490

/-- A circle with center O and a point P inside it -/
structure CircleWithInnerPoint where
  O : ℝ × ℝ  -- Center of the circle
  P : ℝ × ℝ  -- Point inside the circle
  r : ℝ      -- Radius of the circle
  h_inside : dist P O < r  -- P is inside the circle

/-- The theorem stating that if P is inside circle O and distance from P to O is 5,
    then the radius of circle O must be greater than 5 -/
theorem radius_greater_than_distance_to_center 
  (c : CircleWithInnerPoint) (h : dist c.P c.O = 5) : c.r > 5 := by
  sorry

end radius_greater_than_distance_to_center_l1014_101490


namespace tangent_line_at_point_l1014_101477

/-- The equation of a curve -/
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

/-- The point on the curve -/
def point : ℝ × ℝ := (1, 2)

/-- The equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := 3*x - 1

theorem tangent_line_at_point :
  let (x₀, y₀) := point
  (f x₀ = y₀) ∧ 
  (∀ x : ℝ, tangent_line x = f x₀ + (tangent_line x₀ - f x₀) * (x - x₀)) :=
sorry

end tangent_line_at_point_l1014_101477


namespace x_values_l1014_101416

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) :
  x = 4 ∨ x = 6 := by
  sorry

end x_values_l1014_101416


namespace collinear_vectors_m_value_l1014_101474

def a (m : ℝ) : Fin 2 → ℝ := ![2*m, 3]
def b (m : ℝ) : Fin 2 → ℝ := ![m-1, 1]

theorem collinear_vectors_m_value (m : ℝ) :
  (∃ (k : ℝ), a m = k • b m) → m = 3 := by
  sorry

end collinear_vectors_m_value_l1014_101474


namespace square_side_length_l1014_101421

theorem square_side_length (perimeter : ℝ) (h : perimeter = 17.8) :
  let side_length := perimeter / 4
  side_length = 4.45 := by
sorry

end square_side_length_l1014_101421


namespace unique_positive_solution_l1014_101498

theorem unique_positive_solution : 
  ∃! x : ℝ, x > 0 ∧ (1/2) * (4 * x^2 - 4) = (x^2 - 40*x - 8) * (x^2 + 20*x + 4) ∧ x = 20 + Real.sqrt 410 := by
  sorry

end unique_positive_solution_l1014_101498


namespace distance_proof_l1014_101468

/- Define the speeds of A and B in meters per minute -/
def speed_A : ℝ := 60
def speed_B : ℝ := 80

/- Define the rest time of B in minutes -/
def rest_time : ℝ := 14

/- Define the distance between A and B -/
def distance_AB : ℝ := 1680

/- Theorem statement -/
theorem distance_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    speed_A * t + speed_B * t = distance_AB ∧
    (distance_AB / speed_A + distance_AB / speed_B + rest_time) / 2 = t :=
by sorry

#check distance_proof

end distance_proof_l1014_101468


namespace even_function_extension_l1014_101488

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- The given function f defined for x ≤ 0 -/
def f_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≤ 0 → f x = x^2 - 2*x

theorem even_function_extension :
  ∀ f : ℝ → ℝ, EvenFunction f → f_nonpositive f →
  ∀ x : ℝ, x > 0 → f x = x^2 + 2*x :=
sorry

end even_function_extension_l1014_101488


namespace distance_A_to_origin_l1014_101499

/-- The distance between point A(2, 3, 3) and the origin O(0, 0, 0) in a three-dimensional Cartesian coordinate system is √22. -/
theorem distance_A_to_origin : 
  let A : Fin 3 → ℝ := ![2, 3, 3]
  let O : Fin 3 → ℝ := ![0, 0, 0]
  Real.sqrt ((A 0 - O 0)^2 + (A 1 - O 1)^2 + (A 2 - O 2)^2) = Real.sqrt 22 := by
  sorry

end distance_A_to_origin_l1014_101499


namespace max_value_of_f_l1014_101452

noncomputable def f (x : ℝ) : ℝ := 2 * (-1) * Real.log x - 1 / x

theorem max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x ∧ f x = 2 * Real.log 2 - 2 := by
  sorry

end max_value_of_f_l1014_101452


namespace abs_neg_three_eq_three_l1014_101462

theorem abs_neg_three_eq_three : |(-3 : ℝ)| = 3 := by
  sorry

end abs_neg_three_eq_three_l1014_101462


namespace alexis_shopping_problem_l1014_101476

/-- Alexis's shopping problem -/
theorem alexis_shopping_problem (budget initial_amount remaining_amount shirt_cost pants_cost socks_cost belt_cost shoes_cost : ℕ) 
  (h1 : initial_amount = 200)
  (h2 : shirt_cost = 30)
  (h3 : pants_cost = 46)
  (h4 : socks_cost = 11)
  (h5 : belt_cost = 18)
  (h6 : shoes_cost = 41)
  (h7 : remaining_amount = 16)
  (h8 : budget = initial_amount - remaining_amount) :
  budget - (shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost) = 38 := by
  sorry

#check alexis_shopping_problem

end alexis_shopping_problem_l1014_101476


namespace x_squared_minus_y_squared_l1014_101437

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 11 / 17) 
  (h2 : x - y = 1 / 143) : 
  x^2 - y^2 = 11 / 2431 := by
sorry

end x_squared_minus_y_squared_l1014_101437


namespace quadratic_equation_general_form_l1014_101420

/-- A quadratic equation in one variable -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ
  h_quadratic : a ≠ 0

/-- The general form of a quadratic equation -/
def general_form (eq : QuadraticEquation) : Prop :=
  eq.a * eq.x^2 + eq.b * eq.x + eq.c = 0

/-- Theorem: The general form of a quadratic equation in one variable is ax^2 + bx + c = 0 where a ≠ 0 -/
theorem quadratic_equation_general_form (eq : QuadraticEquation) :
  general_form eq :=
sorry

end quadratic_equation_general_form_l1014_101420


namespace triangle_problem_l1014_101473

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  A + B + C = π →
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  -- Given conditions
  (Real.cos (B + C)) / (Real.cos C) = a / (2 * b + c) →
  b = 1 →
  Real.cos C = 2 * Real.sqrt 7 / 7 →
  -- Conclusions
  A = 2 * π / 3 ∧ a = Real.sqrt 7 ∧ c = 2 :=
by sorry

end triangle_problem_l1014_101473


namespace chess_tournament_games_l1014_101426

/-- The number of games in a chess tournament -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- The theorem stating the number of games in the specific tournament -/
theorem chess_tournament_games :
  tournament_games 19 * 2 = 684 := by
  sorry

end chess_tournament_games_l1014_101426


namespace right_triangle_min_std_dev_l1014_101430

theorem right_triangle_min_std_dev (a b c : ℝ) : 
  a > 0 → b > 0 → c = 3 → a^2 + b^2 = c^2 →
  let s := Real.sqrt ((a^2 + b^2 + c^2) / 3 - ((a + b + c) / 3)^2)
  s ≥ Real.sqrt 2 - 1 ∧ 
  (s = Real.sqrt 2 - 1 ↔ a = 3 * Real.sqrt 2 / 2 ∧ b = 3 * Real.sqrt 2 / 2) :=
by sorry

end right_triangle_min_std_dev_l1014_101430


namespace cookies_distribution_l1014_101456

theorem cookies_distribution (cookies_per_person : ℝ) (total_cookies : ℕ) (h1 : cookies_per_person = 24.0) (h2 : total_cookies = 144) :
  (total_cookies : ℝ) / cookies_per_person = 6 := by
  sorry

end cookies_distribution_l1014_101456


namespace carrie_turnip_mixture_l1014_101466

-- Define the ratio of potatoes to turnips
def potatoTurnipRatio : ℚ := 5 / 2

-- Define the total amount of potatoes
def totalPotatoes : ℚ := 20

-- Define the amount of turnips that can be added
def turnipsToAdd : ℚ := totalPotatoes / potatoTurnipRatio

-- Theorem statement
theorem carrie_turnip_mixture :
  turnipsToAdd = 8 := by sorry

end carrie_turnip_mixture_l1014_101466


namespace circle_on_parabola_passes_through_focus_l1014_101423

/-- A circle with center on a parabola y^2 = 4x and tangent to x = -1 passes through (1, 0) -/
theorem circle_on_parabola_passes_through_focus (C : ℝ × ℝ) (r : ℝ) :
  C.2^2 = 4 * C.1 →  -- Center C is on the parabola y^2 = 4x
  abs (C.1 + 1) = r →  -- Circle is tangent to x = -1
  (1 - C.1)^2 + C.2^2 = r^2  -- Circle passes through (1, 0)
  := by sorry

end circle_on_parabola_passes_through_focus_l1014_101423


namespace victors_flower_stickers_l1014_101433

theorem victors_flower_stickers :
  ∀ (flower_stickers animal_stickers : ℕ),
    animal_stickers = flower_stickers - 2 →
    flower_stickers + animal_stickers = 14 →
    flower_stickers = 8 := by
  sorry

end victors_flower_stickers_l1014_101433


namespace stream_speed_l1014_101428

/-- Proves that the speed of the stream is 8 kmph given the conditions of the problem -/
theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 24 →
  (1 / (boat_speed - stream_speed)) = (2 / (boat_speed + stream_speed)) →
  stream_speed = 8 := by
  sorry

end stream_speed_l1014_101428


namespace total_triangles_is_twenty_l1014_101447

/-- A rectangle with diagonals and midpoint segments. -/
structure RectangleWithDiagonals where
  /-- The rectangle has different length sides. -/
  different_sides : Bool
  /-- The diagonals intersect at the center. -/
  diagonals_intersect_center : Bool
  /-- Segments join midpoints of opposite sides. -/
  midpoint_segments : Bool

/-- Count the number of triangles in the rectangle configuration. -/
def count_triangles (r : RectangleWithDiagonals) : ℕ :=
  sorry

/-- Theorem stating that the total number of triangles is 20. -/
theorem total_triangles_is_twenty (r : RectangleWithDiagonals) 
  (h1 : r.different_sides = true)
  (h2 : r.diagonals_intersect_center = true)
  (h3 : r.midpoint_segments = true) : 
  count_triangles r = 20 := by
  sorry

end total_triangles_is_twenty_l1014_101447


namespace intersection_implies_sum_l1014_101484

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x : ℝ | x^2 - p*x + 15 = 0}
def B (q : ℝ) : Set ℝ := {x : ℝ | x^2 - 5*x + q = 0}

-- State the theorem
theorem intersection_implies_sum (p q : ℝ) : A p ∩ B q = {3} → p + q = 14 := by
  sorry

end intersection_implies_sum_l1014_101484


namespace complex_number_quadrant_l1014_101472

theorem complex_number_quadrant : ∃ (x y : ℝ), (Complex.mk x y = (2 - Complex.I)^2) ∧ (x > 0) ∧ (y < 0) := by
  sorry

end complex_number_quadrant_l1014_101472


namespace grandmother_rolls_l1014_101485

def total_rolls : ℕ := 12
def uncle_rolls : ℕ := 4
def neighbor_rolls : ℕ := 3
def remaining_rolls : ℕ := 2

theorem grandmother_rolls : 
  total_rolls - (uncle_rolls + neighbor_rolls + remaining_rolls) = 3 := by
  sorry

end grandmother_rolls_l1014_101485


namespace exists_valid_coloring_l1014_101439

/-- A point on a 2D grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A set of black cells on the grid --/
def BlackCells := Set GridPoint

/-- A line on the grid (vertical, horizontal, or diagonal) --/
inductive GridLine
  | Vertical (x : ℤ)
  | Horizontal (y : ℤ)
  | Diagonal (m : ℤ) (b : ℤ)

/-- The number of black cells on a given line --/
def blackCellsOnLine (cells : BlackCells) (line : GridLine) : ℕ :=
  sorry

/-- The property that a set of black cells satisfies the k-cell condition --/
def satisfiesKCellCondition (cells : BlackCells) (k : ℕ) : Prop :=
  ∀ line : GridLine, blackCellsOnLine cells line = k ∨ blackCellsOnLine cells line = 0

theorem exists_valid_coloring (k : ℕ) : 
  ∃ (cells : BlackCells), cells.Nonempty ∧ Set.Finite cells ∧ satisfiesKCellCondition cells k :=
  sorry

end exists_valid_coloring_l1014_101439


namespace fermat_coprime_l1014_101463

/-- The n-th Fermat number -/
def fermat (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Fermat numbers are pairwise coprime -/
theorem fermat_coprime : ∀ i j : ℕ, i ≠ j → Nat.gcd (fermat i) (fermat j) = 1 := by
  sorry

end fermat_coprime_l1014_101463


namespace visited_neither_country_l1014_101411

theorem visited_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) : 
  total = 60 → iceland = 35 → norway = 23 → both = 31 → 
  total - (iceland + norway - both) = 33 := by
sorry

end visited_neither_country_l1014_101411


namespace mabel_steps_to_helen_l1014_101445

/-- The total number of steps Mabel walks to visit Helen -/
def total_steps (mabel_distance helen_fraction : ℕ) : ℕ :=
  mabel_distance + (helen_fraction * mabel_distance) / 4

/-- Proof that Mabel walks 7875 steps to visit Helen -/
theorem mabel_steps_to_helen :
  total_steps 4500 3 = 7875 := by
  sorry

end mabel_steps_to_helen_l1014_101445


namespace rhombus_other_diagonal_l1014_101402

/-- Given a rhombus with one diagonal of length 30 meters and an area of 600 square meters,
    prove that the length of the other diagonal is 40 meters. -/
theorem rhombus_other_diagonal (d₁ : ℝ) (d₂ : ℝ) (area : ℝ) 
    (h₁ : d₁ = 30)
    (h₂ : area = 600)
    (h₃ : area = d₁ * d₂ / 2) : 
  d₂ = 40 := by
  sorry

end rhombus_other_diagonal_l1014_101402


namespace complex_modulus_equation_l1014_101486

theorem complex_modulus_equation (t : ℝ) (h : t > 0) :
  Complex.abs (8 + t * Complex.I) = 12 → t = 4 * Real.sqrt 5 := by
  sorry

end complex_modulus_equation_l1014_101486


namespace sequence_characterization_l1014_101478

def isValidSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, 0 ≤ a n) ∧
  (∀ n, a n ≤ a (n + 1)) ∧
  (∀ m n, a (m^2 + n^2) = (a m)^2 + (a n)^2)

theorem sequence_characterization (a : ℕ → ℝ) :
  isValidSequence a →
  ((∀ n, a n = 1/2) ∨ (∀ n, a n = 0) ∨ (∀ n, a n = n)) :=
sorry

end sequence_characterization_l1014_101478


namespace both_shooters_hit_probability_l1014_101418

theorem both_shooters_hit_probability
  (prob_A : ℝ)
  (prob_B : ℝ)
  (h_prob_A : prob_A = 0.9)
  (h_prob_B : prob_B = 0.8)
  (h_independent : True)  -- Assumption of independence
  : prob_A * prob_B = 0.72 :=
by sorry

end both_shooters_hit_probability_l1014_101418


namespace diameter_height_ratio_l1014_101403

/-- A cylinder whose lateral surface unfolds into a square -/
structure SquareUnfoldCylinder where
  diameter : ℝ
  height : ℝ
  square_unfold : height = π * diameter

theorem diameter_height_ratio (c : SquareUnfoldCylinder) :
  c.diameter / c.height = 1 / π := by
  sorry

end diameter_height_ratio_l1014_101403


namespace lucas_addition_example_l1014_101427

/-- Lucas's notation for integers -/
def lucas_notation (n : ℤ) : ℕ :=
  if n ≥ 0 then n.natAbs else n.natAbs + 1

/-- Addition in Lucas's notation -/
def lucas_add (a b : ℕ) : ℕ :=
  lucas_notation (-(a : ℤ) + -(b : ℤ))

/-- Theorem: 000 + 0000 = 000000 in Lucas's notation -/
theorem lucas_addition_example : lucas_add 3 4 = 6 := by
  sorry

end lucas_addition_example_l1014_101427
