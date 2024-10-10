import Mathlib

namespace largest_n_satisfying_conditions_l3014_301417

theorem largest_n_satisfying_conditions : ∃ (n : ℤ), n = 181 ∧ 
  (∃ (m : ℤ), n^2 = (m+1)^3 - m^3) ∧ 
  (∃ (k : ℤ), 2*n + 79 = k^2) ∧
  (∀ (n' : ℤ), n' > n → 
    (¬∃ (m : ℤ), n'^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ (k : ℤ), 2*n' + 79 = k^2)) := by
  sorry

end largest_n_satisfying_conditions_l3014_301417


namespace shirt_costs_15_l3014_301419

/-- The cost of one pair of jeans -/
def jeans_cost : ℚ := sorry

/-- The cost of one shirt -/
def shirt_cost : ℚ := sorry

/-- First condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jeans_cost + 2 * shirt_cost = 69

/-- Second condition: 2 pairs of jeans and 3 shirts cost $71 -/
axiom condition2 : 2 * jeans_cost + 3 * shirt_cost = 71

/-- Theorem: The cost of one shirt is $15 -/
theorem shirt_costs_15 : shirt_cost = 15 := by sorry

end shirt_costs_15_l3014_301419


namespace least_expensive_trip_l3014_301479

-- Define the triangle
def triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 5000^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 4000^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 - (C.1 - A.1)^2 - (C.2 - A.2)^2

-- Define travel costs
def car_cost (distance : ℝ) : ℝ := 0.20 * distance
def train_cost (distance : ℝ) : ℝ := 150 + 0.15 * distance

-- Define the total trip cost
def trip_cost (AB BC CA : ℝ) (mode_AB mode_BC mode_CA : Bool) : ℝ :=
  (if mode_AB then train_cost AB else car_cost AB) +
  (if mode_BC then train_cost BC else car_cost BC) +
  (if mode_CA then train_cost CA else car_cost CA)

-- Theorem statement
theorem least_expensive_trip (A B C : ℝ × ℝ) :
  triangle A B C →
  ∃ (mode_AB mode_BC mode_CA : Bool),
    ∀ (other_mode_AB other_mode_BC other_mode_CA : Bool),
      trip_cost 5000 22500 4000 mode_AB mode_BC mode_CA ≤
      trip_cost 5000 22500 4000 other_mode_AB other_mode_BC other_mode_CA ∧
      trip_cost 5000 22500 4000 mode_AB mode_BC mode_CA = 5130 :=
sorry

end least_expensive_trip_l3014_301479


namespace inequality_problem_l3014_301467

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 1/a + 4/b + 9/c ≤ 36/(a + b + c)) : 
  (2*b + 3*c)/(a + b + c) = 13/6 := by
  sorry

end inequality_problem_l3014_301467


namespace matthew_hotdogs_l3014_301406

/-- The number of hotdogs each sister wants -/
def sisters_hotdogs : ℕ := 2

/-- The total number of hotdogs both sisters want -/
def total_sisters_hotdogs : ℕ := 2 * sisters_hotdogs

/-- The number of hotdogs Luke wants -/
def luke_hotdogs : ℕ := 2 * total_sisters_hotdogs

/-- The number of hotdogs Hunter wants -/
def hunter_hotdogs : ℕ := (3 * total_sisters_hotdogs) / 2

/-- The total number of hotdogs Matthew needs to cook -/
def total_hotdogs : ℕ := total_sisters_hotdogs + luke_hotdogs + hunter_hotdogs

theorem matthew_hotdogs : total_hotdogs = 18 := by
  sorry

end matthew_hotdogs_l3014_301406


namespace min_both_mozart_and_bach_l3014_301472

theorem min_both_mozart_and_bach 
  (total : ℕ) 
  (mozart : ℕ) 
  (bach : ℕ) 
  (h1 : total = 100) 
  (h2 : mozart = 87) 
  (h3 : bach = 70) 
  : ℕ :=
by
  sorry

#check min_both_mozart_and_bach

end min_both_mozart_and_bach_l3014_301472


namespace sum_of_exponents_is_eight_l3014_301485

-- Define the expression
def expression (a b c : ℝ) : ℝ := (40 * a^6 * b^8 * c^14) ^ (1/3)

-- Define a function to calculate the sum of exponents outside the radical
def sum_of_exponents_outside_radical (a b c : ℝ) : ℕ :=
  let simplified := expression a b c
  -- This is a placeholder. In a real implementation, we would need to
  -- analyze the simplified expression to determine the exponents.
  8

-- The theorem to prove
theorem sum_of_exponents_is_eight :
  ∀ a b c : ℝ, sum_of_exponents_outside_radical a b c = 8 := by
  sorry

end sum_of_exponents_is_eight_l3014_301485


namespace fourth_root_fifth_power_eighth_l3014_301491

theorem fourth_root_fifth_power_eighth : (((5 ^ (1/2)) ^ 5) ^ (1/4)) ^ 8 = 3125 := by
  sorry

end fourth_root_fifth_power_eighth_l3014_301491


namespace solution_set_inequality_l3014_301451

theorem solution_set_inequality (x : ℝ) : 
  (x + 3) * (1 - x) ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := by
sorry

end solution_set_inequality_l3014_301451


namespace green_ball_probability_l3014_301425

/-- Represents a container of balls -/
structure Container where
  red : Nat
  green : Nat

/-- The probability of selecting a specific container -/
def containerProb : ℚ := 1 / 3

/-- The probability of selecting a green ball from a given container -/
def greenBallProb (c : Container) : ℚ := c.green / (c.red + c.green)

/-- The containers A, B, and C -/
def containerA : Container := ⟨5, 5⟩
def containerB : Container := ⟨3, 3⟩
def containerC : Container := ⟨3, 3⟩

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  containerProb * greenBallProb containerA +
  containerProb * greenBallProb containerB +
  containerProb * greenBallProb containerC = 1 / 2 := by
  sorry

end green_ball_probability_l3014_301425


namespace rescue_center_dogs_l3014_301489

/-- Calculates the number of remaining dogs after a series of additions and adoptions. -/
def remaining_dogs (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) : ℕ :=
  initial + moved_in - first_adoption - second_adoption

/-- Theorem stating that given the specific numbers from the problem, 
    the number of remaining dogs is 200. -/
theorem rescue_center_dogs : 
  remaining_dogs 200 100 40 60 = 200 := by
  sorry

#eval remaining_dogs 200 100 40 60

end rescue_center_dogs_l3014_301489


namespace line_parallel_to_intersection_l3014_301404

structure GeometricSpace where
  Line : Type
  Plane : Type
  is_parallel : Line → Plane → Prop
  intersect : Plane → Plane → Line
  line_parallel : Line → Line → Prop

theorem line_parallel_to_intersection
  (S : GeometricSpace)
  (l : S.Line)
  (p1 p2 : S.Plane)
  (h1 : S.is_parallel l p1)
  (h2 : S.is_parallel l p2)
  (h3 : p1 ≠ p2) :
  S.line_parallel l (S.intersect p1 p2) :=
sorry

end line_parallel_to_intersection_l3014_301404


namespace complement_M_intersect_N_l3014_301458

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 1}
def N : Set ℝ := {x | x^2 - 4 < 0}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.compl M) ∩ N = {x | 1 < x ∧ x < 2} :=
sorry

end complement_M_intersect_N_l3014_301458


namespace rectangle_area_perimeter_relation_l3014_301410

theorem rectangle_area_perimeter_relation :
  ∀ a b : ℕ,
  a > 10 →
  a * b = 5 * (2 * a + 2 * b) →
  2 * a + 2 * b = 90 :=
by
  sorry

end rectangle_area_perimeter_relation_l3014_301410


namespace greatest_M_inequality_l3014_301465

theorem greatest_M_inequality (x y z : ℝ) : 
  ∃ (M : ℝ), M = 2/3 ∧ 
  (∀ (N : ℝ), (∀ (a b c : ℝ), a^4 + b^4 + c^4 + a*b*c*(a + b + c) ≥ N*(a*b + b*c + c*a)^2) → N ≤ M) ∧
  x^4 + y^4 + z^4 + x*y*z*(x + y + z) ≥ M*(x*y + y*z + z*x)^2 := by
  sorry

end greatest_M_inequality_l3014_301465


namespace olivias_correct_answers_l3014_301496

theorem olivias_correct_answers 
  (total_problems : ℕ) 
  (correct_points : ℤ) 
  (incorrect_points : ℤ) 
  (total_score : ℤ) 
  (h1 : total_problems = 15)
  (h2 : correct_points = 6)
  (h3 : incorrect_points = -3)
  (h4 : total_score = 45) :
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points + 
    (total_problems - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 10 := by
  sorry

end olivias_correct_answers_l3014_301496


namespace condition1_correct_condition2_correct_condition3_correct_l3014_301408

-- Define the number of teachers, male students, and female students
def num_teachers : ℕ := 2
def num_male_students : ℕ := 3
def num_female_students : ℕ := 3

-- Define the total number of people
def total_people : ℕ := num_teachers + num_male_students + num_female_students

-- Function to calculate the number of arrangements for condition 1
def arrangements_condition1 : ℕ := sorry

-- Function to calculate the number of arrangements for condition 2
def arrangements_condition2 : ℕ := sorry

-- Function to calculate the number of arrangements for condition 3
def arrangements_condition3 : ℕ := sorry

-- Theorem for condition 1
theorem condition1_correct : arrangements_condition1 = 4320 := by sorry

-- Theorem for condition 2
theorem condition2_correct : arrangements_condition2 = 30240 := by sorry

-- Theorem for condition 3
theorem condition3_correct : arrangements_condition3 = 6720 := by sorry

end condition1_correct_condition2_correct_condition3_correct_l3014_301408


namespace quadratic_sum_real_roots_l3014_301482

/-- A quadratic polynomial with positive leading coefficient and real roots -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  real_roots : b^2 - 4*a*c ≥ 0

/-- The sum of two QuadraticPolynomials -/
def add_poly (P Q : QuadraticPolynomial) : QuadraticPolynomial :=
  { a := P.a + Q.a,
    b := P.b + Q.b,
    c := P.c + Q.c,
    a_pos := by 
      apply add_pos P.a_pos Q.a_pos
    real_roots := sorry }

/-- Two QuadraticPolynomials have a common root -/
def have_common_root (P Q : QuadraticPolynomial) : Prop :=
  ∃ x : ℝ, P.a * x^2 + P.b * x + P.c = 0 ∧ Q.a * x^2 + Q.b * x + Q.c = 0

theorem quadratic_sum_real_roots (P₁ P₂ P₃ : QuadraticPolynomial)
  (h₁₂ : have_common_root P₁ P₂)
  (h₂₃ : have_common_root P₂ P₃)
  (h₁₃ : have_common_root P₁ P₃) :
  ∃ x : ℝ, (P₁.a + P₂.a + P₃.a) * x^2 + (P₁.b + P₂.b + P₃.b) * x + (P₁.c + P₂.c + P₃.c) = 0 :=
sorry

end quadratic_sum_real_roots_l3014_301482


namespace intersection_M_N_l3014_301434

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l3014_301434


namespace line_not_in_second_quadrant_l3014_301426

/-- A line does not pass through the second quadrant if and only if
    its slope is non-negative and its y-intercept is non-positive -/
def not_in_second_quadrant (a b c : ℝ) : Prop :=
  a ≥ 0 ∧ c ≤ 0

theorem line_not_in_second_quadrant (t : ℝ) :
  not_in_second_quadrant (2*t - 3) 2 t → 0 ≤ t ∧ t ≤ 3/2 := by
  sorry

end line_not_in_second_quadrant_l3014_301426


namespace parallelogram_reflection_theorem_l3014_301422

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the reflection across x-axis
def reflectXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

-- Define the reflection across y=x-1
def reflectYEqXMinus1 (p : Point2D) : Point2D :=
  { x := p.y + 1, y := p.x - 1 }

-- Define the composite transformation
def compositeTransform (p : Point2D) : Point2D :=
  reflectYEqXMinus1 (reflectXAxis p)

-- Theorem statement
theorem parallelogram_reflection_theorem (E F G H : Point2D)
  (hE : E = { x := 3, y := 3 })
  (hF : F = { x := 6, y := 7 })
  (hG : G = { x := 9, y := 3 })
  (hH : H = { x := 6, y := -1 }) :
  compositeTransform H = { x := 2, y := 5 } := by sorry

end parallelogram_reflection_theorem_l3014_301422


namespace cube_surface_area_l3014_301446

/-- Given a cube with volume 125 cubic cm, its surface area is 150 square cm. -/
theorem cube_surface_area (volume : ℝ) (side_length : ℝ) (surface_area : ℝ) : 
  volume = 125 → 
  side_length ^ 3 = volume →
  surface_area = 6 * side_length ^ 2 →
  surface_area = 150 := by
sorry

end cube_surface_area_l3014_301446


namespace first_ring_at_three_am_l3014_301492

/-- A clock that rings at regular intervals throughout the day -/
structure RingingClock where
  ring_interval : ℕ  -- Interval between rings in hours
  rings_per_day : ℕ  -- Number of times the clock rings in a day

/-- The time of day in hours (0 to 23) -/
def Time := Fin 24

/-- Calculate the time of the first ring for a given clock -/
def first_ring_time (clock : RingingClock) : Time :=
  ⟨clock.ring_interval, by sorry⟩

theorem first_ring_at_three_am 
  (clock : RingingClock) 
  (h1 : clock.ring_interval = 3) 
  (h2 : clock.rings_per_day = 8) : 
  first_ring_time clock = ⟨3, by sorry⟩ := by
  sorry

end first_ring_at_three_am_l3014_301492


namespace smallest_student_count_l3014_301443

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  eighth : ℕ
  seventh : ℕ

/-- Checks if the given counts satisfy the ratio conditions --/
def satisfies_ratios (counts : GradeCount) : Prop :=
  7 * counts.seventh = 4 * counts.ninth ∧
  9 * counts.eighth = 5 * counts.ninth

/-- The total number of students --/
def total_students (counts : GradeCount) : ℕ :=
  counts.ninth + counts.eighth + counts.seventh

/-- Theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (counts : GradeCount),
    satisfies_ratios counts ∧
    total_students counts = 134 ∧
    (∀ (other : GradeCount), satisfies_ratios other → total_students other ≥ 134) := by
  sorry

end smallest_student_count_l3014_301443


namespace oblique_prism_volume_l3014_301490

/-- The volume of an oblique prism with a parallelogram base and inclined lateral edge -/
theorem oblique_prism_volume
  (base_side1 base_side2 lateral_edge : ℝ)
  (base_angle lateral_angle : ℝ)
  (h_base_side1 : base_side1 = 3)
  (h_base_side2 : base_side2 = 6)
  (h_lateral_edge : lateral_edge = 4)
  (h_base_angle : base_angle = Real.pi / 4)  -- 45°
  (h_lateral_angle : lateral_angle = Real.pi / 6)  -- 30°
  : Real.sqrt 6 * 18 = 
    base_side1 * base_side2 * Real.sin base_angle * 
    (lateral_edge * Real.cos lateral_angle) := by
  sorry


end oblique_prism_volume_l3014_301490


namespace problem_solution_l3014_301450

theorem problem_solution (a b c d m n : ℕ+) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end problem_solution_l3014_301450


namespace circle_equation_l3014_301416

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def is_tangent_to_line (C : Circle) (a b c : ℝ) : Prop :=
  abs (a * C.center.1 + b * C.center.2 + c) = C.radius * Real.sqrt (a^2 + b^2)

def is_tangent_to_x_axis (C : Circle) : Prop :=
  C.center.2 = C.radius

-- State the theorem
theorem circle_equation (C : Circle) :
  C.radius = 1 →
  is_in_first_quadrant C.center →
  is_tangent_to_line C 4 (-3) 0 →
  is_tangent_to_x_axis C →
  ∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2} :=
by sorry

end circle_equation_l3014_301416


namespace tire_repair_tax_l3014_301455

theorem tire_repair_tax (repair_cost : ℚ) (num_tires : ℕ) (final_cost : ℚ) :
  repair_cost = 7 →
  num_tires = 4 →
  final_cost = 30 →
  (final_cost - (repair_cost * num_tires)) / num_tires = (1/2 : ℚ) := by
sorry

end tire_repair_tax_l3014_301455


namespace love_all_girls_l3014_301463

-- Define the girls
inductive Girl
| Sue
| Marcia
| Diana

-- Define the love relation
def loves : Girl → Prop := sorry

-- State the theorem
theorem love_all_girls :
  -- Condition 1: I love at least one of the three girls
  (∃ g : Girl, loves g) →
  -- Condition 2: If I love Sue but not Diana, then I also love Marcia
  (loves Girl.Sue ∧ ¬loves Girl.Diana → loves Girl.Marcia) →
  -- Condition 3: I either love both Diana and Marcia, or I love neither of them
  ((loves Girl.Diana ∧ loves Girl.Marcia) ∨ (¬loves Girl.Diana ∧ ¬loves Girl.Marcia)) →
  -- Condition 4: If I love Diana, then I also love Sue
  (loves Girl.Diana → loves Girl.Sue) →
  -- Conclusion: I love all three girls
  (loves Girl.Sue ∧ loves Girl.Marcia ∧ loves Girl.Diana) :=
by sorry

end love_all_girls_l3014_301463


namespace sum_first_six_primes_mod_seventh_prime_l3014_301470

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

theorem sum_first_six_primes_mod_seventh_prime :
  sumFirstNPrimes 6 % nthPrime 7 = 7 := by sorry

end sum_first_six_primes_mod_seventh_prime_l3014_301470


namespace min_value_of_function_l3014_301438

theorem min_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∀ y : ℝ, y = 4/x + 1/(1-x) → y ≥ 9 :=
by sorry

end min_value_of_function_l3014_301438


namespace factory_output_increase_l3014_301488

/-- Proves that the percentage increase in actual output compared to last year is 11.1% -/
theorem factory_output_increase (a : ℝ) : 
  let last_year_output := a / 1.1
  let this_year_actual := a * 1.01
  (this_year_actual - last_year_output) / last_year_output * 100 = 11.1 := by
  sorry

end factory_output_increase_l3014_301488


namespace one_certain_event_l3014_301440

-- Define the events
inductive Event
  | WaterFreeze : Event
  | RectangleArea : Event
  | CoinToss : Event
  | ExamScore : Event

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.WaterFreeze => False
  | Event.RectangleArea => True
  | Event.CoinToss => False
  | Event.ExamScore => False

-- Theorem statement
theorem one_certain_event :
  (∃! e : Event, isCertain e) :=
sorry

end one_certain_event_l3014_301440


namespace icosahedron_edge_probability_l3014_301439

/-- A regular icosahedron -/
structure Icosahedron where
  vertices : Finset (Fin 12)
  edges : Finset (Fin 30)

/-- The probability of selecting two vertices that form an edge in a regular icosahedron -/
def edge_probability (i : Icosahedron) : ℚ :=
  5 / 11

/-- Theorem: The probability of randomly selecting two vertices that form an edge
    in a regular icosahedron is 5/11 -/
theorem icosahedron_edge_probability (i : Icosahedron) :
  edge_probability i = 5 / 11 := by
  sorry


end icosahedron_edge_probability_l3014_301439


namespace max_value_a_l3014_301469

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 50) 
  (h5 : d > 10) : 
  a ≤ 2924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 ∧ 
    d' > 10 :=
sorry

end max_value_a_l3014_301469


namespace barn_painted_area_l3014_301430

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted for a rectangular barn -/
def totalPaintedArea (d : BarnDimensions) : ℝ :=
  2 * (2 * (d.width * d.height + d.length * d.height) + 2 * (d.width * d.length))

/-- Theorem stating that the total area to be painted for the given barn is 1368 square yards -/
theorem barn_painted_area :
  let d : BarnDimensions := { width := 12, length := 15, height := 6 }
  totalPaintedArea d = 1368 := by
  sorry

end barn_painted_area_l3014_301430


namespace parabola_hyperbola_tangency_l3014_301477

/-- The value of m for which the parabola y = x^2 + 2x + 3 and 
    the hyperbola y^2 - mx^2 = 5 are tangent to each other -/
def tangency_condition : ℝ := -26

/-- The equation of the parabola -/
def parabola (x y : ℝ) : Prop := y = x^2 + 2*x + 3

/-- The equation of the hyperbola -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 5

/-- Theorem stating that the parabola and hyperbola are tangent when m = -26 -/
theorem parabola_hyperbola_tangency :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola tangency_condition x y ∧
  ∀ (x' y' : ℝ), x' ≠ x → y' ≠ y → 
    ¬(parabola x' y' ∧ hyperbola tangency_condition x' y') :=
sorry

end parabola_hyperbola_tangency_l3014_301477


namespace g_sum_at_two_l3014_301476

/-- Given a function g(x) = ax^8 + bx^6 - cx^4 + dx^2 + 5 where g(2) = 4, 
    prove that g(2) + g(-2) = 8 -/
theorem g_sum_at_two (a b c d : ℝ) :
  let g := fun x : ℝ => a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5
  g 2 = 4 → g 2 + g (-2) = 8 := by
  sorry

end g_sum_at_two_l3014_301476


namespace f_property_f_at_two_l3014_301499

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_property (x : ℝ) : f x = (deriv f 1) * Real.exp (x - 1) - f 0 * x + (1/2) * x^2 := sorry

theorem f_at_two : f 2 = Real.exp 2 := by sorry

end f_property_f_at_two_l3014_301499


namespace eight_real_numbers_inequality_l3014_301473

theorem eight_real_numbers_inequality (x : Fin 8 → ℝ) (h : Function.Injective x) :
  ∃ i j : Fin 8, i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (π / 7) := by
  sorry

end eight_real_numbers_inequality_l3014_301473


namespace proportion_equality_l3014_301435

theorem proportion_equality (x y : ℝ) (h1 : y ≠ 0) (h2 : 3 * x = 4 * y) : x / 4 = y / 3 := by
  sorry

end proportion_equality_l3014_301435


namespace cricket_run_rate_theorem_l3014_301431

/-- Represents a cricket game scenario --/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs --/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let firstPartRuns := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.target - firstPartRuns
  remainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given scenario --/
theorem cricket_run_rate_theorem (game : CricketGame) 
    (h1 : game.totalOvers = 50)
    (h2 : game.firstPartOvers = 10)
    (h3 : game.firstPartRunRate = 3.4)
    (h4 : game.target = 282) :
  requiredRunRate game = 6.2 := by
  sorry

#eval requiredRunRate {
  totalOvers := 50,
  firstPartOvers := 10,
  firstPartRunRate := 3.4,
  target := 282
}

end cricket_run_rate_theorem_l3014_301431


namespace sum_equals_140_l3014_301442

theorem sum_equals_140 (p q r s : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (h1 : p^2 + q^2 = 2500)
  (h2 : r^2 + s^2 = 2500)
  (h3 : p * r = 1200)
  (h4 : q * s = 1200) :
  p + q + r + s = 140 := by
sorry

end sum_equals_140_l3014_301442


namespace inscribed_square_area_l3014_301462

/-- The area of a square inscribed in a right triangle with hypotenuse 100 units and one leg 35 units -/
theorem inscribed_square_area (h : ℝ) (l : ℝ) (s : ℝ) 
  (hyp : h = 100)  -- hypotenuse length
  (leg : l = 35)   -- one leg length
  (square : s^2 = l * (h - l)) : -- s is the side length of the inscribed square
  s^2 = 2275 := by
  sorry

end inscribed_square_area_l3014_301462


namespace coefficient_of_x_cubed_l3014_301498

def expansion (x : ℝ) := (1 + x^2) * (1 - x)^5

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expansion))) 0 / 6 = -15 := by sorry

end coefficient_of_x_cubed_l3014_301498


namespace triangle_angle_measure_l3014_301453

theorem triangle_angle_measure (D E F : ℝ) (h1 : D + E + F = 180)
  (h2 : F = 3 * E) (h3 : E = 15) : D = 120 := by
  sorry

end triangle_angle_measure_l3014_301453


namespace set_equality_implies_difference_l3014_301454

theorem set_equality_implies_difference (a b : ℝ) : 
  ({1, a + b, a} : Set ℝ) = {0, b / a, b} → b - a = 2 := by
  sorry

end set_equality_implies_difference_l3014_301454


namespace tangent_product_inequality_l3014_301459

theorem tangent_product_inequality (a b c : ℝ) (α β : ℝ) :
  a + b < 3 * c →
  Real.tan (α / 2) * Real.tan (β / 2) = (a + b - c) / (a + b + c) →
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 := by
  sorry

end tangent_product_inequality_l3014_301459


namespace tangent_parallel_implies_a_equals_e_max_value_when_a_positive_no_extreme_values_when_a_nonpositive_l3014_301401

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - 1 + a / Real.exp x

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - a / Real.exp x

theorem tangent_parallel_implies_a_equals_e (a : ℝ) :
  f_derivative a 1 = 0 → a = Real.exp 1 := by sorry

theorem max_value_when_a_positive (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), f a x = Real.log a ∧ 
  ∀ (y : ℝ), f a y ≤ f a x := by sorry

theorem no_extreme_values_when_a_nonpositive (a : ℝ) (h : a ≤ 0) :
  ∀ (x : ℝ), ∃ (y : ℝ), f a y > f a x := by sorry

end

end tangent_parallel_implies_a_equals_e_max_value_when_a_positive_no_extreme_values_when_a_nonpositive_l3014_301401


namespace fraction_problem_l3014_301480

theorem fraction_problem (t k : ℚ) (f : ℚ) : 
  t = f * (k - 32) → t = 75 → k = 167 → f = 5/9 := by
sorry

end fraction_problem_l3014_301480


namespace rectangle_area_with_inscribed_circle_l3014_301449

/-- The area of a rectangle with an inscribed circle of radius 7 and length-to-width ratio of 3:1 -/
theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * ratio * 2 * r = 588 := by
  sorry

end rectangle_area_with_inscribed_circle_l3014_301449


namespace five_fridays_in_august_l3014_301494

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Returns the number of occurrences of a given day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If July has five Tuesdays, then August has five Fridays -/
theorem five_fridays_in_august 
  (july : Month) 
  (h1 : july.days = 31)
  (h2 : countDayOccurrences july DayOfWeek.Tuesday = 5) :
  ∃ (august : Month), 
    august.days = 31 ∧ 
    august.firstDay = nextDay (nextDay (nextDay july.firstDay)) ∧
    countDayOccurrences august DayOfWeek.Friday = 5 :=
  sorry

end five_fridays_in_august_l3014_301494


namespace product_decimal_places_l3014_301409

/-- A function that returns the number of decimal places in a decimal number -/
def decimal_places (x : ℚ) : ℕ :=
  sorry

/-- The product of two decimal numbers with one and two decimal places respectively has three decimal places -/
theorem product_decimal_places (a b : ℚ) :
  decimal_places a = 1 → decimal_places b = 2 → decimal_places (a * b) = 3 :=
sorry

end product_decimal_places_l3014_301409


namespace sum_base3_equals_11000_l3014_301428

/-- Represents a number in base 3 --/
def Base3 : Type := List Nat

/-- Converts a base 3 number to its decimal representation --/
def to_decimal (n : Base3) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- Addition of two base 3 numbers --/
def add_base3 (a b : Base3) : Base3 :=
  sorry

/-- Theorem: The sum of 2₃, 22₃, 202₃, and 2022₃ is 11000₃ in base 3 --/
theorem sum_base3_equals_11000 :
  let a := [2]
  let b := [2, 2]
  let c := [2, 0, 2]
  let d := [2, 2, 0, 2]
  let result := [1, 1, 0, 0, 0]
  add_base3 (add_base3 (add_base3 a b) c) d = result :=
sorry

end sum_base3_equals_11000_l3014_301428


namespace polygon_sides_l3014_301461

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 2002 → 
  (n - 2) * 180 - 360 < sum_angles ∧ sum_angles < (n - 2) * 180 →
  n = 14 ∨ n = 15 := by
sorry

end polygon_sides_l3014_301461


namespace sin_cos_sum_equals_one_l3014_301429

theorem sin_cos_sum_equals_one :
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end sin_cos_sum_equals_one_l3014_301429


namespace clothes_spending_fraction_l3014_301436

theorem clothes_spending_fraction (initial_amount remaining_amount : ℝ) : 
  initial_amount = 1249.9999999999998 →
  remaining_amount = 500 →
  ∃ (F : ℝ), 
    F > 0 ∧ F < 1 ∧
    remaining_amount = (1 - 1/4) * (1 - 1/5) * (1 - F) * initial_amount ∧
    F = 1/3 := by
  sorry

end clothes_spending_fraction_l3014_301436


namespace sequence_formula_l3014_301403

theorem sequence_formula (a : ℕ+ → ℚ) :
  (∀ n : ℕ+, a (n + 1) / a n = (n + 2 : ℚ) / n) →
  a 1 = 1 →
  ∀ n : ℕ+, a n = n * (n + 1) / 2 := by
sorry

end sequence_formula_l3014_301403


namespace first_part_length_l3014_301415

/-- Proves that given a 60 km trip with two parts, where the second part is traveled at half the speed
    of the first part, and the average speed of the entire trip is 32 km/h, the length of the first
    part of the trip is 30 km. -/
theorem first_part_length
  (total_distance : ℝ)
  (speed_first_part : ℝ)
  (speed_second_part : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : speed_second_part = speed_first_part / 2)
  (h3 : average_speed = 32)
  : ∃ (first_part_length : ℝ),
    first_part_length = 30 ∧
    first_part_length / speed_first_part +
    (total_distance - first_part_length) / speed_second_part =
    total_distance / average_speed :=
by sorry

end first_part_length_l3014_301415


namespace max_appearances_day_numbers_l3014_301441

-- Define the cube size
def n : ℕ := 2018

-- Define a function that returns the number of times a day number appears
def day_number_appearances (i : ℕ) : ℕ :=
  if i ≤ n then
    i * (i + 1) / 2
  else if n < i ∧ i < 2 * n - 1 then
    (i + 1 - n) * (3 * n - i - 1) / 2
  else if 2 * n - 1 ≤ i ∧ i ≤ 3 * n - 2 then
    day_number_appearances (3 * n - 1 - i)
  else
    0

-- Define the maximum day number
def max_day_number : ℕ := 3 * n - 2

-- State the theorem
theorem max_appearances_day_numbers :
  ∀ k : ℕ, k ≤ max_day_number →
    day_number_appearances k ≤ day_number_appearances 3026 ∧
    day_number_appearances k ≤ day_number_appearances 3027 ∧
    (day_number_appearances 3026 = day_number_appearances 3027) :=
by sorry

end max_appearances_day_numbers_l3014_301441


namespace pharmacist_weights_existence_l3014_301418

theorem pharmacist_weights_existence :
  ∃ (a b c : ℝ), 
    a < b ∧ b < c ∧
    a + b = 100 ∧
    a + c = 101 ∧
    b + c = 102 ∧
    a < 90 ∧ b < 90 ∧ c < 90 := by
  sorry

end pharmacist_weights_existence_l3014_301418


namespace units_digit_factorial_sum_l3014_301414

theorem units_digit_factorial_sum : 
  (1 + 2 + 6 + (24 % 10)) % 10 = 3 := by sorry

end units_digit_factorial_sum_l3014_301414


namespace d_negative_iff_b_decreasing_l3014_301466

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- The sequence b_n defined as 2^(a_n) -/
def bSequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 2^(a n)

/-- A decreasing sequence -/
def isDecreasing (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) < b n

theorem d_negative_iff_b_decreasing
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h1 : arithmeticSequence a d)
  (h2 : bSequence a b) :
  d < 0 ↔ isDecreasing b :=
sorry

end d_negative_iff_b_decreasing_l3014_301466


namespace grace_total_pennies_l3014_301444

/-- The value of a coin in pennies -/
def coin_value : ℕ := 10

/-- The value of a nickel in pennies -/
def nickel_value : ℕ := 5

/-- The number of coins Grace has -/
def grace_coins : ℕ := 10

/-- The number of nickels Grace has -/
def grace_nickels : ℕ := 10

/-- The total number of pennies Grace will have after exchanging her coins and nickels -/
theorem grace_total_pennies : 
  grace_coins * coin_value + grace_nickels * nickel_value = 150 := by
  sorry

end grace_total_pennies_l3014_301444


namespace trigonometric_identity_l3014_301484

theorem trigonometric_identity (α : ℝ) (m : ℝ) (h : Real.sin α - Real.cos α = m) :
  (Real.sin (4 * α) + Real.sin (10 * α) - Real.sin (6 * α)) /
  (Real.cos (2 * α) + 1 - 2 * Real.sin (4 * α) ^ 2) = 2 * (1 - m ^ 2) := by
  sorry

end trigonometric_identity_l3014_301484


namespace john_weekly_income_l3014_301464

/-- Represents the number of crab baskets John reels in each time he collects crabs -/
def baskets_per_collection : ℕ := 3

/-- Represents the number of crabs each basket holds -/
def crabs_per_basket : ℕ := 4

/-- Represents the number of times John collects crabs per week -/
def collections_per_week : ℕ := 2

/-- Represents the selling price of each crab in dollars -/
def price_per_crab : ℕ := 3

/-- Calculates John's weekly income from selling crabs -/
def weekly_income : ℕ := baskets_per_collection * crabs_per_basket * collections_per_week * price_per_crab

/-- Theorem stating that John's weekly income from selling crabs is $72 -/
theorem john_weekly_income : weekly_income = 72 := by
  sorry

end john_weekly_income_l3014_301464


namespace not_always_swappable_renumbering_l3014_301400

-- Define a type for cities
def City : Type := ℕ

-- Define a type for the connection list
def ConnectionList : Type := List (City × City)

-- Function to check if a list is valid (placeholder)
def isValidList (list : ConnectionList) : Prop := sorry

-- Function to represent renumbering of cities
def renumber (oldNum newNum : City) (list : ConnectionList) : ConnectionList := sorry

-- Theorem statement
theorem not_always_swappable_renumbering :
  ∃ (list : ConnectionList) (M N : City),
    isValidList list ∧
    (∀ X Y : City, isValidList (renumber X Y list)) ∧
    ¬(isValidList (renumber M N (renumber N M list))) :=
sorry

end not_always_swappable_renumbering_l3014_301400


namespace odd_function_and_inequality_l3014_301407

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (b - 2^x) / (2^(x+1) + a)

theorem odd_function_and_inequality (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1) ∧
  (∀ t, f 2 1 (t^2 - 2*t) + f 2 1 (2*t^2 - 1) < 0 ↔ t > 1 ∨ t < -1/3) :=
sorry

end odd_function_and_inequality_l3014_301407


namespace proportional_relation_l3014_301413

theorem proportional_relation (x y z : ℝ) (k₁ k₂ : ℝ) :
  (∃ m : ℝ, x = m * y^3) →  -- x is directly proportional to y^3
  (∃ n : ℝ, y * z = n) →    -- y is inversely proportional to z
  (x = 5 ∧ z = 16) →        -- x = 5 when z = 16
  (z = 64 → x = 5/64) :=    -- x = 5/64 when z = 64
by sorry

end proportional_relation_l3014_301413


namespace blue_markers_count_l3014_301483

theorem blue_markers_count (total_markers red_markers : ℕ) 
  (h1 : total_markers = 3343)
  (h2 : red_markers = 2315) :
  total_markers - red_markers = 1028 := by
  sorry

end blue_markers_count_l3014_301483


namespace bicyclist_average_speed_l3014_301402

/-- Proves that the average speed of a bicyclist is 18 km/h given the specified conditions -/
theorem bicyclist_average_speed :
  let total_distance : ℝ := 450
  let first_part_distance : ℝ := 300
  let second_part_distance : ℝ := total_distance - first_part_distance
  let first_part_speed : ℝ := 20
  let second_part_speed : ℝ := 15
  let first_part_time : ℝ := first_part_distance / first_part_speed
  let second_part_time : ℝ := second_part_distance / second_part_speed
  let total_time : ℝ := first_part_time + second_part_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 18 := by
  sorry

end bicyclist_average_speed_l3014_301402


namespace sum_of_subtraction_equation_l3014_301487

theorem sum_of_subtraction_equation :
  ∀ A B : ℕ,
    A ≠ B →
    A < 10 →
    B < 10 →
    (80 + A) - (10 * B + 2) = 45 →
    A + B = 11 := by
  sorry

end sum_of_subtraction_equation_l3014_301487


namespace yogurt_expiration_probability_l3014_301447

def total_boxes : ℕ := 6
def expired_boxes : ℕ := 2
def selected_boxes : ℕ := 2

def probability_at_least_one_expired : ℚ := 3/5

theorem yogurt_expiration_probability :
  (Nat.choose total_boxes selected_boxes - Nat.choose (total_boxes - expired_boxes) selected_boxes) /
  Nat.choose total_boxes selected_boxes = probability_at_least_one_expired :=
sorry

end yogurt_expiration_probability_l3014_301447


namespace middle_quad_area_proportion_l3014_301412

-- Define a convex quadrilateral
def ConvexQuadrilateral : Type := Unit

-- Define a function to represent the area of a quadrilateral
def area (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the middle quadrilateral formed by connecting points
def middleQuadrilateral (q : ConvexQuadrilateral) : ConvexQuadrilateral := sorry

-- State the theorem
theorem middle_quad_area_proportion (q : ConvexQuadrilateral) :
  area (middleQuadrilateral q) = (1 / 25) * area q := by sorry

end middle_quad_area_proportion_l3014_301412


namespace inverse_proportion_problem_l3014_301427

-- Define the inverse proportionality relationship
def inverse_proportional (α β : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ α * β = k

-- State the theorem
theorem inverse_proportion_problem (α₁ α₂ β₁ β₂ : ℝ) :
  inverse_proportional α₁ β₁ →
  α₁ = 2 →
  β₁ = 5 →
  β₂ = -10 →
  inverse_proportional α₂ β₂ →
  α₂ = -1 :=
by sorry

end inverse_proportion_problem_l3014_301427


namespace mikes_ride_length_mikes_ride_length_proof_l3014_301493

/-- Proves that Mike's ride was 35 miles long given the taxi fare conditions -/
theorem mikes_ride_length : ℝ → Prop :=
  fun T =>
    let mike_start : ℝ := 3
    let mike_per_mile : ℝ := 0.3
    let mike_surcharge : ℝ := 1.5
    let annie_start : ℝ := 3.5
    let annie_per_mile : ℝ := 0.25
    let annie_toll : ℝ := 5
    let annie_surcharge : ℝ := 2
    let annie_miles : ℝ := 18
    ∀ M : ℝ,
      (mike_start + mike_per_mile * M + mike_surcharge = T) ∧
      (annie_start + annie_per_mile * annie_miles + annie_toll + annie_surcharge = T) →
      M = 35

/-- Proof of the theorem -/
theorem mikes_ride_length_proof : ∀ T : ℝ, mikes_ride_length T :=
  fun T => by
    -- Proof goes here
    sorry

end mikes_ride_length_mikes_ride_length_proof_l3014_301493


namespace quadratic_roots_relation_l3014_301432

theorem quadratic_roots_relation (a b c : ℚ) : 
  (∃ (r s : ℚ), (4 * r^2 + 2 * r - 9 = 0) ∧ 
                 (4 * s^2 + 2 * s - 9 = 0) ∧ 
                 (a * (r - 3)^2 + b * (r - 3) + c = 0) ∧
                 (a * (s - 3)^2 + b * (s - 3) + c = 0)) →
  c = 51 / 4 := by
sorry

end quadratic_roots_relation_l3014_301432


namespace square_of_binomial_l3014_301420

theorem square_of_binomial (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 9*x^2 - 18*x + a = (b*x + c)^2) → a = 9 := by
  sorry

end square_of_binomial_l3014_301420


namespace gcd_2024_2048_l3014_301456

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end gcd_2024_2048_l3014_301456


namespace gold_heart_necklace_cost_gold_heart_necklace_cost_proof_l3014_301448

/-- The cost of a gold heart necklace given the following conditions:
  * Bracelets cost $15 each
  * Personalized coffee mug costs $20
  * Raine buys 3 bracelets, 2 gold heart necklaces, and 1 coffee mug
  * Raine pays with a $100 bill and gets $15 change
-/
theorem gold_heart_necklace_cost : ℝ :=
  let bracelet_cost : ℝ := 15
  let mug_cost : ℝ := 20
  let num_bracelets : ℕ := 3
  let num_necklaces : ℕ := 2
  let num_mugs : ℕ := 1
  let payment : ℝ := 100
  let change : ℝ := 15
  let total_spent : ℝ := payment - change
  let necklace_cost : ℝ := (total_spent - (bracelet_cost * num_bracelets + mug_cost * num_mugs)) / num_necklaces
  10

theorem gold_heart_necklace_cost_proof : gold_heart_necklace_cost = 10 := by
  sorry

end gold_heart_necklace_cost_gold_heart_necklace_cost_proof_l3014_301448


namespace inconsistent_age_sum_l3014_301486

theorem inconsistent_age_sum (total_students : ℕ) (class_avg_age : ℝ)
  (group1_size group2_size group3_size unknown_size : ℕ)
  (group1_avg_age group2_avg_age group3_avg_age : ℝ)
  (unknown_sum_age : ℝ) :
  total_students = 25 →
  class_avg_age = 18 →
  group1_size = 8 →
  group2_size = 10 →
  group3_size = 5 →
  unknown_size = 2 →
  group1_avg_age = 16 →
  group2_avg_age = 20 →
  group3_avg_age = 17 →
  unknown_sum_age = 35 →
  total_students = group1_size + group2_size + group3_size + unknown_size →
  ¬(class_avg_age * total_students =
    group1_avg_age * group1_size + group2_avg_age * group2_size +
    group3_avg_age * group3_size + unknown_sum_age) :=
by sorry

end inconsistent_age_sum_l3014_301486


namespace max_distinct_squares_sum_2100_l3014_301421

/-- The sum of squares of a list of natural numbers -/
def sum_of_squares (lst : List Nat) : Nat :=
  lst.map (· ^ 2) |>.sum

/-- A proposition stating that a list of natural numbers has distinct elements -/
def is_distinct (lst : List Nat) : Prop :=
  lst.Nodup

theorem max_distinct_squares_sum_2100 :
  (∃ (n : Nat) (lst : List Nat), 
    lst.length = n ∧ 
    is_distinct lst ∧ 
    sum_of_squares lst = 2100 ∧
    ∀ (m : Nat) (lst' : List Nat), 
      lst'.length = m ∧ 
      is_distinct lst' ∧ 
      sum_of_squares lst' = 2100 → 
      m ≤ n) ∧
  (∃ (lst : List Nat), 
    lst.length = 17 ∧ 
    is_distinct lst ∧ 
    sum_of_squares lst = 2100) :=
by
  sorry


end max_distinct_squares_sum_2100_l3014_301421


namespace card_sum_difference_l3014_301474

theorem card_sum_difference (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 4)
  (h2 : ∀ m, m ≤ 2*n + 4 → ⌊a m⌋ = m) :
  ∃ i j k l, 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    i ≤ 2*n + 4 ∧ j ≤ 2*n + 4 ∧ k ≤ 2*n + 4 ∧ l ≤ 2*n + 4 ∧
    |a i + a j - a k - a l| < 1 / (n - Real.sqrt (n / 2)) :=
sorry

end card_sum_difference_l3014_301474


namespace equal_chore_time_l3014_301437

/-- Represents the time taken for each chore in minutes -/
structure ChoreTime where
  sweeping : ℕ
  washing : ℕ
  laundry : ℕ

/-- Represents the chores assigned to each child -/
structure Chores where
  rooms : ℕ
  dishes : ℕ
  loads : ℕ

def total_time (ct : ChoreTime) (c : Chores) : ℕ :=
  ct.sweeping * c.rooms + ct.washing * c.dishes + ct.laundry * c.loads

theorem equal_chore_time (ct : ChoreTime) (anna billy : Chores) : 
  ct.sweeping = 3 → 
  ct.washing = 2 → 
  ct.laundry = 9 → 
  anna.rooms = 10 → 
  anna.dishes = 0 → 
  anna.loads = 0 → 
  billy.rooms = 0 → 
  billy.loads = 2 → 
  total_time ct anna = total_time ct billy → 
  billy.dishes = 6 := by
  sorry

end equal_chore_time_l3014_301437


namespace expansion_coefficient_ratio_l3014_301481

theorem expansion_coefficient_ratio (n : ℕ) : 
  (∀ a b : ℝ, (4 : ℝ)^n / (2 : ℝ)^n = 64) → n = 6 := by
  sorry

end expansion_coefficient_ratio_l3014_301481


namespace twenty_six_billion_scientific_notation_l3014_301460

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_six_billion_scientific_notation :
  toScientificNotation (26 * 10^9) = ScientificNotation.mk 2.6 9 sorry := by
  sorry

end twenty_six_billion_scientific_notation_l3014_301460


namespace complex_sum_problem_l3014_301424

/-- Given three complex numbers and conditions, prove that s+u = -1 -/
theorem complex_sum_problem (p q r s t u : ℝ) : 
  q = 5 →
  t = -p - r →
  (p + q * I) + (r + s * I) + (t + u * I) = 4 * I →
  s + u = -1 := by
  sorry

end complex_sum_problem_l3014_301424


namespace large_pizza_has_16_slices_l3014_301445

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := sorry

/-- The number of large pizzas -/
def num_large_pizzas : ℕ := 2

/-- The number of small pizzas -/
def num_small_pizzas : ℕ := 2

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The total number of slices eaten -/
def total_slices_eaten : ℕ := 48

theorem large_pizza_has_16_slices :
  num_large_pizzas * large_pizza_slices + num_small_pizzas * small_pizza_slices = total_slices_eaten →
  large_pizza_slices = 16 := by
  sorry

end large_pizza_has_16_slices_l3014_301445


namespace complement_of_A_in_U_l3014_301433

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem complement_of_A_in_U :
  (U \ A) = {x | (1 ≤ x ∧ x < 2) ∨ x = 5} := by sorry

end complement_of_A_in_U_l3014_301433


namespace coplanar_vectors_lambda_l3014_301457

/-- Given three vectors a, b, and c in ℝ³, if they are coplanar and have specific coordinates,
    then the third coordinate of c equals 9. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) : 
  a = (2, -1, 3) → b = (-1, 4, -2) → c.1 = 7 → c.2.1 = 7 →
  (∃ (m n : ℝ), c = m • a + n • b) →
  c.2.2 = 9 := by
sorry

end coplanar_vectors_lambda_l3014_301457


namespace jason_work_hours_l3014_301423

def after_school_rate : ℝ := 4.00
def saturday_rate : ℝ := 6.00
def total_earnings : ℝ := 88.00
def saturday_hours : ℝ := 8

def total_hours : ℝ := 18

theorem jason_work_hours :
  ∃ (after_school_hours : ℝ),
    after_school_hours * after_school_rate + saturday_hours * saturday_rate = total_earnings ∧
    after_school_hours + saturday_hours = total_hours :=
by sorry

end jason_work_hours_l3014_301423


namespace factorial_fraction_equality_l3014_301471

theorem factorial_fraction_equality : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end factorial_fraction_equality_l3014_301471


namespace sin_product_equals_one_sixty_fourth_l3014_301468

theorem sin_product_equals_one_sixty_fourth :
  (Real.sin (70 * π / 180))^2 * (Real.sin (50 * π / 180))^2 * (Real.sin (10 * π / 180))^2 = 1/64 := by
  sorry

end sin_product_equals_one_sixty_fourth_l3014_301468


namespace smallest_zero_one_divisible_by_225_is_11111111100_smallest_zero_one_divisible_by_225_properties_l3014_301475

/-- A function that checks if all digits of a natural number are 0 or 1 -/
def all_digits_zero_or_one (n : ℕ) : Prop := sorry

/-- A function that returns the smallest natural number with digits 0 or 1 divisible by 225 -/
noncomputable def smallest_zero_one_divisible_by_225 : ℕ := sorry

theorem smallest_zero_one_divisible_by_225_is_11111111100 :
  smallest_zero_one_divisible_by_225 = 11111111100 :=
by
  sorry

theorem smallest_zero_one_divisible_by_225_properties :
  let n := smallest_zero_one_divisible_by_225
  all_digits_zero_or_one n ∧ n % 225 = 0 ∧ 
  ∀ m : ℕ, m < n → ¬(all_digits_zero_or_one m ∧ m % 225 = 0) :=
by
  sorry

end smallest_zero_one_divisible_by_225_is_11111111100_smallest_zero_one_divisible_by_225_properties_l3014_301475


namespace biker_journey_west_distance_l3014_301478

/-- Represents the journey of a biker -/
structure BikerJourney where
  west : ℝ
  north1 : ℝ
  east : ℝ
  north2 : ℝ
  straightLineDistance : ℝ

/-- Theorem stating the distance traveled west given specific journey parameters -/
theorem biker_journey_west_distance (journey : BikerJourney) 
  (h1 : journey.north1 = 5)
  (h2 : journey.east = 4)
  (h3 : journey.north2 = 15)
  (h4 : journey.straightLineDistance = 20.396078054371138) :
  journey.west = 8 := by
  sorry

end biker_journey_west_distance_l3014_301478


namespace expression_value_l3014_301405

theorem expression_value : 2 * Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + 2 * Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 20 := by
  sorry

end expression_value_l3014_301405


namespace f_x_plus_one_l3014_301495

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 + 4*(x + 1) - 5

-- State the theorem
theorem f_x_plus_one (x : ℝ) : f (x + 1) = x^2 + 8*x + 7 := by
  sorry

end f_x_plus_one_l3014_301495


namespace remainder_when_z_plus_3_div_9_is_integer_l3014_301497

theorem remainder_when_z_plus_3_div_9_is_integer (z : ℤ) :
  ∃ k : ℤ, (z + 3) / 9 = k → z ≡ 6 [ZMOD 9] := by
  sorry

end remainder_when_z_plus_3_div_9_is_integer_l3014_301497


namespace cubic_function_value_l3014_301411

/-- Given a cubic function f(x) = ax³ + 3 where f(-2) = -5, prove that f(2) = 11 -/
theorem cubic_function_value (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^3 + 3) 
  (h2 : f (-2) = -5) : f 2 = 11 := by
  sorry

end cubic_function_value_l3014_301411


namespace value_of_d_l3014_301452

theorem value_of_d (a b c d : ℕ+) 
  (h1 : a^2 = c * (d + 29))
  (h2 : b^2 = c * (d - 29)) :
  d = 421 := by
  sorry

end value_of_d_l3014_301452
