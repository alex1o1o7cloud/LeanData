import Mathlib

namespace NUMINAMATH_CALUDE_mirror_pieces_l3856_385682

theorem mirror_pieces : ∃ P : ℕ, 
  (P > 0) ∧ 
  (P / 2 - 3 > 0) ∧
  ((P / 2 - 3) / 3 = 9) ∧
  (P = 60) := by
  sorry

end NUMINAMATH_CALUDE_mirror_pieces_l3856_385682


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3856_385654

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3856_385654


namespace NUMINAMATH_CALUDE_abc_divides_sum_power_seven_l3856_385646

theorem abc_divides_sum_power_seven
  (a b c : ℕ+)
  (h1 : a ∣ b^2)
  (h2 : b ∣ c^2)
  (h3 : c ∣ a^2) :
  (a * b * c) ∣ (a + b + c)^7 :=
by sorry

end NUMINAMATH_CALUDE_abc_divides_sum_power_seven_l3856_385646


namespace NUMINAMATH_CALUDE_circle_symmetry_sum_l3856_385685

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a circle is symmetric with respect to a line -/
def isSymmetric (circle : Circle) (line : Line) : Prop :=
  sorry

/-- The main theorem -/
theorem circle_symmetry_sum (circle : Circle) 
    (l₁ : Line) (l₂ : Line) :
    l₁ = Line.mk 1 (-1) 4 →
    l₂ = Line.mk 1 3 0 →
    isSymmetric circle l₁ →
    isSymmetric circle l₂ →
    circle.D + circle.E = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_sum_l3856_385685


namespace NUMINAMATH_CALUDE_weighted_average_salary_l3856_385604

/-- Represents the categories of employees in the departmental store -/
inductive EmployeeCategory
  | Manager
  | Associate
  | LeadCashier
  | SalesRepresentative

/-- Returns the number of employees for a given category -/
def employeeCount (category : EmployeeCategory) : Nat :=
  match category with
  | .Manager => 9
  | .Associate => 18
  | .LeadCashier => 6
  | .SalesRepresentative => 45

/-- Returns the average salary for a given category -/
def averageSalary (category : EmployeeCategory) : Nat :=
  match category with
  | .Manager => 4500
  | .Associate => 3500
  | .LeadCashier => 3000
  | .SalesRepresentative => 2500

/-- Calculates the total salary for all employees -/
def totalSalary : Nat :=
  (employeeCount .Manager * averageSalary .Manager) +
  (employeeCount .Associate * averageSalary .Associate) +
  (employeeCount .LeadCashier * averageSalary .LeadCashier) +
  (employeeCount .SalesRepresentative * averageSalary .SalesRepresentative)

/-- Calculates the total number of employees -/
def totalEmployees : Nat :=
  employeeCount .Manager +
  employeeCount .Associate +
  employeeCount .LeadCashier +
  employeeCount .SalesRepresentative

/-- Theorem stating that the weighted average salary is $3000 -/
theorem weighted_average_salary :
  totalSalary / totalEmployees = 3000 := by
  sorry


end NUMINAMATH_CALUDE_weighted_average_salary_l3856_385604


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3856_385687

-- Define the universal set U
def U : Finset Char := {'a', 'b', 'c', 'd'}

-- Define set A
def A : Finset Char := {'a', 'b'}

-- Define set B
def B : Finset Char := {'b', 'c', 'd'}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {'a', 'c', 'd'} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3856_385687


namespace NUMINAMATH_CALUDE_axisymmetric_triangle_is_isosceles_l3856_385652

/-- A triangle is axisymmetric if it has an axis of symmetry. -/
def IsAxisymmetric (t : Triangle) : Prop := sorry

/-- A triangle is isosceles if it has at least two sides of equal length. -/
def IsIsosceles (t : Triangle) : Prop := sorry

/-- If a triangle is axisymmetric, then it is isosceles. -/
theorem axisymmetric_triangle_is_isosceles (t : Triangle) :
  IsAxisymmetric t → IsIsosceles t := by
  sorry

end NUMINAMATH_CALUDE_axisymmetric_triangle_is_isosceles_l3856_385652


namespace NUMINAMATH_CALUDE_circle_equation_implies_y_to_x_equals_nine_l3856_385625

theorem circle_equation_implies_y_to_x_equals_nine (x y : ℝ) : 
  x^2 + y^2 - 4*x + 6*y + 13 = 0 → y^x = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_y_to_x_equals_nine_l3856_385625


namespace NUMINAMATH_CALUDE_problem_solution_l3856_385629

theorem problem_solution (x y : ℝ) : 
  x / y = 6 / 3 → y = 27 → x = 54 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3856_385629


namespace NUMINAMATH_CALUDE_prism_volume_l3856_385612

theorem prism_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 150 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3856_385612


namespace NUMINAMATH_CALUDE_determinant_max_value_l3856_385647

open Real

theorem determinant_max_value :
  let det (θ : ℝ) := 
    let a11 := 1
    let a12 := 1
    let a13 := 1
    let a21 := 1
    let a22 := 1 + sin θ ^ 2
    let a23 := 1
    let a31 := 1 + cos θ ^ 2
    let a32 := 1
    let a33 := 1
    a11 * (a22 * a33 - a23 * a32) - 
    a12 * (a21 * a33 - a23 * a31) + 
    a13 * (a21 * a32 - a22 * a31)
  ∀ θ : ℝ, det θ ≤ 1 ∧ ∃ θ₀ : ℝ, det θ₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_determinant_max_value_l3856_385647


namespace NUMINAMATH_CALUDE_centroid_locus_is_hyperbola_l3856_385676

/-- Given two complex points Z₁ and Z₂ with arguments θ and -θ respectively, 
    where 0 < θ < π/2, and the area of triangle OZ₁Z₂ is constant S, 
    prove that the locus of the centroid Z of triangle OZ₁Z₂ forms a hyperbola. -/
theorem centroid_locus_is_hyperbola 
  (θ : ℝ) 
  (h_θ_pos : 0 < θ) 
  (h_θ_lt_pi_half : θ < π/2) 
  (S : ℝ) 
  (h_S_pos : S > 0) 
  (Z₁ Z₂ : ℂ) 
  (h_Z₁_arg : Complex.arg Z₁ = θ) 
  (h_Z₂_arg : Complex.arg Z₂ = -θ) 
  (h_area : abs (Z₁.im * Z₂.re - Z₁.re * Z₂.im) / 2 = S) : 
  ∃ (a b : ℝ), ∀ (Z : ℂ), Z = (Z₁ + Z₂) / 3 → (Z.re / a)^2 - (Z.im / b)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_centroid_locus_is_hyperbola_l3856_385676


namespace NUMINAMATH_CALUDE_square_39_equals_square_40_minus_79_l3856_385674

theorem square_39_equals_square_40_minus_79 : 39^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_39_equals_square_40_minus_79_l3856_385674


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3856_385614

/-- The width of the painting in inches -/
def painting_width : ℝ := 20

/-- The height of the painting in inches -/
def painting_height : ℝ := 30

/-- The frame width on the sides in inches -/
def frame_side_width : ℝ := 5

/-- The frame width on the top and bottom in inches -/
def frame_top_bottom_width : ℝ := 3 * frame_side_width

/-- The area of the painting in square inches -/
def painting_area : ℝ := painting_width * painting_height

/-- The area of the framed painting in square inches -/
def framed_painting_area : ℝ := (painting_width + 2 * frame_side_width) * (painting_height + 2 * frame_top_bottom_width)

/-- The width of the framed painting in inches -/
def framed_painting_width : ℝ := painting_width + 2 * frame_side_width

/-- The height of the framed painting in inches -/
def framed_painting_height : ℝ := painting_height + 2 * frame_top_bottom_width

theorem framed_painting_ratio :
  framed_painting_area = 2 * painting_area ∧
  framed_painting_width / framed_painting_height = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3856_385614


namespace NUMINAMATH_CALUDE_min_altitude_inequality_l3856_385600

/-- The minimum altitude of a triangle, or zero if the points are collinear -/
noncomputable def min_altitude (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The triangle inequality for minimum altitudes -/
theorem min_altitude_inequality (A B C X : ℝ × ℝ) :
  min_altitude A B C ≤ min_altitude A B X + min_altitude A X C + min_altitude X B C := by
  sorry

end NUMINAMATH_CALUDE_min_altitude_inequality_l3856_385600


namespace NUMINAMATH_CALUDE_probability_is_one_third_l3856_385659

/-- A rectangle in the xy-plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability that a randomly chosen point (x,y) from the given rectangle satisfies x > 2y --/
def probability_x_gt_2y (rect : Rectangle) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle := {
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 1
  h_x := by norm_num
  h_y := by norm_num
}

theorem probability_is_one_third :
  probability_x_gt_2y problem_rectangle = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l3856_385659


namespace NUMINAMATH_CALUDE_adoption_fee_is_50_l3856_385681

/-- Represents the adoption fee for the cat -/
def adoption_fee : ℝ := sorry

/-- Represents the total vet visit costs for the first year -/
def vet_costs : ℝ := 500

/-- Represents the monthly food cost -/
def monthly_food_cost : ℝ := 25

/-- Represents the cost of toys Jenny bought -/
def jenny_toy_costs : ℝ := 200

/-- Represents Jenny's total spending on the cat in the first year -/
def jenny_total_spending : ℝ := 625

/-- Theorem stating that the adoption fee is $50 -/
theorem adoption_fee_is_50 : adoption_fee = 50 := by
  sorry

end NUMINAMATH_CALUDE_adoption_fee_is_50_l3856_385681


namespace NUMINAMATH_CALUDE_oil_leaked_before_is_6522_l3856_385628

/-- The amount of oil leaked before engineers started to fix the pipe -/
def oil_leaked_before : ℕ := 11687 - 5165

/-- Theorem stating that the amount of oil leaked before engineers started to fix the pipe is 6522 liters -/
theorem oil_leaked_before_is_6522 : oil_leaked_before = 6522 := by
  sorry

end NUMINAMATH_CALUDE_oil_leaked_before_is_6522_l3856_385628


namespace NUMINAMATH_CALUDE_factoring_expression_l3856_385616

theorem factoring_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l3856_385616


namespace NUMINAMATH_CALUDE_equality_and_inequality_of_exponents_l3856_385630

theorem equality_and_inequality_of_exponents (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (2 : ℝ)^x = (3 : ℝ)^y ∧ (3 : ℝ)^y = (4 : ℝ)^z) :
  2 * x = 4 * z ∧ 2 * x > 3 * y :=
by sorry

end NUMINAMATH_CALUDE_equality_and_inequality_of_exponents_l3856_385630


namespace NUMINAMATH_CALUDE_no_four_digit_sum12_div11and5_l3856_385670

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem no_four_digit_sum12_div11and5 :
  ¬ ∃ n : ℕ, is_four_digit n ∧ digit_sum n = 12 ∧ n % 11 = 0 ∧ n % 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_four_digit_sum12_div11and5_l3856_385670


namespace NUMINAMATH_CALUDE_hiking_rate_theorem_l3856_385699

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  rate_up : ℝ
  time : ℝ
  route_down_length : ℝ
  rate_down_multiplier : ℝ

/-- The hiking scenario satisfies the given conditions -/
def satisfies_conditions (h : HikingScenario) : Prop :=
  h.time = 2 ∧
  h.route_down_length = 12 ∧
  h.rate_down_multiplier = 1.5

/-- The theorem stating that under the given conditions, the rate going up is 4 miles per day -/
theorem hiking_rate_theorem (h : HikingScenario) 
  (hc : satisfies_conditions h) : h.rate_up = 4 := by
  sorry

end NUMINAMATH_CALUDE_hiking_rate_theorem_l3856_385699


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3856_385640

theorem floor_negative_seven_fourths : ⌊(-7/4 : ℚ)⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3856_385640


namespace NUMINAMATH_CALUDE_subset_intersection_cardinality_l3856_385695

theorem subset_intersection_cardinality (n m : ℕ) (Z : Finset ℕ) 
  (A : Fin m → Finset ℕ) : 
  (Z.card = n) →
  (∀ i : Fin m, A i ⊂ Z) →
  (∀ i j : Fin m, i ≠ j → (A i ∩ A j).card = 1) →
  m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_cardinality_l3856_385695


namespace NUMINAMATH_CALUDE_equation_solutions_l3856_385636

theorem equation_solutions :
  (∃ x : ℚ, x - 2 * (x - 4) = 3 * (1 - x) ∧ x = -5/2) ∧
  (∃ x : ℚ, (2 * x + 1) / 3 - (5 * x - 1) / 60 = 1 ∧ x = 39/35) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3856_385636


namespace NUMINAMATH_CALUDE_journey_time_properties_l3856_385658

/-- Represents the bus route system with given conditions -/
structure BusSystem where
  bus_a_interval : ℝ := 20
  total_distance : ℝ := 12
  journey_time : ℝ := 20
  bus_b_delay : ℝ := 10

/-- The maximum journey time function -/
noncomputable def f (bs : BusSystem) (x : ℝ) : ℝ :=
  sorry

/-- Main theorem about the journey time function -/
theorem journey_time_properties (bs : BusSystem) :
  f bs 2 = 23 ∧ f bs 4 = 23 ∧ 
  (∀ x, 0 < x → x < 12 → f bs x ≤ 25) ∧
  f bs 3 = 25 ∧ f bs 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_properties_l3856_385658


namespace NUMINAMATH_CALUDE_mary_money_left_l3856_385611

/-- The amount of money Mary has left after her purchases -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 4 * p
  let total_cost := 4 * drink_cost + medium_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that Mary will have 50 - 10p dollars left after her purchases -/
theorem mary_money_left (p : ℝ) : money_left p = 50 - 10 * p := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l3856_385611


namespace NUMINAMATH_CALUDE_marcus_and_leah_games_l3856_385680

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The function to calculate the number of games where two specific players play together -/
def games_together (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n - 2) (k - 2)

/-- The theorem stating that Marcus and Leah play together in 210 games -/
theorem marcus_and_leah_games : 
  games_together total_players players_per_game = 210 := by
  sorry

#eval games_together total_players players_per_game

end NUMINAMATH_CALUDE_marcus_and_leah_games_l3856_385680


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3856_385608

theorem quadratic_root_in_unit_interval (a b c : ℝ) (h : 2*a + 3*b + 6*c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3856_385608


namespace NUMINAMATH_CALUDE_integral_of_constant_one_equals_one_l3856_385641

-- Define the constant function f(x) = 1
def f : ℝ → ℝ := λ x => 1

-- State the theorem
theorem integral_of_constant_one_equals_one :
  ∫ x in (0:ℝ)..1, f x = 1 := by sorry

end NUMINAMATH_CALUDE_integral_of_constant_one_equals_one_l3856_385641


namespace NUMINAMATH_CALUDE_inequality_proof_l3856_385663

theorem inequality_proof (m n a : ℝ) (h : m > n) : a - m < a - n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3856_385663


namespace NUMINAMATH_CALUDE_petya_win_probability_l3856_385683

/-- The "Heap of Stones" game -/
structure HeapOfStones where
  initialStones : Nat
  minTake : Nat
  maxTake : Nat

/-- A player in the game -/
inductive Player
  | Petya
  | Computer

/-- The state of the game -/
structure GameState where
  stones : Nat
  currentPlayer : Player

/-- The outcome of the game -/
inductive GameOutcome
  | PetyaWins
  | ComputerWins

/-- A strategy for playing the game -/
def Strategy := GameState → Nat

/-- The random strategy that Petya uses -/
def randomStrategy : Strategy := sorry

/-- The optimal strategy that the computer uses -/
def optimalStrategy : Strategy := sorry

/-- Play the game with given strategies -/
def playGame (petyaStrategy : Strategy) (computerStrategy : Strategy) : GameOutcome := sorry

/-- The probability of Petya winning -/
def petyaWinProbability : ℚ := sorry

/-- Main theorem: The probability of Petya winning is 1/256 -/
theorem petya_win_probability :
  let game : HeapOfStones := ⟨16, 1, 4⟩
  petyaWinProbability = 1 / 256 := by sorry

end NUMINAMATH_CALUDE_petya_win_probability_l3856_385683


namespace NUMINAMATH_CALUDE_coronavirus_diameter_scientific_notation_l3856_385684

theorem coronavirus_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000125 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.25 ∧ n = -7 :=
by sorry

end NUMINAMATH_CALUDE_coronavirus_diameter_scientific_notation_l3856_385684


namespace NUMINAMATH_CALUDE_ron_height_l3856_385620

theorem ron_height (dean_height ron_height water_depth : ℝ) : 
  water_depth = 2 * dean_height →
  dean_height = ron_height - 8 →
  water_depth = 12 →
  ron_height = 14 := by
sorry

end NUMINAMATH_CALUDE_ron_height_l3856_385620


namespace NUMINAMATH_CALUDE_coefficient_x5_expansion_l3856_385649

/-- The coefficient of x^5 in the expansion of (1+x)^2(1-x)^5 is -1 -/
theorem coefficient_x5_expansion : Int := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_expansion_l3856_385649


namespace NUMINAMATH_CALUDE_container_volume_transformation_l3856_385609

/-- Represents a rectangular container with dimensions height, length, and width -/
structure Container where
  height : ℝ
  length : ℝ
  width : ℝ

/-- Calculates the volume of a container -/
def volume (c : Container) : ℝ := c.height * c.length * c.width

/-- Creates a new container by scaling the dimensions of an original container -/
def scaleContainer (c : Container) (h_scale l_scale w_scale : ℝ) : Container :=
  { height := c.height * h_scale,
    length := c.length * l_scale,
    width := c.width * w_scale }

theorem container_volume_transformation (original : Container) :
  volume original = 4 →
  volume (scaleContainer original 2 3 4) = 96 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_transformation_l3856_385609


namespace NUMINAMATH_CALUDE_fraction_equality_l3856_385619

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
    (h3 : (4 * a + b) / (a - 4 * b) = 3) : 
  (a + 4 * b) / (4 * a - b) = 9 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3856_385619


namespace NUMINAMATH_CALUDE_farmer_potatoes_l3856_385633

theorem farmer_potatoes (initial_tomatoes picked_tomatoes total_left : ℕ) 
  (h1 : initial_tomatoes = 177)
  (h2 : picked_tomatoes = 53)
  (h3 : total_left = 136) :
  initial_tomatoes - picked_tomatoes + (total_left - (initial_tomatoes - picked_tomatoes)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_farmer_potatoes_l3856_385633


namespace NUMINAMATH_CALUDE_min_value_theorem_l3856_385622

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2*m + n = 1) :
  (1/m + 2/n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2*m₀ + n₀ = 1 ∧ 1/m₀ + 2/n₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3856_385622


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3856_385626

theorem linear_equation_solution (V : ℝ → ℝ) (p q : ℝ) 
  (h1 : ∀ t, V t = p * t + q)
  (h2 : V 0 = 100)
  (h3 : V 10 = 103.5) :
  p = 0.35 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3856_385626


namespace NUMINAMATH_CALUDE_wall_length_is_800_l3856_385665

-- Define the dimensions of a single brick
def brick_length : ℝ := 100
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the wall dimensions
def wall_height : ℝ := 600 -- 6 m converted to cm
def wall_width : ℝ := 22.5

-- Define the number of bricks
def num_bricks : ℕ := 1600

-- Define the volume of a single brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Define the total volume of all bricks
def total_brick_volume : ℝ := brick_volume * num_bricks

-- Theorem stating the length of the wall
theorem wall_length_is_800 : 
  ∃ (wall_length : ℝ), wall_length * wall_height * wall_width = total_brick_volume ∧ wall_length = 800 := by
sorry

end NUMINAMATH_CALUDE_wall_length_is_800_l3856_385665


namespace NUMINAMATH_CALUDE_malcolm_total_followers_l3856_385601

/-- The total number of followers Malcolm has on all social media platforms --/
def total_followers (instagram facebook : ℕ) : ℕ :=
  let twitter := (instagram + facebook) / 2
  let tiktok := 3 * twitter
  let youtube := tiktok + 510
  instagram + facebook + twitter + tiktok + youtube

/-- Theorem stating that Malcolm's total followers across all platforms is 3840 --/
theorem malcolm_total_followers :
  total_followers 240 500 = 3840 := by
  sorry

#eval total_followers 240 500

end NUMINAMATH_CALUDE_malcolm_total_followers_l3856_385601


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l3856_385627

theorem arcsin_equation_solution :
  let f (x : ℝ) := Real.arcsin (x * Real.sqrt 5 / 3) + Real.arcsin (x * Real.sqrt 5 / 6) - Real.arcsin (7 * x * Real.sqrt 5 / 18)
  ∀ x : ℝ, 
    (abs x ≤ 18 / (7 * Real.sqrt 5)) →
    (f x = 0 ↔ x = 0 ∨ x = 8/7 ∨ x = -8/7) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l3856_385627


namespace NUMINAMATH_CALUDE_poster_board_side_length_l3856_385648

/-- Prove that a square poster board that can fit 24 rectangular cards
    measuring 2 inches by 3 inches has a side length of 1 foot. -/
theorem poster_board_side_length :
  ∀ (side_length : ℝ),
  (side_length * side_length = 24 * 2 * 3) →
  (side_length / 12 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_poster_board_side_length_l3856_385648


namespace NUMINAMATH_CALUDE_intersection_and_complement_when_m_2_existence_and_range_of_m_l3856_385692

def A : Set ℝ := {x | (x + 3) / (x - 1) ≤ 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - m*x - 2*m^2 ≤ 0}

theorem intersection_and_complement_when_m_2 :
  (A ∩ B 2 = {x | -2 ≤ x ∧ x < 1}) ∧
  ((Set.univ : Set ℝ) \ B 2 = {x | x < -2 ∨ x > 4}) := by sorry

theorem existence_and_range_of_m :
  ∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, x ∈ A ↔ x ∈ B m) ∧
  (∀ m' : ℝ, (∀ x : ℝ, x ∈ A ↔ x ∈ B m') → m' ≥ 3) := by sorry

end NUMINAMATH_CALUDE_intersection_and_complement_when_m_2_existence_and_range_of_m_l3856_385692


namespace NUMINAMATH_CALUDE_jordan_roller_skates_l3856_385621

/-- The number of pairs of old roller skates in Jordan's driveway -/
def num_roller_skate_pairs : ℕ := 2

theorem jordan_roller_skates :
  let num_cars : ℕ := 2
  let wheels_per_car : ℕ := 4
  let num_bikes : ℕ := 2
  let wheels_per_bike : ℕ := 2
  let num_trash_cans : ℕ := 1
  let wheels_per_trash_can : ℕ := 2
  let num_tricycles : ℕ := 1
  let wheels_per_tricycle : ℕ := 3
  let total_wheels : ℕ := 25
  let wheels_per_skate : ℕ := 2
  num_cars * wheels_per_car +
  num_bikes * wheels_per_bike +
  num_trash_cans * wheels_per_trash_can +
  num_tricycles * wheels_per_tricycle +
  num_roller_skate_pairs * 2 * wheels_per_skate = total_wheels :=
by
  sorry

#check jordan_roller_skates

end NUMINAMATH_CALUDE_jordan_roller_skates_l3856_385621


namespace NUMINAMATH_CALUDE_retail_overhead_expenses_l3856_385617

/-- A problem about calculating overhead expenses in retail --/
theorem retail_overhead_expenses 
  (purchase_price : ℝ) 
  (selling_price : ℝ) 
  (profit_percent : ℝ) 
  (h1 : purchase_price = 225)
  (h2 : selling_price = 300)
  (h3 : profit_percent = 25) :
  ∃ (overhead_expenses : ℝ),
    selling_price = (purchase_price + overhead_expenses) * (1 + profit_percent / 100) ∧
    overhead_expenses = 15 := by
  sorry

end NUMINAMATH_CALUDE_retail_overhead_expenses_l3856_385617


namespace NUMINAMATH_CALUDE_complex_coordinates_l3856_385605

theorem complex_coordinates (z : ℂ) : z = Complex.I * (2 - Complex.I) → (z.re = 1 ∧ z.im = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinates_l3856_385605


namespace NUMINAMATH_CALUDE_isabel_circuit_length_l3856_385624

/-- The length of Isabel's running circuit in meters. -/
def circuit_length : ℕ := 365

/-- The number of times Isabel runs the circuit in the morning. -/
def morning_runs : ℕ := 7

/-- The number of times Isabel runs the circuit in the afternoon. -/
def afternoon_runs : ℕ := 3

/-- The total distance Isabel runs in a week, in meters. -/
def weekly_distance : ℕ := 25550

/-- The number of days in a week. -/
def days_in_week : ℕ := 7

theorem isabel_circuit_length :
  circuit_length * (morning_runs + afternoon_runs) * days_in_week = weekly_distance :=
sorry

end NUMINAMATH_CALUDE_isabel_circuit_length_l3856_385624


namespace NUMINAMATH_CALUDE_max_value_abc_l3856_385645

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  a + a * b + a * b * c ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_l3856_385645


namespace NUMINAMATH_CALUDE_gcf_294_108_l3856_385610

theorem gcf_294_108 : Nat.gcd 294 108 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_294_108_l3856_385610


namespace NUMINAMATH_CALUDE_right_triangle_area_l3856_385643

theorem right_triangle_area (a b c : ℕ) : 
  a = 7 →                  -- One leg is 7
  a * a + b * b = c * c →  -- Pythagorean theorem (right triangle)
  a * b = 168 →            -- Area is 84 (2 * 84 = 168)
  (∃ (S : ℕ), S = 84 ∧ S = a * b / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3856_385643


namespace NUMINAMATH_CALUDE_pedros_daughters_l3856_385660

/-- The number of ice cream flavors available -/
def num_flavors : ℕ := 12

/-- The number of scoops in each child's combo -/
def scoops_per_combo : ℕ := 3

/-- The total number of scoops ordered for each flavor -/
def scoops_per_flavor : ℕ := 2

structure Family where
  num_boys : ℕ
  num_girls : ℕ

/-- Pedro's family satisfies the given conditions -/
def is_valid_family (f : Family) : Prop :=
  f.num_boys > 0 ∧ 
  f.num_girls > f.num_boys ∧
  (f.num_boys + f.num_girls) * scoops_per_combo = num_flavors * scoops_per_flavor ∧
  ∃ (boys_flavors girls_flavors : Finset ℕ), 
    boys_flavors.card = (3 * f.num_boys) / 2 ∧
    girls_flavors.card = (3 * f.num_girls) / 2 ∧
    boys_flavors ∩ girls_flavors = ∅ ∧
    boys_flavors ∪ girls_flavors = Finset.range num_flavors

theorem pedros_daughters (f : Family) (h : is_valid_family f) : f.num_girls = 6 :=
sorry

end NUMINAMATH_CALUDE_pedros_daughters_l3856_385660


namespace NUMINAMATH_CALUDE_min_value_expression_l3856_385669

theorem min_value_expression (x y : ℝ) : 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3856_385669


namespace NUMINAMATH_CALUDE_partnership_investment_l3856_385651

/-- Partnership investment problem -/
theorem partnership_investment (x : ℝ) (y : ℝ) : 
  x > 0 →  -- Raman's investment is positive
  y > 0 →  -- Lakshmi invests after a positive number of months
  y < 12 → -- Lakshmi invests before the end of the year
  (2 * x * (12 - y)) / (x * 12 + 2 * x * (12 - y) + 3 * x * 4) = 1 / 3 →
  y = 6 := by
sorry

end NUMINAMATH_CALUDE_partnership_investment_l3856_385651


namespace NUMINAMATH_CALUDE_fraction_of_books_sold_l3856_385697

/-- Given a collection of books where some were sold and some remained unsold,
    this theorem proves the fraction of books sold. -/
theorem fraction_of_books_sold
  (price_per_book : ℝ)
  (unsold_books : ℕ)
  (total_revenue : ℝ)
  (h1 : price_per_book = 3.5)
  (h2 : unsold_books = 40)
  (h3 : total_revenue = 280.00000000000006) :
  (total_revenue / price_per_book) / ((total_revenue / price_per_book) + unsold_books : ℝ) = 2/3 := by
  sorry

#eval (280.00000000000006 / 3.5) / ((280.00000000000006 / 3.5) + 40)

end NUMINAMATH_CALUDE_fraction_of_books_sold_l3856_385697


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l3856_385656

/-- Represents a parallelogram with side lengths -/
structure Parallelogram where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- The property that opposite sides of a parallelogram are equal -/
def Parallelogram.oppositeSidesEqual (p : Parallelogram) : Prop :=
  p.ab = p.cd ∧ p.bc = p.da

/-- The theorem to be proved -/
theorem parallelogram_side_length 
  (p : Parallelogram) 
  (h1 : p.oppositeSidesEqual) 
  (h2 : p.ab + p.bc + p.cd + p.da = 14) 
  (h3 : p.da = 5) : 
  p.ab = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l3856_385656


namespace NUMINAMATH_CALUDE_triangle_area_l3856_385644

/-- The area of a triangle with vertices at (-4,3), (0,6), and (2,-2) is 19 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (-4, 3)
  let B : ℝ × ℝ := (0, 6)
  let C : ℝ × ℝ := (2, -2)
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = 19 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l3856_385644


namespace NUMINAMATH_CALUDE_existence_of_larger_prime_factor_l3856_385673

theorem existence_of_larger_prime_factor (p : ℕ) (hp : Prime p) (hp_ge_3 : p ≥ 3) :
  ∃ N : ℕ, ∀ x ≥ N, ∃ i ∈ Finset.range ((p + 3) / 2), ∃ q : ℕ, Prime q ∧ q > p ∧ q ∣ (x + i + 1) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_larger_prime_factor_l3856_385673


namespace NUMINAMATH_CALUDE_negation_existence_to_universal_negation_of_existence_proposition_l3856_385661

theorem negation_existence_to_universal (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_existence_to_universal_negation_of_existence_proposition_l3856_385661


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3856_385675

theorem inverse_variation_problem (k : ℝ) (h : k > 0) :
  (∀ x y, x > 0 → y * Real.sqrt x = k) →
  (1/2 * Real.sqrt (1/4) = k) →
  (∃ x, x > 0 ∧ 8 * Real.sqrt x = k ∧ x = 1/1024) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3856_385675


namespace NUMINAMATH_CALUDE_age_difference_l3856_385637

theorem age_difference (a b : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 →  -- Ensuring a and b are single digits
  (10 * a + b) + 5 = 3 * ((10 * b + a) + 5) → -- Condition after 5 years
  (10 * a + b) - (10 * b + a) = 63 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3856_385637


namespace NUMINAMATH_CALUDE_similarity_of_triangles_l3856_385606

-- Define the points
variable (A B C D E F G H O : Point)

-- Define the cyclic quadrilateral ABCD
def is_cyclic_quad (A B C D : Point) : Prop := sorry

-- Define the circle centered at O passing through B and D
def circle_O_passes_through (O B D : Point) : Prop := sorry

-- Define that E and F are on lines BA and BC respectively
def E_on_BA (E B A : Point) : Prop := sorry
def F_on_BC (F B C : Point) : Prop := sorry

-- Define that E and F are distinct from A, B, C
def E_F_distinct (E F A B C : Point) : Prop := sorry

-- Define H as the orthocenter of triangle DEF
def H_orthocenter_DEF (H D E F : Point) : Prop := sorry

-- Define that AC, DO, and EF are concurrent
def lines_concurrent (A C D O E F : Point) : Prop := sorry

-- Define similarity of triangles
def triangles_similar (A B C E H F : Point) : Prop := sorry

-- Theorem statement
theorem similarity_of_triangles 
  (h1 : is_cyclic_quad A B C D)
  (h2 : circle_O_passes_through O B D)
  (h3 : E_on_BA E B A)
  (h4 : F_on_BC F B C)
  (h5 : E_F_distinct E F A B C)
  (h6 : H_orthocenter_DEF H D E F)
  (h7 : lines_concurrent A C D O E F) :
  triangles_similar A B C E H F :=
sorry

end NUMINAMATH_CALUDE_similarity_of_triangles_l3856_385606


namespace NUMINAMATH_CALUDE_unique_line_through_point_with_equal_intercepts_l3856_385613

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using the general form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def equal_intercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c / l.a = -l.c / l.b

theorem unique_line_through_point_with_equal_intercepts :
  ∃! l : Line2D, point_on_line ⟨0, 5⟩ l ∧ equal_intercepts l :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_point_with_equal_intercepts_l3856_385613


namespace NUMINAMATH_CALUDE_prop_negation_false_l3856_385689

theorem prop_negation_false (p q : Prop) : 
  ¬(¬(p ∧ q)) → (p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_prop_negation_false_l3856_385689


namespace NUMINAMATH_CALUDE_point_on_line_line_slope_is_one_line_equation_correct_l3856_385677

/-- A line passing through the point (1, 3) with slope 1 -/
def line (x y : ℝ) : Prop := x - y + 2 = 0

/-- The point (1, 3) lies on the line -/
theorem point_on_line : line 1 3 := by sorry

/-- The slope of the line is 1 -/
theorem line_slope_is_one :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ → line x₁ y₁ → line x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = 1 := by sorry

/-- The equation x - y + 2 = 0 represents the unique line passing through (1, 3) with slope 1 -/
theorem line_equation_correct :
  ∀ (x y : ℝ), (x - y + 2 = 0) ↔ (∃ (m b : ℝ), m = 1 ∧ y = m * (x - 1) + 3) := by sorry

end NUMINAMATH_CALUDE_point_on_line_line_slope_is_one_line_equation_correct_l3856_385677


namespace NUMINAMATH_CALUDE_lines_dont_form_triangle_iff_l3856_385618

/-- A line in 2D space represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The three lines given in the problem -/
def line1 : Line := ⟨4, 1, 4⟩
def line2 (m : ℝ) : Line := ⟨m, 1, 0⟩
def line3 (m : ℝ) : Line := ⟨2, -3*m, 4⟩

/-- The condition for the lines not forming a triangle -/
def lines_dont_form_triangle (m : ℝ) : Prop :=
  are_parallel line1 (line2 m) ∨ 
  are_parallel line1 (line3 m) ∨ 
  are_parallel (line2 m) (line3 m)

theorem lines_dont_form_triangle_iff (m : ℝ) : 
  lines_dont_form_triangle m ↔ m = 4 ∨ m = -1/6 := by sorry

end NUMINAMATH_CALUDE_lines_dont_form_triangle_iff_l3856_385618


namespace NUMINAMATH_CALUDE_ramsey_type_theorem_l3856_385668

theorem ramsey_type_theorem (n r : ℕ) (hn : n > 0) (hr : r > 0) :
  ∃ m : ℕ, m > 0 ∧
  (∀ (A : Fin r → Set ℕ),
    (∀ i j : Fin r, i ≠ j → A i ∩ A j = ∅) →
    (⋃ i : Fin r, A i) = Finset.range m →
    ∃ (i : Fin r) (a b : ℕ), a ∈ A i ∧ b ∈ A i ∧ b < a ∧ a ≤ (n + 1) * b / n) ∧
  (∀ k : ℕ, 0 < k → k < m →
    ∃ (A : Fin r → Set ℕ),
      (∀ i j : Fin r, i ≠ j → A i ∩ A j = ∅) ∧
      (⋃ i : Fin r, A i) = Finset.range k ∧
      ∀ (i : Fin r) (a b : ℕ), a ∈ A i → b ∈ A i → b < a → a > (n + 1) * b / n) ∧
  m = (n + 1) * r :=
by sorry

end NUMINAMATH_CALUDE_ramsey_type_theorem_l3856_385668


namespace NUMINAMATH_CALUDE_water_bucket_ratio_l3856_385666

/-- Given two partially filled buckets of water, a and b, prove that the ratio of water in bucket b 
    to bucket a after transferring 6 liters from b to a is 1:2, given the initial conditions. -/
theorem water_bucket_ratio : 
  ∀ (a b : ℝ),
  a = 13.2 →
  a - 6 = (1/3) * (b + 6) →
  (b - 6) / (a + 6) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_water_bucket_ratio_l3856_385666


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3856_385653

theorem arithmetic_sequence_length (a₁ aₙ d n : ℕ) : 
  a₁ = 6 → aₙ = 154 → d = 4 → aₙ = a₁ + (n - 1) * d → n = 38 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3856_385653


namespace NUMINAMATH_CALUDE_max_intersection_area_theorem_l3856_385607

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- Represents the maximum possible intersection area of two rectangles --/
def max_intersection_area (r1 r2 : Rectangle) : ℕ :=
  min r1.width r2.width * min r1.height r2.height

theorem max_intersection_area_theorem (r1 r2 : Rectangle) :
  r1.width < r1.height →
  r2.width > r2.height →
  2011 < area r1 →
  area r1 < 2020 →
  2011 < area r2 →
  area r2 < 2020 →
  max_intersection_area r1 r2 ≤ 1764 ∧
  ∃ (r1' r2' : Rectangle),
    r1'.width < r1'.height ∧
    r2'.width > r2'.height ∧
    2011 < area r1' ∧
    area r1' < 2020 ∧
    2011 < area r2' ∧
    area r2' < 2020 ∧
    max_intersection_area r1' r2' = 1764 := by
  sorry

#check max_intersection_area_theorem

end NUMINAMATH_CALUDE_max_intersection_area_theorem_l3856_385607


namespace NUMINAMATH_CALUDE_tan_equality_unique_solution_l3856_385632

theorem tan_equality_unique_solution : 
  ∃! (n : ℤ), -100 < n ∧ n < 100 ∧ Real.tan (n * π / 180) = Real.tan (216 * π / 180) :=
by
  -- The unique solution is n = 36
  use 36
  sorry

end NUMINAMATH_CALUDE_tan_equality_unique_solution_l3856_385632


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3856_385664

theorem complex_equation_solution (z : ℂ) :
  z * (Complex.I - Complex.I^2) = 1 + Complex.I^3 → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3856_385664


namespace NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l3856_385631

/-- A trapezoid with a 60° angle that has both inscribed and circumscribed circles -/
structure SpecialTrapezoid where
  /-- The measure of one angle of the trapezoid in degrees -/
  angle : ℝ
  /-- The trapezoid has an inscribed circle -/
  has_inscribed_circle : Prop
  /-- The trapezoid has a circumscribed circle -/
  has_circumscribed_circle : Prop
  /-- The angle measure is 60° -/
  angle_is_60 : angle = 60

/-- The ratio of the bases of the special trapezoid -/
def base_ratio (t : SpecialTrapezoid) : ℝ × ℝ :=
  (1, 3)

/-- Theorem: The ratio of the bases of a special trapezoid is 1:3 -/
theorem special_trapezoid_base_ratio (t : SpecialTrapezoid) :
  base_ratio t = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l3856_385631


namespace NUMINAMATH_CALUDE_equation_equality_l3856_385634

theorem equation_equality 
  (p q r x y z a b c : ℝ) 
  (h1 : p / x = q / y ∧ q / y = r / z) 
  (h2 : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1) :
  p^2 / a^2 + q^2 / b^2 + r^2 / c^2 = (p^2 + q^2 + r^2) / (x^2 + y^2 + z^2) := by
sorry

end NUMINAMATH_CALUDE_equation_equality_l3856_385634


namespace NUMINAMATH_CALUDE_theater_attendance_l3856_385615

/-- The number of men who spent Rs. 3 each on tickets -/
def num_men_standard : ℕ := 8

/-- The amount spent by each of the standard-paying men -/
def standard_price : ℚ := 3

/-- The total amount spent by all men -/
def total_spent : ℚ := 29.25

/-- The extra amount spent by the last man compared to the average -/
def extra_spent : ℚ := 2

theorem theater_attendance :
  ∃ (n : ℕ), n > 0 ∧
  (n : ℚ) * (total_spent / n) = 
    num_men_standard * standard_price + (total_spent / n + extra_spent) ∧
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_theater_attendance_l3856_385615


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3856_385693

theorem circle_area_ratio (X Y : ℝ) (h : (π / 2) * X = (π / 3) * Y) : 
  (π * X^2) / (π * Y^2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3856_385693


namespace NUMINAMATH_CALUDE_tangent_lines_max_area_and_slope_l3856_385672

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y - 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (-6, -5)

-- Define point N
def point_N : ℝ × ℝ := (1, 3)

-- Theorem for tangent lines
theorem tangent_lines :
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ x = -6) ∧
    (∀ x y, l₂ x y ↔ 3*x - 4*y - 2 = 0) ∧
    (∀ l, (∀ x y, l x y → circle_C x y) →
          (l (point_M.1) (point_M.2)) →
          (∃ x₀ y₀, circle_C x₀ y₀ ∧ l x₀ y₀ ∧
            ∀ x y, circle_C x y ∧ l x y → (x, y) = (x₀, y₀)) →
          (l = l₁ ∨ l = l₂)) :=
sorry

-- Theorem for maximum area and slope
theorem max_area_and_slope :
  ∃ (max_area : ℝ) (slope₁ slope₂ : ℝ),
    max_area = 8 ∧
    slope₁ = 2 * Real.sqrt 2 ∧
    slope₂ = -2 * Real.sqrt 2 ∧
    (∀ l : ℝ → ℝ → Prop,
      (l (point_N.1) (point_N.2)) →
      (∃ A B : ℝ × ℝ,
        circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
        l A.1 A.2 ∧ l B.1 B.2 ∧ A ≠ B) →
      (∃ C : ℝ × ℝ, C = point_N) →
      (∃ area : ℝ, area ≤ max_area) ∧
      (∃ k : ℝ, (k = slope₁ ∨ k = slope₂) →
        ∀ x y, l x y ↔ y - point_N.2 = k * (x - point_N.1))) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_max_area_and_slope_l3856_385672


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3856_385667

/-- The universal set U is the set of positive integers less than or equal to a -/
def U (a : ℝ) : Set ℕ := {x : ℕ | x > 0 ∧ x ≤ ⌊a⌋}

/-- Set P -/
def P : Set ℕ := {1, 2, 3}

/-- Set Q -/
def Q : Set ℕ := {4, 5, 6}

/-- The complement of set A in the universal set U -/
def complement (a : ℝ) (A : Set ℕ) : Set ℕ := (U a) \ A

theorem necessary_and_sufficient_condition (a : ℝ) :
  (6 ≤ a ∧ a < 7) ↔ complement a P = Q := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3856_385667


namespace NUMINAMATH_CALUDE_min_trips_for_28_containers_l3856_385655

/-- The minimum number of trips required to transport a given number of containers -/
def min_trips (total_containers : ℕ) (max_per_trip : ℕ) : ℕ :=
  (total_containers + max_per_trip - 1) / max_per_trip

theorem min_trips_for_28_containers :
  min_trips 28 5 = 6 := by
  sorry

#eval min_trips 28 5

end NUMINAMATH_CALUDE_min_trips_for_28_containers_l3856_385655


namespace NUMINAMATH_CALUDE_tenths_minus_hundredths_l3856_385602

theorem tenths_minus_hundredths : (0.5 : ℝ) - (0.05 : ℝ) = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_tenths_minus_hundredths_l3856_385602


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3856_385698

theorem algebraic_expression_value (a b : ℝ) (h1 : a = 3) (h2 : a - b = 1) : a^2 - a*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3856_385698


namespace NUMINAMATH_CALUDE_abs_condition_for_log_half_condition_l3856_385691

-- Define the logarithm with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

-- Statement of the theorem
theorem abs_condition_for_log_half_condition (x : ℝ) :
  (∀ x, |x - 2| < 1 → log_half (x + 2) < 0) ∧
  (∃ x, log_half (x + 2) < 0 ∧ |x - 2| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_abs_condition_for_log_half_condition_l3856_385691


namespace NUMINAMATH_CALUDE_bread_slices_calculation_l3856_385679

/-- Represents the number of pieces a single slice of bread is torn into -/
def pieces_per_slice : ℕ := 4

/-- Represents the total number of bread pieces -/
def total_pieces : ℕ := 8

/-- Calculates the number of original bread slices -/
def original_slices : ℕ := total_pieces / pieces_per_slice

theorem bread_slices_calculation :
  original_slices = 2 := by sorry

end NUMINAMATH_CALUDE_bread_slices_calculation_l3856_385679


namespace NUMINAMATH_CALUDE_series_convergence_l3856_385688

/-- The sum of the infinite series ∑(n=1 to ∞) [(n³+4n²+8n+8) / (3ⁿ·(n³+5))] converges to 1/2. -/
theorem series_convergence : 
  let f : ℕ → ℝ := λ n => (n^3 + 4*n^2 + 8*n + 8) / (3^n * (n^3 + 5))
  ∑' n, f n = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_convergence_l3856_385688


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3856_385623

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3856_385623


namespace NUMINAMATH_CALUDE_solution_for_E_l3856_385603

/-- The function E as defined in the problem -/
def E (a b c : ℚ) : ℚ := a * b^2 + c

/-- Theorem stating that -1/10 is the solution to E(a,4,5) = E(a,6,7) -/
theorem solution_for_E : 
  ∃ a : ℚ, E a 4 5 = E a 6 7 ∧ a = -1/10 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_E_l3856_385603


namespace NUMINAMATH_CALUDE_frog_path_count_l3856_385686

-- Define the octagon and frog movement
def Octagon := Fin 8
def adjacent (v : Octagon) : Set Octagon := {w | (v.val + 1) % 8 = w.val ∨ (v.val + 7) % 8 = w.val}

-- Define the path count function
noncomputable def a (n : ℕ) : ℝ :=
  if n % 2 = 1 then 0
  else ((2 + Real.sqrt 2) ^ ((n / 2) - 1) - (2 - Real.sqrt 2) ^ ((n / 2) - 1)) / Real.sqrt 2

-- State the theorem
theorem frog_path_count :
  ∀ n : ℕ, a n = (if n % 2 = 1 then 0
              else ((2 + Real.sqrt 2) ^ ((n / 2) - 1) - (2 - Real.sqrt 2) ^ ((n / 2) - 1)) / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_frog_path_count_l3856_385686


namespace NUMINAMATH_CALUDE_point_5_4_in_first_quadrant_l3856_385657

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def is_in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The theorem to be proved -/
theorem point_5_4_in_first_quadrant :
  let p : Point := ⟨5, 4⟩
  is_in_first_quadrant p := by
  sorry

end NUMINAMATH_CALUDE_point_5_4_in_first_quadrant_l3856_385657


namespace NUMINAMATH_CALUDE_dealer_gross_profit_l3856_385696

-- Define the parameters
def purchase_price : ℝ := 150
def markup_rate : ℝ := 0.25
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05

-- Define the initial selling price
noncomputable def initial_selling_price : ℝ :=
  purchase_price / (1 - markup_rate)

-- Define the discounted price
noncomputable def discounted_price : ℝ :=
  initial_selling_price * (1 - discount_rate)

-- Define the final selling price (including tax)
noncomputable def final_selling_price : ℝ :=
  discounted_price * (1 + tax_rate)

-- Define the gross profit
noncomputable def gross_profit : ℝ :=
  final_selling_price - purchase_price

-- Theorem statement
theorem dealer_gross_profit :
  gross_profit = 19 := by sorry

end NUMINAMATH_CALUDE_dealer_gross_profit_l3856_385696


namespace NUMINAMATH_CALUDE_brendas_age_l3856_385635

theorem brendas_age (addison janet brenda : ℝ) 
  (h1 : addison = 4 * brenda) 
  (h2 : janet = brenda + 10) 
  (h3 : addison = janet) : 
  brenda = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_brendas_age_l3856_385635


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3856_385694

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * s = 4 * Real.sqrt 241 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3856_385694


namespace NUMINAMATH_CALUDE_intersection_inequality_solution_l3856_385678

/-- Given two lines y = 3x + a and y = -2x + b that intersect at a point with x-coordinate -5,
    the solution set of the inequality 3x + a < -2x + b is {x ∈ ℝ | x < -5}. -/
theorem intersection_inequality_solution (a b : ℝ) :
  (∃ y, 3 * (-5) + a = y ∧ -2 * (-5) + b = y) →
  (∀ x, 3 * x + a < -2 * x + b ↔ x < -5) :=
by sorry

end NUMINAMATH_CALUDE_intersection_inequality_solution_l3856_385678


namespace NUMINAMATH_CALUDE_optimal_pricing_strategy_l3856_385642

/-- Represents the pricing strategy of a merchant -/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  selling_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price based on the list price and purchase discount -/
def purchase_price (m : MerchantPricing) : ℝ :=
  m.list_price * (1 - m.purchase_discount)

/-- Calculates the selling price based on the marked price and selling discount -/
def selling_price (m : MerchantPricing) : ℝ :=
  m.marked_price * (1 - m.selling_discount)

/-- Calculates the profit based on the selling price and purchase price -/
def profit (m : MerchantPricing) : ℝ :=
  selling_price m - purchase_price m

/-- Theorem stating the optimal marked price for the merchant's pricing strategy -/
theorem optimal_pricing_strategy (m : MerchantPricing) 
  (h1 : m.purchase_discount = 0.3)
  (h2 : m.selling_discount = 0.2)
  (h3 : m.profit_margin = 0.3)
  (h4 : profit m = m.profit_margin * selling_price m) :
  m.marked_price = 1.25 * m.list_price := by
  sorry


end NUMINAMATH_CALUDE_optimal_pricing_strategy_l3856_385642


namespace NUMINAMATH_CALUDE_olivia_basketball_cards_l3856_385639

theorem olivia_basketball_cards 
  (basketball_price : ℕ)
  (baseball_decks : ℕ)
  (baseball_price : ℕ)
  (total_paid : ℕ)
  (change : ℕ)
  (h1 : basketball_price = 3)
  (h2 : baseball_decks = 5)
  (h3 : baseball_price = 4)
  (h4 : total_paid = 50)
  (h5 : change = 24) :
  ∃ (x : ℕ), x * basketball_price + baseball_decks * baseball_price = total_paid - change ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_olivia_basketball_cards_l3856_385639


namespace NUMINAMATH_CALUDE_clever_calculation_l3856_385638

theorem clever_calculation :
  (1978 + 250 + 1022 + 750 = 4000) ∧
  (454 + 999 * 999 + 545 = 999000) ∧
  (999 + 998 + 997 + 996 + 1004 + 1003 + 1002 + 1001 = 8000) :=
by sorry

end NUMINAMATH_CALUDE_clever_calculation_l3856_385638


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3856_385650

theorem boys_to_girls_ratio (total_students girls : ℕ) 
  (h1 : total_students = 1040)
  (h2 : girls = 400) :
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3856_385650


namespace NUMINAMATH_CALUDE_second_investment_rate_l3856_385690

def contest_winnings : ℝ := 5000
def first_investment : ℝ := 1800
def first_interest_rate : ℝ := 0.05
def total_interest : ℝ := 298

def second_investment : ℝ := 2 * first_investment - 400

def first_interest : ℝ := first_investment * first_interest_rate

def second_interest : ℝ := total_interest - first_interest

theorem second_investment_rate (second_rate : ℝ) : 
  second_rate * second_investment = second_interest → second_rate = 0.065 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_rate_l3856_385690


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3856_385662

theorem sufficient_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 1 → a + b > 3 ∧ a * b > 2) ∧
  ∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 2 ∧ b > 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3856_385662


namespace NUMINAMATH_CALUDE_x_value_theorem_l3856_385671

theorem x_value_theorem (x y : ℝ) (h : x / (x - 2) = (y^2 + 3*y - 2) / (y^2 + 3*y + 1)) :
  x = 2*y^2 + 6*y + 4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_theorem_l3856_385671
