import Mathlib

namespace triangle_angle_value_l2996_299666

/-- Given a triangle ABC with angle C = 60°, angle A = x, and angle B = 2x,
    where x is also an alternate interior angle formed by a line intersecting two parallel lines,
    prove that x = 40°. -/
theorem triangle_angle_value (A B C : ℝ) (x : ℝ) : 
  A = x → B = 2*x → C = 60 → A + B + C = 180 → x = 40 := by sorry

end triangle_angle_value_l2996_299666


namespace game_c_higher_prob_l2996_299638

/-- A biased coin with probability of heads 3/5 and tails 2/5 -/
structure BiasedCoin where
  p_heads : ℚ
  p_tails : ℚ
  head_prob : p_heads = 3/5
  tail_prob : p_tails = 2/5
  total_prob : p_heads + p_tails = 1

/-- Game C: Win if all three outcomes are the same -/
def prob_win_game_c (coin : BiasedCoin) : ℚ :=
  coin.p_heads^3 + coin.p_tails^3

/-- Game D: Win if first two outcomes are the same and third is different -/
def prob_win_game_d (coin : BiasedCoin) : ℚ :=
  2 * (coin.p_heads^2 * coin.p_tails + coin.p_tails^2 * coin.p_heads)

/-- The main theorem stating that Game C has a 1/25 higher probability of winning -/
theorem game_c_higher_prob (coin : BiasedCoin) :
  prob_win_game_c coin - prob_win_game_d coin = 1/25 := by
  sorry

end game_c_higher_prob_l2996_299638


namespace tan_pi_sixth_minus_alpha_l2996_299689

theorem tan_pi_sixth_minus_alpha (α : ℝ) (h : Real.sin α = 3 * Real.sin (α - π / 3)) :
  Real.tan (π / 6 - α) = -2 * Real.sqrt 3 / 3 := by
  sorry

end tan_pi_sixth_minus_alpha_l2996_299689


namespace largest_prime_divisor_of_sum_of_squares_l2996_299616

theorem largest_prime_divisor_of_sum_of_squares :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (35^2 + 84^2) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (35^2 + 84^2) → q ≤ p :=
by sorry

end largest_prime_divisor_of_sum_of_squares_l2996_299616


namespace simplify_expression_1_expand_expression_2_simplify_expression_3_l2996_299649

-- Part 1
theorem simplify_expression_1 (m n : ℝ) :
  15 * m * n^2 + 5 * m * n * m^3 * n = 15 * m * n^2 + 5 * m^4 * n^2 := by sorry

-- Part 2
theorem expand_expression_2 (x : ℝ) :
  (3 * x + 1) * (2 * x - 5) = 6 * x^2 - 13 * x - 5 := by sorry

-- Part 3
theorem simplify_expression_3 :
  (-0.25)^2024 * 4^2023 = 0.25 := by sorry

end simplify_expression_1_expand_expression_2_simplify_expression_3_l2996_299649


namespace seven_nanometers_in_meters_l2996_299652

-- Define the conversion factor for nanometers to meters
def nanometer_to_meter : ℝ := 1e-9

-- Theorem statement
theorem seven_nanometers_in_meters :
  7 * nanometer_to_meter = 7e-9 := by
  sorry

end seven_nanometers_in_meters_l2996_299652


namespace two_rooks_non_attacking_placements_l2996_299680

/-- The size of a standard chessboard --/
def boardSize : Nat := 8

/-- The total number of squares on the chessboard --/
def totalSquares : Nat := boardSize * boardSize

/-- The number of squares a rook can attack (excluding its own square) --/
def rookAttackSquares : Nat := 2 * boardSize - 1

/-- The number of ways to place two rooks on a chessboard without attacking each other --/
def twoRooksPlacement : Nat := totalSquares * (totalSquares - rookAttackSquares)

theorem two_rooks_non_attacking_placements :
  twoRooksPlacement = 3136 := by
  sorry

end two_rooks_non_attacking_placements_l2996_299680


namespace nicole_cookies_l2996_299673

theorem nicole_cookies (N : ℚ) : 
  (((1 - N) * (1 - 3/5)) = 6/25) → N = 2/5 := by
  sorry

end nicole_cookies_l2996_299673


namespace min_n_for_S_greater_than_1020_l2996_299640

def S (n : ℕ) : ℕ := 2 * (2^n - 1) - n

theorem min_n_for_S_greater_than_1020 :
  ∀ k : ℕ, k < 10 → S k ≤ 1020 ∧ S 10 > 1020 := by sorry

end min_n_for_S_greater_than_1020_l2996_299640


namespace division_problem_l2996_299685

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 5 →
  divisor = 3 * remainder + 3 →
  dividend = 113 →
  dividend = divisor * quotient + remainder →
  (divisor : ℚ) / quotient = 3 / 1 := by sorry

end division_problem_l2996_299685


namespace number_of_boys_number_of_boys_is_17_l2996_299610

theorem number_of_boys (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neither_children : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) 
  (neither_boys : ℕ) : ℕ :=
  by
  have h1 : total_children = 60 := by sorry
  have h2 : happy_children = 30 := by sorry
  have h3 : sad_children = 10 := by sorry
  have h4 : neither_children = 20 := by sorry
  have h5 : girls = 43 := by sorry
  have h6 : happy_boys = 6 := by sorry
  have h7 : sad_girls = 4 := by sorry
  have h8 : neither_boys = 5 := by sorry
  
  exact total_children - girls

theorem number_of_boys_is_17 : number_of_boys 60 30 10 20 43 6 4 5 = 17 := by sorry

end number_of_boys_number_of_boys_is_17_l2996_299610


namespace octagon_diagonals_l2996_299642

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by sorry

end octagon_diagonals_l2996_299642


namespace square_perimeter_l2996_299635

/-- Given a square cut into four equal rectangles that form a shape with perimeter 56,
    prove that the original square's perimeter is 32. -/
theorem square_perimeter (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  2 * (14 * width) = 56 →
  4 * (4 * width) = 32 :=
by
  sorry

#check square_perimeter

end square_perimeter_l2996_299635


namespace james_purchase_cost_l2996_299695

/-- The total cost of James' purchase of shirts and pants -/
def total_cost (num_shirts : ℕ) (shirt_price : ℕ) (pant_price : ℕ) : ℕ :=
  let num_pants := num_shirts / 2
  num_shirts * shirt_price + num_pants * pant_price

/-- Theorem stating that James' purchase costs $100 -/
theorem james_purchase_cost : total_cost 10 6 8 = 100 := by
  sorry

end james_purchase_cost_l2996_299695


namespace max_regions_correct_max_regions_recurrence_max_regions_is_maximum_l2996_299648

/-- The maximum number of regions a plane can be divided into by n rectangles with parallel sides -/
def max_regions (n : ℕ) : ℕ := 2*n^2 - 2*n + 2

/-- Theorem stating that max_regions gives the correct number of regions for n rectangles -/
theorem max_regions_correct (n : ℕ) : 
  max_regions n = 2*n^2 - 2*n + 2 := by sorry

/-- Theorem stating that max_regions satisfies the recurrence relation -/
theorem max_regions_recurrence (n : ℕ) : 
  max_regions (n + 1) = max_regions n + 4*n := by sorry

/-- Theorem stating that max_regions gives the maximum possible number of regions -/
theorem max_regions_is_maximum (n : ℕ) (k : ℕ) :
  k ≤ max_regions n := by sorry

end max_regions_correct_max_regions_recurrence_max_regions_is_maximum_l2996_299648


namespace arrangements_count_l2996_299645

/-- The number of different arrangements for 6 students where two specific students cannot stand together -/
def number_of_arrangements : ℕ := 480

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of students that can be arranged freely -/
def free_students : ℕ := 4

/-- The number of gaps after arranging the free students -/
def number_of_gaps : ℕ := 5

/-- The number of students that cannot stand together -/
def restricted_students : ℕ := 2

theorem arrangements_count :
  number_of_arrangements = 
    (Nat.factorial free_students) * (number_of_gaps * (number_of_gaps - 1)) :=
by sorry

end arrangements_count_l2996_299645


namespace intersection_M_complement_N_l2996_299643

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | 2*x < 2}

theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = Set.Icc 1 3 := by sorry

end intersection_M_complement_N_l2996_299643


namespace last_four_digits_of_5_to_9000_l2996_299658

theorem last_four_digits_of_5_to_9000 (h : 5^300 ≡ 1 [ZMOD 1250]) :
  5^9000 ≡ 1 [ZMOD 1250] := by
  sorry

end last_four_digits_of_5_to_9000_l2996_299658


namespace reasoning_classification_l2996_299639

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive

-- Define the reasoning methods
def method1 : String := "Inferring the properties of a ball from the properties of a circle"
def method2 : String := "Inducing that the sum of the internal angles of all triangles is 180° from the sum of the internal angles of right triangles, isosceles triangles, and equilateral triangles"
def method3 : String := "Deducing that f(x) = sinx is an odd function from f(-x) = -f(x), x ∈ R"
def method4 : String := "Inducing that the sum of the internal angles of a convex polygon is (n-2)•180° from the sum of the internal angles of a triangle, quadrilateral, and pentagon"

-- Define a function to classify reasoning methods
def classifyReasoning (method : String) : ReasoningType := sorry

-- Theorem to prove
theorem reasoning_classification :
  (classifyReasoning method1 = ReasoningType.Analogical) ∧
  (classifyReasoning method2 = ReasoningType.Inductive) ∧
  (classifyReasoning method3 = ReasoningType.Deductive) ∧
  (classifyReasoning method4 = ReasoningType.Inductive) := by
  sorry

end reasoning_classification_l2996_299639


namespace sphere_volume_to_surface_area_l2996_299686

theorem sphere_volume_to_surface_area :
  ∀ (r : ℝ), 
    (4 / 3 : ℝ) * π * r^3 = 4 * Real.sqrt 3 * π → 
    4 * π * r^2 = 12 * π := by
  sorry

end sphere_volume_to_surface_area_l2996_299686


namespace line_parallel_to_plane_relationship_l2996_299687

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (skew_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_relationship
  (a b : Line) (α : Plane)
  (h1 : parallel_line_plane a α)
  (h2 : contained_in_plane b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end line_parallel_to_plane_relationship_l2996_299687


namespace largest_positive_solution_l2996_299608

theorem largest_positive_solution : 
  ∃ (x : ℝ), x > 0 ∧ 
    (2 * x^3 - x^2 - x + 1)^(1 + 1/(2*x + 1)) = 1 ∧ 
    ∀ (y : ℝ), y > 0 → 
      (2 * y^3 - y^2 - y + 1)^(1 + 1/(2*y + 1)) = 1 → 
      y ≤ x :=
by
  -- The proof goes here
  sorry

end largest_positive_solution_l2996_299608


namespace four_blocks_in_six_by_six_grid_l2996_299615

theorem four_blocks_in_six_by_six_grid : 
  let n : ℕ := 6
  let k : ℕ := 4
  let grid_size := n * n
  let combinations := (n.choose k) * (n.choose k) * (k.factorial)
  combinations = 5400 := by
  sorry

end four_blocks_in_six_by_six_grid_l2996_299615


namespace trapezium_other_side_length_l2996_299636

/-- Theorem: In a trapezium with one parallel side of 18 cm, a distance between parallel sides of 10 cm,
    and an area of 190 square centimeters, the length of the other parallel side is 20 cm. -/
theorem trapezium_other_side_length (a b h : ℝ) (h1 : a = 18) (h2 : h = 10) (h3 : (a + b) * h / 2 = 190) :
  b = 20 := by
  sorry

end trapezium_other_side_length_l2996_299636


namespace mechanic_bill_calculation_l2996_299601

/-- Given a mechanic's hourly rate, parts cost, and hours worked, calculate the total bill -/
def total_bill (hourly_rate : ℕ) (parts_cost : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_rate * hours_worked + parts_cost

/-- Theorem: The total bill for a 5-hour job with $45/hour rate and $225 parts cost is $450 -/
theorem mechanic_bill_calculation :
  total_bill 45 225 5 = 450 := by
  sorry

end mechanic_bill_calculation_l2996_299601


namespace total_cost_is_2495_l2996_299688

/-- Represents the quantity of each fruit in kilograms -/
def apple_qty : ℕ := 8
def mango_qty : ℕ := 9
def banana_qty : ℕ := 6
def grape_qty : ℕ := 4
def cherry_qty : ℕ := 3

/-- Represents the rate of each fruit per kilogram -/
def apple_rate : ℕ := 70
def mango_rate : ℕ := 75
def banana_rate : ℕ := 40
def grape_rate : ℕ := 120
def cherry_rate : ℕ := 180

/-- Calculates the total cost of all fruits -/
def total_cost : ℕ := 
  apple_qty * apple_rate + 
  mango_qty * mango_rate + 
  banana_qty * banana_rate + 
  grape_qty * grape_rate + 
  cherry_qty * cherry_rate

/-- Theorem stating that the total cost of all fruits is 2495 -/
theorem total_cost_is_2495 : total_cost = 2495 := by
  sorry

end total_cost_is_2495_l2996_299688


namespace fibonacci_6_l2996_299609

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_6 : fibonacci 5 = 8 := by
  sorry

end fibonacci_6_l2996_299609


namespace gcd_of_squares_l2996_299676

theorem gcd_of_squares : Nat.gcd (101^2 + 203^2 + 307^2) (100^2 + 202^2 + 308^2) = 1 := by
  sorry

end gcd_of_squares_l2996_299676


namespace smallest_divisible_n_l2996_299663

theorem smallest_divisible_n : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m^3 % 450 = 0 ∧ m^4 % 2560 = 0 → n ≤ m) ∧
  n^3 % 450 = 0 ∧ n^4 % 2560 = 0 ∧ n = 60 := by
  sorry

end smallest_divisible_n_l2996_299663


namespace carl_payment_percentage_l2996_299694

theorem carl_payment_percentage (property_damage medical_bills insurance_percentage carl_owes : ℚ)
  (h1 : property_damage = 40000)
  (h2 : medical_bills = 70000)
  (h3 : insurance_percentage = 80/100)
  (h4 : carl_owes = 22000) :
  carl_owes / (property_damage + medical_bills) = 20/100 := by
  sorry

end carl_payment_percentage_l2996_299694


namespace tom_reading_pages_l2996_299618

def pages_read (initial_speed : ℕ) (time : ℕ) (speed_factor : ℕ) : ℕ :=
  initial_speed * speed_factor * time

theorem tom_reading_pages : pages_read 12 2 3 = 72 := by
  sorry

end tom_reading_pages_l2996_299618


namespace equation_root_approximation_l2996_299604

/-- The equation whose root we need to find -/
def equation (x : ℝ) : Prop :=
  (Real.sqrt 5 - Real.sqrt 2) * (1 + x) = (Real.sqrt 6 - Real.sqrt 3) * (1 - x)

/-- The approximate root of the equation -/
def approximate_root : ℝ := -0.068

/-- Theorem stating that the approximate root satisfies the equation within a small error -/
theorem equation_root_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((Real.sqrt 5 - Real.sqrt 2) * (1 + approximate_root) - 
    (Real.sqrt 6 - Real.sqrt 3) * (1 - approximate_root))| < ε :=
sorry

end equation_root_approximation_l2996_299604


namespace hcf_lcm_sum_reciprocal_l2996_299605

theorem hcf_lcm_sum_reciprocal (m n : ℕ+) : 
  Nat.gcd m.val n.val = 6 → 
  Nat.lcm m.val n.val = 210 → 
  m.val + n.val = 60 → 
  (1 : ℚ) / m.val + (1 : ℚ) / n.val = 1 / 21 := by
sorry

end hcf_lcm_sum_reciprocal_l2996_299605


namespace perpendicular_vectors_x_value_l2996_299653

theorem perpendicular_vectors_x_value (x y : ℝ) :
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (3, 2 - x)
  (a.1 * b.1 + a.2 * b.2 = 0) → (x = 3 ∨ x = -1) := by
  sorry

end perpendicular_vectors_x_value_l2996_299653


namespace inequality_solution_set_l2996_299613

theorem inequality_solution_set : 
  {x : ℝ | (x - 2) * (2 * x + 1) > 0} = 
  {x : ℝ | x < -1/2 ∨ x > 2} := by sorry

end inequality_solution_set_l2996_299613


namespace factors_of_N_l2996_299630

/-- The number of natural-number factors of N, where N = 2^4 * 3^3 * 5^2 * 7^2 -/
def num_factors (N : Nat) : Nat :=
  if N = 2^4 * 3^3 * 5^2 * 7^2 then 180 else 0

/-- Theorem stating that the number of natural-number factors of N is 180 -/
theorem factors_of_N :
  ∃ N : Nat, N = 2^4 * 3^3 * 5^2 * 7^2 ∧ num_factors N = 180 :=
by
  sorry

#check factors_of_N

end factors_of_N_l2996_299630


namespace prime_divisibility_l2996_299641

theorem prime_divisibility (p q r : ℕ) : 
  Prime p → Prime q → Prime r → Odd p → (p ∣ q^r + 1) → 
  (2*r ∣ p - 1) ∨ (p ∣ q^2 - 1) := by
  sorry

end prime_divisibility_l2996_299641


namespace unique_nonnegative_solution_l2996_299607

theorem unique_nonnegative_solution (x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  x + y + z = 3 * x * y →
  x^2 + y^2 + z^2 = 3 * x * z →
  x^3 + y^3 + z^3 = 3 * y * z →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end unique_nonnegative_solution_l2996_299607


namespace continuity_at_two_l2996_299674

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / (x^2 - 4)

theorem continuity_at_two :
  ∀ (c : ℝ), ContinuousAt f 2 ↔ c = 7/4 := by sorry

end continuity_at_two_l2996_299674


namespace circle_equation_tangent_lines_l2996_299681

-- Define the circle C
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the point M
def M : ℝ × ℝ := (0, 2)

-- Define the point P
def P : ℝ × ℝ := (3, 2)

-- Theorem for the equation of circle C
theorem circle_equation : 
  ∃ (r : ℝ), M ∈ Circle r ∧ Circle r = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4} :=
sorry

-- Define a tangent line
def TangentLine (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | k * p.1 - p.2 - 3 * k + 2 = 0}

-- Theorem for the equations of tangent lines
theorem tangent_lines :
  ∃ (k₁ k₂ : ℝ), 
    (TangentLine k₁ = {p : ℝ × ℝ | p.2 = 2}) ∧
    (TangentLine k₂ = {p : ℝ × ℝ | 12 * p.1 - 5 * p.2 - 26 = 0}) ∧
    P ∈ TangentLine k₁ ∧ P ∈ TangentLine k₂ ∧
    (∀ (p : ℝ × ℝ), p ∈ TangentLine k₁ ∩ Circle 2 → p = P) ∧
    (∀ (p : ℝ × ℝ), p ∈ TangentLine k₂ ∩ Circle 2 → p = P) :=
sorry

end circle_equation_tangent_lines_l2996_299681


namespace largest_reciprocal_l2996_299646

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/7 → b = 3/4 → c = 2 → d = 8 → e = 100 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end largest_reciprocal_l2996_299646


namespace f_properties_l2996_299698

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - (m / 2) * x^2 + x

theorem f_properties (m : ℝ) :
  (m > 0 ∧ (∀ x > 0, f m x ≤ m * x - 1/2) → m ≥ 1) ∧
  (m = -1 → ∀ x₁ > 0, ∀ x₂ > 0, f m x₁ + f m x₂ = 0 → x₁ + x₂ ≥ Real.sqrt 3 - 1) :=
by sorry

end f_properties_l2996_299698


namespace distance_center_to_plane_l2996_299682

/-- Given a sphere and three points on its surface, calculate the distance from the center to the plane of the triangle formed by the points. -/
theorem distance_center_to_plane (S : Real) (AB BC AC : Real) (h1 : S = 20 * Real.pi) (h2 : BC = 2 * Real.sqrt 3) (h3 : AB = 2) (h4 : AC = 2) : 
  ∃ d : Real, d = 1 ∧ d = Real.sqrt (((S / (4 * Real.pi))^(1/2 : Real))^2 - (BC / (2 * Real.sin (Real.arccos ((AC^2 + AB^2 - BC^2) / (2 * AC * AB)))))^2) :=
by sorry

end distance_center_to_plane_l2996_299682


namespace inner_polygon_perimeter_less_than_outer_l2996_299656

-- Define a type for convex polygons
structure ConvexPolygon where
  -- Add necessary fields (this is a simplified representation)
  perimeter : ℝ

-- Define a relation for one polygon being inside another
def IsInside (inner outer : ConvexPolygon) : Prop :=
  -- Add necessary conditions for one polygon being inside another
  sorry

-- Theorem statement
theorem inner_polygon_perimeter_less_than_outer
  (inner outer : ConvexPolygon)
  (h : IsInside inner outer) :
  inner.perimeter < outer.perimeter :=
sorry

end inner_polygon_perimeter_less_than_outer_l2996_299656


namespace expression_value_l2996_299693

theorem expression_value (a b c d m : ℝ)
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : m = -1)     -- m equals -1
  : 2 * a * b - (c + d) + m^2 = 3 := by
  sorry

end expression_value_l2996_299693


namespace afternoon_rowers_l2996_299684

theorem afternoon_rowers (total : ℕ) (morning : ℕ) (h1 : total = 60) (h2 : morning = 53) :
  total - morning = 7 := by
  sorry

end afternoon_rowers_l2996_299684


namespace find_number_l2996_299633

theorem find_number : ∃ x : ℚ, (x + 32/113) * 113 = 9637 ∧ x = 85 := by sorry

end find_number_l2996_299633


namespace smallest_positive_integer_with_remainders_l2996_299603

theorem smallest_positive_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  ∀ m : ℕ, m > 0 ∧ 
    m % 2 = 1 ∧
    m % 3 = 2 ∧
    m % 4 = 3 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 ∧
    m % 10 = 9 → m ≥ n :=
by
  sorry

end smallest_positive_integer_with_remainders_l2996_299603


namespace two_train_problem_l2996_299679

/-- Prove that given the conditions of the two-train problem, the speed of the second train is 40 km/hr -/
theorem two_train_problem (v : ℝ) : 
  (∀ t : ℝ, 50 * t = v * t + 100) →  -- First train travels 100 km more
  (∀ t : ℝ, 50 * t + v * t = 900) →  -- Total distance is 900 km
  v = 40 := by
  sorry

end two_train_problem_l2996_299679


namespace largest_prime_divisor_factorial_sum_l2996_299632

theorem largest_prime_divisor_factorial_sum : 
  ∃ p : Nat, 
    Nat.Prime p ∧ 
    p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧
    ∀ q : Nat, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by
  -- The proof would go here
  sorry

end largest_prime_divisor_factorial_sum_l2996_299632


namespace regular_tile_area_l2996_299664

/-- Represents the properties of tiles used to cover a wall -/
structure TileInfo where
  regularLength : ℝ
  regularWidth : ℝ
  jumboLength : ℝ
  jumboWidth : ℝ
  totalTiles : ℝ
  regularTiles : ℝ
  jumboTiles : ℝ

/-- Theorem stating the area covered by regular tiles on a wall -/
theorem regular_tile_area (t : TileInfo) (h1 : t.jumboLength = 3 * t.regularLength)
    (h2 : t.jumboWidth = t.regularWidth)
    (h3 : t.jumboTiles = (1/3) * t.totalTiles)
    (h4 : t.regularTiles = (2/3) * t.totalTiles)
    (h5 : t.regularLength * t.regularWidth * t.regularTiles +
          t.jumboLength * t.jumboWidth * t.jumboTiles = 385) :
    t.regularLength * t.regularWidth * t.regularTiles = 154 := by
  sorry

end regular_tile_area_l2996_299664


namespace container_capacity_container_capacity_proof_l2996_299670

theorem container_capacity : ℝ → Prop :=
  fun C =>
    (C > 0) ∧                   -- Capacity is positive
    (1/2 * C + 20 = 3/4 * C) →  -- Adding 20 liters to half-full makes it 3/4 full
    C = 80                      -- The capacity is 80 liters

-- Proof
theorem container_capacity_proof : ∃ C, container_capacity C :=
  sorry

end container_capacity_container_capacity_proof_l2996_299670


namespace cone_lateral_area_l2996_299619

/-- The lateral area of a cone with base radius 3 cm and height 4 cm is 15π cm². -/
theorem cone_lateral_area :
  let r : ℝ := 3  -- radius in cm
  let h : ℝ := 4  -- height in cm
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height
  let lateral_area : ℝ := π * r * l  -- lateral area formula
  lateral_area = 15 * π := by sorry

end cone_lateral_area_l2996_299619


namespace parallel_lines_a_value_l2996_299650

/-- Given two lines L1 and L2, returns true if they are parallel but not coincident -/
def are_parallel_not_coincident (L1 L2 : ℝ → ℝ → Prop) : Prop :=
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, L1 x y ↔ L2 (k * x) (k * y)) ∧
  ¬(∀ x y, L1 x y ↔ L2 x y)

/-- The first line: ax + 2y + 6 = 0 -/
def L1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0

/-- The second line: x + (a - 1)y + (a^2 - 1) = 0 -/
def L2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0

theorem parallel_lines_a_value :
  ∃ a : ℝ, are_parallel_not_coincident (L1 a) (L2 a) ∧ a = -1 :=
by sorry

end parallel_lines_a_value_l2996_299650


namespace odd_even_sum_difference_l2996_299654

def sum_odd (n : ℕ) : ℕ := n^2

def sum_even (n : ℕ) : ℕ := n * (n + 1)

def odd_terms (max : ℕ) : ℕ := (max - 1) / 2 + 1

def even_terms (max : ℕ) : ℕ := (max - 2) / 2 + 1

theorem odd_even_sum_difference :
  sum_odd (odd_terms 2023) - sum_even (even_terms 2020) = 3034 := by
  sorry

end odd_even_sum_difference_l2996_299654


namespace cosine_period_problem_l2996_299657

theorem cosine_period_problem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * x + c) + d) →
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * (x + 2 * π) + c) + d) →
  b = 2 := by
sorry

end cosine_period_problem_l2996_299657


namespace right_triangle_circumradius_l2996_299623

theorem right_triangle_circumradius (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  a^2 + b^2 = c^2 → (c / 2 : ℝ) = 7.5 := by
  sorry

end right_triangle_circumradius_l2996_299623


namespace digitSquareSequenceReaches1Or4_l2996_299624

/-- Sum of squares of digits of a natural number -/
def sumOfSquaresOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence of sum of squares of digits -/
def digitSquareSequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => sumOfSquaresOfDigits (digitSquareSequence start n)

/-- Predicate to check if a number is three digits -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem digitSquareSequenceReaches1Or4 (start : ℕ) (h : isThreeDigit start) :
  ∃ (k : ℕ), digitSquareSequence start k = 1 ∨ digitSquareSequence start k = 4 := by sorry

end digitSquareSequenceReaches1Or4_l2996_299624


namespace f_min_at_three_l2996_299675

/-- The quadratic function to be minimized -/
def f (c : ℝ) : ℝ := 3 * c^2 - 18 * c + 20

/-- Theorem stating that f is minimized at c = 3 -/
theorem f_min_at_three : 
  ∀ x : ℝ, f 3 ≤ f x := by sorry

end f_min_at_three_l2996_299675


namespace martha_apples_l2996_299620

theorem martha_apples (tim harry martha : ℕ) 
  (h1 : martha = tim + 30)
  (h2 : harry = tim / 2)
  (h3 : harry = 19) : 
  martha = 68 := by sorry

end martha_apples_l2996_299620


namespace complex_exponential_sum_l2996_299659

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I := by
  sorry

end complex_exponential_sum_l2996_299659


namespace quadratic_equations_solutions_l2996_299627

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 2 * (x - 1)^2 = 18 ↔ x = 4 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by sorry

end quadratic_equations_solutions_l2996_299627


namespace quadratic_through_points_l2996_299667

/-- A quadratic function that passes through (-1, 2) and (1, y) must have y = 2 -/
theorem quadratic_through_points (a : ℝ) (y : ℝ) : 
  a ≠ 0 → (2 = a * (-1)^2) → (y = a * 1^2) → y = 2 := by
  sorry

end quadratic_through_points_l2996_299667


namespace sum_of_fourth_powers_l2996_299690

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 3)
  (sum_cubes : a^3 + b^3 + c^3 = 4) :
  a^4 + b^4 + c^4 = 6.833 := by sorry

end sum_of_fourth_powers_l2996_299690


namespace expression_equals_negative_one_l2996_299602

theorem expression_equals_negative_one (x y : ℝ) 
  (hx : x ≠ 0) (hxy : x ≠ 2*y ∧ x ≠ -2*y) : 
  (x / (x + 2*y) + 2*y / (x - 2*y)) / (2*y / (x + 2*y) - x / (x - 2*y)) = -1 :=
by sorry

end expression_equals_negative_one_l2996_299602


namespace folded_rectangle_length_l2996_299655

/-- Given a rectangular strip of paper with dimensions 4 × 13, folded to form two rectangles
    with areas P and Q such that P = 2Q, prove that the length of one of the resulting rectangles is 6. -/
theorem folded_rectangle_length (x y : ℝ) (P Q : ℝ) : 
  x + y = 9 →  -- Sum of lengths of the two rectangles
  x + 4 + y = 13 →  -- Total length of the original rectangle
  P = 4 * x →  -- Area of rectangle P
  Q = 4 * y →  -- Area of rectangle Q
  P = 2 * Q →  -- Relationship between areas P and Q
  x = 6 := by sorry

end folded_rectangle_length_l2996_299655


namespace jakesDrinkVolume_l2996_299669

/-- Represents the composition of a drink mixture -/
structure DrinkMixture where
  coke : ℕ
  sprite : ℕ
  mountainDew : ℕ

/-- Calculates the total parts in a drink mixture -/
def totalParts (d : DrinkMixture) : ℕ := d.coke + d.sprite + d.mountainDew

/-- Represents Jake's drink mixture -/
def jakesDrink : DrinkMixture := { coke := 2, sprite := 1, mountainDew := 3 }

/-- The volume of Coke in Jake's drink in ounces -/
def cokeVolume : ℕ := 6

/-- Theorem: Jake's drink has a total volume of 18 ounces -/
theorem jakesDrinkVolume : 
  (cokeVolume * totalParts jakesDrink) / jakesDrink.coke = 18 := by
  sorry

end jakesDrinkVolume_l2996_299669


namespace average_rate_of_change_f_on_0_2_l2996_299660

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the interval [0, 2]
def a : ℝ := 0
def b : ℝ := 2

-- State the theorem
theorem average_rate_of_change_f_on_0_2 :
  (f b - f a) / (b - a) = 2 := by
  sorry

end average_rate_of_change_f_on_0_2_l2996_299660


namespace john_replacement_cost_l2996_299678

/-- Represents the genre of a movie --/
inductive Genre
  | Action
  | Comedy
  | Drama

/-- Represents the popularity of a movie --/
inductive Popularity
  | Popular
  | ModeratelyPopular
  | Unpopular

/-- Represents a movie with its genre and popularity --/
structure Movie where
  genre : Genre
  popularity : Popularity

/-- The trade-in value for a VHS movie based on its genre --/
def tradeInValue (g : Genre) : ℕ :=
  match g with
  | Genre.Action => 3
  | Genre.Comedy => 2
  | Genre.Drama => 1

/-- The purchase price for a DVD based on its popularity --/
def purchasePrice (p : Popularity) : ℕ :=
  match p with
  | Popularity.Popular => 12
  | Popularity.ModeratelyPopular => 8
  | Popularity.Unpopular => 5

/-- The collection of movies John has --/
def johnMovies : List Movie :=
  (List.replicate 20 ⟨Genre.Action, Popularity.Popular⟩) ++
  (List.replicate 30 ⟨Genre.Comedy, Popularity.ModeratelyPopular⟩) ++
  (List.replicate 10 ⟨Genre.Drama, Popularity.Unpopular⟩) ++
  (List.replicate 15 ⟨Genre.Comedy, Popularity.Popular⟩) ++
  (List.replicate 25 ⟨Genre.Action, Popularity.ModeratelyPopular⟩)

/-- The total cost to replace all movies --/
def replacementCost (movies : List Movie) : ℕ :=
  (movies.map (fun m => purchasePrice m.popularity)).sum -
  (movies.map (fun m => tradeInValue m.genre)).sum

/-- Theorem stating the cost to replace all of John's movies --/
theorem john_replacement_cost :
  replacementCost johnMovies = 675 := by
  sorry

end john_replacement_cost_l2996_299678


namespace office_network_connections_l2996_299606

/-- The number of connections in a network of switches where each switch is connected to a fixed number of other switches. -/
def network_connections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 30 switches, where each switch is directly connected to exactly 4 other switches, the total number of connections is 60. -/
theorem office_network_connections :
  network_connections 30 4 = 60 := by
  sorry

end office_network_connections_l2996_299606


namespace vintik_votes_l2996_299621

theorem vintik_votes (total_percentage : ℝ) (shpuntik_votes : ℕ) 
  (h1 : total_percentage = 146)
  (h2 : shpuntik_votes > 1000) :
  ∃ (vintik_votes : ℕ), vintik_votes > 850 := by
  sorry

end vintik_votes_l2996_299621


namespace equation_roots_range_l2996_299600

theorem equation_roots_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
   |x₁ - 2*n| = k * Real.sqrt x₁ ∧
   |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end equation_roots_range_l2996_299600


namespace line_ellipse_intersection_l2996_299647

/-- The line equation -/
def line (a x y : ℝ) : Prop := (a + 1) * x + (3 * a - 1) * y - (6 * a + 2) = 0

/-- The ellipse equation -/
def ellipse (x y m : ℝ) : Prop := x^2 / 16 + y^2 / m = 1

/-- The theorem stating the conditions for the line and ellipse to always have a common point -/
theorem line_ellipse_intersection (a m : ℝ) :
  (∀ x y : ℝ, line a x y → ellipse x y m → False) ↔ 
  (m ∈ Set.Icc (16/7) 16 ∪ Set.Ioi 16) :=
sorry

end line_ellipse_intersection_l2996_299647


namespace complex_power_sum_l2996_299625

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^3000 + 1/z^3000 = 1 := by sorry

end complex_power_sum_l2996_299625


namespace largest_decimal_l2996_299628

theorem largest_decimal : 
  let a := 0.9877
  let b := 0.9789
  let c := 0.9700
  let d := 0.9790
  let e := 0.9709
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) :=
by sorry

end largest_decimal_l2996_299628


namespace employee_average_salary_l2996_299668

theorem employee_average_salary 
  (num_employees : ℕ) 
  (manager_salary : ℕ) 
  (average_increase : ℕ) 
  (h1 : num_employees = 18)
  (h2 : manager_salary = 5800)
  (h3 : average_increase = 200) :
  let total_with_manager := (num_employees + 1) * (average_employee_salary + average_increase)
  let total_without_manager := num_employees * average_employee_salary + manager_salary
  total_with_manager = total_without_manager →
  average_employee_salary = 2000 :=
by
  sorry

end employee_average_salary_l2996_299668


namespace smallest_cube_ending_144_l2996_299631

theorem smallest_cube_ending_144 : ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 144 ∧ ∀ (m : ℕ), m > 0 → m^3 % 1000 = 144 → n ≤ m :=
by sorry

end smallest_cube_ending_144_l2996_299631


namespace watch_cost_price_l2996_299651

/-- Proves that the cost price of a watch is 1500 Rs. given the conditions of the problem -/
theorem watch_cost_price (loss_percentage : ℚ) (gain_percentage : ℚ) (price_difference : ℚ) :
  loss_percentage = 10 / 100 →
  gain_percentage = 5 / 100 →
  price_difference = 225 →
  ∃ (cost_price : ℚ), 
    (1 - loss_percentage) * cost_price + price_difference = (1 + gain_percentage) * cost_price ∧
    cost_price = 1500 := by
  sorry

end watch_cost_price_l2996_299651


namespace budget_increase_is_twenty_percent_l2996_299612

/-- The percentage increase in the gym budget -/
def budget_increase_percentage (original_dodgeball_count : ℕ) (dodgeball_price : ℚ)
  (new_softball_count : ℕ) (softball_price : ℚ) : ℚ :=
  let original_budget := original_dodgeball_count * dodgeball_price
  let new_budget := new_softball_count * softball_price
  ((new_budget - original_budget) / original_budget) * 100

/-- Theorem stating that the budget increase percentage is 20% -/
theorem budget_increase_is_twenty_percent :
  budget_increase_percentage 15 5 10 9 = 20 := by
  sorry

end budget_increase_is_twenty_percent_l2996_299612


namespace root_cube_sum_condition_l2996_299696

theorem root_cube_sum_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁^3 - 6*x₁^2 + a*x₁ + a = 0) ∧ 
    (x₂^3 - 6*x₂^2 + a*x₂ + a = 0) ∧ 
    (x₃^3 - 6*x₃^2 + a*x₃ + a = 0) ∧ 
    ((x₁-3)^3 + (x₂-3)^3 + (x₃-3)^3 = 0)) ↔ 
  (a = -9) :=
sorry

end root_cube_sum_condition_l2996_299696


namespace greatest_bound_of_r2_l2996_299622

/-- The function f(x) = x^2 - r_2x + r_3 -/
def f (r_2 r_3 : ℝ) (x : ℝ) : ℝ := x^2 - r_2*x + r_3

/-- The sequence g_n defined recursively -/
def g (r_2 r_3 : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r_2 r_3 (g r_2 r_3 n)

/-- The property that g_{2i} < g_{2i+1} and g_{2i+1} > g_{2i+2} for 0 ≤ i ≤ 2011 -/
def alternating_property (r_2 r_3 : ℝ) : Prop :=
  ∀ i : ℕ, i ≤ 2011 → g r_2 r_3 (2*i) < g r_2 r_3 (2*i + 1) ∧ g r_2 r_3 (2*i + 1) > g r_2 r_3 (2*i + 2)

/-- The property that there exists j such that g_{i+1} > g_i for all i > j -/
def eventually_increasing (r_2 r_3 : ℝ) : Prop :=
  ∃ j : ℕ, ∀ i : ℕ, i > j → g r_2 r_3 (i + 1) > g r_2 r_3 i

/-- The property that the sequence g_n is unbounded -/
def unbounded_sequence (r_2 r_3 : ℝ) : Prop :=
  ∀ M : ℝ, ∃ N : ℕ, g r_2 r_3 N > M

/-- The main theorem -/
theorem greatest_bound_of_r2 :
  (∃ A : ℝ, ∀ r_2 r_3 : ℝ, 
    alternating_property r_2 r_3 → 
    eventually_increasing r_2 r_3 → 
    unbounded_sequence r_2 r_3 → 
    A ≤ |r_2| ∧ 
    (∀ B : ℝ, (∀ r_2' r_3' : ℝ, 
      alternating_property r_2' r_3' → 
      eventually_increasing r_2' r_3' → 
      unbounded_sequence r_2' r_3' → 
      B ≤ |r_2'|) → B ≤ A)) ∧
  (∀ A : ℝ, (∀ r_2 r_3 : ℝ, 
    alternating_property r_2 r_3 → 
    eventually_increasing r_2 r_3 → 
    unbounded_sequence r_2 r_3 → 
    A ≤ |r_2| ∧ 
    (∀ B : ℝ, (∀ r_2' r_3' : ℝ, 
      alternating_property r_2' r_3' → 
      eventually_increasing r_2' r_3' → 
      unbounded_sequence r_2' r_3' → 
      B ≤ |r_2'|) → B ≤ A)) → A = 2) := by
  sorry

end greatest_bound_of_r2_l2996_299622


namespace opposite_of_negative_2023_l2996_299617

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end opposite_of_negative_2023_l2996_299617


namespace perfect_square_condition_l2996_299683

theorem perfect_square_condition (n : ℤ) : 
  (∃ k : ℤ, n^2 + 6*n + 1 = k^2) ↔ (n = -6 ∨ n = 0) := by
  sorry

end perfect_square_condition_l2996_299683


namespace tunnel_length_l2996_299661

/-- Calculates the length of a tunnel given train and travel information -/
theorem tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) :
  train_length = 2 →
  exit_time = 4 →
  train_speed = 90 →
  (train_speed / 60) * exit_time - train_length = 4 := by
  sorry

#check tunnel_length

end tunnel_length_l2996_299661


namespace lewis_earnings_l2996_299671

/-- Calculates the weekly earnings of a person given the number of weeks worked,
    weekly rent, and final savings. -/
def weekly_earnings (weeks : ℕ) (rent : ℚ) (final_savings : ℚ) : ℚ :=
  (final_savings + weeks * rent) / weeks

theorem lewis_earnings :
  let weeks : ℕ := 1181
  let rent : ℚ := 216
  let final_savings : ℚ := 324775
  weekly_earnings weeks rent final_savings = 490.75 := by sorry

end lewis_earnings_l2996_299671


namespace not_sufficient_not_necessary_l2996_299611

open Set

/-- The set of real numbers x where x³ - 2x > 0 -/
def S : Set ℝ := {x | x^3 - 2*x > 0}

/-- The set of real numbers x where |x + 1| > 3 -/
def T : Set ℝ := {x | |x + 1| > 3}

theorem not_sufficient_not_necessary : ¬(S ⊆ T) ∧ ¬(T ⊆ S) := by sorry

end not_sufficient_not_necessary_l2996_299611


namespace system_solutions_l2996_299697

theorem system_solutions :
  ∀ x y : ℝ,
  (y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2) ∨ 
   (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2)) :=
by sorry

end system_solutions_l2996_299697


namespace five_balls_three_boxes_l2996_299692

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 24 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 24 := by
  sorry

end five_balls_three_boxes_l2996_299692


namespace express_y_in_terms_of_x_l2996_299677

/-- Given the equation 2x + y = 6, prove that y can be expressed as 6 - 2x. -/
theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 6) : y = 6 - 2 * x := by
  sorry

end express_y_in_terms_of_x_l2996_299677


namespace shopping_problem_l2996_299614

theorem shopping_problem (total : ℕ) (stores : ℕ) (initial_amount : ℕ) :
  total = stores ∧ 
  initial_amount = 100 ∧ 
  stores = 6 → 
  ∃ (spent_per_store : ℕ), 
    spent_per_store * stores ≤ initial_amount ∧ 
    spent_per_store > 0 ∧
    initial_amount - spent_per_store * stores ≤ 28 :=
by sorry

#check shopping_problem

end shopping_problem_l2996_299614


namespace instantaneous_velocity_at_3_seconds_l2996_299699

/-- The position function of a particle -/
def s (t : ℝ) : ℝ := 3 * t^2 + t

/-- The velocity function of a particle -/
def v (t : ℝ) : ℝ := 6 * t + 1

theorem instantaneous_velocity_at_3_seconds : v 3 = 19 := by
  sorry

end instantaneous_velocity_at_3_seconds_l2996_299699


namespace terrence_earnings_l2996_299629

def total_earnings : ℕ := 90
def emilee_earnings : ℕ := 25

theorem terrence_earnings (jermaine_earnings terrence_earnings : ℕ) 
  (h1 : jermaine_earnings = terrence_earnings + 5)
  (h2 : jermaine_earnings + terrence_earnings + emilee_earnings = total_earnings) :
  terrence_earnings = 30 := by
  sorry

end terrence_earnings_l2996_299629


namespace positive_terms_count_l2996_299691

/-- The number of positive terms in an arithmetic sequence with general term a_n = 90 - 2n -/
theorem positive_terms_count : ∃ k : ℕ, k = 44 ∧ 
  ∀ n : ℕ+, (90 : ℝ) - 2 * (n : ℝ) > 0 ↔ n ≤ k := by sorry

end positive_terms_count_l2996_299691


namespace shells_not_red_or_green_l2996_299626

theorem shells_not_red_or_green (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 291) (h2 : red = 76) (h3 : green = 49) :
  total - (red + green) = 166 := by
  sorry

end shells_not_red_or_green_l2996_299626


namespace correct_propositions_l2996_299672

theorem correct_propositions :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ∧
  (∀ a b : ℝ, ab ≠ 0 → a ≠ 0) :=
by sorry

end correct_propositions_l2996_299672


namespace school_commute_properties_l2996_299634

/-- Represents the distribution of students' commute times -/
structure CommuteDistribution where
  less_than_20 : Nat
  between_20_and_40 : Nat
  between_41_and_60 : Nat
  more_than_60 : Nat

/-- The given distribution of students' commute times -/
def school_distribution : CommuteDistribution :=
  { less_than_20 := 90
  , between_20_and_40 := 60
  , between_41_and_60 := 10
  , more_than_60 := 20 }

/-- Theorem stating the properties of the school's commute distribution -/
theorem school_commute_properties (d : CommuteDistribution) 
  (h : d = school_distribution) : 
  (d.less_than_20 = 90) ∧ 
  (d.less_than_20 + d.between_20_and_40 + d.between_41_and_60 + d.more_than_60 = 180) ∧ 
  (d.between_41_and_60 + d.more_than_60 = 30) ∧
  ¬(d.between_20_and_40 + d.between_41_and_60 + d.more_than_60 > d.less_than_20) := by
  sorry

#check school_commute_properties

end school_commute_properties_l2996_299634


namespace decimal_expansion_irrational_l2996_299662

/-- Decimal expansion function -/
def decimal_expansion (f : ℕ → ℕ) : ℚ :=
  sorry

/-- Power function -/
def f (n : ℕ) (x : ℕ) : ℕ :=
  x^n

/-- Theorem: The decimal expansion α is irrational for all positive integers n -/
theorem decimal_expansion_irrational (n : ℕ) (h : n > 0) :
  ¬ ∃ (q : ℚ), q = decimal_expansion (f n) :=
sorry

end decimal_expansion_irrational_l2996_299662


namespace rectangle_area_theorem_l2996_299644

theorem rectangle_area_theorem (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧
  l / w = 5 / 2 ∧
  d ^ 2 = (l / 2) ^ 2 + w ^ 2 ∧
  l * w = (5 / 13) * d ^ 2 :=
by sorry

end rectangle_area_theorem_l2996_299644


namespace exists_acute_triangle_with_large_intersection_area_l2996_299665

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The area of a triangle. -/
def area (t : Triangle) : ℝ := sorry

/-- A point is the median of a triangle if it is the midpoint of a side. -/
def is_median (M : Point) (t : Triangle) : Prop := sorry

/-- A point is on the angle bisector if it is equidistant from the two sides forming the angle. -/
def is_angle_bisector (K : Point) (t : Triangle) : Prop := sorry

/-- A point is on the altitude if it forms a right angle with the base of the triangle. -/
def is_altitude (H : Point) (t : Triangle) : Prop := sorry

/-- A triangle is acute if all its angles are less than 90 degrees. -/
def is_acute (t : Triangle) : Prop := sorry

/-- The area of the triangle formed by the intersection points of the median, angle bisector, and altitude. -/
def area_intersection (t : Triangle) (M K H : Point) : ℝ := sorry

/-- There exists an acute triangle where the area of the triangle formed by the intersection points
    of its median, angle bisector, and altitude is greater than 0.499 times the area of the original triangle. -/
theorem exists_acute_triangle_with_large_intersection_area :
  ∃ (t : Triangle) (M K H : Point),
    is_acute t ∧
    is_median M t ∧
    is_angle_bisector K t ∧
    is_altitude H t ∧
    area_intersection t M K H > 0.499 * area t :=
sorry

end exists_acute_triangle_with_large_intersection_area_l2996_299665


namespace quadrilateral_area_inequality_l2996_299637

/-- A quadrilateral with sides a, b, c, d and area S -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  S : ℝ

/-- Predicate for a cyclic quadrilateral with perpendicular diagonals -/
def is_cyclic_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_area_inequality (q : Quadrilateral) :
  q.S ≤ (q.a * q.c + q.b * q.d) / 2 ∧
  (q.S = (q.a * q.c + q.b * q.d) / 2 ↔ is_cyclic_perpendicular_diagonals q) := by
  sorry

end quadrilateral_area_inequality_l2996_299637
