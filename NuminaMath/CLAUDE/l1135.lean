import Mathlib

namespace commercial_fraction_l1135_113513

theorem commercial_fraction (num_programs : ℕ) (program_duration : ℕ) (commercial_time : ℕ) :
  num_programs = 6 →
  program_duration = 30 →
  commercial_time = 45 →
  (commercial_time : ℚ) / (num_programs * program_duration : ℚ) = 1 / 4 := by
  sorry

end commercial_fraction_l1135_113513


namespace solution_in_third_quadrant_implies_k_bound_l1135_113501

theorem solution_in_third_quadrant_implies_k_bound 
  (k : ℝ) 
  (h : ∃ x : ℝ, 
    π < x ∧ x < 3*π/2 ∧ 
    k * Real.cos x + Real.arccos (π/4) = 0) : 
  k > Real.arccos (π/4) := by
sorry

end solution_in_third_quadrant_implies_k_bound_l1135_113501


namespace student_number_choice_l1135_113572

theorem student_number_choice (x : ℝ) : 2 * x - 138 = 104 → x = 121 := by
  sorry

end student_number_choice_l1135_113572


namespace car_price_calculation_l1135_113520

/-- Calculates the price of a car given loan terms and payments -/
theorem car_price_calculation 
  (loan_years : ℕ) 
  (down_payment : ℚ) 
  (monthly_payment : ℚ) 
  (h_loan_years : loan_years = 5)
  (h_down_payment : down_payment = 5000)
  (h_monthly_payment : monthly_payment = 250) :
  down_payment + loan_years * 12 * monthly_payment = 20000 := by
  sorry

#check car_price_calculation

end car_price_calculation_l1135_113520


namespace max_value_of_d_l1135_113563

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_of_d_l1135_113563


namespace xyz_value_l1135_113569

variables (x y z : ℝ)

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
                   (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 6 := by
  sorry

end xyz_value_l1135_113569


namespace parallel_condition_l1135_113511

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define the parallel relation between two lines
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, l₁ a x y ↔ l₂ a x y

-- Theorem statement
theorem parallel_condition (a : ℝ) : parallel a ↔ a = 1 :=
sorry

end parallel_condition_l1135_113511


namespace computer_price_ratio_l1135_113500

theorem computer_price_ratio (c : ℝ) (h1 : c > 0) (h2 : c * 1.3 = 351) :
  (c + 351) / c = 2.3 := by
sorry

end computer_price_ratio_l1135_113500


namespace expression_equals_point_one_l1135_113576

-- Define the expression
def expression : ℝ := (0.000001 ^ (1/2)) ^ (1/3)

-- State the theorem
theorem expression_equals_point_one : expression = 0.1 := by
  sorry

end expression_equals_point_one_l1135_113576


namespace object_is_cylinder_l1135_113533

-- Define the properties of the object
structure GeometricObject where
  front_view : Type
  side_view : Type
  top_view : Type
  front_is_square : front_view = Square
  side_is_square : side_view = Square
  front_side_equal : front_view = side_view
  top_is_circle : top_view = Circle

-- Define the theorem
theorem object_is_cylinder (obj : GeometricObject) : obj = Cylinder := by
  sorry

end object_is_cylinder_l1135_113533


namespace cyclic_sum_inequality_l1135_113521

theorem cyclic_sum_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_sum : x^2 + y^2 + z^2 = 3) :
  (x^2 + y*z) / (x^2 + y*z + 1) + (y^2 + z*x) / (y^2 + z*x + 1) + (z^2 + x*y) / (z^2 + x*y + 1) ≤ 2 := by
  sorry

end cyclic_sum_inequality_l1135_113521


namespace num_true_propositions_even_l1135_113531

/-- A proposition type representing a logical statement. -/
structure Proposition : Type :=
  (is_true : Bool)

/-- A set of four related propositions (original, converse, inverse, and contrapositive). -/
structure RelatedPropositions : Type :=
  (original : Proposition)
  (converse : Proposition)
  (inverse : Proposition)
  (contrapositive : Proposition)

/-- The number of true propositions in a set of related propositions. -/
def num_true_propositions (rp : RelatedPropositions) : Nat :=
  (if rp.original.is_true then 1 else 0) +
  (if rp.converse.is_true then 1 else 0) +
  (if rp.inverse.is_true then 1 else 0) +
  (if rp.contrapositive.is_true then 1 else 0)

/-- Theorem stating that the number of true propositions in a set of related propositions
    can only be 0, 2, or 4. -/
theorem num_true_propositions_even (rp : RelatedPropositions) :
  num_true_propositions rp = 0 ∨ num_true_propositions rp = 2 ∨ num_true_propositions rp = 4 :=
by sorry

end num_true_propositions_even_l1135_113531


namespace quadratic_linear_intersection_l1135_113598

-- Define the quadratic and linear functions
def quadratic (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 1
def linear (a : ℝ) (x : ℝ) : ℝ := a * x

-- State the theorem
theorem quadratic_linear_intersection :
  ∃ (a b : ℝ),
    (∀ x : ℝ, quadratic a b (-2) = linear a (-2)) ∧
    (quadratic a b (-2) = 1) ∧
    (a = -1/2) ∧ (b = -2) ∧
    (∀ y₁ y₂ y₃ : ℝ,
      (quadratic a b 2 = y₁) →
      (quadratic a b b = y₂) →
      (quadratic a b (a - b) = y₃) →
      (y₁ < y₃ ∧ y₃ < y₂)) :=
sorry

end quadratic_linear_intersection_l1135_113598


namespace simplify_cube_roots_l1135_113591

theorem simplify_cube_roots : (64 : ℝ) ^ (1/3) - (216 : ℝ) ^ (1/3) = -2 := by
  sorry

end simplify_cube_roots_l1135_113591


namespace curve_tangent_values_l1135_113583

/-- The curve equation -/
def curve (x a b : ℝ) : ℝ := x^2 + a*x + b

/-- The tangent equation -/
def tangent (x y : ℝ) : Prop := x - y + 1 = 0

/-- Main theorem -/
theorem curve_tangent_values (a b : ℝ) :
  (∀ x y, curve x a b = y → tangent x y) →
  a = 1 ∧ b = 1 := by sorry

end curve_tangent_values_l1135_113583


namespace sum_of_squares_l1135_113529

theorem sum_of_squares (a b c : ℝ) : 
  (a + b + c) / 3 = 10 →
  (a * b * c) ^ (1/3 : ℝ) = 6 →
  3 / (1/a + 1/b + 1/c) = 4 →
  a^2 + b^2 + c^2 = 576 := by
sorry

end sum_of_squares_l1135_113529


namespace green_square_area_percentage_l1135_113509

/-- Represents a square flag with a symmetrical cross -/
structure FlagWithCross where
  side : ℝ
  crossWidth : ℝ
  crossArea : ℝ
  greenSquareArea : ℝ

/-- Properties of the flag with cross -/
def FlagWithCross.properties (flag : FlagWithCross) : Prop :=
  flag.side > 0 ∧
  flag.crossWidth > 0 ∧
  flag.crossWidth < flag.side / 2 ∧
  flag.crossArea = 0.49 * flag.side * flag.side ∧
  flag.greenSquareArea = (flag.side - 2 * flag.crossWidth) * (flag.side - 2 * flag.crossWidth)

/-- Theorem: If the cross occupies 49% of the flag area, the green square occupies 25% -/
theorem green_square_area_percentage (flag : FlagWithCross) 
  (h : flag.properties) : flag.greenSquareArea = 0.25 * flag.side * flag.side := by
  sorry


end green_square_area_percentage_l1135_113509


namespace max_product_under_constraint_l1135_113550

theorem max_product_under_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_constraint : 3 * x + 8 * y = 48) : x * y ≤ 24 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 8 * y = 48 ∧ x * y = 24 := by
  sorry

end max_product_under_constraint_l1135_113550


namespace root_product_equals_32_l1135_113581

theorem root_product_equals_32 : 
  (256 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 32 := by
  sorry

end root_product_equals_32_l1135_113581


namespace operations_sum_2345_l1135_113518

def apply_operations (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  (d1^2 * 1000) + (d2 * d3 * 100) + (d2 * d3 * 10) + (10 - d4)

theorem operations_sum_2345 :
  apply_operations 2345 = 5325 := by
  sorry

end operations_sum_2345_l1135_113518


namespace samson_utility_l1135_113596

/-- Utility function -/
def utility (math_hours : ℝ) (frisbee_hours : ℝ) : ℝ :=
  (math_hours + 2) * frisbee_hours

/-- The problem statement -/
theorem samson_utility (s : ℝ) : 
  utility (10 - 2*s) s = utility (2*s + 4) (3 - s) → s = 3/2 :=
by sorry

end samson_utility_l1135_113596


namespace factorial_expression_equals_100_l1135_113502

-- Define factorial
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem factorial_expression_equals_100 : 
  (factorial 11 - factorial 10) / factorial 9 = 100 := by
  sorry

end factorial_expression_equals_100_l1135_113502


namespace prob_two_red_balls_l1135_113559

/-- The probability of picking two red balls from a bag containing 3 red balls, 2 blue balls,
    and 3 green balls, when 2 balls are picked at random without replacement. -/
theorem prob_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ)
    (h1 : total_balls = red_balls + blue_balls + green_balls)
    (h2 : red_balls = 3)
    (h3 : blue_balls = 2)
    (h4 : green_balls = 3) :
    (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 3 / 28 := by
  sorry

end prob_two_red_balls_l1135_113559


namespace rational_triple_theorem_l1135_113532

/-- The set of triples that satisfy the conditions -/
def valid_triples : Set (ℚ × ℚ × ℚ) :=
  {(1, 1, 1), (1, 2, 2), (2, 4, 4), (2, 3, 6), (3, 3, 3)}

/-- A predicate that checks if a triple of rationals satisfies the conditions -/
def satisfies_conditions (p q r : ℚ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧
  (p + q + r).isInt ∧
  (1/p + 1/q + 1/r).isInt ∧
  (p * q * r).isInt

theorem rational_triple_theorem :
  ∀ p q r : ℚ, satisfies_conditions p q r ↔ (p, q, r) ∈ valid_triples :=
by sorry

end rational_triple_theorem_l1135_113532


namespace steps_to_madison_square_garden_l1135_113541

/-- The number of steps taken to reach Madison Square Garden -/
def total_steps (steps_down : ℕ) (steps_to_msg : ℕ) : ℕ :=
  steps_down + steps_to_msg

/-- Theorem stating the total number of steps taken to reach Madison Square Garden -/
theorem steps_to_madison_square_garden :
  total_steps 676 315 = 991 := by
  sorry

end steps_to_madison_square_garden_l1135_113541


namespace vector_operations_l1135_113542

def a : ℝ × ℝ := (3, 3)
def b : ℝ × ℝ := (1, 4)

theorem vector_operations :
  (2 • a - b = (5, 2)) ∧
  (∃ m : ℝ, m = -2 ∧ ∃ k : ℝ, k • (m • a + b) = 2 • a - b) := by
  sorry

end vector_operations_l1135_113542


namespace triangle_special_case_l1135_113588

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circumcenter and orthocenter
def circumcenter (t : Triangle) : ℝ × ℝ := sorry
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex B
def angle_B (t : Triangle) : ℝ := sorry

-- Main theorem
theorem triangle_special_case (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  distance t.B O = distance t.B H →
  (angle_B t = 60 ∨ angle_B t = 120) :=
by sorry

end triangle_special_case_l1135_113588


namespace tangent_at_one_tangent_through_point_l1135_113507

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + x + 2

-- Theorem for the tangent line at x = 1
theorem tangent_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 
    (x = 1 ∧ y = f 1) ∨ 
    (y - f 1 = (3 * 1^2 - 2 * 1 + 1) * (x - 1)) :=
sorry

-- Theorem for the tangent lines passing through (1,3)
theorem tangent_through_point :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, y = m₁*x + b₁ → (∃ t, f t = y ∧ 3*t^2 - 2*t + 1 = m₁ ∧ x = t)) ∧
    (∀ x y, y = m₂*x + b₂ → (∃ t, f t = y ∧ 3*t^2 - 2*t + 1 = m₂ ∧ x = t)) ∧
    m₁ = 1 ∧ b₁ = 2 ∧ m₂ = 2 ∧ b₂ = 1 :=
sorry

end tangent_at_one_tangent_through_point_l1135_113507


namespace koi_fish_after_six_weeks_l1135_113522

/-- Represents the number of fish in the tank -/
structure FishTank where
  koi : ℕ
  goldfish : ℕ
  angelfish : ℕ

/-- Calculates the total number of fish in the tank -/
def FishTank.total (ft : FishTank) : ℕ := ft.koi + ft.goldfish + ft.angelfish

/-- Represents the daily and weekly changes in fish numbers -/
structure FishChanges where
  koi_per_day : ℕ
  goldfish_per_day : ℕ
  angelfish_per_week : ℕ

/-- Calculates the new fish numbers after a given number of weeks -/
def apply_changes (initial : FishTank) (changes : FishChanges) (weeks : ℕ) : FishTank :=
  { koi := initial.koi + changes.koi_per_day * 7 * weeks,
    goldfish := initial.goldfish + changes.goldfish_per_day * 7 * weeks,
    angelfish := initial.angelfish + changes.angelfish_per_week * weeks }

theorem koi_fish_after_six_weeks
  (initial : FishTank)
  (changes : FishChanges)
  (h_initial_total : initial.total = 450)
  (h_changes : changes = { koi_per_day := 4, goldfish_per_day := 7, angelfish_per_week := 9 })
  (h_final_goldfish : (apply_changes initial changes 6).goldfish = 300)
  (h_final_angelfish : (apply_changes initial changes 6).angelfish = 180) :
  (apply_changes initial changes 6).koi = 486 :=
sorry

end koi_fish_after_six_weeks_l1135_113522


namespace savings_growth_l1135_113503

/-- The amount of money in a savings account after n years -/
def savings_amount (a : ℝ) (n : ℕ) : ℝ :=
  a * (1 + 0.02) ^ n

/-- Theorem: The amount of money in a savings account after n years,
    given an initial deposit of a rubles and a 2% annual interest rate,
    is equal to a × 1.02^n rubles. -/
theorem savings_growth (a : ℝ) (n : ℕ) :
  savings_amount a n = a * 1.02 ^ n :=
by sorry

end savings_growth_l1135_113503


namespace geometric_sequence_eighth_term_l1135_113577

theorem geometric_sequence_eighth_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (fifth_term : a 5 = 11) 
  (eleventh_term : a 11 = 5) : 
  a 8 = Real.sqrt 55 := by
sorry

end geometric_sequence_eighth_term_l1135_113577


namespace vector_angle_condition_l1135_113568

/-- Given two vectors a and b in R², if the angle between them is acute,
    then the second component of b satisfies the given conditions. -/
theorem vector_angle_condition (m : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, m)
  -- The angle between a and b is acute
  (0 < (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) ∧ 
   (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) < 1) →
  m > -1/2 ∧ m ≠ 2 := by
sorry

end vector_angle_condition_l1135_113568


namespace max_value_ratio_l1135_113562

theorem max_value_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c)^2 / (a^2 + b^2 + c^2) = 3 :=
by sorry

end max_value_ratio_l1135_113562


namespace curves_intersection_l1135_113530

/-- The first curve equation -/
def curve1 (x y : ℝ) : Prop :=
  2 * x^2 + 3 * x * y - 2 * y^2 - 6 * x + 3 * y = 0

/-- The second curve equation -/
def curve2 (x y : ℝ) : Prop :=
  3 * x^2 + 7 * x * y + 2 * y^2 - 7 * x + y - 6 = 0

/-- The set of intersection points -/
def intersection_points : Set (ℝ × ℝ) :=
  {(-1, 2), (1, 1), (0, 3/2), (3, 0), (4, -1/2), (5, -1)}

/-- Theorem stating that the given points are the intersection points of the two curves -/
theorem curves_intersection :
  ∀ (p : ℝ × ℝ), p ∈ intersection_points ↔ (curve1 p.1 p.2 ∧ curve2 p.1 p.2) :=
sorry

end curves_intersection_l1135_113530


namespace cousin_arrangement_count_l1135_113553

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- There are 4 rooms -/
def num_rooms : ℕ := 4

/-- The theorem stating the number of ways to arrange the cousins -/
theorem cousin_arrangement_count :
  distribute num_cousins num_rooms = 51 := by sorry

end cousin_arrangement_count_l1135_113553


namespace rabbit_chicken_puzzle_l1135_113515

theorem rabbit_chicken_puzzle (total_animals : ℕ) (rabbit_count : ℕ) : 
  total_animals = 40 →
  4 * rabbit_count = 10 * 2 * (total_animals - rabbit_count) + 8 →
  rabbit_count = 33 := by
sorry

end rabbit_chicken_puzzle_l1135_113515


namespace min_diff_y_x_l1135_113573

theorem min_diff_y_x (x y z : ℤ) 
  (h1 : x < y ∧ y < z) 
  (h2 : Even x)
  (h3 : Odd y ∧ Odd z)
  (h4 : ∀ w, (w : ℤ) ≥ x ∧ Odd w → w - x ≥ 9) :
  ∃ (d : ℤ), d = y - x ∧ ∀ (d' : ℤ), y - x ≤ d' := by
  sorry

end min_diff_y_x_l1135_113573


namespace dual_polyhedron_properties_l1135_113578

/-- A regular polyhedron with its dual -/
structure RegularPolyhedronWithDual where
  G : ℕ  -- number of faces
  P : ℕ  -- number of edges
  B : ℕ  -- number of vertices
  n : ℕ  -- number of edges meeting at each vertex

/-- Properties of the dual of a regular polyhedron -/
def dual_properties (poly : RegularPolyhedronWithDual) : Prop :=
  ∃ (dual_faces dual_edges dual_vertices : ℕ),
    dual_faces = poly.B ∧
    dual_edges = poly.P ∧
    dual_vertices = poly.G

/-- Theorem stating the properties of the dual polyhedron -/
theorem dual_polyhedron_properties (poly : RegularPolyhedronWithDual) :
  dual_properties poly :=
sorry

end dual_polyhedron_properties_l1135_113578


namespace fair_transaction_balance_l1135_113599

/-- Represents the financial transactions at a fair --/
structure FairTransactions where
  initial_amount : ℤ
  ride_expense : ℤ
  game_winnings : ℤ
  food_expense : ℤ
  found_money : ℤ
  final_amount : ℤ

/-- Calculates the total amount spent or gained at the fair --/
def total_spent_or_gained (t : FairTransactions) : ℤ :=
  t.initial_amount - t.final_amount

/-- Theorem stating that the total amount spent or gained is equal to 
    the difference between initial and final amounts --/
theorem fair_transaction_balance (t : FairTransactions) : 
  total_spent_or_gained t = 
    t.ride_expense + t.food_expense - t.game_winnings - t.found_money :=
by
  sorry

#eval total_spent_or_gained {
  initial_amount := 87,
  ride_expense := 25,
  game_winnings := 10,
  food_expense := 12,
  found_money := 5,
  final_amount := 16
}

end fair_transaction_balance_l1135_113599


namespace solution_range_l1135_113540

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 2*m - 2 = 0 ∧ 0 ≤ x ∧ x ≤ 3/2) ↔ 
  -1/2 ≤ m ∧ m ≤ 4 - 2*Real.sqrt 2 := by
sorry

end solution_range_l1135_113540


namespace pencil_cost_l1135_113594

-- Define the cost of a pen and a pencil as real numbers
variable (x y : ℝ)

-- Define the conditions from the problem
def condition1 : Prop := 5 * x + 4 * y = 320
def condition2 : Prop := 3 * x + 6 * y = 246

-- State the theorem to be proved
theorem pencil_cost (h1 : condition1 x y) (h2 : condition2 x y) : y = 15 := by
  sorry

end pencil_cost_l1135_113594


namespace cube_root_of_64_l1135_113549

theorem cube_root_of_64 : ∃ (a : ℝ), a^3 = 64 ∧ a = 4 := by sorry

end cube_root_of_64_l1135_113549


namespace average_growth_rate_satisfies_equation_average_growth_rate_is_twenty_percent_l1135_113535

/-- The average monthly growth rate from March to May for a shopping mall's sales volume. -/
def average_growth_rate : ℝ := 0.2

/-- The sales volume in February in yuan. -/
def february_sales : ℝ := 4000000

/-- The sales volume increase rate from February to March. -/
def march_increase_rate : ℝ := 0.1

/-- The sales volume in May in yuan. -/
def may_sales : ℝ := 6336000

/-- Theorem stating that the calculated average growth rate satisfies the sales volume equation. -/
theorem average_growth_rate_satisfies_equation :
  february_sales * (1 + march_increase_rate) * (1 + average_growth_rate)^2 = may_sales := by sorry

/-- Theorem stating that the average growth rate is indeed 20%. -/
theorem average_growth_rate_is_twenty_percent :
  average_growth_rate = 0.2 := by sorry

end average_growth_rate_satisfies_equation_average_growth_rate_is_twenty_percent_l1135_113535


namespace equation_solution_l1135_113590

theorem equation_solution : 
  ∃ (x : ℤ), 45 - (28 - (37 - (15 - x))) = 56 ∧ x = 122 := by
  sorry

end equation_solution_l1135_113590


namespace martha_family_women_without_daughters_l1135_113516

/-- Represents the family structure of Martha and her descendants -/
structure MarthaFamily where
  daughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of women (daughters and granddaughters) who have no daughters -/
def women_without_daughters (f : MarthaFamily) : ℕ :=
  f.total_descendants - f.daughters_with_children

/-- Theorem stating the number of women without daughters in Martha's family -/
theorem martha_family_women_without_daughters :
  ∀ f : MarthaFamily,
  f.daughters = 8 →
  f.total_descendants = 40 →
  f.daughters_with_children * 8 = f.total_descendants - f.daughters →
  women_without_daughters f = 36 :=
by sorry

end martha_family_women_without_daughters_l1135_113516


namespace running_speed_calculation_l1135_113538

theorem running_speed_calculation (walking_speed running_speed total_distance total_time : ℝ) :
  walking_speed = 4 →
  total_distance = 4 →
  total_time = 0.75 →
  (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time →
  running_speed = 8 := by
  sorry

end running_speed_calculation_l1135_113538


namespace no_triple_with_three_coprime_roots_l1135_113534

theorem no_triple_with_three_coprime_roots : ¬∃ (a b c x₁ x₂ x₃ : ℤ),
  (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
  (Int.gcd x₁ x₂ = 1 ∧ Int.gcd x₁ x₃ = 1 ∧ Int.gcd x₂ x₃ = 1) ∧
  (x₁^3 - a^2*x₁^2 + b^2*x₁ - a*b + 3*c = 0) ∧
  (x₂^3 - a^2*x₂^2 + b^2*x₂ - a*b + 3*c = 0) ∧
  (x₃^3 - a^2*x₃^2 + b^2*x₃ - a*b + 3*c = 0) :=
by
  sorry

end no_triple_with_three_coprime_roots_l1135_113534


namespace f_even_and_decreasing_l1135_113580

def f (x : ℝ) := -x^2

theorem f_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end f_even_and_decreasing_l1135_113580


namespace sine_function_max_min_l1135_113546

theorem sine_function_max_min (a b : ℝ) (h1 : a < 0) :
  (∀ x, a * Real.sin x + b ≤ 3) ∧
  (∀ x, a * Real.sin x + b ≥ -1) ∧
  (∃ x, a * Real.sin x + b = 3) ∧
  (∃ x, a * Real.sin x + b = -1) →
  a = -2 ∧ b = 1 := by
sorry

end sine_function_max_min_l1135_113546


namespace baseball_cards_l1135_113558

theorem baseball_cards (n : ℕ) : ∃ (total : ℕ), 
  (total = 3 * n + 1) ∧ (∃ (k : ℕ), total = 3 * k + 1) := by
  sorry

end baseball_cards_l1135_113558


namespace quotient_problem_l1135_113592

theorem quotient_problem (x : ℝ) (h : x = 0.3) : 0.009 / x = 0.03 := by
  sorry

end quotient_problem_l1135_113592


namespace commercial_length_proof_l1135_113514

theorem commercial_length_proof (x : ℝ) : 
  (3 * x + 11 * 2 = 37) → x = 5 := by
  sorry

end commercial_length_proof_l1135_113514


namespace smallest_regular_polygon_with_28_degree_extension_l1135_113557

/-- The angle (in degrees) at which two extended sides of a regular polygon meet -/
def extended_angle (n : ℕ) : ℚ :=
  180 / n

/-- Theorem stating that 45 is the smallest positive integer n for which
    a regular n-sided polygon has two extended sides meeting at an angle of 28 degrees -/
theorem smallest_regular_polygon_with_28_degree_extension :
  (∀ k : ℕ, k > 0 → k < 45 → extended_angle k ≠ 28) ∧ extended_angle 45 = 28 :=
sorry

end smallest_regular_polygon_with_28_degree_extension_l1135_113557


namespace imaginary_roots_sum_of_magnitudes_l1135_113582

theorem imaginary_roots_sum_of_magnitudes (m : ℝ) : 
  (∃ α β : ℂ, (3 * α^2 - 6*(m - 1)*α + m^2 + 1 = 0) ∧ 
               (3 * β^2 - 6*(m - 1)*β + m^2 + 1 = 0) ∧ 
               (α.im ≠ 0) ∧ (β.im ≠ 0) ∧
               (Complex.abs α + Complex.abs β = 2)) →
  m = Real.sqrt 2 :=
by sorry

end imaginary_roots_sum_of_magnitudes_l1135_113582


namespace percentage_problem_l1135_113555

theorem percentage_problem (x y : ℝ) : 
  x = 0.18 * 4750 →
  y = 1.3 * x →
  y / 8950 * 100 = 12.42 := by
sorry

end percentage_problem_l1135_113555


namespace window_cost_is_700_l1135_113526

/-- The cost of damages caused by Jack -/
def total_damage : ℕ := 1450

/-- The number of tires damaged -/
def num_tires : ℕ := 3

/-- The cost of each tire -/
def tire_cost : ℕ := 250

/-- The cost of the window -/
def window_cost : ℕ := total_damage - (num_tires * tire_cost)

theorem window_cost_is_700 : window_cost = 700 := by
  sorry

end window_cost_is_700_l1135_113526


namespace area_enclosed_theorem_l1135_113554

/-- Represents a configuration of three intersecting circles -/
structure CircleConfiguration where
  radius : ℝ
  centralAngle : ℝ
  numCircles : ℕ

/-- Calculates the area enclosed by the arcs of the circle configuration -/
def areaEnclosedByArcs (config : CircleConfiguration) : ℝ :=
  sorry

/-- Represents the coefficients of the area formula a√b + cπ -/
structure AreaCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the area can be expressed as a√b + cπ with a + b + c = 40.5 -/
theorem area_enclosed_theorem (config : CircleConfiguration) 
  (h1 : config.radius = 5)
  (h2 : config.centralAngle = π / 2)
  (h3 : config.numCircles = 3) :
  ∃ (coef : AreaCoefficients), 
    areaEnclosedByArcs config = coef.a * Real.sqrt coef.b + coef.c * π ∧
    coef.a + coef.b + coef.c = 40.5 :=
  sorry

end area_enclosed_theorem_l1135_113554


namespace sufficient_condition_transitivity_l1135_113585

theorem sufficient_condition_transitivity 
  (C B A : Prop) 
  (h1 : C → B) 
  (h2 : B → A) : 
  C → A := by
  sorry

end sufficient_condition_transitivity_l1135_113585


namespace sum_of_squares_of_roots_l1135_113517

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 6 * x₁ - 9 = 0) → 
  (3 * x₂^2 + 6 * x₂ - 9 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 10) :=
by sorry

end sum_of_squares_of_roots_l1135_113517


namespace expansion_coefficient_x_fifth_l1135_113561

theorem expansion_coefficient_x_fifth (x : ℝ) :
  ∃ (aₙ a₁ a₂ a₃ a₄ a₅ : ℝ),
    x^5 = aₙ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 ∧
    a₄ = 5 := by
  sorry

end expansion_coefficient_x_fifth_l1135_113561


namespace scientific_notation_448000_l1135_113504

theorem scientific_notation_448000 : 448000 = 4.48 * (10 : ℝ) ^ 5 := by
  sorry

end scientific_notation_448000_l1135_113504


namespace inequality_system_implies_a_leq_3_l1135_113525

theorem inequality_system_implies_a_leq_3 :
  (∀ x : ℝ, (4 * (x - 1) > 3 * x - 1 ∧ 5 * x > 3 * x + 2 * a) ↔ x > 3) →
  a ≤ 3 :=
by sorry

end inequality_system_implies_a_leq_3_l1135_113525


namespace abs_func_even_and_increasing_l1135_113566

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_func_even_and_increasing :
  (∀ x : ℝ, abs_func (-x) = abs_func x) ∧
  (∀ x y : ℝ, 0 < x → x < y → abs_func x < abs_func y) :=
by sorry

end abs_func_even_and_increasing_l1135_113566


namespace area_of_section_ABD_l1135_113586

theorem area_of_section_ABD (a : ℝ) (S : ℝ) (V : ℝ) : 
  a > 0 → 0 < S → S < π / 2 → V > 0 →
  let area_ABD := (Real.sqrt 3 / Real.sin S) * (V ^ (2 / 3) * Real.tan S) ^ (1 / 3)
  ∃ (h : ℝ), h > 0 ∧ 
    V = (a ^ 3 / 8) * Real.tan S ∧
    area_ABD = (a ^ 2 * Real.sqrt 3) / (4 * Real.cos S) :=
by sorry

#check area_of_section_ABD

end area_of_section_ABD_l1135_113586


namespace final_number_is_fifty_l1135_113574

/-- Represents the state of the board at any given time -/
structure BoardState where
  ones : Nat
  fours : Nat
  others : List Nat

/-- The operation of replacing two numbers with their Pythagorean sum -/
def replaceTwo (x y : Nat) : Nat :=
  Nat.sqrt (x^2 + y^2)

/-- The process of reducing the board until only one number remains -/
def reduceBoard : BoardState → Nat
| s => if s.ones + s.fours + s.others.length = 1
       then if s.ones = 1 then 1
            else if s.fours = 1 then 4
            else s.others.head!
       else sorry -- recursively apply replaceTwo

theorem final_number_is_fifty :
  ∀ (finalNum : Nat),
  (∃ (s : BoardState), s.ones = 900 ∧ s.fours = 100 ∧ s.others = [] ∧
   reduceBoard s = finalNum) →
  finalNum = 50 := by
  sorry

#check final_number_is_fifty

end final_number_is_fifty_l1135_113574


namespace cube_volume_surface_area_l1135_113547

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 3*x ∧ 6*s^2 = x) → x = 5832 := by
  sorry

end cube_volume_surface_area_l1135_113547


namespace second_platform_speed_l1135_113523

/-- The speed of Alex's platform in ft/s -/
def alex_speed : ℝ := 1

/-- The distance Alex's platform travels before falling, in ft -/
def fall_distance : ℝ := 100

/-- The time Edward arrives after Alex's platform starts, in seconds -/
def edward_arrival_time : ℝ := 60

/-- Edward's calculation time before launching the second platform, in seconds -/
def edward_calc_time : ℝ := 5

/-- The length of both platforms, in ft -/
def platform_length : ℝ := 5

/-- The optimal speed of the second platform that maximizes Alex's transfer time -/
def optimal_speed : ℝ := 1.125

theorem second_platform_speed (v : ℝ) :
  v = optimal_speed ↔
    (v > 0) ∧
    (v * (fall_distance / alex_speed - edward_arrival_time - edward_calc_time) = 
      fall_distance - alex_speed * edward_arrival_time + platform_length) ∧
    (∀ u : ℝ, u > 0 →
      (u * (fall_distance / alex_speed - edward_arrival_time - edward_calc_time) = 
        fall_distance - alex_speed * edward_arrival_time + platform_length) →
      v ≥ u) :=
by sorry

end second_platform_speed_l1135_113523


namespace infinitely_many_primes_2_mod_3_l1135_113587

theorem infinitely_many_primes_2_mod_3 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end infinitely_many_primes_2_mod_3_l1135_113587


namespace seven_valid_positions_l1135_113528

/-- Represents a position where an additional square can be attached --/
inductive Position
| CentralExtension
| OuterEdge
| MiddleEdge

/-- Represents the cross-shaped polygon --/
structure CrossPolygon where
  squares : Fin 6 → Unit  -- Represents the 6 squares in the cross
  additional_positions : Fin 11 → Position  -- Represents the 11 possible positions

/-- Represents a configuration with an additional square attached --/
structure ExtendedPolygon where
  base : CrossPolygon
  additional_square_position : Fin 11

/-- Predicate to check if a configuration can be folded into a cube with one face missing --/
def can_fold_to_cube (ep : ExtendedPolygon) : Prop :=
  sorry  -- Definition of this predicate would depend on the geometry of the problem

/-- The main theorem to be proved --/
theorem seven_valid_positions (cp : CrossPolygon) :
  (∃ (valid_positions : Finset (Fin 11)), 
    valid_positions.card = 7 ∧ 
    (∀ p : Fin 11, p ∈ valid_positions ↔ can_fold_to_cube ⟨cp, p⟩)) :=
  sorry


end seven_valid_positions_l1135_113528


namespace comparison_theorem_l1135_113544

theorem comparison_theorem :
  (-4 / 7 : ℚ) > -2 / 3 ∧ -(-7 : ℤ) > -|(-7 : ℤ)| := by sorry

end comparison_theorem_l1135_113544


namespace eightieth_digit_of_one_seventh_l1135_113506

def decimal_representation_of_one_seventh : List Nat := [1, 4, 2, 8, 5, 7]

theorem eightieth_digit_of_one_seventh : 
  (decimal_representation_of_one_seventh[(80 - 1) % decimal_representation_of_one_seventh.length] = 4) := by
  sorry

end eightieth_digit_of_one_seventh_l1135_113506


namespace star_operation_result_l1135_113567

-- Define the sets A and B
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define the set difference operation
def set_difference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define the * operation
def star_operation (X Y : Set ℝ) : Set ℝ := 
  (set_difference X Y) ∪ (set_difference Y X)

-- Theorem statement
theorem star_operation_result : 
  star_operation A B = {x | (-3 ≤ x ∧ x < 0) ∨ (x > 3)} := by sorry

end star_operation_result_l1135_113567


namespace expected_socks_is_2n_l1135_113512

/-- The expected number of socks taken until a pair is found -/
def expected_socks (n : ℕ) : ℝ :=
  2 * n

/-- Theorem stating that the expected number of socks taken until a pair is found is 2n -/
theorem expected_socks_is_2n (n : ℕ) (h : n > 0) :
  expected_socks n = 2 * n := by
  sorry

end expected_socks_is_2n_l1135_113512


namespace min_exponent_sum_l1135_113548

/-- Given a positive integer A that can be factorized as A = 2^α × 3^β × 5^γ,
    where α, β, and γ are natural numbers, and satisfying the conditions:
    - A/2 is a perfect square
    - A/3 is a perfect cube
    - A/5 is a perfect fifth power
    The minimum value of α + β + γ is 31. -/
theorem min_exponent_sum (A : ℕ+) (α β γ : ℕ) 
  (h_factorization : (A : ℕ) = 2^α * 3^β * 5^γ)
  (h_half_square : ∃ k : ℕ, 2 * k^2 = A)
  (h_third_cube : ∃ k : ℕ, 3 * k^3 = A)
  (h_fifth_power : ∃ k : ℕ, 5 * k^5 = A) :
  α + β + γ ≥ 31 :=
sorry

end min_exponent_sum_l1135_113548


namespace set_equivalences_l1135_113505

/-- The set of non-negative even numbers not greater than 10 -/
def nonNegEvenSet : Set ℕ :=
  {n : ℕ | n % 2 = 0 ∧ n ≤ 10}

/-- The set of prime numbers not greater than 10 -/
def primeSet : Set ℕ :=
  {n : ℕ | Nat.Prime n ∧ n ≤ 10}

/-- The equation x^2 + 2x - 15 = 0 -/
def equation (x : ℝ) : Prop :=
  x^2 + 2*x - 15 = 0

theorem set_equivalences :
  (nonNegEvenSet = {0, 2, 4, 6, 8, 10}) ∧
  (primeSet = {2, 3, 5, 7}) ∧
  ({x : ℝ | equation x} = {-5, 3}) := by
  sorry

end set_equivalences_l1135_113505


namespace smallest_largest_multiples_l1135_113556

theorem smallest_largest_multiples :
  ∃ (smallest largest : ℕ),
    (smallest ≥ 10 ∧ smallest < 100) ∧
    (largest ≥ 100 ∧ largest < 1000) ∧
    (∀ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n → n ≥ smallest) ∧
    (∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n → n ≤ largest) ∧
    2 ∣ smallest ∧ 3 ∣ smallest ∧ 5 ∣ smallest ∧
    2 ∣ largest ∧ 3 ∣ largest ∧ 5 ∣ largest ∧
    smallest = 30 ∧ largest = 990 := by
  sorry

end smallest_largest_multiples_l1135_113556


namespace class_average_l1135_113524

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) (high_score : ℝ) (rest_average : ℝ) :
  total_students = 25 →
  high_scorers = 3 →
  zero_scorers = 3 →
  high_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - high_scorers - zero_scorers
  let total_marks := high_scorers * high_score + zero_scorers * 0 + rest_students * rest_average
  total_marks / total_students = 45.6 := by
sorry

end class_average_l1135_113524


namespace gcd_condition_equivalence_l1135_113571

theorem gcd_condition_equivalence (m n : ℕ+) :
  (∀ (x y : ℕ+), x ∣ m → y ∣ n → Nat.gcd (x + y) (m * n) > 1) ↔ Nat.gcd m n > 1 := by
  sorry

end gcd_condition_equivalence_l1135_113571


namespace angle_properties_l1135_113552

theorem angle_properties (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.tan (2*θ) = -2*Real.sqrt 2) :
  (Real.tan θ = -Real.sqrt 2 / 2) ∧
  ((2 * (Real.cos (θ/2))^2 - Real.sin θ - Real.tan (5*π/4)) / (Real.sqrt 2 * Real.sin (θ + π/4)) = 3 + 2*Real.sqrt 2) := by
  sorry

end angle_properties_l1135_113552


namespace last_two_digits_of_sequence_sum_l1135_113527

/-- The sum of the sequence 8, 88, 888, ..., up to 2008 digits -/
def sequence_sum : ℕ := 8 + 88 * 2007

/-- The last two digits of a number -/
def last_two_digits (n : ℕ) : ℕ := n % 100

/-- Theorem: The last two digits of the sequence sum are 24 -/
theorem last_two_digits_of_sequence_sum :
  last_two_digits sequence_sum = 24 := by sorry

end last_two_digits_of_sequence_sum_l1135_113527


namespace balance_implies_20g_difference_l1135_113510

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem balance_implies_20g_difference 
  (weights : Finset ℕ) 
  (h_weights : weights = Finset.range 40)
  (left_pan right_pan : Finset ℕ)
  (h_left : left_pan ⊆ weights ∧ left_pan.card = 10 ∧ ∀ n ∈ left_pan, is_even n)
  (h_right : right_pan ⊆ weights ∧ right_pan.card = 10 ∧ ∀ n ∈ right_pan, is_odd n)
  (h_balance : left_pan.sum id = right_pan.sum id) :
  ∃ (a b : ℕ), (a ∈ left_pan ∧ b ∈ left_pan ∧ a - b = 20) ∨ 
               (a ∈ right_pan ∧ b ∈ right_pan ∧ a - b = 20) :=
sorry

end balance_implies_20g_difference_l1135_113510


namespace arithmetic_sequence_first_term_l1135_113593

/-- An arithmetic sequence with the given property -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_first_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_prop : ∀ n : ℕ+, a (n + 1) + a (n + 2) = 3 * (n : ℚ) + 5) :
  a 1 = 7 / 4 := by
sorry

end arithmetic_sequence_first_term_l1135_113593


namespace brick_height_calculation_l1135_113543

/-- Calculates the height of a brick given wall dimensions and brick count -/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ) 
  (brick_length brick_width : ℝ) (brick_count : ℕ) :
  wall_length = 27 →
  wall_width = 2 →
  wall_height = 0.75 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_count = 27000 →
  ∃ (brick_height : ℝ), 
    brick_height = (wall_length * wall_width * wall_height) / (brick_length * brick_width * brick_count) ∧
    brick_height = 0.075 := by
  sorry

end brick_height_calculation_l1135_113543


namespace square_difference_equality_l1135_113536

theorem square_difference_equality (a b M : ℝ) : 
  (a + 2*b)^2 = (a - 2*b)^2 + M → M = 8*a*b := by
  sorry

end square_difference_equality_l1135_113536


namespace flowers_lilly_can_buy_l1135_113570

def days_until_birthday : ℕ := 22
def savings_per_day : ℚ := 2
def cost_per_flower : ℚ := 4

theorem flowers_lilly_can_buy :
  (days_until_birthday : ℚ) * savings_per_day / cost_per_flower = 11 := by
  sorry

end flowers_lilly_can_buy_l1135_113570


namespace exponent_operation_l1135_113565

theorem exponent_operation (a : ℝ) : -(-a)^2 * a^4 = -a^6 := by
  sorry

end exponent_operation_l1135_113565


namespace subtraction_of_negative_problem_solution_l1135_113537

theorem subtraction_of_negative (a b : ℤ) : a - (-b) = a + b := by sorry

theorem problem_solution : 2 - (-12) = 14 := by sorry

end subtraction_of_negative_problem_solution_l1135_113537


namespace symmetric_point_correct_l1135_113564

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to x-axis -/
def symmetricXAxis (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

/-- The original point -/
def originalPoint : Point3D :=
  ⟨2, 3, 4⟩

/-- The symmetric point -/
def symmetricPoint : Point3D :=
  ⟨2, -3, -4⟩

theorem symmetric_point_correct : symmetricXAxis originalPoint = symmetricPoint := by
  sorry

end symmetric_point_correct_l1135_113564


namespace f_greater_than_one_range_l1135_113575

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_greater_than_one_range :
  {x₀ : ℝ | f x₀ > 1} = Set.Ioi 1 ∪ Set.Iic (-1) :=
sorry

end f_greater_than_one_range_l1135_113575


namespace cos_period_scaled_cos_third_period_l1135_113595

/-- The period of cosine function with a scaled argument -/
theorem cos_period_scaled (a : ℝ) (ha : a ≠ 0) : 
  let f := fun x => Real.cos (x / a)
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q ∧ q < p → ∃ x, f (x + q) ≠ f x :=
by
  sorry

/-- The period of y = cos(x/3) is 6π -/
theorem cos_third_period : 
  let f := fun x => Real.cos (x / 3)
  ∃ p : ℝ, p = 6 * Real.pi ∧ p > 0 ∧ ∀ x, f (x + p) = f x ∧ 
    ∀ q, 0 < q ∧ q < p → ∃ x, f (x + q) ≠ f x :=
by
  sorry

end cos_period_scaled_cos_third_period_l1135_113595


namespace bus_distance_theorem_l1135_113589

/-- Calculates the total distance traveled by a bus with increasing speed over a given number of hours -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  hours * (2 * initialSpeed + (hours - 1) * speedIncrease) / 2

/-- Theorem stating that a bus with given initial speed and speed increase travels a specific distance in 12 hours -/
theorem bus_distance_theorem (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) :
  initialSpeed = 35 →
  speedIncrease = 2 →
  hours = 12 →
  totalDistance initialSpeed speedIncrease hours = 552 := by
  sorry

#eval totalDistance 35 2 12  -- This should evaluate to 552

end bus_distance_theorem_l1135_113589


namespace tyler_meal_choices_l1135_113560

-- Define the number of options for each food category
def num_meat_options : ℕ := 3
def num_vegetable_options : ℕ := 5
def num_dessert_options : ℕ := 5

-- Define the number of vegetables to be chosen
def num_vegetables_to_choose : ℕ := 2

-- Theorem statement
theorem tyler_meal_choices :
  (num_meat_options) *
  (num_vegetable_options.choose num_vegetables_to_choose) *
  (num_dessert_options) = 150 := by
  sorry


end tyler_meal_choices_l1135_113560


namespace min_lines_8x8_grid_is_14_l1135_113539

/-- The minimum number of straight lines required to separate all points in an 8x8 grid -/
def min_lines_8x8_grid : ℕ := 14

/-- The number of rows in the grid -/
def num_rows : ℕ := 8

/-- The number of columns in the grid -/
def num_columns : ℕ := 8

/-- The total number of points in the grid -/
def total_points : ℕ := num_rows * num_columns

/-- Theorem stating that the minimum number of lines to separate all points in an 8x8 grid is 14 -/
theorem min_lines_8x8_grid_is_14 : 
  min_lines_8x8_grid = (num_rows - 1) + (num_columns - 1) :=
sorry

end min_lines_8x8_grid_is_14_l1135_113539


namespace min_pizzas_to_cover_costs_l1135_113551

def car_cost : ℕ := 8000
def earnings_per_pizza : ℕ := 12
def gas_cost_per_delivery : ℕ := 4
def monthly_maintenance : ℕ := 200

theorem min_pizzas_to_cover_costs : 
  ∃ (p : ℕ), p = 1025 ∧ 
  (p * (earnings_per_pizza - gas_cost_per_delivery) ≥ car_cost + monthly_maintenance) ∧
  ∀ (q : ℕ), q < p → q * (earnings_per_pizza - gas_cost_per_delivery) < car_cost + monthly_maintenance :=
sorry

end min_pizzas_to_cover_costs_l1135_113551


namespace first_valid_year_is_2015_l1135_113508

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ sum_of_digits year = 8

theorem first_valid_year_is_2015 : 
  ∀ year : ℕ, is_valid_year year → year ≥ 2015 :=
sorry

end first_valid_year_is_2015_l1135_113508


namespace right_triangle_inscribed_circle_area_ratio_l1135_113597

theorem right_triangle_inscribed_circle_area_ratio 
  (h a r : ℝ) (h_pos : h > 0) (a_pos : a > 0) (r_pos : r > 0) (h_gt_a : h > a) :
  let A := (1/2) * a * Real.sqrt (h^2 - a^2)
  (π * r^2) / A = 4 * π * A / (a + Real.sqrt (h^2 - a^2) + h)^2 :=
by sorry

end right_triangle_inscribed_circle_area_ratio_l1135_113597


namespace no_multiple_hundred_scores_l1135_113545

/-- Represents the types of wishes available to modify exam scores -/
inductive Wish
  | AddOne      : Wish  -- Add one point to each exam
  | DecreaseOne : Fin 3 → Wish  -- Decrease one exam by 3, increase others by 1

/-- Represents the state of exam scores -/
structure ExamScores where
  russian : ℕ
  physics : ℕ
  math : ℕ
  russian_physics_diff : russian = physics - 3
  physics_math_diff : physics = math - 7

/-- Applies a wish to the exam scores -/
def applyWish (scores : ExamScores) (wish : Wish) : ExamScores :=
  sorry

/-- Checks if more than one exam score is at least 100 -/
def moreThanOneHundred (scores : ExamScores) : Prop :=
  sorry

/-- Main theorem: It's impossible to achieve 100 or more in more than one exam -/
theorem no_multiple_hundred_scores (initial : ExamScores) (wishes : List Wish) :
  ¬∃ (final : ExamScores), (List.foldl applyWish initial wishes = final ∧ moreThanOneHundred final) :=
  sorry

end no_multiple_hundred_scores_l1135_113545


namespace consecutive_integers_sum_properties_l1135_113519

theorem consecutive_integers_sum_properties :
  (∀ k : ℤ, ¬∃ n : ℤ, 12 * k + 78 = n ^ 2) ∧
  (∃ k : ℤ, ∃ n : ℤ, 11 * k + 66 = n ^ 2) := by
  sorry

end consecutive_integers_sum_properties_l1135_113519


namespace inverse_variation_problem_l1135_113579

/-- Given that x and y are positive real numbers, x² and y² vary inversely,
    and y = 5 when x = 2, prove that x = 2/25 when y = 125. -/
theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ (k : ℝ), ∀ x y, x^2 * y^2 = k)
  (h_initial : 2^2 * 5^2 = x^2 * 125^2) :
  y = 125 → x = 2/25 := by
sorry

end inverse_variation_problem_l1135_113579


namespace point_on_line_l1135_113584

/-- A point on a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line (t : ℝ) :
  let p1 : Point := ⟨2, 4⟩
  let p2 : Point := ⟨10, 1⟩
  let p3 : Point := ⟨t, 7⟩
  collinear p1 p2 p3 → t = -6 := by
  sorry


end point_on_line_l1135_113584
