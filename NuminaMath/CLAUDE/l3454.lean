import Mathlib

namespace cab_driver_income_l3454_345418

theorem cab_driver_income (income1 income2 income3 income4 : ℕ) (average : ℚ) :
  income1 = 45 →
  income2 = 50 →
  income3 = 60 →
  income4 = 65 →
  average = 58 →
  ∃ income5 : ℕ, 
    (income1 + income2 + income3 + income4 + income5 : ℚ) / 5 = average ∧
    income5 = 70 := by
  sorry

end cab_driver_income_l3454_345418


namespace complex_number_existence_l3454_345406

theorem complex_number_existence : ∃ z : ℂ, 
  (z + 10 / z).im = 0 ∧ (z + 4).re = (z + 4).im :=
sorry

end complex_number_existence_l3454_345406


namespace mixed_box_weight_l3454_345471

/-- The weight of a box with 100 aluminum balls -/
def weight_aluminum : ℝ := 510

/-- The weight of a box with 100 plastic balls -/
def weight_plastic : ℝ := 490

/-- The number of aluminum balls in the mixed box -/
def num_aluminum : ℕ := 20

/-- The number of plastic balls in the mixed box -/
def num_plastic : ℕ := 80

/-- The total number of balls in each box -/
def total_balls : ℕ := 100

theorem mixed_box_weight : 
  (num_aluminum : ℝ) / total_balls * weight_aluminum + 
  (num_plastic : ℝ) / total_balls * weight_plastic = 494 := by
  sorry

end mixed_box_weight_l3454_345471


namespace remainder_sum_powers_mod_5_l3454_345491

theorem remainder_sum_powers_mod_5 :
  (Nat.pow 9 7 + Nat.pow 4 5 + Nat.pow 3 9) % 5 = 1 := by
  sorry

end remainder_sum_powers_mod_5_l3454_345491


namespace max_at_two_implies_c_six_l3454_345463

/-- The function f(x) defined as x(x-c)^2 --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- Theorem stating that if f(x) has a maximum at x = 2, then c = 6 --/
theorem max_at_two_implies_c_six :
  ∀ c : ℝ, (∀ x : ℝ, f c x ≤ f c 2) → c = 6 :=
by sorry

end max_at_two_implies_c_six_l3454_345463


namespace categorical_variables_l3454_345487

-- Define the variables
def Smoking : Type := String
def Gender : Type := String
def ReligiousBelief : Type := String
def Nationality : Type := String

-- Define what it means for a variable to be categorical
def IsCategorical (α : Type) : Prop := ∃ (categories : Set α), Finite categories ∧ (∀ x : α, x ∈ categories)

-- State the theorem
theorem categorical_variables :
  IsCategorical Gender ∧ IsCategorical ReligiousBelief ∧ IsCategorical Nationality :=
sorry

end categorical_variables_l3454_345487


namespace product_closest_to_315_l3454_345499

def product : ℝ := 3.57 * 9.052 * (6.18 + 3.821)

def options : List ℝ := [200, 300, 315, 400, 500]

theorem product_closest_to_315 :
  ∀ x ∈ options, |product - 315| ≤ |product - x| :=
sorry

end product_closest_to_315_l3454_345499


namespace container_volume_transformation_l3454_345466

/-- A cuboid container with volume measured in gallons -/
structure Container where
  height : ℝ
  length : ℝ
  width : ℝ
  volume : ℝ
  volume_eq : volume = height * length * width

/-- Theorem stating that if a container with 3 gallon volume has its height doubled and length tripled, its new volume will be 18 gallons -/
theorem container_volume_transformation (c : Container) 
  (h_volume : c.volume = 3) :
  let new_container := Container.mk 
    (2 * c.height) 
    (3 * c.length) 
    c.width 
    ((2 * c.height) * (3 * c.length) * c.width)
    (by simp)
  new_container.volume = 18 := by
  sorry

end container_volume_transformation_l3454_345466


namespace xyz_sum_bounds_l3454_345420

theorem xyz_sum_bounds (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  let m := x*y + y*z + z*x
  (∃ (k : ℝ), ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 1 → x*y + y*z + z*x ≤ k) ∧
  (∃ (l : ℝ), ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 1 → l ≤ x*y + y*z + z*x) ∧
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 1 ∧ x*y + y*z + z*x = 1) ∧
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 1 ∧ x*y + y*z + z*x = -1/2) :=
sorry

end xyz_sum_bounds_l3454_345420


namespace rectangle_perimeter_l3454_345401

theorem rectangle_perimeter (area : ℝ) (side_difference : ℝ) (perimeter : ℝ) : 
  area = 500 →
  side_difference = 5 →
  (∃ x : ℝ, x > 0 ∧ x * (x + side_difference) = area) →
  perimeter = 90 :=
by sorry

end rectangle_perimeter_l3454_345401


namespace complex_number_coordinates_l3454_345417

/-- Given a complex number z = (1 + 2i^3) / (2 + i), prove that its coordinates in the complex plane are (0, -1) -/
theorem complex_number_coordinates :
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 2 * i^3) / (2 + i)
  z.re = 0 ∧ z.im = -1 := by sorry

end complex_number_coordinates_l3454_345417


namespace factorization_valid_l3454_345444

-- Define the left-hand side of the equation
def lhs (x : ℝ) : ℝ := -8 * x^2 + 8 * x - 2

-- Define the right-hand side of the equation
def rhs (x : ℝ) : ℝ := -2 * (2 * x - 1)^2

-- Theorem stating that the left-hand side equals the right-hand side for all real x
theorem factorization_valid (x : ℝ) : lhs x = rhs x := by
  sorry

end factorization_valid_l3454_345444


namespace square_area_above_line_l3454_345473

/-- Given a square with vertices at (2,1), (7,1), (7,6), and (2,6),
    and a line connecting points (2,1) and (7,3),
    the fraction of the square's area above this line is 4/5. -/
theorem square_area_above_line : 
  let square_vertices : List (ℝ × ℝ) := [(2,1), (7,1), (7,6), (2,6)]
  let line_points : List (ℝ × ℝ) := [(2,1), (7,3)]
  let total_area : ℝ := 25
  let area_above_line : ℝ := 20
  (area_above_line / total_area) = 4/5 := by sorry

end square_area_above_line_l3454_345473


namespace cost_price_equation_l3454_345441

/-- The cost price of a watch satisfying the given conditions -/
def cost_price : ℝ := 
  let C : ℝ := 2070.31
  C

/-- Theorem stating the equation that the cost price must satisfy -/
theorem cost_price_equation : 
  3 * (0.925 * cost_price + 265) = 3 * cost_price * 1.053 := by
  sorry

end cost_price_equation_l3454_345441


namespace purple_valley_skirts_l3454_345433

/-- The number of skirts in Azure Valley -/
def azure_skirts : ℕ := 60

/-- The number of skirts in Seafoam Valley -/
def seafoam_skirts : ℕ := (2 * azure_skirts) / 3

/-- The number of skirts in Purple Valley -/
def purple_skirts : ℕ := seafoam_skirts / 4

/-- Theorem stating that Purple Valley has 10 skirts -/
theorem purple_valley_skirts : purple_skirts = 10 := by
  sorry

end purple_valley_skirts_l3454_345433


namespace triangle_problem_l3454_345408

theorem triangle_problem (A B C : Real) (a b c S : Real) :
  -- Given conditions
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = Real.sqrt 3 →
  S = Real.sqrt 3 / 2 →
  -- Triangle properties
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  S = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  -- Conclusion
  A = π/3 ∧ b + c = 3 := by
  sorry

end triangle_problem_l3454_345408


namespace card_58_is_six_l3454_345455

/-- Represents a playing card value -/
inductive CardValue
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Converts a natural number to a card value -/
def natToCardValue (n : ℕ) : CardValue :=
  match n % 13 with
  | 0 => CardValue.Ace
  | 1 => CardValue.Two
  | 2 => CardValue.Three
  | 3 => CardValue.Four
  | 4 => CardValue.Five
  | 5 => CardValue.Six
  | 6 => CardValue.Seven
  | 7 => CardValue.Eight
  | 8 => CardValue.Nine
  | 9 => CardValue.Ten
  | 10 => CardValue.Jack
  | 11 => CardValue.Queen
  | _ => CardValue.King

theorem card_58_is_six :
  natToCardValue 57 = CardValue.Six :=
by sorry

end card_58_is_six_l3454_345455


namespace number_of_cats_l3454_345422

/-- Represents the number of cats on the ship. -/
def cats : ℕ := sorry

/-- Represents the number of sailors on the ship. -/
def sailors : ℕ := sorry

/-- Represents the number of cooks on the ship. -/
def cooks : ℕ := 1

/-- Represents the number of captains on the ship. -/
def captains : ℕ := 1

/-- The total number of heads on the ship. -/
def total_heads : ℕ := 16

/-- The total number of legs on the ship. -/
def total_legs : ℕ := 41

/-- Theorem stating that the number of cats on the ship is 5. -/
theorem number_of_cats : cats = 5 := by
  have head_count : cats + sailors + cooks + captains = total_heads := sorry
  have leg_count : 4 * cats + 2 * sailors + 2 * cooks + captains = total_legs := sorry
  sorry

end number_of_cats_l3454_345422


namespace sum_odd_integers_11_to_39_l3454_345465

/-- The sum of odd integers from 11 to 39 (inclusive) is 375 -/
theorem sum_odd_integers_11_to_39 : 
  (Finset.range 15).sum (fun i => 2 * i + 11) = 375 := by
  sorry

end sum_odd_integers_11_to_39_l3454_345465


namespace tan_roots_expression_value_l3454_345434

theorem tan_roots_expression_value (α β : ℝ) :
  (∃ x y : ℝ, x^2 - 4*x - 2 = 0 ∧ y^2 - 4*y - 2 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  Real.cos (α + β)^2 + 2 * Real.sin (α + β) * Real.cos (α + β) - 3 * Real.sin (α + β)^2 = -3/5 :=
by sorry

end tan_roots_expression_value_l3454_345434


namespace voting_stabilizes_l3454_345495

/-- Represents the state of votes in a circular arrangement -/
def VoteState := Vector Bool 25

/-- Represents the next state of votes based on the current state -/
def nextState (current : VoteState) : VoteState :=
  Vector.ofFn (fun i =>
    let prev := current.get ((i - 1 + 25) % 25)
    let next := current.get ((i + 1) % 25)
    let curr := current.get i
    if prev = next then curr else !curr)

/-- Theorem stating that the voting pattern will eventually stabilize -/
theorem voting_stabilizes : ∃ (n : ℕ), ∀ (initial : VoteState),
  ∃ (k : ℕ), k ≤ n ∧ nextState^[k] initial = nextState^[k+1] initial :=
sorry


end voting_stabilizes_l3454_345495


namespace fraction_simplification_l3454_345474

theorem fraction_simplification (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 5) :
  (15 * x^2 * y^3) / (9 * x * y^2 * z) = 2 := by
  sorry

end fraction_simplification_l3454_345474


namespace flower_bed_distance_l3454_345477

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- The total distance walked around a rectangle multiple times -/
def total_distance (length width : ℝ) (times : ℕ) : ℝ :=
  (rectangle_perimeter length width) * times

theorem flower_bed_distance :
  total_distance 5 3 3 = 30 := by sorry

end flower_bed_distance_l3454_345477


namespace statement_consistency_l3454_345480

def Statement : Type := Bool

def statementA (a b c d e : Statement) : Prop :=
  (a = true ∨ b = true ∨ c = true ∨ d = true ∨ e = true) ∧
  ¬(a = true ∧ b = true) ∧ ¬(a = true ∧ c = true) ∧ ¬(a = true ∧ d = true) ∧ ¬(a = true ∧ e = true) ∧
  ¬(b = true ∧ c = true) ∧ ¬(b = true ∧ d = true) ∧ ¬(b = true ∧ e = true) ∧
  ¬(c = true ∧ d = true) ∧ ¬(c = true ∧ e = true) ∧
  ¬(d = true ∧ e = true)

def statementC (a b c d e : Statement) : Prop :=
  a = true ∧ b = true ∧ c = true ∧ d = true ∧ e = true

def statementE (a : Statement) : Prop :=
  a = true

theorem statement_consistency :
  ∀ (a b c d e : Statement),
  (statementA a b c d e ↔ a = true) →
  (statementC a b c d e ↔ c = true) →
  (statementE a ↔ e = true) →
  (a = false ∧ b = true ∧ c = false ∧ d = true ∧ e = false) :=
by sorry

end statement_consistency_l3454_345480


namespace present_age_of_B_l3454_345435

/-- Given three people A, B, and C, whose ages satisfy certain conditions,
    prove that the present age of B is 30 years. -/
theorem present_age_of_B (A B C : ℕ) : 
  A + B + C = 90 →  -- Total present age is 90
  (A - 10) = 1 * x ∧ (B - 10) = 2 * x ∧ (C - 10) = 3 * x →  -- Age ratio 10 years ago
  B = 30 := by
sorry


end present_age_of_B_l3454_345435


namespace complex_equation_solution_l3454_345472

theorem complex_equation_solution (m : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (↑m + 2 * Complex.I) * (2 - Complex.I) = 4 + 3 * Complex.I →
  m = 1 :=
by
  sorry

end complex_equation_solution_l3454_345472


namespace puppy_weight_l3454_345410

/-- Given the weights of a puppy, a smaller cat, and a larger cat, prove that the puppy weighs 5 pounds. -/
theorem puppy_weight (p s l : ℝ) 
  (total_weight : p + s + l = 30)
  (puppy_larger_cat : p + l = 3 * s)
  (puppy_smaller_cat : p + s = l - 5) :
  p = 5 := by
  sorry

end puppy_weight_l3454_345410


namespace prob_skew_lines_l3454_345447

/-- A cube with 8 vertices -/
structure Cube :=
  (vertices : Finset (Fin 8))

/-- A line determined by two vertices of the cube -/
structure Line (c : Cube) :=
  (v1 v2 : Fin 8)
  (h1 : v1 ∈ c.vertices)
  (h2 : v2 ∈ c.vertices)
  (h3 : v1 ≠ v2)

/-- Two lines are skew if they are non-coplanar and non-intersecting -/
def are_skew (c : Cube) (l1 l2 : Line c) : Prop :=
  sorry

/-- The set of all lines determined by any two vertices of the cube -/
def all_lines (c : Cube) : Finset (Line c) :=
  sorry

/-- The probability of an event occurring when choosing two lines from all_lines -/
def probability (c : Cube) (event : Line c → Line c → Prop) : ℚ :=
  sorry

theorem prob_skew_lines (c : Cube) :
  probability c (λ l1 l2 => are_skew c l1 l2) = 29 / 63 :=
sorry

end prob_skew_lines_l3454_345447


namespace triangle_area_theorem_l3454_345469

/-- Given a triangle ABC with an arbitrary point inside it, and three lines drawn through
    this point parallel to the sides of the triangle, dividing it into six parts including
    three triangles with areas S₁, S₂, and S₃, the area of triangle ABC is (√S₁ + √S₂ + √S₃)². -/
theorem triangle_area_theorem (S₁ S₂ S₃ : ℝ) (h₁ : 0 < S₁) (h₂ : 0 < S₂) (h₃ : 0 < S₃) :
  ∃ (S : ℝ), S > 0 ∧ S = (Real.sqrt S₁ + Real.sqrt S₂ + Real.sqrt S₃)^2 :=
by sorry

end triangle_area_theorem_l3454_345469


namespace product_is_twice_square_l3454_345450

theorem product_is_twice_square (a b c d : ℕ+) (h : a * b = 2 * c * d) :
  ∃ (n : ℕ), a * b * c * d = 2 * n^2 := by
  sorry

end product_is_twice_square_l3454_345450


namespace sqrt_588_simplification_l3454_345445

theorem sqrt_588_simplification : Real.sqrt 588 = 14 * Real.sqrt 3 := by
  sorry

end sqrt_588_simplification_l3454_345445


namespace rug_area_l3454_345457

/-- The area of a rug on a rectangular floor with uncovered strips along the edges -/
theorem rug_area (floor_length floor_width strip_width : ℝ) 
  (h1 : floor_length = 12)
  (h2 : floor_width = 10)
  (h3 : strip_width = 3)
  (h4 : floor_length > 0)
  (h5 : floor_width > 0)
  (h6 : strip_width > 0)
  (h7 : 2 * strip_width < floor_length)
  (h8 : 2 * strip_width < floor_width) :
  (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) = 24 := by
  sorry

end rug_area_l3454_345457


namespace dougs_age_l3454_345439

theorem dougs_age (betty_age : ℕ) (doug_age : ℕ) (pack_cost : ℕ) :
  2 * betty_age = pack_cost →
  betty_age + doug_age = 90 →
  20 * pack_cost = 2000 →
  doug_age = 40 := by
sorry

end dougs_age_l3454_345439


namespace bananas_arrangements_l3454_345428

def word_length : ℕ := 7
def a_count : ℕ := 3
def n_count : ℕ := 2

theorem bananas_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial) = 420 := by
  sorry

end bananas_arrangements_l3454_345428


namespace point_on_line_l3454_345456

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line :
  let p1 : Point := ⟨4, 8⟩
  let p2 : Point := ⟨0, -4⟩
  let p3 : Point := ⟨2, 2⟩
  collinear p1 p2 p3 := by sorry

end point_on_line_l3454_345456


namespace outfits_count_l3454_345400

/-- The number of different outfits that can be made from a given number of shirts, ties, and belts. -/
def number_of_outfits (shirts ties belts : ℕ) : ℕ := shirts * ties * belts

/-- Theorem stating that the number of outfits from 8 shirts, 7 ties, and 4 belts is 224. -/
theorem outfits_count : number_of_outfits 8 7 4 = 224 := by
  sorry

end outfits_count_l3454_345400


namespace page_lines_increase_l3454_345440

theorem page_lines_increase (L : ℕ) (h1 : (60 : ℝ) / L = 1 / 3) : L + 60 = 240 := by
  sorry

end page_lines_increase_l3454_345440


namespace burritos_per_box_burritos_problem_l3454_345415

theorem burritos_per_box (total_boxes : ℕ) (fraction_given_away : ℚ) 
  (burritos_eaten_per_day : ℕ) (days_eaten : ℕ) (burritos_left : ℕ) : ℕ :=
let burritos_per_box := 
  (burritos_left + burritos_eaten_per_day * days_eaten) / 
  (total_boxes * (1 - fraction_given_away))
20

theorem burritos_problem : 
  burritos_per_box 3 (1/3) 3 10 10 = 20 := by
sorry

end burritos_per_box_burritos_problem_l3454_345415


namespace whitewashing_cost_is_8154_l3454_345475

/-- Calculates the cost of white washing a room with given dimensions and openings -/
def whitewashingCost (length width height : ℝ) (doorLength doorWidth : ℝ)
  (windowLength windowWidth : ℝ) (windowCount : ℕ) (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let doorArea := doorLength * doorWidth
  let windowArea := windowLength * windowWidth * windowCount
  let netArea := wallArea - doorArea - windowArea
  netArea * costPerSquareFoot

/-- Theorem stating the cost of white washing the room with given specifications -/
theorem whitewashing_cost_is_8154 :
  whitewashingCost 25 15 12 6 3 4 3 3 9 = 8154 := by
  sorry

end whitewashing_cost_is_8154_l3454_345475


namespace scientific_notation_of_number_l3454_345419

def number : ℝ := 308000000

theorem scientific_notation_of_number :
  ∃ (a : ℝ) (n : ℤ), number = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.08 ∧ n = 8 :=
sorry

end scientific_notation_of_number_l3454_345419


namespace total_cost_calculation_l3454_345483

/-- Calculates the total cost of a medical visit given the insurance coverage percentage and out-of-pocket cost -/
theorem total_cost_calculation (insurance_coverage_percent : ℝ) (out_of_pocket_cost : ℝ) : 
  insurance_coverage_percent = 80 → 
  out_of_pocket_cost = 60 → 
  (100 - insurance_coverage_percent) / 100 * (out_of_pocket_cost / ((100 - insurance_coverage_percent) / 100)) = 300 := by
sorry

end total_cost_calculation_l3454_345483


namespace field_trip_cost_l3454_345453

def total_cost (students : ℕ) (teachers : ℕ) (bus_capacity : ℕ) (rental_cost : ℕ) (toll_cost : ℕ) : ℕ :=
  let total_people := students + teachers
  let buses_needed := (total_people + bus_capacity - 1) / bus_capacity
  buses_needed * (rental_cost + toll_cost)

theorem field_trip_cost :
  total_cost 252 8 41 300000 7500 = 2152500 :=
by sorry

end field_trip_cost_l3454_345453


namespace original_number_exists_and_unique_l3454_345423

theorem original_number_exists_and_unique : 
  ∃! x : ℝ, 3 * (2 * x + 9) = 63 := by
  sorry

end original_number_exists_and_unique_l3454_345423


namespace misread_subtraction_l3454_345486

theorem misread_subtraction (x y : Nat) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y = 9 →  -- Two-digit number condition
  10 * x + 6 - 57 = 39 →  -- Misread calculation result
  10 * x + y - 57 = 42    -- Correct calculation result
:= by sorry

end misread_subtraction_l3454_345486


namespace tangent_line_equation_l3454_345432

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Define the point of tangency
def x₀ : ℝ := 2

-- Define the slope of the tangent line
def m : ℝ := 3 * x₀^2 - 2

-- Define the y-intercept of the tangent line
def b : ℝ := f x₀ - m * x₀

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * x + b ↔ y - f x₀ = m * (x - x₀) :=
by sorry

end tangent_line_equation_l3454_345432


namespace intersection_point_satisfies_equations_unique_intersection_point_l3454_345482

/-- The point of intersection of two lines -/
def intersection_point : ℚ × ℚ := (60/23, 50/23)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8*x - 5*y = 10

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 6*x + 2*y = 20

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end intersection_point_satisfies_equations_unique_intersection_point_l3454_345482


namespace cubic_roots_product_l3454_345476

theorem cubic_roots_product (a b c : ℝ) : 
  (x^3 - 15*x^2 + 25*x - 12 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (2 + a) * (2 + b) * (2 + c) = 130 := by
  sorry

end cubic_roots_product_l3454_345476


namespace present_worth_calculation_present_worth_approximation_l3454_345498

/-- Calculates the present worth of an investment given specific interest rates and banker's gain --/
theorem present_worth_calculation (banker_gain : ℝ) : ∃ P : ℝ,
  P * (1.05 * 1.1025 * 1.1255 - 1) = banker_gain :=
by
  sorry

/-- Verifies that the calculated present worth is approximately 114.94 --/
theorem present_worth_approximation (P : ℝ) 
  (h : P * (1.05 * 1.1025 * 1.1255 - 1) = 36) : 
  114.9 < P ∧ P < 115 :=
by
  sorry

end present_worth_calculation_present_worth_approximation_l3454_345498


namespace perpendicular_vectors_x_value_l3454_345468

theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![3, -1]
  let b : Fin 2 → ℝ := ![1, x]
  (∀ i, i < 2 → a i * b i = 0) → x = 3 := by
  sorry

end perpendicular_vectors_x_value_l3454_345468


namespace unknown_number_solution_l3454_345416

theorem unknown_number_solution : 
  ∃ x : ℝ, (4.7 * 13.26 + 4.7 * x + 4.7 * 77.31 = 470) ∧ (abs (x - 9.43) < 0.01) := by
  sorry

end unknown_number_solution_l3454_345416


namespace union_of_A_and_B_l3454_345451

def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x > 1}

theorem union_of_A_and_B : A ∪ B = {x | x > -1} := by sorry

end union_of_A_and_B_l3454_345451


namespace zongzi_sales_theorem_l3454_345478

/-- Represents the sales and profit model for zongzi boxes -/
structure ZongziSales where
  cost : ℝ             -- Cost per box
  min_price : ℝ        -- Minimum selling price
  base_sales : ℝ       -- Base sales at minimum price
  price_sensitivity : ℝ -- Decrease in sales per unit price increase
  max_price : ℝ        -- Maximum allowed selling price
  min_profit : ℝ       -- Minimum desired daily profit

/-- The main theorem about zongzi sales and profit -/
theorem zongzi_sales_theorem (z : ZongziSales)
  (h_cost : z.cost = 40)
  (h_min_price : z.min_price = 45)
  (h_base_sales : z.base_sales = 700)
  (h_price_sensitivity : z.price_sensitivity = 20)
  (h_max_price : z.max_price = 58)
  (h_min_profit : z.min_profit = 6000) :
  (∃ (sales_eq : ℝ → ℝ),
    (∀ x, sales_eq x = -20 * x + 1600) ∧
    (∃ (optimal_price : ℝ) (max_profit : ℝ),
      optimal_price = 60 ∧
      max_profit = 8000 ∧
      (∀ p, z.min_price ≤ p → p ≤ z.max_price →
        (p - z.cost) * (sales_eq p) ≤ max_profit)) ∧
    (∃ (min_boxes : ℝ),
      min_boxes = 440 ∧
      (z.max_price - z.cost) * min_boxes ≥ z.min_profit)) :=
sorry

end zongzi_sales_theorem_l3454_345478


namespace ellipse_equation_proof_l3454_345411

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  fociAxis : ℝ × ℝ
  eccentricity : ℝ
  passingPoint : ℝ × ℝ

/-- The equation of an ellipse given its properties -/
def ellipseEquation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 45 + y^2 / 36 = 1

/-- Theorem stating that an ellipse with the given properties has the specified equation -/
theorem ellipse_equation_proof (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.fociAxis.2 = 0)
  (h3 : e.eccentricity = Real.sqrt 5 / 5)
  (h4 : e.passingPoint = (-5, 4)) :
  ellipseEquation e = fun x y => x^2 / 45 + y^2 / 36 = 1 :=
by
  sorry

end ellipse_equation_proof_l3454_345411


namespace perfume_price_increase_l3454_345412

theorem perfume_price_increase (x : ℝ) : 
  let original_price : ℝ := 1200
  let increased_price : ℝ := original_price * (1 + x / 100)
  let final_price : ℝ := increased_price * (1 - 15 / 100)
  final_price = original_price - 78 → x = 10 :=
by sorry

end perfume_price_increase_l3454_345412


namespace barbaras_candy_count_l3454_345430

/-- Given Barbara's initial candy count and the number of candies she bought,
    prove that her total candy count is the sum of these two quantities. -/
theorem barbaras_candy_count (initial_candies bought_candies : ℕ) :
  initial_candies = 9 →
  bought_candies = 18 →
  initial_candies + bought_candies = 27 :=
by
  sorry

end barbaras_candy_count_l3454_345430


namespace max_piles_is_30_l3454_345488

/-- Represents a configuration of stone piles -/
structure StonePiles where
  piles : List Nat
  sum_stones : (piles.sum = 660)
  size_constraint : ∀ i j, i < piles.length → j < piles.length → 2 * piles[i]! > piles[j]!

/-- Represents a valid split operation on stone piles -/
def split (sp : StonePiles) (index : Nat) (amount : Nat) : Option StonePiles :=
  sorry

/-- The maximum number of piles that can be formed -/
def max_piles : Nat := 30

/-- Theorem stating that the maximum number of piles is 30 -/
theorem max_piles_is_30 :
  ∀ sp : StonePiles,
  (∀ index amount, split sp index amount = none) →
  sp.piles.length ≤ max_piles :=
sorry

end max_piles_is_30_l3454_345488


namespace circle_chords_count_l3454_345493

/-- The number of combinations of n items taken k at a time -/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The number of points on the circle -/
def num_points : ℕ := 10

/-- The number of points needed to form a chord -/
def points_per_chord : ℕ := 2

theorem circle_chords_count :
  choose num_points points_per_chord = 45 := by sorry

end circle_chords_count_l3454_345493


namespace max_take_home_pay_l3454_345403

/-- Represents the income in thousands of dollars -/
def income (x : ℝ) : ℝ := x + 10

/-- Represents the tax rate as a percentage -/
def taxRate (x : ℝ) : ℝ := x

/-- Calculates the take-home pay given the income parameter x -/
def takeHomePay (x : ℝ) : ℝ := 30250 - 10 * (x - 45)^2

/-- Theorem stating that the income yielding the maximum take-home pay is $55,000 -/
theorem max_take_home_pay :
  ∃ (x : ℝ), (∀ (y : ℝ), takeHomePay y ≤ takeHomePay x) ∧ income x = 55 := by
  sorry

end max_take_home_pay_l3454_345403


namespace sum_of_divisors_143_l3454_345454

theorem sum_of_divisors_143 : (Finset.filter (· ∣ 143) (Finset.range 144)).sum id = 168 := by
  sorry

end sum_of_divisors_143_l3454_345454


namespace fifth_term_of_special_sequence_l3454_345442

/-- A sequence where each term after the first is 1/4 of the sum of the term before it and the term after it -/
def SpecialSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = (1 : ℚ) / 4 * (a n + a (n + 2))

theorem fifth_term_of_special_sequence
  (a : ℕ → ℚ)
  (h_seq : SpecialSequence a)
  (h_first : a 1 = 2)
  (h_fourth : a 4 = 50) :
  a 5 = 2798 / 15 := by
sorry

end fifth_term_of_special_sequence_l3454_345442


namespace fraction_simplification_l3454_345443

theorem fraction_simplification (a : ℝ) (h : a ≠ 2) :
  (3 - a) / (a - 2) + 1 = 1 / (a - 2) :=
by sorry

end fraction_simplification_l3454_345443


namespace intersection_of_A_and_B_l3454_345449

-- Define set A
def A : Set ℝ := {x | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l3454_345449


namespace product_mod_seventeen_l3454_345490

theorem product_mod_seventeen : (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 0 := by
  sorry

end product_mod_seventeen_l3454_345490


namespace barefoot_kids_l3454_345470

theorem barefoot_kids (total : ℕ) (with_socks : ℕ) (with_shoes : ℕ) (with_both : ℕ) : 
  total = 22 →
  with_socks = 12 →
  with_shoes = 8 →
  with_both = 6 →
  total - (with_socks + with_shoes - with_both) = 8 :=
by sorry

end barefoot_kids_l3454_345470


namespace julie_initial_savings_l3454_345479

/-- The amount of money Julie saved initially before doing jobs to buy a mountain bike. -/
def initial_savings : ℕ := sorry

/-- The cost of the mountain bike Julie wants to buy. -/
def bike_cost : ℕ := 2345

/-- The number of lawns Julie plans to mow. -/
def lawns_to_mow : ℕ := 20

/-- The payment Julie receives for mowing each lawn. -/
def payment_per_lawn : ℕ := 20

/-- The number of newspapers Julie plans to deliver. -/
def newspapers_to_deliver : ℕ := 600

/-- The payment Julie receives for delivering each newspaper (in cents). -/
def payment_per_newspaper : ℕ := 40

/-- The number of dogs Julie plans to walk. -/
def dogs_to_walk : ℕ := 24

/-- The payment Julie receives for walking each dog. -/
def payment_per_dog : ℕ := 15

/-- The amount of money Julie has left after purchasing the bike. -/
def money_left : ℕ := 155

/-- Theorem stating that Julie's initial savings were $1190. -/
theorem julie_initial_savings :
  initial_savings = 1190 :=
by sorry

end julie_initial_savings_l3454_345479


namespace square_area_proof_l3454_345436

theorem square_area_proof (side_length : ℝ) (rectangle_perimeter : ℝ) : 
  side_length = 8 →
  rectangle_perimeter = 20 →
  (side_length * side_length) = 64 := by
  sorry

end square_area_proof_l3454_345436


namespace alpha_more_advantageous_regular_l3454_345489

/-- Represents a fitness club with a monthly fee -/
structure FitnessClub where
  name : String
  monthlyFee : ℕ

/-- Calculates the number of visits in a year for regular attendance -/
def regularAttendanceVisits : ℕ := 96

/-- Calculates the yearly cost for a fitness club -/
def yearlyCost (club : FitnessClub) : ℕ := club.monthlyFee * 12

/-- Calculates the cost per visit for regular attendance -/
def costPerVisitRegular (club : FitnessClub) : ℚ :=
  (yearlyCost club : ℚ) / regularAttendanceVisits

/-- Alpha and Beta fitness clubs -/
def alpha : FitnessClub := ⟨"Alpha", 999⟩
def beta : FitnessClub := ⟨"Beta", 1299⟩

/-- Theorem stating that Alpha is more advantageous for regular attendance -/
theorem alpha_more_advantageous_regular : 
  costPerVisitRegular alpha < costPerVisitRegular beta := by
  sorry

end alpha_more_advantageous_regular_l3454_345489


namespace quadratic_equation_m_value_l3454_345492

theorem quadratic_equation_m_value (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 9 = (x + 3)^2) → m = 6 := by
sorry

end quadratic_equation_m_value_l3454_345492


namespace min_cars_correct_l3454_345413

/-- Represents the minimum number of cars needed for a given number of adults -/
def min_cars (adults : ℕ) : ℕ :=
  if adults ≤ 5 then 6 else 10

/-- Each car must rest one day a week -/
axiom car_rest_day : ∀ (c : ℕ), c > 0 → ∃ (d : ℕ), d ≤ 7 ∧ c % 7 = d

/-- All adults wish to drive daily -/
axiom adults_drive_daily : ∀ (a : ℕ), a > 0 → ∀ (d : ℕ), d ≤ 7 → ∃ (c : ℕ), c > 0

theorem min_cars_correct (adults : ℕ) (h : adults > 0) :
  ∀ (cars : ℕ), cars < min_cars adults →
    ∃ (d : ℕ), d ≤ 7 ∧ cars - (cars / 7) < adults :=
by sorry

#check min_cars_correct

end min_cars_correct_l3454_345413


namespace trigonometric_product_l3454_345458

theorem trigonometric_product (α : Real) (h : Real.tan α = -2) : 
  Real.sin (π/2 + α) * Real.cos (π + α) = -1/5 := by
  sorry

end trigonometric_product_l3454_345458


namespace intensity_for_three_breaks_l3454_345467

/-- Represents the relationship between breaks and intensity -/
def inverse_proportional (breaks intensity : ℝ) (k : ℝ) : Prop :=
  breaks * intensity = k

theorem intensity_for_three_breaks 
  (k : ℝ) 
  (h1 : inverse_proportional 4 6 k) 
  (h2 : inverse_proportional 3 8 k) : 
  True :=
sorry

end intensity_for_three_breaks_l3454_345467


namespace only_D_opposite_sign_l3454_345427

-- Define the pairs of numbers
def pair_A : ℤ × ℤ := (-(-1), 1)
def pair_B : ℤ × ℤ := ((-1)^2, 1)
def pair_C : ℤ × ℤ := (|(-1)|, 1)
def pair_D : ℤ × ℤ := (-1, 1)

-- Define a function to check if two numbers are opposite in sign
def opposite_sign (a b : ℤ) : Prop := a * b < 0

-- Theorem stating that only pair D contains numbers with opposite signs
theorem only_D_opposite_sign :
  ¬(opposite_sign pair_A.1 pair_A.2) ∧
  ¬(opposite_sign pair_B.1 pair_B.2) ∧
  ¬(opposite_sign pair_C.1 pair_C.2) ∧
  (opposite_sign pair_D.1 pair_D.2) :=
sorry

end only_D_opposite_sign_l3454_345427


namespace tangent_line_at_one_l3454_345446

/-- The function f(x) = x^2(x-2) + 1 -/
def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 2*x*(x - 2) + x^2

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - 1 = 0) :=
sorry

end tangent_line_at_one_l3454_345446


namespace log_sum_fifty_twenty_l3454_345409

theorem log_sum_fifty_twenty : Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 = 3 := by
  sorry

end log_sum_fifty_twenty_l3454_345409


namespace train_length_calculation_l3454_345404

/-- The length of a train given its speed, the speed of a trolley moving in the opposite direction, and the time it takes for the train to pass the trolley. -/
theorem train_length_calculation (train_speed : ℝ) (trolley_speed : ℝ) (passing_time : ℝ) : 
  train_speed = 60 →
  trolley_speed = 12 →
  passing_time = 5.4995600351971845 →
  ∃ (train_length : ℝ), abs (train_length - 109.99) < 0.01 := by
  sorry

end train_length_calculation_l3454_345404


namespace product_as_sum_of_squares_l3454_345426

theorem product_as_sum_of_squares : 85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 := by
  sorry

end product_as_sum_of_squares_l3454_345426


namespace linear_function_straight_line_l3454_345429

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define the property of having a straight line graph
def HasStraightLineGraph (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x y : ℝ, f y - f x = a * (y - x)

-- Define our specific function
def f (x : ℝ) : ℝ := 2 * x + 5

-- State the theorem
theorem linear_function_straight_line :
  (∀ g : ℝ → ℝ, LinearFunction g → HasStraightLineGraph g) →
  LinearFunction f →
  HasStraightLineGraph f :=
by
  sorry

end linear_function_straight_line_l3454_345429


namespace relationship_abc_l3454_345462

theorem relationship_abc (a b c : ℝ) : 
  a = (2/5)^(2/5) → 
  b = (3/5)^(2/5) → 
  c = Real.log (2/5) / Real.log (3/5) → 
  a < b ∧ b < c := by sorry

end relationship_abc_l3454_345462


namespace vector_to_line_parallel_l3454_345460

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if a point lies on a parameterized line -/
def pointOnLine (p : ℝ × ℝ) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = p.1 ∧ l.y t = p.2

/-- Check if two vectors are parallel -/
def vectorsParallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_to_line_parallel (l : ParametricLine) (v : ℝ × ℝ) :
  l.x t = 5 * t + 3 ∧ l.y t = 2 * t - 1 →
  pointOnLine v l ∧ vectorsParallel v (5, 2) →
  v = (-2.5, -1) :=
sorry

end vector_to_line_parallel_l3454_345460


namespace prime_sum_divisibility_l3454_345437

theorem prime_sum_divisibility (p : ℕ) : 
  Prime p → (7^p - 6^p + 2) % 43 = 0 → p = 3 := by
  sorry

end prime_sum_divisibility_l3454_345437


namespace college_graduates_scientific_notation_l3454_345452

theorem college_graduates_scientific_notation :
  ∃ (x : ℝ) (n : ℤ), 
    x ≥ 1 ∧ x < 10 ∧ 
    116000000 = x * (10 : ℝ) ^ n ∧
    x = 1.16 ∧ n = 7 :=
by sorry

end college_graduates_scientific_notation_l3454_345452


namespace v_shaped_to_log_v_shaped_l3454_345405

/-- Definition of a V-shaped function -/
def is_v_shaped (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) ≤ f x₁ + f x₂

/-- Definition of a Logarithmic V-shaped function -/
def is_log_v_shaped (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f x > 0) ∧
  (∀ x₁ x₂ : ℝ, Real.log (f (x₁ + x₂)) < Real.log (f x₁) + Real.log (f x₂))

/-- Theorem: If f is V-shaped and f(x) ≥ 2 for all x, then f is Logarithmic V-shaped -/
theorem v_shaped_to_log_v_shaped (f : ℝ → ℝ) 
    (hv : is_v_shaped f) (hf : ∀ x : ℝ, f x ≥ 2) : 
    is_log_v_shaped f := by
  sorry

end v_shaped_to_log_v_shaped_l3454_345405


namespace correct_team_selection_l3454_345485

def group_A_nurses : ℕ := 4
def group_A_doctors : ℕ := 1
def group_B_nurses : ℕ := 6
def group_B_doctors : ℕ := 2
def members_per_group : ℕ := 2
def total_members : ℕ := 4
def required_doctors : ℕ := 1

def select_team : ℕ := sorry

theorem correct_team_selection :
  select_team = 132 := by sorry

end correct_team_selection_l3454_345485


namespace coordinate_sum_of_h_l3454_345414

theorem coordinate_sum_of_h (g : ℝ → ℝ) (h : ℝ → ℝ) : 
  g 4 = 8 → 
  (∀ x, h x = (g x)^2) → 
  4 + h 4 = 68 := by
sorry

end coordinate_sum_of_h_l3454_345414


namespace multiples_of_six_or_eight_l3454_345481

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_of_six_or_eight (upper_bound : ℕ) (h : upper_bound = 151) : 
  (count_multiples upper_bound 6 + count_multiples upper_bound 8 - 2 * count_multiples upper_bound 24) = 31 := by
  sorry

end multiples_of_six_or_eight_l3454_345481


namespace complex_equation_solution_l3454_345497

theorem complex_equation_solution :
  ∃ (z : ℂ), 3 - 2 * Complex.I * z = 5 + 3 * Complex.I * z ∧ z = (2 * Complex.I) / 5 := by
  sorry

end complex_equation_solution_l3454_345497


namespace opposite_of_neg_six_l3454_345407

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- Theorem: The opposite of -6 is 6. -/
theorem opposite_of_neg_six : opposite (-6) = 6 := by
  sorry

end opposite_of_neg_six_l3454_345407


namespace divisible_by_72_implies_a3_b2_l3454_345448

/-- Represents a five-digit number in the form a679b -/
def five_digit_number (a b : ℕ) : ℕ := a * 10000 + 6790 + b

/-- Checks if a natural number is divisible by another natural number -/
def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem divisible_by_72_implies_a3_b2 :
  ∀ a b : ℕ, 
    a < 10 → b < 10 →
    is_divisible_by (five_digit_number a b) 72 →
    a = 3 ∧ b = 2 := by
  sorry

end divisible_by_72_implies_a3_b2_l3454_345448


namespace tomato_count_l3454_345402

theorem tomato_count (plant1 plant2 plant3 plant4 : ℕ) : 
  plant1 = 8 →
  plant2 = plant1 + 4 →
  plant3 = 3 * (plant1 + plant2) →
  plant4 = plant3 →
  plant1 + plant2 + plant3 + plant4 = 140 :=
by
  sorry

end tomato_count_l3454_345402


namespace polynomial_root_problem_l3454_345461

theorem polynomial_root_problem (a b : ℝ) : 
  (∀ x : ℝ, a*x^4 + (a + b)*x^3 + (b - 2*a)*x^2 + 5*b*x + (12 - a) = 0 ↔ 
    x = 1 ∨ x = -3 ∨ x = 4 ∨ x = -92/297) :=
by sorry

end polynomial_root_problem_l3454_345461


namespace stratified_sampling_properties_l3454_345421

structure School where
  first_year_students : ℕ
  second_year_students : ℕ

def stratified_sample (school : School) (sample_size : ℕ) : 
  ℕ × ℕ :=
  let total_students := school.first_year_students + school.second_year_students
  let first_year_sample := (school.first_year_students * sample_size) / total_students
  let second_year_sample := sample_size - first_year_sample
  (first_year_sample, second_year_sample)

theorem stratified_sampling_properties 
  (school : School)
  (sample_size : ℕ)
  (h1 : school.first_year_students = 1000)
  (h2 : school.second_year_students = 1080)
  (h3 : sample_size = 208) :
  let (first_sample, second_sample) := stratified_sample school sample_size
  -- 1. Students from different grades can be selected simultaneously
  (first_sample > 0 ∧ second_sample > 0) ∧
  -- 2. The number of students selected from each grade is proportional to the grade's population
  (first_sample = 100 ∧ second_sample = 108) ∧
  -- 3. The probability of selection for any student is equal across both grades
  (first_sample / school.first_year_students = second_sample / school.second_year_students) :=
by
  sorry

end stratified_sampling_properties_l3454_345421


namespace cereal_eating_time_l3454_345494

def fat_rate : ℚ := 1 / 25
def thin_rate : ℚ := 1 / 35
def medium_rate : ℚ := 1 / 28
def total_cereal : ℚ := 5

def combined_rate : ℚ := fat_rate + thin_rate + medium_rate

def time_taken : ℚ := total_cereal / combined_rate

theorem cereal_eating_time : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ time_taken - 48 < ε ∧ 48 - time_taken < ε :=
sorry

end cereal_eating_time_l3454_345494


namespace negative_fraction_comparison_l3454_345464

theorem negative_fraction_comparison : -3/4 > -6/5 := by
  sorry

end negative_fraction_comparison_l3454_345464


namespace square_of_binomial_p_l3454_345425

/-- If 9x^2 + 24x + p is the square of a binomial, then p = 16 -/
theorem square_of_binomial_p (p : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 + 24*x + p = (a*x + b)^2) → p = 16 := by
sorry

end square_of_binomial_p_l3454_345425


namespace conference_season_games_l3454_345438

/-- Calculates the number of games in a complete season for a sports conference. -/
def games_in_season (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let divisions := total_teams / teams_per_division
  let intra_division_total := divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams / 2) * teams_per_division * inter_division_games
  intra_division_total + inter_division_total

/-- Theorem stating the number of games in a complete season for the given conference structure. -/
theorem conference_season_games : 
  games_in_season 14 7 3 1 = 175 := by
  sorry

end conference_season_games_l3454_345438


namespace least_n_for_adjacent_probability_l3454_345496

def adjacent_probability (n : ℕ) : ℚ :=
  (4 * n^2 - 4 * n + 8) / (n^2 * (n^2 - 1))

theorem least_n_for_adjacent_probability : 
  (∀ k < 90, adjacent_probability k ≥ 1 / 2015) ∧ 
  adjacent_probability 90 < 1 / 2015 :=
sorry

end least_n_for_adjacent_probability_l3454_345496


namespace swimmer_speed_in_still_water_l3454_345484

/-- Represents the speed of a swimmer in various conditions -/
structure SwimmerSpeed where
  downstream : ℝ
  upstream : ℝ
  stillWater : ℝ

/-- Theorem stating that given the downstream and upstream speeds, 
    we can determine the speed in still water -/
theorem swimmer_speed_in_still_water 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_distance : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 72) 
  (h2 : downstream_time = 4) 
  (h3 : upstream_distance = 36) 
  (h4 : upstream_time = 6) :
  ∃ (s : SwimmerSpeed), 
    s.downstream = downstream_distance / downstream_time ∧
    s.upstream = upstream_distance / upstream_time ∧
    s.stillWater = 12 := by
  sorry

#check swimmer_speed_in_still_water

end swimmer_speed_in_still_water_l3454_345484


namespace peanuts_problem_l3454_345431

/-- The number of peanuts remaining in the jar after a series of distributions and consumptions -/
def peanuts_remaining (initial : ℕ) : ℕ :=
  let brock_ate := initial / 3
  let after_brock := initial - brock_ate
  let per_family := after_brock / 3
  let bonita_per_family := (2 * per_family) / 5
  let after_bonita_per_family := per_family - bonita_per_family
  let after_bonita_total := after_bonita_per_family * 3
  let carlos_ate := after_bonita_total / 5
  after_bonita_total - carlos_ate

/-- Theorem stating that given the initial conditions, 216 peanuts remain in the jar -/
theorem peanuts_problem : peanuts_remaining 675 = 216 := by
  sorry

end peanuts_problem_l3454_345431


namespace remainder_of_S_mod_512_l3454_345424

def R : Finset ℕ := Finset.image (λ n => (3^n) % 512) (Finset.range 12)

def S : ℕ := Finset.sum R id

theorem remainder_of_S_mod_512 : S % 512 = 72 := by sorry

end remainder_of_S_mod_512_l3454_345424


namespace quadratic_root_ratio_l3454_345459

theorem quadratic_root_ratio (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ∃ (x y : ℝ), x = 2022 * y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :
  (2023 * a * c) / (b^2) = 2022 / 2023 := by
sorry

end quadratic_root_ratio_l3454_345459
