import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_fractions_l172_17299

theorem sum_of_fractions : (2 : ℚ) / 20 + (4 : ℚ) / 40 + (5 : ℚ) / 50 = (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l172_17299


namespace NUMINAMATH_CALUDE_no_unique_solution_l172_17249

/-- For a system of two linear equations to have no unique solution, 
    the ratios of coefficients and constants must be equal. -/
theorem no_unique_solution (k e : ℝ) : 
  (∃ k, ¬∃! (x y : ℝ), 4 * (3 * x + 4 * y) = 48 ∧ k * x + e * y = 30) →
  e = 10 := by
sorry

end NUMINAMATH_CALUDE_no_unique_solution_l172_17249


namespace NUMINAMATH_CALUDE_primitive_root_modulo_p_alpha_implies_modulo_p_l172_17250

theorem primitive_root_modulo_p_alpha_implies_modulo_p
  (p : Nat) (α : Nat) (x : Nat)
  (h_prime : Nat.Prime p)
  (h_pos : α > 0)
  (h_primitive_p_alpha : IsPrimitiveRoot x (p ^ α)) :
  IsPrimitiveRoot x p :=
sorry

end NUMINAMATH_CALUDE_primitive_root_modulo_p_alpha_implies_modulo_p_l172_17250


namespace NUMINAMATH_CALUDE_item_prices_l172_17286

theorem item_prices (x y z : ℝ) 
  (eq1 : 3 * x + 5 * y + z = 32) 
  (eq2 : 4 * x + 7 * y + z = 40) : 
  x + y + z = 16 := by
  sorry

end NUMINAMATH_CALUDE_item_prices_l172_17286


namespace NUMINAMATH_CALUDE_complement_of_union_l172_17235

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {2, 3, 5}

-- Define set N
def N : Set Nat := {4, 5}

-- Theorem statement
theorem complement_of_union (h : Set Nat → Set Nat → Set Nat) :
  h M N = {1, 6} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_union_l172_17235


namespace NUMINAMATH_CALUDE_brick_wall_bottom_row_l172_17270

/-- Represents a brick wall with a decreasing number of bricks per row from bottom to top -/
structure BrickWall where
  numRows : Nat
  totalBricks : Nat
  bottomRowBricks : Nat

/-- Calculates the total number of bricks in the wall given the number of bricks in the bottom row -/
def sumBricks (n : Nat) : Nat :=
  List.range n |> List.map (fun i => n - i) |> List.sum

/-- Theorem: A brick wall with 5 rows and 50 total bricks, where each row above the bottom
    has one less brick than the row below, has 12 bricks in the bottom row -/
theorem brick_wall_bottom_row : 
  ∀ (wall : BrickWall), 
    wall.numRows = 5 → 
    wall.totalBricks = 50 → 
    (sumBricks wall.bottomRowBricks = wall.totalBricks) → 
    wall.bottomRowBricks = 12 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_bottom_row_l172_17270


namespace NUMINAMATH_CALUDE_election_votes_l172_17239

theorem election_votes (total_votes : ℕ) (winner_votes loser_votes : ℕ) : 
  winner_votes = (56 : ℕ) * total_votes / 100 →
  loser_votes = total_votes - winner_votes →
  winner_votes - loser_votes = 288 →
  winner_votes = 1344 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l172_17239


namespace NUMINAMATH_CALUDE_garrison_size_l172_17291

/-- The initial number of days the provisions would last -/
def initial_days : ℕ := 28

/-- The number of days that passed before reinforcements arrived -/
def days_before_reinforcement : ℕ := 12

/-- The number of men that arrived as reinforcement -/
def reinforcement : ℕ := 1110

/-- The number of days the provisions would last after reinforcement arrived -/
def remaining_days : ℕ := 10

/-- The initial number of men in the garrison -/
def initial_men : ℕ := 1850

theorem garrison_size :
  ∃ (M : ℕ),
    M * initial_days = 
    (M + reinforcement) * remaining_days + 
    M * days_before_reinforcement ∧
    M = initial_men :=
by sorry

end NUMINAMATH_CALUDE_garrison_size_l172_17291


namespace NUMINAMATH_CALUDE_good_permutations_congruence_l172_17259

/-- Given a prime number p > 3, count_good_permutations p returns the number of permutations
    (a₁, a₂, ..., aₚ₋₁) of (1, 2, ..., p-1) such that p divides the sum of consecutive products. -/
def count_good_permutations (p : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the number of good permutations is congruent to p-1 modulo p(p-1). -/
theorem good_permutations_congruence (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  count_good_permutations p ≡ p - 1 [MOD p * (p - 1)] :=
sorry

end NUMINAMATH_CALUDE_good_permutations_congruence_l172_17259


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l172_17248

theorem sum_of_roots_equals_fourteen : 
  let f : ℝ → ℝ := λ x => (x - 7)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l172_17248


namespace NUMINAMATH_CALUDE_inequality_range_l172_17296

theorem inequality_range (a b x : ℝ) (ha : a ≠ 0) :
  (|2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|)) → x ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l172_17296


namespace NUMINAMATH_CALUDE_inequality_proof_l172_17284

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (1 - 2*x) / Real.sqrt (x * (1 - x)) + 
  (1 - 2*y) / Real.sqrt (y * (1 - y)) + 
  (1 - 2*z) / Real.sqrt (z * (1 - z)) ≥ 
  Real.sqrt (x / (1 - x)) + 
  Real.sqrt (y / (1 - y)) + 
  Real.sqrt (z / (1 - z)) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l172_17284


namespace NUMINAMATH_CALUDE_kaleb_final_amount_l172_17297

def kaleb_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

theorem kaleb_final_amount :
  kaleb_business 4 50 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_final_amount_l172_17297


namespace NUMINAMATH_CALUDE_range_of_m_l172_17227

def has_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c ≥ 0

def p (m : ℝ) : Prop :=
  has_real_roots 1 m 1

def q (m : ℝ) : Prop :=
  ¬(has_real_roots 4 (4*(m-2)) 1)

def exactly_one_true (p q : Prop) : Prop :=
  (p ∧ ¬q) ∨ (¬p ∧ q)

theorem range_of_m : 
  {m : ℝ | exactly_one_true (p m) (q m)} = 
  {m : ℝ | m ≤ -2 ∨ (1 < m ∧ m < 2) ∨ m ≥ 3} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l172_17227


namespace NUMINAMATH_CALUDE_coordinates_of_B_l172_17219

-- Define the square OABC
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 3)

-- Define the property that C is in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Define the property of being a square
def is_square (O A B C : ℝ × ℝ) : Prop :=
  let d₁ := (A.1 - O.1)^2 + (A.2 - O.2)^2
  let d₂ := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let d₃ := (C.1 - B.1)^2 + (C.2 - B.2)^2
  let d₄ := (O.1 - C.1)^2 + (O.2 - C.2)^2
  d₁ = d₂ ∧ d₂ = d₃ ∧ d₃ = d₄

-- Theorem statement
theorem coordinates_of_B :
  ∃ (B C : ℝ × ℝ), is_square O A B C ∧ in_fourth_quadrant C → B = (7, -1) :=
sorry

end NUMINAMATH_CALUDE_coordinates_of_B_l172_17219


namespace NUMINAMATH_CALUDE_other_items_tax_is_ten_percent_l172_17278

/-- Represents the tax rates and spending percentages in Jill's shopping trip -/
structure ShoppingTax where
  clothing_spend : Rat
  food_spend : Rat
  other_spend : Rat
  clothing_tax : Rat
  food_tax : Rat
  total_tax : Rat

/-- The tax rate on other items given the shopping tax structure -/
def other_items_tax_rate (st : ShoppingTax) : Rat :=
  (st.total_tax - st.clothing_tax * st.clothing_spend) / st.other_spend

/-- Theorem stating that the tax rate on other items is 10% -/
theorem other_items_tax_is_ten_percent (st : ShoppingTax) 
  (h1 : st.clothing_spend = 1/2)
  (h2 : st.food_spend = 1/5)
  (h3 : st.other_spend = 3/10)
  (h4 : st.clothing_tax = 1/20)
  (h5 : st.food_tax = 0)
  (h6 : st.total_tax = 11/200) :
  other_items_tax_rate st = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_other_items_tax_is_ten_percent_l172_17278


namespace NUMINAMATH_CALUDE_trigonometric_equality_l172_17230

theorem trigonometric_equality (x y z a : ℝ) 
  (h1 : (Real.sin x + Real.sin y + Real.sin z) / Real.sin (x + y + z) = a)
  (h2 : (Real.cos x + Real.cos y + Real.cos z) / Real.cos (x + y + z) = a) :
  Real.cos (x + y) + Real.cos (y + z) + Real.cos (z + x) = a := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l172_17230


namespace NUMINAMATH_CALUDE_f_of_one_eq_two_l172_17225

def f (x : ℝ) := 3 * x - 1

theorem f_of_one_eq_two : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_eq_two_l172_17225


namespace NUMINAMATH_CALUDE_chords_intersection_theorem_l172_17298

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a chord
structure Chord (c : Circle) where
  p1 : Point
  p2 : Point

-- Define the length of a segment
def segmentLength (p1 p2 : Point) : ℝ := sorry

-- Define a right angle
def isRightAngle (p1 p2 p3 : Point) : Prop := sorry

-- Theorem statement
theorem chords_intersection_theorem (c : Circle) (ab cd : Chord c) (e : Point) :
  isRightAngle ab.p1 e cd.p1 →
  (segmentLength ab.p1 e)^2 + (segmentLength ab.p2 e)^2 + 
  (segmentLength cd.p1 e)^2 + (segmentLength cd.p2 e)^2 = 
  (2 * c.radius)^2 := by
  sorry

end NUMINAMATH_CALUDE_chords_intersection_theorem_l172_17298


namespace NUMINAMATH_CALUDE_exists_m_divides_f_100_l172_17207

def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_m_divides_f_100 :
  ∃ m : ℕ+, 19881 ∣ (3^100 * (m.val + 1) - 1) :=
sorry

end NUMINAMATH_CALUDE_exists_m_divides_f_100_l172_17207


namespace NUMINAMATH_CALUDE_integral_of_even_function_l172_17260

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem -/
theorem integral_of_even_function (a : ℝ) :
  let f := fun x => a * x^2 + (a - 2) * x + a^2
  IsEven f →
  ∫ x in (-a)..a, (x^2 + x + Real.sqrt (4 - x^2)) = 16/3 + 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_integral_of_even_function_l172_17260


namespace NUMINAMATH_CALUDE_find_a_l172_17293

-- Define the solution set
def solutionSet (x : ℝ) : Prop :=
  (-3 < x ∧ x < -1) ∨ x > 2

-- Define the inequality
def inequality (a x : ℝ) : Prop :=
  (x + a) / (x^2 + 4*x + 3) > 0

theorem find_a :
  (∃ a : ℝ, ∀ x : ℝ, inequality a x ↔ solutionSet x) →
  (∃ a : ℝ, a = -2 ∧ ∀ x : ℝ, inequality a x ↔ solutionSet x) :=
by sorry

end NUMINAMATH_CALUDE_find_a_l172_17293


namespace NUMINAMATH_CALUDE_festival_remaining_money_l172_17210

def festival_spending (total_budget food_cost : ℕ) : ℕ :=
  let ride_cost := 3 * food_cost
  let game_cost := ride_cost / 2
  total_budget - (food_cost + ride_cost + game_cost)

theorem festival_remaining_money :
  festival_spending 100 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_festival_remaining_money_l172_17210


namespace NUMINAMATH_CALUDE_circle_op_calculation_l172_17271

-- Define the ⊗ operation
def circle_op (a b : ℚ) : ℚ := (a^2 + b^2) / (a + b)

-- State the theorem
theorem circle_op_calculation : circle_op (circle_op 5 2) 4 = 11375 / 2793 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_calculation_l172_17271


namespace NUMINAMATH_CALUDE_tickets_left_l172_17229

/-- Given that Paul bought eleven tickets and spent three tickets,
    prove that he has eight tickets left. -/
theorem tickets_left (total : ℕ) (spent : ℕ) (left : ℕ) 
    (h1 : total = 11)
    (h2 : spent = 3)
    (h3 : left = total - spent) : left = 8 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l172_17229


namespace NUMINAMATH_CALUDE_absolute_value_inequality_supremum_l172_17254

theorem absolute_value_inequality_supremum :
  (∀ k : ℝ, (∀ x : ℝ, |x + 3| + |x - 1| > k) → k < 4) ∧
  ∀ ε > 0, ∃ x : ℝ, |x + 3| + |x - 1| < 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_supremum_l172_17254


namespace NUMINAMATH_CALUDE_problem_statement_l172_17231

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49/((x - 3)^2) = 23 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l172_17231


namespace NUMINAMATH_CALUDE_wall_painting_fraction_l172_17267

theorem wall_painting_fraction (paint_rate : ℝ) (total_time minutes : ℝ) 
  (h1 : paint_rate * total_time = 1)  -- Can paint whole wall in total_time
  (h2 : minutes / total_time = 1 / 5) -- Minutes is 1/5 of total time
  : paint_rate * minutes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wall_painting_fraction_l172_17267


namespace NUMINAMATH_CALUDE_least_tiles_required_l172_17203

def room_length : Real := 8.16
def room_width : Real := 4.32
def recess_width : Real := 1.24
def recess_length : Real := 2
def protrusion_width : Real := 0.48
def protrusion_length : Real := 0.96

def main_area : Real := room_length * room_width
def recess_area : Real := recess_width * recess_length
def protrusion_area : Real := protrusion_width * protrusion_length
def total_area : Real := main_area + recess_area + protrusion_area

def tile_size : Real := protrusion_width

theorem least_tiles_required :
  ∃ n : ℕ, n = ⌈total_area / (tile_size * tile_size)⌉ ∧ n = 166 := by
  sorry

end NUMINAMATH_CALUDE_least_tiles_required_l172_17203


namespace NUMINAMATH_CALUDE_remainder_theorem_l172_17274

theorem remainder_theorem : (7 * 10^20 + 2^20) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l172_17274


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l172_17277

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope 1 and y-intercept m -/
structure Line where
  m : ℝ

/-- Represents the intersection of an ellipse and a line -/
def Intersection (e : Ellipse) (l : Line) :=
  {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ p.2 = p.1 + l.m}

/-- Theorem stating the range of m for which the line intersects the ellipse at two distinct points forming an acute angle at the origin -/
theorem ellipse_line_intersection_range (e : Ellipse) (l : Line) 
  (h_minor : e.b = 1)
  (h_ecc : Real.sqrt (1 - e.b^2 / e.a^2) = Real.sqrt 3 / 2)
  (h_intersect : ∃ A B, A ∈ Intersection e l ∧ B ∈ Intersection e l ∧ A ≠ B)
  (h_acute : ∃ A B, A ∈ Intersection e l ∧ B ∈ Intersection e l ∧ A ≠ B ∧ 
    0 < Real.arccos ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2))) ∧
    Real.arccos ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2))) < π / 2) :
  (-Real.sqrt 5 < l.m ∧ l.m < -2 * Real.sqrt 10 / 5) ∨ (2 * Real.sqrt 10 / 5 < l.m ∧ l.m < Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l172_17277


namespace NUMINAMATH_CALUDE_sum_of_digits_R50_div_R8_l172_17202

def R (k : ℕ) : ℕ := (10^k - 1) / 9

theorem sum_of_digits_R50_div_R8 : ∃ (q : ℕ), R 50 = q * R 8 ∧ (q.digits 10).sum = 6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_R50_div_R8_l172_17202


namespace NUMINAMATH_CALUDE_gcd_2134_155_in_ternary_is_100_l172_17234

def gcd_2134_155_in_ternary : List Nat :=
  let m := Nat.gcd 2134 155
  Nat.digits 3 m

theorem gcd_2134_155_in_ternary_is_100 : 
  gcd_2134_155_in_ternary = [1, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_gcd_2134_155_in_ternary_is_100_l172_17234


namespace NUMINAMATH_CALUDE_y2_greater_than_y1_l172_17208

/-- The parabola equation y = x² - 2x + 3 -/
def parabola (x y : ℝ) : Prop := y = x^2 - 2*x + 3

theorem y2_greater_than_y1 (y1 y2 : ℝ) 
  (h1 : parabola (-1) y1)
  (h2 : parabola (-2) y2) : 
  y2 > y1 := by
  sorry

end NUMINAMATH_CALUDE_y2_greater_than_y1_l172_17208


namespace NUMINAMATH_CALUDE_equilateral_triangle_sticks_l172_17237

def canFormEquilateralTriangle (n : ℕ) : Prop :=
  ∃ (side : ℕ), 3 * side = n * (n + 1) / 2

theorem equilateral_triangle_sticks (n : ℕ) :
  canFormEquilateralTriangle n ↔ n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sticks_l172_17237


namespace NUMINAMATH_CALUDE_bus_problem_l172_17256

theorem bus_problem (initial_children on_bus off_bus final_children : ℕ) :
  initial_children = 22 →
  on_bus = 40 →
  final_children = 2 →
  initial_children + on_bus - off_bus = final_children →
  off_bus = 60 :=
by sorry

end NUMINAMATH_CALUDE_bus_problem_l172_17256


namespace NUMINAMATH_CALUDE_intersection_sum_l172_17251

theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + c) →
  (∀ x y : ℝ, y = 5 * x + d) →
  16 = 2 * 4 + c →
  16 = 5 * 4 + d →
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l172_17251


namespace NUMINAMATH_CALUDE_cos_330_deg_l172_17257

/-- Cosine of 330 degrees is equal to sqrt(3)/2 -/
theorem cos_330_deg : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_deg_l172_17257


namespace NUMINAMATH_CALUDE_speeding_statistics_l172_17282

structure SpeedingCategory where
  name : Char
  percentMotorists : ℝ
  ticketRate : ℝ

def categoryA : SpeedingCategory := ⟨'A', 0.14, 0.25⟩
def categoryB : SpeedingCategory := ⟨'B', 0.07, 0.55⟩
def categoryC : SpeedingCategory := ⟨'C', 0.04, 0.80⟩
def categoryD : SpeedingCategory := ⟨'D', 0.02, 0.95⟩

def categories : List SpeedingCategory := [categoryA, categoryB, categoryC, categoryD]

theorem speeding_statistics :
  (List.sum (categories.map (λ c => c.percentMotorists)) = 0.27) ∧
  (categoryA.percentMotorists * categoryA.ticketRate = 0.035) ∧
  (categoryB.percentMotorists * categoryB.ticketRate = 0.0385) ∧
  (categoryC.percentMotorists * categoryC.ticketRate = 0.032) ∧
  (categoryD.percentMotorists * categoryD.ticketRate = 0.019) :=
by sorry

end NUMINAMATH_CALUDE_speeding_statistics_l172_17282


namespace NUMINAMATH_CALUDE_final_state_theorem_l172_17264

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the state of the box -/
structure BoxState where
  white : Nat
  black : Nat

/-- The initial state of the box -/
def initialState : BoxState :=
  { white := 2015, black := 2015 }

/-- The final state of the box -/
def finalState : BoxState :=
  { white := 2, black := 1 }

/-- Represents one step of the ball selection process -/
def selectBalls (state : BoxState) : BoxState :=
  sorry

/-- Predicate to check if the process should stop -/
def stopCondition (state : BoxState) : Prop :=
  state.white + state.black = 3

/-- Theorem stating that the process will end with 2 white balls and 1 black ball -/
theorem final_state_theorem (state : BoxState) :
  state = initialState →
  (∃ n : Nat, (selectBalls^[n] state) = finalState ∧ stopCondition (selectBalls^[n] state)) :=
sorry

end NUMINAMATH_CALUDE_final_state_theorem_l172_17264


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l172_17292

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l172_17292


namespace NUMINAMATH_CALUDE_equation_solution_l172_17269

theorem equation_solution : 
  ∃ x : ℚ, (24 - 4 = 3 * (1 + x)) ∧ (x = 17 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l172_17269


namespace NUMINAMATH_CALUDE_task_completion_probability_l172_17263

theorem task_completion_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 3/8) (h2 : p2 = 3/5) (h3 : p3 = 5/9) (h4 : p4 = 7/12) :
  p1 * (1 - p2) * p3 * (1 - p4) = 5/72 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l172_17263


namespace NUMINAMATH_CALUDE_watermelon_not_necessarily_split_l172_17226

/-- Represents a spherical watermelon with given diameter and cut depth. -/
structure Watermelon where
  diameter : ℝ
  cut_depth : ℝ

/-- Determines if the watermelon is necessarily split into at least two pieces. -/
def is_necessarily_split (w : Watermelon) : Prop :=
  ∃ (configuration : ℝ → ℝ → ℝ → Prop),
    ∀ (x y z : ℝ),
      configuration x y z →
      (x^2 + y^2 + z^2 ≤ (w.diameter/2)^2) →
      (|x| ≤ w.cut_depth ∨ |y| ≤ w.cut_depth ∨ |z| ≤ w.cut_depth)

/-- Theorem stating that a watermelon with diameter 20 cm is not necessarily split
    for cut depths of 17 cm and 18 cm. -/
theorem watermelon_not_necessarily_split :
  let w₁ : Watermelon := ⟨20, 17⟩
  let w₂ : Watermelon := ⟨20, 18⟩
  ¬(is_necessarily_split w₁) ∧ ¬(is_necessarily_split w₂) := by
  sorry

end NUMINAMATH_CALUDE_watermelon_not_necessarily_split_l172_17226


namespace NUMINAMATH_CALUDE_angle_terminal_side_cosine_l172_17261

theorem angle_terminal_side_cosine (x : ℝ) (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (-x, -6) ∧ P ∈ {p | ∃ t : ℝ, p = (t * Real.cos α, t * Real.sin α)}) →
  Real.cos α = 4/5 →
  x = -8 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_cosine_l172_17261


namespace NUMINAMATH_CALUDE_coat_price_calculation_l172_17206

/-- Calculates the final price of a coat after two discounts and tax --/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (taxRate : ℝ) : ℝ :=
  let priceAfterDiscount1 := originalPrice * (1 - discount1)
  let priceAfterDiscount2 := priceAfterDiscount1 * (1 - discount2)
  priceAfterDiscount2 * (1 + taxRate)

/-- Theorem stating that the final price of the coat is approximately 84.7 --/
theorem coat_price_calculation :
  let originalPrice : ℝ := 120
  let discount1 : ℝ := 0.30
  let discount2 : ℝ := 0.10
  let taxRate : ℝ := 0.12
  abs (finalPrice originalPrice discount1 discount2 taxRate - 84.7) < 0.1 := by
  sorry

#eval finalPrice 120 0.30 0.10 0.12

end NUMINAMATH_CALUDE_coat_price_calculation_l172_17206


namespace NUMINAMATH_CALUDE_amy_candy_difference_l172_17272

/-- Amy's candy problem -/
theorem amy_candy_difference (initial : ℕ) (given_away : ℕ) (left : ℕ) 
  (h1 : given_away = 6)
  (h2 : left = 5)
  (h3 : initial = given_away + left) :
  given_away - left = 1 := by
  sorry

end NUMINAMATH_CALUDE_amy_candy_difference_l172_17272


namespace NUMINAMATH_CALUDE_difference_of_squares_l172_17204

theorem difference_of_squares (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l172_17204


namespace NUMINAMATH_CALUDE_cos_pi_minus_alpha_l172_17213

theorem cos_pi_minus_alpha (α : Real) (h : Real.sin (α / 2) = 2 / 3) : 
  Real.cos (Real.pi - α) = -1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_alpha_l172_17213


namespace NUMINAMATH_CALUDE_baker_pies_sold_l172_17283

theorem baker_pies_sold (cakes : ℕ) (cake_price pie_price total_earnings : ℚ) 
  (h1 : cakes = 453)
  (h2 : cake_price = 12)
  (h3 : pie_price = 7)
  (h4 : total_earnings = 6318) :
  (total_earnings - cakes * cake_price) / pie_price = 126 :=
by sorry

end NUMINAMATH_CALUDE_baker_pies_sold_l172_17283


namespace NUMINAMATH_CALUDE_problem_statement_l172_17238

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1/2) : 
  m = 100 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l172_17238


namespace NUMINAMATH_CALUDE_equation_solution_l172_17218

theorem equation_solution : ∃ x : ℝ, (x - 3)^2 = x^2 - 9 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l172_17218


namespace NUMINAMATH_CALUDE_two_red_one_blue_probability_l172_17228

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8

theorem two_red_one_blue_probability :
  let prob := (red_marbles * (red_marbles - 1) * blue_marbles + 
               red_marbles * blue_marbles * (red_marbles - 1) + 
               blue_marbles * red_marbles * (red_marbles - 1)) / 
              (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  prob = 44 / 95 := by
sorry

end NUMINAMATH_CALUDE_two_red_one_blue_probability_l172_17228


namespace NUMINAMATH_CALUDE_ellipse_equation_l172_17253

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The condition that an ellipse passes through a point -/
def passes_through (e : Ellipse) (p : Point) : Prop :=
  e.equation p.x p.y

/-- The condition that two foci and one endpoint of the minor axis form an isosceles right triangle -/
def isosceles_right_triangle (e : Ellipse) : Prop :=
  e.a = Real.sqrt 2 * e.b

theorem ellipse_equation (e : Ellipse) (p : Point) 
    (h1 : passes_through e p)
    (h2 : p.x = 1 ∧ p.y = Real.sqrt 2 / 2)
    (h3 : isosceles_right_triangle e) :
    ∀ x y : ℝ, e.equation x y ↔ x^2 / 2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l172_17253


namespace NUMINAMATH_CALUDE_train_speed_problem_l172_17216

/-- Given a train journey with two scenarios:
    1. The train covers a distance in 360 minutes at an unknown average speed.
    2. The same distance can be covered in 90 minutes at a speed of 20 kmph.
    This theorem proves that the average speed in the first scenario is 5 kmph. -/
theorem train_speed_problem (distance : ℝ) (avg_speed : ℝ) : 
  distance = 20 * (90 / 60) → -- The distance is covered in 90 minutes at 20 kmph
  distance = avg_speed * (360 / 60) → -- The same distance is covered in 360 minutes at avg_speed
  avg_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l172_17216


namespace NUMINAMATH_CALUDE_unique_prime_solution_l172_17258

theorem unique_prime_solution :
  ∀ p q r : ℕ,
  Prime p → Prime q → Prime r →
  p + q^2 = r^4 →
  p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l172_17258


namespace NUMINAMATH_CALUDE_order_of_expressions_l172_17224

theorem order_of_expressions : 7^(0.3 : ℝ) > (0.3 : ℝ)^7 ∧ (0.3 : ℝ)^7 > Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l172_17224


namespace NUMINAMATH_CALUDE_allens_mother_age_l172_17222

theorem allens_mother_age (allen_age mother_age : ℕ) : 
  allen_age = mother_age - 25 →
  allen_age + mother_age + 6 = 41 →
  mother_age = 30 := by
sorry

end NUMINAMATH_CALUDE_allens_mother_age_l172_17222


namespace NUMINAMATH_CALUDE_contradictory_statement_l172_17265

theorem contradictory_statement (x : ℝ) :
  (∀ x, x + 3 ≥ 0 → x ≥ -3) ↔ (∀ x, x + 3 < 0 → x < -3) :=
by sorry

end NUMINAMATH_CALUDE_contradictory_statement_l172_17265


namespace NUMINAMATH_CALUDE_quadratic_decreasing_condition_l172_17241

/-- A quadratic function f(x) = -2x^2 + mx - 3 is decreasing on the interval [-1, +∞) if and only if m ≤ -4 -/
theorem quadratic_decreasing_condition (m : ℝ) :
  (∀ x : ℝ, x ≥ -1 → (∀ y : ℝ, y > x → -2*y^2 + m*y - 3 < -2*x^2 + m*x - 3)) ↔ m ≤ -4 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_decreasing_condition_l172_17241


namespace NUMINAMATH_CALUDE_probability_at_least_two_correct_l172_17201

/-- The probability of getting at least two correct answers out of five questions
    with four choices each, when guessing randomly. -/
theorem probability_at_least_two_correct : ℝ := by
  -- Define the number of questions and choices
  let n : ℕ := 5
  let choices : ℕ := 4

  -- Define the probability of a correct guess
  let p : ℝ := 1 / choices

  -- Define the binomial probability function
  let binomial_prob (k : ℕ) : ℝ := (n.choose k) * p^k * (1 - p)^(n - k)

  -- Calculate the probability of getting 0 or 1 correct
  let prob_zero_or_one : ℝ := binomial_prob 0 + binomial_prob 1

  -- The probability of at least two correct is 1 minus the probability of 0 or 1 correct
  let prob_at_least_two : ℝ := 1 - prob_zero_or_one

  -- Prove that this probability is equal to 47/128
  sorry

#eval (47 : ℚ) / 128

end NUMINAMATH_CALUDE_probability_at_least_two_correct_l172_17201


namespace NUMINAMATH_CALUDE_product_increase_l172_17266

theorem product_increase (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a + 1) * (b + 1) = 2 * a * b) : 
  (a^2 - 1) * (b^2 - 1) = 4 * a * b := by
sorry

end NUMINAMATH_CALUDE_product_increase_l172_17266


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l172_17245

theorem arithmetic_sequence_seventh_term
  (a : ℚ) -- First term of the sequence
  (d : ℚ) -- Common difference of the sequence
  (h1 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 20) -- Sum of first five terms
  (h2 : a + 5*d = 8) -- Sixth term
  : a + 6*d = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l172_17245


namespace NUMINAMATH_CALUDE_binary_octal_conversion_l172_17244

/-- Converts a binary number (represented as a list of 0s and 1s) to decimal -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.reverse.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts a decimal number to octal (represented as a list of digits) -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem binary_octal_conversion :
  let binary : List Nat := [1, 0, 1, 1, 1, 1]
  let decimal : Nat := binary_to_decimal binary
  let octal : List Nat := decimal_to_octal decimal
  decimal = 47 ∧ octal = [5, 7] := by sorry

end NUMINAMATH_CALUDE_binary_octal_conversion_l172_17244


namespace NUMINAMATH_CALUDE_square_rectangle_area_multiplier_l172_17294

theorem square_rectangle_area_multiplier :
  let square_perimeter : ℝ := 800
  let rectangle_length : ℝ := 125
  let rectangle_width : ℝ := 64
  let square_side : ℝ := square_perimeter / 4
  let square_area : ℝ := square_side * square_side
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  square_area / rectangle_area = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_multiplier_l172_17294


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l172_17233

theorem triangle_max_perimeter :
  ∀ (x : ℕ),
  x > 0 →
  x + 2*x > 15 →
  x + 15 > 2*x →
  2*x + 15 > x →
  (∀ y : ℕ, y > 0 → y + 2*y > 15 → y + 15 > 2*y → 2*y + 15 > y → x + 2*x + 15 ≥ y + 2*y + 15) →
  x + 2*x + 15 = 57 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l172_17233


namespace NUMINAMATH_CALUDE_inequality_range_l172_17232

theorem inequality_range (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ m : ℝ, (3 * x) / (2 * x + y) + (3 * y) / (x + 2 * y) ≤ m^2 + m) ↔ 
  (m ≤ -2 ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l172_17232


namespace NUMINAMATH_CALUDE_laundry_cost_is_11_l172_17240

/-- The cost of Samantha's laundry given the specified conditions. -/
def laundry_cost : ℚ :=
  let washer_cost : ℚ := 4
  let dryer_cost_per_10_min : ℚ := 1/4
  let wash_loads : ℕ := 2
  let dryer_count : ℕ := 3
  let dryer_time : ℕ := 40

  let wash_cost : ℚ := washer_cost * wash_loads
  let dryer_intervals : ℕ := dryer_time / 10
  let dryer_cost : ℚ := (dryer_cost_per_10_min * dryer_intervals) * dryer_count

  wash_cost + dryer_cost

/-- Theorem stating that the total cost of Samantha's laundry is $11. -/
theorem laundry_cost_is_11 : laundry_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_laundry_cost_is_11_l172_17240


namespace NUMINAMATH_CALUDE_neither_direct_nor_inverse_proportional_l172_17243

/-- A function representing the relationship between x and y --/
def Relationship (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x y, y = f x → (y = k * x ∨ x * y = k)

/-- Equation A: 2x + y = 5 --/
def EquationA (x y : ℝ) : Prop := 2 * x + y = 5

/-- Equation B: 4xy = 12 --/
def EquationB (x y : ℝ) : Prop := 4 * x * y = 12

/-- Equation C: x = 4y --/
def EquationC (x y : ℝ) : Prop := x = 4 * y

/-- Equation D: 2x + 3y = 15 --/
def EquationD (x y : ℝ) : Prop := 2 * x + 3 * y = 15

/-- Equation E: x/y = 4 --/
def EquationE (x y : ℝ) : Prop := x / y = 4

theorem neither_direct_nor_inverse_proportional :
  (¬ Relationship (λ x => 5 - 2 * x)) ∧
  (Relationship (λ x => 3 / (4 * x))) ∧
  (Relationship (λ x => x / 4)) ∧
  (¬ Relationship (λ x => (15 - 2 * x) / 3)) ∧
  (Relationship (λ x => x / 4)) :=
sorry

end NUMINAMATH_CALUDE_neither_direct_nor_inverse_proportional_l172_17243


namespace NUMINAMATH_CALUDE_ceiling_sqrt_165_l172_17247

theorem ceiling_sqrt_165 : ⌈Real.sqrt 165⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_165_l172_17247


namespace NUMINAMATH_CALUDE_parallel_line_equation_l172_17275

/-- Given a line L1 with equation 2x + y - 5 = 0 and a point P (1, -3),
    prove that the line L2 passing through P and parallel to L1
    has the equation 2x + y + 1 = 0 -/
theorem parallel_line_equation (x y : ℝ) :
  let L1 : ℝ × ℝ → Prop := λ (x, y) ↦ 2 * x + y - 5 = 0
  let P : ℝ × ℝ := (1, -3)
  let L2 : ℝ × ℝ → Prop := λ (x, y) ↦ 2 * x + y + 1 = 0
  (∀ (x₁ y₁ x₂ y₂ : ℝ), L1 (x₁, y₁) ∧ L1 (x₂, y₂) → (y₂ - y₁) = -2 * (x₂ - x₁)) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), L2 (x₁, y₁) ∧ L2 (x₂, y₂) → (y₂ - y₁) = -2 * (x₂ - x₁)) →
  L2 P →
  ∀ (a b : ℝ), (∀ (x y : ℝ), a * x + b * y + 1 = 0 ↔ L2 (x, y)) →
  a = 2 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l172_17275


namespace NUMINAMATH_CALUDE_largest_cube_minus_smallest_fifth_l172_17220

theorem largest_cube_minus_smallest_fifth : ∃ (a b : ℕ), 
  (∀ n : ℕ, n^3 < 999 → n ≤ a) ∧ 
  (a^3 < 999) ∧
  (∀ m : ℕ, m^5 > 99 → b ≤ m) ∧ 
  (b^5 > 99) ∧
  (a - b = 6) := by
sorry

end NUMINAMATH_CALUDE_largest_cube_minus_smallest_fifth_l172_17220


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l172_17252

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x + 2| > a) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l172_17252


namespace NUMINAMATH_CALUDE_f_range_characterization_l172_17285

def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem f_range_characterization :
  (∀ x : ℝ, f x > 2 ↔ x < (1/2) ∨ x > (5/2)) ∧
  (∀ x : ℝ, (∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ |a| * f x) ↔ (1/2) ≤ x ∧ x ≤ (5/2)) :=
by sorry

end NUMINAMATH_CALUDE_f_range_characterization_l172_17285


namespace NUMINAMATH_CALUDE_angle_measure_proof_l172_17246

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 6 * (90 - x)) → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l172_17246


namespace NUMINAMATH_CALUDE_perfect_square_base_9_l172_17279

def is_base_9_digit (d : ℕ) : Prop := d < 9

def base_9_to_decimal (a b d : ℕ) : ℕ := 729 * a + 81 * b + 36 + d

theorem perfect_square_base_9 (a b d : ℕ) (ha : a ≠ 0) (hd : is_base_9_digit d) :
  ∃ (k : ℕ), (base_9_to_decimal a b d) = k^2 → d ∈ ({0, 1, 4} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_base_9_l172_17279


namespace NUMINAMATH_CALUDE_extremum_values_l172_17242

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_values (a b : ℝ) :
  f a b 1 = 10 ∧ (deriv (f a b)) 1 = 0 → a = 4 ∧ b = -11 := by
  sorry

end NUMINAMATH_CALUDE_extremum_values_l172_17242


namespace NUMINAMATH_CALUDE_distribute_negative_two_l172_17255

theorem distribute_negative_two (x : ℝ) : -2 * (x + 1) = -2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_two_l172_17255


namespace NUMINAMATH_CALUDE_total_street_lights_l172_17276

theorem total_street_lights (neighborhoods : ℕ) (roads_per_neighborhood : ℕ) (lights_per_side : ℕ) : 
  neighborhoods = 10 → roads_per_neighborhood = 4 → lights_per_side = 250 →
  neighborhoods * roads_per_neighborhood * lights_per_side * 2 = 20000 := by
sorry

end NUMINAMATH_CALUDE_total_street_lights_l172_17276


namespace NUMINAMATH_CALUDE_income_growth_and_projection_l172_17212

/-- Represents the annual growth rate as a real number between 0 and 1 -/
def AnnualGrowthRate := { r : ℝ // 0 < r ∧ r < 1 }

/-- Calculates the future value given initial value, growth rate, and number of years -/
def futureValue (initialValue : ℝ) (rate : AnnualGrowthRate) (years : ℕ) : ℝ :=
  initialValue * (1 + rate.val) ^ years

theorem income_growth_and_projection (initialIncome : ℝ) (finalIncome : ℝ) (years : ℕ) 
  (h1 : initialIncome = 2500)
  (h2 : finalIncome = 3600)
  (h3 : years = 2) :
  ∃ (rate : AnnualGrowthRate),
    (futureValue initialIncome rate years = finalIncome) ∧ 
    (rate.val = 0.2) ∧
    (futureValue finalIncome rate 1 > 4200) := by
  sorry

#check income_growth_and_projection

end NUMINAMATH_CALUDE_income_growth_and_projection_l172_17212


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l172_17295

theorem complex_fraction_simplification :
  (1 * 3 * 5 * 7 * 9) * (10 * 12 * 14 * 16 * 18) / (5 * 6 * 7 * 8 * 9)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l172_17295


namespace NUMINAMATH_CALUDE_adoption_cost_calculation_l172_17288

/-- Calculates the total cost of preparing animals for adoption --/
def total_adoption_cost (cat_prep_cost adult_dog_prep_cost puppy_prep_cost : ℕ) 
                        (num_cats num_adult_dogs num_puppies : ℕ)
                        (additional_costs : List ℝ) : ℝ :=
  (cat_prep_cost * num_cats + 
   adult_dog_prep_cost * num_adult_dogs + 
   puppy_prep_cost * num_puppies : ℝ) + 
  additional_costs.sum

/-- Theorem stating the total cost for the given scenario --/
theorem adoption_cost_calculation 
  (cat_prep_cost : ℕ) (adult_dog_prep_cost : ℕ) (puppy_prep_cost : ℕ)
  (x1 x2 x3 x4 x5 x6 x7 : ℝ) :
  cat_prep_cost = 50 →
  adult_dog_prep_cost = 100 →
  puppy_prep_cost = 150 →
  total_adoption_cost cat_prep_cost adult_dog_prep_cost puppy_prep_cost 2 3 2 [x1, x2, x3, x4, x5, x6, x7] = 
    700 + x1 + x2 + x3 + x4 + x5 + x6 + x7 :=
by
  sorry

end NUMINAMATH_CALUDE_adoption_cost_calculation_l172_17288


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l172_17281

theorem perfect_square_trinomial (x : ℝ) : ∃ (a : ℝ), (x^2 + 4 + 4*x) = (x + a)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l172_17281


namespace NUMINAMATH_CALUDE_no_single_intersection_point_l172_17262

theorem no_single_intersection_point :
  ¬ ∃ (b : ℝ), b ≠ 0 ∧
    (∃! (x : ℝ), bx^2 + 2*x - 3 = 2*x + 5) :=
sorry

end NUMINAMATH_CALUDE_no_single_intersection_point_l172_17262


namespace NUMINAMATH_CALUDE_pants_and_belt_cost_l172_17268

/-- The total cost of a pair of pants and a belt, given their prices -/
def total_cost (pants_price belt_price : ℝ) : ℝ := pants_price + belt_price

theorem pants_and_belt_cost :
  let pants_price : ℝ := 34.0
  let belt_price : ℝ := pants_price + 2.93
  total_cost pants_price belt_price = 70.93 := by
sorry

end NUMINAMATH_CALUDE_pants_and_belt_cost_l172_17268


namespace NUMINAMATH_CALUDE_factorization_ax2_minus_4a_l172_17273

theorem factorization_ax2_minus_4a (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax2_minus_4a_l172_17273


namespace NUMINAMATH_CALUDE_total_spent_is_575_l172_17205

def vacuum_original_cost : ℚ := 250
def vacuum_discount_rate : ℚ := 20 / 100
def dishwasher_cost : ℚ := 450
def combined_discount : ℚ := 75

def total_spent : ℚ :=
  (vacuum_original_cost * (1 - vacuum_discount_rate) + dishwasher_cost) - combined_discount

theorem total_spent_is_575 : total_spent = 575 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_575_l172_17205


namespace NUMINAMATH_CALUDE_max_N_is_seven_l172_17214

def J (k : ℕ) : ℕ := 10^(k+3) + 128

def N (k : ℕ) : ℕ := (J k).factors.count 2

theorem max_N_is_seven : ∀ k : ℕ, k > 0 → N k ≤ 7 ∧ ∃ k₀ : ℕ, k₀ > 0 ∧ N k₀ = 7 :=
sorry

end NUMINAMATH_CALUDE_max_N_is_seven_l172_17214


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l172_17209

theorem quadratic_roots_nature (k : ℝ) : 
  (∃ x y : ℝ, x * y = 12 ∧ x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ y^2 - 4*k*y + 3*k^2 + 1 = 0) →
  (∃ x : ℝ, x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ (∀ y : ℝ, y^2 - 4*k*y + 3*k^2 + 1 = 0 → y = x)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l172_17209


namespace NUMINAMATH_CALUDE_multiple_properties_l172_17290

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 5 * k) 
  (hb : ∃ m : ℤ, b = 10 * m) : 
  (∃ n : ℤ, b = 5 * n) ∧ 
  (∃ p : ℤ, a + b = 5 * p) ∧ 
  (∃ q : ℤ, a + b = 2 * q) := by
  sorry

end NUMINAMATH_CALUDE_multiple_properties_l172_17290


namespace NUMINAMATH_CALUDE_symmetric_point_of_P_l172_17200

/-- The symmetric point of P(1, 3) with respect to the line y=x is (3, 1) -/
theorem symmetric_point_of_P : ∃ (P' : ℝ × ℝ), 
  (P' = (3, 1) ∧ 
   (∀ (Q : ℝ × ℝ), Q.1 = Q.2 → (1 - Q.1)^2 + (3 - Q.2)^2 = (P'.1 - Q.1)^2 + (P'.2 - Q.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_of_P_l172_17200


namespace NUMINAMATH_CALUDE_floor_factorial_ratio_l172_17223

open BigOperators

def factorial (n : ℕ) : ℕ := ∏ i in Finset.range n, i + 1

theorem floor_factorial_ratio : 
  ⌊(factorial 2007 + factorial 2004 : ℚ) / (factorial 2006 + factorial 2005)⌋ = 2006 := by
  sorry

end NUMINAMATH_CALUDE_floor_factorial_ratio_l172_17223


namespace NUMINAMATH_CALUDE_line_cartesian_to_polar_l172_17236

/-- Given a line in Cartesian coordinates x cos α + y sin α = 0,
    its equivalent polar coordinate equation is θ = α - π/2 --/
theorem line_cartesian_to_polar (α : Real) :
  ∀ x y r θ : Real,
  (x * Real.cos α + y * Real.sin α = 0) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  (θ = α - π/2) := by
  sorry

end NUMINAMATH_CALUDE_line_cartesian_to_polar_l172_17236


namespace NUMINAMATH_CALUDE_certain_number_proof_l172_17221

theorem certain_number_proof (x y : ℤ) 
  (eq1 : 4 * x + y = 34) 
  (eq2 : y^2 = 4) : 
  2 * x - y = 14 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l172_17221


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l172_17280

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (i / (1 + i) : ℂ) = a + b * I :=
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l172_17280


namespace NUMINAMATH_CALUDE_integer_divisibility_problem_l172_17289

theorem integer_divisibility_problem (n : ℤ) :
  (∃ k : ℤ, n - 4 = 6 * k) ∧ (∃ m : ℤ, n - 8 = 10 * m) →
  n ≡ 28 [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_integer_divisibility_problem_l172_17289


namespace NUMINAMATH_CALUDE_conditional_structure_correctness_l172_17211

-- Define a conditional structure
structure ConditionalStructure where
  hasTwoExits : Bool
  hasOneEffectiveExit : Bool

-- Define the properties of conditional structures
def conditionalStructureProperties : ConditionalStructure where
  hasTwoExits := true
  hasOneEffectiveExit := true

-- Theorem to prove
theorem conditional_structure_correctness :
  (conditionalStructureProperties.hasTwoExits = true) ∧
  (conditionalStructureProperties.hasOneEffectiveExit = true) := by
  sorry

#check conditional_structure_correctness

end NUMINAMATH_CALUDE_conditional_structure_correctness_l172_17211


namespace NUMINAMATH_CALUDE_root_implies_k_value_l172_17215

theorem root_implies_k_value (k : ℝ) : 
  (2^2 - 3*2 + k = 0) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l172_17215


namespace NUMINAMATH_CALUDE_f_leq_f_f_eq_abs_f_l172_17287

/-- The function f(x) = x^2 + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a

theorem f_leq_f'_iff_a_geq_three_halves (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) (-1), f a x ≤ f' a x) ↔ a ≥ 3/2 := by sorry

theorem f_eq_abs_f'_solutions (a : ℝ) :
  (∀ x : ℝ, f a x = |f' a x|) ↔
  ((a < -1 ∧ (x = -1 ∨ x = 1 - 2*a)) ∨
   (-1 ≤ a ∧ a ≤ 1 ∧ (x = 1 ∨ x = -1 ∨ x = 1 - 2*a ∨ x = -(1 + 2*a))) ∨
   (a > 1 ∧ (x = 1 ∨ x = -(1 + 2*a)))) := by sorry

end NUMINAMATH_CALUDE_f_leq_f_f_eq_abs_f_l172_17287


namespace NUMINAMATH_CALUDE_xy_value_l172_17217

theorem xy_value (x y : ℚ) 
  (eq1 : 5 * x + 3 * y + 5 = 0) 
  (eq2 : 3 * x + 5 * y - 5 = 0) : 
  x * y = -25 / 4 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l172_17217
