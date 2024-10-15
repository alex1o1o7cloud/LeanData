import Mathlib

namespace NUMINAMATH_CALUDE_inspection_team_selection_l1493_149342

theorem inspection_team_selection 
  (total_employees : ℕ) 
  (men : ℕ) 
  (women : ℕ) 
  (team_size : ℕ) 
  (h1 : total_employees = 15)
  (h2 : men = 10)
  (h3 : women = 5)
  (h4 : team_size = 6)
  (h5 : men + women = total_employees)
  (h6 : 2 * women = men) : 
  Nat.choose men 4 * Nat.choose women 2 = 
  (number_of_ways_to_select_team : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inspection_team_selection_l1493_149342


namespace NUMINAMATH_CALUDE_tanning_salon_revenue_l1493_149364

/-- Calculate the revenue of a tanning salon for a calendar month -/
theorem tanning_salon_revenue 
  (first_visit_cost : ℕ) 
  (subsequent_visit_cost : ℕ) 
  (total_customers : ℕ) 
  (second_visit_customers : ℕ) 
  (third_visit_customers : ℕ)
  (h1 : first_visit_cost = 10)
  (h2 : subsequent_visit_cost = 8)
  (h3 : total_customers = 100)
  (h4 : second_visit_customers = 30)
  (h5 : third_visit_customers = 10)
  (h6 : second_visit_customers ≤ total_customers)
  (h7 : third_visit_customers ≤ second_visit_customers) :
  first_visit_cost * total_customers + 
  subsequent_visit_cost * second_visit_customers + 
  subsequent_visit_cost * third_visit_customers = 1320 :=
by sorry

end NUMINAMATH_CALUDE_tanning_salon_revenue_l1493_149364


namespace NUMINAMATH_CALUDE_conic_eccentricity_l1493_149354

/-- The eccentricity of a conic section given by x^2 + y^2/m = 1, where m is the geometric mean of 2 and 8 -/
theorem conic_eccentricity (m : ℝ) : 
  (m^2 = 2 * 8) → 
  (∃ (x y : ℝ), x^2 + y^2/m = 1) → 
  (∃ (e : ℝ), e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l1493_149354


namespace NUMINAMATH_CALUDE_mr_blue_bean_yield_l1493_149382

/-- Calculates the expected bean yield for a rectangular terrain --/
def expected_bean_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * step_length * (width_steps : ℝ) * step_length * yield_per_sqft

/-- Proves that the expected bean yield for Mr. Blue's terrain is 5906.25 pounds --/
theorem mr_blue_bean_yield :
  expected_bean_yield 25 35 3 0.75 = 5906.25 := by
  sorry

#eval expected_bean_yield 25 35 3 0.75

end NUMINAMATH_CALUDE_mr_blue_bean_yield_l1493_149382


namespace NUMINAMATH_CALUDE_runner_area_theorem_l1493_149307

/-- Given a table and three runners, calculates the total area of the runners -/
def total_runner_area (table_area : ℝ) (double_layer_area : ℝ) (triple_layer_area : ℝ) : ℝ :=
  let covered_area := 0.8 * table_area
  let single_layer_area := covered_area - double_layer_area - triple_layer_area
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area

/-- Theorem stating that under the given conditions, the total area of the runners is 168 square inches -/
theorem runner_area_theorem (table_area : ℝ) (double_layer_area : ℝ) (triple_layer_area : ℝ) 
  (h1 : table_area = 175)
  (h2 : double_layer_area = 24)
  (h3 : triple_layer_area = 28) :
  total_runner_area table_area double_layer_area triple_layer_area = 168 := by
  sorry

#eval total_runner_area 175 24 28

end NUMINAMATH_CALUDE_runner_area_theorem_l1493_149307


namespace NUMINAMATH_CALUDE_leonardo_earnings_l1493_149362

/-- Calculates the total earnings for Leonardo over two weeks given the following conditions:
  * Leonardo worked 18 hours in the second week
  * Leonardo worked 13 hours in the first week
  * Leonardo earned $65.70 more in the second week than in the first week
  * His hourly wage remained the same throughout both weeks
-/
def total_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) : ℚ :=
  let hourly_wage := extra_earnings / (hours_week2 - hours_week1 : ℚ)
  (hours_week1 + hours_week2 : ℚ) * hourly_wage

/-- The theorem states that given the specific conditions in the problem,
    Leonardo's total earnings for the two weeks is $407.34. -/
theorem leonardo_earnings :
  total_earnings 13 18 65.70 = 407.34 := by
  sorry

end NUMINAMATH_CALUDE_leonardo_earnings_l1493_149362


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1493_149369

theorem algebraic_expression_equality (a b : ℝ) (h1 : a = 3) (h2 : a - b = 1) :
  a^2 - a*b = a*(a - b) := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1493_149369


namespace NUMINAMATH_CALUDE_total_candidates_l1493_149397

theorem total_candidates (girls : ℕ) (boys_fail_rate : ℝ) (girls_fail_rate : ℝ) (total_fail_rate : ℝ) :
  girls = 900 →
  boys_fail_rate = 0.7 →
  girls_fail_rate = 0.68 →
  total_fail_rate = 0.691 →
  ∃ (total : ℕ), total = 2000 ∧ 
    (boys_fail_rate * (total - girls) + girls_fail_rate * girls) / total = total_fail_rate :=
by sorry

end NUMINAMATH_CALUDE_total_candidates_l1493_149397


namespace NUMINAMATH_CALUDE_blue_to_red_ratio_l1493_149351

theorem blue_to_red_ratio (n : ℕ) (h : n = 13) : 
  (6 * n^3 - 6 * n^2) / (6 * n^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_blue_to_red_ratio_l1493_149351


namespace NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l1493_149337

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by
  sorry

theorem specific_square_root_squared : (Real.sqrt 625681) ^ 2 = 625681 := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l1493_149337


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1493_149360

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 2, 0],
    ![0, 1, 2],
    ![2, 0, 1]]

theorem matrix_equation_solution :
  ∃ (p q r : ℚ), B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 ∧ 
  p = -3 ∧ q = 3 ∧ r = -9 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1493_149360


namespace NUMINAMATH_CALUDE_infinitely_many_invalid_d_l1493_149312

/-- The perimeter difference between the triangle and rectangle -/
def perimeter_difference : ℕ := 504

/-- The length of the shorter side of the rectangle -/
def rectangle_short_side : ℕ := 7

/-- Represents the relationship between the triangle side length, rectangle long side, and d -/
def triangle_rectangle_relation (triangle_side : ℝ) (rectangle_long_side : ℝ) (d : ℝ) : Prop :=
  triangle_side = rectangle_long_side + d

/-- Represents the perimeter relationship between the triangle and rectangle -/
def perimeter_relation (triangle_side : ℝ) (rectangle_long_side : ℝ) : Prop :=
  3 * triangle_side - 2 * (rectangle_long_side + rectangle_short_side) = perimeter_difference

/-- The main theorem stating that there are infinitely many positive integers
    that cannot be valid values for d -/
theorem infinitely_many_invalid_d : ∃ (S : Set ℕ), Set.Infinite S ∧
  ∀ (d : ℕ), d ∈ S →
    ¬∃ (triangle_side rectangle_long_side : ℝ),
      triangle_rectangle_relation triangle_side rectangle_long_side d ∧
      perimeter_relation triangle_side rectangle_long_side :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_invalid_d_l1493_149312


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l1493_149301

def is_integer_fraction (m n : ℕ+) : Prop :=
  ∃ k : ℤ, (n.val ^ 3 + 1 : ℤ) = k * (m.val * n.val - 1)

def solution_set : Set (ℕ+ × ℕ+) :=
  {(2, 1), (3, 1), (1, 2), (2, 2), (5, 2), (1, 3), (5, 3), (3, 5)}

theorem integer_fraction_characterization :
  ∀ m n : ℕ+, is_integer_fraction m n ↔ (m, n) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l1493_149301


namespace NUMINAMATH_CALUDE_solution_replacement_fraction_l1493_149378

theorem solution_replacement_fraction (V : ℝ) (x : ℝ) 
  (h1 : V > 0)
  (h2 : 0 ≤ x ∧ x ≤ 1)
  (h3 : (0.80 * V - 0.80 * x * V) + 0.25 * x * V = 0.35 * V) :
  x = 9 / 11 := by
sorry

end NUMINAMATH_CALUDE_solution_replacement_fraction_l1493_149378


namespace NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l1493_149387

-- Define the universe of goods
variable (Goods : Type)

-- Define predicates for "cheap" and "good quality"
variable (cheap : Goods → Prop)
variable (good_quality : Goods → Prop)

-- State the given condition
variable (h : ∀ g : Goods, cheap g → ¬(good_quality g))

-- Theorem statement
theorem not_cheap_necessary_for_good_quality :
  ∀ g : Goods, good_quality g → ¬(cheap g) :=
by
  sorry

end NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l1493_149387


namespace NUMINAMATH_CALUDE_angle_bisector_exists_l1493_149396

-- Define the basic structures
structure Point :=
  (x : ℝ) (y : ℝ)

structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Define the given lines
def L1 : Line := sorry
def L2 : Line := sorry

-- Define the property of inaccessible intersection
def inaccessibleIntersection (l1 l2 : Line) : Prop := sorry

-- Define angle bisector
def isAngleBisector (bisector : Line) (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem angle_bisector_exists (h : inaccessibleIntersection L1 L2) :
  ∃ bisector : Line, isAngleBisector bisector L1 L2 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_exists_l1493_149396


namespace NUMINAMATH_CALUDE_negative_nine_less_than_negative_sqrt_80_l1493_149353

theorem negative_nine_less_than_negative_sqrt_80 : -9 < -Real.sqrt 80 := by
  sorry

end NUMINAMATH_CALUDE_negative_nine_less_than_negative_sqrt_80_l1493_149353


namespace NUMINAMATH_CALUDE_sqrt_fifteen_div_sqrt_three_eq_sqrt_five_l1493_149348

theorem sqrt_fifteen_div_sqrt_three_eq_sqrt_five : 
  Real.sqrt 15 / Real.sqrt 3 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifteen_div_sqrt_three_eq_sqrt_five_l1493_149348


namespace NUMINAMATH_CALUDE_tan_ratio_given_sin_equality_l1493_149304

theorem tan_ratio_given_sin_equality (α : ℝ) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * (π / 180))) : 
  Real.tan (α + π / 180) / Real.tan (α - π / 180) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_given_sin_equality_l1493_149304


namespace NUMINAMATH_CALUDE_unique_k_value_l1493_149344

theorem unique_k_value : ∃! k : ℝ, ∀ x : ℝ, 
  (x * (2 * x + 3) < k ↔ -5/2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_k_value_l1493_149344


namespace NUMINAMATH_CALUDE_rachel_total_steps_l1493_149367

/-- The total number of steps Rachel took during her trip to the Eiffel Tower -/
def total_steps (steps_up steps_down : ℕ) : ℕ := steps_up + steps_down

/-- Theorem stating that Rachel took 892 steps in total -/
theorem rachel_total_steps : total_steps 567 325 = 892 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_steps_l1493_149367


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l1493_149381

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l1493_149381


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l1493_149368

def arrange_books (geometry_copies : ℕ) (algebra_copies : ℕ) : ℕ :=
  Nat.choose (geometry_copies + algebra_copies - 2) (algebra_copies - 2)

theorem book_arrangement_proof : 
  arrange_books 4 5 = 35 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l1493_149368


namespace NUMINAMATH_CALUDE_tournament_boxes_needed_l1493_149394

/-- A single-elimination tennis tournament -/
structure TennisTournament where
  participants : ℕ
  boxes_per_match : ℕ

/-- The number of boxes needed for a single-elimination tournament -/
def boxes_needed (t : TennisTournament) : ℕ :=
  t.participants - 1

/-- Theorem: A single-elimination tournament with 199 participants needs 198 boxes -/
theorem tournament_boxes_needed :
  ∀ t : TennisTournament, t.participants = 199 ∧ t.boxes_per_match = 1 →
  boxes_needed t = 198 :=
by sorry

end NUMINAMATH_CALUDE_tournament_boxes_needed_l1493_149394


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l1493_149309

theorem sum_reciprocal_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3/(a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l1493_149309


namespace NUMINAMATH_CALUDE_root_difference_quadratic_equation_l1493_149308

theorem root_difference_quadratic_equation : 
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := -12
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  abs (root1 - root2) = 5.5 := by
sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_equation_l1493_149308


namespace NUMINAMATH_CALUDE_game_cost_is_two_l1493_149333

/-- Calculates the cost of a new game based on initial money, allowance, and final amount. -/
def game_cost (initial_money : ℝ) (allowance : ℝ) (final_amount : ℝ) : ℝ :=
  initial_money + allowance - final_amount

/-- Proves that the cost of the new game is $2 given the specific amounts in the problem. -/
theorem game_cost_is_two :
  game_cost 5 5 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_is_two_l1493_149333


namespace NUMINAMATH_CALUDE_cos_seven_pi_fourth_l1493_149329

theorem cos_seven_pi_fourth : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_fourth_l1493_149329


namespace NUMINAMATH_CALUDE_mothers_day_discount_l1493_149311

def original_price : ℝ := 125
def mother_discount : ℝ := 0.1
def additional_discount : ℝ := 0.04
def children_count : ℕ := 4

theorem mothers_day_discount (price : ℝ) (md : ℝ) (ad : ℝ) (cc : ℕ) :
  price > 0 →
  md > 0 →
  ad > 0 →
  cc ≥ 3 →
  price * (1 - md) * (1 - ad) = 108 := by
sorry

end NUMINAMATH_CALUDE_mothers_day_discount_l1493_149311


namespace NUMINAMATH_CALUDE_exists_zero_sum_assignment_l1493_149372

/-- A regular 2n-gon -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- An arrow assignment for a regular 2n-gon -/
def ArrowAssignment (n : ℕ) := 
  (Fin (2*n) × Fin (2*n)) → ℝ × ℝ

/-- The sum of vectors in an arrow assignment -/
def sumVectors (n : ℕ) (assignment : ArrowAssignment n) : ℝ × ℝ := sorry

/-- Theorem stating the existence of a zero-sum arrow assignment -/
theorem exists_zero_sum_assignment (n : ℕ) (polygon : RegularPolygon n) :
  ∃ (assignment : ArrowAssignment n), sumVectors n assignment = (0, 0) := by sorry

end NUMINAMATH_CALUDE_exists_zero_sum_assignment_l1493_149372


namespace NUMINAMATH_CALUDE_dice_probability_l1493_149328

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice re-rolled -/
def numRerolled : ℕ := 3

/-- The total number of possible outcomes when re-rolling -/
def totalOutcomes : ℕ := numFaces ^ numRerolled

/-- The number of ways the re-rolled dice can not match the pair -/
def waysNotMatchingPair : ℕ := (numFaces - 1) ^ numRerolled

/-- The number of ways at least one re-rolled die matches the pair -/
def waysMatchingPair : ℕ := totalOutcomes - waysNotMatchingPair

/-- The number of ways all re-rolled dice match each other -/
def waysAllMatch : ℕ := numFaces

/-- The number of successful outcomes -/
def successfulOutcomes : ℕ := waysMatchingPair + waysAllMatch - 1

/-- The probability of at least three dice showing the same value after re-rolling -/
def probability : ℚ := successfulOutcomes / totalOutcomes

theorem dice_probability : probability = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1493_149328


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1493_149332

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (geom_seq : b^2 = a * c)
  (value_a : a = 5 + 2 * Real.sqrt 3)
  (value_c : c = 5 - 2 * Real.sqrt 3) : 
  b = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1493_149332


namespace NUMINAMATH_CALUDE_function_properties_l1493_149313

def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (-(x + 1))

def is_odd_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f (-(x + 2))

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def equation_solutions (f : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x = x ∧ f y = y ∧ f z = z ∧
    ∀ w, f w = w → w = x ∨ w = y ∨ w = z

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even_shifted f)
  (h2 : is_odd_shifted f)
  (h3 : ∀ x ∈ Set.Icc 0 1, f x = 2^x - 1) :
  symmetric_about f 1 ∧ equation_solutions f := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1493_149313


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1493_149375

-- Define a structure for a rectangular solid
structure RectangularSolid where
  a : ℕ
  b : ℕ
  c : ℕ

-- Define primality
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define volume
def volume (solid : RectangularSolid) : ℕ := solid.a * solid.b * solid.c

-- Define surface area
def surfaceArea (solid : RectangularSolid) : ℕ :=
  2 * (solid.a * solid.b + solid.b * solid.c + solid.c * solid.a)

-- The main theorem
theorem rectangular_solid_surface_area :
  ∀ solid : RectangularSolid,
    isPrime solid.a ∧ isPrime solid.b ∧ isPrime solid.c →
    volume solid = 221 →
    surfaceArea solid = 502 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1493_149375


namespace NUMINAMATH_CALUDE_determine_contents_l1493_149345

-- Define the colors of balls
inductive Color
| White
| Black

-- Define the types of boxes
inductive BoxType
| TwoWhite
| TwoBlack
| OneWhiteOneBlack

-- Define a box with a label and contents
structure Box where
  label : BoxType
  contents : BoxType

-- Define the problem setup
def problem_setup : Prop :=
  ∃ (box1 box2 box3 : Box),
    -- Three boxes with different labels
    box1.label ≠ box2.label ∧ box2.label ≠ box3.label ∧ box1.label ≠ box3.label ∧
    -- Contents don't match labels
    box1.contents ≠ box1.label ∧ box2.contents ≠ box2.label ∧ box3.contents ≠ box3.label ∧
    -- One box has two white balls, one has two black balls, and one has one of each
    (box1.contents = BoxType.TwoWhite ∧ box2.contents = BoxType.TwoBlack ∧ box3.contents = BoxType.OneWhiteOneBlack) ∨
    (box1.contents = BoxType.TwoWhite ∧ box2.contents = BoxType.OneWhiteOneBlack ∧ box3.contents = BoxType.TwoBlack) ∨
    (box1.contents = BoxType.TwoBlack ∧ box2.contents = BoxType.TwoWhite ∧ box3.contents = BoxType.OneWhiteOneBlack) ∨
    (box1.contents = BoxType.TwoBlack ∧ box2.contents = BoxType.OneWhiteOneBlack ∧ box3.contents = BoxType.TwoWhite) ∨
    (box1.contents = BoxType.OneWhiteOneBlack ∧ box2.contents = BoxType.TwoWhite ∧ box3.contents = BoxType.TwoBlack) ∨
    (box1.contents = BoxType.OneWhiteOneBlack ∧ box2.contents = BoxType.TwoBlack ∧ box3.contents = BoxType.TwoWhite)

-- Define the theorem
theorem determine_contents (setup : problem_setup) :
  ∃ (box : Box) (c : Color),
    box.label = BoxType.OneWhiteOneBlack →
    (c = Color.White → box.contents = BoxType.TwoWhite) ∧
    (c = Color.Black → box.contents = BoxType.TwoBlack) :=
sorry

end NUMINAMATH_CALUDE_determine_contents_l1493_149345


namespace NUMINAMATH_CALUDE_simplify_expression_l1493_149356

theorem simplify_expression (x y : ℝ) : (5 - 4*y) - (6 + 5*y - 2*x) = -1 - 9*y + 2*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1493_149356


namespace NUMINAMATH_CALUDE_smallest_b_not_prime_nine_satisfies_condition_nine_is_smallest_l1493_149314

theorem smallest_b_not_prime (b : ℕ) (h : b > 8) :
  (∀ x : ℤ, ¬ Prime (x^4 + b^2 : ℤ)) →
  b ≥ 9 :=
by sorry

theorem nine_satisfies_condition :
  ∀ x : ℤ, ¬ Prime (x^4 + 9^2 : ℤ) :=
by sorry

theorem nine_is_smallest :
  ∀ b : ℕ, b > 8 →
  (∀ x : ℤ, ¬ Prime (x^4 + b^2 : ℤ)) →
  b ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_not_prime_nine_satisfies_condition_nine_is_smallest_l1493_149314


namespace NUMINAMATH_CALUDE_divisibility_implies_inequality_l1493_149310

theorem divisibility_implies_inequality (a k : ℕ+) :
  (a^2 + k : ℕ) ∣ ((a - 1) * a * (a + 1) : ℕ) → k ≥ a :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_inequality_l1493_149310


namespace NUMINAMATH_CALUDE_root_sum_squared_l1493_149399

theorem root_sum_squared (a b : ℝ) : 
  (a^2 + 2*a - 2016 = 0) → 
  (b^2 + 2*b - 2016 = 0) → 
  (a + b = -2) → 
  (a^2 + 3*a + b = 2014) := by
sorry

end NUMINAMATH_CALUDE_root_sum_squared_l1493_149399


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l1493_149330

/-- Given that two dozen apples cost $15.60, prove that four dozen apples at the same rate will cost $31.20. -/
theorem apple_cost_calculation (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) :
  let cost_per_dozen : ℝ := cost_two_dozen / 2
  let cost_four_dozen : ℝ := 4 * cost_per_dozen
  cost_four_dozen = 31.20 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l1493_149330


namespace NUMINAMATH_CALUDE_jennas_age_l1493_149317

/-- Given that Jenna is 5 years older than Darius, their ages sum to 21, and Darius is 8 years old,
    prove that Jenna is 13 years old. -/
theorem jennas_age (jenna_age darius_age : ℕ) 
  (h1 : jenna_age = darius_age + 5)
  (h2 : jenna_age + darius_age = 21)
  (h3 : darius_age = 8) :
  jenna_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_jennas_age_l1493_149317


namespace NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l1493_149389

theorem ice_cream_scoop_arrangements (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l1493_149389


namespace NUMINAMATH_CALUDE_quadratic_inequality_counterexample_l1493_149339

theorem quadratic_inequality_counterexample :
  ∃ (a b c : ℝ), b^2 - 4*a*c ≤ 0 ∧ ∃ (x : ℝ), a*x^2 + b*x + c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_counterexample_l1493_149339


namespace NUMINAMATH_CALUDE_inequality_solution_set_k_value_range_l1493_149306

-- Problem 1
theorem inequality_solution_set (x : ℝ) : 
  -2 * x^2 - x + 6 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3/2 :=
sorry

-- Problem 2
theorem k_value_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) ↔ (k > Real.sqrt 2 ∨ k < -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_k_value_range_l1493_149306


namespace NUMINAMATH_CALUDE_system_solutions_l1493_149365

theorem system_solutions (x y z : ℤ) : 
  x^2 - 9*y^2 - z^2 = 0 ∧ z = x - 3*y →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 3 ∧ y = 1 ∧ z = 0) ∨
  (x = 9 ∧ y = 3 ∧ z = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solutions_l1493_149365


namespace NUMINAMATH_CALUDE_max_ratio_x_y_l1493_149326

theorem max_ratio_x_y (x y a b : ℝ) : 
  x ≥ y ∧ y > 0 →
  0 ≤ a ∧ a ≤ x →
  0 ≤ b ∧ b ≤ y →
  (x - a)^2 + (y - b)^2 = x^2 + b^2 →
  (x - a)^2 + (y - b)^2 = y^2 + a^2 →
  ∃ (c : ℝ), c = x / y ∧ c ≤ 2 * Real.sqrt 3 / 3 ∧
  ∀ (d : ℝ), d = x / y → d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_x_y_l1493_149326


namespace NUMINAMATH_CALUDE_quadratic_roots_sums_l1493_149377

theorem quadratic_roots_sums (p q x₁ x₂ : ℝ) 
  (hq : q ≠ 0)
  (hroots : x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) :
  (1/x₁ + 1/x₂ = -p/q) ∧
  (1/x₁^2 + 1/x₂^2 = (p^2 - 2*q)/q^2) ∧
  (1/x₁^3 + 1/x₂^3 = (p/q^3)*(3*q - p^2)) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_sums_l1493_149377


namespace NUMINAMATH_CALUDE_number_divisibility_l1493_149398

theorem number_divisibility :
  (∀ a : ℕ, 100 ≤ a ∧ a < 1000 → (7 ∣ 1001 * a) ∧ (11 ∣ 1001 * a) ∧ (13 ∣ 1001 * a)) ∧
  (∀ b : ℕ, 1000 ≤ b ∧ b < 10000 → (73 ∣ 10001 * b) ∧ (137 ∣ 10001 * b)) :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_l1493_149398


namespace NUMINAMATH_CALUDE_work_completion_time_l1493_149334

/-- If P people can complete a job in 20 days, then 2P people can complete half of the job in 5 days -/
theorem work_completion_time 
  (P : ℕ) -- number of people
  (full_work_time : ℕ := 20) -- time to complete full work with P people
  (h : P > 0) -- ensure P is positive
  : (2 * P) * 5 = P * full_work_time / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1493_149334


namespace NUMINAMATH_CALUDE_absolute_value_six_point_five_l1493_149374

theorem absolute_value_six_point_five (x : ℝ) : |x| = 6.5 ↔ x = 6.5 ∨ x = -6.5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_six_point_five_l1493_149374


namespace NUMINAMATH_CALUDE_reflection_maps_correctly_l1493_149386

-- Define points in 2D space
def C : Prod ℝ ℝ := (-3, 2)
def D : Prod ℝ ℝ := (-2, 5)
def C' : Prod ℝ ℝ := (3, -2)
def D' : Prod ℝ ℝ := (2, -5)

-- Define reflection across y = -x
def reflect_across_y_eq_neg_x (p : Prod ℝ ℝ) : Prod ℝ ℝ :=
  (-p.2, -p.1)

-- Theorem statement
theorem reflection_maps_correctly :
  reflect_across_y_eq_neg_x C = C' ∧
  reflect_across_y_eq_neg_x D = D' := by
  sorry

end NUMINAMATH_CALUDE_reflection_maps_correctly_l1493_149386


namespace NUMINAMATH_CALUDE_remainder_theorem_l1493_149393

theorem remainder_theorem (P D D' Q Q' R R' : ℤ) 
  (h1 : P = Q * D + R) 
  (h2 : 0 ≤ R ∧ R < D) 
  (h3 : Q = Q' * D' + R') 
  (h4 : 0 ≤ R' ∧ R' < D') : 
  P % (D * D') = R + R' * D :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1493_149393


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1493_149373

theorem more_girls_than_boys (boys girls : ℕ) : 
  boys = 40 →
  girls * 5 = boys * 13 →
  girls > boys →
  girls - boys = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1493_149373


namespace NUMINAMATH_CALUDE_mini_train_length_l1493_149383

/-- The length of a mini-train given its speed and time to cross a pole -/
theorem mini_train_length (speed_kmph : ℝ) (time_seconds : ℝ) : 
  speed_kmph = 75 → time_seconds = 3 → 
  (speed_kmph * 1000 / 3600) * time_seconds = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_mini_train_length_l1493_149383


namespace NUMINAMATH_CALUDE_min_value_of_a_l1493_149395

theorem min_value_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, x > 1 → x + a / (x - 1) ≥ 5) : 
  (∀ b : ℝ, b > 0 → (∀ x : ℝ, x > 1 → x + b / (x - 1) ≥ 5) → b ≥ a) ∧ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1493_149395


namespace NUMINAMATH_CALUDE_polynomial_value_l1493_149347

theorem polynomial_value (x y : ℝ) (h : x + 2*y = 6) : 2*x + 4*y - 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l1493_149347


namespace NUMINAMATH_CALUDE_carla_food_bank_theorem_l1493_149302

/-- Represents the food bank scenario with Carla --/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day1_restock : ℕ
  day2_people : ℕ
  day2_cans_per_person : ℕ
  day2_restock : ℕ

/-- Calculates the total number of cans given away --/
def total_cans_given_away (fb : FoodBank) : ℕ :=
  fb.day1_people * fb.day1_cans_per_person + fb.day2_people * fb.day2_cans_per_person

/-- Theorem stating that the total cans given away is 2500 --/
theorem carla_food_bank_theorem (fb : FoodBank) 
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day1_restock = 1500)
  (h5 : fb.day2_people = 1000)
  (h6 : fb.day2_cans_per_person = 2)
  (h7 : fb.day2_restock = 3000) :
  total_cans_given_away fb = 2500 := by
  sorry

end NUMINAMATH_CALUDE_carla_food_bank_theorem_l1493_149302


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l1493_149300

/-- Given a boat traveling downstream, calculates the speed of the stream. -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) 
  (h1 : boat_speed = 13)
  (h2 : downstream_distance = 69)
  (h3 : downstream_time = 3.6315789473684212) :
  let downstream_speed := downstream_distance / downstream_time
  let stream_speed := downstream_speed - boat_speed
  stream_speed = 6 := by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l1493_149300


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1493_149352

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x₀ : ℝ, x₀^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1493_149352


namespace NUMINAMATH_CALUDE_smallest_n_with_2323_divisible_l1493_149343

def count_divisible (n : ℕ) : ℕ :=
  (n / 2) + (n / 23) - 2 * (n / 46)

theorem smallest_n_with_2323_divisible : ∃ (n : ℕ), n > 0 ∧ count_divisible n = 2323 ∧ ∀ m < n, count_divisible m ≠ 2323 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_2323_divisible_l1493_149343


namespace NUMINAMATH_CALUDE_five_people_lineup_l1493_149391

/-- The number of ways to arrange n people in a line where k people cannot be first -/
def arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n - k) * n.factorial

/-- The problem statement -/
theorem five_people_lineup : arrangements 5 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_people_lineup_l1493_149391


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1493_149355

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 9) 
  (eq2 : x + 4 * y = 16) : 
  18 * x^2 + 20 * x * y + 18 * y^2 = 337 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1493_149355


namespace NUMINAMATH_CALUDE_max_profit_is_45_6_l1493_149346

/-- Profit function for location A -/
def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def profit_B (x : ℝ) : ℝ := 2 * x

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit_is_45_6 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 15 ∧ 
  (∀ y : ℝ, y ≥ 0 → y ≤ 15 → total_profit y ≤ total_profit x) ∧
  total_profit x = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_45_6_l1493_149346


namespace NUMINAMATH_CALUDE_nonIntersectingPolylines_correct_l1493_149385

/-- The number of ways to connect n points on a circle with a non-self-intersecting polyline -/
def nonIntersectingPolylines (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n ≥ 3 then n * 2^(n-3)
  else 0

theorem nonIntersectingPolylines_correct (n : ℕ) (h : n > 1) :
  nonIntersectingPolylines n =
    if n = 2 then 1
    else n * 2^(n-3) := by
  sorry

end NUMINAMATH_CALUDE_nonIntersectingPolylines_correct_l1493_149385


namespace NUMINAMATH_CALUDE_james_calories_per_minute_l1493_149380

/-- Represents the number of calories burned per minute in a spinning class -/
def calories_per_minute (classes_per_week : ℕ) (hours_per_class : ℚ) (total_calories_per_week : ℕ) : ℚ :=
  let minutes_per_week : ℚ := classes_per_week * hours_per_class * 60
  total_calories_per_week / minutes_per_week

/-- Proves that James burns 7 calories per minute in his spinning class -/
theorem james_calories_per_minute :
  calories_per_minute 3 (3/2) 1890 = 7 := by
sorry

end NUMINAMATH_CALUDE_james_calories_per_minute_l1493_149380


namespace NUMINAMATH_CALUDE_students_per_group_l1493_149358

theorem students_per_group (total : ℕ) (not_picked : ℕ) (groups : ℕ) 
  (h1 : total = 17) 
  (h2 : not_picked = 5) 
  (h3 : groups = 3) :
  (total - not_picked) / groups = 4 := by
sorry

end NUMINAMATH_CALUDE_students_per_group_l1493_149358


namespace NUMINAMATH_CALUDE_power_of_256_l1493_149370

theorem power_of_256 : (256 : ℝ) ^ (5/8 : ℝ) = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_of_256_l1493_149370


namespace NUMINAMATH_CALUDE_square_cloth_trimming_l1493_149323

theorem square_cloth_trimming (x : ℝ) : 
  x > 0 →  -- Ensure positive length
  (x - 6) * (x - 5) = 120 → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_square_cloth_trimming_l1493_149323


namespace NUMINAMATH_CALUDE_inequality_holds_l1493_149366

-- Define the real number a
variable (a : ℝ)

-- Define functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f, g, and a
axiom a_gt_one : a > 1
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom f_minus_g : ∀ x, f x - g x = a^x

-- State the theorem
theorem inequality_holds : g 0 < f 2 ∧ f 2 < f 3 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_l1493_149366


namespace NUMINAMATH_CALUDE_marble_arrangement_l1493_149322

/-- Represents the color of a marble -/
inductive Color
  | Green
  | Blue
  | Red

/-- Represents an arrangement of marbles -/
def Arrangement := List Color

/-- Checks if an arrangement satisfies the equal neighbor condition -/
def satisfiesCondition (arr : Arrangement) : Bool :=
  sorry

/-- Counts the number of valid arrangements for a given number of marbles -/
def countArrangements (totalMarbles : Nat) : Nat :=
  sorry

theorem marble_arrangement :
  let greenMarbles : Nat := 6
  let m : Nat := 12  -- maximum number of additional blue and red marbles
  let totalMarbles : Nat := greenMarbles + m
  let N : Nat := countArrangements totalMarbles
  N = 924 ∧ N % 1000 = 924 := by
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_l1493_149322


namespace NUMINAMATH_CALUDE_only_valid_number_l1493_149341

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ 
  n = (n / 10)^2 + (n % 10)^2 + 13

theorem only_valid_number : ∀ n : ℕ, is_valid_number n ↔ n = 54 := by sorry

end NUMINAMATH_CALUDE_only_valid_number_l1493_149341


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1493_149350

theorem polynomial_expansion (z : ℝ) :
  (3 * z^2 - 4 * z + 1) * (2 * z^3 + 3 * z^2 - 5 * z + 2) =
  6 * z^5 + z^4 - 25 * z^3 + 29 * z^2 - 13 * z + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1493_149350


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_values_positive_l1493_149319

theorem sum_of_cubes_and_values_positive (a b c : ℝ) 
  (hab : a + b > 0) (hac : a + c > 0) (hbc : b + c > 0) : 
  (a^3 + a) + (b^3 + b) + (c^3 + c) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_values_positive_l1493_149319


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1493_149331

theorem smallest_sum_of_sequence (X Y Z W : ℤ) : 
  X > 0 → Y > 0 → Z > 0 →  -- X, Y, Z are positive integers
  (∃ d : ℤ, Y - X = d ∧ Z - Y = d) →  -- X, Y, Z form an arithmetic sequence
  (∃ r : ℚ, Z = Y * r ∧ W = Z * r) →  -- Y, Z, W form a geometric sequence
  Z = (7 * Y) / 4 →  -- Z/Y = 7/4
  (∀ X' Y' Z' W' : ℤ, 
    X' > 0 → Y' > 0 → Z' > 0 →
    (∃ d : ℤ, Y' - X' = d ∧ Z' - Y' = d) →
    (∃ r : ℚ, Z' = Y' * r ∧ W' = Z' * r) →
    Z' = (7 * Y') / 4 →
    X + Y + Z + W ≤ X' + Y' + Z' + W') →
  X + Y + Z + W = 97 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1493_149331


namespace NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l1493_149340

theorem smithtown_left_handed_women_percentage
  (total : ℕ)
  (right_handed : ℕ)
  (left_handed : ℕ)
  (men : ℕ)
  (women : ℕ)
  (h1 : right_handed = 3 * left_handed)
  (h2 : men = 3 * (men + women) / 5)
  (h3 : women = 2 * (men + women) / 5)
  (h4 : total = right_handed + left_handed)
  (h5 : total = men + women)
  (h6 : men ≤ right_handed) :
  left_handed * 100 / total = 25 := by
sorry

end NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l1493_149340


namespace NUMINAMATH_CALUDE_doll_count_l1493_149392

theorem doll_count (vera sophie aida : ℕ) : 
  vera = 20 → 
  sophie = 2 * vera → 
  aida = 2 * sophie → 
  vera + sophie + aida = 140 := by
sorry

end NUMINAMATH_CALUDE_doll_count_l1493_149392


namespace NUMINAMATH_CALUDE_billboard_average_l1493_149361

theorem billboard_average (h1 h2 h3 : ℕ) (h1_val : h1 = 17) (h2_val : h2 = 20) (h3_val : h3 = 23) :
  (h1 + h2 + h3) / 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_billboard_average_l1493_149361


namespace NUMINAMATH_CALUDE_pear_price_l1493_149335

/-- Proves that the price of a pear is $60 given the conditions from the problem -/
theorem pear_price (orange pear banana : ℚ) 
  (h1 : orange - pear = banana)
  (h2 : orange + pear = 120)
  (h3 : 200 * banana + 400 * orange = 24000) : 
  pear = 60 := by
  sorry

end NUMINAMATH_CALUDE_pear_price_l1493_149335


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1493_149321

theorem system_of_equations_solutions :
  -- First system
  (∃ x y : ℝ, 4 * x - y = 1 ∧ y = 2 * x + 3 ∧ x = 2 ∧ y = 7) ∧
  -- Second system
  (∃ x y : ℝ, 2 * x - y = 5 ∧ 7 * x - 3 * y = 20 ∧ x = 5 ∧ y = 5) :=
by
  sorry

#check system_of_equations_solutions

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1493_149321


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l1493_149336

/-- The function f(x) = a^(x-2) + 1 passes through the point (2, 2) for any a > 0 and a ≠ 1 -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 1
  f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l1493_149336


namespace NUMINAMATH_CALUDE_dropped_players_not_necessarily_played_each_other_l1493_149390

/-- Represents a round-robin chess tournament --/
structure ChessTournament where
  n : ℕ  -- Total number of participants
  games_played : ℕ  -- Total number of games played
  dropped_players : ℕ  -- Number of players who dropped out

/-- Calculates the total number of games in a round-robin tournament --/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a specific tournament scenario, dropped players didn't necessarily play each other --/
theorem dropped_players_not_necessarily_played_each_other 
  (t : ChessTournament) 
  (h1 : t.games_played = 23) 
  (h2 : t.dropped_players = 2) 
  (h3 : ∃ k : ℕ, t.n = k + t.dropped_players) 
  (h4 : ∃ m : ℕ, m * t.dropped_players = t.games_played - total_games (t.n - t.dropped_players)) :
  ¬ (∀ dropped_player_games : ℕ, dropped_player_games * t.dropped_players = t.games_played - total_games (t.n - t.dropped_players) → dropped_player_games = t.n - t.dropped_players - 1) :=
sorry

end NUMINAMATH_CALUDE_dropped_players_not_necessarily_played_each_other_l1493_149390


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l1493_149388

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_are_different (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens ≠ ones

def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens + ones

theorem unique_two_digit_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
            digits_are_different n ∧ 
            n^2 = (sum_of_digits n)^3 ∧
            n = 27 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l1493_149388


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l1493_149349

/-- There exist positive integers a and b with a > b > 1 such that
    for all positive integers k, there exists a positive integer n
    where an + b is a k-th power of a positive integer. -/
theorem existence_of_special_integers : ∃ (a b : ℕ), 
  a > b ∧ b > 1 ∧ 
  ∀ (k : ℕ), k > 0 → 
    ∃ (n m : ℕ), n > 0 ∧ m > 0 ∧ a * n + b = m ^ k :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l1493_149349


namespace NUMINAMATH_CALUDE_base6_addition_correct_l1493_149371

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The first number in base 6 -/
def num1 : List Nat := [3, 4, 2, 1]

/-- The second number in base 6 -/
def num2 : List Nat := [4, 5, 2, 5]

/-- The expected sum in base 6 -/
def expectedSum : List Nat := [1, 2, 3, 5, 0]

theorem base6_addition_correct :
  decimalToBase6 (base6ToDecimal num1 + base6ToDecimal num2) = expectedSum := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_correct_l1493_149371


namespace NUMINAMATH_CALUDE_red_balls_count_l1493_149359

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 → 
  prob = 1 / 35 →
  (∃ r : ℕ, r ≤ total ∧ 
    prob = (r * (r - 1) * (r - 2)) / (total * (total - 1) * (total - 2))) →
  (∃ r : ℕ, r = 7 ∧ r ≤ total ∧ 
    prob = (r * (r - 1) * (r - 2)) / (total * (total - 1) * (total - 2))) :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l1493_149359


namespace NUMINAMATH_CALUDE_average_monthly_balance_l1493_149357

def monthly_balances : List ℝ := [120, 150, 180, 150, 210, 180]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 165 := by sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l1493_149357


namespace NUMINAMATH_CALUDE_sector_central_angle_l1493_149363

theorem sector_central_angle (circumference : ℝ) (area : ℝ) :
  circumference = 6 →
  area = 2 →
  ∃ (r l : ℝ),
    l + 2*r = 6 ∧
    (1/2) * l * r = 2 ∧
    (l / r = 1 ∨ l / r = 4) :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1493_149363


namespace NUMINAMATH_CALUDE_unique_solution_l1493_149305

/-- The inequality condition for positive real numbers a, b, c, d, and real number x -/
def inequality_condition (a b c d x : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  ((a^3 / (a^3 + 15*b*c*d))^(1/2) : ℝ) ≥ (a^x / (a^x + b^x + c^x + d^x) : ℝ)

/-- The theorem stating that 15/8 is the only solution -/
theorem unique_solution :
  ∀ x : ℝ, (∀ a b c d : ℝ, inequality_condition a b c d x) ↔ x = 15/8 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1493_149305


namespace NUMINAMATH_CALUDE_expression_evaluation_l1493_149376

theorem expression_evaluation :
  let a : ℤ := 2
  let b : ℤ := -1
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = -14 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1493_149376


namespace NUMINAMATH_CALUDE_four_intersections_iff_l1493_149338

/-- The number of intersection points between x^2 + y^2 = a^2 and y = x^2 - a - 1 -/
def intersection_count (a : ℝ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the condition for exactly four intersection points -/
theorem four_intersections_iff (a : ℝ) :
  intersection_count a = 4 ↔ a > -1/2 :=
by sorry

end NUMINAMATH_CALUDE_four_intersections_iff_l1493_149338


namespace NUMINAMATH_CALUDE_fudge_difference_is_14_ounces_l1493_149384

/-- Conversion factor from pounds to ounces -/
def poundsToOunces : ℚ := 16

/-- Marina's fudge in pounds -/
def marinaFudgePounds : ℚ := 4.5

/-- Amount of fudge Lazlo has less than 4 pounds, in ounces -/
def lazloFudgeDifference : ℚ := 6

/-- Calculates the difference in ounces of fudge between Marina and Lazlo -/
def fudgeDifferenceInOunces : ℚ :=
  marinaFudgePounds * poundsToOunces - (4 * poundsToOunces - lazloFudgeDifference)

theorem fudge_difference_is_14_ounces :
  fudgeDifferenceInOunces = 14 := by
  sorry

end NUMINAMATH_CALUDE_fudge_difference_is_14_ounces_l1493_149384


namespace NUMINAMATH_CALUDE_tetrahedron_colorings_l1493_149318

/-- Represents a coloring of the tetrahedron -/
def Coloring := Fin 7 → Bool

/-- The group of rotational symmetries of a tetrahedron -/
def TetrahedronSymmetry : Type := Unit -- Placeholder, actual implementation would be more complex

/-- Action of a symmetry on a coloring -/
def symmetryAction (s : TetrahedronSymmetry) (c : Coloring) : Coloring :=
  sorry

/-- A coloring is considered fixed under a symmetry if it's unchanged by the symmetry's action -/
def isFixed (s : TetrahedronSymmetry) (c : Coloring) : Prop :=
  symmetryAction s c = c

/-- The number of distinct colorings under rotational symmetry -/
def numDistinctColorings : ℕ :=
  sorry

theorem tetrahedron_colorings : numDistinctColorings = 48 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_colorings_l1493_149318


namespace NUMINAMATH_CALUDE_rotate_point_on_circle_l1493_149325

/-- Given a circle with radius 5 centered at the origin, 
    prove that rotating the point (3,4) by 45 degrees counterclockwise 
    results in the point (-√2/2, 7√2/2) -/
theorem rotate_point_on_circle (P Q : ℝ × ℝ) : 
  P.1^2 + P.2^2 = 25 →  -- P is on the circle
  P = (3, 4) →  -- P starts at (3,4)
  Q.1 = P.1 * (Real.sqrt 2 / 2) - P.2 * (Real.sqrt 2 / 2) →  -- Q is P rotated 45°
  Q.2 = P.1 * (Real.sqrt 2 / 2) + P.2 * (Real.sqrt 2 / 2) →
  Q = (-Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_point_on_circle_l1493_149325


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1493_149324

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 3) (hw : w = 4) (hh : h = 5) :
  Real.sqrt (l^2 + w^2 + h^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1493_149324


namespace NUMINAMATH_CALUDE_girls_same_color_marble_l1493_149320

-- Define the total number of marbles
def total_marbles : ℕ := 4

-- Define the number of white marbles
def white_marbles : ℕ := 2

-- Define the number of black marbles
def black_marbles : ℕ := 2

-- Define the number of girls selecting marbles
def girls : ℕ := 2

-- Define the probability of both girls selecting the same colored marble
def prob_same_color : ℚ := 1 / 3

-- Theorem statement
theorem girls_same_color_marble :
  (total_marbles = white_marbles + black_marbles) →
  (white_marbles = black_marbles) →
  (girls = 2) →
  (prob_same_color = 1 / 3) := by
sorry

end NUMINAMATH_CALUDE_girls_same_color_marble_l1493_149320


namespace NUMINAMATH_CALUDE_martyrs_cemetery_distance_l1493_149316

/-- The distance from the school to the Martyrs' Cemetery in meters -/
def distance : ℝ := 180000

/-- The original speed of the car in meters per minute -/
def original_speed : ℝ := 500

/-- The scheduled travel time in minutes -/
def scheduled_time : ℝ := 120

theorem martyrs_cemetery_distance :
  ∃ (d : ℝ) (v : ℝ),
    d = distance ∧
    v = original_speed ∧
    -- Condition 1: Increased speed by 1/5 after 1 hour
    (60 / v + (d - 60 * v) / (6/5 * v) = scheduled_time - 10) ∧
    -- Condition 2: Increased speed by 1/3 after 60 km
    (60000 / v + (d - 60000) / (4/3 * v) = scheduled_time - 20) ∧
    -- Scheduled time is 2 hours
    scheduled_time = 120 :=
by sorry

end NUMINAMATH_CALUDE_martyrs_cemetery_distance_l1493_149316


namespace NUMINAMATH_CALUDE_ball_travel_distance_l1493_149327

/-- The distance traveled by a ball rolling down a ramp -/
def ballDistance (initialDistance : ℕ) (increase : ℕ) (time : ℕ) : ℕ :=
  let lastTerm := initialDistance + (time - 1) * increase
  time * (initialDistance + lastTerm) / 2

/-- Theorem stating the total distance traveled by the ball -/
theorem ball_travel_distance :
  ballDistance 10 8 25 = 2650 := by
  sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l1493_149327


namespace NUMINAMATH_CALUDE_find_d_l1493_149315

theorem find_d : ∃ d : ℝ, 
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * n^2 + 19 * n - 84 = 0) ∧ 
  (5 * (d - ⌊d⌋)^2 - 26 * (d - ⌊d⌋) + 12 = 0) ∧ 
  (0 ≤ d - ⌊d⌋ ∧ d - ⌊d⌋ < 1) ∧
  d = 3.44 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l1493_149315


namespace NUMINAMATH_CALUDE_largest_special_number_l1493_149303

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_special_number : 
  let n := 4731
  ∀ m : ℕ, m > n → 
    ¬(m < 10000 ∧ 
      (∀ i j : ℕ, i < 4 → j < 4 → i ≠ j → (m / 10^i % 10) ≠ (m / 10^j % 10)) ∧
      is_prime (n / 100) ∧
      is_prime ((n / 1000) * 10 + (n % 10)) ∧
      is_prime ((n / 1000) * 10 + (n / 10 % 10)) ∧
      n % 3 = 0 ∧
      ¬(is_prime n)) :=
by sorry

end NUMINAMATH_CALUDE_largest_special_number_l1493_149303


namespace NUMINAMATH_CALUDE_f_composition_value_l1493_149379

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_composition_value : f (3 * f (-1)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1493_149379
