import Mathlib

namespace NUMINAMATH_CALUDE_city_a_sand_amount_l1263_126387

/-- The amount of sand received by City A, given the total amount and amounts received by other cities -/
theorem city_a_sand_amount (total sand_b sand_c sand_d : ℝ) (h1 : total = 95) 
  (h2 : sand_b = 26) (h3 : sand_c = 24.5) (h4 : sand_d = 28) : 
  total - (sand_b + sand_c + sand_d) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_city_a_sand_amount_l1263_126387


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1263_126337

theorem imaginary_part_of_complex_number : 
  Complex.im (1 - Complex.I * Real.sqrt 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1263_126337


namespace NUMINAMATH_CALUDE_probability_sum_10_l1263_126353

-- Define a die roll as a natural number between 1 and 6
def DieRoll : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define a function to check if the sum of three die rolls is 10
def sumIs10 (roll1 roll2 roll3 : DieRoll) : Prop :=
  roll1.val + roll2.val + roll3.val = 10

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := 216

-- Define the number of favorable outcomes (sum is 10)
def favorableOutcomes : ℕ := 27

-- Theorem statement
theorem probability_sum_10 :
  (favorableOutcomes : ℚ) / totalOutcomes = 27 / 216 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_10_l1263_126353


namespace NUMINAMATH_CALUDE_equation_root_l1263_126392

theorem equation_root (x m : ℝ) : 
  (2 / (x - 2) = 1 - m / (x - 2)) → 
  (x > 0) → 
  (m = -2) := by
sorry

end NUMINAMATH_CALUDE_equation_root_l1263_126392


namespace NUMINAMATH_CALUDE_fraction_of_repeating_decimals_l1263_126386

def repeating_decimal_142857 : ℚ := 142857 / 999999
def repeating_decimal_857143 : ℚ := 857143 / 999999

theorem fraction_of_repeating_decimals : 
  (repeating_decimal_142857) / (2 + repeating_decimal_857143) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_repeating_decimals_l1263_126386


namespace NUMINAMATH_CALUDE_decimal_333_to_octal_l1263_126319

def decimal_to_octal (n : Nat) : Nat :=
  let rec aux (m : Nat) (acc : Nat) : Nat :=
    if m = 0 then acc
    else aux (m / 8) (acc * 10 + m % 8)
  aux n 0

theorem decimal_333_to_octal :
  decimal_to_octal 333 = 515 := by
sorry

end NUMINAMATH_CALUDE_decimal_333_to_octal_l1263_126319


namespace NUMINAMATH_CALUDE_fruit_spending_sum_l1263_126391

/-- The total amount Mary spent on fruits after discounts -/
def total_spent : ℝ := 52.09

/-- The amount Mary paid for berries -/
def berries_price : ℝ := 11.08

/-- The amount Mary paid for apples -/
def apples_price : ℝ := 14.33

/-- The amount Mary paid for peaches -/
def peaches_price : ℝ := 9.31

/-- The amount Mary paid for grapes -/
def grapes_price : ℝ := 7.50

/-- The amount Mary paid for bananas -/
def bananas_price : ℝ := 5.25

/-- The amount Mary paid for pineapples -/
def pineapples_price : ℝ := 4.62

/-- Theorem stating that the sum of individual fruit prices equals the total spent -/
theorem fruit_spending_sum :
  berries_price + apples_price + peaches_price + grapes_price + bananas_price + pineapples_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_fruit_spending_sum_l1263_126391


namespace NUMINAMATH_CALUDE_factor_polynomial_l1263_126364

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1263_126364


namespace NUMINAMATH_CALUDE_whitney_max_sets_l1263_126317

/-- Represents the number of items Whitney has -/
structure Inventory where
  tshirts : ℕ
  buttons : ℕ
  stickers : ℕ

/-- Represents the composition of each set -/
structure SetComposition where
  tshirts : ℕ
  buttons : ℕ
  stickers : ℕ

def max_sets (inv : Inventory) (comp : SetComposition) : ℕ :=
  min (inv.tshirts / comp.tshirts)
      (min (inv.buttons / comp.buttons) (inv.stickers / comp.stickers))

/-- Theorem stating that the maximum number of sets Whitney can make is 5 -/
theorem whitney_max_sets :
  let inv : Inventory := { tshirts := 5, buttons := 24, stickers := 12 }
  let comp : SetComposition := { tshirts := 1, buttons := 2, stickers := 1 }
  max_sets inv comp = 5 := by
  sorry

end NUMINAMATH_CALUDE_whitney_max_sets_l1263_126317


namespace NUMINAMATH_CALUDE_octopus_leg_counts_l1263_126385

/-- Represents the possible number of legs an octopus can have -/
inductive LegCount
  | six
  | seven
  | eight

/-- Represents an octopus with a name and a number of legs -/
structure Octopus :=
  (name : String)
  (legs : LegCount)

/-- Determines if an octopus is telling the truth based on its leg count -/
def isTruthful (o : Octopus) : Bool :=
  match o.legs with
  | LegCount.seven => false
  | _ => true

/-- Converts LegCount to a natural number -/
def legCountToNat (lc : LegCount) : Nat :=
  match lc with
  | LegCount.six => 6
  | LegCount.seven => 7
  | LegCount.eight => 8

/-- The main theorem about the octopuses' leg counts -/
theorem octopus_leg_counts (blue green red yellow : Octopus)
  (h1 : blue.name = "Blue" ∧ green.name = "Green" ∧ red.name = "Red" ∧ yellow.name = "Yellow")
  (h2 : (isTruthful blue) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 25))
  (h3 : (isTruthful green) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 26))
  (h4 : (isTruthful red) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 27))
  (h5 : (isTruthful yellow) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 28)) :
  blue.legs = LegCount.seven ∧ green.legs = LegCount.seven ∧ red.legs = LegCount.six ∧ yellow.legs = LegCount.seven :=
sorry

end NUMINAMATH_CALUDE_octopus_leg_counts_l1263_126385


namespace NUMINAMATH_CALUDE_prism_volume_l1263_126310

/-- A right rectangular prism with given face areas has the specified volume -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1263_126310


namespace NUMINAMATH_CALUDE_expression_evaluation_l1263_126328

theorem expression_evaluation :
  let x : ℤ := -2
  (x - 2)^2 - 4*x*(x - 1) + (2*x + 1)*(2*x - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1263_126328


namespace NUMINAMATH_CALUDE_cuttable_triangle_type_l1263_126378

/-- A triangle that can be cut into two parts formable into a rectangle -/
structure CuttableTriangle where
  /-- The triangle can be cut into two parts -/
  can_be_cut : Bool
  /-- The parts can be rearranged into a rectangle -/
  forms_rectangle : Bool

/-- Types of triangles -/
inductive TriangleType
  | Right
  | Isosceles
  | Other

/-- Theorem stating that a cuttable triangle must be right or isosceles -/
theorem cuttable_triangle_type (t : CuttableTriangle) :
  t.can_be_cut ∧ t.forms_rectangle →
  (∃ tt : TriangleType, tt = TriangleType.Right ∨ tt = TriangleType.Isosceles) :=
by
  sorry

end NUMINAMATH_CALUDE_cuttable_triangle_type_l1263_126378


namespace NUMINAMATH_CALUDE_characterize_function_l1263_126325

open Set Function Real

-- Define the interval (1,∞)
def OpenOneInfty : Set ℝ := {x : ℝ | x > 1}

-- Define the property for the function
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ OpenOneInfty → y ∈ OpenOneInfty →
    (x^2 ≤ y ∧ y ≤ x^3) → ((f x)^2 ≤ f y ∧ f y ≤ (f x)^3)

-- The main theorem
theorem characterize_function :
  ∀ f : ℝ → ℝ, (∀ x, x ∈ OpenOneInfty → f x ∈ OpenOneInfty) →
    SatisfiesProperty f →
    ∃ k : ℝ, k > 0 ∧ ∀ x ∈ OpenOneInfty, f x = exp (k * log x) := by
  sorry

end NUMINAMATH_CALUDE_characterize_function_l1263_126325


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1263_126381

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b : V) (x y : ℝ) 
  (h_not_collinear : ¬ ∃ (k : ℝ), b = k • a) 
  (h_eq : (3 * x - 4 * y) • a + (2 * x - 3 * y) • b = 6 • a + 3 • b) :
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l1263_126381


namespace NUMINAMATH_CALUDE_football_team_size_l1263_126326

/-- Represents the number of players on a football team -/
def total_players : ℕ := 70

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 31

/-- Represents the total number of right-handed players -/
def right_handed_total : ℕ := 57

/-- Represents the number of left-handed players (non-throwers) -/
def left_handed_non_throwers : ℕ := (total_players - throwers) / 3

/-- Represents the number of right-handed non-throwers -/
def right_handed_non_throwers : ℕ := right_handed_total - throwers

theorem football_team_size :
  total_players = throwers + left_handed_non_throwers + right_handed_non_throwers ∧
  left_handed_non_throwers * 2 = right_handed_non_throwers ∧
  right_handed_total = throwers + right_handed_non_throwers :=
by
  sorry

end NUMINAMATH_CALUDE_football_team_size_l1263_126326


namespace NUMINAMATH_CALUDE_sector_central_angle_l1263_126316

/-- Given a sector with radius 1 and perimeter 4, its central angle in radians has an absolute value of 2. -/
theorem sector_central_angle (r : ℝ) (L : ℝ) (α : ℝ) : 
  r = 1 → L = 4 → L = r * α + 2 * r → |α| = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1263_126316


namespace NUMINAMATH_CALUDE_arithmetic_progression_log_range_l1263_126357

theorem arithmetic_progression_log_range (x y : ℝ) : 
  (∃ k : ℝ, Real.log 2 - k = Real.log (Real.sin x - 1/3) ∧ 
             Real.log (Real.sin x - 1/3) - k = Real.log (1 - y)) →
  (y ≥ 7/9 ∧ ∀ M : ℝ, ∃ y' ≥ M, 
    ∃ k' : ℝ, Real.log 2 - k' = Real.log (Real.sin x - 1/3) ∧ 
             Real.log (Real.sin x - 1/3) - k' = Real.log (1 - y')) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_log_range_l1263_126357


namespace NUMINAMATH_CALUDE_purple_balls_count_l1263_126383

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (p : ℚ) :
  total = 100 ∧
  white = 10 ∧
  green = 30 ∧
  yellow = 10 ∧
  red = 47 ∧
  p = 1/2 ∧
  p = (white + green + yellow : ℚ) / total →
  ∃ purple : ℕ, purple = 3 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_purple_balls_count_l1263_126383


namespace NUMINAMATH_CALUDE_solve_equation_l1263_126345

theorem solve_equation (b : ℚ) (h : b + b/4 = 10/4) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1263_126345


namespace NUMINAMATH_CALUDE_sandy_paint_area_l1263_126380

/-- The area Sandy needs to paint on her bedroom wall -/
def area_to_paint (wall_height wall_length window_width window_height : ℝ) : ℝ :=
  wall_height * wall_length - window_width * window_height

/-- Theorem stating the area Sandy needs to paint -/
theorem sandy_paint_area :
  area_to_paint 9 12 2 4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sandy_paint_area_l1263_126380


namespace NUMINAMATH_CALUDE_square_area_on_line_and_parabola_l1263_126397

/-- A square with one side on y = x + 4 and two vertices on y² = x has area 18 or 50 -/
theorem square_area_on_line_and_parabola :
  ∀ (A B C D : ℝ × ℝ),
    (∃ (y₁ y₂ : ℝ),
      A.2 = A.1 + 4 ∧
      B.2 = B.1 + 4 ∧
      C = (y₁^2, y₁) ∧
      D = (y₂^2, y₂) ∧
      (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
      (C.1 - B.1)^2 + (C.2 - B.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
      (D.1 - C.1)^2 + (D.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2) →
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 18) ∨ ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 50) :=
by sorry


end NUMINAMATH_CALUDE_square_area_on_line_and_parabola_l1263_126397


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1263_126334

theorem smaller_number_proof (x y m : ℝ) 
  (h1 : x - y = 9)
  (h2 : x + y = 46)
  (h3 : x = m * y) : 
  min x y = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1263_126334


namespace NUMINAMATH_CALUDE_maple_trees_after_planting_l1263_126352

/-- The number of maple trees in the park after planting is equal to the sum of 
    the initial number of trees and the number of newly planted trees. -/
theorem maple_trees_after_planting 
  (initial_trees : ℕ) 
  (planted_trees : ℕ) 
  (h1 : initial_trees = 53) 
  (h2 : planted_trees = 11) : 
  initial_trees + planted_trees = 64 := by
  sorry

#check maple_trees_after_planting

end NUMINAMATH_CALUDE_maple_trees_after_planting_l1263_126352


namespace NUMINAMATH_CALUDE_second_term_arithmetic_sequence_l1263_126355

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 1 + (n * (n - 1) / 2) * (a n - a 1)

theorem second_term_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_first_term : a 1 = -2010) 
  (h_sum_condition : sum_arithmetic_sequence a 2010 / 2010 - sum_arithmetic_sequence a 2008 / 2008 = 2) :
  a 2 = -2008 :=
sorry

end NUMINAMATH_CALUDE_second_term_arithmetic_sequence_l1263_126355


namespace NUMINAMATH_CALUDE_golden_ratio_percentage_l1263_126302

theorem golden_ratio_percentage (a b : ℝ) (h : a > 0) (h' : b > 0) :
  b / a = a / (a + b) → b / a = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_percentage_l1263_126302


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1263_126399

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) - Real.exp x - 5 * x

theorem solution_set_of_inequality :
  {x : ℝ | f (x^2) + f (-x-6) < 0} = {x : ℝ | x < -2 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1263_126399


namespace NUMINAMATH_CALUDE_cookie_jar_spending_ratio_l1263_126367

/-- Proves that the ratio of Martha's spending to Doris' spending is 1:2 --/
theorem cookie_jar_spending_ratio 
  (initial_amount : ℕ) 
  (doris_spent : ℕ) 
  (final_amount : ℕ) 
  (h1 : initial_amount = 24)
  (h2 : doris_spent = 6)
  (h3 : final_amount = 15) :
  ∃ (martha_spent : ℕ), 
    martha_spent = initial_amount - doris_spent - final_amount ∧
    martha_spent * 2 = doris_spent := by
  sorry

#check cookie_jar_spending_ratio

end NUMINAMATH_CALUDE_cookie_jar_spending_ratio_l1263_126367


namespace NUMINAMATH_CALUDE_max_elements_sum_l1263_126305

/-- A shape formed by adding a pyramid to a rectangular prism -/
structure PrismPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_new_faces : Nat
  pyramid_new_edges : Nat
  pyramid_new_vertex : Nat

/-- The total number of exterior elements in the combined shape -/
def total_elements (shape : PrismPyramid) : Nat :=
  (shape.prism_faces - 1 + shape.pyramid_new_faces) +
  (shape.prism_edges + shape.pyramid_new_edges) +
  (shape.prism_vertices + shape.pyramid_new_vertex)

/-- Theorem stating the maximum sum of exterior elements -/
theorem max_elements_sum :
  ∀ shape : PrismPyramid,
  shape.prism_faces = 6 →
  shape.prism_edges = 12 →
  shape.prism_vertices = 8 →
  shape.pyramid_new_faces ≤ 4 →
  shape.pyramid_new_edges ≤ 4 →
  shape.pyramid_new_vertex ≤ 1 →
  total_elements shape ≤ 34 :=
sorry

end NUMINAMATH_CALUDE_max_elements_sum_l1263_126305


namespace NUMINAMATH_CALUDE_equation_solution_l1263_126308

theorem equation_solution (x : ℝ) : x > 0 → (5 * x^(1/4) - 3 * (x / x^(3/4)) = 10 + x^(1/4)) ↔ x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1263_126308


namespace NUMINAMATH_CALUDE_river_flow_volume_l1263_126324

/-- Calculates the volume of water flowing into the sea per minute for a river with given dimensions and flow rate. -/
theorem river_flow_volume 
  (depth : ℝ) 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (h_depth : depth = 3) 
  (h_width : width = 32) 
  (h_flow_rate : flow_rate_kmph = 2) : 
  depth * width * (flow_rate_kmph * 1000 / 60) = 3200 := by
  sorry

#check river_flow_volume

end NUMINAMATH_CALUDE_river_flow_volume_l1263_126324


namespace NUMINAMATH_CALUDE_boxes_with_neither_l1263_126377

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ)
  (h_total : total = 15)
  (h_markers : markers = 10)
  (h_erasers : erasers = 5)
  (h_both : both = 4) :
  total - (markers + erasers - both) = 4 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l1263_126377


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l1263_126372

-- Define the given points and circles
def M : ℝ × ℝ := (2, -2)
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 3
def circle_2 (x y : ℝ) : Prop := x^2 + y^2 + 3*x = 0

-- Define the resulting circle and line
def result_circle (x y : ℝ) : Prop := 3*x^2 + 3*y^2 - 5*x - 14 = 0
def line_AB (x y : ℝ) : Prop := 2*x - 2*y = 3

theorem circle_and_line_theorem :
  -- (1) The result_circle passes through M and intersects with circle_O and circle_2
  (result_circle M.1 M.2) ∧
  (∃ x y : ℝ, result_circle x y ∧ circle_O x y) ∧
  (∃ x y : ℝ, result_circle x y ∧ circle_2 x y) ∧
  -- (2) line_AB is tangent to circle_O at two points
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧
    (∀ x y : ℝ, line_AB x y → circle_O x y → (x, y) = A ∨ (x, y) = B)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_theorem_l1263_126372


namespace NUMINAMATH_CALUDE_intersecting_lines_sum_of_intercepts_l1263_126315

/-- Two lines intersecting at (3, 3) have the sum of their y-intercepts equal to 4 -/
theorem intersecting_lines_sum_of_intercepts (c d : ℝ) : 
  (3 = (1/3) * 3 + c) ∧ (3 = (1/3) * 3 + d) → c + d = 4 := by
  sorry

#check intersecting_lines_sum_of_intercepts

end NUMINAMATH_CALUDE_intersecting_lines_sum_of_intercepts_l1263_126315


namespace NUMINAMATH_CALUDE_salad_ratio_l1263_126344

theorem salad_ratio (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ) : 
  mushrooms = 3 →
  cherry_tomatoes = 2 * mushrooms →
  pickles = 4 * cherry_tomatoes →
  bacon_bits = 4 * pickles →
  red_bacon_bits = 32 →
  (red_bacon_bits : ℚ) / bacon_bits = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_salad_ratio_l1263_126344


namespace NUMINAMATH_CALUDE_train_length_l1263_126335

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length (train_speed : Real) (bridge_crossing_time : Real) (bridge_length : Real) :
  train_speed = 72 * 1000 / 3600 ∧ 
  bridge_crossing_time = 12.099 ∧ 
  bridge_length = 132 →
  train_speed * bridge_crossing_time - bridge_length = 110 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1263_126335


namespace NUMINAMATH_CALUDE_equation_solution_l1263_126343

theorem equation_solution :
  ∃ x : ℚ, (5 * x + 9 * x = 360 - 7 * (x + 4)) ∧ (x = 332 / 21) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1263_126343


namespace NUMINAMATH_CALUDE_activity_participation_l1263_126354

theorem activity_participation (total : ℕ) (books songs movies : ℕ) 
  (books_songs books_movies songs_movies : ℕ) (all_three : ℕ) : 
  total = 200 → 
  books = 80 → 
  songs = 60 → 
  movies = 30 → 
  books_songs = 25 → 
  books_movies = 15 → 
  songs_movies = 20 → 
  all_three = 10 → 
  books + songs + movies - books_songs - books_movies - songs_movies + all_three = 120 :=
by sorry

end NUMINAMATH_CALUDE_activity_participation_l1263_126354


namespace NUMINAMATH_CALUDE_fraction_sum_equals_point_three_l1263_126323

theorem fraction_sum_equals_point_three :
  (2 : ℚ) / 20 + (4 : ℚ) / 40 + (9 : ℚ) / 90 = (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_point_three_l1263_126323


namespace NUMINAMATH_CALUDE_circular_field_diameter_l1263_126342

/-- The diameter of a circular field given the fencing cost per meter and total fencing cost -/
theorem circular_field_diameter 
  (cost_per_meter : ℝ) 
  (total_cost : ℝ) 
  (h_cost : cost_per_meter = 3) 
  (h_total : total_cost = 395.84067435231395) : 
  ∃ (diameter : ℝ), abs (diameter - 42) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_circular_field_diameter_l1263_126342


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l1263_126376

/-- Defines an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Defines a line -/
structure Line where
  k : ℝ
  m : ℝ

/-- Theorem statement -/
theorem ellipse_fixed_point_intersection 
  (E : Ellipse) 
  (h_point : E.a^2 + (3/2)^2 / E.b^2 = 1) 
  (h_ecc : (E.a^2 - E.b^2) / E.a^2 = 1/4) 
  (l : Line) 
  (h_intersect : ∃ (M N : ℝ × ℝ), M ≠ N ∧ 
    M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 ∧
    N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1 ∧
    M.2 = l.k * M.1 + l.m ∧
    N.2 = l.k * N.1 + l.m)
  (h_perp : ∀ (M N : ℝ × ℝ), 
    M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 →
    N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1 →
    M.2 = l.k * M.1 + l.m →
    N.2 = l.k * N.1 + l.m →
    (M.1 - E.a) * (N.1 - E.a) + M.2 * N.2 = 0) :
  l.k * (2/7) + l.m = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l1263_126376


namespace NUMINAMATH_CALUDE_ellipse_equation_form_l1263_126360

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  center_x : ℝ
  center_y : ℝ
  foci_on_axes : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

-- Define the conditions
def satisfies_conditions (e : Ellipse) : Prop :=
  e.center_x = 0 ∧
  e.center_y = 0 ∧
  e.foci_on_axes ∧
  e.eccentricity = Real.sqrt 3 / 2 ∧
  e.passes_through = (2, 0)

-- Define the equation of the ellipse
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.center_x)^2 / e.a^2 + (y - e.center_y)^2 / e.b^2 = 1

-- Theorem statement
theorem ellipse_equation_form (e : Ellipse) :
  satisfies_conditions e →
  (∀ x y, ellipse_equation e x y ↔ (x^2 / 4 + y^2 = 1 ∨ x^2 / 4 + y^2 / 16 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_form_l1263_126360


namespace NUMINAMATH_CALUDE_power_equality_l1263_126338

theorem power_equality (q : ℕ) (h : 81^7 = 3^q) : q = 28 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1263_126338


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1263_126339

theorem sqrt_difference_equality : 
  Real.sqrt (9/2) - Real.sqrt (8/5) = (15 * Real.sqrt 2 - 4 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1263_126339


namespace NUMINAMATH_CALUDE_product_sum_fractions_l1263_126336

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l1263_126336


namespace NUMINAMATH_CALUDE_ava_distance_covered_l1263_126365

/-- Represents the race scenario where Aubrey and Ava are running --/
structure RaceScenario where
  race_length : ℝ  -- Length of the race in kilometers
  ava_remaining : ℝ  -- Distance Ava has left to finish in meters

/-- Calculates the distance Ava has covered in meters --/
def distance_covered (scenario : RaceScenario) : ℝ :=
  scenario.race_length * 1000 - scenario.ava_remaining

/-- Theorem stating that Ava covered 833 meters in the given scenario --/
theorem ava_distance_covered (scenario : RaceScenario)
  (h1 : scenario.race_length = 1)
  (h2 : scenario.ava_remaining = 167) :
  distance_covered scenario = 833 := by
  sorry

end NUMINAMATH_CALUDE_ava_distance_covered_l1263_126365


namespace NUMINAMATH_CALUDE_max_remainder_division_by_nine_l1263_126322

theorem max_remainder_division_by_nine (n : ℕ) : 
  n / 9 = 6 → n % 9 ≤ 8 ∧ ∃ m : ℕ, m / 9 = 6 ∧ m % 9 = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_division_by_nine_l1263_126322


namespace NUMINAMATH_CALUDE_fraction_division_l1263_126301

theorem fraction_division : (4 : ℚ) / 5 / ((8 : ℚ) / 15) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l1263_126301


namespace NUMINAMATH_CALUDE_complex_multiplication_sum_l1263_126393

theorem complex_multiplication_sum (z : ℂ) (a b : ℝ) :
  z = 5 + 3 * I →
  I * z = a + b * I →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_multiplication_sum_l1263_126393


namespace NUMINAMATH_CALUDE_square_root_divided_by_two_l1263_126382

theorem square_root_divided_by_two : Real.sqrt 16 / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_two_l1263_126382


namespace NUMINAMATH_CALUDE_expression_simplification_l1263_126359

/-- Given real numbers x and y, prove that the expression
    ((x² + y²)(x² - y²)) / ((x² + y²) + (x² - y²)) + ((x² + y²) + (x² - y²)) / ((x² + y²)(x² - y²))
    simplifies to (x⁴ + y⁴)² / (2x²(x⁴ - y⁴)) -/
theorem expression_simplification (x y : ℝ) (h : x ≠ 0) :
  let P := x^2 + y^2
  let Q := x^2 - y^2
  (P * Q) / (P + Q) + (P + Q) / (P * Q) = (x^4 + y^4)^2 / (2 * x^2 * (x^4 - y^4)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1263_126359


namespace NUMINAMATH_CALUDE_tims_children_treats_l1263_126332

/-- The total number of treats Tim's children get while trick-or-treating -/
def total_treats (num_children : ℕ) (hours_out : ℕ) (houses_per_hour : ℕ) (treats_per_kid : ℕ) : ℕ :=
  num_children * hours_out * houses_per_hour * treats_per_kid

/-- Theorem stating that Tim's children get 180 treats in total -/
theorem tims_children_treats : 
  total_treats 3 4 5 3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_tims_children_treats_l1263_126332


namespace NUMINAMATH_CALUDE_model_height_is_58_l1263_126346

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the Empire State Building in feet -/
def actual_height : ℕ := 1454

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The height of the scale model rounded to the nearest foot -/
def model_height : ℕ := (round_to_nearest ((actual_height : ℚ) / scale_ratio)).natAbs

theorem model_height_is_58 : model_height = 58 := by sorry

end NUMINAMATH_CALUDE_model_height_is_58_l1263_126346


namespace NUMINAMATH_CALUDE_vector_statements_false_l1263_126398

open RealInnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def is_unit_vector (v : V) : Prop := ‖v‖ = 1

def parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w

theorem vector_statements_false (a₀ : V) (h : is_unit_vector a₀) :
  (∃ (a : V), a ≠ ‖a‖ • a₀) ∧
  (∃ (a : V), parallel a a₀ ∧ a ≠ ‖a‖ • a₀) ∧
  (∃ (a : V), parallel a a₀ ∧ is_unit_vector a ∧ a ≠ a₀) := by
  sorry

end NUMINAMATH_CALUDE_vector_statements_false_l1263_126398


namespace NUMINAMATH_CALUDE_characterization_of_matrices_with_power_in_S_l1263_126384

-- Define the set S
def S : Set (Matrix (Fin 2) (Fin 2) ℝ) :=
  {M | ∃ (a r : ℝ), M = !![a, a+r; a+2*r, a+3*r]}

-- Define the property of M^k being in S for some k > 1
def has_power_in_S (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ (M ^ k) ∈ S

-- Main theorem
theorem characterization_of_matrices_with_power_in_S :
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ),
  M ∈ S → (has_power_in_S M ↔ 
    (∃ (c : ℝ), M = c • !![1, 1; 1, 1]) ∨
    (∃ (c : ℝ), M = c • !![-3, -1; 1, 3])) :=
by sorry

end NUMINAMATH_CALUDE_characterization_of_matrices_with_power_in_S_l1263_126384


namespace NUMINAMATH_CALUDE_max_median_redistribution_l1263_126330

theorem max_median_redistribution (x : ℕ) :
  let initial_amounts : List ℕ := [28, 72, 98, x]
  let total : ℕ := initial_amounts.sum
  let redistributed : ℚ := (total : ℚ) / 4
  (∀ (a : ℕ), a ∈ initial_amounts → (a : ℚ) ≤ redistributed) →
  redistributed ≤ 98 →
  x ≤ 194 →
  (x = 194 → redistributed = 98) :=
by sorry

end NUMINAMATH_CALUDE_max_median_redistribution_l1263_126330


namespace NUMINAMATH_CALUDE_triangle_theorem_l1263_126371

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.a + t.b - t.c) * (t.a + t.b + t.c) = t.a * t.b)
  (h2 : t.c = 2 * t.a * Real.cos t.B)
  (h3 : t.b = 2) : 
  t.C = 2 * Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1263_126371


namespace NUMINAMATH_CALUDE_batsman_average_after_12_innings_l1263_126313

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat

/-- Calculates the average score of a batsman -/
def calculateAverage (b : Batsman) : Nat :=
  b.totalRuns / b.innings

/-- Theorem: A batsman's average after 12 innings is 70 runs -/
theorem batsman_average_after_12_innings (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.totalRuns = calculateAverage b * 11 + 92)
  (h3 : b.averageIncrease = 2)
  : calculateAverage b = 70 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_12_innings_l1263_126313


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l1263_126341

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l1263_126341


namespace NUMINAMATH_CALUDE_min_trees_triangular_plot_l1263_126309

/-- Given a triangular plot with 5 trees planted on each side, 
    the minimum number of trees that can be planted is 12. -/
theorem min_trees_triangular_plot : 
  ∀ (trees_per_side : ℕ), 
  trees_per_side = 5 → 
  (∃ (min_trees : ℕ), 
    min_trees = 12 ∧ 
    ∀ (total_trees : ℕ), 
      (total_trees ≥ min_trees ∧ 
       ∃ (trees_on_edges : ℕ), 
         trees_on_edges = total_trees - 3 ∧ 
         trees_on_edges % 3 = 0 ∧ 
         trees_on_edges / 3 + 1 = trees_per_side)) :=
by sorry

end NUMINAMATH_CALUDE_min_trees_triangular_plot_l1263_126309


namespace NUMINAMATH_CALUDE_least_integer_proof_l1263_126362

/-- The least positive integer divisible by all numbers from 1 to 22 and 25 to 30 -/
def least_integer : ℕ := 1237834741500

/-- The set of divisors from 1 to 30, excluding 23 and 24 -/
def divisors : Set ℕ := {n : ℕ | n ∈ Finset.range 31 ∧ n ≠ 23 ∧ n ≠ 24}

theorem least_integer_proof :
  (∀ n ∈ divisors, least_integer % n = 0) ∧
  (∀ m : ℕ, m < least_integer →
    ∃ k ∈ divisors, m % k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_least_integer_proof_l1263_126362


namespace NUMINAMATH_CALUDE_complete_square_sum_l1263_126373

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 + 6*x - 9 = 0 ↔ (x + b)^2 = c) → 
  b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1263_126373


namespace NUMINAMATH_CALUDE_two_thousand_five_is_334th_term_l1263_126390

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem two_thousand_five_is_334th_term :
  arithmetic_sequence 7 6 334 = 2005 :=
by sorry

end NUMINAMATH_CALUDE_two_thousand_five_is_334th_term_l1263_126390


namespace NUMINAMATH_CALUDE_dart_target_probability_l1263_126312

theorem dart_target_probability (n : ℕ) : 
  (n : ℝ) * π / (n : ℝ)^2 ≥ (1 : ℝ) / 2 → n ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_dart_target_probability_l1263_126312


namespace NUMINAMATH_CALUDE_s_99_digits_l1263_126327

/-- s(n) is the number formed by concatenating the first n perfect squares -/
def s (n : ℕ) : ℕ := sorry

/-- Count the number of digits in a natural number -/
def countDigits (n : ℕ) : ℕ := sorry

/-- The main theorem: s(99) has 353 digits -/
theorem s_99_digits : countDigits (s 99) = 353 := by sorry

end NUMINAMATH_CALUDE_s_99_digits_l1263_126327


namespace NUMINAMATH_CALUDE_infinite_primes_with_property_l1263_126363

theorem infinite_primes_with_property : 
  ∃ (S : Set Nat), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (Set.Infinite S) ∧ 
    (∀ p ∈ S, ∃ n : Nat, ¬(n ∣ (p - 1)) ∧ (p ∣ (Nat.factorial n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_with_property_l1263_126363


namespace NUMINAMATH_CALUDE_tom_spending_l1263_126321

def apple_count : ℕ := 4
def egg_count : ℕ := 6
def bread_count : ℕ := 3
def cheese_count : ℕ := 2
def chicken_count : ℕ := 1

def apple_price : ℚ := 1
def egg_price : ℚ := 0.5
def bread_price : ℚ := 3
def cheese_price : ℚ := 6
def chicken_price : ℚ := 8

def coupon_threshold : ℚ := 40
def coupon_value : ℚ := 10

def total_cost : ℚ :=
  apple_count * apple_price +
  egg_count * egg_price +
  bread_count * bread_price +
  cheese_count * cheese_price +
  chicken_count * chicken_price

theorem tom_spending :
  (if total_cost ≥ coupon_threshold then total_cost - coupon_value else total_cost) = 36 := by
  sorry

end NUMINAMATH_CALUDE_tom_spending_l1263_126321


namespace NUMINAMATH_CALUDE_quadratic_equations_one_common_root_l1263_126395

theorem quadratic_equations_one_common_root 
  (a b c d : ℝ) : 
  (∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + c*x + d = 0) ↔ 
  ((a*d - b*c)*(c - a) = (b - d)^2 ∧ (a*d - b*c)*(c - a) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_one_common_root_l1263_126395


namespace NUMINAMATH_CALUDE_min_keys_required_l1263_126394

/-- Represents a hotel with rooms and guests -/
structure Hotel where
  rooms : ℕ
  guests : ℕ

/-- Represents the key distribution system for the hotel -/
structure KeyDistribution where
  hotel : Hotel
  keys : ℕ
  returningGuests : ℕ

/-- Checks if the key distribution is valid for the hotel -/
def isValidDistribution (kd : KeyDistribution) : Prop :=
  kd.returningGuests ≤ kd.hotel.guests ∧
  kd.returningGuests ≤ kd.hotel.rooms ∧
  kd.keys ≥ kd.hotel.rooms * (kd.hotel.guests - kd.hotel.rooms + 1)

/-- Theorem: The minimum number of keys required for the given hotel scenario is 990 -/
theorem min_keys_required (h : Hotel) (kd : KeyDistribution) 
  (hrooms : h.rooms = 90)
  (hguests : h.guests = 100)
  (hreturning : kd.returningGuests = 90)
  (hhotel : kd.hotel = h)
  (hvalid : isValidDistribution kd) :
  kd.keys ≥ 990 := by
  sorry

end NUMINAMATH_CALUDE_min_keys_required_l1263_126394


namespace NUMINAMATH_CALUDE_least_sum_pqr_l1263_126366

theorem least_sum_pqr (p q r : ℕ) : 
  p > 1 → q > 1 → r > 1 → 
  17 * (p + 1) = 28 * (q + 1) ∧ 28 * (q + 1) = 35 * (r + 1) →
  ∀ p' q' r' : ℕ, 
    p' > 1 → q' > 1 → r' > 1 → 
    17 * (p' + 1) = 28 * (q' + 1) ∧ 28 * (q' + 1) = 35 * (r' + 1) →
    p + q + r ≤ p' + q' + r' ∧ p + q + r = 290 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_pqr_l1263_126366


namespace NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l1263_126350

theorem tripled_base_doubled_exponent 
  (a b x : ℝ) 
  (hb : b ≠ 0) 
  (hr : (3 * a) ^ (2 * b) = a ^ b * x ^ b) : 
  x = 9 * a := by sorry

end NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l1263_126350


namespace NUMINAMATH_CALUDE_one_in_range_of_f_l1263_126303

/-- The function f(x) = x^2 + bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 1

/-- Theorem: For all real numbers b, 1 is always in the range of f(x) = x^2 + bx - 1 -/
theorem one_in_range_of_f (b : ℝ) : ∃ x : ℝ, f b x = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_in_range_of_f_l1263_126303


namespace NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l1263_126374

/-- Number of lines of symmetry for a given shape -/
def linesOfSymmetry (shape : String) : ℕ :=
  match shape with
  | "regular pentagon" => 5
  | "parallelogram" => 0
  | "oval" => 2
  | "right triangle" => 0
  | "regular hexagon" => 6
  | _ => 0

/-- The set of shapes we're considering -/
def shapes : List String := ["regular pentagon", "parallelogram", "oval", "right triangle", "regular hexagon"]

theorem regular_hexagon_most_symmetry :
  ∀ s ∈ shapes, linesOfSymmetry "regular hexagon" ≥ linesOfSymmetry s :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l1263_126374


namespace NUMINAMATH_CALUDE_mary_height_to_grow_l1263_126368

/-- The problem of calculating how much Mary needs to grow to ride Kingda Ka -/
theorem mary_height_to_grow (min_height brother_height : ℝ) (h1 : min_height = 140) 
  (h2 : brother_height = 180) : 
  min_height - (2/3 * brother_height) = 20 := by
  sorry

end NUMINAMATH_CALUDE_mary_height_to_grow_l1263_126368


namespace NUMINAMATH_CALUDE_smallest_b_value_l1263_126331

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 8) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  ∀ c : ℕ+, c < b → ¬(∃ d : ℕ+, d - c = 8 ∧ 
    Nat.gcd ((d^3 + c^3) / (d + c)) (d * c) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1263_126331


namespace NUMINAMATH_CALUDE_age_difference_l1263_126347

theorem age_difference (A B : ℕ) : B = 39 → A + 10 = 2 * (B - 10) → A - B = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1263_126347


namespace NUMINAMATH_CALUDE_tree_planting_growth_rate_l1263_126379

/-- Represents the annual average growth rate of tree planting -/
def annual_growth_rate : ℝ → Prop :=
  λ x => 400 * (1 + x)^2 = 625

/-- Theorem stating the relationship between the number of trees planted
    in the first and third years, and the annual average growth rate -/
theorem tree_planting_growth_rate :
  ∃ x : ℝ, annual_growth_rate x :=
sorry

end NUMINAMATH_CALUDE_tree_planting_growth_rate_l1263_126379


namespace NUMINAMATH_CALUDE_hyperbola_equivalence_l1263_126375

-- Define the equation
def hyperbola_eq (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + y^2) - Real.sqrt ((x + 3)^2 + y^2) = 4

-- Define the standard form of the hyperbola
def hyperbola_standard_form (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1 ∧ x ≤ -2

-- Theorem stating the equivalence
theorem hyperbola_equivalence :
  ∀ x y : ℝ, hyperbola_eq x y ↔ hyperbola_standard_form x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equivalence_l1263_126375


namespace NUMINAMATH_CALUDE_train_length_l1263_126348

theorem train_length (pole_time : ℝ) (tunnel_length tunnel_time : ℝ) :
  pole_time = 20 →
  tunnel_length = 500 →
  tunnel_time = 40 →
  ∃ (train_length : ℝ),
    train_length = pole_time * (train_length + tunnel_length) / tunnel_time ∧
    train_length = 500 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1263_126348


namespace NUMINAMATH_CALUDE_allan_has_one_more_balloon_l1263_126388

/-- Given the number of balloons Allan and Jake have, prove that Allan has one more balloon than Jake. -/
theorem allan_has_one_more_balloon (allan_balloons jake_initial_balloons jake_bought_balloons : ℕ) 
  (h1 : allan_balloons = 6)
  (h2 : jake_initial_balloons = 2)
  (h3 : jake_bought_balloons = 3) :
  allan_balloons - (jake_initial_balloons + jake_bought_balloons) = 1 := by
  sorry

end NUMINAMATH_CALUDE_allan_has_one_more_balloon_l1263_126388


namespace NUMINAMATH_CALUDE_ellipse_dot_product_l1263_126349

/-- An ellipse with given properties and a line intersecting it -/
structure EllipseWithLine where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (1 - b^2 / a^2).sqrt = Real.sqrt 2 / 2
  h_point : b^2 = 1
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  h_A : A.1 = -a ∧ A.2 = 0
  h_B : B.1^2 / a^2 + B.2^2 / b^2 = 1
  h_P : P.1 = a
  h_collinear : ∃ (t : ℝ), B = A + t • (P - A)

/-- The dot product of OB and OP is 2 -/
theorem ellipse_dot_product (e : EllipseWithLine) : e.B.1 * e.P.1 + e.B.2 * e.P.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_l1263_126349


namespace NUMINAMATH_CALUDE_basketball_team_probabilities_l1263_126356

/-- Represents a series of independent events -/
structure EventSeries where
  n : ℕ  -- number of events
  p : ℝ  -- probability of success for each event
  h1 : 0 ≤ p ∧ p ≤ 1  -- probability is between 0 and 1

/-- The probability of k failures before the first success -/
def prob_k_failures_before_success (es : EventSeries) (k : ℕ) : ℝ :=
  (1 - es.p)^k * es.p

/-- The probability of exactly k successes in n events -/
def prob_exactly_k_successes (es : EventSeries) (k : ℕ) : ℝ :=
  (Nat.choose es.n k : ℝ) * es.p^k * (1 - es.p)^(es.n - k)

/-- The expected number of successes in n events -/
def expected_successes (es : EventSeries) : ℝ :=
  es.n * es.p

theorem basketball_team_probabilities :
  ∀ es : EventSeries,
    es.n = 6 ∧ es.p = 1/3 →
    (prob_k_failures_before_success es 2 = 4/27) ∧
    (prob_exactly_k_successes es 3 = 160/729) ∧
    (expected_successes es = 2) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_probabilities_l1263_126356


namespace NUMINAMATH_CALUDE_labeling_existence_condition_l1263_126361

/-- A labeling function for lattice points -/
def LabelingFunction := ℤ × ℤ → ℕ+

/-- The property that a labeling satisfies the distance condition for a given c -/
def SatisfiesDistanceCondition (f : LabelingFunction) (c : ℝ) : Prop :=
  ∀ i : ℕ+, ∀ p q : ℤ × ℤ, f p = i ∧ f q = i → dist p q ≥ c ^ (i : ℝ)

/-- The property that a labeling uses only finitely many labels -/
def UsesFiniteLabels (f : LabelingFunction) : Prop :=
  ∃ n : ℕ, ∀ p : ℤ × ℤ, (f p : ℕ) ≤ n

/-- The main theorem -/
theorem labeling_existence_condition (c : ℝ) :
  (c > 0 ∧ c < Real.sqrt 2) ↔
  (∃ f : LabelingFunction, SatisfiesDistanceCondition f c ∧ UsesFiniteLabels f) :=
sorry

end NUMINAMATH_CALUDE_labeling_existence_condition_l1263_126361


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_l1263_126370

theorem largest_n_binomial_sum : 
  (∃ n : ℕ, (Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 n) ∧ 
   (∀ m : ℕ, m > n → Nat.choose 9 4 + Nat.choose 9 5 ≠ Nat.choose 10 m)) → 
  (∃ n : ℕ, n = 5 ∧ (Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 n) ∧ 
   (∀ m : ℕ, m > n → Nat.choose 9 4 + Nat.choose 9 5 ≠ Nat.choose 10 m)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_l1263_126370


namespace NUMINAMATH_CALUDE_same_color_marble_probability_l1263_126358

/-- The probability of drawing four marbles of the same color from a box containing
    3 orange, 7 purple, and 5 green marbles, without replacement. -/
theorem same_color_marble_probability :
  let total_marbles : ℕ := 3 + 7 + 5
  let orange_marbles : ℕ := 3
  let purple_marbles : ℕ := 7
  let green_marbles : ℕ := 5
  let draw_count : ℕ := 4
  
  (orange_marbles.choose draw_count +
   purple_marbles.choose draw_count +
   green_marbles.choose draw_count : ℚ) /
  (total_marbles.choose draw_count : ℚ) = 210 / 1369 :=
by sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_l1263_126358


namespace NUMINAMATH_CALUDE_cubic_expansion_property_l1263_126340

theorem cubic_expansion_property (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5*x + 4)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) →
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_property_l1263_126340


namespace NUMINAMATH_CALUDE_scientific_notation_of_505000_l1263_126300

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The number to be represented in scientific notation -/
def number : ℝ := 505000

/-- The expected scientific notation representation -/
def expected : ScientificNotation :=
  { coefficient := 5.05
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the scientific notation of 505,000 is 5.05 × 10^5 -/
theorem scientific_notation_of_505000 :
  toScientificNotation number = expected := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_505000_l1263_126300


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1263_126306

theorem power_tower_mod_500 : 2^(2^(2^2)) % 500 = 36 := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1263_126306


namespace NUMINAMATH_CALUDE_xiao_ming_brother_age_l1263_126369

def is_multiple_of_19 (year : ℕ) : Prop := ∃ k : ℕ, year = 19 * k

def has_repeated_digits (year : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ (∃ i j : ℕ, i ≠ j ∧ (year / 10^i) % 10 = d ∧ (year / 10^j) % 10 = d)

def first_non_repeating_year (birth_year : ℕ) (target_year : ℕ) : Prop :=
  ¬(has_repeated_digits target_year) ∧
  ∀ y : ℕ, birth_year ≤ y ∧ y < target_year → has_repeated_digits y

theorem xiao_ming_brother_age :
  ∀ birth_year : ℕ,
    is_multiple_of_19 birth_year →
    first_non_repeating_year birth_year 2013 →
    2013 - birth_year = 18 :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_brother_age_l1263_126369


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l1263_126307

/-- Given 6 consecutive odd numbers whose product is 135135, prove their sum is 48 -/
theorem consecutive_odd_numbers_sum (a b c d e f : ℕ) : 
  (a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) →  -- consecutive
  (∃ k, a = 2*k + 1) →  -- a is odd
  (b = a + 2) → (c = b + 2) → (d = c + 2) → (e = d + 2) → (f = e + 2) →  -- consecutive odd numbers
  (a * b * c * d * e * f = 135135) →  -- product is 135135
  (a + b + c + d + e + f = 48) :=  -- sum is 48
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l1263_126307


namespace NUMINAMATH_CALUDE_problem_solution_l1263_126318

theorem problem_solution (a b c d : ℝ) 
  (h1 : a < b ∧ b < d)
  (h2 : ∀ x, (x - a) * (x - b) * (x - d) / (x - c) ≥ 0 ↔ x ≤ -7 ∨ (30 ≤ x ∧ x ≤ 32)) :
  a + 2*b + 3*c + 4*d = 160 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1263_126318


namespace NUMINAMATH_CALUDE_parking_lot_wheel_count_l1263_126320

def parking_lot_wheels (num_cars : ℕ) (num_bikes : ℕ) (wheels_per_car : ℕ) (wheels_per_bike : ℕ) : ℕ :=
  num_cars * wheels_per_car + num_bikes * wheels_per_bike

theorem parking_lot_wheel_count : parking_lot_wheels 14 10 4 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheel_count_l1263_126320


namespace NUMINAMATH_CALUDE_interior_alternate_angles_equal_implies_parallel_l1263_126311

/-- Two lines in a plane -/
structure Line

/-- A transversal line cutting two other lines -/
structure Transversal

/-- An angle formed by the intersection of lines -/
structure Angle

/-- Defines the concept of interior alternate angles -/
def interior_alternate_angles (l1 l2 : Line) (t : Transversal) (a1 a2 : Angle) : Prop :=
  sorry

/-- Defines parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  sorry

/-- The main theorem: if interior alternate angles are equal, then the lines are parallel -/
theorem interior_alternate_angles_equal_implies_parallel 
  (l1 l2 : Line) (t : Transversal) (a1 a2 : Angle) :
  interior_alternate_angles l1 l2 t a1 a2 → a1 = a2 → parallel l1 l2 :=
sorry

end NUMINAMATH_CALUDE_interior_alternate_angles_equal_implies_parallel_l1263_126311


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1263_126304

/-- Given a parabola and a hyperbola satisfying certain conditions, 
    prove their equations. -/
theorem parabola_hyperbola_equations :
  ∀ (parabola : ℝ → ℝ → Prop) (hyperbola : ℝ → ℝ → Prop),
  (∀ x y, parabola x y → (x = 0 ∧ y = 0)) →  -- vertex at origin
  (∃ x₀, ∀ y, hyperbola x₀ y → parabola x₀ y) →  -- axis of symmetry passes through focus
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y, hyperbola x y ↔ x^2/a^2 - y^2/b^2 = 1) →  -- general form of hyperbola
  hyperbola (3/2) (Real.sqrt 6) →  -- intersection point
  (∀ x y, parabola x y ↔ y^2 = 4*x) ∧  -- equation of parabola
  (∀ x y, hyperbola x y ↔ 4*x^2 - 4*y^2/3 = 1) :=  -- equation of hyperbola
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1263_126304


namespace NUMINAMATH_CALUDE_subgrids_cover_half_board_l1263_126389

/-- Represents a subgrid on the board -/
structure Subgrid where
  rows : ℕ
  cols : ℕ

/-- The board and its properties -/
structure Board where
  n : ℕ
  subgrids : List Subgrid

/-- Calculates the half-perimeter of a subgrid -/
def half_perimeter (s : Subgrid) : ℕ := s.rows + s.cols

/-- Checks if a list of subgrids covers the main diagonal -/
def covers_main_diagonal (b : Board) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ b.n → ∃ s ∈ b.subgrids, half_perimeter s ≥ b.n

/-- Calculates the number of squares covered by a list of subgrids -/
def squares_covered (b : Board) : ℕ :=
  sorry -- Implementation details omitted

/-- Main theorem -/
theorem subgrids_cover_half_board (b : Board) 
  (h_board_size : b.n * b.n = 11 * 60)
  (h_cover_diagonal : covers_main_diagonal b) :
  2 * (squares_covered b) ≥ b.n * b.n := by
  sorry

end NUMINAMATH_CALUDE_subgrids_cover_half_board_l1263_126389


namespace NUMINAMATH_CALUDE_f_even_iff_a_eq_zero_l1263_126329

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + ax for some a ∈ ℝ -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x

/-- Theorem: f is an even function if and only if a = 0 -/
theorem f_even_iff_a_eq_zero (a : ℝ) :
  IsEven (f a) ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_f_even_iff_a_eq_zero_l1263_126329


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_four_l1263_126333

def z (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 8) (m - 2)

theorem pure_imaginary_implies_m_eq_neg_four :
  ∀ m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_four_l1263_126333


namespace NUMINAMATH_CALUDE_pencil_count_l1263_126314

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of these two numbers. -/
theorem pencil_count (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l1263_126314


namespace NUMINAMATH_CALUDE_polynomial_roots_imply_composite_sum_of_squares_l1263_126351

/-- A polynomial with integer coefficients -/
def IntPolynomial (p q : ℤ) : ℝ → ℝ := fun x ↦ x^2 + p*x + q + 1

/-- Definition of a composite number -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem polynomial_roots_imply_composite_sum_of_squares (p q : ℤ) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ (IntPolynomial p q a = 0) ∧ (IntPolynomial p q b = 0)) →
  IsComposite (Int.natAbs (p^2 + q^2)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_imply_composite_sum_of_squares_l1263_126351


namespace NUMINAMATH_CALUDE_washing_machines_removed_per_box_l1263_126396

theorem washing_machines_removed_per_box :
  let num_crates : ℕ := 10
  let boxes_per_crate : ℕ := 6
  let initial_machines_per_box : ℕ := 4
  let total_machines_removed : ℕ := 60
  let total_boxes : ℕ := num_crates * boxes_per_crate
  let machines_removed_per_box : ℕ := total_machines_removed / total_boxes
  machines_removed_per_box = 1 := by
  sorry

end NUMINAMATH_CALUDE_washing_machines_removed_per_box_l1263_126396
