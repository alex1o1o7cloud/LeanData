import Mathlib

namespace equation_solution_l1102_110261

theorem equation_solution : 
  ∃ x : ℝ, (8 * 5.4 - 0.6 * x / 1.2 = 31.000000000000004) ∧ (x = 24.4) :=
by sorry

end equation_solution_l1102_110261


namespace largest_fraction_l1102_110291

theorem largest_fraction :
  let a := (1 : ℚ) / 3
  let b := (1 : ℚ) / 4
  let c := (3 : ℚ) / 8
  let d := (5 : ℚ) / 12
  let e := (7 : ℚ) / 24
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end largest_fraction_l1102_110291


namespace abs_inequality_solution_set_l1102_110270

def solution_set (x : ℝ) := -1 < x ∧ x < 0

theorem abs_inequality_solution_set :
  {x : ℝ | |2*x + 1| < 1} = {x : ℝ | solution_set x} := by sorry

end abs_inequality_solution_set_l1102_110270


namespace consecutive_nonprime_integers_l1102_110204

theorem consecutive_nonprime_integers :
  ∃ n : ℕ,
    100 < n ∧
    n + 4 < 200 ∧
    (¬ Prime n) ∧
    (¬ Prime (n + 1)) ∧
    (¬ Prime (n + 2)) ∧
    (¬ Prime (n + 3)) ∧
    (¬ Prime (n + 4)) ∧
    n + 4 = 148 :=
by sorry

end consecutive_nonprime_integers_l1102_110204


namespace quadratic_shift_sum_l1102_110228

/-- Given a quadratic function f(x) = 3x^2 + 2x - 5, when shifted 6 units to the right,
    the resulting function g(x) = ax^2 + bx + c has coefficients that sum to 60. -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, (3 * (x - 6)^2 + 2 * (x - 6) - 5) = (a * x^2 + b * x + c)) →
  a + b + c = 60 := by
sorry

end quadratic_shift_sum_l1102_110228


namespace student_distribution_problem_l1102_110299

/-- The number of ways to distribute n students among k schools,
    where each school must have at least one student. -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  (n - 1).choose (k - 1) * k.factorial

/-- The specific problem statement -/
theorem student_distribution_problem :
  distribute_students 4 3 = 36 := by
  sorry

end student_distribution_problem_l1102_110299


namespace quadratic_equation_roots_l1102_110295

/-- Given a quadratic equation x^2 - (3+m)x + 3m = 0 with real roots x1 and x2
    satisfying 2x1 - x1x2 + 2x2 = 12, prove that x1 = -6 and x2 = 3 -/
theorem quadratic_equation_roots (m : ℝ) (x1 x2 : ℝ) :
  x1^2 - (3+m)*x1 + 3*m = 0 →
  x2^2 - (3+m)*x2 + 3*m = 0 →
  2*x1 - x1*x2 + 2*x2 = 12 →
  (x1 = -6 ∧ x2 = 3) ∨ (x1 = 3 ∧ x2 = -6) :=
by sorry

end quadratic_equation_roots_l1102_110295


namespace walking_speed_proof_l1102_110254

/-- The walking speed of person A in km/h -/
def a_speed : ℝ := 10

/-- The cycling speed of person B in km/h -/
def b_speed : ℝ := 20

/-- The time difference between A's start and B's start in hours -/
def time_diff : ℝ := 4

/-- The distance at which B catches up with A in km -/
def catch_up_distance : ℝ := 80

theorem walking_speed_proof :
  (catch_up_distance / a_speed = time_diff + catch_up_distance / b_speed) →
  a_speed = 10 := by
  sorry

#check walking_speed_proof

end walking_speed_proof_l1102_110254


namespace andrey_gleb_distance_l1102_110267

/-- Represents the position of a home on a straight street -/
structure Home where
  position : ℝ

/-- The street with four homes -/
structure Street where
  andrey : Home
  borya : Home
  vova : Home
  gleb : Home

/-- The distance between two homes -/
def distance (h1 h2 : Home) : ℝ := |h1.position - h2.position|

/-- The conditions of the problem -/
def valid_street (s : Street) : Prop :=
  distance s.andrey s.borya = 600 ∧
  distance s.vova s.gleb = 600 ∧
  distance s.andrey s.gleb = 3 * distance s.borya s.vova

/-- The theorem to be proved -/
theorem andrey_gleb_distance (s : Street) :
  valid_street s →
  distance s.andrey s.gleb = 1500 ∨ distance s.andrey s.gleb = 1800 :=
sorry

end andrey_gleb_distance_l1102_110267


namespace problem_1_l1102_110252

theorem problem_1 (x : ℝ) : (x - 1)^2 + x*(3 - x) = x + 1 := by
  sorry

end problem_1_l1102_110252


namespace circle_and_line_problem_l1102_110220

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = x^2 - x - 6

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - x + 5*y - 6 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-2, 5)

-- Define the line l (vertical case)
def line_l_vertical (x : ℝ) : Prop := x = -2

-- Define the line l (non-vertical case)
def line_l_nonvertical (x y : ℝ) : Prop := 4*x + 3*y - 7 = 0

-- Theorem statement
theorem circle_and_line_problem :
  ∃ (A B : ℝ × ℝ),
    -- Circle C passes through intersection points of parabola and axes
    (∀ (x y : ℝ), (x = 0 ∨ y = 0) ∧ parabola x y → circle_C x y) ∧
    -- Line l passes through P and intersects C at A and B
    (line_l_vertical A.1 ∨ line_l_nonvertical A.1 A.2) ∧
    (line_l_vertical B.1 ∨ line_l_nonvertical B.1 B.2) ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    -- Tangents at A and B are perpendicular
    (∃ (tA tB : ℝ × ℝ → ℝ × ℝ),
      (tA A = B ∨ tB B = A) →
      (tA A • tB B = 0)) →
    -- Conclusion: Equations of circle C and line l
    (∀ (x y : ℝ), circle_C x y ↔ x^2 + y^2 - x + 5*y - 6 = 0) ∧
    (∀ (x y : ℝ), (x = -2 ∨ 4*x + 3*y - 7 = 0) ↔ (line_l_vertical x ∨ line_l_nonvertical x y))
  := by sorry

end circle_and_line_problem_l1102_110220


namespace tethered_dog_area_tethered_dog_area_exact_l1102_110232

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem tethered_dog_area (side_length : ℝ) (rope_length : ℝ) 
  (h1 : side_length = 2)
  (h2 : rope_length = 3) : 
  ℝ :=
let hexagon_area := 3 * Real.sqrt 3 * side_length ^ 2 / 2
let circle_area := Real.pi * rope_length ^ 2
circle_area - hexagon_area

/-- The main theorem stating the exact area -/
theorem tethered_dog_area_exact : 
  tethered_dog_area 2 3 rfl rfl = 9 * Real.pi - 6 * Real.sqrt 3 :=
sorry

end tethered_dog_area_tethered_dog_area_exact_l1102_110232


namespace thirty_six_has_nine_divisors_l1102_110286

/-- The number of positive divisors of 36 -/
def num_divisors_36 : ℕ := sorry

/-- 36 has exactly 9 positive divisors -/
theorem thirty_six_has_nine_divisors : num_divisors_36 = 9 := by sorry

end thirty_six_has_nine_divisors_l1102_110286


namespace sum_abcd_l1102_110210

theorem sum_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ 
       b + 3 = c + 4 ∧ 
       c + 4 = d + 5 ∧ 
       d + 5 = a + b + c + d + 15) : 
  a + b + c + d = -46/3 := by
sorry

end sum_abcd_l1102_110210


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1102_110241

/-- An isosceles triangle with side lengths 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Two sides are 9, one side is 4
    (b = c) →                  -- The triangle is isosceles
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (a + b + c = 22)           -- The perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 4 9 9 := by
  sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1102_110241


namespace e_2i_in_second_quadrant_l1102_110234

open Complex

theorem e_2i_in_second_quadrant :
  let z : ℂ := Complex.exp (2 * I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end e_2i_in_second_quadrant_l1102_110234


namespace perpendicular_tangents_ratio_l1102_110200

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem perpendicular_tangents_ratio (a b : ℝ) :
  (a * P.1 - b * P.2 - 2 = 0) →  -- Line equation at point P
  (curve P.1 = P.2) →            -- Curve passes through P
  (curve_derivative P.1 * (a / b) = -1) →  -- Perpendicular tangents condition
  a / b = -1/4 := by
sorry

end perpendicular_tangents_ratio_l1102_110200


namespace shaded_grid_percentage_l1102_110227

theorem shaded_grid_percentage (total_squares : ℕ) (shaded_squares : ℕ) 
  (h1 : total_squares = 36) (h2 : shaded_squares = 16) : 
  (shaded_squares : ℚ) / total_squares * 100 = 44.4444444444444444 := by
  sorry

end shaded_grid_percentage_l1102_110227


namespace base4_addition_theorem_l1102_110279

/-- Addition of numbers in base 4 -/
def base4_add (a b c d : ℕ) : ℕ := sorry

/-- Conversion from base 4 to decimal -/
def base4_to_decimal (n : ℕ) : ℕ := sorry

theorem base4_addition_theorem :
  base4_add (base4_to_decimal 2) (base4_to_decimal 23) (base4_to_decimal 132) (base4_to_decimal 1320) = base4_to_decimal 20200 := by
  sorry

end base4_addition_theorem_l1102_110279


namespace complement_of_A_in_U_l1102_110242

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 2}

theorem complement_of_A_in_U : 
  {x : Int | x ∈ U ∧ x ∉ A} = {0, 1} := by sorry

end complement_of_A_in_U_l1102_110242


namespace arielle_age_l1102_110276

theorem arielle_age (elvie_age : ℕ) (total : ℕ) : 
  elvie_age = 10 →
  (∃ (arielle_age : ℕ), 
    elvie_age + arielle_age + elvie_age * arielle_age = total ∧
    total = 131) →
  ∃ (arielle_age : ℕ), arielle_age = 11 :=
by sorry

end arielle_age_l1102_110276


namespace inconsistent_average_and_sum_l1102_110251

theorem inconsistent_average_and_sum :
  let numbers : List ℕ := [54, 55, 57, 58, 62, 62, 63, 65, 65]
  let average : ℕ := 60
  let total_sum : ℕ := average * numbers.length
  let sum_of_numbers : ℕ := numbers.sum
  sum_of_numbers > total_sum := by sorry

end inconsistent_average_and_sum_l1102_110251


namespace plan_y_more_economical_min_megabytes_optimal_l1102_110257

/-- Represents the cost of an internet plan in cents -/
def PlanCost (initial_fee : ℕ) (rate : ℕ) (megabytes : ℕ) : ℕ :=
  initial_fee * 100 + rate * megabytes

/-- The minimum number of megabytes for Plan Y to be more economical than Plan X -/
def MinMegabytes : ℕ := 501

theorem plan_y_more_economical :
  ∀ m : ℕ, m ≥ MinMegabytes →
    PlanCost 25 10 m < PlanCost 0 15 m :=
by
  sorry

theorem min_megabytes_optimal :
  ∀ m : ℕ, m < MinMegabytes →
    PlanCost 0 15 m ≤ PlanCost 25 10 m :=
by
  sorry

end plan_y_more_economical_min_megabytes_optimal_l1102_110257


namespace sqrt_product_simplification_l1102_110244

theorem sqrt_product_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (3 * x) * Real.sqrt (2 * x^2) = Real.sqrt 6 * x^(3/2) :=
sorry

end sqrt_product_simplification_l1102_110244


namespace domino_tiling_triomino_tiling_l_tetromino_tiling_l1102_110207

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a tile --/
structure Tile :=
  (size : ℕ)

/-- Defines a 9x9 chessboard --/
def chessboard_9x9 : Chessboard :=
  ⟨9 * 9⟩

/-- Defines a 2x1 domino --/
def domino : Tile :=
  ⟨2⟩

/-- Defines a 3x1 triomino --/
def triomino : Tile :=
  ⟨3⟩

/-- Defines an L-shaped tetromino --/
def l_tetromino : Tile :=
  ⟨4⟩

/-- Determines if a chessboard can be tiled with a given tile --/
def can_tile (c : Chessboard) (t : Tile) : Prop :=
  c.size % t.size = 0

theorem domino_tiling :
  ¬ can_tile chessboard_9x9 domino :=
sorry

theorem triomino_tiling :
  can_tile chessboard_9x9 triomino :=
sorry

theorem l_tetromino_tiling :
  ¬ can_tile chessboard_9x9 l_tetromino :=
sorry

end domino_tiling_triomino_tiling_l_tetromino_tiling_l1102_110207


namespace logarithm_and_exponent_calculation_l1102_110231

theorem logarithm_and_exponent_calculation :
  (2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2) ∧
  ((0.027 ^ (-1/3 : ℝ)) - ((-1/7 : ℝ)⁻¹) + ((2 + 7/9 : ℝ) ^ (1/2 : ℝ)) - ((Real.sqrt 2 - 1) ^ (0 : ℝ)) = 11) :=
by sorry

end logarithm_and_exponent_calculation_l1102_110231


namespace quadrilateral_side_length_l1102_110222

theorem quadrilateral_side_length (A B C D : ℝ × ℝ) : 
  let angle (p q r : ℝ × ℝ) := Real.arccos ((p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2)) / 
    (Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) * Real.sqrt ((r.1 - q.1)^2 + (r.2 - q.2)^2))
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  angle D A B = π/3 ∧ 
  angle A B C = π/2 ∧ 
  angle B C D = π/2 ∧ 
  dist B C = 2 ∧ 
  dist C D = 3 →
  dist A B = 8 / Real.sqrt 3 := by
sorry


end quadrilateral_side_length_l1102_110222


namespace radical_axis_intersection_squared_distance_l1102_110249

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB : Real)
  (BC : Real)
  (CA : Real)

-- Define the incircle and its touchpoints
structure Incircle :=
  (I : ℝ × ℝ)
  (M N D : ℝ × ℝ)

-- Define point K
def K (t : Triangle) (inc : Incircle) : ℝ × ℝ := sorry

-- Define circumcircles of triangles MAN and KID
def CircumcircleMAN (t : Triangle) (inc : Incircle) : ℝ × ℝ := sorry
def CircumcircleKID (t : Triangle) (inc : Incircle) : ℝ × ℝ := sorry

-- Define the radical axis
def RadicalAxis (c1 c2 : ℝ × ℝ) : ℝ × ℝ → Prop := sorry

-- Define L₁ and L₂
def L₁ (t : Triangle) (ra : ℝ × ℝ → Prop) : ℝ × ℝ := sorry
def L₂ (t : Triangle) (ra : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

-- Distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem radical_axis_intersection_squared_distance
  (t : Triangle)
  (inc : Incircle)
  (h1 : t.AB = 36)
  (h2 : t.BC = 48)
  (h3 : t.CA = 60)
  (h4 : inc.M = sorry)  -- Point where incircle touches AB
  (h5 : inc.N = sorry)  -- Point where incircle touches AC
  (h6 : inc.D = sorry)  -- Point where incircle touches BC
  :
  let k := K t inc
  let c1 := CircumcircleMAN t inc
  let c2 := CircumcircleKID t inc
  let ra := RadicalAxis c1 c2
  let l1 := L₁ t ra
  let l2 := L₂ t ra
  (distance l1 l2)^2 = 720 := by sorry

end radical_axis_intersection_squared_distance_l1102_110249


namespace overall_average_l1102_110268

theorem overall_average (n : ℕ) (avg_first : ℝ) (avg_last : ℝ) (middle : ℝ) :
  n = 25 →
  avg_first = 14 →
  avg_last = 17 →
  middle = 78 →
  (avg_first * 12 + middle + avg_last * 12) / n = 18 := by
sorry

end overall_average_l1102_110268


namespace circle_center_correct_l1102_110256

/-- The center of a circle given by the equation x^2 + y^2 - 2x + 4y = 0 --/
def circle_center : ℝ × ℝ := sorry

/-- The equation of the circle --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y = 0

theorem circle_center_correct :
  let (h, k) := circle_center
  ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 1) ∧ 
  h = 1 ∧ k = -2 := by sorry

end circle_center_correct_l1102_110256


namespace butterfly_equation_equal_roots_l1102_110259

/-- A quadratic equation ax^2 + bx + c = 0 (a ≠ 0) that satisfies the "butterfly" condition (a - b + c = 0) and has two equal real roots implies a = c. -/
theorem butterfly_equation_equal_roots (a b c : ℝ) (ha : a ≠ 0) :
  (a - b + c = 0) →  -- Butterfly condition
  (b^2 - 4*a*c = 0) →  -- Condition for two equal real roots (discriminant = 0)
  a = c := by
  sorry

end butterfly_equation_equal_roots_l1102_110259


namespace jack_morning_letters_l1102_110253

def morning_letters (afternoon_letters : ℕ) (difference : ℕ) : ℕ :=
  afternoon_letters + difference

theorem jack_morning_letters :
  morning_letters 7 1 = 8 := by
  sorry

end jack_morning_letters_l1102_110253


namespace triangle_abc_properties_l1102_110216

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  c = 2 →
  C = π / 3 →
  2 * sin (2 * A) + sin (2 * B + C) = sin C →
  (∃ S : ℝ, S = (2 * Real.sqrt 3) / 3 ∧ S = (1 / 2) * a * b * sin C) ∧
  (∃ P : ℝ, P ≤ 6 ∧ P = a + b + c) :=
by sorry

end triangle_abc_properties_l1102_110216


namespace square_minus_a_nonpositive_l1102_110294

theorem square_minus_a_nonpositive (a : ℝ) (h : a > 4) :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - a ≤ 0 := by sorry

end square_minus_a_nonpositive_l1102_110294


namespace marble_distribution_l1102_110255

theorem marble_distribution (y : ℕ) : 
  let first_friend := 2 * y + 2
  let second_friend := y
  let third_friend := 3 * y - 1
  first_friend + second_friend + third_friend = 6 * y + 1 :=
by sorry

end marble_distribution_l1102_110255


namespace integer_set_equivalence_l1102_110209

theorem integer_set_equivalence (a : ℝ) : 
  (a ≤ 1 ∧ (Set.range (fun n : ℤ => (n : ℝ)) ∩ Set.Icc a (2 - a)).ncard = 3) ↔ 
  -1 < a ∧ a ≤ 0 := by
sorry

end integer_set_equivalence_l1102_110209


namespace luke_gave_five_stickers_l1102_110271

/-- Calculates the number of stickers Luke gave to his sister -/
def stickers_given_to_sister (initial : ℕ) (bought : ℕ) (birthday : ℕ) (used : ℕ) (left : ℕ) : ℕ :=
  initial + bought + birthday - used - left

/-- Proves that Luke gave 5 stickers to his sister -/
theorem luke_gave_five_stickers :
  stickers_given_to_sister 20 12 20 8 39 = 5 := by
  sorry

#eval stickers_given_to_sister 20 12 20 8 39

end luke_gave_five_stickers_l1102_110271


namespace consecutive_integers_sum_of_squares_l1102_110273

theorem consecutive_integers_sum_of_squares : 
  ∃ (b : ℕ), 
    (b > 0) ∧ 
    ((b - 1) * b * (b + 1) = 12 * (3 * b)) → 
    ((b - 1)^2 + b^2 + (b + 1)^2 = 110) :=
by sorry

end consecutive_integers_sum_of_squares_l1102_110273


namespace cubic_expression_value_l1102_110211

theorem cubic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2*x^2 - 12 = -11 := by
  sorry

end cubic_expression_value_l1102_110211


namespace parabola_tangent_intersection_l1102_110219

/-- Parabola tangent intersection theorem -/
theorem parabola_tangent_intersection
  (t₁ t₂ : ℝ) (h : t₁ ≠ t₂) :
  let parabola := fun x : ℝ => x^2 / 4
  let tangent₁ := fun x : ℝ => t₁ * x - t₁^2
  let tangent₂ := fun x : ℝ => t₂ * x - t₂^2
  let intersection_x := t₁ + t₂
  let intersection_y := t₁ * t₂
  (parabola (2 * t₁) = t₁^2) ∧
  (parabola (2 * t₂) = t₂^2) ∧
  (tangent₁ intersection_x = intersection_y) ∧
  (tangent₂ intersection_x = intersection_y) :=
by sorry

end parabola_tangent_intersection_l1102_110219


namespace marble_selection_problem_l1102_110224

theorem marble_selection_problem (n : ℕ) (k : ℕ) (s : ℕ) (t : ℕ) :
  n = 15 ∧ k = 5 ∧ s = 4 ∧ t = 2 →
  (Nat.choose s t) * (Nat.choose (n - s) (k - t)) = 990 := by
  sorry

end marble_selection_problem_l1102_110224


namespace faye_initial_giveaway_l1102_110278

/-- The number of coloring books Faye bought initially -/
def initial_books : ℝ := 48.0

/-- The number of coloring books Faye gave away after the initial giveaway -/
def additional_giveaway : ℝ := 3.0

/-- The number of coloring books Faye has left -/
def remaining_books : ℝ := 11.0

/-- The number of coloring books Faye gave away initially -/
def initial_giveaway : ℝ := initial_books - additional_giveaway - remaining_books

theorem faye_initial_giveaway : initial_giveaway = 34.0 := by
  sorry

end faye_initial_giveaway_l1102_110278


namespace max_freshmen_is_eight_l1102_110221

/-- Represents the relation of knowing each other among freshmen. -/
def Knows (n : ℕ) := Fin n → Fin n → Prop

/-- The property that any 3 people include at least 2 who know each other. -/
def AnyThreeHaveTwoKnown (n : ℕ) (knows : Knows n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    knows a b ∨ knows b c ∨ knows a c

/-- The property that any 4 people include at least 2 who do not know each other. -/
def AnyFourHaveTwoUnknown (n : ℕ) (knows : Knows n) : Prop :=
  ∀ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(knows a b) ∨ ¬(knows a c) ∨ ¬(knows a d) ∨
    ¬(knows b c) ∨ ¬(knows b d) ∨ ¬(knows c d)

/-- The theorem stating that the maximum number of freshmen satisfying the conditions is 8. -/
theorem max_freshmen_is_eight :
  ∀ n : ℕ, (∃ knows : Knows n, AnyThreeHaveTwoKnown n knows ∧ AnyFourHaveTwoUnknown n knows) →
    n ≤ 8 :=
sorry

end max_freshmen_is_eight_l1102_110221


namespace correlation_identification_l1102_110274

-- Define the concept of a relationship
def Relationship : Type := Unit

-- Define specific relationships
def age_wealth : Relationship := ()
def curve_coordinates : Relationship := ()
def apple_production_climate : Relationship := ()
def tree_diameter_height : Relationship := ()

-- Define the property of being correlational
def is_correlational : Relationship → Prop := sorry

-- Define the property of being functional
def is_functional : Relationship → Prop := sorry

-- State that functional relationships are not correlational
axiom functional_not_correlational : 
  ∀ (r : Relationship), is_functional r → ¬is_correlational r

-- State the theorem
theorem correlation_identification :
  is_correlational age_wealth ∧
  is_correlational apple_production_climate ∧
  is_correlational tree_diameter_height ∧
  is_functional curve_coordinates :=
sorry

end correlation_identification_l1102_110274


namespace complex_square_one_plus_i_l1102_110203

theorem complex_square_one_plus_i (i : ℂ) : 
  i ^ 2 = -1 → (1 + i) ^ 2 = 2 * i := by
  sorry

end complex_square_one_plus_i_l1102_110203


namespace power_sum_of_i_l1102_110246

-- Define the imaginary unit i
axiom i : ℂ
axiom i_squared : i^2 = -1

-- State the theorem
theorem power_sum_of_i : i^2023 + i^303 = -2*i := by sorry

end power_sum_of_i_l1102_110246


namespace initial_employees_correct_l1102_110218

/-- Represents the initial number of employees in a company. -/
def initial_employees : ℕ := 450

/-- Represents the monthly salary of each employee in dollars. -/
def salary_per_employee : ℕ := 2000

/-- Represents the fraction of employees remaining after layoffs. -/
def remaining_fraction : ℚ := 2/3

/-- Represents the total amount paid to remaining employees in dollars. -/
def total_paid : ℕ := 600000

/-- Theorem stating that the initial number of employees is correct given the conditions. -/
theorem initial_employees_correct : 
  (initial_employees : ℚ) * remaining_fraction * salary_per_employee = total_paid :=
sorry

end initial_employees_correct_l1102_110218


namespace coefficient_of_linear_term_l1102_110235

theorem coefficient_of_linear_term (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 0) → (a = 1 ∧ b = 3 ∧ c = -1) → b = 3 := by
  sorry

end coefficient_of_linear_term_l1102_110235


namespace function_inequality_l1102_110272

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x ∈ (Set.Ioo 0 (π/2)), HasDerivAt f (f' x) x) :
  (∀ x ∈ (Set.Ioo 0 (π/2)), f' x * sin x - cos x * f x > 0) →
  Real.sqrt 3 * f (π/6) < f (π/3) := by
  sorry

end function_inequality_l1102_110272


namespace yellow_balls_after_loss_l1102_110213

theorem yellow_balls_after_loss (initial_total : ℕ) (current_total : ℕ) (blue : ℕ) (lost : ℕ) : 
  initial_total = 120 →
  current_total = 110 →
  blue = 15 →
  lost = 10 →
  let red := 3 * blue
  let green := red + blue
  let yellow := initial_total - (red + blue + green)
  yellow = 0 := by sorry

end yellow_balls_after_loss_l1102_110213


namespace budget_equipment_percentage_l1102_110258

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  transportation : ℝ
  research_development : ℝ
  utilities : ℝ
  supplies : ℝ
  salaries : ℝ
  equipment : ℝ

/-- Theorem: Given the budget allocation conditions, the percentage spent on equipment is 4% -/
theorem budget_equipment_percentage
  (budget : BudgetAllocation)
  (h1 : budget.transportation = 15)
  (h2 : budget.research_development = 9)
  (h3 : budget.utilities = 5)
  (h4 : budget.supplies = 2)
  (h5 : budget.salaries = (234 / 360) * 100)
  (h6 : budget.transportation + budget.research_development + budget.utilities +
        budget.supplies + budget.salaries + budget.equipment = 100) :
  budget.equipment = 4 := by
  sorry

end budget_equipment_percentage_l1102_110258


namespace point_C_coordinates_l1102_110214

-- Define the points A and B
def A : ℝ × ℝ := (-2, -1)
def B : ℝ × ℝ := (4, 9)

-- Define the condition for point C
def is_point_C (C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
  C.1 = A.1 + t * (B.1 - A.1) ∧
  C.2 = A.2 + t * (B.2 - A.2) ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 16 * ((B.1 - C.1)^2 + (B.2 - C.2)^2)

-- Theorem statement
theorem point_C_coordinates :
  ∃ C : ℝ × ℝ, is_point_C C ∧ C = (-0.8, 1) :=
sorry

end point_C_coordinates_l1102_110214


namespace complement_A_intersect_integers_l1102_110285

def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

theorem complement_A_intersect_integers :
  (Set.univ \ A) ∩ Set.range (Int.cast : ℤ → ℝ) = {-1, 0, 1, 2} := by sorry

end complement_A_intersect_integers_l1102_110285


namespace chapter_length_l1102_110229

theorem chapter_length (pages_per_chapter : ℕ) 
  (h1 : 10 * pages_per_chapter + 20 + 2 * pages_per_chapter = 500) :
  pages_per_chapter = 40 := by
  sorry

end chapter_length_l1102_110229


namespace min_value_quadratic_l1102_110266

theorem min_value_quadratic (x : ℝ) :
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4 * x^2 + 8 * x + 12 → y ≥ y_min ∧ ∃ (x_0 : ℝ), 4 * x_0^2 + 8 * x_0 + 12 = y_min :=
by
  sorry

end min_value_quadratic_l1102_110266


namespace race_time_theorem_l1102_110201

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.distance = 1000 ∧
  (r.runner_a.speed * r.runner_a.time = r.distance) ∧
  (r.runner_b.speed * r.runner_b.time = r.distance - 40 ∨
   r.runner_b.speed * (r.runner_a.time + 10) = r.distance)

/-- The theorem to prove -/
theorem race_time_theorem (r : Race) :
  race_conditions r → r.runner_a.time = 240 := by
  sorry

end race_time_theorem_l1102_110201


namespace intersection_M_complement_N_l1102_110236

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}

theorem intersection_M_complement_N : M ∩ (U \ N) = {4} := by
  sorry

end intersection_M_complement_N_l1102_110236


namespace tractor_production_proof_l1102_110212

/-- The number of tractors produced in October -/
def october_production : ℕ := 1000

/-- The additional number of tractors planned to be produced in November and December -/
def additional_production : ℕ := 2310

/-- The percentage increase of the additional production compared to the original plan -/
def percentage_increase : ℚ := 21 / 100

/-- The monthly growth rate for November and December -/
def monthly_growth_rate : ℚ := 1 / 10

/-- The original annual production plan -/
def original_annual_plan : ℕ := 11000

theorem tractor_production_proof :
  (october_production * (1 + monthly_growth_rate) + october_production * (1 + monthly_growth_rate)^2 = additional_production) ∧
  (original_annual_plan + original_annual_plan * percentage_increase = original_annual_plan + additional_production) :=
by sorry

end tractor_production_proof_l1102_110212


namespace weekend_rain_probability_l1102_110240

/-- The probability of rain on Saturday -/
def prob_rain_saturday : ℝ := 0.4

/-- The probability of rain on Sunday -/
def prob_rain_sunday : ℝ := 0.5

/-- The probabilities are independent -/
axiom independence : True

/-- The probability of rain on at least one day over the weekend -/
def prob_rain_weekend : ℝ := 1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday)

theorem weekend_rain_probability :
  prob_rain_weekend = 0.7 :=
sorry

end weekend_rain_probability_l1102_110240


namespace unique_quadrilateral_from_centers_l1102_110248

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Checks if a circle can be inscribed in a quadrilateral -/
def hasInscribedCircle (q : Quadrilateral) : Prop := sorry

/-- Checks if a circle can be circumscribed around a quadrilateral -/
def hasCircumscribedCircle (q : Quadrilateral) : Prop := sorry

/-- Gets the center of the inscribed circle of a quadrilateral -/
def getInscribedCenter (q : Quadrilateral) : Point2D := sorry

/-- Gets the center of the circumscribed circle of a quadrilateral -/
def getCircumscribedCenter (q : Quadrilateral) : Point2D := sorry

/-- Gets the intersection point of lines connecting midpoints of opposite sides -/
def getMidpointIntersection (q : Quadrilateral) : Point2D := sorry

/-- Theorem: A unique quadrilateral can be determined from its inscribed circle center,
    circumscribed circle center, and the intersection of midpoint lines -/
theorem unique_quadrilateral_from_centers
  (I O M : Point2D) :
  ∃! q : Quadrilateral,
    hasInscribedCircle q ∧
    hasCircumscribedCircle q ∧
    getInscribedCenter q = I ∧
    getCircumscribedCenter q = O ∧
    getMidpointIntersection q = M :=
  sorry

end unique_quadrilateral_from_centers_l1102_110248


namespace transylvanian_must_be_rational_l1102_110262

/-- Represents the state of a person's mind -/
inductive MindState
| Rational
| Lost

/-- Represents a person -/
structure Person where
  mindState : MindState

/-- Represents the claim made by a person -/
def claim (p : Person) : Prop :=
  p.mindState = MindState.Lost

/-- A person with a lost mind cannot make a truthful claim about their condition -/
axiom lost_mind_cannot_claim : ∀ (p : Person), p.mindState = MindState.Lost → ¬(claim p)

/-- The theorem to be proved -/
theorem transylvanian_must_be_rational (p : Person) (makes_claim : claim p) :
  p.mindState = MindState.Rational := by
  sorry

end transylvanian_must_be_rational_l1102_110262


namespace total_cost_calculation_l1102_110269

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 2.44

/-- The number of sandwiches -/
def num_sandwiches : ℕ := 2

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 0.87

/-- The number of sodas -/
def num_sodas : ℕ := 4

/-- The total cost of the order -/
def total_cost : ℚ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem total_cost_calculation :
  total_cost = 8.36 := by sorry

end total_cost_calculation_l1102_110269


namespace base_nine_to_ten_l1102_110250

theorem base_nine_to_ten : 
  (3 * 9^3 + 7 * 9^2 + 2 * 9^1 + 5 * 9^0) = 2777 := by
  sorry

end base_nine_to_ten_l1102_110250


namespace final_temperature_is_correct_l1102_110263

/-- Calculates the final temperature after a series of adjustments --/
def finalTemperature (initial : ℝ) : ℝ :=
  let temp1 := initial * 2
  let temp2 := temp1 - 30
  let temp3 := temp2 * 0.7
  let temp4 := temp3 + 24
  let temp5 := temp4 * 0.9
  let temp6 := temp5 + 8
  let temp7 := temp6 * 1.2
  temp7 - 15

/-- Theorem stating that the final temperature is 58.32 degrees --/
theorem final_temperature_is_correct : 
  abs (finalTemperature 40 - 58.32) < 0.01 := by
  sorry

end final_temperature_is_correct_l1102_110263


namespace factorization_equality_l1102_110283

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := by
  sorry

end factorization_equality_l1102_110283


namespace triangle_angle_inequalities_l1102_110226

theorem triangle_angle_inequalities (α β γ : Real) 
  (h : α + β + γ = π) : 
  (Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) ≤ 1/8) ∧
  (Real.cos α * Real.cos β * Real.cos γ ≤ 1/8) := by
  sorry

end triangle_angle_inequalities_l1102_110226


namespace sum_of_squares_divisible_by_four_l1102_110243

theorem sum_of_squares_divisible_by_four (n : ℤ) :
  ∃ k : ℤ, (2*n)^2 + (2*n + 2)^2 + (2*n + 4)^2 = 4 * k := by
  sorry

end sum_of_squares_divisible_by_four_l1102_110243


namespace jet_distance_l1102_110202

/-- Given a jet that travels 580 miles in 2 hours, prove that it will travel 2900 miles in 10 hours. -/
theorem jet_distance (distance : ℝ) (time : ℝ) (new_time : ℝ) 
    (h1 : distance = 580) 
    (h2 : time = 2) 
    (h3 : new_time = 10) : 
  (distance / time) * new_time = 2900 := by
  sorry

end jet_distance_l1102_110202


namespace d_in_N_l1102_110277

def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n}
def N : Set ℤ := {x | ∃ n : ℤ, |x| = 3 * n + 1}
def P : Set ℤ := {x | ∃ n : ℤ, x = 3 * n - 1}

theorem d_in_N (a b c : ℤ) (ha : a ∈ M) (hb : b ∈ N) (hc : c ∈ P) :
  (a - b + c) ∈ N := by
  sorry

end d_in_N_l1102_110277


namespace peppers_total_weight_l1102_110223

theorem peppers_total_weight : 
  let green_peppers : Float := 0.3333333333333333
  let red_peppers : Float := 0.3333333333333333
  let yellow_peppers : Float := 0.25
  let orange_peppers : Float := 0.5
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.4166666666666665 := by
  sorry

end peppers_total_weight_l1102_110223


namespace fiveDigitIntegersCount_eq_ten_l1102_110237

/-- The number of permutations of n elements with repetitions, where r₁, r₂, ..., rₖ
    are the repetition counts of each repeated element. -/
def permutationsWithRepetition (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

/-- The number of different five-digit integers formed using the digits 3, 3, 3, 8, and 8. -/
def fiveDigitIntegersCount : ℕ :=
  permutationsWithRepetition 5 [3, 2]

theorem fiveDigitIntegersCount_eq_ten : fiveDigitIntegersCount = 10 := by
  sorry

end fiveDigitIntegersCount_eq_ten_l1102_110237


namespace four_isosceles_triangles_l1102_110245

/-- A point on the grid --/
structure Point where
  x : Int
  y : Int

/-- A triangle defined by three points --/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Check if a triangle is isosceles --/
def isIsosceles (t : Triangle) : Bool :=
  let d12 := (t.p1.x - t.p2.x)^2 + (t.p1.y - t.p2.y)^2
  let d23 := (t.p2.x - t.p3.x)^2 + (t.p2.y - t.p3.y)^2
  let d31 := (t.p3.x - t.p1.x)^2 + (t.p3.y - t.p1.y)^2
  d12 = d23 || d23 = d31 || d31 = d12

/-- The list of triangles on the grid --/
def triangles : List Triangle := [
  { p1 := { x := 1, y := 6 }, p2 := { x := 3, y := 6 }, p3 := { x := 2, y := 3 } },
  { p1 := { x := 4, y := 2 }, p2 := { x := 4, y := 4 }, p3 := { x := 6, y := 2 } },
  { p1 := { x := 0, y := 0 }, p2 := { x := 3, y := 1 }, p3 := { x := 6, y := 0 } },
  { p1 := { x := 7, y := 3 }, p2 := { x := 6, y := 5 }, p3 := { x := 9, y := 3 } },
  { p1 := { x := 8, y := 0 }, p2 := { x := 9, y := 2 }, p3 := { x := 10, y := 0 } }
]

theorem four_isosceles_triangles :
  (triangles.filter isIsosceles).length = 4 := by sorry


end four_isosceles_triangles_l1102_110245


namespace work_completion_time_l1102_110280

-- Define the work rates and time worked by Y
def x_rate : ℚ := 1 / 24
def y_rate : ℚ := 1 / 16
def y_days_worked : ℕ := 10

-- Define the theorem
theorem work_completion_time :
  let total_work : ℚ := 1
  let work_done_by_y : ℚ := y_rate * y_days_worked
  let remaining_work : ℚ := total_work - work_done_by_y
  let days_needed_by_x : ℚ := remaining_work / x_rate
  days_needed_by_x = 9 := by sorry

end work_completion_time_l1102_110280


namespace special_function_inequality_l1102_110275

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  has_derivative : Differentiable ℝ f
  symmetric : ∀ x, f x = 6 * x^2 - f (-x)
  derivative_bound : ∀ x, x < 0 → 2 * deriv f x + 1 < 12 * x

/-- The main theorem -/
theorem special_function_inequality (sf : SpecialFunction) :
  ∀ m : ℝ, sf.f (m + 2) ≤ sf.f (-2 * m) + 12 * m + 12 - 9 * m^2 ↔ m ≥ - 2/3 :=
by sorry

end special_function_inequality_l1102_110275


namespace triangle_angle_cosine_l1102_110284

theorem triangle_angle_cosine (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  A + C = 2 * B ∧ -- Given condition
  1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B -- Given condition
  → Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 := by
    sorry

end triangle_angle_cosine_l1102_110284


namespace kim_initial_classes_l1102_110264

def initial_classes (class_duration : ℕ) (dropped_classes : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours / class_duration) + dropped_classes

theorem kim_initial_classes :
  initial_classes 2 1 6 = 4 := by
  sorry

end kim_initial_classes_l1102_110264


namespace no_factors_of_polynomial_l1102_110298

theorem no_factors_of_polynomial (x : ℝ) : 
  let p (x : ℝ) := x^4 - 4*x^2 + 16
  let f1 (x : ℝ) := x^2 + 4
  let f2 (x : ℝ) := x^2 - 1
  let f3 (x : ℝ) := x^2 + 1
  let f4 (x : ℝ) := x^2 + 3*x + 2
  (∃ (y : ℝ), p x = f1 x * y) = False ∧
  (∃ (y : ℝ), p x = f2 x * y) = False ∧
  (∃ (y : ℝ), p x = f3 x * y) = False ∧
  (∃ (y : ℝ), p x = f4 x * y) = False :=
by sorry

end no_factors_of_polynomial_l1102_110298


namespace min_reciprocal_sum_l1102_110225

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 20) :
  (1 / a + 1 / b) ≥ (1 / 5 : ℝ) :=
sorry

end min_reciprocal_sum_l1102_110225


namespace lamps_remaining_lit_l1102_110289

/-- The number of lamps initially lit -/
def total_lamps : ℕ := 1997

/-- Function to count lamps that are multiples of a given number -/
def count_multiples (n : ℕ) : ℕ :=
  (total_lamps - (total_lamps % n)) / n

/-- Function to count lamps that are multiples of two given numbers -/
def count_common_multiples (a b : ℕ) : ℕ :=
  (total_lamps - (total_lamps % (a * b))) / (a * b)

/-- Function to count lamps that are multiples of three given numbers -/
def count_triple_multiples (a b c : ℕ) : ℕ :=
  (total_lamps - (total_lamps % (a * b * c))) / (a * b * c)

/-- The main theorem stating the number of lamps that remain lit -/
theorem lamps_remaining_lit : 
  total_lamps - 
  (count_multiples 2 - count_common_multiples 2 3 - count_common_multiples 2 5 + count_triple_multiples 2 3 5) -
  (count_multiples 3 - count_common_multiples 2 3 - count_common_multiples 3 5 + count_triple_multiples 2 3 5) -
  (count_multiples 5 - count_common_multiples 2 5 - count_common_multiples 3 5 + count_triple_multiples 2 3 5) = 999 := by
  sorry

end lamps_remaining_lit_l1102_110289


namespace total_cost_is_21_93_l1102_110297

/-- The amount Alyssa spent on grapes -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa spent on cherries -/
def cherries_cost : ℚ := 9.85

/-- The total amount Alyssa spent on fruits -/
def total_cost : ℚ := grapes_cost + cherries_cost

/-- Theorem stating that the total cost is equal to $21.93 -/
theorem total_cost_is_21_93 : total_cost = 21.93 := by
  sorry

end total_cost_is_21_93_l1102_110297


namespace pavan_travel_distance_l1102_110282

theorem pavan_travel_distance :
  ∀ (total_distance : ℝ),
  (total_distance / 2 / 30 + total_distance / 2 / 25 = 11) →
  total_distance = 150 := by
sorry

end pavan_travel_distance_l1102_110282


namespace bee_count_l1102_110230

theorem bee_count (initial_bees : ℕ) (incoming_bees : ℕ) : 
  initial_bees = 16 → incoming_bees = 9 → initial_bees + incoming_bees = 25 := by
  sorry

end bee_count_l1102_110230


namespace wednesday_saturday_earnings_difference_l1102_110281

def total_earnings : ℝ := 5182.50
def saturday_earnings : ℝ := 2662.50

theorem wednesday_saturday_earnings_difference :
  saturday_earnings - (total_earnings - saturday_earnings) = 142.50 := by
  sorry

end wednesday_saturday_earnings_difference_l1102_110281


namespace ellipse_equation_l1102_110206

/-- Represents an ellipse with its properties -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- The standard equation of an ellipse -/
def standardEquation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Conditions for the ellipse -/
def ellipseConditions (e : Ellipse) : Prop :=
  e.a = 2 * Real.sqrt 3 ∧
  e.c = Real.sqrt 3 ∧
  e.b^2 = e.a^2 - e.c^2 ∧
  e.a = 3 * e.b ∧
  standardEquation e 3 0

theorem ellipse_equation (e : Ellipse) (h : ellipseConditions e) :
  (∀ x y, standardEquation e x y ↔ x^2 / 12 + y^2 / 9 = 1) ∨
  (∀ x y, standardEquation e x y ↔ x^2 / 9 + y^2 / 12 = 1) :=
sorry

end ellipse_equation_l1102_110206


namespace tenth_term_of_specific_geometric_sequence_l1102_110215

/-- Given a geometric sequence with first term a and second term b,
    this function returns the nth term of the sequence. -/
def geometric_sequence_term (a b : ℚ) (n : ℕ) : ℚ :=
  let r := b / a
  a * r ^ (n - 1)

/-- Theorem stating that the 10th term of the geometric sequence
    with first term 8 and second term -16/3 is -4096/19683. -/
theorem tenth_term_of_specific_geometric_sequence :
  geometric_sequence_term 8 (-16/3) 10 = -4096/19683 := by
  sorry

#eval geometric_sequence_term 8 (-16/3) 10

end tenth_term_of_specific_geometric_sequence_l1102_110215


namespace quadratic_minimum_at_positive_l1102_110239

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 6

-- Theorem statement
theorem quadratic_minimum_at_positive (x : ℝ) :
  ∃ (x_min : ℝ), x_min > 0 ∧ ∀ (y : ℝ), f y ≥ f x_min :=
by
  sorry

end quadratic_minimum_at_positive_l1102_110239


namespace isosceles_triangle_base_length_l1102_110238

/-- An isosceles triangle with perimeter 60 and two equal sides of length x has a base of length 60 - 2x -/
theorem isosceles_triangle_base_length (x : ℝ) (h : x > 0) : 
  let y := 60 - 2*x
  (2*x + y = 60) ∧ (y = -2*x + 60) := by
  sorry

end isosceles_triangle_base_length_l1102_110238


namespace water_pricing_l1102_110290

/-- Water pricing problem -/
theorem water_pricing
  (a : ℝ) -- Previous year's water usage
  (k : ℝ) -- Proportionality coefficient
  (h_a : a > 0) -- Assumption: water usage is positive
  (h_k : k > 0) -- Assumption: coefficient is positive
  :
  -- 1. Revenue function
  let revenue (x : ℝ) := (a + k / (x - 2)) * (x - 1.8)
  -- 2. Minimum water price for 20% increase when k = 0.4a
  ∃ (x : ℝ), x = 2.4 ∧ 
    (∀ y ∈ Set.Icc 2.3 2.6, 
      revenue y ≥ 1.2 * (2.8 * a - 1.8 * a) → y ≥ x) ∧
    k = 0.4 * a →
    revenue x ≥ 1.2 * (2.8 * a - 1.8 * a)
  -- 3. Water price for minimum revenue and minimum revenue when k = 0.8a
  ∧ ∃ (x : ℝ), x = 2.4 ∧
    (∀ y ∈ Set.Icc 2.3 2.6, revenue x ≤ revenue y) ∧
    k = 0.8 * a →
    revenue x = 1.8 * a :=
by
  sorry

end water_pricing_l1102_110290


namespace system_sum_l1102_110288

theorem system_sum (x y z : ℝ) 
  (eq1 : x + y = 4)
  (eq2 : y + z = 6)
  (eq3 : z + x = 8) :
  x + y + z = 9 := by
  sorry

end system_sum_l1102_110288


namespace calculator_profit_l1102_110260

theorem calculator_profit : 
  let selling_price : ℝ := 64
  let profit_percentage : ℝ := 0.6
  let loss_percentage : ℝ := 0.2
  let cost_price1 : ℝ := selling_price / (1 + profit_percentage)
  let cost_price2 : ℝ := selling_price / (1 - loss_percentage)
  let total_cost : ℝ := cost_price1 + cost_price2
  let total_revenue : ℝ := 2 * selling_price
  total_revenue - total_cost = 8 := by
  sorry

end calculator_profit_l1102_110260


namespace race_distance_P_300_l1102_110205

/-- A race between two runners P and Q, where P is faster but Q gets a head start -/
structure Race where
  /-- The speed ratio of P to Q -/
  speed_ratio : ℝ
  /-- The head start given to Q in meters -/
  head_start : ℝ

/-- The distance run by P in the race -/
def distance_P (race : Race) : ℝ :=
  sorry

theorem race_distance_P_300 (race : Race) 
  (h_speed : race.speed_ratio = 1.25)
  (h_head_start : race.head_start = 60)
  (h_tie : distance_P race = distance_P race - race.head_start + race.head_start) :
  distance_P race = 300 :=
sorry

end race_distance_P_300_l1102_110205


namespace twenty_five_percent_problem_l1102_110296

theorem twenty_five_percent_problem : ∃ x : ℝ, (0.75 * 80 = 1.25 * x) ∧ (x = 48) := by
  sorry

end twenty_five_percent_problem_l1102_110296


namespace f_definition_l1102_110265

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 - 4

-- State the theorem
theorem f_definition : 
  (∀ x : ℝ, f (x - 2) = x^2 - 4*x) → 
  (∀ x : ℝ, f x = x^2 - 4) := by sorry

end f_definition_l1102_110265


namespace extra_fruits_count_l1102_110208

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 75

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 35

/-- The number of oranges ordered by the cafeteria -/
def oranges : ℕ := 40

/-- The number of bananas ordered by the cafeteria -/
def bananas : ℕ := 20

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 17

/-- The total number of fruits ordered by the cafeteria -/
def total_fruits : ℕ := red_apples + green_apples + oranges + bananas

/-- The number of extra fruits the cafeteria ended up with -/
def extra_fruits : ℕ := total_fruits - students_wanting_fruit

theorem extra_fruits_count : extra_fruits = 153 := by
  sorry

end extra_fruits_count_l1102_110208


namespace hexadecimal_to_decimal_l1102_110233

theorem hexadecimal_to_decimal (k : ℕ) : k > 0 → (1 * 6^3 + k * 6^1 + 5 = 239) → k = 3 := by
  sorry

end hexadecimal_to_decimal_l1102_110233


namespace complex_product_pure_imaginary_l1102_110287

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop := (z.re = 0) ∧ (z.im ≠ 0)

/-- Given that b is a real number and (1+bi)(2+i) is a pure imaginary number, b equals 2 -/
theorem complex_product_pure_imaginary (b : ℝ) 
  (h : IsPureImaginary ((1 + b * Complex.I) * (2 + Complex.I))) : b = 2 := by
  sorry

end complex_product_pure_imaginary_l1102_110287


namespace factor_polynomial_l1102_110293

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 270 * x^13 = 15 * x^7 * (5 - 18 * x^6) := by
  sorry

end factor_polynomial_l1102_110293


namespace committee_rearrangements_l1102_110292

def word : String := "COMMITTEE"

def vowels : List Char := ['O', 'I', 'E', 'E']
def consonants : List Char := ['C', 'M', 'M', 'T', 'T']

def vowel_arrangements : ℕ := 12
def consonant_m_positions : ℕ := 10
def consonant_t_positions : ℕ := 3

theorem committee_rearrangements :
  (vowel_arrangements * consonant_m_positions * consonant_t_positions) = 360 :=
sorry

end committee_rearrangements_l1102_110292


namespace triangle_3_4_5_l1102_110217

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Theorem stating that line segments of lengths 3, 4, and 5 can form a triangle -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end triangle_3_4_5_l1102_110217


namespace f_properties_imply_a_equals_four_l1102_110247

/-- A function f(x) = x^2 - ax that is decreasing on (-∞, 2] and increasing on (2, +∞) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 - a*x

/-- The property of f being decreasing on (-∞, 2] -/
def decreasing_on_left (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 2 → f a x > f a y

/-- The property of f being increasing on (2, +∞) -/
def increasing_on_right (a : ℝ) : Prop :=
  ∀ x y, 2 < x → x < y → f a x < f a y

/-- Theorem stating that if f(x) = x^2 - ax is decreasing on (-∞, 2] and increasing on (2, +∞), then a = 4 -/
theorem f_properties_imply_a_equals_four :
  ∀ a : ℝ, decreasing_on_left a → increasing_on_right a → a = 4 :=
by
  sorry

end f_properties_imply_a_equals_four_l1102_110247
