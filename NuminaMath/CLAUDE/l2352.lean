import Mathlib

namespace NUMINAMATH_CALUDE_sum_interior_angles_octagon_l2352_235290

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The sum of the interior angles of an octagon is 1080° -/
theorem sum_interior_angles_octagon :
  sum_interior_angles octagon_sides = 1080 := by
  sorry


end NUMINAMATH_CALUDE_sum_interior_angles_octagon_l2352_235290


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l2352_235264

theorem largest_prime_factors_difference (n : Nat) (h : n = 184437) :
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p > q ∧
  (∀ r : Nat, Prime r → r ∣ n → r ≤ p) ∧
  (p ∣ n) ∧ (q ∣ n) ∧ (p - q = 8776) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l2352_235264


namespace NUMINAMATH_CALUDE_painted_cube_probability_l2352_235248

/-- Represents a 3x3x3 cube with two adjacent faces painted -/
structure PaintedCube where
  size : Nat
  painted_faces : Nat

/-- Counts the number of cubes with exactly two painted faces -/
def count_two_painted (cube : PaintedCube) : Nat :=
  4  -- The edge cubes between the two painted faces

/-- Counts the number of cubes with no painted faces -/
def count_no_painted (cube : PaintedCube) : Nat :=
  9  -- The interior cubes not visible from any painted face

/-- Calculates the total number of ways to select two cubes -/
def total_selections (cube : PaintedCube) : Nat :=
  (cube.size^3 * (cube.size^3 - 1)) / 2

/-- The main theorem to prove -/
theorem painted_cube_probability (cube : PaintedCube) 
  (h1 : cube.size = 3) 
  (h2 : cube.painted_faces = 2) : 
  (count_two_painted cube * count_no_painted cube) / total_selections cube = 4 / 39 := by
  sorry


end NUMINAMATH_CALUDE_painted_cube_probability_l2352_235248


namespace NUMINAMATH_CALUDE_brave_children_count_l2352_235226

/-- Represents the arrangement of children on a bench -/
structure BenchArrangement where
  total_children : ℕ
  boy_girl_pairs : ℕ

/-- The initial arrangement with 2 children -/
def initial_arrangement : BenchArrangement :=
  { total_children := 2, boy_girl_pairs := 1 }

/-- The final arrangement with 22 children alternating boy-girl -/
def final_arrangement : BenchArrangement :=
  { total_children := 22, boy_girl_pairs := 21 }

/-- A child is brave if they create two new boy-girl pairs when sitting down -/
def brave_children (initial final : BenchArrangement) : ℕ :=
  (final.boy_girl_pairs - initial.boy_girl_pairs) / 2

theorem brave_children_count :
  brave_children initial_arrangement final_arrangement = 10 := by
  sorry

end NUMINAMATH_CALUDE_brave_children_count_l2352_235226


namespace NUMINAMATH_CALUDE_rhombus_area_example_l2352_235263

/-- Given a rhombus with height h and diagonal d, calculates its area -/
def rhombusArea (h d : ℝ) : ℝ := sorry

/-- Theorem: A rhombus with height 12 cm and diagonal 15 cm has an area of 150 cm² -/
theorem rhombus_area_example : rhombusArea 12 15 = 150 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_example_l2352_235263


namespace NUMINAMATH_CALUDE_sum_of_fractions_integer_l2352_235238

theorem sum_of_fractions_integer (a b : ℤ) :
  (a ≠ 0 ∧ b ≠ 0) →
  (∃ k : ℤ, (a : ℚ) / b + (b : ℚ) / a = k) ↔ (a = b ∨ a = -b) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_integer_l2352_235238


namespace NUMINAMATH_CALUDE_horse_and_saddle_cost_l2352_235229

/-- The total cost of a horse and saddle, given their relative costs -/
def total_cost (saddle_cost : ℕ) (horse_cost_multiplier : ℕ) : ℕ :=
  saddle_cost + horse_cost_multiplier * saddle_cost

/-- Theorem: The total cost of a horse and saddle is $5000 -/
theorem horse_and_saddle_cost :
  total_cost 1000 4 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_horse_and_saddle_cost_l2352_235229


namespace NUMINAMATH_CALUDE_rectangle_side_increase_l2352_235270

theorem rectangle_side_increase (increase_factor : Real) :
  increase_factor > 0 →
  (1 + increase_factor)^2 = 1.8225 →
  increase_factor = 0.35 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_increase_l2352_235270


namespace NUMINAMATH_CALUDE_dimitri_burger_calories_l2352_235242

/-- Given that Dimitri eats 3 burgers per day and each burger has 20 calories,
    prove that the total calories consumed after two days is 120 calories. -/
theorem dimitri_burger_calories : 
  let burgers_per_day : ℕ := 3
  let calories_per_burger : ℕ := 20
  let days : ℕ := 2
  burgers_per_day * calories_per_burger * days = 120 := by
  sorry


end NUMINAMATH_CALUDE_dimitri_burger_calories_l2352_235242


namespace NUMINAMATH_CALUDE_fourth_root_13824000_l2352_235218

theorem fourth_root_13824000 : (62 : ℕ)^4 = 13824000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_13824000_l2352_235218


namespace NUMINAMATH_CALUDE_red_peaches_count_l2352_235207

/-- The number of red peaches in a basket with yellow, green, and red peaches. -/
def num_red_peaches (yellow green red_and_green : ℕ) : ℕ :=
  red_and_green - green

/-- Theorem stating that the number of red peaches is 6. -/
theorem red_peaches_count :
  let yellow : ℕ := 90
  let green : ℕ := 16
  let red_and_green : ℕ := 22
  num_red_peaches yellow green red_and_green = 6 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l2352_235207


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2352_235221

/-- The number of ways to choose 2 boxes out of 4 -/
def choose_empty_boxes : ℕ := 6

/-- The number of ways to place 4 different balls into 2 boxes, with at least one ball in each box -/
def place_balls : ℕ := 14

/-- The total number of ways to place 4 different balls into 4 numbered boxes such that exactly two boxes are empty -/
def total_ways : ℕ := choose_empty_boxes * place_balls

theorem balls_in_boxes :
  total_ways = 84 :=
sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2352_235221


namespace NUMINAMATH_CALUDE_total_fertilizer_used_l2352_235245

/-- The amount of fertilizer used per day for the first 9 days -/
def normal_amount : ℕ := 2

/-- The number of days the florist uses the normal amount of fertilizer -/
def normal_days : ℕ := 9

/-- The extra amount of fertilizer used on the final day -/
def extra_amount : ℕ := 4

/-- The total number of days the florist uses fertilizer -/
def total_days : ℕ := normal_days + 1

/-- Theorem: The total amount of fertilizer used over 10 days is 24 pounds -/
theorem total_fertilizer_used : 
  normal_amount * normal_days + (normal_amount + extra_amount) = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_fertilizer_used_l2352_235245


namespace NUMINAMATH_CALUDE_max_non_managers_proof_l2352_235251

/-- Represents a department in the company -/
structure Department where
  managers : ℕ
  nonManagers : ℕ

/-- The ratio condition for managers to non-managers -/
def validRatio (d : Department) : Prop :=
  (d.managers : ℚ) / d.nonManagers > 7 / 24

/-- The maximum number of non-managers allowed -/
def maxNonManagers : ℕ := 27

/-- The minimum number of managers required -/
def minManagers : ℕ := 8

theorem max_non_managers_proof (d : Department) 
    (h1 : d.managers ≥ minManagers) 
    (h2 : validRatio d) 
    (h3 : d.nonManagers ≤ maxNonManagers) :
    d.nonManagers = maxNonManagers :=
  sorry

#check max_non_managers_proof

end NUMINAMATH_CALUDE_max_non_managers_proof_l2352_235251


namespace NUMINAMATH_CALUDE_original_price_after_discount_l2352_235291

theorem original_price_after_discount (P : ℝ) : 
  P * (1 - 0.2) = P - 50 → P = 250 := by
  sorry

end NUMINAMATH_CALUDE_original_price_after_discount_l2352_235291


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2352_235266

-- Define the universal set U
def U : Set Nat := {1, 3, 5, 6, 8}

-- Define set A
def A : Set Nat := {1, 6}

-- Define set B
def B : Set Nat := {5, 6, 8}

-- Theorem to prove
theorem complement_intersection_equals_set :
  (U \ A) ∩ B = {5, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2352_235266


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2352_235228

/-- A quadratic function f with parameter t -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (x - (t + 2) / 2)^2 - t^2 / 4

/-- The theorem stating the properties of the quadratic function and the value of t -/
theorem quadratic_function_properties (t : ℝ) :
  t ≠ 0 ∧
  f t ((t + 2) / 2) = -t^2 / 4 ∧
  f t 1 = 0 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) (1/2), f t x ≥ -5) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) (1/2), f t x = -5) →
  t = -9/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2352_235228


namespace NUMINAMATH_CALUDE_parametric_equation_solution_l2352_235271

theorem parametric_equation_solution (a b : ℝ) (h1 : a ≠ 2 * b) (h2 : a ≠ -3 * b) :
  ∃! x : ℝ, (a * x - 3) / (b * x + 1) = 2 :=
by
  use 5 / (a - 2 * b)
  sorry

end NUMINAMATH_CALUDE_parametric_equation_solution_l2352_235271


namespace NUMINAMATH_CALUDE_farm_food_calculation_l2352_235257

/-- Given a farm with sheep and horses, calculate the daily food requirement per horse -/
theorem farm_food_calculation (sheep_count horse_count total_food : ℕ) 
  (h1 : sheep_count = 56)
  (h2 : sheep_count = horse_count)
  (h3 : total_food = 12880) :
  total_food / horse_count = 230 := by
sorry

end NUMINAMATH_CALUDE_farm_food_calculation_l2352_235257


namespace NUMINAMATH_CALUDE_triangle_transformation_exists_l2352_235203

/-- Triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Line represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflect a point over a line -/
def reflect (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

/-- Translate a point by a vector -/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Apply reflection and translation to a triangle -/
def transformTriangle (t : Triangle) (l : Line) (v : ℝ × ℝ) : Triangle :=
  { A := translate (reflect t.A l) v
  , B := translate (reflect t.B l) v
  , C := translate (reflect t.C l) v }

theorem triangle_transformation_exists :
  ∃ (l : Line) (v : ℝ × ℝ),
    let t1 : Triangle := { A := (0, 0), B := (15, 0), C := (0, 5) }
    let t2 : Triangle := { A := (17.2, 19.6), B := (26.2, 6.6), C := (22, 21) }
    transformTriangle t2 l v = t1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_transformation_exists_l2352_235203


namespace NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_difference_l2352_235220

theorem quadratic_equation_from_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_abs : |x - y| = 12) : 
  (∀ z : ℝ, z^2 - 10*z - 11 = 0 ↔ z = x ∨ z = y) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_difference_l2352_235220


namespace NUMINAMATH_CALUDE_rectangular_solid_width_l2352_235282

/-- The surface area of a rectangular solid given its length, width, and depth. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a rectangular solid with length 9 meters, depth 5 meters, 
    and surface area 314 square meters is 8 meters. -/
theorem rectangular_solid_width : 
  ∃ (w : ℝ), w = 8 ∧ surface_area 9 w 5 = 314 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_width_l2352_235282


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l2352_235231

theorem coefficient_x3y5_in_expansion : ∀ (x y : ℝ),
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℝ) * x^(8 - k) * y^k) =
  56 * x^3 * y^5 + (Finset.range 9).sum (fun k => if k ≠ 5 then (Nat.choose 8 k : ℝ) * x^(8 - k) * y^k else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l2352_235231


namespace NUMINAMATH_CALUDE_white_go_stones_l2352_235297

theorem white_go_stones (total : ℕ) (difference : ℕ) (white : ℕ) (black : ℕ) : 
  total = 120 →
  difference = 36 →
  white = black + difference →
  total = white + black →
  white = 78 := by
sorry

end NUMINAMATH_CALUDE_white_go_stones_l2352_235297


namespace NUMINAMATH_CALUDE_sequence_general_term_l2352_235246

/-- Given a sequence {a_n} with sum of first n terms S_n = (2/3)a_n + 1/3,
    prove that the general term formula is a_n = (-2)^(n-1) -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = (2/3) * a n + 1/3) →
  ∃ C : ℝ, ∀ n : ℕ, a n = C * (-2)^(n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2352_235246


namespace NUMINAMATH_CALUDE_ax_plus_by_fifth_power_l2352_235241

theorem ax_plus_by_fifth_power (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 6)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = -360 := by sorry

end NUMINAMATH_CALUDE_ax_plus_by_fifth_power_l2352_235241


namespace NUMINAMATH_CALUDE_absolute_sum_of_roots_greater_than_four_l2352_235278

theorem absolute_sum_of_roots_greater_than_four 
  (p : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) 
  (h2 : x₁^2 + p*x₁ + 4 = 0) 
  (h3 : x₂^2 + p*x₂ + 4 = 0) : 
  |x₁ + x₂| > 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_sum_of_roots_greater_than_four_l2352_235278


namespace NUMINAMATH_CALUDE_yellow_score_mixture_l2352_235237

theorem yellow_score_mixture (white_ratio black_ratio total_yellow : ℕ) 
  (h1 : white_ratio = 7)
  (h2 : black_ratio = 6)
  (h3 : total_yellow = 78) :
  (2 : ℚ) / 3 * (white_ratio - black_ratio) / (white_ratio + black_ratio) * total_yellow = 4 := by
  sorry

end NUMINAMATH_CALUDE_yellow_score_mixture_l2352_235237


namespace NUMINAMATH_CALUDE_job_completion_time_l2352_235288

/-- Given that A can complete a job in 10 hours alone and A and D together can complete a job in 5 hours, prove that D can complete the job in 10 hours alone. -/
theorem job_completion_time (a_time : ℝ) (ad_time : ℝ) (d_time : ℝ) 
    (ha : a_time = 10) 
    (had : ad_time = 5) : 
  d_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2352_235288


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2352_235281

-- Define the sets M and N
def M : Set ℝ := {x | x + 1 ≥ 0}
def N : Set ℝ := {x | x - 2 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2352_235281


namespace NUMINAMATH_CALUDE_bingley_has_six_bracelets_l2352_235289

/-- The number of bracelets Bingley has remaining after exchanges with Kelly and his sister -/
def bingleysRemainingBracelets (bingleyInitial : ℕ) (kellyInitial : ℕ) : ℕ :=
  let bingleyAfterKelly := bingleyInitial + kellyInitial / 4
  bingleyAfterKelly - bingleyAfterKelly / 3

/-- Theorem stating that Bingley has 6 bracelets remaining -/
theorem bingley_has_six_bracelets :
  bingleysRemainingBracelets 5 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bingley_has_six_bracelets_l2352_235289


namespace NUMINAMATH_CALUDE_task_completion_probability_l2352_235227

theorem task_completion_probability (p1 p2 : ℚ) 
  (h1 : p1 = 2/3) 
  (h2 : p2 = 3/5) : 
  p1 * (1 - p2) = 4/15 := by
sorry

end NUMINAMATH_CALUDE_task_completion_probability_l2352_235227


namespace NUMINAMATH_CALUDE_converse_correct_l2352_235273

/-- The original statement -/
def original_statement (x : ℝ) : Prop := x^2 = 1 → x = 1

/-- The converse statement -/
def converse_statement (x : ℝ) : Prop := x^2 ≠ 1 → x ≠ 1

/-- Theorem stating that the converse_statement is indeed the converse of the original_statement -/
theorem converse_correct :
  converse_statement = (fun x => ¬(original_statement x)) := by sorry

end NUMINAMATH_CALUDE_converse_correct_l2352_235273


namespace NUMINAMATH_CALUDE_vasya_drove_two_fifths_l2352_235200

/-- Represents the fraction of total distance driven by each person -/
structure DistanceFractions where
  anton : ℝ
  vasya : ℝ
  sasha : ℝ
  dima : ℝ

/-- Conditions of the driving problem -/
def driving_conditions (d : DistanceFractions) : Prop :=
  d.anton = d.vasya / 2 ∧
  d.sasha = d.anton + d.dima ∧
  d.dima = 1 / 10 ∧
  d.anton + d.vasya + d.sasha + d.dima = 1

/-- Theorem stating that Vasya drove 2/5 of the total distance -/
theorem vasya_drove_two_fifths :
  ∀ d : DistanceFractions, driving_conditions d → d.vasya = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_vasya_drove_two_fifths_l2352_235200


namespace NUMINAMATH_CALUDE_combined_score_is_78_l2352_235261

/-- Represents the score of a player in either football or basketball -/
structure PlayerScore where
  name : String
  score : ℕ

/-- Calculates the total score of a list of players -/
def totalScore (players : List PlayerScore) : ℕ :=
  players.foldl (fun acc p => acc + p.score) 0

/-- The combined score of football and basketball games -/
theorem combined_score_is_78 (bruce michael jack sarah andy lily : PlayerScore) :
  bruce.name = "Bruce" ∧ bruce.score = 4 ∧
  michael.name = "Michael" ∧ michael.score = 2 * bruce.score ∧
  jack.name = "Jack" ∧ jack.score = bruce.score - 1 ∧
  sarah.name = "Sarah" ∧ sarah.score = jack.score / 2 ∧
  andy.name = "Andy" ∧ andy.score = 22 ∧
  lily.name = "Lily" ∧ lily.score = andy.score + 18 →
  totalScore [bruce, michael, jack, sarah, andy, lily] = 78 := by
sorry

#eval totalScore [
  {name := "Bruce", score := 4},
  {name := "Michael", score := 8},
  {name := "Jack", score := 3},
  {name := "Sarah", score := 1},
  {name := "Andy", score := 22},
  {name := "Lily", score := 40}
]

end NUMINAMATH_CALUDE_combined_score_is_78_l2352_235261


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_b_greater_than_nine_l2352_235232

/-- If the solution set of the inequality |x-4|-|x+5| ≥ b about x is empty, then b > 9 -/
theorem empty_solution_set_implies_b_greater_than_nine (b : ℝ) :
  (∀ x : ℝ, |x - 4| - |x + 5| < b) → b > 9 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_b_greater_than_nine_l2352_235232


namespace NUMINAMATH_CALUDE_rod_cutting_l2352_235292

/-- Given a rod of length 38.25 meters that can be cut into 45 pieces,
    prove that each piece is 85 centimeters long. -/
theorem rod_cutting (rod_length : Real) (num_pieces : Nat) :
  rod_length = 38.25 ∧ num_pieces = 45 →
  (rod_length / num_pieces) * 100 = 85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l2352_235292


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2352_235252

theorem smallest_solution_of_equation (y : ℝ) :
  (3 * y^2 + 33 * y - 90 = y * (y + 16)) →
  y ≥ -10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2352_235252


namespace NUMINAMATH_CALUDE_cube_roots_opposite_implies_a_eq_neg_three_l2352_235244

theorem cube_roots_opposite_implies_a_eq_neg_three (a : ℝ) :
  (∃ x : ℝ, x^3 = 2*a + 1 ∧ (-x)^3 = 2 - a) → a = -3 := by
sorry

end NUMINAMATH_CALUDE_cube_roots_opposite_implies_a_eq_neg_three_l2352_235244


namespace NUMINAMATH_CALUDE_dispatch_plans_eq_28_l2352_235265

/-- Given a set of athletes with the following properties:
  * There are 9 athletes in total
  * 5 athletes can play basketball
  * 6 athletes can play soccer
This function calculates the number of ways to select one athlete for basketball
and one for soccer. -/
def dispatch_plans (total : Nat) (basketball : Nat) (soccer : Nat) : Nat :=
  sorry

/-- Theorem stating that the number of dispatch plans for the given conditions is 28. -/
theorem dispatch_plans_eq_28 : dispatch_plans 9 5 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_dispatch_plans_eq_28_l2352_235265


namespace NUMINAMATH_CALUDE_wheel_probability_l2352_235217

theorem wheel_probability (P_A P_B P_C P_D : ℚ) : 
  P_A = 1/4 → P_B = 1/3 → P_C = 1/6 → P_A + P_B + P_C + P_D = 1 → P_D = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l2352_235217


namespace NUMINAMATH_CALUDE_quadratic_derivative_condition_l2352_235260

/-- Given a quadratic function f(x) = 3x² + bx + c, prove that if the derivative at x = b is 14, then b = 2 -/
theorem quadratic_derivative_condition (b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + b * x + c
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f (b + Δx) - f b) / Δx) - 14| < ε) → 
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_derivative_condition_l2352_235260


namespace NUMINAMATH_CALUDE_parallelogram_bisector_theorem_l2352_235240

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Representation of a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := sorry

/-- The length between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on a line defined by two other points -/
def isOnLine (p1 p2 p : Point) : Prop := sorry

/-- Check if a line is an angle bisector -/
def isAngleBisector (vertex p1 p2 : Point) : Prop := sorry

/-- The main theorem -/
theorem parallelogram_bisector_theorem (ABCD : Parallelogram) (E F : Point) :
  perimeter ABCD = 32 →
  isAngleBisector ABCD.C ABCD.D ABCD.B →
  isOnLine ABCD.A ABCD.D E →
  isOnLine ABCD.A ABCD.B F →
  distance ABCD.A E = 2 →
  (distance ABCD.B F = 7 ∨ distance ABCD.B F = 9) := by sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_theorem_l2352_235240


namespace NUMINAMATH_CALUDE_min_removals_for_three_by_three_l2352_235224

/-- Represents a 3x3 square figure made of matches -/
structure MatchSquare where
  size : Nat
  total_matches : Nat
  matches_per_side : Nat

/-- Defines the properties of our specific 3x3 match square -/
def three_by_three_square : MatchSquare :=
  { size := 3
  , total_matches := 24
  , matches_per_side := 1 }

/-- Defines what it means for a number of removals to be valid -/
def is_valid_removal (square : MatchSquare) (removals : Nat) : Prop :=
  removals ≤ square.total_matches ∧
  ∀ (x y : Nat), x < square.size ∧ y < square.size →
    ∃ (side : Nat), side < 4 ∧ 
      (removals > (x * square.size + y) * 4 + side)

/-- The main theorem statement -/
theorem min_removals_for_three_by_three (square : MatchSquare) 
  (h1 : square = three_by_three_square) :
  ∃ (n : Nat), is_valid_removal square n ∧
    ∀ (m : Nat), m < n → ¬ is_valid_removal square m :=
  sorry

end NUMINAMATH_CALUDE_min_removals_for_three_by_three_l2352_235224


namespace NUMINAMATH_CALUDE_sqrt_ratio_implies_sum_ratio_l2352_235293

theorem sqrt_ratio_implies_sum_ratio (x y : ℝ) (h : x > 0) (k : y > 0) :
  (Real.sqrt x / Real.sqrt y = 5) → ((x + y) / (2 * y) = 13) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ratio_implies_sum_ratio_l2352_235293


namespace NUMINAMATH_CALUDE_inequality_implication_condition_l2352_235216

theorem inequality_implication_condition :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  ¬(∀ x : ℝ, |x - 1| < 2 → x * (x - 3) < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implication_condition_l2352_235216


namespace NUMINAMATH_CALUDE_cos_75_deg_l2352_235259

/-- Prove that cos 75° = (√6 - √2) / 4 using the angle sum identity for cosine with angles 60° and 15° -/
theorem cos_75_deg : 
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_deg_l2352_235259


namespace NUMINAMATH_CALUDE_monica_savings_l2352_235267

theorem monica_savings (weeks_per_cycle : ℕ) (num_cycles : ℕ) (total_per_cycle : ℚ) 
  (h1 : weeks_per_cycle = 60)
  (h2 : num_cycles = 5)
  (h3 : total_per_cycle = 4500) :
  (num_cycles * total_per_cycle) / (num_cycles * weeks_per_cycle) = 75 := by
  sorry

end NUMINAMATH_CALUDE_monica_savings_l2352_235267


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2352_235269

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 11*x + 30 = 0 →
  2 + x > 4 ∧ 4 + x > 2 ∧ 2 + 4 > x →
  2 + 4 + x = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2352_235269


namespace NUMINAMATH_CALUDE_jelly_cost_l2352_235239

/-- The cost of jelly for sandwiches --/
theorem jelly_cost (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (3 * B + 7 * J) = 378 → 
  (N * J * 7 : ℚ) / 100 = 294 / 100 := by
  sorry

end NUMINAMATH_CALUDE_jelly_cost_l2352_235239


namespace NUMINAMATH_CALUDE_regular_polygon_150_degree_angles_l2352_235255

/-- A regular polygon with interior angles measuring 150° has 12 sides. -/
theorem regular_polygon_150_degree_angles (n : ℕ) : 
  n > 2 → (∀ θ : ℝ, θ = 150 → n * θ = (n - 2) * 180) → n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degree_angles_l2352_235255


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_b_l2352_235294

theorem gcd_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 528 * k) :
  Nat.gcd (2 * b^4 + b^3 + 5 * b^2 + 6 * b + 132).natAbs b.natAbs = 132 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_b_l2352_235294


namespace NUMINAMATH_CALUDE_park_area_l2352_235285

/-- Given a rectangular park with length l and width w, where:
    1) l = 3w + 20
    2) The perimeter is 800 feet
    Prove that the area of the park is 28,975 square feet -/
theorem park_area (w l : ℝ) (h1 : l = 3 * w + 20) (h2 : 2 * l + 2 * w = 800) :
  w * l = 28975 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l2352_235285


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2352_235272

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4) :
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2352_235272


namespace NUMINAMATH_CALUDE_product_closest_to_127_l2352_235206

def product : ℝ := 2.5 * (50.5 + 0.25)

def options : List ℝ := [120, 125, 127, 130, 140]

theorem product_closest_to_127 :
  ∀ x ∈ options, x ≠ 127 → |product - 127| < |product - x| :=
by sorry

end NUMINAMATH_CALUDE_product_closest_to_127_l2352_235206


namespace NUMINAMATH_CALUDE_table_people_count_l2352_235210

/-- The number of seeds taken by n people in the first round -/
def first_round_seeds (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of seeds taken by n people in the second round -/
def second_round_seeds (n : ℕ) : ℕ := first_round_seeds n + n^2

/-- The difference in seeds taken between the second and first rounds -/
def seed_difference (n : ℕ) : ℕ := second_round_seeds n - first_round_seeds n

theorem table_people_count : 
  ∃ n : ℕ, n > 0 ∧ seed_difference n = 100 ∧ 
  (∀ m : ℕ, m > 0 → seed_difference m = 100 → m = n) :=
sorry

end NUMINAMATH_CALUDE_table_people_count_l2352_235210


namespace NUMINAMATH_CALUDE_unique_quadratic_trinomial_l2352_235219

theorem unique_quadratic_trinomial : ∃! (a b c : ℝ), 
  (∀ x : ℝ, (a + 1) * x^2 + b * x + c = 0 → (∃! y : ℝ, y = x)) ∧
  (∀ x : ℝ, a * x^2 + (b + 1) * x + c = 0 → (∃! y : ℝ, y = x)) ∧
  (∀ x : ℝ, a * x^2 + b * x + (c + 1) = 0 → (∃! y : ℝ, y = x)) ∧
  a = 1/8 ∧ b = -3/4 ∧ c = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_trinomial_l2352_235219


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l2352_235283

theorem product_of_sum_and_sum_of_squares (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^2 + b^2 = 6) : 
  a * b = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l2352_235283


namespace NUMINAMATH_CALUDE_last_remaining_card_l2352_235296

/-- The largest power of 2 less than or equal to n -/
def largestPowerOf2 (n : ℕ) : ℕ :=
  (Nat.log2 n).succ

/-- The process of eliminating cards -/
def cardElimination (n : ℕ) : ℕ :=
  let L := largestPowerOf2 n
  2 * (n - 2^L) + 1

theorem last_remaining_card (n : ℕ) (h : n > 0) :
  ∃ (k : ℕ), k ≤ n ∧ cardElimination n = k :=
sorry

end NUMINAMATH_CALUDE_last_remaining_card_l2352_235296


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2352_235280

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) = f x * f y - 2 * x * y) →
  (∀ x : ℝ, f x = 2 * x) ∨ (∀ x : ℝ, f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2352_235280


namespace NUMINAMATH_CALUDE_vector_dot_product_equals_three_l2352_235208

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  ab_length : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1
  bc_length : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def vec_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem vector_dot_product_equals_three 
  (A B C M : ℝ × ℝ) 
  (h : Triangle A B C) 
  (hm : vec_sub B M = vec_scale 2 (vec_sub A M)) : 
  dot_product (vec_sub C M) (vec_sub C A) = 3 := by
  sorry


end NUMINAMATH_CALUDE_vector_dot_product_equals_three_l2352_235208


namespace NUMINAMATH_CALUDE_exists_recurrence_sequence_l2352_235274

-- Define the sequence type
def RecurrenceSequence (x y : ℝ) := ℕ → ℝ

-- Define the recurrence relation property
def SatisfiesRecurrence (a : RecurrenceSequence x y) : Prop :=
  ∀ n : ℕ, a (n + 2) = x * a (n + 1) + y * a n

-- Define the boundedness property
def SatisfiesBoundedness (a : RecurrenceSequence x y) : Prop :=
  ∀ r : ℝ, r > 0 → ∃ i j : ℕ, i > 0 ∧ j > 0 ∧ |a i| < r ∧ r < |a j|

-- Define the non-zero property
def IsNonZero (a : RecurrenceSequence x y) : Prop :=
  ∀ n : ℕ, a n ≠ 0

-- Main theorem
theorem exists_recurrence_sequence :
  ∃ x y : ℝ, ∃ a : RecurrenceSequence x y,
    SatisfiesRecurrence a ∧ SatisfiesBoundedness a ∧ IsNonZero a := by
  sorry

end NUMINAMATH_CALUDE_exists_recurrence_sequence_l2352_235274


namespace NUMINAMATH_CALUDE_trapezoid_area_l2352_235204

/-- Represents a trapezoid ABCD with a circle passing through A, B, and touching C -/
structure TrapezoidWithCircle where
  /-- Length of CD -/
  cd : ℝ
  /-- Length of AE -/
  ae : ℝ
  /-- The circle is centered on diagonal AC -/
  circle_on_diagonal : Bool
  /-- BC is parallel to AD -/
  bc_parallel_ad : Bool
  /-- The circle passes through A and B -/
  circle_through_ab : Bool
  /-- The circle touches CD at C -/
  circle_touches_cd : Bool
  /-- The circle intersects AD at E -/
  circle_intersects_ad : Bool

/-- Calculate the area of the trapezoid ABCD -/
def calculate_area (t : TrapezoidWithCircle) : ℝ :=
  sorry

/-- Theorem stating that the area of the trapezoid ABCD is 204 -/
theorem trapezoid_area (t : TrapezoidWithCircle) 
  (h1 : t.cd = 6 * Real.sqrt 13)
  (h2 : t.ae = 8)
  (h3 : t.circle_on_diagonal)
  (h4 : t.bc_parallel_ad)
  (h5 : t.circle_through_ab)
  (h6 : t.circle_touches_cd)
  (h7 : t.circle_intersects_ad) :
  calculate_area t = 204 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2352_235204


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l2352_235277

theorem inserted_numbers_sum : ∃ (a b : ℝ), 
  4 < a ∧ a < b ∧ b < 16 ∧ 
  (b - a = a - 4) ∧
  (b * b = a * 16) ∧
  a + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l2352_235277


namespace NUMINAMATH_CALUDE_bicycle_race_fraction_l2352_235298

theorem bicycle_race_fraction (total_racers : ℕ) (total_wheels : ℕ) 
  (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) :
  total_racers = 40 →
  total_wheels = 96 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  ∃ (bicycles : ℕ) (tricycles : ℕ),
    bicycles + tricycles = total_racers ∧
    bicycles * bicycle_wheels + tricycles * tricycle_wheels = total_wheels ∧
    (bicycles : ℚ) / total_racers = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_race_fraction_l2352_235298


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l2352_235268

theorem polynomial_equality_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^3 - 1) * (x + 1)^7 = a₀ + a₁*(x + 3) + a₂*(x + 3)^2 + a₃*(x + 3)^3 + 
    a₄*(x + 3)^4 + a₅*(x + 3)^5 + a₆*(x + 3)^6 + a₇*(x + 3)^7 + a₈*(x + 3)^8 + 
    a₉*(x + 3)^9 + a₁₀*(x + 3)^10) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l2352_235268


namespace NUMINAMATH_CALUDE_num_divisors_2310_l2352_235233

/-- The number of positive divisors of a positive integer n -/
def numPositiveDivisors (n : ℕ+) : ℕ := sorry

/-- 2310 as a positive integer -/
def n : ℕ+ := 2310

/-- Theorem: The number of positive divisors of 2310 is 32 -/
theorem num_divisors_2310 : numPositiveDivisors n = 32 := by sorry

end NUMINAMATH_CALUDE_num_divisors_2310_l2352_235233


namespace NUMINAMATH_CALUDE_right_triangle_complex_roots_l2352_235287

theorem right_triangle_complex_roots : 
  ∃! (S : Finset ℂ), 
    (∀ z ∈ S, z ≠ 0 ∧ 
      (z.re * (z^6 - z).re + z.im * (z^6 - z).im = 0)) ∧ 
    S.card = 5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_complex_roots_l2352_235287


namespace NUMINAMATH_CALUDE_inequality_proof_l2352_235286

theorem inequality_proof (a b c d : ℝ) : 
  (a + b + c + d)^2 ≤ 3 * (a^2 + b^2 + c^2 + d^2) + 6 * a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2352_235286


namespace NUMINAMATH_CALUDE_at_least_one_accepted_l2352_235205

theorem at_least_one_accepted (prob_A prob_B : ℝ) 
  (h1 : 0 ≤ prob_A ∧ prob_A ≤ 1)
  (h2 : 0 ≤ prob_B ∧ prob_B ≤ 1)
  (independence : True) -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) = prob_A + prob_B - prob_A * prob_B :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_accepted_l2352_235205


namespace NUMINAMATH_CALUDE_power_simplification_l2352_235254

theorem power_simplification :
  ((5^13 / 5^11)^2 * 5^2) / 2^5 = 15625 / 32 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l2352_235254


namespace NUMINAMATH_CALUDE_symmetry_properties_l2352_235247

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ

-- Define a quadratic function type
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to get the symmetric line about x-axis
def symmetricAboutXAxis (l : Line) : Line :=
  { a := -l.a, b := -l.b }

-- Function to get the symmetric line about y-axis
def symmetricAboutYAxis (l : Line) : Line :=
  { a := -l.a, b := l.b }

-- Function to get the symmetric quadratic function about origin
def symmetricAboutOrigin (q : QuadraticFunction) : QuadraticFunction :=
  { a := q.a, b := -q.b, c := -q.c }

-- Theorem statements
theorem symmetry_properties (l : Line) (q : QuadraticFunction) :
  (symmetricAboutXAxis l = { a := -l.a, b := -l.b }) ∧
  (symmetricAboutYAxis l = { a := -l.a, b := l.b }) ∧
  (symmetricAboutOrigin q = { a := q.a, b := -q.b, c := -q.c }) := by
  sorry


end NUMINAMATH_CALUDE_symmetry_properties_l2352_235247


namespace NUMINAMATH_CALUDE_beach_house_pool_problem_l2352_235211

theorem beach_house_pool_problem (total_people : ℕ) (legs_in_pool : ℕ) (legs_per_person : ℕ) :
  total_people = 14 →
  legs_in_pool = 16 →
  legs_per_person = 2 →
  total_people - (legs_in_pool / legs_per_person) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_beach_house_pool_problem_l2352_235211


namespace NUMINAMATH_CALUDE_profit_maximization_and_cost_l2352_235279

/-- Represents the relationship between selling price and daily sales volume -/
def sales_volume (x : ℝ) : ℝ := -30 * x + 1500

/-- Calculates the daily sales profit -/
def sales_profit (x : ℝ) : ℝ := sales_volume x * (x - 30)

/-- Calculates the daily profit including additional cost a -/
def total_profit (x a : ℝ) : ℝ := sales_volume x * (x - 30 - a)

theorem profit_maximization_and_cost (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 10) :
  (∀ x, sales_profit x ≤ sales_profit 40) ∧
  (∃ x, 40 ≤ x ∧ x ≤ 45 ∧ total_profit x a = 2430) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_and_cost_l2352_235279


namespace NUMINAMATH_CALUDE_factorization_quadratic_l2352_235215

theorem factorization_quadratic (x : ℝ) : x^2 + 2*x = x*(x+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_quadratic_l2352_235215


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_product_l2352_235276

theorem sqrt_sum_difference_product (a b c d : ℝ) :
  Real.sqrt 75 + Real.sqrt 27 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 8 * Real.sqrt 3 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_product_l2352_235276


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l2352_235236

theorem nested_subtraction_simplification : 2 - (-2 - 2) - (-2 - (-2 - 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l2352_235236


namespace NUMINAMATH_CALUDE_bin_game_expectation_l2352_235258

theorem bin_game_expectation (k : ℕ+) : 
  let total_balls : ℕ := 8 + k
  let green_prob : ℚ := 8 / total_balls
  let purple_prob : ℚ := k / total_balls
  let expected_value : ℚ := green_prob * 3 + purple_prob * (-1)
  expected_value = 60 / 100 → k = 12 := by
sorry

end NUMINAMATH_CALUDE_bin_game_expectation_l2352_235258


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2352_235214

/-- A rectangle with whole number dimensions and perimeter 40 has a maximum area of 100 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l > 0 → w > 0 →
  2 * l + 2 * w = 40 →
  l * w ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2352_235214


namespace NUMINAMATH_CALUDE_prob_at_least_one_l2352_235209

/-- The probability of possessing at least one of two independent events,
    given their individual probabilities -/
theorem prob_at_least_one (p_ballpoint p_ink : ℚ) 
  (h_ballpoint : p_ballpoint = 3/5)
  (h_ink : p_ink = 2/3)
  (h_independent : True) -- Assumption of independence
  : p_ballpoint + p_ink - p_ballpoint * p_ink = 13/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_l2352_235209


namespace NUMINAMATH_CALUDE_sin_cos_rational_implies_natural_combination_l2352_235201

theorem sin_cos_rational_implies_natural_combination 
  (x y : ℝ) 
  (h1 : ∃ (a : ℚ), a > 0 ∧ Real.sin x + Real.cos y = a)
  (h2 : ∃ (b : ℚ), b > 0 ∧ Real.sin y + Real.cos x = b) :
  ∃ (m n : ℕ), ∃ (k : ℕ), m * Real.sin x + n * Real.cos x = k := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_rational_implies_natural_combination_l2352_235201


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l2352_235225

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) 
  (h : (2 - 3*I : ℂ) ^ 3 + a * (2 - 3*I : ℂ) ^ 2 - (2 - 3*I : ℂ) + b = 0) : 
  a = -1/2 ∧ b = 91/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l2352_235225


namespace NUMINAMATH_CALUDE_current_velocity_proof_l2352_235256

/-- The velocity of the current in a river, given the following conditions:
  1. A man can row at 5 kmph in still water.
  2. It takes him 1 hour to row to a place and come back.
  3. The place is 2.4 km away. -/
def current_velocity : ℝ := 1

/-- The man's rowing speed in still water (in kmph) -/
def rowing_speed : ℝ := 5

/-- The distance to the destination (in km) -/
def distance : ℝ := 2.4

/-- The total time for the round trip (in hours) -/
def total_time : ℝ := 1

theorem current_velocity_proof :
  (distance / (rowing_speed + current_velocity) +
   distance / (rowing_speed - current_velocity) = total_time) ∧
  (current_velocity > 0) ∧
  (current_velocity < rowing_speed) := by
  sorry

end NUMINAMATH_CALUDE_current_velocity_proof_l2352_235256


namespace NUMINAMATH_CALUDE_sum_of_integers_l2352_235202

theorem sum_of_integers (a b c : ℤ) :
  a = (b + c) / 3 →
  b = (a + c) / 5 →
  c = 35 →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2352_235202


namespace NUMINAMATH_CALUDE_vertical_angles_are_equal_converse_is_false_l2352_235234

-- Define what it means for angles to be vertical
def are_vertical_angles (α β : Real) : Prop := sorry

-- Define what it means for angles to be equal
def are_equal_angles (α β : Real) : Prop := α = β

-- Theorem stating that vertical angles are equal
theorem vertical_angles_are_equal (α β : Real) : 
  are_vertical_angles α β → are_equal_angles α β := by sorry

-- Theorem stating that the converse is false
theorem converse_is_false : 
  ¬(∀ α β : Real, are_equal_angles α β → are_vertical_angles α β) := by sorry

end NUMINAMATH_CALUDE_vertical_angles_are_equal_converse_is_false_l2352_235234


namespace NUMINAMATH_CALUDE_siblings_selection_probability_l2352_235222

theorem siblings_selection_probability 
  (p_ram : ℚ) (p_ravi : ℚ) (p_rina : ℚ)
  (h_ram : p_ram = 4/7)
  (h_ravi : p_ravi = 1/5)
  (h_rina : p_rina = 3/8) :
  p_ram * p_ravi * p_rina = 3/70 := by
sorry

end NUMINAMATH_CALUDE_siblings_selection_probability_l2352_235222


namespace NUMINAMATH_CALUDE_clippings_per_friend_l2352_235249

theorem clippings_per_friend 
  (num_friends : ℕ) 
  (total_glue_drops : ℕ) 
  (glue_drops_per_clipping : ℕ) 
  (h1 : num_friends = 7)
  (h2 : total_glue_drops = 126)
  (h3 : glue_drops_per_clipping = 6) :
  (total_glue_drops / glue_drops_per_clipping) / num_friends = 3 :=
by sorry

end NUMINAMATH_CALUDE_clippings_per_friend_l2352_235249


namespace NUMINAMATH_CALUDE_optimal_store_strategy_l2352_235299

/-- Represents the store's inventory and pricing strategy -/
structure Store where
  total_balls : Nat
  budget : Nat
  basketball_cost : Nat
  volleyball_cost : Nat
  basketball_price_ratio : Rat
  school_basketball_revenue : Nat
  school_volleyball_revenue : Nat
  school_volleyball_count_diff : Int

/-- Represents the store's pricing and purchase strategy -/
structure Strategy where
  basketball_price : Nat
  volleyball_price : Nat
  basketball_count : Nat
  volleyball_count : Nat

/-- Checks if the strategy satisfies all constraints -/
def is_valid_strategy (store : Store) (strategy : Strategy) : Prop :=
  strategy.basketball_count + strategy.volleyball_count = store.total_balls ∧
  strategy.basketball_count * store.basketball_cost + strategy.volleyball_count * store.volleyball_cost ≤ store.budget ∧
  strategy.basketball_price = (strategy.volleyball_price : Rat) * store.basketball_price_ratio ∧
  (store.school_basketball_revenue : Rat) / strategy.basketball_price - 
    (store.school_volleyball_revenue : Rat) / strategy.volleyball_price = store.school_volleyball_count_diff

/-- Calculates the profit after price reduction -/
def profit_after_reduction (store : Store) (strategy : Strategy) : Int :=
  (strategy.basketball_price - 3 - store.basketball_cost) * strategy.basketball_count +
  (strategy.volleyball_price - 2 - store.volleyball_cost) * strategy.volleyball_count

/-- Main theorem: Proves the optimal strategy for the store -/
theorem optimal_store_strategy (store : Store) 
    (h_store : store.total_balls = 200 ∧ 
               store.budget = 5000 ∧ 
               store.basketball_cost = 30 ∧ 
               store.volleyball_cost = 24 ∧ 
               store.basketball_price_ratio = 3/2 ∧
               store.school_basketball_revenue = 1800 ∧
               store.school_volleyball_revenue = 1500 ∧
               store.school_volleyball_count_diff = 10) :
  ∃ (strategy : Strategy),
    is_valid_strategy store strategy ∧
    strategy.basketball_price = 45 ∧
    strategy.volleyball_price = 30 ∧
    strategy.basketball_count = 33 ∧
    strategy.volleyball_count = 167 ∧
    ∀ (other_strategy : Strategy),
      is_valid_strategy store other_strategy →
      profit_after_reduction store strategy ≥ profit_after_reduction store other_strategy :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_store_strategy_l2352_235299


namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_l2352_235262

/-- Represents a rectangular prism made of 1-inch cubes -/
structure CubePrism where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a CubePrism -/
def volume (p : CubePrism) : ℕ := p.width * p.length * p.height

/-- Represents the prism of cubes not touching any tin foil -/
def innerPrism (outer : CubePrism) : CubePrism where
  width := outer.width - 2
  length := (outer.width - 2) / 2
  height := (outer.width - 2) / 2

theorem cubes_not_touching_foil (outer : CubePrism) 
  (h1 : outer.width = 10) 
  (h2 : innerPrism outer = { width := 8, length := 4, height := 4 }) : 
  volume (innerPrism outer) = 128 := by
  sorry

end NUMINAMATH_CALUDE_cubes_not_touching_foil_l2352_235262


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l2352_235275

/-- Given a set of observations with an incorrect mean due to a misrecorded value,
    calculate the corrected mean. -/
def corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean + (correct_value - incorrect_value) / (n : ℚ)

/-- Theorem stating that the corrected mean for the given problem is 45.45 -/
theorem corrected_mean_problem :
  corrected_mean 100 45 20 65 = 45.45 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_problem_l2352_235275


namespace NUMINAMATH_CALUDE_no_consecutive_sum_for_2_14_l2352_235212

theorem no_consecutive_sum_for_2_14 : ¬∃ (k n : ℕ), k > 0 ∧ n > 0 ∧ 2^14 = (k * (2*n + k + 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_sum_for_2_14_l2352_235212


namespace NUMINAMATH_CALUDE_exists_multiple_sum_of_digits_divides_l2352_235250

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For every positive integer n, there exists a multiple of n whose sum of digits divides it -/
theorem exists_multiple_sum_of_digits_divides (n : ℕ+) : 
  ∃ k : ℕ+, (sum_of_digits (k * n) ∣ (k * n)) := by sorry

end NUMINAMATH_CALUDE_exists_multiple_sum_of_digits_divides_l2352_235250


namespace NUMINAMATH_CALUDE_target_word_satisfies_conditions_target_word_is_unique_l2352_235243

/-- Represents a word with multiple meanings -/
structure MultiMeaningWord where
  word : String
  soundsLike : String
  usedInSports : Bool
  usedInPensions : Bool

/-- Represents the conditions for the word we're looking for -/
def wordConditions : MultiMeaningWord → Prop := fun w =>
  w.soundsLike = "festive dance event" ∧
  w.usedInSports = true ∧
  w.usedInPensions = true

/-- The word we're looking for -/
def targetWord : MultiMeaningWord := {
  word := "баллы",
  soundsLike := "festive dance event",
  usedInSports := true,
  usedInPensions := true
}

/-- Theorem stating that our target word satisfies all conditions -/
theorem target_word_satisfies_conditions : 
  wordConditions targetWord := by sorry

/-- Theorem stating that our target word is unique -/
theorem target_word_is_unique :
  ∀ w : MultiMeaningWord, wordConditions w → w = targetWord := by sorry

end NUMINAMATH_CALUDE_target_word_satisfies_conditions_target_word_is_unique_l2352_235243


namespace NUMINAMATH_CALUDE_token_count_after_removal_l2352_235295

/-- Represents a token on the board -/
inductive Token
| White
| Black
| Empty

/-- Represents the board state -/
def Board (n : ℕ) := Fin (2*n) → Fin (2*n) → Token

/-- Counts the number of tokens of a specific type on the board -/
def countTokens (b : Board n) (t : Token) : ℕ := sorry

/-- Performs the token removal process -/
def removeTokens (b : Board n) : Board n := sorry

theorem token_count_after_removal (n : ℕ) (initial_board : Board n) :
  let final_board := removeTokens initial_board
  (countTokens final_board Token.Black ≤ n^2) ∧ 
  (countTokens final_board Token.White ≤ n^2) := by
  sorry

end NUMINAMATH_CALUDE_token_count_after_removal_l2352_235295


namespace NUMINAMATH_CALUDE_log_equation_roots_l2352_235213

theorem log_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ > 0 ∧ x₁ + a > 0 ∧ x₁ + a ≠ 1 ∧
   x₂ > 0 ∧ x₂ + a > 0 ∧ x₂ + a ≠ 1 ∧
   Real.log (2 * x₁) / Real.log (x₁ + a) = 2 ∧
   Real.log (2 * x₂) / Real.log (x₂ + a) = 2) ↔
  (a > 0 ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_log_equation_roots_l2352_235213


namespace NUMINAMATH_CALUDE_mp3_song_count_l2352_235253

theorem mp3_song_count (initial_songs : ℕ) (deleted_songs : ℕ) (added_songs : ℕ) 
  (h1 : initial_songs = 15)
  (h2 : deleted_songs = 8)
  (h3 : added_songs = 50) :
  initial_songs - deleted_songs + added_songs = 57 := by
  sorry

end NUMINAMATH_CALUDE_mp3_song_count_l2352_235253


namespace NUMINAMATH_CALUDE_q_sum_zero_five_l2352_235223

/-- A monic polynomial of degree 5 -/
def MonicPolynomial5 (q : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, q x = x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + q 0

/-- The main theorem -/
theorem q_sum_zero_five
  (q : ℝ → ℝ)
  (monic : MonicPolynomial5 q)
  (h1 : q 1 = 24)
  (h2 : q 2 = 48)
  (h3 : q 3 = 72) :
  q 0 + q 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_q_sum_zero_five_l2352_235223


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2352_235235

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3}

-- Define set P
def P : Set ℕ := {1, 2}

-- Define set Q
def Q : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_intersection_equals_set : 
  (U \ (P ∩ Q)) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2352_235235


namespace NUMINAMATH_CALUDE_product_one_inequality_l2352_235230

theorem product_one_inequality (a b c d e : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_prod : a * b * c * d * e = 1) : 
  a^2 / b^2 + b^2 / c^2 + c^2 / d^2 + d^2 / e^2 + e^2 / a^2 ≥ a + b + c + d + e := by
sorry

end NUMINAMATH_CALUDE_product_one_inequality_l2352_235230


namespace NUMINAMATH_CALUDE_retail_price_calculation_l2352_235284

/-- The retail price of a machine given wholesale price, discount, and profit percentage -/
theorem retail_price_calculation (wholesale_price discount_percent profit_percent : ℝ) 
  (h_wholesale : wholesale_price = 90)
  (h_discount : discount_percent = 10)
  (h_profit : profit_percent = 20) :
  ∃ (retail_price : ℝ), 
    retail_price = 120 ∧ 
    (1 - discount_percent / 100) * retail_price = wholesale_price + (profit_percent / 100 * wholesale_price) := by
  sorry


end NUMINAMATH_CALUDE_retail_price_calculation_l2352_235284
