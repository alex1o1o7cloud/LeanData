import Mathlib

namespace triangle_inequality_and_equality_l134_13433

/-- Triangle ABC with side lengths a, b, c opposite to vertices A, B, C respectively,
    and h being the height from vertex C onto side AB -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_h : 0 < h
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem about the inequality and equality condition -/
theorem triangle_inequality_and_equality (t : Triangle) :
  t.a + t.b ≥ Real.sqrt (t.c^2 + 4*t.h^2) ∧
  (t.a + t.b = Real.sqrt (t.c^2 + 4*t.h^2) ↔ t.a = t.b ∧ t.a^2 + t.b^2 = t.c^2) :=
by sorry

end triangle_inequality_and_equality_l134_13433


namespace equation_solution_l134_13450

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 55 ∧ x = 16 := by
  sorry

end equation_solution_l134_13450


namespace chord_intersection_probability_l134_13403

/-- The probability that a chord intersects the inner circle when two points are chosen randomly
    on the outer circle of two concentric circles with radii 2 and 3 -/
theorem chord_intersection_probability (r₁ r₂ : ℝ) (h₁ : r₁ = 2) (h₂ : r₂ = 3) :
  let θ := 2 * Real.arctan (r₁ / Real.sqrt (r₂^2 - r₁^2))
  (θ / (2 * Real.pi)) = 0.2148 := by sorry

end chord_intersection_probability_l134_13403


namespace birdseed_mix_cost_l134_13412

theorem birdseed_mix_cost (millet_weight : ℝ) (millet_cost : ℝ) (sunflower_weight : ℝ) (mixture_cost : ℝ) :
  millet_weight = 100 →
  millet_cost = 0.60 →
  sunflower_weight = 25 →
  mixture_cost = 0.70 →
  let total_weight := millet_weight + sunflower_weight
  let total_cost := mixture_cost * total_weight
  let millet_total_cost := millet_weight * millet_cost
  let sunflower_total_cost := total_cost - millet_total_cost
  sunflower_total_cost / sunflower_weight = 1.10 := by
sorry

end birdseed_mix_cost_l134_13412


namespace no_snuggly_integers_l134_13464

/-- A two-digit positive integer is snuggly if it equals the sum of its nonzero tens digit and the cube of its units digit. -/
def is_snuggly (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n = (n / 10) + (n % 10)^3

/-- There are no snuggly two-digit positive integers. -/
theorem no_snuggly_integers : ¬∃ n : ℕ, is_snuggly n := by
  sorry

end no_snuggly_integers_l134_13464


namespace pencil_count_problem_l134_13436

/-- Given an initial number of pencils, a number of lost pencils, and a number of gained pencils,
    calculate the final number of pencils. -/
def finalPencilCount (initial lost gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Theorem stating that given the specific values in the problem,
    the final pencil count is 2060. -/
theorem pencil_count_problem :
  finalPencilCount 2015 5 50 = 2060 := by
  sorry

end pencil_count_problem_l134_13436


namespace quadratic_inequality_solution_set_l134_13416

theorem quadratic_inequality_solution_set 
  (a : ℝ) (ha : a < 0) :
  {x : ℝ | 42 * x^2 + a * x - a^2 < 0} = {x : ℝ | a / 7 < x ∧ x < -a / 6} :=
by sorry

end quadratic_inequality_solution_set_l134_13416


namespace stating_max_equations_theorem_l134_13428

/-- 
Represents the maximum number of equations without real roots 
that the first player can guarantee in a game with n equations.
-/
def max_equations_without_real_roots (n : ℕ) : ℕ :=
  if n % 2 = 0 then 0 else (n + 1) / 2

/-- 
Theorem stating the maximum number of equations without real roots 
that the first player can guarantee in the game.
-/
theorem max_equations_theorem (n : ℕ) :
  max_equations_without_real_roots n = 
    if n % 2 = 0 then 0 else (n + 1) / 2 := by
  sorry

end stating_max_equations_theorem_l134_13428


namespace trig_expression_equals_one_l134_13437

theorem trig_expression_equals_one : 
  (Real.sin (20 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 1 := by
  sorry

end trig_expression_equals_one_l134_13437


namespace quadratic_inequality_l134_13472

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 4

-- State the theorem
theorem quadratic_inequality (a x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 3) 
  (hx : x₁ < x₂) 
  (hsum : x₁ + x₂ = 0) : 
  f a x₁ < f a x₂ := by
  sorry


end quadratic_inequality_l134_13472


namespace complement_intersection_theorem_l134_13458

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set Nat := {1, 3, 5, 7}
def N : Set Nat := {5, 6, 7}

theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {2, 4, 8} := by
  sorry

end complement_intersection_theorem_l134_13458


namespace mike_bought_33_books_l134_13463

/-- The number of books Mike bought at a yard sale -/
def books_bought (initial_books final_books books_given_away : ℕ) : ℕ :=
  final_books - (initial_books - books_given_away)

/-- Theorem stating that Mike bought 33 books at the yard sale -/
theorem mike_bought_33_books :
  books_bought 35 56 12 = 33 := by
  sorry

end mike_bought_33_books_l134_13463


namespace invalid_external_diagonals_l134_13411

/-- Checks if three numbers can be the lengths of external diagonals of a right regular prism -/
def are_valid_external_diagonals (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ a^2 + c^2 > b^2

/-- Theorem stating that {5, 7, 9} cannot be the external diagonals of a right regular prism -/
theorem invalid_external_diagonals :
  ¬(are_valid_external_diagonals 5 7 9) := by
  sorry

end invalid_external_diagonals_l134_13411


namespace cube_score_is_40_l134_13469

/-- Represents the score for a unit cube based on the number of painted faces. -/
def score (painted_faces : Nat) : Int :=
  match painted_faces with
  | 3 => 3
  | 2 => 2
  | 1 => 1
  | 0 => -7
  | _ => 0  -- This case should never occur in our problem

/-- The size of one side of the cube. -/
def cube_size : Nat := 4

/-- The total number of unit cubes in the large cube. -/
def total_cubes : Nat := cube_size ^ 3

/-- The number of corner cubes (with 3 painted faces). -/
def corner_cubes : Nat := 8

/-- The number of edge cubes (with 2 painted faces), excluding corners. -/
def edge_cubes : Nat := 12 * (cube_size - 2)

/-- The number of face cubes (with 1 painted face), excluding edges and corners. -/
def face_cubes : Nat := 6 * (cube_size - 2) ^ 2

/-- The number of internal cubes (with 0 painted faces). -/
def internal_cubes : Nat := (cube_size - 2) ^ 3

theorem cube_score_is_40 :
  (corner_cubes * score 3 +
   edge_cubes * score 2 +
   face_cubes * score 1 +
   internal_cubes * score 0) = 40 ∧
  corner_cubes + edge_cubes + face_cubes + internal_cubes = total_cubes :=
sorry

end cube_score_is_40_l134_13469


namespace modified_circle_radius_l134_13484

/-- Given a circle with radius r, prove that if its modified area and circumference
    sum to 180π, then r satisfies the equation r² + 2r - 90 = 0 -/
theorem modified_circle_radius (r : ℝ) : 
  (2 * Real.pi * r^2) + (4 * Real.pi * r) = 180 * Real.pi → 
  r^2 + 2*r - 90 = 0 := by
  sorry


end modified_circle_radius_l134_13484


namespace max_value_theorem_l134_13414

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hsum : a + b + c = 3) :
  (a * b / (a + b)) + (b * c / (b + c)) + (c * a / (c + a)) ≤ 3 / 2 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 3 ∧
    (a' * b' / (a' + b')) + (b' * c' / (b' + c')) + (c' * a' / (c' + a')) = 3 / 2 :=
by sorry

end max_value_theorem_l134_13414


namespace total_distance_in_feet_l134_13425

/-- Conversion factor from miles to feet -/
def miles_to_feet : ℝ := 5280

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- Conversion factor from kilometers to feet -/
def km_to_feet : ℝ := 3280.84

/-- Conversion factor from meters to feet -/
def meters_to_feet : ℝ := 3.28084

/-- Distance walked by Lionel in miles -/
def lionel_distance : ℝ := 4

/-- Distance walked by Esther in yards -/
def esther_distance : ℝ := 975

/-- Distance walked by Niklaus in feet -/
def niklaus_distance : ℝ := 1287

/-- Distance biked by Isabella in kilometers -/
def isabella_distance : ℝ := 18

/-- Distance swam by Sebastian in meters -/
def sebastian_distance : ℝ := 2400

/-- Theorem stating the total combined distance traveled by the friends in feet -/
theorem total_distance_in_feet :
  lionel_distance * miles_to_feet +
  esther_distance * yards_to_feet +
  niklaus_distance +
  isabella_distance * km_to_feet +
  sebastian_distance * meters_to_feet = 89261.136 := by
  sorry

end total_distance_in_feet_l134_13425


namespace triangle_formation_l134_13424

/-- Two lines in the Cartesian coordinate system -/
structure CartesianLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ
  k : ℝ

/-- Condition for two lines to form a triangle with the x-axis -/
def formsTriangle (lines : CartesianLines) : Prop :=
  lines.k ≠ 0 ∧ lines.k ≠ -1/2

/-- Theorem: The given lines form a triangle with the x-axis if and only if k ≠ -1/2 -/
theorem triangle_formation (lines : CartesianLines) 
  (h1 : lines.line1 = fun x ↦ -0.5 * x - 2)
  (h2 : lines.line2 = fun x ↦ lines.k * x + 3) :
  formsTriangle lines ↔ lines.k ≠ -1/2 :=
sorry

end triangle_formation_l134_13424


namespace min_of_quadratic_l134_13410

/-- The quadratic function f(x) = x^2 - 2px + 4q -/
def f (p q x : ℝ) : ℝ := x^2 - 2*p*x + 4*q

/-- Theorem stating that the minimum of f occurs at x = p -/
theorem min_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f p q x_min ≤ f p q x ∧ x_min = p :=
sorry

end min_of_quadratic_l134_13410


namespace paint_mixer_production_time_l134_13443

/-- A paint mixer's production rate and time to complete a job -/
theorem paint_mixer_production_time 
  (days_for_some_drums : ℕ) 
  (total_drums : ℕ) 
  (total_days : ℕ) 
  (h1 : days_for_some_drums = 3)
  (h2 : total_drums = 360)
  (h3 : total_days = 60) :
  total_days = total_drums / (total_drums / total_days) :=
by sorry

end paint_mixer_production_time_l134_13443


namespace quadratic_inequality_solution_set_l134_13493

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 12 ≥ 0} = {x : ℝ | x ≤ -4 ∨ x ≥ 3} := by sorry

end quadratic_inequality_solution_set_l134_13493


namespace relationship_abc_l134_13491

open Real

theorem relationship_abc (a b c : ℝ) (ha : a = 2^(log 2)) (hb : b = 2 + 2*log 2) (hc : c = (log 2)^2) :
  c < a ∧ a < b := by
  sorry

end relationship_abc_l134_13491


namespace equation_equality_l134_13445

theorem equation_equality (x : ℝ) : -x^3 + 7*x^2 + 2*x - 8 = -(x - 2)*(x - 4)*(x - 1) := by
  sorry

end equation_equality_l134_13445


namespace orchard_solution_l134_13467

/-- Represents the number of trees in an orchard -/
structure Orchard where
  peach : ℕ
  apple : ℕ

/-- Conditions for the orchard problem -/
def OrchardConditions (o : Orchard) : Prop :=
  (o.apple = o.peach + 1700) ∧ (o.apple = 3 * o.peach + 200)

/-- Theorem stating the solution to the orchard problem -/
theorem orchard_solution : 
  ∃ o : Orchard, OrchardConditions o ∧ o.peach = 750 ∧ o.apple = 2450 := by
  sorry

end orchard_solution_l134_13467


namespace problem_solution_l134_13449

theorem problem_solution (a b c : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : c > 0) :
  (a * c < b * c) ∧ (a + b + c < b + c) ∧ (c / a > 1) := by
  sorry

end problem_solution_l134_13449


namespace amoeba_count_14_days_l134_13468

/-- Calculates the number of amoebas after a given number of days -/
def amoeba_count (days : ℕ) : ℕ :=
  if days ≤ 2 then 2^(days - 1)
  else 5 * 2^(days - 3)

/-- The number of amoebas after 14 days is 10240 -/
theorem amoeba_count_14_days : amoeba_count 14 = 10240 := by
  sorry

end amoeba_count_14_days_l134_13468


namespace stuffed_toy_dogs_boxes_l134_13461

theorem stuffed_toy_dogs_boxes (dogs_per_box : ℕ) (total_dogs : ℕ) (h1 : dogs_per_box = 4) (h2 : total_dogs = 28) :
  total_dogs / dogs_per_box = 7 := by
  sorry

end stuffed_toy_dogs_boxes_l134_13461


namespace sector_max_area_l134_13471

/-- Given a sector with circumference 8, its area is at most 4 -/
theorem sector_max_area :
  ∀ (r l : ℝ), r > 0 → l > 0 → 2 * r + l = 8 →
  (1 / 2) * l * r ≤ 4 := by
  sorry

end sector_max_area_l134_13471


namespace rectangle_ribbon_length_l134_13413

/-- The length of ribbon required to form a rectangle -/
def ribbon_length (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: The length of ribbon required to form a rectangle with length 20 feet and width 15 feet is 70 feet -/
theorem rectangle_ribbon_length : 
  ribbon_length 20 15 = 70 := by
  sorry

end rectangle_ribbon_length_l134_13413


namespace unique_number_satisfying_conditions_l134_13415

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem unique_number_satisfying_conditions : 
  ∃! n : ℕ, is_two_digit n ∧ 
    ((ends_in n 6 ∨ n % 7 = 0) ∧ ¬(ends_in n 6 ∧ n % 7 = 0)) ∧
    ((n > 26 ∨ ends_in n 8) ∧ ¬(n > 26 ∧ ends_in n 8)) ∧
    ((n % 13 = 0 ∨ n < 27) ∧ ¬(n % 13 = 0 ∧ n < 27)) ∧
    n = 91 := by
  sorry

#check unique_number_satisfying_conditions

end unique_number_satisfying_conditions_l134_13415


namespace students_just_passed_l134_13434

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ)
  (h_total : total = 300)
  (h_first : first_div_percent = 25 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent < 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 63 := by
  sorry

end students_just_passed_l134_13434


namespace complex_magnitude_product_l134_13481

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 5 - 5 * Complex.I) * (2 * Real.sqrt 7 + 4 * Complex.I)) = 20 * Real.sqrt 77 := by
  sorry

end complex_magnitude_product_l134_13481


namespace sqrt_meaningful_range_l134_13487

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 3 * x - 5) ↔ x ≥ 5 / 3 := by
  sorry

end sqrt_meaningful_range_l134_13487


namespace quadratic_root_zero_l134_13473

/-- Given a quadratic equation (a+3)x^2 - 4x + a^2 - 9 = 0 with 0 as a root and a + 3 ≠ 0, prove that a = 3 -/
theorem quadratic_root_zero (a : ℝ) : 
  ((a + 3) * 0^2 - 4 * 0 + a^2 - 9 = 0) → 
  (a + 3 ≠ 0) → 
  (a = 3) := by
sorry

end quadratic_root_zero_l134_13473


namespace min_value_x_plus_2y_l134_13417

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 2*x*y = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ z, z = x + 2*y → z ≥ m :=
sorry

end min_value_x_plus_2y_l134_13417


namespace radical_conjugate_sum_product_l134_13466

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = 0 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 16 → 
  a + b = -16 := by
sorry

end radical_conjugate_sum_product_l134_13466


namespace emmalyn_earnings_l134_13480

/-- The rate Emmalyn charges per meter for painting fences, in dollars -/
def rate : ℚ := 0.20

/-- The number of fences in the neighborhood -/
def num_fences : ℕ := 50

/-- The length of each fence in meters -/
def fence_length : ℕ := 500

/-- The total amount Emmalyn earned in dollars -/
def total_amount : ℚ := rate * (num_fences * fence_length)

theorem emmalyn_earnings :
  total_amount = 5000 := by sorry

end emmalyn_earnings_l134_13480


namespace fixed_point_of_exponential_function_l134_13478

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x y : ℝ, x = -2 ∧ y = 6 ∧ a^(x + 2) + 5 = y :=
sorry

end fixed_point_of_exponential_function_l134_13478


namespace inverse_sum_mod_31_l134_13495

theorem inverse_sum_mod_31 :
  ∃ (a b : ℤ), a ≡ 25 [ZMOD 31] ∧ b ≡ 5 [ZMOD 31] ∧ (a + b) ≡ 30 [ZMOD 31] := by
  sorry

end inverse_sum_mod_31_l134_13495


namespace derivative_of_y_l134_13400

noncomputable def y (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)

theorem derivative_of_y (x : ℝ) :
  deriv y x = 2 * Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) := by sorry

end derivative_of_y_l134_13400


namespace ball_box_theorem_l134_13447

/-- Represents the state of boxes after a number of steps -/
def BoxState := List Nat

/-- Converts a natural number to its septenary (base 7) representation -/
def toSeptenary (n : Nat) : List Nat :=
  sorry

/-- Simulates the ball-placing process for a given number of steps -/
def simulateSteps (steps : Nat) : BoxState :=
  sorry

/-- Counts the number of non-zero elements in a list -/
def countNonZero (l : List Nat) : Nat :=
  sorry

/-- Sums all elements in a list -/
def sumList (l : List Nat) : Nat :=
  sorry

theorem ball_box_theorem (steps : Nat := 3456) :
  let septenaryRep := toSeptenary steps
  let finalState := simulateSteps steps
  countNonZero finalState = countNonZero septenaryRep ∧
  sumList finalState = sumList septenaryRep :=
by sorry

end ball_box_theorem_l134_13447


namespace marble_fraction_after_tripling_l134_13435

theorem marble_fraction_after_tripling (total : ℚ) (h : total > 0) :
  let blue := (2 / 3) * total
  let red := total - blue
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = 3 / 5 := by
sorry

end marble_fraction_after_tripling_l134_13435


namespace max_triangle_area_l134_13496

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The line l passing through F₂(1,0) -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 1

/-- The area of triangle F₁AB -/
def triangle_area (y₁ y₂ : ℝ) : ℝ :=
  |y₁ - y₂|

/-- The theorem stating the maximum area of triangle F₁AB -/
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = 3 ∧
  ∀ (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    trajectory x₁ y₁ →
    trajectory x₂ y₂ →
    line_l m x₁ y₁ →
    line_l m x₂ y₂ →
    x₁ ≠ x₂ →
    triangle_area y₁ y₂ ≤ max_area :=
by sorry

end max_triangle_area_l134_13496


namespace division_remainder_proof_l134_13456

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 158 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
  sorry

end division_remainder_proof_l134_13456


namespace identical_car_in_kindergarten_l134_13444

-- Define the properties of a car
structure Car where
  color : String
  size : String
  hasTrailer : Bool

-- Define the boys and their car collections
def Misha : List Car := [
  { color := "green", size := "small", hasTrailer := false },
  { color := "unknown", size := "small", hasTrailer := false },
  { color := "unknown", size := "unknown", hasTrailer := true }
]

def Vitya : List Car := [
  { color := "unknown", size := "unknown", hasTrailer := false },
  { color := "green", size := "small", hasTrailer := true }
]

def Kolya : List Car := [
  { color := "unknown", size := "big", hasTrailer := false },
  { color := "blue", size := "small", hasTrailer := true }
]

-- Define the theorem
theorem identical_car_in_kindergarten :
  ∃ (c : Car),
    c ∈ Misha ∧ c ∈ Vitya ∧ c ∈ Kolya ∧
    c.color = "green" ∧ c.size = "big" ∧ c.hasTrailer = false :=
by
  sorry

end identical_car_in_kindergarten_l134_13444


namespace basketball_lineup_count_l134_13475

/-- Represents the number of possible line-ups for a basketball team -/
def number_of_lineups (total_players : ℕ) (centers : ℕ) (right_forwards : ℕ) (left_forwards : ℕ) (right_guards : ℕ) (flexible_guards : ℕ) : ℕ :=
  let guard_combinations := flexible_guards * flexible_guards
  guard_combinations * centers * right_forwards * left_forwards

/-- Theorem stating the number of possible line-ups for the given team composition -/
theorem basketball_lineup_count :
  number_of_lineups 10 2 2 2 1 3 = 72 := by
  sorry

end basketball_lineup_count_l134_13475


namespace allan_final_score_l134_13408

/-- Calculates the final score on a test with the given parameters. -/
def final_score (total_questions : ℕ) (correct_answers : ℕ) (points_per_correct : ℚ) (points_per_incorrect : ℚ) : ℚ :=
  let incorrect_answers := total_questions - correct_answers
  (correct_answers : ℚ) * points_per_correct - (incorrect_answers : ℚ) * points_per_incorrect

/-- Theorem stating that Allan's final score is 100 given the test conditions. -/
theorem allan_final_score :
  let total_questions : ℕ := 120
  let correct_answers : ℕ := 104
  let points_per_correct : ℚ := 1
  let points_per_incorrect : ℚ := 1/4
  final_score total_questions correct_answers points_per_correct points_per_incorrect = 100 := by
  sorry

end allan_final_score_l134_13408


namespace phi_difference_squared_l134_13488

theorem phi_difference_squared : ∀ Φ φ : ℝ, 
  Φ ≠ φ → 
  Φ^2 - 2*Φ - 1 = 0 → 
  φ^2 - 2*φ - 1 = 0 → 
  (Φ - φ)^2 = 8 := by
  sorry

end phi_difference_squared_l134_13488


namespace max_value_of_function_l134_13477

theorem max_value_of_function :
  let f : ℝ → ℝ := λ x => (Real.sqrt 3 / 2) * Real.sin (x + Real.pi / 2) + Real.cos (Real.pi / 6 - x)
  ∃ (M : ℝ), M = Real.sqrt 13 / 2 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end max_value_of_function_l134_13477


namespace right_triangle_inequality_l134_13465

theorem right_triangle_inequality (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a^2 = b^2 + c^2 → 
  a ≥ b ∧ a ≥ c → 
  (a^x > b^x + c^x ↔ x > 2) :=
sorry

end right_triangle_inequality_l134_13465


namespace closest_integer_to_ratio_l134_13406

/-- Given two positive real numbers a and b where a > b, and their arithmetic mean
    is equal to twice their geometric mean, prove that the integer closest to a/b is 14. -/
theorem closest_integer_to_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : (a + b) / 2 = 2 * Real.sqrt (a * b)) : 
    ∃ (n : ℤ), n = 14 ∧ ∀ (m : ℤ), |a / b - ↑n| ≤ |a / b - ↑m| :=
by sorry

end closest_integer_to_ratio_l134_13406


namespace dog_food_per_dog_l134_13470

/-- The amount of dog food two dogs eat together per day -/
def total_food : ℝ := 0.25

/-- The number of dogs -/
def num_dogs : ℕ := 2

theorem dog_food_per_dog :
  ∀ (food_per_dog : ℝ),
  (food_per_dog * num_dogs = total_food) →
  (food_per_dog = 0.125) := by
sorry

end dog_food_per_dog_l134_13470


namespace polar_circle_equation_l134_13498

/-- A circle in a polar coordinate system with radius 1 and center at (1, 0) -/
structure PolarCircle where
  center : ℝ × ℝ := (1, 0)
  radius : ℝ := 1

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Predicate to check if a point is on the circle -/
def IsOnCircle (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.ρ = 2 * c.radius * Real.cos p.θ

theorem polar_circle_equation (c : PolarCircle) (p : PolarPoint) 
  (h : IsOnCircle c p) : p.ρ = 2 * Real.cos p.θ := by
  sorry

end polar_circle_equation_l134_13498


namespace grandfather_age_ratio_l134_13474

/-- Given the current ages of Xiao Hong and her grandfather, 
    prove the ratio of their ages last year -/
theorem grandfather_age_ratio (xiao_hong_age grandfather_age : ℕ) 
  (h1 : xiao_hong_age = 8) 
  (h2 : grandfather_age = 64) : 
  (grandfather_age - 1) / (xiao_hong_age - 1) = 9 := by
  sorry

end grandfather_age_ratio_l134_13474


namespace reflected_ray_equation_l134_13401

/-- Given that:
  - Point A is at (-1/2, 0)
  - Point B is at (0, 1)
  - A' is the reflection of A across the y-axis
Prove that the line passing through A' and B has the equation 2x + y - 1 = 0 -/
theorem reflected_ray_equation (A : ℝ × ℝ) (B : ℝ × ℝ) (A' : ℝ × ℝ) :
  A = (-1/2, 0) →
  B = (0, 1) →
  A'.1 = -A.1 →  -- A' is reflection of A across y-axis
  A'.2 = A.2 →   -- A' is reflection of A across y-axis
  ∀ (x y : ℝ), (x = A'.1 ∧ y = A'.2) ∨ (x = B.1 ∧ y = B.2) →
    2 * x + y - 1 = 0 :=
by sorry

end reflected_ray_equation_l134_13401


namespace parabola_point_distance_l134_13402

theorem parabola_point_distance (m n : ℝ) : 
  n^2 = 4*m →                             -- P(m,n) is on the parabola y^2 = 4x
  (m + 1)^2 = (m - 5)^2 + n^2 →           -- Distance from P to x=-1 equals distance from P to A(5,0)
  m = 3 := by sorry

end parabola_point_distance_l134_13402


namespace perimeter_of_square_b_l134_13462

/-- Given a square A with perimeter 40 cm and a square B with area equal to one-third the area of square A, 
    the perimeter of square B is (40√3)/3 cm. -/
theorem perimeter_of_square_b (square_a square_b : Real → Real → Prop) : 
  (∃ side_a, square_a side_a side_a ∧ 4 * side_a = 40) →
  (∃ side_b, square_b side_b side_b ∧ side_b^2 = (side_a^2) / 3) →
  (∃ perimeter_b, perimeter_b = 40 * Real.sqrt 3 / 3) :=
by sorry

end perimeter_of_square_b_l134_13462


namespace existence_of_product_one_derivatives_l134_13452

theorem existence_of_product_one_derivatives 
  (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_diff : DifferentiableOn ℝ f (Set.Ioo 0 1))
  (h_range : Set.range f ⊆ Set.Icc 0 1)
  (h_zero : f 0 = 0)
  (h_one : f 1 = 1) :
  ∃ a b : ℝ, a ∈ Set.Ioo 0 1 ∧ b ∈ Set.Ioo 0 1 ∧ a ≠ b ∧ 
    deriv f a * deriv f b = 1 :=
sorry

end existence_of_product_one_derivatives_l134_13452


namespace driver_net_pay_rate_driver_net_pay_is_25_l134_13459

/-- Calculates the net rate of pay for a driver given specific conditions -/
theorem driver_net_pay_rate (travel_time : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (compensation_rate : ℝ) (fuel_cost : ℝ) : ℝ :=
  let total_distance := travel_time * speed
  let fuel_used := total_distance / fuel_efficiency
  let earnings := compensation_rate * total_distance
  let fuel_expense := fuel_cost * fuel_used
  let net_earnings := earnings - fuel_expense
  let net_rate := net_earnings / travel_time
  net_rate

/-- Proves that the driver's net rate of pay is $25 per hour under the given conditions -/
theorem driver_net_pay_is_25 : 
  driver_net_pay_rate 3 50 25 0.60 2.50 = 25 := by
  sorry

end driver_net_pay_rate_driver_net_pay_is_25_l134_13459


namespace minnows_per_prize_bowl_l134_13489

theorem minnows_per_prize_bowl (total_minnows : ℕ) (total_players : ℕ) (winner_percentage : ℚ) (leftover_minnows : ℕ) :
  total_minnows = 600 →
  total_players = 800 →
  winner_percentage = 15 / 100 →
  leftover_minnows = 240 →
  (total_minnows - leftover_minnows) / (total_players * winner_percentage) = 3 :=
by sorry

end minnows_per_prize_bowl_l134_13489


namespace consecutive_even_numbers_largest_l134_13448

theorem consecutive_even_numbers_largest (n : ℕ) : 
  (∀ k : ℕ, k < 7 → ∃ m : ℕ, n + 2*k = 2*m) →  -- 7 consecutive even numbers
  (n + 12 = 3 * n) →                           -- largest is 3 times the smallest
  (n + 12 = 18) :=                             -- largest number is 18
by sorry

end consecutive_even_numbers_largest_l134_13448


namespace magnitude_of_complex_fraction_l134_13492

theorem magnitude_of_complex_fraction (z : ℂ) : z = (2 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end magnitude_of_complex_fraction_l134_13492


namespace area_inside_circle_outside_square_l134_13490

/-- The area inside a circle of radius √3/3 but outside a square of side length 1, 
    when they share the same center. -/
theorem area_inside_circle_outside_square : 
  let square_side : ℝ := 1
  let circle_radius : ℝ := Real.sqrt 3 / 3
  let circle_area : ℝ := π * circle_radius^2
  let square_area : ℝ := square_side^2
  let area_difference : ℝ := circle_area - square_area
  area_difference = 2 * π / 9 - Real.sqrt 3 / 3 := by
  sorry

end area_inside_circle_outside_square_l134_13490


namespace amusement_park_expenses_l134_13482

theorem amusement_park_expenses (total brought food tshirt left ticket : ℕ) : 
  brought = 75 ∧ 
  food = 13 ∧ 
  tshirt = 23 ∧ 
  left = 9 ∧ 
  brought = food + tshirt + left + ticket → 
  ticket = 30 := by
  sorry

end amusement_park_expenses_l134_13482


namespace megan_folders_l134_13422

/-- Calculates the number of full folders given the initial number of files, 
    number of deleted files, and number of files per folder. -/
def fullFolders (initialFiles : ℕ) (deletedFiles : ℕ) (filesPerFolder : ℕ) : ℕ :=
  ((initialFiles - deletedFiles) / filesPerFolder : ℕ)

/-- Proves that Megan ends up with 15 full folders given the initial conditions. -/
theorem megan_folders : fullFolders 256 67 12 = 15 := by
  sorry

#eval fullFolders 256 67 12

end megan_folders_l134_13422


namespace tetrahedron_division_possible_l134_13483

theorem tetrahedron_division_possible (edge_length : ℝ) (target_length : ℝ) : 
  edge_length > 0 → target_length > 0 → target_length < edge_length →
  ∃ n : ℕ, (1/2 : ℝ)^n * edge_length < target_length := by
  sorry

#check tetrahedron_division_possible 1 (1/100)

end tetrahedron_division_possible_l134_13483


namespace least_multiple_36_with_digit_sum_multiple_9_l134_13404

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem least_multiple_36_with_digit_sum_multiple_9 :
  ∀ k : ℕ, k > 0 → 36 * k ≠ 36 →
    digit_sum (36 * k) % 9 = 0 → digit_sum 36 % 9 = 0 ∧ 36 < 36 * k :=
by sorry

end least_multiple_36_with_digit_sum_multiple_9_l134_13404


namespace lcm_4_8_9_10_l134_13442

theorem lcm_4_8_9_10 : Nat.lcm 4 (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := by
  sorry

end lcm_4_8_9_10_l134_13442


namespace arithmetic_sequence_unique_n_l134_13441

/-- An arithmetic sequence with n terms, where a₁ is the first term and d is the common difference. -/
structure ArithmeticSequence where
  n : ℕ
  a₁ : ℚ
  d : ℚ

/-- The sum of the first k terms of an arithmetic sequence. -/
def sum_first_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k / 2 * (2 * seq.a₁ + (k - 1) * seq.d)

/-- The sum of the last k terms of an arithmetic sequence. -/
def sum_last_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k / 2 * (2 * (seq.a₁ + (seq.n - 1) * seq.d) - (k - 1) * seq.d)

/-- The sum of all terms in an arithmetic sequence. -/
def sum_all (seq : ArithmeticSequence) : ℚ :=
  seq.n / 2 * (2 * seq.a₁ + (seq.n - 1) * seq.d)

theorem arithmetic_sequence_unique_n :
  ∀ seq : ArithmeticSequence,
    sum_first_k seq 3 = 34 →
    sum_last_k seq 3 = 146 →
    sum_all seq = 390 →
    seq.n = 11 := by
  sorry

end arithmetic_sequence_unique_n_l134_13441


namespace fraction_to_decimal_decimal_representation_l134_13485

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = (94 : ℚ) / 10000 := by
  sorry

theorem decimal_representation : (94 : ℚ) / 10000 = 0.0094 := by
  sorry

end fraction_to_decimal_decimal_representation_l134_13485


namespace cos_sin_225_degrees_l134_13438

theorem cos_sin_225_degrees :
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 ∧
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_sin_225_degrees_l134_13438


namespace calories_per_bar_is_48_l134_13432

-- Define the total number of calories
def total_calories : ℕ := 2016

-- Define the number of candy bars
def num_candy_bars : ℕ := 42

-- Define the function to calculate calories per candy bar
def calories_per_bar : ℚ := total_calories / num_candy_bars

-- Theorem to prove
theorem calories_per_bar_is_48 : calories_per_bar = 48 := by
  sorry

end calories_per_bar_is_48_l134_13432


namespace pet_shop_kittens_l134_13405

/-- Represents the pet shop inventory and pricing --/
structure PetShop where
  num_puppies : ℕ
  puppy_price : ℕ
  kitten_price : ℕ
  total_value : ℕ

/-- Calculates the number of kittens in the pet shop --/
def num_kittens (shop : PetShop) : ℕ :=
  (shop.total_value - shop.num_puppies * shop.puppy_price) / shop.kitten_price

/-- Theorem stating that the number of kittens in the given pet shop is 4 --/
theorem pet_shop_kittens :
  let shop : PetShop := {
    num_puppies := 2,
    puppy_price := 20,
    kitten_price := 15,
    total_value := 100
  }
  num_kittens shop = 4 := by
  sorry

end pet_shop_kittens_l134_13405


namespace min_value_x_plus_y_l134_13426

theorem min_value_x_plus_y (x y : ℝ) (h1 : 4/x + 9/y = 1) (h2 : x > 0) (h3 : y > 0) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 4/a + 9/b = 1 → x + y ≤ a + b :=
by sorry

end min_value_x_plus_y_l134_13426


namespace ordering_of_expressions_l134_13453

theorem ordering_of_expressions : e^(0.11 : ℝ) > (1.1 : ℝ)^(1.1 : ℝ) ∧ (1.1 : ℝ)^(1.1 : ℝ) > 1.11 := by
  sorry

end ordering_of_expressions_l134_13453


namespace cube_root_problem_l134_13440

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end cube_root_problem_l134_13440


namespace diving_class_capacity_l134_13420

/-- The number of people that can be accommodated in each diving class -/
def people_per_class : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The number of classes per weekday -/
def classes_per_weekday : ℕ := 2

/-- The number of classes per weekend day -/
def classes_per_weekend_day : ℕ := 4

/-- The number of weeks -/
def weeks : ℕ := 3

/-- The total number of people that can take classes in 3 weeks -/
def total_people : ℕ := 270

/-- Theorem stating that the number of people per class is 5 -/
theorem diving_class_capacity :
  people_per_class = 
    total_people / (weeks * (weekdays * classes_per_weekday + weekend_days * classes_per_weekend_day)) :=
by sorry

end diving_class_capacity_l134_13420


namespace quadratic_roots_relation_l134_13430

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ p q : ℝ, 
    (3 * p^2 - 5 * p - 7 = 0) ∧ 
    (3 * q^2 - 5 * q - 7 = 0) ∧ 
    ((p + 2)^2 + b * (p + 2) + c = 0) ∧
    ((q + 2)^2 + b * (q + 2) + c = 0)) →
  c = 5 := by
sorry

end quadratic_roots_relation_l134_13430


namespace largest_inscribed_triangle_area_l134_13418

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 10) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_height := r
  let triangle_area := (1/2) * diameter * max_height
  triangle_area = 100 := by sorry

end largest_inscribed_triangle_area_l134_13418


namespace sum_of_squares_of_roots_l134_13407

theorem sum_of_squares_of_roots : ∃ (a b c : ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -3 ∧ x ≠ -6 →
    (1 / x + 2 / (x + 3) + 3 / (x + 6) = 1) ↔ (x = a ∨ x = b ∨ x = c)) ∧
  a^2 + b^2 + c^2 = 33 := by
  sorry

end sum_of_squares_of_roots_l134_13407


namespace power_multiplication_subtraction_l134_13457

theorem power_multiplication_subtraction (a : ℝ) : 4 * a * a^3 - a^4 = 3 * a^4 := by
  sorry

end power_multiplication_subtraction_l134_13457


namespace football_game_attendance_l134_13494

theorem football_game_attendance (saturday_attendance : ℕ) 
  (expected_total : ℕ) : saturday_attendance = 80 →
  expected_total = 350 →
  (saturday_attendance + 
   (saturday_attendance - 20) + 
   (saturday_attendance - 20 + 50) + 
   (saturday_attendance + (saturday_attendance - 20))) - 
  expected_total = 40 := by
  sorry

end football_game_attendance_l134_13494


namespace license_plate_increase_l134_13421

theorem license_plate_increase : 
  let old_format := 26^2 * 10^3
  let new_format := 26^4 * 10^4
  (new_format / old_format : ℚ) = 2600 := by
  sorry

end license_plate_increase_l134_13421


namespace parallel_lines_a_value_l134_13460

/-- Given two lines l₁ and l₂, prove that a = -1 -/
theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 2*x + (a-1)*y + a = 0 ↔ a*x + y + 2 = 0) → -- l₁ and l₂ are parallel
  (2*2 ≠ a^2) →                                         -- Additional condition
  a = -1 :=
by sorry

end parallel_lines_a_value_l134_13460


namespace bisection_method_next_interval_l134_13479

def f (x : ℝ) := x^3 - 2*x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x₀ := (a + b) / 2
  f a * f x₀ < 0 → ∃ x ∈ Set.Icc a x₀, f x = 0 :=
by sorry

end bisection_method_next_interval_l134_13479


namespace smallest_valid_staircase_sum_of_digits_90_l134_13431

def is_valid_staircase (n : ℕ) : Prop :=
  ⌈(n : ℚ) / 2⌉ - ⌈(n : ℚ) / 3⌉ = 15

theorem smallest_valid_staircase :
  ∀ m : ℕ, m < 90 → ¬(is_valid_staircase m) ∧ is_valid_staircase 90 :=
by sorry

theorem sum_of_digits_90 : (9 : ℕ) = (9 : ℕ) + (0 : ℕ) :=
by sorry

end smallest_valid_staircase_sum_of_digits_90_l134_13431


namespace quadratic_equation_real_roots_l134_13476

theorem quadratic_equation_real_roots (k : ℕ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) → k = 1 := by
  sorry

end quadratic_equation_real_roots_l134_13476


namespace ellipse_condition_l134_13454

def is_ellipse_equation (m : ℝ) : Prop :=
  (m - 2 > 0) ∧ (6 - m > 0) ∧ (m - 2 ≠ 6 - m)

theorem ellipse_condition (m : ℝ) :
  (is_ellipse_equation m → m ∈ Set.Ioo 2 6) ∧
  (∃ m ∈ Set.Ioo 2 6, ¬is_ellipse_equation m) :=
sorry

end ellipse_condition_l134_13454


namespace total_hangers_count_l134_13439

def pink_hangers : ℕ := 7
def green_hangers : ℕ := 4
def blue_hangers : ℕ := green_hangers - 1
def yellow_hangers : ℕ := blue_hangers - 1
def orange_hangers : ℕ := 2 * pink_hangers
def purple_hangers : ℕ := yellow_hangers + 3
def red_hangers : ℕ := purple_hangers / 2

theorem total_hangers_count :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers +
  orange_hangers + purple_hangers + red_hangers = 37 := by
  sorry

end total_hangers_count_l134_13439


namespace male_cousins_count_l134_13455

/-- Represents the Martin family structure -/
structure MartinFamily where
  michael_sisters : ℕ
  michael_brothers : ℕ
  total_cousins : ℕ

/-- The number of male cousins counted by each female cousin in the Martin family -/
def male_cousins_per_female (family : MartinFamily) : ℕ :=
  family.michael_brothers + 1

/-- Theorem stating the number of male cousins counted by each female cousin -/
theorem male_cousins_count (family : MartinFamily) 
  (h1 : family.michael_sisters = 4)
  (h2 : family.michael_brothers = 6)
  (h3 : family.total_cousins = family.michael_sisters + family.michael_brothers + 2) 
  (h4 : ∃ n : ℕ, 2 * n = family.total_cousins) :
  male_cousins_per_female family = 8 := by
  sorry

#eval male_cousins_per_female { michael_sisters := 4, michael_brothers := 6, total_cousins := 14 }

end male_cousins_count_l134_13455


namespace simplify_and_rationalize_l134_13419

theorem simplify_and_rationalize (x : ℝ) (h : x = 1 / (1 + 1 / (Real.sqrt 2 + 2))) :
  x = (4 + Real.sqrt 2) / 7 := by
  sorry

end simplify_and_rationalize_l134_13419


namespace peanut_distribution_l134_13486

theorem peanut_distribution (x₁ x₂ x₃ x₄ x₅ : ℕ) : 
  x₁ + x₂ + x₃ + x₄ + x₅ = 100 ∧
  x₁ + x₂ = 52 ∧
  x₂ + x₃ = 43 ∧
  x₃ + x₄ = 34 ∧
  x₄ + x₅ = 30 →
  x₁ = 27 ∧ x₂ = 25 ∧ x₃ = 18 ∧ x₄ = 16 ∧ x₅ = 14 :=
by
  sorry

#check peanut_distribution

end peanut_distribution_l134_13486


namespace tangent_line_x_intercept_l134_13427

-- Define the function f(x) = x³ + 4x + 5
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

theorem tangent_line_x_intercept :
  let slope : ℝ := f' 1
  let y_intercept : ℝ := f 1 - slope * 1
  let x_intercept : ℝ := -y_intercept / slope
  x_intercept = -3/7 := by sorry

end tangent_line_x_intercept_l134_13427


namespace max_length_theorem_l134_13423

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a line passing through (0,1)
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the intersection points
def IntersectionPoints (k : ℝ) :
  (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

-- Define the lengths |A₁B₁| and |A₂B₂|
def Length_A1B1 (k : ℝ) : ℝ := sorry
def Length_A2B2 (k : ℝ) : ℝ := sorry

-- The main theorem
theorem max_length_theorem :
  ∃ k : ℝ, Length_A1B1 k = max_length_A1B1 ∧ Length_A2B2 k = 2 * Real.sqrt 30 / 3 :=
sorry

end max_length_theorem_l134_13423


namespace polynomial_remainder_l134_13499

theorem polynomial_remainder (x : ℝ) : 
  (x^5 + 2*x^2 + 3) % (x - 2) = 43 := by
  sorry

end polynomial_remainder_l134_13499


namespace cricket_team_size_l134_13446

/-- Represents the number of players on a cricket team -/
def total_players : ℕ := 55

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 37

/-- Represents the number of right-handed players on the team -/
def right_handed : ℕ := 49

/-- Theorem stating the total number of players on the cricket team -/
theorem cricket_team_size :
  total_players = throwers + (right_handed - throwers) * 3 / 2 :=
by sorry

end cricket_team_size_l134_13446


namespace select_five_from_eight_l134_13429

theorem select_five_from_eight (n m : ℕ) (h1 : n = 8) (h2 : m = 5) :
  Nat.choose n m = 56 := by
  sorry

end select_five_from_eight_l134_13429


namespace max_distance_to_line_l134_13497

/-- Given m ∈ ℝ, prove that the maximum distance from a point P(x,y) satisfying both
    x + m*y = 0 and m*x - y - 2*m + 4 = 0 to the line (x-1)*cos θ + (y-2)*sin θ = 3 is 3 + √5 -/
theorem max_distance_to_line (m : ℝ) :
  let P : ℝ × ℝ := (x, y) 
  ∃ x y : ℝ, x + m*y = 0 ∧ m*x - y - 2*m + 4 = 0 →
  (∀ θ : ℝ, (x - 1)*Real.cos θ + (y - 2)*Real.sin θ ≤ 3 + Real.sqrt 5) ∧
  (∃ θ : ℝ, (x - 1)*Real.cos θ + (y - 2)*Real.sin θ = 3 + Real.sqrt 5) :=
by sorry

end max_distance_to_line_l134_13497


namespace intersection_A_B_l134_13451

-- Define set A
def A : Set ℝ := {x | x - 1 < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 3 := by
  sorry

end intersection_A_B_l134_13451


namespace sin_x_in_terms_of_a_and_b_l134_13409

theorem sin_x_in_terms_of_a_and_b (a b x : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 0 < x) (h4 : x < π/2) (h5 : Real.tan x = (3*a*b) / (a^2 - b^2)) : 
  Real.sin x = (3*a*b) / Real.sqrt (a^4 + 7*a^2*b^2 + b^4) := by
  sorry

end sin_x_in_terms_of_a_and_b_l134_13409
