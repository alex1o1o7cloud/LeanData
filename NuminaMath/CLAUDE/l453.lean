import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l453_45336

-- Define the quadratic functions
def f (x : ℝ) := x^2 - 3*x - 4
def g (x : ℝ) := x^2 - x - 6

-- Define the solution sets
def S₁ : Set ℝ := {x | -1 < x ∧ x < 4}
def S₂ : Set ℝ := {x | x < -2 ∨ x > 3}

-- Theorem statements
theorem solution_set_f : {x : ℝ | f x < 0} = S₁ := by sorry

theorem solution_set_g : {x : ℝ | g x > 0} = S₂ := by sorry

end NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l453_45336


namespace NUMINAMATH_CALUDE_f_derivative_negative_solution_set_l453_45357

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- State the theorem
theorem f_derivative_negative_solution_set :
  {x : ℝ | (deriv f) x < 0} = Set.Ioo (-1) 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_negative_solution_set_l453_45357


namespace NUMINAMATH_CALUDE_halloween_bags_cost_l453_45306

/-- Calculates the minimum cost to buy a given number of items, 
    where items can be bought in packs of 5 or individually --/
def minCost (numItems : ℕ) (packPrice packSize : ℕ) (individualPrice : ℕ) : ℕ :=
  let numPacks := numItems / packSize
  let numIndividuals := numItems % packSize
  numPacks * packPrice + numIndividuals * individualPrice

theorem halloween_bags_cost : 
  let totalStudents : ℕ := 25
  let vampireRequests : ℕ := 11
  let pumpkinRequests : ℕ := 14
  let packPrice : ℕ := 3
  let packSize : ℕ := 5
  let individualPrice : ℕ := 1
  
  vampireRequests + pumpkinRequests = totalStudents →
  
  minCost vampireRequests packPrice packSize individualPrice + 
  minCost pumpkinRequests packPrice packSize individualPrice = 17 := by
  sorry

end NUMINAMATH_CALUDE_halloween_bags_cost_l453_45306


namespace NUMINAMATH_CALUDE_fold_sum_value_l453_45340

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a fold on graph paper -/
structure Fold :=
  (p1 : Point)
  (p2 : Point)
  (q1 : Point)
  (q2 : Point)

/-- The sum of the coordinates of the unknown point in a fold -/
def fold_sum (f : Fold) : ℝ := f.q2.x + f.q2.y

/-- The theorem stating the sum of coordinates of the unknown point -/
theorem fold_sum_value (f : Fold) 
  (h1 : f.p1 = ⟨0, 2⟩)
  (h2 : f.p2 = ⟨4, 0⟩)
  (h3 : f.q1 = ⟨7, 3⟩) :
  fold_sum f = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_fold_sum_value_l453_45340


namespace NUMINAMATH_CALUDE_discount_difference_is_399_l453_45304

def initial_price : ℝ := 8000

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def option1_discounts : List ℝ := [0.25, 0.15, 0.05]
def option2_discounts : List ℝ := [0.35, 0.10, 0.05]

def apply_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem discount_difference_is_399 :
  apply_discounts initial_price option1_discounts -
  apply_discounts initial_price option2_discounts = 399 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_is_399_l453_45304


namespace NUMINAMATH_CALUDE_chef_initial_apples_chef_had_46_apples_l453_45345

/-- The number of apples a chef had initially, given the number of apples left after making pies
    and the difference between the initial and final number of apples. -/
theorem chef_initial_apples (apples_left : ℕ) (difference : ℕ) : ℕ :=
  apples_left + difference

/-- Proof that the chef initially had 46 apples -/
theorem chef_had_46_apples : chef_initial_apples 14 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_chef_initial_apples_chef_had_46_apples_l453_45345


namespace NUMINAMATH_CALUDE_proposition_p_and_q_true_l453_45399

open Real

theorem proposition_p_and_q_true : 
  (∃ φ : ℝ, (φ = π / 2 ∧ 
    (∀ x : ℝ, sin (x + φ) = sin (-x - φ)) ∧
    (∃ ψ : ℝ, ψ ≠ π / 2 ∧ ∀ x : ℝ, sin (x + ψ) = sin (-x - ψ)))) ∧
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < π / 2 ∧ sin x₀ ≠ 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_and_q_true_l453_45399


namespace NUMINAMATH_CALUDE_total_markers_count_l453_45310

/-- The number of red markers Connie has -/
def red_markers : ℕ := 2315

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 1028

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers

/-- Theorem stating that the total number of markers is 3343 -/
theorem total_markers_count : total_markers = 3343 := by
  sorry

end NUMINAMATH_CALUDE_total_markers_count_l453_45310


namespace NUMINAMATH_CALUDE_unique_triple_solution_l453_45347

theorem unique_triple_solution : 
  ∃! (p q r : ℕ), 
    p > 0 ∧ q > 0 ∧ r > 0 ∧
    Nat.Prime p ∧ Nat.Prime q ∧
    (r^2 - 5*q^2) / (p^2 - 1) = 2 ∧
    p = 3 ∧ q = 2 ∧ r = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l453_45347


namespace NUMINAMATH_CALUDE_san_antonio_bound_passes_two_austin_bound_l453_45317

/-- Represents the direction of travel for a bus -/
inductive Direction
  | AustinToSanAntonio
  | SanAntonioToAustin

/-- Represents a bus schedule -/
structure BusSchedule where
  direction : Direction
  departureInterval : ℕ  -- in hours
  departureOffset : ℕ    -- in hours

/-- Represents the bus system between Austin and San Antonio -/
structure BusSystem where
  travelTime : ℕ
  austinToSanAntonioSchedule : BusSchedule
  sanAntonioToAustinSchedule : BusSchedule

/-- Counts the number of buses passed during a journey -/
def countPassedBuses (system : BusSystem) : ℕ :=
  sorry

/-- The main theorem stating that a San Antonio-bound bus passes exactly 2 Austin-bound buses -/
theorem san_antonio_bound_passes_two_austin_bound :
  ∀ (system : BusSystem),
    system.travelTime = 3 ∧
    system.austinToSanAntonioSchedule.direction = Direction.AustinToSanAntonio ∧
    system.austinToSanAntonioSchedule.departureInterval = 2 ∧
    system.austinToSanAntonioSchedule.departureOffset = 0 ∧
    system.sanAntonioToAustinSchedule.direction = Direction.SanAntonioToAustin ∧
    system.sanAntonioToAustinSchedule.departureInterval = 2 ∧
    system.sanAntonioToAustinSchedule.departureOffset = 1 →
    countPassedBuses system = 2 :=
  sorry

end NUMINAMATH_CALUDE_san_antonio_bound_passes_two_austin_bound_l453_45317


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l453_45311

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l453_45311


namespace NUMINAMATH_CALUDE_alice_baking_cake_l453_45378

theorem alice_baking_cake (total_flour : ℕ) (cup_capacity : ℕ) (h1 : total_flour = 750) (h2 : cup_capacity = 125) :
  total_flour / cup_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_alice_baking_cake_l453_45378


namespace NUMINAMATH_CALUDE_gcd_12012_18018_l453_45396

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12012_18018_l453_45396


namespace NUMINAMATH_CALUDE_sum_of_decimals_l453_45375

theorem sum_of_decimals : 2.75 + 0.003 + 0.158 = 2.911 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l453_45375


namespace NUMINAMATH_CALUDE_quadratic_p_value_l453_45348

/-- The quadratic function passing through specific points -/
def quadratic_function (p : ℝ) (x : ℝ) : ℝ := p * x^2 + 5 * x + p

theorem quadratic_p_value :
  ∃ (p : ℝ), 
    (quadratic_function p 0 = -2) ∧ 
    (quadratic_function p (1/2) = 0) ∧ 
    (quadratic_function p 2 = 0) ∧
    (p = -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_p_value_l453_45348


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l453_45391

/-- The cubic function f(x) = x^3 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- Theorem stating that f has exactly 3 zeros if and only if a is in the interval (-∞, -3) -/
theorem f_has_three_zeros (a : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔
  a < -3 :=
sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l453_45391


namespace NUMINAMATH_CALUDE_folded_rectangle_area_l453_45386

/-- Given a rectangle with dimensions 5 by 8, when folded to form a trapezoid
    where corners touch, the area of the resulting trapezoid is 55/2. -/
theorem folded_rectangle_area (rect_width : ℝ) (rect_length : ℝ) 
    (h_width : rect_width = 5)
    (h_length : rect_length = 8)
    (trapezoid_short_base : ℝ)
    (h_short_base : trapezoid_short_base = 3)
    (trapezoid_long_base : ℝ)
    (h_long_base : trapezoid_long_base = rect_length)
    (trapezoid_height : ℝ)
    (h_height : trapezoid_height = rect_width) : 
  (trapezoid_short_base + trapezoid_long_base) * trapezoid_height / 2 = 55 / 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_area_l453_45386


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l453_45307

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) :
  perimeter = 40 →
  area = (perimeter / 4)^2 →
  area = 100 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l453_45307


namespace NUMINAMATH_CALUDE_chosen_number_proof_l453_45334

theorem chosen_number_proof (x : ℝ) : (x / 6) - 189 = 3 → x = 1152 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l453_45334


namespace NUMINAMATH_CALUDE_isosceles_triangle_special_points_l453_45335

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2

/-- Checks if two line segments have equal length -/
def segmentsEqual (A B C D : Point) : Prop :=
  (A.x - B.x)^2 + (A.y - B.y)^2 = (C.x - D.x)^2 + (C.y - D.y)^2

/-- Checks if two line segments are parallel -/
def isParallel (A B C D : Point) : Prop :=
  (B.y - A.y) * (D.x - C.x) = (B.x - A.x) * (D.y - C.y)

/-- Calculates the angle between three points in degrees -/
noncomputable def angle (A B C : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem isosceles_triangle_special_points
  (t : Triangle)
  (M N P Q : Point)
  (h1 : isIsosceles t)
  (h2 : segmentsEqual t.A M M N)
  (h3 : segmentsEqual M N N t.C)
  (h4 : isParallel M Q t.B t.C)
  (h5 : isParallel N P t.A t.B)
  (h6 : segmentsEqual P Q t.B M) :
  angle M Q t.B = 36 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_special_points_l453_45335


namespace NUMINAMATH_CALUDE_distance_by_car_l453_45376

/-- Proves that the distance traveled by car is 6 kilometers -/
theorem distance_by_car (total_distance : ℝ) (h1 : total_distance = 24) :
  total_distance - (1/2 * total_distance + 1/4 * total_distance) = 6 := by
  sorry

#check distance_by_car

end NUMINAMATH_CALUDE_distance_by_car_l453_45376


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l453_45332

/-- Given that x² and y vary inversely, prove that when y = 20 for x = 3,
    then x = 3√10/50 when y = 5000 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) : 
  (∀ x y, x^2 * y = k) →  -- x² and y vary inversely
  (3^2 * 20 = k) →        -- y = 20 when x = 3
  (x^2 * 5000 = k) →      -- y = 5000 for the x we're looking for
  x = 3 * Real.sqrt 10 / 50 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l453_45332


namespace NUMINAMATH_CALUDE_uncovered_area_calculation_l453_45374

/-- Calculates the uncovered area of a rectangular floor when a square carpet is placed on it. -/
def uncovered_area (floor_length floor_width carpet_side : ℝ) : ℝ :=
  floor_length * floor_width - carpet_side * carpet_side

/-- Theorem stating that the uncovered area of a 10m x 8m floor with a 4m x 4m carpet is 64 square meters. -/
theorem uncovered_area_calculation :
  uncovered_area 10 8 4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_calculation_l453_45374


namespace NUMINAMATH_CALUDE_rotation_solutions_l453_45359

-- Define the basic geometric elements
def Point : Type := ℝ × ℝ × ℝ
def Line : Type := Point → Prop
def Plane : Type := Point → Prop

-- Define the given elements
variable (v : Line) -- Second elevation line
variable (P : Point) -- Original point
variable (P₂'' : Point) -- Inverted point parallel to second elevation plane

-- Define the geometric constructions
def rotationCircle (v : Line) (P : Point) : Set Point := sorry
def firstBisectorPlane : Plane := sorry
def planeS (v : Line) (P : Point) : Plane := sorry
def lineH₁ (v : Line) (P : Point) : Line := sorry

-- Define the number of intersections
def numIntersections (circle : Set Point) (line : Line) : ℕ := sorry

-- Define the number of solutions
def numSolutions (v : Line) (P : Point) : ℕ := sorry

-- State the theorem
theorem rotation_solutions (v : Line) (P : Point) :
  numSolutions v P = numIntersections (rotationCircle v P) (lineH₁ v P) := by sorry

end NUMINAMATH_CALUDE_rotation_solutions_l453_45359


namespace NUMINAMATH_CALUDE_simplify_expression_l453_45358

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  x⁻¹ - 3*x + 2 = -(3*x^2 - 2*x - 1) / x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l453_45358


namespace NUMINAMATH_CALUDE_all_triangles_isosceles_l453_45309

-- Define a point on the grid
structure GridPoint where
  x : Int
  y : Int

-- Define a triangle on the grid
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : GridPoint) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : GridTriangle) : Prop :=
  let d12 := squaredDistance t.p1 t.p2
  let d13 := squaredDistance t.p1 t.p3
  let d23 := squaredDistance t.p2 t.p3
  d12 = d13 ∨ d12 = d23 ∨ d13 = d23

-- Define the four triangles
def triangle1 : GridTriangle := ⟨⟨2, 2⟩, ⟨5, 2⟩, ⟨2, 5⟩⟩
def triangle2 : GridTriangle := ⟨⟨1, 1⟩, ⟨4, 1⟩, ⟨1, 4⟩⟩
def triangle3 : GridTriangle := ⟨⟨3, 3⟩, ⟨6, 3⟩, ⟨6, 6⟩⟩
def triangle4 : GridTriangle := ⟨⟨0, 0⟩, ⟨3, 0⟩, ⟨3, 3⟩⟩

-- Theorem: All four triangles are isosceles
theorem all_triangles_isosceles :
  isIsosceles triangle1 ∧
  isIsosceles triangle2 ∧
  isIsosceles triangle3 ∧
  isIsosceles triangle4 := by
  sorry

end NUMINAMATH_CALUDE_all_triangles_isosceles_l453_45309


namespace NUMINAMATH_CALUDE_libby_quarters_l453_45381

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The cost of the dress in dollars -/
def dress_cost : ℕ := 35

/-- The number of quarters Libby has left after paying for the dress -/
def quarters_left : ℕ := 20

/-- The initial number of quarters Libby had -/
def initial_quarters : ℕ := dress_cost * quarters_per_dollar + quarters_left

theorem libby_quarters : initial_quarters = 160 := by
  sorry

end NUMINAMATH_CALUDE_libby_quarters_l453_45381


namespace NUMINAMATH_CALUDE_largest_four_digit_perfect_square_l453_45322

theorem largest_four_digit_perfect_square : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → (∃ m : ℕ, n = m^2) → n ≤ 9261 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_perfect_square_l453_45322


namespace NUMINAMATH_CALUDE_min_questions_to_determine_product_l453_45329

theorem min_questions_to_determine_product (n : ℕ) (h : n > 3) :
  let min_questions_any_three := Int.ceil (n / 3 : ℚ)
  let min_questions_consecutive_three := if n % 3 = 0 then n / 3 else n
  true := by
  sorry

#check min_questions_to_determine_product

end NUMINAMATH_CALUDE_min_questions_to_determine_product_l453_45329


namespace NUMINAMATH_CALUDE_negation_equivalence_l453_45337

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, Real.log x₀ < x₀^2 - 1) ↔ (∀ x : ℝ, Real.log x ≥ x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l453_45337


namespace NUMINAMATH_CALUDE_matt_weight_matt_weight_is_80kg_l453_45326

/-- Given Matt's protein intake and requirements, calculate his weight. -/
theorem matt_weight (protein_percentage : ℝ) (protein_per_kg : ℝ) (powder_per_week : ℝ) : ℝ :=
  let protein_per_day := (powder_per_week / 7) * protein_percentage
  protein_per_day / protein_per_kg

/-- Prove that Matt weighs 80 kilograms given his protein intake and requirements. -/
theorem matt_weight_is_80kg : 
  matt_weight 0.80 2 1400 = 80 := by
  sorry

end NUMINAMATH_CALUDE_matt_weight_matt_weight_is_80kg_l453_45326


namespace NUMINAMATH_CALUDE_optimal_investment_l453_45314

/-- Represents an investment project with profit and loss rates -/
structure Project where
  maxProfitRate : Rat
  maxLossRate : Rat

/-- Represents an investment allocation -/
structure Investment where
  projectA : Rat
  projectB : Rat

def totalInvestment (i : Investment) : Rat :=
  i.projectA + i.projectB

def possibleLoss (p : Project) (i : Rat) : Rat :=
  i * p.maxLossRate

def possibleProfit (p : Project) (i : Rat) : Rat :=
  i * p.maxProfitRate

theorem optimal_investment
  (projectA : Project)
  (projectB : Project)
  (maxInvestment : Rat)
  (maxLoss : Rat)
  (h1 : projectA.maxProfitRate = 1)
  (h2 : projectB.maxProfitRate = 1/2)
  (h3 : projectA.maxLossRate = 3/10)
  (h4 : projectB.maxLossRate = 1/10)
  (h5 : maxInvestment = 100000)
  (h6 : maxLoss = 18000) :
  ∃ (i : Investment),
    totalInvestment i ≤ maxInvestment ∧
    possibleLoss projectA i.projectA + possibleLoss projectB i.projectB ≤ maxLoss ∧
    ∀ (j : Investment),
      totalInvestment j ≤ maxInvestment →
      possibleLoss projectA j.projectA + possibleLoss projectB j.projectB ≤ maxLoss →
      possibleProfit projectA i.projectA + possibleProfit projectB i.projectB ≥
      possibleProfit projectA j.projectA + possibleProfit projectB j.projectB ∧
    i.projectA = 40000 ∧
    i.projectB = 60000 :=
  sorry

#check optimal_investment

end NUMINAMATH_CALUDE_optimal_investment_l453_45314


namespace NUMINAMATH_CALUDE_problem_solved_probability_l453_45338

theorem problem_solved_probability :
  let p_A : ℚ := 1/2  -- Probability of student A solving the problem
  let p_B : ℚ := 1/3  -- Probability of student B solving the problem
  let p_C : ℚ := 1/4  -- Probability of student C solving the problem
  -- The probability that the problem is solved by at least one student
  (1 : ℚ) - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_problem_solved_probability_l453_45338


namespace NUMINAMATH_CALUDE_star_shape_perimeter_star_shape_perimeter_is_4pi_l453_45324

/-- The perimeter of a star-like shape formed by arcs of six unit circles arranged in a regular hexagon configuration --/
theorem star_shape_perimeter : ℝ :=
  let n : ℕ := 6  -- number of coins
  let r : ℝ := 1  -- radius of each coin
  let angle_sum : ℝ := 2 * Real.pi  -- sum of internal angles of a hexagon
  4 * Real.pi

/-- Proof that the perimeter of the star-like shape is 4π --/
theorem star_shape_perimeter_is_4pi : star_shape_perimeter = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_star_shape_perimeter_star_shape_perimeter_is_4pi_l453_45324


namespace NUMINAMATH_CALUDE_gmat_test_problem_l453_45320

theorem gmat_test_problem (first_correct : Real) (second_correct : Real) (neither_correct : Real) :
  first_correct = 85 / 100 →
  second_correct = 65 / 100 →
  neither_correct = 5 / 100 →
  first_correct + second_correct - (1 - neither_correct) = 55 / 100 := by
  sorry

end NUMINAMATH_CALUDE_gmat_test_problem_l453_45320


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l453_45392

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4)^2 = 1/2 * x * triangle_height →
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l453_45392


namespace NUMINAMATH_CALUDE_average_difference_with_input_error_l453_45388

theorem average_difference_with_input_error (n : ℕ) (correct_value wrong_value : ℝ) : 
  n = 30 → correct_value = 75 → wrong_value = 15 → 
  (correct_value - wrong_value) / n = -2 := by
sorry

end NUMINAMATH_CALUDE_average_difference_with_input_error_l453_45388


namespace NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_conditions_sixty_is_smallest_smallest_n_is_sixty_l453_45393

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 45 ∣ n^2 ∧ 1152 ∣ n^4 → n ≥ 60 :=
by sorry

theorem sixty_satisfies_conditions : 45 ∣ 60^2 ∧ 1152 ∣ 60^4 :=
by sorry

theorem sixty_is_smallest : ∀ m : ℕ, m > 0 ∧ 45 ∣ m^2 ∧ 1152 ∣ m^4 → m ≥ 60 :=
by sorry

theorem smallest_n_is_sixty : ∃! n : ℕ, n > 0 ∧ 45 ∣ n^2 ∧ 1152 ∣ n^4 ∧ ∀ m : ℕ, (m > 0 ∧ 45 ∣ m^2 ∧ 1152 ∣ m^4 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_conditions_sixty_is_smallest_smallest_n_is_sixty_l453_45393


namespace NUMINAMATH_CALUDE_no_solution_fibonacci_equation_l453_45385

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem no_solution_fibonacci_equation :
  ∀ n : ℕ, n * (fib n) * (fib (n + 1)) ≠ (fib (n + 2) - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fibonacci_equation_l453_45385


namespace NUMINAMATH_CALUDE_total_items_given_out_l453_45361

/-- The number of groups in Miss Davis's class -/
def num_groups : ℕ := 10

/-- The number of popsicle sticks given to each group -/
def popsicle_sticks_per_group : ℕ := 15

/-- The number of straws given to each group -/
def straws_per_group : ℕ := 20

/-- Theorem stating the total number of items given out by Miss Davis -/
theorem total_items_given_out :
  (num_groups * (popsicle_sticks_per_group + straws_per_group)) = 350 := by
  sorry

end NUMINAMATH_CALUDE_total_items_given_out_l453_45361


namespace NUMINAMATH_CALUDE_count_integers_with_5_or_6_l453_45377

/-- The number of integers among the first 729 positive integers in base 9 
    that contain either 5 or 6 (or both) as a digit -/
def count_with_5_or_6 : ℕ := 386

/-- The base of the number system we're working with -/
def base : ℕ := 9

/-- The number of smallest positive integers we're considering -/
def total_count : ℕ := 729

theorem count_integers_with_5_or_6 :
  count_with_5_or_6 = total_count - (base - 2)^3 ∧
  total_count = base^3 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_5_or_6_l453_45377


namespace NUMINAMATH_CALUDE_square_areas_sum_l453_45368

theorem square_areas_sum (a b c : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 6) :
  a^2 + b^2 + c^2 = 7^2 := by
  sorry

end NUMINAMATH_CALUDE_square_areas_sum_l453_45368


namespace NUMINAMATH_CALUDE_longest_frog_vs_shortest_grasshopper_l453_45354

def frog_jumps : List ℕ := [39, 45, 50]
def grasshopper_jumps : List ℕ := [17, 22, 28, 31]

theorem longest_frog_vs_shortest_grasshopper :
  (List.maximum frog_jumps).get! - (List.minimum grasshopper_jumps).get! = 33 := by
  sorry

end NUMINAMATH_CALUDE_longest_frog_vs_shortest_grasshopper_l453_45354


namespace NUMINAMATH_CALUDE_available_seats_l453_45373

/-- Proves the number of available seats in an auditorium -/
theorem available_seats (total_seats : ℕ) 
  (h1 : total_seats = 500)
  (h2 : 2 * total_seats / 5 = total_seats * 2 / 5)
  (h3 : total_seats / 10 = total_seats * 1 / 10) :
  total_seats - (total_seats * 2 / 5 + total_seats * 1 / 10) = 250 := by
  sorry

#check available_seats

end NUMINAMATH_CALUDE_available_seats_l453_45373


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l453_45300

theorem arctan_equation_solution :
  ∀ y : ℝ, 2 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4 → y = 1210 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l453_45300


namespace NUMINAMATH_CALUDE_total_scoops_l453_45325

def flour_cups : ℚ := 3
def sugar_cups : ℚ := 2
def flour_scoop : ℚ := 1/4
def sugar_scoop : ℚ := 1/3

theorem total_scoops : 
  (flour_cups / flour_scoop + sugar_cups / sugar_scoop : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_scoops_l453_45325


namespace NUMINAMATH_CALUDE_functional_equation_solution_l453_45333

theorem functional_equation_solution (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (2*m + 2*n) = f m * f n) : 
  ∀ x : ℕ, f x = 1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l453_45333


namespace NUMINAMATH_CALUDE_reflect_A_across_x_axis_l453_45356

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflect_A_across_x_axis :
  let A : ℝ × ℝ := (-4, 3)
  reflect_x A = (-4, -3) := by
sorry

end NUMINAMATH_CALUDE_reflect_A_across_x_axis_l453_45356


namespace NUMINAMATH_CALUDE_meals_for_adults_l453_45353

/-- The number of meals initially available for adults -/
def A : ℕ := 18

/-- The number of children that can be fed with all the meals -/
def C : ℕ := 90

/-- Theorem stating that A is the correct number of meals initially available for adults -/
theorem meals_for_adults : 
  (∀ x : ℕ, x * (C / A) = 72 → x = 14) ∧ 
  (A : ℚ) = C / (72 / 14) :=
sorry

end NUMINAMATH_CALUDE_meals_for_adults_l453_45353


namespace NUMINAMATH_CALUDE_total_time_in_hours_l453_45372

def laundry_time : ℕ := 30
def bathroom_cleaning_time : ℕ := 15
def room_cleaning_time : ℕ := 35
def homework_time : ℕ := 40

def minutes_per_hour : ℕ := 60

theorem total_time_in_hours :
  (laundry_time + bathroom_cleaning_time + room_cleaning_time + homework_time) / minutes_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_total_time_in_hours_l453_45372


namespace NUMINAMATH_CALUDE_tiles_for_wall_l453_45365

/-- The number of tiles needed to cover a wall -/
def tiles_needed (tile_size wall_length wall_width : ℕ) : ℕ :=
  (wall_length / tile_size) * (wall_width / tile_size)

/-- Theorem: 432 tiles of size 15 cm × 15 cm are needed to cover a wall of 360 cm × 270 cm -/
theorem tiles_for_wall : tiles_needed 15 360 270 = 432 := by
  sorry

end NUMINAMATH_CALUDE_tiles_for_wall_l453_45365


namespace NUMINAMATH_CALUDE_min_value_exponential_function_l453_45331

theorem min_value_exponential_function :
  (∀ x : ℝ, Real.exp x + 4 * Real.exp (-x) ≥ 4) ∧
  (∃ x : ℝ, Real.exp x + 4 * Real.exp (-x) = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_exponential_function_l453_45331


namespace NUMINAMATH_CALUDE_overtake_at_eight_hours_l453_45382

/-- Represents the chase between a pirate ship and a trading vessel -/
structure ChaseScenario where
  initial_distance : ℝ
  pirate_initial_speed : ℝ
  trading_initial_speed : ℝ
  damage_time : ℝ
  pirate_damaged_distance : ℝ
  trading_damaged_distance : ℝ

/-- The time at which the pirate ship overtakes the trading vessel -/
def overtake_time (scenario : ChaseScenario) : ℝ :=
  sorry

/-- The specific chase scenario described in the problem -/
def given_scenario : ChaseScenario :=
  { initial_distance := 15
  , pirate_initial_speed := 14
  , trading_initial_speed := 10
  , damage_time := 3
  , pirate_damaged_distance := 18
  , trading_damaged_distance := 17 }

/-- Theorem stating that the overtake time for the given scenario is 8 hours -/
theorem overtake_at_eight_hours :
  overtake_time given_scenario = 8 :=
sorry

end NUMINAMATH_CALUDE_overtake_at_eight_hours_l453_45382


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l453_45343

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l453_45343


namespace NUMINAMATH_CALUDE_saturday_extra_calories_l453_45364

def daily_calories : ℕ := 2500
def daily_burn : ℕ := 3000
def weekly_deficit : ℕ := 2500
def days_in_week : ℕ := 7
def regular_days : ℕ := 6

def total_weekly_burn : ℕ := daily_burn * days_in_week
def regular_weekly_intake : ℕ := daily_calories * regular_days
def total_weekly_intake : ℕ := total_weekly_burn - weekly_deficit

theorem saturday_extra_calories :
  total_weekly_intake - regular_weekly_intake - daily_calories = 1000 := by
  sorry

end NUMINAMATH_CALUDE_saturday_extra_calories_l453_45364


namespace NUMINAMATH_CALUDE_perimeter_pentagon_l453_45313

/-- Given a square PQRS and a triangle PZS, this theorem proves the perimeter of pentagon PQRSZ -/
theorem perimeter_pentagon (x : ℝ) : 
  let square_perimeter : ℝ := 120
  let triangle_perimeter : ℝ := 2 * x
  let pentagon_perimeter : ℝ := square_perimeter / 2 + triangle_perimeter - square_perimeter / 4
  pentagon_perimeter = 60 + 2 * x :=
by sorry

end NUMINAMATH_CALUDE_perimeter_pentagon_l453_45313


namespace NUMINAMATH_CALUDE_inequality_exp_l453_45312

theorem inequality_exp (m n : ℝ) (h1 : m > n) (h2 : n > 0) : m * Real.exp m + n < n * Real.exp m + m := by
  sorry

end NUMINAMATH_CALUDE_inequality_exp_l453_45312


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l453_45350

-- Define the angle in degrees
def angle : ℤ := -3290

-- Function to normalize an angle to the range [0, 360)
def normalizeAngle (a : ℤ) : ℤ :=
  a % 360

-- Function to determine the quadrant of a normalized angle
def quadrant (a : ℤ) : ℕ :=
  if 0 ≤ a ∧ a < 90 then 1
  else if 90 ≤ a ∧ a < 180 then 2
  else if 180 ≤ a ∧ a < 270 then 3
  else 4

-- Theorem statement
theorem angle_in_fourth_quadrant :
  quadrant (normalizeAngle angle) = 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l453_45350


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l453_45390

theorem sqrt_sum_inequality (a b : ℝ) 
  (h1 : a + b = 1) 
  (h2 : (a + 1/2) * (b + 1/2) ≥ 0) : 
  Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l453_45390


namespace NUMINAMATH_CALUDE_decoration_time_proof_l453_45363

/-- The time needed for Mia and Billy to decorate Easter eggs -/
def decoration_time (mia_rate : ℕ) (billy_rate : ℕ) (total_eggs : ℕ) : ℚ :=
  total_eggs / (mia_rate + billy_rate)

/-- Theorem stating that Mia and Billy will take 5 hours to decorate 170 eggs -/
theorem decoration_time_proof :
  decoration_time 24 10 170 = 5 := by
  sorry

end NUMINAMATH_CALUDE_decoration_time_proof_l453_45363


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_range_l453_45301

theorem rectangular_solid_volume_range (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  2 * (a * b + b * c + a * c) = 48 →
  4 * (a + b + c) = 36 →
  16 ≤ a * b * c ∧ a * b * c ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_range_l453_45301


namespace NUMINAMATH_CALUDE_count_valid_numbers_eq_441_l453_45383

/-- The count of valid digits for hundreds place (1-4, 7-9) -/
def valid_hundreds : Nat := 7

/-- The count of valid digits for tens place (0-4, 7-9) -/
def valid_tens : Nat := 7

/-- The count of valid digits for units place (1-9) -/
def valid_units : Nat := 9

/-- The count of three-digit whole numbers with no 5's and 6's in the tens and hundreds places -/
def count_valid_numbers : Nat := valid_hundreds * valid_tens * valid_units

theorem count_valid_numbers_eq_441 : count_valid_numbers = 441 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_eq_441_l453_45383


namespace NUMINAMATH_CALUDE_existence_of_comparable_indices_l453_45315

theorem existence_of_comparable_indices
  (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_comparable_indices_l453_45315


namespace NUMINAMATH_CALUDE_triangle_area_l453_45352

/-- The area of a right triangle with vertices at (0, 0), (0, 10), and (-10, 0) is 50 square units,
    given that the points (-3, 7) and (-7, 3) lie on its hypotenuse. -/
theorem triangle_area : 
  let p1 : ℝ × ℝ := (-3, 7)
  let p2 : ℝ × ℝ := (-7, 3)
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (0, 10)
  let v3 : ℝ × ℝ := (-10, 0)
  (p1.1 - p2.1) / (p1.2 - p2.2) = 1 →  -- Slope of the line through p1 and p2 is 1
  (∃ t : ℝ, v2 = p1 + t • (1, 1)) →  -- v2 lies on the line through p1 with slope 1
  (∃ t : ℝ, v3 = p2 + t • (1, 1)) →  -- v3 lies on the line through p2 with slope 1
  (1/2) * (v2.2 - v1.2) * (v1.1 - v3.1) = 50 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l453_45352


namespace NUMINAMATH_CALUDE_exponential_function_max_min_sum_l453_45344

theorem exponential_function_max_min_sum (a : ℝ) (f : ℝ → ℝ) :
  a > 1 →
  (∀ x, f x = a^x) →
  (∃ max min : ℝ, (∀ x ∈ Set.Icc 0 1, f x ≤ max) ∧
                  (∀ x ∈ Set.Icc 0 1, min ≤ f x) ∧
                  max + min = 3) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_max_min_sum_l453_45344


namespace NUMINAMATH_CALUDE_tan_theta_solution_l453_45395

theorem tan_theta_solution (θ : Real) (h1 : 0 < θ * (180 / Real.pi)) 
  (h2 : θ * (180 / Real.pi) < 30) 
  (h3 : Real.tan θ + Real.tan (4 * θ) + Real.tan (6 * θ) = 0) : 
  Real.tan θ = 1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_solution_l453_45395


namespace NUMINAMATH_CALUDE_base10_157_equals_base12_B21_l453_45318

/-- Converts a base 12 number to base 10 -/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 12 + d) 0

/-- Represents 'B' as 11 in base 12 -/
def baseB : Nat := 11

theorem base10_157_equals_base12_B21 :
  157 = base12ToBase10 [baseB, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_base10_157_equals_base12_B21_l453_45318


namespace NUMINAMATH_CALUDE_equation_solution_l453_45328

theorem equation_solution : ∃! x : ℝ, x > 0 ∧ (x - 3) / 12 = 5 / (x - 12) ∧ x = (15 + Real.sqrt 321) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l453_45328


namespace NUMINAMATH_CALUDE_product_of_recurring_decimal_and_seven_l453_45370

theorem product_of_recurring_decimal_and_seven :
  ∃ (x : ℚ), (∃ (n : ℕ), x = (456 : ℚ) / (10^3 - 1)) ∧ 7 * x = 355 / 111 := by
  sorry

end NUMINAMATH_CALUDE_product_of_recurring_decimal_and_seven_l453_45370


namespace NUMINAMATH_CALUDE_digit_sequence_bound_l453_45303

/-- Given a positive integer N with n digits, if all its digits are distinct
    and the sum of any three consecutive digits is divisible by 5, then n ≤ 6. -/
theorem digit_sequence_bound (N : ℕ) (n : ℕ) : 
  (N ≥ 10^(n-1) ∧ N < 10^n) →  -- N is an n-digit number
  (∀ i j, i ≠ j → (N / 10^i) % 10 ≠ (N / 10^j) % 10) →  -- All digits are distinct
  (∀ i, i + 2 < n → ((N / 10^i) % 10 + (N / 10^(i+1)) % 10 + (N / 10^(i+2)) % 10) % 5 = 0) →  -- Sum of any three consecutive digits is divisible by 5
  n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_digit_sequence_bound_l453_45303


namespace NUMINAMATH_CALUDE_statement_a_statement_b_l453_45327

-- Define rationality for real numbers
def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Statement a
theorem statement_a : ∃ (x : ℝ), IsRational (x^7) ∧ IsRational (x^12) ∧ ¬IsRational x := by
  sorry

-- Statement b
theorem statement_b : ∀ (x : ℝ), IsRational (x^9) ∧ IsRational (x^12) → IsRational x := by
  sorry

end NUMINAMATH_CALUDE_statement_a_statement_b_l453_45327


namespace NUMINAMATH_CALUDE_logarithm_sum_inequality_l453_45384

theorem logarithm_sum_inequality : 
  Real.log 6 / Real.log 5 + Real.log 7 / Real.log 6 + Real.log 8 / Real.log 7 + Real.log 5 / Real.log 8 > 4 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_inequality_l453_45384


namespace NUMINAMATH_CALUDE_min_cost_for_48_students_l453_45351

/-- The minimum cost to purchase tickets for a group of students. -/
def min_ticket_cost (num_students : ℕ) (single_price : ℕ) (group_price : ℕ) : ℕ :=
  min
    ((num_students / 10) * group_price + (num_students % 10) * single_price)
    ((num_students / 10 + 1) * group_price)

/-- The minimum cost to purchase tickets for 48 students is 350 yuan. -/
theorem min_cost_for_48_students :
  min_ticket_cost 48 10 70 = 350 := by
  sorry

#eval min_ticket_cost 48 10 70

end NUMINAMATH_CALUDE_min_cost_for_48_students_l453_45351


namespace NUMINAMATH_CALUDE_triangle_pcd_area_l453_45321

/-- Given points P(0, 18), D(3, 18), and C(0, q) in a Cartesian coordinate system,
    where PD and PC are perpendicular sides of triangle PCD,
    prove that the area of triangle PCD is equal to 27 - (3/2)q. -/
theorem triangle_pcd_area (q : ℝ) : 
  let P : ℝ × ℝ := (0, 18)
  let D : ℝ × ℝ := (3, 18)
  let C : ℝ × ℝ := (0, q)
  -- PD and PC are perpendicular
  (D.1 - P.1) * (C.2 - P.2) = 0 →
  -- Area of triangle PCD
  (1/2) * (D.1 - P.1) * (P.2 - C.2) = 27 - (3/2) * q := by
  sorry

end NUMINAMATH_CALUDE_triangle_pcd_area_l453_45321


namespace NUMINAMATH_CALUDE_continuous_function_property_l453_45394

open Real Set

theorem continuous_function_property (d : ℝ) (h_d : d ∈ Ioc 0 1) :
  (∀ f : ℝ → ℝ, ContinuousOn f (Icc 0 1) → f 0 = f 1 →
    ∃ x₀ ∈ Icc 0 (1 - d), f x₀ = f (x₀ + d)) ↔
  ∃ k : ℕ, d = 1 / k :=
by sorry


end NUMINAMATH_CALUDE_continuous_function_property_l453_45394


namespace NUMINAMATH_CALUDE_problem_statement_l453_45346

theorem problem_statement (a : ℤ) 
  (h1 : 0 ≤ a) (h2 : a < 13) 
  (h3 : (51^2018 + a) % 13 = 0) : 
  a = 12 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l453_45346


namespace NUMINAMATH_CALUDE_infinite_sum_equals_four_l453_45305

open BigOperators

theorem infinite_sum_equals_four : 
  ∑' (n : ℕ), (3 * n + 2) / (n * (n + 1) * (n + 3)) = 4 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_four_l453_45305


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l453_45397

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l453_45397


namespace NUMINAMATH_CALUDE_area_of_side_face_l453_45367

/-- Represents a rectangular box with length, width, and height -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Theorem: Area of side face of a rectangular box -/
theorem area_of_side_face (b : Box) 
  (h1 : b.width * b.height = 0.5 * (b.length * b.width))
  (h2 : b.length * b.width = 1.5 * (b.length * b.height))
  (h3 : b.length * b.width * b.height = 5184) :
  b.length * b.height = 288 := by
  sorry

end NUMINAMATH_CALUDE_area_of_side_face_l453_45367


namespace NUMINAMATH_CALUDE_train_speed_calculation_l453_45308

/-- Proves that a train with given length, crossing a bridge of given length in a given time, has a specific speed in km/hr -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ 
  bridge_length = 132 ∧ 
  crossing_time = 13.598912087033037 →
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l453_45308


namespace NUMINAMATH_CALUDE_two_lines_forming_30_degrees_l453_45323

/-- Represents a line in 3D space -/
structure Line3D where
  -- Define necessary properties for a line

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Define necessary properties for a plane

/-- Angle between a line and a plane -/
def angle_line_plane (l : Line3D) (p : Plane3D) : ℝ :=
  sorry

/-- Angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if a line passes through a point -/
def line_passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- The main theorem -/
theorem two_lines_forming_30_degrees (a : Line3D) (α : Plane3D) (P : Point3D) :
  angle_line_plane a α = 30 →
  ∃! (s : Finset Line3D), 
    s.card = 2 ∧ 
    ∀ b ∈ s, line_passes_through b P ∧ 
              angle_between_lines a b = 30 ∧ 
              angle_line_plane b α = 30 :=
sorry

end NUMINAMATH_CALUDE_two_lines_forming_30_degrees_l453_45323


namespace NUMINAMATH_CALUDE_dance_troupe_max_members_l453_45362

theorem dance_troupe_max_members :
  ∀ m : ℤ,
  (∃ k : ℤ, 25 * m = 31 * k + 7) →
  25 * m < 1300 →
  25 * m ≤ 875 :=
by sorry

end NUMINAMATH_CALUDE_dance_troupe_max_members_l453_45362


namespace NUMINAMATH_CALUDE_median_invariance_l453_45342

def judges_scores := Fin 7 → ℝ
def reduced_scores := Fin 5 → ℝ

def median (scores : Fin n → ℝ) : ℝ :=
  sorry

def remove_extremes (scores : judges_scores) : reduced_scores :=
  sorry

theorem median_invariance (scores : judges_scores) :
  median scores = median (remove_extremes scores) :=
sorry

end NUMINAMATH_CALUDE_median_invariance_l453_45342


namespace NUMINAMATH_CALUDE_prob_at_least_one_heart_or_king_l453_45379

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either a heart or a king -/
def target_cards : ℕ := 16

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The probability of drawing at least one heart or king in three draws with replacement -/
theorem prob_at_least_one_heart_or_king :
  1 - (((deck_size - target_cards : ℚ) / deck_size) ^ num_draws) = 1468 / 2197 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_heart_or_king_l453_45379


namespace NUMINAMATH_CALUDE_possible_x_values_l453_45389

def M (x : ℝ) : Set ℝ := {3, 9, 3*x}
def N (x : ℝ) : Set ℝ := {3, x^2}

theorem possible_x_values :
  ∀ x : ℝ, N x ⊆ M x → x = -3 ∨ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_possible_x_values_l453_45389


namespace NUMINAMATH_CALUDE_no_equal_notebooks_l453_45369

theorem no_equal_notebooks : ¬∃ (x : ℝ), x > 0 ∧ 12 / x = 21 / (x + 1.2) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_notebooks_l453_45369


namespace NUMINAMATH_CALUDE_distance_circle_center_to_point_l453_45319

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y - 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 3)

-- Define the given point
def given_point : ℝ × ℝ := (10, 8)

-- Theorem statement
theorem distance_circle_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  (cx - px)^2 + (cy - py)^2 = 89 :=
sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_point_l453_45319


namespace NUMINAMATH_CALUDE_log_arithmetic_mean_implies_geometric_mean_geometric_mean_not_implies_log_arithmetic_mean_l453_45371

-- Define the arithmetic mean of logarithms
def log_arithmetic_mean (x y z : ℝ) : Prop :=
  Real.log y = (Real.log x + Real.log z) / 2

-- Define the geometric mean
def geometric_mean (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem log_arithmetic_mean_implies_geometric_mean
  (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  log_arithmetic_mean x y z → geometric_mean x y z :=
sorry

theorem geometric_mean_not_implies_log_arithmetic_mean :
  ∃ x y z : ℝ, geometric_mean x y z ∧ ¬log_arithmetic_mean x y z :=
sorry

end NUMINAMATH_CALUDE_log_arithmetic_mean_implies_geometric_mean_geometric_mean_not_implies_log_arithmetic_mean_l453_45371


namespace NUMINAMATH_CALUDE_vector_equation_proof_l453_45380

/-- Prove that the given values of a and b satisfy the vector equation -/
theorem vector_equation_proof :
  let a : ℚ := -3/14
  let b : ℚ := 107/14
  let v1 : Fin 2 → ℚ := ![3, 4]
  let v2 : Fin 2 → ℚ := ![1, 6]
  let result : Fin 2 → ℚ := ![7, 45]
  (a • v1 + b • v2) = result :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_proof_l453_45380


namespace NUMINAMATH_CALUDE_length_MN_is_six_l453_45355

-- Define the points
variable (A B C D M N : ℝ)

-- Define the conditions
axiom on_segment : A < C ∧ C < D ∧ D < B
axiom midpoint_M : M = (A + C) / 2
axiom midpoint_N : N = (D + B) / 2
axiom length_AB : B - A = 10
axiom length_CD : D - C = 2

-- Theorem statement
theorem length_MN_is_six : N - M = 6 := by sorry

end NUMINAMATH_CALUDE_length_MN_is_six_l453_45355


namespace NUMINAMATH_CALUDE_pokemon_game_l453_45316

theorem pokemon_game (n : ℕ) : 
  (∃ (m : ℕ), 
    n * m + 11 * (m + 6) = n^2 + 3*n - 2 ∧ 
    m > 0 ∧ 
    (m + 6) > 0) → 
  n = 9 :=
by sorry

end NUMINAMATH_CALUDE_pokemon_game_l453_45316


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l453_45360

theorem quadratic_roots_expression (r s : ℝ) : 
  (2 * r^2 - 3 * r = 11) → 
  (2 * s^2 - 3 * s = 11) → 
  r ≠ s →
  (4 * r^3 - 4 * s^3) / (r - s) = 31 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l453_45360


namespace NUMINAMATH_CALUDE_discount_clinic_savings_l453_45341

theorem discount_clinic_savings (normal_fee : ℝ) : 
  (normal_fee - 2 * (0.3 * normal_fee) = 80) → normal_fee = 200 := by
  sorry

end NUMINAMATH_CALUDE_discount_clinic_savings_l453_45341


namespace NUMINAMATH_CALUDE_second_printer_theorem_l453_45366

/-- The time (in minutes) it takes for the second printer to print 800 flyers -/
def second_printer_time (first_printer_time second_printer_time combined_time : ℚ) : ℚ :=
  30 / 7

/-- Given the specifications of two printers, proves that the second printer
    takes 30/7 minutes to print 800 flyers -/
theorem second_printer_theorem (first_printer_time combined_time : ℚ) 
  (h1 : first_printer_time = 10)
  (h2 : combined_time = 3) :
  second_printer_time first_printer_time (second_printer_time first_printer_time (30/7) combined_time) combined_time = 30 / 7 := by
  sorry

#check second_printer_theorem

end NUMINAMATH_CALUDE_second_printer_theorem_l453_45366


namespace NUMINAMATH_CALUDE_original_number_proof_l453_45339

theorem original_number_proof (x : ℝ) : 
  (x * 1.1 * 1.15 = 632.5) → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l453_45339


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l453_45387

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l453_45387


namespace NUMINAMATH_CALUDE_nonagon_diagonal_count_l453_45398

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonal_count : nonagon_diagonals = (nonagon_sides * (nonagon_sides - 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_count_l453_45398


namespace NUMINAMATH_CALUDE_base_conversion_3012_to_octal_l453_45330

theorem base_conversion_3012_to_octal :
  (3012 : ℕ) = 5 * (8 : ℕ)^3 + 7 * (8 : ℕ)^2 + 0 * (8 : ℕ)^1 + 4 * (8 : ℕ)^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_3012_to_octal_l453_45330


namespace NUMINAMATH_CALUDE_assembly_line_theorem_l453_45349

/-- Represents the assembly line production --/
structure AssemblyLine where
  initial_rate : ℕ
  initial_order : ℕ
  increased_rate : ℕ
  second_order : ℕ

/-- Calculates the overall average output of the assembly line --/
def average_output (line : AssemblyLine) : ℚ :=
  let total_cogs := line.initial_order + line.second_order
  let total_time := (line.initial_order : ℚ) / line.initial_rate + (line.second_order : ℚ) / line.increased_rate
  total_cogs / total_time

/-- Theorem stating that the average output for the given conditions is 40 cogs per hour --/
theorem assembly_line_theorem (line : AssemblyLine) 
    (h1 : line.initial_rate = 30)
    (h2 : line.initial_order = 60)
    (h3 : line.increased_rate = 60)
    (h4 : line.second_order = 60) :
  average_output line = 40 := by
  sorry

end NUMINAMATH_CALUDE_assembly_line_theorem_l453_45349


namespace NUMINAMATH_CALUDE_range_of_f_l453_45302

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define the range
def range : Set ℝ := { y | ∃ x ∈ domain, f x = y }

-- Theorem statement
theorem range_of_f : range = { y | -3 ≤ y ∧ y ≤ 5 } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l453_45302
