import Mathlib

namespace NUMINAMATH_CALUDE_stock_value_ordering_l3984_398416

def initial_investment : ℝ := 200

def alpha_year1 : ℝ := 1.30
def beta_year1 : ℝ := 0.80
def gamma_year1 : ℝ := 1.10
def delta_year1 : ℝ := 0.90

def alpha_year2 : ℝ := 0.85
def beta_year2 : ℝ := 1.30
def gamma_year2 : ℝ := 0.95
def delta_year2 : ℝ := 1.20

def final_alpha : ℝ := initial_investment * alpha_year1 * alpha_year2
def final_beta : ℝ := initial_investment * beta_year1 * beta_year2
def final_gamma : ℝ := initial_investment * gamma_year1 * gamma_year2
def final_delta : ℝ := initial_investment * delta_year1 * delta_year2

theorem stock_value_ordering :
  final_delta < final_beta ∧ final_beta < final_gamma ∧ final_gamma < final_alpha :=
by sorry

end NUMINAMATH_CALUDE_stock_value_ordering_l3984_398416


namespace NUMINAMATH_CALUDE_equation_solutions_l3984_398479

theorem equation_solutions :
  (∃ x : ℝ, x^2 + 2*x = 0 ↔ x = 0 ∨ x = -2) ∧
  (∃ x : ℝ, 4*x^2 - 4*x + 1 = 0 ↔ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3984_398479


namespace NUMINAMATH_CALUDE_problem_solution_l3984_398446

theorem problem_solution (x y : ℝ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : 1/x + 1/y = 1) 
  (h4 : x * y = 9) : 
  y = (9 + 3 * Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3984_398446


namespace NUMINAMATH_CALUDE_sector_area_l3984_398409

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = π / 6) :
  (1 / 2) * r^2 * θ = 3 * π := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l3984_398409


namespace NUMINAMATH_CALUDE_smoking_rate_estimate_l3984_398444

/-- Represents the survey results and conditions -/
structure SurveyData where
  total_students : ℕ
  yes_answers : ℕ
  die_prob : ℚ

/-- Calculates the estimated smoking rate based on survey data -/
def estimate_smoking_rate (data : SurveyData) : ℚ :=
  let estimated_smokers := data.yes_answers / 2
  (estimated_smokers : ℚ) / data.total_students

/-- Theorem stating the estimated smoking rate for the given survey data -/
theorem smoking_rate_estimate (data : SurveyData) 
  (h1 : data.total_students = 300)
  (h2 : data.yes_answers = 80)
  (h3 : data.die_prob = 1/2) :
  ∃ (ε : ℚ), abs (estimate_smoking_rate data - 40/300) < ε ∧ ε < 1/1000 := by
  sorry

end NUMINAMATH_CALUDE_smoking_rate_estimate_l3984_398444


namespace NUMINAMATH_CALUDE_simplify_expression_value_under_condition_independence_condition_l3984_398454

/-- Given algebraic expressions A and B -/
def A (m y : ℝ) : ℝ := 2 * m^2 + 3 * m * y + 2 * y - 1

def B (m y : ℝ) : ℝ := m^2 - m * y

/-- Theorem 1: Simplification of 3A - 2(A + B) -/
theorem simplify_expression (m y : ℝ) :
  3 * A m y - 2 * (A m y + B m y) = 5 * m * y + 2 * y - 1 := by sorry

/-- Theorem 2: Value of 3A - 2(A + B) under specific condition -/
theorem value_under_condition (m y : ℝ) :
  (m - 1)^2 + |y + 2| = 0 →
  3 * A m y - 2 * (A m y + B m y) = -15 := by sorry

/-- Theorem 3: Condition for 3A - 2(A + B) to be independent of y -/
theorem independence_condition (m : ℝ) :
  (∀ y : ℝ, 3 * A m y - 2 * (A m y + B m y) = 5 * m * y + 2 * y - 1) →
  m = -2/5 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_value_under_condition_independence_condition_l3984_398454


namespace NUMINAMATH_CALUDE_gloria_cabin_theorem_l3984_398497

/-- Represents the problem of calculating Gloria's remaining money after buying a cabin --/
def gloria_cabin_problem (cabin_price cash_on_hand cypress_count pine_count maple_count cypress_price pine_price maple_price : ℕ) : Prop :=
  let total_from_trees := cypress_count * cypress_price + pine_count * pine_price + maple_count * maple_price
  let total_amount := total_from_trees + cash_on_hand
  let money_left := total_amount - cabin_price
  money_left = 350

/-- Theorem stating that Gloria will have $350 left after buying the cabin --/
theorem gloria_cabin_theorem : gloria_cabin_problem 129000 150 20 600 24 100 200 300 := by
  sorry

end NUMINAMATH_CALUDE_gloria_cabin_theorem_l3984_398497


namespace NUMINAMATH_CALUDE_decimal_addition_l3984_398471

theorem decimal_addition : (4.358 + 3.892 : ℝ) = 8.250 := by sorry

end NUMINAMATH_CALUDE_decimal_addition_l3984_398471


namespace NUMINAMATH_CALUDE_count_012_in_base3_string_l3984_398447

/-- Represents a base-3 digit -/
inductive Base3Digit
| Zero
| One
| Two

/-- Represents a number in base-3 -/
def Base3Number := List Base3Digit

/-- Converts an integer to its base-3 representation -/
def toBase3 (n : Nat) : Base3Number :=
  sorry

/-- Removes leading zeros from a base-3 number -/
def removeLeadingZeros (n : Base3Number) : Base3Number :=
  sorry

/-- Joins a list of base-3 numbers into a single list of digits -/
def joinBase3Numbers (numbers : List Base3Number) : List Base3Digit :=
  sorry

/-- Counts occurrences of the substring "012" in a list of base-3 digits -/
def count012Occurrences (digits : List Base3Digit) : Nat :=
  sorry

/-- The main theorem -/
theorem count_012_in_base3_string :
  let numbers := List.range 729
  let base3Numbers := numbers.map (fun n => removeLeadingZeros (toBase3 (n + 1)))
  let joinedDigits := joinBase3Numbers base3Numbers
  count012Occurrences joinedDigits = 148 :=
sorry

end NUMINAMATH_CALUDE_count_012_in_base3_string_l3984_398447


namespace NUMINAMATH_CALUDE_parking_space_area_l3984_398458

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  length : ℝ
  width : ℝ
  painted_sides_sum : ℝ
  unpainted_side_length : ℝ

/-- The area of a parking space -/
def area (p : ParkingSpace) : ℝ := p.length * p.width

/-- Theorem: The area of a specific parking space is 126 square feet -/
theorem parking_space_area (p : ParkingSpace) 
  (h1 : p.unpainted_side_length = 9)
  (h2 : p.painted_sides_sum = 37)
  (h3 : p.length = p.unpainted_side_length)
  (h4 : p.painted_sides_sum = p.length + 2 * p.width) :
  area p = 126 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_area_l3984_398458


namespace NUMINAMATH_CALUDE_larger_number_is_eight_l3984_398463

theorem larger_number_is_eight (x y : ℕ) (h1 : x * y = 24) (h2 : x + y = 11) : 
  max x y = 8 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_eight_l3984_398463


namespace NUMINAMATH_CALUDE_cubic_equation_consequence_l3984_398445

theorem cubic_equation_consequence (y : ℝ) (h : y^3 - 3*y = 9) : 
  y^5 - 10*y^2 = -y^2 + 9*y + 27 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_consequence_l3984_398445


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3984_398491

/-- Given a line l: x + 2y + p = 0 (p ∈ ℝ), prove that 2x - y - 1 = 0 is the equation of the line
    passing through the point P(2,3) and perpendicular to l. -/
theorem perpendicular_line_equation (p : ℝ) :
  let l : ℝ → ℝ → Prop := fun x y ↦ x + 2 * y + p = 0
  let perpendicular_line : ℝ → ℝ → Prop := fun x y ↦ 2 * x - y - 1 = 0
  (∀ x y, l x y → (perpendicular_line x y → False) → False) ∧
  perpendicular_line 2 3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3984_398491


namespace NUMINAMATH_CALUDE_egg_carton_problem_l3984_398437

theorem egg_carton_problem (abigail_eggs beatrice_eggs carson_eggs carton_size : ℕ) 
  (h1 : abigail_eggs = 48)
  (h2 : beatrice_eggs = 63)
  (h3 : carson_eggs = 27)
  (h4 : carton_size = 15) :
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  (total_eggs % carton_size = 3) ∧ (total_eggs / carton_size = 9) := by
  sorry

end NUMINAMATH_CALUDE_egg_carton_problem_l3984_398437


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l3984_398401

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 4*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 4

-- Theorem statement
theorem tangent_slope_at_point_one :
  f 1 = -3 ∧ f' 1 = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l3984_398401


namespace NUMINAMATH_CALUDE_shortest_path_on_cube_is_four_l3984_398439

/-- The shortest path on the surface of a regular cube with edge length 2,
    from one corner to the opposite corner. -/
def shortest_path_on_cube : ℝ := 4

/-- Proof that the shortest path on the surface of a regular cube with edge length 2,
    from one corner to the opposite corner, is equal to 4. -/
theorem shortest_path_on_cube_is_four :
  shortest_path_on_cube = 4 := by sorry

end NUMINAMATH_CALUDE_shortest_path_on_cube_is_four_l3984_398439


namespace NUMINAMATH_CALUDE_intersection_point_theorem_l3984_398413

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- A point in 3D space -/
structure Point

/-- The intersection of two planes is a line -/
def plane_intersection (p1 p2 : Plane) : Line :=
  sorry

/-- A point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- A point lies on a plane -/
def point_on_plane (p : Point) (pl : Plane) : Prop :=
  sorry

theorem intersection_point_theorem 
  (α β γ : Plane) 
  (M : Point) :
  let a := plane_intersection α β
  let b := plane_intersection α γ
  let c := plane_intersection β γ
  (point_on_line M a ∧ point_on_line M b) → 
  point_on_line M c :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_theorem_l3984_398413


namespace NUMINAMATH_CALUDE_camel_cost_proof_l3984_398426

/-- The cost of a camel in rupees -/
def camel_cost : ℝ := 5200

/-- The cost of a horse in rupees -/
def horse_cost : ℝ := 2166.67

/-- The cost of an ox in rupees -/
def ox_cost : ℝ := 8666.67

/-- The cost of an elephant in rupees -/
def elephant_cost : ℝ := 13000

theorem camel_cost_proof :
  (10 * camel_cost = 24 * horse_cost) ∧
  (16 * horse_cost = 4 * ox_cost) ∧
  (6 * ox_cost = 4 * elephant_cost) ∧
  (10 * elephant_cost = 130000) →
  camel_cost = 5200 := by
sorry

end NUMINAMATH_CALUDE_camel_cost_proof_l3984_398426


namespace NUMINAMATH_CALUDE_last_integer_in_sequence_l3984_398435

def sequence_term (n : ℕ) : ℚ :=
  (1024000 : ℚ) / (4 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem last_integer_in_sequence :
  ∃ (n : ℕ), (is_integer (sequence_term n) ∧ sequence_term n = 250) ∧
             ∀ (m : ℕ), m > n → ¬ is_integer (sequence_term m) :=
by sorry

end NUMINAMATH_CALUDE_last_integer_in_sequence_l3984_398435


namespace NUMINAMATH_CALUDE_class_average_theorem_l3984_398488

theorem class_average_theorem (total_students : ℕ) 
                               (excluded_students : ℕ) 
                               (excluded_average : ℝ) 
                               (remaining_average : ℝ) : 
  total_students = 56 →
  excluded_students = 8 →
  excluded_average = 20 →
  remaining_average = 90 →
  (total_students * (total_students * remaining_average - excluded_students * remaining_average + excluded_students * excluded_average)) / 
  (total_students * total_students) = 80 := by
sorry

end NUMINAMATH_CALUDE_class_average_theorem_l3984_398488


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l3984_398466

/-- Given two vectors a and b in ℝ², where a = (6, 3) and b = (sinθ, cosθ),
    if a is parallel to b, then sin2θ - 2cos²θ = 2/5 -/
theorem parallel_vectors_trig_identity (θ : ℝ) :
  let a : Fin 2 → ℝ := ![6, 3]
  let b : Fin 2 → ℝ := ![Real.sin θ, Real.cos θ]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l3984_398466


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3984_398403

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3984_398403


namespace NUMINAMATH_CALUDE_product_equals_square_l3984_398465

theorem product_equals_square : 500 * 3986 * 0.3986 * 20 = (3986 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l3984_398465


namespace NUMINAMATH_CALUDE_exists_f_1984_eq_A_l3984_398442

-- Define the function property
def satisfies_property (f : ℤ → ℝ) : Prop :=
  ∀ x y : ℤ, f (x - y^2) = f x + (y^2 - 2*x) * f y

-- State the theorem
theorem exists_f_1984_eq_A (A : ℝ) :
  ∃ f : ℤ → ℝ, satisfies_property f ∧ f 1984 = A :=
sorry

end NUMINAMATH_CALUDE_exists_f_1984_eq_A_l3984_398442


namespace NUMINAMATH_CALUDE_leftover_coin_value_l3984_398408

def quarters_per_roll : ℕ := 45
def dimes_per_roll : ℕ := 55
def james_quarters : ℕ := 95
def james_dimes : ℕ := 173
def lindsay_quarters : ℕ := 140
def lindsay_dimes : ℕ := 285
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftover_coin_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  let leftover_value := (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value
  leftover_value = 5.30 := by sorry

end NUMINAMATH_CALUDE_leftover_coin_value_l3984_398408


namespace NUMINAMATH_CALUDE_max_point_inequality_l3984_398474

noncomputable section

variables (a : ℝ) (x₁ : ℝ)

def f (x : ℝ) : ℝ := Real.log x - 2 * a * x

def g (x : ℝ) : ℝ := f a x + (1/2) * x^2

theorem max_point_inequality (h1 : x₁ > 0) (h2 : IsLocalMax (g a) x₁) :
  (Real.log x₁) / x₁ + 1 / x₁^2 > a :=
sorry

end

end NUMINAMATH_CALUDE_max_point_inequality_l3984_398474


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3984_398406

theorem digit_sum_problem (w x y z : ℕ) : 
  w ≤ 9 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧  -- digits are between 0 and 9
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧  -- all digits are different
  y + w = 10 ∧  -- sum in ones place
  x + y + 1 = 10 ∧  -- sum in tens place with carry
  w + z + 1 = 11  -- sum in hundreds place with carry
  →
  w + x + y + z = 23 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3984_398406


namespace NUMINAMATH_CALUDE_certain_amount_of_seconds_l3984_398495

/-- Given that 12 is to a certain amount of seconds as 16 is to 8 minutes,
    prove that the certain amount of seconds is 360. -/
theorem certain_amount_of_seconds : ∃ X : ℝ, 
  (12 / X = 16 / (8 * 60)) ∧ (X = 360) := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_of_seconds_l3984_398495


namespace NUMINAMATH_CALUDE_sarah_bowling_score_l3984_398487

theorem sarah_bowling_score (greg_score sarah_score : ℕ) : 
  sarah_score = greg_score + 30 →
  (sarah_score + greg_score) / 2 = 95 →
  sarah_score = 110 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bowling_score_l3984_398487


namespace NUMINAMATH_CALUDE_distance_difference_l3984_398481

/-- The distance biked by Bjorn after six hours -/
def bjorn_distance : ℕ := 75

/-- The distance biked by Alberto after six hours -/
def alberto_distance : ℕ := 105

/-- Alberto bikes faster than Bjorn -/
axiom alberto_faster : alberto_distance > bjorn_distance

/-- The difference in distance biked between Alberto and Bjorn after six hours is 30 miles -/
theorem distance_difference : alberto_distance - bjorn_distance = 30 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3984_398481


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3984_398422

-- Define the circles
def circle1 (x y a : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 9 = 0
def circle2 (x y b : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

-- Define the theorem
theorem min_value_of_expression (a b : ℝ) 
  (h1 : ∃ x y, circle1 x y a)
  (h2 : ∃ x y, circle2 x y b)
  (h3 : ∃ t1 t2 t3 : ℝ × ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    (∀ x y, circle1 x y a → (t1.1 * x + t1.2 * y = 1 ∨ t2.1 * x + t2.2 * y = 1 ∨ t3.1 * x + t3.2 * y = 1)) ∧
    (∀ x y, circle2 x y b → (t1.1 * x + t1.2 * y = 1 ∨ t2.1 * x + t2.2 * y = 1 ∨ t3.1 * x + t3.2 * y = 1)))
  (h4 : a ≠ 0)
  (h5 : b ≠ 0) :
  ∃ m : ℝ, m = 1 ∧ ∀ a b : ℝ, a ≠ 0 → b ≠ 0 → 4 / a^2 + 1 / b^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3984_398422


namespace NUMINAMATH_CALUDE_male_response_rate_change_l3984_398499

/- Define the survey data structure -/
structure SurveyData where
  totalCustomers : ℕ
  malePercentage : ℚ
  femalePercentage : ℚ
  totalResponses : ℕ
  maleResponsePercentage : ℚ
  femaleResponsePercentage : ℚ

/- Define the surveys -/
def initialSurvey : SurveyData :=
  { totalCustomers := 100
  , malePercentage := 60 / 100
  , femalePercentage := 40 / 100
  , totalResponses := 10
  , maleResponsePercentage := 50 / 100
  , femaleResponsePercentage := 50 / 100 }

def finalSurvey : SurveyData :=
  { totalCustomers := 90
  , malePercentage := 50 / 100
  , femalePercentage := 50 / 100
  , totalResponses := 27
  , maleResponsePercentage := 30 / 100
  , femaleResponsePercentage := 70 / 100 }

/- Calculate male response rate -/
def maleResponseRate (survey : SurveyData) : ℚ :=
  (survey.maleResponsePercentage * survey.totalResponses) /
  (survey.malePercentage * survey.totalCustomers)

/- Calculate percentage change -/
def percentageChange (initial : ℚ) (final : ℚ) : ℚ :=
  ((final - initial) / initial) * 100

/- Theorem statement -/
theorem male_response_rate_change :
  percentageChange (maleResponseRate initialSurvey) (maleResponseRate finalSurvey) = 113.4 := by
  sorry

end NUMINAMATH_CALUDE_male_response_rate_change_l3984_398499


namespace NUMINAMATH_CALUDE_original_people_count_l3984_398485

/-- The original number of people in the room. -/
def original_people : ℕ := 36

/-- The fraction of people who left initially. -/
def fraction_left : ℚ := 1 / 3

/-- The fraction of remaining people who started dancing. -/
def fraction_dancing : ℚ := 1 / 4

/-- The number of people who were not dancing. -/
def non_dancing_people : ℕ := 18

theorem original_people_count :
  (original_people : ℚ) * (1 - fraction_left) * (1 - fraction_dancing) = non_dancing_people := by
  sorry

end NUMINAMATH_CALUDE_original_people_count_l3984_398485


namespace NUMINAMATH_CALUDE_cornelia_area_is_17_over_6_l3984_398456

/-- Represents an equiangular octagon with alternating side lengths -/
structure EquiangularOctagon where
  side1 : ℝ
  side2 : ℝ

/-- Represents a self-intersecting octagon formed by connecting alternate vertices of an equiangular octagon -/
structure SelfIntersectingOctagon where
  base : EquiangularOctagon

/-- The area enclosed by a self-intersecting octagon -/
def enclosed_area (octagon : SelfIntersectingOctagon) : ℝ := sorry

/-- The theorem stating that the area enclosed by CORNELIA is 17/6 -/
theorem cornelia_area_is_17_over_6 (caroline : EquiangularOctagon) 
  (cornelia : SelfIntersectingOctagon) (h1 : caroline.side1 = Real.sqrt 2) 
  (h2 : caroline.side2 = 1) (h3 : cornelia.base = caroline) : 
  enclosed_area cornelia = 17 / 6 := by sorry

end NUMINAMATH_CALUDE_cornelia_area_is_17_over_6_l3984_398456


namespace NUMINAMATH_CALUDE_max_value_xyz_expression_l3984_398470

theorem max_value_xyz_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * (x + y + z)) / ((x + z)^2 * (y + z)^2) ≤ (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_xyz_expression_l3984_398470


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3984_398430

theorem more_girls_than_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 345 →
  boys = 138 →
  girls > boys →
  total = girls + boys →
  girls - boys = 69 :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3984_398430


namespace NUMINAMATH_CALUDE_seashells_given_to_mike_l3984_398484

theorem seashells_given_to_mike (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 79)
  (h2 : remaining_seashells = 16) :
  initial_seashells - remaining_seashells = 63 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_mike_l3984_398484


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3984_398494

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, prove that the 12th term is 14. -/
theorem arithmetic_sequence_12th_term
  (seq : ArithmeticSequence)
  (h1 : seq.a 7 + seq.a 9 = 15)
  (h2 : seq.a 4 = 1) :
  seq.a 12 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3984_398494


namespace NUMINAMATH_CALUDE_jane_work_days_jane_solo_days_l3984_398443

theorem jane_work_days (john_days : ℝ) (total_days : ℝ) (jane_stop_days : ℝ) : ℝ :=
  let john_rate := 1 / john_days
  let total_work := 1
  let jane_work_days := total_days - jane_stop_days
  let john_solo_work := john_rate * jane_stop_days
  let combined_work := total_work - john_solo_work
  combined_work / (john_rate + 1 / (total_days - jane_stop_days)) / jane_work_days

theorem jane_solo_days 
  (john_days : ℝ) 
  (total_days : ℝ) 
  (jane_stop_days : ℝ) 
  (h1 : john_days = 20)
  (h2 : total_days = 10)
  (h3 : jane_stop_days = 4)
  : jane_work_days john_days total_days jane_stop_days = 12 := by
  sorry

end NUMINAMATH_CALUDE_jane_work_days_jane_solo_days_l3984_398443


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_6_pow_5_l3984_398410

-- Define the cycle of units digits for powers of 7
def units_cycle : List Nat := [7, 9, 3, 1]

-- Theorem statement
theorem units_digit_of_7_pow_6_pow_5 : 
  ∃ (n : Nat), 7^(6^5) ≡ 7 [ZMOD 10] ∧ n = 7^(6^5) % 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_6_pow_5_l3984_398410


namespace NUMINAMATH_CALUDE_ring_arrangement_correct_l3984_398486

/-- The number of ways to arrange 5 rings out of 9 on 5 fingers -/
def ring_arrangements (total_rings : ℕ) (arranged_rings : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings arranged_rings * Nat.factorial arranged_rings * Nat.choose (total_rings - 1) (fingers - 1)

/-- The correct number of arrangements for 9 rings, 5 arranged, on 5 fingers -/
def correct_arrangement : ℕ := 1905120

/-- Theorem stating that the number of arrangements is correct -/
theorem ring_arrangement_correct :
  ring_arrangements 9 5 5 = correct_arrangement := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangement_correct_l3984_398486


namespace NUMINAMATH_CALUDE_derivative_property_l3984_398476

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem derivative_property (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_property_l3984_398476


namespace NUMINAMATH_CALUDE_sqrt_54_minus_sqrt_6_l3984_398414

theorem sqrt_54_minus_sqrt_6 : Real.sqrt 54 - Real.sqrt 6 = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_54_minus_sqrt_6_l3984_398414


namespace NUMINAMATH_CALUDE_map_distance_calculation_map_distance_proof_l3984_398450

theorem map_distance_calculation (scale_map : Real) (scale_actual : Real) (actual_distance : Real) : Real :=
  let scale_factor := scale_actual / scale_map
  let map_distance := actual_distance / scale_factor
  map_distance

theorem map_distance_proof (h1 : map_distance_calculation 0.4 5.3 848 = 64) : 
  ∃ (d : Real), map_distance_calculation 0.4 5.3 848 = d ∧ d = 64 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_calculation_map_distance_proof_l3984_398450


namespace NUMINAMATH_CALUDE_sales_after_reduction_profit_after_optimal_reduction_l3984_398460

/-- Represents a store's sales and pricing strategy -/
structure Store where
  initial_sales : ℕ
  initial_profit : ℝ
  sales_increase : ℝ
  min_profit : ℝ

/-- Calculates the new sales quantity after a price reduction -/
def new_sales (s : Store) (price_reduction : ℝ) : ℝ :=
  s.initial_sales + s.sales_increase * price_reduction

/-- Calculates the new profit per item after a price reduction -/
def new_profit_per_item (s : Store) (price_reduction : ℝ) : ℝ :=
  s.initial_profit - price_reduction

/-- Calculates the total daily profit after a price reduction -/
def daily_profit (s : Store) (price_reduction : ℝ) : ℝ :=
  new_sales s price_reduction * new_profit_per_item s price_reduction

/-- The store's initial conditions -/
def my_store : Store :=
  { initial_sales := 20
  , initial_profit := 40
  , sales_increase := 2
  , min_profit := 25 }

theorem sales_after_reduction (s : Store) :
  new_sales s 3 = 26 :=
sorry

theorem profit_after_optimal_reduction (s : Store) :
  ∃ (x : ℝ), x = 10 ∧ 
    daily_profit s x = 1200 ∧ 
    new_profit_per_item s x ≥ s.min_profit :=
sorry

end NUMINAMATH_CALUDE_sales_after_reduction_profit_after_optimal_reduction_l3984_398460


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l3984_398432

/-- Proves that given specific interest conditions, if the difference between compound and simple interest is 36, the principal is 3600. -/
theorem interest_difference_implies_principal : 
  let rate : ℝ := 10  -- Interest rate (%)
  let time : ℝ := 2   -- Time period in years
  let diff : ℝ := 36  -- Difference between compound and simple interest
  ∀ principal : ℝ,
    (principal * (1 + rate / 100) ^ time - principal) -  -- Compound interest
    (principal * rate * time / 100) =                    -- Simple interest
    diff →
    principal = 3600 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l3984_398432


namespace NUMINAMATH_CALUDE_ruby_apples_l3984_398429

/-- The number of apples Ruby has initially -/
def initial_apples : ℕ := 63

/-- The number of apples Emily takes away -/
def apples_taken : ℕ := 55

/-- The number of apples Ruby has after Emily takes some away -/
def remaining_apples : ℕ := initial_apples - apples_taken

theorem ruby_apples : remaining_apples = 8 := by
  sorry

end NUMINAMATH_CALUDE_ruby_apples_l3984_398429


namespace NUMINAMATH_CALUDE_circle_tangent_to_lines_l3984_398457

/-- A circle with center (0, k) is tangent to lines y = x, y = -x, y = 10, and y = -4x. -/
theorem circle_tangent_to_lines (k : ℝ) (h : k > 10) :
  let r := 10 * Real.sqrt 34 * (Real.sqrt 2 / (Real.sqrt 2 - 1 / Real.sqrt 17)) - 10 * Real.sqrt 2
  ∃ (circle : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ circle ↔ (x^2 + (y - k)^2 = r^2)) ∧
    (∃ (x₁ y₁ : ℝ), (x₁, y₁) ∈ circle ∧ y₁ = x₁) ∧
    (∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ circle ∧ y₂ = -x₂) ∧
    (∃ (x₃ y₃ : ℝ), (x₃, y₃) ∈ circle ∧ y₃ = 10) ∧
    (∃ (x₄ y₄ : ℝ), (x₄, y₄) ∈ circle ∧ y₄ = -4*x₄) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_lines_l3984_398457


namespace NUMINAMATH_CALUDE_student_average_less_than_true_average_l3984_398451

theorem student_average_less_than_true_average 
  (w x y z : ℝ) (hw : w < x) (hx : x < y) (hy : y < z) :
  (w + x + (y + z) / 2) / 3 < (w + x + y + z) / 4 := by
sorry

end NUMINAMATH_CALUDE_student_average_less_than_true_average_l3984_398451


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l3984_398407

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of socks in the drawer -/
def total_socks : ℕ := 8

/-- The number of socks to be chosen -/
def socks_to_choose : ℕ := 4

/-- The number of non-red socks -/
def non_red_socks : ℕ := 7

theorem sock_selection_theorem :
  choose total_socks socks_to_choose - choose non_red_socks socks_to_choose = 35 := by sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l3984_398407


namespace NUMINAMATH_CALUDE_simplify_expression_l3984_398440

theorem simplify_expression (x : ℝ) :
  Real.sqrt (x^6 + 3*x^4 + 2*x^2) = |x| * Real.sqrt ((x^2 + 1) * (x^2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3984_398440


namespace NUMINAMATH_CALUDE_determinant_of_cubic_roots_l3984_398425

theorem determinant_of_cubic_roots (s p q : ℝ) (a b c : ℝ) : 
  (a^3 - s*a^2 + p*a + q = 0) →
  (b^3 - s*b^2 + p*b + q = 0) →
  (c^3 - s*c^2 + p*c + q = 0) →
  (a + b + c = s) →
  (a*b + b*c + a*c = p) →
  (a*b*c = -q) →
  Matrix.det !![1 + a, 1, 1; 1, 1 + b, 1; 1, 1, 1 + c] = p + 3*s := by
sorry

end NUMINAMATH_CALUDE_determinant_of_cubic_roots_l3984_398425


namespace NUMINAMATH_CALUDE_shaded_area_equals_1150_l3984_398441

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The main theorem stating the area of the shaded region -/
theorem shaded_area_equals_1150 :
  let square_side : ℝ := 40
  let p1 : Point := ⟨0, 0⟩
  let p2 : Point := ⟨20, 0⟩
  let p3 : Point := ⟨40, 30⟩
  let p4 : Point := ⟨40, 40⟩
  let p5 : Point := ⟨10, 40⟩
  let p6 : Point := ⟨0, 10⟩
  let square_area := square_side * square_side
  let triangle1_area := triangleArea p2 ⟨40, 0⟩ p3
  let triangle2_area := triangleArea p6 ⟨0, 40⟩ p5
  square_area - (triangle1_area + triangle2_area) = 1150 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_equals_1150_l3984_398441


namespace NUMINAMATH_CALUDE_sector_central_angle_l3984_398483

/-- Given a circular sector with area 4 and arc length 4, its central angle is 2 radians. -/
theorem sector_central_angle (area : ℝ) (arc_length : ℝ) (radius : ℝ) (angle : ℝ) :
  area = 4 →
  arc_length = 4 →
  area = (1 / 2) * radius * arc_length →
  arc_length = radius * angle →
  angle = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3984_398483


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3984_398404

/-- Given that 3y varies inversely as the square of x, prove that y = 5/9 when x = 6, 
    given the initial condition y = 5 when x = 2 -/
theorem inverse_variation_problem (k : ℝ) :
  (∀ x y : ℝ, x ≠ 0 → 3 * y = k / (x^2)) →  -- Inverse variation relationship
  (3 * 5 = k / (2^2)) →                     -- Initial condition
  ∃ y : ℝ, 3 * y = k / (6^2) ∧ y = 5/9      -- Conclusion for x = 6
  := by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3984_398404


namespace NUMINAMATH_CALUDE_triangle_properties_l3984_398417

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle with specific properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.c - t.b) / t.a = Real.cos t.B / Real.cos t.A) 
  (h2 : t.a = 2 * Real.sqrt 5) : 
  t.A = π / 3 ∧ 
  (∃ (S : ℝ), S = 5 * Real.sqrt 3 ∧ ∀ (area : ℝ), area ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3984_398417


namespace NUMINAMATH_CALUDE_game_points_percentage_l3984_398400

theorem game_points_percentage (samanta mark eric : ℕ) : 
  samanta = mark + 8 →
  eric = 6 →
  samanta + mark + eric = 32 →
  (mark - eric : ℚ) / eric * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_game_points_percentage_l3984_398400


namespace NUMINAMATH_CALUDE_helen_cookies_l3984_398431

/-- The number of chocolate chip cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 1081 - 554

/-- The total number of chocolate chip cookies Helen baked -/
def total_cookies : ℕ := 1081

/-- The number of chocolate chip cookies Helen baked this morning -/
def cookies_this_morning : ℕ := 554

theorem helen_cookies : cookies_yesterday = 527 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l3984_398431


namespace NUMINAMATH_CALUDE_sin_neg_pi_half_l3984_398427

theorem sin_neg_pi_half : Real.sin (-π / 2) = -1 := by sorry

end NUMINAMATH_CALUDE_sin_neg_pi_half_l3984_398427


namespace NUMINAMATH_CALUDE_function_symmetry_l3984_398448

/-- Given a function f(x) = a*sin(x) - b*cos(x) where a and b are constants, a ≠ 0,
    and f(x) attains its maximum value at x = π/4, prove that g(x) = f(x + π/4)
    is an even function and its graph is symmetric about the point (3π/2, 0). -/
theorem function_symmetry (a b : ℝ) (h_a : a ≠ 0) :
  let f := fun (x : ℝ) ↦ a * Real.sin x - b * Real.cos x
  let g := fun (x : ℝ) ↦ f (x + π/4)
  (∀ x, f x ≤ f (π/4)) →
  (∀ x, g x = g (-x)) ∧
  (∀ x, g (x + 3*π/2) = -g (-x + 3*π/2)) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l3984_398448


namespace NUMINAMATH_CALUDE_earl_stuffing_rate_l3984_398490

-- Define Earl's stuffing rate (envelopes per minute)
def earl_rate : ℝ := sorry

-- Define Ellen's stuffing rate (envelopes per minute)
def ellen_rate : ℝ := sorry

-- Condition 1: Ellen's rate is 2/3 of Earl's rate
axiom rate_relation : ellen_rate = (2/3) * earl_rate

-- Condition 2: Together they stuff 360 envelopes in 6 minutes
axiom combined_rate : earl_rate + ellen_rate = 360 / 6

-- Theorem to prove
theorem earl_stuffing_rate : earl_rate = 36 := by sorry

end NUMINAMATH_CALUDE_earl_stuffing_rate_l3984_398490


namespace NUMINAMATH_CALUDE_garden_to_land_ratio_l3984_398405

/-- A rectangle with width 3/5 of its length -/
structure Rectangle where
  length : ℝ
  width : ℝ
  width_prop : width = (3/5) * length

theorem garden_to_land_ratio (land garden : Rectangle) : 
  (garden.length * garden.width) / (land.length * land.width) = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_garden_to_land_ratio_l3984_398405


namespace NUMINAMATH_CALUDE_largest_n_squared_sum_largest_n_exists_largest_n_is_three_l3984_398402

theorem largest_n_squared_sum (n : ℕ+) : 
  (∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18) →
  n ≤ 3 :=
by sorry

theorem largest_n_exists : 
  ∃ (x y z : ℕ+), 3^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18 :=
by sorry

theorem largest_n_is_three : 
  (∃ (n : ℕ+), (∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18) ∧
  (∀ (m : ℕ+), (∃ (a b c : ℕ+), m^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 6*a + 6*b + 6*c - 18) → m ≤ n)) →
  (∃ (x y z : ℕ+), 3^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_squared_sum_largest_n_exists_largest_n_is_three_l3984_398402


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3984_398477

theorem polynomial_factorization (x : ℤ) :
  3 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (3 * x^2 + 58 * x + 231) * (x + 7) * (x + 11) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3984_398477


namespace NUMINAMATH_CALUDE_union_A_B_intersect_complement_A_B_l3984_398482

/-- The set A -/
def A : Set ℝ := {x | x < -5 ∨ x > 1}

/-- The set B -/
def B : Set ℝ := {x | -4 < x ∧ x < 3}

/-- Theorem: The union of A and B -/
theorem union_A_B : A ∪ B = {x : ℝ | x < -5 ∨ x > -4} := by sorry

/-- Theorem: The intersection of the complement of A and B -/
theorem intersect_complement_A_B : (Aᶜ) ∩ B = {x : ℝ | -4 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersect_complement_A_B_l3984_398482


namespace NUMINAMATH_CALUDE_floor_abs_sum_equality_l3984_398438

theorem floor_abs_sum_equality : ⌊|(-7.3 : ℝ)|⌋ + |⌊(-7.3 : ℝ)⌋| = 15 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_equality_l3984_398438


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_nine_l3984_398459

theorem missing_digit_divisible_by_nine :
  ∀ x : ℕ,
  x < 10 →
  (13507 + 100 * x) % 9 = 0 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_nine_l3984_398459


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3984_398492

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≥ 10) →
  (∀ x ∈ Set.Icc 1 3, x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, x^2 - a ≤ 0) → (a ≥ 10)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3984_398492


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3984_398424

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 36 ≤ 0 → n ≤ m) ∧ (n^2 - 13*n + 36 ≤ 0) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3984_398424


namespace NUMINAMATH_CALUDE_team_point_difference_l3984_398418

/-- The difference in points between two teams -/
def point_difference (beth_score jan_score judy_score angel_score : ℕ) : ℕ :=
  (beth_score + jan_score) - (judy_score + angel_score)

/-- Theorem stating the point difference between the two teams -/
theorem team_point_difference :
  point_difference 12 10 8 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_team_point_difference_l3984_398418


namespace NUMINAMATH_CALUDE_greatest_number_odd_factors_under_200_l3984_398469

theorem greatest_number_odd_factors_under_200 :
  ∃ (n : ℕ), n < 200 ∧ n = 196 ∧ 
  (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧
  (∃ k : ℕ, n = k^2) :=
sorry

end NUMINAMATH_CALUDE_greatest_number_odd_factors_under_200_l3984_398469


namespace NUMINAMATH_CALUDE_candied_yams_ratio_l3984_398415

/-- The ratio of shoppers who buy candied yams to the total number of shoppers -/
theorem candied_yams_ratio 
  (packages_per_box : ℕ) 
  (boxes_ordered : ℕ) 
  (total_shoppers : ℕ) 
  (h1 : packages_per_box = 25)
  (h2 : boxes_ordered = 5)
  (h3 : total_shoppers = 375) : 
  (boxes_ordered * packages_per_box : ℚ) / total_shoppers = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_candied_yams_ratio_l3984_398415


namespace NUMINAMATH_CALUDE_exists_special_sequence_l3984_398412

/-- A sequence of natural numbers with the property that all natural numbers
    appear exactly once as differences between its members. -/
def special_sequence : Set ℕ → Prop :=
  λ S => (∀ n : ℕ, ∃! (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a > b ∧ a - b = n) ∧
         (∀ a : ℕ, a ∈ S → ∃ b : ℕ, b > a ∧ b ∈ S)

/-- Theorem stating the existence of a special sequence of natural numbers. -/
theorem exists_special_sequence : ∃ S : Set ℕ, special_sequence S :=
  sorry


end NUMINAMATH_CALUDE_exists_special_sequence_l3984_398412


namespace NUMINAMATH_CALUDE_neg_p_necessary_not_sufficient_for_neg_q_l3984_398449

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 1 > 0
def q (x : ℝ) : Prop := (x + 1) * (x - 2) > 0

-- Define the relationship between ¬p and ¬q
theorem neg_p_necessary_not_sufficient_for_neg_q :
  (∀ x, ¬(q x) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) := by
  sorry

end NUMINAMATH_CALUDE_neg_p_necessary_not_sufficient_for_neg_q_l3984_398449


namespace NUMINAMATH_CALUDE_externally_tangent_circles_m_value_l3984_398419

/-- A circle in the 2D plane defined by its equation coefficients -/
structure Circle where
  a : ℝ -- coefficient of x^2
  b : ℝ -- coefficient of y^2
  c : ℝ -- coefficient of x
  d : ℝ -- coefficient of y
  e : ℝ -- constant term

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let center1 := (- c1.c / (2 * c1.a), - c1.d / (2 * c1.b))
  let center2 := (- c2.c / (2 * c2.a), - c2.d / (2 * c2.b))
  let radius1 := Real.sqrt ((c1.c^2 / (4 * c1.a^2) + c1.d^2 / (4 * c1.b^2) - c1.e / c1.a))
  let radius2 := Real.sqrt ((c2.c^2 / (4 * c2.a^2) + c2.d^2 / (4 * c2.b^2) - c2.e / c2.a))
  let distance := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance = radius1 + radius2

/-- The main theorem -/
theorem externally_tangent_circles_m_value :
  ∀ m : ℝ,
  let c1 : Circle := { a := 1, b := 1, c := -2, d := -4, e := m }
  let c2 : Circle := { a := 1, b := 1, c := -8, d := -12, e := 36 }
  are_externally_tangent c1 c2 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_m_value_l3984_398419


namespace NUMINAMATH_CALUDE_square_area_with_diagonal_l3984_398468

/-- Given a square with side length s and diagonal length s + 1, its area is 3 + 2√2 -/
theorem square_area_with_diagonal (s : ℝ) (h : s * Real.sqrt 2 = s + 1) :
  s^2 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_diagonal_l3984_398468


namespace NUMINAMATH_CALUDE_billy_crayons_l3984_398428

theorem billy_crayons (jane_crayons : ℝ) (total_crayons : ℕ) 
  (h1 : jane_crayons = 52.0) 
  (h2 : total_crayons = 114) : 
  ↑total_crayons - jane_crayons = 62 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_l3984_398428


namespace NUMINAMATH_CALUDE_non_monotonic_range_l3984_398496

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem non_monotonic_range (a : ℝ) : 
  (¬ is_monotonic (f a)) ↔ 
  (0 < a ∧ a < 1/7) ∨ (1/3 ≤ a ∧ a < 1) ∨ (a > 1) :=
sorry

end NUMINAMATH_CALUDE_non_monotonic_range_l3984_398496


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3984_398411

theorem purely_imaginary_complex_number (m : ℝ) :
  (m^2 - m : ℂ) + m * I = (0 : ℂ) + (m : ℂ) * I → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3984_398411


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l3984_398478

theorem recurring_decimal_to_fraction :
  ∃ (x : ℚ), x = 4 + 36 / 99 ∧ x = 144 / 33 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l3984_398478


namespace NUMINAMATH_CALUDE_twenty_four_divides_Q_largest_divisor_of_Q_l3984_398480

/-- The product of three consecutive positive even integers -/
def Q (n : ℕ) : ℕ := (2*n) * (2*n + 2) * (2*n + 4)

/-- 24 divides Q for all positive n -/
theorem twenty_four_divides_Q (n : ℕ) (h : n > 0) : 24 ∣ Q n := by sorry

/-- 24 is the largest integer that divides Q for all positive n -/
theorem largest_divisor_of_Q :
  ∀ d : ℕ, (∀ n : ℕ, n > 0 → d ∣ Q n) → d ≤ 24 := by sorry

end NUMINAMATH_CALUDE_twenty_four_divides_Q_largest_divisor_of_Q_l3984_398480


namespace NUMINAMATH_CALUDE_floor_sum_example_l3984_398453

theorem floor_sum_example : ⌊(12.7 : ℝ)⌋ + ⌊(-12.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3984_398453


namespace NUMINAMATH_CALUDE_coat_drive_total_l3984_398436

theorem coat_drive_total (high_school_coats elementary_school_coats : ℕ) 
  (h1 : high_school_coats = 6922)
  (h2 : elementary_school_coats = 2515) :
  high_school_coats + elementary_school_coats = 9437 :=
by sorry

end NUMINAMATH_CALUDE_coat_drive_total_l3984_398436


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l3984_398420

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) :
  ∃ (m : ℝ), m = -1 ∧ ∀ x, (8 * x^2 + 10 * x + 6 = 2) → (3 * x + 2 ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l3984_398420


namespace NUMINAMATH_CALUDE_multiple_of_power_minus_one_l3984_398433

theorem multiple_of_power_minus_one (a b c : ℕ) :
  (∃ k : ℤ, 2^a + 2^b + 1 = k * (2^c - 1)) ↔
  ((a = 0 ∧ b = 0 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3)) :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_power_minus_one_l3984_398433


namespace NUMINAMATH_CALUDE_derivative_of_x4_minus_7_l3984_398489

theorem derivative_of_x4_minus_7 (x : ℝ) :
  deriv (fun x => x^4 - 7) x = 4 * x^3 - 7 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_x4_minus_7_l3984_398489


namespace NUMINAMATH_CALUDE_problem_solution_l3984_398475

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem problem_solution :
  (∀ x : ℝ, a = 1 → (p x a ∧ q x) ↔ x ∈ Set.Ioo 2 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ a ∈ Set.Icc 1 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3984_398475


namespace NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l3984_398473

theorem no_real_solutions_for_sqrt_equation :
  ∀ z : ℝ, ¬(Real.sqrt (5 - 4*z) = 7) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l3984_398473


namespace NUMINAMATH_CALUDE_stating_alice_probability_after_two_turns_l3984_398493

/-- The probability that Alice passes the ball to Bob -/
def alice_pass_prob : ℚ := 2/3

/-- The probability that Bob passes the ball to Alice -/
def bob_pass_prob : ℚ := 1/2

/-- The probability that Alice has the ball after two turns -/
def alice_has_ball_after_two_turns : ℚ := 4/9

/-- 
Theorem stating that given the game rules, the probability 
that Alice has the ball after two turns is 4/9 
-/
theorem alice_probability_after_two_turns : 
  alice_has_ball_after_two_turns = 
    (alice_pass_prob * bob_pass_prob) + ((1 - alice_pass_prob) * (1 - alice_pass_prob)) := by
  sorry

end NUMINAMATH_CALUDE_stating_alice_probability_after_two_turns_l3984_398493


namespace NUMINAMATH_CALUDE_volume_of_special_parallelepiped_l3984_398462

/-- A rectangular parallelepiped with specific properties -/
structure RectangularParallelepiped where
  /-- Side length of the square face -/
  a : ℝ
  /-- Height perpendicular to the square face -/
  b : ℝ
  /-- The diagonal length is 1 -/
  diagonal_eq_one : 2 * a^2 + b^2 = 1
  /-- The surface area is 1 -/
  surface_area_eq_one : 4 * a * b + 2 * a^2 = 1
  /-- Ensure a and b are positive -/
  a_pos : 0 < a
  b_pos : 0 < b

/-- The volume of a rectangular parallelepiped with the given properties is √2/27 -/
theorem volume_of_special_parallelepiped (p : RectangularParallelepiped) :
  p.a^2 * p.b = Real.sqrt 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_special_parallelepiped_l3984_398462


namespace NUMINAMATH_CALUDE_min_value_theorem_l3984_398472

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 20 ∧
  ((x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) = 20 ↔ x = 3 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3984_398472


namespace NUMINAMATH_CALUDE_solution_set_for_a_3_min_value_and_range_l3984_398455

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part 1
theorem solution_set_for_a_3 :
  {x : ℝ | f 3 x ≤ 6} = {x : ℝ | 0 ≤ x ∧ x ≤ 3} :=
sorry

-- Part 2
theorem min_value_and_range :
  (∀ x : ℝ, f a x + g x ≥ 3) ↔ a ∈ Set.Ici 2 :=
sorry

#check solution_set_for_a_3
#check min_value_and_range

end NUMINAMATH_CALUDE_solution_set_for_a_3_min_value_and_range_l3984_398455


namespace NUMINAMATH_CALUDE_boulevard_painting_cost_l3984_398498

/-- Represents a side of the boulevard with house numbers -/
structure BoulevardSide where
  start : ℕ
  diff : ℕ
  count : ℕ

/-- Calculates the sum of digits for all numbers in an arithmetic sequence -/
def sumOfDigits (side : BoulevardSide) : ℕ :=
  sorry

/-- The total cost of painting house numbers on both sides of the boulevard -/
def totalCost (eastSide westSide : BoulevardSide) : ℕ :=
  sumOfDigits eastSide + sumOfDigits westSide

theorem boulevard_painting_cost :
  let eastSide : BoulevardSide := { start := 5, diff := 7, count := 25 }
  let westSide : BoulevardSide := { start := 2, diff := 5, count := 25 }
  totalCost eastSide westSide = 113 :=
sorry

end NUMINAMATH_CALUDE_boulevard_painting_cost_l3984_398498


namespace NUMINAMATH_CALUDE_point_b_coordinates_l3984_398461

/-- Given point A (-1, 5) and vector a (2, 3), if vector AB = 3 * vector a, 
    then the coordinates of point B are (5, 14). -/
theorem point_b_coordinates 
  (A : ℝ × ℝ) 
  (a : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (h1 : A = (-1, 5)) 
  (h2 : a = (2, 3)) 
  (h3 : B.1 - A.1 = 3 * a.1 ∧ B.2 - A.2 = 3 * a.2) : 
  B = (5, 14) := by
sorry


end NUMINAMATH_CALUDE_point_b_coordinates_l3984_398461


namespace NUMINAMATH_CALUDE_work_completion_time_l3984_398467

/-- Given workers A, B, and C, where:
  * A can complete the work in 8 days
  * B can complete the work in 16 days
  * A, B, and C together can complete the work in 4 days
  Prove that C alone can complete the work in 16 days -/
theorem work_completion_time (a b c : ℝ) 
  (ha : a = 1 / 8)  -- A's work rate per day
  (hb : b = 1 / 16) -- B's work rate per day
  (habc : a + b + c = 1 / 4) -- A, B, and C's combined work rate per day
  : c = 1 / 16 := by  -- C's work rate per day (equivalent to completing in 16 days)
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3984_398467


namespace NUMINAMATH_CALUDE_negation_equivalence_l3984_398452

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ (∀ x : ℝ, x > 0 → x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3984_398452


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3984_398464

def is_valid_representation (n : ℕ) (a b : ℕ) : Prop :=
  a > 2 ∧ b > 2 ∧ 2 * a + 1 = n ∧ b + 2 = n

theorem smallest_dual_base_representation : 
  (∃ (a b : ℕ), is_valid_representation 7 a b) ∧ 
  (∀ (n : ℕ), n < 7 → ¬∃ (a b : ℕ), is_valid_representation n a b) :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3984_398464


namespace NUMINAMATH_CALUDE_salary_calculation_l3984_398434

/-- Prove that if a salary is first increased by 10% and then decreased by 5%,
    resulting in Rs. 4180, the original salary was Rs. 4000. -/
theorem salary_calculation (original : ℝ) : 
  (original * 1.1 * 0.95 = 4180) → original = 4000 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l3984_398434


namespace NUMINAMATH_CALUDE_washington_dc_trip_cost_l3984_398423

/-- Calculates the total cost per person for a group trip to Washington D.C. -/
theorem washington_dc_trip_cost 
  (num_friends : ℕ)
  (airfare_hotel_cost : ℚ)
  (food_expenses : ℚ)
  (transportation_expenses : ℚ)
  (smithsonian_tour_cost : ℚ)
  (zoo_entry_fee : ℚ)
  (zoo_spending_allowance : ℚ)
  (river_cruise_cost : ℚ)
  (h1 : num_friends = 15)
  (h2 : airfare_hotel_cost = 13500)
  (h3 : food_expenses = 4500)
  (h4 : transportation_expenses = 3000)
  (h5 : smithsonian_tour_cost = 50)
  (h6 : zoo_entry_fee = 75)
  (h7 : zoo_spending_allowance = 15)
  (h8 : river_cruise_cost = 100) :
  (airfare_hotel_cost + food_expenses + transportation_expenses + 
   num_friends * (smithsonian_tour_cost + zoo_entry_fee + zoo_spending_allowance + river_cruise_cost)) / num_friends = 1640 := by
sorry

end NUMINAMATH_CALUDE_washington_dc_trip_cost_l3984_398423


namespace NUMINAMATH_CALUDE_range_encoding_l3984_398421

/-- Represents a coding scheme for words -/
structure CodeScheme where
  random : Nat
  rand : Nat

/-- Defines the coding for a word given a CodeScheme -/
def encode (scheme : CodeScheme) (word : String) : Nat :=
  sorry

/-- Theorem: Given the coding scheme where 'random' is 123678 and 'rand' is 1236,
    the code for 'range' is 12378 -/
theorem range_encoding (scheme : CodeScheme)
    (h1 : scheme.random = 123678)
    (h2 : scheme.rand = 1236) :
    encode scheme "range" = 12378 :=
  sorry

end NUMINAMATH_CALUDE_range_encoding_l3984_398421
