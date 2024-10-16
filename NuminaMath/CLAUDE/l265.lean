import Mathlib

namespace NUMINAMATH_CALUDE_odd_number_of_odd_sided_faces_l265_26569

-- Define a convex polyhedron
structure ConvexPolyhedron where
  vertices : ℕ
  convex : Bool

-- Define a closed broken line on the polyhedron
structure ClosedBrokenLine where
  polyhedron : ConvexPolyhedron
  passes_all_vertices_once : Bool

-- Define a part of the polyhedron surface
structure SurfacePart where
  polyhedron : ConvexPolyhedron
  broken_line : ClosedBrokenLine
  faces : Finset (Finset ℕ)  -- Each face is represented as a set of its vertices

-- Function to count odd-sided faces in a surface part
def count_odd_sided_faces (part : SurfacePart) : ℕ :=
  (part.faces.filter (λ face => face.card % 2 = 1)).card

-- The main theorem
theorem odd_number_of_odd_sided_faces 
  (poly : ConvexPolyhedron) 
  (line : ClosedBrokenLine) 
  (part : SurfacePart) : 
  poly.vertices = 2003 → 
  poly.convex = true → 
  line.polyhedron = poly → 
  line.passes_all_vertices_once = true → 
  part.polyhedron = poly → 
  part.broken_line = line → 
  count_odd_sided_faces part % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_odd_number_of_odd_sided_faces_l265_26569


namespace NUMINAMATH_CALUDE_water_requirement_l265_26571

/-- The number of households in the village -/
def num_households : ℕ := 10

/-- The total amount of water available in litres -/
def total_water : ℕ := 6000

/-- The number of months the water lasts -/
def num_months : ℕ := 4

/-- The amount of water required per household per month -/
def water_per_household_per_month : ℕ := total_water / (num_households * num_months)

/-- Theorem stating that the water required per household per month is 150 litres -/
theorem water_requirement : water_per_household_per_month = 150 := by
  sorry

end NUMINAMATH_CALUDE_water_requirement_l265_26571


namespace NUMINAMATH_CALUDE_logan_tower_height_l265_26535

/-- The height of the city's water tower in meters -/
def city_tower_height : ℝ := 60

/-- The volume of water the city's water tower can hold in liters -/
def city_tower_volume : ℝ := 150000

/-- The volume of water Logan's miniature water tower can hold in liters -/
def miniature_tower_volume : ℝ := 0.15

/-- The height of Logan's miniature water tower in meters -/
def miniature_tower_height : ℝ := 0.6

/-- Theorem stating that the height of Logan's miniature tower should be 0.6 meters -/
theorem logan_tower_height : miniature_tower_height = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_logan_tower_height_l265_26535


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l265_26572

theorem alcohol_mixture_problem (x : ℝ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : 0.9 * x = 0.54 * (x + 16)) : x = 24 := by
  sorry

#check alcohol_mixture_problem

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l265_26572


namespace NUMINAMATH_CALUDE_renaldo_distance_l265_26534

theorem renaldo_distance :
  ∀ (r : ℝ),
  (r + (1/3 * r + 7) = 27) →
  r = 15 := by
sorry

end NUMINAMATH_CALUDE_renaldo_distance_l265_26534


namespace NUMINAMATH_CALUDE_area_between_squares_l265_26592

/-- The area of the region between two concentric squares -/
theorem area_between_squares (outer_side : ℝ) (inner_side : ℝ) 
  (h_outer : outer_side = 6) 
  (h_inner : inner_side = 4) :
  outer_side ^ 2 - inner_side ^ 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_area_between_squares_l265_26592


namespace NUMINAMATH_CALUDE_inequality_proof_l265_26547

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 / (x + y) + 4 / (y + z) + 9 / (x + z) ≥ 18 / (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l265_26547


namespace NUMINAMATH_CALUDE_cubic_root_sum_l265_26524

theorem cubic_root_sum (d e f : ℕ) (hd : d > 0) (he : e > 0) (hf : f > 0) :
  let x : ℝ := (Real.rpow d (1/3) + Real.rpow e (1/3) + 3) / f
  (27 * x^3 - 15 * x^2 - 9 * x - 3 = 0) →
  d + e + f = 126 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l265_26524


namespace NUMINAMATH_CALUDE_proportional_function_decreases_l265_26512

/-- Proves that for a proportional function y = kx passing through the point (4, -1),
    where k is a non-zero constant, y decreases as x increases. -/
theorem proportional_function_decreases (k : ℝ) (h1 : k ≠ 0) (h2 : k * 4 = -1) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_decreases_l265_26512


namespace NUMINAMATH_CALUDE_average_age_decrease_l265_26582

/-- Proves that replacing a 46-year-old person with a 16-year-old person in a group of 10 decreases the average age by 3 years -/
theorem average_age_decrease (initial_avg : ℝ) : 
  let total_age := 10 * initial_avg
  let new_total_age := total_age - 46 + 16
  let new_avg := new_total_age / 10
  initial_avg - new_avg = 3 := by sorry

end NUMINAMATH_CALUDE_average_age_decrease_l265_26582


namespace NUMINAMATH_CALUDE_triangle_count_4x3_l265_26578

/-- Represents a grid of points in a rectangle --/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a triangle in the grid --/
structure Triangle :=
  (p1 : Nat × Nat)
  (p2 : Nat × Nat)
  (p3 : Nat × Nat)

/-- Function to count the number of unique triangles in a grid with diagonals --/
def countTriangles (g : Grid) : Nat :=
  sorry

/-- Theorem stating that a 4x3 grid with diagonals contains 54 unique triangles --/
theorem triangle_count_4x3 :
  let g : Grid := { rows := 4, cols := 3 }
  countTriangles g = 54 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_4x3_l265_26578


namespace NUMINAMATH_CALUDE_not_all_rationals_repeating_l265_26518

-- Define rational numbers
def Rational : Type := ℚ

-- Define integers
def Integer : Type := ℤ

-- Define repeating decimal
def RepeatingDecimal (x : ℚ) : Prop := sorry

-- Statement that integers are rational numbers
axiom integer_is_rational : Integer → Rational

-- Statement that not all integers are repeating decimals
axiom not_all_integers_repeating : ∃ (n : Integer), ¬(RepeatingDecimal (integer_is_rational n))

-- Theorem to prove
theorem not_all_rationals_repeating : ¬(∀ (q : Rational), RepeatingDecimal q) := by
  sorry

end NUMINAMATH_CALUDE_not_all_rationals_repeating_l265_26518


namespace NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l265_26599

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  day_students : ℕ

/-- Represents the school's student composition -/
structure School where
  initial_boarders : ℕ
  initial_ratio : Ratio
  new_boarders : ℕ

/-- Calculates the new ratio of boarders to day students after new boarders join -/
def new_ratio (school : School) : Ratio :=
  sorry

/-- Theorem stating that the new ratio is 1:2 given the initial conditions -/
theorem new_ratio_is_one_to_two (school : School) 
  (h1 : school.initial_boarders = 120)
  (h2 : school.initial_ratio = Ratio.mk 2 5)
  (h3 : school.new_boarders = 30) :
  new_ratio school = Ratio.mk 1 2 :=
sorry

end NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l265_26599


namespace NUMINAMATH_CALUDE_science_club_problem_l265_26568

theorem science_club_problem (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ)
  (h1 : total = 75)
  (h2 : biology = 42)
  (h3 : chemistry = 38)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_science_club_problem_l265_26568


namespace NUMINAMATH_CALUDE_max_a_for_defined_f_l265_26596

-- Define the function g(x) = |x-2| + |x-a|
def g (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- State the theorem
theorem max_a_for_defined_f :
  (∃ (a_max : ℝ), (∀ (a : ℝ), (∀ (x : ℝ), g a x ≥ 2 * a) → a ≤ a_max) ∧
                  (∀ (x : ℝ), g a_max x ≥ 2 * a_max) ∧
                  a_max = 2/3) :=
sorry

end NUMINAMATH_CALUDE_max_a_for_defined_f_l265_26596


namespace NUMINAMATH_CALUDE_chase_travel_time_l265_26507

/-- Represents the journey from Granville to Salisbury with intermediate stops -/
structure Journey where
  chase_speed : ℝ
  cameron_speed : ℝ
  danielle_speed : ℝ
  chase_scooter_speed : ℝ
  cameron_bike_speed : ℝ
  danielle_time : ℝ

/-- The conditions of the journey -/
def journey_conditions (j : Journey) : Prop :=
  j.cameron_speed = 2 * j.chase_speed ∧
  j.danielle_speed = 3 * j.cameron_speed ∧
  j.cameron_bike_speed = 0.75 * j.cameron_speed ∧
  j.chase_scooter_speed = 1.25 * j.chase_speed ∧
  j.danielle_time = 30

/-- The theorem stating that Chase's travel time is 180 minutes -/
theorem chase_travel_time (j : Journey) 
  (h : journey_conditions j) : 
  (180 : ℝ) * j.chase_speed = j.danielle_speed * j.danielle_time :=
sorry

end NUMINAMATH_CALUDE_chase_travel_time_l265_26507


namespace NUMINAMATH_CALUDE_expression_evaluation_l265_26530

theorem expression_evaluation : 
  1 - (1 / (1 + Real.sqrt 5)) + (1 / (1 - Real.sqrt 5)) = 1 - (Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l265_26530


namespace NUMINAMATH_CALUDE_teachers_survey_l265_26552

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 90)
  (h_heart_trouble : heart_trouble = 50)
  (h_both : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 80 / 3 :=
by sorry

end NUMINAMATH_CALUDE_teachers_survey_l265_26552


namespace NUMINAMATH_CALUDE_tire_circumference_l265_26515

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference
  (revolutions_per_minute : ℝ)
  (car_speed_kmh : ℝ)
  (h1 : revolutions_per_minute = 400)
  (h2 : car_speed_kmh = 48)
  : ∃ (circumference : ℝ), circumference = 2 := by
  sorry

end NUMINAMATH_CALUDE_tire_circumference_l265_26515


namespace NUMINAMATH_CALUDE_polynomial_equality_l265_26550

theorem polynomial_equality (p : ℝ → ℝ) :
  (∀ x, p x + (2*x^6 + 4*x^4 + 6*x^2) = 8*x^4 + 27*x^3 + 33*x^2 + 15*x + 5) →
  (∀ x, p x = -2*x^6 + 4*x^4 + 27*x^3 + 27*x^2 + 15*x + 5) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l265_26550


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l265_26594

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  45 ∣ n^2 ∧ 720 ∣ n^3 ∧ 
  ∀ (m : ℕ), m > 0 → 45 ∣ m^2 → 720 ∣ m^3 → n ≤ m :=
by
  use 60
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l265_26594


namespace NUMINAMATH_CALUDE_circles_common_chord_l265_26520

-- Define the circles
def circle1 (x y a : ℝ) : Prop := (x - a)^2 + (y + 2)^2 = 4
def circle2 (x y b : ℝ) : Prop := (x + b)^2 + (y + 2)^2 = 1

-- Define the condition for intersection
def intersect (a b : ℝ) : Prop := 1 < |a + b| ∧ |a + b| < Real.sqrt 3

-- Define the equation of the common chord
def common_chord (x a b : ℝ) : Prop := (2*a + 2*b)*x + 3 + b^2 - a^2 = 0

-- Theorem statement
theorem circles_common_chord (a b : ℝ) (h : intersect a b) :
  ∀ x y : ℝ, circle1 x y a ∧ circle2 x y b → common_chord x a b :=
sorry

end NUMINAMATH_CALUDE_circles_common_chord_l265_26520


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l265_26595

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((2 * x^2 + y^2) * (4 * x^2 + y^2)).sqrt) / (x * y) ≥ 3 :=
sorry

theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    (((2 * x₀^2 + y₀^2) * (4 * x₀^2 + y₀^2)).sqrt) / (x₀ * y₀) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l265_26595


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l265_26548

/-- Given a geometric sequence with first term 1, last term 9, and middle terms a, b, c, prove that b = 3 -/
theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h_sequence : ∃ (r : ℝ), r > 0 ∧ a = 1 * r ∧ b = a * r ∧ c = b * r ∧ 9 = c * r) : 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l265_26548


namespace NUMINAMATH_CALUDE_complex_number_problem_l265_26542

open Complex

theorem complex_number_problem (z : ℂ) (a b : ℝ) : 
  z = ((1 + I)^2 + 2*(5 - I)) / (3 + I) →
  abs z = Real.sqrt 10 ∧
  (z * (z + a) = b + I → a = -7 ∧ b = -13) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l265_26542


namespace NUMINAMATH_CALUDE_f_properties_l265_26580

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + m

theorem f_properties (m : ℝ) :
  (∀ x > 0, f m x ≤ 0) →
  (m = 1 ∧
   ∀ a b, 0 < a → a < b →
     (f m b - f m a) / (b - a) < 1 / (a * (a + 1))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l265_26580


namespace NUMINAMATH_CALUDE_gwen_homework_problems_l265_26586

/-- The number of math problems Gwen had -/
def math_problems : ℕ := 18

/-- The number of science problems Gwen had -/
def science_problems : ℕ := 11

/-- The number of problems Gwen finished at school -/
def finished_at_school : ℕ := 24

/-- The number of problems Gwen had to do for homework -/
def homework_problems : ℕ := math_problems + science_problems - finished_at_school

theorem gwen_homework_problems :
  homework_problems = 5 := by sorry

end NUMINAMATH_CALUDE_gwen_homework_problems_l265_26586


namespace NUMINAMATH_CALUDE_corn_acres_calculation_l265_26510

def total_land : ℝ := 1634
def beans_ratio : ℝ := 4.5
def wheat_ratio : ℝ := 2.3
def corn_ratio : ℝ := 3.8
def barley_ratio : ℝ := 3.4

theorem corn_acres_calculation :
  let total_ratio := beans_ratio + wheat_ratio + corn_ratio + barley_ratio
  let acres_per_part := total_land / total_ratio
  let corn_acres := corn_ratio * acres_per_part
  ∃ ε > 0, |corn_acres - 443.51| < ε :=
by sorry

end NUMINAMATH_CALUDE_corn_acres_calculation_l265_26510


namespace NUMINAMATH_CALUDE_hiking_campers_l265_26523

theorem hiking_campers (morning_rowing : ℕ) (afternoon_rowing : ℕ) (total_campers : ℕ)
  (h1 : morning_rowing = 41)
  (h2 : afternoon_rowing = 26)
  (h3 : total_campers = 71)
  : total_campers - (morning_rowing + afternoon_rowing) = 4 := by
  sorry

end NUMINAMATH_CALUDE_hiking_campers_l265_26523


namespace NUMINAMATH_CALUDE_milk_glass_density_ratio_l265_26544

/-- Prove that the density of milk is 0.2 times the density of glass -/
theorem milk_glass_density_ratio 
  (m_CT : ℝ) -- mass of empty glass jar
  (m_M : ℝ)  -- mass of milk
  (V_CT : ℝ) -- volume of glass
  (V_M : ℝ)  -- volume of milk
  (h1 : m_CT + m_M = 3 * m_CT) -- mass of full jar is 3 times mass of empty jar
  (h2 : V_M = 10 * V_CT) -- volume of milk is 10 times volume of glass
  : m_M / V_M = 0.2 * (m_CT / V_CT) := by
  sorry

#check milk_glass_density_ratio

end NUMINAMATH_CALUDE_milk_glass_density_ratio_l265_26544


namespace NUMINAMATH_CALUDE_alpha_sufficient_not_necessary_for_beta_l265_26521

theorem alpha_sufficient_not_necessary_for_beta :
  (∀ x : ℝ, x = -1 → x ≤ 0) ∧ 
  (∃ x : ℝ, x ≤ 0 ∧ x ≠ -1) := by
  sorry

end NUMINAMATH_CALUDE_alpha_sufficient_not_necessary_for_beta_l265_26521


namespace NUMINAMATH_CALUDE_credit_card_balance_transfer_l265_26522

theorem credit_card_balance_transfer (G : ℝ) : 
  let gold_limit : ℝ := G
  let platinum_limit : ℝ := 2 * G
  let gold_balance : ℝ := G / 3
  let platinum_balance : ℝ := platinum_limit / 4
  let new_platinum_balance : ℝ := platinum_balance + gold_balance
  let unspent_portion : ℝ := (platinum_limit - new_platinum_balance) / platinum_limit
  unspent_portion = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_credit_card_balance_transfer_l265_26522


namespace NUMINAMATH_CALUDE_x_squared_coefficient_expansion_l265_26551

/-- The coefficient of x^2 in the expansion of (x+1)^5(2x+1) is 20 -/
theorem x_squared_coefficient_expansion : ∃ (p : Polynomial ℤ), 
  p = (X + 1)^5 * (2*X + 1) ∧ p.coeff 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_expansion_l265_26551


namespace NUMINAMATH_CALUDE_lamp_purchasing_problem_l265_26589

/-- Represents a purchasing plan for energy-saving lamps -/
structure LampPlan where
  typeA : ℕ
  typeB : ℕ
  cost : ℕ

/-- Checks if a plan satisfies the given constraints -/
def isValidPlan (plan : LampPlan) : Prop :=
  plan.typeA + plan.typeB = 50 ∧
  2 * plan.typeB ≤ plan.typeA ∧
  plan.typeA ≤ 3 * plan.typeB

/-- Calculates the cost of a plan given the prices of lamps -/
def calculateCost (priceA priceB : ℕ) (plan : LampPlan) : ℕ :=
  priceA * plan.typeA + priceB * plan.typeB

/-- Main theorem to prove -/
theorem lamp_purchasing_problem :
  ∃ (priceA priceB : ℕ) (plans : List LampPlan),
    priceA + 3 * priceB = 26 ∧
    3 * priceA + 2 * priceB = 29 ∧
    priceA = 5 ∧
    priceB = 7 ∧
    plans.length = 4 ∧
    (∀ plan ∈ plans, isValidPlan plan) ∧
    (∃ bestPlan ∈ plans,
      bestPlan.typeA = 37 ∧
      bestPlan.typeB = 13 ∧
      calculateCost priceA priceB bestPlan = 276 ∧
      ∀ plan ∈ plans, calculateCost priceA priceB bestPlan ≤ calculateCost priceA priceB plan) :=
sorry

end NUMINAMATH_CALUDE_lamp_purchasing_problem_l265_26589


namespace NUMINAMATH_CALUDE_max_r_value_l265_26564

theorem max_r_value (r : ℕ) (m n : ℕ → ℤ) 
  (h1 : r ≥ 2)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ r → |m i * n j - m j * n i| = 1) :
  r ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_r_value_l265_26564


namespace NUMINAMATH_CALUDE_max_value_quarter_l265_26549

def f (a b : ℕ) : ℚ := (a : ℚ) / (10 * b + a) + (b : ℚ) / (10 * a + b)

theorem max_value_quarter (a b : ℕ) (ha : 2 ≤ a ∧ a ≤ 8) (hb : 2 ≤ b ∧ b ≤ 8) :
  f a b ≤ 1/4 := by
  sorry

#eval f 2 2  -- To check the function definition

end NUMINAMATH_CALUDE_max_value_quarter_l265_26549


namespace NUMINAMATH_CALUDE_acute_angle_solution_l265_26539

theorem acute_angle_solution : ∃ x : Real, 
  0 < x ∧ 
  x < π / 2 ∧ 
  2 * (Real.sin x)^2 + Real.sin x - Real.sin (2 * x) = 3 * Real.cos x ∧ 
  x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_solution_l265_26539


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l265_26500

theorem arithmetic_calculation : (1 + 2) * (3 - 4) + 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l265_26500


namespace NUMINAMATH_CALUDE_greatest_integer_mike_l265_26576

theorem greatest_integer_mike (n : ℕ) : 
  (∃ k l : ℤ, n = 9 * k - 1 ∧ n = 10 * l - 4) →
  n < 150 →
  (∀ m : ℕ, (∃ k l : ℤ, m = 9 * k - 1 ∧ m = 10 * l - 4) → m < 150 → m ≤ n) →
  n = 86 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_mike_l265_26576


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l265_26538

theorem solution_set_of_inequalities :
  let S := {x : ℝ | x - 2 > 1 ∧ x < 4}
  S = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l265_26538


namespace NUMINAMATH_CALUDE_B_equals_one_four_l265_26591

def A : Set ℝ := {0, 1, 2, 3}

def B (m : ℝ) : Set ℝ := {x | x^2 - 5*x + m = 0}

theorem B_equals_one_four (m : ℝ) : 
  (A ∩ B m = {1}) → B m = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_B_equals_one_four_l265_26591


namespace NUMINAMATH_CALUDE_product_sum_theorem_l265_26528

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a + b + c = 20) : 
  a*b + b*c + c*a = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l265_26528


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l265_26554

def M : Matrix (Fin 2) (Fin 2) ℚ :=
  !![3, 1; 0, -2]

theorem inverse_as_linear_combination :
  ∃ (a b : ℚ), a = 1/6 ∧ b = -1/6 ∧ M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l265_26554


namespace NUMINAMATH_CALUDE_value_of_y_l265_26527

theorem value_of_y : (2010^2 - 2010 + 1) / 2010 = 2009 + 1/2010 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l265_26527


namespace NUMINAMATH_CALUDE_negation_of_set_implication_l265_26545

theorem negation_of_set_implication (A B : Set α) :
  ¬(A ∪ B = A → A ∩ B = B) ↔ (A ∪ B ≠ A → A ∩ B ≠ B) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_set_implication_l265_26545


namespace NUMINAMATH_CALUDE_intersection_k_range_l265_26555

-- Define the line equation
def line_eq (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for intersection at two distinct points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧
  hyperbola_eq x₁ (line_eq k x₁) ∧
  hyperbola_eq x₂ (line_eq k x₂)

-- State the theorem
theorem intersection_k_range :
  ∀ k : ℝ, intersects_at_two_points k ↔ 1 < k ∧ k < Real.sqrt 15 / 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_k_range_l265_26555


namespace NUMINAMATH_CALUDE_symmetry_preserves_circle_l265_26505

/-- A circle in R^2 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in R^2 of the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- The given circle (x-1)^2 + (y-1)^2 = 1 -/
def given_circle : Circle := { center := (1, 1), radius := 1 }

/-- The given line y = 5x - 4 -/
def given_line : Line := { m := 5, b := -4 }

/-- Predicate to check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.m * p.1 + l.b

/-- The symmetrical circle with respect to a line -/
def symmetrical_circle (c : Circle) (l : Line) : Circle :=
  sorry -- Definition of symmetrical circle

theorem symmetry_preserves_circle (c : Circle) (l : Line) :
  point_on_line c.center l →
  symmetrical_circle c l = c := by
  sorry

end NUMINAMATH_CALUDE_symmetry_preserves_circle_l265_26505


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l265_26546

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 4

/-- The total number of people that can ride the wheel at the same time -/
def total_people : ℕ := 20

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := total_people / num_seats

theorem ferris_wheel_capacity : people_per_seat = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l265_26546


namespace NUMINAMATH_CALUDE_enchilada_cost_l265_26506

theorem enchilada_cost (T E : ℝ) 
  (h1 : 2 * T + 3 * E = 7.80)
  (h2 : 3 * T + 5 * E = 12.70) : 
  E = 2.00 := by
sorry

end NUMINAMATH_CALUDE_enchilada_cost_l265_26506


namespace NUMINAMATH_CALUDE_A_99_times_B_l265_26587

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]
def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 0, 1, -1; 1, 0, 0]

theorem A_99_times_B : 
  A^99 * B = !![1, 0, 0; 1, 1, 1; 0, 1, -1] := by sorry

end NUMINAMATH_CALUDE_A_99_times_B_l265_26587


namespace NUMINAMATH_CALUDE_line_parallel_in_perp_planes_l265_26509

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Define the relation of a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_in_perp_planes
  (α β : Plane) (l m n : Line)
  (h1 : perp_plane α β)
  (h2 : l = intersection α β)
  (h3 : in_plane n β)
  (h4 : perp_line n l)
  (h5 : perp_line_plane m α) :
  parallel m n :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_in_perp_planes_l265_26509


namespace NUMINAMATH_CALUDE_mikey_has_56_jelly_beans_l265_26579

def napoleon_jelly_beans : ℕ := 34

def sedrich_jelly_beans (napoleon : ℕ) : ℕ := napoleon + 7

def daphne_jelly_beans (sedrich : ℕ) : ℕ := sedrich - 4

def mikey_jelly_beans (napoleon sedrich daphne : ℕ) : ℕ :=
  (3 * (napoleon + sedrich + daphne)) / 6

theorem mikey_has_56_jelly_beans :
  mikey_jelly_beans napoleon_jelly_beans 
    (sedrich_jelly_beans napoleon_jelly_beans) 
    (daphne_jelly_beans (sedrich_jelly_beans napoleon_jelly_beans)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_mikey_has_56_jelly_beans_l265_26579


namespace NUMINAMATH_CALUDE_matrix_product_result_l265_26536

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range n).foldl
    (λ acc i => acc * !![1, 2*(i+1); 0, 1])
    (!![1, 0; 0, 1])

theorem matrix_product_result :
  matrix_product 50 = !![1, 2550; 0, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_product_result_l265_26536


namespace NUMINAMATH_CALUDE_lizzy_money_theorem_l265_26585

/-- Calculates the final amount Lizzy has after lending money and receiving it back with interest -/
def final_amount (initial : ℝ) (loan : ℝ) (interest_rate : ℝ) : ℝ :=
  initial - loan + loan * (1 + interest_rate)

/-- Theorem stating that given the specific conditions, Lizzy will have $33 -/
theorem lizzy_money_theorem :
  let initial := 30
  let loan := 15
  let interest_rate := 0.2
  final_amount initial loan interest_rate = 33 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_theorem_l265_26585


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l265_26574

theorem min_value_sum_of_fractions (a b : ℤ) (h : a ≠ b) :
  (a^2 + b^2 : ℚ) / (a^2 - b^2) + (a^2 - b^2 : ℚ) / (a^2 + b^2) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l265_26574


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l265_26540

/-- The area of a triangle with base 18 and height 6 is 54 -/
theorem triangle_area : Real → Real → Real → Prop :=
  fun base height area =>
    base = 18 ∧ height = 6 → area = (base * height) / 2 → area = 54

-- The proof is omitted
theorem triangle_area_proof : triangle_area 18 6 54 := by sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l265_26540


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l265_26557

/-- A triangle with given altitudes and median -/
structure Triangle where
  ma : ℝ  -- altitude to side a
  mb : ℝ  -- altitude to side b
  sc : ℝ  -- median to side c
  ma_pos : 0 < ma
  mb_pos : 0 < mb
  sc_pos : 0 < sc

/-- The existence condition for a triangle with given altitudes and median -/
def triangle_exists (t : Triangle) : Prop :=
  t.ma < 2 * t.sc ∧ t.mb < 2 * t.sc

/-- Theorem stating the necessary and sufficient condition for triangle existence -/
theorem triangle_existence_condition (t : Triangle) :
  ∃ (triangle : Triangle), triangle.ma = t.ma ∧ triangle.mb = t.mb ∧ triangle.sc = t.sc ↔ triangle_exists t :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l265_26557


namespace NUMINAMATH_CALUDE_f_three_minus_f_neg_three_l265_26588

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3*x^3 + 4*x

-- State the theorem
theorem f_three_minus_f_neg_three : f 3 - f (-3) = 672 := by
  sorry

end NUMINAMATH_CALUDE_f_three_minus_f_neg_three_l265_26588


namespace NUMINAMATH_CALUDE_cube_root_of_negative_two_sqrt_two_l265_26598

theorem cube_root_of_negative_two_sqrt_two (x : ℝ) :
  x = ((-2 : ℝ) ^ (1/2 : ℝ)) → x = ((-2 * (2 ^ (1/2 : ℝ))) ^ (1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_two_sqrt_two_l265_26598


namespace NUMINAMATH_CALUDE_min_value_of_f_inequality_condition_l265_26560

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 2|

-- Statement 1: The minimum value of f(x) is 4
theorem min_value_of_f : ∃ (x : ℝ), f x = 4 ∧ ∀ (y : ℝ), f y ≥ 4 :=
sorry

-- Statement 2: f(x) ≥ |a+4| - |a-3| for all x if and only if a ≤ 3/2
theorem inequality_condition (a : ℝ) : 
  (∀ (x : ℝ), f x ≥ |a + 4| - |a - 3|) ↔ a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_inequality_condition_l265_26560


namespace NUMINAMATH_CALUDE_percent_both_correct_l265_26597

/-- Given a class where:
    - 70% answered the first question correctly
    - 55% answered the second question correctly
    - 20% answered neither question correctly
    Prove that 45% answered both questions correctly -/
theorem percent_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) :
  p_first = 0.7 →
  p_second = 0.55 →
  p_neither = 0.2 →
  p_first + p_second - (1 - p_neither) = 0.45 := by
sorry

end NUMINAMATH_CALUDE_percent_both_correct_l265_26597


namespace NUMINAMATH_CALUDE_previous_day_visitors_count_l265_26525

/-- The number of visitors to Buckingham Palace on the current day -/
def current_day_visitors : ℕ := 666

/-- The difference in visitors between the current day and the previous day -/
def visitor_difference : ℕ := 566

/-- The number of visitors to Buckingham Palace on the previous day -/
def previous_day_visitors : ℕ := current_day_visitors - visitor_difference

theorem previous_day_visitors_count : previous_day_visitors = 100 := by
  sorry

end NUMINAMATH_CALUDE_previous_day_visitors_count_l265_26525


namespace NUMINAMATH_CALUDE_negative_distribution_l265_26543

theorem negative_distribution (a b c : ℝ) : -(a - b + c) = -a + b - c := by
  sorry

end NUMINAMATH_CALUDE_negative_distribution_l265_26543


namespace NUMINAMATH_CALUDE_cement_for_tess_street_l265_26532

/-- The amount of cement used for Tess's street, given the total cement used and the amount used for Lexi's street. -/
theorem cement_for_tess_street (total_cement : ℝ) (lexi_cement : ℝ) 
  (h1 : total_cement = 15.1)
  (h2 : lexi_cement = 10) : 
  total_cement - lexi_cement = 5.1 := by
  sorry

end NUMINAMATH_CALUDE_cement_for_tess_street_l265_26532


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l265_26508

theorem absolute_value_inequality (x : ℝ) : ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l265_26508


namespace NUMINAMATH_CALUDE_inverse_function_property_l265_26562

-- Define a function f with an inverse
def f_has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- State the theorem
theorem inverse_function_property
  (f : ℝ → ℝ)
  (h_inverse : f_has_inverse f)
  (h_point : f 2 = -1) :
  ∃ f_inv : ℝ → ℝ, f_inv (-1) = 2 ∧ (∀ x, f_inv (f x) = x) ∧ (∀ y, f (f_inv y) = y) :=
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l265_26562


namespace NUMINAMATH_CALUDE_first_rectangle_height_l265_26556

/-- Proves that the height of the first rectangle is 5 inches -/
theorem first_rectangle_height : 
  ∀ (h : ℝ), -- height of the first rectangle
  (4 * h = 3 * 6 + 2) → -- area of first = area of second + 2
  h = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_rectangle_height_l265_26556


namespace NUMINAMATH_CALUDE_sophomore_selection_l265_26559

/-- Calculates the number of sophomores selected for a study tour using proportional allocation -/
theorem sophomore_selection (freshmen sophomore junior total_spots : ℕ) : 
  freshmen = 240 →
  sophomore = 260 →
  junior = 300 →
  total_spots = 40 →
  (sophomore * total_spots) / (freshmen + sophomore + junior) = 26 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_selection_l265_26559


namespace NUMINAMATH_CALUDE_game_value_conversion_l265_26537

/-- Calculates the final value of sold games in USD after multiple currency conversions and fees --/
theorem game_value_conversion (initial_value : ℝ) (usd_to_eur_rate : ℝ) (eur_to_usd_fee : ℝ)
  (value_increase : ℝ) (eur_to_jpy_rate : ℝ) (eur_to_jpy_fee : ℝ) (sell_percentage : ℝ)
  (japan_tax_rate : ℝ) (jpy_to_usd_rate : ℝ) (jpy_to_usd_fee : ℝ) :
  initial_value = 200 →
  usd_to_eur_rate = 0.85 →
  eur_to_usd_fee = 0.03 →
  value_increase = 3 →
  eur_to_jpy_rate = 130 →
  eur_to_jpy_fee = 0.02 →
  sell_percentage = 0.4 →
  japan_tax_rate = 0.1 →
  jpy_to_usd_rate = 0.0085 →
  jpy_to_usd_fee = 0.01 →
  ∃ final_value : ℝ, abs (final_value - 190.93) < 0.01 ∧
  final_value = initial_value * usd_to_eur_rate * (1 - eur_to_usd_fee) * value_increase *
                eur_to_jpy_rate * (1 - eur_to_jpy_fee) * sell_percentage *
                (1 - japan_tax_rate) * jpy_to_usd_rate * (1 - jpy_to_usd_fee) := by
  sorry

end NUMINAMATH_CALUDE_game_value_conversion_l265_26537


namespace NUMINAMATH_CALUDE_tv_cash_price_l265_26563

def installment_plan_cost (down_payment : ℕ) (monthly_payment : ℕ) (num_months : ℕ) : ℕ :=
  down_payment + monthly_payment * num_months

def cash_price (total_installment_cost : ℕ) (savings : ℕ) : ℕ :=
  total_installment_cost - savings

theorem tv_cash_price :
  let down_payment : ℕ := 120
  let monthly_payment : ℕ := 30
  let num_months : ℕ := 12
  let savings : ℕ := 80
  let total_installment_cost : ℕ := installment_plan_cost down_payment monthly_payment num_months
  cash_price total_installment_cost savings = 400 := by
  sorry

end NUMINAMATH_CALUDE_tv_cash_price_l265_26563


namespace NUMINAMATH_CALUDE_division_value_problem_l265_26567

theorem division_value_problem (x : ℝ) (h : (5 / x) * 12 = 10) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l265_26567


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_sixteen_l265_26501

theorem sqrt_difference_equals_negative_sixteen :
  Real.sqrt (16 - 8 * Real.sqrt 2) - Real.sqrt (16 + 8 * Real.sqrt 2) = -16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_sixteen_l265_26501


namespace NUMINAMATH_CALUDE_complex_simplification_l265_26575

theorem complex_simplification (i : ℂ) (h : i * i = -1) : 
  (1 + i) / i = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l265_26575


namespace NUMINAMATH_CALUDE_jelly_cost_theorem_l265_26590

/-- The cost of jelly for all sandwiches is $1.68 --/
theorem jelly_cost_theorem (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (3 * B + 7 * J) = 336 → 
  (N * J * 7 : ℚ) / 100 = 1.68 := by
  sorry

end NUMINAMATH_CALUDE_jelly_cost_theorem_l265_26590


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l265_26504

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side_length : α

/-- Calculates the area of a square -/
def Square.area {α : Type*} [LinearOrderedField α] (s : Square α) : α :=
  s.side_length * s.side_length

/-- Represents the shaded regions in the square -/
structure ShadedRegions (α : Type*) [LinearOrderedField α] where
  small_square_side : α
  medium_square_side : α
  large_square_side : α

/-- Calculates the total shaded area -/
def ShadedRegions.total_area {α : Type*} [LinearOrderedField α] (sr : ShadedRegions α) : α :=
  sr.small_square_side * sr.small_square_side +
  (sr.medium_square_side * sr.medium_square_side - sr.small_square_side * sr.small_square_side) +
  (sr.large_square_side * sr.large_square_side - sr.medium_square_side * sr.medium_square_side)

/-- Theorem: The percentage of shaded area in square ABCD is (36/49) * 100 -/
theorem shaded_area_percentage
  (square : Square ℝ)
  (shaded : ShadedRegions ℝ)
  (h1 : square.side_length = 7)
  (h2 : shaded.small_square_side = 2)
  (h3 : shaded.medium_square_side = 4)
  (h4 : shaded.large_square_side = 6) :
  (shaded.total_area / square.area) * 100 = (36 / 49) * 100 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l265_26504


namespace NUMINAMATH_CALUDE_log_inequality_range_l265_26573

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_inequality_range (a : ℝ) :
  log a (2/5) < 1 ↔ (0 < a ∧ a < 2/5) ∨ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_range_l265_26573


namespace NUMINAMATH_CALUDE_inverse_function_solution_l265_26584

/-- Given a function f(x) = 1 / (ax^2 + bx + c), where a, b, and c are nonzero real constants,
    the solutions to f^(-1)(x) = 1 are x = (-b ± √(b^2 - 4a(c-1))) / (2a) -/
theorem inverse_function_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f := fun x => 1 / (a * x^2 + b * x + c)
  let sol₁ := (-b + Real.sqrt (b^2 - 4*a*(c-1))) / (2*a)
  let sol₂ := (-b - Real.sqrt (b^2 - 4*a*(c-1))) / (2*a)
  (∀ x, f x = 1 ↔ x = sol₁ ∨ x = sol₂) :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_solution_l265_26584


namespace NUMINAMATH_CALUDE_inequality_solution_l265_26565

def solution_set (m : ℝ) : Set ℝ :=
  if m < -4 then {x | -1 < x ∧ x < 1 / (m + 3)}
  else if m = -4 then ∅
  else if m > -4 ∧ m < -3 then {x | 1 / (m + 3) < x ∧ x < -1}
  else if m = -3 then {x | x > -1}
  else {x | x < -1 ∨ x > 1 / (m + 3)}

theorem inequality_solution (m : ℝ) :
  {x : ℝ | ((m + 3) * x - 1) * (x + 1) > 0} = solution_set m := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l265_26565


namespace NUMINAMATH_CALUDE_tie_shirt_ratio_l265_26593

/-- Represents the cost of a school uniform -/
structure UniformCost where
  pants : ℝ
  shirt : ℝ
  tie : ℝ
  socks : ℝ

/-- Calculates the total cost of a given number of uniforms -/
def totalCost (u : UniformCost) (n : ℕ) : ℝ :=
  n * (u.pants + u.shirt + u.tie + u.socks)

/-- Theorem: The ratio of tie cost to shirt cost is 1:5 given the uniform pricing conditions -/
theorem tie_shirt_ratio :
  ∀ (u : UniformCost),
    u.pants = 20 →
    u.shirt = 2 * u.pants →
    u.socks = 3 →
    totalCost u 5 = 355 →
    u.tie / u.shirt = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_tie_shirt_ratio_l265_26593


namespace NUMINAMATH_CALUDE_function_range_theorem_l265_26581

theorem function_range_theorem (a : ℝ) :
  (∃ x : ℝ, (|2*x + 1| + |2*x - 3| < |a - 1|)) →
  (a < -3 ∨ a > 5) :=
by sorry

end NUMINAMATH_CALUDE_function_range_theorem_l265_26581


namespace NUMINAMATH_CALUDE_square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l265_26511

theorem square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three :
  ∀ x : ℝ, x = Real.sqrt 2 + 1 → x^2 - 2*x + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l265_26511


namespace NUMINAMATH_CALUDE_locus_of_point_C_l265_26561

/-- The locus of point C in a triangle ABC with given conditions forms an ellipse -/
theorem locus_of_point_C (A B C : ℝ × ℝ) : 
  (A = (-6, 0) ∧ B = (6, 0)) →  -- Coordinates of A and B
  (dist A B + dist B C + dist C A = 26) →  -- Perimeter condition
  (C.1 ≠ 7 ∧ C.1 ≠ -7) →  -- Exclude points where x = ±7
  (C.1^2 / 49 + C.2^2 / 13 = 1) :=  -- Equation of the ellipse
by sorry

end NUMINAMATH_CALUDE_locus_of_point_C_l265_26561


namespace NUMINAMATH_CALUDE_quadratic_factorization_l265_26577

theorem quadratic_factorization (c d : ℕ) (hc : c > d) :
  (∀ x, x^2 - 20*x + 96 = (x - c) * (x - d)) →
  4*d - c = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l265_26577


namespace NUMINAMATH_CALUDE_age_difference_l265_26503

theorem age_difference : ∀ (a b : ℕ),
  (10 ≤ 10 * a + b) ∧ (10 * a + b < 100) ∧  -- Jack's age is two-digit
  (10 ≤ 10 * b + a) ∧ (10 * b + a < 100) ∧  -- Bill's age is two-digit
  (10 * a + b + 10 = 3 * (10 * b + a + 10))  -- In 10 years, Jack will be 3 times Bill's age
  → (10 * a + b) - (10 * b + a) = 54 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l265_26503


namespace NUMINAMATH_CALUDE_mineral_water_recycling_l265_26558

/-- Calculates the total number of bottles that can be drunk given an initial number of bottles -/
def total_bottles_drunk (initial_bottles : ℕ) : ℕ :=
  sorry

/-- Calculates the initial number of bottles needed to drink a given total number of bottles -/
def initial_bottles_needed (total_drunk : ℕ) : ℕ :=
  sorry

theorem mineral_water_recycling :
  (total_bottles_drunk 1999 = 2665) ∧
  (initial_bottles_needed 3126 = 2345) :=
by sorry

end NUMINAMATH_CALUDE_mineral_water_recycling_l265_26558


namespace NUMINAMATH_CALUDE_quartic_roots_sum_l265_26519

theorem quartic_roots_sum (p q r s : ℂ) : 
  (p^4 = p^2 + p + 2) → 
  (q^4 = q^2 + q + 2) → 
  (r^4 = r^2 + r + 2) → 
  (s^4 = s^2 + s + 2) → 
  p * (q - r)^2 + q * (r - s)^2 + r * (s - p)^2 + s * (p - q)^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_quartic_roots_sum_l265_26519


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l265_26502

-- Define the number of games
def num_games : ℕ := 3

-- Define the total points scored
def total_points : ℕ := 81

-- Define the points per game as a function
def points_per_game : ℕ := total_points / num_games

-- Theorem to prove
theorem melissa_points_per_game : points_per_game = 27 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l265_26502


namespace NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l265_26583

theorem prime_pairs_satisfying_equation :
  ∀ p q n : ℕ,
    Prime p → Prime q → n > 0 →
    p * (p + 1) + q * (q + 1) = n * (n + 1) →
    ((p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) ∨ (p = 2 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l265_26583


namespace NUMINAMATH_CALUDE_parabola_coeff_sum_l265_26570

/-- A parabola with equation y = ax^2 + bx + c, vertex at (2, 3), and passing through (5, 6) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 2
  vertex_y : ℝ := 3
  point_x : ℝ := 5
  point_y : ℝ := 6
  eq_at_vertex : 3 = a * 2^2 + b * 2 + c
  eq_at_point : 6 = a * 5^2 + b * 5 + c

/-- The sum of coefficients a, b, and c equals 4 -/
theorem parabola_coeff_sum (p : Parabola) : p.a + p.b + p.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coeff_sum_l265_26570


namespace NUMINAMATH_CALUDE_unique_fixed_point_of_rotation_invariant_function_l265_26531

/-- A function is rotation-invariant if rotating its graph by π/2 around the origin
    results in the same graph. -/
def RotationInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (-y) = x

theorem unique_fixed_point_of_rotation_invariant_function
  (f : ℝ → ℝ) (h : RotationInvariant f) :
  ∃! x, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_fixed_point_of_rotation_invariant_function_l265_26531


namespace NUMINAMATH_CALUDE_exam_theorem_l265_26513

def exam_problem (total_boys : ℕ) (overall_average : ℚ) (passed_boys : ℕ) (failed_average : ℚ) : Prop :=
  let passed_average : ℚ := (total_boys * overall_average - (total_boys - passed_boys) * failed_average) / passed_boys
  passed_average = 39

theorem exam_theorem : exam_problem 120 36 105 15 := by
  sorry

end NUMINAMATH_CALUDE_exam_theorem_l265_26513


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l265_26529

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line y = -4x --/
def line1 (x y : ℝ) : Prop := y = -4 * x

/-- The line x + y - 1 = 0 --/
def line2 (x y : ℝ) : Prop := x + y - 1 = 0

/-- Check if a point is on a circle --/
def isOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Check if a circle is tangent to a line at a point --/
def isTangent (c : Circle) (p : ℝ × ℝ) : Prop :=
  isOnCircle c p ∧ line2 p.1 p.2

/-- The equation of the circle --/
def circleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 95 = 0

theorem circle_satisfies_conditions :
  ∃! c : Circle,
    (∃ x y : ℝ, line1 x y ∧ c.center = (x, y)) ∧
    isTangent c (3, -2) ∧
    isOnCircle c (1, 12) ∧
    isOnCircle c (7, 10) ∧
    isOnCircle c (-9, 2) ∧
    ∀ x y : ℝ, circleEquation x y ↔ isOnCircle c (x, y) :=
  sorry


end NUMINAMATH_CALUDE_circle_satisfies_conditions_l265_26529


namespace NUMINAMATH_CALUDE_amount_after_two_years_l265_26566

theorem amount_after_two_years (initial_amount : ℝ) (increase_ratio : ℝ) :
  initial_amount = 70400 →
  increase_ratio = 1 / 8 →
  initial_amount * (1 + increase_ratio)^2 = 89070 :=
by sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l265_26566


namespace NUMINAMATH_CALUDE_area_triangle_BCD_l265_26533

/-- Given a triangle ABC and a point D on the line AC extended, 
    prove that the area of triangle BCD can be calculated. -/
theorem area_triangle_BCD 
  (area_ABC : ℝ) 
  (length_AC : ℝ) 
  (length_CD : ℝ) 
  (h_area_ABC : area_ABC = 36)
  (h_length_AC : length_AC = 9)
  (h_length_CD : length_CD = 33) :
  ∃ (area_BCD : ℝ), area_BCD = 132 := by
sorry

end NUMINAMATH_CALUDE_area_triangle_BCD_l265_26533


namespace NUMINAMATH_CALUDE_system_solution_l265_26517

theorem system_solution :
  ∀ x y z t : ℝ,
  (x * y - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18) →
  ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l265_26517


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l265_26553

theorem smallest_root_of_quadratic (x : ℝ) : 
  (10 * x^2 - 66 * x + 56 = 0) → (x ≥ 1.6) := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l265_26553


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l265_26514

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = (1 / 2) * a n

theorem geometric_sequence_a1 (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 4 = 8) : 
  a 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_l265_26514


namespace NUMINAMATH_CALUDE_division_equality_l265_26526

theorem division_equality : 204 / 12.75 = 16 := by
  -- Given condition
  have h1 : 2.04 / 1.275 = 1.6 := by sorry
  
  -- Define the scaling factor
  let scale : ℝ := 100 / 10
  
  -- Prove that 204 / 12.75 = 16
  sorry

end NUMINAMATH_CALUDE_division_equality_l265_26526


namespace NUMINAMATH_CALUDE_problem_statement_l265_26541

theorem problem_statement (p q : Prop) 
  (hp : p ↔ 3 % 2 = 1) 
  (hq : q ↔ 5 % 2 = 0) : 
  p ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l265_26541


namespace NUMINAMATH_CALUDE_total_assignment_plans_l265_26516

def number_of_male_doctors : ℕ := 6
def number_of_female_doctors : ℕ := 4
def number_of_selected_male_doctors : ℕ := 3
def number_of_selected_female_doctors : ℕ := 2
def number_of_regions : ℕ := 5

def assignment_plans : ℕ := 12960

theorem total_assignment_plans :
  (number_of_male_doctors = 6) →
  (number_of_female_doctors = 4) →
  (number_of_selected_male_doctors = 3) →
  (number_of_selected_female_doctors = 2) →
  (number_of_regions = 5) →
  assignment_plans = 12960 :=
by sorry

end NUMINAMATH_CALUDE_total_assignment_plans_l265_26516
