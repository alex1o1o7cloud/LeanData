import Mathlib

namespace NUMINAMATH_CALUDE_steps_to_madison_square_garden_l206_20616

/-- The number of steps taken to reach Madison Square Garden -/
def total_steps (steps_down : ℕ) (steps_to_msg : ℕ) : ℕ :=
  steps_down + steps_to_msg

/-- Theorem stating the total number of steps taken to reach Madison Square Garden -/
theorem steps_to_madison_square_garden :
  total_steps 676 315 = 991 := by
  sorry

end NUMINAMATH_CALUDE_steps_to_madison_square_garden_l206_20616


namespace NUMINAMATH_CALUDE_dual_polyhedron_properties_l206_20649

/-- A regular polyhedron with its dual -/
structure RegularPolyhedronWithDual where
  G : ℕ  -- number of faces
  P : ℕ  -- number of edges
  B : ℕ  -- number of vertices
  n : ℕ  -- number of edges meeting at each vertex

/-- Properties of the dual of a regular polyhedron -/
def dual_properties (poly : RegularPolyhedronWithDual) : Prop :=
  ∃ (dual_faces dual_edges dual_vertices : ℕ),
    dual_faces = poly.B ∧
    dual_edges = poly.P ∧
    dual_vertices = poly.G

/-- Theorem stating the properties of the dual polyhedron -/
theorem dual_polyhedron_properties (poly : RegularPolyhedronWithDual) :
  dual_properties poly :=
sorry

end NUMINAMATH_CALUDE_dual_polyhedron_properties_l206_20649


namespace NUMINAMATH_CALUDE_sufficient_condition_transitivity_l206_20658

theorem sufficient_condition_transitivity 
  (C B A : Prop) 
  (h1 : C → B) 
  (h2 : B → A) : 
  C → A := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_transitivity_l206_20658


namespace NUMINAMATH_CALUDE_angle_properties_l206_20642

theorem angle_properties (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.tan (2*θ) = -2*Real.sqrt 2) :
  (Real.tan θ = -Real.sqrt 2 / 2) ∧
  ((2 * (Real.cos (θ/2))^2 - Real.sin θ - Real.tan (5*π/4)) / (Real.sqrt 2 * Real.sin (θ + π/4)) = 3 + 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l206_20642


namespace NUMINAMATH_CALUDE_color_film_fraction_l206_20631

/-- Given a film festival selection process, this theorem proves the fraction of selected films that are in color. -/
theorem color_film_fraction (x y : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) : 
  let total_bw : ℝ := 20 * x
  let total_color : ℝ := 4 * y
  let selected_bw : ℝ := (y / x) * total_bw / 100
  let selected_color : ℝ := total_color
  (selected_color) / (selected_bw + selected_color) = 20 / (x + 20) := by
  sorry

end NUMINAMATH_CALUDE_color_film_fraction_l206_20631


namespace NUMINAMATH_CALUDE_smallest_regular_polygon_with_28_degree_extension_l206_20695

/-- The angle (in degrees) at which two extended sides of a regular polygon meet -/
def extended_angle (n : ℕ) : ℚ :=
  180 / n

/-- Theorem stating that 45 is the smallest positive integer n for which
    a regular n-sided polygon has two extended sides meeting at an angle of 28 degrees -/
theorem smallest_regular_polygon_with_28_degree_extension :
  (∀ k : ℕ, k > 0 → k < 45 → extended_angle k ≠ 28) ∧ extended_angle 45 = 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_regular_polygon_with_28_degree_extension_l206_20695


namespace NUMINAMATH_CALUDE_running_speed_calculation_l206_20663

theorem running_speed_calculation (walking_speed running_speed total_distance total_time : ℝ) :
  walking_speed = 4 →
  total_distance = 4 →
  total_time = 0.75 →
  (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time →
  running_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_running_speed_calculation_l206_20663


namespace NUMINAMATH_CALUDE_pencil_cost_l206_20655

-- Define the cost of a pen and a pencil as real numbers
variable (x y : ℝ)

-- Define the conditions from the problem
def condition1 : Prop := 5 * x + 4 * y = 320
def condition2 : Prop := 3 * x + 6 * y = 246

-- State the theorem to be proved
theorem pencil_cost (h1 : condition1 x y) (h2 : condition2 x y) : y = 15 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l206_20655


namespace NUMINAMATH_CALUDE_exponent_operation_l206_20650

theorem exponent_operation (a : ℝ) : -(-a)^2 * a^4 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_operation_l206_20650


namespace NUMINAMATH_CALUDE_baseball_cards_l206_20681

theorem baseball_cards (n : ℕ) : ∃ (total : ℕ), 
  (total = 3 * n + 1) ∧ (∃ (k : ℕ), total = 3 * k + 1) := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_l206_20681


namespace NUMINAMATH_CALUDE_bus_distance_theorem_l206_20624

/-- Calculates the total distance traveled by a bus with increasing speed over a given number of hours -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  hours * (2 * initialSpeed + (hours - 1) * speedIncrease) / 2

/-- Theorem stating that a bus with given initial speed and speed increase travels a specific distance in 12 hours -/
theorem bus_distance_theorem (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) :
  initialSpeed = 35 →
  speedIncrease = 2 →
  hours = 12 →
  totalDistance initialSpeed speedIncrease hours = 552 := by
  sorry

#eval totalDistance 35 2 12  -- This should evaluate to 552

end NUMINAMATH_CALUDE_bus_distance_theorem_l206_20624


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_number_l206_20600

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ        -- Total number of items
  sampleSize : ℕ   -- Size of the sample
  step : ℕ         -- Step size for systematic sampling
  first : ℕ        -- First sample number

/-- Generates the nth sample number in a systematic sample -/
def nthSample (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first + (n - 1) * s.step

/-- Checks if a number is in the sample -/
def isInSample (s : SystematicSample) (num : ℕ) : Prop :=
  ∃ n : ℕ, n ≤ s.sampleSize ∧ nthSample s n = num

theorem systematic_sample_fourth_number
  (s : SystematicSample)
  (h_total : s.total = 52)
  (h_size : s.sampleSize = 4)
  (h_7 : isInSample s 7)
  (h_33 : isInSample s 33)
  (h_46 : isInSample s 46) :
  isInSample s 20 :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_number_l206_20600


namespace NUMINAMATH_CALUDE_final_number_is_fifty_l206_20669

/-- Represents the state of the board at any given time -/
structure BoardState where
  ones : Nat
  fours : Nat
  others : List Nat

/-- The operation of replacing two numbers with their Pythagorean sum -/
def replaceTwo (x y : Nat) : Nat :=
  Nat.sqrt (x^2 + y^2)

/-- The process of reducing the board until only one number remains -/
def reduceBoard : BoardState → Nat
| s => if s.ones + s.fours + s.others.length = 1
       then if s.ones = 1 then 1
            else if s.fours = 1 then 4
            else s.others.head!
       else sorry -- recursively apply replaceTwo

theorem final_number_is_fifty :
  ∀ (finalNum : Nat),
  (∃ (s : BoardState), s.ones = 900 ∧ s.fours = 100 ∧ s.others = [] ∧
   reduceBoard s = finalNum) →
  finalNum = 50 := by
  sorry

#check final_number_is_fifty

end NUMINAMATH_CALUDE_final_number_is_fifty_l206_20669


namespace NUMINAMATH_CALUDE_koi_fish_after_six_weeks_l206_20671

/-- Represents the number of fish in the tank -/
structure FishTank where
  koi : ℕ
  goldfish : ℕ
  angelfish : ℕ

/-- Calculates the total number of fish in the tank -/
def FishTank.total (ft : FishTank) : ℕ := ft.koi + ft.goldfish + ft.angelfish

/-- Represents the daily and weekly changes in fish numbers -/
structure FishChanges where
  koi_per_day : ℕ
  goldfish_per_day : ℕ
  angelfish_per_week : ℕ

/-- Calculates the new fish numbers after a given number of weeks -/
def apply_changes (initial : FishTank) (changes : FishChanges) (weeks : ℕ) : FishTank :=
  { koi := initial.koi + changes.koi_per_day * 7 * weeks,
    goldfish := initial.goldfish + changes.goldfish_per_day * 7 * weeks,
    angelfish := initial.angelfish + changes.angelfish_per_week * weeks }

theorem koi_fish_after_six_weeks
  (initial : FishTank)
  (changes : FishChanges)
  (h_initial_total : initial.total = 450)
  (h_changes : changes = { koi_per_day := 4, goldfish_per_day := 7, angelfish_per_week := 9 })
  (h_final_goldfish : (apply_changes initial changes 6).goldfish = 300)
  (h_final_angelfish : (apply_changes initial changes 6).angelfish = 180) :
  (apply_changes initial changes 6).koi = 486 :=
sorry

end NUMINAMATH_CALUDE_koi_fish_after_six_weeks_l206_20671


namespace NUMINAMATH_CALUDE_curve_tangent_values_l206_20696

/-- The curve equation -/
def curve (x a b : ℝ) : ℝ := x^2 + a*x + b

/-- The tangent equation -/
def tangent (x y : ℝ) : Prop := x - y + 1 = 0

/-- Main theorem -/
theorem curve_tangent_values (a b : ℝ) :
  (∀ x y, curve x a b = y → tangent x y) →
  a = 1 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_curve_tangent_values_l206_20696


namespace NUMINAMATH_CALUDE_student_number_choice_l206_20687

theorem student_number_choice (x : ℝ) : 2 * x - 138 = 104 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_student_number_choice_l206_20687


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_72_l206_20628

theorem six_digit_divisible_by_72 (A B : ℕ) : 
  A < 10 →
  B < 10 →
  (A * 100000 + 44610 + B) % 72 = 0 →
  A + B = 12 := by
sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_72_l206_20628


namespace NUMINAMATH_CALUDE_area_enclosed_theorem_l206_20618

/-- Represents a configuration of three intersecting circles -/
structure CircleConfiguration where
  radius : ℝ
  centralAngle : ℝ
  numCircles : ℕ

/-- Calculates the area enclosed by the arcs of the circle configuration -/
def areaEnclosedByArcs (config : CircleConfiguration) : ℝ :=
  sorry

/-- Represents the coefficients of the area formula a√b + cπ -/
structure AreaCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the area can be expressed as a√b + cπ with a + b + c = 40.5 -/
theorem area_enclosed_theorem (config : CircleConfiguration) 
  (h1 : config.radius = 5)
  (h2 : config.centralAngle = π / 2)
  (h3 : config.numCircles = 3) :
  ∃ (coef : AreaCoefficients), 
    areaEnclosedByArcs config = coef.a * Real.sqrt coef.b + coef.c * π ∧
    coef.a + coef.b + coef.c = 40.5 :=
  sorry

end NUMINAMATH_CALUDE_area_enclosed_theorem_l206_20618


namespace NUMINAMATH_CALUDE_cousin_arrangement_count_l206_20617

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- There are 4 rooms -/
def num_rooms : ℕ := 4

/-- The theorem stating the number of ways to arrange the cousins -/
theorem cousin_arrangement_count :
  distribute num_cousins num_rooms = 51 := by sorry

end NUMINAMATH_CALUDE_cousin_arrangement_count_l206_20617


namespace NUMINAMATH_CALUDE_min_distance_MN_l206_20622

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  -- Equation of the hyperbola: x²/4 - y² = 1
  equation : ℝ → ℝ → Prop
  -- One asymptote has equation x - 2y = 0
  asymptote : ℝ → ℝ → Prop
  -- The hyperbola passes through (2√2, 1)
  passes_through : Prop

/-- Represents a point on the hyperbola -/
structure PointOnHyperbola (C : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : C.equation x y

/-- Represents the vertices of the hyperbola -/
structure HyperbolaVertices (C : Hyperbola) where
  A₁ : ℝ × ℝ  -- Left vertex
  A₂ : ℝ × ℝ  -- Right vertex

/-- Function to calculate |MN| given a point P on the hyperbola -/
def distance_MN (C : Hyperbola) (V : HyperbolaVertices C) (P : PointOnHyperbola C) : ℝ :=
  sorry  -- Definition of |MN| calculation

/-- The main theorem to prove -/
theorem min_distance_MN (C : Hyperbola) (V : HyperbolaVertices C) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 3 ∧
  ∀ (P : PointOnHyperbola C), distance_MN C V P ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_MN_l206_20622


namespace NUMINAMATH_CALUDE_rational_triple_theorem_l206_20666

/-- The set of triples that satisfy the conditions -/
def valid_triples : Set (ℚ × ℚ × ℚ) :=
  {(1, 1, 1), (1, 2, 2), (2, 4, 4), (2, 3, 6), (3, 3, 3)}

/-- A predicate that checks if a triple of rationals satisfies the conditions -/
def satisfies_conditions (p q r : ℚ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧
  (p + q + r).isInt ∧
  (1/p + 1/q + 1/r).isInt ∧
  (p * q * r).isInt

theorem rational_triple_theorem :
  ∀ p q r : ℚ, satisfies_conditions p q r ↔ (p, q, r) ∈ valid_triples :=
by sorry

end NUMINAMATH_CALUDE_rational_triple_theorem_l206_20666


namespace NUMINAMATH_CALUDE_unique_zero_implies_t_bound_l206_20609

/-- A cubic function parameterized by t -/
def f (t : ℝ) (x : ℝ) : ℝ := -2 * x^3 + 2 * t * x^2 + 1

/-- The derivative of f with respect to x -/
def f_deriv (t : ℝ) (x : ℝ) : ℝ := -6 * x^2 + 4 * t * x

/-- Theorem stating that if f has a unique zero, then t > -3/2 -/
theorem unique_zero_implies_t_bound (t : ℝ) :
  (∃! x, f t x = 0) → t > -3/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_implies_t_bound_l206_20609


namespace NUMINAMATH_CALUDE_car_price_calculation_l206_20676

/-- Calculates the price of a car given loan terms and payments -/
theorem car_price_calculation 
  (loan_years : ℕ) 
  (down_payment : ℚ) 
  (monthly_payment : ℚ) 
  (h_loan_years : loan_years = 5)
  (h_down_payment : down_payment = 5000)
  (h_monthly_payment : monthly_payment = 250) :
  down_payment + loan_years * 12 * monthly_payment = 20000 := by
  sorry

#check car_price_calculation

end NUMINAMATH_CALUDE_car_price_calculation_l206_20676


namespace NUMINAMATH_CALUDE_unique_m_value_l206_20612

theorem unique_m_value : ∃! m : ℝ, (abs m = 1) ∧ (m - 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l206_20612


namespace NUMINAMATH_CALUDE_baking_soda_cost_is_one_l206_20660

/-- Represents the cost of supplies for a science project. -/
structure SupplyCost where
  students : ℕ
  bowCost : ℕ
  vinegarCost : ℕ
  totalCost : ℕ

/-- Calculates the cost of each box of baking soda. -/
def bakingSodaCost (s : SupplyCost) : ℕ :=
  (s.totalCost - (s.students * (s.bowCost + s.vinegarCost))) / s.students

/-- Theorem stating that the cost of each box of baking soda is $1. -/
theorem baking_soda_cost_is_one (s : SupplyCost)
  (h1 : s.students = 23)
  (h2 : s.bowCost = 5)
  (h3 : s.vinegarCost = 2)
  (h4 : s.totalCost = 184) :
  bakingSodaCost s = 1 := by
  sorry

end NUMINAMATH_CALUDE_baking_soda_cost_is_one_l206_20660


namespace NUMINAMATH_CALUDE_f_even_and_decreasing_l206_20614

def f (x : ℝ) := -x^2

theorem f_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_decreasing_l206_20614


namespace NUMINAMATH_CALUDE_prob_two_red_balls_l206_20682

/-- The probability of picking two red balls from a bag containing 3 red balls, 2 blue balls,
    and 3 green balls, when 2 balls are picked at random without replacement. -/
theorem prob_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ)
    (h1 : total_balls = red_balls + blue_balls + green_balls)
    (h2 : red_balls = 3)
    (h3 : blue_balls = 2)
    (h4 : green_balls = 3) :
    (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_balls_l206_20682


namespace NUMINAMATH_CALUDE_miles_walked_approx_2250_l206_20601

/-- Represents a pedometer with a maximum reading before reset --/
structure Pedometer where
  max_reading : ℕ
  steps_per_mile : ℕ

/-- Represents the pedometer readings over a year --/
structure YearlyReading where
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total miles walked based on pedometer data --/
def total_miles_walked (p : Pedometer) (yr : YearlyReading) : ℚ :=
  let total_steps : ℕ := p.max_reading * yr.resets + yr.final_reading + 1
  (total_steps : ℚ) / p.steps_per_mile

/-- Theorem stating that the total miles walked is approximately 2250 --/
theorem miles_walked_approx_2250 (p : Pedometer) (yr : YearlyReading) :
  p.max_reading = 99999 →
  p.steps_per_mile = 1600 →
  yr.resets = 36 →
  yr.final_reading = 25000 →
  2249 < total_miles_walked p yr ∧ total_miles_walked p yr < 2251 :=
sorry

end NUMINAMATH_CALUDE_miles_walked_approx_2250_l206_20601


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_properties_l206_20675

theorem consecutive_integers_sum_properties :
  (∀ k : ℤ, ¬∃ n : ℤ, 12 * k + 78 = n ^ 2) ∧
  (∃ k : ℤ, ∃ n : ℤ, 11 * k + 66 = n ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_properties_l206_20675


namespace NUMINAMATH_CALUDE_flowers_lilly_can_buy_l206_20610

def days_until_birthday : ℕ := 22
def savings_per_day : ℚ := 2
def cost_per_flower : ℚ := 4

theorem flowers_lilly_can_buy :
  (days_until_birthday : ℚ) * savings_per_day / cost_per_flower = 11 := by
  sorry

end NUMINAMATH_CALUDE_flowers_lilly_can_buy_l206_20610


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l206_20632

-- Define the number of options for each food category
def num_meat_options : ℕ := 3
def num_vegetable_options : ℕ := 5
def num_dessert_options : ℕ := 5

-- Define the number of vegetables to be chosen
def num_vegetables_to_choose : ℕ := 2

-- Theorem statement
theorem tyler_meal_choices :
  (num_meat_options) *
  (num_vegetable_options.choose num_vegetables_to_choose) *
  (num_dessert_options) = 150 := by
  sorry


end NUMINAMATH_CALUDE_tyler_meal_choices_l206_20632


namespace NUMINAMATH_CALUDE_triangle_special_case_l206_20608

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circumcenter and orthocenter
def circumcenter (t : Triangle) : ℝ × ℝ := sorry
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex B
def angle_B (t : Triangle) : ℝ := sorry

-- Main theorem
theorem triangle_special_case (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  distance t.B O = distance t.B H →
  (angle_B t = 60 ∨ angle_B t = 120) :=
by sorry

end NUMINAMATH_CALUDE_triangle_special_case_l206_20608


namespace NUMINAMATH_CALUDE_star_operation_result_l206_20636

-- Define the sets A and B
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define the set difference operation
def set_difference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define the * operation
def star_operation (X Y : Set ℝ) : Set ℝ := 
  (set_difference X Y) ∪ (set_difference Y X)

-- Theorem statement
theorem star_operation_result : 
  star_operation A B = {x | (-3 ≤ x ∧ x < 0) ∨ (x > 3)} := by sorry

end NUMINAMATH_CALUDE_star_operation_result_l206_20636


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l206_20602

theorem max_product_under_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_constraint : 3 * x + 8 * y = 48) : x * y ≤ 24 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 8 * y = 48 ∧ x * y = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l206_20602


namespace NUMINAMATH_CALUDE_object_is_cylinder_l206_20667

-- Define the properties of the object
structure GeometricObject where
  front_view : Type
  side_view : Type
  top_view : Type
  front_is_square : front_view = Square
  side_is_square : side_view = Square
  front_side_equal : front_view = side_view
  top_is_circle : top_view = Circle

-- Define the theorem
theorem object_is_cylinder (obj : GeometricObject) : obj = Cylinder := by
  sorry

end NUMINAMATH_CALUDE_object_is_cylinder_l206_20667


namespace NUMINAMATH_CALUDE_second_platform_speed_l206_20672

/-- The speed of Alex's platform in ft/s -/
def alex_speed : ℝ := 1

/-- The distance Alex's platform travels before falling, in ft -/
def fall_distance : ℝ := 100

/-- The time Edward arrives after Alex's platform starts, in seconds -/
def edward_arrival_time : ℝ := 60

/-- Edward's calculation time before launching the second platform, in seconds -/
def edward_calc_time : ℝ := 5

/-- The length of both platforms, in ft -/
def platform_length : ℝ := 5

/-- The optimal speed of the second platform that maximizes Alex's transfer time -/
def optimal_speed : ℝ := 1.125

theorem second_platform_speed (v : ℝ) :
  v = optimal_speed ↔
    (v > 0) ∧
    (v * (fall_distance / alex_speed - edward_arrival_time - edward_calc_time) = 
      fall_distance - alex_speed * edward_arrival_time + platform_length) ∧
    (∀ u : ℝ, u > 0 →
      (u * (fall_distance / alex_speed - edward_arrival_time - edward_calc_time) = 
        fall_distance - alex_speed * edward_arrival_time + platform_length) →
      v ≥ u) :=
by sorry

end NUMINAMATH_CALUDE_second_platform_speed_l206_20672


namespace NUMINAMATH_CALUDE_expression_equals_point_one_l206_20647

-- Define the expression
def expression : ℝ := (0.000001 ^ (1/2)) ^ (1/3)

-- State the theorem
theorem expression_equals_point_one : expression = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_point_one_l206_20647


namespace NUMINAMATH_CALUDE_min_pizzas_to_cover_costs_l206_20603

def car_cost : ℕ := 8000
def earnings_per_pizza : ℕ := 12
def gas_cost_per_delivery : ℕ := 4
def monthly_maintenance : ℕ := 200

theorem min_pizzas_to_cover_costs : 
  ∃ (p : ℕ), p = 1025 ∧ 
  (p * (earnings_per_pizza - gas_cost_per_delivery) ≥ car_cost + monthly_maintenance) ∧
  ∀ (q : ℕ), q < p → q * (earnings_per_pizza - gas_cost_per_delivery) < car_cost + monthly_maintenance :=
sorry

end NUMINAMATH_CALUDE_min_pizzas_to_cover_costs_l206_20603


namespace NUMINAMATH_CALUDE_root_product_equals_32_l206_20685

theorem root_product_equals_32 : 
  (256 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_32_l206_20685


namespace NUMINAMATH_CALUDE_samson_utility_l206_20678

/-- Utility function -/
def utility (math_hours : ℝ) (frisbee_hours : ℝ) : ℝ :=
  (math_hours + 2) * frisbee_hours

/-- The problem statement -/
theorem samson_utility (s : ℝ) : 
  utility (10 - 2*s) s = utility (2*s + 4) (3 - s) → s = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_samson_utility_l206_20678


namespace NUMINAMATH_CALUDE_curves_intersection_l206_20641

/-- The first curve equation -/
def curve1 (x y : ℝ) : Prop :=
  2 * x^2 + 3 * x * y - 2 * y^2 - 6 * x + 3 * y = 0

/-- The second curve equation -/
def curve2 (x y : ℝ) : Prop :=
  3 * x^2 + 7 * x * y + 2 * y^2 - 7 * x + y - 6 = 0

/-- The set of intersection points -/
def intersection_points : Set (ℝ × ℝ) :=
  {(-1, 2), (1, 1), (0, 3/2), (3, 0), (4, -1/2), (5, -1)}

/-- Theorem stating that the given points are the intersection points of the two curves -/
theorem curves_intersection :
  ∀ (p : ℝ × ℝ), p ∈ intersection_points ↔ (curve1 p.1 p.2 ∧ curve2 p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_curves_intersection_l206_20641


namespace NUMINAMATH_CALUDE_subtraction_of_negative_problem_solution_l206_20626

theorem subtraction_of_negative (a b : ℤ) : a - (-b) = a + b := by sorry

theorem problem_solution : 2 - (-12) = 14 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_problem_solution_l206_20626


namespace NUMINAMATH_CALUDE_stock_price_change_l206_20623

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let day1_price := initial_price * (1 - 0.25)
  let day2_price := day1_price * (1 + 0.35)
  (day2_price - initial_price) / initial_price = 0.0125 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l206_20623


namespace NUMINAMATH_CALUDE_min_diff_y_x_l206_20688

theorem min_diff_y_x (x y z : ℤ) 
  (h1 : x < y ∧ y < z) 
  (h2 : Even x)
  (h3 : Odd y ∧ Odd z)
  (h4 : ∀ w, (w : ℤ) ≥ x ∧ Odd w → w - x ≥ 9) :
  ∃ (d : ℤ), d = y - x ∧ ∀ (d' : ℤ), y - x ≤ d' := by
  sorry

end NUMINAMATH_CALUDE_min_diff_y_x_l206_20688


namespace NUMINAMATH_CALUDE_no_perfect_square_3n_plus_2_17n_l206_20694

theorem no_perfect_square_3n_plus_2_17n :
  ∀ n : ℕ, ¬∃ m : ℕ, 3^n + 2 * 17^n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_3n_plus_2_17n_l206_20694


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l206_20613

/-- Given that x and y are positive real numbers, x² and y² vary inversely,
    and y = 5 when x = 2, prove that x = 2/25 when y = 125. -/
theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ (k : ℝ), ∀ x y, x^2 * y^2 = k)
  (h_initial : 2^2 * 5^2 = x^2 * 125^2) :
  y = 125 → x = 2/25 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l206_20613


namespace NUMINAMATH_CALUDE_point_on_line_l206_20657

/-- A point on a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line (t : ℝ) :
  let p1 : Point := ⟨2, 4⟩
  let p2 : Point := ⟨10, 1⟩
  let p3 : Point := ⟨t, 7⟩
  collinear p1 p2 p3 → t = -6 := by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l206_20657


namespace NUMINAMATH_CALUDE_comparison_theorem_l206_20640

theorem comparison_theorem :
  (-4 / 7 : ℚ) > -2 / 3 ∧ -(-7 : ℤ) > -|(-7 : ℤ)| := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l206_20640


namespace NUMINAMATH_CALUDE_vector_operations_l206_20638

def a : ℝ × ℝ := (3, 3)
def b : ℝ × ℝ := (1, 4)

theorem vector_operations :
  (2 • a - b = (5, 2)) ∧
  (∃ m : ℝ, m = -2 ∧ ∃ k : ℝ, k • (m • a + b) = 2 • a - b) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l206_20638


namespace NUMINAMATH_CALUDE_brick_height_calculation_l206_20639

/-- Calculates the height of a brick given wall dimensions and brick count -/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ) 
  (brick_length brick_width : ℝ) (brick_count : ℕ) :
  wall_length = 27 →
  wall_width = 2 →
  wall_height = 0.75 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_count = 27000 →
  ∃ (brick_height : ℝ), 
    brick_height = (wall_length * wall_width * wall_height) / (brick_length * brick_width * brick_count) ∧
    brick_height = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l206_20639


namespace NUMINAMATH_CALUDE_average_growth_rate_satisfies_equation_average_growth_rate_is_twenty_percent_l206_20637

/-- The average monthly growth rate from March to May for a shopping mall's sales volume. -/
def average_growth_rate : ℝ := 0.2

/-- The sales volume in February in yuan. -/
def february_sales : ℝ := 4000000

/-- The sales volume increase rate from February to March. -/
def march_increase_rate : ℝ := 0.1

/-- The sales volume in May in yuan. -/
def may_sales : ℝ := 6336000

/-- Theorem stating that the calculated average growth rate satisfies the sales volume equation. -/
theorem average_growth_rate_satisfies_equation :
  february_sales * (1 + march_increase_rate) * (1 + average_growth_rate)^2 = may_sales := by sorry

/-- Theorem stating that the average growth rate is indeed 20%. -/
theorem average_growth_rate_is_twenty_percent :
  average_growth_rate = 0.2 := by sorry

end NUMINAMATH_CALUDE_average_growth_rate_satisfies_equation_average_growth_rate_is_twenty_percent_l206_20637


namespace NUMINAMATH_CALUDE_count_linear_inequalities_one_variable_l206_20683

-- Define a structure for an expression
structure Expression where
  is_linear_inequality : Bool
  has_one_variable : Bool

-- Define the six expressions
def expressions : List Expression := [
  { is_linear_inequality := true,  has_one_variable := true  }, -- ①
  { is_linear_inequality := false, has_one_variable := true  }, -- ②
  { is_linear_inequality := false, has_one_variable := true  }, -- ③
  { is_linear_inequality := true,  has_one_variable := true  }, -- ④
  { is_linear_inequality := true,  has_one_variable := true  }, -- ⑤
  { is_linear_inequality := true,  has_one_variable := false }  -- ⑥
]

-- Theorem statement
theorem count_linear_inequalities_one_variable :
  (expressions.filter (fun e => e.is_linear_inequality && e.has_one_variable)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_linear_inequalities_one_variable_l206_20683


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l206_20674

theorem simplify_cube_roots : (64 : ℝ) ^ (1/3) - (216 : ℝ) ^ (1/3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l206_20674


namespace NUMINAMATH_CALUDE_sine_function_max_min_l206_20634

theorem sine_function_max_min (a b : ℝ) (h1 : a < 0) :
  (∀ x, a * Real.sin x + b ≤ 3) ∧
  (∀ x, a * Real.sin x + b ≥ -1) ∧
  (∃ x, a * Real.sin x + b = 3) ∧
  (∃ x, a * Real.sin x + b = -1) →
  a = -2 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_sine_function_max_min_l206_20634


namespace NUMINAMATH_CALUDE_abs_func_even_and_increasing_l206_20635

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_func_even_and_increasing :
  (∀ x : ℝ, abs_func (-x) = abs_func x) ∧
  (∀ x y : ℝ, 0 < x → x < y → abs_func x < abs_func y) :=
by sorry

end NUMINAMATH_CALUDE_abs_func_even_and_increasing_l206_20635


namespace NUMINAMATH_CALUDE_solution_range_l206_20615

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 2*m - 2 = 0 ∧ 0 ≤ x ∧ x ≤ 3/2) ↔ 
  -1/2 ≤ m ∧ m ≤ 4 - 2*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l206_20615


namespace NUMINAMATH_CALUDE_jerry_birthday_games_l206_20699

/-- The number of games Jerry received for his birthday -/
def games_received (initial_games final_games : ℕ) : ℕ :=
  final_games - initial_games

/-- Proof that Jerry received 2 games for his birthday -/
theorem jerry_birthday_games :
  games_received 7 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_birthday_games_l206_20699


namespace NUMINAMATH_CALUDE_polynomial_remainder_l206_20662

theorem polynomial_remainder (x : ℝ) : 
  (x^15 + 3) % (x + 2) = -32765 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l206_20662


namespace NUMINAMATH_CALUDE_sum_of_squares_l206_20693

theorem sum_of_squares (a b c : ℝ) : 
  (a + b + c) / 3 = 10 →
  (a * b * c) ^ (1/3 : ℝ) = 6 →
  3 / (1/a + 1/b + 1/c) = 4 →
  a^2 + b^2 + c^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l206_20693


namespace NUMINAMATH_CALUDE_max_value_implies_a_l206_20620

/-- The function f(x) = 2x^3 - 3x^2 + a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 6 * x^2 - 6 * x

theorem max_value_implies_a (a : ℝ) :
  (∃ (max : ℝ), max = 6 ∧ ∀ (x : ℝ), f a x ≤ max) →
  a = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l206_20620


namespace NUMINAMATH_CALUDE_goose_eggs_count_l206_20630

theorem goose_eggs_count (total_eggs : ℕ) : 
  (1 : ℚ) / 3 * (3 : ℚ) / 4 * (2 : ℚ) / 5 * total_eggs = 120 →
  total_eggs = 1200 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l206_20630


namespace NUMINAMATH_CALUDE_polynomial_division_l206_20611

theorem polynomial_division (x : ℝ) :
  5*x^4 - 9*x^3 + 3*x^2 + 7*x - 6 = (x - 1)*(5*x^3 - 4*x^2 + 7*x + 7) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_l206_20611


namespace NUMINAMATH_CALUDE_dave_spent_43_tickets_l206_20690

/-- The number of tickets Dave started with -/
def initial_tickets : ℕ := 98

/-- The number of tickets Dave had left after buying the stuffed tiger -/
def remaining_tickets : ℕ := 55

/-- The number of tickets Dave spent on the stuffed tiger -/
def spent_tickets : ℕ := initial_tickets - remaining_tickets

theorem dave_spent_43_tickets : spent_tickets = 43 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_43_tickets_l206_20690


namespace NUMINAMATH_CALUDE_area_of_section_ABD_l206_20677

theorem area_of_section_ABD (a : ℝ) (S : ℝ) (V : ℝ) : 
  a > 0 → 0 < S → S < π / 2 → V > 0 →
  let area_ABD := (Real.sqrt 3 / Real.sin S) * (V ^ (2 / 3) * Real.tan S) ^ (1 / 3)
  ∃ (h : ℝ), h > 0 ∧ 
    V = (a ^ 3 / 8) * Real.tan S ∧
    area_ABD = (a ^ 2 * Real.sqrt 3) / (4 * Real.cos S) :=
by sorry

#check area_of_section_ABD

end NUMINAMATH_CALUDE_area_of_section_ABD_l206_20677


namespace NUMINAMATH_CALUDE_percentage_problem_l206_20645

theorem percentage_problem (x y : ℝ) : 
  x = 0.18 * 4750 →
  y = 1.3 * x →
  y / 8950 * 100 = 12.42 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l206_20645


namespace NUMINAMATH_CALUDE_equation_solution_l206_20673

theorem equation_solution : 
  ∃ (x : ℤ), 45 - (28 - (37 - (15 - x))) = 56 ∧ x = 122 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l206_20673


namespace NUMINAMATH_CALUDE_axis_of_symmetry_translated_sine_l206_20629

theorem axis_of_symmetry_translated_sine (k : ℤ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 3)
  ∃ (x : ℝ), x = k * π / 2 + π / 12 ∧
    ∀ (y : ℝ), f (x - y) = f (x + y) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_translated_sine_l206_20629


namespace NUMINAMATH_CALUDE_square_difference_equality_l206_20625

theorem square_difference_equality (a b M : ℝ) : 
  (a + 2*b)^2 = (a - 2*b)^2 + M → M = 8*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l206_20625


namespace NUMINAMATH_CALUDE_vector_angle_condition_l206_20651

/-- Given two vectors a and b in R², if the angle between them is acute,
    then the second component of b satisfies the given conditions. -/
theorem vector_angle_condition (m : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, m)
  -- The angle between a and b is acute
  (0 < (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) ∧ 
   (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) < 1) →
  m > -1/2 ∧ m ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_vector_angle_condition_l206_20651


namespace NUMINAMATH_CALUDE_f_greater_than_one_range_l206_20659

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_greater_than_one_range :
  {x₀ : ℝ | f x₀ > 1} = Set.Ioi 1 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_f_greater_than_one_range_l206_20659


namespace NUMINAMATH_CALUDE_xyz_value_l206_20652

variables (x y z : ℝ)

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
                   (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l206_20652


namespace NUMINAMATH_CALUDE_fair_transaction_balance_l206_20698

/-- Represents the financial transactions at a fair --/
structure FairTransactions where
  initial_amount : ℤ
  ride_expense : ℤ
  game_winnings : ℤ
  food_expense : ℤ
  found_money : ℤ
  final_amount : ℤ

/-- Calculates the total amount spent or gained at the fair --/
def total_spent_or_gained (t : FairTransactions) : ℤ :=
  t.initial_amount - t.final_amount

/-- Theorem stating that the total amount spent or gained is equal to 
    the difference between initial and final amounts --/
theorem fair_transaction_balance (t : FairTransactions) : 
  total_spent_or_gained t = 
    t.ride_expense + t.food_expense - t.game_winnings - t.found_money :=
by
  sorry

#eval total_spent_or_gained {
  initial_amount := 87,
  ride_expense := 25,
  game_winnings := 10,
  food_expense := 12,
  found_money := 5,
  final_amount := 16
}

end NUMINAMATH_CALUDE_fair_transaction_balance_l206_20698


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l206_20646

/-- The function f(x) = -x^2 + 2ax - a - a^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x - a - a^2

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ -2) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = -2) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l206_20646


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_area_ratio_l206_20679

theorem right_triangle_inscribed_circle_area_ratio 
  (h a r : ℝ) (h_pos : h > 0) (a_pos : a > 0) (r_pos : r > 0) (h_gt_a : h > a) :
  let A := (1/2) * a * Real.sqrt (h^2 - a^2)
  (π * r^2) / A = 4 * π * A / (a + Real.sqrt (h^2 - a^2) + h)^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_area_ratio_l206_20679


namespace NUMINAMATH_CALUDE_infinitely_many_primes_2_mod_3_l206_20607

theorem infinitely_many_primes_2_mod_3 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_2_mod_3_l206_20607


namespace NUMINAMATH_CALUDE_imaginary_roots_sum_of_magnitudes_l206_20686

theorem imaginary_roots_sum_of_magnitudes (m : ℝ) : 
  (∃ α β : ℂ, (3 * α^2 - 6*(m - 1)*α + m^2 + 1 = 0) ∧ 
               (3 * β^2 - 6*(m - 1)*β + m^2 + 1 = 0) ∧ 
               (α.im ≠ 0) ∧ (β.im ≠ 0) ∧
               (Complex.abs α + Complex.abs β = 2)) →
  m = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_roots_sum_of_magnitudes_l206_20686


namespace NUMINAMATH_CALUDE_sin_double_angle_problem_l206_20644

theorem sin_double_angle_problem (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (π / 2 - α) = 3 / 5) : Real.sin (2 * α) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_problem_l206_20644


namespace NUMINAMATH_CALUDE_num_true_propositions_even_l206_20604

/-- A proposition type representing a logical statement. -/
structure Proposition : Type :=
  (is_true : Bool)

/-- A set of four related propositions (original, converse, inverse, and contrapositive). -/
structure RelatedPropositions : Type :=
  (original : Proposition)
  (converse : Proposition)
  (inverse : Proposition)
  (contrapositive : Proposition)

/-- The number of true propositions in a set of related propositions. -/
def num_true_propositions (rp : RelatedPropositions) : Nat :=
  (if rp.original.is_true then 1 else 0) +
  (if rp.converse.is_true then 1 else 0) +
  (if rp.inverse.is_true then 1 else 0) +
  (if rp.contrapositive.is_true then 1 else 0)

/-- Theorem stating that the number of true propositions in a set of related propositions
    can only be 0, 2, or 4. -/
theorem num_true_propositions_even (rp : RelatedPropositions) :
  num_true_propositions rp = 0 ∨ num_true_propositions rp = 2 ∨ num_true_propositions rp = 4 :=
by sorry

end NUMINAMATH_CALUDE_num_true_propositions_even_l206_20604


namespace NUMINAMATH_CALUDE_video_game_points_l206_20661

/-- 
Given a video game where:
- Each enemy defeated gives 9 points
- There are 11 enemies total in a level
- You destroy all but 3 enemies

Prove that the number of points earned is 72.
-/
theorem video_game_points : 
  (∀ (points_per_enemy : ℕ) (total_enemies : ℕ) (enemies_left : ℕ),
    points_per_enemy = 9 → 
    total_enemies = 11 → 
    enemies_left = 3 → 
    (total_enemies - enemies_left) * points_per_enemy = 72) :=
by sorry

end NUMINAMATH_CALUDE_video_game_points_l206_20661


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l206_20654

/-- An arithmetic sequence with the given property -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_first_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_prop : ∀ n : ℕ+, a (n + 1) + a (n + 2) = 3 * (n : ℚ) + 5) :
  a 1 = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l206_20654


namespace NUMINAMATH_CALUDE_janet_pill_intake_l206_20680

/-- Represents Janet's pill intake schedule for a month --/
structure PillSchedule where
  multivitamins_per_day : ℕ
  calcium_first_two_weeks : ℕ
  calcium_last_two_weeks : ℕ
  weeks_in_month : ℕ

/-- Calculates the total number of pills Janet takes in a month --/
def total_pills (schedule : PillSchedule) : ℕ :=
  let days_per_period := schedule.weeks_in_month / 2 * 7
  let pills_first_two_weeks := (schedule.multivitamins_per_day + schedule.calcium_first_two_weeks) * days_per_period
  let pills_last_two_weeks := (schedule.multivitamins_per_day + schedule.calcium_last_two_weeks) * days_per_period
  pills_first_two_weeks + pills_last_two_weeks

/-- Theorem stating that Janet's total pill intake for the month is 112 --/
theorem janet_pill_intake :
  ∃ (schedule : PillSchedule),
    schedule.multivitamins_per_day = 2 ∧
    schedule.calcium_first_two_weeks = 3 ∧
    schedule.calcium_last_two_weeks = 1 ∧
    schedule.weeks_in_month = 4 ∧
    total_pills schedule = 112 := by
  sorry

end NUMINAMATH_CALUDE_janet_pill_intake_l206_20680


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l206_20619

theorem cyclic_sum_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_sum : x^2 + y^2 + z^2 = 3) :
  (x^2 + y*z) / (x^2 + y*z + 1) + (y^2 + z*x) / (y^2 + z*x + 1) + (z^2 + x*y) / (z^2 + x*y + 1) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l206_20619


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l206_20627

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to x-axis -/
def symmetricXAxis (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

/-- The original point -/
def originalPoint : Point3D :=
  ⟨2, 3, 4⟩

/-- The symmetric point -/
def symmetricPoint : Point3D :=
  ⟨2, -3, -4⟩

theorem symmetric_point_correct : symmetricXAxis originalPoint = symmetricPoint := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l206_20627


namespace NUMINAMATH_CALUDE_quotient_problem_l206_20653

theorem quotient_problem (x : ℝ) (h : x = 0.3) : 0.009 / x = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l206_20653


namespace NUMINAMATH_CALUDE_no_multiple_hundred_scores_l206_20633

/-- Represents the types of wishes available to modify exam scores -/
inductive Wish
  | AddOne      : Wish  -- Add one point to each exam
  | DecreaseOne : Fin 3 → Wish  -- Decrease one exam by 3, increase others by 1

/-- Represents the state of exam scores -/
structure ExamScores where
  russian : ℕ
  physics : ℕ
  math : ℕ
  russian_physics_diff : russian = physics - 3
  physics_math_diff : physics = math - 7

/-- Applies a wish to the exam scores -/
def applyWish (scores : ExamScores) (wish : Wish) : ExamScores :=
  sorry

/-- Checks if more than one exam score is at least 100 -/
def moreThanOneHundred (scores : ExamScores) : Prop :=
  sorry

/-- Main theorem: It's impossible to achieve 100 or more in more than one exam -/
theorem no_multiple_hundred_scores (initial : ExamScores) (wishes : List Wish) :
  ¬∃ (final : ExamScores), (List.foldl applyWish initial wishes = final ∧ moreThanOneHundred final) :=
  sorry

end NUMINAMATH_CALUDE_no_multiple_hundred_scores_l206_20633


namespace NUMINAMATH_CALUDE_seven_valid_positions_l206_20692

/-- Represents a position where an additional square can be attached --/
inductive Position
| CentralExtension
| OuterEdge
| MiddleEdge

/-- Represents the cross-shaped polygon --/
structure CrossPolygon where
  squares : Fin 6 → Unit  -- Represents the 6 squares in the cross
  additional_positions : Fin 11 → Position  -- Represents the 11 possible positions

/-- Represents a configuration with an additional square attached --/
structure ExtendedPolygon where
  base : CrossPolygon
  additional_square_position : Fin 11

/-- Predicate to check if a configuration can be folded into a cube with one face missing --/
def can_fold_to_cube (ep : ExtendedPolygon) : Prop :=
  sorry  -- Definition of this predicate would depend on the geometry of the problem

/-- The main theorem to be proved --/
theorem seven_valid_positions (cp : CrossPolygon) :
  (∃ (valid_positions : Finset (Fin 11)), 
    valid_positions.card = 7 ∧ 
    (∀ p : Fin 11, p ∈ valid_positions ↔ can_fold_to_cube ⟨cp, p⟩)) :=
  sorry


end NUMINAMATH_CALUDE_seven_valid_positions_l206_20692


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l206_20643

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 
  n = 1008 ∧ 
  n % 18 = 0 ∧ 
  1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m : ℕ, m % 18 = 0 → 1000 ≤ m → m < 10000 → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l206_20643


namespace NUMINAMATH_CALUDE_chocolate_cookies_sold_l206_20621

/-- Proves that the number of chocolate cookies sold is 220 --/
theorem chocolate_cookies_sold (price_chocolate : ℕ) (price_vanilla : ℕ) (vanilla_sold : ℕ) (total_revenue : ℕ) :
  price_chocolate = 1 →
  price_vanilla = 2 →
  vanilla_sold = 70 →
  total_revenue = 360 →
  total_revenue = price_chocolate * (total_revenue - price_vanilla * vanilla_sold) + price_vanilla * vanilla_sold →
  total_revenue - price_vanilla * vanilla_sold = 220 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_cookies_sold_l206_20621


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l206_20648

theorem geometric_sequence_eighth_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (fifth_term : a 5 = 11) 
  (eleventh_term : a 11 = 5) : 
  a 8 = Real.sqrt 55 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l206_20648


namespace NUMINAMATH_CALUDE_last_two_digits_of_sequence_sum_l206_20606

/-- The sum of the sequence 8, 88, 888, ..., up to 2008 digits -/
def sequence_sum : ℕ := 8 + 88 * 2007

/-- The last two digits of a number -/
def last_two_digits (n : ℕ) : ℕ := n % 100

/-- Theorem: The last two digits of the sequence sum are 24 -/
theorem last_two_digits_of_sequence_sum :
  last_two_digits sequence_sum = 24 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sequence_sum_l206_20606


namespace NUMINAMATH_CALUDE_expansion_coefficient_l206_20684

/-- The coefficient of the x^(3/2) term in the expansion of (√x - a/√x)^5 -/
def coefficient (a : ℝ) : ℝ := -5 * a

theorem expansion_coefficient (a : ℝ) :
  coefficient a = 30 → a = -6 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l206_20684


namespace NUMINAMATH_CALUDE_min_lines_8x8_grid_is_14_l206_20664

/-- The minimum number of straight lines required to separate all points in an 8x8 grid -/
def min_lines_8x8_grid : ℕ := 14

/-- The number of rows in the grid -/
def num_rows : ℕ := 8

/-- The number of columns in the grid -/
def num_columns : ℕ := 8

/-- The total number of points in the grid -/
def total_points : ℕ := num_rows * num_columns

/-- Theorem stating that the minimum number of lines to separate all points in an 8x8 grid is 14 -/
theorem min_lines_8x8_grid_is_14 : 
  min_lines_8x8_grid = (num_rows - 1) + (num_columns - 1) :=
sorry

end NUMINAMATH_CALUDE_min_lines_8x8_grid_is_14_l206_20664


namespace NUMINAMATH_CALUDE_no_triple_with_three_coprime_roots_l206_20668

theorem no_triple_with_three_coprime_roots : ¬∃ (a b c x₁ x₂ x₃ : ℤ),
  (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
  (Int.gcd x₁ x₂ = 1 ∧ Int.gcd x₁ x₃ = 1 ∧ Int.gcd x₂ x₃ = 1) ∧
  (x₁^3 - a^2*x₁^2 + b^2*x₁ - a*b + 3*c = 0) ∧
  (x₂^3 - a^2*x₂^2 + b^2*x₂ - a*b + 3*c = 0) ∧
  (x₃^3 - a^2*x₃^2 + b^2*x₃ - a*b + 3*c = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_no_triple_with_three_coprime_roots_l206_20668


namespace NUMINAMATH_CALUDE_quadratic_linear_intersection_l206_20697

-- Define the quadratic and linear functions
def quadratic (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 1
def linear (a : ℝ) (x : ℝ) : ℝ := a * x

-- State the theorem
theorem quadratic_linear_intersection :
  ∃ (a b : ℝ),
    (∀ x : ℝ, quadratic a b (-2) = linear a (-2)) ∧
    (quadratic a b (-2) = 1) ∧
    (a = -1/2) ∧ (b = -2) ∧
    (∀ y₁ y₂ y₃ : ℝ,
      (quadratic a b 2 = y₁) →
      (quadratic a b b = y₂) →
      (quadratic a b (a - b) = y₃) →
      (y₁ < y₃ ∧ y₃ < y₂)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_linear_intersection_l206_20697


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l206_20689

theorem arithmetic_geometric_mean_problem (a b : ℝ) 
  (h1 : (a + b) / 2 = 24) 
  (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 110) : 
  a^2 + b^2 = 1424 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l206_20689


namespace NUMINAMATH_CALUDE_herd_size_l206_20691

theorem herd_size (first_son_fraction : ℚ) (second_son_fraction : ℚ) (third_son_fraction : ℚ) 
  (village_cows : ℕ) (fourth_son_cows : ℕ) :
  first_son_fraction = 1/3 →
  second_son_fraction = 1/6 →
  third_son_fraction = 3/10 →
  village_cows = 10 →
  fourth_son_cows = 9 →
  ∃ (total_cows : ℕ), 
    total_cows = 95 ∧
    (first_son_fraction + second_son_fraction + third_son_fraction) * total_cows + 
    village_cows + fourth_son_cows = total_cows :=
by sorry

end NUMINAMATH_CALUDE_herd_size_l206_20691


namespace NUMINAMATH_CALUDE_cos_period_scaled_cos_third_period_l206_20665

/-- The period of cosine function with a scaled argument -/
theorem cos_period_scaled (a : ℝ) (ha : a ≠ 0) : 
  let f := fun x => Real.cos (x / a)
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q ∧ q < p → ∃ x, f (x + q) ≠ f x :=
by
  sorry

/-- The period of y = cos(x/3) is 6π -/
theorem cos_third_period : 
  let f := fun x => Real.cos (x / 3)
  ∃ p : ℝ, p = 6 * Real.pi ∧ p > 0 ∧ ∀ x, f (x + p) = f x ∧ 
    ∀ q, 0 < q ∧ q < p → ∃ x, f (x + q) ≠ f x :=
by
  sorry

end NUMINAMATH_CALUDE_cos_period_scaled_cos_third_period_l206_20665


namespace NUMINAMATH_CALUDE_cube_root_of_64_l206_20670

theorem cube_root_of_64 : ∃ (a : ℝ), a^3 = 64 ∧ a = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l206_20670


namespace NUMINAMATH_CALUDE_window_cost_is_700_l206_20605

/-- The cost of damages caused by Jack -/
def total_damage : ℕ := 1450

/-- The number of tires damaged -/
def num_tires : ℕ := 3

/-- The cost of each tire -/
def tire_cost : ℕ := 250

/-- The cost of the window -/
def window_cost : ℕ := total_damage - (num_tires * tire_cost)

theorem window_cost_is_700 : window_cost = 700 := by
  sorry

end NUMINAMATH_CALUDE_window_cost_is_700_l206_20605


namespace NUMINAMATH_CALUDE_correlation_index_approaching_one_improves_fitting_l206_20656

/-- The correlation index in regression analysis -/
def correlation_index : ℝ → ℝ := sorry

/-- The fitting effect of a regression model -/
def fitting_effect : ℝ → ℝ := sorry

/-- As the correlation index approaches 1, the fitting effect improves -/
theorem correlation_index_approaching_one_improves_fitting :
  ∀ ε > 0, ∃ δ > 0, ∀ r : ℝ,
    1 - δ < correlation_index r → 
    fitting_effect r > fitting_effect 0 + ε :=
sorry

end NUMINAMATH_CALUDE_correlation_index_approaching_one_improves_fitting_l206_20656
