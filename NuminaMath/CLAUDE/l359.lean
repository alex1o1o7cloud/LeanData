import Mathlib

namespace NUMINAMATH_CALUDE_intersection_with_complement_l359_35935

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 2, 3, 4}
def Q : Set Nat := {3, 4, 5}

theorem intersection_with_complement :
  P ∩ (U \ Q) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l359_35935


namespace NUMINAMATH_CALUDE_opposite_vector_with_magnitude_l359_35957

/-- Given two vectors a and b in ℝ², where a is (-1, 2) and b is in the opposite direction
    to a with magnitude √5, prove that b = (1, -2) -/
theorem opposite_vector_with_magnitude (a b : ℝ × ℝ) : 
  a = (-1, 2) →
  ∃ k : ℝ, k < 0 ∧ b = k • a →
  ‖b‖ = Real.sqrt 5 →
  b = (1, -2) :=
by sorry

end NUMINAMATH_CALUDE_opposite_vector_with_magnitude_l359_35957


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l359_35983

def A : Set ℝ := {2, 3}

def B (m : ℝ) : Set ℝ := {x | m * x - 6 = 0}

theorem subset_implies_m_values (m : ℝ) (h : B m ⊆ A) : m = 0 ∨ m = 2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l359_35983


namespace NUMINAMATH_CALUDE_canada_moose_population_l359_35910

/-- The moose population in Canada, in millions -/
def moose_population : ℝ := 1

/-- The beaver population in Canada, in millions -/
def beaver_population : ℝ := 2 * moose_population

/-- The human population in Canada, in millions -/
def human_population : ℝ := 38

theorem canada_moose_population :
  (beaver_population = 2 * moose_population) →
  (human_population = 19 * beaver_population) →
  (human_population = 38) →
  moose_population = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_canada_moose_population_l359_35910


namespace NUMINAMATH_CALUDE_alexander_paintings_l359_35936

/-- The number of paintings at each new gallery --/
def paintings_per_new_gallery : ℕ := 2

theorem alexander_paintings :
  let first_gallery_paintings : ℕ := 9
  let new_galleries : ℕ := 5
  let pencils_per_painting : ℕ := 4
  let signature_pencils_per_gallery : ℕ := 2
  let total_pencils_used : ℕ := 88
  
  paintings_per_new_gallery = 
    ((total_pencils_used - 
      (signature_pencils_per_gallery * (new_galleries + 1)) - 
      (first_gallery_paintings * pencils_per_painting)) 
     / (new_galleries * pencils_per_painting)) :=
by
  sorry

end NUMINAMATH_CALUDE_alexander_paintings_l359_35936


namespace NUMINAMATH_CALUDE_fair_spending_l359_35966

theorem fair_spending (initial_amount : ℝ) (ride_fraction : ℝ) (dessert_cost : ℝ) : 
  initial_amount = 30 →
  ride_fraction = 1/2 →
  dessert_cost = 5 →
  initial_amount - (ride_fraction * initial_amount) - dessert_cost = 10 := by
sorry

end NUMINAMATH_CALUDE_fair_spending_l359_35966


namespace NUMINAMATH_CALUDE_caleb_ice_cream_purchase_l359_35908

/-- The number of cartons of ice cream Caleb bought -/
def ice_cream_cartons : ℕ := sorry

/-- The number of cartons of frozen yoghurt Caleb bought -/
def frozen_yoghurt_cartons : ℕ := 4

/-- The cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 4

/-- The cost of one carton of frozen yoghurt in dollars -/
def frozen_yoghurt_cost : ℕ := 1

/-- The difference in dollars between the total cost of ice cream and frozen yoghurt -/
def cost_difference : ℕ := 36

theorem caleb_ice_cream_purchase : 
  ice_cream_cartons = 10 ∧
  ice_cream_cartons * ice_cream_cost = 
    frozen_yoghurt_cartons * frozen_yoghurt_cost + cost_difference := by
  sorry

end NUMINAMATH_CALUDE_caleb_ice_cream_purchase_l359_35908


namespace NUMINAMATH_CALUDE_minBrokenLine_l359_35987

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def sameSide (A B : Point) (l : Line) : Prop := sorry

def reflectPoint (A : Point) (l : Line) : Point := sorry

def onLine (X : Point) (l : Line) : Prop := sorry

def intersectionPoint (l : Line) (A B : Point) : Point := sorry

def brokenLineLength (A X B : Point) : ℝ := sorry

-- State the theorem
theorem minBrokenLine (l : Line) (A B : Point) :
  sameSide A B l →
  ∃ X : Point, onLine X l ∧
    ∀ Y : Point, onLine Y l →
      brokenLineLength A X B ≤ brokenLineLength A Y B :=
  by
    sorry

end NUMINAMATH_CALUDE_minBrokenLine_l359_35987


namespace NUMINAMATH_CALUDE_product_value_l359_35928

theorem product_value (x y : ℤ) (h1 : x = 12) (h2 : y = 7) :
  (x - y) * (2 * x + 2 * y) = 190 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l359_35928


namespace NUMINAMATH_CALUDE_melted_prism_to_cube_l359_35915

-- Define the prism's properties
def prism_base_area : Real := 16
def prism_height : Real := 4

-- Define the volume of the prism
def prism_volume : Real := prism_base_area * prism_height

-- Define the edge length of the resulting cube
def cube_edge_length : Real := 4

-- Theorem statement
theorem melted_prism_to_cube :
  prism_volume = cube_edge_length ^ 3 :=
by
  sorry

#check melted_prism_to_cube

end NUMINAMATH_CALUDE_melted_prism_to_cube_l359_35915


namespace NUMINAMATH_CALUDE_field_trip_totals_budget_exceeded_l359_35982

/-- Represents the total number of people and cost for the field trip -/
structure FieldTrip where
  students : ℕ
  teachers : ℕ
  parents : ℕ
  cost : ℕ

/-- Calculates the total number of people and cost for a given vehicle type -/
def vehicleTotal (vehicles : ℕ) (studentsPerVehicle : ℕ) (teachersPerVehicle : ℕ) (parentsPerVehicle : ℕ) (costPerVehicle : ℕ) : FieldTrip :=
  { students := vehicles * studentsPerVehicle,
    teachers := vehicles * teachersPerVehicle,
    parents := vehicles * parentsPerVehicle,
    cost := vehicles * costPerVehicle }

/-- Combines two FieldTrip structures -/
def combineFieldTrips (a b : FieldTrip) : FieldTrip :=
  { students := a.students + b.students,
    teachers := a.teachers + b.teachers,
    parents := a.parents + b.parents,
    cost := a.cost + b.cost }

theorem field_trip_totals : 
  let vans := vehicleTotal 6 10 2 1 100
  let minibusses := vehicleTotal 4 24 3 2 200
  let coachBuses := vehicleTotal 2 48 4 4 350
  let schoolBus := vehicleTotal 1 35 5 3 250
  let total := combineFieldTrips (combineFieldTrips (combineFieldTrips vans minibusses) coachBuses) schoolBus
  total.students = 287 ∧ 
  total.teachers = 37 ∧ 
  total.parents = 25 ∧ 
  total.cost = 2350 :=
by sorry

theorem budget_exceeded (budget : ℕ) : 
  let vans := vehicleTotal 6 10 2 1 100
  let minibusses := vehicleTotal 4 24 3 2 200
  let coachBuses := vehicleTotal 2 48 4 4 350
  let schoolBus := vehicleTotal 1 35 5 3 250
  let total := combineFieldTrips (combineFieldTrips (combineFieldTrips vans minibusses) coachBuses) schoolBus
  budget = 2000 →
  total.cost - budget = 350 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_totals_budget_exceeded_l359_35982


namespace NUMINAMATH_CALUDE_keith_books_l359_35999

theorem keith_books (jason_books : ℕ) (total_books : ℕ) (h1 : jason_books = 21) (h2 : total_books = 41) :
  total_books - jason_books = 20 := by
sorry

end NUMINAMATH_CALUDE_keith_books_l359_35999


namespace NUMINAMATH_CALUDE_negation_of_p_is_existential_l359_35976

-- Define the set of even numbers
def A : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 2 * k}

-- Define the proposition p
def p : Prop := ∀ x : ℤ, (2 * x) ∈ A

-- Theorem statement
theorem negation_of_p_is_existential :
  ¬p ↔ ∃ x : ℤ, (2 * x) ∉ A := by sorry

end NUMINAMATH_CALUDE_negation_of_p_is_existential_l359_35976


namespace NUMINAMATH_CALUDE_green_blue_difference_l359_35955

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the count of disks for each color -/
structure DiskCounts where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total number of disks -/
def totalDisks (counts : DiskCounts) : ℕ :=
  counts.blue + counts.yellow + counts.green

/-- Checks if the given counts match the specified ratio -/
def matchesRatio (counts : DiskCounts) (blueRatio yellowRatio greenRatio : ℕ) : Prop :=
  counts.blue * yellowRatio = counts.yellow * blueRatio ∧
  counts.blue * greenRatio = counts.green * blueRatio

theorem green_blue_difference (counts : DiskCounts) :
  totalDisks counts = 72 →
  matchesRatio counts 3 7 8 →
  counts.green - counts.blue = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l359_35955


namespace NUMINAMATH_CALUDE_electron_transfer_for_N2_production_l359_35949

-- Define the chemical elements and compounds
def Zn : Type := Unit
def H : Type := Unit
def N : Type := Unit
def O : Type := Unit
def HNO3 : Type := Unit
def NH4NO3 : Type := Unit
def H2O : Type := Unit
def ZnNO3_2 : Type := Unit

-- Define the reaction
def reaction : Type := Unit

-- Define Avogadro's constant
def Na : ℕ := sorry

-- Define the electron transfer function
def electron_transfer (r : reaction) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem electron_transfer_for_N2_production (r : reaction) :
  electron_transfer r 1 = 5 * Na := by sorry

end NUMINAMATH_CALUDE_electron_transfer_for_N2_production_l359_35949


namespace NUMINAMATH_CALUDE_sqrt_a_minus_2_real_l359_35948

theorem sqrt_a_minus_2_real (a : ℝ) : (∃ x : ℝ, x^2 = a - 2) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_2_real_l359_35948


namespace NUMINAMATH_CALUDE_jordan_oreos_l359_35920

theorem jordan_oreos (total : ℕ) (h1 : total = 36) : ∃ (jordan : ℕ), 
  jordan + (2 * jordan + 3) = total ∧ jordan = 11 := by
  sorry

end NUMINAMATH_CALUDE_jordan_oreos_l359_35920


namespace NUMINAMATH_CALUDE_diamond_area_is_50_l359_35952

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Represents the diamond-shaped region in the square -/
structure DiamondRegion where
  square : Square
  pointA : Point
  pointB : Point

/-- The area of the diamond-shaped region in a 10x10 square -/
def diamondArea (d : DiamondRegion) : ℝ :=
  sorry

theorem diamond_area_is_50 (d : DiamondRegion) : 
  d.square.side = 10 →
  d.pointA.x = 5 ∧ d.pointA.y = 10 →
  d.pointB.x = 5 ∧ d.pointB.y = 0 →
  diamondArea d = 50 := by
  sorry

end NUMINAMATH_CALUDE_diamond_area_is_50_l359_35952


namespace NUMINAMATH_CALUDE_team_selection_ways_eq_8400_l359_35925

/-- The number of ways to select a team of 4 boys from 8 boys and 3 girls from 10 girls -/
def team_selection_ways : ℕ :=
  (Nat.choose 8 4) * (Nat.choose 10 3)

/-- Theorem stating that the number of ways to select the team is 8400 -/
theorem team_selection_ways_eq_8400 : team_selection_ways = 8400 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_eq_8400_l359_35925


namespace NUMINAMATH_CALUDE_figure_area_is_61_l359_35994

/-- Calculates the area of a figure composed of three rectangles -/
def figure_area (rect1_height rect1_width rect2_height rect2_width rect3_height rect3_width : ℕ) : ℕ :=
  rect1_height * rect1_width + rect2_height * rect2_width + rect3_height * rect3_width

/-- The area of the given figure is 61 square units -/
theorem figure_area_is_61 : figure_area 7 6 3 3 2 5 = 61 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_is_61_l359_35994


namespace NUMINAMATH_CALUDE_remaining_trees_l359_35997

/-- Given a park with an initial number of trees, some of which die and others are cut,
    this theorem proves the number of remaining trees. -/
theorem remaining_trees (initial : ℕ) (dead : ℕ) (cut : ℕ) : 
  initial = 86 → dead = 15 → cut = 23 → initial - (dead + cut) = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_trees_l359_35997


namespace NUMINAMATH_CALUDE_jamie_tax_payment_l359_35981

/-- Calculates the tax amount based on a progressive tax system --/
def calculate_tax (gross_income : ℕ) (deduction : ℕ) : ℕ :=
  let taxable_income := gross_income - deduction
  let first_bracket := min taxable_income 150
  let second_bracket := min (taxable_income - 150) 150
  let third_bracket := max (taxable_income - 300) 0
  0 * first_bracket + 
  (10 * second_bracket) / 100 + 
  (15 * third_bracket) / 100

/-- Theorem stating that Jamie's tax payment is $30 --/
theorem jamie_tax_payment : 
  calculate_tax 450 50 = 30 := by
  sorry

#eval calculate_tax 450 50  -- This should output 30

end NUMINAMATH_CALUDE_jamie_tax_payment_l359_35981


namespace NUMINAMATH_CALUDE_computer_price_proof_l359_35906

theorem computer_price_proof (P : ℝ) 
  (h1 : 1.3 * P = 364)
  (h2 : 2 * P = 560) : 
  P = 280 := by sorry

end NUMINAMATH_CALUDE_computer_price_proof_l359_35906


namespace NUMINAMATH_CALUDE_set_operations_l359_35995

def U : Set Int := {x | |x| < 3}
def A : Set Int := {0, 1, 2}
def B : Set Int := {1, 2}

theorem set_operations :
  (A ∪ B = {0, 1, 2}) ∧
  ((U \ A) ∩ (U \ B) = {-2, -1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l359_35995


namespace NUMINAMATH_CALUDE_reciprocal_sum_equals_one_l359_35932

theorem reciprocal_sum_equals_one : 
  1/2 + 1/3 + 1/12 + 1/18 + 1/72 + 1/108 + 1/216 = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equals_one_l359_35932


namespace NUMINAMATH_CALUDE_toothbrushes_per_patient_l359_35927

/-- Calculates the number of toothbrushes given to each patient in a dental office -/
theorem toothbrushes_per_patient
  (hours_per_day : ℝ)
  (hours_per_visit : ℝ)
  (days_per_week : ℕ)
  (total_toothbrushes : ℕ)
  (h1 : hours_per_day = 8)
  (h2 : hours_per_visit = 0.5)
  (h3 : days_per_week = 5)
  (h4 : total_toothbrushes = 160) :
  (total_toothbrushes : ℝ) / ((hours_per_day / hours_per_visit) * days_per_week) = 2 := by
  sorry

end NUMINAMATH_CALUDE_toothbrushes_per_patient_l359_35927


namespace NUMINAMATH_CALUDE_sequence_property_l359_35901

theorem sequence_property (m : ℤ) (a : ℕ → ℤ) (r s : ℕ) :
  (|m| ≥ 2) →
  (a 1 ≠ 0 ∨ a 2 ≠ 0) →
  (∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n) →
  (r > s) →
  (s ≥ 2) →
  (a r = a s) →
  (a r = a 1) →
  (r - s : ℤ) ≥ |m| :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l359_35901


namespace NUMINAMATH_CALUDE_ball_return_ways_formula_l359_35998

/-- The number of ways a ball can return to the starting person after n passes among 7m people. -/
def ball_return_ways (m n : ℕ) : ℚ :=
  (1 / m : ℚ) * ((m - 1 : ℚ)^n + (m - 1 : ℚ) * (-1)^n)

/-- Theorem stating the formula for the number of ways a ball can return to the starting person. -/
theorem ball_return_ways_formula {m n : ℕ} (hm : m ≥ 3) (hn : n ≥ 2) :
  ∃ (c : ℕ → ℚ), c n = ball_return_ways m n :=
by sorry

end NUMINAMATH_CALUDE_ball_return_ways_formula_l359_35998


namespace NUMINAMATH_CALUDE_sandys_pumpkins_l359_35974

/-- Sandy and Mike grew pumpkins. This theorem proves how many pumpkins Sandy grew. -/
theorem sandys_pumpkins (mike_pumpkins total_pumpkins : ℕ) 
  (h1 : mike_pumpkins = 23)
  (h2 : mike_pumpkins + sandy_pumpkins = total_pumpkins)
  (h3 : total_pumpkins = 74) :
  sandy_pumpkins = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_sandys_pumpkins_l359_35974


namespace NUMINAMATH_CALUDE_present_age_of_A_l359_35947

/-- Given the ages of three people A, B, and C, prove that A's present age is 11 years. -/
theorem present_age_of_A (A B C : ℕ) : 
  A + B + C = 57 → 
  ∃ (x : ℕ), A - 3 = x ∧ B - 3 = 2 * x ∧ C - 3 = 3 * x → 
  A = 11 := by
sorry

end NUMINAMATH_CALUDE_present_age_of_A_l359_35947


namespace NUMINAMATH_CALUDE_only_rectangle_both_symmetric_l359_35961

-- Define the shape type
inductive Shape
  | EquilateralTriangle
  | Angle
  | Rectangle
  | Parallelogram

-- Define axisymmetry property
def isAxisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Angle => true
  | Shape.Rectangle => true
  | Shape.Parallelogram => false

-- Define central symmetry property
def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Angle => false
  | Shape.Rectangle => true
  | Shape.Parallelogram => true

-- Theorem stating that only Rectangle is both axisymmetric and centrally symmetric
theorem only_rectangle_both_symmetric :
  ∀ s : Shape, isAxisymmetric s ∧ isCentrallySymmetric s ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_only_rectangle_both_symmetric_l359_35961


namespace NUMINAMATH_CALUDE_train_length_calculation_l359_35945

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
theorem train_length_calculation (jogger_speed train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ)
    (h1 : jogger_speed = 9 / 3.6) -- Convert 9 km/hr to m/s
    (h2 : train_speed = 45 / 3.6) -- Convert 45 km/hr to m/s
    (h3 : initial_distance = 240)
    (h4 : passing_time = 35) :
    train_speed * passing_time - jogger_speed * passing_time - initial_distance = 110 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l359_35945


namespace NUMINAMATH_CALUDE_quadratic_equation_shift_l359_35942

theorem quadratic_equation_shift (a h k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3 ∧ 
   ∀ x : ℝ, a * (x - h)^2 + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ y₁ y₂ : ℝ, y₁ = 0 ∧ y₂ = 4 ∧ 
   ∀ y : ℝ, a * (y - h - 1)^2 + k = 0 ↔ y = y₁ ∨ y = y₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_shift_l359_35942


namespace NUMINAMATH_CALUDE_james_car_sale_l359_35964

/-- The percentage at which James sold his car -/
def sell_percentage : ℝ → Prop := λ P =>
  let old_car_value : ℝ := 20000
  let new_car_sticker : ℝ := 30000
  let new_car_discount : ℝ := 0.9
  let out_of_pocket : ℝ := 11000
  new_car_sticker * new_car_discount - old_car_value * (P / 100) = out_of_pocket

theorem james_car_sale : 
  sell_percentage 80 := by sorry

end NUMINAMATH_CALUDE_james_car_sale_l359_35964


namespace NUMINAMATH_CALUDE_exam_disturbance_probability_l359_35907

theorem exam_disturbance_probability :
  let n : ℕ := 6  -- number of students
  let p_undisturbed : ℚ := 2 / n * 2 / (n - 1) * 2 / (n - 2) * 2 / (n - 3)
  (1 : ℚ) - p_undisturbed = 43 / 45 :=
by sorry

end NUMINAMATH_CALUDE_exam_disturbance_probability_l359_35907


namespace NUMINAMATH_CALUDE_glued_cubes_surface_area_l359_35965

/-- Represents a 3D shape formed by two glued cubes -/
structure GluedCubes where
  large_cube_side : ℝ
  small_cube_side : ℝ
  glued : Bool

/-- Calculate the surface area of the GluedCubes shape -/
def surface_area (shape : GluedCubes) : ℝ :=
  let large_cube_area := 6 * shape.large_cube_side ^ 2
  let small_cube_area := 5 * shape.small_cube_side ^ 2
  large_cube_area + small_cube_area

/-- The theorem stating that the surface area of the specific GluedCubes shape is 74 -/
theorem glued_cubes_surface_area :
  let shape := GluedCubes.mk 3 1 true
  surface_area shape = 74 := by
  sorry

end NUMINAMATH_CALUDE_glued_cubes_surface_area_l359_35965


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_chord_length_l359_35978

/-- Given a parabola and a circle, prove the length of the chord formed by their intersection -/
theorem parabola_circle_intersection_chord_length :
  ∀ (p : ℝ) (x y : ℝ → ℝ),
    p > 0 →
    (∀ t, y t ^ 2 = 2 * p * x t) →
    (∀ t, (x t - 1) ^ 2 + (y t + 2) ^ 2 = 9) →
    x 0 = 1 ∧ y 0 = -2 →
    ∃ (a b : ℝ), a ≠ b ∧
      x a = -1 ∧ x b = -1 ∧
      (x a - 1) ^ 2 + (y a + 2) ^ 2 = 9 ∧
      (x b - 1) ^ 2 + (y b + 2) ^ 2 = 9 ∧
      (y a - y b) ^ 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_chord_length_l359_35978


namespace NUMINAMATH_CALUDE_min_face_sum_is_16_l359_35960

/-- Represents the arrangement of numbers on a cube's vertices -/
def CubeArrangement := Fin 8 → Fin 8

/-- Check if a given arrangement satisfies the condition that the sum of any three vertices on a face is at least 10 -/
def ValidArrangement (arr : CubeArrangement) : Prop :=
  ∀ (face : Fin 6) (v1 v2 v3 : Fin 4), v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3 →
    (arr (face * 4 + v1) + arr (face * 4 + v2) + arr (face * 4 + v3) : ℕ) ≥ 10

/-- Calculate the sum of numbers on a given face -/
def FaceSum (arr : CubeArrangement) (face : Fin 6) : ℕ :=
  (arr (face * 4) : ℕ) + (arr (face * 4 + 1) : ℕ) + (arr (face * 4 + 2) : ℕ) + (arr (face * 4 + 3) : ℕ)

/-- The main theorem stating that the minimal possible sum on any face is 16 -/
theorem min_face_sum_is_16 :
  ∃ (arr : CubeArrangement), ValidArrangement arr ∧
    (∀ (arr' : CubeArrangement), ValidArrangement arr' →
      ∀ (face : Fin 6), FaceSum arr face ≤ FaceSum arr' face) ∧
    (∃ (face : Fin 6), FaceSum arr face = 16) :=
  sorry

end NUMINAMATH_CALUDE_min_face_sum_is_16_l359_35960


namespace NUMINAMATH_CALUDE_unique_integer_solution_implies_a_range_l359_35958

theorem unique_integer_solution_implies_a_range (a : ℝ) :
  (∃! x : ℤ, (2 * x + 3 > 5 ∧ x - a ≤ 0)) →
  (2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_implies_a_range_l359_35958


namespace NUMINAMATH_CALUDE_pauls_books_count_paul_has_151_books_l359_35933

/-- Calculates the total number of books Paul has after buying new ones -/
def total_books (initial_books new_books : ℕ) : ℕ :=
  initial_books + new_books

/-- Theorem: Paul's total books equal the sum of initial and new books -/
theorem pauls_books_count (initial_books new_books : ℕ) :
  total_books initial_books new_books = initial_books + new_books :=
by sorry

/-- Theorem: Paul now has 151 books -/
theorem paul_has_151_books :
  total_books 50 101 = 151 :=
by sorry

end NUMINAMATH_CALUDE_pauls_books_count_paul_has_151_books_l359_35933


namespace NUMINAMATH_CALUDE_unique_identical_lines_l359_35996

theorem unique_identical_lines : 
  ∃! (a d : ℝ), ∀ (x y : ℝ), (2 * x + a * y + 4 = 0 ↔ d * x - 3 * y + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_identical_lines_l359_35996


namespace NUMINAMATH_CALUDE_exists_m_for_inequality_l359_35970

def sequence_a : ℕ → ℚ
  | 7 => 16/3
  | n+1 => (3 * sequence_a n + 4) / (7 - sequence_a n)
  | _ => 0  -- Define for n < 7 to make the function total

theorem exists_m_for_inequality :
  ∃ m : ℕ, ∀ n ≥ m, sequence_a n > (sequence_a (n-1) + sequence_a (n+1)) / 2 :=
sorry

end NUMINAMATH_CALUDE_exists_m_for_inequality_l359_35970


namespace NUMINAMATH_CALUDE_combined_savings_l359_35900

/-- Calculate combined savings of three employees over four weeks -/
theorem combined_savings (hourly_wage : ℚ) (hours_per_day : ℚ) (days_per_week : ℚ) (weeks : ℚ)
  (robby_save_ratio : ℚ) (jaylen_save_ratio : ℚ) (miranda_save_ratio : ℚ)
  (h1 : hourly_wage = 10)
  (h2 : hours_per_day = 10)
  (h3 : days_per_week = 5)
  (h4 : weeks = 4)
  (h5 : robby_save_ratio = 2/5)
  (h6 : jaylen_save_ratio = 3/5)
  (h7 : miranda_save_ratio = 1/2) :
  let monthly_salary := hourly_wage * hours_per_day * days_per_week * weeks
  let robby_savings := robby_save_ratio * monthly_salary
  let jaylen_savings := jaylen_save_ratio * monthly_salary
  let miranda_savings := miranda_save_ratio * monthly_salary
  robby_savings + jaylen_savings + miranda_savings = 3000 := by
  sorry

end NUMINAMATH_CALUDE_combined_savings_l359_35900


namespace NUMINAMATH_CALUDE_divisibility_theorem_l359_35938

theorem divisibility_theorem (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  ∃ m : ℤ, (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4*k - 1) = m * (n^5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l359_35938


namespace NUMINAMATH_CALUDE_correct_answers_for_given_exam_l359_35940

/-- Represents an exam with a fixed number of questions and scoring rules. -/
structure Exam where
  totalQuestions : ℕ
  correctScore : ℕ
  wrongScore : ℤ

/-- Represents a student's exam attempt. -/
structure ExamAttempt where
  exam : Exam
  correctAnswers : ℕ
  wrongAnswers : ℕ
  totalScore : ℤ

/-- Calculates the total score for an exam attempt. -/
def calculateScore (attempt : ExamAttempt) : ℤ :=
  (attempt.correctAnswers : ℤ) * attempt.exam.correctScore - attempt.wrongAnswers * (-attempt.exam.wrongScore)

/-- Theorem stating the correct number of answers for the given exam conditions. -/
theorem correct_answers_for_given_exam :
  ∀ (attempt : ExamAttempt),
    attempt.exam.totalQuestions = 75 →
    attempt.exam.correctScore = 4 →
    attempt.exam.wrongScore = -1 →
    attempt.correctAnswers + attempt.wrongAnswers = attempt.exam.totalQuestions →
    calculateScore attempt = 125 →
    attempt.correctAnswers = 40 := by
  sorry


end NUMINAMATH_CALUDE_correct_answers_for_given_exam_l359_35940


namespace NUMINAMATH_CALUDE_problem_solution_l359_35919

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (-a*b)^2009 - (c+d)^2010 - e^2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l359_35919


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l359_35922

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_f_odd_and_decreasing_l359_35922


namespace NUMINAMATH_CALUDE_fraction_is_composite_l359_35967

theorem fraction_is_composite : ¬ Nat.Prime ((5^125 - 1) / (5^25 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_composite_l359_35967


namespace NUMINAMATH_CALUDE_all_symmetry_statements_correct_l359_35918

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define symmetry with respect to y-axis
def symmetric_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define symmetry with respect to x-axis
def symmetric_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f x

-- Define symmetry with respect to origin
def symmetric_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- Define symmetry with respect to vertical line x = a
def symmetric_vertical_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem all_symmetry_statements_correct (f : ℝ → ℝ) : 
  (symmetric_y_axis f) ∧ 
  (symmetric_x_axis f) ∧ 
  (symmetric_origin f) ∧ 
  (∀ a : ℝ, symmetric_vertical_line f a → 
    ∃ g : ℝ → ℝ, ∀ x, g (x - a) = f x) :=
by sorry

end NUMINAMATH_CALUDE_all_symmetry_statements_correct_l359_35918


namespace NUMINAMATH_CALUDE_max_m_and_a_value_l359_35937

/-- The function f(x) = |x+3| -/
def f (x : ℝ) : ℝ := |x + 3|

/-- The function g(x) = m - 2|x-11| -/
def g (m : ℝ) (x : ℝ) : ℝ := m - 2*|x - 11|

/-- The theorem stating the maximum value of m and the value of a -/
theorem max_m_and_a_value :
  (∀ x : ℝ, 2 * f x ≥ g m (x + 4)) →
  (∃ t : ℝ, t = 20 ∧ 
    (∀ m' : ℝ, (∀ x : ℝ, 2 * f x ≥ g m' (x + 4)) → m' ≤ t) ∧
    (∀ a : ℝ, a > 0 →
      (∃ x y z : ℝ, 2*x^2 + 3*y^2 + 6*z^2 = a ∧ 
        x + y + z = t/20 ∧
        (∀ x' y' z' : ℝ, 2*x'^2 + 3*y'^2 + 6*z'^2 = a → x' + y' + z' ≤ t/20)) →
      a = 1)) :=
sorry

end NUMINAMATH_CALUDE_max_m_and_a_value_l359_35937


namespace NUMINAMATH_CALUDE_piano_lesson_rate_piano_rate_is_28_l359_35953

/-- Calculates the hourly rate for piano lessons given the conditions -/
theorem piano_lesson_rate (clarinet_rate : ℝ) (clarinet_hours : ℝ) (piano_hours : ℝ) 
  (extra_piano_cost : ℝ) (weeks_per_year : ℕ) : ℝ :=
  let annual_clarinet_cost := clarinet_rate * clarinet_hours * weeks_per_year
  let annual_piano_cost := annual_clarinet_cost + extra_piano_cost
  annual_piano_cost / (piano_hours * weeks_per_year)

/-- The hourly rate for piano lessons is $28 -/
theorem piano_rate_is_28 : 
  piano_lesson_rate 40 3 5 1040 52 = 28 := by
sorry

end NUMINAMATH_CALUDE_piano_lesson_rate_piano_rate_is_28_l359_35953


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l359_35968

theorem not_necessarily_right_triangle (A B C : ℝ) (h : A / B = 3 / 4 ∧ B / C = 4 / 5) :
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l359_35968


namespace NUMINAMATH_CALUDE_younger_person_age_l359_35986

/-- Given two people's ages, proves that the younger person is 12 years old --/
theorem younger_person_age
  (total_age : ℕ)
  (age_difference : ℕ)
  (h1 : total_age = 30)
  (h2 : age_difference = 6) :
  (total_age - age_difference) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_younger_person_age_l359_35986


namespace NUMINAMATH_CALUDE_intersects_x_axis_once_l359_35929

/-- A function f(x) = (k-3)x^2 + 2x + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x^2 + 2 * x + 1

/-- The condition for a quadratic function to have exactly one root -/
def has_one_root (k : ℝ) : Prop :=
  (k = 3) ∨ (4 * (k - 3) * 1 = 2^2)

theorem intersects_x_axis_once (k : ℝ) :
  (∃! x, f k x = 0) ↔ has_one_root k := by sorry

end NUMINAMATH_CALUDE_intersects_x_axis_once_l359_35929


namespace NUMINAMATH_CALUDE_fruit_lovers_count_l359_35921

/-- The number of people who like apple -/
def apple_lovers : ℕ := 40

/-- The number of people who like orange and mango but dislike apple -/
def orange_mango_lovers : ℕ := 7

/-- The number of people who like mango and apple and dislike orange -/
def apple_mango_lovers : ℕ := 10

/-- The total number of people who like apple -/
def total_apple_lovers : ℕ := 47

/-- The number of people who like all three fruits (apple, orange, and mango) -/
def all_fruit_lovers : ℕ := 3

theorem fruit_lovers_count : 
  apple_lovers + (apple_mango_lovers - all_fruit_lovers) + all_fruit_lovers = total_apple_lovers :=
by sorry

end NUMINAMATH_CALUDE_fruit_lovers_count_l359_35921


namespace NUMINAMATH_CALUDE_f_order_l359_35973

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 6 * m * x + 2

-- State the theorem
theorem f_order (m : ℝ) (h : ∀ x, f m x = f m (-x)) :
  f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
  sorry

end NUMINAMATH_CALUDE_f_order_l359_35973


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l359_35992

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 1)
  let b : ℝ × ℝ := (x, 2)
  are_parallel a b → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l359_35992


namespace NUMINAMATH_CALUDE_ceiling_times_self_182_l359_35930

theorem ceiling_times_self_182 :
  ∃! (x : ℝ), ⌈x⌉ * x = 182 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ceiling_times_self_182_l359_35930


namespace NUMINAMATH_CALUDE_det_positive_for_special_matrix_l359_35926

open Matrix

theorem det_positive_for_special_matrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) 
  (h : A + Aᵀ = 1) : 
  0 < det A := by
  sorry

end NUMINAMATH_CALUDE_det_positive_for_special_matrix_l359_35926


namespace NUMINAMATH_CALUDE_ahmed_orange_trees_count_l359_35911

-- Define the number of apple and orange trees for Hassan
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2

-- Define the number of apple trees for Ahmed
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees

-- Define the total number of trees for Hassan
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Define the relationship between Ahmed's and Hassan's total trees
def ahmed_total_trees (ahmed_orange_trees : ℕ) : ℕ := 
  ahmed_apple_trees + ahmed_orange_trees

-- Theorem stating that Ahmed has 8 orange trees
theorem ahmed_orange_trees_count : 
  ∃ (x : ℕ), ahmed_total_trees x = hassan_total_trees + 9 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_orange_trees_count_l359_35911


namespace NUMINAMATH_CALUDE_garden_fencing_theorem_l359_35977

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.width)

theorem garden_fencing_theorem :
  ∀ (garden : RectangularGarden),
    garden.length = 50 →
    garden.length = 2 * garden.width →
    perimeter garden = 150 := by
  sorry

end NUMINAMATH_CALUDE_garden_fencing_theorem_l359_35977


namespace NUMINAMATH_CALUDE_unique_solution_n_times_s_l359_35909

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x * f y + 2 * x) = 2 * x * y + f x

/-- The theorem stating that f(3) = -2 is the only solution -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 3 = -2 := by
  sorry

/-- The number of possible values for f(3) -/
def n : ℕ := 1

/-- The sum of all possible values for f(3) -/
def s : ℝ := -2

/-- The product of n and s -/
theorem n_times_s : n * s = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_n_times_s_l359_35909


namespace NUMINAMATH_CALUDE_cucumber_salad_problem_l359_35917

theorem cucumber_salad_problem (total : ℕ) (ratio : ℕ) : 
  total = 280 → ratio = 3 → ∃ (cucumbers : ℕ), cucumbers * (ratio + 1) = total ∧ cucumbers = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_cucumber_salad_problem_l359_35917


namespace NUMINAMATH_CALUDE_car_fuel_tank_cost_l359_35951

/-- Proves that the cost to fill a car fuel tank is $45 given specific conditions -/
theorem car_fuel_tank_cost : ∃ (F : ℚ),
  (2000 / 500 : ℚ) * F + (3/5) * ((2000 / 500 : ℚ) * F) = 288 ∧ F = 45 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_tank_cost_l359_35951


namespace NUMINAMATH_CALUDE_trout_division_l359_35990

theorem trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) : 
  total_trout = 18 → num_people = 2 → trout_per_person = total_trout / num_people → trout_per_person = 9 := by
  sorry

end NUMINAMATH_CALUDE_trout_division_l359_35990


namespace NUMINAMATH_CALUDE_second_win_proof_l359_35993

/-- Represents the financial transactions of a man and calculates the amount won in the second round --/
def calculate_second_win (initial_amount : ℚ) (first_win : ℚ) : ℚ :=
  let after_first_loss := initial_amount * (2/3)
  let after_first_win := after_first_loss + first_win
  let after_second_loss := after_first_win * (2/3)
  initial_amount - after_second_loss

/-- Proves that the calculated second win amount results in the initial amount --/
theorem second_win_proof (initial_amount : ℚ) (first_win : ℚ) :
  let second_win := calculate_second_win initial_amount first_win
  let final_amount := (((initial_amount * (2/3) + first_win) * (2/3)) + second_win)
  initial_amount = 48.00000000000001 ∧ first_win = 10 →
  final_amount = initial_amount ∧ second_win = 20 := by
  sorry

#eval calculate_second_win 48.00000000000001 10

end NUMINAMATH_CALUDE_second_win_proof_l359_35993


namespace NUMINAMATH_CALUDE_product_remainder_l359_35905

def product : ℕ := 3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93

theorem product_remainder (n : ℕ) (h : n = product) : n % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l359_35905


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l359_35914

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 15.625

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 25

theorem bowling_ball_weight_proof :
  (8 * bowling_ball_weight = 5 * canoe_weight) ∧
  (4 * canoe_weight = 100) →
  bowling_ball_weight = 15.625 := by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l359_35914


namespace NUMINAMATH_CALUDE_probability_less_than_20_l359_35902

theorem probability_less_than_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 120) (h2 : over_30 = 90) :
  let under_20 := total - over_30
  (under_20 : ℚ) / total = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_20_l359_35902


namespace NUMINAMATH_CALUDE_item_frequency_proof_l359_35904

theorem item_frequency_proof (total : ℕ) (second_grade : ℕ) 
  (h1 : total = 400) (h2 : second_grade = 20) : 
  let first_grade := total - second_grade
  (first_grade : ℚ) / total = 95 / 100 ∧ 
  (second_grade : ℚ) / total = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_item_frequency_proof_l359_35904


namespace NUMINAMATH_CALUDE_two_thirds_in_M_l359_35950

open Set

-- Define the sets A and B as open intervals
def A : Set ℝ := Ioo (-4) 1
def B : Set ℝ := Ioo (-2) 5

-- Define M as the intersection of A and B
def M : Set ℝ := A ∩ B

-- Theorem statement
theorem two_thirds_in_M : (2/3 : ℝ) ∈ M := by sorry

end NUMINAMATH_CALUDE_two_thirds_in_M_l359_35950


namespace NUMINAMATH_CALUDE_square_of_sum_l359_35972

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l359_35972


namespace NUMINAMATH_CALUDE_pi_irrational_less_than_neg_three_l359_35943

theorem pi_irrational_less_than_neg_three : 
  Irrational (-Real.pi) ∧ -Real.pi < -3 := by sorry

end NUMINAMATH_CALUDE_pi_irrational_less_than_neg_three_l359_35943


namespace NUMINAMATH_CALUDE_van_distance_theorem_l359_35956

def distance_covered (initial_time : ℝ) (new_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  new_speed * (initial_time * time_ratio)

theorem van_distance_theorem (initial_time : ℝ) (new_speed : ℝ) (time_ratio : ℝ) :
  initial_time = 5 →
  new_speed = 80 →
  time_ratio = 3/2 →
  distance_covered initial_time new_speed time_ratio = 600 := by
    sorry

end NUMINAMATH_CALUDE_van_distance_theorem_l359_35956


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_8_l359_35931

def binary_number : ℕ := 101110100101

theorem remainder_of_binary_div_8 : 
  binary_number % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_8_l359_35931


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l359_35954

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36 = 0

/-- The distance between vertices of the hyperbola -/
def vertex_distance : ℝ := 1

/-- Theorem: The distance between the vertices of the hyperbola given by the equation
    16x^2 + 64x - 4y^2 + 8y + 36 = 0 is equal to 1 -/
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → vertex_distance = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l359_35954


namespace NUMINAMATH_CALUDE_complex_division_fourth_quadrant_l359_35913

theorem complex_division_fourth_quadrant : 
  let i : ℂ := Complex.I
  let z₁ : ℂ := 1 + i
  let z₂ : ℂ := 1 + 2*i
  (z₁ / z₂).re > 0 ∧ (z₁ / z₂).im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_division_fourth_quadrant_l359_35913


namespace NUMINAMATH_CALUDE_angle_rotation_l359_35969

theorem angle_rotation (initial_angle rotation : ℝ) (h1 : initial_angle = 60) (h2 : rotation = 420) :
  (initial_angle - (rotation % 360)) % 360 = 0 :=
by sorry

end NUMINAMATH_CALUDE_angle_rotation_l359_35969


namespace NUMINAMATH_CALUDE_m_range_l359_35941

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, m*x^2 - x + (1/16)*m > 0

-- Define the set of m satisfying the conditions
def S : Set ℝ := {m | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- Theorem statement
theorem m_range : S = {m | 1 < m ∧ m ≤ 2} := by sorry

end NUMINAMATH_CALUDE_m_range_l359_35941


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l359_35923

/-- The function f(x) = x^4 - 2x^3 -/
def f (x : ℝ) : ℝ := x^4 - 2*x^3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4*x^3 - 6*x^2

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2*x + 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l359_35923


namespace NUMINAMATH_CALUDE_line_intercept_ratio_l359_35924

theorem line_intercept_ratio (b : ℝ) (u v : ℝ) 
  (h1 : b ≠ 0)
  (h2 : 0 = 8 * u + b)
  (h3 : 0 = 4 * v + b) :
  u / v = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_ratio_l359_35924


namespace NUMINAMATH_CALUDE_original_mean_calculation_l359_35985

theorem original_mean_calculation (n : ℕ) (decrease : ℝ) (updated_mean : ℝ) :
  n = 50 →
  decrease = 6 →
  updated_mean = 194 →
  (updated_mean + decrease : ℝ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_mean_calculation_l359_35985


namespace NUMINAMATH_CALUDE_some_employees_not_team_leaders_l359_35944

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Employee : U → Prop)
variable (TeamLeader : U → Prop)
variable (MeetsDeadlines : U → Prop)

-- State the theorem
theorem some_employees_not_team_leaders
  (h1 : ∃ x, Employee x ∧ ¬MeetsDeadlines x)
  (h2 : ∀ x, TeamLeader x → MeetsDeadlines x) :
  ∃ x, Employee x ∧ ¬TeamLeader x :=
by
  sorry

end NUMINAMATH_CALUDE_some_employees_not_team_leaders_l359_35944


namespace NUMINAMATH_CALUDE_first_day_duration_l359_35962

def total_distance : ℝ := 115

def day2_distance : ℝ := 6 * 6 + 3 * 3

def day3_distance : ℝ := 7 * 5

def day1_speed : ℝ := 5

theorem first_day_duration : ∃ (hours : ℝ), 
  hours * day1_speed + day2_distance + day3_distance = total_distance ∧ hours = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_day_duration_l359_35962


namespace NUMINAMATH_CALUDE_champagne_glasses_per_guest_l359_35963

/-- Calculates the number of champagne glasses per guest at Ashley's wedding. -/
theorem champagne_glasses_per_guest :
  let num_guests : ℕ := 120
  let servings_per_bottle : ℕ := 6
  let num_bottles : ℕ := 40
  let total_servings : ℕ := num_bottles * servings_per_bottle
  let glasses_per_guest : ℕ := total_servings / num_guests
  glasses_per_guest = 2 := by
  sorry

end NUMINAMATH_CALUDE_champagne_glasses_per_guest_l359_35963


namespace NUMINAMATH_CALUDE_f_is_algebraic_fraction_l359_35975

/-- An algebraic fraction is a ratio of algebraic expressions. -/
def is_algebraic_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, d x ≠ 0 → f x = (n x) / (d x)

/-- The function f(x) = 2/(x+3) for x ≠ -3 -/
def f (x : ℚ) : ℚ := 2 / (x + 3)

/-- Theorem: f(x) = 2/(x+3) is an algebraic fraction -/
theorem f_is_algebraic_fraction : is_algebraic_fraction f :=
sorry

end NUMINAMATH_CALUDE_f_is_algebraic_fraction_l359_35975


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l359_35903

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : Real.log (a + b) = 0) :
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.log (x + y) = 0 ∧ 1 / x + 1 / y = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l359_35903


namespace NUMINAMATH_CALUDE_seeds_per_can_l359_35984

def total_seeds : ℕ := 54
def num_cans : ℕ := 9

theorem seeds_per_can :
  total_seeds / num_cans = 6 :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_can_l359_35984


namespace NUMINAMATH_CALUDE_center_is_B_l359_35991

-- Define the points
variable (A B C D P Q K L : ℝ × ℝ)

-- Define the conditions
axiom on_line : ∃ (t : ℝ), A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1 ∧
  B = (1 - t) • A + t • D ∧
  C = (1 - t) • B + t • D

axiom AB_eq_BC : dist A B = dist B C

axiom perp_B : (B.2 - A.2) * (P.1 - B.1) = (B.1 - A.1) * (P.2 - B.2) ∧
               (B.2 - A.2) * (Q.1 - B.1) = (B.1 - A.1) * (Q.2 - B.2)

axiom perp_C : (C.2 - B.2) * (K.1 - C.1) = (C.1 - B.1) * (K.2 - C.2) ∧
               (C.2 - B.2) * (L.1 - C.1) = (C.1 - B.1) * (L.2 - C.2)

axiom on_circle_AD : dist A P + dist P D = dist A D ∧
                     dist A Q + dist Q D = dist A D

axiom on_circle_BD : dist B K + dist K D = dist B D ∧
                     dist B L + dist L D = dist B D

-- State the theorem
theorem center_is_B : 
  dist B P = dist B K ∧ dist B K = dist B L ∧ dist B L = dist B Q :=
sorry

end NUMINAMATH_CALUDE_center_is_B_l359_35991


namespace NUMINAMATH_CALUDE_min_distance_point_to_line_l359_35988

theorem min_distance_point_to_line (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n - m - n = 3) :
  let d := |m + n| / Real.sqrt 2
  d ≥ 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_distance_point_to_line_l359_35988


namespace NUMINAMATH_CALUDE_problem_solution_l359_35989

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_product : x * y * z = 1)
  (h_eq1 : x + 1 / z = 10)
  (h_eq2 : y + 1 / x = 5) :
  z + 1 / y = 17 / 49 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l359_35989


namespace NUMINAMATH_CALUDE_milk_fraction_problem_l359_35946

theorem milk_fraction_problem (V : ℝ) (h : V > 0) :
  let x := (3 : ℝ) / 5
  let second_cup_milk := (4 : ℝ) / 5 * V
  let second_cup_water := V - second_cup_milk
  let total_milk := x * V + second_cup_milk
  let total_water := (1 - x) * V + second_cup_water
  (total_water / total_milk = 3 / 7) → x = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_milk_fraction_problem_l359_35946


namespace NUMINAMATH_CALUDE_unique_train_journey_l359_35934

/-- Represents a day of the week -/
inductive DayOfWeek
| Saturday
| Sunday
| Monday

/-- Represents the train journey details -/
structure TrainJourney where
  carNumber : Nat
  seatNumber : Nat
  saturdayDate : Nat
  mondayDate : Nat

/-- Checks if the journey satisfies all given conditions -/
def isValidJourney (journey : TrainJourney) : Prop :=
  journey.seatNumber < journey.carNumber ∧
  journey.saturdayDate > journey.carNumber ∧
  journey.mondayDate = journey.carNumber

theorem unique_train_journey : 
  ∀ (journey : TrainJourney), 
    isValidJourney journey → 
    journey.carNumber = 2 ∧ journey.seatNumber = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_train_journey_l359_35934


namespace NUMINAMATH_CALUDE_m_equals_one_iff_z_purely_imaginary_l359_35916

-- Define a complex number
def z (m : ℝ) : ℂ := m^2 * (1 + Complex.I) + m * (Complex.I - 1)

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

-- State the theorem
theorem m_equals_one_iff_z_purely_imaginary :
  ∀ m : ℝ, m = 1 ↔ isPurelyImaginary (z m) := by sorry

end NUMINAMATH_CALUDE_m_equals_one_iff_z_purely_imaginary_l359_35916


namespace NUMINAMATH_CALUDE_percentage_difference_l359_35912

theorem percentage_difference : (0.4 * 60) - (4/5 * 25) = 4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l359_35912


namespace NUMINAMATH_CALUDE_solution_characterization_l359_35979

/-- The set of polynomials that satisfy the given condition -/
def SolutionSet : Set (Polynomial ℤ) :=
  {f | f = Polynomial.monomial 3 1 + Polynomial.monomial 2 1 + Polynomial.monomial 1 1 + Polynomial.monomial 0 1 ∨
       f = Polynomial.monomial 3 1 + Polynomial.monomial 2 2 + Polynomial.monomial 1 2 + Polynomial.monomial 0 2 ∨
       f = Polynomial.monomial 3 2 + Polynomial.monomial 2 1 + Polynomial.monomial 1 2 + Polynomial.monomial 0 1 ∨
       f = Polynomial.monomial 3 2 + Polynomial.monomial 2 2 + Polynomial.monomial 1 1 + Polynomial.monomial 0 2}

/-- The condition that f must satisfy -/
def SatisfiesCondition (f : Polynomial ℤ) : Prop :=
  ∃ g h : Polynomial ℤ, f^4 + 2*f + 2 = (Polynomial.monomial 4 1 + 2*Polynomial.monomial 2 1 + 2)*g + 3*h

theorem solution_characterization :
  ∀ f : Polynomial ℤ, (f ∈ SolutionSet ↔ (SatisfiesCondition f ∧ 
    ∀ f' : Polynomial ℤ, SatisfiesCondition f' → (Polynomial.degree f' ≥ Polynomial.degree f))) :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l359_35979


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l359_35980

theorem smallest_number_with_given_remainders : ∃ (n : ℕ), 
  (n % 19 = 9 ∧ n % 23 = 7) ∧ 
  (∀ m : ℕ, m % 19 = 9 ∧ m % 23 = 7 → n ≤ m) ∧
  n = 161 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l359_35980


namespace NUMINAMATH_CALUDE_mike_pears_l359_35959

theorem mike_pears (jason_pears keith_pears total_pears : ℕ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : total_pears = 105)
  (h4 : ∃ mike_pears : ℕ, jason_pears + keith_pears + mike_pears = total_pears) :
  ∃ mike_pears : ℕ, mike_pears = 12 ∧ jason_pears + keith_pears + mike_pears = total_pears := by
sorry

end NUMINAMATH_CALUDE_mike_pears_l359_35959


namespace NUMINAMATH_CALUDE_complex_equation_solution_l359_35939

theorem complex_equation_solution (x y : ℂ) (hx : x ≠ 0) (hxy : x + 2*y ≠ 0) :
  (x + 2*y) / x = 2*y / (x + 2*y) →
  (x = -y + Complex.I * Real.sqrt 3 * y) ∨ (x = -y - Complex.I * Real.sqrt 3 * y) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l359_35939


namespace NUMINAMATH_CALUDE_sector_angle_l359_35971

theorem sector_angle (R : ℝ) (α : ℝ) : 
  R > 0 ∧ 2 * R + α * R = 6 ∧ (1/2) * R^2 * α = 2 → α = 1 ∨ α = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l359_35971
