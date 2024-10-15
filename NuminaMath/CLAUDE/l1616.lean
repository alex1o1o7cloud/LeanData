import Mathlib

namespace NUMINAMATH_CALUDE_square_area_increase_l1616_161641

/-- The increase in area of a square when its side length increases by 6 -/
theorem square_area_increase (a : ℝ) : 
  (a + 6)^2 - a^2 = 12*a + 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l1616_161641


namespace NUMINAMATH_CALUDE_min_money_lost_l1616_161668

def check_amount : ℕ := 1270
def bill_10 : ℕ := 10
def bill_50 : ℕ := 50
def total_bills_used : ℕ := 15

def money_lost (t f : ℕ) : ℕ :=
  check_amount - (t * bill_10 + f * bill_50)

theorem min_money_lost :
  ∃ (t f : ℕ),
    (t + f = total_bills_used) ∧
    ((t = f + 1) ∨ (t = f - 1)) ∧
    (∀ (t' f' : ℕ),
      (t' + f' = total_bills_used) ∧
      ((t' = f' + 1) ∨ (t' = f' - 1)) →
      money_lost t f ≤ money_lost t' f') ∧
    money_lost t f = 800 :=
by sorry

end NUMINAMATH_CALUDE_min_money_lost_l1616_161668


namespace NUMINAMATH_CALUDE_measure_one_kg_l1616_161624

theorem measure_one_kg (n : ℕ) (h : ¬ 3 ∣ n) : 
  ∃ (k : ℕ), n - 3 * k = 1 ∨ n - 3 * k = 2 :=
sorry

end NUMINAMATH_CALUDE_measure_one_kg_l1616_161624


namespace NUMINAMATH_CALUDE_mass_of_ccl4_produced_l1616_161690

-- Define the chemical equation
def balanced_equation : String := "CaC2 + 4 Cl2O → CCl4 + CaCl2O"

-- Define the number of moles of reaction
def reaction_moles : ℝ := 8

-- Define molar masses
def molar_mass_carbon : ℝ := 12.01
def molar_mass_chlorine : ℝ := 35.45

-- Define the molar mass of CCl4
def molar_mass_ccl4 : ℝ := molar_mass_carbon + 4 * molar_mass_chlorine

-- Theorem statement
theorem mass_of_ccl4_produced : 
  reaction_moles * molar_mass_ccl4 = 1230.48 := by sorry

end NUMINAMATH_CALUDE_mass_of_ccl4_produced_l1616_161690


namespace NUMINAMATH_CALUDE_reseat_twelve_women_l1616_161625

/-- Number of ways to reseat n women under given conditions -/
def T : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => T (n + 2) + T (n + 1) + T n

/-- Theorem stating that the number of ways to reseat 12 women is 927 -/
theorem reseat_twelve_women : T 12 = 927 := by
  sorry

end NUMINAMATH_CALUDE_reseat_twelve_women_l1616_161625


namespace NUMINAMATH_CALUDE_certain_number_proof_l1616_161675

theorem certain_number_proof (x : ℝ) : 0.28 * x + 0.45 * 250 = 224.5 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1616_161675


namespace NUMINAMATH_CALUDE_representative_selection_count_l1616_161696

def female_students : ℕ := 5
def male_students : ℕ := 7
def total_representatives : ℕ := 5
def max_female_representatives : ℕ := 2

theorem representative_selection_count :
  (Nat.choose male_students total_representatives) +
  (Nat.choose female_students 1 * Nat.choose male_students 4) +
  (Nat.choose female_students 2 * Nat.choose male_students 3) = 546 := by
  sorry

end NUMINAMATH_CALUDE_representative_selection_count_l1616_161696


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l1616_161613

/-- The height of a melted ice cream scoop -/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h : ℝ) : 
  r_sphere = 3 →
  r_cylinder = 12 →
  (4 / 3) * π * r_sphere^3 = π * r_cylinder^2 * h →
  h = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_melted_ice_cream_height_l1616_161613


namespace NUMINAMATH_CALUDE_temperature_difference_is_eight_l1616_161699

-- Define the temperatures
def highest_temp : ℤ := 5
def lowest_temp : ℤ := -3

-- Define the temperature difference
def temp_difference : ℤ := highest_temp - lowest_temp

-- Theorem to prove
theorem temperature_difference_is_eight :
  temp_difference = 8 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_is_eight_l1616_161699


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_min_value_is_sqrt_3_l1616_161692

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + 2*x*y - 1 = 0) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 + 2*a*b - 1 = 0 → 2*x + y ≤ 2*a + b :=
by
  sorry

theorem min_value_is_sqrt_3 (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + 2*x*y - 1 = 0) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 2*a*b - 1 = 0 ∧ 2*a + b = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_min_value_is_sqrt_3_l1616_161692


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1616_161626

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length. -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 90) 
  (h2 : train_speed_kmh = 45) 
  (h3 : bridge_length = 285) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1616_161626


namespace NUMINAMATH_CALUDE_truck_travel_distance_l1616_161607

/-- Given a truck that travels 300 miles on 10 gallons of diesel,
    prove that it will travel 450 miles on 15 gallons of diesel. -/
theorem truck_travel_distance (initial_distance : ℝ) (initial_fuel : ℝ) (new_fuel : ℝ)
    (h1 : initial_distance = 300)
    (h2 : initial_fuel = 10)
    (h3 : new_fuel = 15) :
    (initial_distance / initial_fuel) * new_fuel = 450 :=
by sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l1616_161607


namespace NUMINAMATH_CALUDE_pot_height_problem_l1616_161649

/-- Given two similar right-angled triangles, where one triangle has a height of 20 inches and
    a base of 10 inches, and the other triangle has a base of 20 inches,
    prove that the height of the second triangle is 40 inches. -/
theorem pot_height_problem (h₁ h₂ : ℝ) (b₁ b₂ : ℝ) :
  h₁ = 20 → b₁ = 10 → b₂ = 20 → (h₁ / b₁ = h₂ / b₂) → h₂ = 40 :=
by sorry

end NUMINAMATH_CALUDE_pot_height_problem_l1616_161649


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l1616_161662

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def quadrilateral_properties (ABCD : Quadrilateral) : Prop :=
  let (xa, ya) := ABCD.A
  let (xb, yb) := ABCD.B
  let (xc, yc) := ABCD.C
  let (xd, yd) := ABCD.D
  ∃ (angle_BCD : ℝ),
    angle_BCD = 120 ∧
    (xb - xa)^2 + (yb - ya)^2 = 13^2 ∧
    (xc - xb)^2 + (yc - yb)^2 = 6^2 ∧
    (xd - xc)^2 + (yd - yc)^2 = 5^2 ∧
    (xa - xd)^2 + (ya - yd)^2 = 12^2

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the area of quadrilateral ABCD
def quadrilateral_area (ABCD : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral (ABCD : Quadrilateral) :
  quadrilateral_properties ABCD →
  quadrilateral_area ABCD = (15 * Real.sqrt 3) / 2 + triangle_area ABCD.B ABCD.D ABCD.A :=
sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l1616_161662


namespace NUMINAMATH_CALUDE_equation_solution_l1616_161697

theorem equation_solution : 
  ∃ y : ℝ, (7 * y / (y + 4) - 5 / (y + 4) = 2 / (y + 4)) ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1616_161697


namespace NUMINAMATH_CALUDE_container_capacity_l1616_161631

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 54 = 0.75 * C) : C = 120 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l1616_161631


namespace NUMINAMATH_CALUDE_system_solution_l1616_161680

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 7 * y = -9) ∧ (5 * x + 3 * y = -11) ∧ (x = -104/47) ∧ (y = 1/47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1616_161680


namespace NUMINAMATH_CALUDE_student_number_problem_l1616_161695

theorem student_number_problem : ∃ x : ℤ, 2 * x - 148 = 110 ∧ x = 129 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1616_161695


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1616_161637

theorem quadratic_equation_solution (m n : ℤ) :
  m^2 + 2*m*n + 2*n^2 - 4*n + 4 = 0 → m = -2 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1616_161637


namespace NUMINAMATH_CALUDE_largest_n_is_correct_l1616_161618

/-- Represents the coefficients of the quadratic expression 6x^2 + nx + 48 -/
structure QuadraticCoeffs where
  n : ℤ

/-- Represents the coefficients of the linear factors (2x + A)(3x + B) -/
structure LinearFactors where
  A : ℤ
  B : ℤ

/-- Checks if the given linear factors produce the quadratic expression -/
def is_valid_factorization (q : QuadraticCoeffs) (f : LinearFactors) : Prop :=
  (2 * f.B + 3 * f.A = q.n) ∧ (f.A * f.B = 48)

/-- The largest value of n for which the quadratic can be factored -/
def largest_n : ℤ := 99

theorem largest_n_is_correct : 
  (∀ q : QuadraticCoeffs, ∃ f : LinearFactors, is_valid_factorization q f → q.n ≤ largest_n) ∧
  (∃ q : QuadraticCoeffs, ∃ f : LinearFactors, is_valid_factorization q f ∧ q.n = largest_n) :=
sorry

end NUMINAMATH_CALUDE_largest_n_is_correct_l1616_161618


namespace NUMINAMATH_CALUDE_rods_to_furlongs_l1616_161654

/-- Conversion factor from furlongs to rods -/
def furlong_to_rods : ℕ := 50

/-- The number of rods we want to convert -/
def total_rods : ℕ := 1000

/-- The theorem states that 1000 rods is equal to 20 furlongs -/
theorem rods_to_furlongs : 
  (total_rods : ℚ) / furlong_to_rods = 20 := by sorry

end NUMINAMATH_CALUDE_rods_to_furlongs_l1616_161654


namespace NUMINAMATH_CALUDE_extra_workers_for_road_project_l1616_161645

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℕ
  initialWorkers : ℕ
  completedLength : ℝ
  completedDays : ℕ

/-- Calculates the number of extra workers needed to complete the project on time -/
def extraWorkersNeeded (project : RoadProject) : ℕ :=
  sorry

/-- Theorem stating that for the given project parameters, approximately 53 extra workers are needed -/
theorem extra_workers_for_road_project :
  let project : RoadProject := {
    totalLength := 15,
    totalDays := 300,
    initialWorkers := 35,
    completedLength := 2.5,
    completedDays := 100
  }
  ∃ n : ℕ, n ≥ 53 ∧ n ≤ 54 ∧ extraWorkersNeeded project = n :=
sorry

end NUMINAMATH_CALUDE_extra_workers_for_road_project_l1616_161645


namespace NUMINAMATH_CALUDE_absolute_value_calculation_l1616_161671

theorem absolute_value_calculation : |-3| * 2 - (-1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_l1616_161671


namespace NUMINAMATH_CALUDE_tom_jogging_distance_l1616_161693

/-- Tom's jogging rate in miles per minute -/
def jogging_rate : ℚ := 1 / 15

/-- Time Tom jogs in minutes -/
def jogging_time : ℚ := 45

/-- Distance Tom jogs in miles -/
def jogging_distance : ℚ := jogging_rate * jogging_time

theorem tom_jogging_distance :
  jogging_distance = 3 := by sorry

end NUMINAMATH_CALUDE_tom_jogging_distance_l1616_161693


namespace NUMINAMATH_CALUDE_base7_521_equals_260_l1616_161639

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem base7_521_equals_260 :
  base7ToBase10 [1, 2, 5] = 260 := by
  sorry

end NUMINAMATH_CALUDE_base7_521_equals_260_l1616_161639


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_8_l1616_161666

def is_greatest_factorial_under_1000 (n : ℕ) : Prop :=
  n.factorial < 1000 ∧ ∀ m : ℕ, m > n → m.factorial ≥ 1000

theorem sum_of_x_and_y_is_8 :
  ∀ x y : ℕ,
    x > 0 →
    y > 1 →
    is_greatest_factorial_under_1000 x →
    x + y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_8_l1616_161666


namespace NUMINAMATH_CALUDE_parabola_comparison_l1616_161678

theorem parabola_comparison : ∀ x : ℝ, 
  x^2 - (1/3)*x + 3 < x^2 + (1/3)*x + 4 := by sorry

end NUMINAMATH_CALUDE_parabola_comparison_l1616_161678


namespace NUMINAMATH_CALUDE_min_value_of_a_l1616_161694

theorem min_value_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 3 ∧ (1 + a * x) / (x * 2^x) ≥ 1) → 
  a ≥ 7/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1616_161694


namespace NUMINAMATH_CALUDE_sum_inequality_l1616_161685

theorem sum_inequality {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : c + a > d + b := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1616_161685


namespace NUMINAMATH_CALUDE_store_breaks_even_l1616_161669

/-- Represents the financial outcome of selling two items -/
def break_even (selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : Prop :=
  let cost_price_1 := selling_price / (1 + profit_percent / 100)
  let cost_price_2 := selling_price / (1 - loss_percent / 100)
  cost_price_1 + cost_price_2 = selling_price * 2

/-- Theorem: A store breaks even when selling two items at $150 each, 
    with one making 50% profit and the other incurring 25% loss -/
theorem store_breaks_even : break_even 150 50 25 := by
  sorry

end NUMINAMATH_CALUDE_store_breaks_even_l1616_161669


namespace NUMINAMATH_CALUDE_product_factorization_l1616_161620

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product : ℕ := (List.range 9).foldl (λ acc i => acc * factorial (20 + i)) 1

theorem product_factorization :
  ∃ (n : ℕ), n > 0 ∧ product = 825 * n^3 := by sorry

end NUMINAMATH_CALUDE_product_factorization_l1616_161620


namespace NUMINAMATH_CALUDE_triangle_properties_l1616_161600

/-- Triangle ABC with vertices A(0, 3), B(-2, -1), and C(4, 3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
def givenTriangle : Triangle where
  A := (0, 3)
  B := (-2, -1)
  C := (4, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The altitude from side AB -/
def altitudeAB (t : Triangle) : Line :=
  sorry

/-- The point symmetric to C with respect to line AB -/
def symmetricPointC (t : Triangle) : Point :=
  sorry

theorem triangle_properties (t : Triangle) (h : t = givenTriangle) :
  (altitudeAB t = Line.mk 1 2 (-10)) ∧
  (symmetricPointC t = Point.mk (-12/5) (31/5)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1616_161600


namespace NUMINAMATH_CALUDE_hyperbola_property_l1616_161665

/-- The hyperbola with equation x^2 - y^2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p | p.1^2 - p.2^2 = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- The length of the semi-major axis of the hyperbola -/
def a : ℝ := sorry

theorem hyperbola_property (P : ℝ × ℝ) (h₁ : P ∈ Hyperbola)
    (h₂ : (P.1 - F₁.1, P.2 - F₁.2) • (P.1 - F₂.1, P.2 - F₂.2) = 0) :
    ‖(P.1 - F₁.1, P.2 - F₁.2) + (P.1 - F₂.1, P.2 - F₂.2)‖ = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_property_l1616_161665


namespace NUMINAMATH_CALUDE_expression_order_l1616_161638

theorem expression_order (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  (2 * a * b) / (a + b) < Real.sqrt (a * b) ∧
  Real.sqrt (a * b) < (a + b) / 2 ∧
  (a + b) / 2 < Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_order_l1616_161638


namespace NUMINAMATH_CALUDE_problem_solution_l1616_161604

theorem problem_solution : ∃ x : ℝ, 400 * x = 28000 * 100^1 ∧ x = 7000 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1616_161604


namespace NUMINAMATH_CALUDE_find_divisor_l1616_161673

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 689)
  (h2 : quotient = 19)
  (h3 : remainder = 5)
  (h4 : dividend = quotient * (dividend / quotient) + remainder) :
  dividend / quotient = 36 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1616_161673


namespace NUMINAMATH_CALUDE_prime_relation_l1616_161663

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem prime_relation (p q : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h : 13 * p + 1 = q + 2) : 
  q = 39 := by sorry

end NUMINAMATH_CALUDE_prime_relation_l1616_161663


namespace NUMINAMATH_CALUDE_scientific_notation_of_170000_l1616_161614

/-- Definition of scientific notation -/
def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ a ∧ a < 10 ∧ ∃ (x : ℝ), x = a * (10 : ℝ) ^ n

/-- The problem statement -/
theorem scientific_notation_of_170000 :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ 170000 = a * (10 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_170000_l1616_161614


namespace NUMINAMATH_CALUDE_union_of_sets_l1616_161652

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4}
  let B : Set ℕ := {2, 4, 5, 6}
  A ∪ B = {1, 2, 3, 4, 5, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1616_161652


namespace NUMINAMATH_CALUDE_x_value_l1616_161634

/-- Given that 20% of x is 15 less than 15% of 1500, prove that x = 1050 -/
theorem x_value : ∃ x : ℝ, (0.2 * x = 0.15 * 1500 - 15) ∧ x = 1050 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1616_161634


namespace NUMINAMATH_CALUDE_male_female_ratio_l1616_161622

/-- Represents an association with male and female members selling raffle tickets -/
structure Association where
  male_members : ℕ
  female_members : ℕ
  total_tickets : ℕ
  male_tickets : ℕ
  female_tickets : ℕ

/-- The conditions given in the problem -/
def association_conditions (a : Association) : Prop :=
  (a.total_tickets : ℚ) / (a.male_members + a.female_members : ℚ) = 66 ∧
  (a.female_tickets : ℚ) / (a.female_members : ℚ) = 70 ∧
  (a.male_tickets : ℚ) / (a.male_members : ℚ) = 58 ∧
  a.total_tickets = a.male_tickets + a.female_tickets

/-- The theorem stating that under the given conditions, the male to female ratio is 1:2 -/
theorem male_female_ratio (a : Association) (h : association_conditions a) :
  (a.male_members : ℚ) / (a.female_members : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_male_female_ratio_l1616_161622


namespace NUMINAMATH_CALUDE_determine_constant_b_l1616_161643

theorem determine_constant_b (b c : ℝ) : 
  (∀ x, (3*x^2 - 4*x + 8/3)*(2*x^2 + b*x + c) = 6*x^4 - 17*x^3 + 21*x^2 - 16/3*x + 9/3) → 
  b = -3 := by
sorry

end NUMINAMATH_CALUDE_determine_constant_b_l1616_161643


namespace NUMINAMATH_CALUDE_sequence_median_l1616_161661

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def median_position (n : ℕ) : ℕ := (sequence_sum n + 1) / 2

theorem sequence_median :
  ∃ (m : ℕ), m = 106 ∧
  sequence_sum (m - 1) < median_position 150 ∧
  median_position 150 ≤ sequence_sum m :=
sorry

end NUMINAMATH_CALUDE_sequence_median_l1616_161661


namespace NUMINAMATH_CALUDE_lunch_break_duration_l1616_161644

/- Define the workshop as a unit (100%) -/
def workshop : ℝ := 1

/- Define the working rates -/
variable (p : ℝ) -- Paula's painting rate (workshop/hour)
variable (h : ℝ) -- Combined rate of helpers (workshop/hour)

/- Define the lunch break duration in hours -/
variable (L : ℝ)

/- Monday's work -/
axiom monday_work : (9 - L) * (p + h) = 0.6 * workshop

/- Tuesday's work -/
axiom tuesday_work : (7 - L) * h = 0.3 * workshop

/- Wednesday's work -/
axiom wednesday_work : (10 - L) * p = 0.1 * workshop

/- The sum of work done on all three days equals the whole workshop -/
axiom total_work : 0.6 * workshop + 0.3 * workshop + 0.1 * workshop = workshop

/- Theorem: The lunch break is 48 minutes -/
theorem lunch_break_duration : L * 60 = 48 := by
  sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l1616_161644


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1616_161615

/-- An increasing geometric sequence with specific conditions has a common ratio of 2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) > a n) →  -- increasing sequence
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence
  (a 1 + a 5 = 17) →  -- first condition
  (a 2 * a 4 = 16) →  -- second condition
  ∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = q * a n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1616_161615


namespace NUMINAMATH_CALUDE_count_even_factors_div_by_5_l1616_161672

/-- The number of even natural-number factors divisible by 5 of 2^3 * 5^2 * 11^1 -/
def num_even_factors_div_by_5 : ℕ :=
  let n : ℕ := 2^3 * 5^2 * 11^1
  -- Define the function here
  12

theorem count_even_factors_div_by_5 :
  num_even_factors_div_by_5 = 12 := by sorry

end NUMINAMATH_CALUDE_count_even_factors_div_by_5_l1616_161672


namespace NUMINAMATH_CALUDE_present_age_of_B_l1616_161623

/-- Given two people A and B, proves that B's current age is 70 years -/
theorem present_age_of_B (A B : ℕ) : 
  (A + 20 = 2 * (B - 20)) →  -- In 20 years, A will be twice as old as B was 20 years ago
  (A = B + 10) →             -- A is now 10 years older than B
  B = 70 :=                  -- B's current age is 70 years
by
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_l1616_161623


namespace NUMINAMATH_CALUDE_berry_picking_difference_l1616_161609

/-- Represents the berry picking scenario -/
structure BerryPicking where
  total_berries : ℕ
  dima_basket_ratio : ℚ
  sergei_basket_ratio : ℚ
  dima_speed_multiplier : ℕ

/-- Calculates the difference in berries placed in the basket between Dima and Sergei -/
def berry_difference (scenario : BerryPicking) : ℕ :=
  sorry

/-- The main theorem stating the difference in berries placed in the basket -/
theorem berry_picking_difference (scenario : BerryPicking) 
  (h1 : scenario.total_berries = 450)
  (h2 : scenario.dima_basket_ratio = 1/2)
  (h3 : scenario.sergei_basket_ratio = 2/3)
  (h4 : scenario.dima_speed_multiplier = 2) :
  berry_difference scenario = 50 :=
sorry

end NUMINAMATH_CALUDE_berry_picking_difference_l1616_161609


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1616_161656

theorem sum_of_three_numbers : 4.75 + 0.303 + 0.432 = 5.485 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1616_161656


namespace NUMINAMATH_CALUDE_plane_air_time_l1616_161611

/-- Proves that the time the plane spent in the air is 10/3 hours given the problem conditions. -/
theorem plane_air_time (total_distance : ℝ) (icebreaker_speed : ℝ) (plane_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 840)
  (h2 : icebreaker_speed = 20)
  (h3 : plane_speed = 120)
  (h4 : total_time = 22) :
  (total_distance - icebreaker_speed * total_time) / plane_speed = 10 / 3 := by
  sorry

#check plane_air_time

end NUMINAMATH_CALUDE_plane_air_time_l1616_161611


namespace NUMINAMATH_CALUDE_all_vints_are_xaffs_l1616_161659

-- Define the types
variable (Zibb Xaff Yurn Worb Vint : Type)

-- Define the conditions
variable (h1 : Zibb → Xaff)
variable (h2 : Yurn → Xaff)
variable (h3 : Worb → Zibb)
variable (h4 : Yurn → Worb)
variable (h5 : Worb → Vint)
variable (h6 : Vint → Yurn)

-- Theorem to prove
theorem all_vints_are_xaffs : Vint → Xaff := by sorry

end NUMINAMATH_CALUDE_all_vints_are_xaffs_l1616_161659


namespace NUMINAMATH_CALUDE_sarah_candy_problem_l1616_161617

/-- The number of candy pieces Sarah received from neighbors -/
def candy_from_neighbors : ℕ := 66

/-- The number of candy pieces Sarah ate per day -/
def candy_per_day : ℕ := 9

/-- The number of days Sarah's candy lasted -/
def days_candy_lasted : ℕ := 9

/-- The number of candy pieces Sarah received from her older sister -/
def candy_from_sister : ℕ := 15

theorem sarah_candy_problem :
  candy_from_sister = days_candy_lasted * candy_per_day - candy_from_neighbors :=
by sorry

end NUMINAMATH_CALUDE_sarah_candy_problem_l1616_161617


namespace NUMINAMATH_CALUDE_part_one_part_two_l1616_161628

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Part I
theorem part_one :
  let m : ℝ := -1
  {x : ℝ | f x m ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
theorem part_two :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (3/4 : ℝ) 2, f x m ≤ |2*x + 1|) →
  -11/4 ≤ m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1616_161628


namespace NUMINAMATH_CALUDE_train_final_speed_l1616_161612

/-- Given a train with the following properties:
  * Length: 360 meters
  * Initial velocity: 0 m/s (starts from rest)
  * Acceleration: 1 m/s²
  * Time to cross a man on the platform: 20 seconds
Prove that the final speed of the train is 20 m/s. -/
theorem train_final_speed
  (length : ℝ)
  (initial_velocity : ℝ)
  (acceleration : ℝ)
  (time : ℝ)
  (h1 : length = 360)
  (h2 : initial_velocity = 0)
  (h3 : acceleration = 1)
  (h4 : time = 20)
  : initial_velocity + acceleration * time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_final_speed_l1616_161612


namespace NUMINAMATH_CALUDE_coloring_books_remaining_l1616_161640

theorem coloring_books_remaining (initial : Real) (first_giveaway : Real) (second_giveaway : Real) :
  initial = 48.0 →
  first_giveaway = 34.0 →
  second_giveaway = 3.0 →
  initial - first_giveaway - second_giveaway = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_remaining_l1616_161640


namespace NUMINAMATH_CALUDE_intersecting_lines_l1616_161674

theorem intersecting_lines (x y : ℝ) : 
  (2*x - y)^2 - (x + 3*y)^2 = 0 ↔ (x = 4*y ∨ x = -2/3*y) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_l1616_161674


namespace NUMINAMATH_CALUDE_pentagon_area_l1616_161677

-- Define the pentagon
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  angle : ℝ

-- Define the given pentagon
def given_pentagon : Pentagon :=
  { side1 := 18
  , side2 := 25
  , side3 := 30
  , side4 := 28
  , side5 := 22
  , angle := 110 }

-- Define the area calculation function
noncomputable def calculate_area (p : Pentagon) : ℝ := sorry

-- Theorem stating the area of the given pentagon
theorem pentagon_area :
  ∃ ε > 0, |calculate_area given_pentagon - 738| < ε := by sorry

end NUMINAMATH_CALUDE_pentagon_area_l1616_161677


namespace NUMINAMATH_CALUDE_mango_lassi_price_l1616_161679

/-- The cost of a mango lassi at Delicious Delhi restaurant --/
def mango_lassi_cost (samosa_cost pakora_cost tip_percentage total_cost : ℚ) : ℚ :=
  total_cost - (samosa_cost + pakora_cost + (samosa_cost + pakora_cost) * tip_percentage / 100)

/-- Theorem stating the cost of the mango lassi --/
theorem mango_lassi_price :
  mango_lassi_cost 6 12 25 25 = (5/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_mango_lassi_price_l1616_161679


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1616_161670

theorem ellipse_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2 / (k - 4) + y^2 / (10 - k) = 1) ∧ 
  (∀ x y : ℝ, x^2 / (k - 4) + y^2 / (10 - k) = 1 → 
    ∃ a b c : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ c^2 = a^2 - b^2 ∧ c ≠ 0) →
  k > 7 ∧ k < 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1616_161670


namespace NUMINAMATH_CALUDE_security_code_combinations_l1616_161642

theorem security_code_combinations : Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_security_code_combinations_l1616_161642


namespace NUMINAMATH_CALUDE_max_stamps_per_page_l1616_161610

theorem max_stamps_per_page (album1 album2 album3 : ℕ) 
  (h1 : album1 = 945)
  (h2 : album2 = 1260)
  (h3 : album3 = 1575) :
  Nat.gcd album1 (Nat.gcd album2 album3) = 315 :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_per_page_l1616_161610


namespace NUMINAMATH_CALUDE_route2_faster_l1616_161627

-- Define the probabilities and delay times for each route
def prob_green_A : ℚ := 1/2
def prob_green_B : ℚ := 2/3
def delay_A : ℕ := 2
def delay_B : ℕ := 3
def time_green_AB : ℕ := 20

def prob_green_a : ℚ := 3/4
def prob_green_b : ℚ := 2/5
def delay_a : ℕ := 8
def delay_b : ℕ := 5
def time_green_ab : ℕ := 15

-- Define the expected delay for each route
def expected_delay_route1 : ℚ := 
  (1 - prob_green_A) * delay_A + (1 - prob_green_B) * delay_B

def expected_delay_route2 : ℚ := 
  (1 - prob_green_a) * delay_a + (1 - prob_green_b) * delay_b

-- Define the expected travel time for each route
def expected_time_route1 : ℚ := time_green_AB + expected_delay_route1
def expected_time_route2 : ℚ := time_green_ab + expected_delay_route2

-- Theorem statement
theorem route2_faster : expected_time_route2 < expected_time_route1 :=
  sorry

end NUMINAMATH_CALUDE_route2_faster_l1616_161627


namespace NUMINAMATH_CALUDE_cute_six_digit_integers_l1616_161687

def is_permutation (n : ℕ) (digits : List ℕ) : Prop :=
  digits.length = 6 ∧ digits.toFinset = Finset.range 6

def first_k_digits_divisible (digits : List ℕ) : Prop :=
  ∀ k : ℕ, k ≤ 6 → k ∣ (digits.take k).foldl (λ acc d => acc * 10 + d) 0

def is_cute (digits : List ℕ) : Prop :=
  is_permutation 6 digits ∧ first_k_digits_divisible digits

theorem cute_six_digit_integers :
  ∃! (s : Finset (List ℕ)), s.card = 2 ∧ ∀ digits, digits ∈ s ↔ is_cute digits :=
sorry

end NUMINAMATH_CALUDE_cute_six_digit_integers_l1616_161687


namespace NUMINAMATH_CALUDE_gwen_bookcase_distribution_l1616_161621

/-- Given a bookcase with mystery and picture book shelves, 
    calculates the number of books on each shelf. -/
def books_per_shelf (mystery_shelves : ℕ) (picture_shelves : ℕ) (total_books : ℕ) : ℕ :=
  total_books / (mystery_shelves + picture_shelves)

/-- Proves that Gwen's bookcase has 4 books on each shelf. -/
theorem gwen_bookcase_distribution :
  books_per_shelf 5 3 32 = 4 := by
  sorry

#eval books_per_shelf 5 3 32

end NUMINAMATH_CALUDE_gwen_bookcase_distribution_l1616_161621


namespace NUMINAMATH_CALUDE_plane_through_points_l1616_161616

-- Define the points A, B, and C
def A (a : ℝ) : ℝ × ℝ × ℝ := (a, 0, 0)
def B (b : ℝ) : ℝ × ℝ × ℝ := (0, b, 0)
def C (c : ℝ) : ℝ × ℝ × ℝ := (0, 0, c)

-- Define the plane equation
def plane_equation (a b c x y z : ℝ) : Prop :=
  x / a + y / b + z / c = 1

-- Theorem statement
theorem plane_through_points (a b c : ℝ) (h : a * b * c ≠ 0) :
  ∃ (f : ℝ × ℝ × ℝ → Prop),
    (∀ x y z, f (x, y, z) ↔ plane_equation a b c x y z) ∧
    f (A a) ∧ f (B b) ∧ f (C c) :=
sorry

end NUMINAMATH_CALUDE_plane_through_points_l1616_161616


namespace NUMINAMATH_CALUDE_fraction_operations_l1616_161636

theorem fraction_operations :
  let a := 2
  let b := 9
  let c := 5
  let d := 11
  (a / b) * (c / d) = 10 / 99 ∧ (a / b) + (c / d) = 67 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fraction_operations_l1616_161636


namespace NUMINAMATH_CALUDE_series_sum_l1616_161650

/-- r is the positive real solution to x³ - ¼x - 1 = 0 -/
def r : ℝ := sorry

/-- T is the sum of the infinite series r + 2r⁴ + 3r⁷ + 4r¹⁰ + ... -/
noncomputable def T : ℝ := sorry

/-- The equation that r satisfies -/
axiom r_eq : r^3 - (1/4)*r - 1 = 0

/-- The main theorem: T equals 4 / (1 + 4/r) -/
theorem series_sum : T = 4 / (1 + 4/r) := by sorry

end NUMINAMATH_CALUDE_series_sum_l1616_161650


namespace NUMINAMATH_CALUDE_sam_initial_money_l1616_161664

/-- The amount of money Sam had initially -/
def initial_money (num_books : ℕ) (cost_per_book : ℕ) (money_left : ℕ) : ℕ :=
  num_books * cost_per_book + money_left

/-- Theorem stating that Sam's initial money was 79 dollars -/
theorem sam_initial_money :
  initial_money 9 7 16 = 79 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_money_l1616_161664


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l1616_161647

/-- Given a spherical hole with a width of 30 cm at the top and a depth of 10 cm,
    the radius of the sphere that created this hole is 16.25 cm. -/
theorem sphere_radius_from_hole (hole_width : ℝ) (hole_depth : ℝ) (sphere_radius : ℝ) : 
  hole_width = 30 → 
  hole_depth = 10 → 
  sphere_radius = (hole_width ^ 2 / 4 + hole_depth ^ 2) / (2 * hole_depth) → 
  sphere_radius = 16.25 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l1616_161647


namespace NUMINAMATH_CALUDE_polar_equation_represents_line_and_circle_l1616_161667

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.sin θ = Real.sin (2 * θ)

-- Define a line in Cartesian coordinates (x-axis in this case)
def is_line (x y : ℝ) : Prop := y = 0

-- Define a circle in Cartesian coordinates
def is_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem polar_equation_represents_line_and_circle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (is_line x₁ y₁ ∧ polar_equation x₁ 0) ∧
    (is_circle x₂ y₂ ∧ ∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x₂ = ρ * Real.cos θ ∧ y₂ = ρ * Real.sin θ) :=
sorry

end NUMINAMATH_CALUDE_polar_equation_represents_line_and_circle_l1616_161667


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1616_161608

-- Define the sample space
def Ω : Type := Unit

-- Define the events
def both_miss : Set Ω := sorry
def hit_at_least_once : Set Ω := sorry

-- Define the theorem
theorem mutually_exclusive_events : 
  both_miss ∩ hit_at_least_once = ∅ := by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l1616_161608


namespace NUMINAMATH_CALUDE_matrix_product_AB_l1616_161635

def A : Matrix (Fin 4) (Fin 3) ℝ := !![0, -1, 2; 2, 1, 1; 3, 0, 1; 3, 7, 1]
def B : Matrix (Fin 3) (Fin 2) ℝ := !![3, 1; 2, 1; 1, 0]

theorem matrix_product_AB :
  A * B = !![0, -1; 9, 3; 10, 3; 24, 10] := by sorry

end NUMINAMATH_CALUDE_matrix_product_AB_l1616_161635


namespace NUMINAMATH_CALUDE_segment_and_polygon_inequalities_l1616_161655

/-- Segment with projections a and b on perpendicular lines has length c -/
structure Segment where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Polygon with projections a and b on coordinate axes has perimeter P -/
structure Polygon where
  a : ℝ
  b : ℝ
  P : ℝ

/-- Theorem about segment length and polygon perimeter -/
theorem segment_and_polygon_inequalities 
  (s : Segment) (p : Polygon) : 
  s.c ≥ (s.a + s.b) / Real.sqrt 2 ∧ 
  p.P ≥ Real.sqrt 2 * (p.a + p.b) := by
  sorry


end NUMINAMATH_CALUDE_segment_and_polygon_inequalities_l1616_161655


namespace NUMINAMATH_CALUDE_unique_root_condition_l1616_161658

/-- The characteristic equation of a thermal energy process -/
def characteristic_equation (x t : ℝ) : Prop := x^3 - 3*x = t

/-- The condition for a unique root -/
def has_unique_root (t : ℝ) : Prop :=
  ∃! x, characteristic_equation x t

/-- The main theorem about the uniqueness and magnitude of the root -/
theorem unique_root_condition (t : ℝ) :
  has_unique_root t ↔ abs t > 2 ∧ ∀ x, characteristic_equation x t → abs x > 2 :=
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l1616_161658


namespace NUMINAMATH_CALUDE_polygon_equidistant_point_l1616_161601

-- Define a convex polygon
def ConvexPolygon (V : Type*) := V → ℝ × ℝ

-- Define a point inside the polygon
def InsidePoint (P : ConvexPolygon V) (O : ℝ × ℝ) : Prop := sorry

-- Define the property of forming isosceles triangles
def FormsIsoscelesTriangles (P : ConvexPolygon V) (O : ℝ × ℝ) : Prop :=
  ∀ (v1 v2 : V), v1 ≠ v2 → ‖P v1 - O‖ = ‖P v2 - O‖

-- Define the property of being equidistant from all vertices
def EquidistantFromVertices (P : ConvexPolygon V) (O : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (v : V), ‖P v - O‖ = r

-- State the theorem
theorem polygon_equidistant_point {V : Type*} (P : ConvexPolygon V) (O : ℝ × ℝ) :
  InsidePoint P O → FormsIsoscelesTriangles P O → EquidistantFromVertices P O :=
sorry

end NUMINAMATH_CALUDE_polygon_equidistant_point_l1616_161601


namespace NUMINAMATH_CALUDE_x_value_proof_l1616_161630

theorem x_value_proof (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1616_161630


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_prime_terms_l1616_161603

-- Define an arithmetic progression
def ArithmeticProgression (a k : ℕ) : ℕ → ℕ := fun n => a + k * n

-- Define the property of having infinitely many prime terms at prime indices
def HasInfinitelyManyPrimeTermsAtPrimeIndices (seq : ℕ → ℕ) : Prop :=
  ∃ N : ℕ, ∀ p : ℕ, Prime p → p > N → Prime (seq p)

-- State the theorem
theorem arithmetic_progression_with_prime_terms (seq : ℕ → ℕ) :
  (∃ a k : ℕ, seq = ArithmeticProgression a k) →
  HasInfinitelyManyPrimeTermsAtPrimeIndices seq →
  (∃ P : ℕ, Prime P ∧ seq = fun _ => P) ∨ seq = id :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_prime_terms_l1616_161603


namespace NUMINAMATH_CALUDE_school_population_l1616_161682

/-- Given a school with boys, girls, and teachers, prove the total population. -/
theorem school_population (b g t : ℕ) : 
  b = 4 * g ∧ g = 8 * t → b + g + t = (41 * b) / 32 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l1616_161682


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_product_l1616_161689

theorem floor_sqrt_sum_eq_floor_sqrt_product (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_product_l1616_161689


namespace NUMINAMATH_CALUDE_equation_solution_l1616_161660

theorem equation_solution : ∃ x : ℚ, (x - 30) / 3 = (4 - 3*x) / 7 ∧ x = 111/8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1616_161660


namespace NUMINAMATH_CALUDE_house_sale_profit_l1616_161632

-- Define the initial home value
def initial_value : ℝ := 12000

-- Define the profit percentage for the first sale
def profit_percentage : ℝ := 0.20

-- Define the loss percentage for the second sale
def loss_percentage : ℝ := 0.15

-- Define the net profit
def net_profit : ℝ := 2160

-- Theorem statement
theorem house_sale_profit :
  let first_sale_price := initial_value * (1 + profit_percentage)
  let second_sale_price := first_sale_price * (1 - loss_percentage)
  first_sale_price - second_sale_price = net_profit := by
sorry

end NUMINAMATH_CALUDE_house_sale_profit_l1616_161632


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1616_161683

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored : ℚ) / (b.innings + 1 : ℚ)

theorem batsman_average_after_12th_innings 
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 65 = b.average + 3)
  : newAverage b 65 = 32 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1616_161683


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_subset_implies_a_range_l1616_161629

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | -a < x ∧ x < a + 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | 4 - x < a}

-- Part 1
theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = Set.Ioo 2 3 → a = 2 := by sorry

-- Part 2
theorem subset_implies_a_range (a : ℝ) :
  A a ⊆ (Set.univ \ B a) → a ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_subset_implies_a_range_l1616_161629


namespace NUMINAMATH_CALUDE_find_n_l1616_161688

-- Define the polynomial
def p (x y : ℝ) := (x^2 - y)^7

-- Define the fourth term of the expansion
def fourth_term (x y : ℝ) := -35 * x^8 * y^3

-- Define the fifth term of the expansion
def fifth_term (x y : ℝ) := 35 * x^6 * y^4

-- Theorem statement
theorem find_n (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n = 7)
  (h4 : fourth_term m n = fifth_term m n) : n = (49 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1616_161688


namespace NUMINAMATH_CALUDE_triangle_area_l1616_161602

/-- The area of a triangle with base 12 and height 9 is 54 -/
theorem triangle_area : ∀ (base height : ℝ), 
  base = 12 → height = 9 → (1/2 : ℝ) * base * height = 54 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l1616_161602


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_theorem_3_l1616_161691

-- Define the set S
def S (m l : ℝ) := {x : ℝ | m ≤ x ∧ x ≤ l}

-- Define the property that x^2 ∈ S for all x ∈ S
def closed_under_square (m l : ℝ) :=
  ∀ x, x ∈ S m l → x^2 ∈ S m l

-- Theorem 1
theorem theorem_1 (m l : ℝ) (h : closed_under_square m l) :
  m = 1 → S m l = {1} := by sorry

-- Theorem 2
theorem theorem_2 (m l : ℝ) (h : closed_under_square m l) :
  m = -1/2 → 1/4 ≤ l ∧ l ≤ 1 := by sorry

-- Theorem 3
theorem theorem_3 (m l : ℝ) (h : closed_under_square m l) :
  l = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_theorem_3_l1616_161691


namespace NUMINAMATH_CALUDE_prob_same_heads_is_five_thirty_seconds_l1616_161606

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The probability of getting heads on a single penny toss -/
def prob_heads : ℚ := 1/2

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def prob_same_heads : ℚ := 5/32

/-- Theorem stating that the probability of Ephraim getting the same number of heads as Keiko is 5/32 -/
theorem prob_same_heads_is_five_thirty_seconds :
  prob_same_heads = 5/32 := by sorry

end NUMINAMATH_CALUDE_prob_same_heads_is_five_thirty_seconds_l1616_161606


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l1616_161648

def is_valid (x y z : Nat) : Prop :=
  let n := 579000 + 100 * x + 10 * y + z
  n % 5 = 0 ∧ n % 7 = 0 ∧ n % 9 = 0

theorem three_digit_divisibility :
  ∀ x y z : Nat,
    x < 10 ∧ y < 10 ∧ z < 10 →
    is_valid x y z ↔ (x = 6 ∧ y = 0 ∧ z = 0) ∨ 
                     (x = 2 ∧ y = 8 ∧ z = 5) ∨ 
                     (x = 9 ∧ y = 1 ∧ z = 5) :=
by sorry

#check three_digit_divisibility

end NUMINAMATH_CALUDE_three_digit_divisibility_l1616_161648


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1616_161698

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  (x + 2) * (x - 3) = 2 * x - 6

-- Define the general form of the equation
def general_form (x : ℝ) : Prop :=
  x^2 - 3*x = 0

-- Theorem statement
theorem quadratic_equation_solution :
  (∀ x, quadratic_equation x ↔ general_form x) ∧
  (∃ x₁ x₂, x₁ = 0 ∧ x₂ = 3 ∧ ∀ x, general_form x ↔ (x = x₁ ∨ x = x₂)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1616_161698


namespace NUMINAMATH_CALUDE_simplify_fraction_l1616_161605

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1616_161605


namespace NUMINAMATH_CALUDE_otimes_nested_equality_l1616_161686

/-- The custom operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 + 3 - y

/-- Theorem stating that k ⊗ (k ⊗ (k ⊗ k)) = k^3 + 3 - k -/
theorem otimes_nested_equality (k : ℝ) : otimes k (otimes k (otimes k k)) = k^3 + 3 - k := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_equality_l1616_161686


namespace NUMINAMATH_CALUDE_sum_squared_l1616_161619

theorem sum_squared (x y : ℝ) (h1 : 2*x*(x+y) = 58) (h2 : 3*y*(x+y) = 111) : 
  (x + y)^2 = 28561 / 25 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_l1616_161619


namespace NUMINAMATH_CALUDE_shaded_perimeter_is_32_l1616_161651

/-- Given two squares ABCD and BEFG sharing vertex B, with E on BC, G on AB,
    this structure represents the configuration described in the problem. -/
structure SquareConfiguration where
  -- Side length of square ABCD
  x : ℝ
  -- Assertion that E is on BC and G is on AB
  h_e_on_bc : True
  h_g_on_ab : True
  -- Length of CG is 9
  h_cg_length : 9 = 9
  -- Area of shaded region (ABCD - BEFG) is 47
  h_shaded_area : x^2 - (81 - x^2) = 47

/-- Theorem stating that under the given conditions, 
    the perimeter of the shaded region (which is the same as ABCD) is 32. -/
theorem shaded_perimeter_is_32 (config : SquareConfiguration) :
  4 * config.x = 32 := by
  sorry

#check shaded_perimeter_is_32

end NUMINAMATH_CALUDE_shaded_perimeter_is_32_l1616_161651


namespace NUMINAMATH_CALUDE_inscribed_circles_diameter_l1616_161653

/-- A sequence of circles inscribed in a parabola -/
def InscribedCircles (ω : ℕ → Set (ℝ × ℝ)) : Prop :=
  ∀ n : ℕ, 
    -- Each circle is inscribed in the parabola y = x²
    (∀ (x y : ℝ), (x, y) ∈ ω n → y = x^2) ∧
    -- Each circle is tangent to the next one
    (∃ (x y : ℝ), (x, y) ∈ ω n ∧ (x, y) ∈ ω (n + 1)) ∧
    -- The first circle has diameter 1 and touches the parabola at (0,0)
    (n = 1 → (0, 0) ∈ ω 1 ∧ ∃ (x y : ℝ), (x, y) ∈ ω 1 ∧ x^2 + y^2 = 1/4)

/-- The diameter of a circle -/
def Diameter (ω : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The diameter of the nth circle is 2n - 1 -/
theorem inscribed_circles_diameter 
  (ω : ℕ → Set (ℝ × ℝ)) 
  (h : InscribedCircles ω) :
  ∀ n : ℕ, n > 0 → Diameter (ω n) = 2 * n - 1 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circles_diameter_l1616_161653


namespace NUMINAMATH_CALUDE_trig_identity_l1616_161684

theorem trig_identity (α : Real) :
  (∃ P : Real × Real, P.1 = Real.sin 2 ∧ P.2 = Real.cos 2 ∧ 
    P.1^2 + P.2^2 = 1 ∧ Real.sin α = P.2) →
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1616_161684


namespace NUMINAMATH_CALUDE_faye_remaining_money_is_correct_l1616_161681

/-- Calculates Faye's remaining money after her shopping spree -/
def faye_remaining_money (original_money : ℚ) : ℚ :=
  let father_gift := 3 * original_money
  let mother_gift := 2 * father_gift
  let grandfather_gift := 4 * original_money
  let total_money := original_money + father_gift + mother_gift + grandfather_gift
  let muffin_cost := 15 * 1.75
  let cookie_cost := 10 * 2.5
  let juice_cost := 2 * 4
  let candy_cost := 25 * 0.25
  let total_item_cost := muffin_cost + cookie_cost + juice_cost + candy_cost
  let tip := 0.15 * (muffin_cost + cookie_cost)
  let total_spent := total_item_cost + tip
  total_money - total_spent

theorem faye_remaining_money_is_correct : 
  faye_remaining_money 20 = 206.81 := by sorry

end NUMINAMATH_CALUDE_faye_remaining_money_is_correct_l1616_161681


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_2050_l1616_161676

theorem units_digit_of_7_pow_2050 : 7^2050 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_2050_l1616_161676


namespace NUMINAMATH_CALUDE_embroidery_time_l1616_161633

-- Define the stitches per minute
def stitches_per_minute : ℕ := 4

-- Define the number of stitches for each design
def flower_stitches : ℕ := 60
def unicorn_stitches : ℕ := 180
def godzilla_stitches : ℕ := 800

-- Define the number of each design
def num_flowers : ℕ := 50
def num_unicorns : ℕ := 3
def num_godzilla : ℕ := 1

-- Theorem to prove
theorem embroidery_time :
  (num_flowers * flower_stitches + num_unicorns * unicorn_stitches + num_godzilla * godzilla_stitches) / stitches_per_minute = 1085 := by
  sorry

end NUMINAMATH_CALUDE_embroidery_time_l1616_161633


namespace NUMINAMATH_CALUDE_horse_distribution_exists_l1616_161657

/-- Represents the distribution of horses to sons -/
structure Distribution (b₁ b₂ b₃ : ℕ) :=
  (x₁₁ x₁₂ x₁₃ : ℕ)
  (x₂₁ x₂₂ x₂₃ : ℕ)
  (x₃₁ x₃₂ x₃₃ : ℕ)
  (sum_eq_b₁ : x₁₁ + x₂₁ + x₃₁ = b₁)
  (sum_eq_b₂ : x₁₂ + x₂₂ + x₃₂ = b₂)
  (sum_eq_b₃ : x₁₃ + x₂₃ + x₃₃ = b₃)

/-- Represents the value matrix for horses -/
def ValueMatrix := Matrix (Fin 3) (Fin 3) ℚ

/-- The theorem statement -/
theorem horse_distribution_exists :
  ∃ n : ℕ, ∀ b₁ b₂ b₃ : ℕ, ∀ A : ValueMatrix,
    (∀ i j : Fin 3, i ≠ j → A i i > A i j) →
    min b₁ (min b₂ b₃) > n →
    ∃ d : Distribution b₁ b₂ b₃,
      (A 0 0 * d.x₁₁ + A 0 1 * d.x₁₂ + A 0 2 * d.x₁₃ > A 0 0 * d.x₂₁ + A 0 1 * d.x₂₂ + A 0 2 * d.x₂₃) ∧
      (A 0 0 * d.x₁₁ + A 0 1 * d.x₁₂ + A 0 2 * d.x₁₃ > A 0 0 * d.x₃₁ + A 0 1 * d.x₃₂ + A 0 2 * d.x₃₃) ∧
      (A 1 0 * d.x₂₁ + A 1 1 * d.x₂₂ + A 1 2 * d.x₂₃ > A 1 0 * d.x₁₁ + A 1 1 * d.x₁₂ + A 1 2 * d.x₁₃) ∧
      (A 1 0 * d.x₂₁ + A 1 1 * d.x₂₂ + A 1 2 * d.x₂₃ > A 1 0 * d.x₃₁ + A 1 1 * d.x₃₂ + A 1 2 * d.x₃₃) ∧
      (A 2 0 * d.x₃₁ + A 2 1 * d.x₃₂ + A 2 2 * d.x₃₃ > A 2 0 * d.x₁₁ + A 2 1 * d.x₁₂ + A 2 2 * d.x₁₃) ∧
      (A 2 0 * d.x₃₁ + A 2 1 * d.x₃₂ + A 2 2 * d.x₃₃ > A 2 0 * d.x₂₁ + A 2 1 * d.x₂₂ + A 2 2 * d.x₂₃) :=
sorry

end NUMINAMATH_CALUDE_horse_distribution_exists_l1616_161657


namespace NUMINAMATH_CALUDE_no_perfect_power_triple_l1616_161646

theorem no_perfect_power_triple (n r : ℕ) (hn : n ≥ 1) (hr : r ≥ 2) :
  ¬∃ m : ℤ, (n : ℤ) * (n + 1) * (n + 2) = m ^ r :=
sorry

end NUMINAMATH_CALUDE_no_perfect_power_triple_l1616_161646
