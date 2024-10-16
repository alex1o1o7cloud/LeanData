import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l3633_363368

theorem imaginary_part_of_complex_product : Complex.im ((3 * Complex.I - 1) * Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l3633_363368


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l3633_363340

theorem youngest_sibling_age (y : ℝ) : 
  (y + (y + 3) + (y + 6) + (y + 7)) / 4 = 30 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l3633_363340


namespace NUMINAMATH_CALUDE_xyz_maximum_l3633_363397

theorem xyz_maximum (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_one : x + y + z = 1) (sum_inv_eq_ten : 1/x + 1/y + 1/z = 10) :
  xyz ≤ 4/125 ∧ ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ 1/x + 1/y + 1/z = 10 ∧ x*y*z = 4/125 :=
by sorry

end NUMINAMATH_CALUDE_xyz_maximum_l3633_363397


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l3633_363374

/-- An arithmetic sequence of positive real numbers -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r ∧ b n > 0

/-- Theorem: For arithmetic sequence a and geometric sequence b,
    if a_1 = b_1 and a_2015 = b_2015, then a_1008 ≥ b_1008 -/
theorem arithmetic_geometric_inequality
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h_eq1 : a 1 = b 1)
  (h_eq2015 : a 2015 = b 2015) :
  a 1008 ≥ b 1008 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l3633_363374


namespace NUMINAMATH_CALUDE_cargo_passenger_relationship_l3633_363313

/-- Represents a train with passenger cars and cargo cars. -/
structure Train where
  total_cars : ℕ
  passenger_cars : ℕ
  cargo_cars : ℕ

/-- Defines the properties of our specific train. -/
def our_train : Train where
  total_cars := 71
  passenger_cars := 44
  cargo_cars := 25

/-- Theorem stating the relationship between cargo cars and passenger cars. -/
theorem cargo_passenger_relationship (t : Train) 
  (h1 : t.total_cars = t.passenger_cars + t.cargo_cars + 2) 
  (h2 : t.cargo_cars = t.passenger_cars / 2 + (t.cargo_cars - t.passenger_cars / 2)) : 
  t.cargo_cars - t.passenger_cars / 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_cargo_passenger_relationship_l3633_363313


namespace NUMINAMATH_CALUDE_frisbee_throwing_problem_l3633_363332

/-- Frisbee throwing problem -/
theorem frisbee_throwing_problem 
  (bess_distance : ℝ) 
  (bess_throws : ℕ) 
  (holly_throws : ℕ) 
  (total_distance : ℝ) 
  (h1 : bess_distance = 20)
  (h2 : bess_throws = 4)
  (h3 : holly_throws = 5)
  (h4 : total_distance = 200)
  (h5 : bess_distance * bess_throws * 2 + holly_throws * holly_distance = total_distance) :
  holly_distance = 8 := by
  sorry


end NUMINAMATH_CALUDE_frisbee_throwing_problem_l3633_363332


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3633_363353

theorem simple_interest_problem (simple_interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  simple_interest = 4016.25 →
  rate = 0.01 →
  time = 3 →
  principal = simple_interest / (rate * time) →
  principal = 133875 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3633_363353


namespace NUMINAMATH_CALUDE_tile_formation_theorem_l3633_363365

/-- Represents a 4x4 tile --/
def Tile := Matrix (Fin 4) (Fin 4) Bool

/-- Checks if a tile has alternating colors on its outside row and column --/
def hasAlternatingOutside (t : Tile) : Prop :=
  (∀ i, t 0 i ≠ t 0 (i + 1)) ∧
  (∀ i, t i 0 ≠ t (i + 1) 0)

/-- Represents the property that a tile can be formed by combining two pieces --/
def canBeFormedByPieces (t : Tile) : Prop :=
  hasAlternatingOutside t

theorem tile_formation_theorem (t : Tile) :
  ¬(canBeFormedByPieces t) ↔ ¬(hasAlternatingOutside t) :=
sorry

end NUMINAMATH_CALUDE_tile_formation_theorem_l3633_363365


namespace NUMINAMATH_CALUDE_sqrt_seven_less_than_three_l3633_363345

theorem sqrt_seven_less_than_three : Real.sqrt 7 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_less_than_three_l3633_363345


namespace NUMINAMATH_CALUDE_equality_preservation_l3633_363387

theorem equality_preservation (x y : ℝ) (h : x = y) : -1/3 * x + 1 = -1/3 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_equality_preservation_l3633_363387


namespace NUMINAMATH_CALUDE_fruit_store_solution_l3633_363367

/-- Represents the purchase quantities and costs of two types of fruits -/
structure FruitPurchase where
  quantityA : ℕ  -- Quantity of fruit A in kg
  quantityB : ℕ  -- Quantity of fruit B in kg
  totalCost : ℕ  -- Total cost in yuan

/-- The fruit store problem -/
def fruitStoreProblem (purchase1 purchase2 : FruitPurchase) : Prop :=
  ∃ (priceA priceB : ℕ),
    -- Conditions from the first purchase
    purchase1.quantityA * priceA + purchase1.quantityB * priceB = purchase1.totalCost ∧
    -- Conditions from the second purchase
    purchase2.quantityA * priceA + purchase2.quantityB * priceB = purchase2.totalCost ∧
    -- Unique solution condition
    ∀ (x y : ℕ),
      (purchase1.quantityA * x + purchase1.quantityB * y = purchase1.totalCost ∧
       purchase2.quantityA * x + purchase2.quantityB * y = purchase2.totalCost) →
      x = priceA ∧ y = priceB

/-- Theorem stating the solution to the fruit store problem -/
theorem fruit_store_solution :
  fruitStoreProblem
    { quantityA := 60, quantityB := 40, totalCost := 1520 }
    { quantityA := 30, quantityB := 50, totalCost := 1360 } →
  ∃ (priceA priceB : ℕ), priceA = 12 ∧ priceB = 20 := by
  sorry

end NUMINAMATH_CALUDE_fruit_store_solution_l3633_363367


namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l3633_363343

/-- An arithmetic progression with first term 5 and 25th term 173 has common difference 7. -/
theorem arithmetic_progression_common_difference : 
  ∀ (a : ℕ → ℝ), 
    (a 1 = 5) → 
    (a 25 = 173) → 
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) → 
    (a 2 - a 1 = 7) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l3633_363343


namespace NUMINAMATH_CALUDE_perpendicular_lines_angle_relation_l3633_363311

-- Define a dihedral angle
structure DihedralAngle where
  plane_angle : ℝ
  -- Add other necessary properties

-- Define a point inside a dihedral angle
structure PointInDihedralAngle where
  dihedral : DihedralAngle
  -- Add other necessary properties

-- Define the angle formed by perpendicular lines
def perpendicularLinesAngle (p : PointInDihedralAngle) : ℝ := sorry

-- Define the relationship between angles
def isEqualOrComplementary (a b : ℝ) : Prop :=
  a = b ∨ a + b = Real.pi / 2

-- Theorem statement
theorem perpendicular_lines_angle_relation (p : PointInDihedralAngle) :
  isEqualOrComplementary (perpendicularLinesAngle p) p.dihedral.plane_angle := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_angle_relation_l3633_363311


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l3633_363333

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) :
  x = 2 ∧ y = -2 →
  ρ = Real.sqrt (x^2 + y^2) ∧
  θ = -π/4 ∧
  x = ρ * Real.cos θ ∧
  y = ρ * Real.sin θ →
  ρ = 2 * Real.sqrt 2 ∧ θ = -π/4 := by
  sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l3633_363333


namespace NUMINAMATH_CALUDE_cube_paper_expenditure_l3633_363322

-- Define the parameters
def paper_cost_per_kg : ℚ := 60
def cube_edge_length : ℚ := 10
def area_covered_per_kg : ℚ := 20

-- Define the function to calculate the expenditure
def calculate_expenditure (edge_length area_per_kg cost_per_kg : ℚ) : ℚ :=
  6 * edge_length^2 / area_per_kg * cost_per_kg

-- State the theorem
theorem cube_paper_expenditure :
  calculate_expenditure cube_edge_length area_covered_per_kg paper_cost_per_kg = 1800 := by
  sorry

end NUMINAMATH_CALUDE_cube_paper_expenditure_l3633_363322


namespace NUMINAMATH_CALUDE_alyssa_grew_nine_turnips_l3633_363370

/-- The number of turnips Keith grew -/
def keith_turnips : ℕ := 6

/-- The total number of turnips Keith and Alyssa grew together -/
def total_turnips : ℕ := 15

/-- The number of turnips Alyssa grew -/
def alyssa_turnips : ℕ := total_turnips - keith_turnips

theorem alyssa_grew_nine_turnips : alyssa_turnips = 9 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_grew_nine_turnips_l3633_363370


namespace NUMINAMATH_CALUDE_skipping_rope_price_solution_l3633_363369

def skipping_rope_prices (price_A price_B : ℚ) : Prop :=
  (price_B = price_A + 10) ∧
  (3150 / price_A = 3900 / price_B) ∧
  (price_A = 42) ∧
  (price_B = 52)

theorem skipping_rope_price_solution :
  ∃ (price_A price_B : ℚ), skipping_rope_prices price_A price_B :=
sorry

end NUMINAMATH_CALUDE_skipping_rope_price_solution_l3633_363369


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3633_363339

theorem ratio_x_to_y (x y : ℝ) 
  (h1 : (3*x - 2*y) / (2*x + 3*y) = 5/4)
  (h2 : x + y = 5) : 
  x / y = 23/2 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3633_363339


namespace NUMINAMATH_CALUDE_unique_a_for_equal_F_l3633_363314

def F (a b c : ℝ) : ℝ := a * b^2 + c

theorem unique_a_for_equal_F : ∃! a : ℝ, F a 3 (-1) = F a 5 (-3) ∧ a = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_for_equal_F_l3633_363314


namespace NUMINAMATH_CALUDE_witnesses_same_type_l3633_363377

-- Define the types of people
inductive PersonType
| Knight
| Liar

-- Define the statements as functions
def statement_A (X Y : Prop) : Prop := X → Y
def statement_B (X Y : Prop) : Prop := ¬X ∨ Y

-- Main theorem
theorem witnesses_same_type (X Y : Prop) (A B : PersonType) :
  (A = PersonType.Knight ↔ statement_A X Y) →
  (B = PersonType.Knight ↔ statement_B X Y) →
  A = B :=
sorry

end NUMINAMATH_CALUDE_witnesses_same_type_l3633_363377


namespace NUMINAMATH_CALUDE_curve_intersection_tangent_l3633_363301

/-- The value of a for which the curves y = a√x and y = ln√x have a common point
    with the same tangent line. -/
theorem curve_intersection_tangent (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, x > 0 ∧ a * Real.sqrt x = Real.log (Real.sqrt x) ∧
    a / (2 * Real.sqrt x) = 1 / (2 * x)) →
  a = Real.exp (-1) := by
sorry

end NUMINAMATH_CALUDE_curve_intersection_tangent_l3633_363301


namespace NUMINAMATH_CALUDE_original_room_population_l3633_363300

theorem original_room_population (initial_population : ℕ) : 
  (initial_population / 3 : ℚ) = 18 → initial_population = 54 :=
by
  intro h
  sorry

#check original_room_population

end NUMINAMATH_CALUDE_original_room_population_l3633_363300


namespace NUMINAMATH_CALUDE_exponent_properties_l3633_363307

theorem exponent_properties (a b : ℝ) (n : ℕ) :
  (a * b) ^ n = a ^ n * b ^ n ∧
  2 ^ 5 * (-1/2) ^ 5 = -1 ∧
  (-0.125) ^ 2022 * 2 ^ 2021 * 4 ^ 2020 = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_exponent_properties_l3633_363307


namespace NUMINAMATH_CALUDE_f_at_6_l3633_363308

/-- The polynomial f(x) = 3x^6 + 12x^5 + 8x^4 - 3.5x^3 + 7.2x^2 + 5x - 13 -/
def f (x : ℝ) : ℝ := 3*x^6 + 12*x^5 + 8*x^4 - 3.5*x^3 + 7.2*x^2 + 5*x - 13

/-- Theorem stating that f(6) = 243168.2 -/
theorem f_at_6 : f 6 = 243168.2 := by
  sorry

end NUMINAMATH_CALUDE_f_at_6_l3633_363308


namespace NUMINAMATH_CALUDE_contest_scores_order_l3633_363329

theorem contest_scores_order (A B C D : ℕ) 
  (eq1 : A + B = C + D)
  (eq2 : D + B = A + C + 10)
  (eq3 : C = A + D + 5) :
  B > C ∧ C > D ∧ D > A := by
  sorry

end NUMINAMATH_CALUDE_contest_scores_order_l3633_363329


namespace NUMINAMATH_CALUDE_double_root_k_l3633_363325

/-- A cubic equation with a double root -/
def has_double_root (k : ℝ) : Prop :=
  ∃ (r s : ℝ), (∀ x, x^3 + k*x - 128 = (x - r)^2 * (x - s))

/-- The value of k for which x^3 + kx - 128 = 0 has a double root -/
theorem double_root_k : ∃ k : ℝ, has_double_root k ∧ k = -48 := by
  sorry

end NUMINAMATH_CALUDE_double_root_k_l3633_363325


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3633_363390

theorem quadratic_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x < 0 ↔ 0 < x ∧ x < 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3633_363390


namespace NUMINAMATH_CALUDE_quotient_problem_l3633_363319

theorem quotient_problem (y : ℝ) (h : 5 / y = 5.3) : y = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l3633_363319


namespace NUMINAMATH_CALUDE_min_value_theorem_l3633_363326

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z > 0 → z + y = 1 → (2 / (z + 3 * y) + 1 / (z - y)) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3633_363326


namespace NUMINAMATH_CALUDE_regular_polygon_right_triangles_l3633_363324

/-- Given a regular polygon with n sides, if there are 1200 ways to choose
    three vertices that form a right triangle, then n = 50. -/
theorem regular_polygon_right_triangles (n : ℕ) : n > 0 →
  (n / 2 : ℕ) * (n - 2) = 1200 → n = 50 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_right_triangles_l3633_363324


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3633_363355

theorem complex_number_quadrant : 
  let z : ℂ := (Complex.I : ℂ) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3633_363355


namespace NUMINAMATH_CALUDE_charity_raffle_proof_l3633_363396

/-- Calculates the total money raised from a charity raffle and donations. -/
def total_money_raised (num_tickets : ℕ) (ticket_price : ℚ) (donation1 : ℚ) (num_donation1 : ℕ) (donation2 : ℚ) : ℚ :=
  (num_tickets : ℚ) * ticket_price + (num_donation1 : ℚ) * donation1 + donation2

/-- Proves that the total money raised is $100.00 given the specific conditions. -/
theorem charity_raffle_proof :
  let num_tickets : ℕ := 25
  let ticket_price : ℚ := 2
  let donation1 : ℚ := 15
  let num_donation1 : ℕ := 2
  let donation2 : ℚ := 20
  total_money_raised num_tickets ticket_price donation1 num_donation1 donation2 = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_charity_raffle_proof_l3633_363396


namespace NUMINAMATH_CALUDE_ball_count_in_box_l3633_363392

theorem ball_count_in_box (n : ℕ) (yellow_count : ℕ) (prob_yellow : ℚ) : 
  yellow_count = 9 → prob_yellow = 3/10 → (yellow_count : ℚ) / n = prob_yellow → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_in_box_l3633_363392


namespace NUMINAMATH_CALUDE_min_area_inscribed_equilateral_l3633_363372

/-- The minimum area of an inscribed equilateral triangle in a right triangle -/
theorem min_area_inscribed_equilateral (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let min_area := (Real.sqrt 3 * a^2 * b^2) / (4 * (a^2 + b^2 + Real.sqrt 3 * a * b))
  ∀ (D E F : ℝ × ℝ),
    let A := (0, 0)
    let B := (a, 0)
    let C := (0, b)
    (D.1 ≥ 0 ∧ D.1 ≤ a ∧ D.2 = 0) →  -- D is on BC
    (E.1 = 0 ∧ E.2 ≥ 0 ∧ E.2 ≤ b) →  -- E is on CA
    (F.2 = (b / a) * F.1 ∧ F.1 ≥ 0 ∧ F.1 ≤ a) →  -- F is on AB
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = (E.1 - F.1)^2 + (E.2 - F.2)^2 →  -- DEF is equilateral
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = (F.1 - D.1)^2 + (F.2 - D.2)^2 →
    let area := Real.sqrt 3 / 4 * ((D.1 - E.1)^2 + (D.2 - E.2)^2)
    area ≥ min_area :=
by sorry

end NUMINAMATH_CALUDE_min_area_inscribed_equilateral_l3633_363372


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_l3633_363375

theorem floor_plus_self_eq (r : ℝ) : ⌊r⌋ + r = 12.4 ↔ r = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_l3633_363375


namespace NUMINAMATH_CALUDE_porter_previous_painting_price_l3633_363318

/-- The amount Porter made for his previous painting, in dollars. -/
def previous_painting_price : ℕ := sorry

/-- The amount Porter made for his recent painting, in dollars. -/
def recent_painting_price : ℕ := 44000

/-- The relationship between the prices of the two paintings. -/
axiom price_relation : recent_painting_price = 5 * previous_painting_price - 1000

theorem porter_previous_painting_price :
  previous_painting_price = 9000 := by sorry

end NUMINAMATH_CALUDE_porter_previous_painting_price_l3633_363318


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l3633_363373

/-- The definite integral of √(1-x^2) + x from -1 to 1 equals π/2 -/
theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ x in (-1 : ℝ)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l3633_363373


namespace NUMINAMATH_CALUDE_fraction_to_longest_side_is_five_twelfths_l3633_363361

/-- Represents a trapezoid field with corn -/
structure CornField where
  -- Side lengths in clockwise order from a 60° angle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  -- Angles at the non-parallel sides
  angle1 : ℝ
  angle2 : ℝ
  -- Conditions
  side1_eq : side1 = 150
  side2_eq : side2 = 150
  side3_eq : side3 = 200
  side4_eq : side4 = 200
  angle1_eq : angle1 = 60
  angle2_eq : angle2 = 120
  is_trapezoid : angle1 + angle2 = 180

/-- The fraction of the crop brought to the longest side -/
def fractionToLongestSide (field : CornField) : ℚ :=
  5/12

/-- Theorem stating that the fraction of the crop brought to the longest side is 5/12 -/
theorem fraction_to_longest_side_is_five_twelfths (field : CornField) :
  fractionToLongestSide field = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_longest_side_is_five_twelfths_l3633_363361


namespace NUMINAMATH_CALUDE_ticket_ratio_l3633_363385

/-- Prove the ratio of Peyton's tickets to Tate's total tickets -/
theorem ticket_ratio :
  let tate_initial : ℕ := 32
  let tate_bought : ℕ := 2
  let total_tickets : ℕ := 51
  let tate_total : ℕ := tate_initial + tate_bought
  let peyton_tickets : ℕ := total_tickets - tate_total
  (peyton_tickets : ℚ) / tate_total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ticket_ratio_l3633_363385


namespace NUMINAMATH_CALUDE_rohans_salary_l3633_363398

/-- Rohan's monthly salary in rupees -/
def monthly_salary : ℝ := 7500

/-- Percentage of salary spent on food -/
def food_percentage : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_percentage : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_percentage : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_percentage : ℝ := 10

/-- Rohan's savings at the end of the month in rupees -/
def savings : ℝ := 1500

theorem rohans_salary :
  monthly_salary = 7500 ∧
  food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage + (savings / monthly_salary * 100) = 100 :=
by sorry

end NUMINAMATH_CALUDE_rohans_salary_l3633_363398


namespace NUMINAMATH_CALUDE_total_spent_calculation_l3633_363352

/-- Calculates the total amount spent at a restaurant given the food price, sales tax rate, and tip rate. -/
def totalSpent (foodPrice : ℝ) (salesTaxRate : ℝ) (tipRate : ℝ) : ℝ :=
  let priceWithTax := foodPrice * (1 + salesTaxRate)
  let tipAmount := priceWithTax * tipRate
  priceWithTax + tipAmount

/-- Theorem stating that the total amount spent is $184.80 given the specific conditions. -/
theorem total_spent_calculation (foodPrice : ℝ) (salesTaxRate : ℝ) (tipRate : ℝ) 
    (h1 : foodPrice = 140)
    (h2 : salesTaxRate = 0.1)
    (h3 : tipRate = 0.2) : 
  totalSpent foodPrice salesTaxRate tipRate = 184.80 := by
  sorry

#eval totalSpent 140 0.1 0.2

end NUMINAMATH_CALUDE_total_spent_calculation_l3633_363352


namespace NUMINAMATH_CALUDE_coplanar_condition_l3633_363359

open Vector

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

def isCoplanar (p₁ p₂ p₃ p₄ p₅ : V) : Prop :=
  ∃ (a b c d : ℝ), a • (p₂ - p₁) + b • (p₃ - p₁) + c • (p₄ - p₁) + d • (p₅ - p₁) = 0

theorem coplanar_condition (O E F G H I : V) (m : ℝ) :
  (4 • (E - O) - 3 • (F - O) + 6 • (G - O) + m • (H - O) - 2 • (I - O) = 0) →
  (isCoplanar E F G H I ↔ m = -5) := by
  sorry

end NUMINAMATH_CALUDE_coplanar_condition_l3633_363359


namespace NUMINAMATH_CALUDE_double_sum_equals_one_point_five_l3633_363303

/-- The double sum of 1/(mn(m+n+2)) from m=1 to ∞ and n=1 to ∞ equals 1.5 -/
theorem double_sum_equals_one_point_five :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 2))) = (3 / 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_double_sum_equals_one_point_five_l3633_363303


namespace NUMINAMATH_CALUDE_impossibility_of_triangular_section_l3633_363356

theorem impossibility_of_triangular_section (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = 7^2 →
  b^2 + c^2 = 8^2 →
  c^2 + a^2 = 11^2 →
  False :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_triangular_section_l3633_363356


namespace NUMINAMATH_CALUDE_sugar_solution_mixing_l3633_363331

/-- Calculates the percentage of sugar in the resulting solution after replacing
    a portion of an initial sugar solution with another sugar solution. -/
theorem sugar_solution_mixing (initial_sugar_percentage : ℝ)
                               (replacement_portion : ℝ)
                               (replacement_sugar_percentage : ℝ) :
  initial_sugar_percentage = 8 →
  replacement_portion = 1/4 →
  replacement_sugar_percentage = 40 →
  let remaining_portion := 1 - replacement_portion
  let initial_sugar := initial_sugar_percentage * remaining_portion
  let replacement_sugar := replacement_sugar_percentage * replacement_portion
  let final_sugar_percentage := initial_sugar + replacement_sugar
  final_sugar_percentage = 16 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_mixing_l3633_363331


namespace NUMINAMATH_CALUDE_subset_range_of_a_l3633_363315

theorem subset_range_of_a (a : ℝ) : 
  let A := {x : ℝ | 1 ≤ x ∧ x ≤ 5}
  let B := {x : ℝ | a < x ∧ x < a + 1}
  B ⊆ A → 1 ≤ a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_subset_range_of_a_l3633_363315


namespace NUMINAMATH_CALUDE_justine_fewer_than_ylona_l3633_363358

/-- The number of rubber bands each person had initially and after Bailey's distribution --/
structure RubberBands where
  justine_initial : ℕ
  bailey_initial : ℕ
  ylona_initial : ℕ
  bailey_final : ℕ

/-- The conditions of the rubber band problem --/
def rubber_band_problem (rb : RubberBands) : Prop :=
  rb.justine_initial = rb.bailey_initial + 10 ∧
  rb.justine_initial < rb.ylona_initial ∧
  rb.bailey_final = rb.bailey_initial - 4 ∧
  rb.bailey_final = 8 ∧
  rb.ylona_initial = 24

/-- Theorem stating that Justine had 2 fewer rubber bands than Ylona initially --/
theorem justine_fewer_than_ylona (rb : RubberBands) 
  (h : rubber_band_problem rb) : 
  rb.ylona_initial - rb.justine_initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_justine_fewer_than_ylona_l3633_363358


namespace NUMINAMATH_CALUDE_max_a_min_b_for_sin_inequality_l3633_363349

theorem max_a_min_b_for_sin_inequality 
  (h : ∀ x ∈ Set.Ioo 0 (π/2), a * x < Real.sin x ∧ Real.sin x < b * x) :
  (∀ a' : ℝ, (∀ x ∈ Set.Ioo 0 (π/2), a' * x < Real.sin x) → a' ≤ 2/π) ∧
  (∀ b' : ℝ, (∀ x ∈ Set.Ioo 0 (π/2), Real.sin x < b' * x) → b' ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_max_a_min_b_for_sin_inequality_l3633_363349


namespace NUMINAMATH_CALUDE_compound_statement_false_l3633_363386

theorem compound_statement_false (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_compound_statement_false_l3633_363386


namespace NUMINAMATH_CALUDE_arithmetic_progression_solution_l3633_363336

theorem arithmetic_progression_solution (a : ℝ) : 
  (3 - 2*a = a - 6 - 3) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_solution_l3633_363336


namespace NUMINAMATH_CALUDE_average_speed_two_part_trip_l3633_363363

/-- Calculate the average speed of a two-part trip -/
theorem average_speed_two_part_trip (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 50)
  (h2 : distance1 = 25)
  (h3 : speed1 = 66)
  (h4 : speed2 = 33) :
  let time1 := distance1 / speed1
  let time2 := (total_distance - distance1) / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  ∃ ε > 0, |avg_speed - 44| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_speed_two_part_trip_l3633_363363


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l3633_363344

/-- Proves that the loss percentage is 10% given the selling prices with loss and with 10% gain --/
theorem book_sale_loss_percentage 
  (sp_loss : ℝ) 
  (sp_gain : ℝ) 
  (h_sp_loss : sp_loss = 450)
  (h_sp_gain : sp_gain = 550)
  (h_gain_percentage : sp_gain = 1.1 * (sp_gain / 1.1)) : 
  (((sp_gain / 1.1) - sp_loss) / (sp_gain / 1.1)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l3633_363344


namespace NUMINAMATH_CALUDE_smallest_a_for_nonempty_solution_l3633_363323

theorem smallest_a_for_nonempty_solution : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (∃ x : ℝ, 2*|x-3| + |x-4| < a^2 + a) ∧
  (∀ b : ℕ, (b > 0 ∧ b < a) → ¬∃ x : ℝ, 2*|x-3| + |x-4| < b^2 + b) ∧
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_nonempty_solution_l3633_363323


namespace NUMINAMATH_CALUDE_carol_distance_behind_anna_l3633_363376

/-- Represents the position of a runner in a race -/
structure Position :=
  (distance : ℝ)

/-- Represents a runner in the race -/
structure Runner :=
  (speed : ℝ)
  (position : Position)

/-- The race setup -/
structure Race :=
  (length : ℝ)
  (anna : Runner)
  (bridgit : Runner)
  (carol : Runner)

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.length = 100 ∧
  r.anna.speed > 0 ∧
  r.bridgit.speed > 0 ∧
  r.carol.speed > 0 ∧
  r.anna.speed > r.bridgit.speed ∧
  r.bridgit.speed > r.carol.speed ∧
  r.length - r.bridgit.position.distance = 16 ∧
  r.length - r.carol.position.distance = 25 + (r.length - r.bridgit.position.distance)

theorem carol_distance_behind_anna (r : Race) (h : race_conditions r) :
  r.length - r.carol.position.distance = 37 :=
sorry

end NUMINAMATH_CALUDE_carol_distance_behind_anna_l3633_363376


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l3633_363327

theorem sally_pokemon_cards (x : ℕ) : 
  x + 41 - 20 = 48 → x = 27 := by
sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l3633_363327


namespace NUMINAMATH_CALUDE_third_row_sum_is_401_l3633_363379

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the grid -/
def Grid := ℕ → ℕ → ℕ

/-- The size of the grid -/
def gridSize : ℕ := 16

/-- The starting position (centermost) -/
def startPos : Position :=
  { row := 9, col := 9 }

/-- Fills the grid in a clockwise spiral pattern -/
def fillGrid : Grid :=
  sorry

/-- Gets the numbers in a specific row -/
def getRowNumbers (g : Grid) (row : ℕ) : List ℕ :=
  sorry

/-- Theorem: The sum of the greatest and least number in the third row from the top is 401 -/
theorem third_row_sum_is_401 :
  let g := fillGrid
  let thirdRow := getRowNumbers g 3
  (List.maximum thirdRow).getD 0 + (List.minimum thirdRow).getD 0 = 401 := by
  sorry

end NUMINAMATH_CALUDE_third_row_sum_is_401_l3633_363379


namespace NUMINAMATH_CALUDE_square_difference_identity_l3633_363306

theorem square_difference_identity (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l3633_363306


namespace NUMINAMATH_CALUDE_robotics_camp_age_problem_l3633_363383

theorem robotics_camp_age_problem (total_members : ℕ) (girls : ℕ) (boys : ℕ) (adults : ℕ)
  (overall_avg : ℚ) (girls_avg : ℚ) (boys_avg : ℚ) :
  total_members = 60 →
  girls = 30 →
  boys = 20 →
  adults = 10 →
  overall_avg = 18 →
  girls_avg = 16 →
  boys_avg = 17 →
  (total_members * overall_avg - girls * girls_avg - boys * boys_avg) / adults = 26 :=
by sorry

end NUMINAMATH_CALUDE_robotics_camp_age_problem_l3633_363383


namespace NUMINAMATH_CALUDE_hit_rate_calculation_l3633_363393

theorem hit_rate_calculation (p₁ p₂ : ℚ) : 
  (p₁ * (1 - p₂) * (1/3 : ℚ) = 1/18) →
  (p₂ * (2/3 : ℚ) = 4/9) →
  p₁ = 1/2 ∧ p₂ = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_hit_rate_calculation_l3633_363393


namespace NUMINAMATH_CALUDE_property_of_x_l3633_363381

theorem property_of_x (x : ℝ) (h1 : x > 0) :
  (100 - x) / 100 * x = 16 → x = 40 ∨ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_property_of_x_l3633_363381


namespace NUMINAMATH_CALUDE_sin_sum_equals_half_l3633_363335

theorem sin_sum_equals_half : 
  Real.sin (163 * π / 180) * Real.sin (223 * π / 180) + 
  Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_equals_half_l3633_363335


namespace NUMINAMATH_CALUDE_construction_company_higher_utility_l3633_363384

/-- Represents the quality of renovation work -/
structure Quality where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents the cost of renovation work -/
structure Cost where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents the amount of available information about the service provider -/
structure Information where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents a renovation service provider -/
structure ServiceProvider where
  quality : Quality
  cost : Cost
  information : Information

/-- Utility function for renovation service -/
def utilityFunction (α β γ : ℝ) (sp : ServiceProvider) : ℝ :=
  α * sp.quality.value + β * sp.information.value - γ * sp.cost.value

/-- Theorem: Under certain conditions, a construction company can provide higher expected utility -/
theorem construction_company_higher_utility 
  (cc : ServiceProvider) -- construction company
  (prc : ServiceProvider) -- private repair crew
  (α β γ : ℝ) -- utility function parameters
  (h_α : α > 0) -- quality is valued positively
  (h_β : β > 0) -- information is valued positively
  (h_γ : γ > 0) -- cost is valued negatively
  (h_quality : cc.quality.value > prc.quality.value) -- company provides higher quality
  (h_info : cc.information.value > prc.information.value) -- company provides more information
  (h_cost : cc.cost.value > prc.cost.value) -- company is more expensive
  : ∃ (α β γ : ℝ), utilityFunction α β γ cc > utilityFunction α β γ prc :=
sorry

end NUMINAMATH_CALUDE_construction_company_higher_utility_l3633_363384


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3633_363399

theorem simplify_trig_expression (h : π / 2 < 2 ∧ 2 < π) :
  2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3633_363399


namespace NUMINAMATH_CALUDE_investment_revenue_difference_l3633_363338

def banks_investments : ℕ := 8
def banks_revenue_per_investment : ℕ := 500
def elizabeth_investments : ℕ := 5
def elizabeth_revenue_per_investment : ℕ := 900

theorem investment_revenue_difference :
  elizabeth_investments * elizabeth_revenue_per_investment - 
  banks_investments * banks_revenue_per_investment = 500 := by
sorry

end NUMINAMATH_CALUDE_investment_revenue_difference_l3633_363338


namespace NUMINAMATH_CALUDE_thomas_savings_l3633_363364

/-- Thomas's savings scenario --/
theorem thomas_savings (
  weekly_allowance : ℝ)
  (weeks_per_year : ℕ)
  (hours_per_week : ℕ)
  (car_cost : ℝ)
  (weekly_spending : ℝ)
  (additional_savings_needed : ℝ)
  (hourly_wage : ℝ)
  (h1 : weekly_allowance = 50)
  (h2 : weeks_per_year = 52)
  (h3 : hours_per_week = 30)
  (h4 : car_cost = 15000)
  (h5 : weekly_spending = 35)
  (h6 : additional_savings_needed = 2000)
  : hourly_wage = 7.83 := by
  sorry

end NUMINAMATH_CALUDE_thomas_savings_l3633_363364


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_nine_l3633_363310

/-- A quadratic function with roots satisfying specific conditions -/
structure QuadraticWithSpecialRoots where
  a : ℝ
  b : ℝ
  m : ℝ
  n : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_roots : m ≠ n ∧ m > 0 ∧ n > 0
  h_vieta : m + n = a ∧ m * n = b
  h_arithmetic : (m - n = n - (-2)) ∨ (n - m = m - (-2))
  h_geometric : (m / n = n / (-2)) ∨ (n / m = m / (-2))

/-- The sum of coefficients a and b equals 9 -/
theorem sum_of_coefficients_is_nine (q : QuadraticWithSpecialRoots) : q.a + q.b = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_nine_l3633_363310


namespace NUMINAMATH_CALUDE_unique_obtuse_consecutive_triangle_l3633_363382

/-- A triangle with consecutive natural number side lengths is obtuse if and only if 
    the square of the longest side is greater than the sum of squares of the other two sides. -/
def IsObtuseConsecutiveTriangle (x : ℕ) : Prop :=
  (x + 2)^2 > x^2 + (x + 1)^2

/-- There exists exactly one obtuse triangle with consecutive natural number side lengths. -/
theorem unique_obtuse_consecutive_triangle :
  ∃! x : ℕ, IsObtuseConsecutiveTriangle x ∧ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_obtuse_consecutive_triangle_l3633_363382


namespace NUMINAMATH_CALUDE_inequality_proof_l3633_363312

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3633_363312


namespace NUMINAMATH_CALUDE_hypotenuse_of_6_8_triangle_l3633_363334

/-- The Pythagorean theorem for a right-angled triangle -/
def pythagorean_theorem (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

/-- Theorem: In a right-angled triangle with legs of length 6 and 8, the hypotenuse has a length of 10 -/
theorem hypotenuse_of_6_8_triangle :
  ∃ (c : ℝ), pythagorean_theorem 6 8 c ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_of_6_8_triangle_l3633_363334


namespace NUMINAMATH_CALUDE_octagon_diagonals_and_quadratic_positivity_l3633_363321

theorem octagon_diagonals_and_quadratic_positivity :
  (∀ (n : ℕ), n = 8 → (n * (n - 3)) / 2 = 20) ∧
  (∀ (x : ℝ), 2 * x^2 - 2 * x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_octagon_diagonals_and_quadratic_positivity_l3633_363321


namespace NUMINAMATH_CALUDE_f_not_even_l3633_363316

def f (x : ℝ) := x^2 + x

theorem f_not_even : ¬(∀ x : ℝ, f x = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_not_even_l3633_363316


namespace NUMINAMATH_CALUDE_max_integer_solution_inequality_system_negative_six_satisfies_system_max_integer_solution_is_negative_six_l3633_363348

theorem max_integer_solution_inequality_system :
  ∀ x : ℤ, (x + 5 < 0 ∧ (3 * x - 1) / 2 ≥ 2 * x + 1) → x ≤ -6 :=
by
  sorry

theorem negative_six_satisfies_system :
  -6 + 5 < 0 ∧ (3 * (-6) - 1) / 2 ≥ 2 * (-6) + 1 :=
by
  sorry

theorem max_integer_solution_is_negative_six :
  ∃ x : ℤ, x + 5 < 0 ∧ (3 * x - 1) / 2 ≥ 2 * x + 1 ∧
  ∀ y : ℤ, (y + 5 < 0 ∧ (3 * y - 1) / 2 ≥ 2 * y + 1) → y ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_max_integer_solution_inequality_system_negative_six_satisfies_system_max_integer_solution_is_negative_six_l3633_363348


namespace NUMINAMATH_CALUDE_tom_lifting_capacity_l3633_363309

def initial_capacity : ℝ := 80
def training_multiplier : ℝ := 2
def specialization_increase : ℝ := 1.1
def num_hands : ℕ := 2

theorem tom_lifting_capacity : 
  initial_capacity * training_multiplier * specialization_increase * num_hands = 352 := by
  sorry

end NUMINAMATH_CALUDE_tom_lifting_capacity_l3633_363309


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3633_363305

theorem binomial_expansion_coefficient (x : ℝ) :
  (1 + 2*x)^3 = 1 + 6*x + 12*x^2 + 8*x^3 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3633_363305


namespace NUMINAMATH_CALUDE_no_charming_numbers_l3633_363394

/-- A two-digit positive integer is charming if it equals the sum of the square of its tens digit
and the product of its digits. -/
def IsCharming (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = a^2 + a * b

/-- There are no charming two-digit positive integers. -/
theorem no_charming_numbers : ¬∃ n : ℕ, IsCharming n := by
  sorry

end NUMINAMATH_CALUDE_no_charming_numbers_l3633_363394


namespace NUMINAMATH_CALUDE_line_slope_is_one_l3633_363366

-- Define the line using its point-slope form
def line_equation (x y : ℝ) : Prop := y + 1 = x - 2

-- State the theorem
theorem line_slope_is_one :
  ∀ x y : ℝ, line_equation x y → (y - (y + 1)) / (x - (x - 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_is_one_l3633_363366


namespace NUMINAMATH_CALUDE_min_value_expression_l3633_363317

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ≥ 11 / 16 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 2 ∧
    (4 / (x₀ + 2) + (3 * x₀ - 7) / (3 * y₀ + 4)) = 11 / 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3633_363317


namespace NUMINAMATH_CALUDE_smallest_odd_prime_divisor_of_difference_of_squares_l3633_363395

def is_odd_prime (p : Nat) : Prop := Nat.Prime p ∧ p % 2 = 1

theorem smallest_odd_prime_divisor_of_difference_of_squares :
  ∃ (k : Nat), k = 3 ∧
  (∀ (m n : Nat), is_odd_prime m → is_odd_prime n → m < 10 → n < 10 → n < m →
    k ∣ (m^2 - n^2)) ∧
  (∀ (p : Nat), p < k → is_odd_prime p →
    ∃ (m n : Nat), is_odd_prime m ∧ is_odd_prime n ∧ m < 10 ∧ n < 10 ∧ n < m ∧
      ¬(p ∣ (m^2 - n^2))) := by
sorry

end NUMINAMATH_CALUDE_smallest_odd_prime_divisor_of_difference_of_squares_l3633_363395


namespace NUMINAMATH_CALUDE_price_restoration_percentage_l3633_363328

theorem price_restoration_percentage (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := 0.85 * original_price
  let restoration_factor := original_price / reduced_price
  let percentage_increase := (restoration_factor - 1) * 100
  ∃ ε > 0, abs (percentage_increase - 17.65) < ε :=
by sorry

end NUMINAMATH_CALUDE_price_restoration_percentage_l3633_363328


namespace NUMINAMATH_CALUDE_reciprocal_of_golden_ratio_l3633_363320

theorem reciprocal_of_golden_ratio (φ : ℝ) :
  φ = (Real.sqrt 5 + 1) / 2 →
  1 / φ = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_golden_ratio_l3633_363320


namespace NUMINAMATH_CALUDE_arrange_40555_l3633_363389

def digit_arrangements (n : ℕ) : ℕ := 
  if n = 40555 then 12 else 0

theorem arrange_40555 :
  digit_arrangements 40555 = 12 ∧
  (∀ x : ℕ, x ≠ 40555 → digit_arrangements x = 0) :=
sorry

end NUMINAMATH_CALUDE_arrange_40555_l3633_363389


namespace NUMINAMATH_CALUDE_multiplication_sum_equality_l3633_363388

theorem multiplication_sum_equality : 45 * 58 + 45 * 42 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_sum_equality_l3633_363388


namespace NUMINAMATH_CALUDE_sid_shopping_l3633_363380

def shopping_problem (initial_amount : ℕ) (snack_cost : ℕ) (remaining_amount_extra : ℕ) : Prop :=
  let computer_accessories_cost := initial_amount - snack_cost - (initial_amount / 2 + remaining_amount_extra)
  computer_accessories_cost = 12

theorem sid_shopping :
  shopping_problem 48 8 4 :=
sorry

end NUMINAMATH_CALUDE_sid_shopping_l3633_363380


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt13_l3633_363341

theorem closest_integer_to_sqrt13 : 
  ∀ n : ℤ, n ∈ ({2, 3, 4, 5} : Set ℤ) → |n - Real.sqrt 13| ≥ |4 - Real.sqrt 13| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt13_l3633_363341


namespace NUMINAMATH_CALUDE_equilateral_triangle_dot_product_l3633_363378

/-- Equilateral triangle ABC with side length 2 -/
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ‖A - B‖ = 2 ∧ ‖B - C‖ = 2 ∧ ‖C - A‖ = 2

/-- Vector from point P to point Q -/
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := Q - P

theorem equilateral_triangle_dot_product 
  (A B C D E : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_BC : vec B C = 2 • vec B D)
  (h_CA : vec C A = 3 • vec C E)
  (a b : ℝ × ℝ)
  (h_a : a = vec A B)
  (h_b : b = vec A C)
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 2)
  (h_dot : a • b = 2) :
  (1/2 • (a + b)) • (2/3 • b - a) = -1 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_dot_product_l3633_363378


namespace NUMINAMATH_CALUDE_distance_between_runners_l3633_363346

/-- The distance between two runners at the end of a 1 km race -/
theorem distance_between_runners (H J : ℝ) (t : ℝ) 
  (h_distance : 1000 = H * t) 
  (j_distance : 152 = J * t) : 
  1000 - 152 = 848 := by sorry

end NUMINAMATH_CALUDE_distance_between_runners_l3633_363346


namespace NUMINAMATH_CALUDE_third_part_value_l3633_363354

theorem third_part_value (total : ℚ) (ratio1 ratio2 ratio3 : ℚ) 
  (h_total : total = 782)
  (h_ratio1 : ratio1 = 1/2)
  (h_ratio2 : ratio2 = 2/3)
  (h_ratio3 : ratio3 = 3/4) :
  (ratio3 / (ratio1 + ratio2 + ratio3)) * total = 306 :=
by sorry

end NUMINAMATH_CALUDE_third_part_value_l3633_363354


namespace NUMINAMATH_CALUDE_raisins_sum_l3633_363347

-- Define the amounts of yellow and black raisins
def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

-- Define the total amount of raisins
def total_raisins : ℝ := yellow_raisins + black_raisins

-- Theorem statement
theorem raisins_sum : total_raisins = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_raisins_sum_l3633_363347


namespace NUMINAMATH_CALUDE_roots_modulus_one_preserved_l3633_363302

theorem roots_modulus_one_preserved (a b c : ℂ) :
  (∃ α β γ : ℂ, (α^3 + a*α^2 + b*α + c = 0) ∧ 
                (β^3 + a*β^2 + b*β + c = 0) ∧ 
                (γ^3 + a*γ^2 + b*γ + c = 0) ∧
                (Complex.abs α = 1) ∧ (Complex.abs β = 1) ∧ (Complex.abs γ = 1)) →
  (∃ x y z : ℂ, (x^3 + Complex.abs a*x^2 + Complex.abs b*x + Complex.abs c = 0) ∧ 
                (y^3 + Complex.abs a*y^2 + Complex.abs b*y + Complex.abs c = 0) ∧ 
                (z^3 + Complex.abs a*z^2 + Complex.abs b*z + Complex.abs c = 0) ∧
                (Complex.abs x = 1) ∧ (Complex.abs y = 1) ∧ (Complex.abs z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_roots_modulus_one_preserved_l3633_363302


namespace NUMINAMATH_CALUDE_y_value_at_x_2_l3633_363337

theorem y_value_at_x_2 :
  let y₁ := λ x : ℝ => x^2 - 7*x + 6
  let y₂ := λ x : ℝ => 7*x - 3
  let y := λ x : ℝ => y₁ x + x * y₂ x
  y 2 = 18 := by sorry

end NUMINAMATH_CALUDE_y_value_at_x_2_l3633_363337


namespace NUMINAMATH_CALUDE_student_score_average_l3633_363351

/-- Given a student's scores in mathematics, physics, and chemistry, prove that the average of mathematics and chemistry scores is 26. -/
theorem student_score_average (math physics chem : ℕ) : 
  math + physics = 32 →
  chem = physics + 20 →
  (math + chem) / 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_student_score_average_l3633_363351


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l3633_363391

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 4| = 12 ∧ |x₂ - 4| = 12 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l3633_363391


namespace NUMINAMATH_CALUDE_charles_earnings_l3633_363330

def housesitting_rate : ℕ := 15
def dog_walking_rate : ℕ := 22
def housesitting_hours : ℕ := 10
def dogs_walked : ℕ := 3
def hours_per_dog : ℕ := 1

def total_earnings : ℕ := housesitting_rate * housesitting_hours + dog_walking_rate * dogs_walked * hours_per_dog

theorem charles_earnings : total_earnings = 216 := by
  sorry

end NUMINAMATH_CALUDE_charles_earnings_l3633_363330


namespace NUMINAMATH_CALUDE_fermat_little_theorem_general_l3633_363362

theorem fermat_little_theorem_general (p : ℕ) (m : ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, m^p - m = k * p :=
sorry

end NUMINAMATH_CALUDE_fermat_little_theorem_general_l3633_363362


namespace NUMINAMATH_CALUDE_investment_rate_calculation_investment_rate_proof_l3633_363350

/-- Calculates the required interest rate for the remaining investment --/
theorem investment_rate_calculation 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (first_rate : ℝ) 
  (second_rate : ℝ) 
  (desired_income : ℝ) : ℝ :=
  let remaining_investment := total_investment - first_investment - second_investment
  let first_income := first_investment * first_rate / 100
  let second_income := second_investment * second_rate / 100
  let remaining_income := desired_income - first_income - second_income
  let required_rate := remaining_income / remaining_investment * 100
  required_rate

/-- Proves that the required interest rate is approximately 7.05% --/
theorem investment_rate_proof 
  (h1 : total_investment = 15000)
  (h2 : first_investment = 6000)
  (h3 : second_investment = 4500)
  (h4 : first_rate = 3)
  (h5 : second_rate = 4.5)
  (h6 : desired_income = 700) :
  ∃ ε > 0, |investment_rate_calculation total_investment first_investment second_investment first_rate second_rate desired_income - 7.05| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_investment_rate_proof_l3633_363350


namespace NUMINAMATH_CALUDE_man_speed_with_current_l3633_363360

/-- Calculates the man's speed with the current given his speed against the current and the current speed. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Proves that given a man's speed against a current of 9.6 km/hr and a current speed of 3.2 km/hr, 
    the man's speed with the current is 16.0 km/hr. -/
theorem man_speed_with_current :
  speed_with_current 9.6 3.2 = 16.0 := by
  sorry

#eval speed_with_current 9.6 3.2

end NUMINAMATH_CALUDE_man_speed_with_current_l3633_363360


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_equality_l3633_363304

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + d.repeatingPart / (99 : ℚ)

/-- The main theorem stating that the given fraction of repeating decimals equals the specified rational number -/
theorem repeating_decimal_fraction_equality : 
  let a := RepeatingDecimal.mk 0 75
  let b := RepeatingDecimal.mk 2 25
  (toRational a) / (toRational b) = 2475 / 7339 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_equality_l3633_363304


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3633_363371

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 5 / 9 * 11 / 13 = 440 / 2457 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3633_363371


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3633_363342

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 8 = 6) →
  (a 3 * a 8 = 5) →
  a 5 + a 6 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3633_363342


namespace NUMINAMATH_CALUDE_ant_return_probability_l3633_363357

/-- Represents a vertex in a tetrahedron -/
inductive Vertex : Type
  | A | B | C | D

/-- Represents the state of the ant's position -/
structure AntState :=
  (position : Vertex)
  (distance : ℕ)

/-- The probability of choosing any edge at a vertex -/
def edgeProbability : ℚ := 1 / 3

/-- The total distance the ant needs to travel -/
def totalDistance : ℕ := 4

/-- Function to calculate the probability of the ant being at a specific vertex after a certain distance -/
noncomputable def probabilityAtVertex (v : Vertex) (d : ℕ) : ℚ :=
  sorry

/-- Theorem stating the probability of the ant returning to vertex A after 4 moves -/
theorem ant_return_probability :
  probabilityAtVertex Vertex.A totalDistance = 7 / 27 :=
sorry

end NUMINAMATH_CALUDE_ant_return_probability_l3633_363357
