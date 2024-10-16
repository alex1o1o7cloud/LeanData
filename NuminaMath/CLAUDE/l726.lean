import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l726_72637

/-- A geometric sequence with common ratio r satisfying a_n * a_(n+1) = 16^n has r = 4 --/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_prod : ∀ n, a n * a (n + 1) = 16^n) : 
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l726_72637


namespace NUMINAMATH_CALUDE_center_number_l726_72650

/-- Represents a 3x3 grid with numbers from 1 to 9 --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions in the grid share an edge --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Check if the grid satisfies the consecutive number constraint --/
def consecutive_constraint (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ p1 p2 : Fin 3 × Fin 3,
    g p1.1 p1.2 = n ∧ g p2.1 p2.2 = n + 1 ∧ adjacent p1 p2

/-- Sum of corner numbers in the grid --/
def corner_sum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- The main theorem --/
theorem center_number (g : Grid) 
  (unique : ∀ i j k l : Fin 3, g i j = g k l → (i, j) = (k, l))
  (all_numbers : ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n)
  (consec : consecutive_constraint g)
  (corners : corner_sum g = 20) :
  g 1 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_center_number_l726_72650


namespace NUMINAMATH_CALUDE_company_choices_eq_24_l726_72698

/-- The number of ways two students can choose companies with exactly one overlap -/
def company_choices : ℕ :=
  -- Number of ways to choose the shared company
  4 *
  -- Ways for student A to choose the second company
  3 *
  -- Ways for student B to choose the second company
  2

/-- Theorem stating that the number of ways to choose companies with one overlap is 24 -/
theorem company_choices_eq_24 : company_choices = 24 := by
  sorry

end NUMINAMATH_CALUDE_company_choices_eq_24_l726_72698


namespace NUMINAMATH_CALUDE_borrowed_amount_proof_l726_72684

/-- Represents the simple interest calculation for a loan -/
structure LoanInfo where
  principal : ℝ
  rate : ℝ
  time : ℝ
  total_amount : ℝ

/-- Theorem stating that given the loan conditions, the principal amount is 5400 -/
theorem borrowed_amount_proof (loan : LoanInfo) 
  (h1 : loan.rate = 0.06)
  (h2 : loan.time = 9)
  (h3 : loan.total_amount = 8310)
  : loan.principal = 5400 := by
  sorry

#check borrowed_amount_proof

end NUMINAMATH_CALUDE_borrowed_amount_proof_l726_72684


namespace NUMINAMATH_CALUDE_eggs_for_blueberry_and_pecan_pies_l726_72663

theorem eggs_for_blueberry_and_pecan_pies 
  (total_eggs : ℕ) 
  (pumpkin_eggs : ℕ) 
  (apple_eggs : ℕ) 
  (cherry_eggs : ℕ) 
  (h1 : total_eggs = 1820)
  (h2 : pumpkin_eggs = 816)
  (h3 : apple_eggs = 384)
  (h4 : cherry_eggs = 120) :
  total_eggs - (pumpkin_eggs + apple_eggs + cherry_eggs) = 500 :=
by sorry

end NUMINAMATH_CALUDE_eggs_for_blueberry_and_pecan_pies_l726_72663


namespace NUMINAMATH_CALUDE_bouquet_cost_50_l726_72670

/-- Represents the cost function for bouquets at Bella's Blossom Shop -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_price := (36 : ℚ) / 18 * n.min 40
  let extra_price := if n > 40 then (36 : ℚ) / 18 * (n - 40) else 0
  let total_price := base_price + extra_price
  if n > 40 then total_price * (9 / 10) else total_price

theorem bouquet_cost_50 : bouquet_cost 50 = 90 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_50_l726_72670


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l726_72677

def scores : List ℝ := [84, 90, 87, 93, 88, 92]

theorem arithmetic_mean_of_scores : 
  (scores.sum / scores.length : ℝ) = 89 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l726_72677


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l726_72625

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x : ℝ), x = 1 ∧ (a^(x - 1) + 1 = x) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l726_72625


namespace NUMINAMATH_CALUDE_sum_of_coordinates_D_l726_72632

-- Define the points
def C : ℝ × ℝ := (10, 6)
def N : ℝ × ℝ := (4, 8)

-- Define D as a variable point
variable (D : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem sum_of_coordinates_D :
  is_midpoint N C D → D.1 + D.2 = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_D_l726_72632


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l726_72694

theorem negative_fraction_comparison : -1/2 < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l726_72694


namespace NUMINAMATH_CALUDE_distance_ratio_theorem_l726_72671

/-- Represents a square pyramid -/
structure SquarePyramid where
  -- Base side length
  a : ℝ
  -- Height
  h : ℝ
  -- Assume positive dimensions
  a_pos : 0 < a
  h_pos : 0 < h

/-- A point inside the base square -/
structure BasePoint where
  x : ℝ
  y : ℝ
  -- Assume the point is inside the base square
  x_bound : 0 < x ∧ x < 1
  y_bound : 0 < y ∧ y < 1

/-- Sum of distances from a point to the triangular faces -/
noncomputable def sumDistancesToFaces (p : SquarePyramid) (e : BasePoint) : ℝ := sorry

/-- Sum of distances from a point to the base edges -/
noncomputable def sumDistancesToEdges (p : SquarePyramid) (e : BasePoint) : ℝ := sorry

/-- The main theorem -/
theorem distance_ratio_theorem (p : SquarePyramid) (e : BasePoint) :
  sumDistancesToFaces p e / sumDistancesToEdges p e = p.h / p.a := by sorry

end NUMINAMATH_CALUDE_distance_ratio_theorem_l726_72671


namespace NUMINAMATH_CALUDE_darryl_melon_sales_l726_72646

/-- Calculates the total money made from selling melons given the initial quantities,
    prices, dropped/rotten melons, and remaining melons. -/
def total_money_made (initial_cantaloupes initial_honeydews : ℕ)
                     (price_cantaloupe price_honeydew : ℕ)
                     (dropped_cantaloupes rotten_honeydews : ℕ)
                     (remaining_cantaloupes remaining_honeydews : ℕ) : ℕ :=
  let sold_cantaloupes := initial_cantaloupes - remaining_cantaloupes - dropped_cantaloupes
  let sold_honeydews := initial_honeydews - remaining_honeydews - rotten_honeydews
  sold_cantaloupes * price_cantaloupe + sold_honeydews * price_honeydew

/-- Theorem stating that Darryl made $85 from selling melons. -/
theorem darryl_melon_sales : 
  total_money_made 30 27 2 3 2 3 8 9 = 85 := by
  sorry


end NUMINAMATH_CALUDE_darryl_melon_sales_l726_72646


namespace NUMINAMATH_CALUDE_expansion_coefficient_l726_72659

/-- The coefficient of x^2 in the expansion of (x - 2/x)^6 -/
def coefficient_x_squared : ℕ :=
  let n : ℕ := 6
  let k : ℕ := 2
  ((-2)^k : ℤ) * (Nat.choose n k) |>.natAbs

theorem expansion_coefficient :
  coefficient_x_squared = 60 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l726_72659


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l726_72633

theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2/2 + y^2/m = 1) →  -- ellipse equation
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (x^2/a^2 + y^2/b^2 = 1 ↔ x^2/2 + y^2/m = 1) ∧ 
    c^2 = a^2 - b^2 ∧ c/a = 1/2) →  -- eccentricity condition
  m = 3/2 ∨ m = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l726_72633


namespace NUMINAMATH_CALUDE_certain_value_calculation_l726_72640

theorem certain_value_calculation (x : ℝ) (v : ℝ) (h1 : x = 100) (h2 : 0.8 * x + v = x) : v = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_calculation_l726_72640


namespace NUMINAMATH_CALUDE_expected_messages_is_27_l726_72697

/-- Calculates the expected number of greeting messages --/
def expected_messages (total_colleagues : ℕ) 
  (probabilities : List ℝ) (people_counts : List ℕ) : ℝ :=
  List.sum (List.zipWith (· * ·) probabilities people_counts)

/-- Theorem: The expected number of greeting messages is 27 --/
theorem expected_messages_is_27 : 
  let total_colleagues : ℕ := 40
  let probabilities : List ℝ := [1, 0.8, 0.5, 0]
  let people_counts : List ℕ := [8, 15, 14, 3]
  expected_messages total_colleagues probabilities people_counts = 27 := by
  sorry

end NUMINAMATH_CALUDE_expected_messages_is_27_l726_72697


namespace NUMINAMATH_CALUDE_pages_copied_l726_72627

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 4

/-- The amount available in dollars -/
def available_dollars : ℕ := 25

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The number of pages that can be copied for $25 is 625 -/
theorem pages_copied (cost_per_page : ℕ) (available_dollars : ℕ) (cents_per_dollar : ℕ) :
  (available_dollars * cents_per_dollar) / cost_per_page = 625 :=
sorry

end NUMINAMATH_CALUDE_pages_copied_l726_72627


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_lcm_168_l726_72602

theorem consecutive_numbers_with_lcm_168 (a b c : ℕ) : 
  (b = a + 1) → (c = b + 1) → Nat.lcm (Nat.lcm a b) c = 168 → a + b + c = 21 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_with_lcm_168_l726_72602


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l726_72603

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (2 * Real.sin θ - Real.cos θ)

-- Define the Cartesian equation of a line
def line_equation (x y : ℝ) : Prop :=
  2 * y - x = 1

-- Theorem statement
theorem polar_to_cartesian_line :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ →
    y = r * Real.sin θ →
    line_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_polar_to_cartesian_line_l726_72603


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_11_l726_72645

theorem consecutive_integers_around_sqrt_11 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 11) → (Real.sqrt 11 < b) → (a + b = 7) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_11_l726_72645


namespace NUMINAMATH_CALUDE_solve_for_a_l726_72652

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | -5/3 < x ∧ x < 1/3}

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := |a * x - 2| < 3

-- Theorem statement
theorem solve_for_a : 
  (∀ x, x ∈ solution_set a ↔ inequality a x) → a = -3 :=
sorry

end NUMINAMATH_CALUDE_solve_for_a_l726_72652


namespace NUMINAMATH_CALUDE_randy_blocks_left_l726_72655

/-- Given a total number of blocks and the number of blocks used, 
    calculate the number of blocks left. -/
def blocks_left (total : ℕ) (used : ℕ) : ℕ := total - used

/-- Theorem stating that given 59 total blocks and 36 blocks used, 
    23 blocks are left. -/
theorem randy_blocks_left : blocks_left 59 36 = 23 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_left_l726_72655


namespace NUMINAMATH_CALUDE_robyn_cookie_sales_l726_72687

/-- Given that Robyn and Lucy sold a total of 98 packs of cookies,
    and Lucy sold 43 packs, prove that Robyn sold 55 packs. -/
theorem robyn_cookie_sales (total : ℕ) (lucy : ℕ) (robyn : ℕ)
    (h1 : total = 98)
    (h2 : lucy = 43)
    (h3 : total = lucy + robyn) :
  robyn = 55 := by
  sorry

end NUMINAMATH_CALUDE_robyn_cookie_sales_l726_72687


namespace NUMINAMATH_CALUDE_factor_tree_X_value_l726_72666

def factor_tree (X Y Z Q R : ℕ) : Prop :=
  Y = 2 * Q ∧
  Z = 7 * R ∧
  Q = 5 * 3 ∧
  R = 11 * 2 ∧
  X = Y * Z

theorem factor_tree_X_value :
  ∀ X Y Z Q R : ℕ, factor_tree X Y Z Q R → X = 4620 :=
by
  sorry

end NUMINAMATH_CALUDE_factor_tree_X_value_l726_72666


namespace NUMINAMATH_CALUDE_mass_of_iodine_l726_72662

/-- The mass of 3 moles of I₂ given the atomic mass of I -/
theorem mass_of_iodine (atomic_mass_I : ℝ) (h : atomic_mass_I = 126.90) :
  let molar_mass_I2 := 2 * atomic_mass_I
  3 * molar_mass_I2 = 761.40 := by
  sorry

#check mass_of_iodine

end NUMINAMATH_CALUDE_mass_of_iodine_l726_72662


namespace NUMINAMATH_CALUDE_bridge_extension_l726_72668

/-- The width of the river in inches -/
def river_width : ℕ := 487

/-- The length of the existing bridge in inches -/
def existing_bridge_length : ℕ := 295

/-- The additional length needed for the bridge to cross the river -/
def additional_length : ℕ := river_width - existing_bridge_length

theorem bridge_extension :
  additional_length = 192 := by sorry

end NUMINAMATH_CALUDE_bridge_extension_l726_72668


namespace NUMINAMATH_CALUDE_negation_of_all_geq_two_l726_72615

theorem negation_of_all_geq_two :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x₀ : ℝ, x₀ < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_geq_two_l726_72615


namespace NUMINAMATH_CALUDE_hemisphere_center_of_mass_l726_72639

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a hemisphere -/
structure Hemisphere where
  radius : ℝ

/-- Density function for the hemisphere -/
def density (p : Point3D) : ℝ :=
  sorry

/-- Center of mass of a hemisphere -/
def centerOfMass (h : Hemisphere) : Point3D :=
  sorry

/-- Theorem: The center of mass of a hemisphere with radius R and volume density
    proportional to the distance from the origin is located at (0, 0, 2R/5) -/
theorem hemisphere_center_of_mass (h : Hemisphere) :
  let com := centerOfMass h
  com.x = 0 ∧ com.y = 0 ∧ com.z = 2 * h.radius / 5 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_center_of_mass_l726_72639


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l726_72631

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_property :
  (∀ a : ℕ → ℝ, is_geometric_sequence a → satisfies_condition a) ∧
  (∃ a : ℕ → ℝ, satisfies_condition a ∧ ¬is_geometric_sequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l726_72631


namespace NUMINAMATH_CALUDE_episodes_watched_per_day_l726_72665

theorem episodes_watched_per_day 
  (total_episodes : ℕ) 
  (total_days : ℕ) 
  (h1 : total_episodes = 50) 
  (h2 : total_days = 10) 
  (h3 : total_episodes > 0) 
  (h4 : total_days > 0) : 
  (total_episodes : ℚ) / total_days = 1 / 10 := by
  sorry

#check episodes_watched_per_day

end NUMINAMATH_CALUDE_episodes_watched_per_day_l726_72665


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l726_72622

/-- Calculates the length of a bridge given train and crossing parameters -/
theorem bridge_length_calculation
  (train_length : ℝ)
  (initial_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (wind_resistance_factor : ℝ)
  (h_train_length : train_length = 750)
  (h_initial_speed : initial_speed_kmh = 120)
  (h_crossing_time : crossing_time = 45)
  (h_wind_resistance : wind_resistance_factor = 0.9)
  : ∃ (bridge_length : ℝ), bridge_length = 600 :=
by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l726_72622


namespace NUMINAMATH_CALUDE_sequence_limit_l726_72610

/-- The sequence defined by the recurrence relation -/
noncomputable def x : ℕ → ℝ
| 0 => sorry -- x₁ is not specified in the original problem
| n + 1 => Real.sqrt (2 * x n + 3)

/-- The theorem stating that the limit of the sequence is 3 -/
theorem sequence_limit : Filter.Tendsto x Filter.atTop (nhds 3) := by sorry

end NUMINAMATH_CALUDE_sequence_limit_l726_72610


namespace NUMINAMATH_CALUDE_square_of_sum_80_5_l726_72669

theorem square_of_sum_80_5 : (80 + 5)^2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_80_5_l726_72669


namespace NUMINAMATH_CALUDE_circular_garden_area_increase_l726_72691

theorem circular_garden_area_increase : 
  let r₁ : ℝ := 6  -- radius of larger garden
  let r₂ : ℝ := 4  -- radius of smaller garden
  let area₁ := π * r₁^2  -- area of larger garden
  let area₂ := π * r₂^2  -- area of smaller garden
  (area₁ - area₂) / area₂ * 100 = 125
  := by sorry

end NUMINAMATH_CALUDE_circular_garden_area_increase_l726_72691


namespace NUMINAMATH_CALUDE_unique_integer_factorial_division_l726_72649

theorem unique_integer_factorial_division : 
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 60 ∧ (∃ k : ℕ, k * (Nat.factorial n)^(n + 2) = Nat.factorial (n^2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_factorial_division_l726_72649


namespace NUMINAMATH_CALUDE_correct_number_probability_l726_72621

def first_three_digits : ℕ := 3

def last_four_digits : List ℕ := [0, 1, 1, 7]

def permutations_of_last_four : ℕ := 12

theorem correct_number_probability :
  (1 : ℚ) / (first_three_digits * permutations_of_last_four) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_probability_l726_72621


namespace NUMINAMATH_CALUDE_cubic_difference_evaluation_l726_72682

theorem cubic_difference_evaluation : 
  2010^3 - 2007 * 2010^2 - 2007^2 * 2010 + 2007^3 = 36153 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_evaluation_l726_72682


namespace NUMINAMATH_CALUDE_octal_7624_is_decimal_3988_l726_72693

def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem octal_7624_is_decimal_3988 : octal_to_decimal 7624 = 3988 := by
  sorry

end NUMINAMATH_CALUDE_octal_7624_is_decimal_3988_l726_72693


namespace NUMINAMATH_CALUDE_fewer_children_than_adults_l726_72607

theorem fewer_children_than_adults : 
  ∀ (children seniors : ℕ),
  58 + children + seniors = 127 →
  seniors = 2 * children →
  58 - children = 35 := by
sorry

end NUMINAMATH_CALUDE_fewer_children_than_adults_l726_72607


namespace NUMINAMATH_CALUDE_inscribed_square_area_is_2275_l726_72606

/-- A right triangle with an inscribed square -/
structure InscribedSquareTriangle where
  /-- The length of side XY of the triangle -/
  xy : ℝ
  /-- The length of side ZQ of the triangle -/
  zq : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- Condition ensuring the square fits in the triangle -/
  square_fits : square_side ≤ min xy zq

/-- The area of the inscribed square in the triangle -/
def inscribed_square_area (t : InscribedSquareTriangle) : ℝ := t.square_side ^ 2

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area_is_2275 
  (t : InscribedSquareTriangle) 
  (h1 : t.xy = 35) 
  (h2 : t.zq = 65) : 
  inscribed_square_area t = 2275 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_is_2275_l726_72606


namespace NUMINAMATH_CALUDE_jack_and_jill_speed_l726_72617

/-- Jack and Jill's Mountain Climb Theorem -/
theorem jack_and_jill_speed (x : ℝ) : 
  (x^2 - 13*x - 26 = (x^2 - 5*x - 66) / (x + 8)) → 
  (x^2 - 13*x - 26 = 4) := by
  sorry

#check jack_and_jill_speed

end NUMINAMATH_CALUDE_jack_and_jill_speed_l726_72617


namespace NUMINAMATH_CALUDE_car_oil_problem_l726_72681

/-- Represents the relationship between remaining oil and distance traveled for a car -/
def oil_remaining (x : ℝ) : ℝ := 56 - 0.08 * x

/-- The initial amount of oil in the tank -/
def initial_oil : ℝ := 56

/-- The rate of oil consumption per kilometer -/
def consumption_rate : ℝ := 0.08

theorem car_oil_problem :
  (∀ x : ℝ, oil_remaining x = 56 - 0.08 * x) ∧
  (oil_remaining 350 = 28) ∧
  (∃ x : ℝ, oil_remaining x = 8 ∧ x = 600) := by
  sorry


end NUMINAMATH_CALUDE_car_oil_problem_l726_72681


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l726_72642

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Main theorem
theorem fixed_point_on_line (x₁ y₁ x₂ y₂ : ℝ) :
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  perpendicular x₁ y₁ x₂ y₂ →
  ∃ m : ℝ, x₁ = m*y₁ + 8 ∧ x₂ = m*y₂ + 8 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l726_72642


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l726_72630

theorem existence_of_counterexample (x y : ℝ) (h : x > y) :
  ∃ x y : ℝ, x > y ∧ x^2 - 3 ≤ y^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l726_72630


namespace NUMINAMATH_CALUDE_divisible_by_eight_l726_72647

theorem divisible_by_eight (b n : ℕ) (h1 : Even b) (h2 : b > 0) (h3 : n > 1)
  (h4 : ∃ k : ℕ, (b^n - 1) / (b - 1) = k^2) : 
  8 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l726_72647


namespace NUMINAMATH_CALUDE_amanda_almonds_l726_72600

theorem amanda_almonds (total_almonds : ℚ) (num_piles : ℕ) (amanda_piles : ℕ) : 
  total_almonds = 66 / 7 →
  num_piles = 6 →
  amanda_piles = 3 →
  amanda_piles * (total_almonds / num_piles) = 33 / 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_almonds_l726_72600


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l726_72678

noncomputable def f (x : ℝ) : ℝ := (2^x) / (2 * (Real.log 2 - 1) * x)

theorem f_derivative_at_one :
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l726_72678


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l726_72643

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l726_72643


namespace NUMINAMATH_CALUDE_stating_traffic_light_probability_l726_72680

/-- Represents the duration of a traffic light cycle in seconds. -/
def cycleDuration : ℕ := 80

/-- Represents the duration of time when proceeding is allowed (green + yellow) in seconds. -/
def proceedDuration : ℕ := 50

/-- Represents the duration of time when proceeding is not allowed (red) in seconds. -/
def stopDuration : ℕ := 30

/-- Represents the maximum waiting time in seconds for the probability calculation. -/
def maxWaitTime : ℕ := 10

/-- 
Theorem stating that the probability of waiting no more than 10 seconds to proceed 
in the given traffic light cycle is 3/4.
-/
theorem traffic_light_probability : 
  (proceedDuration + maxWaitTime : ℚ) / cycleDuration = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_stating_traffic_light_probability_l726_72680


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l726_72648

theorem quadrilateral_angle_measure (W X Y Z : ℝ) : 
  W > 0 ∧ X > 0 ∧ Y > 0 ∧ Z > 0 →
  W = 3 * X ∧ W = 4 * Y ∧ W = 6 * Z →
  W + X + Y + Z = 360 →
  W = 1440 / 7 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l726_72648


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l726_72690

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- Expected value of a random variable -/
def expected_value (ξ : NormalRandomVariable) : ℝ := ξ.μ

/-- Variance of a random variable -/
def variance (ξ : NormalRandomVariable) : ℝ := ξ.σ^2

/-- Probability of a random variable falling within a certain range -/
def probability (ξ : NormalRandomVariable) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (ξ : NormalRandomVariable) 
  (h1 : expected_value ξ = 3) 
  (h2 : variance ξ = 1) 
  (h3 : probability ξ (ξ.μ - ξ.σ) (ξ.μ + ξ.σ) = 0.683) : 
  probability ξ 2 4 = 0.683 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l726_72690


namespace NUMINAMATH_CALUDE_hexagon_circle_area_ratio_l726_72685

theorem hexagon_circle_area_ratio (r : ℝ) (h : r > 0) :
  (3 * Real.sqrt 3 * r^2 / 2) / (π * r^2) = 3 * Real.sqrt 3 / (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_hexagon_circle_area_ratio_l726_72685


namespace NUMINAMATH_CALUDE_product_of_roots_l726_72623

theorem product_of_roots (x : ℝ) : (x - 1) * (x + 4) = 22 → ∃ y : ℝ, (x - 1) * (x + 4) = 22 ∧ (y - 1) * (y + 4) = 22 ∧ x * y = -26 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l726_72623


namespace NUMINAMATH_CALUDE_davids_physics_marks_l726_72667

/-- Calculates the marks in Physics given marks in other subjects and the average --/
def physics_marks (english : ℕ) (math : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + math + chemistry + biology)

/-- Proves that David's marks in Physics are 82 given his other marks and average --/
theorem davids_physics_marks :
  physics_marks 61 65 67 85 72 = 82 := by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l726_72667


namespace NUMINAMATH_CALUDE_inverse_composition_result_l726_72636

-- Define the function f
def f : Fin 5 → Fin 5
| 1 => 3
| 2 => 5
| 3 => 1
| 4 => 2
| 5 => 4

-- Define the inverse function f⁻¹
def f_inv : Fin 5 → Fin 5
| 1 => 3
| 2 => 4
| 3 => 1
| 4 => 5
| 5 => 2

-- State the theorem
theorem inverse_composition_result :
  f_inv (f_inv (f_inv 5)) = 5 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_result_l726_72636


namespace NUMINAMATH_CALUDE_apples_bought_l726_72661

/-- The number of apples bought by Cecile, Diane, and Emily -/
def total_apples (cecile diane emily : ℕ) : ℕ := cecile + diane + emily

/-- The theorem stating the total number of apples bought -/
theorem apples_bought (cecile diane emily : ℕ) 
  (h1 : cecile = 15)
  (h2 : diane = cecile + 20)
  (h3 : emily = ((cecile + diane) * 13) / 10) :
  total_apples cecile diane emily = 115 := by
  sorry

#check apples_bought

end NUMINAMATH_CALUDE_apples_bought_l726_72661


namespace NUMINAMATH_CALUDE_line_sum_m_b_l726_72616

/-- A line passing through points (-2, 0) and (0, 2) can be represented by y = mx + b -/
def line_equation (x y m b : ℝ) : Prop := y = m * x + b

/-- The line passes through (-2, 0) -/
def point1_condition (m b : ℝ) : Prop := line_equation (-2) 0 m b

/-- The line passes through (0, 2) -/
def point2_condition (m b : ℝ) : Prop := line_equation 0 2 m b

/-- Theorem: For a line passing through (-2, 0) and (0, 2), represented by y = mx + b, m + b = 3 -/
theorem line_sum_m_b : ∀ m b : ℝ, point1_condition m b → point2_condition m b → m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_sum_m_b_l726_72616


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l726_72612

theorem sum_of_roots_quadratic (z : ℂ) : 
  (∃ z₁ z₂ : ℂ, z₁ + z₂ = 16 ∧ z₁ * z₂ = 15 ∧ z₁ ≠ z₂ ∧ z^2 - 16*z + 15 = 0 → z = z₁ ∨ z = z₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l726_72612


namespace NUMINAMATH_CALUDE_biscuit_count_l726_72672

-- Define the dimensions of the dough sheet
def dough_side : ℕ := 12

-- Define the dimensions of each biscuit
def biscuit_side : ℕ := 3

-- Theorem to prove
theorem biscuit_count : (dough_side * dough_side) / (biscuit_side * biscuit_side) = 16 := by
  sorry

end NUMINAMATH_CALUDE_biscuit_count_l726_72672


namespace NUMINAMATH_CALUDE_probability_three_same_color_l726_72664

/-- Represents a person in the block placement scenario -/
structure Person where
  name : String
  blocks : Fin 5 → Color

/-- Represents the colors of the blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | White
  | Green

/-- Represents the result of a single trial -/
structure Trial where
  placements : Person → Fin 6 → Option (Fin 5)

/-- The probability of a specific event occurring in the trial -/
def probability (event : Trial → Prop) : ℚ :=
  sorry

/-- Checks if a trial results in at least one box with 3 blocks of the same color -/
def has_three_same_color (t : Trial) : Prop :=
  sorry

/-- The main theorem stating the probability of the event -/
theorem probability_three_same_color 
  (ang ben jasmin : Person)
  (h1 : ang.name = "Ang" ∧ ben.name = "Ben" ∧ jasmin.name = "Jasmin")
  (h2 : ∀ p : Person, p = ang ∨ p = ben ∨ p = jasmin → 
        ∀ i : Fin 5, ∃! c : Color, p.blocks i = c) :
  probability has_three_same_color = 5 / 216 :=
sorry

end NUMINAMATH_CALUDE_probability_three_same_color_l726_72664


namespace NUMINAMATH_CALUDE_unique_triple_sum_l726_72634

/-- Two-digit positive integer -/
def TwoDigitPositiveInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_triple_sum (a b c : ℕ) 
  (ha : TwoDigitPositiveInt a) 
  (hb : TwoDigitPositiveInt b) 
  (hc : TwoDigitPositiveInt c) 
  (h : a^3 + 3*b^3 + 9*c^3 = 9*a*b*c + 1) : 
  a + b + c = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_triple_sum_l726_72634


namespace NUMINAMATH_CALUDE_solve_stick_problem_l726_72604

def stick_problem (dave_sticks amy_sticks ben_sticks total_sticks : ℕ) : Prop :=
  let total_picked := dave_sticks + amy_sticks + ben_sticks
  let sticks_left := total_sticks - total_picked
  total_picked - sticks_left = 5

theorem solve_stick_problem :
  stick_problem 14 9 12 65 := by
  sorry

end NUMINAMATH_CALUDE_solve_stick_problem_l726_72604


namespace NUMINAMATH_CALUDE_billboard_perimeter_l726_72601

/-- Represents a rectangular billboard --/
structure Billboard where
  length : ℝ
  width : ℝ

/-- The area of a billboard --/
def area (b : Billboard) : ℝ := b.length * b.width

/-- The perimeter of a billboard --/
def perimeter (b : Billboard) : ℝ := 2 * (b.length + b.width)

theorem billboard_perimeter :
  ∀ b : Billboard,
    area b = 91 ∧
    b.width = 7 →
    perimeter b = 40 := by
  sorry

end NUMINAMATH_CALUDE_billboard_perimeter_l726_72601


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l726_72679

theorem mean_equality_implies_z_value : 
  let mean1 := (8 + 14 + 24) / 3
  let mean2 := (16 + z) / 2
  mean1 = mean2 → z = 44 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l726_72679


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l726_72651

theorem modulo_residue_problem : (513 + 3 * 68 + 9 * 289 + 2 * 34 - 10) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l726_72651


namespace NUMINAMATH_CALUDE_function_composition_sqrt2_l726_72676

/-- Given a function f(x) = a * x^2 - √2, where a is a constant,
    if f(f(√2)) = -√2, then a = √2 / 2 -/
theorem function_composition_sqrt2 (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a * x^2 - Real.sqrt 2) →
  f (f (Real.sqrt 2)) = -Real.sqrt 2 →
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_sqrt2_l726_72676


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l726_72654

theorem arctan_tan_difference (θ : Real) : 
  0 ≤ θ ∧ θ ≤ 180 ∧ 
  θ = Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (15 * π / 180)) * 180 / π :=
by sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l726_72654


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l726_72696

theorem simplify_nested_expression (x : ℝ) : 1 - (1 - (1 + (1 - (1 + (1 - x))))) = 3 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l726_72696


namespace NUMINAMATH_CALUDE_jills_income_ratio_l726_72673

/-- Proves that the ratio of Jill's discretionary income to her net monthly salary is 1/5 -/
theorem jills_income_ratio :
  let net_salary : ℚ := 3500
  let discretionary_income : ℚ := 105 / (15/100)
  discretionary_income / net_salary = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_jills_income_ratio_l726_72673


namespace NUMINAMATH_CALUDE_cube_of_square_of_second_smallest_prime_l726_72605

-- Define the second smallest prime number
def second_smallest_prime : Nat := 3

-- Theorem statement
theorem cube_of_square_of_second_smallest_prime : 
  (second_smallest_prime ^ 2) ^ 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_square_of_second_smallest_prime_l726_72605


namespace NUMINAMATH_CALUDE_nonreal_roots_product_l726_72699

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 4*x^3 + 6*x^2 - 4*x + 4 = 4036) → 
  (∃ a b : ℂ, a ≠ b ∧ a.im ≠ 0 ∧ b.im ≠ 0 ∧ 
   (x = a ∨ x = b) ∧ 
   (x^4 - 4*x^3 + 6*x^2 - 4*x + 4 = 4036) ∧
   (a * b = 1 + Real.sqrt 4033)) :=
by sorry

end NUMINAMATH_CALUDE_nonreal_roots_product_l726_72699


namespace NUMINAMATH_CALUDE_set_inclusion_conditions_l726_72660

def P : Set ℝ := {x | x^2 + 4*x = 0}
def Q (m : ℝ) : Set ℝ := {x | x^2 + 2*(m+1)*x + m^2 - 1 = 0}

theorem set_inclusion_conditions :
  (∀ m : ℝ, P ⊆ Q m ↔ m = 1) ∧
  (∀ m : ℝ, Q m ⊆ P ↔ m ≤ -1 ∨ m = 1) := by sorry

end NUMINAMATH_CALUDE_set_inclusion_conditions_l726_72660


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_right_triangle_max_ratio_equality_l726_72628

theorem right_triangle_max_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  (a^2 + b^2 + a*b) / c^2 ≤ 1.5 := by
sorry

theorem right_triangle_max_ratio_equality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧ (a^2 + b^2 + a*b) / c^2 = 1.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_right_triangle_max_ratio_equality_l726_72628


namespace NUMINAMATH_CALUDE_pink_highlighters_count_l726_72644

theorem pink_highlighters_count (total yellow blue : ℕ) (h1 : total = 22) (h2 : yellow = 8) (h3 : blue = 5) :
  ∃ pink : ℕ, pink + yellow + blue = total ∧ pink = 9 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_count_l726_72644


namespace NUMINAMATH_CALUDE_sum_extrema_l726_72688

theorem sum_extrema (x y z w : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ w ≥ 0) 
  (h_eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17/2) :
  (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 → 
    a + b + c + d ≤ 3) ∧
  (x + y + z + w ≥ -2 + 5/2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_extrema_l726_72688


namespace NUMINAMATH_CALUDE_range_of_m_l726_72656

open Real Set

theorem range_of_m (p q : Prop) (m : ℝ) : 
  (∀ x ∈ Icc 0 (π/4), tan x ≤ m) → p ∧
  (∀ x₁ ∈ Icc (-1) 3, ∃ x₂ ∈ Icc 0 2, x₁^2 + m ≥ (1/2)^x₂ - m) → q ∧
  (¬(p ∧ q)) ∧ (p ∨ q) →
  m ∈ Icc (1/8) 1 ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l726_72656


namespace NUMINAMATH_CALUDE_constant_value_l726_72683

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f x + c * f (8 - x) = x

theorem constant_value (f : ℝ → ℝ) (c : ℝ) 
    (h1 : SatisfiesCondition f c) 
    (h2 : f 2 = 2) : 
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l726_72683


namespace NUMINAMATH_CALUDE_polygon_diagonals_l726_72675

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) :
  (n - 3 ≤ 6) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l726_72675


namespace NUMINAMATH_CALUDE_apple_profit_calculation_l726_72658

/-- Calculates the total percentage profit for a shopkeeper selling apples -/
theorem apple_profit_calculation (total_apples : ℝ) (first_portion : ℝ) (second_portion : ℝ)
  (first_profit_rate : ℝ) (second_profit_rate : ℝ) 
  (h1 : total_apples = 100)
  (h2 : first_portion = 0.5 * total_apples)
  (h3 : second_portion = 0.5 * total_apples)
  (h4 : first_profit_rate = 0.25)
  (h5 : second_profit_rate = 0.3) :
  (((first_portion * (1 + first_profit_rate) + second_portion * (1 + second_profit_rate)) - total_apples) / total_apples) * 100 = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_apple_profit_calculation_l726_72658


namespace NUMINAMATH_CALUDE_existence_of_sequence_with_divisors_l726_72689

theorem existence_of_sequence_with_divisors :
  ∃ f : ℕ → ℕ, ∀ k : ℕ, ∃ d : Finset ℕ,
    d.card ≥ k ∧
    ∀ x ∈ d, x > 0 ∧ (f k)^2 + f k + 2023 ≡ 0 [MOD x] :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sequence_with_divisors_l726_72689


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l726_72608

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f 2 x ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f a x ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l726_72608


namespace NUMINAMATH_CALUDE_programmer_is_odd_one_out_l726_72620

/-- Represents a profession --/
inductive Profession
  | Dentist
  | ElementarySchoolTeacher
  | Programmer

/-- Represents whether a profession has special pension benefits --/
def has_special_pension_benefits (p : Profession) : Prop :=
  match p with
  | Profession.Dentist => true
  | Profession.ElementarySchoolTeacher => true
  | Profession.Programmer => false

/-- Theorem stating that the programmer is the odd one out --/
theorem programmer_is_odd_one_out :
  ∃! p : Profession, ¬(has_special_pension_benefits p) :=
sorry


end NUMINAMATH_CALUDE_programmer_is_odd_one_out_l726_72620


namespace NUMINAMATH_CALUDE_distribute_four_students_three_groups_l726_72626

/-- The number of ways to distribute n distinct students into k distinct groups,
    where each student is in exactly one group and each group has at least one member. -/
def distribute_students (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 4 distinct students into 3 distinct groups,
    where each student is in exactly one group and each group has at least one member, is 36. -/
theorem distribute_four_students_three_groups :
  distribute_students 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_four_students_three_groups_l726_72626


namespace NUMINAMATH_CALUDE_binomial_20_2_l726_72657

theorem binomial_20_2 : Nat.choose 20 2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_2_l726_72657


namespace NUMINAMATH_CALUDE_same_color_pen_probability_l726_72618

theorem same_color_pen_probability (blue_pens black_pens : ℕ) 
  (h1 : blue_pens = 8) (h2 : black_pens = 5) : 
  let total_pens := blue_pens + black_pens
  (blue_pens / total_pens)^2 + (black_pens / total_pens)^2 = 89 / 169 := by
  sorry

end NUMINAMATH_CALUDE_same_color_pen_probability_l726_72618


namespace NUMINAMATH_CALUDE_find_x_value_l726_72641

theorem find_x_value (x : ℝ) : 3639 + 11.95 - x = 3054 → x = 596.95 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l726_72641


namespace NUMINAMATH_CALUDE_cookfire_logs_remaining_l726_72624

/-- Represents the number of logs remaining in a cookfire after a given number of hours -/
def logs_remaining (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (hours : ℕ) : ℤ :=
  initial_logs + hours * (add_rate - burn_rate)

/-- Theorem stating the number of logs remaining after x hours for the given cookfire scenario -/
theorem cookfire_logs_remaining (x : ℕ) :
  logs_remaining 8 4 3 x = 8 - x :=
sorry

end NUMINAMATH_CALUDE_cookfire_logs_remaining_l726_72624


namespace NUMINAMATH_CALUDE_recurrence_relation_solution_l726_72619

def a (n : ℕ) : ℤ := 1 + 5 * 3^n - 4 * 2^n

theorem recurrence_relation_solution :
  (∀ n : ℕ, n ≥ 2 → a n = 4 * a (n - 1) - 3 * a (n - 2) + 2^n) ∧
  a 0 = 2 ∧
  a 1 = 8 := by
sorry

end NUMINAMATH_CALUDE_recurrence_relation_solution_l726_72619


namespace NUMINAMATH_CALUDE_sine_difference_inequality_l726_72614

theorem sine_difference_inequality (A B : Real) (hA : 0 ≤ A ∧ A ≤ π) (hB : 0 ≤ B ∧ B ≤ π) :
  |Real.sin A - Real.sin B| ≤ |Real.sin (A - B)| := by
  sorry

end NUMINAMATH_CALUDE_sine_difference_inequality_l726_72614


namespace NUMINAMATH_CALUDE_product_equals_zero_l726_72695

theorem product_equals_zero (b : ℤ) (h : b = 3) : 
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l726_72695


namespace NUMINAMATH_CALUDE_specific_lamp_arrangement_probability_l726_72692

/-- The probability of a specific lamp arrangement and state --/
def specific_arrangement_probability (total_lamps : ℕ) (purple_lamps : ℕ) (green_lamps : ℕ) (lamps_on : ℕ) : ℚ :=
  let total_arrangements := Nat.choose total_lamps purple_lamps * Nat.choose total_lamps lamps_on
  let specific_arrangements := Nat.choose (total_lamps - 2) (purple_lamps - 1) * Nat.choose (total_lamps - 2) (lamps_on - 1)
  specific_arrangements / total_arrangements

/-- The main theorem statement --/
theorem specific_lamp_arrangement_probability :
  specific_arrangement_probability 8 4 4 4 = 4 / 49 := by
  sorry

end NUMINAMATH_CALUDE_specific_lamp_arrangement_probability_l726_72692


namespace NUMINAMATH_CALUDE_tan_beta_value_l726_72686

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan (α + β) = 1/5) : 
  Real.tan β = -9/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l726_72686


namespace NUMINAMATH_CALUDE_whale_first_hour_consumption_l726_72638

def whale_feeding (first_hour : ℕ) : Prop :=
  let second_hour := first_hour + 3
  let third_hour := first_hour + 6
  let fourth_hour := first_hour + 9
  let fifth_hour := first_hour + 12
  (third_hour = 93) ∧ 
  (first_hour + second_hour + third_hour + fourth_hour + fifth_hour = 450)

theorem whale_first_hour_consumption : 
  ∃ (x : ℕ), whale_feeding x ∧ x = 87 :=
sorry

end NUMINAMATH_CALUDE_whale_first_hour_consumption_l726_72638


namespace NUMINAMATH_CALUDE_sqrt_of_negative_eight_squared_l726_72613

theorem sqrt_of_negative_eight_squared : Real.sqrt ((-8)^2) = 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_negative_eight_squared_l726_72613


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l726_72629

theorem pure_imaginary_product (x : ℝ) : 
  (Complex.I * 2 + x) * (Complex.I * 2 + (x + 2)) * (Complex.I * 2 + (x + 4)) ∈ {z : ℂ | z.re = 0} ↔ 
  x = -4 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l726_72629


namespace NUMINAMATH_CALUDE_complement_of_intersection_l726_72611

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l726_72611


namespace NUMINAMATH_CALUDE_rectangle_max_area_l726_72609

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) (perimeter : x + y = 24) :
  x * y ≤ 144 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 24 ∧ a * b = 144 := by
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l726_72609


namespace NUMINAMATH_CALUDE_gravel_cost_theorem_l726_72653

/-- The cost of gravel in dollars per cubic foot -/
def gravel_cost_per_cubic_foot : ℝ := 4

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of gravel in cubic yards -/
def gravel_volume_cubic_yards : ℝ := 8

/-- Theorem stating the cost of 8 cubic yards of gravel -/
theorem gravel_cost_theorem :
  gravel_cost_per_cubic_foot * cubic_yards_to_cubic_feet * gravel_volume_cubic_yards = 864 := by
  sorry

end NUMINAMATH_CALUDE_gravel_cost_theorem_l726_72653


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l726_72635

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x, (1 - a) * x > 3 ↔ x < 3 / (1 - a)) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l726_72635


namespace NUMINAMATH_CALUDE_math_team_combinations_l726_72674

theorem math_team_combinations (girls : ℕ) (boys : ℕ) : 
  girls = 3 → boys = 5 → (girls.choose 2) * (boys.choose 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l726_72674
