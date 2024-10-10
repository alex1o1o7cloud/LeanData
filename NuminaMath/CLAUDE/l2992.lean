import Mathlib

namespace sum_of_ratios_theorem_l2992_299221

theorem sum_of_ratios_theorem (a b c : ℚ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a * b * c ≠ 0) (h5 : a + b + c = 0) : 
  a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 :=
by sorry

end sum_of_ratios_theorem_l2992_299221


namespace trapezoid_perimeter_is_200_l2992_299244

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  AD : ℝ
  BC : ℝ
  angle_BAD : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + t.CD + t.AD + t.BC

/-- Theorem stating that the perimeter of the given trapezoid is 200 units -/
theorem trapezoid_perimeter_is_200 (t : Trapezoid) 
  (h1 : t.AB = 40)
  (h2 : t.CD = 35)
  (h3 : t.AD = 70)
  (h4 : t.BC = 55)
  (h5 : t.angle_BAD = 30 * π / 180)  -- Convert 30° to radians
  : perimeter t = 200 := by
  sorry

#check trapezoid_perimeter_is_200

end trapezoid_perimeter_is_200_l2992_299244


namespace equation_root_implies_m_equals_three_l2992_299249

theorem equation_root_implies_m_equals_three (x m : ℝ) :
  (x ≠ 3) →
  (x / (x - 3) = 2 - m / (3 - x)) →
  m = 3 :=
by sorry

end equation_root_implies_m_equals_three_l2992_299249


namespace restaurant_group_cost_l2992_299224

/-- Represents the cost structure and group composition at a restaurant -/
structure RestaurantGroup where
  adult_meal_cost : ℚ
  adult_drink_cost : ℚ
  adult_dessert_cost : ℚ
  kid_meal_cost : ℚ
  kid_drink_cost : ℚ
  kid_dessert_cost : ℚ
  total_people : ℕ
  num_kids : ℕ

/-- Calculates the total cost for a restaurant group -/
def total_cost (g : RestaurantGroup) : ℚ :=
  let num_adults := g.total_people - g.num_kids
  let adult_cost := num_adults * (g.adult_meal_cost + g.adult_drink_cost + g.adult_dessert_cost)
  let kid_cost := g.num_kids * (g.kid_meal_cost + g.kid_drink_cost + g.kid_dessert_cost)
  adult_cost + kid_cost

/-- Theorem stating that the total cost for the given group is $87.50 -/
theorem restaurant_group_cost :
  let g : RestaurantGroup := {
    adult_meal_cost := 7
    adult_drink_cost := 4
    adult_dessert_cost := 3
    kid_meal_cost := 0
    kid_drink_cost := 2
    kid_dessert_cost := 3/2
    total_people := 13
    num_kids := 9
  }
  total_cost g = 175/2 := by sorry

end restaurant_group_cost_l2992_299224


namespace arithmetic_sequence_line_point_l2992_299258

/-- If k, -1, b are three numbers in arithmetic sequence, 
    then the line y = kx + b passes through the point (1, -2). -/
theorem arithmetic_sequence_line_point (k b : ℝ) : 
  (∃ d : ℝ, k = -1 - d ∧ b = -1 + d) → 
  k * 1 + b = -2 := by
  sorry

end arithmetic_sequence_line_point_l2992_299258


namespace nancy_spend_l2992_299254

/-- The cost of a set of crystal beads in dollars -/
def crystal_cost : ℕ := 9

/-- The cost of a set of metal beads in dollars -/
def metal_cost : ℕ := 10

/-- The number of crystal bead sets Nancy buys -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets Nancy buys -/
def metal_sets : ℕ := 2

/-- The total amount Nancy spends in dollars -/
def total_cost : ℕ := crystal_cost * crystal_sets + metal_cost * metal_sets

theorem nancy_spend : total_cost = 29 := by
  sorry

end nancy_spend_l2992_299254


namespace intersection_of_A_and_B_l2992_299241

def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {4} := by sorry

end intersection_of_A_and_B_l2992_299241


namespace fundraiser_group_composition_l2992_299269

theorem fundraiser_group_composition (p : ℕ) : 
  p > 0 ∧ 
  (p / 2 : ℚ) = (p / 2 : ℕ) ∧ 
  ((p / 2 - 2 : ℚ) / p = 2 / 5) → 
  p / 2 = 10 := by
  sorry

end fundraiser_group_composition_l2992_299269


namespace lightsaber_to_other_toys_ratio_l2992_299220

-- Define the cost of other Star Wars toys
def other_toys_cost : ℕ := 1000

-- Define the total spent
def total_spent : ℕ := 3000

-- Define the cost of the lightsaber
def lightsaber_cost : ℕ := total_spent - other_toys_cost

-- Theorem statement
theorem lightsaber_to_other_toys_ratio :
  (lightsaber_cost : ℚ) / other_toys_cost = 2 := by sorry

end lightsaber_to_other_toys_ratio_l2992_299220


namespace price_decrease_proof_l2992_299282

/-- The original price of an article before a price decrease -/
def original_price : ℝ := 1300

/-- The percentage of the original price after the decrease -/
def price_decrease_percentage : ℝ := 24

/-- The price of the article after the decrease -/
def decreased_price : ℝ := 988

theorem price_decrease_proof : 
  (1 - price_decrease_percentage / 100) * original_price = decreased_price := by
  sorry

end price_decrease_proof_l2992_299282


namespace clothing_retailer_optimal_strategy_l2992_299226

/-- Represents the clothing retailer's purchase and sales data --/
structure ClothingRetailer where
  first_purchase_cost : ℝ
  second_purchase_cost : ℝ
  cost_increase_per_item : ℝ
  base_price : ℝ
  base_sales : ℝ
  price_decrease : ℝ
  sales_increase : ℝ
  daily_profit : ℝ

/-- Theorem stating the initial purchase quantity and price, and the optimal selling price --/
theorem clothing_retailer_optimal_strategy (r : ClothingRetailer)
  (h1 : r.first_purchase_cost = 48000)
  (h2 : r.second_purchase_cost = 100000)
  (h3 : r.cost_increase_per_item = 10)
  (h4 : r.base_price = 300)
  (h5 : r.base_sales = 80)
  (h6 : r.price_decrease = 10)
  (h7 : r.sales_increase = 20)
  (h8 : r.daily_profit = 3600) :
  ∃ (initial_quantity : ℝ) (initial_price : ℝ) (selling_price : ℝ),
    initial_quantity = 200 ∧
    initial_price = 240 ∧
    selling_price = 280 ∧
    (selling_price - (initial_price + r.cost_increase_per_item)) *
      (r.base_sales + (r.base_price - selling_price) / r.price_decrease * r.sales_increase) = r.daily_profit :=
by sorry

end clothing_retailer_optimal_strategy_l2992_299226


namespace shirt_trouser_combinations_l2992_299255

theorem shirt_trouser_combinations (shirt_styles : ℕ) (trouser_colors : ℕ) 
  (h1 : shirt_styles = 4) (h2 : trouser_colors = 3) : 
  shirt_styles * trouser_colors = 12 := by
  sorry

end shirt_trouser_combinations_l2992_299255


namespace circumscribed_circle_equation_l2992_299287

theorem circumscribed_circle_equation (A B C : ℝ × ℝ) :
  A = (4, 1) → B = (6, -3) → C = (-3, 0) →
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = r^2 ↔
      (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = C.1 ∧ y = C.2)) ∧
    center = (1, -3) ∧ r^2 = 25 := by sorry

end circumscribed_circle_equation_l2992_299287


namespace december_sales_fraction_l2992_299277

theorem december_sales_fraction (average_sales : ℝ) (h : average_sales > 0) :
  let other_months_total := 11 * average_sales
  let december_sales := 6 * average_sales
  let annual_sales := other_months_total + december_sales
  december_sales / annual_sales = 6 / 17 := by
sorry

end december_sales_fraction_l2992_299277


namespace hexagon_perimeter_hexagon_perimeter_proof_l2992_299250

/-- The perimeter of a regular hexagon with side length 2 inches is 12 inches. -/
theorem hexagon_perimeter : ℝ → Prop :=
  fun (side_length : ℝ) =>
    side_length = 2 →
    6 * side_length = 12

/-- Proof of the theorem -/
theorem hexagon_perimeter_proof : hexagon_perimeter 2 := by
  sorry

end hexagon_perimeter_hexagon_perimeter_proof_l2992_299250


namespace johns_donation_l2992_299219

/-- Calculates the size of a donation that increases the average contribution by 75% to $100 when added to 10 existing contributions. -/
theorem johns_donation (initial_contributions : ℕ) (increase_percentage : ℚ) (new_average : ℚ) : 
  initial_contributions = 10 → 
  increase_percentage = 75 / 100 → 
  new_average = 100 → 
  (11 : ℚ) * new_average - initial_contributions * (new_average / (1 + increase_percentage)) = 3700 / 7 := by
  sorry

#eval (3700 : ℚ) / 7

end johns_donation_l2992_299219


namespace product_difference_square_l2992_299289

theorem product_difference_square (n : ℤ) : (n - 1) * (n + 1) - n^2 = -1 :=
by sorry

end product_difference_square_l2992_299289


namespace car_trip_duration_l2992_299263

theorem car_trip_duration (initial_speed initial_time remaining_speed average_speed : ℝ) 
  (h1 : initial_speed = 70)
  (h2 : initial_time = 4)
  (h3 : remaining_speed = 60)
  (h4 : average_speed = 65) :
  ∃ (total_time : ℝ), 
    (initial_speed * initial_time + remaining_speed * (total_time - initial_time)) / total_time = average_speed ∧ 
    total_time = 8 := by
sorry

end car_trip_duration_l2992_299263


namespace geometric_sequence_ratio_l2992_299268

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  prop1 : a 5 * a 8 = 6
  prop2 : a 3 + a 10 = 5

/-- The ratio of a_20 to a_13 in the geometric sequence is either 3/2 or 2/3 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 20 / seq.a 13 = 3/2 ∨ seq.a 20 / seq.a 13 = 2/3 := by
  sorry

end geometric_sequence_ratio_l2992_299268


namespace land_of_computation_base_l2992_299200

/-- Represents a number in base s --/
def BaseS (coeffs : List Nat) (s : Nat) : Nat :=
  coeffs.enum.foldl (fun acc (i, a) => acc + a * s^i) 0

/-- The problem statement --/
theorem land_of_computation_base (s : Nat) : 
  s > 1 → 
  BaseS [0, 5, 5] s + BaseS [0, 2, 4] s = BaseS [0, 0, 1, 1] s → 
  s = 7 := by
sorry

end land_of_computation_base_l2992_299200


namespace complex_number_location_l2992_299276

/-- Given a complex number z satisfying (1-i)z = (1+i)^2, 
    prove that z has a negative real part and a positive imaginary part. -/
theorem complex_number_location (z : ℂ) (h : (1 - I) * z = (1 + I)^2) : 
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_location_l2992_299276


namespace molecular_weight_calculation_l2992_299278

/-- The molecular weight of AlCl3 in g/mol -/
def molecular_weight_AlCl3 : ℝ := 132

/-- The number of moles given in the problem -/
def given_moles : ℝ := 4

/-- The total weight of the given moles in grams -/
def total_weight : ℝ := 528

theorem molecular_weight_calculation :
  molecular_weight_AlCl3 * given_moles = total_weight :=
by sorry

end molecular_weight_calculation_l2992_299278


namespace system_solution_l2992_299290

theorem system_solution (x y z : ℝ) : 
  x + y + z = 9 ∧ 
  1/x + 1/y + 1/z = 1 ∧ 
  x*y + x*z + y*z = 27 → 
  x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end system_solution_l2992_299290


namespace jakes_earnings_theorem_l2992_299234

/-- Calculates Jake's weekly earnings based on Jacob's hourly rates and Jake's work schedule. -/
def jakes_weekly_earnings (jacobs_weekday_rate : ℕ) (jacobs_weekend_rate : ℕ) 
  (jakes_weekday_hours : ℕ) (jakes_weekend_hours : ℕ) : ℕ :=
  let jakes_weekday_rate := 3 * jacobs_weekday_rate
  let jakes_weekend_rate := 3 * jacobs_weekend_rate
  let weekday_earnings := jakes_weekday_rate * jakes_weekday_hours * 5
  let weekend_earnings := jakes_weekend_rate * jakes_weekend_hours * 2
  weekday_earnings + weekend_earnings

/-- Theorem stating that Jake's weekly earnings are $960. -/
theorem jakes_earnings_theorem : 
  jakes_weekly_earnings 6 8 8 5 = 960 := by
  sorry

end jakes_earnings_theorem_l2992_299234


namespace sum_of_powers_equals_zero_l2992_299245

theorem sum_of_powers_equals_zero :
  -1^2010 + (-1)^2011 + 1^2012 - 1^2013 + (-1)^2014 = 0 := by
  sorry

end sum_of_powers_equals_zero_l2992_299245


namespace beryllium_hydroxide_formation_l2992_299233

/-- Represents a chemical species in a reaction -/
structure ChemicalSpecies where
  formula : String
  moles : ℚ

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

/-- The balanced chemical equation for the reaction of beryllium carbide with water -/
def berylliumCarbideReaction : ChemicalReaction :=
  { reactants := [
      { formula := "Be2C", moles := 1 },
      { formula := "H2O", moles := 4 }
    ],
    products := [
      { formula := "Be(OH)2", moles := 2 },
      { formula := "CH4", moles := 1 }
    ]
  }

/-- Given 1 mole of Be2C and 4 moles of H2O, 2 moles of Be(OH)2 are formed -/
theorem beryllium_hydroxide_formation :
  ∀ (reaction : ChemicalReaction),
    reaction = berylliumCarbideReaction →
    ∃ (product : ChemicalSpecies),
      product ∈ reaction.products ∧
      product.formula = "Be(OH)2" ∧
      product.moles = 2 :=
by sorry

end beryllium_hydroxide_formation_l2992_299233


namespace science_club_committee_selection_l2992_299207

theorem science_club_committee_selection (total_candidates : Nat) 
  (previously_served : Nat) (committee_size : Nat) 
  (h1 : total_candidates = 20) (h2 : previously_served = 8) 
  (h3 : committee_size = 4) :
  Nat.choose total_candidates committee_size - 
  Nat.choose (total_candidates - previously_served) committee_size = 4350 :=
by
  sorry

end science_club_committee_selection_l2992_299207


namespace salt_price_reduction_l2992_299213

/-- Given a 20% reduction in the price of salt allows 10 kgs more to be purchased for Rs. 400,
    prove that the original price per kg of salt was Rs. 10. -/
theorem salt_price_reduction (P : ℝ) 
  (h1 : P > 0) -- The price is positive
  (h2 : ∃ (X : ℝ), 400 / P = X ∧ 400 / (0.8 * P) = X + 10) -- Condition from the problem
  : P = 10 := by
  sorry

end salt_price_reduction_l2992_299213


namespace chipped_marbles_bag_l2992_299229

def marbleBags : List Nat := [15, 20, 22, 31, 33, 37, 40]

def isValidDistribution (jane : List Nat) (george : List Nat) : Prop :=
  jane.length = 4 ∧ 
  george.length = 2 ∧ 
  (jane.sum : ℚ) = 1.5 * george.sum ∧ 
  (jane.sum + george.sum) % 5 = 0

theorem chipped_marbles_bag (h : ∃ (jane george : List Nat),
  (∀ x ∈ jane ++ george, x ∈ marbleBags) ∧
  isValidDistribution jane george) :
  33 ∈ marbleBags \ (jane ++ george) :=
sorry

end chipped_marbles_bag_l2992_299229


namespace average_of_solutions_is_zero_l2992_299296

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 28}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ x ∈ solutions, x = x₁ ∨ x = x₂ :=
by sorry

end average_of_solutions_is_zero_l2992_299296


namespace zero_in_A_l2992_299266

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : (0 : ℝ) ∈ A := by sorry

end zero_in_A_l2992_299266


namespace intersection_point_equivalence_l2992_299208

theorem intersection_point_equivalence 
  (m n a b : ℝ) 
  (h1 : m * a + 2 * m * b = 5) 
  (h2 : n * a - 2 * n * b = 7) :
  (5 / (2 * m) - a / 2 = b) ∧ (a / 2 - 7 / (2 * n) = b) := by
  sorry

end intersection_point_equivalence_l2992_299208


namespace expansion_simplification_l2992_299222

theorem expansion_simplification (y : ℝ) : (2*y - 3)*(2*y + 3) - (4*y - 1)*(y + 5) = -19*y - 4 := by
  sorry

end expansion_simplification_l2992_299222


namespace system_solution_exists_l2992_299273

theorem system_solution_exists (x y z : ℝ) : 
  (x * y = 8 - 3 * x - 2 * y) →
  (y * z = 8 - 2 * y - 3 * z) →
  (x * z = 35 - 5 * x - 3 * z) →
  ∃ (x : ℝ), x = 8 := by
sorry

end system_solution_exists_l2992_299273


namespace movie_ticket_cost_l2992_299299

theorem movie_ticket_cost (ticket_price : ℕ) (num_students : ℕ) (budget : ℕ) : 
  ticket_price = 29 → num_students = 498 → budget = 1500 → 
  ticket_price * num_students > budget :=
by
  sorry

end movie_ticket_cost_l2992_299299


namespace central_angle_from_arc_length_l2992_299285

/-- Given a circle with radius 12 mm and an arc length of 144 mm, 
    the central angle in radians is equal to 12. -/
theorem central_angle_from_arc_length (R L θ : ℝ) : 
  R = 12 → L = 144 → L = R * θ → θ = 12 := by
  sorry

end central_angle_from_arc_length_l2992_299285


namespace first_stop_passengers_l2992_299242

/-- The number of passengers who got on at the first stop of a bus route -/
def passengers_first_stop : ℕ :=
  sorry

/-- The net change in passengers at the second stop -/
def net_change_second_stop : ℤ := 2

/-- The net change in passengers at the third stop -/
def net_change_third_stop : ℤ := 2

/-- The total number of passengers after the third stop -/
def total_passengers : ℕ := 11

theorem first_stop_passengers :
  passengers_first_stop = 7 :=
sorry

end first_stop_passengers_l2992_299242


namespace sector_arc_length_l2992_299256

/-- Given a circular sector with area 10 cm² and central angle 2 radians,
    the arc length of the sector is 2√10 cm. -/
theorem sector_arc_length (S : ℝ) (α : ℝ) (l : ℝ) :
  S = 10 →  -- Area of the sector
  α = 2 →   -- Central angle in radians
  l = 2 * Real.sqrt 10 -- Arc length
  := by sorry

end sector_arc_length_l2992_299256


namespace alto_saxophone_ratio_l2992_299201

/-- The ratio of alto saxophone players to total saxophone players in a high school band -/
theorem alto_saxophone_ratio (total_students : ℕ) 
  (h1 : total_students = 600)
  (marching_band : ℕ) 
  (h2 : marching_band = total_students / 5)
  (brass_players : ℕ) 
  (h3 : brass_players = marching_band / 2)
  (saxophone_players : ℕ) 
  (h4 : saxophone_players = brass_players / 5)
  (alto_saxophone_players : ℕ) 
  (h5 : alto_saxophone_players = 4) :
  (alto_saxophone_players : ℚ) / saxophone_players = 1 / 3 := by
sorry


end alto_saxophone_ratio_l2992_299201


namespace inequality_condition_l2992_299272

theorem inequality_condition (a x : ℝ) : x^3 + 13*a^2*x > 5*a*x^2 + 9*a^3 ↔ x > a := by
  sorry

end inequality_condition_l2992_299272


namespace tan_alpha_value_l2992_299216

theorem tan_alpha_value (α : Real) 
  (h : (Real.cos (π/4 - α)) / (Real.cos (π/4 + α)) = 1/2) : 
  Real.tan α = -1/3 := by
  sorry

end tan_alpha_value_l2992_299216


namespace tangency_condition_min_area_triangle_l2992_299235

/-- The curve C: x^2 + y^2 - 2x - 2y + 1 = 0 -/
def curve (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line l: bx + ay = ab -/
def line (a b x y : ℝ) : Prop :=
  b*x + a*y = a*b

/-- The line l is tangent to the curve C -/
def is_tangent (a b : ℝ) : Prop :=
  ∃ x y, curve x y ∧ line a b x y

theorem tangency_condition (a b : ℝ) (ha : a > 2) (hb : b > 2) (h_tangent : is_tangent a b) :
  (a - 2) * (b - 2) = 2 :=
sorry

theorem min_area_triangle (a b : ℝ) (ha : a > 2) (hb : b > 2) (h_tangent : is_tangent a b) :
  ∃ area : ℝ, area = 3 + 2 * Real.sqrt 2 ∧ 
  ∀ a' b', a' > 2 → b' > 2 → is_tangent a' b' → (1/2 * a' * b' ≥ area) :=
sorry

end tangency_condition_min_area_triangle_l2992_299235


namespace length_of_A_l2992_299271

def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (3, 6)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def intersect (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (1 - t) • p + t • q = r

theorem length_of_A'B' :
  ∃ A' B' : ℝ × ℝ,
    on_line_y_eq_x A' ∧
    on_line_y_eq_x B' ∧
    intersect A A' C ∧
    intersect B B' C ∧
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = (12 / 7) * Real.sqrt 2 :=
sorry

end length_of_A_l2992_299271


namespace folding_punching_theorem_l2992_299270

/-- Represents a rectangular piece of paper --/
structure Paper where
  width : ℝ
  height : ℝ
  (width_pos : width > 0)
  (height_pos : height > 0)

/-- Represents a fold operation on the paper --/
inductive Fold
  | BottomToTop
  | RightToHalfLeft
  | DiagonalBottomLeftToTopRight

/-- Represents a hole punched in the paper --/
structure Hole where
  x : ℝ
  y : ℝ

/-- Applies a sequence of folds to a paper --/
def applyFolds (p : Paper) (folds : List Fold) : Paper :=
  sorry

/-- Punches a hole in the folded paper --/
def punchHole (p : Paper) : Hole :=
  sorry

/-- Unfolds the paper and calculates the resulting hole pattern --/
def unfoldAndGetHoles (p : Paper) (folds : List Fold) (h : Hole) : List Hole :=
  sorry

/-- Checks if a list of holes is symmetric around the center and along two diagonals --/
def isSymmetricPattern (holes : List Hole) : Prop :=
  sorry

/-- The main theorem stating that the folding and punching process results in 8 symmetric holes --/
theorem folding_punching_theorem (p : Paper) :
  let folds := [Fold.BottomToTop, Fold.RightToHalfLeft, Fold.DiagonalBottomLeftToTopRight]
  let foldedPaper := applyFolds p folds
  let hole := punchHole foldedPaper
  let holePattern := unfoldAndGetHoles p folds hole
  (holePattern.length = 8) ∧ isSymmetricPattern holePattern :=
by
  sorry

end folding_punching_theorem_l2992_299270


namespace expected_throws_in_leap_year_l2992_299237

/-- The expected number of throws for a single day -/
def expected_throws_per_day : ℚ := 8/7

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The expected number of throws in a leap year -/
def expected_throws_leap_year : ℚ := expected_throws_per_day * leap_year_days

theorem expected_throws_in_leap_year :
  expected_throws_leap_year = 2928/7 := by sorry

end expected_throws_in_leap_year_l2992_299237


namespace yellow_shirt_pairs_l2992_299218

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 60 →
  yellow_students = 84 →
  total_students = 144 →
  total_pairs = 72 →
  blue_blue_pairs = 25 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 37 :=
by sorry

end yellow_shirt_pairs_l2992_299218


namespace age_of_a_l2992_299274

/-- Given the ages of four people a, b, c, and d, prove that the age of a is 11 years. -/
theorem age_of_a (A B C D : ℕ) : 
  A + B + C + D = 76 →
  ∃ (k : ℕ), A - 3 = k ∧ B - 3 = 2 * k ∧ C - 3 = 3 * k →
  ∃ (m : ℕ), A - 5 = 3 * m ∧ D - 5 = 4 * m ∧ B - 5 = 5 * m →
  A = 11 := by
  sorry

end age_of_a_l2992_299274


namespace oil_mixture_volume_constant_oil_problem_solution_l2992_299211

/-- Represents the properties of an oil mixture -/
structure OilMixture where
  V_hot : ℝ  -- Volume of hot oil
  V_cold : ℝ  -- Volume of cold oil
  T_hot : ℝ  -- Temperature of hot oil
  T_cold : ℝ  -- Temperature of cold oil
  beta : ℝ  -- Coefficient of thermal expansion

/-- Calculates the final volume of an oil mixture at thermal equilibrium -/
def final_volume (mix : OilMixture) : ℝ :=
  mix.V_hot + mix.V_cold

/-- Theorem stating that the final volume of the oil mixture at thermal equilibrium
    is equal to the sum of the initial volumes -/
theorem oil_mixture_volume_constant (mix : OilMixture) :
  final_volume mix = mix.V_hot + mix.V_cold :=
by sorry

/-- Specific instance of the oil mixture problem -/
def oil_problem : OilMixture :=
  { V_hot := 2
  , V_cold := 1
  , T_hot := 100
  , T_cold := 20
  , beta := 2e-3
  }

/-- The final volume of the specific oil mixture problem is 3 liters -/
theorem oil_problem_solution :
  final_volume oil_problem = 3 :=
by sorry

end oil_mixture_volume_constant_oil_problem_solution_l2992_299211


namespace line_ellipse_intersection_slopes_l2992_299284

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line equation
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = m * x + 4

-- Theorem statement
theorem line_ellipse_intersection_slopes :
  ∀ m : ℝ, (∃ x y : ℝ, ellipse_eq x y ∧ line_eq m x y) →
  m^2 ≥ 0.48 :=
sorry

end line_ellipse_intersection_slopes_l2992_299284


namespace smallest_number_in_sequence_l2992_299236

theorem smallest_number_in_sequence (x : ℝ) : 
  let second := 4 * x
  let third := 2 * second
  (x + second + third) / 3 = 78 →
  x = 18 :=
by
  sorry

end smallest_number_in_sequence_l2992_299236


namespace farm_animals_l2992_299257

theorem farm_animals (total_legs : ℕ) (chicken_count : ℕ) (chicken_legs : ℕ) (buffalo_legs : ℕ) :
  total_legs = 44 →
  chicken_count = 4 →
  chicken_legs = 2 →
  buffalo_legs = 4 →
  ∃ (buffalo_count : ℕ),
    total_legs = chicken_count * chicken_legs + buffalo_count * buffalo_legs ∧
    chicken_count + buffalo_count = 13 :=
by
  sorry

end farm_animals_l2992_299257


namespace not_always_same_digit_sum_l2992_299279

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- State the theorem
theorem not_always_same_digit_sum :
  ∃ (N M : ℕ), 
    (sumOfDigits (N + M) = sumOfDigits N) ∧ 
    (∀ k : ℕ, k > 1 → sumOfDigits (N + k * M) ≠ sumOfDigits N) :=
sorry

end not_always_same_digit_sum_l2992_299279


namespace yogurt_refund_calculation_l2992_299297

theorem yogurt_refund_calculation (total_packs : ℕ) (expired_percentage : ℚ) (price_per_pack : ℚ) : 
  total_packs = 80 →
  expired_percentage = 40 / 100 →
  price_per_pack = 12 →
  (total_packs : ℚ) * expired_percentage * price_per_pack = 384 := by
sorry

end yogurt_refund_calculation_l2992_299297


namespace average_age_combined_l2992_299283

theorem average_age_combined (num_students : ℕ) (avg_age_students : ℝ)
                              (num_teachers : ℕ) (avg_age_teachers : ℝ)
                              (num_parents : ℕ) (avg_age_parents : ℝ) :
  num_students = 40 →
  avg_age_students = 10 →
  num_teachers = 4 →
  avg_age_teachers = 40 →
  num_parents = 60 →
  avg_age_parents = 34 →
  (num_students * avg_age_students + num_teachers * avg_age_teachers + num_parents * avg_age_parents) /
  (num_students + num_teachers + num_parents : ℝ) = 25 := by
  sorry

end average_age_combined_l2992_299283


namespace total_strings_needed_johns_total_strings_l2992_299210

theorem total_strings_needed (num_basses : ℕ) (strings_per_bass : ℕ) 
  (strings_per_guitar : ℕ) (strings_per_8string_guitar : ℕ) : ℕ :=
  let num_guitars := 2 * num_basses
  let num_8string_guitars := num_guitars - 3
  let bass_strings := num_basses * strings_per_bass
  let guitar_strings := num_guitars * strings_per_guitar
  let eight_string_guitar_strings := num_8string_guitars * strings_per_8string_guitar
  bass_strings + guitar_strings + eight_string_guitar_strings

theorem johns_total_strings : 
  total_strings_needed 3 4 6 8 = 72 := by
  sorry

end total_strings_needed_johns_total_strings_l2992_299210


namespace coconut_grove_problem_l2992_299280

/-- Coconut grove problem -/
theorem coconut_grove_problem (x : ℝ) 
  (h1 : 60 * (x + 1) + 120 * x + 180 * (x - 1) = 100 * (3 * x)) : 
  x = 2 := by
  sorry

end coconut_grove_problem_l2992_299280


namespace simplify_and_evaluate_l2992_299206

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 2) : 
  (a^2 - 4*a + 4) / a / (a - 4/a) = 1 - 2 * Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l2992_299206


namespace horner_method_v2_l2992_299202

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 - x + 5

def horner_v2 (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₅ * x + a₄) * x + a₃) * x + a₂

theorem horner_method_v2 :
  horner_v2 2 0 (-3) 2 (-1) 5 2 = 5 := by
  sorry

end horner_method_v2_l2992_299202


namespace amanda_grass_seed_bags_l2992_299264

/-- The number of bags of grass seed needed for a specific lot -/
def grassSeedBags (lotLength lotWidth concreteLength concreteWidth bagCoverage : ℕ) : ℕ :=
  let totalArea := lotLength * lotWidth
  let concreteArea := concreteLength * concreteWidth
  let grassArea := totalArea - concreteArea
  (grassArea + bagCoverage - 1) / bagCoverage

theorem amanda_grass_seed_bags :
  grassSeedBags 120 60 40 40 56 = 100 := by
  sorry

end amanda_grass_seed_bags_l2992_299264


namespace a_squared_gt_one_neither_sufficient_nor_necessary_for_one_over_a_gt_zero_l2992_299203

theorem a_squared_gt_one_neither_sufficient_nor_necessary_for_one_over_a_gt_zero :
  ¬(∀ a : ℝ, a^2 > 1 → 1/a > 0) ∧ ¬(∀ a : ℝ, 1/a > 0 → a^2 > 1) := by
  sorry

end a_squared_gt_one_neither_sufficient_nor_necessary_for_one_over_a_gt_zero_l2992_299203


namespace square_root_fraction_equality_l2992_299286

theorem square_root_fraction_equality :
  Real.sqrt (8^2 + 15^2) / Real.sqrt (49 + 36) = 17 * Real.sqrt 85 / 85 := by
  sorry

end square_root_fraction_equality_l2992_299286


namespace sunflower_contest_total_l2992_299230

/-- Represents the total number of seeds eaten in a sunflower eating contest -/
def total_seeds_eaten (player1 player2 player3 : ℕ) : ℕ :=
  player1 + player2 + player3

/-- Theorem stating the total number of seeds eaten in the contest -/
theorem sunflower_contest_total :
  let player1 := 78
  let player2 := 53
  let player3 := player2 + 30
  total_seeds_eaten player1 player2 player3 = 214 := by
  sorry

end sunflower_contest_total_l2992_299230


namespace x1_x2_ratio_lt_ae_l2992_299209

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / a - Real.exp x

theorem x1_x2_ratio_lt_ae (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) 
  (hf₂ : f a x₂ = 0) : 
  x₁ / x₂ < a * Real.exp 1 := by
  sorry

end x1_x2_ratio_lt_ae_l2992_299209


namespace cube_sum_reciprocal_l2992_299228

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 4) :
  a^3 + 1/a^3 = 2 ∨ a^3 + 1/a^3 = -2 := by
  sorry

end cube_sum_reciprocal_l2992_299228


namespace integer_solutions_quadratic_equation_l2992_299239

theorem integer_solutions_quadratic_equation :
  {(x, y) : ℤ × ℤ | x^2 + y^2 = x + y + 2} =
  {(2, 1), (2, 0), (-1, 1), (-1, 0)} := by sorry

end integer_solutions_quadratic_equation_l2992_299239


namespace parallelogram_height_l2992_299215

theorem parallelogram_height (area base height : ℝ) : 
  area = 612 ∧ base = 34 ∧ area = base * height → height = 18 := by
  sorry

end parallelogram_height_l2992_299215


namespace perpendicular_lines_a_values_l2992_299223

theorem perpendicular_lines_a_values (a : ℝ) : 
  ((3*a + 2) * (5*a - 2) + (1 - 4*a) * (a + 4) = 0) → (a = 0 ∨ a = 1) := by
  sorry

end perpendicular_lines_a_values_l2992_299223


namespace combination_lock_code_l2992_299225

theorem combination_lock_code (x y : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ x ≠ 0 → 
  (x + y + x * y = 10 * x + y) ↔ 
  (y = 9 ∧ x ∈ Finset.range 10 \ {0}) :=
sorry

end combination_lock_code_l2992_299225


namespace invisible_dots_count_l2992_299238

/-- The sum of numbers on a standard six-sided die -/
def standard_die_sum : Nat := 21

/-- The number of dice rolled -/
def num_dice : Nat := 4

/-- The sum of visible numbers on the dice -/
def visible_sum : Nat := 6 + 6 + 4 + 4 + 3 + 2 + 1

/-- The total number of dots on all dice -/
def total_dots : Nat := num_dice * standard_die_sum

theorem invisible_dots_count : total_dots - visible_sum = 58 := by
  sorry

end invisible_dots_count_l2992_299238


namespace romanian_sequence_swaps_l2992_299288

/-- A Romanian sequence is a sequence of 3n letters where I, M, and O each occur exactly n times. -/
def RomanianSequence (n : ℕ) := Vector (Fin 3) (3 * n)

/-- The number of swaps required to transform one sequence into another. -/
def swapsRequired (n : ℕ) (X Y : RomanianSequence n) : ℕ := sorry

/-- There exists a Romanian sequence Y for any Romanian sequence X such that
    at least 3n^2/2 swaps are required to transform X into Y. -/
theorem romanian_sequence_swaps (n : ℕ) :
  ∀ X : RomanianSequence n, ∃ Y : RomanianSequence n,
    swapsRequired n X Y ≥ (3 * n^2) / 2 := by sorry

end romanian_sequence_swaps_l2992_299288


namespace sticks_at_20th_stage_l2992_299247

/-- The number of sticks in the nth stage of the pattern -/
def sticks : ℕ → ℕ
| 0 => 5  -- Initial stage (indexed as 0)
| n + 1 => if n < 10 then sticks n + 3 else sticks n + 4

/-- The theorem stating that the 20th stage (indexed as 19) has 68 sticks -/
theorem sticks_at_20th_stage : sticks 19 = 68 := by
  sorry

end sticks_at_20th_stage_l2992_299247


namespace clock_gains_seven_minutes_per_hour_l2992_299248

/-- A clock that gains time -/
structure GainingClock where
  start_time : Nat  -- Start time in hours (24-hour format)
  end_time : Nat    -- End time in hours (24-hour format)
  total_gain : Nat  -- Total minutes gained

/-- Calculate the minutes gained per hour -/
def minutes_gained_per_hour (clock : GainingClock) : Rat :=
  clock.total_gain / (clock.end_time - clock.start_time)

/-- Theorem: A clock starting at 9 AM, ending at 6 PM, and gaining 63 minutes
    will gain 7 minutes per hour -/
theorem clock_gains_seven_minutes_per_hour 
  (clock : GainingClock) 
  (h1 : clock.start_time = 9)
  (h2 : clock.end_time = 18)
  (h3 : clock.total_gain = 63) :
  minutes_gained_per_hour clock = 7 := by
  sorry

end clock_gains_seven_minutes_per_hour_l2992_299248


namespace geometric_progression_squared_sum_l2992_299214

theorem geometric_progression_squared_sum 
  (q : ℝ) 
  (S : ℝ) 
  (h1 : abs q < 1) 
  (h2 : S = 1 / (1 - q)) : 
  1 / (1 - q^2) = S^2 / (2*S - 1) := by
  sorry

end geometric_progression_squared_sum_l2992_299214


namespace tangent_circle_equation_l2992_299204

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the point (4, -1) on circle C
def point_on_C : Prop := circle_C 4 (-1)

-- Define the new circle with center (a, b) and radius 1
def new_circle (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1

-- Define the tangency condition
def is_tangent (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), circle_C x y ∧ new_circle a b x y

-- The theorem to prove
theorem tangent_circle_equation :
  point_on_C →
  is_tangent 5 (-1) ∨ is_tangent 3 (-1) :=
sorry

end tangent_circle_equation_l2992_299204


namespace opposite_of_negative_two_l2992_299243

theorem opposite_of_negative_two : 
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
sorry

end opposite_of_negative_two_l2992_299243


namespace triangle_side_difference_l2992_299240

theorem triangle_side_difference (a b c : ℝ) : 
  b = 8 → c = 3 → 
  a > 0 → b > 0 → c > 0 →
  a + b > c → a + c > b → b + c > a →
  (∃ (a_min a_max : ℕ), 
    (∀ x : ℕ, (x : ℝ) = a → a_min ≤ x ∧ x ≤ a_max) ∧
    (∀ y : ℕ, y < a_min → (y : ℝ) ≠ a) ∧
    (∀ z : ℕ, z > a_max → (z : ℝ) ≠ a) ∧
    a_max - a_min = 4) :=
by sorry

end triangle_side_difference_l2992_299240


namespace product_remainder_theorem_l2992_299292

def numbers : List Nat := [445876, 985420, 215546, 656452, 387295]

def remainder_sum_squares (nums : List Nat) : Nat :=
  (nums.map (λ n => (n^2) % 8)).sum

theorem product_remainder_theorem :
  (remainder_sum_squares numbers) % 9 = 5 := by
  sorry

end product_remainder_theorem_l2992_299292


namespace park_area_l2992_299262

/-- Proves that a rectangular park with given conditions has an area of 102400 square meters -/
theorem park_area (length breadth : ℝ) (speed : ℝ) (time : ℝ) : 
  length / breadth = 4 →
  speed = 12 →
  time = 8 / 60 →
  2 * (length + breadth) = speed * time * 1000 →
  length * breadth = 102400 :=
by sorry

end park_area_l2992_299262


namespace moving_circle_trajectory_l2992_299253

-- Define the circles O₁ and O₂
def O₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def O₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the trajectory of the center M
def trajectory (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- State the theorem
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x₁ y₁ : ℝ), O₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (1 + r)^2) ∧
    (∀ (x₂ y₂ : ℝ), O₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (9 - r)^2)) →
  trajectory x y :=
by sorry

end moving_circle_trajectory_l2992_299253


namespace rectangle_circumscribed_l2992_299291

/-- Two lines form a rectangle with the coordinate axes that can be circumscribed by a circle -/
theorem rectangle_circumscribed (k : ℝ) : 
  (∃ (x y : ℝ), x + 3*y - 7 = 0 ∧ k*x - y - 2 = 0) →
  (∀ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 → (x + 3*y - 7 = 0 ∨ k*x - y - 2 = 0 ∨ x = 0 ∨ y = 0)) →
  (k = 3) := by
sorry

end rectangle_circumscribed_l2992_299291


namespace other_denomination_is_50_l2992_299281

/-- Proves that the denomination of the other currency notes is 50 given the problem conditions --/
theorem other_denomination_is_50 
  (total_notes : ℕ) 
  (total_amount : ℕ) 
  (amount_other_denom : ℕ) 
  (h_total_notes : total_notes = 85)
  (h_total_amount : total_amount = 5000)
  (h_amount_other_denom : amount_other_denom = 3500) :
  ∃ (x y D : ℕ), 
    x + y = total_notes ∧ 
    100 * x + D * y = total_amount ∧
    D * y = amount_other_denom ∧
    D = 50 := by
  sorry

#check other_denomination_is_50

end other_denomination_is_50_l2992_299281


namespace ellipse_equation_l2992_299294

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (2 * a * b = 4) →
  (a^2 - b^2 = 3) →
  (a = 2 ∧ b = 1) := by sorry

end ellipse_equation_l2992_299294


namespace billy_has_24_balloons_l2992_299231

/-- The number of water balloons Billy is left with after the water balloon fight -/
def billys_balloons (total_packs : ℕ) (balloons_per_pack : ℕ) (num_people : ℕ) 
  (extra_milly : ℕ) (extra_tamara : ℕ) (extra_floretta : ℕ) : ℕ :=
  (total_packs * balloons_per_pack) / num_people

/-- Theorem stating that Billy is left with 24 water balloons -/
theorem billy_has_24_balloons : 
  billys_balloons 12 8 4 11 9 4 = 24 := by
  sorry

#eval billys_balloons 12 8 4 11 9 4

end billy_has_24_balloons_l2992_299231


namespace factorial_sum_equality_l2992_299251

theorem factorial_sum_equality : 7 * Nat.factorial 6 + 6 * Nat.factorial 5 + 2 * Nat.factorial 5 = 6000 := by
  sorry

end factorial_sum_equality_l2992_299251


namespace intercepts_correct_l2992_299275

/-- The line equation is 5x - 2y - 10 = 0 -/
def line_equation (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

/-- Proof that the x-intercept and y-intercept are correct for the given line equation -/
theorem intercepts_correct : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept := by sorry

end intercepts_correct_l2992_299275


namespace cos_negative_330_degrees_l2992_299232

theorem cos_negative_330_degrees : Real.cos (-(330 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end cos_negative_330_degrees_l2992_299232


namespace coach_votes_l2992_299261

theorem coach_votes (num_coaches : ℕ) (num_voters : ℕ) (votes_per_voter : ℕ) 
  (h1 : num_coaches = 36)
  (h2 : num_voters = 60)
  (h3 : votes_per_voter = 3)
  (h4 : num_voters * votes_per_voter % num_coaches = 0) :
  (num_voters * votes_per_voter) / num_coaches = 5 := by
sorry

end coach_votes_l2992_299261


namespace absolute_value_and_exponents_l2992_299295

theorem absolute_value_and_exponents : 
  |(-3 : ℝ)| + (Real.pi + 1)^(0 : ℝ) - (1/3 : ℝ)^(-1 : ℝ) = 1 := by
  sorry

end absolute_value_and_exponents_l2992_299295


namespace sphere_volume_rectangular_solid_l2992_299267

theorem sphere_volume_rectangular_solid (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b = 1 →
  b * c = 2 →
  a * c = 2 →
  (4 / 3) * Real.pi * ((a^2 + b^2 + c^2).sqrt / 2)^3 = Real.pi * Real.sqrt 6 := by
  sorry

end sphere_volume_rectangular_solid_l2992_299267


namespace price_quantity_difference_l2992_299252

/-- Given a price increase and quantity reduction, proves the difference in cost -/
theorem price_quantity_difference (P Q : ℝ) (h_pos_P : P > 0) (h_pos_Q : Q > 0) : 
  (P * 1.1 * (Q * 0.8)) - (P * Q) = -0.12 * (P * Q) := by
  sorry

#check price_quantity_difference

end price_quantity_difference_l2992_299252


namespace perpendicular_trapezoid_midline_l2992_299260

/-- A trapezoid with perpendicular diagonals -/
structure PerpendicularTrapezoid where
  /-- One diagonal of the trapezoid -/
  diagonal1 : ℝ
  /-- The angle between the other diagonal and the base -/
  angle : ℝ
  /-- The diagonals are perpendicular -/
  perpendicular : True
  /-- One diagonal is 6 units long -/
  diagonal1_length : diagonal1 = 6
  /-- The other diagonal forms a 30° angle with the base -/
  angle_is_30 : angle = 30 * π / 180

/-- The midline of a trapezoid with perpendicular diagonals -/
def midline (t : PerpendicularTrapezoid) : ℝ := sorry

/-- Theorem: The midline of a trapezoid with perpendicular diagonals,
    where one diagonal is 6 units long and the other forms a 30° angle with the base,
    is 6 units long -/
theorem perpendicular_trapezoid_midline (t : PerpendicularTrapezoid) :
  midline t = 6 := by sorry

end perpendicular_trapezoid_midline_l2992_299260


namespace direct_product_is_group_l2992_299259

/-- Given two groups G and H, their direct product is also a group. -/
theorem direct_product_is_group {G H : Type*} [Group G] [Group H] :
  Group (G × H) :=
by sorry

end direct_product_is_group_l2992_299259


namespace jules_blocks_to_walk_l2992_299298

-- Define the given constants
def vacation_cost : ℚ := 1000
def family_members : ℕ := 5
def start_fee : ℚ := 2
def per_block_fee : ℚ := 1.25
def num_dogs : ℕ := 20

-- Define Jules' contribution
def jules_contribution : ℚ := vacation_cost / family_members

-- Define the function to calculate earnings based on number of blocks
def earnings (blocks : ℕ) : ℚ := num_dogs * (start_fee + per_block_fee * blocks)

-- Theorem statement
theorem jules_blocks_to_walk :
  ∃ (blocks : ℕ), earnings blocks ≥ jules_contribution ∧
    ∀ (b : ℕ), b < blocks → earnings b < jules_contribution :=
by sorry

end jules_blocks_to_walk_l2992_299298


namespace geometric_sequence_ratio_l2992_299246

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The common ratio of a geometric sequence is the constant factor between successive terms. -/
def CommonRatio (a : ℕ → ℚ) : ℚ :=
  a 1 / a 0

theorem geometric_sequence_ratio :
  ∀ a : ℕ → ℚ,
  IsGeometricSequence a →
  a 0 = 25 →
  a 1 = -50 →
  a 2 = 100 →
  a 3 = -200 →
  CommonRatio a = -2 := by
  sorry

end geometric_sequence_ratio_l2992_299246


namespace least_n_with_gcd_conditions_l2992_299293

theorem least_n_with_gcd_conditions : 
  ∃ (n : ℕ), n > 1000 ∧ 
  Nat.gcd 63 (n + 120) = 21 ∧ 
  Nat.gcd (n + 63) 120 = 60 ∧
  (∀ m : ℕ, m > 1000 ∧ m < n → 
    Nat.gcd 63 (m + 120) ≠ 21 ∨ 
    Nat.gcd (m + 63) 120 ≠ 60) ∧
  n = 1917 :=
by sorry

end least_n_with_gcd_conditions_l2992_299293


namespace constant_zero_unique_solution_l2992_299227

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the derivative of a function
noncomputable def derivative (f : RealFunction) : RealFunction :=
  λ x => deriv f x

-- State the theorem
theorem constant_zero_unique_solution :
  ∃! f : RealFunction, ∀ x : ℝ, f x = derivative f x ∧ f x = 0 :=
sorry

end constant_zero_unique_solution_l2992_299227


namespace anthony_jim_shoe_difference_l2992_299265

-- Define the number of shoe pairs for each person
def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

-- Theorem statement
theorem anthony_jim_shoe_difference :
  anthony_shoes - jim_shoes = 2 := by
  sorry

end anthony_jim_shoe_difference_l2992_299265


namespace quadratic_equation_roots_l2992_299212

theorem quadratic_equation_roots (x y : ℝ) : 
  x + y = 10 →
  |x - y| = 4 →
  x * y = 21 →
  x^2 - 10*x + 21 = 0 ∧ y^2 - 10*y + 21 = 0 :=
by sorry

end quadratic_equation_roots_l2992_299212


namespace sin_product_equals_one_sixteenth_l2992_299205

theorem sin_product_equals_one_sixteenth :
  Real.sin (18 * π / 180) * Real.sin (42 * π / 180) *
  Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 1 / 16 := by
  sorry

end sin_product_equals_one_sixteenth_l2992_299205


namespace even_five_digit_numbers_l2992_299217

def set1 : Finset ℕ := {1, 3, 5}
def set2 : Finset ℕ := {2, 4, 6, 8}

def is_valid_selection (s : Finset ℕ) : Prop :=
  s.card = 5 ∧ (s ∩ set1).card = 2 ∧ (s ∩ set2).card = 3

def is_even (n : ℕ) : Prop := n % 2 = 0

def count_even_numbers : ℕ := sorry

theorem even_five_digit_numbers :
  count_even_numbers = 864 :=
sorry

end even_five_digit_numbers_l2992_299217
