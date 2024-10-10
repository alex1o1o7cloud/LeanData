import Mathlib

namespace arithmetic_geometric_mean_inequality_l3796_379617

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end arithmetic_geometric_mean_inequality_l3796_379617


namespace B_power_48_l3796_379608

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, 1; 0, -1, 0]

theorem B_power_48 : 
  B ^ 48 = !![0, 0, 0; 0, 1, 0; 0, 0, 1] := by sorry

end B_power_48_l3796_379608


namespace distance_of_symmetric_points_on_parabola_l3796_379601

-- Define the parabola
def parabola (x : ℝ) : ℝ := 3 - x^2

-- Define the symmetry line
def symmetryLine (x y : ℝ) : Prop := x + y = 0

-- Define a point on the parabola
def pointOnParabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

-- Define symmetry with respect to the line x + y = 0
def symmetricPoints (p q : ℝ × ℝ) : Prop :=
  q.1 = p.2 ∧ q.2 = p.1

-- The main theorem
theorem distance_of_symmetric_points_on_parabola (A B : ℝ × ℝ) :
  pointOnParabola A →
  pointOnParabola B →
  A ≠ B →
  symmetricPoints A B →
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3 * Real.sqrt 2 :=
sorry

end distance_of_symmetric_points_on_parabola_l3796_379601


namespace farm_dogs_left_l3796_379682

/-- Given a farm with dogs and farmhands, calculates the number of dogs left after a morning walk. -/
def dogs_left_after_walk (total_dogs : ℕ) (dog_houses : ℕ) (farmhands : ℕ) (dogs_per_farmhand : ℕ) : ℕ :=
  total_dogs - farmhands * dogs_per_farmhand

/-- Proves that given the specific conditions of the farm, 144 dogs are left after the morning walk. -/
theorem farm_dogs_left : dogs_left_after_walk 156 22 6 2 = 144 := by
  sorry

#eval dogs_left_after_walk 156 22 6 2

end farm_dogs_left_l3796_379682


namespace shortest_distance_proof_l3796_379651

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 - y - 2 * Real.log (Real.sqrt x) = 0

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 4*y + 1 = 0

-- Define the shortest distance function
noncomputable def shortest_distance : ℝ := (Real.sqrt 2 / 2) * (1 + Real.log 2)

-- Theorem statement
theorem shortest_distance_proof :
  ∀ (x y : ℝ), curve x y →
  ∃ (d : ℝ), d ≥ 0 ∧ 
    (∀ (x' y' : ℝ), line x' y' → 
      d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) ∧
    d = shortest_distance :=
sorry

end shortest_distance_proof_l3796_379651


namespace arithmetic_calculations_l3796_379685

theorem arithmetic_calculations :
  ((-8) - 5 + (-4) - (-10) = -7) ∧
  (18 - 6 / (-2) * (-1/3) = 17) :=
by sorry

end arithmetic_calculations_l3796_379685


namespace complex_number_in_fourth_quadrant_l3796_379652

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - 7*I) / (4 - I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l3796_379652


namespace point_p_coordinates_l3796_379629

/-- A point on the x-axis with distance 3 from the origin -/
structure PointP where
  x : ℝ
  y : ℝ
  on_x_axis : y = 0
  distance_3 : x^2 + y^2 = 3^2

/-- The coordinates of point P are either (-3,0) or (3,0) -/
theorem point_p_coordinates (p : PointP) : (p.x = -3 ∧ p.y = 0) ∨ (p.x = 3 ∧ p.y = 0) := by
  sorry

end point_p_coordinates_l3796_379629


namespace partnership_capital_share_l3796_379658

theorem partnership_capital_share :
  let total_profit : ℚ := 2430
  let a_profit : ℚ := 810
  let a_share : ℚ := 1/3
  let b_share : ℚ := 1/4
  let d_share (c_share : ℚ) : ℚ := 1 - (a_share + b_share + c_share)
  ∀ c_share : ℚ,
    (a_share / 1 = a_profit / total_profit) →
    (a_share + b_share + c_share + d_share c_share = 1) →
    c_share = 5/24 :=
by sorry

end partnership_capital_share_l3796_379658


namespace zongzi_price_proof_l3796_379628

-- Define the unit price of type B zongzi
def unit_price_B : ℝ := 4

-- Define the conditions
def amount_A : ℝ := 1200
def amount_B : ℝ := 800
def quantity_difference : ℕ := 50

-- Theorem statement
theorem zongzi_price_proof :
  -- Conditions
  (amount_A = (2 * unit_price_B) * ((amount_B / unit_price_B) - quantity_difference)) ∧
  (amount_B = unit_price_B * (amount_B / unit_price_B)) →
  -- Conclusion
  unit_price_B = 4 := by
  sorry


end zongzi_price_proof_l3796_379628


namespace factory_employees_count_l3796_379613

/-- Represents the profit calculation for a t-shirt factory --/
def factory_profit (num_employees : ℕ) : ℚ :=
  let shirts_per_employee := 20
  let shirt_price := 35
  let hourly_wage := 12
  let per_shirt_bonus := 5
  let hours_per_shift := 8
  let nonemployee_expenses := 1000
  let total_shirts := num_employees * shirts_per_employee
  let revenue := total_shirts * shirt_price
  let employee_pay := num_employees * (hourly_wage * hours_per_shift + per_shirt_bonus * shirts_per_employee)
  revenue - employee_pay - nonemployee_expenses

/-- The number of employees that results in the given profit --/
theorem factory_employees_count : 
  ∃ (n : ℕ), factory_profit n = 9080 ∧ n = 20 := by
  sorry


end factory_employees_count_l3796_379613


namespace distance_between_ports_l3796_379665

/-- The distance between two ports given ship and current speeds and time difference -/
theorem distance_between_ports (ship_speed : ℝ) (current_speed : ℝ) (time_diff : ℝ) :
  ship_speed > current_speed →
  ship_speed = 24 →
  current_speed = 3 →
  time_diff = 5 →
  ∃ (distance : ℝ),
    distance / (ship_speed - current_speed) - distance / (ship_speed + current_speed) = time_diff ∧
    distance = 200 / 3 := by
  sorry

end distance_between_ports_l3796_379665


namespace cafeteria_sales_comparison_l3796_379619

def arithmetic_growth (initial : ℝ) (increment : ℝ) (periods : ℕ) : ℝ :=
  initial + increment * periods

def geometric_growth (initial : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  initial * (1 + rate) ^ periods

theorem cafeteria_sales_comparison
  (initial : ℝ)
  (increment : ℝ)
  (rate : ℝ)
  (h1 : initial > 0)
  (h2 : increment > 0)
  (h3 : rate > 0)
  (h4 : arithmetic_growth initial increment 8 = geometric_growth initial rate 8) :
  arithmetic_growth initial increment 4 > geometric_growth initial rate 4 :=
by sorry

end cafeteria_sales_comparison_l3796_379619


namespace rod_and_rope_problem_l3796_379627

theorem rod_and_rope_problem (x y : ℝ) : 
  (x - y = 5 ∧ y - (1/2) * x = 5) ↔ 
  (x > y ∧ x - y = 5 ∧ y > (1/2) * x ∧ y - (1/2) * x = 5) :=
sorry

end rod_and_rope_problem_l3796_379627


namespace geometric_sequence_property_l3796_379691

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_sum : a 4 + a 8 = -2) : 
  a 4^2 + 2 * a 6^2 + a 6 * a 10 = 4 := by
sorry

end geometric_sequence_property_l3796_379691


namespace regular_hexagon_most_symmetry_l3796_379639

-- Define the types of polygons
inductive Polygon
  | RegularPentagon
  | IrregularHexagon
  | RegularHexagon
  | IrregularPentagon
  | EquilateralTriangle

-- Function to get the number of lines of symmetry for each polygon
def linesOfSymmetry (p : Polygon) : ℕ :=
  match p with
  | Polygon.RegularPentagon => 5
  | Polygon.IrregularHexagon => 0
  | Polygon.RegularHexagon => 6
  | Polygon.IrregularPentagon => 0
  | Polygon.EquilateralTriangle => 3

-- Theorem stating that the regular hexagon has the most lines of symmetry
theorem regular_hexagon_most_symmetry :
  ∀ p : Polygon, linesOfSymmetry Polygon.RegularHexagon ≥ linesOfSymmetry p :=
by sorry

end regular_hexagon_most_symmetry_l3796_379639


namespace solution_set_equality_l3796_379641

theorem solution_set_equality : 
  {x : ℝ | 1 < |x + 2| ∧ |x + 2| < 5} = 
  {x : ℝ | -7 < x ∧ x < -3} ∪ {x : ℝ | -1 < x ∧ x < 3} := by sorry

end solution_set_equality_l3796_379641


namespace pencils_for_classroom_l3796_379684

/-- Given a classroom with 4 children where each child receives 2 pencils,
    prove that the teacher needs to give out 8 pencils in total. -/
theorem pencils_for_classroom (num_children : ℕ) (pencils_per_child : ℕ) 
  (h1 : num_children = 4) (h2 : pencils_per_child = 2) : 
  num_children * pencils_per_child = 8 := by
  sorry

end pencils_for_classroom_l3796_379684


namespace diminished_number_divisibility_l3796_379631

def smallest_number : ℕ := 1013
def diminished_number : ℕ := smallest_number - 5

def divisors : Set ℕ := {1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 16, 18, 21, 24, 28, 36, 42, 48, 56, 63, 72, 84, 96, 112, 126, 144, 168, 192, 252, 336, 504, 1008}

theorem diminished_number_divisibility :
  (∀ n ∈ divisors, diminished_number % n = 0) ∧
  (∀ m : ℕ, m > 0 → m ∉ divisors → diminished_number % m ≠ 0) :=
sorry

end diminished_number_divisibility_l3796_379631


namespace taco_castle_parking_lot_l3796_379680

/-- The number of Volkswagen Bugs in the parking lot of Taco Castle -/
def volkswagen_bugs (dodge ford toyota : ℕ) : ℕ :=
  toyota / 2

theorem taco_castle_parking_lot (dodge ford toyota : ℕ) 
  (h1 : ford = dodge / 3)
  (h2 : ford = toyota * 2)
  (h3 : dodge = 60) :
  volkswagen_bugs dodge ford toyota = 5 := by
sorry

end taco_castle_parking_lot_l3796_379680


namespace total_degrees_theorem_l3796_379692

/-- Represents the budget allocation percentages for different sectors -/
structure BudgetAllocation where
  microphotonics : Float
  homeElectronics : Float
  foodAdditives : Float
  geneticallyModifiedMicroorganisms : Float
  industrialLubricants : Float
  artificialIntelligence : Float
  nanotechnology : Float

/-- Calculates the degrees in a circle graph for a given percentage -/
def percentageToDegrees (percentage : Float) : Float :=
  percentage * 3.6

/-- Calculates the total degrees for basic astrophysics, artificial intelligence, and nanotechnology -/
def totalDegrees (allocation : BudgetAllocation) : Float :=
  let basicAstrophysics := 100 - (allocation.microphotonics + allocation.homeElectronics + 
    allocation.foodAdditives + allocation.geneticallyModifiedMicroorganisms + 
    allocation.industrialLubricants + allocation.artificialIntelligence + allocation.nanotechnology)
  percentageToDegrees basicAstrophysics + 
  percentageToDegrees allocation.artificialIntelligence + 
  percentageToDegrees allocation.nanotechnology

/-- Theorem: The total degrees for basic astrophysics, artificial intelligence, and nanotechnology is 117.36 -/
theorem total_degrees_theorem (allocation : BudgetAllocation) 
  (h1 : allocation.microphotonics = 12.3)
  (h2 : allocation.homeElectronics = 17.8)
  (h3 : allocation.foodAdditives = 9.4)
  (h4 : allocation.geneticallyModifiedMicroorganisms = 21.7)
  (h5 : allocation.industrialLubricants = 6.2)
  (h6 : allocation.artificialIntelligence = 4.1)
  (h7 : allocation.nanotechnology = 5.3) :
  totalDegrees allocation = 117.36 := by
  sorry

end total_degrees_theorem_l3796_379692


namespace modulo_residue_problem_l3796_379670

theorem modulo_residue_problem :
  (250 * 15 - 337 * 5 + 22) % 13 = 7 := by
  sorry

end modulo_residue_problem_l3796_379670


namespace hands_count_l3796_379672

/-- The number of students in Peter's class, including Peter. -/
def total_students : ℕ := 11

/-- The number of hands each student has. -/
def hands_per_student : ℕ := 2

/-- The number of hands in Peter's class, not including his. -/
def hands_in_class : ℕ := (total_students - 1) * hands_per_student

theorem hands_count : hands_in_class = 20 := by
  sorry

end hands_count_l3796_379672


namespace congruence_problem_l3796_379646

theorem congruence_problem (y : ℤ) 
  (h1 : (2 + y) % (2^4) = 2^3 % (2^4))
  (h2 : (4 + y) % (4^3) = 4^2 % (4^3))
  (h3 : (6 + y) % (6^3) = 6^2 % (6^3)) :
  y % 48 = 44 := by
  sorry

end congruence_problem_l3796_379646


namespace sum_of_absolute_coefficients_l3796_379689

-- Define the polynomial coefficients
variable (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)

-- Define the polynomial equation
def polynomial_equation (x : ℝ) : Prop :=
  (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6

-- State the theorem
theorem sum_of_absolute_coefficients :
  (∀ x, polynomial_equation a₀ a₁ a₂ a₃ a₄ a₅ a₆ x) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
  sorry

end sum_of_absolute_coefficients_l3796_379689


namespace initial_hno3_concentration_l3796_379643

/-- Proves that the initial concentration of HNO3 is 35% given the problem conditions -/
theorem initial_hno3_concentration
  (initial_volume : ℝ)
  (pure_hno3_added : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 60)
  (h2 : pure_hno3_added = 18)
  (h3 : final_concentration = 50)
  : ∃ (initial_concentration : ℝ),
    initial_concentration = 35 ∧
    (initial_concentration / 100) * initial_volume + pure_hno3_added =
    (final_concentration / 100) * (initial_volume + pure_hno3_added) :=
by sorry

end initial_hno3_concentration_l3796_379643


namespace square_side_length_l3796_379630

/-- Given a square with diagonal length 2√2, prove that its side length is 2. -/
theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ (s : ℝ), s * s = (d * d) / 2 ∧ s = 2 := by
  sorry

end square_side_length_l3796_379630


namespace max_product_sum_11_l3796_379698

theorem max_product_sum_11 :
  ∃ (a b : ℕ), a + b = 11 ∧
  ∀ (x y : ℕ), x + y = 11 → x * y ≤ a * b ∧
  a * b = 30 :=
sorry

end max_product_sum_11_l3796_379698


namespace interval_length_implies_difference_l3796_379636

/-- Given an inequality a ≤ 3x + 5 ≤ b, where the length of the interval of solutions is 15, prove that b - a = 45 -/
theorem interval_length_implies_difference (a b : ℝ) : 
  (∃ x : ℝ, a ≤ 3*x + 5 ∧ 3*x + 5 ≤ b) → 
  ((b - 5) / 3 - (a - 5) / 3 = 15) → 
  b - a = 45 := by sorry

end interval_length_implies_difference_l3796_379636


namespace geometric_sequence_condition_l3796_379602

/-- A sequence {a_n} with sum of first n terms S_n = p^n + q, where p ≠ 0 and p ≠ 1, 
    is geometric if and only if q = -1 -/
theorem geometric_sequence_condition (p : ℝ) (q : ℝ) (h_p_nonzero : p ≠ 0) (h_p_not_one : p ≠ 1) :
  let a : ℕ → ℝ := fun n => (p^n + q) - (p^(n-1) + q)
  let S : ℕ → ℝ := fun n => p^n + q
  (∀ n : ℕ, n ≥ 2 → a (n+1) / a n = a 2 / a 1) ↔ q = -1 := by
  sorry

end geometric_sequence_condition_l3796_379602


namespace probability_circle_or_square_l3796_379642

def total_figures : ℕ := 10
def num_circles : ℕ := 3
def num_squares : ℕ := 4
def num_triangles : ℕ := 3

theorem probability_circle_or_square :
  (num_circles + num_squares : ℚ) / total_figures = 7 / 10 :=
by sorry

end probability_circle_or_square_l3796_379642


namespace intersection_sum_l3796_379669

theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 6 + a) → 
  (6 = (1/3) * 3 + b) → 
  a + b = 6 := by
sorry

end intersection_sum_l3796_379669


namespace pet_store_bird_dog_ratio_l3796_379607

/-- Given a pet store with dogs, cats, birds, and fish, prove the ratio of birds to dogs. -/
theorem pet_store_bird_dog_ratio 
  (dogs : ℕ) 
  (cats : ℕ) 
  (birds : ℕ) 
  (fish : ℕ) 
  (h1 : dogs = 6) 
  (h2 : cats = dogs / 2) 
  (h3 : fish = 3 * dogs) 
  (h4 : dogs + cats + birds + fish = 39) : 
  birds / dogs = 2 := by
  sorry

end pet_store_bird_dog_ratio_l3796_379607


namespace min_pizzas_for_johns_van_l3796_379644

/-- The minimum whole number of pizzas needed to recover the van's cost -/
def min_pizzas (van_cost : ℕ) (earnings_per_pizza : ℕ) (gas_cost : ℕ) : ℕ :=
  (van_cost + (earnings_per_pizza - gas_cost - 1)) / (earnings_per_pizza - gas_cost)

theorem min_pizzas_for_johns_van :
  min_pizzas 8000 15 4 = 728 := by sorry

end min_pizzas_for_johns_van_l3796_379644


namespace sum_interior_angles_pentagon_l3796_379610

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides. -/
def pentagon_sides : ℕ := 5

/-- The sum of the interior angles of a pentagon is 540 degrees. -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon_sides = 540 := by
  sorry

end sum_interior_angles_pentagon_l3796_379610


namespace gratuity_percentage_is_twenty_percent_l3796_379605

def number_of_people : ℕ := 6
def total_bill : ℚ := 720
def average_cost_before_gratuity : ℚ := 100

theorem gratuity_percentage_is_twenty_percent :
  let total_before_gratuity : ℚ := number_of_people * average_cost_before_gratuity
  let gratuity_amount : ℚ := total_bill - total_before_gratuity
  gratuity_amount / total_before_gratuity = 1/5 := by
sorry

end gratuity_percentage_is_twenty_percent_l3796_379605


namespace basketball_lineup_count_l3796_379647

/-- The number of ways to choose a starting lineup for a basketball team -/
def starting_lineup_count (total_members : ℕ) (center_capable : ℕ) : ℕ :=
  center_capable * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose a starting lineup for a specific basketball team -/
theorem basketball_lineup_count :
  starting_lineup_count 12 4 = 31680 :=
by sorry

end basketball_lineup_count_l3796_379647


namespace divisibility_by_24_l3796_379695

theorem divisibility_by_24 (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) : 
  24 ∣ p^2 - 1 := by
sorry

end divisibility_by_24_l3796_379695


namespace inequality_of_powers_l3796_379635

theorem inequality_of_powers (m n : ℕ) : (5 + 3 * Real.sqrt 2) ^ m ≠ (3 + 5 * Real.sqrt 2) ^ n := by
  sorry

end inequality_of_powers_l3796_379635


namespace chess_club_officers_l3796_379675

/-- The number of members in the Chess Club -/
def total_members : ℕ := 25

/-- The number of officers to be selected -/
def num_officers : ℕ := 3

/-- Function to calculate the number of ways to select officers -/
def select_officers (total : ℕ) (officers : ℕ) : ℕ :=
  let case1 := (total - 2) * (total - 3) * (total - 4)  -- Neither Alice nor Bob
  let case2 := 3 * 2 * (total - 3)  -- Both Alice and Bob
  case1 + case2

/-- Theorem stating the number of ways to select officers -/
theorem chess_club_officers :
  select_officers total_members num_officers = 10758 := by
  sorry

end chess_club_officers_l3796_379675


namespace imaginary_part_of_i_squared_times_i_minus_one_l3796_379606

theorem imaginary_part_of_i_squared_times_i_minus_one (i : ℂ) : 
  i^2 = -1 → Complex.im (i^2 * (i - 1)) = -1 := by sorry

end imaginary_part_of_i_squared_times_i_minus_one_l3796_379606


namespace water_amount_in_sport_formulation_l3796_379674

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio := ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio := 
  let f := standard_ratio.flavoring
  let c := standard_ratio.corn_syrup
  let w := standard_ratio.water
  ⟨f, f * 4, f * 15⟩

/-- Calculates the amount of water given the amount of corn syrup -/
def water_amount (corn_syrup_amount : ℚ) : ℚ :=
  (corn_syrup_amount * sport_ratio.water) / sport_ratio.corn_syrup

/-- Theorem: The amount of water in the sport formulation is 7.5 ounces when there are 2 ounces of corn syrup -/
theorem water_amount_in_sport_formulation :
  water_amount 2 = 7.5 := by sorry

end water_amount_in_sport_formulation_l3796_379674


namespace unique_solution_condition_l3796_379657

theorem unique_solution_condition (a c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + 2) ↔ c ≠ 4 := by
  sorry

end unique_solution_condition_l3796_379657


namespace punch_bowl_capacity_l3796_379622

/-- Proves that the total capacity of a punch bowl is 72 cups given the specified conditions -/
theorem punch_bowl_capacity 
  (lemonade : ℕ) 
  (cranberry : ℕ) 
  (h1 : lemonade * 5 = cranberry * 3) 
  (h2 : cranberry = lemonade + 18) : 
  lemonade + cranberry = 72 := by
  sorry

#check punch_bowl_capacity

end punch_bowl_capacity_l3796_379622


namespace committee_selection_probability_l3796_379640

theorem committee_selection_probability :
  let total_members : ℕ := 9
  let english_teachers : ℕ := 3
  let select_count : ℕ := 2

  let total_combinations := total_members.choose select_count
  let english_combinations := english_teachers.choose select_count

  (english_combinations : ℚ) / total_combinations = 1 / 12 :=
by sorry

end committee_selection_probability_l3796_379640


namespace fast_food_constant_and_variables_l3796_379693

/-- A linear pricing model for fast food boxes -/
structure FastFoodPricing where
  cost_per_box : ℝ  -- Cost per box in yuan
  num_boxes : ℝ     -- Number of boxes purchased
  total_cost : ℝ    -- Total cost in yuan
  pricing_model : total_cost = cost_per_box * num_boxes

/-- Theorem stating that in a FastFoodPricing model, the constant is the cost per box,
    and the variables are the number of boxes and the total cost -/
theorem fast_food_constant_and_variables (model : FastFoodPricing) :
  (∃ (k : ℝ), k = model.cost_per_box ∧ k ≠ 0) ∧
  (∀ (n s : ℝ), n = model.num_boxes ∧ s = model.total_cost →
    s = model.cost_per_box * n) :=
sorry

end fast_food_constant_and_variables_l3796_379693


namespace g_6_eq_1_l3796_379626

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the conditions on f
axiom f_1 : f 1 = 1
axiom f_add_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_add_1 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 1 - x

-- State the theorem to be proved
theorem g_6_eq_1 : g 6 = 1 := by sorry

end g_6_eq_1_l3796_379626


namespace fixed_point_of_log_function_l3796_379612

-- Define the logarithm function with base a
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x + 2) + 3
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_base a (x + 2) + 3

-- Theorem statement
theorem fixed_point_of_log_function (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
  f a (-1) = 3 := by sorry

end fixed_point_of_log_function_l3796_379612


namespace mopping_time_is_30_l3796_379679

def vacuum_time : ℕ := 45
def dusting_time : ℕ := 60
def cat_brushing_time : ℕ := 5
def num_cats : ℕ := 3
def total_free_time : ℕ := 3 * 60
def remaining_free_time : ℕ := 30

def total_cleaning_time : ℕ := total_free_time - remaining_free_time

def other_tasks_time : ℕ := vacuum_time + dusting_time + (cat_brushing_time * num_cats)

theorem mopping_time_is_30 : 
  total_cleaning_time - other_tasks_time = 30 := by sorry

end mopping_time_is_30_l3796_379679


namespace book_code_is_mirror_l3796_379648

/-- Represents the coding system --/
structure CodeSystem where
  book : String
  mirror : String
  board : String
  writing_item : String

/-- The given coding rules --/
def given_code : CodeSystem :=
  { book := "certain_item",
    mirror := "board",
    board := "board",
    writing_item := "2" }

/-- Theorem: The code for 'book' is 'mirror' --/
theorem book_code_is_mirror (code : CodeSystem) (h1 : code.book = "certain_item") 
  (h2 : code.mirror = "board") : code.book = "mirror" :=
by sorry

end book_code_is_mirror_l3796_379648


namespace ratio_equation_solution_l3796_379620

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 20 - 7 * a) : 
  a = 20 / 11 := by
sorry

end ratio_equation_solution_l3796_379620


namespace four_digit_number_problem_l3796_379600

theorem four_digit_number_problem :
  ∀ N : ℕ,
  (1000 ≤ N) ∧ (N < 10000) →
  (∃ x y : ℕ,
    (1 ≤ x) ∧ (x ≤ 9) ∧
    (100 ≤ y) ∧ (y < 1000) ∧
    (N = 1000 * x + y) ∧
    (N / y = 3) ∧
    (N % y = 8)) →
  (N = 1496 ∨ N = 2996) :=
by sorry

end four_digit_number_problem_l3796_379600


namespace student_tickets_sold_l3796_379667

theorem student_tickets_sold (adult_price student_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 6)
  (h2 : student_price = 3)
  (h3 : total_tickets = 846)
  (h4 : total_revenue = 3846) :
  ∃ (adult_tickets student_tickets : ℕ),
    adult_tickets + student_tickets = total_tickets ∧
    adult_price * adult_tickets + student_price * student_tickets = total_revenue ∧
    student_tickets = 410 := by
  sorry

end student_tickets_sold_l3796_379667


namespace largest_positive_integer_for_binary_op_l3796_379655

def binary_op (n : Int) : Int := n - (n * 5)

theorem largest_positive_integer_for_binary_op :
  ∀ n : ℕ+, n > 1 → binary_op n.val ≥ 14 :=
by
  sorry

end largest_positive_integer_for_binary_op_l3796_379655


namespace area_above_line_ratio_l3796_379688

/-- Given two positive real numbers a and b, where a > b > 1/2 * a,
    and two squares with side lengths a and b placed next to each other,
    with the larger square having its lower left corner at (0,0) and
    the smaller square having its lower left corner at (a,0),
    if the area above the line passing through (0,a) and (a+b,0) in both squares is 2013,
    and (a,b) is the unique pair maximizing a+b,
    then a/b = ∛5√3. -/
theorem area_above_line_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hba : b > (1/2) * a) (harea : (a^3 / (2*(a+b))) + (a*b/2) = 2013)
  (hmax : ∀ (x y : ℝ), x > 0 → y > 0 → x > y → y > (1/2) * x →
    (x^3 / (2*(x+y))) + (x*y/2) = 2013 → x + y ≤ a + b) :
  a / b = (3 : ℝ)^(1/5) :=
sorry

end area_above_line_ratio_l3796_379688


namespace percentage_difference_l3796_379663

theorem percentage_difference (w q y z P : ℝ) : 
  w = q * (1 - P / 100) →
  q = y * 0.6 →
  z = y * 0.54 →
  z = w * 1.5 →
  P = 78.4 := by
sorry

end percentage_difference_l3796_379663


namespace sin_cos_sum_one_l3796_379671

theorem sin_cos_sum_one (x : ℝ) :
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0 ∨ x = Real.pi / 2) := by
  sorry

end sin_cos_sum_one_l3796_379671


namespace sum_of_squares_l3796_379624

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 := by
  sorry

end sum_of_squares_l3796_379624


namespace f_properties_l3796_379615

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

-- Define the range of f
def range_f : Set ℝ := {y : ℝ | ∃ x, f x = y}

-- Theorem statement
theorem f_properties :
  (range_f = {y : ℝ | y ≥ 3/2}) ∧
  (∀ a : ℝ, a ∈ range_f → |a - 1| + |a + 1| > 3/(2*a) ∧ 3/(2*a) > 7/2 - 2*a) :=
sorry

end f_properties_l3796_379615


namespace lemonade_production_l3796_379632

/-- Given that John can prepare 15 lemonades from 3 lemons, 
    prove that he can make 90 lemonades from 18 lemons. -/
theorem lemonade_production (initial_lemons : ℕ) (initial_lemonades : ℕ) (new_lemons : ℕ) : 
  initial_lemons = 3 → initial_lemonades = 15 → new_lemons = 18 →
  (new_lemons * initial_lemonades / initial_lemons : ℕ) = 90 := by
  sorry

end lemonade_production_l3796_379632


namespace prime_sum_product_l3796_379656

theorem prime_sum_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 105 → p * q = 206 := by
  sorry

end prime_sum_product_l3796_379656


namespace sandbox_width_l3796_379609

/-- The width of a rectangle given its length and area -/
theorem sandbox_width (length : ℝ) (area : ℝ) (h1 : length = 312) (h2 : area = 45552) :
  area / length = 146 := by
  sorry

end sandbox_width_l3796_379609


namespace tan_3285_degrees_l3796_379666

theorem tan_3285_degrees : Real.tan (3285 * π / 180) = 1 := by
  sorry

end tan_3285_degrees_l3796_379666


namespace sufficient_but_not_necessary_l3796_379687

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (x = -y → x^2 - y^2 - x - y = 0) ∧ 
  ¬(x^2 - y^2 - x - y = 0 → x = -y) := by
  sorry

end sufficient_but_not_necessary_l3796_379687


namespace division_problem_l3796_379650

theorem division_problem (A : ℕ) : 
  (11 / A = 3) ∧ (11 % A = 2) → A = 3 := by sorry

end division_problem_l3796_379650


namespace meters_not_most_appropriate_for_map_distance_l3796_379659

-- Define a type for units of measurement
inductive MeasurementUnit
| Meters
| Centimeters

-- Define a function to determine the most appropriate unit for map distances
def mostAppropriateUnitForMapDistance : MeasurementUnit := sorry

-- Theorem stating that meters is not the most appropriate unit
theorem meters_not_most_appropriate_for_map_distance :
  mostAppropriateUnitForMapDistance ≠ MeasurementUnit.Meters := by
  sorry

end meters_not_most_appropriate_for_map_distance_l3796_379659


namespace opposite_teal_is_blue_l3796_379661

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Teal | Violet

-- Define a cube type
structure Cube where
  faces : Fin 6 → Color
  unique_colors : ∀ i j, i ≠ j → faces i ≠ faces j

-- Define the views
def view1 (c : Cube) : Prop :=
  c.faces 0 = Color.Yellow ∧ c.faces 1 = Color.Blue ∧ c.faces 2 = Color.Orange

def view2 (c : Cube) : Prop :=
  c.faces 0 = Color.Yellow ∧ c.faces 1 = Color.Black ∧ c.faces 2 = Color.Orange

def view3 (c : Cube) : Prop :=
  c.faces 0 = Color.Yellow ∧ c.faces 1 = Color.Violet ∧ c.faces 2 = Color.Orange

-- Theorem statement
theorem opposite_teal_is_blue (c : Cube) 
  (h1 : view1 c) (h2 : view2 c) (h3 : view3 c) :
  ∃ i j, i ≠ j ∧ c.faces i = Color.Teal ∧ c.faces j = Color.Blue ∧ 
  (∀ k, k ≠ i → k ≠ j → c.faces k ≠ Color.Teal ∧ c.faces k ≠ Color.Blue) :=
sorry

end opposite_teal_is_blue_l3796_379661


namespace initial_mixture_volume_l3796_379603

/-- Proves that the initial volume of a mixture is 425 litres given the conditions -/
theorem initial_mixture_volume :
  ∀ (V : ℝ),
  (V > 0) →
  (0.10 * V = V * 0.10) →
  (0.10 * V + 25 = 0.15 * (V + 25)) →
  V = 425 :=
λ V hV_pos hWater_ratio hNew_ratio =>
  sorry

#check initial_mixture_volume

end initial_mixture_volume_l3796_379603


namespace greatest_common_divisor_with_same_remainder_l3796_379697

theorem greatest_common_divisor_with_same_remainder (a b c : ℕ) (h : a < b ∧ b < c) :
  ∃ (d : ℕ), d > 0 ∧ d = Nat.gcd (b - a) (c - b) ∧
  ∀ (k : ℕ), k > d → ¬(∃ (r : ℕ), a % k = r ∧ b % k = r ∧ c % k = r) := by
  sorry

#check greatest_common_divisor_with_same_remainder 25 57 105

end greatest_common_divisor_with_same_remainder_l3796_379697


namespace gcd_5040_13860_l3796_379690

theorem gcd_5040_13860 : Nat.gcd 5040 13860 = 420 := by
  sorry

end gcd_5040_13860_l3796_379690


namespace inequality_system_solution_l3796_379694

theorem inequality_system_solution (a : ℝ) :
  (∃ x : ℝ, x + a ≥ 0 ∧ 1 - 2*x > x - 2) ↔ a > -1 := by
  sorry

end inequality_system_solution_l3796_379694


namespace sports_equipment_purchasing_l3796_379677

/-- Represents the price and quantity information for sports equipment --/
structure EquipmentInfo where
  price_a : ℕ
  price_b : ℕ
  total_budget : ℕ
  total_units : ℕ

/-- Theorem about sports equipment purchasing --/
theorem sports_equipment_purchasing (info : EquipmentInfo) 
  (h1 : 3 * info.price_a + info.price_b = 500)
  (h2 : info.price_a + 2 * info.price_b = 250)
  (h3 : info.total_budget = 2700)
  (h4 : info.total_units = 25) :
  info.price_a = 150 ∧ 
  info.price_b = 50 ∧
  (∀ m : ℕ, m * info.price_a + (info.total_units - m) * info.price_b ≤ info.total_budget → m ≤ 14) ∧
  (∀ m : ℕ, 12 ≤ m → m ≤ 14 → m * info.price_a + (info.total_units - m) * info.price_b ≥ 2450) := by
  sorry

end sports_equipment_purchasing_l3796_379677


namespace line_intersection_l3796_379645

theorem line_intersection : ∃! p : ℚ × ℚ, 
  5 * p.1 - 3 * p.2 = 7 ∧ 
  8 * p.1 + 2 * p.2 = 22 :=
by
  -- The point (40/17, 27/17) satisfies both equations
  have h1 : 5 * (40/17) - 3 * (27/17) = 7 := by sorry
  have h2 : 8 * (40/17) + 2 * (27/17) = 22 := by sorry

  -- Prove uniqueness
  sorry

end line_intersection_l3796_379645


namespace total_money_after_redistribution_l3796_379621

/-- Represents the money redistribution process among three friends. -/
def moneyRedistribution (initialAmy : ℝ) (initialJan : ℝ) (initialToy : ℝ) : ℝ :=
  let afterAmy := initialAmy - 2 * (initialJan + initialToy) + 3 * initialJan + 3 * initialToy
  let afterJan := 3 * (initialAmy - 2 * (initialJan + initialToy)) + 
                  (3 * initialJan - 2 * (initialAmy - 2 * (initialJan + initialToy) + 3 * initialToy)) + 
                  3 * 3 * initialToy
  let afterToy := 27  -- Given condition
  afterAmy + afterJan + afterToy

/-- Theorem stating that the total amount after redistribution is 243 when Toy starts and ends with 27. -/
theorem total_money_after_redistribution :
  ∀ (initialAmy : ℝ) (initialJan : ℝ),
  moneyRedistribution initialAmy initialJan 27 = 243 :=
by
  sorry

#eval moneyRedistribution 0 0 27  -- For verification

end total_money_after_redistribution_l3796_379621


namespace projection_theorem_l3796_379660

def vector_a : ℝ × ℝ := (-2, -4)

theorem projection_theorem (b : ℝ × ℝ) 
  (angle_ab : Real.cos (120 * π / 180) = -1/2)
  (magnitude_b : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5) :
  let projection := (Real.sqrt ((vector_a.1)^2 + (vector_a.2)^2)) * 
                    Real.cos (120 * π / 180)
  projection = -Real.sqrt 5 := by
  sorry

end projection_theorem_l3796_379660


namespace condition_sufficient_not_necessary_l3796_379611

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b, (a - b)^3 * b^2 > 0 → a > b) ∧
  (∃ a b, a > b ∧ (a - b)^3 * b^2 ≤ 0) :=
by sorry

end condition_sufficient_not_necessary_l3796_379611


namespace difference_calculations_l3796_379637

theorem difference_calculations (d1 d2 d3 : Int) 
  (h1 : d1 = -15)
  (h2 : d2 = 405)
  (h3 : d3 = 1280) :
  let sum := d1 + d2 + d3
  let product := d1 * d2 * d3
  let avg_squares := ((d1^2 + d2^2 + d3^2) : ℚ) / 3
  sum = 1670 ∧ 
  product = -7728000 ∧ 
  avg_squares = 600883 + 1/3 ∧
  (product : ℚ) - avg_squares = -8328883 - 1/3 := by
sorry

#eval (-15 : Int) + 405 + 1280
#eval (-15 : Int) * 405 * 1280
#eval ((-15 : ℚ)^2 + 405^2 + 1280^2) / 3
#eval (-7728000 : ℚ) - (((-15 : ℚ)^2 + 405^2 + 1280^2) / 3)

end difference_calculations_l3796_379637


namespace eunji_uncle_money_l3796_379681

/-- The amount of money Eunji received from her uncle -/
def uncle_money : ℕ := sorry

/-- The amount of money Eunji received from her mother -/
def mother_money : ℕ := 550

/-- The total amount of money Eunji has after receiving money from her mother -/
def total_money : ℕ := 1000

/-- Theorem stating that Eunji received 900 won from her uncle -/
theorem eunji_uncle_money :
  uncle_money = 900 ∧
  uncle_money / 2 + mother_money = total_money :=
sorry

end eunji_uncle_money_l3796_379681


namespace johns_out_of_pocket_expense_l3796_379696

/-- Calculates the amount John paid out of pocket for a new computer and accessories,
    given the costs and the sale of his PlayStation. -/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value : ℝ)
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400)
  (discount_rate : ℝ)
  (h4 : discount_rate = 0.2) :
  computer_cost + accessories_cost - playstation_value * (1 - discount_rate) = 580 := by
sorry


end johns_out_of_pocket_expense_l3796_379696


namespace bess_frisbee_throws_l3796_379662

/-- The problem of determining how many times Bess throws the Frisbee -/
theorem bess_frisbee_throws :
  ∀ (bess_throw_distance : ℕ) 
    (holly_throw_distance : ℕ) 
    (holly_throw_count : ℕ) 
    (total_distance : ℕ),
  bess_throw_distance = 20 →
  holly_throw_distance = 8 →
  holly_throw_count = 5 →
  total_distance = 200 →
  ∃ (bess_throw_count : ℕ),
    bess_throw_count * (2 * bess_throw_distance) + 
    holly_throw_count * holly_throw_distance = total_distance ∧
    bess_throw_count = 4 := by
  sorry

end bess_frisbee_throws_l3796_379662


namespace weight_difference_e_d_l3796_379618

/-- Given weights of individuals A, B, C, D, and E, prove that E weighs 3 kg more than D -/
theorem weight_difference_e_d (w_a w_b w_c w_d w_e : ℝ) : 
  (w_a + w_b + w_c) / 3 = 60 →
  (w_a + w_b + w_c + w_d) / 4 = 65 →
  (w_b + w_c + w_d + w_e) / 4 = 64 →
  w_a = 87 →
  w_e - w_d = 3 := by
sorry

end weight_difference_e_d_l3796_379618


namespace car_journey_speed_l3796_379678

def car_journey (v : ℝ) : Prop :=
  let first_part_time : ℝ := 1
  let first_part_speed : ℝ := 40
  let second_part_time : ℝ := 0.5
  let third_part_time : ℝ := 2
  let total_time : ℝ := first_part_time + second_part_time + third_part_time
  let total_distance : ℝ := first_part_speed * first_part_time + v * (second_part_time + third_part_time)
  let average_speed : ℝ := 54.285714285714285
  total_distance / total_time = average_speed

theorem car_journey_speed : car_journey 60 := by
  sorry

end car_journey_speed_l3796_379678


namespace line_circle_intersection_l3796_379676

/-- A line y = kx + 3 intersects a circle (x - 3)^2 + (y - 2)^2 = 4 at two points M and N. 
    If the distance between M and N is at least 2, then k is outside the interval (3 - 2√2, 3 + 2√2). -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    (M.1 - 3)^2 + (M.2 - 2)^2 = 4 ∧
    (N.1 - 3)^2 + (N.2 - 2)^2 = 4 ∧
    M.2 = k * M.1 + 3 ∧
    N.2 = k * N.1 + 3 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 4) →
  k < 3 - 2 * Real.sqrt 2 ∨ k > 3 + 2 * Real.sqrt 2 :=
by sorry

end line_circle_intersection_l3796_379676


namespace smallest_k_no_real_roots_l3796_379668

theorem smallest_k_no_real_roots : ∀ k : ℤ,
  (∀ x : ℝ, (1/2 : ℝ) * x^2 + 3*x + (k : ℝ) ≠ 0) ↔ k ≥ 5 :=
by sorry

end smallest_k_no_real_roots_l3796_379668


namespace sammy_gift_wrapping_l3796_379634

/-- The number of gifts Sammy can wrap -/
def num_gifts : ℕ := 8

/-- The length of ribbon required for each gift in meters -/
def ribbon_per_gift : ℝ := 1.5

/-- The total length of Tom's ribbon in meters -/
def total_ribbon : ℝ := 15

/-- The length of ribbon left after wrapping all gifts in meters -/
def ribbon_left : ℝ := 3

/-- Theorem stating that the number of gifts Sammy can wrap is correct -/
theorem sammy_gift_wrapping :
  (↑num_gifts : ℝ) * ribbon_per_gift = total_ribbon - ribbon_left :=
by sorry

end sammy_gift_wrapping_l3796_379634


namespace min_value_of_expression_lower_bound_achievable_l3796_379616

theorem min_value_of_expression (x y : ℝ) : (x * y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

theorem lower_bound_achievable : ∃ x y : ℝ, (x * y - 1)^2 + (x + y)^2 = 1 := by
  sorry

end min_value_of_expression_lower_bound_achievable_l3796_379616


namespace no_solution_implies_a_geq_6_l3796_379654

theorem no_solution_implies_a_geq_6 (a : ℝ) : 
  (∀ x : ℝ, ¬(2*x - a > 0 ∧ 3*x - 4 < 5)) → a ≥ 6 := by
  sorry

end no_solution_implies_a_geq_6_l3796_379654


namespace function_parameters_l3796_379604

/-- Given a function f(x) = 2sin(ωx + φ) with the specified properties, prove that ω = 2 and φ = π/3 -/
theorem function_parameters (ω φ : ℝ) (f : ℝ → ℝ) : 
  ω > 0 →
  |φ| < π/2 →
  (∀ x, f x = 2 * Real.sin (ω * x + φ)) →
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ π) →
  f 0 = Real.sqrt 3 →
  ω = 2 ∧ φ = π/3 := by
sorry

end function_parameters_l3796_379604


namespace smallest_number_l3796_379638

-- Define the base conversion function
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def A : Nat := to_decimal [1, 1, 1, 1] 2
def B : Nat := to_decimal [0, 1, 2] 6
def C : Nat := to_decimal [0, 0, 0, 1] 4
def D : Nat := to_decimal [1, 0, 1] 8

-- Theorem statement
theorem smallest_number : A < B ∧ A < C ∧ A < D := by
  sorry

end smallest_number_l3796_379638


namespace consecutive_primes_in_sequence_l3796_379673

theorem consecutive_primes_in_sequence (a b : ℕ) (h1 : a > b) (h2 : b > 1) :
  ∃ n : ℕ, n ≥ 2 → 
    ¬(Nat.Prime ((a^n - 1) / (b^n - 1)) ∧ Nat.Prime ((a^(n+1) - 1) / (b^(n+1) - 1))) :=
by sorry

end consecutive_primes_in_sequence_l3796_379673


namespace seventh_term_of_geometric_sequence_l3796_379653

/-- Given a geometric sequence with first term a and common ratio r,
    this function returns the nth term of the sequence. -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- Theorem stating that the 7th term of a geometric sequence
    with first term 5 and second term -1 is 1/3125 -/
theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 5
  let a₂ : ℚ := -1
  let r : ℚ := a₂ / a₁
  geometric_term a₁ r 7 = 1 / 3125 := by
  sorry


end seventh_term_of_geometric_sequence_l3796_379653


namespace vector_lines_correct_l3796_379649

/-- Vector field in R³ -/
def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ := (0, 9*z, -4*y)

/-- Vector lines of the given vector field -/
def vector_lines (x y z C₁ C₂ : ℝ) : Prop :=
  9 * z^2 + 4 * y^2 = C₁ ∧ x = C₂

/-- Theorem stating that the vector lines are correct for the given vector field -/
theorem vector_lines_correct :
  ∀ (x y z C₁ C₂ : ℝ),
    vector_lines x y z C₁ C₂ ↔
    ∃ (t : ℝ), (x, y, z) = (C₂, 
                            9 * t * (vector_field x y z).2.1, 
                            -4 * t * (vector_field x y z).2.2) :=
sorry

end vector_lines_correct_l3796_379649


namespace quadratic_root_implies_coefficient_l3796_379664

theorem quadratic_root_implies_coefficient 
  (b c : ℝ) 
  (h : ∃ x : ℂ, x^2 + b*x + c = 0 ∧ x = 2 + I) : 
  b = -4 := by
sorry

end quadratic_root_implies_coefficient_l3796_379664


namespace age_digits_product_l3796_379699

/-- A function that returns the digits of a two-digit number -/
def digits (n : ℕ) : List ℕ :=
  [n / 10, n % 10]

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- A function that checks if a number is a power of another number -/
def isPowerOf (n : ℕ) (base : ℕ) : Prop :=
  ∃ k : ℕ, n = base ^ k

/-- A function that calculates the sum of a list of numbers -/
def sum (l : List ℕ) : ℕ :=
  l.foldl (·+·) 0

/-- A function that calculates the product of a list of numbers -/
def product (l : List ℕ) : ℕ :=
  l.foldl (·*·) 1

theorem age_digits_product : 
  ∃ (x y : ℕ),
    isTwoDigit x ∧ 
    isTwoDigit y ∧ 
    isPowerOf x 5 ∧ 
    isPowerOf y 2 ∧ 
    Odd (sum (digits x ++ digits y)) → 
    product (digits x ++ digits y) = 240 := by
  sorry

end age_digits_product_l3796_379699


namespace principal_is_800_l3796_379633

/-- Calculates the principal amount given the final amount, interest rate, and time -/
def calculate_principal (amount : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  amount / (1 + rate * time)

/-- Theorem stating that the principal is 800 given the problem conditions -/
theorem principal_is_800 : 
  let amount : ℚ := 896
  let rate : ℚ := 5 / 100
  let time : ℚ := 12 / 5
  calculate_principal amount rate time = 800 := by sorry

end principal_is_800_l3796_379633


namespace digit_equality_l3796_379623

theorem digit_equality (n k : ℕ) : 
  (10^(k-1) ≤ n^n ∧ n^n < 10^k) ∧ 
  (10^(n-1) ≤ k^k ∧ k^k < 10^n) ↔ 
  ((n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9)) :=
by sorry

end digit_equality_l3796_379623


namespace stating_pacific_area_rounded_l3796_379625

/-- The area of the Pacific Ocean in square kilometers -/
def pacific_area : ℕ := 19996800

/-- Conversion factor from square kilometers to ten thousand square kilometers -/
def ten_thousand : ℕ := 10000

/-- Rounds a natural number to the nearest multiple of ten thousand -/
def round_to_nearest_ten_thousand (n : ℕ) : ℕ :=
  (n + 5000) / 10000 * 10000

/-- 
Theorem stating that the area of the Pacific Ocean, when converted to 
ten thousand square kilometers and rounded to the nearest ten thousand, 
is equal to 2000 ten thousand square kilometers
-/
theorem pacific_area_rounded : 
  round_to_nearest_ten_thousand (pacific_area / ten_thousand) = 2000 := by
  sorry


end stating_pacific_area_rounded_l3796_379625


namespace problem_statement_l3796_379683

theorem problem_statement : ((16^15 / 16^14)^3 * 8^3) / 2^9 = 4096 := by
  sorry

end problem_statement_l3796_379683


namespace average_hours_upside_down_per_month_l3796_379686

/-- The number of inches Alex needs to grow to ride the roller coaster -/
def height_difference : ℚ := 54 - 48

/-- Alex's normal growth rate in inches per month -/
def normal_growth_rate : ℚ := 1 / 3

/-- Alex's growth rate in inches per hour when hanging upside down -/
def upside_down_growth_rate : ℚ := 1 / 12

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Theorem stating the average number of hours Alex needs to hang upside down per month -/
theorem average_hours_upside_down_per_month :
  (height_difference - normal_growth_rate * months_per_year) / (upside_down_growth_rate * months_per_year) = 2 := by
  sorry

end average_hours_upside_down_per_month_l3796_379686


namespace square_triangle_count_l3796_379614

theorem square_triangle_count (total_shapes : ℕ) (total_edges : ℕ) 
  (h_total_shapes : total_shapes = 35)
  (h_total_edges : total_edges = 120) :
  ∃ (squares triangles : ℕ),
    squares + triangles = total_shapes ∧
    4 * squares + 3 * triangles = total_edges ∧
    squares = 20 ∧
    triangles = 15 := by
  sorry

end square_triangle_count_l3796_379614
