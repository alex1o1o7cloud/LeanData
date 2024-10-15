import Mathlib

namespace NUMINAMATH_CALUDE_yangtze_length_scientific_notation_l1399_139910

/-- The length of the Yangtze River in meters -/
def yangtze_length : ℕ := 6300000

/-- The scientific notation representation of the Yangtze River's length -/
def yangtze_scientific : ℝ := 6.3 * (10 ^ 6)

theorem yangtze_length_scientific_notation : 
  (yangtze_length : ℝ) = yangtze_scientific := by sorry

end NUMINAMATH_CALUDE_yangtze_length_scientific_notation_l1399_139910


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l1399_139986

theorem regular_polygon_properties (exterior_angle : ℝ) 
  (h1 : exterior_angle = 15)
  (h2 : exterior_angle > 0) :
  ∃ (n : ℕ) (sum_interior : ℝ),
    n = 24 ∧ 
    sum_interior = 3960 ∧
    n * exterior_angle = 360 ∧
    sum_interior = 180 * (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l1399_139986


namespace NUMINAMATH_CALUDE_marble_problem_solution_l1399_139965

/-- Represents the number of marbles of each color in a box -/
structure MarbleBox where
  red : ℕ
  green : ℕ
  yellow : ℕ
  other : ℕ

/-- The conditions of the marble problem -/
def marble_problem (box : MarbleBox) : Prop :=
  box.red = 20 ∧
  box.green = 3 * box.red ∧
  box.yellow = box.green / 5 ∧
  box.red + box.green + box.yellow + box.other = 4 * box.green

/-- The theorem stating the solution to the marble problem -/
theorem marble_problem_solution (box : MarbleBox) :
  marble_problem box → box.other = 148 := by
  sorry


end NUMINAMATH_CALUDE_marble_problem_solution_l1399_139965


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l1399_139905

theorem complex_modulus_theorem (z : ℂ) (h : z + 3 / z = 0) : Complex.abs z = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l1399_139905


namespace NUMINAMATH_CALUDE_households_with_only_bike_l1399_139990

/-- Proves the number of households with only a bike given the provided information -/
theorem households_with_only_bike
  (total_households : ℕ)
  (households_without_car_or_bike : ℕ)
  (households_with_both_car_and_bike : ℕ)
  (households_with_car : ℕ)
  (h1 : total_households = 90)
  (h2 : households_without_car_or_bike = 11)
  (h3 : households_with_both_car_and_bike = 22)
  (h4 : households_with_car = 44) :
  total_households - households_without_car_or_bike - households_with_car - households_with_both_car_and_bike = 35 :=
by sorry

end NUMINAMATH_CALUDE_households_with_only_bike_l1399_139990


namespace NUMINAMATH_CALUDE_last_fish_is_sudak_l1399_139997

-- Define the types of fish
inductive Fish : Type
| Perch : Fish
| Pike : Fish
| Sudak : Fish

-- Define the initial fish counts
def initial_perches : Nat := 6
def initial_pikes : Nat := 7
def initial_sudaks : Nat := 8

-- Define the eating rules
def can_eat (eater prey : Fish) : Prop :=
  match eater, prey with
  | Fish.Perch, Fish.Pike => True
  | Fish.Pike, Fish.Pike => True
  | Fish.Pike, Fish.Perch => True
  | _, _ => False

-- Define the restriction on eating fish that have eaten an odd number
def odd_eater_restriction (f : Fish → Nat) : Prop :=
  ∀ fish, f fish % 2 = 1 → ∀ other, ¬(can_eat other fish)

-- Define the theorem
theorem last_fish_is_sudak :
  ∃ (final_state : Fish → Nat),
    (final_state Fish.Perch + final_state Fish.Pike + final_state Fish.Sudak = 1) ∧
    (final_state Fish.Sudak = 1) ∧
    (∃ (intermediate_state : Fish → Nat),
      odd_eater_restriction intermediate_state ∧
      (∀ fish, final_state fish ≤ intermediate_state fish) ∧
      (intermediate_state Fish.Perch ≤ initial_perches) ∧
      (intermediate_state Fish.Pike ≤ initial_pikes) ∧
      (intermediate_state Fish.Sudak ≤ initial_sudaks)) :=
sorry

end NUMINAMATH_CALUDE_last_fish_is_sudak_l1399_139997


namespace NUMINAMATH_CALUDE_rectangle_area_l1399_139951

-- Define the points
variable (P Q R S T U : Point)

-- Define the rectangle PQRS
def is_rectangle (P Q R S : Point) : Prop := sorry

-- Define the trisection of angle S
def trisects_angle (S T U : Point) : Prop := sorry

-- Define that T is on PQ
def point_on_line (T P Q : Point) : Prop := sorry

-- Define that U is on PS
def point_on_line_2 (U P S : Point) : Prop := sorry

-- Define the length of QT
def length_QT (Q T : Point) : ℝ := sorry

-- Define the length of PU
def length_PU (P U : Point) : ℝ := sorry

-- Define the area of a rectangle
def area_rectangle (P Q R S : Point) : ℝ := sorry

-- Theorem statement
theorem rectangle_area (P Q R S T U : Point) 
  (h1 : is_rectangle P Q R S)
  (h2 : trisects_angle S T U)
  (h3 : point_on_line T P Q)
  (h4 : point_on_line_2 U P S)
  (h5 : length_QT Q T = 8)
  (h6 : length_PU P U = 4) :
  area_rectangle P Q R S = 64 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1399_139951


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1399_139916

-- Define the equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (6 - m) = 1 ∧ 
  m - 2 > 0 ∧ 6 - m > 0 ∧ m - 2 ≠ 6 - m

-- Define the condition
def condition (m : ℝ) : Prop :=
  2 < m ∧ m < 6

-- Theorem stating that the condition is necessary but not sufficient
theorem condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → condition m) ∧
  ¬(∀ m : ℝ, condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1399_139916


namespace NUMINAMATH_CALUDE_jakes_peaches_l1399_139952

theorem jakes_peaches (steven jill jake : ℕ) : 
  steven = 16 →
  jake < steven →
  jake = jill + 9 →
  ∃ (l : ℕ), jake = l + 9 :=
by sorry

end NUMINAMATH_CALUDE_jakes_peaches_l1399_139952


namespace NUMINAMATH_CALUDE_pencil_price_l1399_139901

theorem pencil_price (total_pencils : ℕ) (total_cost : ℚ) (h1 : total_pencils = 10) (h2 : total_cost = 2) :
  total_cost / total_pencils = 1/5 := by
sorry

end NUMINAMATH_CALUDE_pencil_price_l1399_139901


namespace NUMINAMATH_CALUDE_willies_stickers_l1399_139999

theorem willies_stickers (starting_stickers : Real) (received_stickers : Real) :
  starting_stickers = 36.0 →
  received_stickers = 7.0 →
  starting_stickers + received_stickers = 43.0 := by
  sorry

end NUMINAMATH_CALUDE_willies_stickers_l1399_139999


namespace NUMINAMATH_CALUDE_car_demand_and_profit_l1399_139924

-- Define the total demand function
def R (x : ℕ) : ℚ := (1/2) * x * (x + 1) * (39 - 2*x)

-- Define the purchase price function
def W (x : ℕ) : ℚ := 150000 + 2000*x

-- Define the constraints
def valid_x (x : ℕ) : Prop := x > 0 ∧ x ≤ 6

-- Define the demand function
def g (x : ℕ) : ℚ := -3*x^2 + 40*x

-- Define the monthly profit function
def f (x : ℕ) : ℚ := (185000 - W x) * g x

theorem car_demand_and_profit 
  (h : ∀ x, valid_x x → R x - R (x-1) = g x) :
  (∀ x, valid_x x → g x = -3*x^2 + 40*x) ∧ 
  (∀ x, valid_x x → f x ≤ f 5) ∧
  (f 5 = 3125000) := by
  sorry


end NUMINAMATH_CALUDE_car_demand_and_profit_l1399_139924


namespace NUMINAMATH_CALUDE_acute_triangle_inequality_l1399_139906

theorem acute_triangle_inequality (A B C : Real) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hABC : A + B + C = π) (hAcute : A < π/2 ∧ B < π/2 ∧ C < π/2) :
  (Real.sin A + Real.sin B + Real.sin C) * (1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C)
  ≤ π * (1 / A + 1 / B + 1 / C) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_inequality_l1399_139906


namespace NUMINAMATH_CALUDE_apple_eating_contest_difference_l1399_139904

/-- Represents the result of an apple eating contest -/
structure ContestResult where
  numStudents : Nat
  applesCounts : List Nat
  maxEater : Nat
  minEater : Nat

/-- Theorem stating the difference between the maximum and minimum number of apples eaten -/
theorem apple_eating_contest_difference (result : ContestResult)
  (h1 : result.numStudents = 8)
  (h2 : result.applesCounts.length = result.numStudents)
  (h3 : result.maxEater ∈ result.applesCounts)
  (h4 : result.minEater ∈ result.applesCounts)
  (h5 : ∀ x ∈ result.applesCounts, x ≤ result.maxEater ∧ x ≥ result.minEater) :
  result.maxEater - result.minEater = 8 :=
by sorry

end NUMINAMATH_CALUDE_apple_eating_contest_difference_l1399_139904


namespace NUMINAMATH_CALUDE_full_price_revenue_is_3000_total_tickets_correct_total_revenue_correct_l1399_139933

/-- Represents the revenue from full-price tickets at a concert venue. -/
def revenue_full_price (total_tickets : ℕ) (total_revenue : ℕ) : ℕ :=
  let p : ℕ := 20  -- Price of a full-price ticket
  let f : ℕ := 150 -- Number of full-price tickets
  f * p

/-- Theorem stating that the revenue from full-price tickets is $3000. -/
theorem full_price_revenue_is_3000 :
  revenue_full_price 250 3500 = 3000 := by
  sorry

/-- Verifies that the total number of tickets is correct. -/
theorem total_tickets_correct (f h q : ℕ) :
  f + h + q = 250 := by
  sorry

/-- Verifies that the total revenue is correct. -/
theorem total_revenue_correct (f h q p : ℕ) :
  f * p + h * (p / 2) + q * (p / 4) = 3500 := by
  sorry

end NUMINAMATH_CALUDE_full_price_revenue_is_3000_total_tickets_correct_total_revenue_correct_l1399_139933


namespace NUMINAMATH_CALUDE_f_f_neg_two_equals_one_l1399_139995

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 2
  else if x < 3 then 2^x
  else Real.log x

-- State the theorem
theorem f_f_neg_two_equals_one : f (f (-2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_f_neg_two_equals_one_l1399_139995


namespace NUMINAMATH_CALUDE_penny_money_left_l1399_139963

/-- Calculates the amount of money Penny has left after purchasing socks and a hat. -/
def money_left (initial_amount : ℝ) (num_sock_pairs : ℕ) (sock_pair_cost : ℝ) (hat_cost : ℝ) : ℝ :=
  initial_amount - (num_sock_pairs * sock_pair_cost + hat_cost)

/-- Proves that Penny has $5 left after her purchases. -/
theorem penny_money_left :
  money_left 20 4 2 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_penny_money_left_l1399_139963


namespace NUMINAMATH_CALUDE_geometric_propositions_l1399_139985

/-- Two lines in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne : point1 ≠ point2

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ
  nonzero : normal ≠ (0, 0, 0)

/-- Perpendicularity between lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Perpendicularity between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Parallelism between lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallelism between planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem geometric_propositions :
  ∃ (l1 l2 l3 : Line3D) (p1 p2 p3 : Plane3D),
    (¬∀ l1 l2 l3, perpendicular_lines l1 l3 → perpendicular_lines l2 l3 → parallel_lines l1 l2) ∧
    (∀ l1 l2 p, perpendicular_line_plane l1 p → perpendicular_line_plane l2 p → parallel_lines l1 l2) ∧
    (∀ p1 p2 l, perpendicular_line_plane l p1 → perpendicular_line_plane l p2 → parallel_planes p1 p2) ∧
    (¬∀ p1 p2 p3, perpendicular_planes p1 p3 → perpendicular_planes p2 p3 → perpendicular_planes p1 p2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_propositions_l1399_139985


namespace NUMINAMATH_CALUDE_percentage_problem_l1399_139957

theorem percentage_problem (N : ℝ) (P : ℝ) 
  (h1 : 0.3 * N = 120) 
  (h2 : (P / 100) * N = 160) : 
  P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1399_139957


namespace NUMINAMATH_CALUDE_arithmetic_mean_root_mean_square_inequality_l1399_139907

theorem arithmetic_mean_root_mean_square_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) ∧
  ((a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_root_mean_square_inequality_l1399_139907


namespace NUMINAMATH_CALUDE_notebook_pen_cost_l1399_139940

theorem notebook_pen_cost :
  ∀ (n p : ℕ),
  15 * n + 4 * p = 160 →
  n > p →
  n + p = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_pen_cost_l1399_139940


namespace NUMINAMATH_CALUDE_four_digit_no_repeat_count_five_digit_no_repeat_div_by_5_count_l1399_139937

/-- The count of four-digit numbers with no repeated digits -/
def fourDigitNoRepeat : Nat :=
  5 * 4 * 3 * 2

/-- The count of five-digit numbers with no repeated digits and divisible by 5 -/
def fiveDigitNoRepeatDivBy5 : Nat :=
  2 * (4 * 4 * 3 * 2)

theorem four_digit_no_repeat_count :
  fourDigitNoRepeat = 120 := by sorry

theorem five_digit_no_repeat_div_by_5_count :
  fiveDigitNoRepeatDivBy5 = 216 := by sorry

end NUMINAMATH_CALUDE_four_digit_no_repeat_count_five_digit_no_repeat_div_by_5_count_l1399_139937


namespace NUMINAMATH_CALUDE_odd_function_extension_l1399_139993

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x < 0, f x = 1 - Real.exp (-x + 1)) :
  ∀ x > 0, f x = Real.exp (x + 1) - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1399_139993


namespace NUMINAMATH_CALUDE_mascot_costs_l1399_139928

/-- The cost of an Auspicious Mascot Plush Toy -/
def plush_toy_cost : ℝ := 80

/-- The cost of an Auspicious Mascot Metal Ornament -/
def metal_ornament_cost : ℝ := 100

/-- The total cost of Plush Toys purchased -/
def plush_toy_total : ℝ := 6400

/-- The total cost of Metal Ornaments purchased -/
def metal_ornament_total : ℝ := 4000

theorem mascot_costs :
  (metal_ornament_cost = plush_toy_cost + 20) ∧
  (plush_toy_total / plush_toy_cost = 2 * (metal_ornament_total / metal_ornament_cost)) ∧
  (plush_toy_cost = 80) ∧
  (metal_ornament_cost = 100) := by
  sorry

end NUMINAMATH_CALUDE_mascot_costs_l1399_139928


namespace NUMINAMATH_CALUDE_john_percentage_increase_l1399_139980

/-- Represents the data for John's work at two hospitals --/
structure HospitalData where
  patients_first_hospital : ℕ  -- patients per day at first hospital
  total_patients_per_year : ℕ  -- total patients per year
  days_per_week : ℕ           -- working days per week
  weeks_per_year : ℕ          -- working weeks per year

/-- Calculates the percentage increase in patients at the second hospital compared to the first --/
def percentage_increase (data : HospitalData) : ℚ :=
  let total_working_days := data.days_per_week * data.weeks_per_year
  let patients_second_hospital := (data.total_patients_per_year - data.patients_first_hospital * total_working_days) / total_working_days
  ((patients_second_hospital - data.patients_first_hospital) / data.patients_first_hospital) * 100

/-- Theorem stating that given John's work conditions, the percentage increase is 20% --/
theorem john_percentage_increase :
  let john_data : HospitalData := {
    patients_first_hospital := 20,
    total_patients_per_year := 11000,
    days_per_week := 5,
    weeks_per_year := 50
  }
  percentage_increase john_data = 20 := by
  sorry


end NUMINAMATH_CALUDE_john_percentage_increase_l1399_139980


namespace NUMINAMATH_CALUDE_inequality_solution_l1399_139909

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ 
  ((1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) ∨ (8 < x ∧ x < 10)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1399_139909


namespace NUMINAMATH_CALUDE_black_faces_alignment_l1399_139973

/-- Represents a cube with one black face and five white faces -/
structure Cube where
  blackFace : Fin 6

/-- Represents the 8x8 grid of cubes -/
def Grid := Fin 8 → Fin 8 → Cube

/-- Rotates all cubes in a given row -/
def rotateRow (g : Grid) (row : Fin 8) : Grid :=
  sorry

/-- Rotates all cubes in a given column -/
def rotateColumn (g : Grid) (col : Fin 8) : Grid :=
  sorry

/-- Checks if all cubes have their black faces pointing in the same direction -/
def allFacingSameDirection (g : Grid) : Prop :=
  sorry

/-- The main theorem stating that it's always possible to make all black faces point in the same direction -/
theorem black_faces_alignment (g : Grid) :
  ∃ (ops : List (Sum (Fin 8) (Fin 8))), 
    let finalGrid := ops.foldl (λ acc op => match op with
      | Sum.inl row => rotateRow acc row
      | Sum.inr col => rotateColumn acc col) g
    allFacingSameDirection finalGrid :=
  sorry

end NUMINAMATH_CALUDE_black_faces_alignment_l1399_139973


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l1399_139955

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l1399_139955


namespace NUMINAMATH_CALUDE_map_scale_l1399_139902

/-- Given a map scale where 15 cm represents 90 km, 
    prove that 20 cm on the map represents 120 km in reality. -/
theorem map_scale (scale : ℝ → ℝ) 
  (h1 : scale 15 = 90) -- 15 cm on map represents 90 km in reality
  (h2 : ∀ x : ℝ, scale x = (x / 15) * 90) -- scale is linear
  : scale 20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l1399_139902


namespace NUMINAMATH_CALUDE_sector_perimeter_example_l1399_139936

/-- The perimeter of a sector with given radius and central angle -/
def sector_perimeter (r : ℝ) (θ : ℝ) : ℝ := r * θ + 2 * r

/-- Theorem: The perimeter of a sector with radius 1.5 and central angle 2 radians is 6 -/
theorem sector_perimeter_example : sector_perimeter 1.5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_example_l1399_139936


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1399_139982

theorem polynomial_evaluation (x : ℤ) (h : x = -2) : x^3 - x^2 + x - 1 = -15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1399_139982


namespace NUMINAMATH_CALUDE_surface_dots_eq_105_l1399_139923

/-- Represents a standard die -/
structure Die where
  faces : Fin 6 → Nat
  sum_21 : (faces 0) + (faces 1) + (faces 2) + (faces 3) + (faces 4) + (faces 5) = 21

/-- Represents the solid made of glued dice -/
structure DiceSolid where
  dice : Fin 7 → Die
  glued_faces_same : ∀ (i j : Fin 7) (f1 f2 : Fin 6), 
    (dice i).faces f1 = (dice j).faces f2 → i ≠ j

def surface_dots (solid : DiceSolid) : Nat :=
  sorry

theorem surface_dots_eq_105 (solid : DiceSolid) : 
  surface_dots solid = 105 := by
  sorry

end NUMINAMATH_CALUDE_surface_dots_eq_105_l1399_139923


namespace NUMINAMATH_CALUDE_base_conversion_645_to_base_5_l1399_139964

theorem base_conversion_645_to_base_5 :
  (1 * 5^4 + 0 * 5^3 + 4 * 5^1 + 0 * 5^0 : ℕ) = 645 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_645_to_base_5_l1399_139964


namespace NUMINAMATH_CALUDE_probability_of_all_even_sums_l1399_139958

/-- Represents a tile with a number from 1 to 9 -/
def Tile := Fin 9

/-- Represents a player's selection of three tiles -/
def Selection := Fin 3 → Tile

/-- The set of all possible selections -/
def AllSelections := Fin 3 → Selection

/-- Checks if a selection results in an even sum -/
def isEvenSum (s : Selection) : Prop :=
  ∃ (a b c : Tile), s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ (a.val + b.val + c.val) % 2 = 0

/-- Checks if all three players have even sums -/
def allEvenSums (selections : AllSelections) : Prop :=
  ∀ i : Fin 3, isEvenSum (selections i)

/-- The total number of possible ways to distribute the tiles -/
def totalOutcomes : ℕ := 1680

/-- The number of favorable outcomes where all players have even sums -/
def favorableOutcomes : ℕ := 400

theorem probability_of_all_even_sums :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 21 := by sorry

end NUMINAMATH_CALUDE_probability_of_all_even_sums_l1399_139958


namespace NUMINAMATH_CALUDE_valid_hexagonal_star_exists_l1399_139926

/-- A configuration of numbers in the hexagonal star format -/
structure HexagonalStar where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  g : Nat

/-- Check if a number is 2, 3, or 5 -/
def isValidNumber (n : Nat) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

/-- Check if all numbers in the configuration are valid -/
def allValidNumbers (hs : HexagonalStar) : Prop :=
  isValidNumber hs.a ∧ isValidNumber hs.b ∧ isValidNumber hs.c ∧
  isValidNumber hs.d ∧ isValidNumber hs.e ∧ isValidNumber hs.f ∧
  isValidNumber hs.g

/-- Check if all triangles have the same sum -/
def allTrianglesSameSum (hs : HexagonalStar) : Prop :=
  let sum1 := hs.a + hs.b + hs.g
  let sum2 := hs.b + hs.c + hs.g
  let sum3 := hs.c + hs.d + hs.g
  let sum4 := hs.d + hs.e + hs.g
  let sum5 := hs.e + hs.f + hs.g
  let sum6 := hs.f + hs.a + hs.g
  sum1 = sum2 ∧ sum2 = sum3 ∧ sum3 = sum4 ∧ sum4 = sum5 ∧ sum5 = sum6

/-- Theorem: There exists a valid hexagonal star configuration -/
theorem valid_hexagonal_star_exists : ∃ (hs : HexagonalStar), 
  allValidNumbers hs ∧ allTrianglesSameSum hs := by
  sorry

end NUMINAMATH_CALUDE_valid_hexagonal_star_exists_l1399_139926


namespace NUMINAMATH_CALUDE_remainder_theorem_l1399_139900

theorem remainder_theorem (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1399_139900


namespace NUMINAMATH_CALUDE_tan_1500_deg_l1399_139938

theorem tan_1500_deg (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1500 * π / 180) → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_tan_1500_deg_l1399_139938


namespace NUMINAMATH_CALUDE_unique_prime_seven_digit_number_l1399_139941

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def seven_digit_number (B : ℕ) : ℕ := 2024050 + B

theorem unique_prime_seven_digit_number :
  ∃! B : ℕ, B < 10 ∧ is_prime (seven_digit_number B) ∧ seven_digit_number B = 2024051 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_seven_digit_number_l1399_139941


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l1399_139917

theorem triangle_side_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + c^2) / b^2 ≥ 1 ∧ ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧ (a'^2 + c'^2) / b'^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l1399_139917


namespace NUMINAMATH_CALUDE_function_property_l1399_139969

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y)
  (h2 : f 1000 = 6) : 
  f 800 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_function_property_l1399_139969


namespace NUMINAMATH_CALUDE_cube_volume_proof_l1399_139949

theorem cube_volume_proof (a b c : ℝ) 
  (h1 : a^2 + b^2 = 81)
  (h2 : a^2 + c^2 = 169)
  (h3 : c^2 + b^2 = 196) :
  a * b * c = 18 * Real.sqrt 71 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_proof_l1399_139949


namespace NUMINAMATH_CALUDE_inequality_solution_l1399_139956

theorem inequality_solution (x : ℝ) : 
  x ≠ 0 → (x > (9 : ℝ) / x ↔ (x > -3 ∧ x < 0) ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1399_139956


namespace NUMINAMATH_CALUDE_selena_leftover_money_l1399_139943

def meal_cost (tip steak_price steak_count burger_price burger_count
               ice_cream_price ice_cream_count steak_tax burger_tax ice_cream_tax : ℝ) : ℝ :=
  let steak_total := steak_price * steak_count * (1 + steak_tax)
  let burger_total := burger_price * burger_count * (1 + burger_tax)
  let ice_cream_total := ice_cream_price * ice_cream_count * (1 + ice_cream_tax)
  steak_total + burger_total + ice_cream_total

theorem selena_leftover_money :
  let tip := 99
  let steak_price := 24
  let steak_count := 2
  let burger_price := 3.5
  let burger_count := 2
  let ice_cream_price := 2
  let ice_cream_count := 3
  let steak_tax := 0.07
  let burger_tax := 0.06
  let ice_cream_tax := 0.08
  tip - meal_cost tip steak_price steak_count burger_price burger_count
                  ice_cream_price ice_cream_count steak_tax burger_tax ice_cream_tax = 33.74 := by
  sorry

end NUMINAMATH_CALUDE_selena_leftover_money_l1399_139943


namespace NUMINAMATH_CALUDE_paper_folding_thickness_l1399_139984

/-- The thickness of a folded paper after a given number of folds. -/
def thickness (initial_thickness : ℝ) (num_folds : ℕ) : ℝ :=
  initial_thickness * (2 ^ num_folds)

/-- Theorem stating that folding a 0.1 mm thick paper 5 times results in 3.2 mm thickness. -/
theorem paper_folding_thickness :
  thickness 0.1 5 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_thickness_l1399_139984


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l1399_139954

/-- Triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The given conditions for the triangle -/
def satisfiesConditions (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.b + t.c - t.a) = 3 * t.b * t.c ∧
  Real.sin t.A = 2 * Real.sin t.B * Real.cos t.C

/-- Theorem: If a triangle satisfies the given conditions, then it is equilateral -/
theorem triangle_is_equilateral (t : Triangle) 
  (h : satisfiesConditions t) : t.isEquilateral := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l1399_139954


namespace NUMINAMATH_CALUDE_three_color_theorem_l1399_139987

/-- A complete graph with n vertices -/
def CompleteGraph (n : ℕ) := { v : Fin n // True }

/-- An edge in a complete graph connects any two distinct vertices -/
def Edge (n : ℕ) := { e : CompleteGraph n × CompleteGraph n // e.1 ≠ e.2 }

/-- A coloring assignment for vertices -/
def Coloring (n : ℕ) := CompleteGraph n → Fin 3

/-- A valid coloring ensures no two adjacent vertices have the same color -/
def ValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (e : Edge n), c e.1.1 ≠ c e.1.2

theorem three_color_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ (c : Coloring n), ValidColoring n c :=
sorry

end NUMINAMATH_CALUDE_three_color_theorem_l1399_139987


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_l1399_139939

/-- The cost of attractions at an amusement park. -/
structure AttractionCosts where
  roller_coaster : ℕ
  log_ride : ℕ
  ferris_wheel : ℕ

/-- The number of tickets Antonieta has and needs. -/
structure AntonietaTickets where
  current : ℕ
  needed : ℕ

/-- Theorem stating the cost of the Ferris wheel given the other costs and ticket information. -/
theorem ferris_wheel_cost (costs : AttractionCosts) (antonieta : AntonietaTickets) : 
  costs.roller_coaster = 5 →
  costs.log_ride = 7 →
  antonieta.current = 2 →
  antonieta.needed = 16 →
  costs.ferris_wheel = 6 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_l1399_139939


namespace NUMINAMATH_CALUDE_product_of_base8_digits_8675_l1399_139920

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 8675 (base 10) is 0 -/
theorem product_of_base8_digits_8675 :
  productOfList (toBase8 8675) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base8_digits_8675_l1399_139920


namespace NUMINAMATH_CALUDE_modulo_17_intercepts_l1399_139914

/-- Prove the x-intercept, y-intercept, and their sum for the equation 5x ≡ 3y - 1 (mod 17) -/
theorem modulo_17_intercepts :
  ∃ (x₀ y₀ : ℕ), 
    x₀ < 17 ∧ 
    y₀ < 17 ∧
    (5 * x₀) % 17 = 16 ∧ 
    (3 * y₀) % 17 = 1 ∧
    x₀ = 1 ∧ 
    y₀ = 6 ∧ 
    x₀ + y₀ = 7 :=
by sorry

end NUMINAMATH_CALUDE_modulo_17_intercepts_l1399_139914


namespace NUMINAMATH_CALUDE_ladder_problem_l1399_139998

theorem ladder_problem (ladder_length height : Real) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : Real), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1399_139998


namespace NUMINAMATH_CALUDE_area_of_region_l1399_139968

/-- The area enclosed by the region defined by x^2 + y^2 - 4x + 2y = -2 is 3π -/
theorem area_of_region (x y : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 2*y = -2) → 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    center = (2, -1) ∧ 
    r = Real.sqrt 3 ∧ 
    (∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2) ∧
    π * r^2 = 3 * π) :=
by sorry

end NUMINAMATH_CALUDE_area_of_region_l1399_139968


namespace NUMINAMATH_CALUDE_optimal_departure_time_l1399_139975

/-- The travel scenario for two people A and B -/
structure TravelScenario where
  distance : ℝ
  walking_speed : ℝ
  travel_speed : ℝ
  walking_distance : ℝ

/-- The theorem stating the optimal departure time -/
theorem optimal_departure_time (scenario : TravelScenario) 
  (h1 : scenario.distance = 15)
  (h2 : scenario.walking_speed = 1)
  (h3 : scenario.travel_speed > scenario.walking_speed)
  (h4 : scenario.walking_distance = 60 / 11)
  (h5 : scenario.travel_speed * (3 / 11) = scenario.distance - scenario.walking_distance) :
  3 / 11 = scenario.walking_distance / scenario.walking_speed := by
  sorry

#check optimal_departure_time

end NUMINAMATH_CALUDE_optimal_departure_time_l1399_139975


namespace NUMINAMATH_CALUDE_divisible_by_five_l1399_139961

theorem divisible_by_five (a b : ℕ) : 
  (∃ k : ℕ, a * b = 5 * k) → (∃ m : ℕ, a = 5 * m) ∨ (∃ n : ℕ, b = 5 * n) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_five_l1399_139961


namespace NUMINAMATH_CALUDE_second_number_is_30_l1399_139953

theorem second_number_is_30 (x y : ℤ) : 
  y = x + 4 →  -- The second number is 4 more than the first
  x + y = 56 → -- The sum of the two numbers is 56
  y = 30       -- The second number is 30
:= by sorry

end NUMINAMATH_CALUDE_second_number_is_30_l1399_139953


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1399_139946

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  2*x - (3*y - 5*x) + 7*y = 7*x + 4*y :=
by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  (-x^2 + 4*x) - 3*x^2 + 2*(2*x^2 - 3*x) = -2*x :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1399_139946


namespace NUMINAMATH_CALUDE_jan_skips_proof_l1399_139974

/-- Calculates the total number of skips in a given time period after doubling the initial speed -/
def total_skips (initial_speed : ℕ) (time : ℕ) : ℕ :=
  2 * initial_speed * time

/-- Proves that given an initial speed of 70 skips per minute, which doubles after training,
    the total number of skips in 5 minutes is equal to 700 -/
theorem jan_skips_proof :
  total_skips 70 5 = 700 := by
  sorry

#eval total_skips 70 5

end NUMINAMATH_CALUDE_jan_skips_proof_l1399_139974


namespace NUMINAMATH_CALUDE_factorial_product_less_than_factorial_sum_l1399_139959

theorem factorial_product_less_than_factorial_sum {n : ℕ} (k : ℕ) (a : Fin n → ℕ) 
  (h_pos : ∀ i, a i > 0) (h_sum : (Finset.univ.sum a) < k) : 
  (Finset.univ.prod (λ i => Nat.factorial (a i))) < Nat.factorial k := by
sorry

end NUMINAMATH_CALUDE_factorial_product_less_than_factorial_sum_l1399_139959


namespace NUMINAMATH_CALUDE_gcd_888_1147_l1399_139970

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_888_1147_l1399_139970


namespace NUMINAMATH_CALUDE_solution_set_f_positive_solution_set_f_leq_g_l1399_139925

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + (2 - m) * x - m

/-- The function g(x) defined in the problem -/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - x + 2 * m

/-- Theorem for part (1) of the problem -/
theorem solution_set_f_positive :
  {x : ℝ | f 1 x > 0} = {x : ℝ | x > 1/2 ∨ x < -1} := by sorry

/-- Theorem for part (2) of the problem -/
theorem solution_set_f_leq_g (m : ℝ) (h : m > 0) :
  {x : ℝ | f m x ≤ g m x} = {x : ℝ | -3 ≤ x ∧ x ≤ m} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_solution_set_f_leq_g_l1399_139925


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1399_139981

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - k*x₁ - 4 = 0 →
  x₂^2 - k*x₂ - 4 = 0 →
  x₁^2 + x₂^2 + x₁*x₂ = 6 →
  k^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1399_139981


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l1399_139918

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  /-- The semi-major axis length -/
  a : ℝ
  /-- The semi-minor axis length -/
  b : ℝ
  /-- The focal distance -/
  c : ℝ
  /-- Assumption that a > b > 0 -/
  h₁ : a > b ∧ b > 0
  /-- Relationship between a, b, and c -/
  h₂ : c^2 = a^2 - b^2

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.b^2 + y^2 / e.a^2 = 1

/-- Theorem stating the equation of the ellipse given the conditions -/
theorem ellipse_equation_from_conditions
  (e : Ellipse)
  (focus_on_y_axis : e.c = e.a * (1/2))
  (focal_length : 2 * e.c = 8) :
  ellipse_equation e = fun x y ↦ x^2 / 48 + y^2 / 64 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l1399_139918


namespace NUMINAMATH_CALUDE_complement_of_intersection_l1399_139983

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {2, 3, 4}

-- Theorem statement
theorem complement_of_intersection :
  (Aᶜ : Set ℕ) ∩ B = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l1399_139983


namespace NUMINAMATH_CALUDE_negation_of_implication_l1399_139921

theorem negation_of_implication (x y : ℝ) :
  ¬(x + y = 1 → x * y ≤ 1) ↔ (x + y = 1 → x * y > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1399_139921


namespace NUMINAMATH_CALUDE_distance_AB_is_abs_x_minus_one_l1399_139932

/-- Represents a point on a number line with a rational coordinate -/
structure Point where
  coord : ℚ

/-- Calculates the distance between two points on a number line -/
def distance (p q : Point) : ℚ :=
  |p.coord - q.coord|

theorem distance_AB_is_abs_x_minus_one (x : ℚ) :
  let A : Point := ⟨x⟩
  let B : Point := ⟨1⟩
  let C : Point := ⟨-1⟩
  distance A B = |x - 1| := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_abs_x_minus_one_l1399_139932


namespace NUMINAMATH_CALUDE_problem_solution_l1399_139913

/-- Represents the number of students in different groups -/
structure StudentGroups where
  total : ℕ
  chinese : ℕ
  math : ℕ
  both : ℕ

/-- Calculates the number of students in neither group -/
def studentsInNeither (g : StudentGroups) : ℕ :=
  g.total - (g.chinese + g.math - g.both)

/-- Theorem statement for the given problem -/
theorem problem_solution (g : StudentGroups) 
  (h1 : g.total = 50)
  (h2 : g.chinese = 15)
  (h3 : g.math = 20)
  (h4 : g.both = 8) :
  studentsInNeither g = 23 := by
  sorry

/-- Example usage of the theorem -/
example : studentsInNeither ⟨50, 15, 20, 8⟩ = 23 := by
  apply problem_solution ⟨50, 15, 20, 8⟩
  repeat' rfl


end NUMINAMATH_CALUDE_problem_solution_l1399_139913


namespace NUMINAMATH_CALUDE_jeff_payment_correct_l1399_139927

/-- The amount Jeff paid when picking up the Halloween costumes -/
def jeff_payment (num_costumes : ℕ) 
                 (deposit_rate : ℝ) 
                 (price_increase : ℝ) 
                 (last_year_price : ℝ) 
                 (jeff_discount : ℝ) 
                 (friend_discount : ℝ) : ℝ :=
  let this_year_price := last_year_price * (1 + price_increase)
  let total_cost := num_costumes * this_year_price
  let discounts := jeff_discount * this_year_price + friend_discount * this_year_price
  let adjusted_cost := total_cost - discounts
  let deposit := deposit_rate * adjusted_cost
  adjusted_cost - deposit

/-- Theorem stating that Jeff's payment matches the calculated amount -/
theorem jeff_payment_correct : 
  jeff_payment 3 0.1 0.4 250 0.15 0.1 = 866.25 := by
  sorry


end NUMINAMATH_CALUDE_jeff_payment_correct_l1399_139927


namespace NUMINAMATH_CALUDE_function_monotonic_decreasing_l1399_139929

/-- The function f(x) = 3x^2 - 2ln(x) is monotonically decreasing on the interval (0, √3/3) -/
theorem function_monotonic_decreasing (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * Real.log x
  0 < x → x < Real.sqrt 3 / 3 → StrictMonoOn f (Set.Ioo 0 (Real.sqrt 3 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_function_monotonic_decreasing_l1399_139929


namespace NUMINAMATH_CALUDE_nested_root_simplification_l1399_139977

theorem nested_root_simplification (b : ℝ) :
  (((b^16)^(1/8))^(1/4))^6 * (((b^16)^(1/4))^(1/8))^6 = b^6 := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l1399_139977


namespace NUMINAMATH_CALUDE_orange_harvest_days_l1399_139945

/-- The number of days required to harvest a given number of sacks of ripe oranges -/
def harvest_days (total_sacks : ℕ) (sacks_per_day : ℕ) : ℕ :=
  total_sacks / sacks_per_day

/-- Theorem stating that it takes 25 days to harvest 2050 sacks of ripe oranges when harvesting 82 sacks per day -/
theorem orange_harvest_days : harvest_days 2050 82 = 25 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_days_l1399_139945


namespace NUMINAMATH_CALUDE_correct_num_students_l1399_139989

/-- The number of students in the class -/
def num_students : ℕ := 20

/-- The cost of one pack of instant noodles in yuan -/
def noodle_cost : ℚ := 3.5

/-- The cost of one sausage in yuan -/
def sausage_cost : ℚ := 7.5

/-- The total amount spent in yuan -/
def total_spent : ℚ := 290

/-- Theorem stating that the number of students is correct given the problem conditions -/
theorem correct_num_students :
  (num_students : ℚ) * (2 * noodle_cost + sausage_cost) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_correct_num_students_l1399_139989


namespace NUMINAMATH_CALUDE_kata_friends_and_money_l1399_139947

/-- The number of friends Káta has -/
def n : ℕ := sorry

/-- The amount of money Káta has for gifts (in Kč) -/
def x : ℕ := sorry

/-- The cost of a hair clip (in Kč) -/
def hair_clip_cost : ℕ := 28

/-- The cost of a teddy bear (in Kč) -/
def teddy_bear_cost : ℕ := 42

/-- The amount left after buying hair clips (in Kč) -/
def hair_clip_remainder : ℕ := 29

/-- The amount short after buying teddy bears (in Kč) -/
def teddy_bear_shortage : ℕ := 13

theorem kata_friends_and_money :
  (n * hair_clip_cost + hair_clip_remainder = x) ∧
  (n * teddy_bear_cost - teddy_bear_shortage = x) →
  n = 3 ∧ x = 113 := by
  sorry

end NUMINAMATH_CALUDE_kata_friends_and_money_l1399_139947


namespace NUMINAMATH_CALUDE_minutes_on_eleventh_day_l1399_139996

/-- The number of minutes Gage skated each day for the first 6 days -/
def minutes_per_day_first_6 : ℕ := 80

/-- The number of minutes Gage skated each day for the next 4 days -/
def minutes_per_day_next_4 : ℕ := 95

/-- The total number of days Gage has skated -/
def total_days : ℕ := 10

/-- The desired average number of minutes per day -/
def desired_average : ℕ := 90

/-- The total number of days including the day in question -/
def total_days_with_extra : ℕ := total_days + 1

/-- Theorem stating the number of minutes Gage must skate on the eleventh day -/
theorem minutes_on_eleventh_day :
  (total_days_with_extra * desired_average) - (6 * minutes_per_day_first_6 + 4 * minutes_per_day_next_4) = 130 := by
  sorry

end NUMINAMATH_CALUDE_minutes_on_eleventh_day_l1399_139996


namespace NUMINAMATH_CALUDE_max_d_value_l1399_139966

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ 
  (5000000 + d * 100000 + 500000 + 2200 + 20 + e) % 44 = 0

theorem max_d_value :
  ∃ (d : ℕ), is_valid_number d 6 ∧
  ∀ (d' e : ℕ), is_valid_number d' e → d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l1399_139966


namespace NUMINAMATH_CALUDE_number_thought_of_l1399_139962

theorem number_thought_of (x : ℝ) : (x / 5 + 6 = 65) → x = 295 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l1399_139962


namespace NUMINAMATH_CALUDE_coin_regrouping_l1399_139919

/-- The total number of coins remains the same after regrouping -/
theorem coin_regrouping (x : ℕ) : 
  (12 + 17 + 23 + 8 : ℕ) = 60 ∧ 
  x > 0 ∧
  60 % x = 0 →
  60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_coin_regrouping_l1399_139919


namespace NUMINAMATH_CALUDE_cafe_chair_distribution_l1399_139991

/-- Given a cafe with indoor and outdoor tables, prove the number of chairs at each indoor table. -/
theorem cafe_chair_distribution (indoor_tables outdoor_tables : ℕ) 
  (chairs_per_outdoor_table : ℕ) (total_chairs : ℕ) :
  indoor_tables = 9 →
  outdoor_tables = 11 →
  chairs_per_outdoor_table = 3 →
  total_chairs = 123 →
  ∃ (chairs_per_indoor_table : ℕ),
    chairs_per_indoor_table = 10 ∧
    total_chairs = indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table :=
by
  sorry

end NUMINAMATH_CALUDE_cafe_chair_distribution_l1399_139991


namespace NUMINAMATH_CALUDE_trent_tadpoles_l1399_139976

theorem trent_tadpoles (caught : ℕ) (kept : ℕ) (released_percentage : ℚ) : 
  released_percentage = 75 / 100 →
  kept = 45 →
  kept = (1 - released_percentage) * caught →
  caught = 180 :=
by sorry

end NUMINAMATH_CALUDE_trent_tadpoles_l1399_139976


namespace NUMINAMATH_CALUDE_fiftieth_term_is_198_l1399_139994

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
theorem fiftieth_term_is_198 :
  arithmeticSequenceTerm 2 4 50 = 198 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_198_l1399_139994


namespace NUMINAMATH_CALUDE_license_plate_equality_l1399_139978

def florida_plates : ℕ := 26^2 * 10^3 * 26^1
def north_dakota_plates : ℕ := 26^3 * 10^3

theorem license_plate_equality :
  florida_plates = north_dakota_plates :=
by sorry

end NUMINAMATH_CALUDE_license_plate_equality_l1399_139978


namespace NUMINAMATH_CALUDE_parabola_equation_l1399_139967

/-- A parabola with focus F and a point M satisfying given conditions -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  M : ℝ × ℝ
  h_p_pos : p > 0
  h_F_focus : F = (p/2, 0)
  h_M_on_C : M.2^2 = 2*p*M.1
  h_M_x : M.1 = 4
  h_MF_dist : Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 5

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (C : Parabola) : C.M.2^2 = 18 * C.M.1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1399_139967


namespace NUMINAMATH_CALUDE_divisibility_by_square_of_n_minus_one_l1399_139960

theorem divisibility_by_square_of_n_minus_one (n : ℕ) (h : n > 2) :
  ∃ k : ℤ, (n : ℤ)^(n - 1) - 1 = k * (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_square_of_n_minus_one_l1399_139960


namespace NUMINAMATH_CALUDE_picture_placement_l1399_139908

theorem picture_placement (wall_width picture_width : ℝ) 
  (hw : wall_width = 19) 
  (hp : picture_width = 3) : 
  (wall_width - picture_width) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l1399_139908


namespace NUMINAMATH_CALUDE_equation_solutions_l1399_139931

def f (x : ℝ) : ℝ := (x - 4)^6 + (x - 6)^6

theorem equation_solutions :
  {x : ℝ | f x = 64} = {4, 6} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1399_139931


namespace NUMINAMATH_CALUDE_negation_of_implication_l1399_139934

theorem negation_of_implication (p q : Prop) : ¬(p → q) ↔ (p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1399_139934


namespace NUMINAMATH_CALUDE_pens_distribution_l1399_139971

def number_of_friends (kendra_packs tony_packs pens_per_pack pens_kept_per_person : ℕ) : ℕ :=
  (kendra_packs + tony_packs) * pens_per_pack - 2 * pens_kept_per_person

theorem pens_distribution (kendra_packs tony_packs pens_per_pack pens_kept_per_person : ℕ) 
  (h1 : kendra_packs = 4)
  (h2 : tony_packs = 2)
  (h3 : pens_per_pack = 3)
  (h4 : pens_kept_per_person = 2) :
  number_of_friends kendra_packs tony_packs pens_per_pack pens_kept_per_person = 14 := by
  sorry

end NUMINAMATH_CALUDE_pens_distribution_l1399_139971


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1399_139948

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 16 = -6 ∧ a 2 * a 16 = 2) →
  (a 2 * a 16) / a 9 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1399_139948


namespace NUMINAMATH_CALUDE_point_on_line_iff_vector_sum_l1399_139992

/-- 
Given distinct points A and B, and a point O not on line AB,
a point C is on line AB if and only if there exist real numbers x and y 
such that OC = x * OA + y * OB and x + y = 1.
-/
theorem point_on_line_iff_vector_sum (O A B C : EuclideanSpace ℝ (Fin 3)) 
  (hAB : A ≠ B) (hO : O ∉ affineSpan ℝ {A, B}) : 
  C ∈ affineSpan ℝ {A, B} ↔ 
  ∃ (x y : ℝ), C - O = x • (A - O) + y • (B - O) ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_iff_vector_sum_l1399_139992


namespace NUMINAMATH_CALUDE_placemat_length_l1399_139950

theorem placemat_length (R : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) :
  R = 5 →
  n = 8 →
  w = 2 →
  y = 2 * R * Real.sin (π / (2 * n)) →
  y = 5 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_placemat_length_l1399_139950


namespace NUMINAMATH_CALUDE_horner_v3_equals_16_l1399_139903

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 7x^7 + 5x^5 + 4x^4 + 2x^2 + x + 2 -/
def f : List ℝ := [7, 0, 5, 4, 2, 1, 2]

/-- v_3 is the fourth intermediate value in Horner's method -/
def v_3 (coeffs : List ℝ) (x : ℝ) : ℝ :=
  (coeffs.take 4).foldl (fun acc a => acc * x + a) 0

theorem horner_v3_equals_16 :
  v_3 f 1 = 16 := by sorry

end NUMINAMATH_CALUDE_horner_v3_equals_16_l1399_139903


namespace NUMINAMATH_CALUDE_danielles_rooms_l1399_139942

theorem danielles_rooms (heidi grant danielle : ℕ) : 
  heidi = 3 * danielle →
  grant = heidi / 9 →
  grant = 2 →
  danielle = 6 := by
sorry

end NUMINAMATH_CALUDE_danielles_rooms_l1399_139942


namespace NUMINAMATH_CALUDE_point_on_graph_l1399_139915

/-- The linear function f(x) = 3x + 1 -/
def f (x : ℝ) : ℝ := 3 * x + 1

/-- The point (2, 7) -/
def point : ℝ × ℝ := (2, 7)

/-- Theorem: The point (2, 7) lies on the graph of f(x) = 3x + 1 -/
theorem point_on_graph : f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l1399_139915


namespace NUMINAMATH_CALUDE_private_schools_in_district_a_l1399_139944

/-- Represents the number of schools in Veenapaniville -/
def total_schools : Nat := 50

/-- Represents the number of public schools -/
def public_schools : Nat := 25

/-- Represents the number of parochial schools -/
def parochial_schools : Nat := 16

/-- Represents the number of private independent schools -/
def private_schools : Nat := 9

/-- Represents the number of schools in District A -/
def district_a_schools : Nat := 18

/-- Represents the number of schools in District B -/
def district_b_schools : Nat := 17

/-- Represents the number of private independent schools in District B -/
def district_b_private_schools : Nat := 2

/-- Theorem stating that the number of private independent schools in District A is 2 -/
theorem private_schools_in_district_a :
  total_schools = public_schools + parochial_schools + private_schools →
  district_a_schools + district_b_schools < total_schools →
  (total_schools - district_a_schools - district_b_schools) % 3 = 0 →
  private_schools - district_b_private_schools - (total_schools - district_a_schools - district_b_schools) / 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_private_schools_in_district_a_l1399_139944


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1399_139911

theorem right_triangle_third_side : ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  ((a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4)) →
  c = 5 ∨ c = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1399_139911


namespace NUMINAMATH_CALUDE_part_one_part_two_l1399_139972

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 4|

-- Part I
theorem part_one : 
  ∀ x : ℝ, (f 1 x ≤ 2 * |x - 4|) ↔ (x < 1.5) := by sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f a x ≥ 3) ↔ (a ≤ -7 ∨ a ≥ -1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1399_139972


namespace NUMINAMATH_CALUDE_problem_statement_l1399_139935

theorem problem_statement (s x y : ℝ) 
  (h1 : s > 0) 
  (h2 : s ≠ 1) 
  (h3 : x * y ≠ 0) 
  (h4 : s * x > y) : 
  s < y / x := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1399_139935


namespace NUMINAMATH_CALUDE_two_by_one_prism_net_removable_squares_l1399_139988

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of a rectangular prism --/
structure PrismNet where
  squares : ℕ

/-- Function to create a net from a rectangular prism --/
def createNet (prism : RectangularPrism) : PrismNet :=
  { squares := 2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height) }

/-- Function to count removable squares in a net --/
def countRemovableSquares (net : PrismNet) : ℕ := sorry

/-- Theorem stating that a 2×1×1 prism net has exactly 5 removable squares --/
theorem two_by_one_prism_net_removable_squares :
  let prism : RectangularPrism := { length := 2, width := 1, height := 1 }
  let net := createNet prism
  countRemovableSquares net = 5 ∧ net.squares - 1 = 9 := by sorry

end NUMINAMATH_CALUDE_two_by_one_prism_net_removable_squares_l1399_139988


namespace NUMINAMATH_CALUDE_cone_lateral_area_l1399_139930

/-- The lateral area of a cone with a central angle of 60° and base radius of 8 is 384π. -/
theorem cone_lateral_area : 
  ∀ (r : ℝ) (central_angle : ℝ) (lateral_area : ℝ),
  r = 8 →
  central_angle = 60 * π / 180 →
  lateral_area = π * r * (2 * π * r) / central_angle →
  lateral_area = 384 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l1399_139930


namespace NUMINAMATH_CALUDE_valid_seats_29x29_l1399_139912

/-- Represents a grid of seats -/
def Grid (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if two positions in the grid are adjacent -/
def adjacent (n : ℕ) (p q : Fin n × Fin n) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Counts the number of valid seats in an n x n grid -/
def validSeats (n : ℕ) : ℕ := sorry

/-- The main theorem to be proved -/
theorem valid_seats_29x29 :
  validSeats 29 = 421 :=
sorry

end NUMINAMATH_CALUDE_valid_seats_29x29_l1399_139912


namespace NUMINAMATH_CALUDE_bus_trip_speed_l1399_139979

/-- The average speed of a bus trip, given the conditions of the problem -/
def average_speed : ℝ → Prop :=
  fun v =>
    v > 0 ∧
    560 / v - 560 / (v + 10) = 2

theorem bus_trip_speed : ∃ v : ℝ, average_speed v ∧ v = 50 :=
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l1399_139979


namespace NUMINAMATH_CALUDE_charlie_flutes_l1399_139922

theorem charlie_flutes (charlie_flutes : ℕ) (charlie_horns : ℕ) (charlie_harps : ℕ) 
  (carli_flutes : ℕ) (carli_horns : ℕ) (carli_harps : ℕ) : 
  charlie_horns = 2 →
  charlie_harps = 1 →
  carli_flutes = 2 * charlie_flutes →
  carli_horns = charlie_horns / 2 →
  carli_harps = 0 →
  charlie_flutes + charlie_horns + charlie_harps + carli_flutes + carli_horns + carli_harps = 7 →
  charlie_flutes = 1 := by
sorry

end NUMINAMATH_CALUDE_charlie_flutes_l1399_139922
