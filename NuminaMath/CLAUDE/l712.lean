import Mathlib

namespace alex_money_left_l712_71203

/-- Calculates the amount of money Alex has left after deductions --/
theorem alex_money_left (weekly_income : ℕ) (tax_rate : ℚ) (water_bill : ℕ) (tithe_rate : ℚ) : 
  weekly_income = 500 →
  tax_rate = 1/10 →
  water_bill = 55 →
  tithe_rate = 1/10 →
  ↑weekly_income - (↑weekly_income * tax_rate + ↑water_bill + ↑weekly_income * tithe_rate) = 345 := by
sorry

end alex_money_left_l712_71203


namespace frog_jumps_l712_71281

-- Define the hexagon vertices
inductive Vertex : Type
| A | B | C | D | E | F

-- Define the neighbor relation
def isNeighbor : Vertex → Vertex → Prop :=
  sorry

-- Define the number of paths from A to C in n jumps
def numPaths (n : ℕ) : ℕ :=
  if n % 2 = 0 then (1/3) * (4^(n/2) - 1) else 0

-- Define the number of paths from A to C in n jumps avoiding D
def numPathsAvoidD (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3^(n/2 - 1) else 0

-- Define the survival probability after n jumps with mine at D
def survivalProb (n : ℕ) : ℚ :=
  if n % 2 = 0 then (3/4)^(n/2 - 1) else (3/4)^((n-1)/2)

-- Define the expected lifespan
def expectedLifespan : ℚ := 9

-- Main theorem
theorem frog_jumps :
  (∀ n : ℕ, numPaths n = if n % 2 = 0 then (1/3) * (4^(n/2) - 1) else 0) ∧
  (∀ n : ℕ, numPathsAvoidD n = if n % 2 = 0 then 3^(n/2 - 1) else 0) ∧
  (∀ n : ℕ, survivalProb n = if n % 2 = 0 then (3/4)^(n/2 - 1) else (3/4)^((n-1)/2)) ∧
  expectedLifespan = 9 :=
sorry

end frog_jumps_l712_71281


namespace smallest_integer_above_sqrt_sum_power_l712_71272

theorem smallest_integer_above_sqrt_sum_power : 
  ∃ n : ℕ, n = 3742 ∧ (∀ m : ℕ, m < n → (m : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^6) ∧
  n > (Real.sqrt 5 + Real.sqrt 3)^6 := by
  sorry

end smallest_integer_above_sqrt_sum_power_l712_71272


namespace range_of_a_l712_71278

def set_A : Set ℝ := {x : ℝ | (3 * x) / (x + 1) ≤ 2}

def set_B (a : ℝ) : Set ℝ := {x : ℝ | a - 2 < x ∧ x < 2 * a + 1}

theorem range_of_a (a : ℝ) :
  set_A = set_B a → a ∈ Set.Ioo (1/2 : ℝ) 1 := by
  sorry

end range_of_a_l712_71278


namespace smallest_number_divisible_by_all_l712_71223

def is_divisible_by_all (n : ℕ) (divisors : List ℕ) : Prop :=
  ∀ d ∈ divisors, (n + 3) % d = 0

theorem smallest_number_divisible_by_all : 
  ∀ n : ℕ, n < 6303 → ¬(is_divisible_by_all n [70, 100, 84]) ∧ 
  is_divisible_by_all 6303 [70, 100, 84] :=
sorry

end smallest_number_divisible_by_all_l712_71223


namespace triangle_inequality_l712_71211

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l712_71211


namespace ab_negative_sufficient_not_necessary_for_hyperbola_l712_71238

/-- A conic section represented by the equation ax^2 + by^2 = c -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to determine if a conic section is a hyperbola -/
def is_hyperbola (conic : ConicSection) : Prop :=
  sorry -- Definition of hyperbola

/-- Theorem stating that ab < 0 is sufficient but not necessary for a hyperbola -/
theorem ab_negative_sufficient_not_necessary_for_hyperbola :
  ∀ (conic : ConicSection),
    (∀ (conic : ConicSection), conic.a * conic.b < 0 → is_hyperbola conic) ∧
    ¬(∀ (conic : ConicSection), is_hyperbola conic → conic.a * conic.b < 0) :=
by
  sorry


end ab_negative_sufficient_not_necessary_for_hyperbola_l712_71238


namespace max_intersected_cells_8x8_l712_71208

/-- Represents a chessboard with a given number of rows and columns. -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a straight line on a chessboard. -/
structure StraightLine

/-- The number of cells intersected by a straight line on a chessboard. -/
def intersectedCells (board : Chessboard) (line : StraightLine) : Nat :=
  sorry

/-- The maximum number of cells that can be intersected by any straight line on a given chessboard. -/
def maxIntersectedCells (board : Chessboard) : Nat :=
  sorry

/-- Theorem stating that the maximum number of cells intersected by a straight line on an 8x8 chessboard is 15. -/
theorem max_intersected_cells_8x8 :
  maxIntersectedCells (Chessboard.mk 8 8) = 15 :=
by sorry

end max_intersected_cells_8x8_l712_71208


namespace point_on_line_l712_71222

/-- Given three points in a 2D plane, this function checks if they are collinear --/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem point_on_line : are_collinear (0, 4) (-6, 1) (4, 6) := by sorry

end point_on_line_l712_71222


namespace set_intersection_equality_l712_71273

def A : Set ℝ := {x | -5 < x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -7 < x ∧ x < a}
def C (b : ℝ) : Set ℝ := {x | b < x ∧ x < 2}

theorem set_intersection_equality (a b : ℝ) :
  A ∩ B a = C b → a + b = -3 := by
  sorry

end set_intersection_equality_l712_71273


namespace pencil_cost_l712_71201

/-- The cost of a pencil given the total cost with an eraser and the price difference -/
theorem pencil_cost (total : ℝ) (difference : ℝ) (h1 : total = 3.4) (h2 : difference = 3) :
  ∃ (pencil eraser : ℝ),
    pencil + eraser = total ∧
    pencil = eraser + difference ∧
    pencil = 3.2 := by
  sorry

end pencil_cost_l712_71201


namespace b_lending_rate_to_c_l712_71240

/-- Given the lending scenario between A, B, and C, prove that B's lending rate to C is 12.5% --/
theorem b_lending_rate_to_c (principal : ℝ) (a_rate : ℝ) (b_gain : ℝ) (time : ℝ) :
  principal = 3150 →
  a_rate = 8 →
  b_gain = 283.5 →
  time = 2 →
  ∃ (b_rate : ℝ),
    b_rate = 12.5 ∧
    b_gain = (principal * b_rate / 100 * time) - (principal * a_rate / 100 * time) :=
by sorry

end b_lending_rate_to_c_l712_71240


namespace initial_number_of_students_l712_71217

theorem initial_number_of_students :
  ∀ (n : ℕ) (W : ℝ),
    W = n * 28 →
    W + 10 = (n + 1) * 27.4 →
    n = 29 :=
by sorry

end initial_number_of_students_l712_71217


namespace jerrys_shelf_l712_71221

theorem jerrys_shelf (books : ℕ) (added_figures : ℕ) (difference : ℕ) : 
  books = 7 → added_figures = 2 → difference = 2 →
  ∃ initial_figures : ℕ, 
    initial_figures = 3 ∧ 
    books = (initial_figures + added_figures) + difference :=
by sorry

end jerrys_shelf_l712_71221


namespace building_height_l712_71258

/-- Given a flagpole and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 65)
  : (flagpole_height * building_shadow) / flagpole_shadow = 26 := by
  sorry

#check building_height

end building_height_l712_71258


namespace fish_problem_l712_71297

theorem fish_problem (west north left : ℕ) (E : ℕ) : 
  west = 1800 →
  north = 500 →
  left = 2870 →
  (3 * E) / 5 + west / 4 + north = left →
  E = 3200 :=
by
  sorry

end fish_problem_l712_71297


namespace abs_inequality_exponential_inequality_l712_71267

-- Problem 1
theorem abs_inequality (x : ℝ) :
  |x - 1| > 2 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi 3 :=
sorry

-- Problem 2
theorem exponential_inequality (a x : ℝ) (h : 0 < a ∧ a < 1) :
  a^(1 - x) < a^(x + 1) ↔ x ∈ Set.Iio 0 :=
sorry

end abs_inequality_exponential_inequality_l712_71267


namespace davids_biology_marks_l712_71279

theorem davids_biology_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (average : ℕ)
  (h1 : english = 36)
  (h2 : mathematics = 35)
  (h3 : physics = 42)
  (h4 : chemistry = 57)
  (h5 : average = 45)
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) :
  biology = 55 :=
by
  sorry

end davids_biology_marks_l712_71279


namespace simplify_and_evaluate_l712_71276

theorem simplify_and_evaluate (x : ℝ) (h : x = 1 / (Real.sqrt 3 - 2)) :
  x^2 + 4*x - 4 = -5 := by
  sorry

end simplify_and_evaluate_l712_71276


namespace larger_sphere_radius_l712_71239

theorem larger_sphere_radius (r : ℝ) (n : ℕ) (h : r = 3 ∧ n = 9) :
  (n * (4 / 3 * Real.pi * r^3))^(1/3) = 9 := by
  sorry

end larger_sphere_radius_l712_71239


namespace second_grade_selection_theorem_l712_71213

/-- Represents the school population --/
structure School :=
  (total_students : ℕ)
  (first_grade_male_prob : ℝ)

/-- Represents the sampling method --/
structure Sampling :=
  (total_volunteers : ℕ)
  (method : String)

/-- Calculates the number of students selected from the second grade --/
def second_grade_selection (s : School) (samp : Sampling) : ℕ :=
  sorry

theorem second_grade_selection_theorem (s : School) (samp : Sampling) :
  s.total_students = 4000 →
  s.first_grade_male_prob = 0.2 →
  samp.total_volunteers = 100 →
  samp.method = "stratified" →
  second_grade_selection s samp = 30 :=
sorry

end second_grade_selection_theorem_l712_71213


namespace sum_nonnegative_implies_one_nonnegative_l712_71299

theorem sum_nonnegative_implies_one_nonnegative (a b : ℝ) :
  a + b ≥ 0 → (a ≥ 0 ∨ b ≥ 0) := by
  sorry

end sum_nonnegative_implies_one_nonnegative_l712_71299


namespace f_properties_l712_71231

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) := Real.log ((a^x) - (b^x))

theorem f_properties (h1 : a > 1) (h2 : b > 0) (h3 : b < 1) :
  -- 1. Domain of f is (0, +∞)
  (∀ x > 0, (a^x) - (b^x) > 0) ∧
  -- 2. f is strictly increasing on its domain
  (∀ x y, 0 < x ∧ x < y → f a b x < f a b y) ∧
  -- 3. f(x) > 0 for all x > 1 iff a - b ≥ 1
  (∀ x > 1, f a b x > 0) ↔ a - b ≥ 1 :=
sorry

end f_properties_l712_71231


namespace redistribution_theorem_l712_71251

/-- The number of trucks after redistribution of oil containers -/
def num_trucks_after_redistribution : ℕ :=
  let initial_trucks_1 : ℕ := 7
  let boxes_per_truck_1 : ℕ := 20
  let initial_trucks_2 : ℕ := 5
  let boxes_per_truck_2 : ℕ := 12
  let containers_per_box : ℕ := 8
  let containers_per_truck_after : ℕ := 160
  let total_boxes : ℕ := initial_trucks_1 * boxes_per_truck_1 + initial_trucks_2 * boxes_per_truck_2
  let total_containers : ℕ := total_boxes * containers_per_box
  total_containers / containers_per_truck_after

theorem redistribution_theorem :
  num_trucks_after_redistribution = 10 :=
by sorry

end redistribution_theorem_l712_71251


namespace egg_distribution_l712_71207

theorem egg_distribution (total_eggs : ℕ) (num_adults num_boys num_girls : ℕ) 
  (eggs_per_adult eggs_per_girl : ℕ) :
  total_eggs = 36 →
  num_adults = 3 →
  num_boys = 10 →
  num_girls = 7 →
  eggs_per_adult = 3 →
  eggs_per_girl = 1 →
  ∃ (eggs_per_boy : ℕ),
    total_eggs = num_adults * eggs_per_adult + num_boys * eggs_per_boy + num_girls * eggs_per_girl ∧
    eggs_per_boy = eggs_per_girl + 1 :=
by
  sorry

end egg_distribution_l712_71207


namespace inverse_fraction_ratio_l712_71268

noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

theorem inverse_fraction_ratio (a b c d : ℝ) :
  (∀ x, g (((a * x + b) / (c * x + d)) : ℝ) = x) →
  a / c = -4 := by
  sorry

end inverse_fraction_ratio_l712_71268


namespace wheel_moves_200cm_per_rotation_l712_71233

/-- Represents the properties of a rotating wheel -/
structure RotatingWheel where
  rotations_per_minute : ℕ
  distance_per_hour : ℕ

/-- Calculates the distance moved during each rotation of the wheel -/
def distance_per_rotation (wheel : RotatingWheel) : ℚ :=
  wheel.distance_per_hour / (wheel.rotations_per_minute * 60)

/-- Theorem stating that a wheel with the given properties moves 200 cm per rotation -/
theorem wheel_moves_200cm_per_rotation (wheel : RotatingWheel) 
    (h1 : wheel.rotations_per_minute = 10)
    (h2 : wheel.distance_per_hour = 120000) : 
  distance_per_rotation wheel = 200 := by
  sorry

end wheel_moves_200cm_per_rotation_l712_71233


namespace abe_age_sum_l712_71285

/-- The sum of Abe's present age and his age 7 years ago is 31, given that Abe's present age is 19. -/
theorem abe_age_sum : 
  let present_age : ℕ := 19
  let years_ago : ℕ := 7
  present_age + (present_age - years_ago) = 31 := by sorry

end abe_age_sum_l712_71285


namespace sequence_property_l712_71247

def is_valid_sequence (s : List Nat) : Prop :=
  (∀ x ∈ s, x = 0 ∨ x = 1) ∧ 
  (∀ i j, i + 4 < s.length → j + 4 < s.length → 
    (List.take 5 (List.drop i s) ≠ List.take 5 (List.drop j s) ∨ i = j)) ∧
  (∀ x, x = 0 ∨ x = 1 → 
    ¬(∀ i j, i + 4 < (s ++ [x]).length → j + 4 < (s ++ [x]).length → 
      (List.take 5 (List.drop i (s ++ [x])) ≠ List.take 5 (List.drop j (s ++ [x])) ∨ i = j)))

theorem sequence_property (s : List Nat) (h : is_valid_sequence s) (h_length : s.length ≥ 8) :
  List.take 4 s = List.take 4 (List.reverse s) :=
sorry

end sequence_property_l712_71247


namespace competition_score_l712_71256

theorem competition_score (total_judges : Nat) (highest_score lowest_score avg_score : ℝ) :
  total_judges = 9 →
  highest_score = 86 →
  lowest_score = 45 →
  avg_score = 76 →
  (total_judges * avg_score - highest_score - lowest_score) / (total_judges - 2) = 79 := by
  sorry

end competition_score_l712_71256


namespace modulus_of_complex_fraction_l712_71242

theorem modulus_of_complex_fraction : Complex.abs ((2 - Complex.I) / (1 + Complex.I)) = Real.sqrt 10 / 2 := by
  sorry

end modulus_of_complex_fraction_l712_71242


namespace fixed_points_are_corresponding_l712_71216

/-- A type representing a geometric figure -/
structure Figure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Similarity transformation between two figures -/
def similarity (F1 F2 : Figure) : Prop :=
  sorry

/-- A point in a geometric figure -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Defines if a point is fixed under a similarity transformation -/
def is_fixed_point (p : Point) (F1 F2 : Figure) : Prop :=
  sorry

/-- Defines if two points are corresponding in similar figures -/
def are_corresponding (p1 p2 : Point) (F1 F2 : Figure) : Prop :=
  sorry

/-- Main theorem: Fixed points of three similar figures are their corresponding points -/
theorem fixed_points_are_corresponding 
  (F1 F2 F3 : Figure) 
  (h1 : similarity F1 F2) 
  (h2 : similarity F2 F3) 
  (h3 : similarity F3 F1) 
  (p1 : Point) 
  (p2 : Point) 
  (p3 : Point) 
  (hf1 : is_fixed_point p1 F1 F2) 
  (hf2 : is_fixed_point p2 F2 F3) 
  (hf3 : is_fixed_point p3 F3 F1) : 
  are_corresponding p1 p2 F1 F2 ∧ 
  are_corresponding p2 p3 F2 F3 ∧ 
  are_corresponding p3 p1 F3 F1 :=
sorry

end fixed_points_are_corresponding_l712_71216


namespace surface_area_difference_l712_71219

/-- The difference between the sum of surface areas of smaller cubes and the surface area of a larger cube -/
theorem surface_area_difference (larger_cube_volume : ℝ) (num_smaller_cubes : ℕ) (smaller_cube_volume : ℝ) : 
  larger_cube_volume = 343 →
  num_smaller_cubes = 343 →
  smaller_cube_volume = 1 →
  (num_smaller_cubes : ℝ) * (6 * smaller_cube_volume ^ (2/3)) - 6 * larger_cube_volume ^ (2/3) = 1764 := by
  sorry

end surface_area_difference_l712_71219


namespace total_wheels_in_lot_l712_71282

/-- The number of wheels on a standard car -/
def wheels_per_car : ℕ := 4

/-- The number of cars in the parking lot -/
def cars_in_lot : ℕ := 12

/-- Theorem: The total number of car wheels in the parking lot is 48 -/
theorem total_wheels_in_lot : cars_in_lot * wheels_per_car = 48 := by
  sorry

end total_wheels_in_lot_l712_71282


namespace coefficient_of_x5y2_l712_71235

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the polynomial (x^2 + 3x - y)^5
def polynomial (x y : ℤ) : ℤ := (x^2 + 3*x - y)^5

-- Theorem statement
theorem coefficient_of_x5y2 :
  ∃ (coeff : ℤ), coeff = 90 ∧
  ∀ (x y : ℤ), 
    ∃ (rest : ℤ), 
      polynomial x y = coeff * x^5 * y^2 + rest ∧ 
      (∀ (a b : ℕ), a ≤ 5 ∧ b ≤ 2 ∧ (a, b) ≠ (5, 2) → 
        ∃ (other_terms : ℤ), rest = other_terms * x^a * y^b + other_terms) :=
sorry

end coefficient_of_x5y2_l712_71235


namespace inequality_proof_l712_71243

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  a^(a^2 + 2*c*a) * b^(b^2 + 2*a*b) * c^(c^2 + 2*b*c) ≥ 1/3 := by
  sorry

end inequality_proof_l712_71243


namespace randys_pig_feed_per_week_l712_71255

/-- Calculates the amount of pig feed needed per week given the daily feed per pig, number of pigs, and days in a week. -/
def pig_feed_per_week (feed_per_pig_per_day : ℕ) (num_pigs : ℕ) (days_in_week : ℕ) : ℕ :=
  feed_per_pig_per_day * num_pigs * days_in_week

/-- Proves that Randy's pigs will be fed 140 pounds of pig feed per week. -/
theorem randys_pig_feed_per_week :
  let feed_per_pig_per_day : ℕ := 10
  let num_pigs : ℕ := 2
  let days_in_week : ℕ := 7
  pig_feed_per_week feed_per_pig_per_day num_pigs days_in_week = 140 := by
  sorry

end randys_pig_feed_per_week_l712_71255


namespace parabola_vertex_locus_l712_71226

/-- The locus of the vertex of a parabola with specific constraints -/
theorem parabola_vertex_locus (a b s t : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + 1) →  -- Parabola equation
  (8 * a^2 + 4 * a * b = b^3) →       -- Constraint on a and b
  (s = -b / (2 * a)) →                -- x-coordinate of vertex
  (t = (4 * a - b^2) / (4 * a)) →     -- y-coordinate of vertex
  (s * t = 1) :=                      -- Locus equation
by sorry

end parabola_vertex_locus_l712_71226


namespace binomial_30_3_l712_71274

theorem binomial_30_3 : (30 : ℕ).choose 3 = 4060 := by sorry

end binomial_30_3_l712_71274


namespace sin_15_minus_sin_75_l712_71246

theorem sin_15_minus_sin_75 : 
  Real.sin (15 * π / 180) - Real.sin (75 * π / 180) = -Real.sqrt 6 / 2 := by
  sorry

end sin_15_minus_sin_75_l712_71246


namespace digit_product_sum_l712_71254

/-- A function that checks if a number is a three-digit number with all digits the same -/
def isTripleDigit (n : Nat) : Prop :=
  ∃ d, d ∈ Finset.range 10 ∧ n = d * 100 + d * 10 + d

/-- A function that converts a two-digit number to its decimal representation -/
def twoDigitToDecimal (a b : Nat) : Nat := 10 * a + b

theorem digit_product_sum : 
  ∃ (A B C D E : Nat), 
    A ∈ Finset.range 10 ∧ 
    B ∈ Finset.range 10 ∧ 
    C ∈ Finset.range 10 ∧ 
    D ∈ Finset.range 10 ∧ 
    E ∈ Finset.range 10 ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (twoDigitToDecimal A B) * (twoDigitToDecimal C D) = E * 100 + E * 10 + E ∧
    A + B + C + D + E = 21 :=
sorry

end digit_product_sum_l712_71254


namespace line_segment_b_value_l712_71261

/-- Given a line segment with slope -3/2 from (0, b) to (8, 0), prove b = 12 -/
theorem line_segment_b_value (b : ℝ) : 
  (∀ x y, 0 ≤ x → x ≤ 8 → y = b - (3/2) * x) → 
  (b - (3/2) * 8 = 0) → 
  b = 12 := by
  sorry


end line_segment_b_value_l712_71261


namespace petes_number_l712_71292

theorem petes_number : ∃ x : ℚ, 5 * (3 * x + 15) = 245 ∧ x = 34 / 3 := by
  sorry

end petes_number_l712_71292


namespace zhang_wang_sum_difference_l712_71287

/-- Sum of arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n - 1) * d) / 2

/-- Sum of 26 consecutive odd numbers starting from 27 -/
def zhang_sum : ℕ := arithmetic_sum 27 2 26

/-- Sum of 26 consecutive natural numbers starting from 26 -/
def wang_sum : ℕ := arithmetic_sum 26 1 26

theorem zhang_wang_sum_difference :
  zhang_sum - wang_sum = 351 := by sorry

end zhang_wang_sum_difference_l712_71287


namespace all_grids_have_uniform_subgrid_l712_71200

def Grid := Fin 5 → Fin 6 → Bool

def hasUniformSubgrid (g : Grid) : Prop :=
  ∃ (i j : Fin 4), 
    (g i j = g (i + 1) j ∧ 
     g i j = g i (j + 1) ∧ 
     g i j = g (i + 1) (j + 1))

theorem all_grids_have_uniform_subgrid :
  ∀ (g : Grid), hasUniformSubgrid g :=
sorry

end all_grids_have_uniform_subgrid_l712_71200


namespace problem_solution_l712_71244

def smallest_positive_integer : ℕ := 1

def opposite_is_self (b : ℤ) : Prop := -b = b

def largest_negative_integer : ℤ := -1

theorem problem_solution (a b c : ℤ) 
  (ha : a = smallest_positive_integer)
  (hb : opposite_is_self b)
  (hc : c = largest_negative_integer + 3) :
  (2*a + 3*c) * b = 0 := by
  sorry

end problem_solution_l712_71244


namespace divisibility_equivalence_l712_71224

theorem divisibility_equivalence (a b : ℤ) :
  (29 ∣ 3*a + 2*b) ↔ (29 ∣ 11*a + 17*b) := by
  sorry

end divisibility_equivalence_l712_71224


namespace parabola_properties_l712_71234

-- Define the parabola
def parabola (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- Define the roots of the parabola
def roots (b c : ℝ) : Set ℝ := {x | parabola b c x = 0}

-- Theorem statement
theorem parabola_properties :
  ∀ b c : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + x₂ = 4 ∧ {x₁, x₂} ⊆ roots b c) →
  (b = 4 ∧ c > -4) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ {x₁, x₂} ⊆ roots b c ∧ |x₁ - x₂| = 2 → c = -3) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ {x₁, x₂} ⊆ roots b c ∧
    (c = (1 + Real.sqrt 17) / 2 ∨ c = (1 - Real.sqrt 17) / 2) ∧
    |c| = c - parabola b c 2) :=
sorry

end parabola_properties_l712_71234


namespace multiplication_fraction_product_l712_71289

theorem multiplication_fraction_product : 11 * (1 / 17) * 34 = 22 := by
  sorry

end multiplication_fraction_product_l712_71289


namespace parabola_shift_theorem_l712_71225

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically --/
def shift_parabola (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_shift_theorem :
  let original := Parabola.mk (-1) 0 1
  let shifted := shift_parabola original 2 (-2)
  shifted.a = -1 ∧ shifted.h = 2 ∧ shifted.k = -1 := by
  sorry

end parabola_shift_theorem_l712_71225


namespace sqrt_difference_equals_21_over_10_l712_71241

theorem sqrt_difference_equals_21_over_10 :
  Real.sqrt (25 / 4) - Real.sqrt (4 / 25) = 21 / 10 := by
  sorry

end sqrt_difference_equals_21_over_10_l712_71241


namespace smallest_positive_angle_proof_l712_71271

/-- The smallest positive angle with the same terminal side as 400° -/
def smallest_positive_angle : ℝ := 40

/-- The set of angles with the same terminal side as 400° -/
def angle_set (k : ℤ) : ℝ := 400 + k * 360

theorem smallest_positive_angle_proof :
  ∀ k : ℤ, angle_set k > 0 → smallest_positive_angle ≤ angle_set k :=
sorry

end smallest_positive_angle_proof_l712_71271


namespace wrapping_paper_fraction_l712_71229

theorem wrapping_paper_fraction (total_fraction : ℚ) (num_presents : ℕ) 
  (h1 : total_fraction = 3 / 10)
  (h2 : num_presents = 3) :
  total_fraction / num_presents = 1 / 10 := by
  sorry

end wrapping_paper_fraction_l712_71229


namespace a_is_perfect_square_l712_71248

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 7 * a (n + 1) - a n - 2

theorem a_is_perfect_square : ∀ n : ℕ, ∃ k : ℤ, a n = k^2 := by
  sorry

end a_is_perfect_square_l712_71248


namespace triangle_heights_semiperimeter_inequality_l712_71298

/-- Given a triangle with heights m_a, m_b, m_c and semiperimeter s,
    prove that the sum of squares of the heights is less than or equal to
    the square of the semiperimeter. -/
theorem triangle_heights_semiperimeter_inequality 
  (m_a m_b m_c s : ℝ) 
  (h_pos_a : 0 < m_a) (h_pos_b : 0 < m_b) (h_pos_c : 0 < m_c) (h_pos_s : 0 < s)
  (h_heights : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    m_a = (2 * (s * (s - a) * (s - b) * (s - c))^(1/2)) / a ∧
    m_b = (2 * (s * (s - a) * (s - b) * (s - c))^(1/2)) / b ∧
    m_c = (2 * (s * (s - a) * (s - b) * (s - c))^(1/2)) / c ∧
    s = (a + b + c) / 2) :
  m_a^2 + m_b^2 + m_c^2 ≤ s^2 := by sorry

end triangle_heights_semiperimeter_inequality_l712_71298


namespace min_value_of_w_l712_71220

theorem min_value_of_w (x y : ℝ) :
  let w := 3 * x^2 + 5 * y^2 + 8 * x - 10 * y + 34
  w ≥ 71 / 3 :=
by sorry

end min_value_of_w_l712_71220


namespace count_four_digit_snappy_divisible_by_25_l712_71264

def is_snappy (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 1000 + b * 100 + b * 10 + a ∧ a < 10 ∧ b < 10

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem count_four_digit_snappy_divisible_by_25 :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_four_digit n ∧ is_snappy n ∧ n % 25 = 0) ∧
    s.card = 3 :=
sorry

end count_four_digit_snappy_divisible_by_25_l712_71264


namespace fraction_equals_zero_l712_71253

theorem fraction_equals_zero (x : ℝ) : (x - 2) / (x + 5) = 0 → x = 2 := by
  sorry

end fraction_equals_zero_l712_71253


namespace division_problem_l712_71290

theorem division_problem (Ω : ℕ) : 
  Ω ≤ 9 ∧ Ω ≥ 1 →
  (∃ (n : ℕ), n ≥ 10 ∧ n < 50 ∧ 504 / Ω = n + 2 * Ω) →
  Ω = 7 := by
sorry

end division_problem_l712_71290


namespace remainder_plus_three_l712_71293

/-- f(x) represents the remainder of x divided by 3 -/
def f (x : ℕ) : ℕ := x % 3

/-- For all natural numbers x, f(x+3) = f(x) -/
theorem remainder_plus_three (x : ℕ) : f (x + 3) = f x := by
  sorry

end remainder_plus_three_l712_71293


namespace preschool_nap_problem_l712_71277

theorem preschool_nap_problem (initial_kids : ℕ) (awake_after_first_round : ℕ) (awake_after_second_round : ℕ) : 
  initial_kids = 20 →
  awake_after_first_round = initial_kids - initial_kids / 2 →
  awake_after_second_round = awake_after_first_round - awake_after_first_round / 2 →
  awake_after_second_round = 5 :=
by sorry

end preschool_nap_problem_l712_71277


namespace initial_boys_count_l712_71249

theorem initial_boys_count (initial_total : ℕ) (initial_boys : ℕ) (final_boys : ℕ) : 
  initial_boys = initial_total / 2 →                   -- Initially, 50% are boys
  final_boys = initial_boys - 3 →                      -- 3 boys leave
  final_boys * 10 = 4 * initial_total →                -- After changes, 40% are boys
  initial_boys = 15 := by
sorry

end initial_boys_count_l712_71249


namespace oblique_drawing_properties_l712_71232

-- Define the intuitive drawing using the oblique method
structure ObliqueDrawing where
  x_scale : ℝ
  y_scale : ℝ
  angle : ℝ

-- Define the properties of the oblique drawing
def is_valid_oblique_drawing (d : ObliqueDrawing) : Prop :=
  d.x_scale = 1 ∧ d.y_scale = 1/2 ∧ (d.angle = 135 ∨ d.angle = 45)

-- Theorem stating the properties of oblique drawing
theorem oblique_drawing_properties (d : ObliqueDrawing) 
  (h : is_valid_oblique_drawing d) : 
  d.x_scale = 1 ∧ 
  d.y_scale = 1/2 ∧ 
  (d.angle = 135 ∨ d.angle = 45) ∧ 
  ∃ (d' : ObliqueDrawing), is_valid_oblique_drawing d' ∧ d' ≠ d :=
sorry


end oblique_drawing_properties_l712_71232


namespace shoe_size_age_game_l712_71288

theorem shoe_size_age_game (shoe_size age : ℕ) : 
  let current_year := 1952
  let birth_year := current_year - age
  let game_result := ((shoe_size + 7) * 2 + 5) * 50 + 1711 - birth_year
  game_result = 5059 → shoe_size = 43 ∧ age = 50 := by
sorry

end shoe_size_age_game_l712_71288


namespace starting_lineup_combinations_l712_71283

/-- The number of players in the team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def num_starters : ℕ := 5

/-- The number of quadruplets that must be in the starting lineup -/
def quadruplets_in_lineup : ℕ := 3

/-- The number of ways to choose the starting lineup -/
def ways_to_choose_lineup : ℕ := Nat.choose num_quadruplets quadruplets_in_lineup * 
  Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)

theorem starting_lineup_combinations : ways_to_choose_lineup = 264 := by sorry

end starting_lineup_combinations_l712_71283


namespace floor_ceil_problem_l712_71206

theorem floor_ceil_problem : (⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉) * 2 = 62 := by
  sorry

end floor_ceil_problem_l712_71206


namespace computer_usage_difference_l712_71275

/-- The difference in computer usage between two weeks -/
def usage_difference (last_week : ℕ) (this_week_daily : ℕ) : ℕ :=
  last_week - (this_week_daily * 7)

/-- Theorem stating the difference in computer usage -/
theorem computer_usage_difference :
  usage_difference 91 8 = 35 := by
  sorry

end computer_usage_difference_l712_71275


namespace v₃_value_l712_71202

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 5*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

/-- The value of x -/
def x : ℝ := 5

/-- The definition of v₃ -/
def v₃ : ℝ := (((5*x + 2)*x + 3.5)*x - 2.6)

/-- Theorem stating that v₃ equals 689.9 -/
theorem v₃_value : v₃ = 689.9 := by
  sorry

end v₃_value_l712_71202


namespace check_amount_error_l712_71228

theorem check_amount_error (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 →  -- x and y are two-digit numbers
  y - x = 18 →                         -- difference is $17.82
  ∃ x y : ℕ, y = 2 * x                 -- y can be twice x
:= by sorry

end check_amount_error_l712_71228


namespace isabella_escalator_time_l712_71295

/-- Represents the time it takes Isabella to ride an escalator under different conditions -/
def EscalatorTime (walk_time_stopped : ℝ) (walk_time_moving : ℝ) : Prop :=
  ∃ (escalator_speed : ℝ) (isabella_speed : ℝ),
    escalator_speed > 0 ∧
    isabella_speed > 0 ∧
    walk_time_stopped * isabella_speed = walk_time_moving * (isabella_speed + escalator_speed) ∧
    walk_time_stopped / escalator_speed = 45

theorem isabella_escalator_time :
  EscalatorTime 90 30 :=
sorry

end isabella_escalator_time_l712_71295


namespace cylinder_cone_volume_l712_71259

/-- Given a cylinder and cone with the same base and height, and a combined volume of 48cm³,
    prove that the volume of the cylinder is 36cm³ and the volume of the cone is 12cm³. -/
theorem cylinder_cone_volume (cylinder_volume cone_volume : ℝ) : 
  cylinder_volume + cone_volume = 48 →
  cylinder_volume = 3 * cone_volume →
  cylinder_volume = 36 ∧ cone_volume = 12 := by
sorry

end cylinder_cone_volume_l712_71259


namespace arithmetic_equality_l712_71260

theorem arithmetic_equality : 1 - 0.2 + 0.03 - 0.004 = 0.826 := by
  sorry

end arithmetic_equality_l712_71260


namespace exists_special_function_l712_71257

open Function Set

/-- A function f: ℝ → ℝ satisfying specific properties --/
structure SpecialFunction where
  f : ℝ → ℝ
  increasing : Monotone f
  composite_increasing : Monotone (f ∘ f)
  not_fixed_point : ∀ a : ℝ, f a ≠ a
  involutive : ∀ x : ℝ, f (f x) = x

/-- Theorem stating the existence of a function satisfying the required properties --/
theorem exists_special_function : ∃ sf : SpecialFunction, True := by
  sorry

end exists_special_function_l712_71257


namespace max_distance_on_circle_l712_71214

-- Define the circle Ω
def Ω : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + y^2 - 2*x - 4*y = 0}

-- Define the points that the circle passes through
def origin : ℝ × ℝ := (0, 0)
def point1 : ℝ × ℝ := (2, 4)
def point2 : ℝ × ℝ := (3, 3)

-- Theorem statement
theorem max_distance_on_circle :
  origin ∈ Ω ∧ point1 ∈ Ω ∧ point2 ∈ Ω →
  ∃ (max_dist : ℝ),
    (∀ p ∈ Ω, Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ max_dist) ∧
    (∃ q ∈ Ω, Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) = max_dist) ∧
    max_dist = 2 * Real.sqrt 5 :=
sorry

end max_distance_on_circle_l712_71214


namespace two_digit_three_digit_sum_l712_71210

theorem two_digit_three_digit_sum : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 100 ≤ y ∧ y < 1000 ∧ 
  1000 * x + y = 11 * x * y ∧ 
  x + y = 919 := by sorry

end two_digit_three_digit_sum_l712_71210


namespace robin_candy_packages_l712_71236

/-- Given the total number of candy pieces and the number of pieces per package,
    calculate the number of candy packages. -/
def candy_packages (total_pieces : ℕ) (pieces_per_package : ℕ) : ℕ :=
  total_pieces / pieces_per_package

/-- Theorem stating that Robin has 45 packages of candy. -/
theorem robin_candy_packages :
  candy_packages 405 9 = 45 := by
  sorry

#eval candy_packages 405 9

end robin_candy_packages_l712_71236


namespace rectangle_area_rectangle_area_proof_l712_71265

theorem rectangle_area (square_area : Real) (rectangle_breadth : Real) : Real :=
  let square_side : Real := Real.sqrt square_area
  let circle_radius : Real := square_side
  let rectangle_length : Real := circle_radius / 4
  let rectangle_area : Real := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 1225 10 = 87.5 := by
  sorry

end rectangle_area_rectangle_area_proof_l712_71265


namespace largest_b_no_real_roots_l712_71204

theorem largest_b_no_real_roots : 
  ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 15 ≠ 0) → b ≤ 7 :=
by
  sorry

end largest_b_no_real_roots_l712_71204


namespace app_difference_proof_l712_71266

/-- Calculates the difference between added and deleted apps -/
def appDifference (initial final added : ℕ) : ℕ :=
  added - ((initial + added) - final)

theorem app_difference_proof (initial final added : ℕ) 
  (h1 : initial = 21)
  (h2 : final = 24)
  (h3 : added = 89) :
  appDifference initial final added = 3 := by
  sorry

#eval appDifference 21 24 89

end app_difference_proof_l712_71266


namespace prime_difference_divisibility_l712_71286

theorem prime_difference_divisibility 
  (p₁ p₂ p₃ p₄ q₁ q₂ q₃ q₄ : ℕ) 
  (hp : p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄)
  (hq : q₁ < q₂ ∧ q₂ < q₃ ∧ q₃ < q₄)
  (hp_prime : Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄)
  (hq_prime : Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄)
  (hp_diff : p₄ - p₁ = 8)
  (hq_diff : q₄ - q₁ = 8)
  (hp_gt_5 : p₁ > 5)
  (hq_gt_5 : q₁ > 5) :
  30 ∣ (p₁ - q₁) := by
sorry

end prime_difference_divisibility_l712_71286


namespace f_always_positive_sum_reciprocals_geq_nine_l712_71284

-- Problem 1
def f (x : ℝ) : ℝ := x^6 - x^3 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, f x > 0 := by
  sorry

-- Problem 2
theorem sum_reciprocals_geq_nine {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c = 1) : 1/a + 1/b + 1/c ≥ 9 := by
  sorry

end f_always_positive_sum_reciprocals_geq_nine_l712_71284


namespace arithmetic_sequence_second_term_l712_71262

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the sum of the first and third terms is 8,
    prove that the second term is 4. -/
theorem arithmetic_sequence_second_term
  (a : ℕ → ℝ)
  (h_arithmetic : IsArithmeticSequence a)
  (h_sum : a 0 + a 2 = 8) :
  a 1 = 4 := by
sorry

end arithmetic_sequence_second_term_l712_71262


namespace function_minimum_l712_71245

theorem function_minimum (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x - 3*a - 9 ≥ 0) →
  (1^2 + a*1 - 3*a - 9 = 4) :=
by
  sorry

end function_minimum_l712_71245


namespace f_value_at_7_6_l712_71296

def periodic_function (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f (x + period) = f x

theorem f_value_at_7_6 (f : ℝ → ℝ) (h1 : periodic_function f 4) 
  (h2 : ∀ x ∈ Set.Icc (-2) 2, f x = x + 1) : 
  f 7.6 = 0.6 := by
  sorry

end f_value_at_7_6_l712_71296


namespace dollar_calculation_l712_71252

-- Define the $ operation
def dollar (a b : ℝ) : ℝ := (a^2 - b^2)^2

-- Theorem statement
theorem dollar_calculation (x : ℝ) : 
  dollar (x^3 + x) (x - x^3) = 16 * x^8 := by
  sorry

end dollar_calculation_l712_71252


namespace count_valid_triples_l712_71270

def valid_triple (x y z : ℕ+) : Prop :=
  Nat.lcm x.val y.val = 180 ∧
  Nat.lcm x.val z.val = 420 ∧
  Nat.lcm y.val z.val = 1260

theorem count_valid_triples :
  ∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)),
    (∀ t ∈ s, valid_triple t.1 t.2.1 t.2.2) ∧
    s.card = 4 :=
sorry

end count_valid_triples_l712_71270


namespace interest_calculation_l712_71280

/-- Calculates the simple interest and proves the interest credited is 63 cents. -/
theorem interest_calculation (initial_savings : ℝ) (interest_rate : ℝ) (time : ℝ) 
  (additional_deposit : ℝ) (total_amount : ℝ) : ℝ :=
  let interest := initial_savings * interest_rate * time
  let amount_after_interest := initial_savings + interest
  let amount_after_deposit := amount_after_interest + additional_deposit
  let interest_credited := total_amount - (initial_savings + additional_deposit)
by
  have h1 : initial_savings = 500 := by sorry
  have h2 : interest_rate = 0.03 := by sorry
  have h3 : time = 1/4 := by sorry
  have h4 : additional_deposit = 15 := by sorry
  have h5 : total_amount = 515.63 := by sorry
  
  -- Prove that the interest credited is 63 cents
  sorry

#eval (515.63 - (500 + 15)) * 100 -- Should evaluate to 63.0

end interest_calculation_l712_71280


namespace solve_simultaneous_equations_l712_71209

theorem solve_simultaneous_equations :
  ∀ x y : ℝ,
  (x / 5 + 7 = y / 4 - 7) →
  (x / 3 - 4 = y / 2 + 4) →
  (x = -660 ∧ y = -472) :=
by
  sorry

end solve_simultaneous_equations_l712_71209


namespace one_and_two_thirds_of_number_is_45_l712_71205

theorem one_and_two_thirds_of_number_is_45 : ∃ x : ℚ, (5 / 3) * x = 45 ∧ x = 27 := by
  sorry

end one_and_two_thirds_of_number_is_45_l712_71205


namespace factory_workers_count_l712_71250

/-- The total number of workers in the factory -/
def total_workers : ℕ := 900

/-- The number of workers in Workshop B -/
def workshop_b_workers : ℕ := 300

/-- The total sample size -/
def total_sample : ℕ := 45

/-- The number of people sampled from Workshop A -/
def sample_a : ℕ := 20

/-- The number of people sampled from Workshop C -/
def sample_c : ℕ := 10

/-- The number of people sampled from Workshop B -/
def sample_b : ℕ := total_sample - sample_a - sample_c

theorem factory_workers_count :
  (sample_b : ℚ) / workshop_b_workers = (total_sample : ℚ) / total_workers :=
by sorry

end factory_workers_count_l712_71250


namespace cars_without_features_l712_71218

theorem cars_without_features (total : ℕ) (airbags : ℕ) (power_windows : ℕ) (both : ℕ)
  (h_total : total = 65)
  (h_airbags : airbags = 45)
  (h_power_windows : power_windows = 30)
  (h_both : both = 12) :
  total - (airbags + power_windows - both) = 2 :=
by sorry

end cars_without_features_l712_71218


namespace p_sufficient_not_necessary_for_q_l712_71291

-- Define the set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define the set Q
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem stating that "x ∈ P" is a sufficient but not necessary condition for "x ∈ Q"
theorem p_sufficient_not_necessary_for_q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by sorry

end p_sufficient_not_necessary_for_q_l712_71291


namespace factors_360_divisible_by_3_not_5_l712_71294

def factors_divisible_by_3_not_5 (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ∣ n ∧ 3 ∣ x ∧ ¬(5 ∣ x)) (Finset.range (n + 1))).card

theorem factors_360_divisible_by_3_not_5 :
  factors_divisible_by_3_not_5 360 = 8 := by
  sorry

end factors_360_divisible_by_3_not_5_l712_71294


namespace girls_insects_count_l712_71215

/-- The number of insects collected by boys -/
def boys_insects : ℕ := 200

/-- The number of groups the class was divided into -/
def num_groups : ℕ := 4

/-- The number of insects each group received -/
def insects_per_group : ℕ := 125

/-- The number of insects collected by girls -/
def girls_insects : ℕ := num_groups * insects_per_group - boys_insects

theorem girls_insects_count : girls_insects = 300 := by
  sorry

end girls_insects_count_l712_71215


namespace equation_solution_l712_71227

theorem equation_solution : 
  ∃ (x₁ x₂ : ℚ), (x₁ = 1/6 ∧ x₂ = -1/4) ∧ 
  (∀ x : ℚ, 4*x*(6*x - 1) = 1 - 6*x ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solution_l712_71227


namespace haley_recycling_cans_l712_71269

theorem haley_recycling_cans (total_cans : ℕ) (difference : ℕ) (cans_in_bag : ℕ) :
  total_cans = 9 →
  difference = 2 →
  total_cans - cans_in_bag = difference →
  cans_in_bag = 7 := by
sorry

end haley_recycling_cans_l712_71269


namespace set_operations_and_range_l712_71230

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

theorem set_operations_and_range :
  (A ∩ B = {x | 2 < x ∧ x ≤ 3}) ∧
  (A ∪ (U \ B) = {x | x ≤ 3 ∨ x ≥ 4}) ∧
  (∀ a : ℝ, B ∩ C a = C a → 2 < a ∧ a < 3) := by sorry

end set_operations_and_range_l712_71230


namespace quadratic_roots_fourth_power_sum_l712_71212

/-- For a quadratic equation x² - 2ax - 1/a² = 0 with roots x₁ and x₂,
    prove that x₁⁴ + x₂⁴ = 16 + 8√2 if and only if a = ± ∛∛(1/8) -/
theorem quadratic_roots_fourth_power_sum (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*a*x₁ - 1/a^2 = 0 → 
  x₂^2 - 2*a*x₂ - 1/a^2 = 0 → 
  (x₁^4 + x₂^4 = 16 + 8*Real.sqrt 2) ↔ 
  (a = Real.rpow (1/8) (1/8) ∨ a = -Real.rpow (1/8) (1/8)) := by
sorry

end quadratic_roots_fourth_power_sum_l712_71212


namespace no_snow_probability_l712_71237

theorem no_snow_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end no_snow_probability_l712_71237


namespace infinite_sum_equals_five_twentyfourths_l712_71263

/-- The sum of the infinite series n / (n^4 - 4n^2 + 8) from n = 1 to infinity is equal to 5/24. -/
theorem infinite_sum_equals_five_twentyfourths :
  ∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 - 4*(n : ℝ)^2 + 8) = 5 / 24 := by
  sorry

end infinite_sum_equals_five_twentyfourths_l712_71263
