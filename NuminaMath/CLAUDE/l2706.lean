import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2706_270613

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 = (x^2 + 3*x - 4) * q + (-51*x + 52) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2706_270613


namespace NUMINAMATH_CALUDE_car_speed_proof_l2706_270677

/-- Proves that a car's speed is 60 km/h if it takes 12 seconds longer to travel 1 km compared to 75 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v * 3600 = 1 / 75 * 3600 + 12) ↔ v = 60 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l2706_270677


namespace NUMINAMATH_CALUDE_yuna_and_friends_count_l2706_270678

/-- Given a line of people where Yuna is 4th from the front and 6th from the back,
    the total number of people in the line is 9. -/
theorem yuna_and_friends_count (people : ℕ) (yuna_position_front yuna_position_back : ℕ) :
  yuna_position_front = 4 →
  yuna_position_back = 6 →
  people = 9 :=
by sorry

end NUMINAMATH_CALUDE_yuna_and_friends_count_l2706_270678


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l2706_270620

/-- Given a point P with coordinates (x, -4), prove that if the distance from the x-axis to P
    is half the distance from the y-axis to P, then the distance from the y-axis to P is 8 units. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -4)
  let dist_to_x_axis := |P.2|
  let dist_to_y_axis := |P.1|
  dist_to_x_axis = (1/2) * dist_to_y_axis →
  dist_to_y_axis = 8 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l2706_270620


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2706_270641

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 4(x+2)^2 + 16y^2 = 64 is 2√5 -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (C.1 + 2)^2 / 16 + C.2^2 / 4 = 1 ∧  -- C is on the ellipse
    (D.1 + 2)^2 / 16 + D.2^2 / 4 = 1 ∧  -- D is on the ellipse
    C.2 = 0 ∧                           -- C is on the x-axis (major axis)
    D.1 = -2 ∧                          -- D is on the y-axis (minor axis)
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2706_270641


namespace NUMINAMATH_CALUDE_road_travel_cost_l2706_270660

/-- Calculate the cost of traveling two perpendicular roads on a rectangular lawn -/
theorem road_travel_cost (lawn_length lawn_width road_width cost_per_sqm : ℝ) :
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 15 ∧ 
  cost_per_sqm = 3 →
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 5625 := by
  sorry


end NUMINAMATH_CALUDE_road_travel_cost_l2706_270660


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l2706_270638

def jelly_bean_piles (initial_amount : ℕ) (amount_eaten : ℕ) (pile_weight : ℕ) : ℕ :=
  (initial_amount - amount_eaten) / pile_weight

theorem jelly_bean_problem :
  jelly_bean_piles 36 6 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l2706_270638


namespace NUMINAMATH_CALUDE_prob_different_colors_specific_l2706_270674

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  black : ℕ
  white : ℕ
  yellow : ℕ

/-- Calculates the total number of balls in a box -/
def Box.total (b : Box) : ℕ := b.red + b.black + b.white + b.yellow

/-- The probability of drawing two balls of different colors from two boxes -/
def prob_different_colors (boxA boxB : Box) : ℚ :=
  1 - (boxA.black * boxB.black + boxA.white * boxB.white : ℚ) / 
      ((boxA.total * boxB.total) : ℚ)

/-- The main theorem stating the probability of drawing different colored balls -/
theorem prob_different_colors_specific : 
  let boxA : Box := { red := 3, black := 3, white := 3, yellow := 0 }
  let boxB : Box := { red := 0, black := 2, white := 2, yellow := 2 }
  prob_different_colors boxA boxB = 7/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_different_colors_specific_l2706_270674


namespace NUMINAMATH_CALUDE_fraction_sum_product_l2706_270606

theorem fraction_sum_product : 
  24 * (243 / 3 + 49 / 7 + 16 / 8 + 4 / 2 + 2) = 2256 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_l2706_270606


namespace NUMINAMATH_CALUDE_rational_inequalities_l2706_270615

theorem rational_inequalities (a b : ℚ) : 
  ((a + b < a) → (b < 0)) ∧ ((a - b < a) → (b > 0)) := by sorry

end NUMINAMATH_CALUDE_rational_inequalities_l2706_270615


namespace NUMINAMATH_CALUDE_sqrt_2_times_sqrt_12_minus_2_between_2_and_3_l2706_270623

theorem sqrt_2_times_sqrt_12_minus_2_between_2_and_3 :
  2 < Real.sqrt 2 * Real.sqrt 12 - 2 ∧ Real.sqrt 2 * Real.sqrt 12 - 2 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_times_sqrt_12_minus_2_between_2_and_3_l2706_270623


namespace NUMINAMATH_CALUDE_inequality_solution_l2706_270651

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 5 / (x + 4) ≥ 1) ↔ (x ∈ Set.Icc (-4 : ℝ) (-2) ∨ x ∈ Set.Ico (-2 : ℝ) 5) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2706_270651


namespace NUMINAMATH_CALUDE_cube_root_neg_eight_plus_sqrt_nine_equals_one_l2706_270668

theorem cube_root_neg_eight_plus_sqrt_nine_equals_one :
  ((-8 : ℝ) ^ (1/3 : ℝ)) + (9 : ℝ).sqrt = 1 := by sorry

end NUMINAMATH_CALUDE_cube_root_neg_eight_plus_sqrt_nine_equals_one_l2706_270668


namespace NUMINAMATH_CALUDE_gcd_of_N_is_12_l2706_270659

def N (a b c d : ℕ) : ℤ :=
  (a - b) * (c - d) * (a - c) * (b - d) * (a - d) * (b - c)

theorem gcd_of_N_is_12 :
  ∃ (k : ℕ), ∀ (a b c d : ℕ), 
    (∃ (n : ℤ), N a b c d = 12 * n) ∧
    (∀ (m : ℕ), m > 12 → ¬(∃ (l : ℤ), N a b c d = m * l)) :=
sorry

end NUMINAMATH_CALUDE_gcd_of_N_is_12_l2706_270659


namespace NUMINAMATH_CALUDE_college_students_count_l2706_270621

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 210) :
  boys + girls = 546 := by
sorry

end NUMINAMATH_CALUDE_college_students_count_l2706_270621


namespace NUMINAMATH_CALUDE_water_speed_calculation_l2706_270685

/-- The speed of water in a river where a person who can swim at 4 km/h in still water
    takes 8 hours to swim 16 km against the current. -/
def water_speed : ℝ :=
  let still_water_speed : ℝ := 4
  let distance : ℝ := 16
  let time : ℝ := 8
  2

theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ)
    (h1 : still_water_speed = 4)
    (h2 : distance = 16)
    (h3 : time = 8)
    (h4 : distance = (still_water_speed - water_speed) * time) :
  water_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l2706_270685


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l2706_270605

theorem binomial_expansion_terms (n : ℕ) : (Finset.range (2 * n + 1)).card = 2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l2706_270605


namespace NUMINAMATH_CALUDE_leaf_fall_problem_l2706_270607

/-- The rate of leaves falling per hour in the second and third hour -/
def leaf_fall_rate (first_hour : ℕ) (average : ℚ) : ℚ :=
  (3 * average - first_hour) / 2

theorem leaf_fall_problem (first_hour : ℕ) (average : ℚ) :
  first_hour = 7 →
  average = 5 →
  leaf_fall_rate first_hour average = 4 := by
sorry

end NUMINAMATH_CALUDE_leaf_fall_problem_l2706_270607


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l2706_270687

-- Define the function f with domain (-1, 0)
def f : ℝ → ℝ := sorry

-- Define the composite function g(x) = f(2x+1)
def g (x : ℝ) : ℝ := f (2 * x + 1)

-- Theorem statement
theorem domain_of_composite_function :
  (∀ x, f x ≠ 0 → -1 < x ∧ x < 0) →
  (∀ x, g x ≠ 0 → -1 < x ∧ x < -1/2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l2706_270687


namespace NUMINAMATH_CALUDE_translate_line_upward_5_units_l2706_270693

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a linear function vertically by a given amount -/
def translateVertically (f : LinearFunction) (amount : ℝ) : LinearFunction :=
  { slope := f.slope, yIntercept := f.yIntercept + amount }

theorem translate_line_upward_5_units :
  let original : LinearFunction := { slope := 2, yIntercept := -4 }
  let translated := translateVertically original 5
  translated = { slope := 2, yIntercept := 1 } := by
  sorry

end NUMINAMATH_CALUDE_translate_line_upward_5_units_l2706_270693


namespace NUMINAMATH_CALUDE_alien_home_planet_abductees_l2706_270602

def total_abducted : ℕ := 1000
def return_percentage : ℚ := 528 / 1000
def to_zog : ℕ := 135
def to_xelbor : ℕ := 88
def to_qyruis : ℕ := 45

theorem alien_home_planet_abductees :
  total_abducted - 
  (↑(total_abducted) * return_percentage).floor - 
  to_zog - 
  to_xelbor - 
  to_qyruis = 204 := by
  sorry

end NUMINAMATH_CALUDE_alien_home_planet_abductees_l2706_270602


namespace NUMINAMATH_CALUDE_endpoint_sum_l2706_270636

/-- Given a line segment with one endpoint (1, 2) and midpoint (5, 6),
    the sum of coordinates of the other endpoint is 19. -/
theorem endpoint_sum (x y : ℝ) : 
  (1 + x) / 2 = 5 ∧ (2 + y) / 2 = 6 → x + y = 19 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_l2706_270636


namespace NUMINAMATH_CALUDE_birds_in_tree_l2706_270625

theorem birds_in_tree (initial_birds : ℕ) : 
  initial_birds + 21 = 35 → initial_birds = 14 := by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2706_270625


namespace NUMINAMATH_CALUDE_vec_b_is_correct_l2706_270610

def vec_a : ℝ × ℝ := (6, -8)
def vec_b : ℝ × ℝ := (-4, -3)
def vec_c : ℝ × ℝ := (1, 0)

theorem vec_b_is_correct : 
  (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 0) ∧ 
  (vec_b.1^2 + vec_b.2^2 = 25) ∧
  (vec_b.1 * vec_c.1 + vec_b.2 * vec_c.2 < 0) ∧
  (∀ x y : ℝ, (vec_a.1 * x + vec_a.2 * y = 0) ∧ 
              (x^2 + y^2 = 25) ∧ 
              (x * vec_c.1 + y * vec_c.2 < 0) → 
              (x, y) = vec_b) :=
by sorry

end NUMINAMATH_CALUDE_vec_b_is_correct_l2706_270610


namespace NUMINAMATH_CALUDE_max_b_value_l2706_270664

def is_lattice_point (x y : ℤ) : Prop := True

def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x : ℤ, 1 ≤ x → x ≤ 150 → ¬(is_lattice_point x (line_equation m x).num)

theorem max_b_value :
  ∃ b : ℚ, b = 50/149 ∧
    (∀ m : ℚ, 1/3 < m → m < b → no_lattice_points m) ∧
    (∀ b' : ℚ, b < b' → ∃ m : ℚ, 1/3 < m ∧ m < b' ∧ ¬(no_lattice_points m)) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l2706_270664


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l2706_270622

theorem triangle_side_and_area (a b c : ℝ) (B : ℝ) (h_a : a = 8) (h_b : b = 7) (h_B : B = Real.pi / 3) :
  c^2 - 4*c - 25 = 0 ∧ 
  ∃ S : ℝ, S = (1/2) * a * c * Real.sin B :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l2706_270622


namespace NUMINAMATH_CALUDE_medal_winners_combinations_l2706_270695

theorem medal_winners_combinations (semifinalists : ℕ) (eliminated : ℕ) (medals : ℕ) : 
  semifinalists = 8 →
  eliminated = 2 →
  medals = 3 →
  Nat.choose (semifinalists - eliminated) medals = 20 :=
by sorry

end NUMINAMATH_CALUDE_medal_winners_combinations_l2706_270695


namespace NUMINAMATH_CALUDE_cut_triangles_perimeter_sum_l2706_270686

theorem cut_triangles_perimeter_sum (large_perimeter hexagon_perimeter : ℝ) :
  large_perimeter = 60 →
  hexagon_perimeter = 40 →
  ∃ (x y z : ℝ),
    x + y + z = large_perimeter / 3 - hexagon_perimeter / 3 ∧
    3 * (x + y + z) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_cut_triangles_perimeter_sum_l2706_270686


namespace NUMINAMATH_CALUDE_cody_age_l2706_270617

theorem cody_age (grandmother_age : ℕ) (age_ratio : ℕ) (cody_age : ℕ) : 
  grandmother_age = 84 →
  grandmother_age = age_ratio * cody_age →
  age_ratio = 6 →
  cody_age = 14 := by
sorry

end NUMINAMATH_CALUDE_cody_age_l2706_270617


namespace NUMINAMATH_CALUDE_david_started_with_at_least_six_iphones_l2706_270631

/-- Represents the number of cell phones in various categories -/
structure CellPhoneInventory where
  samsung_end : ℕ
  iphone_end : ℕ
  samsung_damaged : ℕ
  iphone_defective : ℕ
  total_sold : ℕ

/-- Given the end-of-day inventory and sales data, proves that David started with at least 6 iPhones -/
theorem david_started_with_at_least_six_iphones 
  (inventory : CellPhoneInventory)
  (h1 : inventory.samsung_end = 10)
  (h2 : inventory.iphone_end = 5)
  (h3 : inventory.samsung_damaged = 2)
  (h4 : inventory.iphone_defective = 1)
  (h5 : inventory.total_sold = 4) :
  ∃ (initial_iphones : ℕ), initial_iphones ≥ 6 ∧ 
    initial_iphones ≥ inventory.iphone_end + inventory.iphone_defective :=
by sorry

end NUMINAMATH_CALUDE_david_started_with_at_least_six_iphones_l2706_270631


namespace NUMINAMATH_CALUDE_twenty_sixth_card_is_red_l2706_270642

-- Define the color type
inductive Color
  | Black
  | Red

-- Define the card sequence type
def CardSequence := Nat → Color

-- Define the property of no two consecutive cards being the same color
def AlternatingColors (seq : CardSequence) : Prop :=
  ∀ n : Nat, seq n ≠ seq (n + 1)

-- Define the problem conditions
def ProblemConditions (seq : CardSequence) : Prop :=
  AlternatingColors seq ∧
  seq 10 = Color.Red ∧
  seq 11 = Color.Red ∧
  seq 25 = Color.Black

-- State the theorem
theorem twenty_sixth_card_is_red (seq : CardSequence) 
  (h : ProblemConditions seq) : seq 26 = Color.Red := by
  sorry


end NUMINAMATH_CALUDE_twenty_sixth_card_is_red_l2706_270642


namespace NUMINAMATH_CALUDE_sum_of_three_integers_with_product_625_l2706_270644

theorem sum_of_three_integers_with_product_625 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 625 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 51 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_integers_with_product_625_l2706_270644


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2706_270647

def g (x : ℝ) := |x - 1| - |x - 2|

theorem empty_solution_set_range (a : ℝ) :
  (∀ x : ℝ, g x < a^2 + a + 1) →
  a ∈ Set.Ioi 0 ∪ Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2706_270647


namespace NUMINAMATH_CALUDE_trig_simplification_l2706_270671

theorem trig_simplification :
  (Real.cos (20 * π / 180) * Real.sqrt (1 - Real.cos (40 * π / 180))) / Real.cos (50 * π / 180) = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_simplification_l2706_270671


namespace NUMINAMATH_CALUDE_find_a_value_l2706_270694

noncomputable section

open Set Real

def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem find_a_value : ∃ (a : ℝ), 
  (∀ x ∈ (Ioo 0 2), StrictAntiOn (f a) (Ioo 0 2)) ∧
  (∀ x ∈ (Ioi 2), StrictMonoOn (f a) (Ioi 2)) ∧
  a = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_find_a_value_l2706_270694


namespace NUMINAMATH_CALUDE_parallelogram_angles_l2706_270646

/-- Given a parallelogram with perimeter to larger diagonal ratio k,
    where the larger diagonal divides one angle in the ratio 1:2,
    prove that its angles are 3 arccos((2+k)/(2k)) and π - 3 arccos((2+k)/(2k)). -/
theorem parallelogram_angles (k : ℝ) (k_pos : k > 0) :
  ∃ (angle₁ angle₂ : ℝ),
    angle₁ = 3 * Real.arccos ((2 + k) / (2 * k)) ∧
    angle₂ = Real.pi - 3 * Real.arccos ((2 + k) / (2 * k)) ∧
    angle₁ + angle₂ = Real.pi ∧
    angle₁ > 0 ∧ angle₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_angles_l2706_270646


namespace NUMINAMATH_CALUDE_unique_positive_number_l2706_270614

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x^2 + x = 210 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l2706_270614


namespace NUMINAMATH_CALUDE_triangle_exists_l2706_270698

/-- A triangle with given circumradius, centroid-circumcenter distance, and centroid-altitude distance --/
structure TriangleWithCircumcenter where
  r : ℝ  -- radius of circumscribed circle
  KS : ℝ  -- distance from circumcenter to centroid
  d : ℝ  -- distance from centroid to altitude

/-- Conditions for the existence of a triangle with given parameters --/
def triangle_existence_conditions (t : TriangleWithCircumcenter) : Prop :=
  t.d ≤ 2 * t.KS ∧ 
  t.r ≥ 3 * t.d / 2 ∧ 
  |Real.sqrt (4 * t.r^2 - 9 * t.d^2) - 3 * Real.sqrt (4 * t.KS^2 - t.d^2)| < 4 * t.r

/-- Theorem stating the existence of a triangle with given parameters --/
theorem triangle_exists (t : TriangleWithCircumcenter) : 
  triangle_existence_conditions t ↔ ∃ (triangle : Type), true :=
sorry

end NUMINAMATH_CALUDE_triangle_exists_l2706_270698


namespace NUMINAMATH_CALUDE_prob_two_red_faces_8x8x8_l2706_270690

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_length : ℕ
  total_cubes : ℕ
  edge_cubes : ℕ

/-- The probability of selecting a cube with exactly two red faces -/
def prob_two_red_faces (c : CutCube) : ℚ :=
  (12 * c.edge_cubes) / c.total_cubes

/-- Theorem: The probability of selecting a cube with exactly two red faces
    from a 8x8x8 cut cube is 9/64 -/
theorem prob_two_red_faces_8x8x8 :
  ∃ c : CutCube, c.side_length = 8 ∧ c.total_cubes = 512 ∧ c.edge_cubes = 6 ∧
  prob_two_red_faces c = 9/64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_faces_8x8x8_l2706_270690


namespace NUMINAMATH_CALUDE_pen_sales_profit_l2706_270637

/-- Calculates the total profit and profit percent for a pen sales scenario --/
def calculate_profit_and_percent (total_pens : ℕ) (marked_price : ℚ) 
  (discount_tier1 : ℚ) (discount_tier2 : ℚ) (discount_tier3 : ℚ)
  (pens_tier1 : ℕ) (pens_tier2 : ℕ)
  (sell_discount1 : ℚ) (sell_discount2 : ℚ)
  (pens_sold1 : ℕ) (pens_sold2 : ℕ) : ℚ × ℚ :=
  sorry

theorem pen_sales_profit :
  let total_pens : ℕ := 150
  let marked_price : ℚ := 240 / 100
  let discount_tier1 : ℚ := 5 / 100
  let discount_tier2 : ℚ := 10 / 100
  let discount_tier3 : ℚ := 15 / 100
  let pens_tier1 : ℕ := 50
  let pens_tier2 : ℕ := 50
  let sell_discount1 : ℚ := 4 / 100
  let sell_discount2 : ℚ := 2 / 100
  let pens_sold1 : ℕ := 75
  let pens_sold2 : ℕ := 75
  let (profit, percent) := calculate_profit_and_percent total_pens marked_price 
    discount_tier1 discount_tier2 discount_tier3
    pens_tier1 pens_tier2
    sell_discount1 sell_discount2
    pens_sold1 pens_sold2
  profit = 2520 / 100 ∧ abs (percent - 778 / 10000) < 1 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_pen_sales_profit_l2706_270637


namespace NUMINAMATH_CALUDE_center_sum_coords_l2706_270669

/-- Defines a circle with the equation x^2 + y^2 = 6x - 8y + 24 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x - 8*y + 24

/-- Defines the center of a circle -/
def is_center (h x y : ℝ) : Prop :=
  ∀ (a b : ℝ), circle_equation a b → (a - x)^2 + (b - y)^2 = h^2

theorem center_sum_coords :
  ∃ (x y : ℝ), is_center 7 x y ∧ x + y = -1 :=
sorry

end NUMINAMATH_CALUDE_center_sum_coords_l2706_270669


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l2706_270604

noncomputable def m (a x : ℝ) : ℝ × ℝ := (a * Real.cos x, Real.cos x)
noncomputable def n (b x : ℝ) : ℝ × ℝ := (2 * Real.cos x, b * Real.sin x)

noncomputable def f (a b x : ℝ) : ℝ := (m a x).1 * (n b x).1 + (m a x).2 * (n b x).2

theorem vector_dot_product_problem (a b : ℝ) :
  (∃ x, f a b x = 2) ∧ 
  (f a b (π/3) = 1/2 + Real.sqrt 3/2) →
  (∃ x_min ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f a b x_min ≤ f a b x) ∧
  (∃ x_max ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f a b x ≤ f a b x_max) ∧
  (∀ θ, 0 < θ ∧ θ < π ∧ f a b (θ/2) = 3/2 → Real.tan θ = -(4 + Real.sqrt 7)/3) :=
by sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l2706_270604


namespace NUMINAMATH_CALUDE_prob_king_hearts_or_spade_l2706_270608

-- Define the total number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of spades in the deck
def num_spades : ℕ := 13

-- Define the probability of drawing the King of Hearts
def prob_king_hearts : ℚ := 1 / total_cards

-- Define the probability of drawing a Spade
def prob_spade : ℚ := num_spades / total_cards

-- Theorem to prove
theorem prob_king_hearts_or_spade :
  prob_king_hearts + prob_spade = 7 / 26 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_hearts_or_spade_l2706_270608


namespace NUMINAMATH_CALUDE_school_population_l2706_270650

/-- Given a school population where:
  * b is the number of boys
  * g is the number of girls
  * t is the number of teachers
  * There are twice as many boys as girls
  * There are four times as many girls as teachers
Prove that the total population is 13t -/
theorem school_population (b g t : ℕ) (h1 : b = 2 * g) (h2 : g = 4 * t) :
  b + g + t = 13 * t := by
  sorry

end NUMINAMATH_CALUDE_school_population_l2706_270650


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_l2706_270632

/-- Calculates the net profit from a lemonade stand --/
theorem lemonade_stand_profit
  (glasses_per_gallon : ℕ)
  (cost_per_gallon : ℚ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ)
  (h1 : glasses_per_gallon = 16)
  (h2 : cost_per_gallon = 7/2)
  (h3 : gallons_made = 2)
  (h4 : price_per_glass = 1)
  (h5 : glasses_drunk = 5)
  (h6 : glasses_unsold = 6) :
  (gallons_made * glasses_per_gallon - glasses_drunk - glasses_unsold) * price_per_glass -
  (gallons_made * cost_per_gallon) = 14 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_stand_profit_l2706_270632


namespace NUMINAMATH_CALUDE_simplify_expression_l2706_270662

theorem simplify_expression (a : ℝ) (h : a ≠ 2) :
  (a - 2) * ((a^2 - 4) / (a^2 - 4*a + 4)) = a + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2706_270662


namespace NUMINAMATH_CALUDE_operation_result_l2706_270600

def operation (n : ℕ) : ℕ := 2 * n + 1

def iterate_operation (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => operation (iterate_operation n k)

theorem operation_result (x : ℕ) :
  ¬(∃ (y : ℕ), iterate_operation x 100 = 1980 * y) ∧
  (∃ (x : ℕ), ∃ (y : ℕ), iterate_operation x 100 = 1981 * y) := by
  sorry


end NUMINAMATH_CALUDE_operation_result_l2706_270600


namespace NUMINAMATH_CALUDE_problem_solution_l2706_270616

-- Define proposition p
def p : Prop := ∀ a : ℝ, ∃ x : ℝ, a^(x + 1) = 1 ∧ x = 0

-- Define proposition q
def q : Prop := ∀ f : ℝ → ℝ, (∀ x : ℝ, f x = f (-x)) → 
  (∀ x : ℝ, f (x + 1) = f (2 - x))

-- Theorem statement
theorem problem_solution : (¬p ∧ ¬q) → (p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2706_270616


namespace NUMINAMATH_CALUDE_f_properties_l2706_270676

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * log x

theorem f_properties (m : ℝ) (h : m ≥ 1) :
  (∃! (x : ℝ), x > 0 ∧ f m x = x^2 - (m + 1) * x) ∧
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f m x ≤ f m y) ∧
  (∃ (x : ℝ), x > 0 ∧ f m x = (m/2) * (1 - log m)) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l2706_270676


namespace NUMINAMATH_CALUDE_lastDigitOf8To19_eq_2_l2706_270619

/-- The last digit of 2^n for n > 0 -/
def lastDigitOfPowerOf2 (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 6

/-- The last digit of 8^19 -/
def lastDigitOf8To19 : ℕ :=
  lastDigitOfPowerOf2 57

theorem lastDigitOf8To19_eq_2 : lastDigitOf8To19 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lastDigitOf8To19_eq_2_l2706_270619


namespace NUMINAMATH_CALUDE_sin_inequality_l2706_270689

open Real

theorem sin_inequality (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) :
  let f := fun θ => (sin θ)^3 / (2 * θ - sin (2 * θ))
  f α > f β := by
sorry

end NUMINAMATH_CALUDE_sin_inequality_l2706_270689


namespace NUMINAMATH_CALUDE_christines_second_dog_weight_l2706_270643

/-- The weight of Christine's second dog -/
def weight_second_dog (cat_weights : List ℕ) (additional_weight : ℕ) : ℕ :=
  let total_cat_weight := cat_weights.sum
  let first_dog_weight := total_cat_weight + additional_weight
  2 * (first_dog_weight - total_cat_weight)

/-- Theorem: Christine's second dog weighs 16 pounds -/
theorem christines_second_dog_weight :
  weight_second_dog [7, 10, 13] 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_christines_second_dog_weight_l2706_270643


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l2706_270626

theorem algebraic_expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (a + 1) / a / (a - (1 + 2 * a^2) / (3 * a)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l2706_270626


namespace NUMINAMATH_CALUDE_triangle_radii_relations_l2706_270663

/-- Triangle properties -/
class Triangle (α : Type*) [LinearOrderedField α] :=
  (a b c : α)
  (t : α)
  (s : α)
  (ρ : α)
  (ρa ρb ρc : α)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)
  (semiperimeter : s = (a + b + c) / 2)
  (area_positive : 0 < t)

/-- Theorem about relationships between inradius, exradii, semiperimeter, and area of a triangle -/
theorem triangle_radii_relations {α : Type*} [LinearOrderedField α] (T : Triangle α) :
  T.ρa * T.ρb + T.ρb * T.ρc + T.ρc * T.ρa = T.s^2 ∧
  1 / T.ρ = 1 / T.ρa + 1 / T.ρb + 1 / T.ρc ∧
  T.ρ * T.ρa * T.ρb * T.ρc = T.t^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_radii_relations_l2706_270663


namespace NUMINAMATH_CALUDE_composite_function_equation_l2706_270692

theorem composite_function_equation (δ φ : ℝ → ℝ) (x : ℝ) 
  (h1 : ∀ y, δ y = 4 * y + 9)
  (h2 : ∀ y, φ y = 9 * y + 8)
  (h3 : δ (φ x) = 11) :
  x = -5/6 := by
sorry

end NUMINAMATH_CALUDE_composite_function_equation_l2706_270692


namespace NUMINAMATH_CALUDE_max_npn_value_l2706_270624

def is_two_digit_same_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ (n / 10 = n % 10)

def is_one_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def is_three_digit_npn (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

theorem max_npn_value :
  ∀ (mm m npn : ℕ),
    is_two_digit_same_digits mm →
    is_one_digit m →
    is_three_digit_npn npn →
    mm * m = npn →
    npn ≤ 729 :=
sorry

end NUMINAMATH_CALUDE_max_npn_value_l2706_270624


namespace NUMINAMATH_CALUDE_wednesday_tips_calculation_l2706_270629

/-- Represents Hallie's work data for a day -/
structure WorkDay where
  hours : ℕ
  tips : ℕ

/-- Calculates the total earnings for a given work day with an hourly rate -/
def dailyEarnings (day : WorkDay) (hourlyRate : ℕ) : ℕ :=
  day.hours * hourlyRate + day.tips

theorem wednesday_tips_calculation (hourlyRate : ℕ) (monday tuesday wednesday : WorkDay) 
    (totalEarnings : ℕ) : 
    hourlyRate = 10 →
    monday.hours = 7 →
    monday.tips = 18 →
    tuesday.hours = 5 →
    tuesday.tips = 12 →
    wednesday.hours = 7 →
    totalEarnings = 240 →
    totalEarnings = dailyEarnings monday hourlyRate + 
                    dailyEarnings tuesday hourlyRate + 
                    dailyEarnings wednesday hourlyRate →
    wednesday.tips = 20 := by
  sorry

#check wednesday_tips_calculation

end NUMINAMATH_CALUDE_wednesday_tips_calculation_l2706_270629


namespace NUMINAMATH_CALUDE_xfx_nonnegative_set_l2706_270611

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem xfx_nonnegative_set (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_monotone : monotone_decreasing_on f (Set.Iic 0))
  (h_f2 : f 2 = 0) :
  {x : ℝ | x * f x ≥ 0} = Set.Icc (-2) 0 ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_xfx_nonnegative_set_l2706_270611


namespace NUMINAMATH_CALUDE_log_xy_value_l2706_270681

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x * y) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l2706_270681


namespace NUMINAMATH_CALUDE_xixi_apples_count_l2706_270683

/-- The number of students in Teacher Xixi's class -/
def xixi_students : ℕ := 12

/-- The number of students in Teacher Shanshan's class -/
def shanshan_students : ℕ := xixi_students

/-- The number of apples Teacher Xixi prepared -/
def xixi_apples : ℕ := 72

/-- The number of oranges Teacher Shanshan prepared -/
def shanshan_oranges : ℕ := 60

theorem xixi_apples_count : xixi_apples = 72 := by
  have h1 : xixi_apples = shanshan_students * 6 := sorry
  have h2 : shanshan_oranges = xixi_students * 3 + 12 := sorry
  have h3 : shanshan_oranges = shanshan_students * 5 := sorry
  sorry

end NUMINAMATH_CALUDE_xixi_apples_count_l2706_270683


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l2706_270653

-- Define the type for 2D vectors
def Vector2D := ℝ × ℝ

-- Define vector addition
def add (v w : Vector2D) : Vector2D :=
  (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def smul (r : ℝ) (v : Vector2D) : Vector2D :=
  (r * v.1, r * v.2)

-- Define dot product
def dot (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_problem (a b : Vector2D) 
  (h1 : add (smul 2 a) b = (1, 6)) 
  (h2 : add a (smul 2 b) = (-4, 9)) : 
  dot a b = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l2706_270653


namespace NUMINAMATH_CALUDE_test_score_ratio_l2706_270618

theorem test_score_ratio (total_questions : ℕ) (score : ℕ) (correct_answers : ℕ)
  (h1 : total_questions = 100)
  (h2 : score = 79)
  (h3 : correct_answers = 93)
  (h4 : correct_answers ≤ total_questions) :
  (total_questions - correct_answers) / correct_answers = 7 / 93 := by
sorry

end NUMINAMATH_CALUDE_test_score_ratio_l2706_270618


namespace NUMINAMATH_CALUDE_final_positions_l2706_270661

structure Person where
  cash : Int
  has_car : Bool

def initial_c : Person := { cash := 15000, has_car := true }
def initial_d : Person := { cash := 17000, has_car := false }

def transaction1 (c d : Person) : Person × Person :=
  ({ cash := c.cash + 16000, has_car := false },
   { cash := d.cash - 16000, has_car := true })

def transaction2 (c d : Person) : Person × Person :=
  ({ cash := c.cash - 14000, has_car := true },
   { cash := d.cash + 14000, has_car := false })

def transaction3 (c d : Person) : Person × Person :=
  ({ cash := c.cash + 15500, has_car := false },
   { cash := d.cash - 15500, has_car := true })

theorem final_positions :
  let (c1, d1) := transaction1 initial_c initial_d
  let (c2, d2) := transaction2 c1 d1
  let (c3, d3) := transaction3 c2 d2
  c3.cash = 32500 ∧ ¬c3.has_car ∧ d3.cash = -500 ∧ d3.has_car := by
  sorry

end NUMINAMATH_CALUDE_final_positions_l2706_270661


namespace NUMINAMATH_CALUDE_sector_area_l2706_270656

theorem sector_area (α : Real) (l : Real) (S : Real) :
  α = π / 9 →
  l = π / 3 →
  S = (1 / 2) * l * (l / α) →
  S = π / 2 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l2706_270656


namespace NUMINAMATH_CALUDE_parabola_directrix_l2706_270654

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2 + 4

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = 63 / 16

/-- Theorem: The directrix of the parabola y = 4x^2 + 4 is y = 63/16 -/
theorem parabola_directrix : ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p q : ℝ × ℝ, p.2 = 4 * p.1^2 + 4 → 
    (p.1 - x)^2 + (p.2 - y)^2 = (p.2 - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2706_270654


namespace NUMINAMATH_CALUDE_total_distance_is_490_l2706_270667

/-- Represents a segment of the journey -/
structure JourneySegment where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled in a journey segment -/
def distanceTraveled (segment : JourneySegment) : ℝ :=
  segment.speed * segment.time

/-- Represents the entire journey -/
def Journey : List JourneySegment := [
  { speed := 90, time := 2 },
  { speed := 60, time := 1 },
  { speed := 100, time := 2.5 }
]

/-- Theorem: The total distance traveled in the journey is 490 km -/
theorem total_distance_is_490 : 
  (Journey.map distanceTraveled).sum = 490 := by sorry

end NUMINAMATH_CALUDE_total_distance_is_490_l2706_270667


namespace NUMINAMATH_CALUDE_inverse_proposition_reciprocals_l2706_270603

/-- The inverse proposition of "If ab = 1, then a and b are reciprocals" -/
theorem inverse_proposition_reciprocals (a b : ℝ) :
  (a ≠ 0 ∧ b ≠ 0 ∧ a * b = 1 → a = 1 / b ∧ b = 1 / a) →
  (a = 1 / b ∧ b = 1 / a → a * b = 1) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_reciprocals_l2706_270603


namespace NUMINAMATH_CALUDE_rubber_band_calculation_l2706_270672

/-- The number of rubber bands in a small ball -/
def small_ball_rubber_bands : ℕ := 50

/-- The number of rubber bands in a large ball -/
def large_ball_rubber_bands : ℕ := 300

/-- The total number of rubber bands -/
def total_rubber_bands : ℕ := 5000

/-- The number of small balls made -/
def small_balls_made : ℕ := 22

/-- The number of large balls that can be made with remaining rubber bands -/
def large_balls_possible : ℕ := 13

theorem rubber_band_calculation :
  small_ball_rubber_bands * small_balls_made +
  large_ball_rubber_bands * large_balls_possible = total_rubber_bands :=
by sorry

end NUMINAMATH_CALUDE_rubber_band_calculation_l2706_270672


namespace NUMINAMATH_CALUDE_parabola_midpoint_distance_l2706_270630

/-- Given a parabola y² = 4x and a line passing through its focus,
    intersecting the parabola at points A and B with |AB| = 7,
    the distance from the midpoint M of AB to the directrix is 7/2. -/
theorem parabola_midpoint_distance (x₁ y₁ x₂ y₂ : ℝ) :
  y₁^2 = 4*x₁ →
  y₂^2 = 4*x₂ →
  (x₁ - 1)^2 + y₁^2 = (x₂ - 1)^2 + y₂^2 →  -- line passes through focus (1, 0)
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 49 →         -- |AB| = 7
  (((x₁ + x₂)/2 + 1) : ℝ) = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_distance_l2706_270630


namespace NUMINAMATH_CALUDE_group_average_before_new_member_l2706_270633

theorem group_average_before_new_member (group : Finset ℕ) (group_sum : ℕ) (new_member : ℕ) :
  Finset.card group = 7 →
  group_sum / Finset.card group = 20 →
  new_member = 56 →
  group_sum / Finset.card group = 20 := by
sorry

end NUMINAMATH_CALUDE_group_average_before_new_member_l2706_270633


namespace NUMINAMATH_CALUDE_positive_solution_between_one_and_two_l2706_270635

def f (x : ℝ) := x^2 + 3*x - 5

theorem positive_solution_between_one_and_two :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  have h1 : f 1 < 0 := by sorry
  have h2 : f 2 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_positive_solution_between_one_and_two_l2706_270635


namespace NUMINAMATH_CALUDE_joeys_rope_length_l2706_270648

/-- Given that the ratio of Joey's rope length to Chad's rope length is 8:3,
    and Chad's rope is 21 cm long, prove that Joey's rope is 56 cm long. -/
theorem joeys_rope_length (chad_rope_length : ℝ) (ratio : ℚ) :
  chad_rope_length = 21 →
  ratio = 8 / 3 →
  ∃ joey_rope_length : ℝ,
    joey_rope_length / chad_rope_length = ratio ∧
    joey_rope_length = 56 :=
by sorry

end NUMINAMATH_CALUDE_joeys_rope_length_l2706_270648


namespace NUMINAMATH_CALUDE_circle_radius_decrease_l2706_270627

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let original_area := π * r^2
  let new_area := 0.58 * original_area
  let new_radius := Real.sqrt (new_area / π)
  (r - new_radius) / r = 1 - Real.sqrt 0.58 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_decrease_l2706_270627


namespace NUMINAMATH_CALUDE_max_value_on_interval_l2706_270665

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 8*x^2 + 2

-- State the theorem
theorem max_value_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 3 ∧ 
  (∀ x, x ∈ Set.Icc (-1) 3 → f x ≤ f c) ∧
  f c = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l2706_270665


namespace NUMINAMATH_CALUDE_green_ball_count_l2706_270679

/-- Represents the number of balls of each color --/
structure BallCount where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The conditions of the problem --/
def validBallCount (bc : BallCount) : Prop :=
  bc.red + bc.blue + bc.green = 50 ∧
  ∀ (subset : ℕ), subset ≤ 50 → subset ≥ 34 → bc.red > 0 ∧
  ∀ (subset : ℕ), subset ≤ 50 → subset ≥ 35 → bc.blue > 0 ∧
  ∀ (subset : ℕ), subset ≤ 50 → subset ≥ 36 → bc.green > 0

/-- The theorem to be proved --/
theorem green_ball_count (bc : BallCount) (h : validBallCount bc) :
  15 ≤ bc.green ∧ bc.green ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_count_l2706_270679


namespace NUMINAMATH_CALUDE_eric_sibling_product_l2706_270673

/-- Represents a family with a given number of sisters and brothers -/
structure Family where
  sisters : ℕ
  brothers : ℕ

/-- Calculates the number of sisters and brothers for a sibling in the family -/
def siblingCounts (f : Family) : ℕ × ℕ :=
  (f.sisters + 1, f.brothers)

theorem eric_sibling_product (emmas_family : Family)
    (h1 : emmas_family.sisters = 4)
    (h2 : emmas_family.brothers = 6) :
    let (S, B) := siblingCounts emmas_family
    S * B = 30 := by
  sorry

end NUMINAMATH_CALUDE_eric_sibling_product_l2706_270673


namespace NUMINAMATH_CALUDE_complex_expression_odd_exponent_l2706_270657

theorem complex_expression_odd_exponent (n : ℕ) (h : Odd n) :
  (((1 + Complex.I) / (1 - Complex.I)) ^ (2 * n) + 
   ((1 - Complex.I) / (1 + Complex.I)) ^ (2 * n)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_odd_exponent_l2706_270657


namespace NUMINAMATH_CALUDE_journey_takes_eight_hours_l2706_270601

/-- Represents the journey with three people A, B, and C --/
structure Journey where
  totalDistance : ℝ
  carSpeed : ℝ
  walkSpeed : ℝ
  t1 : ℝ  -- time A and C drive together
  t2 : ℝ  -- time A drives back
  t3 : ℝ  -- time A and B drive while C walks

/-- The conditions of the journey --/
def journeyConditions (j : Journey) : Prop :=
  j.totalDistance = 100 ∧
  j.carSpeed = 25 ∧
  j.walkSpeed = 5 ∧
  j.carSpeed * j.t1 - j.carSpeed * j.t2 + j.carSpeed * j.t3 = j.totalDistance ∧
  j.walkSpeed * j.t1 + j.walkSpeed * j.t2 + j.carSpeed * j.t3 = j.totalDistance ∧
  j.carSpeed * j.t1 + j.walkSpeed * j.t2 + j.walkSpeed * j.t3 = j.totalDistance

/-- The theorem stating that the journey takes 8 hours --/
theorem journey_takes_eight_hours (j : Journey) (h : journeyConditions j) :
  j.t1 + j.t2 + j.t3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_journey_takes_eight_hours_l2706_270601


namespace NUMINAMATH_CALUDE_root_sum_cube_product_l2706_270658

theorem root_sum_cube_product (α β : ℝ) : 
  α^2 - 2*α - 4 = 0 → β^2 - 2*β - 4 = 0 → α ≠ β → α^3 + 8*β + 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_cube_product_l2706_270658


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2023rd_term_l2706_270699

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2023rd_term 
  (p q : ℝ) 
  (h1 : arithmetic_sequence p (3*p - q - p) 1 = p)
  (h2 : arithmetic_sequence p (3*p - q - p) 2 = 3*p - q)
  (h3 : arithmetic_sequence p (3*p - q - p) 3 = 9)
  (h4 : arithmetic_sequence p (3*p - q - p) 4 = 3*p + q) :
  arithmetic_sequence p (3*p - q - p) 2023 = 18189 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2023rd_term_l2706_270699


namespace NUMINAMATH_CALUDE_treys_total_time_l2706_270639

/-- Represents the number of tasks for cleaning the house -/
def cleaning_tasks : ℕ := 7

/-- Represents the number of tasks for taking a shower -/
def shower_tasks : ℕ := 1

/-- Represents the number of tasks for making dinner -/
def dinner_tasks : ℕ := 4

/-- Represents the time in minutes required for each task -/
def time_per_task : ℕ := 10

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating that the total time to complete Trey's list is 2 hours -/
theorem treys_total_time :
  (cleaning_tasks + shower_tasks + dinner_tasks) * time_per_task / minutes_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_treys_total_time_l2706_270639


namespace NUMINAMATH_CALUDE_same_score_probability_l2706_270645

theorem same_score_probability (p_A p_B : ℝ) 
  (h_A : p_A = 0.6) 
  (h_B : p_B = 0.8) 
  (h_independent : True) -- Representing independence of events
  (h_score : ℕ → ℝ) 
  (h_score_success : h_score 1 = 2) 
  (h_score_fail : h_score 0 = 0) : 
  p_A * p_B + (1 - p_A) * (1 - p_B) = 0.56 := by
sorry

end NUMINAMATH_CALUDE_same_score_probability_l2706_270645


namespace NUMINAMATH_CALUDE_sine_increasing_omega_range_l2706_270628

/-- Given that y = sin(ωx) is increasing on the interval [-π/3, π/3], 
    the range of values for ω is (0, 3/2]. -/
theorem sine_increasing_omega_range (ω : ℝ) : 
  (∀ x ∈ Set.Icc (-π/3) (π/3), 
    Monotone (fun x => Real.sin (ω * x))) → 
  ω ∈ Set.Ioo 0 (3/2) :=
sorry

end NUMINAMATH_CALUDE_sine_increasing_omega_range_l2706_270628


namespace NUMINAMATH_CALUDE_total_frogs_is_48_l2706_270634

/-- The number of frogs in Pond A -/
def frogs_in_pond_a : ℕ := 32

/-- The number of frogs in Pond B -/
def frogs_in_pond_b : ℕ := frogs_in_pond_a / 2

/-- The total number of frogs in both ponds -/
def total_frogs : ℕ := frogs_in_pond_a + frogs_in_pond_b

theorem total_frogs_is_48 : total_frogs = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_frogs_is_48_l2706_270634


namespace NUMINAMATH_CALUDE_letter_F_perimeter_is_19_l2706_270609

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of the letter F given the specified conditions -/
def letter_F_perimeter (large : Rectangle) (small : Rectangle) (offset : ℝ) : ℝ :=
  2 * large.height + -- vertical sides of large rectangle
  (large.width - small.width) + -- uncovered top of large rectangle
  small.width -- bottom of small rectangle

/-- Theorem stating that the perimeter of the letter F is 19 inches -/
theorem letter_F_perimeter_is_19 :
  let large : Rectangle := { width := 2, height := 6 }
  let small : Rectangle := { width := 2, height := 2 }
  let offset : ℝ := 1
  letter_F_perimeter large small offset = 19 := by
  sorry

#eval letter_F_perimeter { width := 2, height := 6 } { width := 2, height := 2 } 1

end NUMINAMATH_CALUDE_letter_F_perimeter_is_19_l2706_270609


namespace NUMINAMATH_CALUDE_pencil_distribution_l2706_270697

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (max_students : ℕ) :
  num_pens = 2010 →
  max_students = 30 →
  num_pens % max_students = 0 →
  num_pencils % max_students = 0 →
  ∃ k : ℕ, num_pencils = 30 * k :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2706_270697


namespace NUMINAMATH_CALUDE_negation_of_all_cars_are_fast_l2706_270670

variable (U : Type) -- Universe of discourse
variable (car : U → Prop) -- Predicate for being a car
variable (fast : U → Prop) -- Predicate for being fast

theorem negation_of_all_cars_are_fast :
  ¬(∀ x, car x → fast x) ↔ ∃ x, car x ∧ ¬(fast x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_cars_are_fast_l2706_270670


namespace NUMINAMATH_CALUDE_original_number_proof_l2706_270666

theorem original_number_proof (x : ℝ) : 3 * (2 * x + 5) = 123 ↔ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2706_270666


namespace NUMINAMATH_CALUDE_marion_score_l2706_270696

/-- Given a 40-item exam, prove Marion's score based on Ella's performance -/
theorem marion_score (total_items : ℕ) (ella_incorrect : ℕ) (marion_bonus : ℕ) :
  total_items = 40 →
  ella_incorrect = 4 →
  marion_bonus = 6 →
  (total_items - ella_incorrect) / 2 + marion_bonus = 24 := by
  sorry

#check marion_score

end NUMINAMATH_CALUDE_marion_score_l2706_270696


namespace NUMINAMATH_CALUDE_sum_O_eq_1000_l2706_270684

/-- O(n) is the sum of the odd digits of n -/
def O (n : ℕ) : ℕ := sorry

/-- The sum of O(n) for n from 1 to 200 -/
def sum_O : ℕ := (Finset.range 200).sum (fun i => O (i + 1))

/-- Theorem: The sum of O(n) for n from 1 to 200 is equal to 1000 -/
theorem sum_O_eq_1000 : sum_O = 1000 := by sorry

end NUMINAMATH_CALUDE_sum_O_eq_1000_l2706_270684


namespace NUMINAMATH_CALUDE_max_ships_on_board_l2706_270640

/-- Represents a ship placement on a board -/
structure ShipPlacement where
  board_size : Nat × Nat
  ship_size : Nat × Nat
  ship_count : Nat

/-- Checks if a ship placement is valid -/
def is_valid_placement (p : ShipPlacement) : Prop :=
  p.board_size.1 = 10 ∧
  p.board_size.2 = 10 ∧
  p.ship_size.1 = 1 ∧
  p.ship_size.2 = 4 ∧
  p.ship_count ≤ 25

/-- Theorem stating the maximum number of ships -/
theorem max_ships_on_board :
  ∃ (p : ShipPlacement), is_valid_placement p ∧
    ∀ (q : ShipPlacement), is_valid_placement q → q.ship_count ≤ p.ship_count :=
sorry

end NUMINAMATH_CALUDE_max_ships_on_board_l2706_270640


namespace NUMINAMATH_CALUDE_expand_product_l2706_270688

theorem expand_product (x : ℝ) : 2 * (x + 2) * (x + 3) * (x + 4) = 2 * x^3 + 18 * x^2 + 52 * x + 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2706_270688


namespace NUMINAMATH_CALUDE_power_exceeds_thresholds_l2706_270682

theorem power_exceeds_thresholds : ∃ (n1 n2 n3 m1 m2 m3 : ℕ), 
  (1.01 : ℝ) ^ n1 > 1000000000000 ∧
  (1.001 : ℝ) ^ n2 > 1000000000000 ∧
  (1.000001 : ℝ) ^ n3 > 1000000000000 ∧
  (1.01 : ℝ) ^ m1 > 1000000000000000000 ∧
  (1.001 : ℝ) ^ m2 > 1000000000000000000 ∧
  (1.000001 : ℝ) ^ m3 > 1000000000000000000 :=
by sorry

end NUMINAMATH_CALUDE_power_exceeds_thresholds_l2706_270682


namespace NUMINAMATH_CALUDE_cover_triangles_l2706_270675

/-- The side length of the small equilateral triangle -/
def small_side : ℝ := 0.5

/-- The side length of the large equilateral triangle -/
def large_side : ℝ := 10

/-- The minimum number of small triangles needed to cover the large triangle -/
def min_triangles : ℕ := 400

theorem cover_triangles : 
  ∀ (n : ℕ), n * (small_side^2 * Real.sqrt 3 / 4) ≥ large_side^2 * Real.sqrt 3 / 4 → n ≥ min_triangles :=
by sorry

end NUMINAMATH_CALUDE_cover_triangles_l2706_270675


namespace NUMINAMATH_CALUDE_min_max_cubic_linear_exists_y_min_max_zero_min_max_value_is_zero_l2706_270612

theorem min_max_cubic_linear (y : ℝ) : 
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ |x^3 - x*y| = 0) ∨ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y| > 0) :=
sorry

theorem exists_y_min_max_zero : 
  ∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y| ≤ 0 :=
sorry

theorem min_max_value_is_zero : 
  ∃ (y : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y| ≤ 0) ∧ 
  (∀ (y' : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y'| ≤ 0) → y' = y) :=
sorry

end NUMINAMATH_CALUDE_min_max_cubic_linear_exists_y_min_max_zero_min_max_value_is_zero_l2706_270612


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2706_270691

theorem geometric_sequence_sum (a₁ a₆ a₈ : ℚ) (h₁ : a₁ = 4096) (h₆ : a₆ = 4) (h₈ : a₈ = 1/4) :
  let r := (a₆ / a₁) ^ (1/5)
  let a₄ := a₁ * r^3
  let a₅ := a₁ * r^4
  a₄ + a₅ = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2706_270691


namespace NUMINAMATH_CALUDE_right_pyramid_surface_area_l2706_270680

/-- Represents a right pyramid with a parallelogram base -/
structure RightPyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_angle : ℝ
  height : ℝ

/-- Calculates the total surface area of a right pyramid -/
def total_surface_area (p : RightPyramid) : ℝ :=
  sorry

theorem right_pyramid_surface_area :
  let p := RightPyramid.mk 12 14 (π / 3) 15
  total_surface_area p = 168 * Real.sqrt 3 + 216 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_right_pyramid_surface_area_l2706_270680


namespace NUMINAMATH_CALUDE_range_of_k_in_linear_system_l2706_270655

/-- Given a system of linear equations and an inequality constraint,
    prove the range of the parameter k. -/
theorem range_of_k_in_linear_system (x y k : ℝ) :
  (2 * x - y = k + 1) →
  (x - y = -3) →
  (x + y > 2) →
  k > -4.5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_in_linear_system_l2706_270655


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2706_270649

theorem system_solution_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hc : c ≠ 0)
  (eq1 : 8 * x - 5 * y = c) (eq2 : 10 * y - 16 * x = d) : d / c = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2706_270649


namespace NUMINAMATH_CALUDE_prob_intersects_inner_is_one_third_l2706_270652

/-- Two concentric circles with radii 1 and 2 -/
structure ConcentricCircles where
  inner_radius : ℝ := 1
  outer_radius : ℝ := 2

/-- A chord on the outer circle -/
structure Chord where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Function to determine if a chord intersects the inner circle -/
def intersects_inner_circle (c : ConcentricCircles) (ch : Chord) : Prop :=
  sorry

/-- Function to calculate the probability of a random chord intersecting the inner circle -/
noncomputable def probability_intersects_inner (c : ConcentricCircles) : ℝ :=
  sorry

/-- Theorem stating that the probability of a random chord intersecting the inner circle is 1/3 -/
theorem prob_intersects_inner_is_one_third (c : ConcentricCircles) :
  probability_intersects_inner c = 1/3 :=
sorry

end NUMINAMATH_CALUDE_prob_intersects_inner_is_one_third_l2706_270652
