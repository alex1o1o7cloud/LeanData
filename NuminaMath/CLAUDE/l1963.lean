import Mathlib

namespace NUMINAMATH_CALUDE_probability_of_graduate_degree_l1963_196339

theorem probability_of_graduate_degree (G C N : ℕ) : 
  G * 8 = N →
  C * 3 = N * 2 →
  (G : ℚ) / (G + C) = 3 / 19 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_graduate_degree_l1963_196339


namespace NUMINAMATH_CALUDE_science_fair_participants_l1963_196398

/-- The number of unique students participating in the Science Fair --/
def unique_students (robotics astronomy chemistry all_three : ℕ) : ℕ :=
  robotics + astronomy + chemistry - 2 * all_three

/-- Theorem stating the number of unique students in the Science Fair --/
theorem science_fair_participants : unique_students 15 10 12 2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_science_fair_participants_l1963_196398


namespace NUMINAMATH_CALUDE_inequality_1_inequality_2_l1963_196366

-- First inequality
theorem inequality_1 (x : ℝ) : (2*x - 1)/3 - (9*x + 2)/6 ≤ 1 ↔ x ≥ -2 := by sorry

-- Second system of inequalities
theorem inequality_2 (x : ℝ) : 
  (x - 3*(x - 2) ≥ 4 ∧ (2*x - 1)/5 < (x + 1)/2) ↔ -7 < x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_1_inequality_2_l1963_196366


namespace NUMINAMATH_CALUDE_log_order_l1963_196322

theorem log_order (a b c : ℝ) (ha : a = Real.log 3 / Real.log 4)
  (hb : b = Real.log 4 / Real.log 3) (hc : c = Real.log (4/3) / Real.log (3/4)) :
  b > a ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_log_order_l1963_196322


namespace NUMINAMATH_CALUDE_stocking_stuffers_l1963_196317

theorem stocking_stuffers (num_kids : ℕ) (candy_canes_per_stocking : ℕ) (beanie_babies_per_stocking : ℕ) (total_stuffers : ℕ) : 
  num_kids = 3 → 
  candy_canes_per_stocking = 4 → 
  beanie_babies_per_stocking = 2 → 
  total_stuffers = 21 → 
  (total_stuffers - (candy_canes_per_stocking + beanie_babies_per_stocking) * num_kids) / num_kids = 1 :=
by sorry

end NUMINAMATH_CALUDE_stocking_stuffers_l1963_196317


namespace NUMINAMATH_CALUDE_raviraj_journey_l1963_196309

def journey (initial_south distance_after_first_turn second_north final_west distance_to_home : ℝ) : Prop :=
  initial_south = 20 ∧
  second_north = 20 ∧
  final_west = 20 ∧
  distance_to_home = 30 ∧
  distance_after_first_turn + final_west = distance_to_home

theorem raviraj_journey :
  ∀ initial_south distance_after_first_turn second_north final_west distance_to_home,
    journey initial_south distance_after_first_turn second_north final_west distance_to_home →
    distance_after_first_turn = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_raviraj_journey_l1963_196309


namespace NUMINAMATH_CALUDE_gcd_count_for_product_180_l1963_196316

theorem gcd_count_for_product_180 : 
  ∃ (S : Finset ℕ), 
    (∀ a b : ℕ, a > 0 → b > 0 → Nat.gcd a b * Nat.lcm a b = 180 → 
      Nat.gcd a b ∈ S) ∧ 
    (∀ n ∈ S, ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ Nat.gcd a b * Nat.lcm a b = 180 ∧ 
      Nat.gcd a b = n) ∧
    Finset.card S = 7 :=
by sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_180_l1963_196316


namespace NUMINAMATH_CALUDE_work_completion_time_l1963_196301

/-- Given two workers a and b, where:
    1. a and b can finish the work together in 30 days
    2. a alone can finish the work in 60 days
    3. a and b worked together for 20 days before b left
    This theorem proves that a finishes the remaining work in 20 days after b left. -/
theorem work_completion_time 
  (total_work : ℝ) 
  (a_rate : ℝ) 
  (b_rate : ℝ) 
  (h1 : a_rate + b_rate = total_work / 30)
  (h2 : a_rate = total_work / 60)
  (h3 : (a_rate + b_rate) * 20 = 2 * total_work / 3) :
  (total_work / 3) / a_rate = 20 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l1963_196301


namespace NUMINAMATH_CALUDE_vectors_perpendicular_distance_AB_l1963_196327

-- Define the line and parabola
def line (x y : ℝ) : Prop := y = x - 2
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define points A and B as intersections
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define O as the origin
def O : ℝ × ℝ := (0, 0)

-- Vector from O to A
def OA : ℝ × ℝ := (A.1 - O.1, A.2 - O.2)

-- Vector from O to B
def OB : ℝ × ℝ := (B.1 - O.1, B.2 - O.2)

-- Theorem 1: OA ⊥ OB
theorem vectors_perpendicular : OA.1 * OB.1 + OA.2 * OB.2 = 0 := by sorry

-- Theorem 2: |AB| = 2√10
theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_distance_AB_l1963_196327


namespace NUMINAMATH_CALUDE_no_solution_for_certain_a_l1963_196341

-- Define the equation
def equation (x a : ℝ) : ℝ := 6 * abs (x - 4*a) + abs (x - a^2) + 5*x - 4*a

-- State the theorem
theorem no_solution_for_certain_a :
  ∀ a : ℝ, (a < -12 ∨ a > 0) → ¬∃ x : ℝ, equation x a = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_certain_a_l1963_196341


namespace NUMINAMATH_CALUDE_min_extracted_tablets_l1963_196374

/-- Represents the contents of a medicine box -/
structure MedicineBox where
  tabletA : Nat
  tabletB : Nat

/-- Represents the minimum number of tablets extracted -/
structure ExtractedTablets where
  minA : Nat
  minB : Nat

/-- Given a medicine box with 10 tablets of each kind and a minimum extraction of 12 tablets,
    proves that the minimum number of tablets of each kind among the extracted is 2 for A and 1 for B -/
theorem min_extracted_tablets (box : MedicineBox) (min_extraction : Nat) :
  box.tabletA = 10 → box.tabletB = 10 → min_extraction = 12 →
  ∃ (extracted : ExtractedTablets),
    extracted.minA = 2 ∧ extracted.minB = 1 ∧
    extracted.minA + extracted.minB ≤ min_extraction ∧
    extracted.minA ≤ box.tabletA ∧ extracted.minB ≤ box.tabletB := by
  sorry

end NUMINAMATH_CALUDE_min_extracted_tablets_l1963_196374


namespace NUMINAMATH_CALUDE_final_red_probability_zero_l1963_196302

/-- Represents the color of a marble -/
inductive Color
| Red
| Blue

/-- Represents the state of the jar -/
structure JarState :=
  (red : Nat)
  (blue : Nat)

/-- Represents the result of drawing two marbles -/
inductive DrawResult
| SameColor (c : Color)
| DifferentColors

/-- Simulates drawing two marbles from the jar -/
def draw (state : JarState) : DrawResult := sorry

/-- Updates the jar state based on the draw result -/
def updateJar (state : JarState) (result : DrawResult) : JarState := sorry

/-- Simulates the entire process of drawing and updating three times -/
def process (initialState : JarState) : JarState := sorry

/-- The probability of the final marble being red -/
def finalRedProbability (initialState : JarState) : Real := sorry

/-- Theorem stating that the probability of the final marble being red is 0 -/
theorem final_red_probability_zero :
  finalRedProbability ⟨2, 2⟩ = 0 := by sorry

end NUMINAMATH_CALUDE_final_red_probability_zero_l1963_196302


namespace NUMINAMATH_CALUDE_distribute_6_4_l1963_196368

/-- The number of ways to distribute n identical objects among k classes,
    with each class receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- Theorem stating that there are 10 ways to distribute 6 spots among 4 classes,
    with each class receiving at least one spot. -/
theorem distribute_6_4 : distribute 6 4 = 10 := by sorry

end NUMINAMATH_CALUDE_distribute_6_4_l1963_196368


namespace NUMINAMATH_CALUDE_amount_saved_calculation_l1963_196395

def initial_amount : ℕ := 6000
def pen_cost : ℕ := 3200
def eraser_cost : ℕ := 1000
def candy_cost : ℕ := 500

theorem amount_saved_calculation :
  initial_amount - (pen_cost + eraser_cost + candy_cost) = 1300 := by
  sorry

end NUMINAMATH_CALUDE_amount_saved_calculation_l1963_196395


namespace NUMINAMATH_CALUDE_parabola_equation_l1963_196385

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form x^2 = 2py -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line passing through a point M and tangent to a parabola at two points A and B -/
structure TangentLine where
  M : Point
  A : Point
  B : Point
  parabola : Parabola
  h1 : M.x = 2
  h2 : M.y = -2 * parabola.p
  h3 : A.x^2 = 2 * parabola.p * A.y
  h4 : B.x^2 = 2 * parabola.p * B.y

/-- The main theorem to prove -/
theorem parabola_equation (t : TangentLine) 
  (h : (t.A.y + t.B.y) / 2 = 6) : 
  t.parabola.p = 1 ∨ t.parabola.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1963_196385


namespace NUMINAMATH_CALUDE_sqrt_360000_l1963_196356

theorem sqrt_360000 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_l1963_196356


namespace NUMINAMATH_CALUDE_polynomial_division_l1963_196382

theorem polynomial_division (x : ℝ) : 
  x^6 - 14*x^4 + 8*x^3 - 26*x^2 + 14*x - 3 = 
  (x - 3) * (x^5 + 3*x^4 - 5*x^3 - 7*x^2 - 47*x - 7) + (-24) := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_l1963_196382


namespace NUMINAMATH_CALUDE_principal_calculation_l1963_196312

/-- Given a principal amount, prove that it equals 2600 if the simple interest
    at 4% for 5 years is 2080 less than the principal. -/
theorem principal_calculation (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2080 → P = 2600 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1963_196312


namespace NUMINAMATH_CALUDE_certain_number_proof_l1963_196349

theorem certain_number_proof (original : ℝ) (certain : ℝ) : 
  original = 50 → (1/5 : ℝ) * original - 5 = certain → certain = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1963_196349


namespace NUMINAMATH_CALUDE_rico_justin_dog_difference_l1963_196392

theorem rico_justin_dog_difference (justin_dogs : ℕ) (camden_dog_legs : ℕ) (camden_rico_ratio : ℚ) :
  justin_dogs = 14 →
  camden_dog_legs = 72 →
  camden_rico_ratio = 3/4 →
  ∃ (rico_dogs : ℕ), rico_dogs - justin_dogs = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rico_justin_dog_difference_l1963_196392


namespace NUMINAMATH_CALUDE_intersection_points_min_distance_l1963_196330

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x + y + 1 = 0
def C₃ (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Theorem for intersection points
theorem intersection_points :
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y ↔ (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = -1)) :=
sorry

-- Theorem for minimum distance
theorem min_distance :
  (∃ d : ℝ, d = Real.sqrt 2 - 1 ∧
    ∀ x₁ y₁ x₂ y₂ : ℝ, C₂ x₁ y₁ → C₃ x₂ y₂ →
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ d) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_min_distance_l1963_196330


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1963_196357

theorem rectangle_area_equals_perimeter (x : ℝ) :
  let length : ℝ := 4 * x
  let width : ℝ := x + 7
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (length > 0 ∧ width > 0 ∧ area = perimeter) → x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1963_196357


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l1963_196389

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l1963_196389


namespace NUMINAMATH_CALUDE_range_of_m_l1963_196350

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m + 3) + y^2 / (7*m - 3) = 1 ∧ m + 3 > 0 ∧ 7*m - 3 < 0

def q (m : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (5 - 2*m)^x₁ < (5 - 2*m)^x₂

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (m ≤ -3 ∨ (3/7 ≤ m ∧ m < 2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1963_196350


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1963_196384

/-- A geometric sequence with common ratio q > 1 and positive first term -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ a 1 > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) 
    (h : GeometricSequence a q) 
    (eq : a 2 * a 4 + a 4 * a 10 - a 4 * a 6 - a 5 * a 5 = 9) :
  a 3 - a 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1963_196384


namespace NUMINAMATH_CALUDE_correct_selection_ways_l1963_196332

/-- The number of university graduates --/
def total_graduates : ℕ := 10

/-- The number of graduates to be selected --/
def selected_graduates : ℕ := 3

/-- The function that calculates the number of ways to select graduates --/
def selection_ways (total : ℕ) (select : ℕ) (at_least_AB : Bool) (exclude_C : Bool) : ℕ := sorry

/-- The theorem stating the correct number of selection ways --/
theorem correct_selection_ways : 
  selection_ways total_graduates selected_graduates true true = 49 := by sorry

end NUMINAMATH_CALUDE_correct_selection_ways_l1963_196332


namespace NUMINAMATH_CALUDE_jennifer_sweets_distribution_l1963_196376

theorem jennifer_sweets_distribution (green blue yellow : ℕ) 
  (h1 : green = 212)
  (h2 : blue = 310)
  (h3 : yellow = 502)
  (friends : ℕ)
  (h4 : friends = 3) :
  (green + blue + yellow) / (friends + 1) = 256 := by
sorry

end NUMINAMATH_CALUDE_jennifer_sweets_distribution_l1963_196376


namespace NUMINAMATH_CALUDE_division_problem_l1963_196311

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 686) (h2 : quotient = 19) (h3 : remainder = 2) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 36 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1963_196311


namespace NUMINAMATH_CALUDE_xyz_product_abs_l1963_196338

theorem xyz_product_abs (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_eq : x + 2/y = y + 2/z ∧ y + 2/z = z + 2/x) :
  |x * y * z| = 2 := by sorry

end NUMINAMATH_CALUDE_xyz_product_abs_l1963_196338


namespace NUMINAMATH_CALUDE_area_of_triangle_FNV_l1963_196365

-- Define the rectangle EFGH
structure Rectangle where
  EF : ℝ
  EH : ℝ

-- Define the trapezoid KWFG
structure Trapezoid where
  KF : ℝ
  WG : ℝ
  height : ℝ
  area : ℝ

-- Define the theorem
theorem area_of_triangle_FNV (rect : Rectangle) (trap : Trapezoid) :
  rect.EF = 15 ∧
  trap.KF = 5 ∧
  trap.WG = 5 ∧
  trap.area = 150 ∧
  trap.KF = trap.WG →
  (1 / 2 : ℝ) * (1 / 2 : ℝ) * (trap.KF + rect.EF) * rect.EH = 125 := by
  sorry


end NUMINAMATH_CALUDE_area_of_triangle_FNV_l1963_196365


namespace NUMINAMATH_CALUDE_soccer_team_red_cards_l1963_196345

theorem soccer_team_red_cards 
  (total_players : ℕ) 
  (players_without_cautions : ℕ) 
  (yellow_cards_per_cautioned_player : ℕ) 
  (yellow_cards_per_red_card : ℕ) 
  (h1 : total_players = 11) 
  (h2 : players_without_cautions = 5) 
  (h3 : yellow_cards_per_cautioned_player = 1) 
  (h4 : yellow_cards_per_red_card = 2) : 
  (total_players - players_without_cautions) * yellow_cards_per_cautioned_player / yellow_cards_per_red_card = 3 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_red_cards_l1963_196345


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1963_196333

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1963_196333


namespace NUMINAMATH_CALUDE_square_value_l1963_196326

theorem square_value : ∃ (square : ℤ), 9210 - 9124 = 210 - square ∧ square = 124 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l1963_196326


namespace NUMINAMATH_CALUDE_tom_payment_l1963_196381

/-- The total amount Tom paid to the shopkeeper for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 1235 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 75 = 1235 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_l1963_196381


namespace NUMINAMATH_CALUDE_pencil_ratio_l1963_196353

/-- Given the number of pencils for Tyrah, Tim, and Sarah, prove the ratio of Tim's to Sarah's pencils -/
theorem pencil_ratio (sarah_pencils tyrah_pencils tim_pencils : ℕ) 
  (h1 : tyrah_pencils = 6 * sarah_pencils)
  (h2 : tyrah_pencils = 12)
  (h3 : tim_pencils = 16) :
  tim_pencils / sarah_pencils = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_ratio_l1963_196353


namespace NUMINAMATH_CALUDE_average_height_calculation_l1963_196373

theorem average_height_calculation (north_count : ℕ) (north_avg : ℝ) 
  (south_count : ℕ) (south_avg : ℝ) : 
  north_count = 300 → 
  south_count = 200 → 
  north_avg = 1.60 → 
  south_avg = 1.50 → 
  let total_count := north_count + south_count
  let total_height := north_count * north_avg + south_count * south_avg
  (total_height / total_count : ℝ) = 1.56 := by sorry

end NUMINAMATH_CALUDE_average_height_calculation_l1963_196373


namespace NUMINAMATH_CALUDE_a_gt_one_iff_a_gt_zero_l1963_196399

theorem a_gt_one_iff_a_gt_zero {a : ℝ} : a > 1 ↔ a > 0 := by sorry

end NUMINAMATH_CALUDE_a_gt_one_iff_a_gt_zero_l1963_196399


namespace NUMINAMATH_CALUDE_floor_neg_three_point_seven_l1963_196315

/-- The floor function, which returns the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem: The floor of -3.7 is -4 -/
theorem floor_neg_three_point_seven :
  floor (-3.7) = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_three_point_seven_l1963_196315


namespace NUMINAMATH_CALUDE_highway_length_l1963_196335

/-- The length of a highway given two cars starting from opposite ends -/
theorem highway_length (speed1 speed2 : ℝ) (time : ℝ) (h1 : speed1 = 54)
  (h2 : speed2 = 57) (h3 : time = 3) :
  speed1 * time + speed2 * time = 333 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_l1963_196335


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1963_196361

theorem arctan_equation_solution (x : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/15) + Real.arctan (1/x) = π/4 → x = 53/4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1963_196361


namespace NUMINAMATH_CALUDE_biggest_measure_for_containers_l1963_196388

theorem biggest_measure_for_containers (a b c : ℕ) 
  (ha : a = 496) (hb : b = 403) (hc : c = 713) : 
  Nat.gcd a (Nat.gcd b c) = 31 := by
  sorry

end NUMINAMATH_CALUDE_biggest_measure_for_containers_l1963_196388


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1963_196303

theorem polynomial_division_theorem (z : ℂ) : 
  ∃ (r : ℂ), 4*z^5 - 3*z^4 + 2*z^3 - 5*z^2 + 9*z - 4 = 
  (z + 3) * (4*z^4 - 15*z^3 + 47*z^2 - 146*z + 447) + r ∧ 
  Complex.abs r < Complex.abs (z + 3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1963_196303


namespace NUMINAMATH_CALUDE_value_144_is_square_iff_b_gt_4_l1963_196320

/-- The value of 144 in base b -/
def value_in_base_b (b : ℕ) : ℕ := b^2 + 4*b + 4

/-- A number is a perfect square if it has an integer square root -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m^2 = n

theorem value_144_is_square_iff_b_gt_4 (b : ℕ) :
  is_perfect_square (value_in_base_b b) ↔ b > 4 :=
sorry

end NUMINAMATH_CALUDE_value_144_is_square_iff_b_gt_4_l1963_196320


namespace NUMINAMATH_CALUDE_middle_number_value_l1963_196364

theorem middle_number_value 
  (a b c d e f g h i j k : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 10.5)
  (h2 : (f + g + h + i + j + k) / 6 = 11.4)
  (h3 : (a + b + c + d + e + f + g + h + i + j + k) / 11 = 9.9)
  (h4 : a + b + c = i + j + k) :
  f = 22.5 := by
    sorry

end NUMINAMATH_CALUDE_middle_number_value_l1963_196364


namespace NUMINAMATH_CALUDE_james_stickers_l1963_196386

theorem james_stickers (x : ℕ) : x + 22 = 61 → x = 39 := by
  sorry

end NUMINAMATH_CALUDE_james_stickers_l1963_196386


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1963_196331

theorem a_plus_b_value (a b : ℝ) (h1 : a^2 = 16) (h2 : |b| = 3) (h3 : a + b < 0) :
  a + b = -1 ∨ a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1963_196331


namespace NUMINAMATH_CALUDE_bryan_stones_sale_l1963_196351

/-- The total money Bryan received from selling his precious stones collection -/
def total_money (num_emeralds num_rubies num_sapphires : ℕ) 
  (price_emerald price_ruby price_sapphire : ℕ) : ℕ :=
  num_emeralds * price_emerald + num_rubies * price_ruby + num_sapphires * price_sapphire

/-- Theorem stating that Bryan received $17555 for his precious stones collection -/
theorem bryan_stones_sale : 
  total_money 3 2 3 1785 2650 2300 = 17555 := by
  sorry

#eval total_money 3 2 3 1785 2650 2300

end NUMINAMATH_CALUDE_bryan_stones_sale_l1963_196351


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1963_196308

theorem reciprocal_of_negative_three :
  (1 : ℝ) / (-3 : ℝ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1963_196308


namespace NUMINAMATH_CALUDE_consecutive_factorials_divisible_by_61_l1963_196314

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem consecutive_factorials_divisible_by_61 (k : ℕ) :
  (∃ m : ℕ, factorial (k - 2) + factorial (k - 1) + factorial k = 61 * m) →
  k ≥ 61 := by
sorry

end NUMINAMATH_CALUDE_consecutive_factorials_divisible_by_61_l1963_196314


namespace NUMINAMATH_CALUDE_alternating_color_probability_alternating_color_probability_proof_l1963_196360

/-- The probability of drawing 10 balls with alternating colors from a box containing 5 white and 5 black balls -/
theorem alternating_color_probability : ℚ :=
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  let successful_arrangements : ℕ := 2
  1 / 126

/-- Proof that the probability of drawing 10 balls with alternating colors from a box containing 5 white and 5 black balls is 1/126 -/
theorem alternating_color_probability_proof :
  alternating_color_probability = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_alternating_color_probability_proof_l1963_196360


namespace NUMINAMATH_CALUDE_sum_a_b_equals_negative_two_l1963_196359

theorem sum_a_b_equals_negative_two (a b : ℝ) :
  |a - 1| + (b + 3)^2 = 0 → a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_negative_two_l1963_196359


namespace NUMINAMATH_CALUDE_daniel_elsa_distance_diff_l1963_196334

/-- Calculates the difference in distance traveled between two cyclists given their speeds and times on different tracks. -/
def distance_difference (daniel_plain_speed elsa_plain_speed : ℝ)
                        (plain_time : ℝ)
                        (daniel_hilly_speed elsa_hilly_speed : ℝ)
                        (hilly_time : ℝ) : ℝ :=
  let daniel_total := daniel_plain_speed * plain_time + daniel_hilly_speed * hilly_time
  let elsa_total := elsa_plain_speed * plain_time + elsa_hilly_speed * hilly_time
  daniel_total - elsa_total

/-- The difference in distance traveled between Daniel and Elsa is 7 miles. -/
theorem daniel_elsa_distance_diff :
  distance_difference 20 18 3 16 15 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_daniel_elsa_distance_diff_l1963_196334


namespace NUMINAMATH_CALUDE_line_parallel_plane_condition_l1963_196393

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)

-- The main theorem
theorem line_parallel_plane_condition :
  -- If a line is parallel to a plane, then it's not contained in the plane
  (∀ (l : Line) (p : Plane), parallel l p → ¬(contained_in l p)) ∧
  -- There exists a line and a plane such that the line is not contained in the plane
  -- but also not parallel to it (i.e., it intersects the plane)
  (∃ (l : Line) (p : Plane), ¬(contained_in l p) ∧ ¬(parallel l p) ∧ intersects l p) :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_condition_l1963_196393


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1963_196318

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1963_196318


namespace NUMINAMATH_CALUDE_problem_solution_l1963_196375

theorem problem_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^3 / y = 2)
  (h2 : y^3 / z = 4)
  (h3 : z^3 / x = 8) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1963_196375


namespace NUMINAMATH_CALUDE_book_recipient_sequences_l1963_196363

theorem book_recipient_sequences (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 3) :
  (n * (n - 1) * (n - 2)) = 2730 :=
by sorry

end NUMINAMATH_CALUDE_book_recipient_sequences_l1963_196363


namespace NUMINAMATH_CALUDE_distribute_six_books_l1963_196352

/-- The number of ways to distribute n different books among two people, 
    with each person getting one book. -/
def distribute_books (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem stating that distributing 6 different books among two people, 
    with each person getting one book, can be done in 30 different ways. -/
theorem distribute_six_books : distribute_books 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_books_l1963_196352


namespace NUMINAMATH_CALUDE_mityas_age_l1963_196358

theorem mityas_age (shura_age mitya_age : ℝ) : 
  (mitya_age = shura_age + 11) →
  (mitya_age - shura_age = 2 * (shura_age - (mitya_age - shura_age))) →
  mitya_age = 27.5 := by
sorry

end NUMINAMATH_CALUDE_mityas_age_l1963_196358


namespace NUMINAMATH_CALUDE_percentage_increase_l1963_196390

theorem percentage_increase (original : ℝ) (new : ℝ) : 
  original = 60 → new = 150 → (new - original) / original * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1963_196390


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1963_196380

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 99) 
  (h2 : a*b + b*c + a*c = 131) : 
  a + b + c = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1963_196380


namespace NUMINAMATH_CALUDE_inverse_g_equals_two_l1963_196337

/-- Given nonzero constants a and b, and a function g defined as g(x) = 1 / (2ax + b),
    prove that the inverse of g evaluated at 1 / (4a + b) is equal to 2. -/
theorem inverse_g_equals_two (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let g := fun x => 1 / (2 * a * x + b)
  Function.invFun g (1 / (4 * a + b)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_equals_two_l1963_196337


namespace NUMINAMATH_CALUDE_thirtieth_number_in_base12_l1963_196300

/-- Converts a decimal number to its base 12 representation --/
def toBase12 (n : ℕ) : List ℕ :=
  if n < 12 then [n]
  else (n % 12) :: toBase12 (n / 12)

/-- Interprets a list of digits as a number in base 12 --/
def fromBase12 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 12 * acc) 0

theorem thirtieth_number_in_base12 :
  toBase12 30 = [6, 2] ∧ fromBase12 [6, 2] = 30 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_number_in_base12_l1963_196300


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_12_l1963_196306

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ+) : ℕ+ := sorry

theorem eleventh_number_with_digit_sum_12 :
  nth_number_with_digit_sum_12 11 = 147 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_12_l1963_196306


namespace NUMINAMATH_CALUDE_tan_105_degrees_l1963_196378

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l1963_196378


namespace NUMINAMATH_CALUDE_pet_store_combinations_l1963_196396

def num_puppies : ℕ := 20
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 12
def num_rabbits : ℕ := 5
def num_customers : ℕ := 4

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * num_rabbits * Nat.factorial num_customers = 288000 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l1963_196396


namespace NUMINAMATH_CALUDE_parabola_and_line_theorem_l1963_196377

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem parabola_and_line_theorem (para : Parabola) (l : Line) (P A B : Point) :
  (para.p * 2 = 4) →  -- Distance between focus and directrix is 4
  (P.x = 1 ∧ P.y = -1) →  -- P is at (1, -1)
  (A.y ^ 2 = 2 * para.p * A.x ∧ B.y ^ 2 = 2 * para.p * B.x) →  -- A and B are on the parabola
  (l.a * A.x + l.b * A.y + l.c = 0 ∧ l.a * B.x + l.b * B.y + l.c = 0) →  -- A and B are on the line
  (P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2) →  -- P is midpoint of AB
  (∀ x y, y ^ 2 = 8 * x ↔ y ^ 2 = 2 * para.p * x) ∧  -- Parabola equation is y² = 8x
  (∀ x y, 4 * x + y - 3 = 0 ↔ l.a * x + l.b * y + l.c = 0)  -- Line equation is 4x + y - 3 = 0
:= by sorry

end NUMINAMATH_CALUDE_parabola_and_line_theorem_l1963_196377


namespace NUMINAMATH_CALUDE_not_enough_ribbons_l1963_196346

/-- Represents the number of ribbons needed for a gift --/
structure RibbonRequirement where
  typeA : ℕ
  typeB : ℕ

/-- Represents the available ribbon supply --/
structure RibbonSupply where
  typeA : ℤ
  typeB : ℤ

def gift_count : ℕ := 8

def initial_supply : RibbonSupply := ⟨10, 12⟩

def requirement_gifts_1_to_4 : RibbonRequirement := ⟨2, 1⟩
def requirement_gifts_5_to_8 : RibbonRequirement := ⟨1, 3⟩

def total_ribbons_needed (req1 req2 : RibbonRequirement) : RibbonRequirement :=
  ⟨req1.typeA * 4 + req2.typeA * 4, req1.typeB * 4 + req2.typeB * 4⟩

def remaining_ribbons (supply : RibbonSupply) (needed : RibbonRequirement) : RibbonSupply :=
  ⟨supply.typeA - needed.typeA, supply.typeB - needed.typeB⟩

theorem not_enough_ribbons :
  let total_needed := total_ribbons_needed requirement_gifts_1_to_4 requirement_gifts_5_to_8
  let remaining := remaining_ribbons initial_supply total_needed
  remaining.typeA < 0 ∧ remaining.typeB < 0 ∧
  remaining.typeA = -2 ∧ remaining.typeB = -4 :=
by sorry

#check not_enough_ribbons

end NUMINAMATH_CALUDE_not_enough_ribbons_l1963_196346


namespace NUMINAMATH_CALUDE_four_numbers_problem_l1963_196397

theorem four_numbers_problem (a b c d : ℝ) : 
  a + b + c + d = 45 ∧ 
  a + 2 = b - 2 ∧ 
  a + 2 = 2 * c ∧ 
  a + 2 = d / 2 → 
  a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_problem_l1963_196397


namespace NUMINAMATH_CALUDE_f_monotonicity_and_max_k_l1963_196307

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + Real.log x) / x

theorem f_monotonicity_and_max_k :
  (∀ m : ℝ, m ≥ 1 → ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f m x₁ > f m x₂) ∧
  (∀ m : ℝ, m < 1 → ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → x₂ ≤ Real.exp (1 - m) → f m x₁ < f m x₂) ∧
  (∀ m : ℝ, m < 1 → ∀ x₁ x₂ : ℝ, Real.exp (1 - m) < x₁ → x₁ < x₂ → f m x₁ > f m x₂) ∧
  (∀ x : ℝ, x > 1 → 6 / (x + 1) < f 4 x) ∧
  (∀ k : ℕ, k > 6 → ∃ x : ℝ, x > 1 ∧ k / (x + 1) ≥ f 4 x) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_max_k_l1963_196307


namespace NUMINAMATH_CALUDE_complement_union_A_B_l1963_196379

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

theorem complement_union_A_B : 
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l1963_196379


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_range_l1963_196328

-- Problem 1
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ -1 < x ∧ x < 2) →
  a = 1 ∧ b = -2 :=
sorry

-- Problem 2
theorem quadratic_inequality_range (c : ℝ) :
  (∀ x : ℝ, x ≥ 1 → x^2 + 3*x - c > 0) →
  c < 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_range_l1963_196328


namespace NUMINAMATH_CALUDE_union_of_sets_l1963_196321

theorem union_of_sets (a b : ℕ) (M N : Set ℕ) : 
  M = {3, 2^a} → N = {a, b} → M ∩ N = {2} → M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1963_196321


namespace NUMINAMATH_CALUDE_trig_identity_l1963_196369

theorem trig_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1963_196369


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l1963_196391

theorem right_triangle_arithmetic_sequence (a b c : ℝ) (area : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a < b → b < c →
  c - b = b - a →
  a^2 + b^2 = c^2 →
  area = (1/2) * a * b →
  area = 1350 →
  (a, b, c) = (45, 60, 75) := by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l1963_196391


namespace NUMINAMATH_CALUDE_february_2013_days_l1963_196304

/-- A function that determines if a given year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  year % 4 = 0

/-- A function that returns the number of days in February for a given year -/
def daysInFebruary (year : ℕ) : ℕ :=
  if isLeapYear year then 29 else 28

/-- Theorem stating that February in 2013 has 28 days -/
theorem february_2013_days : daysInFebruary 2013 = 28 := by
  sorry

#eval daysInFebruary 2013

end NUMINAMATH_CALUDE_february_2013_days_l1963_196304


namespace NUMINAMATH_CALUDE_fraction_division_simplify_fraction_divide_fractions_l1963_196319

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction (n d : ℤ) (hd : d ≠ 0) :
  (n / d : ℚ) = (n / gcd n d) / (d / gcd n d) :=
by sorry

theorem divide_fractions : (5 / 6 : ℚ) / (-9 / 10) = -25 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_simplify_fraction_divide_fractions_l1963_196319


namespace NUMINAMATH_CALUDE_plane_distance_ratio_l1963_196347

/-- Proves the ratio of plane distance to total distance -/
theorem plane_distance_ratio (total : ℝ) (bus : ℝ) (train : ℝ) (plane : ℝ) :
  total = 900 →
  train = (2/3) * bus →
  bus = 360 →
  plane = total - (bus + train) →
  plane / total = 1/3 := by
sorry

end NUMINAMATH_CALUDE_plane_distance_ratio_l1963_196347


namespace NUMINAMATH_CALUDE_regular_square_pyramid_volume_l1963_196362

/-- A regular square pyramid with side edge length 2√3 and angle 60° between side edge and base has volume 6 -/
theorem regular_square_pyramid_volume (side_edge : ℝ) (angle : ℝ) : 
  side_edge = 2 * Real.sqrt 3 →
  angle = π / 3 →
  let height := side_edge * Real.sin angle
  let base_area := (side_edge^2) / 2
  let volume := (1/3) * base_area * height
  volume = 6 := by sorry

end NUMINAMATH_CALUDE_regular_square_pyramid_volume_l1963_196362


namespace NUMINAMATH_CALUDE_book_reading_time_l1963_196313

theorem book_reading_time (total_books : ℕ) (books_per_week : ℕ) (weeks : ℕ) : 
  total_books = 30 → books_per_week = 6 → weeks * books_per_week = total_books → weeks = 5 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l1963_196313


namespace NUMINAMATH_CALUDE_total_knitting_time_l1963_196371

/-- Represents the time in hours to knit each item of clothing --/
structure KnittingTime where
  hat : ℝ
  scarf : ℝ
  sweater : ℝ
  mitten : ℝ
  sock : ℝ

/-- Calculates the total time to knit one complete outfit --/
def outfitTime (t : KnittingTime) : ℝ :=
  t.hat + t.scarf + t.sweater + 2 * t.mitten + 2 * t.sock

/-- Theorem stating the total time to knit 3 outfits --/
theorem total_knitting_time (t : KnittingTime)
  (hat_time : t.hat = 2)
  (scarf_time : t.scarf = 3)
  (sweater_time : t.sweater = 6)
  (mitten_time : t.mitten = 1)
  (sock_time : t.sock = 1.5) :
  3 * outfitTime t = 48 := by
  sorry


end NUMINAMATH_CALUDE_total_knitting_time_l1963_196371


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1963_196340

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1963_196340


namespace NUMINAMATH_CALUDE_loan_interest_rate_l1963_196343

/-- Given a loan of $220 repaid with $242 after one year, prove the annual interest rate is 10% -/
theorem loan_interest_rate : 
  let principal : ℝ := 220
  let total_repayment : ℝ := 242
  let interest_rate : ℝ := (total_repayment - principal) / principal * 100
  interest_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_rate_l1963_196343


namespace NUMINAMATH_CALUDE_stone_150_is_6_l1963_196394

/-- Represents the counting pattern described in the problem -/
def stone_count (n : ℕ) : ℕ := 
  if n ≤ 12 then n 
  else if n ≤ 23 then 24 - n 
  else stone_count ((n - 1) % 22 + 1)

/-- The total number of stones -/
def total_stones : ℕ := 12

/-- The number at which we want to find the corresponding stone -/
def target_count : ℕ := 150

/-- Theorem stating that the stone counted as 150 is originally stone number 6 -/
theorem stone_150_is_6 : 
  ∃ (k : ℕ), k ≤ total_stones ∧ stone_count target_count = stone_count k ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_stone_150_is_6_l1963_196394


namespace NUMINAMATH_CALUDE_employee_earnings_l1963_196342

/-- Calculates the total earnings for an employee based on their work schedule and pay rates. -/
theorem employee_earnings (regular_rate : ℝ) (overtime_multiplier : ℝ) (regular_hours : ℝ) 
  (first_three_days_hours : ℝ) (last_two_days_multiplier : ℝ) : 
  regular_rate = 30 →
  overtime_multiplier = 1.5 →
  regular_hours = 40 →
  first_three_days_hours = 6 →
  last_two_days_multiplier = 2 →
  let overtime_rate := regular_rate * overtime_multiplier
  let last_two_days_hours := first_three_days_hours * last_two_days_multiplier
  let total_hours := first_three_days_hours * 3 + last_two_days_hours * 2
  let overtime_hours := max (total_hours - regular_hours) 0
  let regular_pay := min total_hours regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  regular_pay + overtime_pay = 1290 := by
sorry

end NUMINAMATH_CALUDE_employee_earnings_l1963_196342


namespace NUMINAMATH_CALUDE_remainder_2345678901_mod_102_l1963_196372

theorem remainder_2345678901_mod_102 : 2345678901 % 102 = 65 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2345678901_mod_102_l1963_196372


namespace NUMINAMATH_CALUDE_vector_addition_l1963_196354

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![1, -1]

theorem vector_addition :
  (vector_a + vector_b) = ![2, 1] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l1963_196354


namespace NUMINAMATH_CALUDE_inequality_proof_l1963_196310

theorem inequality_proof (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x^4 + y^4 + z^4 = 1) :
  x^3 / (1 - x^8) + y^3 / (1 - y^8) + z^3 / (1 - z^8) ≥ 9/8 * Real.rpow 3 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1963_196310


namespace NUMINAMATH_CALUDE_triangle_side_length_l1963_196325

-- Define a triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real
  x : Real
  y : Real
  z : Real

-- State the theorem
theorem triangle_side_length (t : Triangle) : 
  t.y = 7 → 
  t.z = 3 → 
  Real.cos (t.Y - t.Z) = 17/32 → 
  t.x = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1963_196325


namespace NUMINAMATH_CALUDE_tom_remaining_pieces_l1963_196323

/-- The number of boxes Tom initially bought -/
def initial_boxes : ℕ := 12

/-- The number of boxes Tom gave to his little brother -/
def boxes_given : ℕ := 7

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 6

/-- Theorem: Tom still has 30 pieces of candy -/
theorem tom_remaining_pieces : 
  (initial_boxes - boxes_given) * pieces_per_box = 30 := by
  sorry

end NUMINAMATH_CALUDE_tom_remaining_pieces_l1963_196323


namespace NUMINAMATH_CALUDE_triangle_area_l1963_196329

theorem triangle_area (base height : ℝ) (h1 : base = 4) (h2 : height = 6) :
  (base * height) / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1963_196329


namespace NUMINAMATH_CALUDE_a_7_not_prime_l1963_196387

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Sequence defined by the recursive formula -/
def a : ℕ → ℕ
  | 0 => 1  -- a_1 is a positive integer
  | n + 1 => a n + reverseDigits (a n)

theorem a_7_not_prime : ∃ (k : ℕ), k > 1 ∧ k < a 7 ∧ a 7 % k = 0 := by sorry

end NUMINAMATH_CALUDE_a_7_not_prime_l1963_196387


namespace NUMINAMATH_CALUDE_license_plate_count_l1963_196383

/-- The number of digits in a license plate -/
def num_digits : ℕ := 5

/-- The number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- The number of consonants in the alphabet -/
def num_consonants : ℕ := 21

/-- The number of positions where the consonant pair can be placed -/
def consonant_pair_positions : ℕ := 6

/-- The number of distinct license plates -/
def num_license_plates : ℕ := 
  consonant_pair_positions * digit_choices ^ num_digits * (num_consonants * (num_consonants - 1))

theorem license_plate_count : num_license_plates = 2520000000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1963_196383


namespace NUMINAMATH_CALUDE_square_sum_geq_two_l1963_196305

theorem square_sum_geq_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_two_l1963_196305


namespace NUMINAMATH_CALUDE_quadratic_root_m_l1963_196370

theorem quadratic_root_m (m : ℝ) : ((-1 : ℝ)^2 + m * (-1) - 5 = 0) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_l1963_196370


namespace NUMINAMATH_CALUDE_french_students_count_l1963_196344

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) (french : ℕ) : 
  total = 60 →
  german = 22 →
  both = 9 →
  neither = 6 →
  french + german - both = total - neither →
  french = 41 :=
by sorry

end NUMINAMATH_CALUDE_french_students_count_l1963_196344


namespace NUMINAMATH_CALUDE_different_course_selections_count_l1963_196355

/-- The number of courses available to choose from -/
def num_courses : ℕ := 4

/-- The number of courses each person must choose -/
def courses_per_person : ℕ := 2

/-- The number of people choosing courses -/
def num_people : ℕ := 2

/-- Represents the ways two people can choose courses differently -/
def different_course_selections : ℕ := 30

/-- Theorem stating the number of ways two people can choose courses differently -/
theorem different_course_selections_count :
  (num_courses = 4) →
  (courses_per_person = 2) →
  (num_people = 2) →
  (different_course_selections = 30) :=
by sorry

end NUMINAMATH_CALUDE_different_course_selections_count_l1963_196355


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1963_196367

theorem sqrt_x_minus_one_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1963_196367


namespace NUMINAMATH_CALUDE_cosine_of_angle_l1963_196348

/-- Given two vectors a and b in ℝ², prove that the cosine of the angle between them is (3√10) / 10 -/
theorem cosine_of_angle (a b : ℝ × ℝ) (h1 : a = (3, 3)) (h2 : (2 • b) - a = (-1, 1)) :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_l1963_196348


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l1963_196324

-- Define the two lines in parametric form
def line1 (t : ℝ) : ℝ × ℝ := (1 - 2*t, 2 + 6*t)
def line2 (u : ℝ) : ℝ × ℝ := (3 + u, 8 + 3*u)

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem lines_intersect_at_point :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line1 t = p) ∧ (∃ u : ℝ, line2 u = p) ∧ p = intersection_point :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l1963_196324


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1963_196336

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 4 = 0) : 
  (a^2 - 3) * (a + 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1963_196336
