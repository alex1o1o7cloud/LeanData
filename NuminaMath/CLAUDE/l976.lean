import Mathlib

namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l976_97687

theorem simplify_sqrt_product : 
  Real.sqrt (5 * 3) * Real.sqrt (3^5 * 5^2) = 135 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l976_97687


namespace NUMINAMATH_CALUDE_min_value_of_ab_l976_97619

theorem min_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 8) :
  a * b ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_ab_l976_97619


namespace NUMINAMATH_CALUDE_probability_arts_and_sciences_is_two_thirds_l976_97639

/-- Represents a class subject -/
inductive Subject
  | Mathematics
  | Chinese
  | Politics
  | Geography
  | English
  | History
  | PhysicalEducation

/-- Represents the time of day for a class -/
inductive TimeOfDay
  | Morning
  | Afternoon

/-- Defines the class schedule -/
def schedule : TimeOfDay → List Subject
  | TimeOfDay.Morning => [Subject.Mathematics, Subject.Chinese, Subject.Politics, Subject.Geography]
  | TimeOfDay.Afternoon => [Subject.English, Subject.History, Subject.PhysicalEducation]

/-- Determines if a subject is related to arts and sciences -/
def isArtsAndSciences : Subject → Bool
  | Subject.Politics => true
  | Subject.History => true
  | Subject.Geography => true
  | _ => false

/-- The probability of selecting at least one arts and sciences class -/
def probabilityArtsAndSciences : ℚ := 2/3

theorem probability_arts_and_sciences_is_two_thirds :
  probabilityArtsAndSciences = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_arts_and_sciences_is_two_thirds_l976_97639


namespace NUMINAMATH_CALUDE_steven_has_14_peaches_l976_97622

/-- The number of peaches each person has -/
structure PeachCount where
  steven : ℕ
  jake : ℕ
  jill : ℕ

/-- Given conditions about peach counts -/
def peach_conditions (p : PeachCount) : Prop :=
  p.jake + 6 = p.steven ∧ 
  p.jake = p.jill + 3 ∧ 
  p.jill = 5

/-- Theorem stating Steven has 14 peaches -/
theorem steven_has_14_peaches (p : PeachCount) 
  (h : peach_conditions p) : p.steven = 14 := by
  sorry

end NUMINAMATH_CALUDE_steven_has_14_peaches_l976_97622


namespace NUMINAMATH_CALUDE_solution_set_inequality_l976_97692

theorem solution_set_inequality (x : ℝ) :
  (x ≠ 2) → (1 / (x - 2) > -2 ↔ x ∈ Set.Iio (3/2) ∪ Set.Ioi 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l976_97692


namespace NUMINAMATH_CALUDE_price_after_discount_l976_97661

def original_price : ℕ := 76
def discount : ℕ := 25

theorem price_after_discount :
  original_price - discount = 51 := by sorry

end NUMINAMATH_CALUDE_price_after_discount_l976_97661


namespace NUMINAMATH_CALUDE_original_number_l976_97616

theorem original_number (y : ℚ) : (1 - (1 / y) = 5 / 4) → y = -4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l976_97616


namespace NUMINAMATH_CALUDE_valid_assignment_l976_97620

/-- Represents the squares in the grid -/
inductive Square
| One | Nine | A | B | C | D | E | F | G

/-- Represents the direction of arrows -/
inductive Direction
| Right | RightUp | Up | Down

/-- Define the arrow directions for each square -/
def arrowDirection (s : Square) : Option Direction :=
  match s with
  | Square.One => some Direction.Right
  | Square.B => some Direction.RightUp
  | Square.E => some Direction.Right
  | Square.C => some Direction.Right
  | Square.D => some Direction.Up
  | Square.A => some Direction.Down
  | Square.G => some Direction.Right
  | Square.F => some Direction.Right
  | Square.Nine => none

/-- Define the next square based on the current square and arrow direction -/
def nextSquare (s : Square) : Option Square :=
  match s, arrowDirection s with
  | Square.One, some Direction.Right => some Square.B
  | Square.B, some Direction.RightUp => some Square.E
  | Square.E, some Direction.Right => some Square.C
  | Square.C, some Direction.Right => some Square.D
  | Square.D, some Direction.Up => some Square.A
  | Square.A, some Direction.Down => some Square.G
  | Square.G, some Direction.Right => some Square.F
  | Square.F, some Direction.Right => some Square.Nine
  | _, _ => none

/-- The assignment of numbers to squares -/
def assignment : Square → Nat
| Square.One => 1
| Square.Nine => 9
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7

/-- Theorem stating that the assignment is valid -/
theorem valid_assignment :
  (∀ s : Square, s ≠ Square.One ∧ s ≠ Square.Nine →
    2 ≤ assignment s ∧ assignment s ≤ 8) ∧
  (∀ s : Square, s ≠ Square.Nine →
    ∃ next : Square, nextSquare s = some next ∧
    assignment next = assignment s + 1) :=
sorry

end NUMINAMATH_CALUDE_valid_assignment_l976_97620


namespace NUMINAMATH_CALUDE_trains_crossing_time_l976_97600

/-- The time it takes for two trains moving in opposite directions to cross each other -/
theorem trains_crossing_time (length_A length_B speed_A speed_B : ℝ) : 
  length_A = 108 →
  length_B = 112 →
  speed_A = 50 * (1000 / 3600) →
  speed_B = 82 * (1000 / 3600) →
  let total_length := length_A + length_B
  let relative_speed := speed_A + speed_B
  let crossing_time := total_length / relative_speed
  ∃ ε > 0, |crossing_time - 6| < ε :=
by
  sorry

#check trains_crossing_time

end NUMINAMATH_CALUDE_trains_crossing_time_l976_97600


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l976_97691

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 2*p^2 - p + 2 = 0) → 
  (q^3 - 2*q^2 - q + 2 = 0) → 
  (r^3 - 2*r^2 - r + 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l976_97691


namespace NUMINAMATH_CALUDE_correct_ranking_l976_97699

-- Define the team members
inductive TeamMember
| David
| Emma
| Frank

-- Define the experience relation
def has_more_experience (a b : TeamMember) : Prop := sorry

-- Define the most experienced member
def is_most_experienced (m : TeamMember) : Prop :=
  ∀ x : TeamMember, x ≠ m → has_more_experience m x

-- Define the statements
def statement_I : Prop := has_more_experience TeamMember.Frank TeamMember.Emma
def statement_II : Prop := has_more_experience TeamMember.David TeamMember.Frank
def statement_III : Prop := is_most_experienced TeamMember.Frank

-- Define the condition that exactly one statement is true
def exactly_one_true : Prop :=
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ ¬statement_II ∧ statement_III)

-- The theorem to prove
theorem correct_ranking (h : exactly_one_true) :
  has_more_experience TeamMember.David TeamMember.Emma ∧
  has_more_experience TeamMember.Emma TeamMember.Frank :=
sorry

end NUMINAMATH_CALUDE_correct_ranking_l976_97699


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l976_97684

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- The probability of not having two consecutive 1s in a sequence of length 15 -/
theorem probability_no_consecutive_ones : 
  (validSequences 15 : ℚ) / (totalSequences 15 : ℚ) = 1597 / 32768 := by
  sorry

#eval validSequences 15
#eval totalSequences 15

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l976_97684


namespace NUMINAMATH_CALUDE_smallest_sum_of_c_and_d_l976_97674

theorem smallest_sum_of_c_and_d (c d : ℝ) (hc : c > 0) (hd : d > 0)
  (h1 : ∃ x : ℝ, x^2 + c*x + 3*d = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*d*x + c = 0) :
  c + d ≥ 16 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_c_and_d_l976_97674


namespace NUMINAMATH_CALUDE_julie_newspaper_sheets_l976_97667

/-- The number of sheets used to print one newspaper -/
def sheets_per_newspaper (boxes : ℕ) (packages_per_box : ℕ) (sheets_per_package : ℕ) (total_newspapers : ℕ) : ℕ :=
  (boxes * packages_per_box * sheets_per_package) / total_newspapers

/-- Proof that Julie uses 25 sheets to print one newspaper -/
theorem julie_newspaper_sheets : 
  sheets_per_newspaper 2 5 250 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_julie_newspaper_sheets_l976_97667


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l976_97645

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 1/b = 4 → x + 2*y ≤ a + 2*b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l976_97645


namespace NUMINAMATH_CALUDE_inequality_system_solution_l976_97679

theorem inequality_system_solution (x a : ℝ) : 
  (1 - x < -1) ∧ (x - 1 > a) ∧ (∀ y, (1 - y < -1 ∧ y - 1 > a) ↔ y > 2) →
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l976_97679


namespace NUMINAMATH_CALUDE_square_completion_and_max_value_l976_97640

theorem square_completion_and_max_value :
  -- Part 1: Completing the square
  ∀ a : ℝ, a^2 + 4*a + 4 = (a + 2)^2 ∧
  -- Part 2: Factorization using completion of squares
  ∀ a : ℝ, a^2 - 24*a + 143 = (a - 11)*(a - 13) ∧
  -- Part 3: Maximum value of quadratic function
  ∀ a : ℝ, -1/4*a^2 + 2*a - 1 ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_square_completion_and_max_value_l976_97640


namespace NUMINAMATH_CALUDE_complex_square_root_l976_97670

theorem complex_square_root (z : ℂ) : 
  z^2 = -3 - 4*I ∧ z.re < 0 ∧ z.im > 0 → z = -1 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_l976_97670


namespace NUMINAMATH_CALUDE_total_candies_is_96_l976_97612

/-- The number of candies Adam has -/
def adam_candies : ℕ := 6

/-- The number of candies James has -/
def james_candies : ℕ := 3 * adam_candies

/-- The number of candies Rubert has -/
def rubert_candies : ℕ := 4 * james_candies

/-- The total number of candies -/
def total_candies : ℕ := adam_candies + james_candies + rubert_candies

theorem total_candies_is_96 : total_candies = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_is_96_l976_97612


namespace NUMINAMATH_CALUDE_machine_purchase_price_machine_purchase_price_is_14000_l976_97680

/-- Proves that the purchase price of a machine is 14000 given the specified conditions -/
theorem machine_purchase_price : ℝ → Prop :=
  fun purchase_price =>
    let repair_cost : ℝ := 5000
    let transport_cost : ℝ := 1000
    let profit_percentage : ℝ := 50
    let selling_price : ℝ := 30000
    let total_cost : ℝ := purchase_price + repair_cost + transport_cost
    let profit_multiplier : ℝ := (100 + profit_percentage) / 100
    selling_price = profit_multiplier * total_cost →
    purchase_price = 14000

/-- The purchase price of the machine is 14000 -/
theorem machine_purchase_price_is_14000 : machine_purchase_price 14000 := by
  sorry

end NUMINAMATH_CALUDE_machine_purchase_price_machine_purchase_price_is_14000_l976_97680


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l976_97643

/-- Represents the number of bottle caps in various situations -/
structure BottleCaps where
  found : ℕ
  thrown_away : ℕ
  current : ℕ

/-- Given Danny's bottle cap collection data, prove that he found 1 more than he threw away -/
theorem danny_bottle_caps (caps : BottleCaps)
  (h1 : caps.found = 36)
  (h2 : caps.thrown_away = 35)
  (h3 : caps.current = 22) :
  caps.found - caps.thrown_away = 1 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l976_97643


namespace NUMINAMATH_CALUDE_total_footprints_pogo_and_grimzi_footprints_l976_97650

/-- Calculates the total number of footprints left by two creatures on their respective planets -/
theorem total_footprints (pogo_footprints_per_meter : ℕ) 
                         (grimzi_footprints_per_six_meters : ℕ) 
                         (distance : ℕ) : ℕ :=
  let pogo_total := pogo_footprints_per_meter * distance
  let grimzi_total := grimzi_footprints_per_six_meters * (distance / 6)
  pogo_total + grimzi_total

/-- Proves that the combined total number of footprints left by Pogo and Grimzi is 27,000 -/
theorem pogo_and_grimzi_footprints : 
  total_footprints 4 3 6000 = 27000 := by
  sorry

end NUMINAMATH_CALUDE_total_footprints_pogo_and_grimzi_footprints_l976_97650


namespace NUMINAMATH_CALUDE_deepak_present_age_l976_97658

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age. -/
theorem deepak_present_age 
  (rahul_deepak_ratio : ℚ) 
  (rahul_future_age : ℕ) 
  (years_difference : ℕ) :
  rahul_deepak_ratio = 4 / 3 →
  rahul_future_age = 42 →
  years_difference = 6 →
  ∃ (deepak_age : ℕ), deepak_age = 27 :=
by sorry

end NUMINAMATH_CALUDE_deepak_present_age_l976_97658


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l976_97607

/-- Represents the contents of a stationery box -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents a person who uses the stationery box -/
structure Person where
  name : String
  box : StationeryBox
  pagesPerLetter : ℕ

theorem stationery_box_sheets (ann sue : Person) : 
  ann.name = "Ann" →
  sue.name = "Sue" →
  ann.pagesPerLetter = 1 →
  sue.pagesPerLetter = 3 →
  ann.box = sue.box →
  ann.box.sheets - ann.box.envelopes = 50 →
  sue.box.envelopes - sue.box.sheets / 3 = 50 →
  ann.box.sheets = 150 := by
sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l976_97607


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l976_97690

-- Define the types for points and triangles
def Point := ℝ × ℝ
def Triangle := Point × Point × Point

-- Define a function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Define a predicate for parallel lines
def parallel (p1 p2 q1 q2 : Point) : Prop := sorry

-- Define a predicate for congruent triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

theorem triangle_construction_theorem 
  (ABC A₁B₁C₁ : Triangle) 
  (h_equal_area : triangleArea ABC = triangleArea A₁B₁C₁) :
  ∃ (A₂B₂C₂ : Triangle),
    congruent A₂B₂C₂ A₁B₁C₁ ∧ 
    parallel (ABC.1) (A₂B₂C₂.1) (ABC.2.1) (A₂B₂C₂.2.1) ∧
    parallel (ABC.2.1) (A₂B₂C₂.2.1) (ABC.2.2) (A₂B₂C₂.2.2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l976_97690


namespace NUMINAMATH_CALUDE_cube_coloring_count_l976_97603

/-- Represents a coloring scheme for a cube -/
structure CubeColoring where
  /-- The number of faces on the cube -/
  faces : Nat
  /-- The number of available colors -/
  colors : Nat
  /-- The number of faces already colored -/
  colored_faces : Nat
  /-- Function to check if a coloring scheme is valid -/
  is_valid : (List Nat) → Bool

/-- Counts the number of valid coloring schemes for a cube -/
def count_valid_colorings (c : CubeColoring) : Nat :=
  sorry

/-- Theorem stating that there are exactly 13 valid coloring schemes for a cube
    with 6 faces, 5 colors, and 3 faces already colored -/
theorem cube_coloring_count :
  ∃ (c : CubeColoring),
    c.faces = 6 ∧
    c.colors = 5 ∧
    c.colored_faces = 3 ∧
    count_valid_colorings c = 13 :=
  sorry

end NUMINAMATH_CALUDE_cube_coloring_count_l976_97603


namespace NUMINAMATH_CALUDE_ms_leech_class_boys_l976_97686

/-- Proves that the number of boys in Ms. Leech's class is 10 -/
theorem ms_leech_class_boys (total_students : ℕ) (total_cups : ℕ) (cups_per_boy : ℕ) :
  total_students = 30 →
  total_cups = 90 →
  cups_per_boy = 5 →
  ∃ (boys : ℕ),
    boys * 3 = total_students ∧
    boys * cups_per_boy = total_cups / 2 ∧
    boys = 10 :=
by sorry

end NUMINAMATH_CALUDE_ms_leech_class_boys_l976_97686


namespace NUMINAMATH_CALUDE_car_trip_duration_l976_97693

theorem car_trip_duration : ∀ (x : ℝ),
  let d1 : ℝ := 70 * 4  -- distance covered in first segment
  let d2 : ℝ := 60 * 5  -- distance covered in second segment
  let d3 : ℝ := 50 * x  -- distance covered in third segment
  let total_distance : ℝ := d1 + d2 + d3
  let total_time : ℝ := 4 + 5 + x
  let average_speed : ℝ := 58
  average_speed = total_distance / total_time →
  total_time = 16.25 :=
by sorry


end NUMINAMATH_CALUDE_car_trip_duration_l976_97693


namespace NUMINAMATH_CALUDE_fifth_minus_fourth_cube_volume_l976_97629

/-- The volume of a cube with side length n -/
def cube_volume (n : ℕ) : ℕ := n ^ 3

/-- The difference in volume between two cubes in the sequence -/
def volume_difference (m n : ℕ) : ℕ := cube_volume m - cube_volume n

theorem fifth_minus_fourth_cube_volume : volume_difference 5 4 = 61 := by
  sorry

end NUMINAMATH_CALUDE_fifth_minus_fourth_cube_volume_l976_97629


namespace NUMINAMATH_CALUDE_prob_same_color_top_three_l976_97647

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of cards of each color (red or black) in a standard deck -/
def cardsPerColor : ℕ := standardDeckSize / 2

/-- The probability of drawing three cards of the same color from the top of a randomly arranged standard deck -/
def probSameColorTopThree : ℚ :=
  (2 * cardsPerColor * (cardsPerColor - 1) * (cardsPerColor - 2)) /
  (standardDeckSize * (standardDeckSize - 1) * (standardDeckSize - 2))

theorem prob_same_color_top_three :
  probSameColorTopThree = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_top_three_l976_97647


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l976_97609

theorem fraction_sum_equals_decimal : 
  (4 : ℚ) / 100 - 8 / 10 + 3 / 1000 + 2 / 10000 = -0.7568 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l976_97609


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l976_97655

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (40 * π / 180) + Real.tan (60 * π / 180)) / Real.cos (10 * π / 180) =
  Real.sqrt 3 * ((1/2 * Real.cos (10 * π / 180) + Real.sqrt 3 / 2) / (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (40 * π / 180))) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l976_97655


namespace NUMINAMATH_CALUDE_car_distance_proof_l976_97631

/-- Proves that the distance a car needs to cover is 630 km, given the original time, 
    new time factor, and new speed. -/
theorem car_distance_proof (original_time : ℝ) (new_time_factor : ℝ) (new_speed : ℝ) : 
  original_time = 6 → 
  new_time_factor = 3 / 2 → 
  new_speed = 70 → 
  original_time * new_time_factor * new_speed = 630 := by
  sorry

#check car_distance_proof

end NUMINAMATH_CALUDE_car_distance_proof_l976_97631


namespace NUMINAMATH_CALUDE_total_budget_is_40_l976_97644

/-- The total budget for Samuel and Kevin's cinema outing -/
def total_budget : ℕ :=
  let samuel_ticket := 14
  let samuel_snacks := 6
  let kevin_ticket := 14
  let kevin_drinks := 2
  let kevin_food := 4
  samuel_ticket + samuel_snacks + kevin_ticket + kevin_drinks + kevin_food

/-- Theorem stating that the total budget for the outing is $40 -/
theorem total_budget_is_40 : total_budget = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_budget_is_40_l976_97644


namespace NUMINAMATH_CALUDE_variance_scaling_l976_97641

def variance (data : List ℝ) : ℝ := sorry

theorem variance_scaling (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  let data₁ := [a₁, a₂, a₃, a₄, a₅, a₆]
  let data₂ := [2*a₁, 2*a₂, 2*a₃, 2*a₄, 2*a₅, 2*a₆]
  variance data₁ = 2 → variance data₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_variance_scaling_l976_97641


namespace NUMINAMATH_CALUDE_shadow_length_sequence_l976_97675

/-- Represents the position of a person relative to a street lamp -/
inductive Position
  | Before
  | Under
  | After

/-- Represents the length of a shadow -/
inductive ShadowLength
  | Long
  | Short

/-- A street lamp as a fixed light source -/
structure StreetLamp where
  position : ℝ × ℝ  -- (x, y) coordinates
  height : ℝ

/-- A person walking past a street lamp -/
structure Person where
  height : ℝ

/-- Calculates the shadow length based on the person's position relative to the lamp -/
def shadowLength (lamp : StreetLamp) (person : Person) (pos : Position) : ShadowLength :=
  sorry

/-- Theorem stating how the shadow length changes as a person walks under a street lamp -/
theorem shadow_length_sequence (lamp : StreetLamp) (person : Person) :
  shadowLength lamp person Position.Before = ShadowLength.Long ∧
  shadowLength lamp person Position.Under = ShadowLength.Short ∧
  shadowLength lamp person Position.After = ShadowLength.Long :=
sorry

end NUMINAMATH_CALUDE_shadow_length_sequence_l976_97675


namespace NUMINAMATH_CALUDE_min_dot_product_ep_qp_l976_97660

/-- The ellipse defined by x^2/36 + y^2/9 = 1 -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

/-- The fixed point E -/
def E : ℝ × ℝ := (3, 0)

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- The squared distance between two points -/
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The minimum value of EP · QP is 6 -/
theorem min_dot_product_ep_qp :
  ∃ (min : ℝ),
    (∀ (P Q : ℝ × ℝ),
      is_on_ellipse P.1 P.2 →
      is_on_ellipse Q.1 Q.2 →
      dot_product (P.1 - E.1, P.2 - E.2) (Q.1 - P.1, Q.2 - P.2) = 0 →
      dot_product (P.1 - E.1, P.2 - E.2) (Q.1 - P.1, Q.2 - P.2) ≥ min) ∧
    min = 6 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_ep_qp_l976_97660


namespace NUMINAMATH_CALUDE_johns_furniture_purchase_l976_97683

theorem johns_furniture_purchase (chair_price table_price couch_price total_price : ℝ) :
  chair_price > 0 ∧
  table_price = 3 * chair_price ∧
  couch_price = 5 * table_price ∧
  total_price = chair_price + table_price + couch_price ∧
  total_price = 380 →
  couch_price = 300 := by
  sorry

end NUMINAMATH_CALUDE_johns_furniture_purchase_l976_97683


namespace NUMINAMATH_CALUDE_exists_non_polynomial_satisfying_inequality_l976_97681

-- Define a periodic function with period 2
def periodic_function (k : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, k (x + 2) = k x

-- Define a bounded function
def bounded_function (k : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ x : ℝ, |k x| ≤ M

-- Define a non-constant function
def non_constant_function (k : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, k x ≠ k y

-- Main theorem
theorem exists_non_polynomial_satisfying_inequality :
  ∃ f : ℝ → ℝ, 
    (∀ x : ℝ, (x - 1) * f (x + 1) - (x + 1) * f (x - 1) ≥ 4 * x * (x^2 - 1)) ∧
    (∃ k : ℝ → ℝ, 
      (∀ x : ℝ, f x = x^3 + x * k x) ∧
      periodic_function k ∧
      bounded_function k ∧
      non_constant_function k) :=
sorry

end NUMINAMATH_CALUDE_exists_non_polynomial_satisfying_inequality_l976_97681


namespace NUMINAMATH_CALUDE_calculator_theorem_l976_97656

/-- Represents the state of the calculator as a 4-tuple of real numbers -/
def CalculatorState := Fin 4 → ℝ

/-- Applies the transformation to a given state -/
def transform (s : CalculatorState) : CalculatorState :=
  fun i => match i with
  | 0 => s 0 - s 1
  | 1 => s 1 - s 2
  | 2 => s 2 - s 3
  | 3 => s 3 - s 0

/-- Applies the transformation n times to a given state -/
def transformN (s : CalculatorState) (n : ℕ) : CalculatorState :=
  match n with
  | 0 => s
  | n + 1 => transform (transformN s n)

/-- Checks if any number in the state is greater than 1985 -/
def hasLargeNumber (s : CalculatorState) : Prop :=
  ∃ i : Fin 4, s i > 1985

/-- Main theorem statement -/
theorem calculator_theorem (s : CalculatorState) 
  (h : ∃ i j : Fin 4, s i ≠ s j) : 
  ∃ n : ℕ, hasLargeNumber (transformN s n) := by
sorry

end NUMINAMATH_CALUDE_calculator_theorem_l976_97656


namespace NUMINAMATH_CALUDE_prob_A_wins_four_consecutive_prob_need_fifth_game_prob_C_ultimate_winner_l976_97635

/-- Represents a player in the badminton game --/
inductive Player : Type
  | A
  | B
  | C

/-- Represents the state of the game --/
structure GameState :=
  (current_players : List Player)
  (bye_player : Player)
  (eliminated_player : Option Player)

/-- The probability of a player winning a single game --/
def win_probability : ℚ := 1/2

/-- The initial game state --/
def initial_state : GameState :=
  { current_players := [Player.A, Player.B],
    bye_player := Player.C,
    eliminated_player := none }

/-- Calculates the probability of a specific game outcome --/
def outcome_probability (num_games : ℕ) : ℚ :=
  (win_probability ^ num_games : ℚ)

/-- Theorem stating the probability of A winning four consecutive games --/
theorem prob_A_wins_four_consecutive :
  outcome_probability 4 = 1/16 := by sorry

/-- Theorem stating the probability of needing a fifth game --/
theorem prob_need_fifth_game :
  1 - 4 * outcome_probability 4 = 3/4 := by sorry

/-- Theorem stating the probability of C being the ultimate winner --/
theorem prob_C_ultimate_winner :
  7/16 = 1 - 2 * (outcome_probability 4 + 7 * outcome_probability 5) := by sorry

end NUMINAMATH_CALUDE_prob_A_wins_four_consecutive_prob_need_fifth_game_prob_C_ultimate_winner_l976_97635


namespace NUMINAMATH_CALUDE_unique_triangle_configuration_l976_97611

/-- Represents a stick with a positive length -/
structure Stick where
  length : ℝ
  positive : length > 0

/-- Represents a triangle formed by three sticks -/
structure Triangle where
  a : Stick
  b : Stick
  c : Stick
  valid : a.length + b.length > c.length ∧
          a.length + c.length > b.length ∧
          b.length + c.length > a.length

/-- A configuration of 15 sticks forming 5 triangles -/
structure Configuration where
  sticks : Fin 15 → Stick
  triangles : Fin 5 → Triangle
  uses_all_sticks : ∀ s : Fin 15, ∃ t : Fin 5, (triangles t).a = sticks s ∨
                                               (triangles t).b = sticks s ∨
                                               (triangles t).c = sticks s

/-- Theorem stating that there's only one way to form 5 triangles from 15 sticks -/
theorem unique_triangle_configuration (c1 c2 : Configuration) : c1 = c2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_configuration_l976_97611


namespace NUMINAMATH_CALUDE_sequence_inequality_l976_97605

theorem sequence_inequality (x : ℕ → ℝ) 
  (h1 : x 1 = 3)
  (h2 : ∀ n : ℕ, 4 * x (n + 1) - 3 * x n < 2)
  (h3 : ∀ n : ℕ, 2 * x (n + 1) - x n < 2) :
  ∀ n : ℕ, 2 + (1/2)^n < x (n + 1) ∧ x (n + 1) < 2 + (3/4)^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l976_97605


namespace NUMINAMATH_CALUDE_problem_solving_probability_l976_97688

theorem problem_solving_probability 
  (kyle_prob : ℚ) 
  (david_prob : ℚ) 
  (catherine_prob : ℚ) 
  (h1 : kyle_prob = 1/3) 
  (h2 : david_prob = 2/7) 
  (h3 : catherine_prob = 5/9) : 
  kyle_prob * catherine_prob * (1 - david_prob) = 25/189 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l976_97688


namespace NUMINAMATH_CALUDE_correct_calculation_l976_97626

theorem correct_calculation (x : ℚ) : x - 13/5 = 9/7 → x + 13/5 = 227/35 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l976_97626


namespace NUMINAMATH_CALUDE_sum_of_fractions_between_18_and_19_l976_97608

theorem sum_of_fractions_between_18_and_19 :
  let a : ℚ := 2 + 3/8
  let b : ℚ := 4 + 1/3
  let c : ℚ := 5 + 2/21
  let d : ℚ := 6 + 1/11
  18 < a + b + c + d ∧ a + b + c + d < 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_between_18_and_19_l976_97608


namespace NUMINAMATH_CALUDE_math_team_combinations_l976_97653

theorem math_team_combinations (girls : ℕ) (boys : ℕ) : girls = 4 → boys = 5 → (girls.choose 3) * (boys.choose 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l976_97653


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_eq_2_subset_complement_iff_m_geq_3_l976_97689

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x : ℝ | x > 2^m}
def B : Set ℝ := {x : ℝ | -4 < x - 4 ∧ x - 4 < 4}

-- Theorem for part (1)
theorem intersection_and_union_when_m_eq_2 :
  (A 2 ∩ B = {x : ℝ | 4 < x ∧ x < 8}) ∧
  (A 2 ∪ B = {x : ℝ | x > 0}) := by sorry

-- Theorem for part (2)
theorem subset_complement_iff_m_geq_3 (m : ℝ) :
  A m ⊆ (Set.univ \ B) ↔ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_eq_2_subset_complement_iff_m_geq_3_l976_97689


namespace NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l976_97649

theorem cauchy_schwarz_like_inequality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l976_97649


namespace NUMINAMATH_CALUDE_event_selection_methods_l976_97657

def total_students : ℕ := 5
def selected_students : ℕ := 4
def num_days : ℕ := 3
def friday_attendees : ℕ := 2
def saturday_attendees : ℕ := 1
def sunday_attendees : ℕ := 1

theorem event_selection_methods :
  (Nat.choose total_students friday_attendees) *
  (Nat.choose (total_students - friday_attendees) saturday_attendees) *
  (Nat.choose (total_students - friday_attendees - saturday_attendees) sunday_attendees) = 60 := by
  sorry

end NUMINAMATH_CALUDE_event_selection_methods_l976_97657


namespace NUMINAMATH_CALUDE_exists_universal_source_l976_97685

/-- A directed graph where every pair of vertices is connected by a directed edge. -/
structure CompleteDigraph (V : Type*) [Fintype V] [DecidableEq V] :=
  (edge : V → V → Prop)
  (complete : ∀ (u v : V), u ≠ v → edge u v ∨ edge v u)

/-- A path of length at most 2 between two vertices. -/
def PathOfLengthAtMostTwo {V : Type*} (edge : V → V → Prop) (u v : V) : Prop :=
  edge u v ∨ ∃ w, edge u w ∧ edge w v

/-- 
In a complete directed graph, there exists a vertex from which 
every other vertex can be reached by a path of length at most 2.
-/
theorem exists_universal_source {V : Type*} [Fintype V] [DecidableEq V] 
  (G : CompleteDigraph V) : 
  ∃ (u : V), ∀ (v : V), u ≠ v → PathOfLengthAtMostTwo G.edge u v :=
sorry

end NUMINAMATH_CALUDE_exists_universal_source_l976_97685


namespace NUMINAMATH_CALUDE_unique_polynomial_composition_l976_97646

theorem unique_polynomial_composition (a b c : ℝ) (n : ℕ) (h : a ≠ 0) :
  ∃! Q : Polynomial ℝ,
    (Polynomial.degree Q = n) ∧
    (∀ x : ℝ, Q.eval (a * x^2 + b * x + c) = a * (Q.eval x)^2 + b * (Q.eval x) + c) := by
  sorry

end NUMINAMATH_CALUDE_unique_polynomial_composition_l976_97646


namespace NUMINAMATH_CALUDE_sarah_apples_left_l976_97601

def apples_left (initial : ℕ) (teachers : ℕ) (friends : ℕ) (eaten : ℕ) : ℕ :=
  initial - (teachers + friends + eaten)

theorem sarah_apples_left :
  apples_left 25 16 5 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sarah_apples_left_l976_97601


namespace NUMINAMATH_CALUDE_remaining_work_time_for_x_l976_97615

-- Define the work rates and work durations
def x_rate : ℚ := 1 / 30
def y_rate : ℚ := 1 / 15
def z_rate : ℚ := 1 / 20
def y_work_days : ℕ := 10
def z_work_days : ℕ := 5

-- Define the theorem
theorem remaining_work_time_for_x :
  let total_work : ℚ := 1
  let work_done_by_y : ℚ := y_rate * y_work_days
  let work_done_by_z : ℚ := z_rate * z_work_days
  let remaining_work : ℚ := total_work - (work_done_by_y + work_done_by_z)
  remaining_work / x_rate = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_remaining_work_time_for_x_l976_97615


namespace NUMINAMATH_CALUDE_scientific_notation_of_2102000_l976_97695

theorem scientific_notation_of_2102000 :
  ∃ (a : ℝ) (n : ℤ), 2102000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.102 ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2102000_l976_97695


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l976_97617

theorem average_of_a_and_b (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 70)
  (h3 : c - a = 50) :
  (a + b) / 2 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l976_97617


namespace NUMINAMATH_CALUDE_projectile_max_height_l976_97624

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 116

/-- Theorem stating that the maximum height reached by the projectile is 116 feet -/
theorem projectile_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l976_97624


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l976_97628

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ),
    n % 4 = 0 ∧
    n % 9 = 0 ∧
    n = 94444 + 90000 * k ∧
    k ≥ 0

theorem smallest_valid_number_last_four_digits :
  ∃ (n : ℕ),
    is_valid_number n ∧
    (∀ m, is_valid_number m → n ≤ m) ∧
    n % 10000 = 4444 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l976_97628


namespace NUMINAMATH_CALUDE_matrix_product_equality_l976_97632

theorem matrix_product_equality : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 3, -1; 1, 5, -2; 0, 6, 2]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![3, -1, 0; 2, 1, -4; 5, 0, 1]
  A * B = !![7, 1, -13; 3, 4, -22; 22, 6, -22] := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l976_97632


namespace NUMINAMATH_CALUDE_complex_number_real_part_eq_imaginary_part_l976_97697

theorem complex_number_real_part_eq_imaginary_part (a : ℝ) : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 - 2*i) * (a + i)
  (z.re + z.im = 0) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_part_eq_imaginary_part_l976_97697


namespace NUMINAMATH_CALUDE_vector_equations_true_l976_97665

-- Define a vector space over the real numbers
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define points A, B, and C
variable (A B C : V)

-- Theorem statement
theorem vector_equations_true :
  (a + b = b + a) ∧
  (-(-a) = a) ∧
  ((B - A) + (C - B) + (A - C) = 0) ∧
  (a + (-a) = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_equations_true_l976_97665


namespace NUMINAMATH_CALUDE_min_positive_period_of_tan_2x_l976_97672

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem min_positive_period_of_tan_2x :
  ∃ (T : ℝ), T > 0 ∧ T = π / 2 ∧
  (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T' ≥ T) :=
sorry

end NUMINAMATH_CALUDE_min_positive_period_of_tan_2x_l976_97672


namespace NUMINAMATH_CALUDE_only_6_8_10_is_right_triangle_l976_97664

-- Define a function to check if three numbers can form a right triangle
def isRightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that only (6, 8, 10) forms a right triangle among the given sets
theorem only_6_8_10_is_right_triangle :
  ¬(isRightTriangle 4 5 6) ∧
  ¬(isRightTriangle 5 7 9) ∧
  isRightTriangle 6 8 10 ∧
  ¬(isRightTriangle 7 8 9) :=
sorry

end NUMINAMATH_CALUDE_only_6_8_10_is_right_triangle_l976_97664


namespace NUMINAMATH_CALUDE_first_player_wins_l976_97673

/-- Represents the state of the game with two piles of tokens -/
structure GameState :=
  (pile1 : Nat) (pile2 : Nat)

/-- Defines a valid move in the game -/
inductive ValidMove : GameState → GameState → Prop
  | single_pile (s t : GameState) (i : Fin 2) (n : Nat) :
      n > 0 →
      (i = 0 → t.pile1 = s.pile1 - n ∧ t.pile2 = s.pile2) →
      (i = 1 → t.pile1 = s.pile1 ∧ t.pile2 = s.pile2 - n) →
      ValidMove s t
  | both_piles (s t : GameState) (x y : Nat) :
      x > 0 →
      y > 0 →
      (x + y) % 2015 = 0 →
      t.pile1 = s.pile1 - x →
      t.pile2 = s.pile2 - y →
      ValidMove s t

/-- Defines the winning condition -/
def IsWinningState (s : GameState) : Prop :=
  ∀ t : GameState, ¬ValidMove s t

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → GameState),
    let initial_state := GameState.mk 10000 20000
    ∀ (opponent_move : GameState → GameState),
      ValidMove initial_state (strategy initial_state) →
      (∀ s, ValidMove s (opponent_move s) → ValidMove (opponent_move s) (strategy (opponent_move s))) →
      ∃ n : Nat, IsWinningState (Nat.iterate (λ s => strategy (opponent_move s)) n (strategy initial_state)) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l976_97673


namespace NUMINAMATH_CALUDE_range_of_a_l976_97652

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - x + a ≥ 0) → a ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l976_97652


namespace NUMINAMATH_CALUDE_sum_equals_twelve_l976_97676

theorem sum_equals_twelve (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_twelve_l976_97676


namespace NUMINAMATH_CALUDE_solve_refrigerator_problem_l976_97618

def refrigerator_problem (purchase_price installation_cost transport_cost selling_price : ℚ) : Prop :=
  let discount_rate : ℚ := 20 / 100
  let profit_rate : ℚ := 10 / 100
  let labelled_price : ℚ := purchase_price / (1 - discount_rate)
  let total_cost : ℚ := labelled_price + installation_cost + transport_cost
  (1 + profit_rate) * total_cost = selling_price

theorem solve_refrigerator_problem :
  refrigerator_problem 17500 250 125 24475 := by sorry

end NUMINAMATH_CALUDE_solve_refrigerator_problem_l976_97618


namespace NUMINAMATH_CALUDE_exponent_subtraction_minus_fifteen_l976_97625

theorem exponent_subtraction_minus_fifteen :
  (23^11 / 23^8) - 15 = 12152 := by sorry

end NUMINAMATH_CALUDE_exponent_subtraction_minus_fifteen_l976_97625


namespace NUMINAMATH_CALUDE_cubic_factorization_l976_97659

theorem cubic_factorization (x : ℝ) : x^3 + 5*x^2 + 6*x = x*(x+2)*(x+3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l976_97659


namespace NUMINAMATH_CALUDE_market_demand_growth_rate_bound_l976_97662

theorem market_demand_growth_rate_bound
  (a : Fin 4 → ℝ)  -- Market demand sequence for 4 years
  (p₁ p₂ p₃ : ℝ)   -- Percentage increases between consecutive years
  (h₁ : p₁ + p₂ + p₃ = 1)  -- Condition on percentage increases
  (h₂ : ∀ i : Fin 3, a (i + 1) = a i * (1 + [p₁, p₂, p₃].get i))  -- Relation between consecutive demands
  : ∃ p : ℝ, (∀ i : Fin 3, a (i + 1) = a i * (1 + p)) ∧ p ≤ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_market_demand_growth_rate_bound_l976_97662


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l976_97666

/-- A rectangle with length thrice its breadth and area 675 square meters has a perimeter of 120 meters. -/
theorem rectangle_perimeter (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let area := l * b
  area = 675 →
  2 * (l + b) = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l976_97666


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l976_97698

-- Define the population
structure Population where
  grades : List String
  students : List String

-- Define the sampling methods
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Stratified
  | Systematic

-- Define the survey requirements
structure SurveyRequirements where
  proportional_sampling : Bool
  multiple_grades : Bool

-- Define a function to determine the most appropriate sampling method
def most_appropriate_method (pop : Population) (req : SurveyRequirements) : SamplingMethod :=
  sorry

-- Theorem stating that stratified sampling is most appropriate
-- for a population with multiple grades and proportional sampling requirement
theorem stratified_sampling_most_appropriate 
  (pop : Population) 
  (req : SurveyRequirements) :
  pop.grades.length > 1 → 
  req.proportional_sampling = true → 
  req.multiple_grades = true → 
  most_appropriate_method pop req = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l976_97698


namespace NUMINAMATH_CALUDE_intersection_A_B_l976_97627

def A : Set Int := {1, 2, 3, 4, 5}

def B : Set Int := {x | (x - 1) / (4 - x) > 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l976_97627


namespace NUMINAMATH_CALUDE_min_value_inequality_l976_97669

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((2 * x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l976_97669


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l976_97694

theorem youngest_sibling_age (a b c d : ℕ) : 
  a + b + c + d = 180 →
  b = a + 2 →
  c = a + 4 →
  d = a + 6 →
  Even a →
  Even b →
  Even c →
  Even d →
  a = 42 := by
sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l976_97694


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l976_97610

/-- 
Given a quadratic equation kx^2 - 6x + 9 = 0, this theorem states that
for the equation to have two distinct real roots, k must be less than 1
and not equal to 0.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6*x + 9 = 0 ∧ k * y^2 - 6*y + 9 = 0) ↔ 
  (k < 1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l976_97610


namespace NUMINAMATH_CALUDE_inscribed_prism_volume_l976_97638

/-- Regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  /-- Radius of the sphere -/
  R : ℝ
  /-- Distance from vertex A to point D on the sphere -/
  AD : ℝ
  /-- Assertion that CD is a diameter of the sphere -/
  is_diameter : Bool

/-- Volume of the inscribed prism -/
def prism_volume (p : InscribedPrism) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific inscribed prism -/
theorem inscribed_prism_volume :
  ∀ (p : InscribedPrism),
    p.R = 3 ∧ p.AD = 2 * Real.sqrt 6 ∧ p.is_diameter = true →
    prism_volume p = 6 * Real.sqrt 15 :=
  sorry

end NUMINAMATH_CALUDE_inscribed_prism_volume_l976_97638


namespace NUMINAMATH_CALUDE_jasons_quarters_l976_97606

/-- Given an initial amount of quarters and an additional amount,
    calculate the total number of quarters. -/
def total_quarters (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Jason's total number of quarters -/
theorem jasons_quarters :
  total_quarters 49 25 = 74 := by
  sorry

end NUMINAMATH_CALUDE_jasons_quarters_l976_97606


namespace NUMINAMATH_CALUDE_contractor_engagement_l976_97604

/-- Represents the contractor's engagement problem -/
def ContractorProblem (daily_wage : ℚ) (daily_fine : ℚ) (total_earnings : ℚ) (absent_days : ℕ) : Prop :=
  ∃ (working_days : ℕ),
    daily_wage * working_days - daily_fine * absent_days = total_earnings ∧
    working_days + absent_days = 30

/-- The theorem states that given the problem conditions, the total engagement days is 30 -/
theorem contractor_engagement :
  ContractorProblem 25 7.5 425 10 :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_l976_97604


namespace NUMINAMATH_CALUDE_natalie_shopping_result_l976_97663

/-- Calculates the amount of money Natalie has left after shopping -/
def money_left (initial_amount jumper_price tshirt_price heels_price jumper_discount_rate sales_tax_rate : ℚ) : ℚ :=
  let discounted_jumper_price := jumper_price * (1 - jumper_discount_rate)
  let total_before_tax := discounted_jumper_price + tshirt_price + heels_price
  let total_after_tax := total_before_tax * (1 + sales_tax_rate)
  initial_amount - total_after_tax

/-- Theorem stating that Natalie has $18.62 left after shopping -/
theorem natalie_shopping_result : 
  money_left 100 25 15 40 (10/100) (5/100) = 18.62 := by
  sorry

end NUMINAMATH_CALUDE_natalie_shopping_result_l976_97663


namespace NUMINAMATH_CALUDE_triangle_side_ratio_bounds_l976_97651

theorem triangle_side_ratio_bounds (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  let t := (a + b + c) / Real.sqrt (a * b + b * c + c * a)
  Real.sqrt 3 ≤ t ∧ t < 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_bounds_l976_97651


namespace NUMINAMATH_CALUDE_transformed_area_l976_97678

-- Define the matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 2; 4, -5]

-- Define the area of the original region R
def area_R : ℝ := 15

-- Theorem statement
theorem transformed_area :
  let det_A := Matrix.det A
  area_R * |det_A| = 345 := by
sorry

end NUMINAMATH_CALUDE_transformed_area_l976_97678


namespace NUMINAMATH_CALUDE_hiker_cyclist_catchup_time_l976_97637

/-- Proves that a hiker catches up to a cyclist in 10 minutes under specific conditions -/
theorem hiker_cyclist_catchup_time :
  let hiker_speed : ℝ := 4  -- km/h
  let cyclist_speed : ℝ := 12  -- km/h
  let stop_time : ℝ := 5 / 60  -- hours (5 minutes converted to hours)
  
  let distance_cyclist : ℝ := cyclist_speed * stop_time
  let distance_hiker : ℝ := hiker_speed * stop_time
  let distance_between : ℝ := distance_cyclist - distance_hiker
  
  let catchup_time : ℝ := distance_between / hiker_speed

  catchup_time * 60 = 10  -- Convert hours to minutes
  := by sorry

end NUMINAMATH_CALUDE_hiker_cyclist_catchup_time_l976_97637


namespace NUMINAMATH_CALUDE_triangle_properties_l976_97668

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

/-- The given triangle satisfies the specified conditions -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (2 * t.c - t.b) * Real.cos t.A ∧
  t.a = Real.sqrt 7 ∧
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.A = π / 3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l976_97668


namespace NUMINAMATH_CALUDE_solution_equality_l976_97634

theorem solution_equality (a : ℝ) : 
  (∃ x, 2 - a - x = 0 ∧ 2*x + 1 = 3) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_equality_l976_97634


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l976_97677

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ = -6 and a₇ = a₅ + 4, prove that a₁ = -10 -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = -6) 
  (h_a7 : a 7 = a 5 + 4) : 
  a 1 = -10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l976_97677


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l976_97671

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l976_97671


namespace NUMINAMATH_CALUDE_distance_walked_when_meeting_l976_97696

/-- 
Given two people walking towards each other from a distance of 50 miles,
each at a constant speed of 5 miles per hour, prove that one person
will have walked 25 miles when they meet.
-/
theorem distance_walked_when_meeting 
  (initial_distance : ℝ) 
  (speed : ℝ) 
  (h1 : initial_distance = 50)
  (h2 : speed = 5) : 
  (initial_distance / (2 * speed)) * speed = 25 :=
by sorry

end NUMINAMATH_CALUDE_distance_walked_when_meeting_l976_97696


namespace NUMINAMATH_CALUDE_rhombus_point_d_y_coord_rhombus_point_d_x_coord_l976_97633

/-- A rhombus ABCD with specific properties -/
structure Rhombus where
  /-- The y-coordinate of point B -/
  b : ℝ
  /-- The x-coordinate of point D -/
  x : ℝ
  /-- The y-coordinate of point D -/
  y : ℝ
  /-- B is on the negative half of the y-axis -/
  h_b_neg : b < 0
  /-- ABCD is a rhombus -/
  h_is_rhombus : True  -- This is a placeholder, as we can't directly express "is_rhombus" without further definitions
  /-- The intersection of diagonals M is on the x-axis -/
  h_m_on_x_axis : True  -- This is a placeholder, as we can't directly express this geometric property without further definitions

/-- The main theorem about the y-coordinate of point D -/
theorem rhombus_point_d_y_coord (r : Rhombus) : r.y = -r.b - 1 := by sorry

/-- The x-coordinate of point D can be any real number -/
theorem rhombus_point_d_x_coord (r : Rhombus) : r.x ∈ Set.univ := by sorry

end NUMINAMATH_CALUDE_rhombus_point_d_y_coord_rhombus_point_d_x_coord_l976_97633


namespace NUMINAMATH_CALUDE_professors_simultaneous_probability_l976_97630

/-- The duration the cafeteria is open, in minutes -/
def cafeteria_open_duration : ℕ := 120

/-- The duration of each professor's lunch, in minutes -/
def lunch_duration : ℕ := 15

/-- The latest possible start time for lunch, in minutes after the cafeteria opens -/
def latest_start_time : ℕ := cafeteria_open_duration - lunch_duration

/-- The probability that two professors are in the cafeteria simultaneously -/
theorem professors_simultaneous_probability : 
  (lunch_duration * latest_start_time : ℚ) / (latest_start_time^2 : ℚ) = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_professors_simultaneous_probability_l976_97630


namespace NUMINAMATH_CALUDE_unique_number_satisfying_equation_l976_97636

theorem unique_number_satisfying_equation : ∃! x : ℝ, ((x^3)^(1/3) * 4) / 2 + 5 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_equation_l976_97636


namespace NUMINAMATH_CALUDE_quadratic_points_range_l976_97654

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * a * x + 3

-- Define the theorem
theorem quadratic_points_range (a m y₁ y₂ : ℝ) :
  a > 0 →
  y₁ < y₂ →
  f a (m - 1) = y₁ →
  f a m = y₂ →
  m > -3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_points_range_l976_97654


namespace NUMINAMATH_CALUDE_stream_speed_l976_97621

/-- Given a canoe that rows upstream at 6 km/hr and downstream at 10 km/hr, 
    the speed of the stream is 2 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 6)
  (h2 : downstream_speed = 10) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 2 := by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l976_97621


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l976_97613

/-- The perimeter of a semicircle with radius 14 cm is equal to 14π + 28 cm. -/
theorem semicircle_perimeter :
  let r : ℝ := 14
  let π : ℝ := Real.pi
  let semicircle_perimeter : ℝ := r * π + 2 * r
  semicircle_perimeter = 14 * π + 28 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l976_97613


namespace NUMINAMATH_CALUDE_max_prime_factors_b_l976_97648

theorem max_prime_factors_b (a b : ℕ+) 
  (h_gcd : (Nat.gcd a b).factors.length = 10)
  (h_lcm : (Nat.lcm a b).factors.length = 25)
  (h_fewer : (b.val.factors.length : ℤ) < a.val.factors.length) :
  b.val.factors.length ≤ 17 :=
sorry

end NUMINAMATH_CALUDE_max_prime_factors_b_l976_97648


namespace NUMINAMATH_CALUDE_jerry_sticker_count_jerry_has_36_stickers_l976_97623

theorem jerry_sticker_count (fred_stickers : ℕ) (george_diff : ℕ) (jerry_multiplier : ℕ) : ℕ :=
  let george_stickers := fred_stickers - george_diff
  let jerry_stickers := jerry_multiplier * george_stickers
  jerry_stickers

theorem jerry_has_36_stickers : 
  jerry_sticker_count 18 6 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jerry_sticker_count_jerry_has_36_stickers_l976_97623


namespace NUMINAMATH_CALUDE_x_y_inequalities_l976_97642

theorem x_y_inequalities (x y : ℝ) (h1 : x - y > x + 1) (h2 : x + y < y - 2) :
  x < -2 ∧ y < -1 := by sorry

end NUMINAMATH_CALUDE_x_y_inequalities_l976_97642


namespace NUMINAMATH_CALUDE_starting_lineup_count_l976_97682

theorem starting_lineup_count :
  let team_size : ℕ := 12
  let lineup_size : ℕ := 5
  let captain_count : ℕ := 1
  let other_players_count : ℕ := lineup_size - captain_count
  team_size * Nat.choose (team_size - captain_count) other_players_count = 3960 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l976_97682


namespace NUMINAMATH_CALUDE_x1_value_l976_97602

theorem x1_value (x₁ x₂ x₃ : ℝ) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h2 : (1 - x₁)^2 + 2*(x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/2) :
  x₁ = (3 * Real.sqrt 2 - 3) / 7 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l976_97602


namespace NUMINAMATH_CALUDE_tailors_hourly_rate_l976_97614

theorem tailors_hourly_rate (num_shirts : ℕ) (num_pants : ℕ) (shirt_time : ℚ) (total_cost : ℚ) :
  num_shirts = 10 →
  num_pants = 12 →
  shirt_time = 3/2 →
  total_cost = 1530 →
  (num_shirts * shirt_time + num_pants * (2 * shirt_time)) * (total_cost / (num_shirts * shirt_time + num_pants * (2 * shirt_time))) = 30 := by
  sorry

end NUMINAMATH_CALUDE_tailors_hourly_rate_l976_97614
