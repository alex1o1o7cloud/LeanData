import Mathlib

namespace NUMINAMATH_CALUDE_work_completion_time_l3831_383196

/-- Given that A can do a work in 12 days and A and B together can do the work in 8 days,
    prove that B can do the work alone in 24 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 12) (hab : 1 / a + 1 / b = 1 / 8) : b = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3831_383196


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3831_383131

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (9 * bowling_ball_weight = 5 * canoe_weight) →
    (4 * canoe_weight = 120) →
    bowling_ball_weight = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3831_383131


namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l3831_383197

/-- Given f(x) = (1/3)x³ + 2x + 1, prove that f'(-1) = 3 -/
theorem derivative_at_negative_one (f : ℝ → ℝ) (hf : ∀ x, f x = (1/3) * x^3 + 2*x + 1) :
  (deriv f) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l3831_383197


namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l3831_383186

theorem three_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l3831_383186


namespace NUMINAMATH_CALUDE_total_cost_after_discounts_l3831_383103

-- Define the original costs and discount percentages
def laptop_original_cost : ℚ := 800
def accessories_original_cost : ℚ := 200
def laptop_discount_percent : ℚ := 15
def accessories_discount_percent : ℚ := 10

-- Define the function to calculate the discounted price
def discounted_price (original_cost : ℚ) (discount_percent : ℚ) : ℚ :=
  original_cost * (1 - discount_percent / 100)

-- Theorem statement
theorem total_cost_after_discounts :
  discounted_price laptop_original_cost laptop_discount_percent +
  discounted_price accessories_original_cost accessories_discount_percent = 860 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_after_discounts_l3831_383103


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3831_383180

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line l
def line_l (p m : ℝ) (x y : ℝ) : Prop := x = m*y + p/2

-- Define points A and B on the parabola and line
def point_on_parabola_and_line (p m : ℝ) (x y : ℝ) : Prop :=
  parabola p x y ∧ line_l p m x y

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁*x₂ + y₁*y₂ = -3

-- Define the distance between two points
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Define the minimization condition
def is_minimum (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∀ x₁' y₁' x₂' y₂', point_on_parabola_and_line p ((x₁' + x₂')/2) x₁' y₁' →
    point_on_parabola_and_line p ((x₁' + x₂')/2) x₂' y₂' →
    dot_product_condition x₁' y₁' x₂' y₂' →
    (|x₁' - p/2| + 4*|x₂' - p/2| ≥ |x₁ - p/2| + 4*|x₂ - p/2|)

-- The main theorem
theorem parabola_intersection_theorem (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabola p x₁ y₁ →
  parabola p x₂ y₂ →
  (∃ m : ℝ, line_l p m x₁ y₁ ∧ line_l p m x₂ y₂) →
  dot_product_condition x₁ y₁ x₂ y₂ →
  is_minimum p x₁ y₁ x₂ y₂ →
  distance x₁ y₁ x₂ y₂ = 9/2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3831_383180


namespace NUMINAMATH_CALUDE_original_plums_count_l3831_383173

theorem original_plums_count (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 4 → total = 21 → initial + added = total → initial = 17 := by
sorry

end NUMINAMATH_CALUDE_original_plums_count_l3831_383173


namespace NUMINAMATH_CALUDE_rainfall_problem_l3831_383162

/-- Rainfall problem -/
theorem rainfall_problem (sunday monday tuesday : ℝ) 
  (h1 : tuesday = 2 * monday)
  (h2 : monday = sunday + 3)
  (h3 : sunday + monday + tuesday = 25) :
  sunday = 4 := by
sorry

end NUMINAMATH_CALUDE_rainfall_problem_l3831_383162


namespace NUMINAMATH_CALUDE_negative_two_less_than_negative_one_l3831_383190

theorem negative_two_less_than_negative_one : -2 < -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_less_than_negative_one_l3831_383190


namespace NUMINAMATH_CALUDE_pascal_contest_participants_l3831_383139

theorem pascal_contest_participants (male_count : ℕ) (ratio_male : ℕ) (ratio_female : ℕ) : 
  male_count = 21 → ratio_male = 3 → ratio_female = 7 → 
  male_count + (male_count * ratio_female / ratio_male) = 70 := by
sorry

end NUMINAMATH_CALUDE_pascal_contest_participants_l3831_383139


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3831_383113

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, (y = f x) ∧ (f' x = 4) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3831_383113


namespace NUMINAMATH_CALUDE_stating_total_dark_triangles_formula_l3831_383150

/-- 
Given a sequence of figures formed by an increasing number of dark equilateral triangles,
this function represents the total number of dark triangles used in the first n figures.
-/
def total_dark_triangles (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- 
Theorem stating that the total number of dark triangles used in the first n figures
of the sequence is (n(n+1)(n+2))/6.
-/
theorem total_dark_triangles_formula (n : ℕ) :
  total_dark_triangles n = n * (n + 1) * (n + 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_stating_total_dark_triangles_formula_l3831_383150


namespace NUMINAMATH_CALUDE_train_speed_problem_l3831_383159

/-- Given two trains traveling towards each other from cities 100 miles apart,
    with one train traveling at 30 mph and the trains meeting after 4/3 hours,
    prove that the speed of the other train is 45 mph. -/
theorem train_speed_problem (distance : ℝ) (speed1 : ℝ) (time : ℝ) (speed2 : ℝ) :
  distance = 100 →
  speed1 = 30 →
  time = 4/3 →
  distance = speed1 * time + speed2 * time →
  speed2 = 45 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3831_383159


namespace NUMINAMATH_CALUDE_sam_winning_probability_l3831_383117

theorem sam_winning_probability :
  let hit_prob : ℚ := 2/5
  let miss_prob : ℚ := 3/5
  let p : ℚ := hit_prob + miss_prob * miss_prob * p
  p = 5/8 := by sorry

end NUMINAMATH_CALUDE_sam_winning_probability_l3831_383117


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l3831_383137

/-- The number of different arrangements for 5 students and 2 teachers in a row,
    with exactly 2 students between the teachers. -/
def photo_arrangements : ℕ := 960

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students between the teachers -/
def students_between : ℕ := 2

theorem photo_arrangement_count :
  photo_arrangements = 960 ∧
  num_students = 5 ∧
  num_teachers = 2 ∧
  students_between = 2 := by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l3831_383137


namespace NUMINAMATH_CALUDE_simplify_expression_l3831_383122

theorem simplify_expression (x y : ℝ) : 7*x + 8*y - 3*x + 4*y + 10 = 4*x + 12*y + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3831_383122


namespace NUMINAMATH_CALUDE_car_speed_l3831_383125

/-- Given a car that travels 495 km in 5 hours, its speed is 99 km/h. -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 495 ∧ time = 5 ∧ speed = distance / time → speed = 99 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_l3831_383125


namespace NUMINAMATH_CALUDE_apple_ratio_is_half_l3831_383115

/-- The number of apples Anna ate on Tuesday -/
def tuesday_apples : ℕ := 4

/-- The number of apples Anna ate on Wednesday -/
def wednesday_apples : ℕ := 2 * tuesday_apples

/-- The total number of apples Anna ate over the three days -/
def total_apples : ℕ := 14

/-- The number of apples Anna ate on Thursday -/
def thursday_apples : ℕ := total_apples - tuesday_apples - wednesday_apples

/-- The ratio of apples eaten on Thursday to Tuesday -/
def thursday_to_tuesday_ratio : ℚ := thursday_apples / tuesday_apples

theorem apple_ratio_is_half : thursday_to_tuesday_ratio = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_apple_ratio_is_half_l3831_383115


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l3831_383132

/-- Proves that for a rectangular field with a square pond, given specific conditions, the ratio of length to width is 2:1 -/
theorem field_length_width_ratio (field_length field_width pond_side : ℝ) : 
  field_length = 28 →
  pond_side = 7 →
  field_length * field_width = 8 * pond_side * pond_side →
  field_length / field_width = 2 := by
  sorry

#check field_length_width_ratio

end NUMINAMATH_CALUDE_field_length_width_ratio_l3831_383132


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l3831_383134

theorem part_to_whole_ratio (N : ℝ) (P : ℝ) : 
  (1 / 4 : ℝ) * P = 10 →
  (40 / 100 : ℝ) * N = 120 →
  P / ((2 / 5 : ℝ) * N) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l3831_383134


namespace NUMINAMATH_CALUDE_polynomial_equality_l3831_383108

theorem polynomial_equality (P : ℝ → ℝ) :
  (∀ a b c : ℝ, P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)) →
  ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3831_383108


namespace NUMINAMATH_CALUDE_triangle_forming_sets_l3831_383151

/-- A function that checks if three numbers can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of numbers we're checking --/
def sets : List (ℝ × ℝ × ℝ) := [
  (1, 2, 3),
  (2, 3, 4),
  (3, 4, 5),
  (3, 6, 9)
]

/-- The theorem stating which sets can form triangles --/
theorem triangle_forming_sets :
  (∀ (a b c : ℝ), (a, b, c) ∈ sets → can_form_triangle a b c) ↔
  (∃ (a b c : ℝ), (a, b, c) ∈ sets ∧ can_form_triangle a b c ∧ (a, b, c) = (2, 3, 4)) ∧
  (∃ (a b c : ℝ), (a, b, c) ∈ sets ∧ can_form_triangle a b c ∧ (a, b, c) = (3, 4, 5)) ∧
  (∀ (a b c : ℝ), (a, b, c) ∈ sets → (a, b, c) ≠ (2, 3, 4) → (a, b, c) ≠ (3, 4, 5) → ¬can_form_triangle a b c) :=
sorry

end NUMINAMATH_CALUDE_triangle_forming_sets_l3831_383151


namespace NUMINAMATH_CALUDE_e1_e2_form_basis_l3831_383114

def e1 : ℝ × ℝ := (-1, 2)
def e2 : ℝ × ℝ := (5, 7)

def is_non_collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 ≠ 0

def forms_basis (v w : ℝ × ℝ) : Prop :=
  is_non_collinear v w

theorem e1_e2_form_basis : forms_basis e1 e2 := by
  sorry

end NUMINAMATH_CALUDE_e1_e2_form_basis_l3831_383114


namespace NUMINAMATH_CALUDE_new_shoes_lifespan_l3831_383168

/-- Proves that the lifespan of new shoes is 2 years given the costs and conditions -/
theorem new_shoes_lifespan (repair_cost : ℝ) (repair_lifespan : ℝ) (new_cost : ℝ) (cost_increase_percentage : ℝ) :
  repair_cost = 14.50 →
  repair_lifespan = 1 →
  new_cost = 32.00 →
  cost_increase_percentage = 10.344827586206897 →
  let new_lifespan := new_cost / (repair_cost * (1 + cost_increase_percentage / 100))
  new_lifespan = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_shoes_lifespan_l3831_383168


namespace NUMINAMATH_CALUDE_baseball_game_total_baseball_game_total_is_643_l3831_383158

/-- Represents the statistics of a baseball team for a single day -/
structure DayStats where
  misses : ℕ
  hits : ℕ
  singles : ℕ
  doubles : ℕ
  triples : ℕ
  homeRuns : ℕ

/-- Represents the statistics of a baseball team for three days -/
structure TeamStats where
  day1 : DayStats
  day2 : DayStats
  day3 : DayStats

theorem baseball_game_total (teamA teamB : TeamStats) : ℕ :=
  let totalMisses := teamA.day1.misses + teamA.day2.misses + teamA.day3.misses +
                     teamB.day1.misses + teamB.day2.misses + teamB.day3.misses
  let totalSingles := teamA.day1.singles + teamA.day2.singles + teamA.day3.singles +
                      teamB.day1.singles + teamB.day2.singles + teamB.day3.singles
  let totalDoubles := teamA.day1.doubles + teamA.day2.doubles + teamA.day3.doubles +
                      teamB.day1.doubles + teamB.day2.doubles + teamB.day3.doubles
  let totalTriples := teamA.day1.triples + teamA.day2.triples + teamA.day3.triples +
                      teamB.day1.triples + teamB.day2.triples + teamB.day3.triples
  let totalHomeRuns := teamA.day1.homeRuns + teamA.day2.homeRuns + teamA.day3.homeRuns +
                       teamB.day1.homeRuns + teamB.day2.homeRuns + teamB.day3.homeRuns
  totalMisses + totalSingles + totalDoubles + totalTriples + totalHomeRuns

theorem baseball_game_total_is_643 :
  let teamA : TeamStats := {
    day1 := { misses := 60, hits := 30, singles := 15, doubles := 0, triples := 0, homeRuns := 15 },
    day2 := { misses := 68, hits := 17, singles := 11, doubles := 6, triples := 0, homeRuns := 0 },
    day3 := { misses := 100, hits := 20, singles := 10, doubles := 0, triples := 5, homeRuns := 5 }
  }
  let teamB : TeamStats := {
    day1 := { misses := 90, hits := 30, singles := 15, doubles := 0, triples := 0, homeRuns := 15 },
    day2 := { misses := 56, hits := 28, singles := 19, doubles := 9, triples := 0, homeRuns := 0 },
    day3 := { misses := 120, hits := 24, singles := 12, doubles := 0, triples := 6, homeRuns := 6 }
  }
  baseball_game_total teamA teamB = 643 := by
  sorry

#check baseball_game_total_is_643

end NUMINAMATH_CALUDE_baseball_game_total_baseball_game_total_is_643_l3831_383158


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3831_383187

/-- Proves that the speed of a boat in still water is 24 km/hr, given the speed of the stream and the boat's downstream travel information. -/
theorem boat_speed_in_still_water :
  let stream_speed : ℝ := 4
  let downstream_distance : ℝ := 140
  let downstream_time : ℝ := 5
  let downstream_speed : ℝ := downstream_distance / downstream_time
  let boat_speed_still_water : ℝ := downstream_speed - stream_speed
  boat_speed_still_water = 24 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3831_383187


namespace NUMINAMATH_CALUDE_sales_theorem_l3831_383107

def sales_problem (last_four_months : List ℕ) (sixth_month : ℕ) (average : ℕ) : Prop :=
  let total_six_months := average * 6
  let sum_last_four := last_four_months.sum
  let first_month := total_six_months - (sum_last_four + sixth_month)
  first_month = 5420

theorem sales_theorem :
  sales_problem [5660, 6200, 6350, 6500] 7070 6200 := by
  sorry

end NUMINAMATH_CALUDE_sales_theorem_l3831_383107


namespace NUMINAMATH_CALUDE_number_of_cut_cubes_l3831_383171

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a set of cubes resulting from cutting a larger cube -/
structure CutCube where
  original : Cube
  pieces : List Cube
  all_same_size : Bool

/-- The volume of a cube -/
def volume (c : Cube) : ℕ := c.edge ^ 3

/-- The total volume of a list of cubes -/
def total_volume (cubes : List Cube) : ℕ :=
  cubes.map volume |>.sum

/-- Theorem: The number of smaller cubes obtained by cutting a 4cm cube is 57 -/
theorem number_of_cut_cubes : ∃ (cut : CutCube), 
  cut.original.edge = 4 ∧ 
  cut.all_same_size = false ∧
  (∀ c ∈ cut.pieces, c.edge > 0) ∧
  total_volume cut.pieces = volume cut.original ∧
  cut.pieces.length = 57 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cut_cubes_l3831_383171


namespace NUMINAMATH_CALUDE_savings_difference_l3831_383167

def initial_amount : ℝ := 10000

def option1_discounts : List ℝ := [0.20, 0.20, 0.10]
def option2_discounts : List ℝ := [0.40, 0.05, 0.05]

def apply_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ acc d => acc * (1 - d)) amount

theorem savings_difference : 
  apply_discounts initial_amount option1_discounts - 
  apply_discounts initial_amount option2_discounts = 345 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l3831_383167


namespace NUMINAMATH_CALUDE_difference_x_y_l3831_383163

theorem difference_x_y (x y : ℤ) (h1 : x + y = 20) (h2 : x = 28) : x - y = 36 := by
  sorry

end NUMINAMATH_CALUDE_difference_x_y_l3831_383163


namespace NUMINAMATH_CALUDE_inverse_proposition_correct_l3831_383185

-- Define the type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define what it means for two lines to be parallel
def parallel (l₁ l₂ : Line) : Prop := sorry

-- Define what it means for angles to be supplementary
def supplementary_angles (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 180

-- Define the original proposition
def original_proposition (l₁ l₂ : Line) (θ₁ θ₂ : ℝ) : Prop :=
  parallel l₁ l₂ → supplementary_angles θ₁ θ₂

-- Define the inverse proposition
def inverse_proposition (l₁ l₂ : Line) (θ₁ θ₂ : ℝ) : Prop :=
  supplementary_angles θ₁ θ₂ → parallel l₁ l₂

-- Theorem stating that the inverse proposition is correct
theorem inverse_proposition_correct :
  ∀ (l₁ l₂ : Line) (θ₁ θ₂ : ℝ),
    inverse_proposition l₁ l₂ θ₁ θ₂ =
    (supplementary_angles θ₁ θ₂ → parallel l₁ l₂) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_correct_l3831_383185


namespace NUMINAMATH_CALUDE_mooncake_packing_l3831_383189

theorem mooncake_packing :
  ∃ (x y : ℕ), 
    9 * x + 4 * y = 35 ∧ 
    (∀ (a b : ℕ), 9 * a + 4 * b = 35 → x + y ≤ a + b) ∧
    x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_mooncake_packing_l3831_383189


namespace NUMINAMATH_CALUDE_exactly_one_cuddly_number_l3831_383183

/-- A two-digit positive integer -/
def TwoDigitPositiveInteger (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- The tens digit of a two-digit number -/
def TensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- The units digit of a two-digit number -/
def UnitsDigit (n : ℕ) : ℕ :=
  n % 10

/-- A cuddly number is equal to the sum of its nonzero tens digit and the square of its units digit -/
def IsCuddly (n : ℕ) : Prop :=
  TwoDigitPositiveInteger n ∧ n = TensDigit n + (UnitsDigit n)^2

/-- There exists exactly one two-digit positive integer that is cuddly -/
theorem exactly_one_cuddly_number : ∃! n : ℕ, IsCuddly n :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_cuddly_number_l3831_383183


namespace NUMINAMATH_CALUDE_computational_not_basic_l3831_383164

/-- The set of basic algorithmic statements -/
def BasicAlgorithmicStatements : Set String :=
  {"assignment", "conditional", "loop", "input", "output"}

/-- Proposition: Computational statements are not basic algorithmic statements -/
theorem computational_not_basic : "computational" ∉ BasicAlgorithmicStatements := by
  sorry

end NUMINAMATH_CALUDE_computational_not_basic_l3831_383164


namespace NUMINAMATH_CALUDE_intersection_condition_l3831_383195

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

-- State the theorem
theorem intersection_condition (a : ℝ) :
  M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l3831_383195


namespace NUMINAMATH_CALUDE_total_legs_on_farm_l3831_383106

/-- The number of legs for each animal type -/
def legs_per_animal (animal : String) : ℕ :=
  match animal with
  | "chicken" => 2
  | "sheep" => 4
  | _ => 0

/-- The total number of animals on the farm -/
def total_animals : ℕ := 12

/-- The number of chickens on the farm -/
def num_chickens : ℕ := 5

/-- Theorem stating the total number of animal legs on the farm -/
theorem total_legs_on_farm : 
  (num_chickens * legs_per_animal "chicken") + 
  ((total_animals - num_chickens) * legs_per_animal "sheep") = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_on_farm_l3831_383106


namespace NUMINAMATH_CALUDE_price_reduction_sales_increase_l3831_383119

theorem price_reduction_sales_increase 
  (price_reduction : Real) 
  (revenue_increase : Real) 
  (sales_increase : Real) : 
  price_reduction = 0.35 → 
  revenue_increase = 0.17 → 
  (1 - price_reduction) * (1 + sales_increase) = 1 + revenue_increase → 
  sales_increase = 0.8 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_sales_increase_l3831_383119


namespace NUMINAMATH_CALUDE_opposite_face_of_B_l3831_383184

/-- Represents a square on the 3x3 grid --/
inductive Square
| A | B | C | D | E | F | G | H | I

/-- Represents the cube formed by folding the grid --/
structure Cube where
  faces : List Square
  open_face : Square

/-- Determines if two squares are adjacent in the 3x3 grid --/
def adjacent (s1 s2 : Square) : Prop := sorry

/-- Determines if two squares are opposite when folded into a cube --/
def opposite (c : Cube) (s1 s2 : Square) : Prop := sorry

/-- The main theorem to be proved --/
theorem opposite_face_of_B (c : Cube) : 
  c.open_face = Square.F → 
  c.faces.length = 5 → 
  opposite c Square.B Square.I := by sorry

end NUMINAMATH_CALUDE_opposite_face_of_B_l3831_383184


namespace NUMINAMATH_CALUDE_pinocchio_problem_l3831_383194

theorem pinocchio_problem (x : ℕ) : 
  x ≠ 0 ∧ x < 10 ∧ (x + x + 1) * x = 111 * x → x = 5 :=
by sorry

end NUMINAMATH_CALUDE_pinocchio_problem_l3831_383194


namespace NUMINAMATH_CALUDE_two_leq_three_l3831_383148

theorem two_leq_three : 2 ≤ 3 := by sorry

end NUMINAMATH_CALUDE_two_leq_three_l3831_383148


namespace NUMINAMATH_CALUDE_valid_configuration_exists_l3831_383182

/-- A configuration is a function from positions (1 to 6) to numbers (1 to 6) -/
def Configuration := Fin 6 → Fin 6

/-- A line is a triple of positions -/
def Line := Fin 3 → Fin 6

/-- The set of all lines in the diagram -/
def lines : Finset Line := sorry

/-- The sum of numbers on a line for a given configuration -/
def lineSum (c : Configuration) (l : Line) : Nat :=
  (l 0).val + 1 + (l 1).val + 1 + (l 2).val + 1

/-- A configuration is valid if it's a bijection and all line sums are 10 -/
def isValidConfiguration (c : Configuration) : Prop :=
  Function.Bijective c ∧ ∀ l ∈ lines, lineSum c l = 10

/-- There exists a valid configuration -/
theorem valid_configuration_exists : ∃ c : Configuration, isValidConfiguration c := by
  sorry

end NUMINAMATH_CALUDE_valid_configuration_exists_l3831_383182


namespace NUMINAMATH_CALUDE_factorization_x4_3x2_1_l3831_383149

theorem factorization_x4_3x2_1 (x : ℝ) :
  x^4 - 3*x^2 + 1 = (x^2 + x - 1) * (x^2 - x - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_x4_3x2_1_l3831_383149


namespace NUMINAMATH_CALUDE_complex_power_difference_zero_l3831_383101

theorem complex_power_difference_zero : Complex.I ^ 2 = -1 → (1 + Complex.I)^20 - (1 - Complex.I)^20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_zero_l3831_383101


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l3831_383140

/-- Two vectors are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- The problem statement -/
theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, 4) (x, 6) → x = 3 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l3831_383140


namespace NUMINAMATH_CALUDE_shell_collection_l3831_383144

theorem shell_collection (laurie_shells : ℕ) (h1 : laurie_shells = 36) :
  ∃ (ben_shells alan_shells : ℕ),
    ben_shells = laurie_shells / 3 ∧
    alan_shells = ben_shells * 4 ∧
    alan_shells = 48 := by
  sorry

end NUMINAMATH_CALUDE_shell_collection_l3831_383144


namespace NUMINAMATH_CALUDE_factor_x4_plus_16_l3831_383165

theorem factor_x4_plus_16 (x : ℝ) : x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_plus_16_l3831_383165


namespace NUMINAMATH_CALUDE_cucumber_weight_problem_l3831_383100

theorem cucumber_weight_problem (initial_water_percentage : Real)
                                (final_water_percentage : Real)
                                (final_weight : Real) :
  initial_water_percentage = 0.99 →
  final_water_percentage = 0.95 →
  final_weight = 20 →
  ∃ initial_weight : Real,
    initial_weight = 100 ∧
    (1 - initial_water_percentage) * initial_weight =
    (1 - final_water_percentage) * final_weight :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_problem_l3831_383100


namespace NUMINAMATH_CALUDE_functional_equation_properties_l3831_383153

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (f 0 = 0) ∧ (f 1 = 0) ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l3831_383153


namespace NUMINAMATH_CALUDE_algorithm_can_contain_all_structures_l3831_383142

/-- Represents the types of logical structures in algorithms -/
inductive LogicalStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents an algorithm -/
structure Algorithm where
  structures : List LogicalStructure

/-- Theorem stating that an algorithm can contain all three types of logical structures -/
theorem algorithm_can_contain_all_structures :
  ∃ (a : Algorithm), (LogicalStructure.Sequential ∈ a.structures) ∧
                     (LogicalStructure.Conditional ∈ a.structures) ∧
                     (LogicalStructure.Loop ∈ a.structures) :=
by sorry


end NUMINAMATH_CALUDE_algorithm_can_contain_all_structures_l3831_383142


namespace NUMINAMATH_CALUDE_congruent_count_l3831_383193

theorem congruent_count (n : ℕ) : 
  (Finset.filter (fun x => x % 7 = 3) (Finset.range 300)).card = 43 :=
by sorry

end NUMINAMATH_CALUDE_congruent_count_l3831_383193


namespace NUMINAMATH_CALUDE_common_tangent_lines_C₁_C₂_l3831_383126

/-- Circle C₁ with equation x² + y² - 2x = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 = 0}

/-- Circle C₂ with equation x² + (y - √3)² = 4 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - Real.sqrt 3)^2 = 4}

/-- The number of common tangent lines between two circles -/
def commonTangentLines (c1 c2 : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- Theorem stating that the number of common tangent lines between C₁ and C₂ is 2 -/
theorem common_tangent_lines_C₁_C₂ :
  commonTangentLines C₁ C₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_common_tangent_lines_C₁_C₂_l3831_383126


namespace NUMINAMATH_CALUDE_second_shot_probability_l3831_383157

/-- Probability of scoring in the next shot if the previous shot was successful -/
def p_success : ℚ := 3/4

/-- Probability of scoring in the next shot if the previous shot was missed -/
def p_miss : ℚ := 1/4

/-- Probability of scoring in the first shot -/
def p_first : ℚ := 3/4

/-- The probability of scoring in the second shot -/
def p_second : ℚ := p_first * p_success + (1 - p_first) * p_miss

theorem second_shot_probability : p_second = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_second_shot_probability_l3831_383157


namespace NUMINAMATH_CALUDE_tangent_line_range_l3831_383192

/-- Given a circle and a line, if there exists a point on the line such that
    the tangents from this point to the circle form a 60° angle,
    then the parameter k in the line equation is between -2√2 and 2√2. -/
theorem tangent_line_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x + y + k = 0 ∧ 
   ∃ (p : ℝ × ℝ), p.1 + p.2 + k = 0 ∧ 
   ∃ (a b : ℝ × ℝ), a.1^2 + a.2^2 = 1 ∧ b.1^2 + b.2^2 = 1 ∧ 
   ((p.1 - a.1)*(b.1 - a.1) + (p.2 - a.2)*(b.2 - a.2))^2 = 
   ((p.1 - a.1)^2 + (p.2 - a.2)^2) * ((b.1 - a.1)^2 + (b.2 - a.2)^2) / 4) →
  -2 * Real.sqrt 2 ≤ k ∧ k ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_range_l3831_383192


namespace NUMINAMATH_CALUDE_steve_commute_time_l3831_383175

-- Define the parameters
def distance_to_work : ℝ := 35
def speed_back : ℝ := 17.5

-- Define the theorem
theorem steve_commute_time :
  let speed_to_work : ℝ := speed_back / 2
  let time_to_work : ℝ := distance_to_work / speed_to_work
  let time_from_work : ℝ := distance_to_work / speed_back
  let total_time : ℝ := time_to_work + time_from_work
  total_time = 6 := by sorry

end NUMINAMATH_CALUDE_steve_commute_time_l3831_383175


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l3831_383118

theorem sum_of_three_squares (n : ℕ+) : ¬ ∃ x y z : ℤ, x^2 + y^2 + z^2 = 8 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l3831_383118


namespace NUMINAMATH_CALUDE_mollys_age_condition_mollys_age_proof_l3831_383109

/-- Molly's present age -/
def mollys_present_age : ℕ := 12

/-- Condition: Molly's age in 18 years will be 5 times her age 6 years ago -/
theorem mollys_age_condition : 
  mollys_present_age + 18 = 5 * (mollys_present_age - 6) :=
by sorry

/-- Proof that Molly's present age is 12 years old -/
theorem mollys_age_proof : 
  mollys_present_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_mollys_age_condition_mollys_age_proof_l3831_383109


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3831_383136

theorem line_passes_through_point :
  ∀ (m : ℝ), m * (-1) - 2 + m + 2 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3831_383136


namespace NUMINAMATH_CALUDE_additional_spend_needed_l3831_383138

-- Define the minimum spend for free delivery
def min_spend : ℝ := 35

-- Define the prices and quantities of items
def chicken_price : ℝ := 6
def chicken_quantity : ℝ := 1.5
def lettuce_price : ℝ := 3
def cherry_tomatoes_price : ℝ := 2.5
def sweet_potato_price : ℝ := 0.75
def sweet_potato_quantity : ℕ := 4
def broccoli_price : ℝ := 2
def broccoli_quantity : ℕ := 2
def brussel_sprouts_price : ℝ := 2.5

-- Calculate the total cost of items in the cart
def total_cost : ℝ :=
  chicken_price * chicken_quantity +
  lettuce_price +
  cherry_tomatoes_price +
  sweet_potato_price * sweet_potato_quantity +
  broccoli_price * broccoli_quantity +
  brussel_sprouts_price

-- Theorem: The difference between min_spend and total_cost is 11
theorem additional_spend_needed : min_spend - total_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_additional_spend_needed_l3831_383138


namespace NUMINAMATH_CALUDE_dress_shop_inventory_l3831_383143

def total_dresses (red : ℕ) (blue : ℕ) : ℕ := red + blue

theorem dress_shop_inventory : 
  let red : ℕ := 83
  let blue : ℕ := red + 34
  total_dresses red blue = 200 := by
sorry

end NUMINAMATH_CALUDE_dress_shop_inventory_l3831_383143


namespace NUMINAMATH_CALUDE_sign_selection_theorem_l3831_383174

theorem sign_selection_theorem (n : ℕ) (a : ℕ → ℕ) 
  (h_n : n ≥ 2)
  (h_a : ∀ k ∈ Finset.range n, 0 < a k ∧ a k ≤ k + 1)
  (h_even : Even (Finset.sum (Finset.range n) a)) :
  ∃ f : ℕ → Int, (∀ k, f k = 1 ∨ f k = -1) ∧ 
    Finset.sum (Finset.range n) (λ k => (f k) * (a k)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_selection_theorem_l3831_383174


namespace NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l3831_383188

/-- The area of the largest equilateral triangle inscribed in a circle of radius 10 -/
theorem largest_inscribed_equilateral_triangle_area (r : ℝ) (h : r = 10) :
  let s := r * (3 / Real.sqrt 3)
  let area := (s^2 * Real.sqrt 3) / 4
  area = 75 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l3831_383188


namespace NUMINAMATH_CALUDE_divide_by_six_multiply_by_twelve_l3831_383124

theorem divide_by_six_multiply_by_twelve (x : ℝ) : (x / 6) * 12 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_divide_by_six_multiply_by_twelve_l3831_383124


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3831_383152

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- Given points P(x,-3) and Q(4,y) that are symmetric with respect to the x-axis,
    prove that x + y = 7 -/
theorem symmetric_points_sum (x y : ℝ) :
  symmetric_x_axis (x, -3) (4, y) → x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3831_383152


namespace NUMINAMATH_CALUDE_circle_properties_l3831_383198

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 6*y - 3 = 0

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- The radius of the circle -/
def CircleRadius : ℝ := 4

/-- Theorem stating that the given equation represents a circle with the specified center and radius -/
theorem circle_properties :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = CircleRadius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3831_383198


namespace NUMINAMATH_CALUDE_calculation_proof_l3831_383121

theorem calculation_proof : |Real.sqrt 3 - 2| - 2 * Real.tan (π / 3) + (π - 2023) ^ 0 + Real.sqrt 27 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3831_383121


namespace NUMINAMATH_CALUDE_equation_solutions_l3831_383120

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 2) = 2 * (x + 2) ↔ x = -2 ∨ x = 2) ∧
  (∀ x : ℝ, 3 * x^2 - x - 1 = 0 ↔ x = (1 + Real.sqrt 13) / 6 ∨ x = (1 - Real.sqrt 13) / 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3831_383120


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3831_383160

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 5 / (3 + 4 * I)
  Complex.im z = -(4 / 5) := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3831_383160


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l3831_383169

theorem smallest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → 1013 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l3831_383169


namespace NUMINAMATH_CALUDE_circle_radius_from_sum_of_circumference_and_area_l3831_383104

theorem circle_radius_from_sum_of_circumference_and_area :
  ∀ r : ℝ, r > 0 →
    2 * Real.pi * r + Real.pi * r^2 = 530.929158456675 →
    r = Real.sqrt 170 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_sum_of_circumference_and_area_l3831_383104


namespace NUMINAMATH_CALUDE_sequence_correct_l3831_383146

def sequence_formula (n : ℕ) : ℤ := (-1)^(n+1) * 2^(n-1)

theorem sequence_correct : ∀ n : ℕ, n ≥ 1 ∧ n ≤ 6 →
  match n with
  | 1 => sequence_formula n = 1
  | 2 => sequence_formula n = -2
  | 3 => sequence_formula n = 4
  | 4 => sequence_formula n = -8
  | 5 => sequence_formula n = 16
  | 6 => sequence_formula n = -32
  | _ => True
  :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_correct_l3831_383146


namespace NUMINAMATH_CALUDE_arithmetic_sequence_vertex_l3831_383130

/-- Given that a, b, c, d form an arithmetic sequence and (a, d) is the vertex of f(x) = x^2 - 2x,
    prove that b + c = 0 -/
theorem arithmetic_sequence_vertex (a b c d : ℝ) : 
  (∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r) →  -- arithmetic sequence condition
  (a = 1 ∧ d = -1) →                              -- vertex condition
  b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_vertex_l3831_383130


namespace NUMINAMATH_CALUDE_point_b_coordinates_l3831_383102

/-- Given a line segment AB parallel to the x-axis with length 3 and point A at coordinates (-1, 2),
    the coordinates of point B are either (-4, 2) or (2, 2). -/
theorem point_b_coordinates :
  ∀ (A B : ℝ × ℝ),
  A = (-1, 2) →
  norm (B.1 - A.1) = 3 →
  B.2 = A.2 →
  B = (-4, 2) ∨ B = (2, 2) :=
by sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l3831_383102


namespace NUMINAMATH_CALUDE_arrangement_count_is_36_l3831_383105

/-- The number of ways to arrange 5 students in a row with specific conditions -/
def arrangement_count : ℕ :=
  let n : ℕ := 5  -- Total number of students
  let special_pair : ℕ := 2  -- Number of students that must be adjacent (A and B)
  let non_end_student : ℕ := 1  -- Number of students that can't be at the ends (A)
  -- The actual count calculation would go here
  36

/-- Theorem stating that the number of arrangements under given conditions is 36 -/
theorem arrangement_count_is_36 : arrangement_count = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_36_l3831_383105


namespace NUMINAMATH_CALUDE_compound_molar_mass_l3831_383112

/-- Given a compound where 6 moles weighs 612 grams, prove its molar mass is 102 grams per mole -/
theorem compound_molar_mass (mass : ℝ) (moles : ℝ) (h1 : mass = 612) (h2 : moles = 6) :
  mass / moles = 102 := by
  sorry

end NUMINAMATH_CALUDE_compound_molar_mass_l3831_383112


namespace NUMINAMATH_CALUDE_road_length_is_10km_l3831_383111

/-- Represents the road construction project -/
structure RoadProject where
  totalDays : ℕ
  initialWorkers : ℕ
  daysElapsed : ℕ
  completedLength : ℝ
  extraWorkers : ℕ

/-- Calculates the total length of the road given the project parameters -/
def calculateRoadLength (project : RoadProject) : ℝ :=
  sorry

/-- Theorem stating that the road length is 10 km given the specific project conditions -/
theorem road_length_is_10km (project : RoadProject) 
  (h1 : project.totalDays = 300)
  (h2 : project.initialWorkers = 30)
  (h3 : project.daysElapsed = 100)
  (h4 : project.completedLength = 2)
  (h5 : project.extraWorkers = 30) :
  calculateRoadLength project = 10 := by
  sorry

end NUMINAMATH_CALUDE_road_length_is_10km_l3831_383111


namespace NUMINAMATH_CALUDE_class_size_proof_l3831_383155

theorem class_size_proof :
  ∀ n : ℕ,
  20 < n ∧ n < 30 →
  ∃ x : ℕ, n = 3 * x →
  ∃ y : ℕ, n = 4 * y →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_class_size_proof_l3831_383155


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_l3831_383127

theorem largest_multiple_of_seven (n : ℤ) : n = 77 ↔ 
  (∃ k : ℤ, n = 7 * k) ∧ 
  (-n > -80) ∧
  (∀ m : ℤ, (∃ j : ℤ, m = 7 * j) → (-m > -80) → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_l3831_383127


namespace NUMINAMATH_CALUDE_self_repeating_mod_1000_numbers_l3831_383129

/-- A three-digit number that remains unchanged when raised to any natural power modulo 1000 -/
def self_repeating_mod_1000 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∀ k : ℕ, k > 0 → n^k ≡ n [MOD 1000]

/-- The only three-digit numbers that remain unchanged when raised to any natural power modulo 1000 are 625 and 376 -/
theorem self_repeating_mod_1000_numbers :
  ∀ n : ℕ, self_repeating_mod_1000 n ↔ n = 625 ∨ n = 376 := by sorry

end NUMINAMATH_CALUDE_self_repeating_mod_1000_numbers_l3831_383129


namespace NUMINAMATH_CALUDE_delta_negative_two_three_l3831_383147

-- Define the Delta operation
def Delta (a b : ℝ) : ℝ := a * b^2 + b + 1

-- Theorem statement
theorem delta_negative_two_three : Delta (-2) 3 = -14 := by
  sorry

end NUMINAMATH_CALUDE_delta_negative_two_three_l3831_383147


namespace NUMINAMATH_CALUDE_percentage_problem_l3831_383177

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 800 →
  0.4 * N = (P / 100) * 650 + 190 →
  P = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3831_383177


namespace NUMINAMATH_CALUDE_intersection_equality_implies_range_l3831_383123

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Define the range of a
def range_a : Set ℝ := {a | a = 1 ∨ a ≤ -1}

-- Theorem statement
theorem intersection_equality_implies_range (a : ℝ) : 
  A ∩ B a = B a → a ∈ range_a :=
sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_range_l3831_383123


namespace NUMINAMATH_CALUDE_plums_given_to_sam_l3831_383156

/-- Given Melanie's plum picking and sharing scenario, prove the number of plums given to Sam. -/
theorem plums_given_to_sam 
  (original_plums : ℕ) 
  (plums_left : ℕ) 
  (h1 : original_plums = 7)
  (h2 : plums_left = 4)
  : original_plums - plums_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_plums_given_to_sam_l3831_383156


namespace NUMINAMATH_CALUDE_value_of_B_l3831_383179

/-- Given the value assignments for letters and words, prove the value of B --/
theorem value_of_B (T L A B : ℤ) : 
  T = 15 →
  B + A + L + L = 40 →
  L + A + B = 25 →
  A + L + L = 30 →
  B = 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_B_l3831_383179


namespace NUMINAMATH_CALUDE_conditional_prob_A_given_B_l3831_383199

-- Define the sample space
def S : Set (Fin 3 → Fin 2) := {f | ∀ i, f i < 2}

-- Define event A: "the second digit is '0'"
def A : Set (Fin 3 → Fin 2) := {f ∈ S | f 1 = 0}

-- Define event B: "the first digit is '0'"
def B : Set (Fin 3 → Fin 2) := {f ∈ S | f 0 = 0}

-- Define the probability measure
variable (P : Set (Fin 3 → Fin 2) → ℝ)

-- Axioms for probability measure
axiom prob_nonneg : ∀ E ⊆ S, P E ≥ 0
axiom prob_total : P S = 1
axiom prob_countable_additivity :
  ∀ (E : ℕ → Set (Fin 3 → Fin 2)), (∀ n, E n ⊆ S) → (∀ m n, m ≠ n → E m ∩ E n = ∅) →
    P (⋃ n, E n) = ∑' n, P (E n)

-- Theorem: The conditional probability P(A|B) = 1/2
theorem conditional_prob_A_given_B :
  P (A ∩ B) / P B = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_conditional_prob_A_given_B_l3831_383199


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3831_383154

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x > 2, x^3 - 8 > 0) ↔ (∃ x > 2, x^3 - 8 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3831_383154


namespace NUMINAMATH_CALUDE_porter_monthly_earnings_l3831_383166

/-- Porter's daily wage in dollars -/
def daily_wage : ℕ := 8

/-- Number of regular working days per week -/
def regular_days : ℕ := 5

/-- Overtime bonus rate as a percentage -/
def overtime_bonus_rate : ℕ := 50

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Calculate Porter's monthly earnings with overtime every week -/
def monthly_earnings_with_overtime : ℕ :=
  let regular_weekly_earnings := daily_wage * regular_days
  let overtime_daily_earnings := daily_wage + (daily_wage * overtime_bonus_rate / 100)
  let weekly_earnings_with_overtime := regular_weekly_earnings + overtime_daily_earnings
  weekly_earnings_with_overtime * weeks_per_month

theorem porter_monthly_earnings :
  monthly_earnings_with_overtime = 208 := by
  sorry

end NUMINAMATH_CALUDE_porter_monthly_earnings_l3831_383166


namespace NUMINAMATH_CALUDE_sine_sum_problem_l3831_383191

theorem sine_sum_problem (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.tan (α - π/4) = 1/3) :
  Real.sin (π/4 + α) = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_problem_l3831_383191


namespace NUMINAMATH_CALUDE_equation_implies_equilateral_l3831_383170

/-- A triangle with sides a, b, c and opposite angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The radius of the circumscribed circle of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The equation that the triangle satisfies -/
def satisfies_equation (t : Triangle) : Prop :=
  (t.a * Real.cos t.α + t.b * Real.cos t.β + t.c * Real.cos t.γ) /
  (t.a * Real.sin t.β + t.b * Real.sin t.γ + t.c * Real.sin t.α) =
  perimeter t / (9 * circumradius t)

/-- A triangle is equilateral if all its sides are equal -/
def is_equilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

/-- The main theorem -/
theorem equation_implies_equilateral (t : Triangle) :
  satisfies_equation t → is_equilateral t :=
by sorry

end NUMINAMATH_CALUDE_equation_implies_equilateral_l3831_383170


namespace NUMINAMATH_CALUDE_max_value_of_g_l3831_383178

def g (x : ℝ) : ℝ := 5 * x - x^5

theorem max_value_of_g :
  ∃ (max : ℝ), max = 4 ∧
  ∀ x : ℝ, 0 ≤ x → x ≤ Real.sqrt 5 → g x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3831_383178


namespace NUMINAMATH_CALUDE_g_composition_value_l3831_383172

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^2 - 6

-- State the theorem
theorem g_composition_value : g (g 2) = 394 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_value_l3831_383172


namespace NUMINAMATH_CALUDE_garden_area_l3831_383116

/-- A rectangular garden with specific walking measurements -/
structure Garden where
  length : ℝ
  width : ℝ
  length_walk : length * 30 = 1200
  perimeter_walk : (2 * length + 2 * width) * 12 = 1200

/-- The area of the garden is 400 square meters -/
theorem garden_area (g : Garden) : g.length * g.width = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l3831_383116


namespace NUMINAMATH_CALUDE_f_two_zero_l3831_383176

/-- A mapping f that takes a point (x,y) to (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that f(2,0) = (2,2) -/
theorem f_two_zero : f (2, 0) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_f_two_zero_l3831_383176


namespace NUMINAMATH_CALUDE_flower_problem_l3831_383110

theorem flower_problem (total : ℕ) (roses_fraction : ℚ) (carnations : ℕ) (tulips : ℕ) :
  total = 40 →
  roses_fraction = 2 / 5 →
  carnations = 14 →
  tulips = total - (roses_fraction * total + carnations) →
  tulips = 10 := by
sorry

end NUMINAMATH_CALUDE_flower_problem_l3831_383110


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l3831_383128

theorem consecutive_negative_integers_product_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = -105 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l3831_383128


namespace NUMINAMATH_CALUDE_andrei_valentin_distance_at_finish_l3831_383133

/-- Represents the race scenario with three runners -/
structure RaceScenario where
  race_distance : ℝ
  andrei_boris_gap : ℝ
  boris_valentin_gap : ℝ

/-- Calculates the distance between Andrei and Valentin at Andrei's finish -/
def distance_andrei_valentin (scenario : RaceScenario) : ℝ :=
  scenario.race_distance - (scenario.race_distance - scenario.andrei_boris_gap - scenario.boris_valentin_gap)

/-- Theorem stating the distance between Andrei and Valentin when Andrei finishes -/
theorem andrei_valentin_distance_at_finish (scenario : RaceScenario) 
  (h1 : scenario.race_distance = 1000)
  (h2 : scenario.andrei_boris_gap = 100)
  (h3 : scenario.boris_valentin_gap = 50) :
  distance_andrei_valentin scenario = 145 := by
  sorry

#eval distance_andrei_valentin ⟨1000, 100, 50⟩

end NUMINAMATH_CALUDE_andrei_valentin_distance_at_finish_l3831_383133


namespace NUMINAMATH_CALUDE_mean_home_runs_l3831_383135

theorem mean_home_runs : 
  let players_with_5 := 3
  let players_with_6 := 4
  let players_with_8 := 2
  let players_with_9 := 1
  let players_with_11 := 1
  let total_players := players_with_5 + players_with_6 + players_with_8 + players_with_9 + players_with_11
  let total_home_runs := 5 * players_with_5 + 6 * players_with_6 + 8 * players_with_8 + 9 * players_with_9 + 11 * players_with_11
  (total_home_runs : ℚ) / total_players = 75 / 11 := by
sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3831_383135


namespace NUMINAMATH_CALUDE_digit_for_divisibility_by_6_l3831_383145

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem digit_for_divisibility_by_6 :
  ∃ B : ℕ, B < 10 ∧ is_divisible_by_6 (5170 + B) ∧ (B = 2 ∨ B = 8) :=
by sorry

end NUMINAMATH_CALUDE_digit_for_divisibility_by_6_l3831_383145


namespace NUMINAMATH_CALUDE_venny_car_cost_l3831_383161

def original_price : ℝ := 37500

def discount_percentage : ℝ := 40

theorem venny_car_cost : ℝ := by
  -- Define the amount Venny spent as 40% of the original price
  let amount_spent := (discount_percentage / 100) * original_price
  
  -- Prove that this amount is equal to $15,000
  sorry

end NUMINAMATH_CALUDE_venny_car_cost_l3831_383161


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3831_383181

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 26) 
  (h2 : 4 * (a + b + c) = 28) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3831_383181


namespace NUMINAMATH_CALUDE_tv_price_difference_l3831_383141

theorem tv_price_difference (budget : ℝ) (initial_discount : ℝ) (percentage_discount : ℝ) : 
  budget = 1000 →
  initial_discount = 100 →
  percentage_discount = 0.2 →
  budget - (budget - initial_discount) * (1 - percentage_discount) = 280 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_difference_l3831_383141
