import Mathlib

namespace NUMINAMATH_CALUDE_student_sums_problem_l3046_304640

theorem student_sums_problem (total : ℕ) (correct : ℕ) (incorrect : ℕ) 
  (h1 : total = 96)
  (h2 : incorrect = 3 * correct)
  (h3 : total = correct + incorrect) :
  correct = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_sums_problem_l3046_304640


namespace NUMINAMATH_CALUDE_problem_statement_l3046_304638

theorem problem_statement (m n N k : ℕ) :
  (n^2 + 1)^(2^k) * (44*n^3 + 11*n^2 + 10*n + 2) = N^m →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3046_304638


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l3046_304622

theorem surrounding_circles_radius (R : ℝ) (n : ℕ) (r : ℝ) :
  R = 2 ∧ n = 6 ∧ r > 0 →
  (R + r)^2 = (2 * r)^2 + (2 * r)^2 - 2 * (2 * r) * (2 * r) * Real.cos (2 * Real.pi / n) →
  r = (2 + 2 * Real.sqrt 11) / 11 := by
  sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l3046_304622


namespace NUMINAMATH_CALUDE_parking_space_savings_l3046_304627

/-- Proves the yearly savings when renting a parking space monthly instead of weekly -/
theorem parking_space_savings (weekly_rate : ℕ) (monthly_rate : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) :
  weekly_rate = 10 →
  monthly_rate = 42 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  weekly_rate * weeks_per_year - monthly_rate * months_per_year = 16 := by
sorry

end NUMINAMATH_CALUDE_parking_space_savings_l3046_304627


namespace NUMINAMATH_CALUDE_train_length_l3046_304625

theorem train_length (platform1_length platform2_length : ℝ)
                     (time1 time2 : ℝ)
                     (h1 : platform1_length = 110)
                     (h2 : platform2_length = 250)
                     (h3 : time1 = 15)
                     (h4 : time2 = 20)
                     (h5 : time1 > 0)
                     (h6 : time2 > 0) :
  let train_length := (platform2_length * time1 - platform1_length * time2) / (time2 - time1)
  train_length = 310 := by
sorry

end NUMINAMATH_CALUDE_train_length_l3046_304625


namespace NUMINAMATH_CALUDE_exists_20_digit_singular_l3046_304663

/-- A number is singular if it's a 2n-digit perfect square, and both its first n digits
    and last n digits are also perfect squares. --/
def is_singular (x : ℕ) : Prop :=
  ∃ (n : ℕ), 
    (x ≥ 10^(2*n - 1)) ∧ 
    (x < 10^(2*n)) ∧
    (∃ (y : ℕ), x = y^2) ∧
    (∃ (a b : ℕ), 
      x = a * 10^n + b ∧
      (∃ (c : ℕ), a = c^2) ∧
      (∃ (d : ℕ), b = d^2) ∧
      (a ≥ 10^(n-1)) ∧
      (b > 0))

/-- There exists a 20-digit singular number. --/
theorem exists_20_digit_singular : ∃ (x : ℕ), is_singular x ∧ (x ≥ 10^19) ∧ (x < 10^20) :=
sorry

end NUMINAMATH_CALUDE_exists_20_digit_singular_l3046_304663


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3046_304661

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 4) :
  ∃ d : ℝ, d = -1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3046_304661


namespace NUMINAMATH_CALUDE_megan_current_seashells_l3046_304610

-- Define the number of seashells Megan currently has
def current_seashells : ℕ := 19

-- Define the number of additional seashells Megan needs
def additional_seashells : ℕ := 6

-- Define the total number of seashells Megan will have after adding more
def total_seashells : ℕ := 25

-- Theorem stating that Megan currently has 19 seashells
theorem megan_current_seashells : 
  current_seashells = total_seashells - additional_seashells :=
by
  sorry

end NUMINAMATH_CALUDE_megan_current_seashells_l3046_304610


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3046_304656

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x * Real.sin x < 2^x - 1) ↔
  (∃ x : ℝ, x > 0 ∧ x * Real.sin x ≥ 2^x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3046_304656


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3046_304606

theorem max_value_of_expression (x : ℝ) : 
  x^4 / (x^8 + 4*x^6 + 2*x^4 + 8*x^2 + 16) ≤ 1/31 ∧ 
  ∃ y : ℝ, y^4 / (y^8 + 4*y^6 + 2*y^4 + 8*y^2 + 16) = 1/31 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3046_304606


namespace NUMINAMATH_CALUDE_tv_cost_l3046_304675

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 840 → 
  furniture_fraction = 3/4 → 
  tv_cost = savings * (1 - furniture_fraction) → 
  tv_cost = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_cost_l3046_304675


namespace NUMINAMATH_CALUDE_total_monthly_earnings_l3046_304693

/-- Represents an apartment floor --/
structure Floor :=
  (rooms : ℕ)
  (rent : ℝ)
  (occupancy : ℝ)

/-- Calculates the monthly earnings for a floor --/
def floorEarnings (f : Floor) : ℝ :=
  f.rooms * f.rent * f.occupancy

/-- Represents an apartment building --/
structure Building :=
  (floors : List Floor)

/-- Calculates the total monthly earnings for a building --/
def buildingEarnings (b : Building) : ℝ :=
  (b.floors.map floorEarnings).sum

/-- The first building --/
def building1 : Building :=
  { floors := [
    { rooms := 5, rent := 15, occupancy := 0.8 },
    { rooms := 6, rent := 25, occupancy := 0.75 },
    { rooms := 9, rent := 30, occupancy := 0.5 },
    { rooms := 4, rent := 60, occupancy := 0.85 }
  ] }

/-- The second building --/
def building2 : Building :=
  { floors := [
    { rooms := 7, rent := 20, occupancy := 0.9 },
    { rooms := 8, rent := 42.5, occupancy := 0.7 }, -- Average rent for the second floor
    { rooms := 6, rent := 60, occupancy := 0.6 }
  ] }

/-- The main theorem --/
theorem total_monthly_earnings :
  buildingEarnings building1 + buildingEarnings building2 = 1091.5 := by
  sorry

end NUMINAMATH_CALUDE_total_monthly_earnings_l3046_304693


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l3046_304649

theorem set_inclusion_implies_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {0, 1} → 
  B = {-1, 0, a+3} → 
  A ⊆ B → 
  a = -2 := by sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l3046_304649


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3046_304605

/-- Given a rectangle with perimeter 40, its maximum area is 100 -/
theorem rectangle_max_area :
  ∀ w l : ℝ,
  w > 0 → l > 0 →
  2 * (w + l) = 40 →
  ∀ w' l' : ℝ,
  w' > 0 → l' > 0 →
  2 * (w' + l') = 40 →
  w * l ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3046_304605


namespace NUMINAMATH_CALUDE_solve_equation_l3046_304607

-- Define the function F
def F (a b c d : ℕ) : ℕ := a^b + c * d

-- State the theorem
theorem solve_equation : ∃ x : ℕ, F 2 x 4 11 = 300 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3046_304607


namespace NUMINAMATH_CALUDE_largest_valid_number_l3046_304634

def is_valid_number (n : ℕ) : Prop :=
  (n > 0) ∧
  (∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ ≠ d₂ → d₁ ≠ 0 ∧ d₂ ≠ 0) ∧
  (n.digits 10).sum = 17

theorem largest_valid_number :
  ∀ m : ℕ, is_valid_number m → m ≤ 98 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3046_304634


namespace NUMINAMATH_CALUDE_product_of_red_is_red_l3046_304644

-- Define the color type
inductive Color : Type
  | Red : Color
  | Blue : Color

-- Define the coloring function
def coloring : ℕ+ → Color := sorry

-- Define the conditions
axiom all_colored : ∀ n : ℕ+, (coloring n = Color.Red) ∨ (coloring n = Color.Blue)
axiom sum_different_colors : ∀ m n : ℕ+, coloring m ≠ coloring n → coloring (m + n) = Color.Blue
axiom product_different_colors : ∀ m n : ℕ+, coloring m ≠ coloring n → coloring (m * n) = Color.Red

-- State the theorem
theorem product_of_red_is_red :
  ∀ m n : ℕ+, coloring m = Color.Red → coloring n = Color.Red → coloring (m * n) = Color.Red :=
sorry

end NUMINAMATH_CALUDE_product_of_red_is_red_l3046_304644


namespace NUMINAMATH_CALUDE_ABABABA_probability_l3046_304630

/-- The number of tiles marked A -/
def num_A : ℕ := 4

/-- The number of tiles marked B -/
def num_B : ℕ := 3

/-- The total number of tiles -/
def total_tiles : ℕ := num_A + num_B

/-- The number of favorable arrangements (ABABABA) -/
def favorable_arrangements : ℕ := 1

/-- The probability of the specific arrangement ABABABA -/
def probability_ABABABA : ℚ := favorable_arrangements / (total_tiles.choose num_A)

theorem ABABABA_probability : probability_ABABABA = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ABABABA_probability_l3046_304630


namespace NUMINAMATH_CALUDE_rhombus_area_70_l3046_304642

/-- The area of a rhombus with given vertices -/
theorem rhombus_area_70 : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (10, 0), (0, -3.5), (-10, 0)]
  let diag1 : ℝ := |3.5 - (-3.5)|
  let diag2 : ℝ := |10 - (-10)|
  (diag1 * diag2) / 2 = 70 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_70_l3046_304642


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3046_304667

/-- The x-intercept of a line passing through two given points is -3/2 -/
theorem x_intercept_of_line (p1 p2 : ℝ × ℝ) : 
  p1 = (-1, 1) → p2 = (0, 3) → 
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = m * x + b ↔ (x, y) ∈ ({p1, p2} : Set (ℝ × ℝ))) → 
  (0 = m * (-3/2) + b) := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3046_304667


namespace NUMINAMATH_CALUDE_smallest_odd_with_five_prime_factors_l3046_304670

def is_prime (n : ℕ) : Prop := sorry

def has_exactly_n_prime_factors (m n : ℕ) : Prop := sorry

theorem smallest_odd_with_five_prime_factors :
  ∀ n : ℕ, n % 2 = 1 → has_exactly_n_prime_factors n 5 → n ≥ 15015 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_with_five_prime_factors_l3046_304670


namespace NUMINAMATH_CALUDE_four_digit_sum_4360_l3046_304662

def is_valid_insertion (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ 
    ((a = 2 ∧ b = 1 ∧ d = 5) ∨ (a = 2 ∧ c = 1 ∧ d = 5))

theorem four_digit_sum_4360 :
  ∀ (n₁ n₂ : ℕ), is_valid_insertion n₁ → is_valid_insertion n₂ → n₁ + n₂ = 4360 →
    ((n₁ = 2195 ∧ n₂ = 2165) ∨ (n₁ = 2185 ∧ n₂ = 2175) ∨ (n₁ = 2215 ∧ n₂ = 2145)) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_4360_l3046_304662


namespace NUMINAMATH_CALUDE_grey_perimeter_fraction_five_strips_l3046_304654

/-- A square divided into strips -/
structure StrippedSquare where
  num_strips : ℕ
  num_grey_strips : ℕ
  h_grey_strips : num_grey_strips ≤ num_strips

/-- The fraction of the perimeter that is grey -/
def grey_perimeter_fraction (s : StrippedSquare) : ℚ :=
  s.num_grey_strips / s.num_strips

/-- Theorem: In a square divided into 5 strips with 2 grey strips, 
    the fraction of the perimeter that is grey is 2/5 -/
theorem grey_perimeter_fraction_five_strips 
  (s : StrippedSquare) 
  (h_five_strips : s.num_strips = 5)
  (h_two_grey : s.num_grey_strips = 2) : 
  grey_perimeter_fraction s = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_grey_perimeter_fraction_five_strips_l3046_304654


namespace NUMINAMATH_CALUDE_grid_toothpicks_l3046_304646

/-- Calculates the total number of toothpicks in a rectangular grid. -/
def total_toothpicks (height width : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  (horizontal_lines * width) + (vertical_lines * height)

/-- Theorem stating that a 30x15 rectangular grid of toothpicks uses 945 toothpicks. -/
theorem grid_toothpicks : total_toothpicks 30 15 = 945 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_l3046_304646


namespace NUMINAMATH_CALUDE_unique_solution_for_all_z_l3046_304617

theorem unique_solution_for_all_z (x : ℚ) : 
  (∀ z : ℚ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_all_z_l3046_304617


namespace NUMINAMATH_CALUDE_impossibleArrangement_l3046_304674

/-- Represents a 3x3 grid filled with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- The given fixed grid arrangement -/
def fixedGrid : Grid :=
  fun i j => Fin.mk ((i.val * 3 + j.val) % 9 + 1) (by sorry)

/-- Two positions are adjacent if they share a side -/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Two numbers are neighbors in a grid if they are in adjacent positions -/
def neighbors (g : Grid) (x y : Fin 9) : Prop :=
  ∃ (p q : Fin 3 × Fin 3), g p.1 p.2 = x ∧ g q.1 q.2 = y ∧ adjacent p q

theorem impossibleArrangement :
  ¬∃ (g₂ g₃ : Grid),
    (∀ x y : Fin 9, (neighbors fixedGrid x y ∨ neighbors g₂ x y ∨ neighbors g₃ x y) →
      ¬(neighbors fixedGrid x y ∧ neighbors g₂ x y) ∧
      ¬(neighbors fixedGrid x y ∧ neighbors g₃ x y) ∧
      ¬(neighbors g₂ x y ∧ neighbors g₃ x y)) :=
by sorry

end NUMINAMATH_CALUDE_impossibleArrangement_l3046_304674


namespace NUMINAMATH_CALUDE_first_hour_speed_l3046_304648

theorem first_hour_speed 
  (total_time : ℝ) 
  (average_speed : ℝ) 
  (second_part_time : ℝ) 
  (second_part_speed : ℝ) 
  (h1 : total_time = 4) 
  (h2 : average_speed = 55) 
  (h3 : second_part_time = 3) 
  (h4 : second_part_speed = 60) : 
  ∃ (first_hour_speed : ℝ), first_hour_speed = 40 ∧ 
    average_speed * total_time = first_hour_speed * (total_time - second_part_time) + 
      second_part_speed * second_part_time :=
by sorry

end NUMINAMATH_CALUDE_first_hour_speed_l3046_304648


namespace NUMINAMATH_CALUDE_max_isosceles_triangles_2017gon_l3046_304690

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- Represents a division of a polygon into triangular regions using diagonals -/
structure PolygonDivision (n : ℕ) where
  polygon : RegularPolygon n
  num_triangles : ℕ
  num_diagonals : ℕ
  diagonals_dont_intersect : Bool

/-- Represents the number of isosceles triangles in a polygon division -/
def num_isosceles_triangles (d : PolygonDivision n) : ℕ :=
  sorry

/-- Theorem: The maximum number of isosceles triangles in a specific polygon division -/
theorem max_isosceles_triangles_2017gon :
  ∀ (d : PolygonDivision 2017),
    d.num_triangles = 2015 →
    d.num_diagonals = 2014 →
    d.diagonals_dont_intersect = true →
    num_isosceles_triangles d ≤ 2010 :=
  sorry

end NUMINAMATH_CALUDE_max_isosceles_triangles_2017gon_l3046_304690


namespace NUMINAMATH_CALUDE_exists_positive_x_hash_equals_63_l3046_304637

/-- Definition of the # operation -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating the existence of a positive real number x such that 3 # x = 63 -/
theorem exists_positive_x_hash_equals_63 : ∃ x : ℝ, x > 0 ∧ hash 3 x = 63 := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_x_hash_equals_63_l3046_304637


namespace NUMINAMATH_CALUDE_bottom_level_legos_l3046_304639

/-- Represents a 3-level pyramid with decreasing lego sides -/
structure LegoPyramid where
  bottom : ℕ  -- Number of legos per side on the bottom level
  mid : ℕ     -- Number of legos per side on the middle level
  top : ℕ     -- Number of legos per side on the top level

/-- Calculates the total number of legos in the pyramid -/
def totalLegos (p : LegoPyramid) : ℕ :=
  p.bottom ^ 2 + p.mid ^ 2 + p.top ^ 2

/-- Theorem: The bottom level of a 3-level pyramid with 110 total legos has 7 legos per side -/
theorem bottom_level_legos :
  ∃ (p : LegoPyramid),
    p.mid = p.bottom - 1 ∧
    p.top = p.bottom - 2 ∧
    totalLegos p = 110 ∧
    p.bottom = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_bottom_level_legos_l3046_304639


namespace NUMINAMATH_CALUDE_toby_friend_wins_l3046_304698

/-- Juggling contest between Toby and his friend -/
theorem toby_friend_wins (toby_rotations : ℕ → ℕ) (friend_apples : ℕ) (friend_rotations : ℕ → ℕ) : 
  friend_apples = 4 ∧ 
  (∀ n, friend_rotations n = 101) ∧ 
  (∀ n, toby_rotations n = 80) → 
  friend_apples * friend_rotations 0 = 404 ∧ 
  ∀ k, k * toby_rotations 0 ≤ friend_apples * friend_rotations 0 :=
by sorry

end NUMINAMATH_CALUDE_toby_friend_wins_l3046_304698


namespace NUMINAMATH_CALUDE_john_finish_distance_ahead_of_steve_john_finishes_two_meters_ahead_l3046_304692

/-- Calculates how many meters ahead John finishes compared to Steve in a race --/
theorem john_finish_distance_ahead_of_steve 
  (initial_distance_behind : ℝ) 
  (john_speed : ℝ) 
  (steve_speed : ℝ) 
  (final_push_time : ℝ) : ℝ :=
  let john_distance := john_speed * final_push_time
  let steve_distance := steve_speed * final_push_time
  let steve_effective_distance := steve_distance + initial_distance_behind
  john_distance - steve_effective_distance

/-- Proves that John finishes 2 meters ahead of Steve given the race conditions --/
theorem john_finishes_two_meters_ahead : 
  john_finish_distance_ahead_of_steve 15 4.2 3.8 42.5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_finish_distance_ahead_of_steve_john_finishes_two_meters_ahead_l3046_304692


namespace NUMINAMATH_CALUDE_harry_anna_pencil_ratio_l3046_304657

/-- Proves that the ratio of Harry's initial pencils to Anna's pencils is 2:1 --/
theorem harry_anna_pencil_ratio :
  ∀ (anna_pencils : ℕ) (harry_initial : ℕ) (harry_lost : ℕ) (harry_left : ℕ),
    anna_pencils = 50 →
    harry_initial = anna_pencils * harry_left / (anna_pencils - harry_lost) →
    harry_lost = 19 →
    harry_left = 81 →
    harry_initial / anna_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_harry_anna_pencil_ratio_l3046_304657


namespace NUMINAMATH_CALUDE_extra_discount_is_four_percent_l3046_304619

/-- Calculates the percentage of extra discount given initial price, first discount, and final price -/
def extra_discount_percentage (initial_price first_discount final_price : ℚ) : ℚ :=
  let price_after_first_discount := initial_price - first_discount
  let extra_discount_amount := price_after_first_discount - final_price
  (extra_discount_amount / price_after_first_discount) * 100

/-- Theorem stating that the extra discount percentage is 4% given the problem conditions -/
theorem extra_discount_is_four_percent :
  extra_discount_percentage 50 2.08 46 = 4 := by
  sorry

end NUMINAMATH_CALUDE_extra_discount_is_four_percent_l3046_304619


namespace NUMINAMATH_CALUDE_fox_jeans_price_l3046_304682

/-- The regular price of Fox jeans -/
def F : ℝ := 15

/-- The regular price of Pony jeans -/
def pony_price : ℝ := 18

/-- The discount rate on Pony jeans -/
def pony_discount : ℝ := 0.1

/-- The sum of the two discount rates -/
def total_discount : ℝ := 0.22

/-- The total savings when purchasing 5 pairs of jeans -/
def total_savings : ℝ := 9

/-- The number of Fox jeans purchased -/
def fox_count : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_count : ℕ := 2

theorem fox_jeans_price :
  F = 15 ∧
  pony_price = 18 ∧
  pony_discount = 0.1 ∧
  total_discount = 0.22 ∧
  total_savings = 9 ∧
  fox_count = 3 ∧
  pony_count = 2 →
  F = 15 := by sorry

end NUMINAMATH_CALUDE_fox_jeans_price_l3046_304682


namespace NUMINAMATH_CALUDE_arc_ray_configuration_theorem_l3046_304696

/-- Given a geometric configuration with circular arcs and rays, 
    we define constants u_ij and v_ij. This theorem proves a relationship between these constants. -/
theorem arc_ray_configuration_theorem 
  (u12 v12 u23 v23 : ℝ) 
  (h1 : u12 = v12) 
  (h2 : u12 = v23) 
  (h3 : u23 = v12) : 
  u23 = v23 := by sorry

end NUMINAMATH_CALUDE_arc_ray_configuration_theorem_l3046_304696


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3046_304624

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + 2 * b₀ = 1 ∧ 1 / a₀ + 1 / b₀ = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3046_304624


namespace NUMINAMATH_CALUDE_P_intersect_M_l3046_304685

def P : Set ℝ := {y | ∃ x, y = x^2 - 3*x + 1}

def M : Set ℝ := {y | ∃ x ∈ Set.Icc (-2) 5, y = Real.sqrt (x + 2) * Real.sqrt (5 - x)}

theorem P_intersect_M : P ∩ M = Set.Icc (-5/4) 5 := by sorry

end NUMINAMATH_CALUDE_P_intersect_M_l3046_304685


namespace NUMINAMATH_CALUDE_compute_expression_l3046_304671

theorem compute_expression : 16 * (125 / 2 + 25 / 4 + 9 / 16 + 1) = 1125 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3046_304671


namespace NUMINAMATH_CALUDE_bug_path_tiles_l3046_304611

def floor_width : ℕ := 10
def floor_length : ℕ := 17

theorem bug_path_tiles : 
  floor_width + floor_length - Nat.gcd floor_width floor_length = 26 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l3046_304611


namespace NUMINAMATH_CALUDE_remainder_is_neg_one_l3046_304608

/-- The polynomial x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 -/
def f (x : ℂ) : ℂ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

/-- The polynomial x^100 + x^75 + x^50 + x^25 + 1 -/
def g (x : ℂ) : ℂ := x^100 + x^75 + x^50 + x^25 + 1

/-- The theorem stating that the remainder of g(x) divided by f(x) is -1 -/
theorem remainder_is_neg_one : ∃ q : ℂ → ℂ, ∀ x : ℂ, g x = f x * q x + (-1) := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_neg_one_l3046_304608


namespace NUMINAMATH_CALUDE_syrup_cost_is_fifty_cents_l3046_304609

/-- The cost of a Build Your Own Hot Brownie dessert --/
def dessert_cost (brownie_cost ice_cream_cost nuts_cost syrup_cost : ℚ) : ℚ :=
  brownie_cost + 2 * ice_cream_cost + nuts_cost + 2 * syrup_cost

/-- Theorem: The syrup cost is $0.50 per serving --/
theorem syrup_cost_is_fifty_cents :
  ∃ (syrup_cost : ℚ),
    dessert_cost 2.5 1 1.5 syrup_cost = 7 ∧
    syrup_cost = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_syrup_cost_is_fifty_cents_l3046_304609


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3046_304665

/-- Represents the supermarket's mineral water sales scenario -/
structure MineralWaterSales where
  costPrice : ℝ
  initialSellingPrice : ℝ
  initialMonthlySales : ℝ
  salesIncrease : ℝ
  targetMonthlyProfit : ℝ

/-- Calculates the monthly profit given a price reduction -/
def monthlyProfit (s : MineralWaterSales) (priceReduction : ℝ) : ℝ :=
  let newPrice := s.initialSellingPrice - priceReduction
  let newSales := s.initialMonthlySales + s.salesIncrease * priceReduction
  (newPrice - s.costPrice) * newSales

/-- Theorem stating that a 7 yuan price reduction achieves the target monthly profit -/
theorem price_reduction_achieves_target_profit (s : MineralWaterSales) 
    (h1 : s.costPrice = 24)
    (h2 : s.initialSellingPrice = 36)
    (h3 : s.initialMonthlySales = 60)
    (h4 : s.salesIncrease = 10)
    (h5 : s.targetMonthlyProfit = 650) :
    monthlyProfit s 7 = s.targetMonthlyProfit := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3046_304665


namespace NUMINAMATH_CALUDE_suitcase_lock_settings_l3046_304673

/-- The number of digits on each dial -/
def num_digits : ℕ := 10

/-- The number of dials on the lock -/
def num_dials : ℕ := 4

/-- The number of choices for each dial after the first -/
def choices_after_first : ℕ := num_digits - 1

/-- The total number of possible settings for the lock -/
def total_settings : ℕ := num_digits * choices_after_first^(num_dials - 1)

/-- Theorem stating that the total number of settings is 7290 -/
theorem suitcase_lock_settings : total_settings = 7290 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_lock_settings_l3046_304673


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_product_l3046_304626

theorem arithmetic_sequence_sum_product (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n / 2) * (a 1 + a n)) →  -- Definition of S_n
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- Arithmetic sequence property
  a 7 < 0 →
  a 8 > 0 →
  a 8 > |a 7| →
  S 13 * S 14 < 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_product_l3046_304626


namespace NUMINAMATH_CALUDE_school_referendum_l3046_304613

theorem school_referendum (total_students : ℕ) (first_issue : ℕ) (second_issue : ℕ) (against_both : ℕ)
  (h1 : total_students = 150)
  (h2 : first_issue = 110)
  (h3 : second_issue = 95)
  (h4 : against_both = 15) :
  first_issue + second_issue - (total_students - against_both) = 70 := by
  sorry

end NUMINAMATH_CALUDE_school_referendum_l3046_304613


namespace NUMINAMATH_CALUDE_toms_age_ratio_l3046_304683

theorem toms_age_ratio (T N : ℝ) (h1 : T > 0) (h2 : N > 0) 
  (h3 : T - N = 2 * (T - 3 * N)) : T / N = 5 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l3046_304683


namespace NUMINAMATH_CALUDE_buses_met_count_l3046_304612

/-- Represents the schedule of buses between Moscow and Voronezh -/
structure BusSchedule where
  moscow_departure_minute : Nat
  voronezh_departure_minute : Nat
  travel_time_hours : Nat

/-- Calculates the number of buses from Voronezh that a bus from Moscow will meet -/
def buses_met (schedule : BusSchedule) : Nat :=
  2 * schedule.travel_time_hours

/-- Theorem stating that a bus from Moscow will meet 16 buses from Voronezh -/
theorem buses_met_count (schedule : BusSchedule) 
  (h1 : schedule.moscow_departure_minute = 0)
  (h2 : schedule.voronezh_departure_minute = 30)
  (h3 : schedule.travel_time_hours = 8) : 
  buses_met schedule = 16 := by
  sorry

#eval buses_met { moscow_departure_minute := 0, voronezh_departure_minute := 30, travel_time_hours := 8 }

end NUMINAMATH_CALUDE_buses_met_count_l3046_304612


namespace NUMINAMATH_CALUDE_sara_frosting_cans_l3046_304664

/-- Represents the data for each day's baking and frosting --/
structure DayData where
  baked : ℕ
  eaten : ℕ
  frostingPerCake : ℕ

/-- Calculates the total frosting cans needed for the remaining cakes --/
def totalFrostingCans (data : List DayData) : ℕ :=
  data.foldl (fun acc day => acc + (day.baked - day.eaten) * day.frostingPerCake) 0

/-- The main theorem stating the total number of frosting cans needed --/
theorem sara_frosting_cans : 
  let bakingData : List DayData := [
    ⟨7, 4, 2⟩,
    ⟨12, 6, 3⟩,
    ⟨8, 3, 4⟩,
    ⟨10, 2, 3⟩,
    ⟨15, 3, 2⟩
  ]
  totalFrostingCans bakingData = 92 := by
  sorry

#eval totalFrostingCans [
  ⟨7, 4, 2⟩,
  ⟨12, 6, 3⟩,
  ⟨8, 3, 4⟩,
  ⟨10, 2, 3⟩,
  ⟨15, 3, 2⟩
]

end NUMINAMATH_CALUDE_sara_frosting_cans_l3046_304664


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_a_equals_two_l3046_304659

-- Define the point P
def P (a : ℤ) : ℝ × ℝ := (a - 1, a - 3)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant_implies_a_equals_two (a : ℤ) :
  in_fourth_quadrant (P a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_a_equals_two_l3046_304659


namespace NUMINAMATH_CALUDE_cone_volume_l3046_304679

/-- Given a cone with slant height 5 and lateral area 20π, its volume is 16π -/
theorem cone_volume (s : ℝ) (l : ℝ) (v : ℝ) 
  (h_slant : s = 5)
  (h_lateral : l = 20 * Real.pi) :
  v = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3046_304679


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3046_304614

theorem inequality_system_solution (x : ℝ) :
  x - 1 > 0 ∧ (2 * x + 1) / 3 ≤ 3 → 1 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3046_304614


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3046_304632

theorem simplify_square_roots : 
  Real.sqrt 18 * Real.sqrt 72 + Real.sqrt 200 = 36 + 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3046_304632


namespace NUMINAMATH_CALUDE_recurring_decimal_equiv_recurring_decimal_lowest_terms_l3046_304616

def recurring_decimal : ℚ := 433 / 990

theorem recurring_decimal_equiv : recurring_decimal = 0.4375375375375375375375375375375 := by sorry

theorem recurring_decimal_lowest_terms : ∀ a b : ℤ, (a : ℚ) / b = recurring_decimal → Nat.gcd a.natAbs b.natAbs = 1 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_equiv_recurring_decimal_lowest_terms_l3046_304616


namespace NUMINAMATH_CALUDE_right_angled_triangle_l3046_304629

theorem right_angled_triangle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_cosine_sum : Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = 1) : 
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l3046_304629


namespace NUMINAMATH_CALUDE_reciprocals_proportional_l3046_304672

/-- If x and y are directly proportional, then their reciprocals are also directly proportional -/
theorem reciprocals_proportional {x y : ℝ} (h : ∃ k : ℝ, y = k * x) :
  ∃ c : ℝ, (1 / y) = c * (1 / x) :=
by sorry

end NUMINAMATH_CALUDE_reciprocals_proportional_l3046_304672


namespace NUMINAMATH_CALUDE_proportion_solution_l3046_304641

/-- Given a proportion 0.75 : x :: 5 : 11, prove that x = 1.65 -/
theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 11) : x = 1.65 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3046_304641


namespace NUMINAMATH_CALUDE_square_difference_theorem_l3046_304623

theorem square_difference_theorem (a b A : ℝ) : 
  (5*a + 3*b)^2 = (5*a - 3*b)^2 + A → A = 60*a*b := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l3046_304623


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3046_304601

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  let s := d / Real.sqrt 2
  s * s = 72 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3046_304601


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l3046_304631

theorem jelly_bean_probability (p_red p_orange : ℝ) 
  (h_red : p_red = 0.25)
  (h_orange : p_orange = 0.35)
  (h_sum : p_red + p_orange + (p_yellow + p_green) = 1) :
  p_yellow + p_green = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l3046_304631


namespace NUMINAMATH_CALUDE_supplementary_angle_measure_l3046_304651

theorem supplementary_angle_measure (angle : ℝ) (supplementary : ℝ) (complementary : ℝ) : 
  angle = 45 →
  angle + supplementary = 180 →
  angle + complementary = 90 →
  supplementary = 3 * complementary →
  supplementary = 135 := by
sorry

end NUMINAMATH_CALUDE_supplementary_angle_measure_l3046_304651


namespace NUMINAMATH_CALUDE_distance_probability_l3046_304699

/-- Represents a city in our problem -/
inductive City : Type
| Bangkok : City
| CapeTown : City
| Honolulu : City
| London : City

/-- The distance between two cities in miles -/
def distance (c1 c2 : City) : ℕ :=
  match c1, c2 with
  | City.Bangkok, City.CapeTown => 6300
  | City.Bangkok, City.Honolulu => 6609
  | City.Bangkok, City.London => 5944
  | City.CapeTown, City.Honolulu => 11535
  | City.CapeTown, City.London => 5989
  | City.Honolulu, City.London => 7240
  | _, _ => 0  -- Same city or reverse order

/-- The total number of unique city pairs -/
def totalPairs : ℕ := 6

/-- The number of city pairs with distance less than 8000 miles -/
def pairsLessThan8000 : ℕ := 5

/-- The probability of selecting two cities with distance less than 8000 miles -/
def probability : ℚ := 5 / 6

theorem distance_probability :
  (probability : ℚ) = (pairsLessThan8000 : ℚ) / (totalPairs : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_distance_probability_l3046_304699


namespace NUMINAMATH_CALUDE_coffee_price_correct_l3046_304681

/-- The price of a cup of coffee satisfying the given conditions -/
def coffee_price : ℝ := 6

/-- The price of a piece of cheesecake -/
def cheesecake_price : ℝ := 10

/-- The discount rate applied to the set of coffee and cheesecake -/
def discount_rate : ℝ := 0.25

/-- The final price of the set (coffee + cheesecake) with discount -/
def discounted_set_price : ℝ := 12

/-- Theorem stating that the coffee price satisfies the given conditions -/
theorem coffee_price_correct :
  (1 - discount_rate) * (coffee_price + cheesecake_price) = discounted_set_price := by
  sorry

end NUMINAMATH_CALUDE_coffee_price_correct_l3046_304681


namespace NUMINAMATH_CALUDE_greatest_cars_with_ac_no_stripes_l3046_304658

theorem greatest_cars_with_ac_no_stripes (total : Nat) (no_ac : Nat) (min_stripes : Nat)
  (h1 : total = 100)
  (h2 : no_ac = 37)
  (h3 : min_stripes = 51)
  (h4 : min_stripes ≤ total)
  (h5 : no_ac < total) :
  (total - no_ac) - min_stripes = 12 :=
sorry

end NUMINAMATH_CALUDE_greatest_cars_with_ac_no_stripes_l3046_304658


namespace NUMINAMATH_CALUDE_hypotenuse_square_of_right_triangle_from_polynomial_roots_l3046_304620

/-- Given complex numbers a, b, and c that are zeros of a polynomial P(z) = z³ + pz² + qz + r,
    if |a|² + |b|² + |c|² = 300 and they form a right triangle in the complex plane,
    then the square of the hypotenuse h² = 400. -/
theorem hypotenuse_square_of_right_triangle_from_polynomial_roots
  (a b c : ℂ) (p q r : ℂ) :
  (a^3 + p*a^2 + q*a + r = 0) →
  (b^3 + p*b^2 + q*b + r = 0) →
  (c^3 + p*c^2 + q*c + r = 0) →
  Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 300 →
  ∃ (h : ℝ), (Complex.abs (a - c))^2 + (Complex.abs (b - c))^2 = h^2 →
  h^2 = 400 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_square_of_right_triangle_from_polynomial_roots_l3046_304620


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_sum_l3046_304602

theorem seventh_root_of_unity_sum (z : ℂ) :
  z ^ 7 = 1 ∧ z ≠ 1 →
  z + z^2 + z^4 = (-1 + Complex.I * Real.sqrt 11) / 2 ∨
  z + z^2 + z^4 = (-1 - Complex.I * Real.sqrt 11) / 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_sum_l3046_304602


namespace NUMINAMATH_CALUDE_cooper_savings_l3046_304666

/-- Calculates the total savings for a given daily savings amount and number of days. -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Proves that saving $34 daily for 365 days results in a total savings of $12,410. -/
theorem cooper_savings :
  totalSavings 34 365 = 12410 := by
  sorry

end NUMINAMATH_CALUDE_cooper_savings_l3046_304666


namespace NUMINAMATH_CALUDE_brians_books_l3046_304684

/-- The number of chapters in the first book Brian read -/
def book1_chapters : ℕ := 20

/-- The total number of chapters Brian read -/
def total_chapters : ℕ := 75

/-- The number of identical books Brian read -/
def identical_books : ℕ := 2

theorem brians_books (x : ℕ) : 
  book1_chapters + identical_books * x + (book1_chapters + identical_books * x) / 2 = total_chapters → 
  x = 15 :=
by sorry

end NUMINAMATH_CALUDE_brians_books_l3046_304684


namespace NUMINAMATH_CALUDE_parabola_closest_point_l3046_304653

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the distance between two points
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 - x2)^2 + (y1 - y2)^2

-- Theorem statement
theorem parabola_closest_point (a : ℝ) :
  (∀ x y : ℝ, parabola x y →
    ∃ xv yv : ℝ, parabola xv yv ∧
      ∀ x' y' : ℝ, parabola x' y' →
        distance_squared xv yv 0 a ≤ distance_squared x' y' 0 a) →
  a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_closest_point_l3046_304653


namespace NUMINAMATH_CALUDE_first_month_sale_l3046_304647

/-- Given the sales data for 6 months and the average sale, prove the sale amount for the first month -/
theorem first_month_sale
  (sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ)
  (average_sale : ℕ)
  (h1 : sales_2 = 6500)
  (h2 : sales_3 = 9855)
  (h3 : sales_4 = 7230)
  (h4 : sales_5 = 7000)
  (h5 : sales_6 = 11915)
  (h6 : average_sale = 7500)
  : ∃ (sales_1 : ℕ), sales_1 = 2500 ∧ 
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / 6 = average_sale :=
by sorry

end NUMINAMATH_CALUDE_first_month_sale_l3046_304647


namespace NUMINAMATH_CALUDE_power_of_product_l3046_304652

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3046_304652


namespace NUMINAMATH_CALUDE_square_to_acute_triangle_with_different_sides_l3046_304621

/-- A part of a square -/
structure SquarePart where
  -- Add necessary fields

/-- A triangle formed from parts of a square -/
structure TriangleFromSquare where
  parts : Finset SquarePart
  -- Add necessary fields for angles and sides

/-- Represents a square that can be cut into parts -/
structure CuttableSquare where
  side : ℝ
  -- Add other necessary fields

/-- Predicate to check if a triangle has acute angles -/
def has_acute_angles (t : TriangleFromSquare) : Prop :=
  sorry

/-- Predicate to check if a triangle has different sides -/
def has_different_sides (t : TriangleFromSquare) : Prop :=
  sorry

/-- Theorem stating that a square can be cut into 3 parts to form a specific triangle -/
theorem square_to_acute_triangle_with_different_sides :
  ∃ (s : CuttableSquare) (t : TriangleFromSquare),
    t.parts.card = 3 ∧
    has_acute_angles t ∧
    has_different_sides t :=
  sorry

end NUMINAMATH_CALUDE_square_to_acute_triangle_with_different_sides_l3046_304621


namespace NUMINAMATH_CALUDE_truck_speed_through_tunnel_l3046_304668

/-- Calculates the speed of a truck passing through a tunnel -/
theorem truck_speed_through_tunnel 
  (truck_length : ℝ) 
  (tunnel_length : ℝ) 
  (exit_time : ℝ) 
  (feet_per_mile : ℝ) 
  (h1 : truck_length = 66) 
  (h2 : tunnel_length = 330) 
  (h3 : exit_time = 6) 
  (h4 : feet_per_mile = 5280) :
  (truck_length + tunnel_length) / exit_time * 3600 / feet_per_mile = 45 := by
sorry

end NUMINAMATH_CALUDE_truck_speed_through_tunnel_l3046_304668


namespace NUMINAMATH_CALUDE_prime_square_minus_one_div_24_l3046_304655

theorem prime_square_minus_one_div_24 (p : Nat) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  24 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_div_24_l3046_304655


namespace NUMINAMATH_CALUDE_exists_special_expression_l3046_304635

/-- Represents an arithmetic expression with ones and alternating operations -/
inductive Expression
  | one : Expression
  | add : Expression → Expression → Expression
  | mul : Expression → Expression → Expression

/-- Evaluates an expression -/
def evaluate : Expression → ℕ
  | Expression.one => 1
  | Expression.add e1 e2 => evaluate e1 + evaluate e2
  | Expression.mul e1 e2 => evaluate e1 * evaluate e2

/-- Swaps addition and multiplication operations in an expression -/
def swap_operations : Expression → Expression
  | Expression.one => Expression.one
  | Expression.add e1 e2 => Expression.mul (swap_operations e1) (swap_operations e2)
  | Expression.mul e1 e2 => Expression.add (swap_operations e1) (swap_operations e2)

/-- Theorem stating the existence of an expression satisfying the problem conditions -/
theorem exists_special_expression : 
  ∃ e : Expression, evaluate e = 2014 ∧ evaluate (swap_operations e) = 2014 := by
  sorry


end NUMINAMATH_CALUDE_exists_special_expression_l3046_304635


namespace NUMINAMATH_CALUDE_triangle_construction_exists_unique_l3046_304645

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Checks if a triangle is isosceles with a given angle at the base -/
def isIsosceles (t : Triangle) (baseAngle : ℝ) : Prop :=
  sorry

/-- Checks if a point is the third vertex of an isosceles triangle -/
def isThirdVertex (base : Point × Point) (apex : Point) (baseAngle : ℝ) : Prop :=
  sorry

/-- The main theorem stating the existence and uniqueness of the construction -/
theorem triangle_construction_exists_unique 
  (A₁ B₁ C₁ : Point) : 
  ∃! (ABC : Triangle), 
    isIsosceles ⟨ABC.B, ABC.C, A₁⟩ (π/4) ∧
    isIsosceles ⟨ABC.C, ABC.A, B₁⟩ (π/4) ∧
    isIsosceles ⟨ABC.A, ABC.B, C₁⟩ (π/4) :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_exists_unique_l3046_304645


namespace NUMINAMATH_CALUDE_nickel_chocolates_l3046_304677

/-- Given that Robert ate 7 chocolates and 4 more than Nickel, prove that Nickel ate 3 chocolates. -/
theorem nickel_chocolates (robert : ℕ) (nickel : ℕ) 
  (h1 : robert = 7)
  (h2 : robert = nickel + 4) : 
  nickel = 3 := by
  sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l3046_304677


namespace NUMINAMATH_CALUDE_mod_37_5_l3046_304678

theorem mod_37_5 : 37 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_37_5_l3046_304678


namespace NUMINAMATH_CALUDE_bird_reserve_theorem_l3046_304650

/-- Represents the composition of birds in the Goshawk-Eurasian Nature Reserve -/
structure BirdReserve where
  total : ℝ
  hawks : ℝ
  paddyfield_warblers : ℝ
  kingfishers : ℝ

/-- The conditions of the bird reserve -/
def reserve_conditions (b : BirdReserve) : Prop :=
  b.hawks = 0.3 * b.total ∧
  b.paddyfield_warblers = 0.4 * (b.total - b.hawks) ∧
  b.kingfishers = 0.25 * b.paddyfield_warblers

/-- The theorem to be proved -/
theorem bird_reserve_theorem (b : BirdReserve) 
  (h : reserve_conditions b) : 
  (b.total - b.hawks - b.paddyfield_warblers - b.kingfishers) / b.total = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_bird_reserve_theorem_l3046_304650


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l3046_304604

/-- Given the conditions of a company's financial performance over two years,
    prove that the profit percentage in the previous year was 10%. -/
theorem profit_percentage_previous_year
  (R : ℝ) -- Revenues in the previous year
  (P : ℝ) -- Profits in the previous year
  (h1 : R > 0) -- Assume positive revenue
  (h2 : P > 0) -- Assume positive profit
  (h3 : 0.8 * R * 0.12 = 0.96 * P) -- Condition from the problem
  : P / R = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l3046_304604


namespace NUMINAMATH_CALUDE_complex_number_properties_l3046_304603

theorem complex_number_properties (z : ℂ) (h : z = 1 + I) : 
  (Complex.abs z = Real.sqrt 2) ∧ 
  (z ≠ 1 - I) ∧
  (z.im ≠ 1) ∧
  (0 < z.re ∧ 0 < z.im) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3046_304603


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3046_304686

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) : 
  (1 + 1 / (x + 1)) * ((x + 1) / (x^2 + 4*x + 4)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3046_304686


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_02008_l3046_304615

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Rounds a float to a specified number of significant figures -/
def roundToSigFigs (x : Float) (sigFigs : Nat) : Float :=
  sorry

/-- Converts a float to scientific notation -/
def toScientificNotation (x : Float) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundScientificNotation (sn : ScientificNotation) (sigFigs : Nat) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_02008 :
  let original := 0.02008
  let scientificNotation := toScientificNotation original
  let rounded := roundScientificNotation scientificNotation 3
  rounded = ScientificNotation.mk 2.01 (-2) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_02008_l3046_304615


namespace NUMINAMATH_CALUDE_square_remainder_is_square_l3046_304688

theorem square_remainder_is_square (N : ℤ) : ∃ (a b : ℤ), 
  ((N = 8 * a + b ∨ N = 8 * a - b) ∧ (b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4)) →
  ∃ (k : ℤ), N^2 % 16 = k^2 := by
sorry

end NUMINAMATH_CALUDE_square_remainder_is_square_l3046_304688


namespace NUMINAMATH_CALUDE_equal_distribution_payout_l3046_304633

def earnings : List ℝ := [30, 35, 45, 55, 65]

theorem equal_distribution_payout (earnings : List ℝ) : 
  earnings = [30, 35, 45, 55, 65] →
  List.length earnings = 5 →
  List.sum earnings / 5 = 46 →
  65 - (List.sum earnings / 5) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_payout_l3046_304633


namespace NUMINAMATH_CALUDE_integral_x_zero_to_one_l3046_304600

theorem integral_x_zero_to_one :
  ∫ x in (0 : ℝ)..1, x = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_integral_x_zero_to_one_l3046_304600


namespace NUMINAMATH_CALUDE_fourth_power_of_nested_square_root_l3046_304628

theorem fourth_power_of_nested_square_root : (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_nested_square_root_l3046_304628


namespace NUMINAMATH_CALUDE_festival_attendance_l3046_304689

theorem festival_attendance (total : ℕ) (d1 d2 d3 d4 : ℕ) : 
  total = 3600 →
  d2 = d1 / 2 →
  d3 = 3 * d1 →
  d4 = 2 * d2 →
  d1 + d2 + d3 + d4 = total →
  (total : ℚ) / 4 = 900 := by
  sorry

end NUMINAMATH_CALUDE_festival_attendance_l3046_304689


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3046_304660

theorem fourth_root_equation_solutions :
  {x : ℝ | x > 0 ∧ (x^(1/4) : ℝ) = 15 / (8 - (x^(1/4) : ℝ))} = {625, 81} := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3046_304660


namespace NUMINAMATH_CALUDE_rectangle_area_is_220_l3046_304636

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of rectangle PQRS -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculate the area of a rectangle given three of its vertices -/
def rectangleArea (rect : Rectangle) : ℝ :=
  let width := abs (rect.Q.x - rect.P.x)
  let height := abs (rect.Q.y - rect.R.y)
  width * height

/-- Theorem: The area of rectangle PQRS with given vertices is 220 -/
theorem rectangle_area_is_220 : ∃ (S : Point),
  let rect : Rectangle := {
    P := { x := 15, y := 55 },
    Q := { x := 26, y := 55 },
    R := { x := 26, y := 35 },
    S := S
  }
  rectangleArea rect = 220 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_220_l3046_304636


namespace NUMINAMATH_CALUDE_equation_solutions_l3046_304618

theorem equation_solutions :
  (∀ x : ℝ, 25 * x^2 = 81 ↔ x = 9/5 ∨ x = -9/5) ∧
  (∀ x : ℝ, (x - 2)^2 = 25 ↔ x = 7 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3046_304618


namespace NUMINAMATH_CALUDE_first_year_growth_rate_is_15_percent_l3046_304687

def initial_investment : ℝ := 80
def additional_investment : ℝ := 28
def second_year_growth_rate : ℝ := 0.10
def final_portfolio_value : ℝ := 132

theorem first_year_growth_rate_is_15_percent :
  ∃ r : ℝ, 
    (initial_investment * (1 + r) + additional_investment) * (1 + second_year_growth_rate) = final_portfolio_value ∧
    r = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_first_year_growth_rate_is_15_percent_l3046_304687


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3046_304643

theorem exponential_equation_solution :
  ∀ x : ℝ, (10 : ℝ)^x * (1000 : ℝ)^(2*x) = (100 : ℝ)^6 → x = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3046_304643


namespace NUMINAMATH_CALUDE_triangle_area_angle_relation_l3046_304676

theorem triangle_area_angle_relation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let A := Real.sqrt 3 / 12 * (a^2 + c^2 - b^2)
  (∃ (B : ℝ), 0 < B ∧ B < π ∧ A = 1/2 * a * c * Real.sin B) → 
  (∃ (B : ℝ), 0 < B ∧ B < π ∧ B = π/6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_angle_relation_l3046_304676


namespace NUMINAMATH_CALUDE_paiges_dresser_capacity_l3046_304695

/-- Calculates the total number of clothing pieces a dresser can hold. -/
def dresser_capacity (drawers : ℕ) (pieces_per_drawer : ℕ) : ℕ :=
  drawers * pieces_per_drawer

/-- Proves that Paige's dresser can hold 8 pieces of clothing. -/
theorem paiges_dresser_capacity :
  dresser_capacity 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_paiges_dresser_capacity_l3046_304695


namespace NUMINAMATH_CALUDE_function_composition_equality_l3046_304669

theorem function_composition_equality (f g : ℝ → ℝ) (b : ℝ) : 
  (∀ x, f x = x / 6 + 2) → 
  (∀ x, g x = 5 - 2 * x) → 
  f (g b) = 4 → 
  b = -7 / 2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3046_304669


namespace NUMINAMATH_CALUDE_calculation_proof_l3046_304680

theorem calculation_proof : 8 - (7.14 * (1/3) - 2 * (2/9) / 2.5) + 0.1 = 6.62 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3046_304680


namespace NUMINAMATH_CALUDE_min_cos_sum_with_tan_product_l3046_304697

theorem min_cos_sum_with_tan_product (x y m : ℝ) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (hm : m > 2) 
  (h : Real.tan x * Real.tan y = m) : 
  Real.cos x + Real.cos y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_cos_sum_with_tan_product_l3046_304697


namespace NUMINAMATH_CALUDE_consecutive_integers_permutation_divisibility_l3046_304694

theorem consecutive_integers_permutation_divisibility
  (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (m : ℕ → ℕ) (hm : ∀ i ∈ Finset.range p, m (i + 1) = m i + 1)
  (σ : Fin p → Fin p) (hσ : Function.Bijective σ) :
  ∃ k l : Fin p, k ≠ l ∧ p ∣ (m k * m (σ k) - m l * m (σ l)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_permutation_divisibility_l3046_304694


namespace NUMINAMATH_CALUDE_three_tangents_condition_l3046_304691

/-- The curve function f(x) = x³ + 3x² + ax + a - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a*x + a - 2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*x + a

/-- The tangent line equation passing through (0, 2) -/
def tangent_line (a : ℝ) (x₀ : ℝ) (x : ℝ) : ℝ :=
  f_deriv a x₀ * (x - x₀) + f a x₀

/-- The condition for a point x₀ to be on a tangent line passing through (0, 2) -/
def tangent_condition (a : ℝ) (x₀ : ℝ) : Prop :=
  tangent_line a x₀ 0 = 2

/-- The main theorem stating the condition for exactly three tangent lines -/
theorem three_tangents_condition (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    tangent_condition a x₁ ∧ tangent_condition a x₂ ∧ tangent_condition a x₃ ∧
    (∀ x : ℝ, tangent_condition a x → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  4 < a ∧ a < 5 :=
sorry

end NUMINAMATH_CALUDE_three_tangents_condition_l3046_304691
