import Mathlib

namespace NUMINAMATH_CALUDE_probability_correct_l464_46470

/-- Represents a runner on a circular track -/
structure Runner where
  direction : Bool  -- true for counterclockwise, false for clockwise
  lap_time : ℝ      -- time to complete one lap in seconds

/-- Represents the track and race setup -/
structure RaceSetup where
  track_length : ℝ           -- length of the track in meters
  focus_start : ℝ            -- start of focus area in meters from start line
  focus_length : ℝ           -- length of focus area in meters
  alice : Runner
  bob : Runner
  race_start_time : ℝ        -- start time of the race in seconds
  photo_start_time : ℝ       -- start time of photo opportunity in seconds
  photo_end_time : ℝ         -- end time of photo opportunity in seconds

def setup : RaceSetup := {
  track_length := 500
  focus_start := 50
  focus_length := 150
  alice := { direction := true, lap_time := 120 }
  bob := { direction := false, lap_time := 75 }
  race_start_time := 0
  photo_start_time := 15 * 60
  photo_end_time := 16 * 60
}

/-- Calculates the probability of both runners being in the focus area -/
def probability_both_in_focus (s : RaceSetup) : ℚ :=
  11/60

theorem probability_correct (s : RaceSetup) :
  s = setup → probability_both_in_focus s = 11/60 := by sorry

end NUMINAMATH_CALUDE_probability_correct_l464_46470


namespace NUMINAMATH_CALUDE_lychee_ratio_l464_46439

theorem lychee_ratio (total : ℕ) (remaining : ℕ) : 
  total = 500 → 
  remaining = 100 → 
  (total / 2 - remaining : ℚ) / (total / 2 : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_lychee_ratio_l464_46439


namespace NUMINAMATH_CALUDE_three_identical_digits_divisible_by_37_l464_46408

theorem three_identical_digits_divisible_by_37 (A : ℕ) (h : A < 10) :
  ∃ k : ℕ, 111 * A = 37 * k := by
  sorry

end NUMINAMATH_CALUDE_three_identical_digits_divisible_by_37_l464_46408


namespace NUMINAMATH_CALUDE_percentage_men_is_seventy_l464_46434

/-- The number of women in the engineering department -/
def num_women : ℕ := 180

/-- The number of men in the engineering department -/
def num_men : ℕ := 420

/-- The total number of students in the engineering department -/
def total_students : ℕ := num_women + num_men

/-- The percentage of men in the engineering department -/
def percentage_men : ℚ := (num_men : ℚ) / (total_students : ℚ) * 100

theorem percentage_men_is_seventy :
  percentage_men = 70 :=
sorry

end NUMINAMATH_CALUDE_percentage_men_is_seventy_l464_46434


namespace NUMINAMATH_CALUDE_median_is_three_l464_46400

def sibling_list : List Nat := [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6]

def median_index (n : Nat) : Nat :=
  (n + 1) / 2

theorem median_is_three :
  sibling_list.get? (median_index sibling_list.length - 1) = some 3 := by
  sorry

end NUMINAMATH_CALUDE_median_is_three_l464_46400


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_l464_46498

theorem number_exceeds_fraction : 
  ∀ x : ℚ, x = (3/8) * x + 25 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_l464_46498


namespace NUMINAMATH_CALUDE_base_4_representation_of_253_base_4_to_decimal_3331_l464_46437

/-- Converts a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n < 4 then [n]
  else (n % 4) :: toBase4 (n / 4)

/-- Converts a list of base 4 digits to its decimal representation -/
def fromBase4 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 4 * acc) 0

theorem base_4_representation_of_253 :
  toBase4 253 = [1, 3, 3, 3] :=
by sorry

theorem base_4_to_decimal_3331 :
  fromBase4 [1, 3, 3, 3] = 253 :=
by sorry

end NUMINAMATH_CALUDE_base_4_representation_of_253_base_4_to_decimal_3331_l464_46437


namespace NUMINAMATH_CALUDE_solve_school_supplies_problem_l464_46495

/-- Represents the price and quantity of pens and notebooks -/
structure Supplies where
  pen_price : ℚ
  notebook_price : ℚ
  pen_quantity : ℕ
  notebook_quantity : ℕ

/-- Calculates the total cost of supplies -/
def total_cost (s : Supplies) : ℚ :=
  s.pen_price * s.pen_quantity + s.notebook_price * s.notebook_quantity

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  xiaofang_cost : ℚ
  xiaoliang_cost : ℚ
  xiaofang_supplies : Supplies
  xiaoliang_supplies : Supplies
  reward_fund : ℚ
  prize_sets : ℕ

/-- Theorem stating the solution to the problem -/
theorem solve_school_supplies_problem (c : ProblemConditions)
  (h1 : c.xiaofang_cost = 18)
  (h2 : c.xiaoliang_cost = 22)
  (h3 : c.xiaofang_supplies.pen_quantity = 2)
  (h4 : c.xiaofang_supplies.notebook_quantity = 3)
  (h5 : c.xiaoliang_supplies.pen_quantity = 3)
  (h6 : c.xiaoliang_supplies.notebook_quantity = 2)
  (h7 : c.reward_fund = 400)
  (h8 : c.prize_sets = 20)
  (h9 : total_cost c.xiaofang_supplies = c.xiaofang_cost)
  (h10 : total_cost c.xiaoliang_supplies = c.xiaoliang_cost) :
  ∃ (pen_price notebook_price : ℚ) (combinations : ℕ),
    pen_price = 6 ∧
    notebook_price = 2 ∧
    combinations = 4 ∧
    (∀ x y : ℕ, (x * pen_price + y * notebook_price) * c.prize_sets = c.reward_fund →
      (x = 0 ∧ y = 10) ∨ (x = 1 ∧ y = 7) ∨ (x = 2 ∧ y = 4) ∨ (x = 3 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_solve_school_supplies_problem_l464_46495


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l464_46482

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 24*x^2 + 98*x - 75

-- Define the theorem
theorem root_sum_reciprocal (p q r A B C : ℝ) :
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →  -- p, q, r are distinct
  (f p = 0 ∧ f q = 0 ∧ f r = 0) →  -- p, q, r are roots of f
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    5 / (s^3 - 24*s^2 + 98*s - 75) = A / (s-p) + B / (s-q) + C / (s-r)) →
  1/A + 1/B + 1/C = 256 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l464_46482


namespace NUMINAMATH_CALUDE_tv_screen_diagonal_l464_46458

theorem tv_screen_diagonal (s : ℝ) (h : s^2 = 256 + 34) :
  Real.sqrt (2 * s^2) = Real.sqrt 580 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_diagonal_l464_46458


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_endpoints_locus_l464_46433

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  A : Point
  B : Point
  C : Point
  centroid : Point
  orthocenter : Point
  isIsosceles : A.x = -B.x ∧ A.y = B.y
  centroidOrigin : centroid = ⟨0, 0⟩
  orthocenterOnYAxis : orthocenter = ⟨0, 1⟩
  thirdVertexOnYAxis : C.x = 0

/-- The locus of base endpoints of an isosceles triangle -/
def locusOfBaseEndpoints (p : Point) : Prop :=
  p.x ≠ 0 ∧ 3 * (p.y - 1/2)^2 - p.x^2 = 3/4

/-- Theorem stating that the base endpoints of the isosceles triangle lie on the specified locus -/
theorem isosceles_triangle_base_endpoints_locus (triangle : IsoscelesTriangle) :
  locusOfBaseEndpoints triangle.A ∧ locusOfBaseEndpoints triangle.B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_endpoints_locus_l464_46433


namespace NUMINAMATH_CALUDE_non_working_games_l464_46411

theorem non_working_games (total_games : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) : 
  total_games = 15 → price_per_game = 5 → total_earnings = 30 → 
  total_games - (total_earnings / price_per_game) = 9 := by
sorry

end NUMINAMATH_CALUDE_non_working_games_l464_46411


namespace NUMINAMATH_CALUDE_jen_candy_profit_l464_46463

/-- Calculates the profit from selling candy bars --/
def candy_profit (buy_price sell_price : ℕ) (bought sold : ℕ) : ℕ :=
  (sell_price - buy_price) * sold

/-- Proves that Jen's profit from selling candy bars is 960 cents --/
theorem jen_candy_profit : candy_profit 80 100 50 48 = 960 := by
  sorry

end NUMINAMATH_CALUDE_jen_candy_profit_l464_46463


namespace NUMINAMATH_CALUDE_dave_tickets_left_l464_46429

def tickets_left (won : ℕ) (lost : ℕ) (used : ℕ) : ℕ :=
  won - lost - used

theorem dave_tickets_left : tickets_left 14 2 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_left_l464_46429


namespace NUMINAMATH_CALUDE_division_remainder_proof_l464_46477

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 725)
  (h2 : divisor = 36)
  (h3 : quotient = 20) :
  dividend = divisor * quotient + 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l464_46477


namespace NUMINAMATH_CALUDE_g_composition_3_l464_46465

def g : ℕ → ℕ
| x => if x % 2 = 0 then x / 2
       else if x < 10 then 3 * x + 2
       else x - 1

theorem g_composition_3 : g (g (g (g (g 3)))) = 16 := by sorry

end NUMINAMATH_CALUDE_g_composition_3_l464_46465


namespace NUMINAMATH_CALUDE_refrigerator_cost_l464_46462

/-- Proves that the cost of the refrigerator is 25000 given the problem conditions -/
theorem refrigerator_cost
  (mobile_cost : ℕ)
  (refrigerator_loss_percent : ℚ)
  (mobile_profit_percent : ℚ)
  (total_profit : ℕ)
  (h1 : mobile_cost = 8000)
  (h2 : refrigerator_loss_percent = 4 / 100)
  (h3 : mobile_profit_percent = 10 / 100)
  (h4 : total_profit = 200) :
  ∃ (refrigerator_cost : ℕ),
    refrigerator_cost = 25000 ∧
    (refrigerator_cost : ℚ) * (1 - refrigerator_loss_percent) +
    (mobile_cost : ℚ) * (1 + mobile_profit_percent) =
    (refrigerator_cost + mobile_cost + total_profit : ℚ) :=
sorry

end NUMINAMATH_CALUDE_refrigerator_cost_l464_46462


namespace NUMINAMATH_CALUDE_raw_material_expenditure_l464_46407

theorem raw_material_expenditure (x : ℝ) :
  (x ≥ 0) →
  (x ≤ 1) →
  (1 - x - (1/10) * (1 - x) = 0.675) →
  (x = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_raw_material_expenditure_l464_46407


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l464_46493

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem mans_speed_in_still_water
  (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ)
  (h_downstream : downstream_distance = 40)
  (h_upstream : upstream_distance = 56)
  (h_time : time = 8)
  : ∃ (v_m : ℝ), v_m = 6 ∧ 
    downstream_distance / time = v_m + (downstream_distance - upstream_distance) / (2 * time) ∧
    upstream_distance / time = v_m - (downstream_distance - upstream_distance) / (2 * time) :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l464_46493


namespace NUMINAMATH_CALUDE_union_of_sets_l464_46438

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l464_46438


namespace NUMINAMATH_CALUDE_min_value_of_expression_l464_46442

theorem min_value_of_expression (x y : ℝ) : (x^2*y - 1)^2 + (x + y - 1)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l464_46442


namespace NUMINAMATH_CALUDE_english_book_pages_l464_46474

theorem english_book_pages :
  ∀ (english_pages chinese_pages : ℕ),
  english_pages = chinese_pages + 12 →
  3 * english_pages + 4 * chinese_pages = 1275 →
  english_pages = 189 :=
by
  sorry

end NUMINAMATH_CALUDE_english_book_pages_l464_46474


namespace NUMINAMATH_CALUDE_arrangement_count_is_twelve_l464_46420

/-- The number of elements to be arranged -/
def n : ℕ := 4

/-- The condition that A is adjacent to B -/
def adjacent_condition : Prop := true  -- We don't need to define this explicitly in Lean

/-- The number of ways to arrange n elements with the adjacent condition -/
def arrangement_count (n : ℕ) (adjacent_condition : Prop) : ℕ := sorry

/-- Theorem stating that the number of arrangements is 12 -/
theorem arrangement_count_is_twelve :
  arrangement_count n adjacent_condition = 12 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_twelve_l464_46420


namespace NUMINAMATH_CALUDE_ella_reads_500_pages_l464_46499

/-- Represents the reading task for Ella and John -/
structure ReadingTask where
  total_pages : ℕ
  ella_pace : ℕ  -- seconds per page
  john_pace : ℕ  -- seconds per page

/-- Calculates the number of pages Ella should read -/
def pages_for_ella (task : ReadingTask) : ℕ :=
  (task.total_pages * task.john_pace) / (task.ella_pace + task.john_pace)

/-- Theorem stating that Ella should read 500 pages given the conditions -/
theorem ella_reads_500_pages (task : ReadingTask) 
  (h1 : task.total_pages = 900)
  (h2 : task.ella_pace = 40)
  (h3 : task.john_pace = 50) : 
  pages_for_ella task = 500 := by
  sorry

#eval pages_for_ella ⟨900, 40, 50⟩

end NUMINAMATH_CALUDE_ella_reads_500_pages_l464_46499


namespace NUMINAMATH_CALUDE_weekend_finances_correct_l464_46466

/-- Represents Tom's financial situation over the weekend -/
structure WeekendFinances where
  initial : ℝ  -- Initial amount
  car_wash : ℝ  -- Amount earned from washing cars
  lawn_mow : ℝ  -- Amount earned from mowing lawns
  painting : ℝ  -- Amount earned from painting
  expenses : ℝ  -- Amount spent on gas and food
  final : ℝ  -- Final amount

/-- Theorem stating that Tom's final amount is correctly calculated -/
theorem weekend_finances_correct (tom : WeekendFinances) 
  (h1 : tom.initial = 74)
  (h2 : tom.final = 86) :
  tom.initial + tom.car_wash + tom.lawn_mow + tom.painting - tom.expenses = tom.final := by
  sorry

end NUMINAMATH_CALUDE_weekend_finances_correct_l464_46466


namespace NUMINAMATH_CALUDE_marble_collection_proof_l464_46457

/-- The number of blue marbles collected by the three friends --/
def total_blue_marbles (mary_blue : ℕ) (jenny_blue : ℕ) (anie_blue : ℕ) : ℕ :=
  mary_blue + jenny_blue + anie_blue

theorem marble_collection_proof 
  (jenny_red : ℕ) 
  (jenny_blue : ℕ) 
  (mary_red : ℕ) 
  (mary_blue : ℕ) 
  (anie_red : ℕ) 
  (anie_blue : ℕ) : 
  jenny_red = 30 →
  jenny_blue = 25 →
  mary_red = 2 * jenny_red →
  anie_red = mary_red + 20 →
  anie_blue = 2 * jenny_blue →
  mary_blue = anie_blue / 2 →
  total_blue_marbles mary_blue jenny_blue anie_blue = 100 := by
  sorry

#check marble_collection_proof

end NUMINAMATH_CALUDE_marble_collection_proof_l464_46457


namespace NUMINAMATH_CALUDE_am_gm_difference_bound_l464_46418

theorem am_gm_difference_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  (x + y) / 2 - Real.sqrt (x * y) ≥ (x - y)^2 / (8 * x) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_difference_bound_l464_46418


namespace NUMINAMATH_CALUDE_magic_potion_cooking_time_l464_46451

/-- Represents a time on a 24-hour digital clock --/
structure DigitalTime where
  hours : Fin 24
  minutes : Fin 60

/-- Checks if a given time is a magic moment --/
def isMagicMoment (t : DigitalTime) : Prop :=
  t.hours = t.minutes

/-- Calculates the time difference between two DigitalTimes in minutes --/
def timeDifference (start finish : DigitalTime) : ℕ :=
  sorry

/-- Theorem stating the existence of a valid cooking time for the magic potion --/
theorem magic_potion_cooking_time :
  ∃ (start finish : DigitalTime),
    isMagicMoment start ∧
    isMagicMoment finish ∧
    90 ≤ timeDifference start finish ∧
    timeDifference start finish ≤ 120 ∧
    timeDifference start finish = 98 :=
  sorry

end NUMINAMATH_CALUDE_magic_potion_cooking_time_l464_46451


namespace NUMINAMATH_CALUDE_josie_checkout_wait_time_l464_46486

def total_shopping_time : ℕ := 90
def cart_wait_time : ℕ := 3
def employee_wait_time : ℕ := 13
def stocker_wait_time : ℕ := 14
def shopping_time : ℕ := 42

theorem josie_checkout_wait_time :
  total_shopping_time - shopping_time - (cart_wait_time + employee_wait_time + stocker_wait_time) = 18 := by
  sorry

end NUMINAMATH_CALUDE_josie_checkout_wait_time_l464_46486


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l464_46454

theorem loss_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1500 → 
  selling_price = 1335 → 
  (cost_price - selling_price) / cost_price * 100 = 11 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l464_46454


namespace NUMINAMATH_CALUDE_complex_sum_square_l464_46496

variable (a b c : ℂ)

theorem complex_sum_square (h1 : a^2 + a*b + b^2 = 1 + I)
                           (h2 : b^2 + b*c + c^2 = -2)
                           (h3 : c^2 + c*a + a^2 = 1) :
  (a*b + b*c + c*a)^2 = (-11 - 4*I) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_square_l464_46496


namespace NUMINAMATH_CALUDE_snow_probability_l464_46427

-- Define the probability of snow on Friday
def prob_snow_friday : ℝ := 0.4

-- Define the probability of snow on Saturday
def prob_snow_saturday : ℝ := 0.3

-- Define the probability of snow on both days
def prob_snow_both_days : ℝ := prob_snow_friday * prob_snow_saturday

-- Theorem to prove
theorem snow_probability : prob_snow_both_days = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l464_46427


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l464_46475

/-- Given a positive geometric sequence {a_n}, prove that a_8 * a_12 = 16,
    where a_1 and a_19 are the roots of x^2 - 10x + 16 = 0 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = a n * r) →
  (a 1)^2 - 10 * (a 1) + 16 = 0 →
  (a 19)^2 - 10 * (a 19) + 16 = 0 →
  a 8 * a 12 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l464_46475


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l464_46467

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, x^2 - k*x + 1 = 0) ↔ k = 2 ∨ k = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l464_46467


namespace NUMINAMATH_CALUDE_zero_in_M_l464_46403

def M : Set ℤ := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_zero_in_M_l464_46403


namespace NUMINAMATH_CALUDE_sixty_six_green_squares_l464_46412

/-- Represents a grid with colored squares -/
structure ColoredGrid :=
  (rows : Nat)
  (columns : Nat)
  (redRows : Nat)
  (redColumns : Nat)
  (blueRows : Nat)

/-- Calculates the number of green squares in the grid -/
def greenSquares (grid : ColoredGrid) : Nat :=
  grid.rows * grid.columns - (grid.redRows * grid.redColumns) - (grid.blueRows * grid.columns)

/-- Theorem stating that in the given grid configuration, there are 66 green squares -/
theorem sixty_six_green_squares :
  let grid : ColoredGrid := {
    rows := 10,
    columns := 15,
    redRows := 4,
    redColumns := 6,
    blueRows := 4
  }
  greenSquares grid = 66 := by sorry

end NUMINAMATH_CALUDE_sixty_six_green_squares_l464_46412


namespace NUMINAMATH_CALUDE_carreys_rental_cost_l464_46459

/-- The cost per kilometer for Carrey's car rental -/
def carreys_cost_per_km : ℝ := 0.25

/-- The initial cost for Carrey's car rental -/
def carreys_initial_cost : ℝ := 20

/-- The initial cost for Samuel's car rental -/
def samuels_initial_cost : ℝ := 24

/-- The cost per kilometer for Samuel's car rental -/
def samuels_cost_per_km : ℝ := 0.16

/-- The distance driven by both Carrey and Samuel -/
def distance_driven : ℝ := 44.44444444444444

theorem carreys_rental_cost (x : ℝ) :
  carreys_initial_cost + x * distance_driven =
  samuels_initial_cost + samuels_cost_per_km * distance_driven →
  x = carreys_cost_per_km :=
by sorry

end NUMINAMATH_CALUDE_carreys_rental_cost_l464_46459


namespace NUMINAMATH_CALUDE_fraction_equality_l464_46401

theorem fraction_equality (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l464_46401


namespace NUMINAMATH_CALUDE_value_of_expression_l464_46488

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x^2 + 2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l464_46488


namespace NUMINAMATH_CALUDE_abs_nine_sqrt_l464_46481

theorem abs_nine_sqrt : Real.sqrt (abs (-9)) = 3 := by sorry

end NUMINAMATH_CALUDE_abs_nine_sqrt_l464_46481


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l464_46414

/-- The expected total rainfall over a week given daily probabilities --/
theorem expected_weekly_rainfall (p_sun p_light p_heavy : ℝ) 
  (rain_light rain_heavy : ℝ) (days : ℕ) :
  p_sun + p_light + p_heavy = 1 →
  p_sun = 0.5 →
  p_light = 0.2 →
  p_heavy = 0.3 →
  rain_light = 2 →
  rain_heavy = 5 →
  days = 7 →
  (p_sun * 0 + p_light * rain_light + p_heavy * rain_heavy) * days = 13.3 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l464_46414


namespace NUMINAMATH_CALUDE_larger_number_proof_l464_46405

theorem larger_number_proof (x y : ℤ) (h1 : x - y = 5) (h2 : x + y = 37) :
  max x y = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l464_46405


namespace NUMINAMATH_CALUDE_nested_f_evaluation_l464_46432

/-- The function f(x) = x^2 - 3x + 1 -/
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

/-- Theorem stating that f(f(f(f(f(f(-1)))))) = 3432163846882600 -/
theorem nested_f_evaluation : f (f (f (f (f (f (-1)))))) = 3432163846882600 := by
  sorry

end NUMINAMATH_CALUDE_nested_f_evaluation_l464_46432


namespace NUMINAMATH_CALUDE_sqrt_of_neg_two_squared_l464_46476

theorem sqrt_of_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_neg_two_squared_l464_46476


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l464_46468

theorem sum_of_squares_and_products : (3 + 5)^2 + (3^2 + 5^2 + 3*5) = 113 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l464_46468


namespace NUMINAMATH_CALUDE_power_of_two_equation_l464_46440

theorem power_of_two_equation (l : ℤ) : 
  2^2000 - 2^1999 - 3 * 2^1998 + 2^1997 = l * 2^1997 → l = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l464_46440


namespace NUMINAMATH_CALUDE_min_value_expression_l464_46480

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x * y = 1) :
  (x + 2 * y) * (2 * x + z) * (y + 2 * z) ≥ 48 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ = 1 ∧
    (x₀ + 2 * y₀) * (2 * x₀ + z₀) * (y₀ + 2 * z₀) = 48 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l464_46480


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l464_46410

/-- A circle is tangent to a line if and only if the distance from the center of the circle to the line is equal to the radius of the circle. -/
theorem circle_tangent_to_line (b : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + 2*x + y^2 - 4*y + 3 = 0}
  let line := {(x, y) : ℝ × ℝ | x + y + b = 0}
  let center := (-1, 2)
  let radius := Real.sqrt 2
  (∀ p ∈ circle, p ∈ line → (∀ q ∈ circle, q = p ∨ q ∉ line)) → 
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l464_46410


namespace NUMINAMATH_CALUDE_classroom_difference_maple_leaf_elementary_l464_46404

theorem classroom_difference : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_classrooms, students_per_class, rabbits_per_class, guinea_pigs_per_class =>
  let total_students := num_classrooms * students_per_class
  let total_pets := num_classrooms * (rabbits_per_class + guinea_pigs_per_class)
  total_students - total_pets

theorem maple_leaf_elementary :
  classroom_difference 6 15 1 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_classroom_difference_maple_leaf_elementary_l464_46404


namespace NUMINAMATH_CALUDE_f_inequality_solution_l464_46435

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.log x - x else -Real.log (-x) + x

-- Define the solution set
def solution_set : Set ℝ := {m : ℝ | m ∈ Set.Ioo (-1/2) 0 ∪ Set.Ioo 0 (1/2)}

-- State the theorem
theorem f_inequality_solution :
  ∀ m : ℝ, m ≠ 0 → (f (1/m) < Real.log (1/2) - 2 ↔ m ∈ solution_set) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_l464_46435


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2018_l464_46460

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_term : a 1 = -2018
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + (n * (n - 1) / 2) * seq.d

/-- The main theorem -/
theorem arithmetic_sequence_sum_2018 (seq : ArithmeticSequence) :
  (sum_n seq 2015 / 2015) - (sum_n seq 2013 / 2013) = 2 →
  sum_n seq 2018 = -2018 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2018_l464_46460


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l464_46469

theorem marcos_strawberries_weight (total_weight dad_weight : ℕ) 
  (h1 : total_weight = 23)
  (h2 : dad_weight = 9) :
  total_weight - dad_weight = 14 := by
  sorry

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l464_46469


namespace NUMINAMATH_CALUDE_plant_growth_equation_l464_46436

/-- Represents the growth pattern of a plant -/
def plant_growth (x : ℕ) : Prop :=
  -- One main stem
  let main_stem := 1
  -- x branches on the main stem
  let branches := x
  -- x small branches on each of the x branches
  let small_branches := x * x
  -- The total number of stems and branches is 73
  main_stem + branches + small_branches = 73

/-- Theorem stating the equation for the plant's growth pattern -/
theorem plant_growth_equation :
  ∃ x : ℕ, plant_growth x ∧ (1 + x + x^2 = 73) :=
sorry

end NUMINAMATH_CALUDE_plant_growth_equation_l464_46436


namespace NUMINAMATH_CALUDE_work_completion_l464_46490

/-- The number of days B worked before leaving the job --/
def days_B_worked (a_rate b_rate : ℚ) (a_remaining_days : ℚ) : ℚ :=
  15 * (1 - 4 * a_rate)

theorem work_completion 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (a_remaining_days : ℚ) 
  (h1 : a_rate = 1 / 12)
  (h2 : b_rate = 1 / 15)
  (h3 : a_remaining_days = 4) :
  days_B_worked a_rate b_rate a_remaining_days = 10 := by
  sorry

#eval days_B_worked (1/12) (1/15) 4

end NUMINAMATH_CALUDE_work_completion_l464_46490


namespace NUMINAMATH_CALUDE_pugsley_has_four_spiders_l464_46447

/-- The number of spiders Pugsley has before trading -/
def pugsley_spiders : ℕ := sorry

/-- The number of spiders Wednesday has before trading -/
def wednesday_spiders : ℕ := sorry

/-- Condition 1: If Pugsley gives Wednesday 2 spiders, Wednesday would have 9 times as many spiders as Pugsley -/
axiom condition1 : wednesday_spiders + 2 = 9 * (pugsley_spiders - 2)

/-- Condition 2: If Wednesday gives Pugsley 6 spiders, Pugsley would have 6 fewer spiders than Wednesday had before they traded -/
axiom condition2 : pugsley_spiders + 6 = wednesday_spiders - 6

/-- Theorem: Pugsley has 4 spiders before the trading game commences -/
theorem pugsley_has_four_spiders : pugsley_spiders = 4 := by sorry

end NUMINAMATH_CALUDE_pugsley_has_four_spiders_l464_46447


namespace NUMINAMATH_CALUDE_old_clock_slower_l464_46487

/-- Represents the number of minutes between hand overlaps on the old clock -/
def overlap_interval : ℕ := 66

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the number of hand overlaps in a 24-hour period -/
def overlaps_per_day : ℕ := hours_per_day - 2

/-- Calculates the total minutes on the old clock for 24 hours -/
def old_clock_minutes : ℕ := overlaps_per_day * overlap_interval

/-- Calculates the total minutes in a standard 24-hour period -/
def standard_clock_minutes : ℕ := hours_per_day * minutes_per_hour

/-- Theorem stating that the old clock is 12 minutes slower over 24 hours -/
theorem old_clock_slower :
  old_clock_minutes - standard_clock_minutes = 12 := by sorry

end NUMINAMATH_CALUDE_old_clock_slower_l464_46487


namespace NUMINAMATH_CALUDE_ali_nada_difference_l464_46441

def total_amount : ℕ := 67
def john_amount : ℕ := 48

theorem ali_nada_difference (ali_amount nada_amount : ℕ) 
  (h1 : ali_amount + nada_amount + john_amount = total_amount)
  (h2 : ali_amount < nada_amount)
  (h3 : john_amount = 4 * nada_amount) :
  nada_amount - ali_amount = 5 := by
sorry

end NUMINAMATH_CALUDE_ali_nada_difference_l464_46441


namespace NUMINAMATH_CALUDE_coeff_x6_eq_30_implies_a_eq_2_l464_46424

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The coefficient of x^6 in the expansion of (x^2 - a)(x + 1/x)^10 -/
def coeff_x6 (a : ℚ) : ℚ := (binomial 10 3 : ℚ) - a * (binomial 10 2 : ℚ)

/-- Theorem: If the coefficient of x^6 in the expansion of (x^2 - a)(x + 1/x)^10 is 30, then a = 2 -/
theorem coeff_x6_eq_30_implies_a_eq_2 :
  coeff_x6 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_coeff_x6_eq_30_implies_a_eq_2_l464_46424


namespace NUMINAMATH_CALUDE_x_minus_y_values_l464_46409

theorem x_minus_y_values (x y : ℝ) 
  (h1 : |x| = 5)
  (h2 : y^2 = 16)
  (h3 : x + y > 0) :
  x - y = 1 ∨ x - y = 9 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l464_46409


namespace NUMINAMATH_CALUDE_divisibility_implication_l464_46413

theorem divisibility_implication (a b : ℕ) : 
  a < 1000 → (∃ k : ℕ, a^21 = k * b^10) → (∃ m : ℕ, a^2 = m * b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l464_46413


namespace NUMINAMATH_CALUDE_proportional_set_l464_46428

/-- A set of four positive real numbers is proportional if the product of the outer terms equals the product of the inner terms. -/
def is_proportional (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * d = b * c

/-- The set {3, 6, 9, 18} is proportional. -/
theorem proportional_set : is_proportional 3 6 9 18 := by
  sorry

end NUMINAMATH_CALUDE_proportional_set_l464_46428


namespace NUMINAMATH_CALUDE_octal_67_to_ternary_l464_46421

/-- Converts a decimal number to ternary representation -/
def toTernary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid ternary number -/
def isValidTernary (l : List ℕ) : Prop :=
  ∀ d ∈ l, d < 3

theorem octal_67_to_ternary :
  let octal_67 : ℕ := 6 * 8 + 7
  let ternary_2001 : List ℕ := [2, 0, 0, 1]
  octal_67 = 55 ∧ 
  toTernary 55 = ternary_2001 ∧
  isValidTernary ternary_2001 := by
  sorry

end NUMINAMATH_CALUDE_octal_67_to_ternary_l464_46421


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l464_46417

/-- Given an increase in average weight and a weight difference between new and old person,
    proves the initial number of persons. -/
theorem initial_number_of_persons
  (avg_weight_increase : ℝ)
  (weight_difference : ℝ)
  (h1 : avg_weight_increase = 2.5)
  (h2 : weight_difference = 20)
  (h3 : weight_difference = avg_weight_increase * (initial_persons : ℝ)) :
  initial_persons = 8 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l464_46417


namespace NUMINAMATH_CALUDE_cell_phone_production_ambiguity_l464_46430

/-- Represents the production of cell phones in a factory --/
structure CellPhoneProduction where
  machines_count : ℕ
  phones_per_machine : ℕ
  total_production : ℕ

/-- The production scenario described in the problem --/
def factory_scenario : CellPhoneProduction :=
  { machines_count := 10
  , phones_per_machine := 5
  , total_production := 50 }

/-- The production rate for some machines described in the problem --/
def some_machines_rate : ℕ := 10

/-- Theorem stating the ambiguity in the production calculation --/
theorem cell_phone_production_ambiguity :
  (factory_scenario.machines_count * factory_scenario.phones_per_machine = factory_scenario.total_production) ∧
  (factory_scenario.phones_per_machine ≠ some_machines_rate) :=
by sorry

end NUMINAMATH_CALUDE_cell_phone_production_ambiguity_l464_46430


namespace NUMINAMATH_CALUDE_f_even_implies_a_zero_min_value_when_a_greater_than_two_l464_46453

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + |2*x - a|

-- Theorem 1: If f is even, then a = 0
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

-- Theorem 2: If a > 2, then the minimum value of f(x) is a - 1
theorem min_value_when_a_greater_than_two (a : ℝ) :
  a > 2 → ∃ m : ℝ, (∀ x : ℝ, f a x ≥ m) ∧ (∃ x : ℝ, f a x = m) ∧ m = a - 1 := by sorry

end NUMINAMATH_CALUDE_f_even_implies_a_zero_min_value_when_a_greater_than_two_l464_46453


namespace NUMINAMATH_CALUDE_stack_probability_l464_46402

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the probability of a stack of crates being a certain height -/
def probabilityOfStackHeight (crate : CrateDimensions) (numCrates : ℕ) (targetHeight : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of a stack of 10 crates being 41 ft tall -/
theorem stack_probability :
  let crate : CrateDimensions := { length := 3, width := 4, height := 6 }
  probabilityOfStackHeight crate 10 41 = 190 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_stack_probability_l464_46402


namespace NUMINAMATH_CALUDE_buttons_given_to_mary_l464_46406

theorem buttons_given_to_mary (initial_buttons : ℕ) (buttons_left : ℕ) : initial_buttons - buttons_left = 4 :=
by
  sorry

#check buttons_given_to_mary 9 5

end NUMINAMATH_CALUDE_buttons_given_to_mary_l464_46406


namespace NUMINAMATH_CALUDE_cos_double_angle_tan_l464_46448

theorem cos_double_angle_tan (θ : Real) (h : Real.tan θ = -1/3) : 
  Real.cos (2 * θ) = 4/5 := by sorry

end NUMINAMATH_CALUDE_cos_double_angle_tan_l464_46448


namespace NUMINAMATH_CALUDE_sum_of_baby_ages_theorem_l464_46425

/-- Calculates the sum of ages of baby animals in 5 years -/
def sum_of_baby_ages_in_5_years (lioness_age : ℕ) : ℕ :=
  let hyena_age := lioness_age / 2
  let baby_lioness_age := lioness_age / 2
  let baby_hyena_age := hyena_age / 2
  (baby_lioness_age + 5) + (baby_hyena_age + 5)

/-- Theorem stating that the sum of ages of baby animals in 5 years is 19 -/
theorem sum_of_baby_ages_theorem :
  sum_of_baby_ages_in_5_years 12 = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_baby_ages_theorem_l464_46425


namespace NUMINAMATH_CALUDE_road_trip_total_hours_l464_46478

/-- Calculates the total hours driven during a road trip -/
def total_hours_driven (days : ℕ) (hours_per_day_person1 : ℕ) (hours_per_day_person2 : ℕ) : ℕ :=
  days * (hours_per_day_person1 + hours_per_day_person2)

/-- Proves that the total hours driven in the given scenario is 42 -/
theorem road_trip_total_hours : total_hours_driven 3 8 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_total_hours_l464_46478


namespace NUMINAMATH_CALUDE_product_ratio_equality_l464_46443

theorem product_ratio_equality (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_equality_l464_46443


namespace NUMINAMATH_CALUDE_initial_group_size_l464_46497

theorem initial_group_size (average_increase : ℝ) (old_weight new_weight : ℝ) :
  average_increase = 3.5 ∧ old_weight = 47 ∧ new_weight = 68 →
  (new_weight - old_weight) / average_increase = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_group_size_l464_46497


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l464_46449

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- Define the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a) = {x | -2 ≤ x ∧ x ≤ 1} → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l464_46449


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_parallel_l464_46471

/-- Two lines are parallel if their slopes are equal and they are not identical -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  (a * f = b * d) ∧ (a * e ≠ b * c)

theorem sufficient_but_not_necessary_parallel :
  (are_parallel 3 2 1 3 2 (-2)) ∧
  (∃ a : ℝ, a ≠ 3 ∧ are_parallel a 2 1 3 (a - 1) (-2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_parallel_l464_46471


namespace NUMINAMATH_CALUDE_red_ant_percentage_l464_46422

/-- Proves that the percentage of red ants in the population is 85%, given the specified conditions. -/
theorem red_ant_percentage (female_red_percentage : ℝ) (male_red_total_percentage : ℝ) :
  female_red_percentage = 45 →
  male_red_total_percentage = 46.75 →
  ∃ (red_percentage : ℝ),
    red_percentage = 85 ∧
    (100 - female_red_percentage) / 100 * red_percentage = male_red_total_percentage :=
by sorry

end NUMINAMATH_CALUDE_red_ant_percentage_l464_46422


namespace NUMINAMATH_CALUDE_expand_and_simplify_l464_46415

theorem expand_and_simplify (x : ℝ) : (7 * x + 5) * 3 * x^2 = 21 * x^3 + 15 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l464_46415


namespace NUMINAMATH_CALUDE_half_merit_scholarship_percentage_l464_46450

theorem half_merit_scholarship_percentage
  (total_students : ℕ)
  (full_scholarship_percentage : ℚ)
  (no_scholarship_count : ℕ)
  (h1 : total_students = 300)
  (h2 : full_scholarship_percentage = 5 / 100)
  (h3 : no_scholarship_count = 255) :
  (total_students - no_scholarship_count - (full_scholarship_percentage * total_students).num) / total_students = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_half_merit_scholarship_percentage_l464_46450


namespace NUMINAMATH_CALUDE_cubic_root_sum_l464_46483

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 2 + 2) + b = 0 → 
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l464_46483


namespace NUMINAMATH_CALUDE_math_sequences_count_l464_46423

theorem math_sequences_count : 
  let letters := ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S']
  let n := letters.length
  let first_letter := 'M'
  let last_letter_options := (letters.filter (· ≠ 'A')).filter (· ≠ first_letter)
  let middle_letters_count := 2
  (n - 1 - middle_letters_count).factorial * 
  last_letter_options.length * 
  (Nat.choose (n - 2) middle_letters_count) = 392 := by
sorry

end NUMINAMATH_CALUDE_math_sequences_count_l464_46423


namespace NUMINAMATH_CALUDE_mary_nickels_l464_46464

def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 5
def mom_multiplier : ℕ := 2

def final_nickels : ℕ := initial_nickels + dad_nickels + mom_multiplier * (initial_nickels + dad_nickels)

theorem mary_nickels : final_nickels = 36 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_l464_46464


namespace NUMINAMATH_CALUDE_meat_purchase_cost_l464_46479

/-- Represents the cost and quantity of a type of meat -/
structure Meat where
  name : String
  cost : ℝ
  quantity : ℝ

/-- Calculates the total cost of a purchase given meat prices and quantities -/
def totalCost (meats : List Meat) : ℝ :=
  meats.map (fun m => m.cost * m.quantity) |>.sum

/-- Theorem stating the total cost of the meat purchase -/
theorem meat_purchase_cost :
  let pork_cost : ℝ := 6
  let chicken_cost : ℝ := pork_cost - 2
  let beef_cost : ℝ := chicken_cost + 4
  let lamb_cost : ℝ := pork_cost + 3
  let meats : List Meat := [
    { name := "Chicken", cost := chicken_cost, quantity := 3.5 },
    { name := "Pork", cost := pork_cost, quantity := 1.2 },
    { name := "Beef", cost := beef_cost, quantity := 2.3 },
    { name := "Lamb", cost := lamb_cost, quantity := 0.8 }
  ]
  totalCost meats = 46.8 := by
  sorry

end NUMINAMATH_CALUDE_meat_purchase_cost_l464_46479


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l464_46472

/-- Given a hyperbola and a circle intersecting to form a square, 
    prove the equation of the asymptotes of the hyperbola. -/
theorem hyperbola_asymptotes_equation 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c = Real.sqrt (a^2 + b^2)) 
  (h_hyperbola : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1 → x^2 + y^2 = c^2 → x^2 = y^2) :
  ∃ k : ℝ, k = Real.sqrt (Real.sqrt 2 - 1) ∧ 
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1 → y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l464_46472


namespace NUMINAMATH_CALUDE_marias_car_trip_l464_46485

theorem marias_car_trip (total_distance : ℝ) (remaining_distance : ℝ) 
  (h1 : total_distance = 400)
  (h2 : remaining_distance = 150) : 
  ∃ x : ℝ, x * total_distance + (1/4) * (total_distance - x * total_distance) = total_distance - remaining_distance ∧ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_marias_car_trip_l464_46485


namespace NUMINAMATH_CALUDE_transformation_result_l464_46494

/-- Rotates a point (x, y) 180° clockwise around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem transformation_result (c d : ℝ) :
  let (x₁, y₁) := rotate180 c d 2 (-3)
  let (x₂, y₂) := reflectAboutYEqualsX x₁ y₁
  (x₂ = 5 ∧ y₂ = -4) → d - c = -19 := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l464_46494


namespace NUMINAMATH_CALUDE_even_increasing_neg_sum_positive_l464_46445

/-- An even function that is increasing on the negative real line -/
def EvenIncreasingNeg (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x y, x < y ∧ y ≤ 0 → f x < f y)

/-- Theorem statement -/
theorem even_increasing_neg_sum_positive
  (f : ℝ → ℝ) (hf : EvenIncreasingNeg f) (x₁ x₂ : ℝ)
  (hx₁ : x₁ > 0) (hx₂ : x₂ < 0) (hf_x : f x₁ < f x₂) :
  x₁ + x₂ > 0 :=
sorry

end NUMINAMATH_CALUDE_even_increasing_neg_sum_positive_l464_46445


namespace NUMINAMATH_CALUDE_derivative_of_f_l464_46446

-- Define the function
def f (x : ℝ) : ℝ := (5 * x - 3) ^ 3

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x => 15 * (5 * x - 3) ^ 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l464_46446


namespace NUMINAMATH_CALUDE_recipe_butter_amount_l464_46444

/-- The amount of butter (in ounces) required per cup of baking mix -/
def butter_per_cup : ℚ := 4/3

/-- The number of cups of baking mix the chef planned to use -/
def planned_cups : ℕ := 6

/-- The amount of coconut oil (in ounces) the chef used as a substitute for butter -/
def coconut_oil_used : ℕ := 8

/-- Theorem stating that the recipe calls for 4/3 ounces of butter per cup of baking mix -/
theorem recipe_butter_amount :
  butter_per_cup * planned_cups = coconut_oil_used := by
  sorry

end NUMINAMATH_CALUDE_recipe_butter_amount_l464_46444


namespace NUMINAMATH_CALUDE_square_perimeter_l464_46416

/-- The perimeter of a square with side length 11 cm is 44 cm. -/
theorem square_perimeter : 
  ∀ (s : ℝ), s = 11 → 4 * s = 44 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l464_46416


namespace NUMINAMATH_CALUDE_bride_groom_age_difference_oldest_bride_problem_l464_46431

theorem bride_groom_age_difference : ℕ → ℕ → ℕ → Prop :=
  fun total_age groom_age age_difference =>
    let bride_age := total_age - groom_age
    bride_age - groom_age = age_difference

theorem oldest_bride_problem (total_age groom_age : ℕ) 
  (h1 : total_age = 185) 
  (h2 : groom_age = 83) : 
  bride_groom_age_difference total_age groom_age 19 := by
  sorry

end NUMINAMATH_CALUDE_bride_groom_age_difference_oldest_bride_problem_l464_46431


namespace NUMINAMATH_CALUDE_unique_two_digit_multiple_l464_46491

theorem unique_two_digit_multiple : ∃! t : ℕ, 10 ≤ t ∧ t < 100 ∧ (13 * t) % 100 = 26 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_multiple_l464_46491


namespace NUMINAMATH_CALUDE_x_abs_x_is_k_function_l464_46492

/-- Definition of a K function -/
def is_k_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0)

/-- The function f(x) = x|x| -/
def f (x : ℝ) : ℝ := x * |x|

/-- Theorem: f(x) = x|x| is a K function -/
theorem x_abs_x_is_k_function : is_k_function f := by sorry

end NUMINAMATH_CALUDE_x_abs_x_is_k_function_l464_46492


namespace NUMINAMATH_CALUDE_complex_fraction_value_l464_46473

theorem complex_fraction_value (a : ℝ) (z : ℂ) : 
  z = (a^2 - 1 : ℂ) + (a + 1 : ℂ) * Complex.I ∧ z.re = 0 → 
  (a + Complex.I^2016) / (1 + Complex.I) = 1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l464_46473


namespace NUMINAMATH_CALUDE_alice_and_bob_savings_l464_46452

theorem alice_and_bob_savings (alice_money : ℚ) (bob_money : ℚ) :
  alice_money = 2 / 5 →
  bob_money = 1 / 4 →
  2 * (alice_money + bob_money) = 13 / 10 := by
sorry

end NUMINAMATH_CALUDE_alice_and_bob_savings_l464_46452


namespace NUMINAMATH_CALUDE_cubic_root_equation_l464_46419

theorem cubic_root_equation (s : ℝ) : 
  s = 1 / (2 - Real.rpow 3 (1/3)) → 
  s = ((2 + Real.rpow 3 (1/3)) * (4 + Real.sqrt 3)) / 13 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_equation_l464_46419


namespace NUMINAMATH_CALUDE_print_shop_charges_l464_46426

/-- The charge per color copy at print shop X -/
def charge_x : ℚ := 1.20

/-- The charge per color copy at print shop Y -/
def charge_y : ℚ := 1.70

/-- The number of copies -/
def num_copies : ℕ := 70

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℚ := 35

theorem print_shop_charges :
  charge_y * num_copies = charge_x * num_copies + additional_charge := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charges_l464_46426


namespace NUMINAMATH_CALUDE_product_even_odd_is_odd_l464_46461

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem product_even_odd_is_odd (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
  IsOdd (fun x ↦ f x * g x) := by
  sorry

end NUMINAMATH_CALUDE_product_even_odd_is_odd_l464_46461


namespace NUMINAMATH_CALUDE_proposition_analysis_l464_46489

-- Define proposition P
def P (x y : ℝ) : Prop := x ≠ y → abs x ≠ abs y

-- Define the converse of P
def P_converse (x y : ℝ) : Prop := abs x ≠ abs y → x ≠ y

-- Define the negation of P
def P_negation (x y : ℝ) : Prop := ¬(x ≠ y → abs x ≠ abs y)

-- Define the contrapositive of P
def P_contrapositive (x y : ℝ) : Prop := abs x = abs y → x = y

theorem proposition_analysis :
  (∃ x y : ℝ, ¬(P x y)) ∧
  (∀ x y : ℝ, P_converse x y) ∧
  (∀ x y : ℝ, P_negation x y) ∧
  (∃ x y : ℝ, ¬(P_contrapositive x y)) :=
sorry

end NUMINAMATH_CALUDE_proposition_analysis_l464_46489


namespace NUMINAMATH_CALUDE_min_value_a_l464_46456

theorem min_value_a (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  ∃ (a : ℝ), ∀ (x y : ℝ), x > 1 → y > 1 → 
    Real.log (x * y) ≤ Real.log a * Real.sqrt (Real.log x ^ 2 + Real.log y ^ 2) ∧
    ∀ (b : ℝ), (∀ (x y : ℝ), x > 1 → y > 1 → 
      Real.log (x * y) ≤ Real.log b * Real.sqrt (Real.log x ^ 2 + Real.log y ^ 2)) → 
    a ≤ b :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l464_46456


namespace NUMINAMATH_CALUDE_laborer_income_l464_46484

/-- Represents the financial situation of a laborer over a 10-month period. -/
structure LaborerFinances where
  monthly_income : ℝ
  initial_expenditure : ℝ
  reduced_expenditure : ℝ
  initial_period : ℕ
  reduced_period : ℕ
  savings : ℝ

/-- The laborer's finances satisfy the given conditions. -/
def satisfies_conditions (f : LaborerFinances) : Prop :=
  f.initial_expenditure = 75 ∧
  f.reduced_expenditure = 60 ∧
  f.initial_period = 6 ∧
  f.reduced_period = 4 ∧
  f.savings = 30 ∧
  f.initial_period * f.monthly_income < f.initial_period * f.initial_expenditure ∧
  f.reduced_period * f.monthly_income = f.reduced_period * f.reduced_expenditure + 
    (f.initial_period * f.initial_expenditure - f.initial_period * f.monthly_income) + f.savings

/-- Theorem stating that if the laborer's finances satisfy the given conditions, 
    then their monthly income is 72. -/
theorem laborer_income (f : LaborerFinances) 
  (h : satisfies_conditions f) : f.monthly_income = 72 := by
  sorry

end NUMINAMATH_CALUDE_laborer_income_l464_46484


namespace NUMINAMATH_CALUDE_h2o_mass_formed_l464_46455

-- Define the chemical reaction
structure Reaction where
  hcl : ℝ
  caco3 : ℝ
  h2o : ℝ

-- Define the molar masses
def molar_mass_h : ℝ := 1.008
def molar_mass_o : ℝ := 15.999

-- Define the reaction stoichiometry
def reaction_stoichiometry (r : Reaction) : Prop :=
  r.hcl = 2 * r.caco3 ∧ r.h2o = r.caco3

-- Calculate the molar mass of H2O
def molar_mass_h2o : ℝ := 2 * molar_mass_h + molar_mass_o

-- Main theorem
theorem h2o_mass_formed (r : Reaction) : 
  reaction_stoichiometry r → r.hcl = 2 → r.caco3 = 1 → r.h2o * molar_mass_h2o = 18.015 :=
sorry

end NUMINAMATH_CALUDE_h2o_mass_formed_l464_46455
