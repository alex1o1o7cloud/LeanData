import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_expression_l3310_331016

theorem max_value_of_expression (a b : ℝ) (h : a^2 + b^2 = 9) :
  ∃ (max : ℝ), max = 5 ∧ ∀ (x y : ℝ), x^2 + y^2 = 9 → x * y - y + x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3310_331016


namespace NUMINAMATH_CALUDE_hiking_rate_ratio_l3310_331062

/-- Prove the ratio of hiking rates for a mountain trip -/
theorem hiking_rate_ratio 
  (rate_up : ℝ) 
  (time_up : ℝ) 
  (distance_down : ℝ) 
  (rate_up_is_4 : rate_up = 4)
  (time_up_is_2 : time_up = 2)
  (distance_down_is_12 : distance_down = 12)
  (time_equal : time_up = distance_down / (distance_down / time_up * rate_up)) :
  distance_down / (time_up * rate_up) / rate_up = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_hiking_rate_ratio_l3310_331062


namespace NUMINAMATH_CALUDE_remainder_after_addition_l3310_331005

theorem remainder_after_addition : Int.mod (3452179 + 50) 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_addition_l3310_331005


namespace NUMINAMATH_CALUDE_complement_of_40_30_l3310_331038

/-- The complement of an angle is the difference between 90 degrees and the angle. -/
def complementOfAngle (angle : ℚ) : ℚ := 90 - angle

/-- Represents 40 degrees and 30 minutes in decimal degrees. -/
def angleA : ℚ := 40 + 30 / 60

theorem complement_of_40_30 :
  complementOfAngle angleA = 49 + 30 / 60 := by sorry

end NUMINAMATH_CALUDE_complement_of_40_30_l3310_331038


namespace NUMINAMATH_CALUDE_exitCell_l3310_331095

/-- Represents a cell on the 4x4 grid --/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the four possible directions --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- The state of the game at any point --/
structure GameState :=
  (position : Cell)
  (arrows : Cell → Direction)

/-- Applies a single move to the game state --/
def move (state : GameState) : GameState :=
  sorry

/-- Checks if a cell is on the boundary of the grid --/
def isBoundary (cell : Cell) : Bool :=
  sorry

/-- Plays the game until the piece exits the grid --/
def playUntilExit (initialState : GameState) : Cell :=
  sorry

theorem exitCell :
  let initialArrows : Cell → Direction := sorry
  let initialState : GameState := {
    position := ⟨2, 1⟩,  -- C2 in 0-based indexing
    arrows := initialArrows
  }
  playUntilExit initialState = ⟨0, 1⟩  -- A2 in 0-based indexing
:= by sorry

end NUMINAMATH_CALUDE_exitCell_l3310_331095


namespace NUMINAMATH_CALUDE_smallest_c_is_22_l3310_331044

/-- A polynomial with three positive integer roots -/
structure PolynomialWithThreeRoots where
  c : ℤ
  d : ℤ
  root1 : ℤ
  root2 : ℤ
  root3 : ℤ
  root1_pos : root1 > 0
  root2_pos : root2 > 0
  root3_pos : root3 > 0
  is_root1 : root1^3 - c*root1^2 + d*root1 - 2310 = 0
  is_root2 : root2^3 - c*root2^2 + d*root2 - 2310 = 0
  is_root3 : root3^3 - c*root3^2 + d*root3 - 2310 = 0

/-- The smallest possible value of c for a polynomial with three positive integer roots -/
def smallest_c : ℤ := 22

/-- Theorem stating that 22 is the smallest possible value of c -/
theorem smallest_c_is_22 (p : PolynomialWithThreeRoots) : p.c ≥ smallest_c := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_is_22_l3310_331044


namespace NUMINAMATH_CALUDE_average_after_removing_two_l3310_331037

def initial_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,12]

def remove_element (list : List ℕ) (elem : ℕ) : List ℕ :=
  list.filter (λ x => x ≠ elem)

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem average_after_removing_two :
  average (remove_element initial_list 2) = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_average_after_removing_two_l3310_331037


namespace NUMINAMATH_CALUDE_median_squares_sum_l3310_331001

/-- Given a triangle with sides a, b, c and corresponding medians s_a, s_b, s_c,
    the sum of the squares of the medians is equal to 3/4 times the sum of the squares of the sides. -/
theorem median_squares_sum (a b c s_a s_b s_c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_s_a : s_a^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_s_b : s_b^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (h_s_c : s_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  s_a^2 + s_b^2 + s_c^2 = 3/4 * (a^2 + b^2 + c^2) := by
  sorry


end NUMINAMATH_CALUDE_median_squares_sum_l3310_331001


namespace NUMINAMATH_CALUDE_buses_passed_count_l3310_331032

/-- Represents the frequency of bus departures in hours -/
structure BusSchedule where
  austin_to_san_antonio : ℕ
  san_antonio_to_austin : ℕ

/-- Represents the journey details -/
structure JourneyDetails where
  trip_duration : ℕ
  same_highway : Bool

/-- Calculates the number of buses passed during the journey -/
def buses_passed (schedule : BusSchedule) (journey : JourneyDetails) : ℕ :=
  sorry

theorem buses_passed_count 
  (schedule : BusSchedule)
  (journey : JourneyDetails)
  (h1 : schedule.austin_to_san_antonio = 2)
  (h2 : schedule.san_antonio_to_austin = 3)
  (h3 : journey.trip_duration = 8)
  (h4 : journey.same_highway = true) :
  buses_passed schedule journey = 4 :=
sorry

end NUMINAMATH_CALUDE_buses_passed_count_l3310_331032


namespace NUMINAMATH_CALUDE_equation_solutions_l3310_331092

theorem equation_solutions : 
  let f (x : ℝ) := 
    10 / (Real.sqrt (x - 10) - 10) + 
    2 / (Real.sqrt (x - 10) - 5) + 
    14 / (Real.sqrt (x - 10) + 5) + 
    20 / (Real.sqrt (x - 10) + 10)
  ∀ x : ℝ, f x = 0 ↔ (x = 190 / 9 ∨ x = 5060 / 256) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3310_331092


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3310_331035

theorem consecutive_integers_product_sum (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 930 → x + (x + 1) = 61 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3310_331035


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3310_331020

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: Given an arithmetic sequence with S_15 = 30 and a_7 = 1, then S_9 = -9 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h1 : seq.S 15 = 30)
  (h2 : seq.a 7 = 1) :
  seq.S 9 = -9 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3310_331020


namespace NUMINAMATH_CALUDE_book_purchase_theorem_l3310_331080

theorem book_purchase_theorem (total_A total_B both only_B : ℕ) 
  (h1 : total_A = 2 * total_B)
  (h2 : both = 500)
  (h3 : both = 2 * only_B) :
  total_A - both = 1000 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_theorem_l3310_331080


namespace NUMINAMATH_CALUDE_find_m_l3310_331031

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}

theorem find_m : ∃ m : ℝ, (U \ A m = {1, 2}) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3310_331031


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3310_331053

theorem polar_to_rectangular_conversion :
  let r : ℝ := 6
  let θ : ℝ := 5 * π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -3 * Real.sqrt 2) ∧ (y = -3 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3310_331053


namespace NUMINAMATH_CALUDE_bills_omelet_preparation_time_l3310_331056

/-- Represents the time in minutes for various tasks in omelet preparation --/
structure OmeletPreparationTime where
  chop_pepper : ℕ
  chop_onion : ℕ
  grate_cheese : ℕ
  assemble_and_cook : ℕ

/-- Represents the quantities of ingredients and omelets --/
structure OmeletQuantities where
  peppers : ℕ
  onions : ℕ
  omelets : ℕ

/-- Calculates the total time for omelet preparation given preparation times and quantities --/
def total_preparation_time (prep_time : OmeletPreparationTime) (quantities : OmeletQuantities) : ℕ :=
  prep_time.chop_pepper * quantities.peppers +
  prep_time.chop_onion * quantities.onions +
  prep_time.grate_cheese * quantities.omelets +
  prep_time.assemble_and_cook * quantities.omelets

/-- Theorem stating that Bill's total preparation time for five omelets is 50 minutes --/
theorem bills_omelet_preparation_time :
  let prep_time : OmeletPreparationTime := {
    chop_pepper := 3,
    chop_onion := 4,
    grate_cheese := 1,
    assemble_and_cook := 5
  }
  let quantities : OmeletQuantities := {
    peppers := 4,
    onions := 2,
    omelets := 5
  }
  total_preparation_time prep_time quantities = 50 := by
  sorry

end NUMINAMATH_CALUDE_bills_omelet_preparation_time_l3310_331056


namespace NUMINAMATH_CALUDE_sports_club_overlap_l3310_331082

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h1 : total = 28)
  (h2 : badminton = 17)
  (h3 : tennis = 19)
  (h4 : neither = 2)
  : badminton + tennis - (total - neither) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l3310_331082


namespace NUMINAMATH_CALUDE_skips_per_meter_l3310_331011

theorem skips_per_meter
  (p q r s t u : ℝ)
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (skip_jump : p * q⁻¹ = 1)
  (jump_foot : r * s⁻¹ = 1)
  (foot_meter : t * u⁻¹ = 1) :
  (t * r * p) * (u * s * q)⁻¹ = 1 := by
sorry

end NUMINAMATH_CALUDE_skips_per_meter_l3310_331011


namespace NUMINAMATH_CALUDE_min_value_abc_l3310_331063

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 27) :
  54 ≤ 3 * a + 6 * b + 9 * c ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ 3 * a₀ + 6 * b₀ + 9 * c₀ = 54 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l3310_331063


namespace NUMINAMATH_CALUDE_average_price_of_rackets_l3310_331064

/-- The average price of a pair of rackets given total sales and number of pairs sold -/
theorem average_price_of_rackets (total_sales : ℝ) (num_pairs : ℕ) (h1 : total_sales = 490) (h2 : num_pairs = 50) :
  total_sales / num_pairs = 9.80 := by
  sorry

end NUMINAMATH_CALUDE_average_price_of_rackets_l3310_331064


namespace NUMINAMATH_CALUDE_inheritance_sum_l3310_331066

theorem inheritance_sum (n : ℕ) : n = 36 → (n * (n + 1)) / 2 = 666 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_sum_l3310_331066


namespace NUMINAMATH_CALUDE_sheets_per_day_l3310_331029

def sheets_per_pad : ℕ := 60
def working_days_per_week : ℕ := 5

theorem sheets_per_day :
  sheets_per_pad / working_days_per_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_day_l3310_331029


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_complex_fraction_l3310_331014

theorem sum_real_imag_parts_complex_fraction : ∃ (z : ℂ), 
  z = (3 - 3 * Complex.I) / (1 - Complex.I) ∧ 
  z.re + z.im = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_complex_fraction_l3310_331014


namespace NUMINAMATH_CALUDE_sqrt_point_three_six_equals_point_six_l3310_331028

theorem sqrt_point_three_six_equals_point_six : Real.sqrt 0.36 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_point_three_six_equals_point_six_l3310_331028


namespace NUMINAMATH_CALUDE_range_of_a_l3310_331047

/-- Given a real number a, we define the following propositions: -/
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0

def q (x a : ℝ) : Prop := x > a

/-- The main theorem stating the range of values for a -/
theorem range_of_a (a : ℝ) :
  (∀ x, ¬(p x) → ¬(q x a)) →  -- Sufficient condition
  ¬(∀ x, ¬(q x a) → ¬(p x)) →  -- Not necessary condition
  a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3310_331047


namespace NUMINAMATH_CALUDE_work_completion_time_l3310_331002

theorem work_completion_time 
  (b_time : ℝ) 
  (ab_time : ℝ) 
  (h1 : b_time = 6) 
  (h2 : ab_time = 10/3) : 
  ∃ a_time : ℝ, a_time = 15/2 ∧ 1/a_time + 1/b_time = 1/ab_time :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3310_331002


namespace NUMINAMATH_CALUDE_train_passing_time_l3310_331076

/-- Proves that a train of given length and speed takes a specific time to pass a stationary point -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 280 →
  train_speed_kmh = 63 →
  passing_time = 16 →
  train_length / (train_speed_kmh * 1000 / 3600) = passing_time := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l3310_331076


namespace NUMINAMATH_CALUDE_carls_garden_area_l3310_331058

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : ℕ
  post_distance : ℕ
  longer_side_posts : ℕ
  shorter_side_posts : ℕ

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℕ :=
  (g.shorter_side_posts - 1) * (g.longer_side_posts - 1) * g.post_distance * g.post_distance

/-- Theorem stating the area of Carl's garden -/
theorem carls_garden_area :
  ∀ g : Garden,
    g.total_posts = 36 ∧
    g.post_distance = 6 ∧
    g.longer_side_posts = 3 * g.shorter_side_posts ∧
    g.total_posts = 2 * (g.longer_side_posts + g.shorter_side_posts - 2) →
    garden_area g = 2016 := by
  sorry

end NUMINAMATH_CALUDE_carls_garden_area_l3310_331058


namespace NUMINAMATH_CALUDE_luthers_line_has_17_pieces_l3310_331021

/-- Represents Luther's latest clothing line -/
structure ClothingLine where
  silk_pieces : ℕ
  cashmere_pieces : ℕ
  blended_pieces : ℕ

/-- Calculates the total number of pieces in the clothing line -/
def total_pieces (line : ClothingLine) : ℕ :=
  line.silk_pieces + line.cashmere_pieces + line.blended_pieces

/-- Theorem: Luther's latest line has 17 pieces -/
theorem luthers_line_has_17_pieces :
  ∃ (line : ClothingLine),
    line.silk_pieces = 10 ∧
    line.cashmere_pieces = line.silk_pieces / 2 ∧
    line.blended_pieces = 2 ∧
    total_pieces line = 17 := by
  sorry

end NUMINAMATH_CALUDE_luthers_line_has_17_pieces_l3310_331021


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3310_331017

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (h1 : ‖a - b‖ = Real.sqrt 3)
  (h2 : ‖a + b‖ = ‖2 • a - b‖) :
  ‖b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3310_331017


namespace NUMINAMATH_CALUDE_gcd_lcm_consecutive_naturals_l3310_331010

theorem gcd_lcm_consecutive_naturals (m : ℕ) (h : m > 0) :
  let n := m + 1
  (Nat.gcd m n = 1) ∧ (Nat.lcm m n = m * n) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_consecutive_naturals_l3310_331010


namespace NUMINAMATH_CALUDE_carolyn_piano_practice_time_l3310_331000

/-- Given Carolyn's practice schedule, prove she practices piano for 20 minutes daily. -/
theorem carolyn_piano_practice_time :
  ∀ (piano_time : ℕ),
    (∃ (violin_time : ℕ), violin_time = 3 * piano_time) →
    (∃ (weekly_practice : ℕ), weekly_practice = 6 * (piano_time + 3 * piano_time)) →
    (∃ (monthly_practice : ℕ), monthly_practice = 4 * 6 * (piano_time + 3 * piano_time)) →
    4 * 6 * (piano_time + 3 * piano_time) = 1920 →
    piano_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_piano_practice_time_l3310_331000


namespace NUMINAMATH_CALUDE_problem_statement_l3310_331008

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b)^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3310_331008


namespace NUMINAMATH_CALUDE_compute_expression_l3310_331004

theorem compute_expression : 9 * (2 / 3) ^ 4 = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3310_331004


namespace NUMINAMATH_CALUDE_arith_progression_poly_j_value_l3310_331026

/-- A polynomial of degree 4 with four distinct real roots in arithmetic progression -/
structure ArithProgressionPoly where
  j : ℝ
  k : ℝ
  roots : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  arithmetic_progression : ∃ (a d : ℝ), ∀ i, roots i = a + i * d
  is_root : ∀ i, (roots i)^4 + j * (roots i)^2 + k * (roots i) + 400 = 0

/-- The value of j in an ArithProgressionPoly is -40 -/
theorem arith_progression_poly_j_value (p : ArithProgressionPoly) : p.j = -40 := by
  sorry

end NUMINAMATH_CALUDE_arith_progression_poly_j_value_l3310_331026


namespace NUMINAMATH_CALUDE_special_item_identification_l3310_331072

/-- Represents the result of a yes/no question -/
inductive Answer
| Yes
| No

/-- Converts an Answer to a natural number (0 for Yes, 1 for No) -/
def answerToNat (a : Answer) : Nat :=
  match a with
  | Answer.Yes => 0
  | Answer.No => 1

/-- Represents the set of items -/
def Items : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7}

/-- The function to determine the special item based on three answers -/
def determineSpecialItem (a₁ a₂ a₃ : Answer) : Nat :=
  answerToNat a₁ + 2 * answerToNat a₂ + 4 * answerToNat a₃

theorem special_item_identification :
  ∀ (special : Nat),
  special ∈ Items →
  ∃ (a₁ a₂ a₃ : Answer),
  determineSpecialItem a₁ a₂ a₃ = special ∧
  ∀ (other : Nat),
  other ∈ Items →
  other ≠ special →
  determineSpecialItem a₁ a₂ a₃ ≠ other :=
sorry

end NUMINAMATH_CALUDE_special_item_identification_l3310_331072


namespace NUMINAMATH_CALUDE_next_coincidence_after_lcm_robinsons_next_busy_day_l3310_331041

/-- Represents a periodic event --/
structure PeriodicEvent where
  period : ℕ

/-- Calculates the least common multiple (LCM) of a list of natural numbers --/
def lcmList (list : List ℕ) : ℕ :=
  list.foldl Nat.lcm 1

/-- Theorem: The next coincidence of periodic events occurs after their LCM --/
theorem next_coincidence_after_lcm (events : List PeriodicEvent) : 
  let periods := events.map (·.period)
  let nextCoincidence := lcmList periods
  ∀ t : ℕ, t < nextCoincidence → ¬ (∀ e ∈ events, t % e.period = 0) :=
by sorry

/-- Robinson Crusoe's activities --/
def robinsons_activities : List PeriodicEvent := [
  { period := 2 },  -- Water replenishment
  { period := 3 },  -- Fruit collection
  { period := 5 }   -- Hunting
]

/-- Theorem: Robinson's next busy day is 30 days after the current busy day --/
theorem robinsons_next_busy_day :
  lcmList (robinsons_activities.map (·.period)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_next_coincidence_after_lcm_robinsons_next_busy_day_l3310_331041


namespace NUMINAMATH_CALUDE_total_amount_proof_l3310_331089

/-- Given that r has two-thirds of the total amount and r has Rs. 2400,
    prove that the total amount p, q, and r have among themselves is Rs. 3600. -/
theorem total_amount_proof (r p q : ℕ) (h1 : r = 2400) (h2 : r * 3 = (p + q + r) * 2) :
  p + q + r = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l3310_331089


namespace NUMINAMATH_CALUDE_lowest_sample_number_48_8_48_l3310_331006

/-- Calculates the lowest number in a systematic sample. -/
def lowestSampleNumber (totalPopulation : ℕ) (sampleSize : ℕ) (highestNumber : ℕ) : ℕ :=
  highestNumber - (totalPopulation / sampleSize) * (sampleSize - 1)

/-- Theorem: In a systematic sampling of 8 students from 48, with highest number 48, the lowest is 6. -/
theorem lowest_sample_number_48_8_48 :
  lowestSampleNumber 48 8 48 = 6 := by
  sorry

#eval lowestSampleNumber 48 8 48

end NUMINAMATH_CALUDE_lowest_sample_number_48_8_48_l3310_331006


namespace NUMINAMATH_CALUDE_triangle_properties_l3310_331093

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (3 * t.c - t.b) * Real.cos t.A ∧
  t.a = 2 * Real.sqrt 2 ∧
  (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = (2 * Real.sqrt 2) / 3 ∧ t.b + t.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3310_331093


namespace NUMINAMATH_CALUDE_markeesha_cracker_sales_l3310_331094

theorem markeesha_cracker_sales :
  ∀ (friday saturday sunday : ℕ),
    friday = 30 →
    saturday = 2 * friday →
    friday + saturday + sunday = 135 →
    saturday - sunday = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_markeesha_cracker_sales_l3310_331094


namespace NUMINAMATH_CALUDE_SR_equals_15_l3310_331018

/-- Triangle PQR with point S on PR --/
structure TrianglePQRWithS where
  /-- Length of PQ --/
  PQ : ℝ
  /-- Length of QR --/
  QR : ℝ
  /-- Length of PS --/
  PS : ℝ
  /-- Length of QS --/
  QS : ℝ
  /-- PQ equals QR --/
  eq_PQ_QR : PQ = QR
  /-- PQ equals 10 --/
  eq_PQ_10 : PQ = 10
  /-- PS equals 6 --/
  eq_PS_6 : PS = 6
  /-- QS equals 5 --/
  eq_QS_5 : QS = 5

/-- The length of SR in the given triangle configuration --/
def SR (t : TrianglePQRWithS) : ℝ := 15

/-- Theorem: The length of SR is 15 in the given triangle configuration --/
theorem SR_equals_15 (t : TrianglePQRWithS) : SR t = 15 := by
  sorry

end NUMINAMATH_CALUDE_SR_equals_15_l3310_331018


namespace NUMINAMATH_CALUDE_factorization_of_3x2_minus_12y2_l3310_331091

theorem factorization_of_3x2_minus_12y2 (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_3x2_minus_12y2_l3310_331091


namespace NUMINAMATH_CALUDE_tree_table_profit_l3310_331039

/-- Calculates the profit from selling tables made from chopped trees --/
theorem tree_table_profit
  (trees : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ)
  (h1 : trees = 30)
  (h2 : planks_per_tree = 25)
  (h3 : planks_per_table = 15)
  (h4 : price_per_table = 300)
  (h5 : labor_cost = 3000)
  : (trees * planks_per_tree / planks_per_table) * price_per_table - labor_cost = 12000 := by
  sorry

#check tree_table_profit

end NUMINAMATH_CALUDE_tree_table_profit_l3310_331039


namespace NUMINAMATH_CALUDE_g_g_eq_5_has_two_solutions_l3310_331012

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 6

theorem g_g_eq_5_has_two_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g (g x₁) = 5 ∧ g (g x₂) = 5 ∧
  ∀ (x : ℝ), g (g x) = 5 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_g_g_eq_5_has_two_solutions_l3310_331012


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l3310_331085

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningScore : Nat

/-- Calculates the new average of a batsman after an additional inning -/
def newAverage (performance : BatsmanPerformance) : Nat :=
  (performance.totalRuns + performance.lastInningScore) / (performance.innings + 1)

/-- Theorem: If a batsman's average increases by 5 after scoring 85 in the 11th inning,
    then his new average is 35 -/
theorem batsman_average_theorem (performance : BatsmanPerformance) 
  (h1 : performance.innings = 10)
  (h2 : performance.lastInningScore = 85)
  (h3 : performance.averageIncrease = 5) :
  newAverage performance = 35 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l3310_331085


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3310_331098

/-- The quadratic equation x^2 + 6mx + m has exactly one real root if and only if m = 1/9 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x, x^2 + 6*m*x + m = 0) ↔ m = 1/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3310_331098


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3310_331022

/-- Given two vectors on a plane with specific properties, prove the magnitude of a third vector. -/
theorem vector_magnitude_problem (a b m : ℝ × ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = -1/2) →  -- angle between a and b is 120°
  (a.1^2 + a.2^2 = 1) →             -- magnitude of a is 1
  (b.1^2 + b.2^2 = 4) →             -- magnitude of b is 2
  (m.1 * a.1 + m.2 * a.2 = 1) →     -- m · a = 1
  (m.1 * b.1 + m.2 * b.2 = 1) →     -- m · b = 1
  m.1^2 + m.2^2 = 7/3 :=            -- |m|^2 = (√21/3)^2 = 21/9 = 7/3
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3310_331022


namespace NUMINAMATH_CALUDE_pyramid_edges_l3310_331088

/-- Represents a pyramid with a polygonal base -/
structure Pyramid where
  base_sides : ℕ

/-- The number of vertices in a pyramid -/
def num_vertices (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of faces in a pyramid -/
def num_faces (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of edges in a pyramid -/
def num_edges (p : Pyramid) : ℕ := p.base_sides + p.base_sides

theorem pyramid_edges (p : Pyramid) :
  num_vertices p + num_faces p = 16 → num_edges p = 14 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edges_l3310_331088


namespace NUMINAMATH_CALUDE_initial_water_amount_l3310_331065

/-- Proves that the initial amount of water in the tank was 100 L given the conditions of the rainstorm. -/
theorem initial_water_amount (flow_rate : ℝ) (duration : ℝ) (total_after : ℝ) : 
  flow_rate = 2 → duration = 90 → total_after = 280 → 
  total_after - (flow_rate * duration) = 100 := by
sorry

end NUMINAMATH_CALUDE_initial_water_amount_l3310_331065


namespace NUMINAMATH_CALUDE_max_profit_at_100_l3310_331015

-- Define the cost function
def C (x : ℕ) : ℚ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

-- Define the profit function
def L (x : ℕ) : ℚ :=
  if x < 80 then -(1/3) * x^2 + 40 * x - 250
  else 1200 - (x + 10000 / x)

-- Theorem statement
theorem max_profit_at_100 :
  ∀ x : ℕ, x > 0 → L x ≤ 1000 ∧ L 100 = 1000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_100_l3310_331015


namespace NUMINAMATH_CALUDE_square_extension_theorem_l3310_331087

/-- A configuration of points derived from a unit square. -/
structure SquareExtension where
  /-- The unit square ABCD -/
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  /-- Extension point E on AB extended -/
  E : ℝ × ℝ
  /-- Extension point F on DA extended -/
  F : ℝ × ℝ
  /-- Point G on ray FC such that FG = FE -/
  G : ℝ × ℝ
  /-- Point H on ray FC such that FH = 1 -/
  H : ℝ × ℝ
  /-- Intersection of FE and line through G parallel to CE -/
  J : ℝ × ℝ
  /-- Intersection of FE and line through H parallel to CJ -/
  K : ℝ × ℝ

  /-- ABCD forms a unit square -/
  h_unit_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)
  /-- BE = 1 -/
  h_BE : E = (2, 0)
  /-- AF = 5/9 -/
  h_AF : F = (0, 5/9)
  /-- FG = FE -/
  h_FG_eq_FE : dist F G = dist F E
  /-- FH = 1 -/
  h_FH : dist F H = 1
  /-- G is on ray FC -/
  h_G_on_FC : ∃ t : ℝ, t > 0 ∧ G = F + t • (C - F)
  /-- H is on ray FC -/
  h_H_on_FC : ∃ t : ℝ, t > 0 ∧ H = F + t • (C - F)
  /-- Line through G is parallel to CE -/
  h_G_parallel_CE : (G.2 - J.2) / (G.1 - J.1) = (C.2 - E.2) / (C.1 - E.1)
  /-- Line through H is parallel to CJ -/
  h_H_parallel_CJ : (H.2 - K.2) / (H.1 - K.1) = (C.2 - J.2) / (C.1 - J.1)

/-- The main theorem stating that FK = 349/97 in the given configuration. -/
theorem square_extension_theorem (se : SquareExtension) : dist se.F se.K = 349/97 := by
  sorry

end NUMINAMATH_CALUDE_square_extension_theorem_l3310_331087


namespace NUMINAMATH_CALUDE_x_squared_minus_x_greater_cube_sum_greater_l3310_331097

-- Part 1
theorem x_squared_minus_x_greater (x : ℝ) : x^2 - x > x - 2 := by sorry

-- Part 2
theorem cube_sum_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by sorry

end NUMINAMATH_CALUDE_x_squared_minus_x_greater_cube_sum_greater_l3310_331097


namespace NUMINAMATH_CALUDE_journey_distance_ratio_l3310_331069

/-- Proves that the ratio of North distance to East distance is 2:1 given the problem conditions --/
theorem journey_distance_ratio :
  let south_distance : ℕ := 40
  let east_distance : ℕ := south_distance + 20
  let total_distance : ℕ := 220
  let north_distance : ℕ := total_distance - south_distance - east_distance
  (north_distance : ℚ) / east_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_ratio_l3310_331069


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3310_331052

/-- The lateral surface area of a cylinder with given base circumference and height -/
def lateral_surface_area (base_circumference : ℝ) (height : ℝ) : ℝ :=
  base_circumference * height

/-- Theorem: The lateral surface area of a cylinder with base circumference 5cm and height 2cm is 10 cm² -/
theorem cylinder_lateral_surface_area :
  lateral_surface_area 5 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3310_331052


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_three_l3310_331054

/-- The probability that at least one of three independent events occurs -/
theorem prob_at_least_one_of_three (p₁ p₂ p₃ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) 
  (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) 
  (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1) : 
  1 - (1 - p₁) * (1 - p₂) * (1 - p₃) = 
  1 - ((1 - p₁) * (1 - p₂) * (1 - p₃)) :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_three_l3310_331054


namespace NUMINAMATH_CALUDE_chef_potatoes_per_week_l3310_331048

/-- Calculates the total number of potatoes used by a chef in one week -/
def total_potatoes_per_week (lunch_potatoes : ℕ) (work_days : ℕ) : ℕ :=
  let dinner_potatoes := 2 * lunch_potatoes
  let lunch_total := lunch_potatoes * work_days
  let dinner_total := dinner_potatoes * work_days
  lunch_total + dinner_total

/-- Proves that the chef uses 90 potatoes in one week -/
theorem chef_potatoes_per_week :
  total_potatoes_per_week 5 6 = 90 :=
by
  sorry

#eval total_potatoes_per_week 5 6

end NUMINAMATH_CALUDE_chef_potatoes_per_week_l3310_331048


namespace NUMINAMATH_CALUDE_sequence_A_l3310_331049

theorem sequence_A (a : ℕ → ℕ) : 
  (a 1 = 2) → 
  (∀ n : ℕ, a (n + 1) = a n + n + 1) → 
  (a 20 = 211) :=
by sorry


end NUMINAMATH_CALUDE_sequence_A_l3310_331049


namespace NUMINAMATH_CALUDE_smallest_reducible_fraction_l3310_331030

theorem smallest_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (n - 17 : ℤ) ≠ 0 ∧
  (7 * n + 2 : ℤ) ≠ 0 ∧
  (∃ (k : ℤ), k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (7 * n + 2)) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    (m - 17 : ℤ) = 0 ∨
    (7 * m + 2 : ℤ) = 0 ∨
    (∀ (k : ℤ), k > 1 → ¬(k ∣ (m - 17) ∧ k ∣ (7 * m + 2)))) ∧
  n = 28 :=
by sorry

end NUMINAMATH_CALUDE_smallest_reducible_fraction_l3310_331030


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3310_331027

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a2_eq_9 : a 2 = 9
  a5_eq_33 : a 5 = 33

/-- The common difference of an arithmetic sequence is 8 given the conditions -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence) :
  ∃ d : ℝ, (∀ n, seq.a (n + 1) - seq.a n = d) ∧ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3310_331027


namespace NUMINAMATH_CALUDE_bryden_payment_proof_l3310_331013

/-- The amount a collector pays for state quarters as a percentage of face value -/
def collector_payment_percentage : ℝ := 1500

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 7

/-- The face value of a single state quarter in dollars -/
def quarter_face_value : ℝ := 0.25

/-- The amount Bryden receives for his quarters in dollars -/
def bryden_payment : ℝ := 26.25

theorem bryden_payment_proof :
  (collector_payment_percentage / 100) * (bryden_quarters * quarter_face_value) = bryden_payment :=
by sorry

end NUMINAMATH_CALUDE_bryden_payment_proof_l3310_331013


namespace NUMINAMATH_CALUDE_stick_cutting_l3310_331033

theorem stick_cutting (n : ℕ) : (1 : ℝ) / 2^n = 1 / 64 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_stick_cutting_l3310_331033


namespace NUMINAMATH_CALUDE_solve_for_y_l3310_331078

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 4*x - 1 = y - 2) (h2 : x = -3) : y = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3310_331078


namespace NUMINAMATH_CALUDE_job_completion_time_l3310_331077

/-- Given workers p and q, where p can complete a job in 4 days and q's daily work rate
    is one-third of p's, prove that p and q working together can complete the job in 3 days. -/
theorem job_completion_time (p q : ℝ) 
    (hp : p = 1 / 4)  -- p's daily work rate
    (hq : q = 1 / 3 * p) : -- q's daily work rate relative to p
    1 / (p + q) = 3 := by
  sorry


end NUMINAMATH_CALUDE_job_completion_time_l3310_331077


namespace NUMINAMATH_CALUDE_max_intersections_circle_line_parabola_l3310_331024

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in a 2D plane --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum number of intersections between a circle and a line --/
def max_intersections_circle_line : ℕ := 2

/-- The maximum number of intersections between a parabola and a line --/
def max_intersections_parabola_line : ℕ := 2

/-- The maximum number of intersections between a circle and a parabola --/
def max_intersections_circle_parabola : ℕ := 4

/-- Theorem: The maximum number of intersections between a circle, a line, and a parabola is 8 --/
theorem max_intersections_circle_line_parabola 
  (c : Circle) (l : Line) (p : Parabola) : 
  max_intersections_circle_line + 
  max_intersections_parabola_line + 
  max_intersections_circle_parabola = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_line_parabola_l3310_331024


namespace NUMINAMATH_CALUDE_binomial_150_150_l3310_331003

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l3310_331003


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l3310_331096

/-- The perimeter of a regular hexagon with radius √3 is 6√3 -/
theorem regular_hexagon_perimeter (r : ℝ) (h : r = Real.sqrt 3) : 
  6 * r = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l3310_331096


namespace NUMINAMATH_CALUDE_koschei_coins_l3310_331046

theorem koschei_coins : ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 10 = 7 ∧ n % 12 = 9 ∧ n = 357 := by
  sorry

end NUMINAMATH_CALUDE_koschei_coins_l3310_331046


namespace NUMINAMATH_CALUDE_fifteen_multiple_and_divisor_of_itself_l3310_331075

theorem fifteen_multiple_and_divisor_of_itself : 
  ∃ n : ℕ, n % 15 = 0 ∧ 15 % n = 0 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_multiple_and_divisor_of_itself_l3310_331075


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l3310_331067

theorem x_squared_plus_y_squared_equals_four 
  (h : (x^2 + y^2 + 1) * (x^2 + y^2 - 3) = 5) : 
  x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l3310_331067


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l3310_331068

/-- Calculates the total number of grandchildren for a grandmother with the given family structure -/
def total_grandchildren (num_daughters num_sons sons_per_daughter daughters_per_son : ℕ) : ℕ :=
  (num_daughters * sons_per_daughter) + (num_sons * daughters_per_son)

/-- Theorem: Grandma Olga's total number of grandchildren is 33 -/
theorem grandma_olga_grandchildren :
  total_grandchildren 3 3 6 5 = 33 := by
  sorry

#eval total_grandchildren 3 3 6 5

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l3310_331068


namespace NUMINAMATH_CALUDE_emily_marbles_l3310_331007

theorem emily_marbles (E : ℕ) : 
  (3 * E - (3 * E / 2 + 1) = 8) → E = 6 := by
  sorry

end NUMINAMATH_CALUDE_emily_marbles_l3310_331007


namespace NUMINAMATH_CALUDE_polynomial_factor_d_value_l3310_331061

theorem polynomial_factor_d_value :
  ∀ d : ℚ,
  (∀ x : ℚ, (3 * x + 4 = 0) → (5 * x^3 + 17 * x^2 + d * x + 28 = 0)) →
  d = 233 / 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_d_value_l3310_331061


namespace NUMINAMATH_CALUDE_problem_solution_l3310_331036

theorem problem_solution (a b : ℕ+) : 
  Nat.lcm a b = 2520 → 
  Nat.gcd a b = 24 → 
  a = 240 → 
  b = 252 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3310_331036


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3310_331051

/-- The number of ways to arrange books of different languages on a shelf. -/
def arrange_books (arabic : ℕ) (german : ℕ) (spanish : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial arabic) * (Nat.factorial german) * (Nat.factorial spanish)

/-- Theorem: The number of ways to arrange 10 books (2 Arabic, 4 German, 4 Spanish) on a shelf,
    keeping books of the same language together, is equal to 6912. -/
theorem book_arrangement_theorem :
  arrange_books 2 4 4 = 6912 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3310_331051


namespace NUMINAMATH_CALUDE_divisibility_conditions_l3310_331034

theorem divisibility_conditions :
  (∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1) → n = 1) ∧
  (∀ n : ℕ, n ≥ 1 → Odd n → (n ∣ 3^n + 1) → n = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l3310_331034


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3310_331090

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (2*m + 1, -1/2)
  let b : ℝ × ℝ := (2*m, 1)
  are_parallel a b → m = -1/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3310_331090


namespace NUMINAMATH_CALUDE_union_equality_condition_min_value_of_expression_min_value_achieved_l3310_331071

-- Option B
theorem union_equality_condition (A B : Set α) :
  (A ∪ B = B) ↔ (A ∩ B = A) := by sorry

-- Option D
theorem min_value_of_expression {x y : ℝ} (hx : x > 1) (hy : y > 1) (hxy : x + y = x * y) :
  (2 * x / (x - 1) + 4 * y / (y - 1)) ≥ 6 + 4 * Real.sqrt 2 := by sorry

-- Theorem stating that the minimum value is achieved
theorem min_value_achieved {x y : ℝ} (hx : x > 1) (hy : y > 1) (hxy : x + y = x * y) :
  ∃ (x₀ y₀ : ℝ), x₀ > 1 ∧ y₀ > 1 ∧ x₀ + y₀ = x₀ * y₀ ∧
    (2 * x₀ / (x₀ - 1) + 4 * y₀ / (y₀ - 1)) = 6 + 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_union_equality_condition_min_value_of_expression_min_value_achieved_l3310_331071


namespace NUMINAMATH_CALUDE_greatest_possible_award_l3310_331083

theorem greatest_possible_award (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ) 
  (h1 : total_prize = 400)
  (h2 : num_winners = 20)
  (h3 : min_award = 20)
  (h4 : 2 * (total_prize / 5) = 3 * (num_winners / 5) * min_award) : 
  ∃ (max_award : ℕ), max_award = 100 ∧ 
  (∀ (award : ℕ), award ≤ max_award ∧ 
    (∃ (distribution : List ℕ), 
      distribution.length = num_winners ∧
      (∀ x ∈ distribution, min_award ≤ x) ∧
      distribution.sum = total_prize ∧
      award ∈ distribution)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_award_l3310_331083


namespace NUMINAMATH_CALUDE_spherical_coordinates_of_point_A_l3310_331050

theorem spherical_coordinates_of_point_A :
  let x : ℝ := (3 * Real.sqrt 3) / 2
  let y : ℝ := 9 / 2
  let z : ℝ := 3
  let r : ℝ := Real.sqrt (x^2 + y^2 + z^2)
  let θ : ℝ := Real.arctan (y / x)
  let φ : ℝ := Real.arccos (z / r)
  (r = 6) ∧ (θ = π / 3) ∧ (φ = π / 3) := by sorry

end NUMINAMATH_CALUDE_spherical_coordinates_of_point_A_l3310_331050


namespace NUMINAMATH_CALUDE_pokemon_card_solution_l3310_331079

def pokemon_card_problem (initial_cards : ℕ) : Prop :=
  let after_trade := initial_cards - 5 + 3
  let after_giving := after_trade - 9
  let final_cards := after_giving + 2
  final_cards = 4

theorem pokemon_card_solution :
  pokemon_card_problem 13 := by sorry

end NUMINAMATH_CALUDE_pokemon_card_solution_l3310_331079


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l3310_331060

/-- Given 7 bugs, each eating 4 flowers, prove that the total number of flowers eaten is 28. -/
theorem bugs_eating_flowers :
  let number_of_bugs : ℕ := 7
  let flowers_per_bug : ℕ := 4
  number_of_bugs * flowers_per_bug = 28 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l3310_331060


namespace NUMINAMATH_CALUDE_max_rectangles_in_modified_grid_l3310_331025

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular cut-out --/
structure Cutout :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the area of a grid --/
def gridArea (g : Grid) : ℕ :=
  g.rows * g.cols

/-- Calculates the area of a cutout --/
def cutoutArea (c : Cutout) : ℕ :=
  c.width * c.height

/-- Calculates the remaining area after cutouts --/
def remainingArea (g : Grid) (cutouts : List Cutout) : ℕ :=
  gridArea g - (cutouts.map cutoutArea).sum

/-- Theorem: Maximum number of 1x3 rectangles in modified 8x8 grid --/
theorem max_rectangles_in_modified_grid :
  let initial_grid : Grid := ⟨8, 8⟩
  let cutouts : List Cutout := [⟨2, 2⟩, ⟨2, 2⟩, ⟨2, 2⟩]
  let remaining_cells := remainingArea initial_grid cutouts
  (remaining_cells / 3 : ℕ) = 17 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangles_in_modified_grid_l3310_331025


namespace NUMINAMATH_CALUDE_train_passengers_count_l3310_331009

/-- Represents the number of passengers in each carriage of a train -/
structure TrainCarriages :=
  (c1 c2 c3 c4 c5 : ℕ)

/-- Defines the condition for the number of neighbours a passenger has -/
def valid_neighbours (tc : TrainCarriages) : Prop :=
  ∀ i : Fin 5, 
    let neighbours := match i with
      | 0 => tc.c1 - 1 + tc.c2
      | 1 => tc.c1 + tc.c2 - 1 + tc.c3
      | 2 => tc.c2 + tc.c3 - 1 + tc.c4
      | 3 => tc.c3 + tc.c4 - 1 + tc.c5
      | 4 => tc.c4 + tc.c5 - 1
    (neighbours = 5 ∨ neighbours = 10)

/-- The main theorem stating that under the given conditions, 
    the total number of passengers is 17 -/
theorem train_passengers_count (tc : TrainCarriages) 
  (h1 : tc.c1 ≥ 1 ∧ tc.c2 ≥ 1 ∧ tc.c3 ≥ 1 ∧ tc.c4 ≥ 1 ∧ tc.c5 ≥ 1)
  (h2 : valid_neighbours tc) : 
  tc.c1 + tc.c2 + tc.c3 + tc.c4 + tc.c5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_count_l3310_331009


namespace NUMINAMATH_CALUDE_cone_cross_section_area_l3310_331045

/-- Represents a cone with height and base radius -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a cross-section of a cone -/
structure CrossSection where
  distanceFromCenter : ℝ

/-- Calculates the area of a cross-section passing through the vertex of a cone -/
def crossSectionArea (c : Cone) (cs : CrossSection) : ℝ :=
  sorry

theorem cone_cross_section_area 
  (c : Cone) 
  (cs : CrossSection) 
  (h1 : c.height = 20) 
  (h2 : c.baseRadius = 25) 
  (h3 : cs.distanceFromCenter = 12) : 
  crossSectionArea c cs = 500 := by
  sorry

end NUMINAMATH_CALUDE_cone_cross_section_area_l3310_331045


namespace NUMINAMATH_CALUDE_ruby_height_l3310_331084

/-- Given the heights of various people, prove Ruby's height -/
theorem ruby_height
  (janet_height : ℕ)
  (charlene_height : ℕ)
  (pablo_height : ℕ)
  (ruby_height : ℕ)
  (h1 : janet_height = 62)
  (h2 : charlene_height = 2 * janet_height)
  (h3 : pablo_height = charlene_height + 70)
  (h4 : ruby_height = pablo_height - 2)
  : ruby_height = 192 := by
  sorry

end NUMINAMATH_CALUDE_ruby_height_l3310_331084


namespace NUMINAMATH_CALUDE_number_difference_l3310_331019

theorem number_difference (L S : ℕ) (h1 : L > S) (h2 : L = 6 * S + 15) (h3 : L = 1656) : L - S = 1383 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3310_331019


namespace NUMINAMATH_CALUDE_composite_figure_area_l3310_331059

/-- The area of a composite figure with specific properties -/
theorem composite_figure_area : 
  let equilateral_triangle_area := Real.sqrt 3 / 4
  let rectangle_area := 1
  let right_triangle_area := 1 / 2
  2 * equilateral_triangle_area + rectangle_area + right_triangle_area = Real.sqrt 3 / 2 + 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_composite_figure_area_l3310_331059


namespace NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l3310_331081

/-- Represents a repeating decimal where the decimal part repeats infinitely -/
def RepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / (999 : ℚ)

/-- The fraction 0.888... -/
def a : ℚ := RepeatingDecimal 0 888

/-- The fraction 1.222... -/
def b : ℚ := RepeatingDecimal 1 222

/-- Theorem stating that the ratio of 0.888... to 1.222... is equal to 8/11 -/
theorem ratio_of_repeating_decimals : a / b = 8 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l3310_331081


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3310_331043

/-- Represents a right circular cone -/
structure RightCircularCone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ  -- Angle from a fixed reference line on the cone surface

/-- Calculates the shortest distance between two points on the surface of a cone -/
def shortestDistanceOnCone (cone : RightCircularCone) (p1 p2 : ConePoint) : ℝ := sorry

/-- Theorem stating the shortest distance between two specific points on a cone -/
theorem shortest_distance_on_specific_cone :
  let cone : RightCircularCone := ⟨450, 300 * Real.sqrt 3⟩
  let p1 : ConePoint := ⟨200, 0⟩
  let p2 : ConePoint := ⟨300 * Real.sqrt 3, π⟩
  shortestDistanceOnCone cone p1 p2 = 200 + 300 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l3310_331043


namespace NUMINAMATH_CALUDE_bolzano_weierstrass_l3310_331070

-- Define a bounded sequence
def BoundedSequence (a : ℕ → ℝ) : Prop :=
  ∃ (M : ℝ), ∀ (n : ℕ), |a n| ≤ M

-- Define a limit point
def LimitPoint (a : ℕ → ℝ) (x : ℝ) : Prop :=
  ∀ (ε : ℝ), ε > 0 → ∀ (N : ℕ), ∃ (n : ℕ), n ≥ N ∧ |a n - x| < ε

-- Bolzano-Weierstrass theorem
theorem bolzano_weierstrass (a : ℕ → ℝ) :
  BoundedSequence a → ∃ (x : ℝ), LimitPoint a x :=
sorry

end NUMINAMATH_CALUDE_bolzano_weierstrass_l3310_331070


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_l3310_331099

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 1 + a 9 = 10) :
  a 5 = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_l3310_331099


namespace NUMINAMATH_CALUDE_slope_of_line_from_equation_l3310_331055

theorem slope_of_line_from_equation (x y : ℝ) (h : (4 / x) + (5 / y) = 0) :
  ∃ m : ℝ, m = -5/4 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (4 / x₁ + 5 / y₁ = 0) → (4 / x₂ + 5 / y₂ = 0) → x₁ ≠ x₂ →
    (y₂ - y₁) / (x₂ - x₁) = m :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_from_equation_l3310_331055


namespace NUMINAMATH_CALUDE_tan_75_degrees_l3310_331073

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l3310_331073


namespace NUMINAMATH_CALUDE_amy_school_year_work_hours_l3310_331074

/-- Calculates the number of hours Amy needs to work per week during the school year -/
def school_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℚ)
  (school_year_weeks : ℕ) (school_year_earnings : ℚ) : ℚ :=
  let hourly_rate := summer_earnings / (summer_weeks * summer_hours_per_week : ℚ)
  let total_school_year_hours := school_year_earnings / hourly_rate
  total_school_year_hours / school_year_weeks

/-- Proves that Amy needs to work approximately 18 hours per week during the school year -/
theorem amy_school_year_work_hours :
  let result := school_year_hours_per_week 10 36 3000 30 4500
  18 < result ∧ result < 19 := by
  sorry

end NUMINAMATH_CALUDE_amy_school_year_work_hours_l3310_331074


namespace NUMINAMATH_CALUDE_sum_binomial_congruence_l3310_331040

theorem sum_binomial_congruence (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (∑' j, Nat.choose p j * Nat.choose (p + j) j) ≡ (2^p + 1) [ZMOD p^2] := by
  sorry

end NUMINAMATH_CALUDE_sum_binomial_congruence_l3310_331040


namespace NUMINAMATH_CALUDE_mother_hubbard_children_l3310_331042

theorem mother_hubbard_children (total_bar : ℚ) (children : ℕ) : 
  total_bar = 1 →
  (total_bar - total_bar / 3) = (children * (total_bar / 12)) →
  children = 8 := by
  sorry

end NUMINAMATH_CALUDE_mother_hubbard_children_l3310_331042


namespace NUMINAMATH_CALUDE_distance_between_students_l3310_331086

/-- The distance between two students after 4 hours, given they start from the same point
    and walk in opposite directions with speeds of 6 km/hr and 9 km/hr respectively. -/
theorem distance_between_students (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 6)
  (h2 : speed2 = 9)
  (h3 : time = 4) :
  speed1 * time + speed2 * time = 60 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_students_l3310_331086


namespace NUMINAMATH_CALUDE_michaels_brother_money_l3310_331023

theorem michaels_brother_money (michael_initial : ℕ) (brother_initial : ℕ) (candy_cost : ℕ) : 
  michael_initial = 42 →
  brother_initial = 17 →
  candy_cost = 3 →
  brother_initial + michael_initial / 2 - candy_cost = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_michaels_brother_money_l3310_331023


namespace NUMINAMATH_CALUDE_inequality_proof_l3310_331057

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((2*a + b + c)^2 / (2*a^2 + (b + c)^2)) + 
  ((2*b + c + a)^2 / (2*b^2 + (c + a)^2)) + 
  ((2*c + a + b)^2 / (2*c^2 + (a + b)^2)) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3310_331057
