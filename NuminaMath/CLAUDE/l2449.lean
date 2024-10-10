import Mathlib

namespace staircase_polygon_perimeter_staircase_polygon_area_l2449_244953

/-- A polygonal region formed by removing a 3x4 rectangle from an 8x12 rectangle -/
structure StaircasePolygon where
  width : ℕ := 12
  height : ℕ := 8
  small_width : ℕ := 3
  small_height : ℕ := 4
  area : ℕ := 86
  stair_side_length : ℕ := 1
  stair_side_count : ℕ := 12

/-- The perimeter of a StaircasePolygon is 44 -/
theorem staircase_polygon_perimeter (p : StaircasePolygon) : 
  p.width + p.height + (p.width - p.small_width) + (p.height - p.small_height) + p.stair_side_count * p.stair_side_length = 44 := by
  sorry

/-- The area of a StaircasePolygon is consistent with its dimensions -/
theorem staircase_polygon_area (p : StaircasePolygon) :
  p.area = p.width * p.height - p.small_width * p.small_height := by
  sorry

end staircase_polygon_perimeter_staircase_polygon_area_l2449_244953


namespace stating_min_natives_correct_stating_min_natives_sufficient_l2449_244992

/-- Represents the minimum number of natives required for the joke-sharing problem. -/
def min_natives (k : ℕ) : ℕ := 2^k

/-- 
Theorem stating that min_natives(k) is the smallest number of natives needed
for each native to know at least k jokes (apart from their own) after crossing the river.
-/
theorem min_natives_correct (k : ℕ) :
  ∀ N : ℕ, (∀ native : Fin N, 
    (∃ known_jokes : Finset (Fin N), 
      known_jokes.card ≥ k ∧ 
      native ∉ known_jokes ∧
      (∀ joke ∈ known_jokes, joke ≠ native))) 
    → N ≥ min_natives k :=
by
  sorry

/-- 
Theorem stating that min_natives(k) is sufficient for each native to know
at least k jokes (apart from their own) after crossing the river.
-/
theorem min_natives_sufficient (k : ℕ) :
  ∃ crossing_strategy : Unit,
    ∀ native : Fin (min_natives k),
      ∃ known_jokes : Finset (Fin (min_natives k)),
        known_jokes.card ≥ k ∧
        native ∉ known_jokes ∧
        (∀ joke ∈ known_jokes, joke ≠ native) :=
by
  sorry

end stating_min_natives_correct_stating_min_natives_sufficient_l2449_244992


namespace divisible_by_24_l2449_244934

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end divisible_by_24_l2449_244934


namespace smallest_longer_leg_length_l2449_244993

/-- Represents a 30-60-90 triangle --/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hypotenuse_eq : hypotenuse = 2 * shorterLeg
  longerLeg_eq : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of three connected 30-60-90 triangles --/
structure TriangleSequence where
  largest : Triangle30_60_90
  middle : Triangle30_60_90
  smallest : Triangle30_60_90
  connection1 : largest.longerLeg = middle.hypotenuse
  connection2 : middle.longerLeg = smallest.hypotenuse
  largest_hypotenuse : largest.hypotenuse = 12
  middle_special : middle.hypotenuse = middle.longerLeg

theorem smallest_longer_leg_length (seq : TriangleSequence) : 
  seq.smallest.longerLeg = 4.5 * Real.sqrt 3 := by
  sorry

end smallest_longer_leg_length_l2449_244993


namespace slurpee_change_l2449_244991

/-- Calculates the change received when buying Slurpees -/
theorem slurpee_change (money_given : ℕ) (slurpee_cost : ℕ) (slurpees_bought : ℕ) : 
  money_given = 20 → slurpee_cost = 2 → slurpees_bought = 6 →
  money_given - (slurpee_cost * slurpees_bought) = 8 := by
  sorry

end slurpee_change_l2449_244991


namespace stock_shares_calculation_l2449_244947

/-- Represents the number of shares for each stock --/
structure StockShares where
  v : ℕ
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the range of shares --/
def calculateRange (shares : StockShares) : ℕ :=
  max shares.v (max shares.w (max shares.x (max shares.y shares.z))) -
  min shares.v (min shares.w (min shares.x (min shares.y shares.z)))

/-- The main theorem to prove --/
theorem stock_shares_calculation (initial : StockShares) (y : ℕ) :
  initial.v = 68 →
  initial.w = 112 →
  initial.x = 56 →
  initial.z = 45 →
  initial.y = y →
  let final : StockShares := {
    v := initial.v,
    w := initial.w,
    x := initial.x - 20,
    y := initial.y + 23,
    z := initial.z
  }
  calculateRange final - calculateRange initial = 14 →
  y = 50 := by
  sorry

#check stock_shares_calculation

end stock_shares_calculation_l2449_244947


namespace geometric_sequence_condition_l2449_244981

/-- A geometric sequence with first term 1, second term a, and third term 16 -/
def is_geometric_sequence (a : ℝ) : Prop := ∃ r : ℝ, r ≠ 0 ∧ a = r ∧ 16 = r * r

/-- The condition is necessary but not sufficient -/
theorem geometric_sequence_condition (a : ℝ) :
  (is_geometric_sequence a → a = 4) ∧ ¬(a = 4 → is_geometric_sequence a) :=
sorry

end geometric_sequence_condition_l2449_244981


namespace total_cost_proof_l2449_244910

def hand_mitts_cost : ℚ := 14
def apron_cost : ℚ := 16
def utensils_cost : ℚ := 10
def knife_cost : ℚ := 2 * utensils_cost
def discount_rate : ℚ := 0.25
def tax_rate : ℚ := 0.08
def num_recipients : ℕ := 8

def total_cost : ℚ :=
  let set_cost := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let total_before_discount := num_recipients * set_cost
  let discounted_total := total_before_discount * (1 - discount_rate)
  discounted_total * (1 + tax_rate)

theorem total_cost_proof : total_cost = 388.8 := by
  sorry

end total_cost_proof_l2449_244910


namespace unpainted_cubes_6x6x6_l2449_244999

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_cubes : Nat
  painted_per_face : Nat
  strip_width : Nat
  strip_length : Nat

/-- Calculate the number of unpainted cubes in a painted cube -/
def unpainted_cubes (c : PaintedCube) : Nat :=
  sorry

/-- Theorem stating the number of unpainted cubes in the specific problem -/
theorem unpainted_cubes_6x6x6 :
  let c : PaintedCube := {
    size := 6,
    total_cubes := 216,
    painted_per_face := 10,
    strip_width := 2,
    strip_length := 5
  }
  unpainted_cubes c = 186 := by
  sorry

end unpainted_cubes_6x6x6_l2449_244999


namespace slower_train_speed_l2449_244921

/-- Proves the speed of a slower train given specific conditions --/
theorem slower_train_speed
  (train_length : ℝ)
  (faster_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 60)
  (h2 : faster_speed = 48)
  (h3 : passing_time = 36)
  : ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * (5 / 18) * passing_time = 2 * train_length :=
by
  sorry

#check slower_train_speed

end slower_train_speed_l2449_244921


namespace binomial_coefficient_divisible_by_prime_l2449_244961

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) : 
  Prime p → 0 < k → k < p → ∃ m : ℕ, Nat.choose p k = p * m := by
  sorry

end binomial_coefficient_divisible_by_prime_l2449_244961


namespace marks_buttons_l2449_244969

theorem marks_buttons (x : ℕ) : 
  (x + 3*x) / 2 = 28 → x = 14 := by
  sorry

end marks_buttons_l2449_244969


namespace nancy_chip_distribution_l2449_244902

/-- The number of tortilla chips Nancy initially had -/
def initial_chips : ℕ := 22

/-- The number of tortilla chips Nancy gave to her brother -/
def brother_chips : ℕ := 7

/-- The number of tortilla chips Nancy kept for herself -/
def nancy_chips : ℕ := 10

/-- The number of tortilla chips Nancy gave to her sister -/
def sister_chips : ℕ := initial_chips - brother_chips - nancy_chips

theorem nancy_chip_distribution :
  sister_chips = 5 := by sorry

end nancy_chip_distribution_l2449_244902


namespace solution_set_inequality_l2449_244935

theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ x ∈ Set.Icc (-1/2) (-1/3)) → 
  (∀ x, ax^2 - b*x - 1 < 0 ↔ x ∈ Set.Ioo 2 3) := by
  sorry

end solution_set_inequality_l2449_244935


namespace log_sum_and_product_imply_average_l2449_244986

theorem log_sum_and_product_imply_average (x y : ℝ) : 
  x > 0 → y > 0 → (Real.log x / Real.log y + Real.log y / Real.log x = 4) → x * y = 81 → 
  (x + y) / 2 = 15 := by
sorry

end log_sum_and_product_imply_average_l2449_244986


namespace valid_lineup_count_l2449_244918

/-- The total number of players in the basketball team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def quadruplets : ℕ := 4

/-- The size of the starting lineup -/
def lineup_size : ℕ := 6

/-- The maximum number of quadruplets allowed in the starting lineup -/
def max_quadruplets : ℕ := 2

/-- The number of ways to choose the starting lineup with the given restrictions -/
def valid_lineups : ℕ := 7062

theorem valid_lineup_count : 
  (Nat.choose total_players lineup_size) - 
  (Nat.choose quadruplets 3 * Nat.choose (total_players - quadruplets) (lineup_size - 3) +
   Nat.choose quadruplets 4 * Nat.choose (total_players - quadruplets) (lineup_size - 4)) = 
  valid_lineups :=
sorry

end valid_lineup_count_l2449_244918


namespace sqrt_two_triangle_one_l2449_244988

-- Define the triangle operation
def triangle (a b : ℝ) : ℝ := a^2 - a*b

-- Theorem statement
theorem sqrt_two_triangle_one :
  triangle (Real.sqrt 2) 1 = 2 - Real.sqrt 2 := by
  sorry

end sqrt_two_triangle_one_l2449_244988


namespace card_draw_probability_l2449_244942

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)

/-- The probability of drawing a specific sequence of cards -/
def draw_probability (d : Deck) (spades : Nat) (tens : Nat) (queens : Nat) : Rat :=
  let p1 := spades / d.cards
  let p2 := tens / (d.cards - 1)
  let p3 := queens / (d.cards - 2)
  p1 * p2 * p3

/-- The theorem to prove -/
theorem card_draw_probability :
  let d := Deck.mk 52
  let spades := 13
  let tens := 4
  let queens := 4
  draw_probability d spades tens queens = 17 / 11050 :=
by
  sorry


end card_draw_probability_l2449_244942


namespace probability_of_red_ball_l2449_244905

-- Define the total number of balls
def total_balls : ℕ := 7

-- Define the number of red balls
def red_balls : ℕ := 2

-- Define the number of black balls
def black_balls : ℕ := 4

-- Define the number of white balls
def white_balls : ℕ := 1

-- Define the probability of drawing a red ball
def prob_red_ball : ℚ := red_balls / total_balls

-- Theorem statement
theorem probability_of_red_ball :
  prob_red_ball = 2 / 7 := by sorry

end probability_of_red_ball_l2449_244905


namespace new_person_weight_l2449_244980

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℚ) (replaced_weight : ℚ) :
  initial_count = 10 →
  weight_increase = 5/2 →
  replaced_weight = 50 →
  ∃ (new_weight : ℚ),
    new_weight = replaced_weight + (initial_count * weight_increase) ∧
    new_weight = 75 :=
by sorry

end new_person_weight_l2449_244980


namespace quadratic_equation_equivalence_l2449_244967

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end quadratic_equation_equivalence_l2449_244967


namespace half_height_of_triangular_prism_l2449_244976

/-- Given a triangular prism with volume 576 cm³ and base area 3 cm², 
    half of its height is 96 cm. -/
theorem half_height_of_triangular_prism (volume : ℝ) (base_area : ℝ) (height : ℝ) :
  volume = 576 ∧ base_area = 3 ∧ volume = base_area * height →
  height / 2 = 96 := by
  sorry

end half_height_of_triangular_prism_l2449_244976


namespace another_rational_right_triangle_with_same_area_l2449_244932

/-- Given a right triangle with rational sides and area S, 
    there exists another right triangle with rational sides and area S -/
theorem another_rational_right_triangle_with_same_area 
  (a b c S : ℚ) : 
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem for right triangle
  (S = (1/2) * a * b) →  -- Area formula
  (∃ (a' b' c' : ℚ), 
    a'^2 + b'^2 = c'^2 ∧  -- New triangle is right-angled
    (1/2) * a' * b' = S ∧  -- New triangle has the same area
    (a' ≠ a ∨ b' ≠ b ∨ c' ≠ c))  -- New triangle is different from the original
  := by sorry

end another_rational_right_triangle_with_same_area_l2449_244932


namespace parabola_c_value_l2449_244960

-- Define the parabola equation
def parabola (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Theorem statement
theorem parabola_c_value :
  ∀ b c : ℝ,
  (parabola 2 b c = 12) ∧ (parabola (-2) b c = 8) →
  c = 6 := by
sorry

end parabola_c_value_l2449_244960


namespace volume_T_coefficients_qr_ps_ratio_l2449_244925

/-- A right rectangular prism with edge lengths 2, 4, and 6 units -/
structure RectangularPrism where
  length : ℝ := 2
  width : ℝ := 4
  height : ℝ := 6

/-- The set of points within distance r from any point in the prism -/
def T (C : RectangularPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume function of T(r) -/
def volume_T (C : RectangularPrism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume function -/
structure VolumeCoefficients where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ

theorem volume_T_coefficients (C : RectangularPrism) :
  ∃ (coeff : VolumeCoefficients),
    ∀ r : ℝ, volume_T C r = coeff.P * r^3 + coeff.Q * r^2 + coeff.R * r + coeff.S :=
  sorry

theorem qr_ps_ratio (C : RectangularPrism) (coeff : VolumeCoefficients)
    (h : ∀ r : ℝ, volume_T C r = coeff.P * r^3 + coeff.Q * r^2 + coeff.R * r + coeff.S) :
    coeff.Q * coeff.R / (coeff.P * coeff.S) = 16.5 :=
  sorry

end volume_T_coefficients_qr_ps_ratio_l2449_244925


namespace train_speed_l2449_244939

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 2500) (h2 : time = 100) :
  length / time = 25 := by
  sorry

end train_speed_l2449_244939


namespace longest_watching_time_l2449_244914

structure Show where
  episodes : ℕ
  minutesPerEpisode : ℕ
  speed : ℚ

def watchingTimePerDay (s : Show) (days : ℕ) : ℚ :=
  (s.episodes * s.minutesPerEpisode : ℚ) / (s.speed * (days * 60))

theorem longest_watching_time (showA showB showC : Show) (days : ℕ) :
  showA.episodes = 20 ∧ 
  showA.minutesPerEpisode = 30 ∧ 
  showA.speed = 1.2 ∧
  showB.episodes = 25 ∧ 
  showB.minutesPerEpisode = 45 ∧ 
  showB.speed = 1 ∧
  showC.episodes = 30 ∧ 
  showC.minutesPerEpisode = 40 ∧ 
  showC.speed = 0.9 ∧
  days = 5 →
  watchingTimePerDay showC days > watchingTimePerDay showA days ∧
  watchingTimePerDay showC days > watchingTimePerDay showB days :=
sorry

end longest_watching_time_l2449_244914


namespace song_listens_proof_l2449_244909

/-- Given a song with an initial number of listens that doubles each month for 3 months,
    resulting in a total of 900,000 listens, prove that the initial number of listens is 60,000. -/
theorem song_listens_proof (L : ℕ) : 
  (L + 2*L + 4*L + 8*L = 900000) → L = 60000 := by
  sorry

end song_listens_proof_l2449_244909


namespace teachers_day_theorem_l2449_244982

/-- A directed graph with 200 vertices where each vertex has exactly one outgoing edge -/
structure TeacherGraph where
  vertices : Finset (Fin 200)
  edges : Fin 200 → Fin 200
  edge_property : ∀ v, v ∈ vertices → edges v ≠ v

/-- An independent set in the graph -/
def IndependentSet (G : TeacherGraph) (S : Finset (Fin 200)) : Prop :=
  ∀ u v, u ∈ S → v ∈ S → u ≠ v → G.edges u ≠ v

/-- The theorem stating that there exists an independent set of size at least 67 -/
theorem teachers_day_theorem (G : TeacherGraph) :
  ∃ S : Finset (Fin 200), IndependentSet G S ∧ S.card ≥ 67 := by
  sorry


end teachers_day_theorem_l2449_244982


namespace complex_fraction_equality_l2449_244904

theorem complex_fraction_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end complex_fraction_equality_l2449_244904


namespace pizza_combinations_l2449_244916

def num_toppings : ℕ := 8

theorem pizza_combinations (n : ℕ) (h : n = num_toppings) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

#eval num_toppings.choose 1 + num_toppings.choose 2 + num_toppings.choose 3

end pizza_combinations_l2449_244916


namespace journey_takes_four_days_l2449_244958

/-- Represents the journey of a young man returning home from vacation. -/
structure Journey where
  totalDistance : ℕ
  firstLegDistance : ℕ
  secondLegDistance : ℕ
  totalDays : ℕ
  remainingDays : ℕ

/-- Checks if the journey satisfies the given conditions. -/
def isValidJourney (j : Journey) : Prop :=
  j.totalDistance = j.firstLegDistance + j.secondLegDistance ∧
  j.firstLegDistance = 246 ∧
  j.secondLegDistance = 276 ∧
  j.totalDays - j.remainingDays = j.remainingDays / 2 + 1 ∧
  j.totalDays > 0 ∧
  j.remainingDays > 0

/-- Theorem stating that the journey takes 4 days in total. -/
theorem journey_takes_four_days :
  ∃ (j : Journey), isValidJourney j ∧ j.totalDays = 4 :=
sorry


end journey_takes_four_days_l2449_244958


namespace nonagon_diagonals_l2449_244955

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals :
  diagonals_in_nonagon = 27 := by sorry

end nonagon_diagonals_l2449_244955


namespace negative_one_greater_than_negative_two_l2449_244926

theorem negative_one_greater_than_negative_two : -1 > -2 := by
  sorry

end negative_one_greater_than_negative_two_l2449_244926


namespace percentage_needed_is_35_l2449_244963

/-- The percentage of total marks needed to pass, given Pradeep's score, 
    the marks he fell short by, and the maximum marks. -/
def percentage_to_pass (pradeep_score : ℕ) (marks_short : ℕ) (max_marks : ℕ) : ℚ :=
  ((pradeep_score + marks_short : ℚ) / max_marks) * 100

/-- Theorem stating that the percentage needed to pass is 35% -/
theorem percentage_needed_is_35 (pradeep_score marks_short max_marks : ℕ) 
  (h1 : pradeep_score = 185)
  (h2 : marks_short = 25)
  (h3 : max_marks = 600) :
  percentage_to_pass pradeep_score marks_short max_marks = 35 := by
  sorry

#eval percentage_to_pass 185 25 600

end percentage_needed_is_35_l2449_244963


namespace complex_power_equality_smallest_power_is_minimal_l2449_244949

/-- The smallest positive integer n for which (a+bi)^(n+1) = (a-bi)^(n+1) holds for some positive real a and b -/
def smallest_power : ℕ := 3

theorem complex_power_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.mk a b)^4 = (Complex.mk a (-b))^4 → b / a = 1 :=
by sorry

theorem smallest_power_is_minimal (n : ℕ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  n < smallest_power →
  (Complex.mk a b)^(n + 1) ≠ (Complex.mk a (-b))^(n + 1) :=
by sorry

#check smallest_power
#check complex_power_equality
#check smallest_power_is_minimal

end complex_power_equality_smallest_power_is_minimal_l2449_244949


namespace linear_coefficient_of_equation_l2449_244984

theorem linear_coefficient_of_equation : ∃ (a b c : ℝ), 
  (∀ x, (2*x + 1)*(x - 3) = x^2 + 1) → 
  (∀ x, a*x^2 + b*x + c = 0) ∧ 
  b = -5 := by
  sorry

end linear_coefficient_of_equation_l2449_244984


namespace root_equation_implies_expression_value_l2449_244962

theorem root_equation_implies_expression_value (m : ℝ) : 
  m^2 + 2*m - 1 = 0 → 2*m^2 + 4*m + 2021 = 2023 := by
  sorry

end root_equation_implies_expression_value_l2449_244962


namespace rectangles_in_7x4_grid_l2449_244968

/-- The number of rectangles in a grid -/
def num_rectangles (columns rows : ℕ) : ℕ :=
  (columns + 1).choose 2 * (rows + 1).choose 2

/-- Theorem: In a 7x4 grid, the number of rectangles is 280 -/
theorem rectangles_in_7x4_grid :
  num_rectangles 7 4 = 280 := by sorry

end rectangles_in_7x4_grid_l2449_244968


namespace solve_for_a_l2449_244927

theorem solve_for_a : ∃ a : ℝ, (2 - 3 * (a + 1) = 2 * 1) ∧ (a = -1) := by
  sorry

end solve_for_a_l2449_244927


namespace units_digit_of_31_cubed_plus_13_cubed_l2449_244901

theorem units_digit_of_31_cubed_plus_13_cubed : ∃ n : ℕ, 31^3 + 13^3 = 10 * n + 8 := by
  sorry

end units_digit_of_31_cubed_plus_13_cubed_l2449_244901


namespace twentieth_term_of_specific_sequence_l2449_244972

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem twentieth_term_of_specific_sequence :
  arithmetic_sequence 2 3 20 = 59 := by
  sorry

end twentieth_term_of_specific_sequence_l2449_244972


namespace polygon_sides_count_polygon_sides_count_proof_l2449_244903

theorem polygon_sides_count : ℕ → Prop :=
  fun n =>
    (n - 2) * 180 = 2 * 360 →
    n = 6

-- The proof is omitted
theorem polygon_sides_count_proof : ∃ n : ℕ, polygon_sides_count n :=
  sorry

end polygon_sides_count_polygon_sides_count_proof_l2449_244903


namespace adam_gave_seven_boxes_l2449_244924

/-- The number of boxes Adam gave to his little brother -/
def boxes_given (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) : ℕ :=
  (total_boxes * pieces_per_box - pieces_left) / pieces_per_box

/-- Proof that Adam gave 7 boxes to his little brother -/
theorem adam_gave_seven_boxes :
  boxes_given 13 6 36 = 7 := by
  sorry

end adam_gave_seven_boxes_l2449_244924


namespace exists_quadratic_with_two_n_roots_l2449_244908

-- Define a quadratic polynomial
def QuadraticPolynomial : Type := ℝ → ℝ

-- Define the property of having 2n distinct real roots for n-fold composition
def HasTwoNRoots (f : QuadraticPolynomial) : Prop :=
  ∀ n : ℕ, ∃! (roots : Finset ℝ), (roots.card = 2 * n) ∧ 
    (∀ x ∈ roots, (f^[n]) x = 0) ∧
    (∀ x : ℝ, (f^[n]) x = 0 → x ∈ roots)

-- The theorem to be proved
theorem exists_quadratic_with_two_n_roots :
  ∃ f : QuadraticPolynomial, HasTwoNRoots f :=
sorry


end exists_quadratic_with_two_n_roots_l2449_244908


namespace calculation_difference_l2449_244917

def correct_calculation : ℤ := 12 - (3 * 4)

def incorrect_calculation : ℤ := 12 - 3 * 4

theorem calculation_difference :
  correct_calculation - incorrect_calculation = 0 := by
  sorry

end calculation_difference_l2449_244917


namespace elmo_has_24_books_l2449_244998

def elmos_books (elmo_multiplier laura_multiplier stu_books : ℕ) : ℕ :=
  elmo_multiplier * (laura_multiplier * stu_books)

theorem elmo_has_24_books :
  elmos_books 3 2 4 = 24 := by
  sorry

end elmo_has_24_books_l2449_244998


namespace chess_tournament_games_l2449_244970

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

#check chess_tournament_games

end chess_tournament_games_l2449_244970


namespace equation_solution_range_l2449_244987

theorem equation_solution_range (x m : ℝ) : 
  x / (x - 1) - 2 = (3 * m) / (2 * x - 2) → 
  x > 0 → 
  x ≠ 1 → 
  m < 4/3 ∧ m ≠ 2/3 := by
  sorry

end equation_solution_range_l2449_244987


namespace unique_solution_sqrt_equation_l2449_244983

/-- The equation √(x+2) + 2√(x-1) + 3√(3x-2) = 10 has a unique solution x = 2 -/
theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, (x + 2 ≥ 0) ∧ (x - 1 ≥ 0) ∧ (3*x - 2 ≥ 0) ∧
  (Real.sqrt (x + 2) + 2 * Real.sqrt (x - 1) + 3 * Real.sqrt (3*x - 2) = 10) ∧
  x = 2 := by
  sorry

end unique_solution_sqrt_equation_l2449_244983


namespace log_sum_equals_two_l2449_244950

theorem log_sum_equals_two (a b : ℝ) (h1 : 2^a = Real.sqrt 10) (h2 : 5^b = Real.sqrt 10) :
  1/a + 1/b = 2 := by
  sorry

end log_sum_equals_two_l2449_244950


namespace coin_flips_count_l2449_244966

theorem coin_flips_count (heads : ℕ) (tails : ℕ) : heads = 65 → tails = heads + 81 → heads + tails = 211 := by
  sorry

end coin_flips_count_l2449_244966


namespace sum_of_digits_of_large_number_l2449_244929

/-- The sum of the digits of 10^93 - 93 -/
def sum_of_digits : ℕ := 826

/-- The number represented by 10^93 - 93 -/
def large_number : ℕ := 10^93 - 93

/-- Function to calculate the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem sum_of_digits_of_large_number :
  digit_sum large_number = sum_of_digits :=
sorry

end sum_of_digits_of_large_number_l2449_244929


namespace arithmetic_geometric_sequence_l2449_244995

theorem arithmetic_geometric_sequence : 
  ∃ (a b c d e : ℚ), 
    -- The five numbers
    a = 4 ∧ b = 8 ∧ c = 12 ∧ d = 16 ∧ e = 64/3 ∧
    -- First four form an arithmetic progression
    (b - a = c - b) ∧ (c - b = d - c) ∧
    -- Sum of first four is 40
    (a + b + c + d = 40) ∧
    -- Last three form a geometric progression
    (c^2 = b * d) ∧
    -- Product of outer terms of geometric progression is 32 times the second number
    (c * e = 32 * b) := by
  sorry

end arithmetic_geometric_sequence_l2449_244995


namespace sum_of_integers_l2449_244907

theorem sum_of_integers : 7 + (-19) + 13 + (-31) = -30 := by sorry

end sum_of_integers_l2449_244907


namespace geometric_sequence_roots_l2449_244937

theorem geometric_sequence_roots (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^2 - m*a + 2) * (a^2 - n*a + 2) = 0 ∧
    (b^2 - m*b + 2) * (b^2 - n*b + 2) = 0 ∧
    (c^2 - m*c + 2) * (c^2 - n*c + 2) = 0 ∧
    (d^2 - m*d + 2) * (d^2 - n*d + 2) = 0 ∧
    a = 1/2 ∧
    ∃ r : ℝ, b = a*r ∧ c = b*r ∧ d = c*r) →
  |m - n| = 3/2 := by
  sorry

end geometric_sequence_roots_l2449_244937


namespace rhombus_diagonal_length_l2449_244945

/-- Given a rhombus with an area equal to that of a square with side length 8,
    and one diagonal of length 8, prove that the other diagonal has length 16. -/
theorem rhombus_diagonal_length :
  ∀ (d1 : ℝ),
  (d1 * 8 / 2 = 8 * 8) →
  d1 = 16 := by
sorry

end rhombus_diagonal_length_l2449_244945


namespace exercise_time_is_1910_l2449_244994

/-- The total exercise time for Javier, Sanda, Luis, and Nita -/
def total_exercise_time : ℕ :=
  let javier := 50 * 10
  let sanda := 90 * 3 + 75 * 2 + 45 * 4
  let luis := 60 * 5 + 30 * 3
  let nita := 100 * 2 + 55 * 4
  javier + sanda + luis + nita

/-- Theorem stating that the total exercise time is 1910 minutes -/
theorem exercise_time_is_1910 : total_exercise_time = 1910 := by
  sorry

end exercise_time_is_1910_l2449_244994


namespace residue_of_negative_1237_mod_29_l2449_244906

theorem residue_of_negative_1237_mod_29 : Int.mod (-1237) 29 = 10 := by
  sorry

end residue_of_negative_1237_mod_29_l2449_244906


namespace inscribed_cube_volume_l2449_244920

theorem inscribed_cube_volume (large_cube_side : ℝ) (sphere_diameter : ℝ) 
  (small_cube_diagonal : ℝ) (small_cube_side : ℝ) (small_cube_volume : ℝ) :
  large_cube_side = 12 →
  sphere_diameter = large_cube_side →
  small_cube_diagonal = sphere_diameter →
  small_cube_diagonal = small_cube_side * Real.sqrt 3 →
  small_cube_side = 12 / Real.sqrt 3 →
  small_cube_volume = small_cube_side ^ 3 →
  small_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l2449_244920


namespace inequality_problem_l2449_244913

theorem inequality_problem :
  -- Part 1: Maximum value of m
  (∃ M : ℝ, (∀ m : ℝ, (∃ x : ℝ, |x - 2| - |x + 3| ≥ |m + 1|) → m ≤ M) ∧
    (∃ x : ℝ, |x - 2| - |x + 3| ≥ |M + 1|) ∧
    M = 4) ∧
  -- Part 2: Inequality for positive a, b, c
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + 2*b + c = 4 →
    1 / (a + b) + 1 / (b + c) ≥ 1) :=
by sorry

end inequality_problem_l2449_244913


namespace matrix_power_difference_l2449_244922

/-- Given a 2x2 matrix B, prove that B^30 - 3B^29 equals the specified result -/
theorem matrix_power_difference (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B = ![![2, 4], ![0, 1]]) : 
  B^30 - 3 * B^29 = ![![-2, 0], ![0, 2]] := by
  sorry

end matrix_power_difference_l2449_244922


namespace trigonometric_expression_evaluation_l2449_244996

theorem trigonometric_expression_evaluation :
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) /
  (Real.sin (12 * π / 180) * (4 * Real.cos (12 * π / 180) ^ 2 - 2)) = -4 * Real.sqrt 3 :=
by sorry

end trigonometric_expression_evaluation_l2449_244996


namespace goods_train_speed_is_36_l2449_244930

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 36

/-- The speed of the express train in km/h -/
def express_train_speed : ℝ := 90

/-- The time difference between the departure of the two trains in hours -/
def time_difference : ℝ := 6

/-- The time it takes for the express train to catch up with the goods train in hours -/
def catch_up_time : ℝ := 4

/-- Theorem stating that the speed of the goods train is 36 km/h -/
theorem goods_train_speed_is_36 :
  goods_train_speed * (time_difference + catch_up_time) = express_train_speed * catch_up_time :=
by sorry

end goods_train_speed_is_36_l2449_244930


namespace perfect_square_trinomial_l2449_244975

/-- A trinomial x^2 + kx + 9 is a perfect square if and only if k = 6 or k = -6 -/
theorem perfect_square_trinomial (k : ℝ) : 
  (∃ (a b : ℝ), ∀ x, x^2 + k*x + 9 = (a*x + b)^2) ↔ (k = 6 ∨ k = -6) := by
  sorry

end perfect_square_trinomial_l2449_244975


namespace circle_radius_c_value_l2449_244956

theorem circle_radius_c_value (c : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → 
  c = 16 := by
  sorry

end circle_radius_c_value_l2449_244956


namespace place_mat_length_l2449_244959

theorem place_mat_length (r : ℝ) (n : ℕ) (h_r : r = 5) (h_n : n = 8) : 
  let x := 2 * r * Real.sin (π / (2 * n))
  x = r * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

#check place_mat_length

end place_mat_length_l2449_244959


namespace geometric_jump_sequence_ratio_range_l2449_244938

/-- A sequence is a jump sequence if for any three consecutive terms,
    the product (a_i - a_i+2)(a_i+2 - a_i+1) is positive. -/
def is_jump_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

/-- A sequence is geometric with ratio q if each term is q times the previous term. -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_jump_sequence_ratio_range
  (a : ℕ → ℝ) (q : ℝ) (h_geom : is_geometric_sequence a q) (h_jump : is_jump_sequence a) :
  q ∈ Set.Ioo (-1 : ℝ) 0 :=
sorry

end geometric_jump_sequence_ratio_range_l2449_244938


namespace fraction_to_whole_number_l2449_244919

theorem fraction_to_whole_number : 
  (∃ n : ℤ, (12 : ℚ) / 2 = n) ∧
  (∀ n : ℤ, (8 : ℚ) / 6 ≠ n) ∧
  (∀ n : ℤ, (9 : ℚ) / 5 ≠ n) ∧
  (∀ n : ℤ, (10 : ℚ) / 4 ≠ n) ∧
  (∀ n : ℤ, (11 : ℚ) / 3 ≠ n) := by
  sorry

end fraction_to_whole_number_l2449_244919


namespace sum_of_p_and_q_l2449_244944

theorem sum_of_p_and_q (p q : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ y : ℝ, 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14)) →
  p + q = 69 := by
sorry

end sum_of_p_and_q_l2449_244944


namespace doctor_nurse_ratio_l2449_244931

theorem doctor_nurse_ratio (total : ℕ) (nurses : ℕ) (h1 : total = 280) (h2 : nurses = 180) :
  (total - nurses) / (Nat.gcd (total - nurses) nurses) = 5 ∧
  nurses / (Nat.gcd (total - nurses) nurses) = 9 :=
by sorry

end doctor_nurse_ratio_l2449_244931


namespace chicken_count_l2449_244957

theorem chicken_count (east : ℕ) (west : ℕ) : 
  east = 40 → 
  (east : ℚ) + west * (1 - 1/4 - 1/3) = (1/2 : ℚ) * (east + west) → 
  east + west = 280 := by
sorry

end chicken_count_l2449_244957


namespace min_value_expression_l2449_244977

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 32) :
  x^2 + 4*x*y + 4*y^2 + 2*z^2 ≥ 96 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 32 ∧ x₀^2 + 4*x₀*y₀ + 4*y₀^2 + 2*z₀^2 = 96 :=
by sorry

end min_value_expression_l2449_244977


namespace cindy_hit_eight_l2449_244985

-- Define the set of players
inductive Player : Type
| Alice : Player
| Ben : Player
| Cindy : Player
| Dave : Player
| Ellen : Player

-- Define the score function
def score : Player → ℕ
| Player.Alice => 10
| Player.Ben => 6
| Player.Cindy => 9
| Player.Dave => 15
| Player.Ellen => 19

-- Define the set of possible scores on the dartboard
def dartboard_scores : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define a function to check if a player's score can be composed of two different dartboard scores
def valid_score (p : Player) : Prop :=
  ∃ (a b : ℕ), a ∈ dartboard_scores ∧ b ∈ dartboard_scores ∧ a ≠ b ∧ a + b = score p

-- Theorem: Cindy is the only player who could have hit the section worth 8 points
theorem cindy_hit_eight :
  (∀ p : Player, valid_score p) →
  (∃! p : Player, ∃ (a : ℕ), a ∈ dartboard_scores ∧ a ≠ 8 ∧ a + 8 = score p) ∧
  (∃ (a : ℕ), a ∈ dartboard_scores ∧ a ≠ 8 ∧ a + 8 = score Player.Cindy) :=
by sorry

end cindy_hit_eight_l2449_244985


namespace mistaken_addition_problem_l2449_244911

/-- Given a two-digit number and conditions from the problem, prove it equals 49. -/
theorem mistaken_addition_problem (A B : ℕ) : 
  B = 9 →
  A * 10 + 6 + 253 = 299 →
  A * 10 + B = 49 :=
by sorry

end mistaken_addition_problem_l2449_244911


namespace regular_survey_rate_l2449_244973

/-- Proves that the regular rate for completing a survey is Rs. 30 given the specified conditions. -/
theorem regular_survey_rate
  (total_surveys : ℕ)
  (cellphone_rate_factor : ℚ)
  (cellphone_surveys : ℕ)
  (total_earnings : ℕ)
  (h1 : total_surveys = 100)
  (h2 : cellphone_rate_factor = 1.20)
  (h3 : cellphone_surveys = 50)
  (h4 : total_earnings = 3300) :
  ∃ (regular_rate : ℚ),
    regular_rate = 30 ∧
    regular_rate * (total_surveys - cellphone_surveys : ℚ) +
    (regular_rate * cellphone_rate_factor) * cellphone_surveys = total_earnings := by
  sorry

end regular_survey_rate_l2449_244973


namespace periodic_function_equality_l2449_244979

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β), if f(3) = 3, then f(2016) = -3 -/
theorem periodic_function_equality (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)
  f 3 = 3 → f 2016 = -3 := by sorry

end periodic_function_equality_l2449_244979


namespace geometric_sequence_sum_l2449_244964

/-- Given a geometric sequence {a_n} with positive terms, where 4a_3, a_5, and 2a_4 form an arithmetic sequence, and a_1 = 1, prove that S_4 = 15 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence
  4 * a 3 + a 5 = 2 * (2 * a 4) →  -- Arithmetic sequence condition
  a 1 = 1 →  -- First term is 1
  a 1 + a 2 + a 3 + a 4 = 15 :=  -- S_4 = 15
by sorry

end geometric_sequence_sum_l2449_244964


namespace statue_weight_proof_l2449_244943

/-- Given a set of statues carved from a marble block, prove the weight of each remaining statue. -/
theorem statue_weight_proof (initial_weight discarded_weight first_statue_weight second_statue_weight : ℝ)
  (h1 : initial_weight = 80)
  (h2 : discarded_weight = 22)
  (h3 : first_statue_weight = 10)
  (h4 : second_statue_weight = 18)
  (h5 : initial_weight ≥ first_statue_weight + second_statue_weight + discarded_weight) :
  let remaining_weight := initial_weight - first_statue_weight - second_statue_weight - discarded_weight
  (remaining_weight / 2 : ℝ) = 15 := by
  sorry

end statue_weight_proof_l2449_244943


namespace two_numbers_problem_l2449_244915

theorem two_numbers_problem :
  ∃! (a b : ℕ), a > b ∧ (a / b : ℚ) * 6 = 10 ∧ (a - b : ℤ) + 4 = 10 := by
  sorry

end two_numbers_problem_l2449_244915


namespace min_value_fraction_l2449_244941

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 5 → 1/x + 4/(y+1) ≤ 1/a + 4/(b+1)) ∧
  (1/x + 4/(y+1) = 3/2) :=
sorry

end min_value_fraction_l2449_244941


namespace parallelogram_z_range_l2449_244928

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -2)

-- Define the function z
def z (x y : ℝ) : ℝ := 2*x - 5*y

-- Statement of the theorem
theorem parallelogram_z_range :
  ∀ (x y : ℝ), 
  (∃ (t₁ t₂ t₃ : ℝ), 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ 0 ≤ t₃ ∧ t₁ + t₂ + t₃ ≤ 1 ∧
    (x, y) = t₁ • A + t₂ • B + t₃ • C + (1 - t₁ - t₂ - t₃) • (C + A - B)) →
  -14 ≤ z x y ∧ z x y ≤ 20 :=
by sorry


end parallelogram_z_range_l2449_244928


namespace percentage_difference_l2449_244997

theorem percentage_difference : (0.7 * 40) - (4 / 5 * 25) = 8 := by
  sorry

end percentage_difference_l2449_244997


namespace sqrt_74_between_consecutive_integers_product_l2449_244989

theorem sqrt_74_between_consecutive_integers_product : ∃ (n : ℕ), 
  n > 0 ∧ 
  (n : ℝ) < Real.sqrt 74 ∧ 
  Real.sqrt 74 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 72 := by
sorry

end sqrt_74_between_consecutive_integers_product_l2449_244989


namespace cube_of_ten_expansion_l2449_244951

theorem cube_of_ten_expansion : 9^3 + 3*(9^2) + 3*9 + 1 = 1000 := by sorry

end cube_of_ten_expansion_l2449_244951


namespace sin_shift_l2449_244965

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by
  sorry

end sin_shift_l2449_244965


namespace consecutive_numbers_sum_l2449_244971

theorem consecutive_numbers_sum (n : ℕ) : 
  n + (n + 1) + (n + 2) = 60 → (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end consecutive_numbers_sum_l2449_244971


namespace rhombus_properties_l2449_244948

structure Rhombus (O : ℝ × ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  side_length : ℝ
  OB_length : ℝ
  OD_length : ℝ
  is_rhombus : side_length = 4 ∧ OB_length = 6 ∧ OD_length = 6

def on_semicircle (A : ℝ × ℝ) : Prop :=
  let (x, y) := A
  (x - 2)^2 + y^2 = 4 ∧ 2 ≤ x ∧ x ≤ 4

theorem rhombus_properties (r : Rhombus (0, 0)) :
  (|r.A.1 * r.C.1 + r.A.2 * r.C.2| = 36) ∧
  (∀ A', on_semicircle A' → 
    ∃ C', r.C = C' → (C' = (5, 5) ∨ C' = (5, -5))) :=
sorry

end rhombus_properties_l2449_244948


namespace storm_damage_conversion_l2449_244912

/-- Converts Canadian dollars to American dollars given exchange rates -/
def storm_damage_in_usd (damage_cad : ℝ) (cad_to_eur : ℝ) (eur_to_usd : ℝ) : ℝ :=
  damage_cad * cad_to_eur * eur_to_usd

/-- Theorem: The storm damage in USD is 40.5 million given the conditions -/
theorem storm_damage_conversion :
  storm_damage_in_usd 45000000 0.75 1.2 = 40500000 := by
  sorry

end storm_damage_conversion_l2449_244912


namespace percentage_x_more_than_y_l2449_244974

theorem percentage_x_more_than_y : 
  ∀ (x y z : ℝ),
  y = 1.2 * z →
  z = 250 →
  x + y + z = 925 →
  (x - y) / y * 100 = 25 := by
sorry

end percentage_x_more_than_y_l2449_244974


namespace max_late_all_days_l2449_244936

theorem max_late_all_days (total_late : ℕ) (late_monday : ℕ) (late_tuesday : ℕ) (late_wednesday : ℕ)
  (h_total : total_late = 30)
  (h_monday : late_monday = 20)
  (h_tuesday : late_tuesday = 13)
  (h_wednesday : late_wednesday = 7) :
  ∃ (x : ℕ), x ≤ 5 ∧ 
    x ≤ late_monday ∧ 
    x ≤ late_tuesday ∧ 
    x ≤ late_wednesday ∧
    (late_monday - x) + (late_tuesday - x) + (late_wednesday - x) + x ≤ total_late ∧
    ∀ (y : ℕ), y > x → 
      (y > late_monday ∨ y > late_tuesday ∨ y > late_wednesday ∨
       (late_monday - y) + (late_tuesday - y) + (late_wednesday - y) + y > total_late) :=
by sorry

end max_late_all_days_l2449_244936


namespace sara_final_quarters_l2449_244954

/-- Calculates the final number of quarters Sara has after a series of transactions -/
def sara_quarters (initial : ℕ) (from_dad : ℕ) (spent : ℕ) (dollars_from_mom : ℕ) (quarters_per_dollar : ℕ) : ℕ :=
  initial + from_dad - spent + dollars_from_mom * quarters_per_dollar

/-- Theorem stating that Sara ends up with 63 quarters -/
theorem sara_final_quarters : 
  sara_quarters 21 49 15 2 4 = 63 := by
  sorry

end sara_final_quarters_l2449_244954


namespace systematic_sampling_probabilities_l2449_244940

/-- Systematic sampling probabilities -/
theorem systematic_sampling_probabilities
  (population : ℕ)
  (sample_size : ℕ)
  (removed : ℕ)
  (h_pop : population = 1005)
  (h_sample : sample_size = 50)
  (h_removed : removed = 5) :
  (removed : ℚ) / population = 5 / 1005 ∧
  (sample_size : ℚ) / population = 50 / 1005 :=
sorry

end systematic_sampling_probabilities_l2449_244940


namespace average_speed_two_segments_l2449_244923

/-- Calculate the average speed of a two-segment journey -/
theorem average_speed_two_segments 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 50) 
  (h2 : speed1 = 20) 
  (h3 : distance2 = 20) 
  (h4 : speed2 = 40) : 
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 70 / 3 := by
  sorry

#eval (70 : ℚ) / 3

end average_speed_two_segments_l2449_244923


namespace x_value_l2449_244900

theorem x_value : ∃ x : ℝ, (3 * x = (16 - x) + 4) ∧ (x = 5) := by
  sorry

end x_value_l2449_244900


namespace calculation_proof_l2449_244946

theorem calculation_proof : (10^8 / (2 * 10^5)) - 50 = 450 := by
  sorry

end calculation_proof_l2449_244946


namespace cos_shift_right_l2449_244933

theorem cos_shift_right (x : ℝ) :
  2 * Real.cos (2 * (x - π/8)) = 2 * Real.cos (2 * x - π/4) :=
by sorry

end cos_shift_right_l2449_244933


namespace tan_equality_periodic_l2449_244978

theorem tan_equality_periodic (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (348 * π / 180) → n = -12 :=
by sorry

end tan_equality_periodic_l2449_244978


namespace box_surface_areas_and_cost_l2449_244952

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular box -/
def surfaceArea (box : BoxDimensions) : ℝ :=
  2 * (box.length * box.width + box.length * box.height + box.width * box.height)

/-- Theorem about the surface areas of two boxes and their cost -/
theorem box_surface_areas_and_cost 
  (a b c : ℝ) 
  (small_box : BoxDimensions := ⟨a, b, c⟩)
  (large_box : BoxDimensions := ⟨2*a, 2*b, 1.5*c⟩)
  (cardboard_cost_per_sqm : ℝ := 15) : 
  (surfaceArea small_box + surfaceArea large_box = 10*a*b + 8*b*c + 8*a*c) ∧ 
  (surfaceArea large_box - surfaceArea small_box = 6*a*b + 4*b*c + 4*a*c) ∧
  (a = 20 → b = 10 → c = 15 → 
    cardboard_cost_per_sqm * (surfaceArea small_box + surfaceArea large_box) / 10000 = 8.4) :=
by sorry


end box_surface_areas_and_cost_l2449_244952


namespace solve_for_b_l2449_244990

theorem solve_for_b (x b : ℝ) : 
  (10 * x + b) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 →
  x = 0.3 →
  b = 2 := by
sorry

end solve_for_b_l2449_244990
