import Mathlib

namespace theater_seat_count_l3536_353627

/-- Represents the number of seats in a theater with a specific seating arrangement. -/
def theaterSeats (firstRowSeats : ℕ) (lastRowSeats : ℕ) : ℕ :=
  let additionalRows := (lastRowSeats - firstRowSeats) / 2
  let totalRows := additionalRows + 1
  let sumAdditionalSeats := additionalRows * (2 + (lastRowSeats - firstRowSeats)) / 2
  firstRowSeats * totalRows + sumAdditionalSeats

/-- Theorem stating that a theater with the given seating arrangement has 3434 seats. -/
theorem theater_seat_count :
  theaterSeats 12 128 = 3434 :=
by sorry

end theater_seat_count_l3536_353627


namespace polynomial_coefficient_sum_l3536_353654

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 6) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 32 := by
sorry

end polynomial_coefficient_sum_l3536_353654


namespace min_words_for_spanish_exam_l3536_353667

/-- Represents the Spanish vocabulary exam scenario -/
structure SpanishExam where
  total_words : ℕ
  min_score_percent : ℕ

/-- Calculates the minimum number of words needed to achieve the desired score -/
def min_words_needed (exam : SpanishExam) : ℕ :=
  (exam.min_score_percent * exam.total_words + 99) / 100

/-- Theorem stating the minimum number of words needed for the given exam conditions -/
theorem min_words_for_spanish_exam :
  let exam : SpanishExam := { total_words := 500, min_score_percent := 85 }
  min_words_needed exam = 425 := by
  sorry

#eval min_words_needed { total_words := 500, min_score_percent := 85 }

end min_words_for_spanish_exam_l3536_353667


namespace other_root_of_quadratic_l3536_353657

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - k = 0 ∧ x = 3) → 
  (∃ y : ℝ, y^2 - 2*y - k = 0 ∧ y = -1) :=
by sorry

end other_root_of_quadratic_l3536_353657


namespace no_function_exists_l3536_353641

theorem no_function_exists : ¬∃ (a : ℕ → ℕ), (a 0 = 0) ∧ (∀ n : ℕ, a n = n - a (a n)) := by
  sorry

end no_function_exists_l3536_353641


namespace valid_routes_count_l3536_353619

-- Define the cities
inductive City : Type
| P | Q | R | S | T | U

-- Define the roads
inductive Road : Type
| PQ | PS | PT | PU | QR | QS | RS | RT | SU | UT

-- Define a route as a list of roads
def Route := List Road

-- Function to check if a route is valid
def isValidRoute (r : Route) : Bool :=
  -- Implementation details omitted
  sorry

-- Function to count valid routes
def countValidRoutes : Nat :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem valid_routes_count :
  countValidRoutes = 15 :=
sorry

end valid_routes_count_l3536_353619


namespace ordered_pairs_count_l3536_353645

theorem ordered_pairs_count : 
  ∃! (pairs : List (ℤ × ℕ)), 
    (∀ (x : ℤ) (y : ℕ), (x, y) ∈ pairs ↔ 
      (∃ (m : ℕ), y = m^2 ∧ y = (x - 90)^2 - 4907)) ∧ 
    pairs.length = 4 := by
  sorry

end ordered_pairs_count_l3536_353645


namespace concentric_circles_radii_difference_l3536_353688

theorem concentric_circles_radii_difference 
  (r R : ℝ) 
  (h_positive : r > 0) 
  (h_ratio : π * R^2 = 4 * π * r^2) : 
  R - r = r := by
sorry

end concentric_circles_radii_difference_l3536_353688


namespace product_probabilities_l3536_353637

/-- The probability of a product having a defect -/
def p₁ : ℝ := 0.1

/-- The probability of the controller detecting an existing defect -/
def p₂ : ℝ := 0.8

/-- The probability of the controller mistakenly rejecting a non-defective product -/
def p₃ : ℝ := 0.3

/-- The probability of a product being mistakenly rejected -/
def P_A₁ : ℝ := (1 - p₁) * p₃

/-- The probability of a product being passed into finished goods with a defect -/
def P_A₂ : ℝ := p₁ * (1 - p₂)

/-- The probability of a product being rejected -/
def P_A₃ : ℝ := p₁ * p₂ + (1 - p₁) * p₃

theorem product_probabilities :
  P_A₁ = 0.27 ∧ P_A₂ = 0.02 ∧ P_A₃ = 0.35 := by
  sorry

end product_probabilities_l3536_353637


namespace train_crossing_time_l3536_353681

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 4 := by
  sorry

#check train_crossing_time

end train_crossing_time_l3536_353681


namespace day300_is_saturday_l3536_353651

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in the year 2004 -/
structure Date2004 where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Function to advance a date by a given number of days -/
def advanceDate (d : Date2004) (days : Nat) : Date2004 :=
  sorry

/-- The 50th day of 2004 is a Monday -/
def day50 : Date2004 :=
  { dayNumber := 50, dayOfWeek := DayOfWeek.Monday }

theorem day300_is_saturday :
  (advanceDate day50 250).dayOfWeek = DayOfWeek.Saturday :=
sorry

end day300_is_saturday_l3536_353651


namespace town_street_lights_l3536_353626

/-- Calculates the total number of street lights in a town -/
def total_street_lights (num_neighborhoods : ℕ) (roads_per_neighborhood : ℕ) (lights_per_side : ℕ) : ℕ :=
  num_neighborhoods * roads_per_neighborhood * lights_per_side * 2

theorem town_street_lights :
  total_street_lights 10 4 250 = 20000 := by
  sorry

end town_street_lights_l3536_353626


namespace hcf_problem_l3536_353683

/-- Given two positive integers with HCF H and LCM (H * 13 * 14),
    where the larger number is 350, prove that H = 70 -/
theorem hcf_problem (a b : ℕ) (H : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a = 350)
  (h4 : H = Nat.gcd a b) (h5 : Nat.lcm a b = H * 13 * 14) : H = 70 := by
  sorry

end hcf_problem_l3536_353683


namespace male_25_plus_percentage_proof_l3536_353602

/-- The percentage of male students in a graduating class -/
def male_percentage : ℝ := 0.4

/-- The percentage of female students who are 25 years old or older -/
def female_25_plus_percentage : ℝ := 0.4

/-- The probability of randomly selecting a student less than 25 years old -/
def under_25_probability : ℝ := 0.56

/-- The percentage of male students who are 25 years old or older -/
def male_25_plus_percentage : ℝ := 0.5

theorem male_25_plus_percentage_proof :
  male_25_plus_percentage = 0.5 :=
by sorry

end male_25_plus_percentage_proof_l3536_353602


namespace number_puzzle_l3536_353699

theorem number_puzzle : ∃! x : ℤ, x - 2 + 4 = 9 := by
  sorry

end number_puzzle_l3536_353699


namespace coefficient_of_x3y5_l3536_353647

/-- The coefficient of x^3y^5 in the expansion of (2/3x - y/3)^8 -/
def coefficient : ℚ := -448/6561

/-- The binomial expansion of (a + b)^n -/
def binomial_expansion (a b : ℚ) (n : ℕ) (k : ℕ) : ℚ := 
  (n.choose k) * (a^(n-k)) * (b^k)

theorem coefficient_of_x3y5 :
  coefficient = binomial_expansion (2/3) (-1/3) 8 5 := by
  sorry

end coefficient_of_x3y5_l3536_353647


namespace ellipse_hyperbola_product_l3536_353608

theorem ellipse_hyperbola_product (a b : ℝ) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |a * b| = Real.sqrt 867.75 := by
sorry

end ellipse_hyperbola_product_l3536_353608


namespace geometric_sequence_minimum_value_l3536_353663

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_exists : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1)
  (h_relation : a 6 = a 5 + 2 * a 4) :
  ∃ m n : ℕ, 1 / m + 4 / n = 3 / 2 ∧
    ∀ k l : ℕ, k > 0 ∧ l > 0 → 1 / k + 4 / l ≥ 3 / 2 :=
sorry

end geometric_sequence_minimum_value_l3536_353663


namespace function_upper_bound_condition_l3536_353631

theorem function_upper_bound_condition (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → a * x - x^2 ≤ 1) ↔ a ≤ 2 :=
by sorry

end function_upper_bound_condition_l3536_353631


namespace intersection_point_sum_l3536_353615

/-- The x-coordinate of the intersection point of two lines -/
def a : ℝ := 5.5

/-- The y-coordinate of the intersection point of two lines -/
def b : ℝ := 2.5

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := y = -x + 8

/-- The second line equation -/
def line2 (x y : ℝ) : Prop := 173 * y = -289 * x + 2021

theorem intersection_point_sum :
  line1 a b ∧ line2 a b → a + b = 8 := by
  sorry

end intersection_point_sum_l3536_353615


namespace second_number_value_l3536_353660

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 220 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a → 
  b = 60 := by
sorry

end second_number_value_l3536_353660


namespace blue_or_green_probability_l3536_353620

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  red : ℕ
  yellow : ℕ
  blue : ℕ
  green : ℕ
  all_faces : sides = red + yellow + blue + green

/-- The probability of an event -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

theorem blue_or_green_probability (d : ColoredDie)
    (h : d.sides = 10 ∧ d.red = 5 ∧ d.yellow = 3 ∧ d.blue = 1 ∧ d.green = 1) :
    probability (d.blue + d.green) d.sides = 1/5 := by
  sorry

end blue_or_green_probability_l3536_353620


namespace mink_babies_problem_l3536_353624

/-- Represents the problem of determining the number of babies each mink had --/
theorem mink_babies_problem (initial_minks : ℕ) (coats_made : ℕ) (skins_per_coat : ℕ) 
  (h1 : initial_minks = 30)
  (h2 : coats_made = 7)
  (h3 : skins_per_coat = 15) :
  ∃ babies_per_mink : ℕ, 
    (initial_minks + initial_minks * babies_per_mink) / 2 = coats_made * skins_per_coat ∧ 
    babies_per_mink = 6 := by
  sorry

end mink_babies_problem_l3536_353624


namespace bedroom_curtain_length_l3536_353612

theorem bedroom_curtain_length :
  let total_fabric_area : ℝ := 16 * 12
  let living_room_curtain_area : ℝ := 4 * 6
  let bedroom_curtain_width : ℝ := 2
  let remaining_fabric_area : ℝ := 160
  let bedroom_curtain_area : ℝ := total_fabric_area - living_room_curtain_area - remaining_fabric_area
  bedroom_curtain_area / bedroom_curtain_width = 4 := by
  sorry

end bedroom_curtain_length_l3536_353612


namespace square_area_ratio_l3536_353678

theorem square_area_ratio (p1 p2 : ℕ) (h1 : p1 = 32) (h2 : p2 = 20) : 
  (p1 / 4) ^ 2 / (p2 / 4) ^ 2 = 64 / 25 := by sorry

end square_area_ratio_l3536_353678


namespace caesars_rental_cost_is_800_l3536_353609

/-- Caesar's room rental cost -/
def caesars_rental_cost : ℝ := sorry

/-- Caesar's per-person meal cost -/
def caesars_meal_cost : ℝ := 30

/-- Venus Hall's room rental cost -/
def venus_rental_cost : ℝ := 500

/-- Venus Hall's per-person meal cost -/
def venus_meal_cost : ℝ := 35

/-- Number of guests at which the costs are equal -/
def equal_cost_guests : ℕ := 60

theorem caesars_rental_cost_is_800 :
  caesars_rental_cost = 800 :=
by
  have h : caesars_rental_cost + caesars_meal_cost * equal_cost_guests =
           venus_rental_cost + venus_meal_cost * equal_cost_guests :=
    sorry
  sorry

end caesars_rental_cost_is_800_l3536_353609


namespace triangle_abc_obtuse_l3536_353687

theorem triangle_abc_obtuse (A B C : ℝ) (a b c : ℝ) : 
  B = 2 * A → a = 1 → b = 4/3 → 0 < A → A < π → B > π/2 := by
  sorry

end triangle_abc_obtuse_l3536_353687


namespace fraction_equality_l3536_353616

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 := by
  sorry

end fraction_equality_l3536_353616


namespace fraction_denominator_l3536_353628

theorem fraction_denominator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (2 * y) / x + (3 * y) / 10 = 0.7 * y) : x = 5 := by
  sorry

end fraction_denominator_l3536_353628


namespace fourth_grade_students_l3536_353606

theorem fourth_grade_students (initial_students : ℕ) (left_students : ℕ) (increase_percentage : ℚ) : 
  initial_students = 10 →
  left_students = 4 →
  increase_percentage = 70/100 →
  (initial_students - left_students + (initial_students - left_students) * increase_percentage).floor = 10 :=
by sorry

end fourth_grade_students_l3536_353606


namespace dave_had_18_tickets_l3536_353644

/-- Calculates the number of tickets Dave had left after playing games and receiving tickets from a friend -/
def daves_tickets : ℕ :=
  let first_set := 14 - 2
  let second_set := 8 - 5
  let third_set := (first_set * 3) - 15
  let total_after_games := first_set + second_set + third_set
  let after_buying_toys := total_after_games - 25
  after_buying_toys + 7

/-- Theorem stating that Dave had 18 tickets left -/
theorem dave_had_18_tickets : daves_tickets = 18 := by
  sorry

end dave_had_18_tickets_l3536_353644


namespace expression_evaluation_l3536_353676

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := 1
  2 * (x - 2*y)^2 - (2*y + x) * (-2*y + x) = 5 := by sorry

end expression_evaluation_l3536_353676


namespace min_value_geometric_sequence_l3536_353642

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 4b₃ is -9/8 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) (s : ℝ) :
  b₁ = 2 → 
  b₂ = b₁ * s →
  b₃ = b₂ * s →
  (∃ (x : ℝ), 3 * b₂ + 4 * b₃ ≥ x) →
  (∀ (x : ℝ), (3 * b₂ + 4 * b₃ ≥ x) → x ≤ -9/8) :=
by sorry

end min_value_geometric_sequence_l3536_353642


namespace hotel_arrangement_l3536_353693

/-- The number of ways to distribute n distinct objects into k distinct containers,
    where each container must have at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to partition n distinct objects into k non-empty subsets. -/
def stirling2 (n k : ℕ) : ℕ := sorry

theorem hotel_arrangement :
  distribute 5 3 = 150 :=
by
  -- Define distribute in terms of stirling2 and factorial
  have h1 : ∀ n k, distribute n k = stirling2 n k * Nat.factorial k
  sorry
  
  -- Use the specific values for our problem
  have h2 : stirling2 5 3 = 25
  sorry
  
  -- Apply the definitions and properties
  rw [h1]
  simp [h2]
  -- The proof is completed by computation
  sorry

end hotel_arrangement_l3536_353693


namespace zigzag_angle_in_rectangle_l3536_353670

theorem zigzag_angle_in_rectangle (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 = 10)
  (h2 : angle2 = 14)
  (h3 : angle3 = 26)
  (h4 : angle4 = 33) :
  ∃ θ : ℝ, θ = 11 ∧ 
  (90 - angle1) + (90 - angle3) + θ = 180 ∧
  (180 - (90 - angle1) - angle2) + (180 - (90 - angle3) - angle4) + θ = 180 :=
by sorry

end zigzag_angle_in_rectangle_l3536_353670


namespace discount_amount_l3536_353674

theorem discount_amount (t_shirt_price backpack_price cap_price total_before_discount total_after_discount : ℕ) : 
  t_shirt_price = 30 →
  backpack_price = 10 →
  cap_price = 5 →
  total_before_discount = t_shirt_price + backpack_price + cap_price →
  total_after_discount = 43 →
  total_before_discount - total_after_discount = 2 := by
sorry

end discount_amount_l3536_353674


namespace base4_equals_base2_l3536_353611

-- Define a function to convert a number from base 4 to decimal
def base4ToDecimal (n : ℕ) : ℕ := sorry

-- Define a function to convert a number from base 2 to decimal
def base2ToDecimal (n : ℕ) : ℕ := sorry

theorem base4_equals_base2 : base4ToDecimal 1010 = base2ToDecimal 1000100 := by sorry

end base4_equals_base2_l3536_353611


namespace unique_solution_sum_l3536_353604

def star_operation (m n : ℕ) : ℕ := m^n + m*n

theorem unique_solution_sum (m n : ℕ) 
  (hm : m ≥ 2) 
  (hn : n ≥ 2) 
  (h_star : star_operation m n = 64) : 
  m + n = 6 := by sorry

end unique_solution_sum_l3536_353604


namespace grid_product_theorem_l3536_353671

theorem grid_product_theorem : ∃ (a b c d e f g h i : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = 120 ∧
   d * e * f = 120 ∧
   g * h * i = 120 ∧
   a * d * g = 120 ∧
   b * e * h = 120 ∧
   c * f * i = 120) ∧
  (∀ (p : ℕ), (∃ (x y z u v w : ℕ),
    x ≠ y ∧ x ≠ z ∧ x ≠ u ∧ x ≠ v ∧ x ≠ w ∧
    y ≠ z ∧ y ≠ u ∧ y ≠ v ∧ y ≠ w ∧
    z ≠ u ∧ z ≠ v ∧ z ≠ w ∧
    u ≠ v ∧ u ≠ w ∧
    v ≠ w ∧
    x * y * z = p ∧ u * v * w = p) → p ≥ 120) :=
by sorry

end grid_product_theorem_l3536_353671


namespace two_candles_burn_time_l3536_353692

/-- Burning time of candle 1 -/
def burn_time_1 : ℕ := 30

/-- Burning time of candle 2 -/
def burn_time_2 : ℕ := 40

/-- Burning time of candle 3 -/
def burn_time_3 : ℕ := 50

/-- Time all three candles burn simultaneously -/
def time_all_three : ℕ := 10

/-- Time only one candle burns -/
def time_one_candle : ℕ := 20

/-- Theorem stating that exactly two candles burn simultaneously for 35 minutes -/
theorem two_candles_burn_time :
  (burn_time_1 + burn_time_2 + burn_time_3) - (3 * time_all_three + time_one_candle) = 70 :=
by sorry

end two_candles_burn_time_l3536_353692


namespace expression_simplification_l3536_353666

theorem expression_simplification (x : ℝ) : 
  (((x+1)^3*(x^2-x+1)^3)/(x^3+1)^3)^3 * (((x-1)^3*(x^2+x+1)^3)/(x^3-1)^3)^3 = 1 :=
by sorry

end expression_simplification_l3536_353666


namespace soybean_price_l3536_353618

/-- Proves that the price of soybean is 20.5 given the conditions of the mixture problem -/
theorem soybean_price (peas_price : ℝ) (mixture_price : ℝ) (ratio : ℝ) :
  peas_price = 16 →
  ratio = 2 →
  mixture_price = 19 →
  (peas_price + ratio * (20.5 : ℝ)) / (1 + ratio) = mixture_price := by
sorry

end soybean_price_l3536_353618


namespace trailing_zeros_310_factorial_l3536_353668

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem trailing_zeros_310_factorial :
  trailingZeros 310 = 76 := by
  sorry

end trailing_zeros_310_factorial_l3536_353668


namespace inscribed_circle_square_area_l3536_353625

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 12 * x - 4 * y + 20

-- Define the square
structure Square where
  side_length : ℝ
  parallel_to_x_axis : Prop

-- Define the inscribed circle
structure InscribedCircle where
  equation : (ℝ → ℝ → Prop)
  square : Square

-- Theorem statement
theorem inscribed_circle_square_area 
  (circle : InscribedCircle) 
  (h : circle.equation = circle_equation) :
  circle.square.side_length^2 = 80 :=
sorry

end inscribed_circle_square_area_l3536_353625


namespace root_sum_equation_l3536_353613

theorem root_sum_equation (n m : ℝ) (hn : n ≠ 0) 
  (hroot : n^2 + m*n + 3*n = 0) : m + n = -3 := by
  sorry

end root_sum_equation_l3536_353613


namespace prob_two_defective_shipment_l3536_353673

/-- The probability of selecting two defective smartphones from a shipment -/
def prob_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1)

/-- Theorem: The probability of selecting two defective smartphones from a 
    shipment of 250 smartphones, of which 76 are defective, is equal to 
    (76/250) * (75/249) -/
theorem prob_two_defective_shipment : 
  prob_two_defective 250 76 = 76 / 250 * 75 / 249 := by
  sorry

#eval prob_two_defective 250 76

end prob_two_defective_shipment_l3536_353673


namespace vector_collinearity_l3536_353675

theorem vector_collinearity (m : ℝ) : 
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-1, 2)
  (∃ (k : ℝ), k ≠ 0 ∧ (m * a.1 + 4 * b.1, m * a.2 + 4 * b.2) = k • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  m = -2 :=
by sorry

end vector_collinearity_l3536_353675


namespace fred_red_marbles_l3536_353694

/-- Fred's marble collection --/
structure MarbleCollection where
  total : ℕ
  darkBlue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Theorem: Fred has 60 red marbles --/
theorem fred_red_marbles (m : MarbleCollection) : m.red = 60 :=
  by
  have h1 : m.total = 120 := by sorry
  have h2 : m.darkBlue = m.total / 4 := by sorry
  have h3 : m.red = 2 * m.darkBlue := by sorry
  have h4 : m.green = 10 := by sorry
  have h5 : m.yellow = 5 := by sorry
  
  -- Proof
  sorry


end fred_red_marbles_l3536_353694


namespace triangle_area_l3536_353672

/-- The area of a triangle composed of two right-angled triangles -/
theorem triangle_area (base1 height1 base2 height2 : ℝ) 
  (h1 : base1 = 1) (h2 : height1 = 1) 
  (h3 : base2 = 2) (h4 : height2 = 1) : 
  (1/2 * base1 * height1) + (1/2 * base2 * height2) = (3/2 : ℝ) := by
  sorry

end triangle_area_l3536_353672


namespace a7_value_in_arithmetic_sequence_l3536_353691

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a7_value_in_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_sum : a 3 + a 5 = 10) : 
  a 7 = 8 := by
sorry

end a7_value_in_arithmetic_sequence_l3536_353691


namespace scientific_notation_equality_l3536_353690

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.000000023 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.3 ∧ n = -8 := by
  sorry

end scientific_notation_equality_l3536_353690


namespace hannah_son_cutting_rate_l3536_353643

/-- The number of strands Hannah's son can cut per minute -/
def sonCuttingRate (totalStrands : ℕ) (hannahRate : ℕ) (totalTime : ℕ) : ℕ :=
  (totalStrands - hannahRate * totalTime) / totalTime

theorem hannah_son_cutting_rate :
  sonCuttingRate 22 8 2 = 3 := by
  sorry

end hannah_son_cutting_rate_l3536_353643


namespace steps_to_rockefeller_center_l3536_353622

theorem steps_to_rockefeller_center 
  (total_steps : ℕ) 
  (steps_to_times_square : ℕ) 
  (h1 : total_steps = 582) 
  (h2 : steps_to_times_square = 228) : 
  total_steps - steps_to_times_square = 354 := by
  sorry

end steps_to_rockefeller_center_l3536_353622


namespace paper_clip_distribution_l3536_353697

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) :
  total_clips = 81 →
  num_boxes = 9 →
  total_clips = num_boxes * clips_per_box →
  clips_per_box = 9 := by
  sorry

end paper_clip_distribution_l3536_353697


namespace second_container_capacity_l3536_353600

/-- Represents a container with dimensions and sand capacity -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  sandCapacity : ℝ

/-- Theorem stating the sand capacity of the second container -/
theorem second_container_capacity 
  (c1 : Container) 
  (c2 : Container) 
  (h1 : c1.height = 3)
  (h2 : c1.width = 4)
  (h3 : c1.length = 6)
  (h4 : c1.sandCapacity = 72)
  (h5 : c2.height = 3 * c1.height)
  (h6 : c2.width = 2 * c1.width)
  (h7 : c2.length = c1.length) :
  c2.sandCapacity = 432 := by
  sorry


end second_container_capacity_l3536_353600


namespace gcd_of_256_450_720_l3536_353664

theorem gcd_of_256_450_720 : Nat.gcd 256 (Nat.gcd 450 720) = 18 := by
  sorry

end gcd_of_256_450_720_l3536_353664


namespace opera_ticket_price_increase_l3536_353635

theorem opera_ticket_price_increase (initial_price new_price : ℝ) 
  (h1 : initial_price = 85)
  (h2 : new_price = 102) :
  (new_price - initial_price) / initial_price * 100 = 20 := by
  sorry

end opera_ticket_price_increase_l3536_353635


namespace irrationality_of_lambda_l3536_353684

theorem irrationality_of_lambda (n : ℕ) : 
  Irrational (Real.sqrt (3 * n^2 + 2 * n + 2)) := by sorry

end irrationality_of_lambda_l3536_353684


namespace quadratic_product_is_square_l3536_353630

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Derivative of a quadratic polynomial -/
def QuadraticPolynomial.deriv (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

theorem quadratic_product_is_square (f g : QuadraticPolynomial) 
  (h : ∀ x : ℝ, f.deriv x * g.deriv x ≥ |f.eval x| + |g.eval x|) :
  ∃ h : QuadraticPolynomial, ∀ x : ℝ, f.eval x * g.eval x = (h.eval x)^2 := by
  sorry

end quadratic_product_is_square_l3536_353630


namespace triangle_perimeter_l3536_353621

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ 
  (c^2 - 6*c + 8 = 0) ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  a + b + c = 13 := by
sorry

end triangle_perimeter_l3536_353621


namespace pen_cost_l3536_353665

/-- The cost of a pen given Elizabeth's budget and purchasing constraints -/
theorem pen_cost (total_budget : ℝ) (pencil_cost : ℝ) (pencil_count : ℕ) (pen_count : ℕ) :
  total_budget = 20 →
  pencil_cost = 1.6 →
  pencil_count = 5 →
  pen_count = 6 →
  (pencil_count * pencil_cost + pen_count * ((total_budget - pencil_count * pencil_cost) / pen_count) = total_budget) →
  (total_budget - pencil_count * pencil_cost) / pen_count = 2 := by
sorry

end pen_cost_l3536_353665


namespace symmetric_circle_equation_l3536_353685

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original circle -/
def original_circle : Circle :=
  { center := (-2, -1), radius := 2 }

/-- The symmetric circle with respect to x-axis -/
def symmetric_circle : Circle :=
  { center := reflect_x original_circle.center, radius := original_circle.radius }

/-- Equation of a circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem symmetric_circle_equation :
  ∀ x y : ℝ, circle_equation symmetric_circle x y ↔ (x + 2)^2 + (y - 1)^2 = 4 := by
  sorry

end symmetric_circle_equation_l3536_353685


namespace simplest_quadratic_radical_l3536_353638

theorem simplest_quadratic_radical : 
  let options : List ℝ := [Real.sqrt (1/2), Real.sqrt 8, Real.sqrt 15, Real.sqrt 20]
  ∀ x ∈ options, x ≠ Real.sqrt 15 → 
    ∃ y z : ℕ, (y > 1 ∧ z > 1 ∧ x = Real.sqrt y * z) ∨ 
              (y > 1 ∧ z > 1 ∧ x = (Real.sqrt y) / z) :=
by sorry

end simplest_quadratic_radical_l3536_353638


namespace brick_surface_area_l3536_353648

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 8 cm x 6 cm x 2 cm brick is 152 cm² -/
theorem brick_surface_area :
  surface_area 8 6 2 = 152 := by
sorry

end brick_surface_area_l3536_353648


namespace daria_credit_card_debt_l3536_353680

def couch_price : ℝ := 800
def couch_discount : ℝ := 0.10
def table_price : ℝ := 120
def table_discount : ℝ := 0.05
def lamp_price : ℝ := 50
def rug_price : ℝ := 250
def rug_discount : ℝ := 0.20
def bookshelf_price : ℝ := 180
def bookshelf_discount : ℝ := 0.15
def artwork_price : ℝ := 100
def artwork_discount : ℝ := 0.25
def savings : ℝ := 500

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_cost : ℝ :=
  discounted_price couch_price couch_discount +
  discounted_price table_price table_discount +
  lamp_price +
  discounted_price rug_price rug_discount +
  discounted_price bookshelf_price bookshelf_discount +
  discounted_price artwork_price artwork_discount

theorem daria_credit_card_debt :
  total_cost - savings = 812 := by sorry

end daria_credit_card_debt_l3536_353680


namespace f_shifted_passes_through_point_one_zero_l3536_353662

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Define the shifted function f(x-1)
def f_shifted (a : ℝ) (x : ℝ) : ℝ := f a (x - 1)

-- Theorem statement
theorem f_shifted_passes_through_point_one_zero (a : ℝ) :
  f_shifted a 1 = 0 := by
  sorry

end f_shifted_passes_through_point_one_zero_l3536_353662


namespace tangent_and_roots_l3536_353698

noncomputable section

def F (x : ℝ) := x * Real.log x

def tangent_line (x y : ℝ) := 2 * x - y - Real.exp 1 = 0

def has_two_roots (t : ℝ) :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    Real.exp (-2) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧
    F x₁ = t ∧ F x₂ = t

theorem tangent_and_roots :
  (∀ x y, F x = y → x = Real.exp 1 → tangent_line x y) ∧
  (∀ t, has_two_roots t ↔ -Real.exp (-1) < t ∧ t ≤ -2 * Real.exp (-2)) :=
sorry

end tangent_and_roots_l3536_353698


namespace binomial_coefficient_property_l3536_353661

theorem binomial_coefficient_property :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243 :=
by
  sorry

end binomial_coefficient_property_l3536_353661


namespace correct_operation_l3536_353629

theorem correct_operation (a : ℝ) : -a + 5*a = 4*a := by
  sorry

end correct_operation_l3536_353629


namespace min_value_theorem_l3536_353650

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y ≤ 2) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 4 ∧
  ∀ (z : ℝ), z = 2 / (x + 3 * y) + 1 / (x - y) → z ≥ min_val :=
sorry

end min_value_theorem_l3536_353650


namespace skee_ball_tickets_count_l3536_353659

/-- The number of tickets Tom won playing 'skee ball' -/
def skee_ball_tickets : ℕ := sorry

/-- The number of tickets Tom won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- The number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- The number of tickets Tom has left -/
def remaining_tickets : ℕ := 50

theorem skee_ball_tickets_count : skee_ball_tickets = 25 := by
  sorry

end skee_ball_tickets_count_l3536_353659


namespace geometric_series_sum_l3536_353656

theorem geometric_series_sum (a b : ℝ) (h : b ≠ 1) (h2 : b ≠ 0) :
  (∑' n, a / b^n) = 2 →
  (∑' n, a / (2*a + b)^n) = 2/5 := by
sorry

end geometric_series_sum_l3536_353656


namespace min_value_theorem_l3536_353614

/-- Given a function f(x) = (1/3)ax³ + (1/2)bx² - x with a > 0 and b > 0,
    if f has a local minimum at x = 1, then the minimum value of (1/a) + (4/b) is 9 -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ → ℝ := λ x ↦ (1/3) * a * x^3 + (1/2) * b * x^2 - x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) →
  (∀ p q : ℝ, p > 0 → q > 0 → p + q = 1 → (1/p) + (4/q) ≥ 9) ∧
  (∃ p q : ℝ, p > 0 ∧ q > 0 ∧ p + q = 1 ∧ (1/p) + (4/q) = 9) :=
by sorry


end min_value_theorem_l3536_353614


namespace tan_theta_in_terms_of_x_l3536_353652

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2) 
  (h_x_pos : x > 0) 
  (h_cos : Real.cos (θ / 3) = Real.sqrt ((x + 2) / (3 * x))) : 
  Real.tan θ = 
    (Real.sqrt (1 - ((4 * (x + 2) ^ (3/2) - 3 * Real.sqrt (3 * x) * Real.sqrt (x + 2)) / 
      (3 * Real.sqrt (3 * x ^ 3))) ^ 2)) / 
    ((4 * (x + 2) ^ (3/2) - 3 * Real.sqrt (3 * x) * Real.sqrt (x + 2)) / 
      (3 * Real.sqrt (3 * x ^ 3))) := by
  sorry

end tan_theta_in_terms_of_x_l3536_353652


namespace only_unit_circle_has_nontrivial_solution_l3536_353607

theorem only_unit_circle_has_nontrivial_solution :
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ Real.sqrt (a^2 + b^2) = 1 ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = a - b → a = 0 ∧ b = 0) ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = 3 * (a + b) → a = 0 ∧ b = 0) := by
  sorry

end only_unit_circle_has_nontrivial_solution_l3536_353607


namespace total_flights_climbed_l3536_353653

/-- Represents a landmark with flights of stairs going up and down -/
structure Landmark where
  name : String
  flightsUp : ℕ
  flightsDown : ℕ

/-- Calculates the total flights for a landmark -/
def totalFlights (l : Landmark) : ℕ := l.flightsUp + l.flightsDown

/-- The landmarks Rachel visited -/
def landmarks : List Landmark := [
  { name := "Eiffel Tower", flightsUp := 347, flightsDown := 216 },
  { name := "Notre-Dame Cathedral", flightsUp := 178, flightsDown := 165 },
  { name := "Leaning Tower of Pisa", flightsUp := 294, flightsDown := 172 },
  { name := "Colosseum", flightsUp := 122, flightsDown := 93 },
  { name := "Sagrada Familia", flightsUp := 267, flightsDown := 251 },
  { name := "Park Güell", flightsUp := 134, flightsDown := 104 }
]

/-- Theorem: The total number of flights Rachel climbed is 2343 -/
theorem total_flights_climbed : (landmarks.map totalFlights).sum = 2343 := by
  sorry

end total_flights_climbed_l3536_353653


namespace rectangular_plot_breadth_l3536_353623

/-- 
Given a rectangular plot where:
- The area is 23 times its breadth
- The difference between the length and breadth is 10 metres

This theorem proves that the breadth of the plot is 13 metres.
-/
theorem rectangular_plot_breadth (length breadth : ℝ) 
  (h1 : length * breadth = 23 * breadth) 
  (h2 : length - breadth = 10) : 
  breadth = 13 := by
sorry

end rectangular_plot_breadth_l3536_353623


namespace roof_area_l3536_353689

/-- Calculates the area of a rectangular roof given the conditions --/
theorem roof_area (width : ℝ) (length : ℝ) : 
  length = 4 * width → 
  length - width = 42 → 
  width * length = 784 := by
  sorry

end roof_area_l3536_353689


namespace rv_parking_probability_l3536_353677

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars that have already parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces required for the RV -/
def required_adjacent_spaces : ℕ := 3

/-- The probability of being able to park the RV -/
def parking_probability : ℚ := 232 / 323

theorem rv_parking_probability :
  let empty_spaces := total_spaces - parked_cars
  let total_arrangements := Nat.choose total_spaces parked_cars
  let valid_arrangements := total_arrangements - Nat.choose (empty_spaces + parked_cars - required_adjacent_spaces + 1) empty_spaces
  (valid_arrangements : ℚ) / total_arrangements = parking_probability := by
  sorry

end rv_parking_probability_l3536_353677


namespace green_yarn_length_l3536_353605

theorem green_yarn_length :
  ∀ (green_length red_length : ℕ),
  red_length = 3 * green_length + 8 →
  green_length + red_length = 632 →
  green_length = 156 :=
by
  sorry

end green_yarn_length_l3536_353605


namespace remainder_17_63_mod_7_l3536_353610

theorem remainder_17_63_mod_7 : 17^63 % 7 = 6 := by sorry

end remainder_17_63_mod_7_l3536_353610


namespace debby_vacation_pictures_l3536_353695

/-- The number of pictures Debby took at the zoo -/
def zoo_pictures : ℕ := 24

/-- The number of pictures Debby took at the museum -/
def museum_pictures : ℕ := 12

/-- The number of pictures Debby deleted -/
def deleted_pictures : ℕ := 14

/-- The total number of pictures Debby took during her vacation -/
def total_pictures : ℕ := zoo_pictures + museum_pictures

/-- The number of pictures Debby still has from her vacation -/
def remaining_pictures : ℕ := total_pictures - deleted_pictures

theorem debby_vacation_pictures : remaining_pictures = 22 := by
  sorry

end debby_vacation_pictures_l3536_353695


namespace stockholm_malmo_distance_l3536_353658

/-- The scale factor of the map, representing kilometers per centimeter. -/
def scale : ℝ := 12

/-- The distance between Stockholm and Malmö on the map, in centimeters. -/
def map_distance : ℝ := 120

/-- The actual distance between Stockholm and Malmö, in kilometers. -/
def actual_distance : ℝ := map_distance * scale

/-- Theorem stating that the actual distance between Stockholm and Malmö is 1440 km. -/
theorem stockholm_malmo_distance : actual_distance = 1440 := by
  sorry

end stockholm_malmo_distance_l3536_353658


namespace article_cost_l3536_353679

/-- Proves that if selling an article for 350 gains 5% more than selling it for 340, then the cost is 140 -/
theorem article_cost (sell_price_high : ℝ) (sell_price_low : ℝ) (cost : ℝ) :
  sell_price_high = 350 ∧
  sell_price_low = 340 ∧
  (sell_price_high - cost) = (sell_price_low - cost) * 1.05 →
  cost = 140 := by
  sorry

end article_cost_l3536_353679


namespace leo_marbles_count_l3536_353603

/-- The number of marbles in each pack -/
def marbles_per_pack : ℕ := 10

/-- The fraction of packs given to Manny -/
def manny_fraction : ℚ := 1/4

/-- The fraction of packs given to Neil -/
def neil_fraction : ℚ := 1/8

/-- The number of packs Leo kept for himself -/
def leo_packs : ℕ := 25

/-- The total number of packs Leo had initially -/
def total_packs : ℕ := 40

/-- The total number of marbles Leo had initially -/
def total_marbles : ℕ := total_packs * marbles_per_pack

theorem leo_marbles_count :
  manny_fraction * total_packs + neil_fraction * total_packs + leo_packs = total_packs ∧
  total_marbles = 400 :=
sorry

end leo_marbles_count_l3536_353603


namespace sum_reciprocal_lower_bound_l3536_353696

theorem sum_reciprocal_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b ≤ 4) :
  1/a + 1/b ≥ 1 := by
  sorry

end sum_reciprocal_lower_bound_l3536_353696


namespace emily_age_l3536_353636

theorem emily_age :
  ∀ (e g : ℕ),
  g = 15 * e →
  g - e = 70 →
  e = 5 :=
by
  sorry

end emily_age_l3536_353636


namespace reflection_yoz_plane_l3536_353632

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the original point P
def P : Point3D := ⟨3, 1, 5⟩

-- Define the function for reflection across the yOz plane
def reflectYOZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

-- Theorem statement
theorem reflection_yoz_plane :
  reflectYOZ P = ⟨-3, 1, 5⟩ := by
  sorry

end reflection_yoz_plane_l3536_353632


namespace distance_from_point_on_number_line_l3536_353649

theorem distance_from_point_on_number_line :
  ∀ x : ℝ, |x - (-3)| = 4 ↔ x = -7 ∨ x = 1 := by
sorry

end distance_from_point_on_number_line_l3536_353649


namespace equation_solutions_count_l3536_353682

theorem equation_solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => 4 * p.1 + 7 * p.2 = 588 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 589) (Finset.range 589))).card = 21 :=
sorry

end equation_solutions_count_l3536_353682


namespace first_employee_wage_is_12_l3536_353634

/-- The hourly wage of the first employee -/
def first_employee_wage : ℝ := sorry

/-- The hourly wage of the second employee -/
def second_employee_wage : ℝ := 22

/-- The hourly subsidy for hiring the second employee -/
def hourly_subsidy : ℝ := 6

/-- The number of hours worked per week -/
def hours_per_week : ℝ := 40

/-- The weekly savings by hiring the first employee -/
def weekly_savings : ℝ := 160

theorem first_employee_wage_is_12 :
  first_employee_wage = 12 :=
by
  have h1 : hours_per_week * (second_employee_wage - hourly_subsidy) - 
            hours_per_week * first_employee_wage = weekly_savings := by sorry
  sorry

end first_employee_wage_is_12_l3536_353634


namespace ellipse_and_slopes_l3536_353640

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the foci F1 and F2 -/
def foci (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  F1.1 < 0 ∧ F2.1 > 0 ∧ F1.2 = 0 ∧ F2.2 = 0 ∧ F1.1^2 = F2.1^2 ∧ F2.1^2 = a^2 - b^2

/-- Definition of the circles intersecting on C -/
def circles_intersect_on_C (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, ellipse_C P.1 P.2 a b ∧ 
    (P.1 - F1.1)^2 + (P.2 - F1.2)^2 = 9 ∧
    (P.1 - F2.1)^2 + (P.2 - F2.2)^2 = 1

/-- Definition of point A -/
def point_A (a b : ℝ) : ℝ × ℝ := (0, b)

/-- Definition of angle F1AF2 -/
def angle_F1AF2 (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  let A := point_A a b
  Real.cos (2 * Real.pi / 3) = 
    ((F1.1 - A.1) * (F2.1 - A.1) + (F1.2 - A.2) * (F2.2 - A.2)) /
    (Real.sqrt ((F1.1 - A.1)^2 + (F1.2 - A.2)^2) * Real.sqrt ((F2.1 - A.1)^2 + (F2.2 - A.2)^2))

/-- Definition of line l -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y + 1 = k * (x - 2)

/-- Definition of points M and N -/
def points_M_N (a b k : ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, ellipse_C M.1 M.2 a b ∧ ellipse_C N.1 N.2 a b ∧
    line_l k M.1 M.2 ∧ line_l k N.1 N.2 ∧ M ≠ N

/-- Main theorem -/
theorem ellipse_and_slopes (a b : ℝ) (F1 F2 : ℝ × ℝ) (k : ℝ) :
  ellipse_C 0 b a b →
  foci F1 F2 a b →
  circles_intersect_on_C F1 F2 a b →
  angle_F1AF2 F1 F2 a b →
  points_M_N a b k →
  (a = 2 ∧ b = 1) ∧
  (∃ k1 k2 : ℝ, k1 + k2 = -1) :=
sorry

end ellipse_and_slopes_l3536_353640


namespace island_puzzle_l3536_353633

-- Define the types of people
inductive PersonType
| Truthful
| Liar

-- Define the genders
inductive Gender
| Boy
| Girl

-- Define a person
structure Person where
  type : PersonType
  gender : Gender

-- Define the statements made by A and B
def statement_A (a b : Person) : Prop :=
  a.type = PersonType.Truthful → b.type = PersonType.Liar

def statement_B (a b : Person) : Prop :=
  b.gender = Gender.Boy → a.gender = Gender.Girl

-- Theorem to prove
theorem island_puzzle :
  ∃ (a b : Person),
    (statement_A a b ↔ a.type = PersonType.Truthful) ∧
    (statement_B a b ↔ b.type = PersonType.Liar) ∧
    a.type = PersonType.Truthful ∧
    a.gender = Gender.Boy ∧
    b.type = PersonType.Liar ∧
    b.gender = Gender.Boy :=
  sorry

end island_puzzle_l3536_353633


namespace larger_number_problem_l3536_353646

theorem larger_number_problem (x y : ℝ) 
  (h1 : 5 * y = 7 * x) 
  (h2 : y - x = 10) : 
  y = 35 := by
sorry

end larger_number_problem_l3536_353646


namespace best_fit_line_slope_for_given_data_l3536_353601

/-- Data point representing height and weight measurements -/
structure DataPoint where
  height : ℝ
  weight : ℝ

/-- Calculate the slope of the best-fit line for given data points -/
def bestFitLineSlope (data : List DataPoint) : ℝ :=
  sorry

/-- Theorem stating that the slope of the best-fit line for the given data is 0.525 -/
theorem best_fit_line_slope_for_given_data :
  let data := [
    DataPoint.mk 150 50,
    DataPoint.mk 160 55,
    DataPoint.mk 170 60.5
  ]
  bestFitLineSlope data = 0.525 := by
  sorry

end best_fit_line_slope_for_given_data_l3536_353601


namespace sum_reciprocals_bound_l3536_353617

theorem sum_reciprocals_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) :
  ∃ S : Set ℝ, S = { x | x ≥ 9 ∧ ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' + b' + c' = 1 ∧ x = 1/a' + 1/b' + 1/c' } :=
by sorry

end sum_reciprocals_bound_l3536_353617


namespace symmetric_point_on_number_line_l3536_353686

/-- Given points A, B, and C on a number line, where A represents √7, B represents 1,
    and C is symmetric to A with respect to B, prove that C represents 2 - √7. -/
theorem symmetric_point_on_number_line (A B C : ℝ) : 
  A = Real.sqrt 7 → B = 1 → (A + C) / 2 = B → C = 2 - Real.sqrt 7 := by
  sorry

end symmetric_point_on_number_line_l3536_353686


namespace f_fixed_points_l3536_353655

def f (x : ℝ) : ℝ := x^2 - 5*x

theorem f_fixed_points : 
  {x : ℝ | f (f x) = f x} = {0, -2, 5, 6} := by sorry

end f_fixed_points_l3536_353655


namespace sqrt_equation_solution_l3536_353669

theorem sqrt_equation_solution : ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 := by
  sorry

end sqrt_equation_solution_l3536_353669


namespace jacks_age_problem_l3536_353639

theorem jacks_age_problem (jack_age_2010 : ℕ) (mother_age_multiplier : ℕ) : 
  jack_age_2010 = 12 →
  mother_age_multiplier = 3 →
  ∃ (years_after_2010 : ℕ), 
    (jack_age_2010 + years_after_2010) * 2 = (jack_age_2010 * mother_age_multiplier + years_after_2010) ∧
    years_after_2010 = 12 :=
by sorry

end jacks_age_problem_l3536_353639
