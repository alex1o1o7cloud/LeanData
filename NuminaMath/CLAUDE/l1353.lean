import Mathlib

namespace correct_ticket_count_l1353_135321

/-- The number of stations between Ernakulam and Chennai -/
def num_stations : ℕ := 50

/-- The number of different train routes -/
def num_routes : ℕ := 3

/-- The number of second class tickets needed for one route -/
def tickets_per_route : ℕ := num_stations * (num_stations - 1) / 2

/-- The total number of second class tickets needed for all routes -/
def total_tickets : ℕ := num_routes * tickets_per_route

theorem correct_ticket_count : total_tickets = 3675 := by
  sorry

end correct_ticket_count_l1353_135321


namespace cameron_tour_theorem_l1353_135347

def cameron_tour_problem (questions_per_tourist : ℕ) (total_tours : ℕ) 
  (group1_size : ℕ) (group2_size : ℕ) (group3_size : ℕ) 
  (inquisitive_factor : ℕ) (total_questions : ℕ) : Prop :=
  let group1_questions := group1_size * questions_per_tourist
  let group2_questions := group2_size * questions_per_tourist
  let group3_questions := group3_size * questions_per_tourist + 
                          (inquisitive_factor - 1) * questions_per_tourist
  let remaining_questions := total_questions - (group1_questions + group2_questions + group3_questions)
  let last_group_size := remaining_questions / questions_per_tourist
  last_group_size = 7

theorem cameron_tour_theorem : 
  cameron_tour_problem 2 4 6 11 8 3 68 := by
  sorry

end cameron_tour_theorem_l1353_135347


namespace quadratic_real_roots_l1353_135363

theorem quadratic_real_roots (k m : ℝ) : 
  (∃ x : ℝ, x^2 + (2*k - 3*m)*x + (k^2 - 5*k*m + 6*m^2) = 0) ↔ k ≥ (15/8)*m :=
by sorry

end quadratic_real_roots_l1353_135363


namespace kids_bike_wheels_count_l1353_135338

/-- The number of wheels on a kid's bike -/
def kids_bike_wheels : ℕ := 4

/-- The number of regular bikes -/
def regular_bikes : ℕ := 7

/-- The number of children's bikes -/
def children_bikes : ℕ := 11

/-- The number of wheels on a regular bike -/
def regular_bike_wheels : ℕ := 2

/-- The total number of wheels observed -/
def total_wheels : ℕ := 58

theorem kids_bike_wheels_count :
  regular_bikes * regular_bike_wheels + children_bikes * kids_bike_wheels = total_wheels :=
by sorry

end kids_bike_wheels_count_l1353_135338


namespace value_of_expression_l1353_135332

theorem value_of_expression (m n : ℤ) (h : m - n = 1) : (m - n)^2 - 2*m + 2*n = -1 := by
  sorry

end value_of_expression_l1353_135332


namespace m_range_l1353_135362

theorem m_range : 
  (∀ x, (|x - m| < 1 ↔ 1/3 < x ∧ x < 1/2)) → 
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
by sorry

end m_range_l1353_135362


namespace greatest_whole_number_inequality_l1353_135366

theorem greatest_whole_number_inequality :
  ∀ x : ℤ, (5 * x - 4 < 3 - 2 * x) → x ≤ 0 :=
by
  sorry

end greatest_whole_number_inequality_l1353_135366


namespace circle_and_line_properties_l1353_135354

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x = 0 ∨ 3*x - 4*y - 8 = 0

-- Theorem statement
theorem circle_and_line_properties :
  ∃ (a : ℝ),
    -- Circle C has its center on the x-axis
    (∀ x y : ℝ, circle_C x y → y = 0) ∧
    -- Circle C passes through the point (0, √3)
    circle_C 0 (Real.sqrt 3) ∧
    -- Circle C is tangent to the line x=-1
    (∀ y : ℝ, circle_C (-1) y → (x : ℝ) → x = -1 → (circle_C x y → x = -1)) ∧
    -- Line l passes through the point (0,-2)
    line_l 0 (-2) ∧
    -- The chord intercepted by circle C on line l has a length of 2√3
    (∃ x₁ y₁ x₂ y₂ : ℝ, 
      line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ 
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 12) :=
by
  sorry

end circle_and_line_properties_l1353_135354


namespace inequality_and_range_l1353_135309

theorem inequality_and_range (a b c m : ℝ) 
  (h1 : a + b + c + 2 - 2*m = 0)
  (h2 : a^2 + (1/4)*b^2 + (1/9)*c^2 + m - 1 = 0) :
  (a^2 + (1/4)*b^2 + (1/9)*c^2 ≥ (a + b + c)^2 / 14) ∧ 
  (-5/2 ≤ m ∧ m ≤ 1) := by
sorry

end inequality_and_range_l1353_135309


namespace set_membership_and_inclusion_l1353_135328

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2*k + 1}

theorem set_membership_and_inclusion :
  (8 ∈ A ∧ 9 ∈ A ∧ 10 ∉ A) ∧ (∀ x : ℤ, x ∈ A → x ∈ B) := by sorry

end set_membership_and_inclusion_l1353_135328


namespace amelia_wins_probability_l1353_135319

/-- Probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 1/4

/-- Probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 3/7

/-- Maximum number of rounds -/
def max_rounds : ℕ := 5

/-- The probability that Amelia wins the coin toss game -/
def amelia_wins_prob : ℚ := 223/784

/-- Theorem stating that the probability of Amelia winning is 223/784 -/
theorem amelia_wins_probability : 
  amelia_wins_prob = p_amelia * (1 - p_blaine) + 
    (1 - p_amelia) * (1 - p_blaine) * p_amelia * (1 - p_blaine) + 
    (1 - p_amelia) * (1 - p_blaine) * (1 - p_amelia) * (1 - p_blaine) * p_amelia := by
  sorry

#check amelia_wins_probability

end amelia_wins_probability_l1353_135319


namespace square_of_six_y_minus_four_l1353_135301

theorem square_of_six_y_minus_four (y : ℝ) (h : 3 * y^2 + 6 = 5 * y + 15) : 
  (6 * y - 4)^2 = 134 := by
  sorry

end square_of_six_y_minus_four_l1353_135301


namespace percentage_of_male_employees_l1353_135360

theorem percentage_of_male_employees
  (total_employees : ℕ)
  (males_below_50 : ℕ)
  (h_total : total_employees = 5200)
  (h_below_50 : males_below_50 = 1170)
  (h_half_above_50 : males_below_50 = (total_employees * (percentage_males / 100) / 2)) :
  percentage_males = 45 :=
by
  sorry

end percentage_of_male_employees_l1353_135360


namespace sales_difference_l1353_135326

-- Define the regular day sales quantities
def regular_croissants : ℕ := 10
def regular_muffins : ℕ := 10
def regular_sourdough : ℕ := 6
def regular_wholewheat : ℕ := 4

-- Define the Monday sales quantities
def monday_croissants : ℕ := 8
def monday_muffins : ℕ := 6
def monday_sourdough : ℕ := 15
def monday_wholewheat : ℕ := 10

-- Define the regular prices
def price_croissant : ℚ := 2.5
def price_muffin : ℚ := 1.75
def price_sourdough : ℚ := 4.25
def price_wholewheat : ℚ := 5

-- Define the discount rate
def discount_rate : ℚ := 0.1

-- Calculate the daily average sales
def daily_average : ℚ :=
  regular_croissants * price_croissant +
  regular_muffins * price_muffin +
  regular_sourdough * price_sourdough +
  regular_wholewheat * price_wholewheat

-- Calculate the Monday sales with discount
def monday_sales : ℚ :=
  monday_croissants * price_croissant * (1 - discount_rate) +
  monday_muffins * price_muffin * (1 - discount_rate) +
  monday_sourdough * price_sourdough * (1 - discount_rate) +
  monday_wholewheat * price_wholewheat * (1 - discount_rate)

-- State the theorem
theorem sales_difference : monday_sales - daily_average = 41.825 := by sorry

end sales_difference_l1353_135326


namespace smiley_face_tulips_l1353_135343

theorem smiley_face_tulips : 
  let red_tulips_per_eye : ℕ := 8
  let red_tulips_for_smile : ℕ := 18
  let yellow_tulips_multiplier : ℕ := 9
  let number_of_eyes : ℕ := 2

  let total_red_tulips : ℕ := red_tulips_per_eye * number_of_eyes + red_tulips_for_smile
  let total_yellow_tulips : ℕ := yellow_tulips_multiplier * red_tulips_for_smile
  let total_tulips : ℕ := total_red_tulips + total_yellow_tulips

  total_tulips = 196 := by sorry

end smiley_face_tulips_l1353_135343


namespace max_value_problem_l1353_135303

theorem max_value_problem (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (a / (a + 1)) + (b / (b + 2)) ≤ (5 - 2 * Real.sqrt 2) / 4 := by
  sorry

end max_value_problem_l1353_135303


namespace percent_greater_relative_to_sum_l1353_135304

/-- Given two real numbers M and N, this theorem states that the percentage
    by which M is greater than N, relative to their sum, is (100(M-N))/(M+N). -/
theorem percent_greater_relative_to_sum (M N : ℝ) :
  (M - N) / (M + N) * 100 = (100 * (M - N)) / (M + N) := by sorry

end percent_greater_relative_to_sum_l1353_135304


namespace root_implies_t_value_l1353_135330

theorem root_implies_t_value (t : ℝ) : 
  (3 * (((-15 - Real.sqrt 145) / 6) ^ 2) + 15 * ((-15 - Real.sqrt 145) / 6) + t = 0) → 
  t = 20 / 3 := by
sorry

end root_implies_t_value_l1353_135330


namespace total_pencils_l1353_135388

/-- Given the number of pencils in different locations, prove the total number of pencils. -/
theorem total_pencils (drawer : ℕ) (desk_initial : ℕ) (desk_added : ℕ) :
  drawer = 43 →
  desk_initial = 19 →
  desk_added = 16 →
  drawer + desk_initial + desk_added = 78 :=
by sorry

end total_pencils_l1353_135388


namespace forgot_lawns_count_l1353_135396

def lawn_problem (total_lawns : ℕ) (earnings_per_lawn : ℕ) (actual_earnings : ℕ) : ℕ :=
  total_lawns - (actual_earnings / earnings_per_lawn)

theorem forgot_lawns_count :
  lawn_problem 17 4 32 = 9 := by
  sorry

end forgot_lawns_count_l1353_135396


namespace proposition_equivalence_l1353_135384

theorem proposition_equivalence (A : Set α) (x y : α) :
  (x ∈ A → y ∉ A) ↔ (y ∈ A → x ∉ A) := by
  sorry

end proposition_equivalence_l1353_135384


namespace complement_B_union_A_when_a_is_1_A_subset_B_iff_a_in_range_l1353_135375

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2*x + a ∧ 2*x + a ≤ 3}
def B : Set ℝ := {x | -1/2 < x ∧ x < 2}

-- Theorem for part (1)
theorem complement_B_union_A_when_a_is_1 :
  (Set.univ \ B) ∪ A 1 = {x | x ≤ 1 ∨ x ≥ 2} := by sorry

-- Theorem for part (2)
theorem A_subset_B_iff_a_in_range (a : ℝ) :
  A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end complement_B_union_A_when_a_is_1_A_subset_B_iff_a_in_range_l1353_135375


namespace age_problem_l1353_135327

theorem age_problem (A B C : ℕ) : 
  (A + B + C) / 3 = 26 →
  (A + C) / 2 = 29 →
  B = 20 := by
sorry

end age_problem_l1353_135327


namespace wall_width_proof_l1353_135358

theorem wall_width_proof (width height length volume : ℝ) : 
  height = 6 * width →
  length = 7 * height →
  volume = length * width * height →
  volume = 16128 →
  width = (384 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end wall_width_proof_l1353_135358


namespace bernardo_wins_with_92_l1353_135318

def game_sequence (M : ℕ) : ℕ → ℕ 
| 0 => M
| 1 => 3 * M
| 2 => 3 * M + 40
| 3 => 9 * M + 120
| 4 => 9 * M + 160
| 5 => 27 * M + 480
| _ => 0

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bernardo_wins_with_92 :
  ∃ (M : ℕ), 
    M ≥ 1 ∧ 
    M ≤ 1000 ∧ 
    game_sequence M 5 < 3000 ∧ 
    game_sequence M 5 + 40 ≥ 3000 ∧
    sum_of_digits M = 11 ∧
    (∀ (N : ℕ), N < M → 
      (game_sequence N 5 < 3000 → game_sequence N 5 + 40 < 3000) ∨ 
      game_sequence N 5 ≥ 3000) :=
by
  use 92
  sorry

#eval game_sequence 92 5  -- Should output 2964
#eval game_sequence 92 5 + 40  -- Should output 3004
#eval sum_of_digits 92  -- Should output 11

end bernardo_wins_with_92_l1353_135318


namespace quadratic_equation_unique_solution_l1353_135345

/-- The quadratic equation ax^2 - x - 1 = 0 has exactly one solution in the interval (0, 1) if and only if a > 2 -/
theorem quadratic_equation_unique_solution (a : ℝ) : 
  (∃! x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 - x - 1 = 0) ↔ a > 2 :=
by sorry

end quadratic_equation_unique_solution_l1353_135345


namespace rectangle_area_difference_l1353_135353

/-- Given a large rectangle of dimensions A × B containing a smaller rectangle of dimensions a × b,
    the difference between the total area of yellow regions and green regions is A × b - a × B. -/
theorem rectangle_area_difference (A B a b : ℝ) (hA : A > 0) (hB : B > 0) (ha : a > 0) (hb : b > 0)
  (ha_le_A : a ≤ A) (hb_le_B : b ≤ B) :
  A * b - a * B = A * b - a * B := by sorry

end rectangle_area_difference_l1353_135353


namespace unknown_number_problem_l1353_135383

theorem unknown_number_problem (x : ℚ) : 
  x + (2/3) * x - (1/3) * (x + (2/3) * x) = 10 ↔ x = 9 := by
  sorry

end unknown_number_problem_l1353_135383


namespace constant_function_invariant_l1353_135379

-- Define g as a function from ℝ to ℝ
def g : ℝ → ℝ := λ x => 5

-- Theorem statement
theorem constant_function_invariant (x : ℝ) : g (3 * x - 7) = 5 := by
  sorry

end constant_function_invariant_l1353_135379


namespace lcm_factor_proof_l1353_135308

theorem lcm_factor_proof (A B : ℕ) (x : ℕ) (h1 : Nat.gcd A B = 23) (h2 : A = 391) 
  (h3 : Nat.lcm A B = 23 * 17 * x) : x = 17 := by
  sorry

end lcm_factor_proof_l1353_135308


namespace power_equation_solution_l1353_135310

theorem power_equation_solution : ∃ x : ℕ, 2^4 + 3 = 5^2 - x ∧ x = 6 := by
  sorry

end power_equation_solution_l1353_135310


namespace sin_300_degrees_l1353_135376

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l1353_135376


namespace number_divided_by_eight_l1353_135398

theorem number_divided_by_eight : ∀ x : ℝ, x / 8 = 4 → x = 32 := by sorry

end number_divided_by_eight_l1353_135398


namespace moon_speed_conversion_l1353_135370

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 1.02

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 3672 := by
  sorry

end moon_speed_conversion_l1353_135370


namespace andy_final_position_l1353_135359

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Represents the state of Andy the Ant -/
structure AntState where
  position : Point
  direction : Direction
  moveCount : Nat

/-- Performs a single move for Andy the Ant -/
def move (state : AntState) : AntState :=
  sorry

/-- Performs n moves for Andy the Ant -/
def moveN (n : Nat) (state : AntState) : AntState :=
  sorry

/-- The main theorem to prove -/
theorem andy_final_position :
  let initialState : AntState := {
    position := { x := -10, y := 10 },
    direction := Direction.East,
    moveCount := 0
  }
  let finalState := moveN 2030 initialState
  finalState.position = { x := -3054, y := 3053 } :=
sorry

end andy_final_position_l1353_135359


namespace largest_n_divisibility_l1353_135316

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > n → ¬(m + 12 ∣ m^3 + 144)) ∧ 
  (n + 12 ∣ n^3 + 144) ∧ 
  n = 132 := by
  sorry

end largest_n_divisibility_l1353_135316


namespace reflection_sum_l1353_135387

/-- Given that the point (-4, 2) is reflected across the line y = mx + b to the point (6, -2),
    prove that m + b = 0 -/
theorem reflection_sum (m b : ℝ) : 
  (∃ (x y : ℝ), y = m * x + b ∧ 
   (x - (-4))^2 + (y - 2)^2 = (x - 6)^2 + (y - (-2))^2 ∧
   (x - (-4)) * (6 - x) + (y - 2) * (-2 - y) = 0) →
  m + b = 0 := by
sorry

end reflection_sum_l1353_135387


namespace linear_equation_solution_l1353_135374

theorem linear_equation_solution (x y : ℝ) : 5 * x + y = 4 → y = 4 - 5 * x := by
  sorry

end linear_equation_solution_l1353_135374


namespace binomial_rv_p_value_l1353_135361

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean : ℝ
  std_dev : ℝ

/-- Theorem: For a binomial random variable with mean 200 and standard deviation 10, p = 1/2 -/
theorem binomial_rv_p_value (X : BinomialRV) 
  (h_mean : X.mean = 200)
  (h_std_dev : X.std_dev = 10) :
  X.p = 1/2 := by
  sorry

end binomial_rv_p_value_l1353_135361


namespace tan_22_5_deg_identity_l1353_135368

theorem tan_22_5_deg_identity : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by sorry

end tan_22_5_deg_identity_l1353_135368


namespace square_roots_equality_l1353_135334

theorem square_roots_equality (m : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (2*m - 4)^2 = k ∧ (3*m - 1)^2 = k) → (m = -3 ∨ m = 1) :=
by sorry

end square_roots_equality_l1353_135334


namespace speed_conversion_l1353_135378

/-- Conversion factor from kilometers per hour to meters per second -/
def kmph_to_mps : ℚ := 5 / 18

/-- The given speed in kilometers per hour -/
def speed_kmph : ℚ := 216

/-- The speed in meters per second -/
def speed_mps : ℚ := speed_kmph * kmph_to_mps

theorem speed_conversion :
  speed_mps = 60 := by sorry

end speed_conversion_l1353_135378


namespace cubic_expression_equality_l1353_135339

theorem cubic_expression_equality (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 73/3 := by
  sorry

end cubic_expression_equality_l1353_135339


namespace fraction_product_simplification_l1353_135364

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end fraction_product_simplification_l1353_135364


namespace math_english_time_difference_l1353_135315

/-- Represents an exam with a number of questions and a duration in hours -/
structure Exam where
  questions : ℕ
  duration : ℚ

/-- Calculates the time per question in minutes for a given exam -/
def timePerQuestion (e : Exam) : ℚ :=
  (e.duration * 60) / e.questions

theorem math_english_time_difference :
  let english : Exam := { questions := 30, duration := 1 }
  let math : Exam := { questions := 15, duration := 1.5 }
  timePerQuestion math - timePerQuestion english = 4 := by
  sorry

end math_english_time_difference_l1353_135315


namespace square_root_divided_by_15_equals_4_l1353_135313

theorem square_root_divided_by_15_equals_4 (x : ℝ) : 
  (Real.sqrt x) / 15 = 4 → x = 3600 := by
  sorry

end square_root_divided_by_15_equals_4_l1353_135313


namespace smallest_y_for_inequality_l1353_135331

theorem smallest_y_for_inequality : ∃ (y : ℕ), y > 0 ∧ (y^6 : ℚ) / (y^3 : ℚ) > 80 ∧ ∀ (z : ℕ), z > 0 → (z^6 : ℚ) / (z^3 : ℚ) > 80 → y ≤ z :=
by sorry

end smallest_y_for_inequality_l1353_135331


namespace empty_box_weight_l1353_135306

def box_weight_problem (initial_weight : ℝ) (half_removed_weight : ℝ) : Prop :=
  ∃ (apple_weight : ℝ) (num_apples : ℕ) (box_weight : ℝ),
    initial_weight = box_weight + apple_weight * num_apples ∧
    half_removed_weight = box_weight + apple_weight * (num_apples / 2) ∧
    box_weight = 1

theorem empty_box_weight :
  box_weight_problem 9 5 := by
  sorry

end empty_box_weight_l1353_135306


namespace downward_parabola_m_range_l1353_135333

/-- A parabola that opens downwards -/
structure DownwardParabola where
  m : ℝ
  eq : ℝ → ℝ := fun x ↦ (m + 3) * x^2 + 1
  opens_downward : m + 3 < 0

/-- The range of m for a downward opening parabola -/
theorem downward_parabola_m_range (p : DownwardParabola) : p.m < -3 := by
  sorry

end downward_parabola_m_range_l1353_135333


namespace geometric_sequence_problem_l1353_135394

theorem geometric_sequence_problem (a b c r : ℤ) : 
  (b = a * r ∧ c = a * r^2) →  -- geometric sequence condition
  (r ≠ 0) →                   -- non-zero ratio
  (c = a + 56) →              -- given condition
  b = 21 := by
sorry

end geometric_sequence_problem_l1353_135394


namespace max_volume_triangular_pyramid_l1353_135342

/-- A triangular pyramid with vertex S and base ABC -/
structure TriangularPyramid where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

/-- The volume of a triangular pyramid -/
def volume (t : TriangularPyramid) : ℝ := sorry

/-- The conditions given in the problem -/
def satisfiesConditions (t : TriangularPyramid) : Prop :=
  t.SA = 4 ∧
  t.SB ≥ 7 ∧
  t.SC ≥ 9 ∧
  t.AB = 5 ∧
  t.BC ≤ 6 ∧
  t.AC ≤ 8

/-- The theorem stating the maximum volume of the triangular pyramid -/
theorem max_volume_triangular_pyramid :
  ∀ t : TriangularPyramid, satisfiesConditions t → volume t ≤ 8 * Real.sqrt 6 :=
sorry

end max_volume_triangular_pyramid_l1353_135342


namespace length_ad_is_12_95_l1353_135348

/-- A quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side CD -/
  cd : ℝ
  /-- Angle B in radians -/
  angle_b : ℝ
  /-- Angle C in radians -/
  angle_c : ℝ
  /-- Condition: AB = 6 -/
  hab : ab = 6
  /-- Condition: BC = 8 -/
  hbc : bc = 8
  /-- Condition: CD = 15 -/
  hcd : cd = 15
  /-- Condition: Angle B is obtuse -/
  hb_obtuse : π / 2 < angle_b ∧ angle_b < π
  /-- Condition: Angle C is obtuse -/
  hc_obtuse : π / 2 < angle_c ∧ angle_c < π
  /-- Condition: sin C = 4/5 -/
  hsin_c : Real.sin angle_c = 4/5
  /-- Condition: cos B = -4/5 -/
  hcos_b : Real.cos angle_b = -4/5

/-- The length of side AD in the quadrilateral ABCD -/
def lengthAD (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that the length of side AD is 12.95 -/
theorem length_ad_is_12_95 (q : Quadrilateral) : lengthAD q = 12.95 := by
  sorry

end length_ad_is_12_95_l1353_135348


namespace evie_shell_collection_l1353_135324

theorem evie_shell_collection (daily_shells : ℕ) : 
  (6 * daily_shells - 2 = 58) → daily_shells = 10 := by
  sorry

end evie_shell_collection_l1353_135324


namespace absolute_value_equation_unique_solution_l1353_135346

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 9| = |x - 3| := by
sorry

end absolute_value_equation_unique_solution_l1353_135346


namespace ibrahim_purchase_l1353_135385

/-- The amount of money Ibrahim lacks to purchase an MP3 player and a CD -/
def money_lacking (mp3_cost cd_cost savings father_contribution : ℕ) : ℕ :=
  (mp3_cost + cd_cost) - (savings + father_contribution)

/-- Theorem: Ibrahim lacks 64 euros -/
theorem ibrahim_purchase :
  money_lacking 120 19 55 20 = 64 := by
  sorry

end ibrahim_purchase_l1353_135385


namespace inequality_and_equality_condition_l1353_135314

theorem inequality_and_equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt 2 * (Real.sqrt (a * (a + b)^3) + b * Real.sqrt (a^2 + b^2)) ≤ 3 * (a^2 + b^2) ∧
  (Real.sqrt 2 * (Real.sqrt (a * (a + b)^3) + b * Real.sqrt (a^2 + b^2)) = 3 * (a^2 + b^2) ↔ a = b) :=
by sorry

end inequality_and_equality_condition_l1353_135314


namespace range_of_A_l1353_135337

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_A : ∀ a : ℝ, a ∈ A ↔ -1 ≤ a ∧ a ≤ 3 := by
  sorry

end range_of_A_l1353_135337


namespace choir_members_count_l1353_135399

theorem choir_members_count :
  ∃ n : ℕ,
    (n + 4) % 10 = 0 ∧
    (n + 5) % 11 = 0 ∧
    200 < n ∧
    n < 300 ∧
    n = 226 := by
  sorry

end choir_members_count_l1353_135399


namespace quartic_polynomial_unique_l1353_135305

/-- A monic quartic polynomial with real coefficients -/
def QuarticPolynomial (a b c d : ℝ) : ℝ → ℂ :=
  fun x ↦ (x^4 : ℂ) + a * (x^3 : ℂ) + b * (x^2 : ℂ) + c * (x : ℂ) + d

theorem quartic_polynomial_unique
  (q : ℝ → ℂ)
  (h_monic : ∀ x, q x = (x^4 : ℂ) + (a * x^3 : ℂ) + (b * x^2 : ℂ) + (c * x : ℂ) + d)
  (h_root : q (5 - 3*I) = 0)
  (h_constant : q 0 = -150) :
  q = QuarticPolynomial (-658/34) (19206/34) (-3822/17) (-150) :=
by sorry

end quartic_polynomial_unique_l1353_135305


namespace eggs_produced_this_year_l1353_135340

/-- Calculates the total egg production for this year given last year's production and additional eggs produced. -/
def total_eggs_this_year (last_year_production additional_eggs : ℕ) : ℕ :=
  last_year_production + additional_eggs

/-- Theorem stating that the total eggs produced this year is 4636. -/
theorem eggs_produced_this_year : 
  total_eggs_this_year 1416 3220 = 4636 := by
  sorry

end eggs_produced_this_year_l1353_135340


namespace tangent_line_implies_a_value_l1353_135323

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_implies_a_value (a : ℝ) :
  (∀ x, x ≠ 0 → (f a x - f a 0) / (x - 0) ≤ 2) ∧
  (∀ x, x ≠ 0 → (f a x - f a 0) / (x - 0) ≥ 2) →
  a = 2 := by
  sorry

end tangent_line_implies_a_value_l1353_135323


namespace mitch_hourly_rate_l1353_135367

/-- Mitch's hourly rate calculation --/
theorem mitch_hourly_rate :
  ∀ (weekday_hours_per_day : ℕ) 
    (weekend_hours_per_day : ℕ) 
    (weekday_count : ℕ) 
    (weekend_count : ℕ) 
    (weekend_multiplier : ℕ) 
    (weekly_earnings : ℕ),
  weekday_hours_per_day = 5 →
  weekend_hours_per_day = 3 →
  weekday_count = 5 →
  weekend_count = 2 →
  weekend_multiplier = 2 →
  weekly_earnings = 111 →
  (weekly_earnings : ℚ) / 
    (weekday_hours_per_day * weekday_count + 
     weekend_hours_per_day * weekend_count * weekend_multiplier) = 3 := by
  sorry

#check mitch_hourly_rate

end mitch_hourly_rate_l1353_135367


namespace fair_haired_women_percentage_l1353_135365

/-- Given a company where:
  * 20% of employees are women with fair hair
  * 50% of employees have fair hair
  Prove that 40% of fair-haired employees are women -/
theorem fair_haired_women_percentage
  (total_employees : ℕ)
  (women_fair_hair_percent : ℚ)
  (fair_hair_percent : ℚ)
  (h1 : women_fair_hair_percent = 20 / 100)
  (h2 : fair_hair_percent = 50 / 100) :
  (women_fair_hair_percent * total_employees) / (fair_hair_percent * total_employees) = 40 / 100 :=
sorry

end fair_haired_women_percentage_l1353_135365


namespace angle_DAE_is_10_degrees_l1353_135386

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem angle_DAE_is_10_degrees 
  (A B C D O E: ℝ × ℝ) 
  (triangle : Triangle A B C) 
  (h1 : angle A C B = 60)
  (h2 : angle C B A = 70)
  (h3 : D.1 = B.1 + (C.1 - B.1) * ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / ((C.1 - B.1)^2 + (C.2 - B.2)^2))
  (h4 : D.2 = B.2 + (C.2 - B.2) * ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / ((C.1 - B.1)^2 + (C.2 - B.2)^2))
  (h5 : ∃ r, Circle O r = {A, B, C})
  (h6 : E.1 = 2 * O.1 - A.1 ∧ E.2 = 2 * O.2 - A.2) :
  angle D A E = 10 := by
sorry


end angle_DAE_is_10_degrees_l1353_135386


namespace election_invalid_votes_l1353_135395

theorem election_invalid_votes 
  (total_polled : ℕ) 
  (losing_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h_total : total_polled = 90083)
  (h_losing : losing_percentage = 45 / 100)
  (h_difference : vote_difference = 9000) :
  total_polled - (vote_difference / (1/2 - losing_percentage)) = 83 := by
sorry

end election_invalid_votes_l1353_135395


namespace rectangle_area_sum_l1353_135377

theorem rectangle_area_sum (a b c d : ℝ) 
  (ha : a = 20) (hb : b = 40) (hc : c = 48) (hd : d = 42) :
  a + b + c + d = 150 := by
  sorry

end rectangle_area_sum_l1353_135377


namespace cube_symmetry_properties_change_l1353_135371

/-- Represents the symmetrical properties of a geometric object -/
structure SymmetryProperties where
  planes : ℕ
  axes : ℕ
  center : Bool

/-- Represents the different painting configurations of a cube -/
inductive CubePainting
  | Unpainted
  | OneFace
  | TwoFacesParallel
  | TwoFacesAdjacent
  | ThreeFacesMeetingAtVertex
  | ThreeFacesNotMeetingAtVertex

/-- Returns the symmetry properties for a given cube painting configuration -/
def symmetryPropertiesForCube (painting : CubePainting) : SymmetryProperties :=
  match painting with
  | .Unpainted => { planes := 9, axes := 9, center := true }
  | .OneFace => { planes := 4, axes := 1, center := false }
  | .TwoFacesParallel => { planes := 5, axes := 3, center := true }
  | .TwoFacesAdjacent => { planes := 2, axes := 1, center := false }
  | .ThreeFacesMeetingAtVertex => { planes := 3, axes := 0, center := false }
  | .ThreeFacesNotMeetingAtVertex => { planes := 2, axes := 1, center := false }

theorem cube_symmetry_properties_change (painting : CubePainting) :
  symmetryPropertiesForCube painting ≠ symmetryPropertiesForCube CubePainting.Unpainted :=
by sorry

end cube_symmetry_properties_change_l1353_135371


namespace consecutive_non_multiple_of_five_product_l1353_135397

theorem consecutive_non_multiple_of_five_product (k : ℤ) :
  (∃ m : ℤ, (5*k + 1) * (5*k + 2) * (5*k + 3) = 5*m + 1) ∨
  (∃ n : ℤ, (5*k + 2) * (5*k + 3) * (5*k + 4) = 5*n - 1) :=
by sorry

end consecutive_non_multiple_of_five_product_l1353_135397


namespace complement_A_in_U_l1353_135300

def U : Set ℝ := {x | x^2 ≤ 4}
def A : Set ℝ := {x | |x + 1| ≤ 1}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end complement_A_in_U_l1353_135300


namespace stratified_sampling_male_count_l1353_135320

theorem stratified_sampling_male_count :
  let total_students : ℕ := 980
  let male_students : ℕ := 560
  let sample_size : ℕ := 280
  let sample_ratio : ℚ := sample_size / total_students
  sample_ratio * male_students = 160 := by sorry

end stratified_sampling_male_count_l1353_135320


namespace manicure_total_cost_l1353_135389

theorem manicure_total_cost (manicure_cost : ℝ) (tip_percentage : ℝ) : 
  manicure_cost = 30 →
  tip_percentage = 30 →
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := by
  sorry

end manicure_total_cost_l1353_135389


namespace angle_CAD_is_15_degrees_l1353_135351

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Represents a square in 2D space -/
structure Square where
  B : Point2D
  C : Point2D
  D : Point2D
  E : Point2D

/-- Calculates the angle between three points in degrees -/
def angle (p1 p2 p3 : Point2D) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if a quadrilateral is a square -/
def isSquare (s : Square) : Prop := sorry

/-- Theorem: In a coplanar configuration where ABC is an equilateral triangle 
    and BCDE is a square, the measure of angle CAD is 15 degrees -/
theorem angle_CAD_is_15_degrees 
  (A B C D E : Point2D) 
  (triangle : Triangle) 
  (square : Square) : 
  triangle.A = A ∧ triangle.B = B ∧ triangle.C = C ∧
  square.B = B ∧ square.C = C ∧ square.D = D ∧ square.E = E ∧
  isEquilateral triangle ∧ 
  isSquare square → 
  angle C A D = 15 := by
  sorry

end angle_CAD_is_15_degrees_l1353_135351


namespace greatest_consecutive_integers_sum_91_l1353_135382

theorem greatest_consecutive_integers_sum_91 :
  (∀ n : ℕ, n > 182 → ¬ (∃ a : ℤ, (Finset.range n).sum (λ i => a + i) = 91)) ∧
  (∃ a : ℤ, (Finset.range 182).sum (λ i => a + i) = 91) :=
by sorry

end greatest_consecutive_integers_sum_91_l1353_135382


namespace subway_construction_equation_l1353_135336

/-- Represents the subway construction scenario -/
structure SubwayConstruction where
  total_length : ℝ
  extra_meters_per_day : ℝ
  days_saved : ℝ
  original_plan : ℝ

/-- The equation holds for the given subway construction scenario -/
def equation_holds (sc : SubwayConstruction) : Prop :=
  sc.total_length / sc.original_plan - sc.total_length / (sc.original_plan + sc.extra_meters_per_day) = sc.days_saved

/-- Theorem stating that the equation holds for the specific scenario described in the problem -/
theorem subway_construction_equation :
  ∀ (sc : SubwayConstruction),
    sc.total_length = 120 ∧
    sc.extra_meters_per_day = 5 ∧
    sc.days_saved = 4 →
    equation_holds sc :=
by
  sorry

#check subway_construction_equation

end subway_construction_equation_l1353_135336


namespace half_perimeter_area_rectangle_existence_l1353_135369

/-- Given a rectangle with sides a and b, this theorem proves the existence of another rectangle
    with half the perimeter and half the area, based on the discriminant of the resulting quadratic equation. -/
theorem half_perimeter_area_rectangle_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = (a + b) / 2 ∧ x * y = (a * b) / 2 ↔
  ((a + b)^2 - 4 * (a * b)) ≥ 0 :=
by sorry

end half_perimeter_area_rectangle_existence_l1353_135369


namespace parallel_iff_a_eq_one_l1353_135380

/-- Two lines in the form ax - y + b = 0 and cx - y + d = 0 are parallel if and only if a = c -/
def are_parallel (a c : ℝ) : Prop := a = c

/-- The condition for the given lines to be parallel -/
def parallel_condition (a : ℝ) : Prop := are_parallel a (1/a)

theorem parallel_iff_a_eq_one (a : ℝ) : 
  parallel_condition a ↔ a = 1 :=
sorry

end parallel_iff_a_eq_one_l1353_135380


namespace arithmetic_sequence_common_difference_l1353_135349

/-- 
Given an arithmetic sequence {a_n} with first term a₁ = 19 and integer common difference d,
if the 6th term is negative and the 5th term is non-negative, then the common difference is -4.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h1 : a 1 = 19)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : a 6 < 0)
  (h4 : a 5 ≥ 0) :
  d = -4 := by
  sorry

end arithmetic_sequence_common_difference_l1353_135349


namespace binomial_multiplication_l1353_135352

theorem binomial_multiplication (x : ℝ) : (4*x + 3) * (2*x - 7) = 8*x^2 - 22*x - 21 := by
  sorry

end binomial_multiplication_l1353_135352


namespace isosceles_triangle_area_l1353_135393

/-- An isosceles triangle with specific measurements -/
structure IsoscelesTriangle where
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The inradius of the triangle -/
  inradius : ℝ
  /-- One of the angles of the triangle in degrees -/
  angle : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : Bool
  /-- The perimeter is 20 cm -/
  perimeterIs20 : perimeter = 20
  /-- The inradius is 2.5 cm -/
  inradiusIs2_5 : inradius = 2.5
  /-- One angle is 40 degrees -/
  angleIs40 : angle = 40
  /-- The triangle is confirmed to be isosceles -/
  isIsoscelesTrue : isIsosceles = true

/-- The area of the isosceles triangle is 25 cm² -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : 
  t.inradius * (t.perimeter / 2) = 25 := by
  sorry

end isosceles_triangle_area_l1353_135393


namespace final_value_calculation_l1353_135322

theorem final_value_calculation : 
  let initial_value := 52
  let first_increase := initial_value * 1.20
  let second_decrease := first_increase * 0.90
  let final_increase := second_decrease * 1.15
  final_increase = 64.584 := by
sorry

end final_value_calculation_l1353_135322


namespace reflection_composition_l1353_135392

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ := 
  let p' := (p.1, p.2 - 2)
  let p'' := (p'.2, p'.1)
  (p''.1, p''.2 + 2)

theorem reflection_composition (D : ℝ × ℝ) (h : D = (5, 2)) : 
  reflect_line (reflect_x D) = (-4, 7) := by sorry

end reflection_composition_l1353_135392


namespace f_properties_l1353_135329

-- Define the function f(x) = lg |sin x|
noncomputable def f (x : ℝ) : ℝ := Real.log (|Real.sin x|)

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = f x) ∧                        -- f is even
  (∀ x, f (x + π) = f x) ∧                     -- f has period π
  (∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y) -- f is monotonically increasing on (0, π/2)
  := by sorry

end f_properties_l1353_135329


namespace expected_potato_yield_l1353_135357

/-- Calculates the expected potato yield from a rectangular garden --/
theorem expected_potato_yield
  (length_steps : ℕ)
  (width_steps : ℕ)
  (step_length : ℝ)
  (yield_per_sqft : ℝ)
  (h1 : length_steps = 18)
  (h2 : width_steps = 25)
  (h3 : step_length = 3)
  (h4 : yield_per_sqft = 0.75)
  : ↑length_steps * step_length * (↑width_steps * step_length) * yield_per_sqft = 3037.5 := by
  sorry

end expected_potato_yield_l1353_135357


namespace chenny_candy_problem_l1353_135312

theorem chenny_candy_problem (initial_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ) : 
  initial_candies = 10 →
  num_friends = 7 →
  candies_per_friend = 2 →
  num_friends * candies_per_friend - initial_candies = 4 := by
  sorry

end chenny_candy_problem_l1353_135312


namespace cubic_sum_over_product_l1353_135355

theorem cubic_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 18)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 21 := by
sorry

end cubic_sum_over_product_l1353_135355


namespace absolute_value_inequality_range_l1353_135372

theorem absolute_value_inequality_range :
  ∀ a : ℝ, (∀ x : ℝ, |x + 3| + |x - 1| ≥ a) ↔ a ≤ 4 :=
by
  sorry

end absolute_value_inequality_range_l1353_135372


namespace square_rectangle_perimeter_equality_l1353_135373

theorem square_rectangle_perimeter_equality :
  ∀ (square_side : ℝ) (rect_length rect_area : ℝ),
    square_side = 15 →
    rect_length = 18 →
    rect_area = 216 →
    4 * square_side = 2 * (rect_length + (rect_area / rect_length)) := by
  sorry

end square_rectangle_perimeter_equality_l1353_135373


namespace max_sum_of_coefficients_l1353_135311

theorem max_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + b * A.2 = 1) ∧ 
    (a * B.1 + b * B.2 = 1) ∧ 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧ 
    (A ≠ B)) →
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + b * A.2 = 1) ∧ 
    (a * B.1 + b * B.2 = 1) ∧ 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧ 
    (abs (A.1 * B.2 - A.2 * B.1) = 1)) →
  a + b ≤ 2 :=
sorry

end max_sum_of_coefficients_l1353_135311


namespace complex_magnitude_squared_l1353_135381

theorem complex_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z ^ 2 = 3 - 5*I) : 
  Complex.abs z ^ 2 = 17/3 := by
  sorry

end complex_magnitude_squared_l1353_135381


namespace number_is_composite_l1353_135307

theorem number_is_composite : ∃ (k : ℕ), k > 1 ∧ k ∣ (53 * 83 * 109 + 40 * 66 * 96) := by
  -- We claim that 149 divides the given number
  use 149
  constructor
  · -- 149 > 1
    norm_num
  · -- 149 divides the given number
    sorry


end number_is_composite_l1353_135307


namespace hyperbola_asymptote_implies_m_eq_six_l1353_135356

/-- Represents a hyperbola with equation x²/m - y²/6 = 1 -/
structure Hyperbola (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / m - y^2 / 6 = 1

/-- Represents an asymptote of a hyperbola -/
structure Asymptote (m : ℝ) where
  slope : ℝ
  eq : ∀ (x y : ℝ), y = slope * x

/-- 
If a hyperbola with equation x²/m - y²/6 = 1 has an asymptote y = x,
then m = 6
-/
theorem hyperbola_asymptote_implies_m_eq_six (m : ℝ) 
  (h : Hyperbola m) 
  (a : Asymptote m) 
  (ha : a.slope = 1) : m = 6 := by
  sorry

end hyperbola_asymptote_implies_m_eq_six_l1353_135356


namespace find_constant_b_l1353_135344

theorem find_constant_b (d e : ℚ) :
  (∀ x : ℚ, (7 * x^2 - 2 * x + 4/3) * (d * x^2 + b * x + e) = 28 * x^4 - 10 * x^3 + 18 * x^2 - 8 * x + 5/3) →
  b = -2/7 := by
sorry

end find_constant_b_l1353_135344


namespace geometric_sequence_common_ratio_l1353_135302

/-- Given a geometric sequence {a_n} where a_2020 = 8a_2017, prove that the common ratio q is 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 2020 = 8 * a 2017 →         -- given condition
  q = 2 :=                      -- conclusion to prove
by
  sorry

end geometric_sequence_common_ratio_l1353_135302


namespace fraction_decomposition_l1353_135390

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 7) (h2 : x ≠ -2) :
  let f := (2 * x + 4) / (x^2 - 5*x - 14)
  let g := 2 / (x - 7) + 0 / (x + 2)
  (x^2 - 5*x - 14 = (x - 7) * (x + 2)) → f = g :=
by
  sorry

end fraction_decomposition_l1353_135390


namespace quadratic_always_positive_l1353_135350

theorem quadratic_always_positive (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end quadratic_always_positive_l1353_135350


namespace coefficient_x4y_value_l1353_135335

/-- The coefficient of x^4y in the expansion of (x^2 + y + 3)^6 -/
def coefficient_x4y (x y : ℕ) : ℕ :=
  (Nat.choose 6 4) * (Nat.choose 4 3) * (3^3)

/-- Theorem stating that the coefficient of x^4y in (x^2 + y + 3)^6 is 1620 -/
theorem coefficient_x4y_value :
  ∀ x y, coefficient_x4y x y = 1620 := by
  sorry

#eval coefficient_x4y 0 0  -- To check the result

end coefficient_x4y_value_l1353_135335


namespace fraction_simplification_l1353_135325

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by
  sorry

end fraction_simplification_l1353_135325


namespace jack_flyers_l1353_135341

theorem jack_flyers (total : ℕ) (rose : ℕ) (left : ℕ) (h1 : total = 1236) (h2 : rose = 320) (h3 : left = 796) :
  total - (rose + left) = 120 := by
  sorry

end jack_flyers_l1353_135341


namespace expression_value_l1353_135391

theorem expression_value : ∀ a b : ℝ, 
  (a * (1 : ℝ)^4 + b * (1 : ℝ)^2 + 2 = -3) → 
  (a * (-1 : ℝ)^4 + b * (-1 : ℝ)^2 - 2 = -7) := by
  sorry

end expression_value_l1353_135391


namespace counterexample_five_l1353_135317

theorem counterexample_five : 
  ∃ n : ℕ, ¬(3 ∣ n) ∧ ¬(Prime (n^2 - 1)) ∧ n = 5 :=
by sorry

end counterexample_five_l1353_135317
