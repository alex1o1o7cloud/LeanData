import Mathlib

namespace NUMINAMATH_CALUDE_circle_triangle_area_l1489_148994

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  -- We'll assume a simple representation for this problem
  -- More complex representations might be needed for general use
  point : Point
  direction : Point

def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.x - c2.center.x) ^ 2 + (c1.center.y - c2.center.y) ^ 2 = (c1.radius + c2.radius) ^ 2

def internally_tangent_to_line (c : Circle) (l : Line) : Prop :=
  -- This is a simplification; in reality, we'd need more complex calculations
  c.center.y = l.point.y + c.radius

def externally_tangent_to_line (c : Circle) (l : Line) : Prop :=
  -- This is a simplification; in reality, we'd need more complex calculations
  c.center.y = l.point.y - c.radius

def between_points_on_line (p q r : Point) (l : Line) : Prop :=
  -- This is a simplification; in reality, we'd need more complex calculations
  p.x < q.x ∧ q.x < r.x

def triangle_area (p q r : Point) : ℝ :=
  0.5 * |p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)|

theorem circle_triangle_area :
  ∀ (P Q R : Circle) (l : Line) (P' Q' R' : Point),
    P.radius = 1 →
    Q.radius = 3 →
    R.radius = 5 →
    internally_tangent_to_line P l →
    externally_tangent_to_line Q l →
    externally_tangent_to_line R l →
    between_points_on_line P' Q' R' l →
    externally_tangent P Q →
    externally_tangent Q R →
    triangle_area P.center Q.center R.center = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_area_l1489_148994


namespace NUMINAMATH_CALUDE_unknown_number_value_l1489_148959

theorem unknown_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 315 * 7) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l1489_148959


namespace NUMINAMATH_CALUDE_shark_sightings_difference_l1489_148931

theorem shark_sightings_difference (daytona_sightings cape_may_sightings : ℕ) 
  (h1 : daytona_sightings = 26)
  (h2 : cape_may_sightings = 7)
  (h3 : daytona_sightings > 3 * cape_may_sightings) :
  daytona_sightings - 3 * cape_may_sightings = 5 := by
sorry

end NUMINAMATH_CALUDE_shark_sightings_difference_l1489_148931


namespace NUMINAMATH_CALUDE_same_color_probability_l1489_148972

def total_plates : ℕ := 11
def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def selected_plates : ℕ := 3

theorem same_color_probability : 
  (Nat.choose red_plates selected_plates : ℚ) / (Nat.choose total_plates selected_plates) = 4 / 33 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l1489_148972


namespace NUMINAMATH_CALUDE_ordering_abc_l1489_148917

noncomputable def a : ℝ := 0.98 + Real.sin 0.01
noncomputable def b : ℝ := Real.exp (-0.01)
noncomputable def c : ℝ := (Real.log 2022) / (Real.log 2023) / (Real.log 2021) / (Real.log 2023)

theorem ordering_abc : c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l1489_148917


namespace NUMINAMATH_CALUDE_log_simplification_l1489_148978

theorem log_simplification (a b c d x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : x > 0) (h6 : y > 0) :
  Real.log (2 * a / (3 * b)) + Real.log (3 * b / (4 * c)) + Real.log (4 * c / (5 * d)) - Real.log (10 * a * y / (3 * d * x)) = Real.log (3 * x / (25 * y)) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l1489_148978


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l1489_148964

def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)

theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l1489_148964


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l1489_148962

theorem mean_equality_implies_z_value :
  let x₁ : ℚ := 7
  let x₂ : ℚ := 11
  let x₃ : ℚ := 23
  let y₁ : ℚ := 15
  let mean_xyz : ℚ := (x₁ + x₂ + x₃) / 3
  let mean_yz : ℚ := (y₁ + z) / 2
  mean_xyz = mean_yz → z = 37 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l1489_148962


namespace NUMINAMATH_CALUDE_fraction_change_theorem_l1489_148976

theorem fraction_change_theorem (a b c d e f x y : ℚ) 
  (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + y) = c / d) 
  (h4 : (a + 2*x) / (b + 2*y) = e / f) 
  (h5 : d ≠ c) (h6 : f ≠ e) : 
  x = (b*c - a*d) / (d - c) ∧ 
  y = (b*e - a*f) / (2*f - 2*e) := by
sorry

end NUMINAMATH_CALUDE_fraction_change_theorem_l1489_148976


namespace NUMINAMATH_CALUDE_car_speed_l1489_148998

/-- Given a car that travels 325 miles in 5 hours, its speed is 65 miles per hour -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 325) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : speed = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l1489_148998


namespace NUMINAMATH_CALUDE_problem_solution_l1489_148950

theorem problem_solution : 
  (12345679^2 * 81 - 1) / 11111111 / 10 * 9 - 8 = 10000000000 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1489_148950


namespace NUMINAMATH_CALUDE_chess_tournament_games_14_l1489_148960

/-- The number of games played in a chess tournament -/
def chess_tournament_games (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a chess tournament with 14 players where each player plays every other player once,
    the total number of games played is 91. -/
theorem chess_tournament_games_14 :
  chess_tournament_games 14 = 91 := by
  sorry

#eval chess_tournament_games 14  -- This should output 91

end NUMINAMATH_CALUDE_chess_tournament_games_14_l1489_148960


namespace NUMINAMATH_CALUDE_tammys_climbing_speed_l1489_148932

/-- Tammy's mountain climbing problem -/
theorem tammys_climbing_speed 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (h1 : total_time = 14) 
  (h2 : total_distance = 52) 
  (h3 : speed_difference = 0.5) 
  (h4 : time_difference = 2) :
  ∃ (v : ℝ), 
    v * (total_time / 2 + time_difference) + 
    (v + speed_difference) * (total_time / 2 - time_difference) = total_distance ∧
    v + speed_difference = 4 := by
  sorry


end NUMINAMATH_CALUDE_tammys_climbing_speed_l1489_148932


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1489_148968

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  (Complex.re ((1 : ℂ) + i) / i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1489_148968


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1489_148996

/-- Given a line y = mx + b, if the point (-2, 0) is reflected to (6, 4) across this line, then m + b = 4 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 6 ∧ y = 4 ∧ 
    (x - (-2))^2 + (y - 0)^2 = ((x + 2)/2 - (m * ((x + (-2))/2) + b))^2 + 
    ((y + 0)/2 - ((x + (-2))/(2*m) + b))^2) → 
  m + b = 4 :=
by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1489_148996


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1489_148952

theorem parallel_vectors_sum (m n : ℝ) : 
  let a : Fin 3 → ℝ := ![(-2 : ℝ), 3, -1]
  let b : Fin 3 → ℝ := ![4, m, n]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  m + n = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1489_148952


namespace NUMINAMATH_CALUDE_allan_correct_answers_l1489_148915

theorem allan_correct_answers (total_questions : ℕ) 
  (correct_points : ℚ) (incorrect_penalty : ℚ) (final_score : ℚ) :
  total_questions = 120 →
  correct_points = 1 →
  incorrect_penalty = 1/4 →
  final_score = 100 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    (correct_answers : ℚ) * correct_points + 
    ((total_questions - correct_answers) : ℚ) * (-incorrect_penalty) = final_score ∧
    correct_answers = 104 := by
  sorry

end NUMINAMATH_CALUDE_allan_correct_answers_l1489_148915


namespace NUMINAMATH_CALUDE_sector_area_l1489_148969

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 12) (h2 : central_angle = 2) :
  let radius := perimeter / (2 + central_angle)
  (1/2) * central_angle * radius^2 = 9 := by sorry

end NUMINAMATH_CALUDE_sector_area_l1489_148969


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1489_148995

theorem circle_equation_proof (x y : ℝ) :
  (∃ (h k r : ℝ), r > 0 ∧ ∀ x y, x^2 + y^2 + 1 = 2*x + 4*y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  (∃ (h k : ℝ), ∀ x y, x^2 + y^2 + 1 = 2*x + 4*y ↔ (x - h)^2 + (y - k)^2 = 4) :=
by sorry

#check circle_equation_proof

end NUMINAMATH_CALUDE_circle_equation_proof_l1489_148995


namespace NUMINAMATH_CALUDE_quarterly_insurance_payment_l1489_148965

theorem quarterly_insurance_payment 
  (annual_payment : ℕ) 
  (quarters_per_year : ℕ) 
  (h1 : annual_payment = 1512) 
  (h2 : quarters_per_year = 4) : 
  annual_payment / quarters_per_year = 378 := by
sorry

end NUMINAMATH_CALUDE_quarterly_insurance_payment_l1489_148965


namespace NUMINAMATH_CALUDE_stone_skipping_l1489_148966

theorem stone_skipping (throw1 throw2 throw3 throw4 throw5 : ℕ) : 
  throw5 = 8 ∧ 
  throw2 = throw1 + 2 ∧ 
  throw3 = 2 * throw2 ∧ 
  throw4 = throw3 - 3 ∧ 
  throw5 = throw4 + 1 →
  throw1 + throw2 + throw3 + throw4 + throw5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_stone_skipping_l1489_148966


namespace NUMINAMATH_CALUDE_max_sum_with_square_diff_l1489_148908

theorem max_sum_with_square_diff (a b : ℤ) (h : a^2 - b^2 = 144) :
  ∃ (d : ℤ), d = a + b ∧ d ≤ 72 ∧ ∃ (a' b' : ℤ), a'^2 - b'^2 = 144 ∧ a' + b' = 72 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_square_diff_l1489_148908


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l1489_148963

theorem nested_subtraction_simplification (x : ℝ) : 1 - (2 - (3 - (4 - (5 - (6 - x))))) = x - 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l1489_148963


namespace NUMINAMATH_CALUDE_ticket_probability_problem_l1489_148957

theorem ticket_probability_problem : ∃! n : ℕ, 
  1 ≤ n ∧ n ≤ 20 ∧ 
  (↑(Finset.filter (λ x => x % n = 0) (Finset.range 20)).card / 20 : ℚ) = 3/10 ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ticket_probability_problem_l1489_148957


namespace NUMINAMATH_CALUDE_population_function_time_to_reach_1_2_million_max_growth_rate_20_years_l1489_148905

-- Define the initial population and growth rate
def initial_population : ℝ := 1000000
def annual_growth_rate : ℝ := 0.012

-- Define the population function
def population (years : ℕ) : ℝ := initial_population * (1 + annual_growth_rate) ^ years

-- Theorem 1: Population function
theorem population_function (years : ℕ) : 
  population years = 100 * (1.012 ^ years) * 10000 := by sorry

-- Theorem 2: Time to reach 1.2 million
theorem time_to_reach_1_2_million : 
  ∃ y : ℕ, y ≥ 16 ∧ y < 17 ∧ population y ≥ 1200000 ∧ population (y-1) < 1200000 := by sorry

-- Theorem 3: Maximum growth rate for 20 years
theorem max_growth_rate_20_years (max_rate : ℝ) : 
  (∀ rate : ℝ, rate ≤ max_rate → initial_population * (1 + rate) ^ 20 ≤ 1200000) ↔ 
  max_rate ≤ 0.009 := by sorry

end NUMINAMATH_CALUDE_population_function_time_to_reach_1_2_million_max_growth_rate_20_years_l1489_148905


namespace NUMINAMATH_CALUDE_rope_sections_l1489_148935

/-- Given a rope of 50 feet, prove that after using 1/5 for art and giving half of the remainder
    to a friend, the number of 2-foot sections that can be cut from the remaining rope is 10. -/
theorem rope_sections (total_rope : ℝ) (art_fraction : ℝ) (friend_fraction : ℝ) (section_length : ℝ) :
  total_rope = 50 ∧
  art_fraction = 1/5 ∧
  friend_fraction = 1/2 ∧
  section_length = 2 →
  (total_rope - art_fraction * total_rope) * (1 - friend_fraction) / section_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_rope_sections_l1489_148935


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l1489_148954

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part 1: Minimum value when a = 1
theorem min_value_when_a_is_one :
  ∃ (m : ℝ), ∀ (x : ℝ), f 1 x ≥ m ∧ ∃ (y : ℝ), f 1 y = m ∧ m = 1 :=
sorry

-- Part 2: Range of a for f(x) ≥ a when x ∈ [-1, +∞)
theorem range_of_a :
  ∀ (a : ℝ), (∀ (x : ℝ), x ≥ -1 → f a x ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l1489_148954


namespace NUMINAMATH_CALUDE_max_carlson_jars_l1489_148928

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamJars where
  carlsonWeights : List Nat  -- List of weights of Carlson's jars
  babyWeights : List Nat     -- List of weights of Baby's jars

/-- Checks if the given JamJars satisfies the initial condition -/
def satisfiesInitialCondition (jars : JamJars) : Prop :=
  jars.carlsonWeights.sum = 13 * jars.babyWeights.sum

/-- Checks if the given JamJars satisfies the final condition after transfer -/
def satisfiesFinalCondition (jars : JamJars) : Prop :=
  let minWeight := jars.carlsonWeights.minimum?
  match minWeight with
  | some w => (jars.carlsonWeights.sum - w) = 8 * (jars.babyWeights.sum + w)
  | none => False

/-- Theorem stating the maximum number of jars Carlson could have initially had -/
theorem max_carlson_jars :
  ∀ jars : JamJars,
    satisfiesInitialCondition jars →
    satisfiesFinalCondition jars →
    jars.carlsonWeights.length ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_max_carlson_jars_l1489_148928


namespace NUMINAMATH_CALUDE_bill_difference_zero_l1489_148921

theorem bill_difference_zero (anna_tip : ℝ) (anna_percent : ℝ) 
  (ben_tip : ℝ) (ben_percent : ℝ) 
  (h1 : anna_tip = 5) 
  (h2 : anna_percent = 25 / 100)
  (h3 : ben_tip = 3)
  (h4 : ben_percent = 15 / 100)
  (h5 : anna_tip = anna_percent * anna_bill)
  (h6 : ben_tip = ben_percent * ben_bill) :
  anna_bill - ben_bill = 0 :=
by
  sorry

#check bill_difference_zero

end NUMINAMATH_CALUDE_bill_difference_zero_l1489_148921


namespace NUMINAMATH_CALUDE_betty_oranges_l1489_148923

/-- The number of boxes Betty has -/
def num_boxes : ℕ := 3

/-- The number of oranges in each box -/
def oranges_per_box : ℕ := 8

/-- The total number of oranges Betty has -/
def total_oranges : ℕ := num_boxes * oranges_per_box

theorem betty_oranges : total_oranges = 24 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_l1489_148923


namespace NUMINAMATH_CALUDE_min_value_a_plus_9b_l1489_148997

theorem min_value_a_plus_9b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 10 * a * b) :
  8/5 ≤ a + 9*b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 10 * a₀ * b₀ ∧ a₀ + 9*b₀ = 8/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_9b_l1489_148997


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l1489_148951

theorem cubic_equation_sum (r s t : ℝ) : 
  r^3 - 7*r^2 + 11*r = 13 →
  s^3 - 7*s^2 + 11*s = 13 →
  t^3 - 7*t^2 + 11*t = 13 →
  (r+s)/t + (s+t)/r + (t+r)/s = 38/13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l1489_148951


namespace NUMINAMATH_CALUDE_james_milk_consumption_l1489_148903

/-- The amount of milk James drank, given his initial amount, the conversion rate from gallons to ounces, and the remaining amount. -/
def milk_drank (initial_gallons : ℕ) (ounces_per_gallon : ℕ) (remaining_ounces : ℕ) : ℕ :=
  initial_gallons * ounces_per_gallon - remaining_ounces

/-- Theorem stating that James drank 13 ounces of milk. -/
theorem james_milk_consumption :
  milk_drank 3 128 371 = 13 := by
  sorry

end NUMINAMATH_CALUDE_james_milk_consumption_l1489_148903


namespace NUMINAMATH_CALUDE_probability_two_females_l1489_148956

theorem probability_two_females (total : ℕ) (females : ℕ) (males : ℕ) 
  (h1 : total = females + males)
  (h2 : total = 8)
  (h3 : females = 5)
  (h4 : males = 3) :
  (Nat.choose females 2 : ℚ) / (Nat.choose total 2) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_l1489_148956


namespace NUMINAMATH_CALUDE_clothing_tax_rate_l1489_148907

theorem clothing_tax_rate 
  (total : ℝ) 
  (clothing_spend : ℝ) 
  (food_spend : ℝ) 
  (other_spend : ℝ) 
  (other_tax_rate : ℝ) 
  (total_tax_rate : ℝ) :
  clothing_spend = 0.6 * total →
  food_spend = 0.1 * total →
  other_spend = 0.3 * total →
  other_tax_rate = 0.08 →
  total_tax_rate = 0.048 →
  ∃ (clothing_tax_rate : ℝ),
    clothing_tax_rate * clothing_spend + other_tax_rate * other_spend = total_tax_rate * total ∧
    clothing_tax_rate = 0.04 :=
by sorry

end NUMINAMATH_CALUDE_clothing_tax_rate_l1489_148907


namespace NUMINAMATH_CALUDE_bubble_gum_count_l1489_148984

/-- The cost of a single piece of bubble gum in cents -/
def cost_per_piece : ℕ := 18

/-- The total cost of all pieces of bubble gum in cents -/
def total_cost : ℕ := 2448

/-- The number of pieces of bubble gum -/
def num_pieces : ℕ := total_cost / cost_per_piece

theorem bubble_gum_count : num_pieces = 136 := by
  sorry

end NUMINAMATH_CALUDE_bubble_gum_count_l1489_148984


namespace NUMINAMATH_CALUDE_marble_sculpture_first_week_cut_l1489_148993

/-- Proves that the percentage of marble cut away in the first week is 30% --/
theorem marble_sculpture_first_week_cut (
  original_weight : ℝ)
  (second_week_cut : ℝ)
  (third_week_cut : ℝ)
  (final_weight : ℝ)
  (h1 : original_weight = 250)
  (h2 : second_week_cut = 20)
  (h3 : third_week_cut = 25)
  (h4 : final_weight = 105)
  : ∃ (first_week_cut : ℝ),
    first_week_cut = 30 ∧
    final_weight = original_weight * 
      (1 - first_week_cut / 100) * 
      (1 - second_week_cut / 100) * 
      (1 - third_week_cut / 100) := by
  sorry


end NUMINAMATH_CALUDE_marble_sculpture_first_week_cut_l1489_148993


namespace NUMINAMATH_CALUDE_max_remainder_eleven_l1489_148943

theorem max_remainder_eleven (x : ℕ+) : ∃ (q r : ℕ), x = 11 * q + r ∧ r ≤ 10 ∧ ∀ (r' : ℕ), x = 11 * q + r' → r' ≤ r :=
sorry

end NUMINAMATH_CALUDE_max_remainder_eleven_l1489_148943


namespace NUMINAMATH_CALUDE_two_numbers_problem_l1489_148948

theorem two_numbers_problem (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b = 6) (h5 : a / b = 6) : a * b - (a - b) = 6 / 49 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l1489_148948


namespace NUMINAMATH_CALUDE_white_animals_more_than_cats_l1489_148930

theorem white_animals_more_than_cats (C W : ℕ) (h1 : C > 0) (h2 : W > 0) : W > C :=
  by
  -- Define the number of white cats (WC)
  have h3 : C / 3 = W / 6 :=
    -- Every third cat is white and every sixth white animal is a cat
    sorry
  
  -- Prove that W = 2C
  have h4 : W = 2 * C :=
    sorry

  -- Conclude that W > C
  sorry


end NUMINAMATH_CALUDE_white_animals_more_than_cats_l1489_148930


namespace NUMINAMATH_CALUDE_vector_sum_proof_l1489_148973

def v1 : Fin 3 → ℤ := ![- 7, 3, 5]
def v2 : Fin 3 → ℤ := ![4, - 1, - 6]
def v3 : Fin 3 → ℤ := ![1, 8, 2]

theorem vector_sum_proof :
  (v1 + v2 + v3) = ![- 2, 10, 1] := by sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l1489_148973


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1489_148919

-- Define the polynomial
def f (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30

-- Define the divisor
def g (x : ℝ) : ℝ := 4 * x - 8

-- Theorem statement
theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, f x = g x * q x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1489_148919


namespace NUMINAMATH_CALUDE_sqrt_81_div_3_l1489_148975

theorem sqrt_81_div_3 : Real.sqrt 81 / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_div_3_l1489_148975


namespace NUMINAMATH_CALUDE_preimage_of_two_one_l1489_148924

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * p.1 + p.2, p.1 - 2 * p.2)

/-- Theorem stating that (1, 0) is the pre-image of (2, 1) under f -/
theorem preimage_of_two_one :
  f (1, 0) = (2, 1) ∧ ∀ p : ℝ × ℝ, f p = (2, 1) → p = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_two_one_l1489_148924


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1489_148989

theorem quadratic_inequality_solution (a m : ℝ) : 
  (∀ x : ℝ, (a * x^2 + 6 * x - a^2 < 0) ↔ (x < 1 ∨ x > m)) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1489_148989


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1489_148939

universe u

def U : Set (Fin 6) := {1, 2, 3, 4, 5, 6}
def P : Set (Fin 6) := {1, 2, 3, 4}
def Q : Set (Fin 6) := {3, 4, 5, 6}

theorem intersection_complement_equality :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1489_148939


namespace NUMINAMATH_CALUDE_percentage_without_scholarship_l1489_148936

/-- Represents the percentage of students who won't get a scholarship in a school with a given ratio of boys to girls and scholarship rates. -/
theorem percentage_without_scholarship
  (boy_girl_ratio : ℚ)
  (boy_scholarship_rate : ℚ)
  (girl_scholarship_rate : ℚ)
  (h1 : boy_girl_ratio = 5 / 6)
  (h2 : boy_scholarship_rate = 1 / 4)
  (h3 : girl_scholarship_rate = 1 / 5) :
  (1 - (boy_girl_ratio * boy_scholarship_rate + girl_scholarship_rate) / (boy_girl_ratio + 1)) * 100 =
  (1 - (1.25 + 1.2) / 11) * 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_without_scholarship_l1489_148936


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_l1489_148925

structure Line3D where
  -- Placeholder for 3D line properties

structure Plane3D where
  -- Placeholder for 3D plane properties

def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

theorem perpendicular_planes_parallel (m : Line3D) (α β : Plane3D) :
  perpendicular m α → perpendicular m β → parallel α β := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_l1489_148925


namespace NUMINAMATH_CALUDE_complex_repair_cost_is_50_l1489_148953

/-- Represents Jim's bike shop financials for a month -/
structure BikeShop where
  tire_repair_price : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repair_price : ℕ
  complex_repairs_count : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ
  total_profit : ℕ

/-- Calculates the cost of parts for each complex repair -/
def complex_repair_cost (shop : BikeShop) : ℕ :=
  let tire_repair_profit := (shop.tire_repair_price - shop.tire_repair_cost) * shop.tire_repairs_count
  let complex_repairs_revenue := shop.complex_repair_price * shop.complex_repairs_count
  let total_revenue := tire_repair_profit + shop.retail_profit + complex_repairs_revenue
  let profit_before_complex_costs := total_revenue - shop.fixed_expenses
  let complex_repairs_profit := shop.total_profit - (profit_before_complex_costs - complex_repairs_revenue)
  (complex_repairs_revenue - complex_repairs_profit) / shop.complex_repairs_count

theorem complex_repair_cost_is_50 (shop : BikeShop)
  (h1 : shop.tire_repair_price = 20)
  (h2 : shop.tire_repair_cost = 5)
  (h3 : shop.tire_repairs_count = 300)
  (h4 : shop.complex_repair_price = 300)
  (h5 : shop.complex_repairs_count = 2)
  (h6 : shop.retail_profit = 2000)
  (h7 : shop.fixed_expenses = 4000)
  (h8 : shop.total_profit = 3000) :
  complex_repair_cost shop = 50 := by
  sorry

end NUMINAMATH_CALUDE_complex_repair_cost_is_50_l1489_148953


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1489_148933

theorem coefficient_x_squared_in_expansion : 
  (Finset.range 11).sum (fun k => (Nat.choose 10 k) * (2^k) * if k = 2 then 1 else 0) = 180 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1489_148933


namespace NUMINAMATH_CALUDE_gcd_of_2_powers_l1489_148922

theorem gcd_of_2_powers : Nat.gcd (2^2018 - 1) (2^2029 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_2_powers_l1489_148922


namespace NUMINAMATH_CALUDE_max_value_of_linear_function_max_value_achieved_l1489_148979

-- Define the linear function
def f (x : ℝ) : ℝ := -x + 3

-- State the theorem
theorem max_value_of_linear_function :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x ≤ 3 :=
by
  sorry

-- State that the maximum is achieved
theorem max_value_achieved :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_linear_function_max_value_achieved_l1489_148979


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coeff_l1489_148949

theorem quadratic_rational_root_even_coeff
  (a b c : ℤ) (x : ℚ)
  (h_a_nonzero : a ≠ 0)
  (h_root : a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coeff_l1489_148949


namespace NUMINAMATH_CALUDE_x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l1489_148937

theorem x_lt_1_necessary_not_sufficient_for_ln_x_lt_0 :
  (∀ x : ℝ, Real.log x < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ Real.log x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l1489_148937


namespace NUMINAMATH_CALUDE_river_width_l1489_148955

/-- Proves that given a river with specified depth, flow rate, and discharge volume,
    the width of the river is 25 meters. -/
theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (discharge_volume : ℝ) :
  depth = 8 →
  flow_rate_kmph = 8 →
  discharge_volume = 26666.666666666668 →
  (discharge_volume / (depth * (flow_rate_kmph * 1000 / 60))) = 25 := by
  sorry


end NUMINAMATH_CALUDE_river_width_l1489_148955


namespace NUMINAMATH_CALUDE_largest_value_l1489_148977

theorem largest_value (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : x + y = 1) :
  (1/2 < x^2 + y^2) ∧ (2*x*y < x^2 + y^2) ∧ (x < x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l1489_148977


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l1489_148961

theorem max_area_inscribed_rectangle (d : ℝ) (h : d = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = d^2 → x * y ≤ d^2 / 2 :=
by
  sorry

#check max_area_inscribed_rectangle

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l1489_148961


namespace NUMINAMATH_CALUDE_outdoor_section_length_l1489_148990

/-- Given a rectangular outdoor section with area 35 square feet and width 7 feet, 
    the length of the section is 5 feet. -/
theorem outdoor_section_length : 
  ∀ (area width length : ℝ), 
    area = 35 → 
    width = 7 → 
    area = width * length → 
    length = 5 := by
sorry

end NUMINAMATH_CALUDE_outdoor_section_length_l1489_148990


namespace NUMINAMATH_CALUDE_total_toys_count_l1489_148910

def bill_toys : ℕ := 60

def hana_toys : ℕ := (5 * bill_toys) / 6

def hash_toys : ℕ := hana_toys / 2 + 9

def total_toys : ℕ := bill_toys + hana_toys + hash_toys

theorem total_toys_count : total_toys = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_count_l1489_148910


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_iff_m_gt_two_l1489_148988

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (1 + Complex.I) * (m - 2 * Complex.I)

-- Define the condition for z to be in the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_first_quadrant_iff_m_gt_two (m : ℝ) :
  is_in_first_quadrant (z m) ↔ m > 2 := by
  sorry


end NUMINAMATH_CALUDE_z_in_first_quadrant_iff_m_gt_two_l1489_148988


namespace NUMINAMATH_CALUDE_harmonious_expressions_l1489_148906

-- Define the concept of a harmonious algebraic expression
def is_harmonious (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b ∧
  ∃ y ∈ Set.Icc a b, ∀ z ∈ Set.Icc a b, f y ≥ f z

-- Theorem statement
theorem harmonious_expressions :
  let a := -2
  let b := 2
  -- Part 1
  ¬ is_harmonious (fun x => |x - 1|) a b ∧
  -- Part 2
  ¬ is_harmonious (fun x => -x + 1) a b ∧
  is_harmonious (fun x => -x^2 + 2) a b ∧
  ¬ is_harmonious (fun x => x^2 + |x| - 4) a b ∧
  -- Part 3
  ∀ c : ℝ, is_harmonious (fun x => c / (|x| + 1) - 2) a b ↔ (0 ≤ c ∧ c ≤ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_harmonious_expressions_l1489_148906


namespace NUMINAMATH_CALUDE_number_of_male_students_l1489_148945

theorem number_of_male_students 
  (total_average : ℝ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (num_female : ℕ) 
  (h1 : total_average = 90) 
  (h2 : male_average = 84) 
  (h3 : female_average = 92) 
  (h4 : num_female = 24) :
  ∃ (num_male : ℕ), 
    num_male = 8 ∧ 
    (num_male : ℝ) * male_average + (num_female : ℝ) * female_average = 
      ((num_male : ℝ) + (num_female : ℝ)) * total_average :=
by sorry

end NUMINAMATH_CALUDE_number_of_male_students_l1489_148945


namespace NUMINAMATH_CALUDE_special_integers_count_l1489_148941

/-- The sum of all positive divisors of n including twice the greatest prime divisor of n -/
def g (n : ℕ) : ℕ := sorry

/-- The count of integers j such that 1 ≤ j ≤ 5000 and g(j) = j + 2√j + 1 -/
def count_special_integers : ℕ := sorry

theorem special_integers_count :
  count_special_integers = 19 := by sorry

end NUMINAMATH_CALUDE_special_integers_count_l1489_148941


namespace NUMINAMATH_CALUDE_infinite_points_in_region_l1489_148981

theorem infinite_points_in_region : 
  ∃ (S : Set (ℚ × ℚ)), 
    (∀ (p : ℚ × ℚ), p ∈ S ↔ 
      (0 < p.1 ∧ 0 < p.2) ∧ 
      (p.1^2 + p.2^2 ≤ 16) ∧ 
      (p.1 ≤ 3 ∧ p.2 ≤ 3)) ∧ 
    Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_infinite_points_in_region_l1489_148981


namespace NUMINAMATH_CALUDE_prove_a_value_l1489_148986

-- Define the operation for integers
def star_op (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- Theorem statement
theorem prove_a_value (h : star_op 21 9 = 160) : 21 = 21 := by
  sorry

end NUMINAMATH_CALUDE_prove_a_value_l1489_148986


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_two_l1489_148927

theorem sqrt_sum_equals_two_sqrt_two :
  Real.sqrt (5 - 2 * Real.sqrt 6) + Real.sqrt (5 + 2 * Real.sqrt 6) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_two_l1489_148927


namespace NUMINAMATH_CALUDE_sum_equals_300_l1489_148985

theorem sum_equals_300 : 156 + 44 + 26 + 74 = 300 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_300_l1489_148985


namespace NUMINAMATH_CALUDE_sum_of_y_values_l1489_148909

theorem sum_of_y_values (x y : ℝ) : 
  x^2 + x^2*y^2 + x^2*y^4 = 525 ∧ x + x*y + x*y^2 = 35 → 
  ∃ y₁ y₂ : ℝ, (x^2 + x^2*y₁^2 + x^2*y₁^4 = 525 ∧ x + x*y₁ + x*y₁^2 = 35) ∧
             (x^2 + x^2*y₂^2 + x^2*y₂^4 = 525 ∧ x + x*y₂ + x*y₂^2 = 35) ∧
             y₁ + y₂ = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l1489_148909


namespace NUMINAMATH_CALUDE_smallest_base_for_perfect_square_l1489_148926

theorem smallest_base_for_perfect_square : 
  ∀ b : ℕ, b > 4 → (∃ n : ℕ, 3 * b + 4 = n^2) → b ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_perfect_square_l1489_148926


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1489_148914

theorem quadratic_solution_sum (a b : ℝ) : 
  (5 * (a + b * Complex.I)^2 + 4 * (a + b * Complex.I) + 1 = 0 ∧
   5 * (a - b * Complex.I)^2 + 4 * (a - b * Complex.I) + 1 = 0) →
  a + b^2 = -9/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1489_148914


namespace NUMINAMATH_CALUDE_tangent_intersection_for_specific_circles_l1489_148946

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Finds the x-coordinate of the intersection point between the common tangent line
    of two circles and the x-axis -/
def tangentIntersectionX (c1 c2 : Circle) : ℝ :=
  sorry

theorem tangent_intersection_for_specific_circles :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (12, 0), radius := 5 }
  tangentIntersectionX c1 c2 = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_for_specific_circles_l1489_148946


namespace NUMINAMATH_CALUDE_max_diff_squares_consecutive_integers_l1489_148913

theorem max_diff_squares_consecutive_integers (n : ℤ) : 
  n + (n + 1) < 150 → (n + 1)^2 - n^2 ≤ 149 := by
  sorry

end NUMINAMATH_CALUDE_max_diff_squares_consecutive_integers_l1489_148913


namespace NUMINAMATH_CALUDE_min_subset_size_for_sum_l1489_148967

theorem min_subset_size_for_sum (n : ℕ+) :
  let M := Finset.range (2 * n)
  ∃ k : ℕ+, (∀ A : Finset ℕ, A ⊆ M → A.card = k →
    ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b + c + d = 4 * n + 1) ∧
  (∀ k' : ℕ+, k' < k →
    ∃ A : Finset ℕ, A ⊆ M ∧ A.card = k' ∧
    ∀ a b c d : ℕ, a ∈ A → b ∈ A → c ∈ A → d ∈ A →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    a + b + c + d ≠ 4 * n + 1) ∧
  k = n + 3 :=
by sorry

end NUMINAMATH_CALUDE_min_subset_size_for_sum_l1489_148967


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l1489_148900

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 30) (h2 : n2 = 50) (h3 : avg1 = 40) (h4 : avg2 = 90) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 71.25 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l1489_148900


namespace NUMINAMATH_CALUDE_bicycle_speed_l1489_148999

/-- Given a journey with two modes of transport (on foot and by bicycle), 
    calculate the speed of the bicycle. -/
theorem bicycle_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (foot_distance : ℝ) 
  (foot_speed : ℝ) 
  (h1 : total_distance = 80) 
  (h2 : total_time = 7) 
  (h3 : foot_distance = 32) 
  (h4 : foot_speed = 8) :
  (total_distance - foot_distance) / (total_time - foot_distance / foot_speed) = 16 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_speed_l1489_148999


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1489_148934

/-- Represents the outcome of tossing three coins -/
inductive CoinToss
  | HHH
  | HHT
  | HTH
  | HTT
  | THH
  | THT
  | TTH
  | TTT

/-- The event of getting no more than one heads -/
def noMoreThanOneHeads (t : CoinToss) : Prop :=
  t = CoinToss.HTT ∨ t = CoinToss.THT ∨ t = CoinToss.TTH ∨ t = CoinToss.TTT

/-- The event of getting at least two heads -/
def atLeastTwoHeads (t : CoinToss) : Prop :=
  t = CoinToss.HHH ∨ t = CoinToss.HHT ∨ t = CoinToss.HTH ∨ t = CoinToss.THH

/-- Theorem stating that the two events are mutually exclusive -/
theorem mutually_exclusive_events :
  ∀ t : CoinToss, ¬(noMoreThanOneHeads t ∧ atLeastTwoHeads t) :=
by
  sorry


end NUMINAMATH_CALUDE_mutually_exclusive_events_l1489_148934


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1489_148983

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem geometric_sequence_property (b : ℕ → ℝ) :
  is_geometric_sequence b →
  b 9 = (3 + 5) / 2 →
  b 1 * b 17 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1489_148983


namespace NUMINAMATH_CALUDE_platform_length_l1489_148982

/-- The length of a platform given train speed and crossing times -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) : 
  train_speed = 72 →
  platform_time = 30 →
  man_time = 19 →
  (train_speed * 1000 / 3600) * platform_time - (train_speed * 1000 / 3600) * man_time = 220 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l1489_148982


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1489_148912

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

-- Define the solution set
def S : Set ℝ := {2} ∪ {x | x > 6}

-- Theorem statement
theorem solution_set_of_inequality :
  {x : ℝ | f x ≥ x^2 - 8*x + 15} = S :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1489_148912


namespace NUMINAMATH_CALUDE_laura_workout_speed_l1489_148958

theorem laura_workout_speed :
  ∃! x : ℝ, x > 0 ∧ (30 / (3 * x + 2) + 3 / x = (230 - 10) / 60) := by
  sorry

end NUMINAMATH_CALUDE_laura_workout_speed_l1489_148958


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1489_148980

/-- Given a geometric sequence with first term 1000 and sixth term 125,
    prove that the third term is equal to 301. -/
theorem geometric_sequence_third_term :
  ∀ (a : ℝ) (r : ℝ),
    a = 1000 →
    a * r^5 = 125 →
    a * r^2 = 301 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1489_148980


namespace NUMINAMATH_CALUDE_fraction_simplification_l1489_148991

theorem fraction_simplification : 
  (1 / 4 - 1 / 5) / (1 / 3 - 1 / 6) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1489_148991


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_stamp_books_l1489_148992

theorem largest_common_divisor_of_stamp_books : ∃ (n : ℕ), n > 0 ∧ n ∣ 900 ∧ n ∣ 1200 ∧ n ∣ 1500 ∧ ∀ (m : ℕ), m > n → ¬(m ∣ 900 ∧ m ∣ 1200 ∧ m ∣ 1500) := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_stamp_books_l1489_148992


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_450_l1489_148911

theorem least_integer_greater_than_sqrt_450 : 
  (∀ n : ℤ, n ≤ ⌊Real.sqrt 450⌋ → n < 22) ∧ 22 > Real.sqrt 450 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_450_l1489_148911


namespace NUMINAMATH_CALUDE_range_of_a_l1489_148987

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∪ B a = B a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1489_148987


namespace NUMINAMATH_CALUDE_billion_scientific_notation_l1489_148938

/-- Represents the value of one billion -/
def billion : ℝ := 10^9

/-- The given amount in billions -/
def amount : ℝ := 4.15

theorem billion_scientific_notation : 
  amount * billion = 4.15 * 10^9 := by sorry

end NUMINAMATH_CALUDE_billion_scientific_notation_l1489_148938


namespace NUMINAMATH_CALUDE_fraction_always_defined_l1489_148916

theorem fraction_always_defined (y : ℝ) : (y^2 + 1) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_fraction_always_defined_l1489_148916


namespace NUMINAMATH_CALUDE_deepak_age_l1489_148902

/-- Given that the ratio of Rahul's age to Deepak's age is 4:3, 
    and Rahul's age after 4 years will be 32, 
    prove that Deepak's current age is 21 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 4 = 32 →
  deepak_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l1489_148902


namespace NUMINAMATH_CALUDE_stratified_sampling_low_income_l1489_148947

/-- Represents the number of households sampled from a given group -/
def sampleSize (totalSize : ℕ) (groupSize : ℕ) (sampledHighIncome : ℕ) (totalHighIncome : ℕ) : ℕ :=
  (sampledHighIncome * groupSize) / totalHighIncome

theorem stratified_sampling_low_income 
  (totalHouseholds : ℕ) 
  (highIncomeHouseholds : ℕ) 
  (lowIncomeHouseholds : ℕ) 
  (sampledHighIncome : ℕ) :
  totalHouseholds = 500 →
  highIncomeHouseholds = 125 →
  lowIncomeHouseholds = 95 →
  sampledHighIncome = 25 →
  sampleSize totalHouseholds lowIncomeHouseholds sampledHighIncome highIncomeHouseholds = 19 := by
  sorry

#check stratified_sampling_low_income

end NUMINAMATH_CALUDE_stratified_sampling_low_income_l1489_148947


namespace NUMINAMATH_CALUDE_johns_hats_cost_l1489_148944

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- The total cost of John's hats -/
def total_cost : ℕ := weeks_of_different_hats * days_per_week * cost_per_hat

theorem johns_hats_cost :
  total_cost = 700 := by sorry

end NUMINAMATH_CALUDE_johns_hats_cost_l1489_148944


namespace NUMINAMATH_CALUDE_binomial_10_choose_5_l1489_148918

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_5_l1489_148918


namespace NUMINAMATH_CALUDE_lune_area_l1489_148904

/-- The area of the region inside a semicircle of diameter 2, outside a semicircle of diameter 4,
    and outside an inscribed square with side length 2 is equal to -π + 2. -/
theorem lune_area (π : ℝ) (h : π > 0) : 
  let small_semicircle_area := (1/2) * π * (2/2)^2
  let large_semicircle_area := (1/2) * π * (4/2)^2
  let square_area := 2^2
  let sector_area := (1/4) * large_semicircle_area
  small_semicircle_area - sector_area - square_area = -π + 2 := by
  sorry

end NUMINAMATH_CALUDE_lune_area_l1489_148904


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l1489_148942

theorem unique_root_quadratic (k : ℝ) :
  (∃! a : ℝ, (k^2 - 9) * a^2 - 2*(k + 1)*a + 1 = 0) →
  (k = 3 ∨ k = -3 ∨ k = -5) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l1489_148942


namespace NUMINAMATH_CALUDE_probability_same_color_problem_l1489_148901

/-- Probability of drawing two balls of the same color with replacement -/
def probability_same_color (green red blue : ℕ) : ℚ :=
  let total := green + red + blue
  (green^2 + red^2 + blue^2) / total^2

/-- Theorem: The probability of drawing two balls of the same color is 29/81 -/
theorem probability_same_color_problem :
  probability_same_color 8 6 4 = 29 / 81 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_problem_l1489_148901


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1489_148970

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def mersenne_prime (p : ℕ) : Prop := is_prime p ∧ ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_under_500 :
  ∃ p : ℕ, mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, mersenne_prime q → q < 500 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1489_148970


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_5_l1489_148929

theorem smallest_four_digit_mod_8_5 : ∃ (n : ℕ), 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 5 → m ≥ n) ∧
  (n = 1005) := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_5_l1489_148929


namespace NUMINAMATH_CALUDE_max_subsets_exists_444_subsets_l1489_148971

/-- A structure representing a collection of 3-element subsets of a 1000-element set. -/
structure SubsetCollection where
  /-- The underlying 1000-element set -/
  base : Finset (Fin 1000)
  /-- The collection of 3-element subsets -/
  subsets : Finset (Finset (Fin 1000))
  /-- Each subset has exactly 3 elements -/
  three_element : ∀ s ∈ subsets, Finset.card s = 3
  /-- Each subset is a subset of the base set -/
  subset_of_base : ∀ s ∈ subsets, s ⊆ base
  /-- The union of any 5 subsets has at least 12 elements -/
  union_property : ∀ (five_subsets : Finset (Finset (Fin 1000))), 
    five_subsets ⊆ subsets → Finset.card five_subsets = 5 → 
    Finset.card (Finset.biUnion five_subsets id) ≥ 12

/-- The maximum number of three-element subsets satisfying the given conditions is 444. -/
theorem max_subsets (sc : SubsetCollection) : Finset.card sc.subsets ≤ 444 := by
  sorry

/-- There exists a collection of 444 three-element subsets satisfying the given conditions. -/
theorem exists_444_subsets : ∃ sc : SubsetCollection, Finset.card sc.subsets = 444 := by
  sorry

end NUMINAMATH_CALUDE_max_subsets_exists_444_subsets_l1489_148971


namespace NUMINAMATH_CALUDE_wood_measurement_l1489_148940

theorem wood_measurement (x y : ℝ) : 
  (y = x + 4.5 ∧ 0.5 * y = x - 1) ↔ 
  (∃ (wood_length rope_length : ℝ), 
    wood_length = x ∧ 
    rope_length = y ∧ 
    rope_length - wood_length = 4.5 ∧ 
    0.5 * rope_length - wood_length = -1) :=
by sorry

end NUMINAMATH_CALUDE_wood_measurement_l1489_148940


namespace NUMINAMATH_CALUDE_rhombus_area_theorem_l1489_148920

/-- A rhombus with perpendicular bisecting diagonals -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  perpendicular_bisectors : Bool

/-- Calculate the area of a rhombus given its properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem: The area of a rhombus with side length √117 and diagonals differing by 8 units is 101 -/
theorem rhombus_area_theorem (r : Rhombus) 
    (h1 : r.side_length = Real.sqrt 117)
    (h2 : r.diagonal_difference = 8)
    (h3 : r.perpendicular_bisectors = true) : 
  rhombus_area r = 101 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_theorem_l1489_148920


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1489_148974

def a (n : ℕ) : ℕ := 2 * Nat.factorial n + n

theorem max_gcd_consecutive_terms (n : ℕ) : Nat.gcd (a n) (a (n + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1489_148974
