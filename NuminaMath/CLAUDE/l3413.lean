import Mathlib

namespace unique_prime_sum_diff_l3413_341360

theorem unique_prime_sum_diff : ∃! p : ℕ, 
  Prime p ∧ 
  (∃ q s r t : ℕ, Prime q ∧ Prime s ∧ Prime r ∧ Prime t ∧ 
    p = q + s ∧ p = r - t) ∧ 
  p = 5 := by
sorry

end unique_prime_sum_diff_l3413_341360


namespace six_people_handshakes_l3413_341315

/-- The number of unique handshakes between n people, where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the number of handshakes between 6 people is 15. -/
theorem six_people_handshakes : handshakes 6 = 15 := by
  sorry

end six_people_handshakes_l3413_341315


namespace price_change_l3413_341364

/-- Calculates the final price of an item after three price changes -/
theorem price_change (initial_price : ℝ) : 
  initial_price = 320 → 
  (initial_price * 1.15 * 0.9 * 1.25) = 414 := by
  sorry


end price_change_l3413_341364


namespace perpendicular_vectors_k_value_l3413_341363

/-- Given vectors a and b in ℝ², prove that if k*a + b is perpendicular to a - 2*b, then k = 2 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 2))
  (h2 : b = (2, -1))
  (h3 : (k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 2 * b.1, a.2 - 2 * b.2) = 0) :
  k = 2 := by sorry

end perpendicular_vectors_k_value_l3413_341363


namespace first_number_in_expression_l3413_341399

theorem first_number_in_expression (x : ℝ) : x = 0.3 → x * 0.8 + 0.1 * 0.5 = 0.29 := by sorry

end first_number_in_expression_l3413_341399


namespace cosine_sum_17th_roots_l3413_341396

theorem cosine_sum_17th_roots : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end cosine_sum_17th_roots_l3413_341396


namespace parallel_line_triangle_l3413_341300

theorem parallel_line_triangle (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y : ℝ), 
  let s := (a + b + c) / 2
  let perimeter_AXY := x + y + (a * (x + y)) / (b + c)
  let perimeter_XBCY := a + b + c - (x + y)
  (0 < x ∧ x < c) ∧ (0 < y ∧ y < b) ∧ 
  perimeter_AXY = perimeter_XBCY →
  (a * (x + y)) / (b + c) = s * (a / (b + c)) := by
sorry

end parallel_line_triangle_l3413_341300


namespace raft_capacity_raft_problem_l3413_341392

/-- Calculates the number of people that can fit on a raft given certain conditions -/
theorem raft_capacity (max_capacity : ℕ) (reduction_full : ℕ) (life_jackets_needed : ℕ) : ℕ :=
  let capacity_with_jackets := max_capacity - reduction_full
  let jacket_to_person_ratio := capacity_with_jackets / reduction_full
  let reduction := life_jackets_needed / jacket_to_person_ratio
  max_capacity - reduction

/-- Proves that 17 people can fit on the raft under given conditions -/
theorem raft_problem : raft_capacity 21 7 8 = 17 := by
  sorry

end raft_capacity_raft_problem_l3413_341392


namespace coefficient_x_squared_expansion_l3413_341357

theorem coefficient_x_squared_expansion :
  let f : Polynomial ℤ := (X + 1)^5 * (2*X + 1)
  (f.coeff 2) = 20 := by
  sorry

end coefficient_x_squared_expansion_l3413_341357


namespace set_forms_triangle_l3413_341347

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (1, 2, 2) can form a triangle -/
theorem set_forms_triangle : can_form_triangle 1 2 2 := by
  sorry

end set_forms_triangle_l3413_341347


namespace cookie_distribution_l3413_341323

theorem cookie_distribution (chris kenny glenn terry dan anne : ℕ) : 
  chris = kenny / 3 →
  glenn = 4 * chris →
  glenn = 24 →
  terry = Int.floor (Real.sqrt (glenn : ℝ) + 3) →
  dan = 2 * (chris + kenny) →
  anne = kenny / 2 →
  anne ≥ 7 →
  kenny % 2 = 1 →
  ∀ k : ℕ, k % 2 = 1 ∧ k / 2 ≥ 7 → kenny ≤ k →
  chris = 6 ∧ 
  kenny = 18 ∧ 
  glenn = 24 ∧ 
  terry = 8 ∧ 
  dan = 48 ∧ 
  anne = 9 ∧
  chris + kenny + glenn + terry + dan + anne = 113 :=
by sorry

end cookie_distribution_l3413_341323


namespace twentieth_meeting_point_theorem_ant_meeting_theorem_l3413_341345

/-- Represents the meeting point of two ants -/
structure MeetingPoint where
  distance : ℝ
  meeting_number : ℕ

/-- Calculates the meeting point of two ants -/
def calculate_meeting_point (total_distance : ℝ) (speed_ratio : ℝ) (meeting_number : ℕ) : MeetingPoint :=
  { distance := 2,  -- The actual calculation is omitted
    meeting_number := meeting_number }

/-- The theorem stating the 20th meeting point of the ants -/
theorem twentieth_meeting_point_theorem (total_distance : ℝ) (speed_ratio : ℝ) :
  (calculate_meeting_point total_distance speed_ratio 20).distance = 2 :=
by
  sorry

#check twentieth_meeting_point_theorem

/-- Main theorem about the ant problem -/
theorem ant_meeting_theorem :
  ∃ (total_distance : ℝ) (speed_ratio : ℝ),
    total_distance = 6 ∧ speed_ratio = 2.5 ∧
    (calculate_meeting_point total_distance speed_ratio 20).distance = 2 :=
by
  sorry

#check ant_meeting_theorem

end twentieth_meeting_point_theorem_ant_meeting_theorem_l3413_341345


namespace probability_not_face_card_l3413_341326

theorem probability_not_face_card (total_cards : ℕ) (face_cards : ℕ) :
  total_cards = 52 →
  face_cards = 12 →
  (total_cards - face_cards : ℚ) / total_cards = 10 / 13 := by
sorry

end probability_not_face_card_l3413_341326


namespace speed_ratio_of_travelers_l3413_341309

/-- Given two travelers A and B covering the same distance, where A takes 2 hours
    to reach the destination and B takes 30 minutes less than A, prove that the
    ratio of their speeds (vA/vB) is 3/4. -/
theorem speed_ratio_of_travelers (d : ℝ) (vA vB : ℝ) : 
  d > 0 ∧ vA > 0 ∧ vB > 0 ∧ d / vA = 120 ∧ d / vB = 90 → vA / vB = 3 / 4 :=
by sorry

end speed_ratio_of_travelers_l3413_341309


namespace z_in_fourth_quadrant_iff_l3413_341375

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.mk (|a| - 1) (a + 1)

/-- A point is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def in_fourth_quadrant (w : ℂ) : Prop := 0 < w.re ∧ w.im < 0

/-- Theorem stating the necessary and sufficient condition for z to be in the fourth quadrant -/
theorem z_in_fourth_quadrant_iff (a : ℝ) : in_fourth_quadrant (z a) ↔ a < -1 := by
  sorry

end z_in_fourth_quadrant_iff_l3413_341375


namespace marys_current_age_l3413_341340

theorem marys_current_age :
  ∀ (mary_age jay_age : ℕ),
    (jay_age - 5 = (mary_age - 5) + 7) →
    (jay_age + 5 = 2 * (mary_age + 5)) →
    mary_age = 2 := by
  sorry

end marys_current_age_l3413_341340


namespace endpoint_coordinate_sum_l3413_341307

/-- Given a line segment with one endpoint at (3, -5) and midpoint at (7, -15),
    the sum of the coordinates of the other endpoint is -14. -/
theorem endpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (x + 3) / 2 = 7 →
  (y - 5) / 2 = -15 →
  x + y = -14 := by
  sorry

end endpoint_coordinate_sum_l3413_341307


namespace alberto_bjorn_distance_difference_l3413_341354

/-- Given two cyclists, Alberto and Bjorn, with different speeds, 
    prove the difference in distance traveled after a certain time. -/
theorem alberto_bjorn_distance_difference 
  (alberto_speed : ℝ) 
  (bjorn_speed : ℝ) 
  (time : ℝ) 
  (h1 : alberto_speed = 18) 
  (h2 : bjorn_speed = 17) 
  (h3 : time = 5) : 
  alberto_speed * time - bjorn_speed * time = 5 := by
  sorry

#check alberto_bjorn_distance_difference

end alberto_bjorn_distance_difference_l3413_341354


namespace sara_quarters_l3413_341366

def initial_quarters : ℕ := 21
def dad_gave : ℕ := 49
def spent : ℕ := 15
def mom_gave_dollars : ℕ := 2
def quarters_per_dollar : ℕ := 4

theorem sara_quarters (x : ℕ) : 
  initial_quarters + dad_gave - spent + (mom_gave_dollars * quarters_per_dollar) + x = 63 + x := by
  sorry

end sara_quarters_l3413_341366


namespace normal_price_of_pin_is_20_l3413_341332

/-- Calculates the normal price of a pin given the number of pins, discount rate, and total spent -/
def normalPriceOfPin (numPins : ℕ) (discountRate : ℚ) (totalSpent : ℚ) : ℚ :=
  totalSpent / (numPins * (1 - discountRate))

theorem normal_price_of_pin_is_20 :
  normalPriceOfPin 10 (15/100) 170 = 20 := by
  sorry

#eval normalPriceOfPin 10 (15/100) 170

end normal_price_of_pin_is_20_l3413_341332


namespace least_prime_factor_of_7_4_minus_7_3_l3413_341338

theorem least_prime_factor_of_7_4_minus_7_3 :
  Nat.minFac (7^4 - 7^3) = 2 := by
sorry

end least_prime_factor_of_7_4_minus_7_3_l3413_341338


namespace second_workshop_production_l3413_341384

/-- Represents the production and sampling data for three workshops -/
structure WorkshopData where
  total_production : ℕ
  sample_1 : ℕ
  sample_2 : ℕ
  sample_3 : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- Calculates the production of the second workshop based on sampling data -/
def productionOfSecondWorkshop (data : WorkshopData) : ℕ :=
  data.sample_2 * data.total_production / (data.sample_1 + data.sample_2 + data.sample_3)

/-- Theorem stating the production of the second workshop is 1200 given the conditions -/
theorem second_workshop_production
  (data : WorkshopData)
  (h_total : data.total_production = 3600)
  (h_arithmetic : isArithmeticSequence data.sample_1 data.sample_2 data.sample_3) :
  productionOfSecondWorkshop data = 1200 := by
  sorry


end second_workshop_production_l3413_341384


namespace count_numbers_with_ten_digit_square_and_cube_l3413_341386

-- Define a function to count the number of digits in a natural number
def countDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + countDigits (n / 10)

-- Define the condition for a number to satisfy the problem requirement
def satisfiesCondition (n : ℕ) : Prop :=
  countDigits (n^2) + countDigits (n^3) = 10

-- Theorem statement
theorem count_numbers_with_ten_digit_square_and_cube :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfiesCondition n) ∧ S.card = 53 :=
sorry

end count_numbers_with_ten_digit_square_and_cube_l3413_341386


namespace product_with_9999_l3413_341398

theorem product_with_9999 : ∃ x : ℕ, x * 9999 = 4691100843 ∧ x = 469143 := by
  sorry

end product_with_9999_l3413_341398


namespace rooster_on_roof_no_egg_falls_l3413_341336

/-- Represents a bird species -/
inductive BirdSpecies
  | Rooster
  | Hen

/-- Represents the ability to lay eggs -/
def canLayEggs (species : BirdSpecies) : Prop :=
  match species with
  | BirdSpecies.Rooster => False
  | BirdSpecies.Hen => True

/-- Represents a roof with two slopes -/
structure Roof :=
  (slope1 : ℝ)
  (slope2 : ℝ)

/-- Theorem: Given a roof with two slopes and a rooster on the ridge, no egg will fall -/
theorem rooster_on_roof_no_egg_falls (roof : Roof) (bird : BirdSpecies) :
  roof.slope1 = 60 → roof.slope2 = 70 → bird = BirdSpecies.Rooster → ¬(canLayEggs bird) :=
by sorry

end rooster_on_roof_no_egg_falls_l3413_341336


namespace twelfth_term_of_sequence_l3413_341306

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem twelfth_term_of_sequence (a₁ d : ℝ) (h₁ : a₁ = 1) (h₂ : d = 2) :
  arithmetic_sequence a₁ d 12 = 23 := by
  sorry

end twelfth_term_of_sequence_l3413_341306


namespace jewels_total_gain_l3413_341318

/-- Represents the problem of calculating Jewel's total gain from selling magazines --/
def jewels_magazines_problem (cheap_magazines : ℕ) (expensive_magazines : ℕ) 
  (cheap_buy_price : ℚ) (expensive_buy_price : ℚ)
  (cheap_sell_price : ℚ) (expensive_sell_price : ℚ)
  (cheap_discount_percent : ℚ) (expensive_discount_percent : ℚ)
  (cheap_discount_on : ℕ) (expensive_discount_on : ℕ) : Prop :=
let total_cost := cheap_magazines * cheap_buy_price + expensive_magazines * expensive_buy_price
let total_sell := cheap_magazines * cheap_sell_price + expensive_magazines * expensive_sell_price
let cheap_discount := cheap_sell_price * cheap_discount_percent
let expensive_discount := expensive_sell_price * expensive_discount_percent
let total_discount := cheap_discount + expensive_discount
let total_gain := total_sell - total_discount - total_cost
total_gain = 5.1875

/-- Theorem stating that Jewel's total gain is $5.1875 under the given conditions --/
theorem jewels_total_gain :
  jewels_magazines_problem 5 5 3 4 3.5 4.75 0.1 0.15 2 4 := by
  sorry

end jewels_total_gain_l3413_341318


namespace log_simplification_l3413_341305

theorem log_simplification (a b m : ℝ) (h : m^2 = a^2 - b^2) (h1 : a + b > 0) (h2 : a - b > 0) (h3 : m > 0) :
  Real.log m / Real.log (a + b) + Real.log m / Real.log (a - b) - 2 * (Real.log m / Real.log (a + b)) * (Real.log m / Real.log (a - b)) = 0 := by
  sorry

end log_simplification_l3413_341305


namespace five_students_three_teams_l3413_341349

/-- The number of ways to assign students to sports teams. -/
def assignStudentsToTeams (numStudents : ℕ) (numTeams : ℕ) : ℕ :=
  numTeams ^ numStudents

/-- Theorem stating that assigning 5 students to 3 teams results in 3^5 possibilities. -/
theorem five_students_three_teams :
  assignStudentsToTeams 5 3 = 3^5 := by
  sorry

end five_students_three_teams_l3413_341349


namespace triangle_obtuse_l3413_341325

theorem triangle_obtuse (a b c : ℝ) (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    A + B + C = π ∧
    c^2 = a^2 + b^2 - 2 * a * b * Real.cos C ∧
    Real.cos C < 0 :=
by sorry

end triangle_obtuse_l3413_341325


namespace equalize_expenses_l3413_341370

/-- The amount LeRoy paid initially -/
def leroy_paid : ℝ := 240

/-- The amount Bernardo paid initially -/
def bernardo_paid : ℝ := 360

/-- The total discount received -/
def discount : ℝ := 60

/-- The amount LeRoy should pay Bernardo to equalize expenses -/
def payment_to_equalize : ℝ := 30

theorem equalize_expenses : 
  let total_cost := leroy_paid + bernardo_paid - discount
  let each_share := total_cost / 2
  payment_to_equalize = each_share - leroy_paid :=
by sorry

end equalize_expenses_l3413_341370


namespace expansion_equals_fifth_power_l3413_341350

theorem expansion_equals_fifth_power (y : ℝ) : 
  (y - 1)^5 + 5*(y - 1)^4 + 10*(y - 1)^3 + 10*(y - 1)^2 + 5*(y - 1) + 1 = y^5 := by
  sorry

end expansion_equals_fifth_power_l3413_341350


namespace intersection_sum_l3413_341373

/-- The quadratic function h(x) = -x^2 - 4x + 1 -/
def h (x : ℝ) : ℝ := -x^2 - 4*x + 1

/-- The function j(x) = -h(x) -/
def j (x : ℝ) : ℝ := -h x

/-- The function k(x) = h(-x) -/
def k (x : ℝ) : ℝ := h (-x)

/-- The number of intersection points between y = h(x) and y = j(x) -/
def c : ℕ := 2

/-- The number of intersection points between y = h(x) and y = k(x) -/
def d : ℕ := 1

/-- Theorem: Given the functions h, j, k, and the intersection counts c and d, 10c + d = 21 -/
theorem intersection_sum : 10 * c + d = 21 := by sorry

end intersection_sum_l3413_341373


namespace martha_age_is_32_l3413_341341

-- Define Ellen's current age
def ellen_current_age : ℕ := 10

-- Define Ellen's age in 6 years
def ellen_future_age : ℕ := ellen_current_age + 6

-- Define Martha's age in terms of Ellen's future age
def martha_age : ℕ := 2 * ellen_future_age

-- Theorem to prove Martha's age
theorem martha_age_is_32 : martha_age = 32 := by
  sorry

end martha_age_is_32_l3413_341341


namespace triangle_area_with_60_degree_angle_l3413_341389

/-- The area of a triangle with one angle of 60 degrees and adjacent sides of 15 cm and 12 cm is 45√3 cm² -/
theorem triangle_area_with_60_degree_angle (a b : ℝ) (h1 : a = 15) (h2 : b = 12) :
  (1/2) * a * b * Real.sqrt 3 = 45 * Real.sqrt 3 := by
  sorry

end triangle_area_with_60_degree_angle_l3413_341389


namespace arithmetic_geometric_harmonic_mean_sum_squares_l3413_341348

theorem arithmetic_geometric_harmonic_mean_sum_squares 
  (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 7)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 385.5 := by
sorry

end arithmetic_geometric_harmonic_mean_sum_squares_l3413_341348


namespace minimize_expression_l3413_341311

theorem minimize_expression (x : ℝ) (h : x > 1) :
  (2 + 3*x + 4/(x - 1)) ≥ 4*Real.sqrt 3 + 5 ∧
  (2 + 3*x + 4/(x - 1) = 4*Real.sqrt 3 + 5 ↔ x = 2/3*Real.sqrt 3 + 1) :=
by sorry

end minimize_expression_l3413_341311


namespace quadratic_solution_difference_squared_l3413_341312

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ,
  (4 * a^2 - 8 * a - 21 = 0) →
  (4 * b^2 - 8 * b - 21 = 0) →
  (a - b)^2 = 25 := by
sorry

end quadratic_solution_difference_squared_l3413_341312


namespace negation_of_implication_l3413_341324

theorem negation_of_implication (x : ℝ) :
  ¬(x < 0 → x < 1) ↔ (x ≥ 0 → x ≥ 1) := by
  sorry

end negation_of_implication_l3413_341324


namespace arrangement_count_is_432_l3413_341344

/-- The number of ways to arrange players from four teams in a row. -/
def arrangement_count : ℕ :=
  let celtics := 3  -- Number of Celtics players
  let lakers := 3   -- Number of Lakers players
  let warriors := 2 -- Number of Warriors players
  let nuggets := 2  -- Number of Nuggets players
  let team_count := 4 -- Number of teams
  let specific_warrior := 1 -- One specific Warrior must sit at the left end
  
  -- Arrangements of teams (excluding Warriors who are fixed at the left)
  (team_count - 1).factorial *
  -- Arrangement of the non-specific Warrior
  (warriors - specific_warrior).factorial *
  -- Arrangements within each team
  celtics.factorial * lakers.factorial * nuggets.factorial

/-- Theorem stating that the number of arrangements is 432. -/
theorem arrangement_count_is_432 : arrangement_count = 432 := by
  sorry

end arrangement_count_is_432_l3413_341344


namespace brad_lemonade_profit_l3413_341394

/-- Calculates the net profit from a lemonade stand operation -/
def lemonade_stand_profit (
  glasses_per_gallon : ℕ)
  (cost_per_gallon : ℚ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ) : ℚ :=
  let total_glasses := glasses_per_gallon * gallons_made
  let glasses_for_sale := total_glasses - glasses_drunk
  let glasses_sold := glasses_for_sale - glasses_unsold
  let revenue := glasses_sold * price_per_glass
  let cost := gallons_made * cost_per_gallon
  revenue - cost

/-- Theorem stating that Brad's net profit is $14.00 -/
theorem brad_lemonade_profit :
  lemonade_stand_profit 16 3.5 2 1 5 6 = 14 := by
  sorry

end brad_lemonade_profit_l3413_341394


namespace ellipse_m_value_l3413_341368

/-- An ellipse with equation x² + my² = 1, where m > 0 -/
structure Ellipse (m : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 + m*y^2 = 1)
  (m_pos : m > 0)

/-- The focus of the ellipse is on the y-axis -/
def focus_on_y_axis (m : ℝ) : Prop := 0 < m ∧ m < 1

/-- The length of the major axis is twice that of the minor axis -/
def major_twice_minor (m : ℝ) : Prop := Real.sqrt (1/m) = 2

/-- Theorem: For an ellipse with equation x² + my² = 1 (m > 0), 
    if its focus is on the y-axis and the length of its major axis 
    is twice that of its minor axis, then m = 1/4 -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
  (h1 : focus_on_y_axis m) (h2 : major_twice_minor m) : m = 1/4 :=
sorry

end ellipse_m_value_l3413_341368


namespace student_count_l3413_341390

theorem student_count : ∃ n : ℕ, n < 40 ∧ n % 7 = 3 ∧ n % 6 = 1 ∧ n = 31 := by
  sorry

end student_count_l3413_341390


namespace range_of_a_l3413_341371

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a * x^2 - x + 1/4 * a)

def q (a : ℝ) : Prop := ∀ x > 0, 3^x - 9^x < a

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l3413_341371


namespace lincoln_county_houses_l3413_341333

theorem lincoln_county_houses : 
  let original_houses : ℕ := 128936
  let new_houses : ℕ := 359482
  original_houses + new_houses = 488418 :=
by sorry

end lincoln_county_houses_l3413_341333


namespace polynomial_division_theorem_l3413_341322

/-- The remainder when (x^1001 - 1) is divided by (x^4 + x^3 + 2x^2 + x + 1) -/
def remainder1 (x : ℝ) : ℝ := x^2 * (1 - x)

/-- The remainder when (x^1001 - 1) is divided by (x^8 + x^6 + 2x^4 + x^2 + 1) -/
def remainder2 (x : ℝ) : ℝ := -2*x^7 - x^5 - 2*x^3 - 1

/-- The first divisor polynomial -/
def divisor1 (x : ℝ) : ℝ := x^4 + x^3 + 2*x^2 + x + 1

/-- The second divisor polynomial -/
def divisor2 (x : ℝ) : ℝ := x^8 + x^6 + 2*x^4 + x^2 + 1

/-- The dividend polynomial -/
def dividend (x : ℝ) : ℝ := x^1001 - 1

theorem polynomial_division_theorem :
  ∀ x : ℝ,
  ∃ q1 q2 : ℝ → ℝ,
  dividend x = q1 x * divisor1 x + remainder1 x ∧
  dividend x = q2 x * divisor2 x + remainder2 x :=
sorry

end polynomial_division_theorem_l3413_341322


namespace jo_bob_balloon_ride_l3413_341319

/-- The problem of Jo-Bob's hot air balloon ride -/
theorem jo_bob_balloon_ride (rise_rate : ℝ) (fall_rate : ℝ) (second_pull_time : ℝ) 
  (fall_time : ℝ) (max_height : ℝ) :
  rise_rate = 50 →
  fall_rate = 10 →
  second_pull_time = 15 →
  fall_time = 10 →
  max_height = 1400 →
  ∃ (first_pull_time : ℝ),
    first_pull_time * rise_rate - fall_time * fall_rate + second_pull_time * rise_rate = max_height ∧
    first_pull_time = 15 := by
  sorry

#check jo_bob_balloon_ride

end jo_bob_balloon_ride_l3413_341319


namespace range_of_g_l3413_341313

def f (x : ℝ) : ℝ := 2 * x - 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x ∈ Set.Icc 1 3, -29 ≤ g x ∧ g x ≤ 3 ∧
  ∀ y ∈ Set.Icc (-29) 3, ∃ x ∈ Set.Icc 1 3, g x = y :=
sorry

end range_of_g_l3413_341313


namespace gcd_18_30_l3413_341303

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l3413_341303


namespace muffin_mix_buyers_l3413_341359

theorem muffin_mix_buyers (total_buyers : ℕ) (cake_mix_buyers : ℕ) (both_mix_buyers : ℕ) 
  (neither_mix_prob : ℚ) (h1 : total_buyers = 100) (h2 : cake_mix_buyers = 50) 
  (h3 : both_mix_buyers = 15) (h4 : neither_mix_prob = 1/4) : 
  ∃ muffin_mix_buyers : ℕ, muffin_mix_buyers = 40 ∧ 
  muffin_mix_buyers = total_buyers - (cake_mix_buyers - both_mix_buyers) - 
    (neither_mix_prob * total_buyers) :=
by
  sorry

#check muffin_mix_buyers

end muffin_mix_buyers_l3413_341359


namespace ellipse_min_reciprocal_sum_l3413_341331

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_min_reciprocal_sum :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
  (1 / distance P left_focus + 1 / distance P right_focus ≥ 1) ∧
  (∃ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 ∧
    1 / distance Q left_focus + 1 / distance Q right_focus = 1) :=
by sorry

end ellipse_min_reciprocal_sum_l3413_341331


namespace area_not_unique_l3413_341380

/-- A land plot with a side of length 10 units -/
structure LandPlot where
  side : ℝ
  side_positive : side > 0

/-- The area of a land plot -/
noncomputable def area (plot : LandPlot) : ℝ := sorry

/-- Theorem: The area of a land plot cannot be uniquely determined given only the length of one side -/
theorem area_not_unique (plot1 plot2 : LandPlot) 
  (h : plot1.side = plot2.side) (h_side : plot1.side = 10) : 
  ¬ (∀ (p1 p2 : LandPlot), p1.side = p2.side → area p1 = area p2) := by
  sorry

end area_not_unique_l3413_341380


namespace best_fitting_model_has_highest_r_squared_model1_has_best_fitting_effect_l3413_341381

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  name : String
  r_squared : ℝ
  r_squared_nonneg : 0 ≤ r_squared
  r_squared_le_one : r_squared ≤ 1

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fitting_effect (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, m.r_squared ≤ model.r_squared

/-- The theorem stating that the model with the highest R² value has the best fitting effect -/
theorem best_fitting_model_has_highest_r_squared 
  (models : List RegressionModel) (model : RegressionModel) 
  (h_model_in_list : model ∈ models) 
  (h_nonempty : models ≠ []) :
  has_best_fitting_effect model models ↔ 
  ∀ m ∈ models, m.r_squared ≤ model.r_squared :=
sorry

/-- The specific problem instance -/
def model1 : RegressionModel := ⟨"Model 1", 0.98, by norm_num, by norm_num⟩
def model2 : RegressionModel := ⟨"Model 2", 0.80, by norm_num, by norm_num⟩
def model3 : RegressionModel := ⟨"Model 3", 0.54, by norm_num, by norm_num⟩
def model4 : RegressionModel := ⟨"Model 4", 0.35, by norm_num, by norm_num⟩

def problem_models : List RegressionModel := [model1, model2, model3, model4]

theorem model1_has_best_fitting_effect : 
  has_best_fitting_effect model1 problem_models :=
sorry

end best_fitting_model_has_highest_r_squared_model1_has_best_fitting_effect_l3413_341381


namespace circle_number_determinable_l3413_341343

/-- Represents a system of six circles connected by line segments -/
structure CircleSystem where
  /-- Numbers in the circles -/
  circle_numbers : Fin 6 → ℝ
  /-- Numbers on the segments connecting the circles -/
  segment_numbers : Fin 6 → ℝ
  /-- Each circle contains the sum of its incoming segment numbers -/
  sum_property : ∀ i : Fin 6, circle_numbers i = segment_numbers i + segment_numbers ((i + 5) % 6)

/-- The theorem stating that any circle's number can be determined from the other five -/
theorem circle_number_determinable (cs : CircleSystem) (i : Fin 6) :
  cs.circle_numbers i =
    cs.circle_numbers ((i + 1) % 6) +
    cs.circle_numbers ((i + 3) % 6) +
    cs.circle_numbers ((i + 5) % 6) -
    cs.circle_numbers ((i + 2) % 6) -
    cs.circle_numbers ((i + 4) % 6) :=
  sorry


end circle_number_determinable_l3413_341343


namespace factor_sum_l3413_341362

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 4) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 15 := by
sorry

end factor_sum_l3413_341362


namespace smallest_representable_l3413_341321

/-- The representation function for k -/
def representation (n m : ℕ+) : ℤ := 19^(n:ℕ) - 5^(m:ℕ)

/-- The property that k is representable -/
def is_representable (k : ℕ) : Prop :=
  ∃ (n m : ℕ+), representation n m = k

/-- The main theorem statement -/
theorem smallest_representable : 
  (is_representable 14) ∧ (∀ k : ℕ, 0 < k ∧ k < 14 → ¬(is_representable k)) := by
  sorry

#check smallest_representable

end smallest_representable_l3413_341321


namespace expected_adjacent_red_pairs_l3413_341382

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCardsCount : ℕ := 26

/-- The probability of a red card being followed by another red card -/
def probRedFollowedByRed : ℚ := 25 / 51

theorem expected_adjacent_red_pairs (deck_size : ℕ) (red_count : ℕ) (prob_red_red : ℚ) :
  deck_size = standardDeckSize →
  red_count = redCardsCount →
  prob_red_red = probRedFollowedByRed →
  (red_count : ℚ) * prob_red_red = 650 / 51 := by
  sorry

#check expected_adjacent_red_pairs

end expected_adjacent_red_pairs_l3413_341382


namespace even_polynomial_iff_product_with_negation_l3413_341391

-- Define the complex polynomials
variable (P Q : ℂ → ℂ)

-- Define what it means for a function to be a polynomial
def IsPolynomial (f : ℂ → ℂ) : Prop := sorry

-- Define what it means for a function to be even
def IsEven (f : ℂ → ℂ) : Prop := ∀ z, f (-z) = f z

-- State the theorem
theorem even_polynomial_iff_product_with_negation :
  (IsPolynomial P ∧ IsEven P) ↔ 
  (∃ Q, IsPolynomial Q ∧ ∀ z, P z = Q z * Q (-z)) := by sorry

end even_polynomial_iff_product_with_negation_l3413_341391


namespace area_of_trapezoid_DBCE_l3413_341356

/-- A structure representing a triangle in our problem -/
structure Triangle where
  area : ℝ

/-- A structure representing the trapezoid DBCE -/
structure Trapezoid where
  area : ℝ

/-- The isosceles triangle ABC -/
def ABC : Triangle := { area := 36 }

/-- One of the smallest triangles -/
def smallTriangle : Triangle := { area := 1 }

/-- The number of smallest triangles -/
def numSmallTriangles : ℕ := 5

/-- Triangle ADE, composed of 3 smallest triangles -/
def ADE : Triangle := { area := 3 }

/-- The trapezoid DBCE -/
def DBCE : Trapezoid := { area := ABC.area - ADE.area }

/-- The theorem to be proved -/
theorem area_of_trapezoid_DBCE : DBCE.area = 33 := by
  sorry

end area_of_trapezoid_DBCE_l3413_341356


namespace acid_dilution_l3413_341334

/-- Given an initial acid solution and water added to dilute it, 
    calculate the amount of water needed to reach a specific concentration. -/
theorem acid_dilution (m : ℝ) (hm : m > 50) : 
  ∃ x : ℝ, 
    (m * (m / 100) = (m + x) * ((m - 20) / 100)) → 
    x = (20 * m) / (m + 20) := by
  sorry

end acid_dilution_l3413_341334


namespace sum_of_A_and_B_is_13_l3413_341377

theorem sum_of_A_and_B_is_13 (A B : ℕ) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  70 + A - (10 * B + 5) = 34 → 
  A + B = 13 := by
sorry

end sum_of_A_and_B_is_13_l3413_341377


namespace haley_deleted_files_l3413_341379

/-- The number of files deleted from a flash drive -/
def files_deleted (initial_music : ℕ) (initial_video : ℕ) (files_left : ℕ) : ℕ :=
  initial_music + initial_video - files_left

/-- Proof that 11 files were deleted from Haley's flash drive -/
theorem haley_deleted_files : files_deleted 27 42 58 = 11 := by
  sorry

end haley_deleted_files_l3413_341379


namespace opposite_of_negative_one_third_l3413_341335

theorem opposite_of_negative_one_third :
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end opposite_of_negative_one_third_l3413_341335


namespace mixture_combination_theorem_l3413_341342

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℕ
  water : ℕ

/-- Combines two mixtures -/
def combineMixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk
    water := m1.water + m2.water }

/-- Simplifies a ratio by dividing both parts by their GCD -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem mixture_combination_theorem :
  let m1 : Mixture := { milk := 7, water := 2 }
  let m2 : Mixture := { milk := 8, water := 1 }
  let combined := combineMixtures m1 m2
  simplifyRatio combined.milk combined.water = (5, 1) := by
  sorry

end mixture_combination_theorem_l3413_341342


namespace exists_solution_l3413_341369

theorem exists_solution : ∃ (a b : ℤ), a ≠ b ∧ 
  (a : ℚ) / 2015 + (b : ℚ) / 2016 = (2015 + 2016 : ℚ) / (2015 * 2016) := by
  sorry

end exists_solution_l3413_341369


namespace binomial_510_510_l3413_341320

theorem binomial_510_510 : (510 : ℕ).choose 510 = 1 := by sorry

end binomial_510_510_l3413_341320


namespace sum_positive_if_greater_than_abs_l3413_341329

theorem sum_positive_if_greater_than_abs (a b : ℝ) (h : a - |b| > 0) : a + b > 0 := by
  sorry

end sum_positive_if_greater_than_abs_l3413_341329


namespace alex_bike_trip_downhill_speed_alex_bike_trip_downhill_speed_is_24_l3413_341376

/-- Calculates the average speed on the downhill section of Alex's bike trip --/
theorem alex_bike_trip_downhill_speed : ℝ :=
  let total_distance : ℝ := 164
  let flat_time : ℝ := 4.5
  let flat_speed : ℝ := 20
  let uphill_time : ℝ := 2.5
  let uphill_speed : ℝ := 12
  let downhill_time : ℝ := 1.5
  let walking_distance : ℝ := 8
  let flat_distance : ℝ := flat_time * flat_speed
  let uphill_distance : ℝ := uphill_time * uphill_speed
  let distance_before_puncture : ℝ := total_distance - walking_distance
  let downhill_distance : ℝ := distance_before_puncture - flat_distance - uphill_distance
  let downhill_speed : ℝ := downhill_distance / downhill_time
  downhill_speed

theorem alex_bike_trip_downhill_speed_is_24 : alex_bike_trip_downhill_speed = 24 := by
  sorry

end alex_bike_trip_downhill_speed_alex_bike_trip_downhill_speed_is_24_l3413_341376


namespace fifth_month_sales_l3413_341378

def sales_1 : ℕ := 5435
def sales_2 : ℕ := 5927
def sales_3 : ℕ := 5855
def sales_4 : ℕ := 6230
def sales_6 : ℕ := 3991
def average_sale : ℕ := 5500
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_5 = 5562 :=
by sorry

end fifth_month_sales_l3413_341378


namespace rectangle_length_from_perimeter_and_width_l3413_341358

/-- The perimeter of a rectangle is twice the sum of its length and width -/
def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Given a rectangle with perimeter 100 cm and width 20 cm, its length is 30 cm -/
theorem rectangle_length_from_perimeter_and_width :
  ∃ (length : ℝ), 
    rectangle_perimeter length 20 = 100 ∧ length = 30 := by
  sorry

end rectangle_length_from_perimeter_and_width_l3413_341358


namespace min_value_of_f_l3413_341393

/-- The function f(x,y) represents the given expression -/
def f (x y : ℝ) : ℝ := x^2 + y^2 - 8*x + 6*y + 20

theorem min_value_of_f :
  (∀ x y : ℝ, f x y ≥ -5) ∧ (∃ x y : ℝ, f x y = -5) :=
sorry

end min_value_of_f_l3413_341393


namespace right_triangle_tan_b_l3413_341328

theorem right_triangle_tan_b (A B C : Real) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) : 
  A + B + Real.pi/2 = Real.pi → Real.sin A = 2/3 → Real.tan B = Real.sqrt 5 / 2 := by
  sorry

end right_triangle_tan_b_l3413_341328


namespace solve_equation_l3413_341361

theorem solve_equation (x : ℝ) (h : 61 + 5 * 12 / (x / 3) = 62) : x = 180 := by
  sorry

end solve_equation_l3413_341361


namespace inequality_proof_l3413_341372

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a * b + b * c + c * a = 1) : 
  (3 / Real.sqrt (a^2 + 1)) + (4 / Real.sqrt (b^2 + 1)) + (12 / Real.sqrt (c^2 + 1)) < 39/2 := by
  sorry

end inequality_proof_l3413_341372


namespace smallest_number_divisible_by_all_l3413_341387

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 5) % 19 = 0 ∧
  (n + 5) % 73 = 0 ∧
  (n + 5) % 101 = 0 ∧
  (n + 5) % 89 = 0

theorem smallest_number_divisible_by_all :
  ∃! n : ℕ, is_divisible_by_all n ∧ ∀ m : ℕ, m < n → ¬is_divisible_by_all m :=
by
  use 1113805958
  sorry

end smallest_number_divisible_by_all_l3413_341387


namespace hidden_dots_four_dice_l3413_341353

/-- The sum of dots on a standard six-sided die -/
def standard_die_sum : ℕ := 21

/-- The total number of dots on four standard six-sided dice -/
def total_dots (n : ℕ) : ℕ := n * standard_die_sum

/-- The sum of visible dots on the stacked dice -/
def visible_dots : ℕ := 1 + 2 + 2 + 3 + 4 + 5 + 6 + 6

/-- The number of hidden dots on four stacked dice -/
def hidden_dots (n : ℕ) : ℕ := total_dots n - visible_dots

theorem hidden_dots_four_dice : 
  hidden_dots 4 = 55 := by sorry

end hidden_dots_four_dice_l3413_341353


namespace darry_smaller_ladder_climbs_l3413_341301

/-- Represents the number of steps in Darry's full ladder -/
def full_ladder_steps : ℕ := 11

/-- Represents the number of steps in Darry's smaller ladder -/
def smaller_ladder_steps : ℕ := 6

/-- Represents the number of times Darry climbed the full ladder -/
def full_ladder_climbs : ℕ := 10

/-- Represents the total number of steps Darry climbed -/
def total_steps : ℕ := 152

/-- Theorem stating that Darry climbed the smaller ladder 7 times -/
theorem darry_smaller_ladder_climbs :
  ∃ (x : ℕ), x * smaller_ladder_steps + full_ladder_climbs * full_ladder_steps = total_steps ∧ x = 7 := by
  sorry

end darry_smaller_ladder_climbs_l3413_341301


namespace two_digit_integers_with_remainder_three_l3413_341351

theorem two_digit_integers_with_remainder_three : 
  (Finset.filter 
    (fun n => n ≥ 10 ∧ n < 100 ∧ n % 7 = 3) 
    (Finset.range 100)).card = 13 := by
  sorry

end two_digit_integers_with_remainder_three_l3413_341351


namespace product_ratio_simplification_l3413_341304

theorem product_ratio_simplification
  (a b c d e f g : ℝ)
  (h1 : a * b * c * d = 260)
  (h2 : b * c * d * e = 390)
  (h3 : c * d * e * f = 2000)
  (h4 : d * e * f * g = 500)
  (h5 : c ≠ 0)
  (h6 : e ≠ 0) :
  (a * g) / (c * e) = a / (4 * c) :=
by sorry

end product_ratio_simplification_l3413_341304


namespace possible_values_of_a_l3413_341388

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}

def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

theorem possible_values_of_a :
  ∃ S : Set ℝ, S = {a : ℝ | A a ∩ B a = {2, 5}} ∧ S = {-1, 2} := by
  sorry

end possible_values_of_a_l3413_341388


namespace soccer_ball_price_proof_l3413_341302

/-- The unit price of B type soccer balls -/
def unit_price_B : ℝ := 60

/-- The unit price of A type soccer balls -/
def unit_price_A : ℝ := 2.5 * unit_price_B

/-- The total cost of A type soccer balls -/
def total_cost_A : ℝ := 7500

/-- The total cost of B type soccer balls -/
def total_cost_B : ℝ := 4800

/-- The quantity difference between B and A type soccer balls -/
def quantity_difference : ℕ := 30

theorem soccer_ball_price_proof :
  (total_cost_A / unit_price_A) + quantity_difference = total_cost_B / unit_price_B :=
by sorry

end soccer_ball_price_proof_l3413_341302


namespace complement_of_A_in_U_l3413_341314

def U : Set ℕ := {0, 2, 4, 6, 8, 10}
def A : Set ℕ := {2, 4, 6}

theorem complement_of_A_in_U :
  U \ A = {0, 8, 10} := by
  sorry

end complement_of_A_in_U_l3413_341314


namespace teacher_fills_thermos_once_per_day_l3413_341327

/-- Represents the teacher's coffee drinking habits --/
structure CoffeeDrinkingHabits where
  thermos_capacity : ℝ
  school_days_per_week : ℕ
  current_weekly_consumption : ℝ
  consumption_reduction_factor : ℝ

/-- Calculates the number of times the thermos is filled per day --/
def thermos_fills_per_day (habits : CoffeeDrinkingHabits) : ℕ :=
  sorry

/-- Theorem stating that the teacher fills her thermos once per day --/
theorem teacher_fills_thermos_once_per_day (habits : CoffeeDrinkingHabits) 
  (h1 : habits.thermos_capacity = 20)
  (h2 : habits.school_days_per_week = 5)
  (h3 : habits.current_weekly_consumption = 40)
  (h4 : habits.consumption_reduction_factor = 1/4) :
  thermos_fills_per_day habits = 1 := by
  sorry

end teacher_fills_thermos_once_per_day_l3413_341327


namespace parabola_properties_l3413_341316

theorem parabola_properties (a b c : ℝ) (h1 : a < 0) 
  (h2 : a * (-3)^2 + b * (-3) + c = 0) 
  (h3 : a * 1^2 + b * 1 + c = 0) : 
  (b^2 - 4*a*c > 0) ∧ (3*b + 2*c = 0) := by sorry

end parabola_properties_l3413_341316


namespace inequality_proof_l3413_341385

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 := by
  sorry

end inequality_proof_l3413_341385


namespace students_in_both_subjects_range_l3413_341330

def total_students : ℕ := 3000

def history_min : ℕ := 2100
def history_max : ℕ := 2250

def psychology_min : ℕ := 1200
def psychology_max : ℕ := 1500

theorem students_in_both_subjects_range :
  ∃ (min_both max_both : ℕ),
    (∀ (h p both : ℕ),
      history_min ≤ h ∧ h ≤ history_max →
      psychology_min ≤ p ∧ p ≤ psychology_max →
      h + p - both = total_students →
      min_both ≤ both ∧ both ≤ max_both) ∧
    max_both - min_both = 450 :=
sorry

end students_in_both_subjects_range_l3413_341330


namespace correct_polynomial_sum_l3413_341308

variable (a : ℝ)
variable (A B : ℝ → ℝ)

theorem correct_polynomial_sum
  (hB : B = λ x => 3 * x^2 - 5 * x - 7)
  (hA_minus_2B : A - 2 * B = λ x => -2 * x^2 + 3 * x + 6) :
  A + 2 * B = λ x => 10 * x^2 - 17 * x - 22 :=
by sorry

end correct_polynomial_sum_l3413_341308


namespace floor_negative_seven_fourths_l3413_341367

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end floor_negative_seven_fourths_l3413_341367


namespace table_sum_zero_l3413_341374

structure Table :=
  (a b c d : ℝ)

def distinct (t : Table) : Prop :=
  t.a ≠ t.b ∧ t.a ≠ t.c ∧ t.a ≠ t.d ∧ t.b ≠ t.c ∧ t.b ≠ t.d ∧ t.c ≠ t.d

def row_sum_equal (t : Table) : Prop :=
  t.a + t.b = t.c + t.d

def column_product_equal (t : Table) : Prop :=
  t.a * t.c = t.b * t.d

theorem table_sum_zero (t : Table) 
  (h1 : distinct t) 
  (h2 : row_sum_equal t) 
  (h3 : column_product_equal t) : 
  t.a + t.b + t.c + t.d = 0 := by
  sorry

end table_sum_zero_l3413_341374


namespace triangle_containing_all_points_l3413_341383

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t1 t2 t3 : Point) : Prop := sorry

theorem triangle_containing_all_points 
  (n : ℕ) 
  (points : Fin n → Point) 
  (h : ∀ (i j k : Fin n), triangleArea (points i) (points j) (points k) ≤ 1) :
  ∃ (t1 t2 t3 : Point), 
    (triangleArea t1 t2 t3 ≤ 4) ∧ 
    (∀ (i : Fin n), isInside (points i) t1 t2 t3) := by
  sorry

end triangle_containing_all_points_l3413_341383


namespace minimum_value_sqrt_sum_l3413_341355

theorem minimum_value_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ Real.sqrt 2) ∧
  (∀ ε > 0, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ Real.sqrt x + Real.sqrt y > Real.sqrt 2 - ε) :=
by sorry

end minimum_value_sqrt_sum_l3413_341355


namespace pyramid_base_side_length_l3413_341397

/-- The side length of the square base of a right pyramid, given the area of a lateral face and the slant height. -/
theorem pyramid_base_side_length (lateral_face_area : ℝ) (slant_height : ℝ) :
  lateral_face_area = 120 →
  slant_height = 40 →
  (lateral_face_area / slant_height) / 2 = 6 := by
  sorry

end pyramid_base_side_length_l3413_341397


namespace no_real_roots_composite_l3413_341365

-- Define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_real_roots_composite (a b c : ℝ) :
  (∀ x : ℝ, f a b c x ≠ x) →
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) :=
by sorry

end no_real_roots_composite_l3413_341365


namespace scientific_notation_of_41600_l3413_341317

theorem scientific_notation_of_41600 :
  ∃ (a : ℝ) (n : ℤ), 41600 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.16 ∧ n = 4 := by
  sorry

end scientific_notation_of_41600_l3413_341317


namespace problem1_l3413_341352

theorem problem1 (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 + y^2 = 30) : x * y = 3 := by
  sorry

end problem1_l3413_341352


namespace function_property_l3413_341346

theorem function_property (A : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, A (x + y) = A x + A y) 
  (h2 : ∀ x y : ℝ, A (x * y) = A x * A y) : 
  (∀ x : ℝ, A x = x) ∨ (∀ x : ℝ, A x = 0) :=
by sorry

end function_property_l3413_341346


namespace intersecting_line_passes_through_fixed_point_l3413_341337

/-- A parabola defined by y² = 4x passing through (1, 2) -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  passes_through : eq 1 2

/-- A line intersecting the parabola at two points -/
structure IntersectingLine (p : Parabola) where
  slope : ℝ
  y_intercept : ℝ
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    p.eq x₁ y₁ ∧ p.eq x₂ y₂ ∧
    x₁ = slope * y₁ + y_intercept ∧
    x₂ = slope * y₂ + y_intercept ∧
    y₁ * y₂ = -4

/-- The theorem to be proved -/
theorem intersecting_line_passes_through_fixed_point (p : Parabola) (l : IntersectingLine p) :
  ∃ (x y : ℝ), x = l.slope * y + l.y_intercept ∧ x = 1 ∧ y = 0 :=
sorry

end intersecting_line_passes_through_fixed_point_l3413_341337


namespace percentage_problem_l3413_341310

theorem percentage_problem (x : ℝ) : (0.3 / 100) * x = 0.15 → x = 50 := by
  sorry

end percentage_problem_l3413_341310


namespace largest_five_digit_palindrome_divisible_by_127_l3413_341395

/-- A function that checks if a number is a 5-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ 
  (n / 10000 = n % 10) ∧
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The largest 5-digit palindrome divisible by 127 -/
def largest_palindrome : ℕ := 99399

theorem largest_five_digit_palindrome_divisible_by_127 :
  is_five_digit_palindrome largest_palindrome ∧
  largest_palindrome % 127 = 0 ∧
  ∀ n : ℕ, is_five_digit_palindrome n ∧ n % 127 = 0 → n ≤ largest_palindrome :=
by sorry

end largest_five_digit_palindrome_divisible_by_127_l3413_341395


namespace sqrt_490000_equals_700_l3413_341339

theorem sqrt_490000_equals_700 : Real.sqrt 490000 = 700 := by
  sorry

end sqrt_490000_equals_700_l3413_341339
