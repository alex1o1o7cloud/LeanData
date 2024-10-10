import Mathlib

namespace sheila_work_hours_l3513_351332

/-- Represents Sheila's work schedule and earnings -/
structure WorkSchedule where
  hoursPerDayMWF : ℕ
  totalWeeklyEarnings : ℕ
  hourlyRate : ℕ

/-- Calculates the number of hours Sheila works on Tuesday and Thursday -/
def hoursTueThu (schedule : WorkSchedule) : ℕ :=
  let knownDaysHours := 3 * schedule.hoursPerDayMWF
  let knownDaysEarnings := knownDaysHours * schedule.hourlyRate
  let remainingEarnings := schedule.totalWeeklyEarnings - knownDaysEarnings
  remainingEarnings / schedule.hourlyRate

/-- Theorem stating that Sheila works 12 hours on Tuesday and Thursday -/
theorem sheila_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.hoursPerDayMWF = 8)
  (h2 : schedule.totalWeeklyEarnings = 432)
  (h3 : schedule.hourlyRate = 12) : 
  hoursTueThu schedule = 12 := by
  sorry

end sheila_work_hours_l3513_351332


namespace gcd_180_450_l3513_351376

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end gcd_180_450_l3513_351376


namespace infinite_solutions_condition_l3513_351384

theorem infinite_solutions_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end infinite_solutions_condition_l3513_351384


namespace unique_four_digit_number_with_reverse_property_l3513_351352

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def reverse_digits (n : ℕ) : ℕ :=
  let d₁ := n / 1000
  let d₂ := (n / 100) % 10
  let d₃ := (n / 10) % 10
  let d₄ := n % 10
  d₄ * 1000 + d₃ * 100 + d₂ * 10 + d₁

theorem unique_four_digit_number_with_reverse_property :
  ∃! n : ℕ, is_four_digit n ∧ n + 7182 = reverse_digits n :=
by
  -- The proof goes here
  sorry

end unique_four_digit_number_with_reverse_property_l3513_351352


namespace jelly_bean_ratio_l3513_351392

def napoleon_jelly_beans : ℕ := 17
def mikey_jelly_beans : ℕ := 19

def sedrich_jelly_beans : ℕ := napoleon_jelly_beans + 4

def total_jelly_beans : ℕ := napoleon_jelly_beans + sedrich_jelly_beans

theorem jelly_bean_ratio :
  total_jelly_beans = 2 * mikey_jelly_beans :=
by sorry

end jelly_bean_ratio_l3513_351392


namespace cubic_equation_consequence_l3513_351363

theorem cubic_equation_consequence (y : ℝ) (h : y^3 - 3*y = 9) : 
  y^5 - 10*y^2 = -y^2 + 9*y + 27 := by
sorry

end cubic_equation_consequence_l3513_351363


namespace coefficient_a7_equals_negative_eight_l3513_351305

/-- Given x^8 = a₀ + a₁(x+1) + a₂(x+1)² + ... + a₈(x+1)⁸, prove that a₇ = -8 -/
theorem coefficient_a7_equals_negative_eight (x : ℝ) (a : Fin 9 → ℝ) :
  x^8 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + 
        a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 + a 8 * (x + 1)^8 →
  a 7 = -8 := by
sorry

end coefficient_a7_equals_negative_eight_l3513_351305


namespace min_distance_to_origin_l3513_351343

/-- The line equation 4x + 3y - 10 = 0 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y - 10 = 0

/-- The theorem stating the minimum value of m^2 + n^2 for points on the line -/
theorem min_distance_to_origin :
  ∀ m n : ℝ, line_equation m n → ∀ x y : ℝ, line_equation x y → m^2 + n^2 ≤ x^2 + y^2 ∧
  ∃ m₀ n₀ : ℝ, line_equation m₀ n₀ ∧ m₀^2 + n₀^2 = 4 :=
sorry

end min_distance_to_origin_l3513_351343


namespace probability_1AECD_l3513_351329

-- Define the structure of the license plate
structure LicensePlate where
  digit : Fin 10
  vowel1 : Fin 5
  vowel2 : Fin 5
  consonant1 : Fin 21
  consonant2 : Fin 21
  different_consonants : consonant1 ≠ consonant2

-- Define the total number of possible license plates
def total_plates : ℕ := 10 * 5 * 5 * 21 * 20

-- Define the probability of a specific plate
def probability_specific_plate : ℚ := 1 / total_plates

-- Theorem to prove
theorem probability_1AECD :
  probability_specific_plate = 1 / 105000 :=
sorry

end probability_1AECD_l3513_351329


namespace square_minus_double_eq_one_implies_double_square_minus_quadruple_l3513_351347

theorem square_minus_double_eq_one_implies_double_square_minus_quadruple (m : ℝ) :
  m^2 - 2*m = 1 → 2*m^2 - 4*m = 2 := by
  sorry

end square_minus_double_eq_one_implies_double_square_minus_quadruple_l3513_351347


namespace ratio_calculation_l3513_351301

theorem ratio_calculation (P M Q R N : ℚ) :
  R = (40 / 100) * M →
  M = (25 / 100) * Q →
  Q = (30 / 100) * P →
  N = (60 / 100) * P →
  R / N = 1 / 20 := by
sorry

end ratio_calculation_l3513_351301


namespace sum_of_fifth_powers_divisible_by_15_l3513_351313

theorem sum_of_fifth_powers_divisible_by_15 
  (a b c d e : ℤ) 
  (h : a + b + c + d + e = 0) : 
  ∃ k : ℤ, a^5 + b^5 + c^5 + d^5 + e^5 = 15 * k := by
  sorry

end sum_of_fifth_powers_divisible_by_15_l3513_351313


namespace system_solution_l3513_351366

theorem system_solution (a₁ a₂ b₁ b₂ : ℝ) :
  (∃ x y : ℝ, a₁ * x + b₁ * y = 21 ∧ a₂ * x + b₂ * y = 12 ∧ x = 3 ∧ y = 6) →
  (∃ m n : ℝ, a₁ * (2 * m + n) + b₁ * (m - n) = 21 ∧ 
              a₂ * (2 * m + n) + b₂ * (m - n) = 12 ∧
              m = 3 ∧ n = -3) :=
by sorry

end system_solution_l3513_351366


namespace polynomial_equality_l3513_351306

theorem polynomial_equality : 
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 102^5 := by
  sorry

end polynomial_equality_l3513_351306


namespace math_city_intersections_l3513_351377

/-- The number of intersections for n non-parallel streets --/
def intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of streets in Math City --/
def num_streets : ℕ := 10

/-- The number of streets with tunnels --/
def streets_with_tunnels : ℕ := 2

/-- The number of intersections bypassed by each tunnel --/
def bypassed_per_tunnel : ℕ := 1

theorem math_city_intersections :
  intersections num_streets - streets_with_tunnels * bypassed_per_tunnel = 43 := by
  sorry

end math_city_intersections_l3513_351377


namespace reciprocal_of_negative_two_l3513_351339

theorem reciprocal_of_negative_two :
  (∃ x : ℚ, -2 * x = 1) ∧ (∀ x : ℚ, -2 * x = 1 → x = -1/2) :=
by sorry

end reciprocal_of_negative_two_l3513_351339


namespace whitewash_cost_l3513_351334

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
theorem whitewash_cost (room_length room_width room_height : ℝ)
                       (door_length door_width : ℝ)
                       (window_length window_width : ℝ)
                       (rate : ℝ) :
  room_length = 25 →
  room_width = 15 →
  room_height = 12 →
  door_length = 6 →
  door_width = 3 →
  window_length = 4 →
  window_width = 3 →
  rate = 5 →
  (2 * (room_length * room_height + room_width * room_height) -
   (door_length * door_width + 3 * window_length * window_width)) * rate = 4530 := by
  sorry

#check whitewash_cost

end whitewash_cost_l3513_351334


namespace hyperbola_condition_l3513_351396

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (m - 6) = 1 ∧ (m - 2) * (m - 6) < 0

/-- The theorem stating the condition for the equation to represent a hyperbola -/
theorem hyperbola_condition (m : ℝ) : is_hyperbola m ↔ 2 < m ∧ m < 6 := by
  sorry

end hyperbola_condition_l3513_351396


namespace similar_triangles_count_l3513_351312

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Represents an altitude of a triangle -/
structure Altitude :=
  (base apex foot : Point)

/-- Checks if a line is an altitude of a triangle -/
def isAltitude (alt : Altitude) (t : Triangle) : Prop := sorry

/-- Represents the intersection of two lines -/
def lineIntersection (p1 p2 q1 q2 : Point) : Point := sorry

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem -/
theorem similar_triangles_count 
  (ABC : Triangle) 
  (h_acute : isAcute ABC)
  (AL : Altitude)
  (h_AL : isAltitude AL ABC)
  (BM : Altitude)
  (h_BM : isAltitude BM ABC)
  (D : Point)
  (h_D : D = lineIntersection AL.foot BM.foot ABC.A ABC.B) :
  ∃ (pairs : List (Triangle × Triangle)), 
    (∀ (p : Triangle × Triangle), p ∈ pairs → areSimilar p.1 p.2) ∧ 
    pairs.length = 10 ∧
    (∀ (t1 t2 : Triangle), areSimilar t1 t2 → (t1, t2) ∈ pairs ∨ (t2, t1) ∈ pairs) :=
sorry

end similar_triangles_count_l3513_351312


namespace quadratic_root_value_l3513_351333

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 8 * x + 1 = 0

-- Define the root form
def root_form (x m n p : ℝ) : Prop := x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p

-- Theorem statement
theorem quadratic_root_value :
  ∃ (m n p : ℕ+),
    (∀ x : ℝ, quadratic_equation x ↔ root_form x m n p) ∧
    Nat.gcd (Nat.gcd m.val n.val) p.val = 1 ∧
    n = 13 := by sorry

end quadratic_root_value_l3513_351333


namespace percentage_difference_l3513_351381

theorem percentage_difference (x y : ℝ) (h : y = x * (1 + 0.8181818181818181)) :
  x = y * 0.55 :=
sorry

end percentage_difference_l3513_351381


namespace parent_payment_calculation_l3513_351308

/-- Calculates the amount each parent has to pay in different currencies --/
theorem parent_payment_calculation
  (former_salary : ℝ)
  (raise_percentage : ℝ)
  (tax_rate : ℝ)
  (num_kids : ℕ)
  (usd_to_eur : ℝ)
  (usd_to_gbp : ℝ)
  (usd_to_jpy : ℝ)
  (h1 : former_salary = 60000)
  (h2 : raise_percentage = 0.25)
  (h3 : tax_rate = 0.10)
  (h4 : num_kids = 15)
  (h5 : usd_to_eur = 0.85)
  (h6 : usd_to_gbp = 0.75)
  (h7 : usd_to_jpy = 110) :
  let new_salary := former_salary * (1 + raise_percentage)
  let after_tax_salary := new_salary * (1 - tax_rate)
  let amount_per_parent := after_tax_salary / num_kids
  (amount_per_parent / usd_to_eur = 5294.12) ∧
  (amount_per_parent / usd_to_gbp = 6000) ∧
  (amount_per_parent * usd_to_jpy = 495000) :=
by sorry

end parent_payment_calculation_l3513_351308


namespace hanna_erasers_l3513_351378

/-- Given information about erasers owned by Tanya, Rachel, and Hanna -/
theorem hanna_erasers (tanya_erasers : ℕ) (tanya_red_erasers : ℕ) (rachel_erasers : ℕ) (hanna_erasers : ℕ) : 
  tanya_erasers = 20 →
  tanya_red_erasers = tanya_erasers / 2 →
  rachel_erasers = tanya_red_erasers / 2 - 3 →
  hanna_erasers = 2 * rachel_erasers →
  hanna_erasers = 4 := by
sorry

end hanna_erasers_l3513_351378


namespace not_p_sufficient_not_necessary_for_not_q_l3513_351355

-- Define the conditions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, not_p x → not_q x) ∧ 
  ¬(∀ x, not_q x → not_p x) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l3513_351355


namespace negation_of_universal_proposition_l3513_351328

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^3 > x^(1/3)) ↔ (∃ x : ℝ, x > 1 ∧ x^3 ≤ x^(1/3)) :=
by sorry

end negation_of_universal_proposition_l3513_351328


namespace complex_number_in_first_quadrant_l3513_351302

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - Complex.I) * (1 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by
  sorry

end complex_number_in_first_quadrant_l3513_351302


namespace scientific_notation_of_18860000_l3513_351395

theorem scientific_notation_of_18860000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 18860000 = a * (10 : ℝ) ^ n ∧ a = 1.886 ∧ n = 7 := by
  sorry

end scientific_notation_of_18860000_l3513_351395


namespace negation_square_positive_l3513_351385

theorem negation_square_positive :
  (¬ ∀ n : ℕ, n^2 > 0) ↔ (∃ n : ℕ, n^2 ≤ 0) := by sorry

end negation_square_positive_l3513_351385


namespace base8_157_equals_base10_111_l3513_351358

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- Theorem: The base-8 number 157 is equal to the base-10 number 111 --/
theorem base8_157_equals_base10_111 : base8_to_base10 157 = 111 := by
  sorry

end base8_157_equals_base10_111_l3513_351358


namespace water_tank_theorem_l3513_351335

/-- Represents the water tank problem --/
def WaterTankProblem (maxCapacity initialLossRate initialLossDuration
                      secondaryLossRate secondaryLossDuration
                      refillRate refillDuration : ℕ) : Prop :=
  let initialLoss := initialLossRate * initialLossDuration
  let secondaryLoss := secondaryLossRate * secondaryLossDuration
  let totalLoss := initialLoss + secondaryLoss
  let remainingWater := maxCapacity - totalLoss
  let refillAmount := refillRate * refillDuration
  let finalWaterAmount := remainingWater + refillAmount
  maxCapacity - finalWaterAmount = 140000

/-- The water tank theorem --/
theorem water_tank_theorem : WaterTankProblem 350000 32000 5 10000 10 40000 3 := by
  sorry

end water_tank_theorem_l3513_351335


namespace dance_pairs_correct_l3513_351393

/-- The number of ways to form dance pairs given specific knowledge constraints -/
def dance_pairs (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then
    Nat.choose n r * (Nat.factorial n / Nat.factorial (n - r))
  else 0

/-- Theorem stating the correct number of dance pairs -/
theorem dance_pairs_correct (n : ℕ) (r : ℕ) (h : r ≤ n) :
  dance_pairs n r = Nat.choose n r * (Nat.factorial n / Nat.factorial (n - r)) :=
by sorry

end dance_pairs_correct_l3513_351393


namespace cubic_km_to_cubic_m_l3513_351304

/-- Proves that 1 cubic kilometer equals 1,000,000,000 cubic meters -/
theorem cubic_km_to_cubic_m : 
  (∀ (km m : ℝ), km = 1 ∧ m = 1000 ∧ km * 1000 = m) → 
  (1 : ℝ)^3 * 1000^3 = 1000000000 := by
  sorry

end cubic_km_to_cubic_m_l3513_351304


namespace equation_solution_l3513_351303

theorem equation_solution (x : ℝ) :
  x > 6 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 18 := by
  sorry

end equation_solution_l3513_351303


namespace quadratic_root_relation_l3513_351338

theorem quadratic_root_relation (p : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + 3 = 0 ∧ y^2 + p*y + 3 = 0 ∧ y = 3*x) → 
  (p = 4 ∨ p = -4) := by
sorry

end quadratic_root_relation_l3513_351338


namespace remaining_water_l3513_351397

theorem remaining_water (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 11/4 → remaining = initial - used → remaining = 1/4 := by sorry

end remaining_water_l3513_351397


namespace complex_magnitude_one_l3513_351356

theorem complex_magnitude_one (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w^2 + 1/w^2 = s) : 
  Complex.abs w = 1 := by sorry

end complex_magnitude_one_l3513_351356


namespace piglet_straws_l3513_351369

theorem piglet_straws (total_straws : ℕ) (adult_pig_fraction : ℚ) (piglet_fraction : ℚ) (num_piglets : ℕ) :
  total_straws = 300 →
  adult_pig_fraction = 7 / 15 →
  piglet_fraction = 2 / 5 →
  num_piglets = 20 →
  (piglet_fraction * total_straws) / num_piglets = 6 := by
  sorry

end piglet_straws_l3513_351369


namespace function_growth_l3513_351349

theorem function_growth (f : ℕ+ → ℝ) 
  (h : ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (h4 : f 4 ≥ 25) :
  ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 := by
sorry

end function_growth_l3513_351349


namespace inequality_proof_l3513_351324

theorem inequality_proof (x : ℝ) (h1 : x > 4/3) (h2 : x ≠ -5) (h3 : x ≠ 4/3) :
  (6*x^2 + 18*x - 60) / ((3*x - 4)*(x + 5)) < 2 :=
by
  sorry

end inequality_proof_l3513_351324


namespace max_x_on_3x3_grid_l3513_351373

/-- Represents a 3x3 grid where X's can be placed. -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Checks if three X's are aligned in any direction on the grid. -/
def hasThreeAligned (g : Grid) : Bool :=
  sorry

/-- Counts the number of X's placed on the grid. -/
def countX (g : Grid) : Nat :=
  sorry

/-- Theorem stating the maximum number of X's that can be placed on a 3x3 grid
    without three X's aligning vertically, horizontally, or diagonally is 4. -/
theorem max_x_on_3x3_grid :
  (∃ g : Grid, ¬hasThreeAligned g ∧ countX g = 4) ∧
  (∀ g : Grid, ¬hasThreeAligned g → countX g ≤ 4) :=
sorry

end max_x_on_3x3_grid_l3513_351373


namespace power_of_five_preceded_by_coprimes_l3513_351386

theorem power_of_five_preceded_by_coprimes (x : ℕ) : 
  (5^x - 1 - (5^x / 5 - 1) = 7812500) → x = 10 := by
  sorry

end power_of_five_preceded_by_coprimes_l3513_351386


namespace amelia_remaining_money_l3513_351383

-- Define the given amounts and percentages
def initial_amount : ℝ := 60
def first_course_cost : ℝ := 15
def second_course_additional_cost : ℝ := 5
def dessert_percentage : ℝ := 0.25
def drink_percentage : ℝ := 0.20

-- Define the theorem
theorem amelia_remaining_money :
  let second_course_cost := first_course_cost + second_course_additional_cost
  let dessert_cost := dessert_percentage * second_course_cost
  let first_three_courses_cost := first_course_cost + second_course_cost + dessert_cost
  let drink_cost := drink_percentage * first_three_courses_cost
  let total_cost := first_three_courses_cost + drink_cost
  initial_amount - total_cost = 12 := by sorry

end amelia_remaining_money_l3513_351383


namespace symmetry_implies_sum_l3513_351346

/-- Two points are symmetric about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are equal. -/
def symmetric_about_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_implies_sum (a b : ℝ) :
  symmetric_about_y_axis (a - 2, 3) (1, b + 1) → a + b = 3 := by
  sorry

end symmetry_implies_sum_l3513_351346


namespace systematic_sampling_removal_l3513_351353

/-- The number of students that need to be initially removed in a systematic sampling -/
def studentsToRemove (totalStudents sampleSize : ℕ) : ℕ :=
  totalStudents % sampleSize

theorem systematic_sampling_removal (totalStudents sampleSize : ℕ) 
  (h1 : totalStudents = 1387)
  (h2 : sampleSize = 9)
  (h3 : sampleSize > 0) :
  studentsToRemove totalStudents sampleSize = 1 := by
  sorry

end systematic_sampling_removal_l3513_351353


namespace triangle_inscribed_circle_properties_l3513_351327

/-- Given a triangle ABC with semi-perimeter s, inscribed circle center O₁,
    inscribed circle radius r₁, and circumscribed circle radius R -/
theorem triangle_inscribed_circle_properties 
  (A B C O₁ : ℝ × ℝ) (r₁ R s a b c : ℝ) :
  let AO₁ := Real.sqrt ((A.1 - O₁.1)^2 + (A.2 - O₁.2)^2)
  let BO₁ := Real.sqrt ((B.1 - O₁.1)^2 + (B.2 - O₁.2)^2)
  let CO₁ := Real.sqrt ((C.1 - O₁.1)^2 + (C.2 - O₁.2)^2)
  -- Conditions
  s = (a + b + c) / 2 →
  -- Theorem statements
  AO₁^2 = (s / (s - a)) * b * c ∧
  AO₁ * BO₁ * CO₁ = 4 * R * r₁^2 := by
sorry

end triangle_inscribed_circle_properties_l3513_351327


namespace eleven_times_digit_sum_l3513_351361

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem eleven_times_digit_sum :
  ∀ n : ℕ, n = 11 * sum_of_digits n ↔ n = 0 ∨ n = 198 := by sorry

end eleven_times_digit_sum_l3513_351361


namespace transportation_is_car_l3513_351371

/-- Represents different modes of transportation -/
inductive TransportMode
  | Walking
  | Bicycle
  | Car

/-- Definition of a transportation mode with its speed -/
structure Transportation where
  mode : TransportMode
  speed : ℝ  -- Speed in kilometers per hour

/-- Theorem stating that a transportation with speed 70 km/h is a car -/
theorem transportation_is_car (t : Transportation) (h : t.speed = 70) : t.mode = TransportMode.Car := by
  sorry


end transportation_is_car_l3513_351371


namespace problem_solution_l3513_351364

theorem problem_solution (x y : ℝ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : 1/x + 1/y = 1) 
  (h4 : x * y = 9) : 
  y = (9 + 3 * Real.sqrt 5) / 2 := by
sorry

end problem_solution_l3513_351364


namespace function_property_l3513_351315

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def symmetric_about_origin (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f (-x - 1)

-- State the theorem
theorem function_property (h1 : is_even f) (h2 : symmetric_about_origin f) (h3 : f 0 = 1) :
  f (-1) + f 2 = -1 := by sorry

end function_property_l3513_351315


namespace simplify_expression_l3513_351326

theorem simplify_expression (x : ℝ) : (5 - 2*x^2) - (7 - 3*x^2) = -2 + x^2 := by
  sorry

end simplify_expression_l3513_351326


namespace shirley_sold_at_least_20_boxes_l3513_351323

/-- The number of cases Shirley needs to deliver -/
def num_cases : ℕ := 5

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 4

/-- The number of extra boxes, which is unknown but non-negative -/
def extra_boxes : ℕ := sorry

/-- The total number of boxes Shirley sold -/
def total_boxes : ℕ := num_cases * boxes_per_case + extra_boxes

theorem shirley_sold_at_least_20_boxes : total_boxes ≥ 20 := by
  sorry

end shirley_sold_at_least_20_boxes_l3513_351323


namespace polynomial_functional_equation_l3513_351391

theorem polynomial_functional_equation (p : ℝ → ℝ) 
  (h1 : p 3 = 10)
  (h2 : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) :
  ∀ x : ℝ, p x = x^2 + 1 := by sorry

end polynomial_functional_equation_l3513_351391


namespace intersection_locus_l3513_351320

/-- Given two fixed points A(a, 0) and B(b, 0) on the x-axis, and a moving point C(0, c) on the y-axis,
    prove that the locus of the intersection point of line BC and line l (which passes through the origin
    and is perpendicular to AC) satisfies the equation (x - b/2)²/(b²/4) + y²/(ab/4) = 1 -/
theorem intersection_locus (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  ∃ (x y : ℝ → ℝ), ∀ (c : ℝ),
    let l := {p : ℝ × ℝ | p.2 = (a / c) * p.1}
    let bc := {p : ℝ × ℝ | p.1 / b + p.2 / c = 1}
    let intersection := Set.inter l bc
    (x c, y c) ∈ intersection ∧
    (x c - b/2)^2 / (b^2/4) + (y c)^2 / (a*b/4) = 1 :=
sorry

end intersection_locus_l3513_351320


namespace rectangle_rotation_path_length_l3513_351341

/-- The length of the path traveled by point A in a rectangle ABCD after three 90° rotations -/
theorem rectangle_rotation_path_length (AB BC : ℝ) (h1 : AB = 3) (h2 : BC = 5) : 
  let diagonal := Real.sqrt (AB^2 + BC^2)
  let single_rotation_arc := π * diagonal / 2
  3 * single_rotation_arc = (3 * π * Real.sqrt 34) / 2 := by
  sorry

end rectangle_rotation_path_length_l3513_351341


namespace three_Y_five_l3513_351389

def Y (a b : ℤ) : ℤ := b + 10*a - a^2 - b^2

theorem three_Y_five : Y 3 5 = 1 := by
  sorry

end three_Y_five_l3513_351389


namespace samuels_birds_berries_l3513_351354

/-- The number of berries a single bird eats per day -/
def berries_per_day : ℕ := 7

/-- The number of birds Samuel has -/
def samuels_birds : ℕ := 5

/-- The number of days we're considering -/
def days : ℕ := 4

/-- Theorem: Samuel's birds eat 140 berries in 4 days -/
theorem samuels_birds_berries : 
  berries_per_day * samuels_birds * days = 140 := by
  sorry

end samuels_birds_berries_l3513_351354


namespace time_from_velocity_and_displacement_l3513_351380

/-- Given an object with acceleration a, initial velocity V₀, 
    final velocity V, and displacement S, prove the time t 
    taken to reach V from V₀ -/
theorem time_from_velocity_and_displacement 
  (a V₀ V S t : ℝ) 
  (hv : V = a * t + V₀) 
  (hs : S = (1/3) * a * t^3 + V₀ * t) :
  t = (V - V₀) / a :=
sorry

end time_from_velocity_and_displacement_l3513_351380


namespace volume_is_360_l3513_351340

/-- A rectangular parallelepiped with edge lengths 4, 6, and 15 -/
structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  length_eq : length = 4
  width_eq : width = 6
  height_eq : height = 15

/-- The volume of a rectangular parallelepiped -/
def volume (rp : RectangularParallelepiped) : ℝ :=
  rp.length * rp.width * rp.height

/-- Theorem: The volume of the given rectangular parallelepiped is 360 cubic units -/
theorem volume_is_360 (rp : RectangularParallelepiped) : volume rp = 360 := by
  sorry

end volume_is_360_l3513_351340


namespace seller_total_loss_l3513_351374

/-- Represents the total loss of a seller in a transaction with a counterfeit banknote -/
def seller_loss (item_cost change_given fake_note_value real_note_value : ℕ) : ℕ :=
  item_cost + change_given + real_note_value

/-- Theorem stating the total loss of the seller in the given scenario -/
theorem seller_total_loss :
  let item_cost : ℕ := 20
  let customer_payment : ℕ := 100
  let change_given : ℕ := customer_payment - item_cost
  let fake_note_value : ℕ := 100
  let real_note_value : ℕ := 100
  seller_loss item_cost change_given fake_note_value real_note_value = 200 := by
  sorry

#eval seller_loss 20 80 100 100

end seller_total_loss_l3513_351374


namespace coffee_blend_price_l3513_351390

/-- Proves the price of the second blend of coffee given the conditions of the problem -/
theorem coffee_blend_price
  (total_blend : ℝ)
  (target_price : ℝ)
  (first_blend_price : ℝ)
  (first_blend_amount : ℝ)
  (h1 : total_blend = 20)
  (h2 : target_price = 8.4)
  (h3 : first_blend_price = 9)
  (h4 : first_blend_amount = 8)
  : ∃ (second_blend_price : ℝ),
    second_blend_price = 8 ∧
    total_blend * target_price =
      first_blend_amount * first_blend_price +
      (total_blend - first_blend_amount) * second_blend_price :=
by sorry

end coffee_blend_price_l3513_351390


namespace largest_negative_angle_l3513_351316

-- Define the function for angles with the same terminal side as -2002°
def sameTerminalSide (k : ℤ) : ℝ := k * 360 - 2002

-- Theorem statement
theorem largest_negative_angle :
  ∃ (k : ℤ), sameTerminalSide k = -202 ∧
  ∀ (m : ℤ), sameTerminalSide m < 0 → sameTerminalSide m ≤ -202 :=
by sorry

end largest_negative_angle_l3513_351316


namespace sqrt_400_div_2_l3513_351367

theorem sqrt_400_div_2 : Real.sqrt 400 / 2 = 10 := by
  sorry

end sqrt_400_div_2_l3513_351367


namespace map_distance_example_l3513_351399

/-- Given a map scale and an actual distance, calculates the distance on the map -/
def map_distance (scale : ℚ) (actual_distance : ℚ) : ℚ :=
  actual_distance * scale

/-- Theorem: For a map with scale 1:5000000 and actual distance 400km, the map distance is 8cm -/
theorem map_distance_example : 
  let scale : ℚ := 1 / 5000000
  let actual_distance : ℚ := 400 * 100000  -- 400km in cm
  map_distance scale actual_distance = 8 := by
  sorry

#eval map_distance (1 / 5000000) (400 * 100000)

end map_distance_example_l3513_351399


namespace vertical_multiplication_puzzle_l3513_351357

theorem vertical_multiplication_puzzle :
  ∀ a b : ℕ,
    10 < a ∧ a < 20 →
    10 < b ∧ b < 20 →
    100 ≤ a * b ∧ a * b < 1000 →
    (a * b) / 100 = 2 →
    a * b % 10 = 7 →
    (a = 13 ∧ b = 19) ∨ (a = 19 ∧ b = 13) :=
by sorry

end vertical_multiplication_puzzle_l3513_351357


namespace equation_solution_l3513_351350

theorem equation_solution (x y : ℝ) : 
  (4 * x + y = 9) → (y = 9 - 4 * x) := by
sorry

end equation_solution_l3513_351350


namespace unique_equal_intercept_line_l3513_351322

/-- A line with equal intercepts on both axes passing through (2,3) -/
def EqualInterceptLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ a : ℝ, p.1 + p.2 = a ∧ (2 : ℝ) + (3 : ℝ) = a}

/-- The theorem stating that there is exactly one line with equal intercepts passing through (2,3) -/
theorem unique_equal_intercept_line : 
  ∃! a : ℝ, (2 : ℝ) + (3 : ℝ) = a ∧ EqualInterceptLine = {p : ℝ × ℝ | p.1 + p.2 = a} :=
by sorry

end unique_equal_intercept_line_l3513_351322


namespace common_face_sum_l3513_351300

/-- Represents a cube with numbers at its vertices -/
structure NumberedCube where
  vertices : Fin 8 → Nat
  additional : Fin 8 → Nat

/-- The sum of numbers from 1 to n -/
def sum_to_n (n : Nat) : Nat := n * (n + 1) / 2

/-- The sum of numbers on a face of the cube -/
def face_sum (cube : NumberedCube) (face : Fin 6) : Nat :=
  sorry -- Definition of face sum

/-- The theorem stating the common sum on each face -/
theorem common_face_sum (cube : NumberedCube) : 
  (∀ (i j : Fin 6), face_sum cube i = face_sum cube j) → 
  (∀ (i : Fin 8), cube.vertices i ∈ Finset.range 9) →
  (∀ (i : Fin 6), face_sum cube i = 9) :=
sorry

end common_face_sum_l3513_351300


namespace shell_collection_ratio_l3513_351372

theorem shell_collection_ratio :
  ∀ (ben_shells laurie_shells alan_shells : ℕ),
    alan_shells = 4 * ben_shells →
    laurie_shells = 36 →
    alan_shells = 48 →
    ben_shells.gcd laurie_shells = ben_shells →
    (ben_shells : ℚ) / laurie_shells = 1 / 3 := by
  sorry

end shell_collection_ratio_l3513_351372


namespace geometric_sequence_constant_l3513_351321

theorem geometric_sequence_constant (a : ℕ → ℝ) (c : ℝ) :
  a 1 = 2 →
  (∀ n : ℕ, a (n + 1) = a n + c * n) →
  (∃ r : ℝ, r ≠ 1 ∧ a 2 = r * a 1 ∧ a 3 = r * a 2) →
  c = 2 := by
sorry

end geometric_sequence_constant_l3513_351321


namespace largest_non_sum_of_5_and_6_l3513_351318

def is_sum_of_5_and_6 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

theorem largest_non_sum_of_5_and_6 :
  (∀ n : ℕ, n > 19 → n ≤ 50 → is_sum_of_5_and_6 n) ∧
  ¬is_sum_of_5_and_6 19 := by
  sorry

end largest_non_sum_of_5_and_6_l3513_351318


namespace sin_cos_sum_identity_l3513_351330

open Real

theorem sin_cos_sum_identity : 
  sin (15 * π / 180) * cos (75 * π / 180) + cos (15 * π / 180) * sin (105 * π / 180) = 1 := by
  sorry

end sin_cos_sum_identity_l3513_351330


namespace plane_equation_proof_l3513_351345

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Parametric representation of the plane -/
def parametricPlane (s t : ℝ) : Point3D :=
  { x := 2 + 2*s - 3*t
  , y := 1 - 2*s
  , z := 4 + 3*s + 4*t }

/-- The plane equation we want to prove -/
def targetPlane : Plane :=
  { a := 8
  , b := 17
  , c := 6
  , d := -57 }

theorem plane_equation_proof :
  (∀ s t : ℝ, pointOnPlane (parametricPlane s t) targetPlane) ∧
  targetPlane.a > 0 ∧
  Int.gcd (Int.natAbs targetPlane.a) (Int.gcd (Int.natAbs targetPlane.b) (Int.gcd (Int.natAbs targetPlane.c) (Int.natAbs targetPlane.d))) = 1 :=
by sorry

end plane_equation_proof_l3513_351345


namespace smallest_digit_divisible_by_11_l3513_351325

theorem smallest_digit_divisible_by_11 : 
  ∃ (d : Nat), d < 10 ∧ 
    (∀ (x : Nat), x < d → ¬(489000 + x * 100 + 7).ModEq 0 11) ∧
    (489000 + d * 100 + 7).ModEq 0 11 :=
by sorry

end smallest_digit_divisible_by_11_l3513_351325


namespace sum_of_s_and_u_l3513_351307

-- Define complex numbers
variable (p q r s t u : ℝ)

-- Define the conditions
def complex_sum_condition (p q r s t u : ℝ) : Prop :=
  Complex.mk (p + r + t) (q + s + u) = Complex.I * (-7)

-- Theorem statement
theorem sum_of_s_and_u 
  (h1 : q = 5)
  (h2 : t = -p - r)
  (h3 : complex_sum_condition p q r s t u) :
  s + u = -12 := by sorry

end sum_of_s_and_u_l3513_351307


namespace M_inter_N_empty_l3513_351365

/-- Set M of complex numbers -/
def M : Set ℂ :=
  {z | ∃ t : ℝ, t ≠ -1 ∧ t ≠ 0 ∧ z = (t / (1 + t) : ℂ) + Complex.I * ((1 + t) / t : ℂ)}

/-- Set N of complex numbers -/
def N : Set ℂ :=
  {z | ∃ t : ℝ, |t| ≤ 1 ∧ z = (Real.sqrt 2 : ℂ) * (Complex.cos (Real.arcsin t) + Complex.I * Complex.cos (Real.arccos t))}

/-- Theorem stating that the intersection of M and N is empty -/
theorem M_inter_N_empty : M ∩ N = ∅ := by
  sorry

end M_inter_N_empty_l3513_351365


namespace complex_equation_sum_l3513_351342

theorem complex_equation_sum (x y : ℝ) : 
  (x + 2 * Complex.I) * Complex.I = y - Complex.I⁻¹ → x + y = -1 := by
  sorry

end complex_equation_sum_l3513_351342


namespace unique_solution_for_class_representatives_l3513_351370

theorem unique_solution_for_class_representatives (m n : ℕ) : 
  10 ≥ m ∧ m > n ∧ n ≥ 4 →
  ((m - n)^2 = m + n) ↔ (m = 10 ∧ n = 6) :=
by sorry

end unique_solution_for_class_representatives_l3513_351370


namespace tangent_parallel_points_l3513_351394

/-- The function f(x) = x^3 + x + 2 -/
def f (x : ℝ) : ℝ := x^3 + x + 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_parallel_points :
  ∀ x y : ℝ, (f x = y ∧ f' x = 4) ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 4) :=
sorry

end tangent_parallel_points_l3513_351394


namespace percentage_problem_l3513_351319

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x * (x / 100) = 9) : x = 30 := by
  sorry

end percentage_problem_l3513_351319


namespace arithmetic_geometric_sequence_l3513_351348

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 = 2 →                     -- a_1 = 2
  d ≠ 0 →                       -- d ≠ 0
  (a 2) ^ 2 = a 1 * a 5 →       -- a_1, a_2, a_5 form a geometric sequence
  d = 4 :=
by sorry

end arithmetic_geometric_sequence_l3513_351348


namespace floor_sqrt_150_l3513_351382

theorem floor_sqrt_150 : ⌊Real.sqrt 150⌋ = 12 := by sorry

end floor_sqrt_150_l3513_351382


namespace base6_multiplication_addition_l3513_351337

/-- Converts a base 6 number represented as a list of digits to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6, represented as a list of digits -/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- The main theorem statement -/
theorem base6_multiplication_addition :
  let a := base6ToBase10 [1, 1, 1]  -- 111₆
  let b := 2
  let c := base6ToBase10 [2, 0, 2]  -- 202₆
  base10ToBase6 (a * b + c) = [4, 2, 4] := by
  sorry


end base6_multiplication_addition_l3513_351337


namespace cube_skew_pairs_l3513_351311

/-- A cube with 8 vertices and 28 lines passing through any two vertices -/
structure Cube :=
  (vertices : Nat)
  (lines : Nat)
  (h_vertices : vertices = 8)
  (h_lines : lines = 28)

/-- The number of sets of 4 points not in the same plane in the cube -/
def sets_of_four_points (c : Cube) : Nat := 58

/-- The number of pairs of skew lines contributed by each set of 4 points -/
def skew_pairs_per_set : Nat := 3

/-- The total number of pairs of skew lines in the cube -/
def total_skew_pairs (c : Cube) : Nat :=
  (sets_of_four_points c) * skew_pairs_per_set

/-- Theorem: The number of pairs of skew lines in the cube is 174 -/
theorem cube_skew_pairs (c : Cube) : total_skew_pairs c = 174 := by
  sorry

end cube_skew_pairs_l3513_351311


namespace min_value_theorem_l3513_351387

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * Real.sqrt x + 2 / x^2 ≥ 5 ∧
  (3 * Real.sqrt x + 2 / x^2 = 5 ↔ x = 1) := by
  sorry

end min_value_theorem_l3513_351387


namespace two_digit_number_ratio_l3513_351398

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  tens_valid : tens ≥ 1 ∧ tens ≤ 9
  units_valid : units ≤ 9

def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ :=
  10 * n.tens + n.units

def TwoDigitNumber.interchanged (n : TwoDigitNumber) : ℕ :=
  10 * n.units + n.tens

theorem two_digit_number_ratio (n : TwoDigitNumber) 
  (h1 : n.value - n.interchanged = 36)
  (h2 : (n.tens + n.units) - (n.tens - n.units) = 8) :
  n.tens = 2 * n.units :=
sorry

end two_digit_number_ratio_l3513_351398


namespace base_9_to_3_conversion_l3513_351336

def to_base_3 (n : ℕ) : ℕ := sorry

def from_base_9 (n : ℕ) : ℕ := sorry

theorem base_9_to_3_conversion :
  to_base_3 (from_base_9 745) = 211112 := by sorry

end base_9_to_3_conversion_l3513_351336


namespace hexagon_segment_probability_l3513_351309

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of short diagonals in a regular hexagon -/
def num_short_diagonals : ℕ := 6

/-- The number of long diagonals in a regular hexagon -/
def num_long_diagonals : ℕ := 3

/-- The total number of elements in set S -/
def total_elements : ℕ := num_sides + num_short_diagonals + num_long_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := 11 / 35

theorem hexagon_segment_probability :
  (num_sides * (num_sides - 1) + num_short_diagonals * (num_short_diagonals - 1) + num_long_diagonals * (num_long_diagonals - 1)) / (total_elements * (total_elements - 1)) = prob_same_length :=
sorry

end hexagon_segment_probability_l3513_351309


namespace cos_two_alpha_zero_l3513_351368

theorem cos_two_alpha_zero (α : Real) (h : Real.sin (π/6 - α) = Real.cos (π/6 + α)) : 
  Real.cos (2 * α) = 0 := by
sorry

end cos_two_alpha_zero_l3513_351368


namespace smallest_n_mod_20_sum_l3513_351362

theorem smallest_n_mod_20_sum (n : ℕ) : n ≥ 9 ↔ 
  ∀ (S : Finset ℤ), S.card = n → 
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      (a + b) % 20 = (c + d) % 20 :=
sorry

end smallest_n_mod_20_sum_l3513_351362


namespace bread_loaves_l3513_351310

theorem bread_loaves (slices_per_loaf : ℕ) (num_friends : ℕ) (slices_per_friend : ℕ) :
  slices_per_loaf = 15 →
  num_friends = 10 →
  slices_per_friend = 6 →
  (num_friends * slices_per_friend) / slices_per_loaf = 4 :=
by sorry

end bread_loaves_l3513_351310


namespace kimberley_firewood_l3513_351344

def firewood_problem (total houston ela : ℕ) : Prop :=
  total = 35 ∧ houston = 12 ∧ ela = 13

theorem kimberley_firewood (total houston ela : ℕ) 
  (h : firewood_problem total houston ela) : 
  total - (houston + ela) = 10 :=
by sorry

end kimberley_firewood_l3513_351344


namespace specific_hyperbola_conjugate_axis_length_l3513_351351

/-- Represents a hyperbola with equation x^2 - y^2/m = 1 -/
structure Hyperbola where
  m : ℝ
  focus : ℝ × ℝ

/-- The length of the conjugate axis of a hyperbola -/
def conjugate_axis_length (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the length of the conjugate axis for a specific hyperbola -/
theorem specific_hyperbola_conjugate_axis_length :
  ∀ (h : Hyperbola), 
  h.m > 0 ∧ h.focus = (-3, 0) → 
  conjugate_axis_length h = 4 * Real.sqrt 2 := by sorry

end specific_hyperbola_conjugate_axis_length_l3513_351351


namespace cross_product_result_l3513_351317

def u : ℝ × ℝ × ℝ := (-3, 4, 2)
def v : ℝ × ℝ × ℝ := (8, -5, 6)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  (a₂ * b₃ - a₃ * b₂, a₃ * b₁ - a₁ * b₃, a₁ * b₂ - a₂ * b₁)

theorem cross_product_result : cross_product u v = (34, -34, -17) := by
  sorry

end cross_product_result_l3513_351317


namespace min_value_theorem_l3513_351331

theorem min_value_theorem (a b : ℝ) (h1 : a * b - 2 * a - b + 1 = 0) (h2 : a > 1) :
  ∀ x y : ℝ, x * y - 2 * x - y + 1 = 0 → x > 1 → (a + 3) * (b + 2) ≤ (x + 3) * (y + 2) ∧
  ∃ a₀ b₀ : ℝ, a₀ * b₀ - 2 * a₀ - b₀ + 1 = 0 ∧ a₀ > 1 ∧ (a₀ + 3) * (b₀ + 2) = 25 :=
sorry

end min_value_theorem_l3513_351331


namespace triangle_inequality_l3513_351375

theorem triangle_inequality (a b c S r R : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0 ∧ r > 0 ∧ R > 0) :
  (9 * r) / (2 * S) ≤ (1 / a + 1 / b + 1 / c) ∧ (1 / a + 1 / b + 1 / c) ≤ (9 * R) / (4 * S) := by
  sorry

end triangle_inequality_l3513_351375


namespace jimin_has_most_candy_left_l3513_351314

def jimin_fraction : ℚ := 1/9
def taehyung_fraction : ℚ := 1/3
def hoseok_fraction : ℚ := 1/6

theorem jimin_has_most_candy_left : 
  jimin_fraction < taehyung_fraction ∧ 
  jimin_fraction < hoseok_fraction :=
by sorry

end jimin_has_most_candy_left_l3513_351314


namespace f_properties_l3513_351360

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x ∈ Set.Icc (-3) (-2), f x ≤ -4) ∧
  (∀ x ∈ Set.Icc (-3) (-2), f x ≥ -7) ∧
  (∃ x ∈ Set.Icc (-3) (-2), f x = -4) ∧
  (∃ x ∈ Set.Icc (-3) (-2), f x = -7) := by
  sorry

#check f_properties

end f_properties_l3513_351360


namespace dividend_calculation_l3513_351379

theorem dividend_calculation (quotient divisor : ℝ) (h1 : quotient = 0.0012000000000000001) (h2 : divisor = 17) :
  quotient * divisor = 0.0204000000000000027 := by
  sorry

end dividend_calculation_l3513_351379


namespace special_cone_volume_l3513_351359

/-- A cone with base area π and lateral surface in the shape of a semicircle -/
structure SpecialCone where
  /-- The radius of the base of the cone -/
  r : ℝ
  /-- The height of the cone -/
  h : ℝ
  /-- The slant height of the cone -/
  l : ℝ
  /-- The base area is π -/
  base_area : π * r^2 = π
  /-- The lateral surface is a semicircle -/
  lateral_surface : π * l = 2 * π * r

/-- The volume of the special cone is (√3/3)π -/
theorem special_cone_volume (c : SpecialCone) : 
  (1/3) * π * c.r^2 * c.h = (Real.sqrt 3 / 3) * π := by
  sorry


end special_cone_volume_l3513_351359


namespace boys_in_class_l3513_351388

theorem boys_in_class (total : ℕ) (girls_fraction : ℚ) (boys : ℕ) : 
  total = 160 → 
  girls_fraction = 1/4 → 
  boys = total - (girls_fraction * total).num → 
  boys = 120 := by
sorry

end boys_in_class_l3513_351388
