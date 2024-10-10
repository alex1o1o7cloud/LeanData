import Mathlib

namespace max_value_of_x_l2802_280232

theorem max_value_of_x (x : ℕ) : 
  x > 0 ∧ 
  ∃ k : ℕ, x = 4 * k ∧ 
  x^3 < 1728 →
  x ≤ 8 ∧ ∃ y : ℕ, y > 0 ∧ ∃ m : ℕ, y = 4 * m ∧ y^3 < 1728 ∧ y = 8 :=
by sorry

end max_value_of_x_l2802_280232


namespace pencil_packing_problem_l2802_280206

theorem pencil_packing_problem :
  ∃ (a k m : ℤ),
    200 ≤ a ∧ a ≤ 300 ∧
    a % 10 = 7 ∧
    a % 12 = 9 ∧
    a = 60 * m + 57 ∧
    (m = 3 ∨ m = 4) :=
by sorry

end pencil_packing_problem_l2802_280206


namespace sock_pair_count_l2802_280238

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_pairs (white brown blue : ℕ) : ℕ :=
  white * brown + brown * blue + white * blue

/-- Theorem: The number of ways to choose a pair of socks with different colors
    from 4 white, 4 brown, and 2 blue socks is 32 -/
theorem sock_pair_count :
  different_color_pairs 4 4 2 = 32 := by
  sorry

end sock_pair_count_l2802_280238


namespace interior_point_distance_l2802_280227

-- Define the rectangle and point
def Rectangle (E F G H : ℝ × ℝ) : Prop := sorry

def InteriorPoint (P : ℝ × ℝ) (E F G H : ℝ × ℝ) : Prop := 
  Rectangle E F G H ∧ sorry

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem interior_point_distance 
  (E F G H P : ℝ × ℝ) 
  (h_rect : Rectangle E F G H)
  (h_interior : InteriorPoint P E F G H)
  (h_PE : distance P E = 5)
  (h_PH : distance P H = 12)
  (h_PG : distance P G = 13) :
  distance P F = 12 := by
  sorry

end interior_point_distance_l2802_280227


namespace exists_solution_with_y_twelve_l2802_280240

theorem exists_solution_with_y_twelve :
  ∃ (x z t : ℕ+), x + 12 + z + t = 15 := by
sorry

end exists_solution_with_y_twelve_l2802_280240


namespace program_cost_is_40_92_l2802_280236

/-- Represents the cost calculation for a computer program run -/
def program_cost_calculation (milliseconds_per_second : ℝ) 
                             (os_overhead : ℝ) 
                             (cost_per_millisecond : ℝ) 
                             (tape_mounting_cost : ℝ) 
                             (program_runtime_seconds : ℝ) : ℝ :=
  let total_milliseconds := program_runtime_seconds * milliseconds_per_second
  os_overhead + (cost_per_millisecond * total_milliseconds) + tape_mounting_cost

/-- Theorem stating that the total cost for the given program run is $40.92 -/
theorem program_cost_is_40_92 : 
  program_cost_calculation 1000 1.07 0.023 5.35 1.5 = 40.92 := by
  sorry

end program_cost_is_40_92_l2802_280236


namespace A_xor_B_equals_one_three_l2802_280299

-- Define the ⊕ operation
def setXor (M P : Set ℝ) : Set ℝ := {x | x ∈ M ∨ x ∈ P ∧ x ∉ M ∩ P}

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

-- Theorem statement
theorem A_xor_B_equals_one_three : setXor A B = {1, 3} := by sorry

end A_xor_B_equals_one_three_l2802_280299


namespace reseating_arrangements_l2802_280269

/-- Number of ways to reseat n people in n+2 seats with restrictions -/
def T : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| n + 3 => T (n + 2) + T (n + 1)

/-- There are 8 seats in total -/
def total_seats : ℕ := 8

/-- There are 6 people to be seated -/
def num_people : ℕ := 6

theorem reseating_arrangements :
  T num_people = 13 :=
sorry

end reseating_arrangements_l2802_280269


namespace negation_of_universal_proposition_l2802_280228

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end negation_of_universal_proposition_l2802_280228


namespace ginos_brown_bears_l2802_280293

theorem ginos_brown_bears :
  ∀ (total white black brown : ℕ),
    total = 66 →
    white = 24 →
    black = 27 →
    total = white + black + brown →
    brown = 15 := by
  sorry

end ginos_brown_bears_l2802_280293


namespace zero_in_interval_l2802_280298

/-- The function f(x) = log_a x + x - b -/
noncomputable def f (a b x : ℝ) : ℝ := (Real.log x) / (Real.log a) + x - b

/-- The theorem stating that the zero of f(x) lies in (2, 3) -/
theorem zero_in_interval (a b : ℝ) (ha : 0 < a) (ha' : a ≠ 1) 
  (hab : 2 < a ∧ a < 3 ∧ 3 < b ∧ b < 4) :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f a b x₀ = 0 :=
sorry

end zero_in_interval_l2802_280298


namespace susan_vacation_pay_missed_l2802_280276

/-- Calculate the pay missed during Susan's vacation --/
theorem susan_vacation_pay_missed
  (vacation_length : ℕ) -- Length of vacation in weeks
  (work_days_per_week : ℕ) -- Number of work days per week
  (paid_vacation_days : ℕ) -- Number of paid vacation days
  (hourly_rate : ℚ) -- Hourly pay rate
  (hours_per_day : ℕ) -- Number of work hours per day
  (h1 : vacation_length = 2)
  (h2 : work_days_per_week = 5)
  (h3 : paid_vacation_days = 6)
  (h4 : hourly_rate = 15)
  (h5 : hours_per_day = 8) :
  (vacation_length * work_days_per_week - paid_vacation_days) * (hourly_rate * hours_per_day) = 480 :=
by sorry

end susan_vacation_pay_missed_l2802_280276


namespace waynes_age_l2802_280214

theorem waynes_age (birth_year_julia : ℕ) (current_year : ℕ) : 
  birth_year_julia = 1979 → current_year = 2021 →
  ∃ (age_wayne age_peter age_julia : ℕ),
    age_julia = current_year - birth_year_julia ∧
    age_peter = age_julia - 2 ∧
    age_wayne = age_peter - 3 ∧
    age_wayne = 37 :=
by sorry

end waynes_age_l2802_280214


namespace tournament_has_25_players_l2802_280202

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- number of players not in the lowest 5
  total_players : ℕ := n + 5
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  points_top_n : ℕ := (n * (n - 1)) / 2
  points_bottom_5 : ℕ := 10

/-- The theorem stating that a tournament satisfying the given conditions must have 25 players -/
theorem tournament_has_25_players (t : Tournament) : t.total_players = 25 := by
  sorry

#check tournament_has_25_players

end tournament_has_25_players_l2802_280202


namespace extreme_value_implies_m_eq_two_l2802_280253

/-- The function f(x) = x³ - (3/2)x² + m has an extreme value of 3/2 in the interval (0, 2) -/
def has_extreme_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo 0 2, f x = 3/2 ∧ ∀ y ∈ Set.Ioo 0 2, f y ≤ f x

/-- The main theorem stating that if f(x) = x³ - (3/2)x² + m has an extreme value of 3/2 
    in the interval (0, 2), then m = 2 -/
theorem extreme_value_implies_m_eq_two :
  ∀ m : ℝ, has_extreme_value (fun x => x^3 - (3/2)*x^2 + m) m → m = 2 :=
by
  sorry

end extreme_value_implies_m_eq_two_l2802_280253


namespace greatest_integer_radius_l2802_280205

theorem greatest_integer_radius (r : ℕ) : (∀ n : ℕ, n > r → (n : ℝ)^2 * Real.pi ≥ 75 * Real.pi) ∧ r^2 * Real.pi < 75 * Real.pi → r = 8 := by
  sorry

end greatest_integer_radius_l2802_280205


namespace fourth_to_second_quadrant_l2802_280265

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Predicate to check if a point is in the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that if P(a,b) is in the fourth quadrant, then Q(-a,-b) is in the second quadrant -/
theorem fourth_to_second_quadrant (p : Point) :
  is_in_fourth_quadrant p → is_in_second_quadrant (Point.mk (-p.x) (-p.y)) := by
  sorry

end fourth_to_second_quadrant_l2802_280265


namespace cube_volume_from_surface_area_l2802_280215

theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  s > 0 →
  6 * s^2 = 864 →
  s^3 = 1728 := by
sorry

end cube_volume_from_surface_area_l2802_280215


namespace function_properties_l2802_280226

/-- Given that y+6 is directly proportional to x+1 and when x=3, y=2 -/
def proportional_function (x y : ℝ) : Prop :=
  ∃ (k : ℝ), y + 6 = k * (x + 1) ∧ 2 + 6 = k * (3 + 1)

theorem function_properties :
  ∀ x y m : ℝ,
  proportional_function x y →
  (y = 2*x - 4 ∧
   (proportional_function m (-2) → m = 1) ∧
   ¬proportional_function 1 (-3)) :=
by sorry

end function_properties_l2802_280226


namespace costs_equal_at_60_guests_l2802_280262

/-- The number of guests for which the costs of Caesar's and Venus Hall are equal -/
def equal_cost_guests : ℕ := 60

/-- Caesar's room rental cost -/
def caesars_rental : ℕ := 800

/-- Caesar's per-meal cost -/
def caesars_meal : ℕ := 30

/-- Venus Hall's room rental cost -/
def venus_rental : ℕ := 500

/-- Venus Hall's per-meal cost -/
def venus_meal : ℕ := 35

/-- Theorem stating that the costs are equal for the given number of guests -/
theorem costs_equal_at_60_guests : 
  caesars_rental + caesars_meal * equal_cost_guests = 
  venus_rental + venus_meal * equal_cost_guests :=
by sorry

end costs_equal_at_60_guests_l2802_280262


namespace tangent_line_to_ln_curve_l2802_280291

theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ k * x₀ = Real.log x₀ ∧ k = 1 / x₀) → k = 1 / Real.exp 1 := by
  sorry

end tangent_line_to_ln_curve_l2802_280291


namespace issac_utensils_count_l2802_280277

/-- The total number of writing utensils bought by Issac -/
def total_utensils (num_pens : ℕ) (num_pencils : ℕ) : ℕ :=
  num_pens + num_pencils

/-- Theorem stating the total number of writing utensils Issac bought -/
theorem issac_utensils_count :
  ∀ (num_pens : ℕ) (num_pencils : ℕ),
    num_pens = 16 →
    num_pencils = 5 * num_pens + 12 →
    total_utensils num_pens num_pencils = 108 :=
by
  sorry

end issac_utensils_count_l2802_280277


namespace number_divided_by_expression_equals_one_l2802_280213

theorem number_divided_by_expression_equals_one :
  ∃ x : ℝ, x / (5 + 3 / 0.75) = 1 ∧ x = 9 := by
  sorry

end number_divided_by_expression_equals_one_l2802_280213


namespace arithmetic_sequence_specific_term_l2802_280281

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_specific_term
  (seq : ArithmeticSequence)
  (m : ℕ)
  (h1 : seq.S (m - 2) = -4)
  (h2 : seq.S m = 0)
  (h3 : seq.S (m + 2) = 12) :
  seq.a m = 3 :=
sorry

end arithmetic_sequence_specific_term_l2802_280281


namespace balloon_arrangement_count_l2802_280237

/-- The number of letters in the word "BALLOON" -/
def n : ℕ := 7

/-- The number of times the letter 'L' appears in "BALLOON" -/
def l_count : ℕ := 2

/-- The number of times the letter 'O' appears in "BALLOON" -/
def o_count : ℕ := 2

/-- The number of unique arrangements of the letters in "BALLOON" -/
def balloon_arrangements : ℕ := n.factorial / (l_count.factorial * o_count.factorial)

theorem balloon_arrangement_count : balloon_arrangements = 1260 := by
  sorry

end balloon_arrangement_count_l2802_280237


namespace bogatyr_age_l2802_280258

/-- Represents the ages of five wine brands -/
structure WineAges where
  carlo_rosi : ℕ
  franzia : ℕ
  twin_valley : ℕ
  beaulieu_vineyard : ℕ
  bogatyr : ℕ

/-- Defines the relationships between wine ages -/
def valid_wine_ages (ages : WineAges) : Prop :=
  ages.carlo_rosi = 40 ∧
  ages.franzia = 3 * ages.carlo_rosi ∧
  ages.carlo_rosi = 4 * ages.twin_valley ∧
  ages.beaulieu_vineyard = ages.twin_valley / 2 ∧
  ages.bogatyr = 2 * ages.franzia

/-- Theorem: Given the relationships between wine ages, Bogatyr's age is 240 years -/
theorem bogatyr_age (ages : WineAges) (h : valid_wine_ages ages) : ages.bogatyr = 240 := by
  sorry

end bogatyr_age_l2802_280258


namespace adam_students_count_l2802_280222

/-- The number of students Adam teaches per year (except for the first year) -/
def studentsPerYear : ℕ := 50

/-- The number of students Adam teaches in the first year -/
def studentsFirstYear : ℕ := 40

/-- The total number of years Adam teaches -/
def totalYears : ℕ := 10

/-- The total number of students Adam teaches over the given period -/
def totalStudents : ℕ := studentsFirstYear + studentsPerYear * (totalYears - 1)

theorem adam_students_count : totalStudents = 490 := by
  sorry

end adam_students_count_l2802_280222


namespace total_fruits_is_112_l2802_280252

/-- The number of apples and pears satisfying the given conditions -/
def total_fruits (apples pears : ℕ) : Prop :=
  ∃ (bags : ℕ),
    (5 * bags + 4 = apples) ∧
    (3 * bags = pears - 12) ∧
    (7 * bags = apples) ∧
    (3 * bags + 12 = pears)

/-- Theorem stating that the total number of fruits is 112 -/
theorem total_fruits_is_112 :
  ∃ (apples pears : ℕ), total_fruits apples pears ∧ apples + pears = 112 :=
sorry

end total_fruits_is_112_l2802_280252


namespace min_value_expression_l2802_280244

/-- Given that x₁ and x₂ are the roots of the equations x + exp x = 3 and x + log x = 3 respectively,
    and x₁ + x₂ = a + b where a and b are positive real numbers,
    prove that the minimum value of (7b² + 1) / (ab) is 2. -/
theorem min_value_expression (x₁ x₂ a b : ℝ) : 
  (∃ (x : ℝ), x + Real.exp x = 3 ∧ x = x₁) →
  (∃ (x : ℝ), x + Real.log x = 3 ∧ x = x₂) →
  x₁ + x₂ = a + b →
  a > 0 →
  b > 0 →
  (∀ c d : ℝ, c > 0 → d > 0 → c + d = a + b → (7 * d^2 + 1) / (c * d) ≥ 2) ∧
  (∃ e f : ℝ, e > 0 ∧ f > 0 ∧ e + f = a + b ∧ (7 * f^2 + 1) / (e * f) = 2) :=
by sorry

end min_value_expression_l2802_280244


namespace factorization_equality_l2802_280295

theorem factorization_equality (a b c d : ℝ) :
  a * (b - c)^3 + b * (c - d)^3 + c * (d - a)^3 + d * (a - b)^3 = 
  (a - b) * (b - c) * (c - d) * (d - a) * (a + b + c + d) := by
  sorry

end factorization_equality_l2802_280295


namespace cubic_root_ratio_l2802_280263

theorem cubic_root_ratio (a b c d : ℝ) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 2 ∨ x = 3) → 
  c / d = -1 / 6 := by
sorry

end cubic_root_ratio_l2802_280263


namespace intersection_count_l2802_280207

/-- Two circles in a plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- The number of intersection points between two circles -/
def intersectionPoints (circles : TwoCircles) : ℕ :=
  sorry

/-- Theorem: The number of intersection points between the given circles is 4 -/
theorem intersection_count : 
  let circles : TwoCircles := {
    center1 := (0, 3),
    radius1 := 3,
    center2 := (3/2, 0),
    radius2 := 3/2
  }
  intersectionPoints circles = 4 := by sorry

end intersection_count_l2802_280207


namespace max_area_and_front_wall_length_l2802_280280

/-- The material cost function for the house -/
def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

/-- The constraint on the material cost -/
def cost_constraint (x y : ℝ) : Prop := material_cost x y ≤ 32000

/-- The area of the house -/
def house_area (x y : ℝ) : ℝ := x * y

/-- Theorem stating the maximum area and corresponding front wall length -/
theorem max_area_and_front_wall_length :
  ∃ (x y : ℝ), 
    cost_constraint x y ∧ 
    ∀ (x' y' : ℝ), cost_constraint x' y' → house_area x' y' ≤ house_area x y ∧
    house_area x y = 100 ∧
    x = 20 / 3 :=
sorry

end max_area_and_front_wall_length_l2802_280280


namespace triangle_altitude_l2802_280246

theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 960 ∧ base = 48 ∧ area = (1/2) * base * altitude →
  altitude = 40 := by
sorry

end triangle_altitude_l2802_280246


namespace club_membership_count_l2802_280250

theorem club_membership_count :
  let tennis : ℕ := 138
  let baseball : ℕ := 255
  let both : ℕ := 94
  let neither : ℕ := 11
  tennis + baseball - both + neither = 310 :=
by sorry

end club_membership_count_l2802_280250


namespace set_operations_and_range_l2802_280233

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

theorem set_operations_and_range :
  ∀ a : ℝ, (B ∪ C a = C a) →
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 4}) ∧
  (A ∪ B = {x : ℝ | x ≥ 2}) ∧
  ((U \ (A ∪ B)) = {x : ℝ | 0 < x ∧ x < 2}) ∧
  ((U \ A) ∩ B = {x : ℝ | x ≥ 4}) ∧
  (∀ x : ℝ, x > -6 ↔ ∃ y : ℝ, y ∈ C x ∧ y ∉ B) :=
by sorry

end set_operations_and_range_l2802_280233


namespace power_multiplication_division_equality_l2802_280242

theorem power_multiplication_division_equality : (15^2 * 8^3) / 256 = 450 := by sorry

end power_multiplication_division_equality_l2802_280242


namespace no_zeros_implies_a_less_than_negative_one_l2802_280264

theorem no_zeros_implies_a_less_than_negative_one (a : ℝ) :
  (∀ x : ℝ, 4^x - 2^(x+1) - a ≠ 0) → a < -1 :=
by sorry

end no_zeros_implies_a_less_than_negative_one_l2802_280264


namespace circle_intersection_theorem_l2802_280284

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y + 2*x - 4*y + a = 0

-- Define the midpoint M
def midpoint_M (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

-- Define the chord length
def chord_length (l : ℝ) : Prop :=
  l = 2 * Real.sqrt 7

-- Main theorem
theorem circle_intersection_theorem (a : ℝ) :
  (∃ x y : ℝ, circle_C a x y ∧ midpoint_M x y) →
  (a < 3 ∧
   ∃ k b : ℝ, k = 1 ∧ b = 1 ∧ ∀ x y : ℝ, y = k*x + b) ∧
  (∀ l : ℝ, chord_length l →
    ∀ x y : ℝ, circle_C a x y ↔ (x+1)^2 + (y-2)^2 = 9) :=
sorry

end circle_intersection_theorem_l2802_280284


namespace product_of_five_terms_l2802_280285

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_five_terms
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a3 : a 3 = -1) :
  a 1 * a 2 * a 3 * a 4 * a 5 = -1 :=
by sorry

end product_of_five_terms_l2802_280285


namespace inequality_preservation_l2802_280278

theorem inequality_preservation (a b : ℝ) : a < b → 1 - a > 1 - b := by
  sorry

end inequality_preservation_l2802_280278


namespace fraction_exceeding_by_30_l2802_280220

theorem fraction_exceeding_by_30 (x : ℚ) : 
  48 = 48 * x + 30 → x = 3 / 8 := by
  sorry

end fraction_exceeding_by_30_l2802_280220


namespace initial_investment_is_200_l2802_280254

/-- Represents the simple interest calculation -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating that given the conditions, the initial investment is $200 -/
theorem initial_investment_is_200 
  (P : ℝ) 
  (h1 : simpleInterest P (1/15) 3 = 240) 
  (h2 : simpleInterest 150 (1/15) 6 = 210) : 
  P = 200 := by
  sorry

#check initial_investment_is_200

end initial_investment_is_200_l2802_280254


namespace count_solutions_eq_338350_l2802_280266

/-- The number of distinct integer solutions to |x| + |y| < 100 -/
def count_solutions : ℕ :=
  (Finset.sum (Finset.range 100) (fun k => (k + 1)^2) : ℕ)

/-- Theorem stating the correct number of solutions -/
theorem count_solutions_eq_338350 : count_solutions = 338350 := by
  sorry

end count_solutions_eq_338350_l2802_280266


namespace projection_line_equation_l2802_280268

/-- The line l passing through a point P that is the projection of the origin onto l -/
structure ProjectionLine where
  -- The coordinates of point P
  px : ℝ
  py : ℝ
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- P is on the line
  point_on_line : a * px + b * py + c = 0
  -- P is the projection of the origin onto the line
  is_projection : a * px + b * py = 0

/-- The equation of the line l given the projection point P(-2, 1) -/
theorem projection_line_equation (l : ProjectionLine) 
  (h1 : l.px = -2) 
  (h2 : l.py = 1) : 
  l.a = 2 ∧ l.b = -1 ∧ l.c = 5 := by
  sorry

end projection_line_equation_l2802_280268


namespace long_track_five_times_short_track_l2802_280247

/-- Represents the lengths of the short and long tracks -/
structure TrackLengths where
  short : ℝ
  long : ℝ

/-- Represents the training schedule for a week -/
structure WeekSchedule where
  days : ℕ
  longTracksPerDay : ℕ
  shortTracksPerDay : ℕ

/-- Calculates the total distance run in a week -/
def totalDistance (t : TrackLengths) (w : WeekSchedule) : ℝ :=
  w.days * (w.longTracksPerDay * t.long + w.shortTracksPerDay * t.short)

theorem long_track_five_times_short_track 
  (t : TrackLengths) 
  (w1 w2 : WeekSchedule) 
  (h1 : w1.days = 6 ∧ w1.longTracksPerDay = 1 ∧ w1.shortTracksPerDay = 2)
  (h2 : w2.days = 7 ∧ w2.longTracksPerDay = 1 ∧ w2.shortTracksPerDay = 1)
  (h3 : totalDistance t w1 = 5000)
  (h4 : totalDistance t w1 = totalDistance t w2) :
  t.long = 5 * t.short := by
  sorry

end long_track_five_times_short_track_l2802_280247


namespace min_value_of_f_l2802_280297

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end min_value_of_f_l2802_280297


namespace office_employee_count_l2802_280287

/-- Proves that the total number of employees in an office is 100 given specific salary and employee count conditions. -/
theorem office_employee_count :
  let avg_salary : ℚ := 720
  let officer_salary : ℚ := 1320
  let manager_salary : ℚ := 840
  let worker_salary : ℚ := 600
  let officer_count : ℕ := 10
  let manager_count : ℕ := 20
  ∃ (worker_count : ℕ),
    (officer_count : ℚ) * officer_salary + (manager_count : ℚ) * manager_salary + (worker_count : ℚ) * worker_salary =
    ((officer_count + manager_count + worker_count) : ℚ) * avg_salary ∧
    officer_count + manager_count + worker_count = 100 :=
by sorry

end office_employee_count_l2802_280287


namespace max_xy_value_l2802_280274

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 6 * x + 8 * y = 72) (h4 : x = 2 * y) :
  ∃ (max_xy : ℝ), max_xy = 25.92 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 6 * x' + 8 * y' = 72 → x' = 2 * y' → x' * y' ≤ max_xy :=
by sorry

end max_xy_value_l2802_280274


namespace function_inequality_l2802_280289

/-- Given functions f and g, prove that if f(x) ≥ g(x) - exp(x) for all x ≥ 1, then a ≥ 1/(2*exp(1)) -/
theorem function_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → a * x - Real.exp x ≥ Real.log x / x - Real.exp x) →
  a ≥ 1 / (2 * Real.exp 1) := by
sorry

end function_inequality_l2802_280289


namespace x_minus_y_equals_three_l2802_280294

theorem x_minus_y_equals_three 
  (h1 : 3 * x - 5 * y = 5) 
  (h2 : x / (x + y) = 5 / 7) : 
  x - y = 3 := by sorry

end x_minus_y_equals_three_l2802_280294


namespace right_triangle_third_side_l2802_280261

theorem right_triangle_third_side (x y z : ℝ) : 
  (x > 0 ∧ y > 0 ∧ z > 0) →  -- positive sides
  (x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2) →  -- right triangle condition
  (|x - 4| + Real.sqrt (y - 3) = 0) →  -- given equation
  (z = 5 ∨ z = Real.sqrt 7) := by
sorry

end right_triangle_third_side_l2802_280261


namespace ellipse_and_line_intersection_l2802_280259

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem ellipse_and_line_intersection
  (C : Ellipse)
  (h_point : C.a^2 * (6 / C.a^2 + 1 / C.b^2) = C.a^2) -- Point (√6, 1) lies on the ellipse
  (h_focus : C.a^2 - C.b^2 = 4) -- Left focus is at (-2, 0)
  (m : ℝ)
  (h_distinct : ∃ (A B : Point), A ≠ B ∧
    C.a^2 * ((A.x^2 / C.a^2) + (A.y^2 / C.b^2)) = C.a^2 ∧
    C.a^2 * ((B.x^2 / C.a^2) + (B.y^2 / C.b^2)) = C.a^2 ∧
    A.y = A.x + m ∧ B.y = B.x + m)
  (h_midpoint : ∃ (M : Point), M.x^2 + M.y^2 = 1 ∧
    ∃ (A B : Point), A ≠ B ∧
      C.a^2 * ((A.x^2 / C.a^2) + (A.y^2 / C.b^2)) = C.a^2 ∧
      C.a^2 * ((B.x^2 / C.a^2) + (B.y^2 / C.b^2)) = C.a^2 ∧
      A.y = A.x + m ∧ B.y = B.x + m ∧
      M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2) :
  C.a = 2 * Real.sqrt 2 ∧ C.b = 2 ∧ m = 3 * Real.sqrt 5 / 5 ∨ m = -3 * Real.sqrt 5 / 5 := by
  sorry

end ellipse_and_line_intersection_l2802_280259


namespace sector_area_l2802_280217

-- Define the parameters
def arc_length : ℝ := 1
def radius : ℝ := 4

-- Define the theorem
theorem sector_area : 
  let θ := arc_length / radius
  (1/2) * radius^2 * θ = 2 := by sorry

end sector_area_l2802_280217


namespace table_tennis_tournament_l2802_280245

theorem table_tennis_tournament (x : ℕ) :
  let sixth_graders := 2 * x
  let seventh_graders := x
  let total_participants := sixth_graders + seventh_graders
  let total_matches := total_participants * (total_participants - 1) / 2
  let matches_between_grades := sixth_graders * seventh_graders
  let matches_among_sixth := sixth_graders * (sixth_graders - 1) / 2
  let matches_among_seventh := seventh_graders * (seventh_graders - 1) / 2
  let matches_won_by_sixth := matches_among_sixth + matches_between_grades / 2
  let matches_won_by_seventh := matches_among_seventh + matches_between_grades / 2
  matches_won_by_seventh = (matches_won_by_sixth * 14) / 10 →
  total_participants = 9 :=
by sorry

end table_tennis_tournament_l2802_280245


namespace ones_digit_of_nine_to_27_l2802_280257

def ones_digit (n : ℕ) : ℕ := n % 10

def power_of_nine_ones_digit (n : ℕ) : ℕ :=
  if n % 2 = 1 then 9 else 1

theorem ones_digit_of_nine_to_27 :
  ones_digit (9^27) = 9 :=
by
  sorry

end ones_digit_of_nine_to_27_l2802_280257


namespace scrap_iron_average_l2802_280234

theorem scrap_iron_average (total_friends : Nat) (total_average : ℝ) (ivan_amount : ℝ) :
  total_friends = 5 →
  total_average = 55 →
  ivan_amount = 43 →
  let total_amount := total_friends * total_average
  let remaining_amount := total_amount - ivan_amount
  let remaining_friends := total_friends - 1
  (remaining_amount / remaining_friends : ℝ) = 58 := by
  sorry

end scrap_iron_average_l2802_280234


namespace angle_problem_l2802_280210

theorem angle_problem (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180)
  (h2 : angle2 + angle3 + angle4 = 180)
  (h3 : angle1 = 70)
  (h4 : angle3 = 40) : 
  angle4 = 30 := by
  sorry

end angle_problem_l2802_280210


namespace smallest_n_for_irreducible_fractions_l2802_280273

theorem smallest_n_for_irreducible_fractions : 
  ∃ (n : ℕ), n = 35 ∧ 
  (∀ k : ℕ, 7 ≤ k ∧ k ≤ 31 → Nat.gcd k (n + k + 2) = 1) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, 7 ≤ k ∧ k ≤ 31 ∧ Nat.gcd k (m + k + 2) ≠ 1) := by
  sorry

end smallest_n_for_irreducible_fractions_l2802_280273


namespace sum_of_integers_l2802_280225

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 120) : 
  x.val + y.val = 2 * Real.sqrt 34 := by
  sorry

end sum_of_integers_l2802_280225


namespace max_necklaces_is_five_l2802_280288

/-- Represents the number of beads of each color required for a single necklace -/
structure NecklacePattern where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Represents the total number of beads available for each color -/
structure AvailableBeads where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Calculates the maximum number of complete necklaces that can be made -/
def maxNecklaces (pattern : NecklacePattern) (available : AvailableBeads) : ℕ :=
  min (available.green / pattern.green)
      (min (available.white / pattern.white)
           (available.orange / pattern.orange))

/-- Theorem stating that given the specific bead counts, the maximum number of necklaces is 5 -/
theorem max_necklaces_is_five :
  let pattern := NecklacePattern.mk 9 6 3
  let available := AvailableBeads.mk 45 45 45
  maxNecklaces pattern available = 5 := by
  sorry

end max_necklaces_is_five_l2802_280288


namespace sum_of_squares_of_roots_l2802_280219

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 9*x + 14 = 0 → ∃ s₁ s₂ : ℝ, s₁ + s₂ = 9 ∧ s₁ * s₂ = 14 ∧ s₁^2 + s₂^2 = 53 :=
by sorry

end sum_of_squares_of_roots_l2802_280219


namespace total_is_41X_l2802_280204

/-- Represents the number of people in different categories of a community -/
structure Community where
  children : ℕ
  teenagers : ℕ
  women : ℕ
  men : ℕ

/-- Defines a community with the given relationships between categories -/
def specialCommunity (X : ℕ) : Community where
  children := X
  teenagers := 4 * X
  women := 3 * (4 * X)
  men := 2 * (3 * (4 * X))

/-- Calculates the total number of people in a community -/
def totalPeople (c : Community) : ℕ :=
  c.children + c.teenagers + c.women + c.men

/-- Theorem stating that the total number of people in the special community is 41X -/
theorem total_is_41X (X : ℕ) :
  totalPeople (specialCommunity X) = 41 * X := by
  sorry

end total_is_41X_l2802_280204


namespace book_selling_price_l2802_280235

/-- Calculates the selling price of a book given its cost price and profit percentage. -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem stating that a book with a cost price of $60 and a profit percentage of 30% has a selling price of $78. -/
theorem book_selling_price :
  selling_price 60 30 = 78 := by
  sorry

end book_selling_price_l2802_280235


namespace arithmetic_progression_cube_sum_l2802_280292

theorem arithmetic_progression_cube_sum (k x y z : ℤ) :
  (x < y ∧ y < z) →  -- x, y, z form an increasing sequence
  (z - y = y - x) →  -- x, y, z form an arithmetic progression
  (k * y^3 = x^3 + z^3) →  -- given equation
  ∃ t : ℤ, k = 2 * (3 * t^2 + 1) := by
sorry

end arithmetic_progression_cube_sum_l2802_280292


namespace rectangular_distance_problem_l2802_280203

-- Define the rectangular distance function
def rectangular_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define points A, O, and B
def A : ℝ × ℝ := (-1, 3)
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the line equation
def on_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

theorem rectangular_distance_problem :
  (rectangular_distance A.1 A.2 O.1 O.2 = 4) ∧
  (∃ min_dist : ℝ, min_dist = 3 ∧
    ∀ x y : ℝ, on_line x y →
      rectangular_distance B.1 B.2 x y ≥ min_dist) :=
by sorry

end rectangular_distance_problem_l2802_280203


namespace expand_expression_l2802_280251

theorem expand_expression (x : ℝ) : 20 * (3 * x + 7 - 2 * x^2) = 60 * x + 140 - 40 * x^2 := by
  sorry

end expand_expression_l2802_280251


namespace calf_grazing_area_increase_l2802_280224

/-- The additional area a calf can graze when its rope is increased from 10 m to 35 m -/
theorem calf_grazing_area_increase : 
  let initial_radius : ℝ := 10
  let increased_radius : ℝ := 35
  let additional_area := π * increased_radius^2 - π * initial_radius^2
  additional_area = 1125 * π := by
  sorry

end calf_grazing_area_increase_l2802_280224


namespace arithmetic_mean_problem_l2802_280231

theorem arithmetic_mean_problem (x y : ℝ) : 
  ((x + 10) + 18 + 3*x + 12 + (3*x + 6) + y) / 6 = 26 → x = 90/7 :=
by sorry

end arithmetic_mean_problem_l2802_280231


namespace equation_has_real_roots_l2802_280229

theorem equation_has_real_roots (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) + 2 * x :=
sorry

end equation_has_real_roots_l2802_280229


namespace bad_carrots_l2802_280279

/-- Given the number of carrots picked by Carol and her mother, and the number of good carrots,
    calculate the number of bad carrots. -/
theorem bad_carrots (carol_carrots mother_carrots good_carrots : ℕ) : 
  carol_carrots = 29 → mother_carrots = 16 → good_carrots = 38 →
  carol_carrots + mother_carrots - good_carrots = 7 := by
  sorry

#check bad_carrots

end bad_carrots_l2802_280279


namespace bacteria_urea_phenol_red_l2802_280248

/-- Represents the color of the phenol red indicator -/
inductive IndicatorColor
| Blue
| Red
| Black
| Brown

/-- Represents the pH level of the medium -/
inductive pHLevel
| Acidic
| Neutral
| Alkaline

/-- Represents a culture medium -/
structure CultureMedium where
  nitrogenSource : String
  indicator : String
  pH : pHLevel

/-- Represents the bacterial culture -/
structure BacterialCulture where
  medium : CultureMedium
  bacteriaPresent : Bool

/-- Function to determine the color of phenol red based on pH -/
def phenolRedColor (pH : pHLevel) : IndicatorColor :=
  match pH with
  | pHLevel.Alkaline => IndicatorColor.Red
  | _ => IndicatorColor.Blue  -- Simplified for this problem

/-- Main theorem to prove -/
theorem bacteria_urea_phenol_red 
  (culture : BacterialCulture)
  (h1 : culture.medium.nitrogenSource = "urea")
  (h2 : culture.medium.indicator = "phenol red")
  (h3 : culture.bacteriaPresent = true) :
  phenolRedColor culture.medium.pH = IndicatorColor.Red :=
sorry

end bacteria_urea_phenol_red_l2802_280248


namespace expansion_sum_l2802_280272

theorem expansion_sum (d : ℝ) (h : d ≠ 0) :
  let expansion := (15*d + 21 + 17*d^2) * (3*d + 4)
  ∃ (a b c e : ℝ), expansion = a*d^3 + b*d^2 + c*d + e ∧ a + b + c + e = 371 := by
  sorry

end expansion_sum_l2802_280272


namespace range_of_m_l2802_280282

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem range_of_m (m : ℝ) : (A ∩ B m = B m) → m ≤ 3 := by
  sorry

end range_of_m_l2802_280282


namespace minimum_school_payment_l2802_280239

/-- The minimum amount a school should pay for cinema tickets -/
theorem minimum_school_payment
  (individual_price : ℝ)
  (group_price : ℝ)
  (group_size : ℕ)
  (student_discount : ℝ)
  (num_students : ℕ)
  (h1 : individual_price = 6)
  (h2 : group_price = 40)
  (h3 : group_size = 10)
  (h4 : student_discount = 0.1)
  (h5 : num_students = 1258) :
  ∃ (min_payment : ℝ),
    min_payment = 4536 ∧
    min_payment ≤ (↑(num_students / group_size) * group_price * (1 - student_discount)) + 
                  (↑(num_students % group_size) * individual_price * (1 - student_discount)) :=
by
  sorry

#eval 1258 / 10 * 40 * 0.9

end minimum_school_payment_l2802_280239


namespace parallel_vectors_m_value_l2802_280256

theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (-2, m) →
  (∃ (k : ℝ), a = k • b) →
  m = -4 := by
  sorry

end parallel_vectors_m_value_l2802_280256


namespace point_direction_form_equation_l2802_280286

/-- The point-direction form equation of a line with direction vector (2, -3) passing through the point (1, 0) -/
theorem point_direction_form_equation (x y : ℝ) : 
  let direction_vector : ℝ × ℝ := (2, -3)
  let point : ℝ × ℝ := (1, 0)
  let line_equation := (x - point.1) / direction_vector.1 = y / direction_vector.2
  line_equation = ((x - 1) / 2 = y / (-3))
  := by sorry

end point_direction_form_equation_l2802_280286


namespace total_buyers_is_140_l2802_280218

/-- The number of buyers in a grocery store over three consecutive days -/
structure BuyerCount where
  day_before_yesterday : ℕ
  yesterday : ℕ
  today : ℕ

/-- Conditions for the buyer count problem -/
def buyer_count_conditions (b : BuyerCount) : Prop :=
  b.today = b.yesterday + 40 ∧
  b.yesterday = b.day_before_yesterday / 2 ∧
  b.day_before_yesterday = 50

/-- The total number of buyers over three days -/
def total_buyers (b : BuyerCount) : ℕ :=
  b.day_before_yesterday + b.yesterday + b.today

/-- Theorem stating that given the conditions, the total number of buyers is 140 -/
theorem total_buyers_is_140 (b : BuyerCount) (h : buyer_count_conditions b) :
  total_buyers b = 140 := by
  sorry

end total_buyers_is_140_l2802_280218


namespace package_cost_theorem_l2802_280271

/-- The cost of a 12-roll package of paper towels -/
def package_cost : ℝ := 9

/-- The cost of one roll sold individually -/
def individual_roll_cost : ℝ := 1

/-- The percent savings per roll for the 12-roll package -/
def percent_savings : ℝ := 0.25

/-- The number of rolls in a package -/
def rolls_per_package : ℕ := 12

theorem package_cost_theorem : 
  package_cost = rolls_per_package * (individual_roll_cost * (1 - percent_savings)) :=
by sorry

end package_cost_theorem_l2802_280271


namespace smallest_positive_integer_congruence_l2802_280216

theorem smallest_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 3457) % 15 = 1537 % 15 ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 3457) % 15 = 1537 % 15 → x ≤ y :=
by sorry

end smallest_positive_integer_congruence_l2802_280216


namespace function_composition_l2802_280211

theorem function_composition (f : ℝ → ℝ) (x : ℝ) : 
  (∀ y, f y = y^2 + 2*y - 1) → f (x - 1) = x^2 - 2 := by
  sorry

end function_composition_l2802_280211


namespace min_value_cubic_function_l2802_280275

/-- A cubic function f(x) = (a/3)x^3 + bx^2 + cx + d is monotonically increasing on ℝ 
    if and only if its derivative is non-negative for all x ∈ ℝ -/
def monotonically_increasing (a b c : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * b * x + c ≥ 0

/-- The theorem stating the minimum value of (a + 2b + 3c)/(b - a) 
    for a monotonically increasing cubic function with a < b -/
theorem min_value_cubic_function (a b c : ℝ) 
    (h1 : a < b) 
    (h2 : monotonically_increasing a b c) : 
  (a + 2*b + 3*c) / (b - a) ≥ 8 + 6 * Real.sqrt 2 :=
sorry

end min_value_cubic_function_l2802_280275


namespace sin_cos_105_15_identity_l2802_280223

theorem sin_cos_105_15_identity : 
  Real.sin (105 * π / 180) * Real.sin (15 * π / 180) - 
  Real.cos (105 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
sorry

end sin_cos_105_15_identity_l2802_280223


namespace inequality_condition_l2802_280283

theorem inequality_condition (a b : ℝ) (h : a * Real.sqrt a > b * Real.sqrt b) : a > b ∧ b > 0 := by
  sorry

end inequality_condition_l2802_280283


namespace max_intersections_circle_quadrilateral_l2802_280212

/-- A circle in a 2D plane -/
structure Circle where
  -- We don't need to define the specifics of a circle for this problem

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  -- We don't need to define the specifics of a quadrilateral for this problem

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- The maximum number of intersections between a line segment and a circle -/
def max_intersections_line_circle : ℕ := 2

/-- Theorem: The maximum number of intersection points between a circle and a quadrilateral is 8 -/
theorem max_intersections_circle_quadrilateral (c : Circle) (q : Quadrilateral) :
  (quadrilateral_sides * max_intersections_line_circle) = 8 := by
  sorry

#check max_intersections_circle_quadrilateral

end max_intersections_circle_quadrilateral_l2802_280212


namespace eight_b_plus_one_composite_l2802_280208

theorem eight_b_plus_one_composite (a b : ℕ) (h1 : a > b) (h2 : a - b = 5 * b^2 - 4 * a^2) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ 8 * b + 1 = x * y :=
sorry

end eight_b_plus_one_composite_l2802_280208


namespace classroom_students_count_l2802_280260

theorem classroom_students_count :
  ∃! n : ℕ, n < 60 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 53 :=
by sorry

end classroom_students_count_l2802_280260


namespace projection_squared_magnitude_l2802_280230

-- Define the 3D Cartesian coordinate system
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define point A
def A : Point3D := ⟨3, 7, -4⟩

-- Define point B as the projection of A onto the xOz plane
def B : Point3D := ⟨A.x, 0, A.z⟩

-- Define the squared magnitude of a vector
def squaredMagnitude (p : Point3D) : ℝ :=
  p.x^2 + p.y^2 + p.z^2

-- Theorem statement
theorem projection_squared_magnitude :
  squaredMagnitude B = 25 := by sorry

end projection_squared_magnitude_l2802_280230


namespace one_solution_condition_l2802_280209

theorem one_solution_condition (a : ℝ) :
  (∃! x : ℝ, x ≠ -4 ∧ x ≠ 1 ∧ |x + 1| = |x - 4| + a) ↔ a ∈ Set.Ioo (-5 : ℝ) (-1) ∪ Set.Ioo (-1 : ℝ) 5 :=
by sorry

end one_solution_condition_l2802_280209


namespace game_ends_in_45_rounds_l2802_280243

/-- Represents the state of the game with token counts for each player -/
structure GameState where
  playerA : ℕ
  playerB : ℕ
  playerC : ℕ

/-- Applies one round of the game rules to the current state -/
def applyRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (any player has 0 tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def countRounds (initialState : GameState) : ℕ :=
  sorry

theorem game_ends_in_45_rounds :
  let initialState : GameState := ⟨18, 16, 15⟩
  countRounds initialState = 45 := by
  sorry

end game_ends_in_45_rounds_l2802_280243


namespace zero_geometric_mean_with_one_l2802_280201

def geometric_mean (list : List ℝ) : ℝ := (list.prod) ^ (1 / list.length)

theorem zero_geometric_mean_with_one {n : ℕ} (h : n > 1) :
  let list : List ℝ := 1 :: List.replicate (n - 1) 0
  geometric_mean list = 0 := by
  sorry

end zero_geometric_mean_with_one_l2802_280201


namespace annika_hiking_time_l2802_280267

/-- Annika's hiking problem -/
theorem annika_hiking_time (rate : ℝ) (initial_distance : ℝ) (total_distance_east : ℝ) : 
  rate = 10 →
  initial_distance = 2.75 →
  total_distance_east = 3.625 →
  (2 * total_distance_east) * rate = 72.5 := by
  sorry

end annika_hiking_time_l2802_280267


namespace students_called_back_l2802_280200

theorem students_called_back (girls : ℕ) (boys : ℕ) (didnt_make_cut : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : didnt_make_cut = 17) :
  girls + boys - didnt_make_cut = 26 := by
  sorry

end students_called_back_l2802_280200


namespace changhyeok_snacks_l2802_280270

theorem changhyeok_snacks :
  ∀ (s d : ℕ),
  s + d = 12 →
  1000 * s + 1300 * d = 15000 →
  s = 2 := by
sorry

end changhyeok_snacks_l2802_280270


namespace uncovered_area_calculation_l2802_280255

theorem uncovered_area_calculation (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 10 →
  small_square_side = 4 →
  large_square_side^2 - 2 * small_square_side^2 = 68 :=
by sorry

end uncovered_area_calculation_l2802_280255


namespace power_fraction_simplification_l2802_280296

theorem power_fraction_simplification :
  (3^1024 + 5 * 3^1022) / (3^1024 - 3^1022) = 7/4 := by
  sorry

end power_fraction_simplification_l2802_280296


namespace smallest_number_property_l2802_280221

/-- The smallest positive integer that is not prime, not a square, and has no prime factor less than 60 -/
def smallest_number : ℕ := 290977

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a perfect square -/
def is_square (n : ℕ) : Prop := sorry

/-- A function that returns the smallest prime factor of a number -/
def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem smallest_number_property : 
  ¬ is_prime smallest_number ∧ 
  ¬ is_square smallest_number ∧ 
  smallest_prime_factor smallest_number > 59 ∧
  ∀ m : ℕ, m < smallest_number → 
    is_prime m ∨ is_square m ∨ smallest_prime_factor m ≤ 59 := by sorry

end smallest_number_property_l2802_280221


namespace insect_eggs_l2802_280290

def base_6_to_10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6 + c

theorem insect_eggs : base_6_to_10 2 5 3 = 105 := by sorry

end insect_eggs_l2802_280290


namespace fence_painting_problem_l2802_280241

theorem fence_painting_problem (initial_people : ℕ) (initial_time : ℝ) (new_time : ℝ) :
  initial_people = 8 →
  initial_time = 3 →
  new_time = 2 →
  ∃ (new_people : ℕ), 
    (initial_people : ℝ) * initial_time = (new_people : ℝ) * new_time ∧ 
    new_people = 12 :=
by sorry

end fence_painting_problem_l2802_280241


namespace smallest_positive_constant_inequality_l2802_280249

theorem smallest_positive_constant_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (∃ c : ℝ, c > 0 ∧ ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 →
    Real.sqrt (x * y * z) + c * Real.sqrt (|x - y|) ≥ (x + y + z) / 3) ∧
  (∀ c : ℝ, c > 0 ∧ (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 →
    Real.sqrt (x * y * z) + c * Real.sqrt (|x - y|) ≥ (x + y + z) / 3) → c ≥ 1) ∧
  (Real.sqrt (x * y * z) + Real.sqrt (|x - y|) ≥ (x + y + z) / 3) :=
by
  sorry

end smallest_positive_constant_inequality_l2802_280249
