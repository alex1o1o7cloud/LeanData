import Mathlib

namespace NUMINAMATH_CALUDE_john_payment_l2197_219731

def lawyer_fee (upfront_fee : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) (prep_time_multiplier : ℕ) : ℕ :=
  upfront_fee + hourly_rate * (court_hours + prep_time_multiplier * court_hours)

theorem john_payment (upfront_fee : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) (prep_time_multiplier : ℕ) :
  upfront_fee = 1000 →
  hourly_rate = 100 →
  court_hours = 50 →
  prep_time_multiplier = 2 →
  lawyer_fee upfront_fee hourly_rate court_hours prep_time_multiplier / 2 = 8000 :=
by
  sorry

#check john_payment

end NUMINAMATH_CALUDE_john_payment_l2197_219731


namespace NUMINAMATH_CALUDE_rope_length_l2197_219755

theorem rope_length : 
  ∀ (L : ℝ), 
    L > 0 →  -- The rope has a positive length
    (L - 3)^2 + 48^2 = L^2 →  -- Pythagorean theorem applied to the right triangle
    L = 388.5 :=
by
  sorry

end NUMINAMATH_CALUDE_rope_length_l2197_219755


namespace NUMINAMATH_CALUDE_find_number_l2197_219703

theorem find_number (x : ℝ) : ((x * 14) / 100) = 0.045374000000000005 → x = 0.3241 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2197_219703


namespace NUMINAMATH_CALUDE_straight_lines_parabolas_disjoint_l2197_219708

-- Define the set of all straight lines
def StraightLines : Set (ℝ → ℝ) := {f | ∃ a b : ℝ, ∀ x, f x = a * x + b}

-- Define the set of all parabolas
def Parabolas : Set (ℝ → ℝ) := {f | ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c}

-- Theorem statement
theorem straight_lines_parabolas_disjoint : StraightLines ∩ Parabolas = ∅ := by
  sorry

end NUMINAMATH_CALUDE_straight_lines_parabolas_disjoint_l2197_219708


namespace NUMINAMATH_CALUDE_gunther_working_time_l2197_219752

/-- Gunther's typing speed in words per minute -/
def typing_speed : ℚ := 160 / 3

/-- Total words Gunther types in a working day -/
def total_words : ℕ := 25600

/-- Gunther's working time in minutes -/
def working_time : ℕ := 480

theorem gunther_working_time :
  (total_words : ℚ) / typing_speed = working_time := by sorry

end NUMINAMATH_CALUDE_gunther_working_time_l2197_219752


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l2197_219712

theorem subtraction_of_fractions : 
  (5 : ℚ) / 6 - 1 / 6 - 1 / 4 = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l2197_219712


namespace NUMINAMATH_CALUDE_probability_of_two_queens_or_at_least_one_king_l2197_219790

-- Define a standard deck
def standard_deck : ℕ := 52

-- Define the number of queens in a deck
def queens_in_deck : ℕ := 4

-- Define the number of kings in a deck
def kings_in_deck : ℕ := 4

-- Define the probability of the event
def prob_two_queens_or_at_least_one_king : ℚ := 2 / 13

-- State the theorem
theorem probability_of_two_queens_or_at_least_one_king :
  let p := (queens_in_deck * (queens_in_deck - 1) / 2 +
            kings_in_deck * (standard_deck - kings_in_deck) +
            kings_in_deck * (kings_in_deck - 1) / 2) /
           (standard_deck * (standard_deck - 1) / 2)
  p = prob_two_queens_or_at_least_one_king :=
by sorry

end NUMINAMATH_CALUDE_probability_of_two_queens_or_at_least_one_king_l2197_219790


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2197_219772

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 3 > 0) ↔ (∃ x : ℝ, x^2 + x + 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2197_219772


namespace NUMINAMATH_CALUDE_election_majority_l2197_219784

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 900 →
  winning_percentage = 70 / 100 →
  (total_votes : ℚ) * winning_percentage - (total_votes : ℚ) * (1 - winning_percentage) = 360 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l2197_219784


namespace NUMINAMATH_CALUDE_parallelogram_area_in_regular_hexagon_l2197_219774

/-- The area of the parallelogram formed by connecting every second vertex of a regular hexagon --/
theorem parallelogram_area_in_regular_hexagon (side_length : ℝ) (h : side_length = 12) :
  let large_triangle_area := Real.sqrt 3 / 4 * (2 * side_length) ^ 2
  let small_triangle_area := Real.sqrt 3 / 4 * side_length ^ 2
  large_triangle_area - 3 * small_triangle_area = 36 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_in_regular_hexagon_l2197_219774


namespace NUMINAMATH_CALUDE_candies_remaining_l2197_219768

theorem candies_remaining (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  red = 40 →
  yellow = 3 * red - 20 →
  blue = yellow / 2 →
  red + blue = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_candies_remaining_l2197_219768


namespace NUMINAMATH_CALUDE_marathon_day3_miles_l2197_219782

/-- Represents the marathon runner's training schedule over 3 days -/
structure MarathonTraining where
  total_miles : ℝ
  day1_percent : ℝ
  day2_percent : ℝ

/-- Calculates the miles run on day 3 given the training schedule -/
def miles_on_day3 (mt : MarathonTraining) : ℝ :=
  mt.total_miles - (mt.total_miles * mt.day1_percent) - ((mt.total_miles - (mt.total_miles * mt.day1_percent)) * mt.day2_percent)

/-- Theorem stating that given the specific training schedule, the miles run on day 3 is 28 -/
theorem marathon_day3_miles :
  let mt : MarathonTraining := ⟨70, 0.2, 0.5⟩
  miles_on_day3 mt = 28 := by
  sorry

end NUMINAMATH_CALUDE_marathon_day3_miles_l2197_219782


namespace NUMINAMATH_CALUDE_aiguo_seashells_l2197_219742

/-- The number of seashells collected by Aiguo, Vail, and Stefan satisfies the given conditions -/
def seashell_collection (aiguo vail stefan : ℕ) : Prop :=
  stefan = vail + 16 ∧ 
  vail + 5 = aiguo ∧ 
  aiguo + vail + stefan = 66

/-- Aiguo had 20 seashells -/
theorem aiguo_seashells :
  ∃ (vail stefan : ℕ), seashell_collection 20 vail stefan := by
  sorry

end NUMINAMATH_CALUDE_aiguo_seashells_l2197_219742


namespace NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l2197_219748

theorem angle_with_supplement_four_times_complement : ∃ (x : ℝ), 
  x = 60 ∧ 
  (180 - x) = 4 * (90 - x) := by
  sorry

end NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l2197_219748


namespace NUMINAMATH_CALUDE_total_bottle_caps_l2197_219726

theorem total_bottle_caps (bottle_caps_per_child : ℕ) (number_of_children : ℕ) 
  (h1 : bottle_caps_per_child = 5) 
  (h2 : number_of_children = 9) : 
  bottle_caps_per_child * number_of_children = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_bottle_caps_l2197_219726


namespace NUMINAMATH_CALUDE_cube_triangle_areas_sum_l2197_219749

/-- Represents a 3D point in space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- The vertices of a 2x2x2 cube -/
def cubeVertices : List Point3D := [
  ⟨0, 0, 0⟩, ⟨0, 0, 2⟩, ⟨0, 2, 0⟩, ⟨0, 2, 2⟩,
  ⟨2, 0, 0⟩, ⟨2, 0, 2⟩, ⟨2, 2, 0⟩, ⟨2, 2, 2⟩
]

/-- All possible triangles formed by the vertices of the cube -/
def cubeTriangles : List Triangle3D := sorry

/-- Calculates the area of a triangle in 3D space -/
def triangleArea (t : Triangle3D) : ℝ := sorry

/-- The sum of areas of all triangles formed by the cube vertices -/
def totalArea : ℝ := (cubeTriangles.map triangleArea).sum

/-- The theorem to be proved -/
theorem cube_triangle_areas_sum :
  ∃ (m n p : ℕ), totalArea = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 7728 := by
  sorry

end NUMINAMATH_CALUDE_cube_triangle_areas_sum_l2197_219749


namespace NUMINAMATH_CALUDE_unique_prime_satisfying_condition_l2197_219741

theorem unique_prime_satisfying_condition : 
  ∃! p : ℕ, Prime p ∧ Prime (4 * p^2 + 1) ∧ Prime (6 * p^2 + 1) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_satisfying_condition_l2197_219741


namespace NUMINAMATH_CALUDE_solve_inequality_m_neg_four_solve_inequality_x_greater_than_one_l2197_219788

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| - |2*x + m|

-- Theorem for part (1)
theorem solve_inequality_m_neg_four :
  ∀ x : ℝ, f x (-4) < 0 ↔ x < 5/3 ∨ x > 3 := by sorry

-- Theorem for part (2)
theorem solve_inequality_x_greater_than_one :
  ∀ m : ℝ, (∀ x : ℝ, x > 1 → f x m < 0) ↔ m ≥ -2 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_m_neg_four_solve_inequality_x_greater_than_one_l2197_219788


namespace NUMINAMATH_CALUDE_base_b_is_ten_l2197_219754

/-- Given that 1304 in base b, when squared, equals 99225 in base b, prove that b = 10 -/
theorem base_b_is_ten (b : ℕ) (h : b > 1) : 
  (b^3 + 3*b^2 + 4)^2 = 9*b^4 + 9*b^3 + 2*b^2 + 2*b + 5 → b = 10 := by
  sorry

#check base_b_is_ten

end NUMINAMATH_CALUDE_base_b_is_ten_l2197_219754


namespace NUMINAMATH_CALUDE_arrangements_for_six_people_l2197_219777

/-- The number of people in the line -/
def n : ℕ := 6

/-- The number of arrangements of n people in a line where two specific people
    must stand next to each other and two other specific people must not stand
    next to each other -/
def arrangements (n : ℕ) : ℕ := 
  2 * (n - 2).factorial * ((n - 2) * (n - 3))

/-- Theorem stating that the number of arrangements for 6 people
    under the given conditions is 144 -/
theorem arrangements_for_six_people : arrangements n = 144 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_for_six_people_l2197_219777


namespace NUMINAMATH_CALUDE_total_card_value_is_244_l2197_219717

def jenny_initial_cards : ℕ := 6
def jenny_rare_percentage : ℚ := 1/2
def orlando_extra_cards : ℕ := 2
def orlando_rare_percentage : ℚ := 2/5
def richard_card_multiplier : ℕ := 3
def richard_rare_percentage : ℚ := 1/4
def jenny_additional_cards : ℕ := 4
def holographic_card_value : ℕ := 15
def first_edition_card_value : ℕ := 8
def rare_card_value : ℕ := 10
def non_rare_card_value : ℕ := 3

def total_card_value : ℕ := sorry

theorem total_card_value_is_244 : total_card_value = 244 := by sorry

end NUMINAMATH_CALUDE_total_card_value_is_244_l2197_219717


namespace NUMINAMATH_CALUDE_sin_30_sin_75_minus_sin_60_cos_105_l2197_219794

theorem sin_30_sin_75_minus_sin_60_cos_105 :
  Real.sin (30 * π / 180) * Real.sin (75 * π / 180) -
  Real.sin (60 * π / 180) * Real.cos (105 * π / 180) =
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_sin_75_minus_sin_60_cos_105_l2197_219794


namespace NUMINAMATH_CALUDE_polynomial_existence_l2197_219724

theorem polynomial_existence : ∃ (f : ℝ → ℝ), 
  (∃ (a b c d e g h : ℝ), ∀ x, f x = a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + g*x + h) ∧ 
  (∀ x, f (Real.sin x) + f (Real.cos x) = 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_existence_l2197_219724


namespace NUMINAMATH_CALUDE_milk_price_calculation_l2197_219744

/-- Calculates the price per gallon of milk given the daily production, 
    number of days, and total income. -/
def price_per_gallon (daily_production : ℕ) (days : ℕ) (total_income : ℚ) : ℚ :=
  total_income / (daily_production * days)

/-- Theorem stating that the price per gallon of milk is $3.05 given the conditions. -/
theorem milk_price_calculation : 
  price_per_gallon 200 30 18300 = 305/100 := by
  sorry

#eval price_per_gallon 200 30 18300

end NUMINAMATH_CALUDE_milk_price_calculation_l2197_219744


namespace NUMINAMATH_CALUDE_smallest_x_value_l2197_219720

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (240 + x)) : 
  ∀ z : ℕ+, z < x → ¬∃ w : ℕ+, (3 : ℚ) / 4 = w / (240 + z) :=
by sorry

#check smallest_x_value

end NUMINAMATH_CALUDE_smallest_x_value_l2197_219720


namespace NUMINAMATH_CALUDE_number_exceeding_half_by_80_l2197_219740

theorem number_exceeding_half_by_80 (x : ℝ) : x = 0.5 * x + 80 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_half_by_80_l2197_219740


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2197_219767

/-- Represents the quantities of sugar types A, B, and C -/
structure SugarQuantities where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given quantities satisfy the problem constraints -/
def satisfiesConstraints (q : SugarQuantities) : Prop :=
  q.a ≥ 0 ∧ q.b ≥ 0 ∧ q.c ≥ 0 ∧
  q.a + q.b + q.c = 1500 ∧
  (8 * q.a + 15 * q.b + 20 * q.c) / 1500 = 14

/-- There are infinitely many solutions to the sugar problem -/
theorem infinitely_many_solutions :
  ∀ ε > 0, ∃ q₁ q₂ : SugarQuantities,
    satisfiesConstraints q₁ ∧
    satisfiesConstraints q₂ ∧
    q₁ ≠ q₂ ∧
    ‖q₁.a - q₂.a‖ < ε ∧
    ‖q₁.b - q₂.b‖ < ε ∧
    ‖q₁.c - q₂.c‖ < ε :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2197_219767


namespace NUMINAMATH_CALUDE_solve_custom_equation_l2197_219761

-- Define the custom operation
def custom_op (m n : ℤ) : ℤ := n^2 - m

-- Theorem statement
theorem solve_custom_equation :
  ∀ x : ℤ, custom_op x 3 = 5 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_custom_equation_l2197_219761


namespace NUMINAMATH_CALUDE_representatives_count_l2197_219776

/-- The number of ways to select 3 representatives from 3 boys and 2 girls, 
    such that at least one girl is included. -/
def select_representatives (num_boys num_girls num_representatives : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of ways to select 3 representatives from 3 boys and 2 girls, 
    such that at least one girl is included, is equal to 9. -/
theorem representatives_count : select_representatives 3 2 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_representatives_count_l2197_219776


namespace NUMINAMATH_CALUDE_jerrys_collection_cost_l2197_219783

/-- The amount of money Jerry needs to finish his action figure collection -/
def jerrysMoney (currentFigures : ℕ) (totalRequired : ℕ) (costPerFigure : ℕ) : ℕ :=
  (totalRequired - currentFigures) * costPerFigure

/-- Proof that Jerry needs $72 to finish his collection -/
theorem jerrys_collection_cost : jerrysMoney 7 16 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_collection_cost_l2197_219783


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2197_219756

theorem triangle_perimeter (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sum : |a + b - c| + |b + c - a| + |c + a - b| = 12) : 
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2197_219756


namespace NUMINAMATH_CALUDE_new_job_bonus_calculation_l2197_219759

/-- Represents Maisy's job options and earnings -/
structure JobOption where
  hours_per_week : ℕ
  hourly_wage : ℕ
  bonus : ℕ

/-- Calculates the weekly earnings for a job option -/
def weekly_earnings (job : JobOption) : ℕ :=
  job.hours_per_week * job.hourly_wage + job.bonus

theorem new_job_bonus_calculation (current_job new_job : JobOption) 
  (h1 : current_job.hours_per_week = 8)
  (h2 : current_job.hourly_wage = 10)
  (h3 : current_job.bonus = 0)
  (h4 : new_job.hours_per_week = 4)
  (h5 : new_job.hourly_wage = 15)
  (h6 : weekly_earnings new_job = weekly_earnings current_job + 15) :
  new_job.bonus = 15 := by
  sorry

#check new_job_bonus_calculation

end NUMINAMATH_CALUDE_new_job_bonus_calculation_l2197_219759


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l2197_219710

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point over the x-axis -/
def reflect_over_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The sum of coordinate values of two points -/
def sum_of_coordinates (p1 p2 : Point) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let c : Point := ⟨x, 8⟩
  let d : Point := reflect_over_x_axis c
  sum_of_coordinates c d = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l2197_219710


namespace NUMINAMATH_CALUDE_factorize_difference_of_squares_factorize_quadratic_l2197_219725

-- Theorem 1
theorem factorize_difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

-- Theorem 2
theorem factorize_quadratic (x : ℝ) : 2*x^2 - 20*x + 50 = 2*(x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorize_difference_of_squares_factorize_quadratic_l2197_219725


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2197_219735

theorem simple_interest_problem (P R : ℝ) (h : P * (R + 10) * 8 / 100 - P * R * 8 / 100 = 150) : P = 187.50 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2197_219735


namespace NUMINAMATH_CALUDE_tank_filling_time_l2197_219722

theorem tank_filling_time (fill_rate_A fill_rate_B : ℚ) : 
  fill_rate_A = 1 / 60 →
  15 * fill_rate_B + 15 * (fill_rate_A + fill_rate_B) = 1 →
  fill_rate_B = 1 / 40 :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l2197_219722


namespace NUMINAMATH_CALUDE_fraction_problem_l2197_219769

theorem fraction_problem (x : ℝ) : 
  (0.60 * x * 100 = 36) → x = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2197_219769


namespace NUMINAMATH_CALUDE_cricketer_average_score_l2197_219795

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (average_all : ℚ) 
  (average_last : ℚ) 
  (last_matches : ℕ) : 
  total_matches = 10 → 
  average_all = 389/10 → 
  average_last = 137/4 → 
  last_matches = 4 → 
  (total_matches * average_all - last_matches * average_last) / (total_matches - last_matches) = 42 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l2197_219795


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l2197_219732

theorem divisible_by_twelve (a b c d : ℤ) : 
  12 ∣ ((b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l2197_219732


namespace NUMINAMATH_CALUDE_nancy_carrots_l2197_219789

/-- Calculates the total number of carrots Nancy has -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  initial - thrown_out + new_picked

/-- Proves that Nancy's total carrots is 31 given the problem conditions -/
theorem nancy_carrots : total_carrots 12 2 21 = 31 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_l2197_219789


namespace NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l2197_219762

/-- Represents the number of family members -/
def family_members : ℕ := 5

/-- Represents the number of front seats in the van -/
def front_seats : ℕ := 2

/-- Represents the number of back seats in the van -/
def back_seats : ℕ := 3

/-- Represents the number of adults who can drive -/
def potential_drivers : ℕ := 2

/-- Calculates the number of possible seating arrangements -/
def seating_arrangements : ℕ :=
  potential_drivers * (family_members - 1) * (back_seats.factorial)

theorem lopez_family_seating_arrangements :
  seating_arrangements = 48 :=
sorry

end NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l2197_219762


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2197_219739

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 12) 
  (h3 : S = a / (1 - r)) : 
  a = 16 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2197_219739


namespace NUMINAMATH_CALUDE_union_of_sets_l2197_219793

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_of_sets :
  ∃ x : ℝ, (A x ∩ B x = {9}) → (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2197_219793


namespace NUMINAMATH_CALUDE_sum_of_integers_l2197_219733

theorem sum_of_integers (x y : ℕ+) 
  (h_diff : x.val - y.val = 8)
  (h_prod : x.val * y.val = 120) : 
  x.val + y.val = 15 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2197_219733


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2197_219792

/-- The number of dots on each side of the square grid -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in a square grid -/
def numRectangles (n : ℕ) : ℕ := (n + 1).choose 2 * (n + 1).choose 2

/-- Theorem stating the number of rectangles in a 5x5 grid -/
theorem rectangles_in_5x5_grid : numRectangles gridSize = 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2197_219792


namespace NUMINAMATH_CALUDE_cubic_identity_l2197_219775

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2197_219775


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2197_219715

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 - 3, t^4 - t^2 - 9*t + 6)

-- Define the self-intersection point
def intersection_point : ℝ × ℝ := (6, 6)

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = intersection_point :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2197_219715


namespace NUMINAMATH_CALUDE_box_volume_increase_l2197_219780

-- Define the properties of the rectangular box
def rectangular_box (l w h : ℝ) : Prop :=
  l * w * h = 5400 ∧
  2 * (l * w + w * h + h * l) = 2352 ∧
  4 * (l + w + h) = 240

-- State the theorem
theorem box_volume_increase (l w h : ℝ) :
  rectangular_box l w h →
  (l + 2) * (w + 2) * (h + 2) = 8054 :=
by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2197_219780


namespace NUMINAMATH_CALUDE_training_schedule_days_l2197_219709

/-- Calculates the number of days required to complete a training schedule. -/
def trainingDays (totalHours : ℕ) (multiplicationMinutes : ℕ) (divisionMinutes : ℕ) : ℕ :=
  let totalMinutes := totalHours * 60
  let dailyMinutes := multiplicationMinutes + divisionMinutes
  totalMinutes / dailyMinutes

/-- Proves that the training schedule takes 10 days to complete. -/
theorem training_schedule_days :
  trainingDays 5 10 20 = 10 := by
  sorry

#eval trainingDays 5 10 20

end NUMINAMATH_CALUDE_training_schedule_days_l2197_219709


namespace NUMINAMATH_CALUDE_uneaten_fish_l2197_219745

def fish_cells : List Nat := [3, 4, 16, 12, 20, 6]

def cat_eating_rate : Nat := 3

theorem uneaten_fish (eaten_count : Nat) (total_time : Nat) :
  eaten_count = 5 →
  total_time * cat_eating_rate = (fish_cells.take eaten_count).sum →
  total_time > 0 →
  (fish_cells.take eaten_count).sum % cat_eating_rate = 1 →
  fish_cells[eaten_count]! = 6 := by
  sorry

end NUMINAMATH_CALUDE_uneaten_fish_l2197_219745


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l2197_219758

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- A point (x, y) is a reflection of (x₀, y₀) across the x-axis if x = x₀ and y = -y₀ -/
def is_reflection_x_axis (x y x₀ y₀ : ℝ) : Prop :=
  x = x₀ ∧ y = -y₀

theorem reflected_ray_equation :
  is_reflection_x_axis 2 (-1) 2 1 →
  line_equation 2 (-1) 4 5 x y ↔ 3 * x - y - 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l2197_219758


namespace NUMINAMATH_CALUDE_root_in_interval_l2197_219729

def f (x : ℝ) := x^3 - x - 1

theorem root_in_interval :
  ∃ r ∈ Set.Icc 1.25 1.5, f r = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2197_219729


namespace NUMINAMATH_CALUDE_prob_two_red_crayons_l2197_219773

/-- The probability of selecting 2 red crayons from a jar containing 6 crayons (3 red, 2 blue, 1 green) -/
theorem prob_two_red_crayons (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) :
  total = 6 →
  red = 3 →
  blue = 2 →
  green = 1 →
  (Nat.choose red 2 : ℚ) / (Nat.choose total 2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_crayons_l2197_219773


namespace NUMINAMATH_CALUDE_string_length_around_cylinder_specific_string_length_l2197_219707

/-- 
Given a cylindrical post with circumference C, height H, and a string making n complete loops 
around it from bottom to top, the length of the string L is given by L = n * √(C² + (H/n)²)
-/
theorem string_length_around_cylinder (C H : ℝ) (n : ℕ) (h1 : C > 0) (h2 : H > 0) (h3 : n > 0) :
  let L := n * Real.sqrt (C^2 + (H/n)^2)
  L = n * Real.sqrt (C^2 + (H/n)^2) := by sorry

/-- 
For the specific case where C = 6, H = 18, and n = 3, prove that the string length is 18√2
-/
theorem specific_string_length :
  let C : ℝ := 6
  let H : ℝ := 18
  let n : ℕ := 3
  let L := n * Real.sqrt (C^2 + (H/n)^2)
  L = 18 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_string_length_around_cylinder_specific_string_length_l2197_219707


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2197_219757

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4 * Complex.I) * z = Complex.abs (4 + 3 * Complex.I)) :
  z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2197_219757


namespace NUMINAMATH_CALUDE_equation_roots_opposite_signs_l2197_219705

theorem equation_roots_opposite_signs (a b c d m : ℝ) (hd : d ≠ 0) :
  (∀ x, (x^2 - (b+1)*x) / ((a-1)*x - (c+d)) = (m-2) / (m+2)) →
  (∃ r : ℝ, r ≠ 0 ∧ (r^2 - (b+1)*r = 0) ∧ (-r^2 - (b+1)*(-r) = 0)) →
  m = 2*(a-b-2) / (a+b) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_opposite_signs_l2197_219705


namespace NUMINAMATH_CALUDE_bus_stop_problem_l2197_219730

theorem bus_stop_problem (boys girls : ℕ) : 
  (boys = 2 * (girls - 15)) →
  (girls - 15 = 5 * (boys - 45)) →
  (boys = 50 ∧ girls = 40) := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l2197_219730


namespace NUMINAMATH_CALUDE_g_zero_iff_a_eq_four_thirds_l2197_219764

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x - 4

-- State the theorem
theorem g_zero_iff_a_eq_four_thirds :
  ∀ a : ℝ, g a = 0 ↔ a = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_g_zero_iff_a_eq_four_thirds_l2197_219764


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2197_219770

theorem arithmetic_expression_equality : 7 / 2 - 3 - 5 + 3 * 4 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2197_219770


namespace NUMINAMATH_CALUDE_money_division_l2197_219799

theorem money_division (a b c total : ℝ) : 
  (∀ x, b = 0.65 * x → c = 0.40 * x → a = x) →  -- For each rupee A has, B has 0.65 and C has 0.40
  c = 32 →  -- C's share is 32 rupees
  total = a + b + c →  -- Total is the sum of all shares
  total = 164 := by sorry

end NUMINAMATH_CALUDE_money_division_l2197_219799


namespace NUMINAMATH_CALUDE_percent_equality_l2197_219779

theorem percent_equality (x y : ℝ) (P : ℝ) (h1 : y = 0.25 * x) 
  (h2 : (P / 100) * (x - y) = 0.15 * (x + y)) : P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l2197_219779


namespace NUMINAMATH_CALUDE_expression_simplification_l2197_219771

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 - a / (a + 1)) / ((a^2 - 1) / (a^2 + 2*a + 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2197_219771


namespace NUMINAMATH_CALUDE_mode_of_scores_l2197_219787

def Scores := List Nat

def count (n : Nat) (scores : Scores) : Nat :=
  scores.filter (· = n) |>.length

def isMode (n : Nat) (scores : Scores) : Prop :=
  ∀ m, count n scores ≥ count m scores

theorem mode_of_scores (scores : Scores) 
  (h1 : scores.all (· ≤ 120))
  (h2 : count 91 scores = 5)
  (h3 : ∀ n, n ≠ 91 → count n scores ≤ 5) :
  isMode 91 scores :=
sorry

end NUMINAMATH_CALUDE_mode_of_scores_l2197_219787


namespace NUMINAMATH_CALUDE_systematic_sampling_survey_c_count_l2197_219711

theorem systematic_sampling_survey_c_count 
  (total_population : Nat) 
  (sample_size : Nat) 
  (first_number : Nat) 
  (survey_c_lower_bound : Nat) 
  (survey_c_upper_bound : Nat) 
  (h1 : total_population = 1000)
  (h2 : sample_size = 50)
  (h3 : first_number = 8)
  (h4 : survey_c_lower_bound = 751)
  (h5 : survey_c_upper_bound = 1000) :
  (Finset.filter (fun n => 
    let term := first_number + (n - 1) * (total_population / sample_size)
    term ≥ survey_c_lower_bound ∧ term ≤ survey_c_upper_bound
  ) (Finset.range sample_size)).card = 12 := by
  sorry

#check systematic_sampling_survey_c_count

end NUMINAMATH_CALUDE_systematic_sampling_survey_c_count_l2197_219711


namespace NUMINAMATH_CALUDE_chord_length_from_arc_and_angle_l2197_219714

theorem chord_length_from_arc_and_angle (m : ℝ) (h : m > 0) :
  let arc_length := m
  let central_angle : ℝ := 120 * π / 180
  let radius := arc_length / central_angle
  let chord_length := 2 * radius * Real.sin (central_angle / 2)
  chord_length = (3 * Real.sqrt 3 / (4 * π)) * m :=
by sorry

end NUMINAMATH_CALUDE_chord_length_from_arc_and_angle_l2197_219714


namespace NUMINAMATH_CALUDE_stamp_cost_l2197_219760

theorem stamp_cost (total_cost : ℕ) (num_stamps : ℕ) (h1 : total_cost = 136) (h2 : num_stamps = 4) :
  total_cost / num_stamps = 34 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_l2197_219760


namespace NUMINAMATH_CALUDE_job_selection_probability_l2197_219778

theorem job_selection_probability (carol_prob bernie_prob : ℚ) 
  (h_carol : carol_prob = 4/5)
  (h_bernie : bernie_prob = 3/5) : 
  carol_prob * bernie_prob = 12/25 := by
sorry

end NUMINAMATH_CALUDE_job_selection_probability_l2197_219778


namespace NUMINAMATH_CALUDE_element_in_set_M_l2197_219746

def U : Finset Nat := {1, 2, 3, 4, 5}

theorem element_in_set_M (M : Finset Nat) 
  (h : (U \ M) = {1, 2}) : 3 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_M_l2197_219746


namespace NUMINAMATH_CALUDE_x_plus_2y_squared_equals_half_l2197_219702

theorem x_plus_2y_squared_equals_half (x y : ℝ) 
  (h : 8*y^4 + 4*x^2*y^2 + 4*x*y^2 + 2*x^3 + 2*y^2 + 2*x = x^2 + 1) : 
  x + 2*y^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_squared_equals_half_l2197_219702


namespace NUMINAMATH_CALUDE_only_white_balls_drawn_is_random_variable_l2197_219738

/-- A bag containing white and red balls -/
structure Bag where
  white_balls : ℕ
  red_balls : ℕ

/-- The options for potential random variables -/
inductive DrawOption
  | BallsDrawn
  | WhiteBallsDrawn
  | TotalBallsDrawn
  | TotalBallsInBag

/-- Definition of a random variable in this context -/
def is_random_variable (option : DrawOption) (bag : Bag) (num_drawn : ℕ) : Prop :=
  match option with
  | DrawOption.BallsDrawn => num_drawn ≠ num_drawn
  | DrawOption.WhiteBallsDrawn => true
  | DrawOption.TotalBallsDrawn => num_drawn ≠ num_drawn
  | DrawOption.TotalBallsInBag => bag.white_balls + bag.red_balls ≠ bag.white_balls + bag.red_balls

/-- The main theorem stating that only the number of white balls drawn is a random variable -/
theorem only_white_balls_drawn_is_random_variable (bag : Bag) (num_drawn : ℕ) :
  bag.white_balls = 5 → bag.red_balls = 3 → num_drawn = 3 →
  ∀ (option : DrawOption), is_random_variable option bag num_drawn ↔ option = DrawOption.WhiteBallsDrawn :=
by sorry

end NUMINAMATH_CALUDE_only_white_balls_drawn_is_random_variable_l2197_219738


namespace NUMINAMATH_CALUDE_no_scalene_equilateral_triangle_no_right_equilateral_triangle_l2197_219791

-- Define a triangle
structure Triangle where
  sides : Fin 3 → ℝ
  angles : Fin 3 → ℝ

-- Define properties of triangles
def Triangle.isScalene (t : Triangle) : Prop :=
  ∀ i j : Fin 3, i ≠ j → t.sides i ≠ t.sides j

def Triangle.isEquilateral (t : Triangle) : Prop :=
  ∀ i j : Fin 3, t.sides i = t.sides j

def Triangle.isRight (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i = 90

-- Theorem: A scalene equilateral triangle is impossible
theorem no_scalene_equilateral_triangle :
  ¬∃ t : Triangle, t.isScalene ∧ t.isEquilateral :=
sorry

-- Theorem: A right equilateral triangle is impossible
theorem no_right_equilateral_triangle :
  ¬∃ t : Triangle, t.isRight ∧ t.isEquilateral :=
sorry

end NUMINAMATH_CALUDE_no_scalene_equilateral_triangle_no_right_equilateral_triangle_l2197_219791


namespace NUMINAMATH_CALUDE_boy_scouts_permission_slips_l2197_219797

theorem boy_scouts_permission_slips 
  (total_permission : Real) 
  (boy_scouts_percentage : Real) 
  (girl_scouts_permission : Real) :
  total_permission = 0.7 →
  boy_scouts_percentage = 0.6 →
  girl_scouts_permission = 0.625 →
  (total_permission - ((1 - boy_scouts_percentage) * girl_scouts_permission)) / boy_scouts_percentage = 0.75 := by
sorry

end NUMINAMATH_CALUDE_boy_scouts_permission_slips_l2197_219797


namespace NUMINAMATH_CALUDE_johns_pictures_l2197_219786

/-- The number of pictures John drew and colored -/
def num_pictures : ℕ := 10

/-- The time it takes John to draw one picture (in hours) -/
def drawing_time : ℝ := 2

/-- The time it takes John to color one picture (in hours) -/
def coloring_time : ℝ := drawing_time * 0.7

/-- The total time John spent on all pictures (in hours) -/
def total_time : ℝ := 34

theorem johns_pictures :
  (drawing_time + coloring_time) * num_pictures = total_time := by sorry

end NUMINAMATH_CALUDE_johns_pictures_l2197_219786


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2197_219785

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ i : ℕ, i > 0 ∧ i ≤ 10 → n % i = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ 10 → m % i = 0) → m ≥ n) ∧
  n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2197_219785


namespace NUMINAMATH_CALUDE_root_product_expression_l2197_219765

theorem root_product_expression (p q : ℝ) 
  (α β γ δ : ℂ) 
  (hαβ : α^2 + p*α = 1 ∧ β^2 + p*β = 1) 
  (hγδ : γ^2 + q*γ = -1 ∧ δ^2 + q*δ = -1) : 
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = p^2 - q^2 := by
  sorry

end NUMINAMATH_CALUDE_root_product_expression_l2197_219765


namespace NUMINAMATH_CALUDE_consecutive_divisible_numbers_exist_l2197_219734

theorem consecutive_divisible_numbers_exist : ∃ (n : ℕ),
  (∀ (i : Fin 11), ∃ (k : ℕ), n + i.val = k * (2 * i.val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_divisible_numbers_exist_l2197_219734


namespace NUMINAMATH_CALUDE_least_bamboo_sticks_l2197_219701

/-- Represents the number of bamboo sticks each panda takes initially -/
structure BambooDistribution where
  s1 : ℕ
  s2 : ℕ
  s3 : ℕ
  s4 : ℕ

/-- Represents the final number of bamboo sticks each panda has -/
structure FinalDistribution where
  p1 : ℕ
  p2 : ℕ
  p3 : ℕ
  p4 : ℕ

/-- Calculates the final distribution based on the initial distribution -/
def calculateFinalDistribution (initial : BambooDistribution) : FinalDistribution :=
  { p1 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + (8 * initial.s4) / 9
  , p2 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + initial.s4 / 9
  , p3 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + initial.s4 / 9
  , p4 := (2 * initial.s1) / 3 + initial.s2 / 2 + initial.s3 / 6 + initial.s4 / 9
  }

/-- Checks if the final distribution satisfies the 4:3:2:1 ratio -/
def isValidRatio (final : FinalDistribution) : Prop :=
  4 * final.p4 = final.p1 ∧
  3 * final.p4 = final.p2 ∧
  2 * final.p4 = final.p3

/-- The main theorem stating the least possible total number of bamboo sticks -/
theorem least_bamboo_sticks :
  ∃ (initial : BambooDistribution),
    let final := calculateFinalDistribution initial
    isValidRatio final ∧
    initial.s1 + initial.s2 + initial.s3 + initial.s4 = 93 ∧
    ∀ (other : BambooDistribution),
      let otherFinal := calculateFinalDistribution other
      isValidRatio otherFinal →
      other.s1 + other.s2 + other.s3 + other.s4 ≥ 93 :=
by sorry


end NUMINAMATH_CALUDE_least_bamboo_sticks_l2197_219701


namespace NUMINAMATH_CALUDE_sum_first_25_odd_numbers_l2197_219798

/-- The sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ :=
  n * n

/-- The 25th odd number -/
def last_odd_number (n : ℕ) : ℕ :=
  2 * n - 1

theorem sum_first_25_odd_numbers :
  sum_odd_numbers 25 = 625 :=
sorry

end NUMINAMATH_CALUDE_sum_first_25_odd_numbers_l2197_219798


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l2197_219704

def n : ℕ := 2^100 * 3^4 * 5 * 7

theorem smallest_number_with_2020_divisors :
  (∀ m : ℕ, m < n → (Nat.divisors m).card ≠ 2020) ∧
  (Nat.divisors n).card = 2020 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l2197_219704


namespace NUMINAMATH_CALUDE_fraction_equality_l2197_219727

theorem fraction_equality : (1722^2 - 1715^2) / (1729^2 - 1708^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2197_219727


namespace NUMINAMATH_CALUDE_dan_helmet_craters_l2197_219700

/-- The number of craters in helmets owned by Dan, Daniel, and Rin. -/
structure HelmetsWithCraters where
  dan : ℕ
  daniel : ℕ
  rin : ℕ

/-- The conditions of the helmet crater problem. -/
def helmet_crater_conditions (h : HelmetsWithCraters) : Prop :=
  h.dan = h.daniel + 10 ∧
  h.rin = h.dan + h.daniel + 15 ∧
  h.rin = 75

/-- The theorem stating that Dan's helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (h : HelmetsWithCraters) 
  (hc : helmet_crater_conditions h) : h.dan = 35 := by
  sorry

end NUMINAMATH_CALUDE_dan_helmet_craters_l2197_219700


namespace NUMINAMATH_CALUDE_A_power_2023_l2197_219716

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, -1, 0;
     1,  0, 0;
     0,  0, 1]

theorem A_power_2023 :
  A ^ 2023 = !![0,  1, 0;
                -1,  0, 0;
                 0,  0, 1] := by sorry

end NUMINAMATH_CALUDE_A_power_2023_l2197_219716


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l2197_219753

/-- A convex polygon inscribed in a circle -/
structure InscribedPolygon where
  sides : ℕ
  sides_ge_3 : sides ≥ 3

/-- The maximum number of intersections between two inscribed polygons -/
def max_intersections (P₁ P₂ : InscribedPolygon) : ℕ := P₁.sides * P₂.sides

/-- Theorem: Maximum intersections between two inscribed polygons -/
theorem max_intersections_theorem (P₁ P₂ : InscribedPolygon) 
  (h : P₁.sides ≤ P₂.sides) : 
  max_intersections P₁ P₂ = P₁.sides * P₂.sides := by sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l2197_219753


namespace NUMINAMATH_CALUDE_minimum_pipes_needed_l2197_219723

/-- Represents the cutting methods for a 6m steel pipe -/
inductive CuttingMethod
  | method2 -- 4 pieces of 0.8m and 1 piece of 2.5m
  | method3 -- 1 piece of 0.8m and 2 pieces of 2.5m

/-- Represents the number of pieces obtained from each cutting method -/
def piecesObtained (m : CuttingMethod) : (ℕ × ℕ) :=
  match m with
  | CuttingMethod.method2 => (4, 1)
  | CuttingMethod.method3 => (1, 2)

theorem minimum_pipes_needed :
  ∃ (x y : ℕ),
    x * (piecesObtained CuttingMethod.method2).1 + y * (piecesObtained CuttingMethod.method3).1 = 100 ∧
    x * (piecesObtained CuttingMethod.method2).2 + y * (piecesObtained CuttingMethod.method3).2 = 32 ∧
    x + y = 28 ∧
    ∀ (a b : ℕ),
      a * (piecesObtained CuttingMethod.method2).1 + b * (piecesObtained CuttingMethod.method3).1 = 100 →
      a * (piecesObtained CuttingMethod.method2).2 + b * (piecesObtained CuttingMethod.method3).2 = 32 →
      a + b ≥ 28 := by
  sorry

end NUMINAMATH_CALUDE_minimum_pipes_needed_l2197_219723


namespace NUMINAMATH_CALUDE_mass_percentage_h_in_water_l2197_219747

/-- The mass percentage of hydrogen in water, considering isotopic composition --/
theorem mass_percentage_h_in_water (h1_abundance : Real) (h2_abundance : Real)
  (h1_mass : Real) (h2_mass : Real) (o_mass : Real)
  (h1_abundance_val : h1_abundance = 0.9998)
  (h2_abundance_val : h2_abundance = 0.0002)
  (h1_mass_val : h1_mass = 1)
  (h2_mass_val : h2_mass = 2)
  (o_mass_val : o_mass = 16) :
  let avg_h_mass := h1_abundance * h1_mass + h2_abundance * h2_mass
  let water_mass := 2 * avg_h_mass + o_mass
  let mass_percentage := (2 * avg_h_mass) / water_mass * 100
  ∃ ε > 0, |mass_percentage - 11.113| < ε :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_h_in_water_l2197_219747


namespace NUMINAMATH_CALUDE_square_sum_equality_l2197_219766

theorem square_sum_equality (x y P Q : ℝ) :
  x^2 + y^2 = (x + y)^2 + P ∧ x^2 + y^2 = (x - y)^2 + Q →
  P = -2*x*y ∧ Q = 2*x*y := by
sorry

end NUMINAMATH_CALUDE_square_sum_equality_l2197_219766


namespace NUMINAMATH_CALUDE_bad_carrots_count_l2197_219737

/-- The number of bad carrots in Faye's garden -/
def bad_carrots (faye_carrots mother_carrots good_carrots : ℕ) : ℕ :=
  faye_carrots + mother_carrots - good_carrots

/-- Theorem: The number of bad carrots is 16 -/
theorem bad_carrots_count : bad_carrots 23 5 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l2197_219737


namespace NUMINAMATH_CALUDE_sean_blocks_l2197_219796

theorem sean_blocks (initial_blocks : ℕ) (eaten_blocks : ℕ) (remaining_blocks : ℕ) : 
  initial_blocks = 55 → eaten_blocks = 29 → remaining_blocks = initial_blocks - eaten_blocks → 
  remaining_blocks = 26 := by
sorry

end NUMINAMATH_CALUDE_sean_blocks_l2197_219796


namespace NUMINAMATH_CALUDE_problem_1_l2197_219718

theorem problem_1 : 6 * (1/3 - 1/2) - 3^2 / (-12) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2197_219718


namespace NUMINAMATH_CALUDE_encryption_assignment_exists_l2197_219750

/-- Represents a user on the platform -/
structure User :=
  (id : Nat)

/-- Represents an encryption key -/
structure EncryptionKey :=
  (id : Nat)

/-- Represents a messaging channel between two users -/
structure Channel :=
  (user1 : User)
  (user2 : User)
  (key : EncryptionKey)

/-- The total number of users on the platform -/
def totalUsers : Nat := 105

/-- The total number of available encryption keys -/
def totalKeys : Nat := 100

/-- A function that assigns an encryption key to a channel between two users -/
def assignKey : User → User → EncryptionKey := sorry

/-- Theorem stating that there exists a key assignment satisfying the required property -/
theorem encryption_assignment_exists :
  ∃ (assignKey : User → User → EncryptionKey),
    ∀ (u1 u2 u3 u4 : User),
      u1 ≠ u2 ∧ u1 ≠ u3 ∧ u1 ≠ u4 ∧ u2 ≠ u3 ∧ u2 ≠ u4 ∧ u3 ≠ u4 →
        ¬(assignKey u1 u2 = assignKey u1 u3 ∧
          assignKey u1 u2 = assignKey u1 u4 ∧
          assignKey u1 u2 = assignKey u2 u3 ∧
          assignKey u1 u2 = assignKey u2 u4 ∧
          assignKey u1 u2 = assignKey u3 u4) :=
by sorry

end NUMINAMATH_CALUDE_encryption_assignment_exists_l2197_219750


namespace NUMINAMATH_CALUDE_log_equation_solution_l2197_219728

theorem log_equation_solution (x : ℝ) (h : x > 0) : 
  2 * (Real.log x / Real.log 6) = 1 - (Real.log 3 / Real.log 6) ↔ x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2197_219728


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2197_219736

/-- 
Given a quadratic function f(x) = ax^2 + bx + 5 with a ≠ 0,
if there exist two distinct points (x₁, 2023) and (x₂, 2023) on the graph of f,
then f(x₁ + x₂) = 5
-/
theorem quadratic_function_property (a b x₁ x₂ : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + 5
  (f x₁ = 2023) → (f x₂ = 2023) → (x₁ ≠ x₂) → f (x₁ + x₂) = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2197_219736


namespace NUMINAMATH_CALUDE_fresh_to_dried_grapes_l2197_219713

/-- Given fresh grapes with 60% water content and dried grapes with 20% water content,
    prove that 15 kg of dried grapes comes from 30 kg of fresh grapes. -/
theorem fresh_to_dried_grapes (fresh_water_content : ℝ) (dried_water_content : ℝ) 
  (dried_weight : ℝ) (fresh_weight : ℝ) : 
  fresh_water_content = 0.6 →
  dried_water_content = 0.2 →
  dried_weight = 15 →
  (1 - fresh_water_content) * fresh_weight = (1 - dried_water_content) * dried_weight →
  fresh_weight = 30 := by
sorry

end NUMINAMATH_CALUDE_fresh_to_dried_grapes_l2197_219713


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2197_219781

theorem weight_of_new_person
  (n : ℕ)
  (initial_weight : ℝ)
  (replaced_weight : ℝ)
  (average_increase : ℝ)
  (h1 : n = 10)
  (h2 : replaced_weight = 70)
  (h3 : average_increase = 4) :
  initial_weight / n + average_increase = (initial_weight - replaced_weight + replaced_weight + n * average_increase) / n :=
by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2197_219781


namespace NUMINAMATH_CALUDE_initial_pencils_count_l2197_219743

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Tim added to the drawer -/
def pencils_added : ℕ := 3

/-- The total number of pencils after Tim's addition -/
def total_pencils : ℕ := 5

theorem initial_pencils_count : initial_pencils = 2 :=
  by sorry

end NUMINAMATH_CALUDE_initial_pencils_count_l2197_219743


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2197_219751

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s ^ 2 = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2197_219751


namespace NUMINAMATH_CALUDE_first_year_interest_rate_is_four_percent_l2197_219763

/-- Calculates the final amount after two years of compound interest -/
def finalAmount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial * (1 + rate1) * (1 + rate2)

/-- Theorem stating that given the conditions, the first year interest rate must be 4% -/
theorem first_year_interest_rate_is_four_percent 
  (initial : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (final : ℝ) 
  (h1 : initial = 7000)
  (h2 : rate2 = 0.05)
  (h3 : final = 7644)
  (h4 : finalAmount initial rate1 rate2 = final) : 
  rate1 = 0.04 := by
  sorry

#check first_year_interest_rate_is_four_percent

end NUMINAMATH_CALUDE_first_year_interest_rate_is_four_percent_l2197_219763


namespace NUMINAMATH_CALUDE_hydrochloric_acid_mixture_l2197_219721

def total_mass : ℝ := 600
def final_concentration : ℝ := 0.15
def concentration_1 : ℝ := 0.3
def concentration_2 : ℝ := 0.1
def mass_1 : ℝ := 150
def mass_2 : ℝ := 450

theorem hydrochloric_acid_mixture :
  mass_1 + mass_2 = total_mass ∧
  (concentration_1 * mass_1 + concentration_2 * mass_2) / total_mass = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_hydrochloric_acid_mixture_l2197_219721


namespace NUMINAMATH_CALUDE_multiple_of_two_three_five_l2197_219706

theorem multiple_of_two_three_five : ∃ n : ℕ, 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n :=
  by
  use 30
  sorry

end NUMINAMATH_CALUDE_multiple_of_two_three_five_l2197_219706


namespace NUMINAMATH_CALUDE_defective_clock_correct_time_fraction_l2197_219719

/-- Represents a 12-hour digital clock with a defect that displays 1 instead of 2 --/
structure DefectiveClock :=
  (hours : Fin 12)
  (minutes : Fin 60)

/-- Checks if the given hour is displayed correctly --/
def hour_correct (h : Fin 12) : Bool :=
  h ≠ 2 ∧ h ≠ 12

/-- Checks if the given minute is displayed correctly --/
def minute_correct (m : Fin 60) : Bool :=
  m % 10 ≠ 2 ∧ m / 10 ≠ 2

/-- The fraction of the day during which the clock displays the correct time --/
def correct_time_fraction (clock : DefectiveClock) : ℚ :=
  (5 : ℚ) / 8

theorem defective_clock_correct_time_fraction :
  ∀ (clock : DefectiveClock),
  correct_time_fraction clock = (5 : ℚ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_defective_clock_correct_time_fraction_l2197_219719
