import Mathlib

namespace greater_number_in_ratio_l1099_109935

theorem greater_number_in_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a / b = 3 / 4 → a + b = 21 → max a b = 12 := by
  sorry

end greater_number_in_ratio_l1099_109935


namespace distribute_four_among_five_l1099_109939

/-- The number of ways to distribute n identical objects among k people,
    where each person receives at most one object and all objects must be distributed. -/
def distribute (n k : ℕ) : ℕ :=
  if n = k - 1 then k else 0

theorem distribute_four_among_five :
  distribute 4 5 = 5 := by
  sorry

end distribute_four_among_five_l1099_109939


namespace cos_alpha_value_l1099_109985

theorem cos_alpha_value (α : Real) :
  (∃ P : Real × Real, P.1 = -3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  Real.cos α = -3/5 := by
sorry

end cos_alpha_value_l1099_109985


namespace walking_speed_calculation_l1099_109919

/-- Proves that the walking speed is 4 km/hr given the problem conditions -/
theorem walking_speed_calculation (total_distance : ℝ) (total_time : ℝ) (running_speed : ℝ) :
  total_distance = 8 →
  total_time = 1.5 →
  running_speed = 8 →
  ∃ (walking_speed : ℝ),
    walking_speed > 0 ∧
    (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time ∧
    walking_speed = 4 := by
  sorry

end walking_speed_calculation_l1099_109919


namespace exists_M_with_properties_l1099_109991

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The theorem stating the existence of M with the required properties -/
theorem exists_M_with_properties : 
  ∃ M : ℕ, M^2 = 36^50 * 50^36 ∧ sum_of_digits M = 36 := by sorry

end exists_M_with_properties_l1099_109991


namespace odd_sum_prob_is_five_thirteenths_l1099_109931

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of even sides on each die -/
def num_even_sides : ℕ := 3

/-- The number of odd sides on each die -/
def num_odd_sides : ℕ := 3

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_sides ^ num_dice

/-- The number of outcomes where all dice show odd numbers -/
def all_odd_outcomes : ℕ := num_odd_sides ^ num_dice

/-- The number of outcomes where the product of dice values is even -/
def even_product_outcomes : ℕ := total_outcomes - all_odd_outcomes

/-- The probability of rolling an odd sum given an even product -/
def prob_odd_sum_given_even_product : ℚ := 5 / 13

theorem odd_sum_prob_is_five_thirteenths :
  prob_odd_sum_given_even_product = 5 / 13 := by sorry

end odd_sum_prob_is_five_thirteenths_l1099_109931


namespace triangle_problem_l1099_109926

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  cos (A - π/3) = 2 * cos A →
  b = 2 →
  (1/2) * b * c * sin A = 3 * sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c * cos A →
  cos (2*C) = 1 - a^2 / (6 * b^2) →
  (a = 2 * sqrt 7 ∧ (B = π/12 ∨ B = 7*π/12)) := by sorry

end triangle_problem_l1099_109926


namespace painting_cost_in_cny_l1099_109949

-- Define exchange rates
def usd_to_nad : ℝ := 7
def usd_to_cny : ℝ := 6

-- Define the cost of the painting in Namibian dollars
def painting_cost_nad : ℝ := 105

-- Theorem to prove
theorem painting_cost_in_cny :
  (painting_cost_nad / usd_to_nad) * usd_to_cny = 90 := by
  sorry

end painting_cost_in_cny_l1099_109949


namespace min_exercise_hours_l1099_109977

/-- Represents the exercise data for a month -/
structure ExerciseData where
  days_20min : Nat
  days_40min : Nat
  days_2hours : Nat
  min_exercise_time : Nat
  max_exercise_time : Nat

/-- Calculates the minimum number of hours exercised in a month -/
def min_hours_exercised (data : ExerciseData) : Rat :=
  let hours_2hours := data.days_2hours * 2
  let hours_40min := (data.days_40min - data.days_2hours) * 2 / 3
  let hours_20min := (data.days_20min - data.days_40min) * 1 / 3
  hours_2hours + hours_40min + hours_20min

/-- Theorem stating the minimum number of hours exercised -/
theorem min_exercise_hours (data : ExerciseData) 
  (h1 : data.days_20min = 26)
  (h2 : data.days_40min = 24)
  (h3 : data.days_2hours = 4)
  (h4 : data.min_exercise_time = 20)
  (h5 : data.max_exercise_time = 120) :
  min_hours_exercised data = 22 := by
  sorry

end min_exercise_hours_l1099_109977


namespace fraction_ordering_l1099_109937

theorem fraction_ordering : (25 : ℚ) / 19 < 21 / 16 ∧ 21 / 16 < 23 / 17 := by sorry

end fraction_ordering_l1099_109937


namespace simon_is_ten_l1099_109903

def alvin_age : ℕ := 30

def simon_age : ℕ := alvin_age / 2 - 5

theorem simon_is_ten : simon_age = 10 := by
  sorry

end simon_is_ten_l1099_109903


namespace max_value_sqrt_sum_l1099_109908

theorem max_value_sqrt_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 2) : 
  (Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2 - a) * (2 - b) * (2 - c))) ≤ 4 :=
sorry

end max_value_sqrt_sum_l1099_109908


namespace min_value_A_over_C_l1099_109920

theorem min_value_A_over_C (x : ℝ) (A C : ℝ) (h1 : x^2 + 1/x^2 = A) (h2 : x + 1/x = C)
  (h3 : A > 0) (h4 : C > 0) (h5 : ∀ y : ℝ, y > 0 → y + 1/y ≥ 2) :
  A / C ≥ 1 ∧ ∃ x₀ : ℝ, x₀ > 0 ∧ (x₀^2 + 1/x₀^2) / (x₀ + 1/x₀) = 1 := by
  sorry

end min_value_A_over_C_l1099_109920


namespace percentage_to_pass_l1099_109904

def max_marks : ℕ := 780
def passing_marks : ℕ := 234

theorem percentage_to_pass : 
  (passing_marks : ℝ) / max_marks * 100 = 30 := by sorry

end percentage_to_pass_l1099_109904


namespace complex_fraction_equality_l1099_109938

theorem complex_fraction_equality : ∃ (i : ℂ), i * i = -1 ∧ (7 + i) / (3 + 4 * i) = 1 - i := by
  sorry

end complex_fraction_equality_l1099_109938


namespace domain_transformation_l1099_109963

/-- Given that the domain of f(x^2 - 1) is [0, 3], prove that the domain of f(2x - 1) is [0, 9/2] -/
theorem domain_transformation (f : ℝ → ℝ) :
  (∀ y, f y ≠ 0 → 0 ≤ y + 1 ∧ y + 1 ≤ 3) →
  (∀ x, f (2*x - 1) ≠ 0 → 0 ≤ x ∧ x ≤ 9/2) :=
sorry

end domain_transformation_l1099_109963


namespace inequality_proof_equality_condition_l1099_109999

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) : 
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3/2 :=
sorry

theorem equality_condition (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) : 
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) = 3/2 ↔ 
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end inequality_proof_equality_condition_l1099_109999


namespace square_room_tiles_and_triangles_l1099_109973

theorem square_room_tiles_and_triangles (n : ℕ) : 
  n > 0 →  -- Ensure the room has a positive side length
  (2 * n - 1 = 57) →  -- Total tiles on diagonals
  (n^2 = 841 ∧ 4 = 4) :=  -- Total tiles and number of triangles
by sorry

end square_room_tiles_and_triangles_l1099_109973


namespace greatest_integer_fraction_inequality_l1099_109972

theorem greatest_integer_fraction_inequality : 
  ∀ x : ℤ, (8 : ℚ) / 11 > (x : ℚ) / 15 ↔ x ≤ 10 :=
sorry

end greatest_integer_fraction_inequality_l1099_109972


namespace circle_center_radius_sum_l1099_109900

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 16 = -y^2 + 6*x + 36

-- Define the center and radius of the circle
def is_center_and_radius (a b r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), is_center_and_radius a b r ∧ a + b + r = 5 + Real.sqrt 65 :=
sorry

end circle_center_radius_sum_l1099_109900


namespace allison_total_items_l1099_109961

/-- Represents the number of craft items bought by a person -/
structure CraftItems where
  glueSticks : ℕ
  constructionPaper : ℕ

/-- The problem setup -/
def craftProblem (marie allison : CraftItems) : Prop :=
  allison.glueSticks = marie.glueSticks + 8 ∧
  marie.constructionPaper = 6 * allison.constructionPaper ∧
  marie.glueSticks = 15 ∧
  marie.constructionPaper = 30

/-- The theorem to prove -/
theorem allison_total_items (marie allison : CraftItems) 
  (h : craftProblem marie allison) : 
  allison.glueSticks + allison.constructionPaper = 28 := by
  sorry


end allison_total_items_l1099_109961


namespace power_of_sum_squares_and_abs_l1099_109987

theorem power_of_sum_squares_and_abs (a b : ℝ) : 
  (a - 4)^2 + |2 - b| = 0 → a^b = 16 := by
  sorry

end power_of_sum_squares_and_abs_l1099_109987


namespace average_age_of_ten_students_l1099_109959

theorem average_age_of_ten_students
  (total_students : ℕ)
  (average_age_all : ℚ)
  (num_group1 : ℕ)
  (average_age_group1 : ℚ)
  (age_last_student : ℕ)
  (h1 : total_students = 25)
  (h2 : average_age_all = 25)
  (h3 : num_group1 = 14)
  (h4 : average_age_group1 = 28)
  (h5 : age_last_student = 13)
  : ∃ (average_age_group2 : ℚ),
    average_age_group2 = 22 ∧
    average_age_group2 * (total_students - num_group1 - 1) =
      total_students * average_age_all - num_group1 * average_age_group1 - age_last_student :=
by sorry

end average_age_of_ten_students_l1099_109959


namespace geometric_sequence_property_l1099_109928

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end geometric_sequence_property_l1099_109928


namespace relay_race_arrangements_l1099_109922

/-- The number of female students --/
def total_students : ℕ := 6

/-- The number of students to be selected for the relay race --/
def selected_students : ℕ := 4

/-- A function to calculate the number of arrangements when only one of A or B participates --/
def one_participates : ℕ := 2 * (total_students - 2).choose (selected_students - 1) * (selected_students).factorial

/-- A function to calculate the number of arrangements when both A and B participate --/
def both_participate : ℕ := selected_students.choose 2 * (selected_students - 1).factorial

/-- The total number of different arrangements --/
def total_arrangements : ℕ := one_participates + both_participate

theorem relay_race_arrangements :
  total_arrangements = 264 := by sorry

end relay_race_arrangements_l1099_109922


namespace tan_sum_simplification_l1099_109951

theorem tan_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (40 * π / 180) + Real.tan (50 * π / 180)) / Real.sin (30 * π / 180) = 
  2 * (Real.cos (40 * π / 180) * Real.cos (50 * π / 180) + 
       Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) / 
      (Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * 
       Real.cos (40 * π / 180) * Real.cos (50 * π / 180)) := by
  sorry

end tan_sum_simplification_l1099_109951


namespace rectangle_tiling_exists_l1099_109957

/-- A tiling of a rectangle using two layers of 1 × 2 bricks -/
structure Tiling (n m : ℕ) :=
  (layer1 : Fin n → Fin (2*m) → Bool)
  (layer2 : Fin n → Fin (2*m) → Bool)

/-- Predicate to check if a tiling is valid -/
def is_valid_tiling (n m : ℕ) (t : Tiling n m) : Prop :=
  (∀ i j, t.layer1 i j ∨ t.layer2 i j) ∧ 
  (∀ i j, ¬(t.layer1 i j ∧ t.layer2 i j))

/-- Main theorem: A valid tiling exists for any rectangle n × 2m where n > 1 -/
theorem rectangle_tiling_exists (n m : ℕ) (h : n > 1) : 
  ∃ t : Tiling n m, is_valid_tiling n m t :=
sorry

end rectangle_tiling_exists_l1099_109957


namespace profit_fluctuation_l1099_109980

theorem profit_fluctuation (march_profit : ℝ) (april_may_decrease : ℝ) :
  let april_profit := march_profit * 1.5
  let may_profit := april_profit * (1 - april_may_decrease / 100)
  let june_profit := may_profit * 1.5
  june_profit = march_profit * 1.8 →
  april_may_decrease = 20 := by
sorry

end profit_fluctuation_l1099_109980


namespace swimming_problem_l1099_109918

/-- Represents the daily swimming distances of Jamir, Sarah, and Julien -/
structure SwimmingDistances where
  julien : ℕ
  sarah : ℕ
  jamir : ℕ

/-- Calculates the total distance swam by all three swimmers in a week -/
def weeklyTotalDistance (d : SwimmingDistances) : ℕ :=
  7 * (d.julien + d.sarah + d.jamir)

/-- The swimming problem statement -/
theorem swimming_problem (d : SwimmingDistances) : 
  d.sarah = 2 * d.julien →
  d.jamir = d.sarah + 20 →
  weeklyTotalDistance d = 1890 →
  d.julien = 50 := by
  sorry

end swimming_problem_l1099_109918


namespace factorial_product_simplification_l1099_109934

theorem factorial_product_simplification (a : ℝ) :
  (1 * 1) * (2 * 1 * a) * (3 * 2 * 1 * a^3) * (4 * 3 * 2 * 1 * a^6) * (5 * 4 * 3 * 2 * 1 * a^10) = 34560 * a^20 := by
  sorry

end factorial_product_simplification_l1099_109934


namespace intersection_M_N_l1099_109964

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc 1 3 := by
  sorry

end intersection_M_N_l1099_109964


namespace sticker_count_l1099_109986

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := 30

/-- The number of stickers Steven has -/
def steven_stickers : ℕ := 3 * ryan_stickers

/-- The number of stickers Terry has -/
def terry_stickers : ℕ := steven_stickers + 20

/-- The total number of stickers Ryan, Steven, and Terry have altogether -/
def total_stickers : ℕ := ryan_stickers + steven_stickers + terry_stickers

theorem sticker_count : total_stickers = 230 := by
  sorry

end sticker_count_l1099_109986


namespace intersecting_circles_chord_length_l1099_109936

/-- Given two circles with radii 10 and 7, whose centers are 15 units apart,
    and a point P where the circles intersect, if a line is drawn through P
    such that QP = PR, then QP^2 = 10800/35. -/
theorem intersecting_circles_chord_length 
  (O₁ O₂ P Q R : ℝ × ℝ) -- Points in 2D plane
  (h₁ : dist O₁ O₂ = 15) -- Centers are 15 units apart
  (h₂ : dist O₁ P = 10) -- Radius of first circle
  (h₃ : dist O₂ P = 7)  -- Radius of second circle
  (h₄ : dist Q P = dist P R) -- QP = PR
  : (dist Q P)^2 = 10800/35 := by
  sorry

end intersecting_circles_chord_length_l1099_109936


namespace a_range_l1099_109969

def A (a : ℝ) : Set ℝ := {x | x^2 - 2*x + a > 0}

theorem a_range (a : ℝ) : (1 ∉ A a) ↔ a ≤ 1 := by
  sorry

end a_range_l1099_109969


namespace vegetables_in_box_l1099_109948

/-- Given a box with cabbages and radishes, we define the total number of vegetables -/
def total_vegetables (num_cabbages num_radishes : ℕ) : ℕ :=
  num_cabbages + num_radishes

/-- Theorem: In a box with 3 cabbages and 2 radishes, there are 5 vegetables in total -/
theorem vegetables_in_box : total_vegetables 3 2 = 5 := by
  sorry

end vegetables_in_box_l1099_109948


namespace bird_wings_problem_l1099_109967

theorem bird_wings_problem :
  ∃! (x y z : ℕ), 2 * x + 4 * y + 3 * z = 70 ∧ x = 2 * y := by
  sorry

end bird_wings_problem_l1099_109967


namespace root_sum_reciprocal_l1099_109970

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a^2 - a + 2 = 0) → 
  (b^3 - 2*b^2 - b + 2 = 0) → 
  (c^3 - 2*c^2 - c + 2 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = -19 / 16) :=
by sorry

end root_sum_reciprocal_l1099_109970


namespace five_greater_than_two_sqrt_five_l1099_109940

theorem five_greater_than_two_sqrt_five : 5 > 2 * Real.sqrt 5 := by
  sorry

end five_greater_than_two_sqrt_five_l1099_109940


namespace units_digit_sum_of_powers_l1099_109923

theorem units_digit_sum_of_powers : (42^5 + 24^5 + 2^5) % 10 = 8 := by
  sorry

end units_digit_sum_of_powers_l1099_109923


namespace inequality_proof_l1099_109942

theorem inequality_proof (n : ℕ) (x : ℝ) (h1 : n > 0) (h2 : x ≥ n^2) :
  n * Real.sqrt (x - n^2) ≤ x / 2 := by
  sorry

end inequality_proof_l1099_109942


namespace point_location_l1099_109932

theorem point_location (m n : ℝ) : 2^m + 2^n < 2 * Real.sqrt 2 → m + n < 1 := by
  sorry

end point_location_l1099_109932


namespace twenty_fifth_is_monday_l1099_109993

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Checks if a given number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 == 0

/-- Represents a month with its dates -/
structure Month where
  dates : List Date
  threeEvenSaturdays : ∃ (d1 d2 d3 : Date),
    d1 ∈ dates ∧ d2 ∈ dates ∧ d3 ∈ dates ∧
    d1.dayOfWeek = DayOfWeek.Saturday ∧
    d2.dayOfWeek = DayOfWeek.Saturday ∧
    d3.dayOfWeek = DayOfWeek.Saturday ∧
    isEven d1.day ∧ isEven d2.day ∧ isEven d3.day ∧
    d1.day ≠ d2.day ∧ d2.day ≠ d3.day ∧ d1.day ≠ d3.day

/-- Theorem: In a month where three Saturdays fall on even dates, 
    the 25th day of that month is a Monday -/
theorem twenty_fifth_is_monday (m : Month) : 
  ∃ (d : Date), d ∈ m.dates ∧ d.day = 25 ∧ d.dayOfWeek = DayOfWeek.Monday := by
  sorry

end twenty_fifth_is_monday_l1099_109993


namespace smaller_number_between_5_and_8_l1099_109983

theorem smaller_number_between_5_and_8 :
  (5 ≤ 8) ∧ (∀ x : ℝ, 5 ≤ x ∧ x ≤ 8 → 5 ≤ x) :=
by sorry

end smaller_number_between_5_and_8_l1099_109983


namespace sum_of_powers_equals_product_l1099_109907

theorem sum_of_powers_equals_product (n : ℕ) : 
  5^n + 5^n + 5^n + 5^n = 4 * 5^n := by sorry

end sum_of_powers_equals_product_l1099_109907


namespace problem_1_l1099_109941

theorem problem_1 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (1) * ((-2*a)^3 * (-a*b^2)^3 - 4*a*b^2 * (2*a^5*b^4 + 1/2*a*b^3 - 5)) / (-2*a*b) = a*b^4 - 10*b :=
sorry

end problem_1_l1099_109941


namespace m_range_is_open_interval_l1099_109996

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

/-- The range of m for which z = (m+3) + (m-1)i is in the fourth quadrant -/
def m_range : Set ℝ := {m : ℝ | in_fourth_quadrant ((m + 3) + (m - 1) * Complex.I)}

theorem m_range_is_open_interval : 
  m_range = Set.Ioo (-3) 1 := by sorry

end m_range_is_open_interval_l1099_109996


namespace shopkeeper_gain_percentage_l1099_109921

/-- Calculates the gain percentage of a shopkeeper using a false weight --/
theorem shopkeeper_gain_percentage 
  (true_weight : ℝ) 
  (false_weight : ℝ) 
  (h1 : true_weight = 1000) 
  (h2 : false_weight = 980) : 
  (true_weight - false_weight) / true_weight * 100 = 2 := by
  sorry

end shopkeeper_gain_percentage_l1099_109921


namespace chess_players_per_game_l1099_109910

theorem chess_players_per_game (total_players : Nat) (total_games : Nat) (players_per_game : Nat) : 
  total_players = 8 → 
  total_games = 28 → 
  (total_players.choose players_per_game) = total_games → 
  players_per_game = 2 := by
sorry

end chess_players_per_game_l1099_109910


namespace valid_pairs_l1099_109978

def is_valid_pair (m n : ℕ+) : Prop :=
  let d := Nat.gcd m n
  m + n^2 + d^3 = m * n * d

theorem valid_pairs :
  ∀ m n : ℕ+, is_valid_pair m n ↔ (m = 4 ∧ n = 2) ∨ (m = 4 ∧ n = 6) ∨ (m = 5 ∧ n = 2) ∨ (m = 5 ∧ n = 3) :=
sorry

end valid_pairs_l1099_109978


namespace value_of_expression_l1099_109929

theorem value_of_expression (a b : ℝ) (h : 2 * a - b = -1) : 
  2021 + 4 * a - 2 * b = 2019 := by
  sorry

end value_of_expression_l1099_109929


namespace julia_bought_399_balls_l1099_109956

/-- The number of balls Julia bought -/
def total_balls (red_packs yellow_packs green_packs balls_per_pack : ℕ) : ℕ :=
  (red_packs + yellow_packs + green_packs) * balls_per_pack

/-- Proof that Julia bought 399 balls -/
theorem julia_bought_399_balls :
  total_balls 3 10 8 19 = 399 := by
  sorry

end julia_bought_399_balls_l1099_109956


namespace unique_functional_equation_l1099_109952

theorem unique_functional_equation (f : ℕ+ → ℕ+)
  (h : ∀ m n : ℕ+, f (f m + f n) = m + n) :
  f 1988 = 1988 := by
  sorry

end unique_functional_equation_l1099_109952


namespace necessary_but_not_sufficient_l1099_109975

/-- Represents an ellipse with the given equation -/
structure Ellipse (k : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 / (k - 4) + y^2 / (10 - k) = 1)

/-- Condition for the foci to be on the x-axis -/
def foci_on_x_axis (k : ℝ) : Prop :=
  k - 4 > 10 - k

/-- The main theorem stating that 4 < k < 10 is necessary but not sufficient -/
theorem necessary_but_not_sufficient :
  ∃ k : ℝ, 4 < k ∧ k < 10 ∧
  (∀ k' : ℝ, (∃ e : Ellipse k', foci_on_x_axis k') → 4 < k' ∧ k' < 10) ∧
  ¬(∀ k' : ℝ, 4 < k' ∧ k' < 10 → ∃ e : Ellipse k', foci_on_x_axis k') :=
sorry

end necessary_but_not_sufficient_l1099_109975


namespace quadratic_coefficient_determination_l1099_109901

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient_determination
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h_f : f = QuadraticFunction a b c)
  (h_point : f 0 = 3)
  (h_vertex : ∃ (k : ℝ), f 2 = -1 ∧ ∀ x, f x ≥ f 2) :
  a = 1 := by
  sorry

end quadratic_coefficient_determination_l1099_109901


namespace martian_puzzle_l1099_109966

-- Define the Martian type
inductive Martian
| Red
| Blue

-- Define the state of the Martians
structure MartianState where
  total : Nat
  initialRed : Nat
  currentRed : Nat

-- Define the properties of the Martians' answers
def validAnswerSequence (state : MartianState) : Prop :=
  state.total = 2018 ∧
  ∀ i : Nat, i < state.total → 
    (i + 1 = state.initialRed + i - state.initialRed + 1)

-- Define the theorem
theorem martian_puzzle :
  ∀ state : MartianState,
    validAnswerSequence state →
    (state.initialRed = 0 ∨ state.initialRed = 1) :=
by
  sorry

end martian_puzzle_l1099_109966


namespace chess_board_configurations_l1099_109912

/-- Represents a chess board configuration -/
def ChessBoard := Fin 5 → Fin 5

/-- The number of ways to arrange n distinct items -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to place pawns on the board -/
def num_pawn_placements : ℕ := factorial 5

/-- The number of ways to assign distinct pawns to positions -/
def num_pawn_assignments : ℕ := factorial 5

/-- The total number of valid configurations -/
def total_configurations : ℕ := num_pawn_placements * num_pawn_assignments

/-- Theorem stating the total number of valid configurations -/
theorem chess_board_configurations :
  total_configurations = 14400 := by sorry

end chess_board_configurations_l1099_109912


namespace compare_star_operations_l1099_109927

-- Define the new operation
def star (a b : ℤ) : ℚ := (a * b : ℚ) - (a : ℚ) / (b : ℚ)

-- Theorem statement
theorem compare_star_operations : star 6 (-3) < star 4 (-4) := by
  sorry

end compare_star_operations_l1099_109927


namespace tangent_line_slope_logarithm_inequality_l1099_109945

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Theorem for the tangent line
theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = k * x₀ ∧ (deriv f) x₀ = k) ↔ k = 1 / Real.exp 1 :=
sorry

-- Theorem for the inequality
theorem logarithm_inequality (a x : ℝ) (ha : a ≥ 1) (hx : x > 0) :
  f x ≤ a * x + (a - 1) / x - 1 :=
sorry

end tangent_line_slope_logarithm_inequality_l1099_109945


namespace probability_of_one_in_pascal_triangle_l1099_109998

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Counts the number of 1s in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ :=
  sorry

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The probability of randomly selecting a 1 from the first 15 rows of Pascal's Triangle is 29/120 -/
theorem probability_of_one_in_pascal_triangle : 
  (countOnes 15 : ℚ) / (totalElements 15 : ℚ) = 29 / 120 :=
sorry

end probability_of_one_in_pascal_triangle_l1099_109998


namespace park_area_l1099_109916

/-- Given a rectangular park with length to breadth ratio of 1:2, where a cyclist completes one round along the boundary in 6 minutes at an average speed of 6 km/hr, prove that the area of the park is 20,000 square meters. -/
theorem park_area (length width : ℝ) (average_speed : ℝ) (time_taken : ℝ) : 
  length > 0 ∧ 
  width > 0 ∧ 
  length = (1/2) * width ∧ 
  average_speed = 6 ∧ 
  time_taken = 1/10 ∧ 
  2 * (length + width) = average_speed * time_taken * 1000 →
  length * width = 20000 := by sorry

end park_area_l1099_109916


namespace clare_bought_four_loaves_l1099_109981

def clares_bread_purchase (initial_money : ℕ) (milk_cartons : ℕ) (bread_cost : ℕ) (milk_cost : ℕ) (money_left : ℕ) : ℕ :=
  ((initial_money - money_left) - (milk_cartons * milk_cost)) / bread_cost

theorem clare_bought_four_loaves :
  clares_bread_purchase 47 2 2 2 35 = 4 := by
  sorry

end clare_bought_four_loaves_l1099_109981


namespace min_distance_is_zero_l1099_109924

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := x^2 - 5*x + 4

-- Define the distance function between the two graphs
def distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_distance_is_zero :
  ∃ (x : ℝ), distance x = 0 ∧ ∀ (y : ℝ), distance y ≥ 0 :=
sorry

end min_distance_is_zero_l1099_109924


namespace smallest_number_with_three_prime_factors_ge_10_l1099_109958

def is_prime (n : ℕ) : Prop := sorry

def has_exactly_three_prime_factors (n : ℕ) : Prop := sorry

def all_prime_factors_ge_10 (n : ℕ) : Prop := sorry

theorem smallest_number_with_three_prime_factors_ge_10 :
  ∀ n : ℕ, (has_exactly_three_prime_factors n ∧ all_prime_factors_ge_10 n) → n ≥ 2431 :=
by sorry

end smallest_number_with_three_prime_factors_ge_10_l1099_109958


namespace negation_equivalence_l1099_109930

theorem negation_equivalence :
  (¬ ∃ x₀ > 0, x₀^2 - 5*x₀ + 6 > 0) ↔ (∀ x > 0, x^2 - 5*x + 6 ≤ 0) :=
by sorry

end negation_equivalence_l1099_109930


namespace problem_solution_l1099_109962

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x - a

theorem problem_solution :
  -- Part 1: Find the value of a
  (∃ a : ℝ, f a 0 = -5 ∧ a = -5) ∧
  -- Part 2: Find the equation of the tangent line
  (∃ x y : ℝ,
    -- Point M(x, y) is on the curve f
    y = f (-5) x ∧
    -- Tangent line at M is parallel to 3x + 2y + 2 = 0
    f' (-5) x = -3/2 ∧
    -- Equation of the tangent line
    (24 : ℝ) * x + 16 * y - 37 = 0) :=
by sorry

end problem_solution_l1099_109962


namespace prob_at_least_six_heads_in_eight_flips_l1099_109911

def num_flips : ℕ := 8
def min_heads : ℕ := 6

-- Probability of getting at least min_heads in num_flips flips of a fair coin
def prob_at_least_heads : ℚ :=
  (Finset.sum (Finset.range (num_flips - min_heads + 1))
    (λ i => Nat.choose num_flips (num_flips - i))) / 2^num_flips

theorem prob_at_least_six_heads_in_eight_flips :
  prob_at_least_heads = 37 / 256 := by
  sorry

end prob_at_least_six_heads_in_eight_flips_l1099_109911


namespace union_equality_condition_l1099_109989

open Set

theorem union_equality_condition (a : ℝ) :
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
  let B : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
  (A ∪ B = A) ↔ a ∈ Set.Icc (-2) 0 := by
  sorry

end union_equality_condition_l1099_109989


namespace a_3_value_l1099_109968

def a (n : ℕ) : ℚ := (-1)^n * (n : ℚ) / (n + 1)

theorem a_3_value : a 3 = -3/4 := by
  sorry

end a_3_value_l1099_109968


namespace kopek_payment_l1099_109905

theorem kopek_payment (n : ℕ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end kopek_payment_l1099_109905


namespace influenza_transmission_rate_l1099_109953

theorem influenza_transmission_rate (initial_infected : ℕ) (total_infected : ℕ) : 
  initial_infected = 4 →
  total_infected = 256 →
  ∃ (x : ℕ), 
    x > 0 ∧
    initial_infected + initial_infected * x + (initial_infected + initial_infected * x) * x = total_infected →
    x = 7 :=
by
  sorry

end influenza_transmission_rate_l1099_109953


namespace parentheses_removal_l1099_109943

theorem parentheses_removal (a b c : ℝ) : -3*a - (2*b - c) = -3*a - 2*b + c := by
  sorry

end parentheses_removal_l1099_109943


namespace inverse_f_at_3_l1099_109982

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the domain of f
def domain (x : ℝ) : Prop := -2 ≤ x ∧ x < 0

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), 
    (∀ x, domain x → f_inv (f x) = x) ∧
    (∀ y, ∃ x, domain x ∧ f x = y → f_inv y = x) ∧
    f_inv 3 = -1 := by
  sorry

end inverse_f_at_3_l1099_109982


namespace remaining_payment_example_l1099_109950

/-- Given a deposit percentage and amount, calculates the remaining amount to be paid -/
def remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) : ℚ :=
  let total_cost := deposit_amount / deposit_percentage
  total_cost - deposit_amount

/-- Theorem: Given a 10% deposit of $130, the remaining amount to be paid is $1170 -/
theorem remaining_payment_example : 
  remaining_payment (1/10) 130 = 1170 := by
  sorry

end remaining_payment_example_l1099_109950


namespace right_trapezoid_area_l1099_109960

/-- The area of a right trapezoid with specific base proportions -/
theorem right_trapezoid_area : ∀ (lower_base : ℝ),
  lower_base > 0 →
  let upper_base := (3 / 5) * lower_base
  let height := (lower_base - upper_base) / 2
  (lower_base - 8 = height) →
  (1 / 2) * (lower_base + upper_base) * height = 192 := by
  sorry

end right_trapezoid_area_l1099_109960


namespace derivative_exp_cos_l1099_109965

/-- The derivative of e^x * cos(x) is e^x * (cos(x) - sin(x)) -/
theorem derivative_exp_cos (x : ℝ) : 
  deriv (λ x => Real.exp x * Real.cos x) x = Real.exp x * (Real.cos x - Real.sin x) := by
  sorry

end derivative_exp_cos_l1099_109965


namespace circumscribed_circle_twice_inscribed_l1099_109994

/-- Given a square, the area of its circumscribed circle is twice the area of its inscribed circle -/
theorem circumscribed_circle_twice_inscribed (a : ℝ) (ha : a > 0) :
  let square_side := 2 * a
  let inscribed_radius := a
  let circumscribed_radius := a * Real.sqrt 2
  (π * circumscribed_radius ^ 2) = 2 * (π * inscribed_radius ^ 2) := by
  sorry

end circumscribed_circle_twice_inscribed_l1099_109994


namespace parade_runner_time_l1099_109988

/-- The time taken for a runner to travel from the front to the end of a moving parade -/
theorem parade_runner_time (parade_length : ℝ) (parade_speed : ℝ) (runner_speed : ℝ) :
  parade_length = 2 →
  parade_speed = 3 →
  runner_speed = 6 →
  (parade_length / (runner_speed - parade_speed)) * 60 = 40 := by
  sorry

end parade_runner_time_l1099_109988


namespace no_real_roots_quadratic_l1099_109906

theorem no_real_roots_quadratic (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 - 2 * x + 1 ≠ 0) → m > 2 := by
  sorry

end no_real_roots_quadratic_l1099_109906


namespace two_propositions_true_l1099_109990

theorem two_propositions_true : 
  (¬(∀ x : ℝ, x^2 > 0)) ∧ 
  (∃ x : ℝ, x^2 ≤ x) ∧ 
  (∀ M N : Set α, ∀ x : α, x ∈ M ∩ N → x ∈ M ∧ x ∈ N) := by
  sorry

end two_propositions_true_l1099_109990


namespace quadratic_discriminant_l1099_109974

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + (5 + 1/2)x - 2 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := -2

theorem quadratic_discriminant : discriminant a b c = 281/4 := by
  sorry

end quadratic_discriminant_l1099_109974


namespace subway_speed_increase_l1099_109947

-- Define the speed function
def speed (s : ℝ) : ℝ := s^2 + 2*s

-- State the theorem
theorem subway_speed_increase (s : ℝ) : 
  0 ≤ s ∧ s ≤ 7 → 
  speed s = speed 5 + 28 → 
  s = 7 := by
  sorry

end subway_speed_increase_l1099_109947


namespace intersection_M_N_l1099_109984

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem intersection_M_N : ∀ x : ℝ, x ∈ (M ∩ N) ↔ 2 < x ∧ x ≤ 3 := by sorry

end intersection_M_N_l1099_109984


namespace max_value_of_f_l1099_109925

-- Define the function f
def f (x : ℝ) : ℝ := -x^4 + 2*x^2 + 3

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 4 := by
  sorry

end max_value_of_f_l1099_109925


namespace factor_polynomial_l1099_109914

theorem factor_polynomial (x : ℝ) : 46 * x^3 - 115 * x^7 = -23 * x^3 * (5 * x^4 - 2) := by
  sorry

end factor_polynomial_l1099_109914


namespace puzzle_solution_l1099_109946

theorem puzzle_solution :
  ∀ (S I A L T : ℕ),
  S ≠ 0 →
  S ≠ I ∧ S ≠ A ∧ S ≠ L ∧ S ≠ T ∧
  I ≠ A ∧ I ≠ L ∧ I ≠ T ∧
  A ≠ L ∧ A ≠ T ∧
  L ≠ T →
  10 * S + I < 100 →
  1000 * S + 100 * A + 10 * L + T < 10000 →
  (10 * S + I) * (10 * S + I) = 1000 * S + 100 * A + 10 * L + T →
  S = 9 ∧ I = 8 ∧ A = 6 ∧ L = 0 ∧ T = 4 :=
by sorry

end puzzle_solution_l1099_109946


namespace max_integer_squared_inequality_l1099_109917

theorem max_integer_squared_inequality : ∃ (n : ℕ),
  n = 30499 ∧ 
  n^2 ≤ 160 * 170 * 180 * 190 ∧
  ∀ (m : ℕ), m > n → m^2 > 160 * 170 * 180 * 190 := by
  sorry

end max_integer_squared_inequality_l1099_109917


namespace sum_of_p_and_q_l1099_109909

theorem sum_of_p_and_q (p q : ℝ) (h_distinct : p ≠ q) (h_greater : p > q) :
  let M := !![2, -5, 8; 1, p, q; 1, q, p]
  Matrix.det M = 0 → p + q = -13/2 := by
  sorry

end sum_of_p_and_q_l1099_109909


namespace median_salary_is_25000_l1099_109995

/-- Represents a position in the company -/
inductive Position
  | CEO
  | SeniorVicePresident
  | Manager
  | AssistantManager
  | Clerk

/-- Represents the salary distribution in the company -/
def salary_distribution : List (Position × Nat × Nat) :=
  [(Position.CEO, 1, 135000),
   (Position.SeniorVicePresident, 4, 95000),
   (Position.Manager, 12, 80000),
   (Position.AssistantManager, 8, 55000),
   (Position.Clerk, 38, 25000)]

/-- The total number of employees in the company -/
def total_employees : Nat := 63

/-- Calculates the median salary given the salary distribution and total number of employees -/
def median_salary (dist : List (Position × Nat × Nat)) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary is $25,000 -/
theorem median_salary_is_25000 :
  median_salary salary_distribution total_employees = 25000 := by
  sorry

end median_salary_is_25000_l1099_109995


namespace square_difference_equality_l1099_109954

theorem square_difference_equality : (1 + 2)^2 - (1^2 + 2^2) = 4 := by
  sorry

end square_difference_equality_l1099_109954


namespace chemistry_textbook_weight_l1099_109976

/-- The weight of the geometry textbook in pounds -/
def geometry_weight : ℝ := 0.62

/-- The additional weight of the chemistry textbook compared to the geometry textbook in pounds -/
def additional_weight : ℝ := 6.5

/-- The weight of the chemistry textbook in pounds -/
def chemistry_weight : ℝ := geometry_weight + additional_weight

theorem chemistry_textbook_weight : chemistry_weight = 7.12 := by
  sorry

end chemistry_textbook_weight_l1099_109976


namespace subset_condition_disjoint_condition_l1099_109944

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for B ⊆ A
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem for A ∩ B = ∅
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end subset_condition_disjoint_condition_l1099_109944


namespace greatest_integer_solution_l1099_109933

theorem greatest_integer_solution : 
  ∃ (x : ℤ), (8 - 6*x > 26) ∧ (∀ (y : ℤ), y > x → 8 - 6*y ≤ 26) := by
  sorry

end greatest_integer_solution_l1099_109933


namespace unique_scenario_l1099_109913

/-- Represents the type of islander -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents the possible responses to the question -/
inductive Response
  | Yes
  | No

/-- Represents the scenario of two islanders -/
structure IslandScenario where
  askedIslander : IslanderType
  otherIslander : IslanderType
  response : Response

/-- Determines if a given scenario is consistent with the rules of knights and liars -/
def isConsistentScenario (scenario : IslandScenario) : Prop :=
  match scenario.askedIslander, scenario.response with
  | IslanderType.Knight, Response.Yes => scenario.askedIslander = IslanderType.Knight ∨ scenario.otherIslander = IslanderType.Knight
  | IslanderType.Knight, Response.No => scenario.askedIslander ≠ IslanderType.Knight ∧ scenario.otherIslander ≠ IslanderType.Knight
  | IslanderType.Liar, Response.Yes => scenario.askedIslander ≠ IslanderType.Knight ∧ scenario.otherIslander ≠ IslanderType.Knight
  | IslanderType.Liar, Response.No => scenario.askedIslander = IslanderType.Knight ∨ scenario.otherIslander = IslanderType.Knight

/-- Determines if a given scenario provides definitive information about both islanders -/
def providesDefinitiveInfo (scenario : IslandScenario) : Prop :=
  isConsistentScenario scenario ∧
  ∀ (altScenario : IslandScenario),
    isConsistentScenario altScenario →
    scenario.askedIslander = altScenario.askedIslander ∧
    scenario.otherIslander = altScenario.otherIslander

/-- The main theorem: The only scenario that satisfies all conditions is when the asked islander is a liar and the other is a knight -/
theorem unique_scenario :
  ∃! (scenario : IslandScenario),
    isConsistentScenario scenario ∧
    providesDefinitiveInfo scenario ∧
    scenario.askedIslander = IslanderType.Liar ∧
    scenario.otherIslander = IslanderType.Knight :=
  sorry

end unique_scenario_l1099_109913


namespace min_side_b_in_special_triangle_l1099_109992

/-- 
Given a triangle ABC where:
- Angles A, B, and C form an arithmetic sequence
- Sides opposite to angles A, B, and C are a, b, and c respectively
- 3ac + b² = 25
This theorem states that the minimum value of side b is 5/2
-/
theorem min_side_b_in_special_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → c > 0 →  -- Ensuring positive side lengths
  2 * B = A + C →  -- Arithmetic sequence condition
  A + B + C = π →  -- Sum of angles in a triangle
  3 * a * c + b^2 = 25 →  -- Given condition
  b ≥ 5/2 ∧ ∃ (a₀ c₀ : ℝ), a₀ > 0 ∧ c₀ > 0 ∧ 3 * a₀ * c₀ + (5/2)^2 = 25 := by
  sorry

end min_side_b_in_special_triangle_l1099_109992


namespace sum_of_fourth_and_fifth_terms_l1099_109915

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fourth_and_fifth_terms
  (a : ℕ → ℕ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 10)
  (h_third : a 3 = 17)
  (h_sixth : a 6 = 38) :
  a 4 + a 5 = 55 :=
by sorry

end sum_of_fourth_and_fifth_terms_l1099_109915


namespace parallelogram_area_l1099_109902

/-- The area of a parallelogram with vertices at (1, 1), (7, 1), (4, 9), and (10, 9) is 48 square units. -/
theorem parallelogram_area : ℝ := by
  -- Define the vertices
  let v1 : ℝ × ℝ := (1, 1)
  let v2 : ℝ × ℝ := (7, 1)
  let v3 : ℝ × ℝ := (4, 9)
  let v4 : ℝ × ℝ := (10, 9)

  -- Define the parallelogram
  let parallelogram := [v1, v2, v3, v4]

  -- Calculate the area
  let area := 48

  -- Prove that the area of the parallelogram is 48 square units
  sorry

end parallelogram_area_l1099_109902


namespace value_of_a_l1099_109979

-- Define sets A and B
def A : Set ℝ := {x : ℝ | |x| = 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- Theorem statement
theorem value_of_a (a : ℝ) : A ⊇ B a → a = 1 ∨ a = 0 ∨ a = -1 := by
  sorry

end value_of_a_l1099_109979


namespace daal_consumption_reduction_l1099_109955

theorem daal_consumption_reduction (old_price new_price : ℝ) 
  (hold_price : old_price = 16) 
  (hnew_price : new_price = 20) : 
  (new_price - old_price) / old_price * 100 = 25 := by
  sorry

end daal_consumption_reduction_l1099_109955


namespace arithmetic_mean_fractions_l1099_109971

theorem arithmetic_mean_fractions : 
  let a := 7 / 11
  let b := 9 / 11
  let c := 8 / 11
  c = (a + b) / 2 := by sorry

end arithmetic_mean_fractions_l1099_109971


namespace kelly_carrot_harvest_l1099_109997

/-- The weight of Kelly's harvested carrots -/
def kelly_carrot_weight (bed1 bed2 bed3 carrots_per_pound : ℕ) : ℚ :=
  (bed1 + bed2 + bed3 : ℚ) / carrots_per_pound

/-- Theorem: Kelly harvested 39 pounds of carrots -/
theorem kelly_carrot_harvest :
  kelly_carrot_weight 55 101 78 6 = 39 := by
  sorry

end kelly_carrot_harvest_l1099_109997
