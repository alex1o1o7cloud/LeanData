import Mathlib

namespace unique_valid_number_l1748_174842

def is_valid_number (n : ℕ) : Prop :=
  -- The number is six digits long
  100000 ≤ n ∧ n < 1000000 ∧
  -- It begins with digit 1
  n / 100000 = 1 ∧
  -- It ends with digit 7
  n % 10 = 7 ∧
  -- If the last digit is decreased by 1 and moved to the first place,
  -- the resulting number is five times the original number
  (6 * 100000 + n / 10) = 5 * n

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 142857 :=
sorry

end unique_valid_number_l1748_174842


namespace fundraising_total_donation_l1748_174871

def total_donation (days : ℕ) (initial_donors : ℕ) (initial_donation : ℕ) : ℕ :=
  let rec donation_sum (d : ℕ) (donors : ℕ) (avg_donation : ℕ) (acc : ℕ) : ℕ :=
    if d = 0 then acc
    else donation_sum (d - 1) (donors * 2) (avg_donation + 5) (acc + donors * avg_donation)
  donation_sum days initial_donors initial_donation 0

theorem fundraising_total_donation :
  total_donation 5 10 10 = 8000 :=
by sorry

end fundraising_total_donation_l1748_174871


namespace distance_to_place_l1748_174876

/-- Calculates the distance to a place given rowing speed, current velocity, and round trip time -/
theorem distance_to_place (rowing_speed current_velocity : ℝ) (round_trip_time : ℝ) : 
  rowing_speed = 5 → 
  current_velocity = 1 → 
  round_trip_time = 1 → 
  (rowing_speed + current_velocity) * (rowing_speed - current_velocity) * round_trip_time / 
  (rowing_speed + current_velocity + rowing_speed - current_velocity) = 2.4 := by
sorry

end distance_to_place_l1748_174876


namespace angle_S_measure_l1748_174819

/-- Represents a convex pentagon with specific angle properties -/
structure ConvexPentagon where
  -- Angle measures
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  -- Convexity and angle sum property
  sum_angles : P + Q + R + S + T = 540
  -- Angle congruence properties
  PQR_congruent : P = Q ∧ Q = R
  ST_congruent : S = T
  -- Relation between P and S
  P_less_than_S : P + 30 = S

/-- 
Theorem: In a convex pentagon with the given properties, 
the measure of angle S is 126 degrees.
-/
theorem angle_S_measure (pentagon : ConvexPentagon) : pentagon.S = 126 := by
  sorry

end angle_S_measure_l1748_174819


namespace senior_trip_fraction_l1748_174855

theorem senior_trip_fraction (total_students : ℝ) (seniors : ℝ) (juniors : ℝ)
  (h1 : juniors = (2/3) * seniors)
  (h2 : (1/4) * juniors + seniors * x = (1/2) * total_students)
  (h3 : total_students = seniors + juniors)
  (h4 : x ≥ 0 ∧ x ≤ 1) :
  x = 2/3 := by
  sorry

end senior_trip_fraction_l1748_174855


namespace score_sum_theorem_l1748_174865

def total_score (keith larry danny emma fiona : ℝ) : ℝ :=
  keith + larry + danny + emma + fiona

theorem score_sum_theorem (keith larry danny emma fiona : ℝ) 
  (h1 : keith = 3.5)
  (h2 : larry = 3.2 * keith)
  (h3 : danny = larry + 5.7)
  (h4 : emma = 2 * danny - 1.2)
  (h5 : fiona = (keith + larry + danny + emma) / 4) :
  total_score keith larry danny emma fiona = 80.25 := by
  sorry

end score_sum_theorem_l1748_174865


namespace certain_value_proof_l1748_174884

theorem certain_value_proof (x y : ℕ) : 
  x + y = 50 → x = 30 → y = 20 → 2 * (x - y) = 20 := by
  sorry

end certain_value_proof_l1748_174884


namespace arcade_tickets_l1748_174820

theorem arcade_tickets (whack_a_mole skee_ball spent remaining : ℕ) :
  skee_ball = 25 ∧ spent = 7 ∧ remaining = 50 →
  whack_a_mole + skee_ball = remaining + spent →
  whack_a_mole = 7 := by
sorry

end arcade_tickets_l1748_174820


namespace talia_drive_distance_l1748_174858

/-- Represents the total distance Talia drives in a day -/
def total_distance (house_to_park park_to_store house_to_store : ℝ) : ℝ :=
  house_to_park + park_to_store + house_to_store

/-- Theorem stating the total distance Talia drives -/
theorem talia_drive_distance :
  ∀ (house_to_park park_to_store house_to_store : ℝ),
    house_to_park = 5 →
    park_to_store = 3 →
    house_to_store = 8 →
    total_distance house_to_park park_to_store house_to_store = 16 := by
  sorry

end talia_drive_distance_l1748_174858


namespace grandmother_mother_age_ratio_l1748_174887

/-- Represents the ages of Grace, her mother, and her grandmother -/
structure FamilyAges where
  grace : ℕ
  mother : ℕ
  grandmother : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.grace = 60 ∧
  ages.mother = 80 ∧
  ages.grace * 8 = ages.grandmother * 3 ∧
  ∃ k : ℕ, ages.grandmother = k * ages.mother

/-- The theorem to be proved -/
theorem grandmother_mother_age_ratio 
  (ages : FamilyAges) 
  (h : problem_conditions ages) : 
  ages.grandmother / ages.mother = 2 := by
  sorry


end grandmother_mother_age_ratio_l1748_174887


namespace cube_arrangement_theorem_l1748_174823

/-- Represents a cube with colored faces -/
structure Cube where
  blue_faces : Nat
  red_faces : Nat

/-- Represents an arrangement of cubes into a larger cube -/
structure CubeArrangement where
  cubes : List Cube
  visible_red_faces : Nat
  visible_blue_faces : Nat

/-- The theorem to be proved -/
theorem cube_arrangement_theorem 
  (cubes : List Cube) 
  (first_arrangement : CubeArrangement) :
  (cubes.length = 8) →
  (∀ c ∈ cubes, c.blue_faces = 2 ∧ c.red_faces = 4) →
  (first_arrangement.cubes = cubes) →
  (first_arrangement.visible_red_faces = 8) →
  (first_arrangement.visible_blue_faces = 16) →
  (∃ second_arrangement : CubeArrangement,
    second_arrangement.cubes = cubes ∧
    second_arrangement.visible_red_faces = 24 ∧
    second_arrangement.visible_blue_faces = 0) :=
by sorry

end cube_arrangement_theorem_l1748_174823


namespace diverse_dates_2013_l1748_174867

/-- A date in the format DD/MM/YY -/
structure Date where
  day : Nat
  month : Nat
  year : Nat

/-- Check if a date is valid (day between 1 and 31, month between 1 and 12) -/
def Date.isValid (d : Date) : Prop :=
  1 ≤ d.day ∧ d.day ≤ 31 ∧ 1 ≤ d.month ∧ d.month ≤ 12

/-- Convert a date to a list of digits -/
def Date.toDigits (d : Date) : List Nat :=
  (d.day / 10) :: (d.day % 10) :: (d.month / 10) :: (d.month % 10) :: (d.year / 10) :: [d.year % 10]

/-- Check if a date is diverse (contains all digits from 0 to 5 exactly once) -/
def Date.isDiverse (d : Date) : Prop :=
  let digits := d.toDigits
  ∀ n : Nat, n ≤ 5 → (digits.count n = 1)

/-- The main theorem: there are exactly 2 diverse dates in 2013 -/
theorem diverse_dates_2013 :
  ∃! (dates : List Date), 
    (∀ d ∈ dates, d.year = 13 ∧ d.isValid ∧ d.isDiverse) ∧ 
    dates.length = 2 := by
  sorry

end diverse_dates_2013_l1748_174867


namespace initial_game_cost_l1748_174836

theorem initial_game_cost (triple_value : ℝ → ℝ) (sold_percentage : ℝ) (sold_amount : ℝ) :
  triple_value = (λ x => 3 * x) →
  sold_percentage = 0.4 →
  sold_amount = 240 →
  ∃ (initial_cost : ℝ), sold_percentage * triple_value initial_cost = sold_amount ∧ initial_cost = 200 :=
by sorry

end initial_game_cost_l1748_174836


namespace max_ab_value_max_ab_value_achieved_l1748_174815

theorem max_ab_value (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) → 
  a * b ≤ (1/4 : ℝ) := by
  sorry

theorem max_ab_value_achieved (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) → 
  (∃ a' b' : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a' * x + b'| ≤ 1) ∧ a' * b' = (1/4 : ℝ)) := by
  sorry

end max_ab_value_max_ab_value_achieved_l1748_174815


namespace num_segments_collinear_points_l1748_174814

/-- The number of distinct segments formed by n collinear points -/
def num_segments (n : ℕ) : ℕ := n.choose 2

/-- Theorem: For n distinct collinear points, the number of distinct segments is n choose 2 -/
theorem num_segments_collinear_points (n : ℕ) (h : n ≥ 2) :
  num_segments n = n.choose 2 := by sorry

end num_segments_collinear_points_l1748_174814


namespace jelly_bean_probability_l1748_174830

/-- The probability of selecting a non-red jelly bean from a bag -/
theorem jelly_bean_probability : 
  let red : ℕ := 7
  let green : ℕ := 9
  let yellow : ℕ := 10
  let blue : ℕ := 12
  let purple : ℕ := 5
  let total : ℕ := red + green + yellow + blue + purple
  let non_red : ℕ := green + yellow + blue + purple
  (non_red : ℚ) / total = 36 / 43 := by
  sorry

end jelly_bean_probability_l1748_174830


namespace martha_cards_remaining_l1748_174879

theorem martha_cards_remaining (initial_cards : ℝ) (cards_to_emily : ℝ) (cards_to_olivia : ℝ) :
  initial_cards = 76.5 →
  cards_to_emily = 3.1 →
  cards_to_olivia = 5.2 →
  initial_cards - (cards_to_emily + cards_to_olivia) = 68.2 := by
sorry

end martha_cards_remaining_l1748_174879


namespace media_team_selection_count_l1748_174817

/-- The number of domestic media teams -/
def domestic_teams : ℕ := 6

/-- The number of foreign media teams -/
def foreign_teams : ℕ := 3

/-- The total number of teams to be selected -/
def selected_teams : ℕ := 3

/-- Represents whether domestic teams can ask questions consecutively -/
def consecutive_domestic : Prop := False

theorem media_team_selection_count : ℕ := by
  sorry

end media_team_selection_count_l1748_174817


namespace calculation_proof_l1748_174860

theorem calculation_proof : (24 / (8 + 2 - 5)) * 7 = 33.6 := by
  sorry

end calculation_proof_l1748_174860


namespace decimal_sum_equals_fraction_l1748_174874

/-- Represents a repeating decimal with a given numerator and denominator -/
def repeating_decimal (n : ℕ) (d : ℕ) : ℚ := n / d

/-- The sum of three specific repeating decimals -/
def decimal_sum : ℚ :=
  repeating_decimal 1 3 + repeating_decimal 2 99 + repeating_decimal 4 9999

theorem decimal_sum_equals_fraction : decimal_sum = 10581 / 29889 := by
  sorry

end decimal_sum_equals_fraction_l1748_174874


namespace unique_solution_cubic_system_l1748_174834

theorem unique_solution_cubic_system (x y z : ℝ) :
  x + y + z = 3 →
  x^2 + y^2 + z^2 = 3 →
  x^3 + y^3 + z^3 = 3 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_solution_cubic_system_l1748_174834


namespace vet_formula_portions_l1748_174889

/-- Calculates the total number of formula portions needed for puppies -/
def total_formula_portions (num_puppies : ℕ) (num_days : ℕ) (feedings_per_day : ℕ) : ℕ :=
  num_puppies * num_days * feedings_per_day

/-- Theorem: The vet gave Sandra 105 portions of formula for her puppies -/
theorem vet_formula_portions : total_formula_portions 7 5 3 = 105 := by
  sorry

end vet_formula_portions_l1748_174889


namespace simple_interest_rate_example_l1748_174809

/-- Given a principal amount, final amount, and time period, 
    calculate the simple interest rate as a percentage. -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  (amount - principal) * 100 / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is 7.6% -/
theorem simple_interest_rate_example : 
  simple_interest_rate 25000 34500 5 = 76/10 := by
  sorry

end simple_interest_rate_example_l1748_174809


namespace perpendicular_vectors_implies_m_half_l1748_174844

/-- Given two vectors a and b in R², if a is perpendicular to b,
    then the second component of a is equal to 1/2. -/
theorem perpendicular_vectors_implies_m_half (a b : ℝ × ℝ) :
  a.1 = 1 →
  a.2 = m →
  b.1 = -1 →
  b.2 = 2 →
  a.1 * b.1 + a.2 * b.2 = 0 →
  m = 1/2 := by
sorry

end perpendicular_vectors_implies_m_half_l1748_174844


namespace weight_of_b_l1748_174843

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 41) :
  b = 27 := by
sorry

end weight_of_b_l1748_174843


namespace quadratic_sum_l1748_174856

/-- A quadratic function g(x) = dx^2 + ex + f -/
def g (d e f x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  g d e f 0 = 5 → g d e f 2 = 3 → d + e + 3 * f = 14 := by
  sorry

end quadratic_sum_l1748_174856


namespace intersection_max_value_l1748_174805

/-- The polynomial function f(x) = x^6 - 10x^5 + 30x^4 - 20x^3 + 50x^2 - 24x + 48 -/
def f (x : ℝ) : ℝ := x^6 - 10*x^5 + 30*x^4 - 20*x^3 + 50*x^2 - 24*x + 48

/-- The line function g(x) = 8x -/
def g (x : ℝ) : ℝ := 8*x

theorem intersection_max_value :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (∀ x : ℝ, f x = g x → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, f y = g y → y ≤ x) →
  (∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, f y = g y → x = 6) :=
sorry

end intersection_max_value_l1748_174805


namespace remainder_theorem_l1748_174864

theorem remainder_theorem : ∃ q : ℕ, 2^404 + 404 = (2^203 + 2^101 + 1) * q + 403 := by
  sorry

end remainder_theorem_l1748_174864


namespace problem_solution_l1748_174853

theorem problem_solution (x y : ℝ) (h1 : y > 2*x) (h2 : x > 0) (h3 : x/y + y/x = 8) :
  (x + y) / (x - y) = -Real.sqrt (5/3) := by
sorry

end problem_solution_l1748_174853


namespace court_cases_guilty_l1748_174890

theorem court_cases_guilty (total : ℕ) (dismissed : ℕ) (delayed : ℕ) : 
  total = 27 → dismissed = 3 → delayed = 2 → 
  ∃ (guilty : ℕ), guilty = total - dismissed - (3 * (total - dismissed) / 4) - delayed ∧ guilty = 4 := by
sorry

end court_cases_guilty_l1748_174890


namespace paris_time_correct_l1748_174896

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : hours < 24
  mValid : minutes < 60

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a datetime with date and time -/
structure DateTime where
  date : Date
  time : Time

def time_difference : ℤ := -7

def beijing_time : DateTime := {
  date := { year := 2023, month := 10, day := 26 },
  time := { hours := 5, minutes := 0, hValid := by sorry, mValid := by sorry }
}

/-- Calculates the Paris time given the Beijing time and time difference -/
def calculate_paris_time (beijing : DateTime) (diff : ℤ) : DateTime :=
  sorry

theorem paris_time_correct :
  let paris_time := calculate_paris_time beijing_time time_difference
  paris_time.date.day = 25 ∧
  paris_time.date.month = 10 ∧
  paris_time.time.hours = 22 ∧
  paris_time.time.minutes = 0 :=
by sorry

end paris_time_correct_l1748_174896


namespace line_slope_intercept_sum_l1748_174857

/-- Given a line with slope -3 passing through (2, 5), prove m + b = 8 --/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -3 → 
  5 = m * 2 + b → 
  m + b = 8 := by
sorry

end line_slope_intercept_sum_l1748_174857


namespace book_price_percentage_l1748_174850

theorem book_price_percentage (suggested_retail_price : ℝ) : 
  suggested_retail_price > 0 →
  let marked_price := 0.7 * suggested_retail_price
  let purchase_price := 0.5 * marked_price
  purchase_price / suggested_retail_price = 0.35 := by sorry

end book_price_percentage_l1748_174850


namespace unique_solution_implies_sqrt_three_l1748_174802

theorem unique_solution_implies_sqrt_three (a : ℝ) :
  (∃! x : ℝ, x^2 + a * |x| + a^2 - 3 = 0) → a = Real.sqrt 3 := by
  sorry

end unique_solution_implies_sqrt_three_l1748_174802


namespace infinite_geometric_series_ratio_l1748_174885

theorem infinite_geometric_series_ratio
  (a : ℝ)  -- first term
  (S : ℝ)  -- sum of the series
  (h1 : a = 328)
  (h2 : S = 2009)
  (h3 : S = a / (1 - r))  -- formula for sum of infinite geometric series
  : r = 41 / 49 :=
by sorry

end infinite_geometric_series_ratio_l1748_174885


namespace perp_line_plane_relation_l1748_174892

-- Define the concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the perpendicularity relations
def perp_to_countless_lines (L : Line) (α : Plane) : Prop := sorry
def perp_to_plane (L : Line) (α : Plane) : Prop := sorry

-- State the theorem
theorem perp_line_plane_relation (L : Line) (α : Plane) :
  (perp_to_plane L α → perp_to_countless_lines L α) ∧
  ∃ L α, perp_to_countless_lines L α ∧ ¬perp_to_plane L α :=
sorry

end perp_line_plane_relation_l1748_174892


namespace smallest_n_value_l1748_174813

-- Define the cost of purple candy
def purple_cost : ℕ := 20

-- Define the quantities of other candies
def red_quantity : ℕ := 12
def green_quantity : ℕ := 14
def blue_quantity : ℕ := 15

-- Define the theorem
theorem smallest_n_value :
  ∃ (n : ℕ), n > 0 ∧ 
  (purple_cost * n) % red_quantity = 0 ∧
  (purple_cost * n) % green_quantity = 0 ∧
  (purple_cost * n) % blue_quantity = 0 ∧
  (∀ (m : ℕ), m > 0 → 
    (purple_cost * m) % red_quantity = 0 →
    (purple_cost * m) % green_quantity = 0 →
    (purple_cost * m) % blue_quantity = 0 →
    m ≥ n) ∧
  n = 21 :=
by sorry

end smallest_n_value_l1748_174813


namespace largest_value_l1748_174821

theorem largest_value (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) :
  b > 1/2 ∧ b > 2*a*b ∧ b > a^2 + b^2 := by
  sorry

end largest_value_l1748_174821


namespace percentage_equality_l1748_174859

theorem percentage_equality (x : ℝ) : 
  (15 / 100) * 75 = (x / 100) * 450 → x = 2.5 := by
sorry

end percentage_equality_l1748_174859


namespace xiaoyue_speed_l1748_174851

/-- Prove that Xiaoyue's average speed is 50 km/h given the conditions of the problem -/
theorem xiaoyue_speed (x : ℝ) 
  (h1 : x > 0)  -- Xiaoyue's speed is positive
  (h2 : 20 / x - 18 / (1.2 * x) = 1 / 10) : x = 50 := by
  sorry

#check xiaoyue_speed

end xiaoyue_speed_l1748_174851


namespace mixed_doubles_selection_methods_l1748_174804

def male_athletes : ℕ := 5
def female_athletes : ℕ := 6
def selected_male : ℕ := 2
def selected_female : ℕ := 2

theorem mixed_doubles_selection_methods :
  (Nat.choose male_athletes selected_male) *
  (Nat.choose female_athletes selected_female) *
  (Nat.factorial selected_male) = 300 := by
sorry

end mixed_doubles_selection_methods_l1748_174804


namespace sum_of_angles_l1748_174831

theorem sum_of_angles (α β : Real) : 
  (∃ x y : Real, x^2 - 3 * Real.sqrt 3 * x + 4 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  (0 < α ∧ α < π/2) →
  (0 < β ∧ β < π/2) →
  α + β = 2*π/3 := by
sorry

end sum_of_angles_l1748_174831


namespace cross_section_area_formula_l1748_174816

/-- Regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) :=
  (edge_length : a > 0)

/-- Plane passing through the midpoint of an edge and perpendicular to an adjacent edge -/
structure CrossSectionPlane (t : RegularTetrahedron a) :=
  (passes_through_midpoint : Bool)
  (perpendicular_to_adjacent : Bool)

/-- The area of the cross-section formed by the plane -/
def cross_section_area (t : RegularTetrahedron a) (p : CrossSectionPlane t) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_formula (a : ℝ) (t : RegularTetrahedron a) (p : CrossSectionPlane t) :
  p.passes_through_midpoint ∧ p.perpendicular_to_adjacent →
  cross_section_area t p = (a^2 * Real.sqrt 2) / 16 :=
sorry

end cross_section_area_formula_l1748_174816


namespace nested_root_simplification_l1748_174846

theorem nested_root_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x^2 * Real.sqrt (x^3 * Real.sqrt (x^4))) = (x^9)^(1/4) := by
  sorry

end nested_root_simplification_l1748_174846


namespace pens_per_student_l1748_174828

theorem pens_per_student (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) 
  (h1 : total_pens = 1001)
  (h2 : total_pencils = 910)
  (h3 : max_students = 91) :
  total_pens / max_students = 11 := by
sorry

end pens_per_student_l1748_174828


namespace seokjin_pencils_used_l1748_174837

/-- The number of pencils Seokjin used -/
def pencils_seokjin_used : ℕ := 9

theorem seokjin_pencils_used 
  (initial_pencils : ℕ) 
  (pencils_given : ℕ) 
  (final_pencils : ℕ) 
  (h1 : initial_pencils = 12)
  (h2 : pencils_given = 4)
  (h3 : final_pencils = 7) :
  pencils_seokjin_used = initial_pencils - final_pencils + pencils_given :=
by sorry

#check seokjin_pencils_used

end seokjin_pencils_used_l1748_174837


namespace min_area_inscribed_equilateral_l1748_174829

/-- The minimum area of an inscribed equilateral triangle in a right triangle -/
theorem min_area_inscribed_equilateral (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let min_area := (Real.sqrt 3 * a^2 * b^2) / (4 * (a^2 + b^2 + Real.sqrt 3 * a * b))
  ∀ (D E F : ℝ × ℝ),
    let A := (0, 0)
    let B := (a, 0)
    let C := (0, b)
    (D.1 ≥ 0 ∧ D.1 ≤ a ∧ D.2 = 0) →  -- D is on BC
    (E.1 = 0 ∧ E.2 ≥ 0 ∧ E.2 ≤ b) →  -- E is on CA
    (F.2 = (b / a) * F.1 ∧ F.1 ≥ 0 ∧ F.1 ≤ a) →  -- F is on AB
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = (E.1 - F.1)^2 + (E.2 - F.2)^2 →  -- DEF is equilateral
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = (F.1 - D.1)^2 + (F.2 - D.2)^2 →
    let area := Real.sqrt 3 / 4 * ((D.1 - E.1)^2 + (D.2 - E.2)^2)
    area ≥ min_area :=
by sorry

end min_area_inscribed_equilateral_l1748_174829


namespace four_tangent_lines_with_equal_intercepts_l1748_174811

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  let d := |l.a * x₀ + l.b * y₀ + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d = c.radius

/-- Check if a line has equal intercepts on both axes -/
def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c/l.a = -l.c/l.b

/-- The main theorem -/
theorem four_tangent_lines_with_equal_intercepts :
  let c : Circle := { center := (3, 3), radius := Real.sqrt 8 }
  ∃ (lines : Finset Line),
    lines.card = 4 ∧
    ∀ l ∈ lines, isTangent l c ∧ hasEqualIntercepts l ∧
    ∀ l', isTangent l' c → hasEqualIntercepts l' → l' ∈ lines :=
sorry

end four_tangent_lines_with_equal_intercepts_l1748_174811


namespace smallest_odd_prime_divisor_of_difference_of_squares_l1748_174825

def is_odd_prime (p : Nat) : Prop := Nat.Prime p ∧ p % 2 = 1

theorem smallest_odd_prime_divisor_of_difference_of_squares :
  ∃ (k : Nat), k = 3 ∧
  (∀ (m n : Nat), is_odd_prime m → is_odd_prime n → m < 10 → n < 10 → n < m →
    k ∣ (m^2 - n^2)) ∧
  (∀ (p : Nat), p < k → is_odd_prime p →
    ∃ (m n : Nat), is_odd_prime m ∧ is_odd_prime n ∧ m < 10 ∧ n < 10 ∧ n < m ∧
      ¬(p ∣ (m^2 - n^2))) := by
sorry

end smallest_odd_prime_divisor_of_difference_of_squares_l1748_174825


namespace same_grade_percentage_l1748_174882

theorem same_grade_percentage :
  let total_students : ℕ := 40
  let same_grade_A : ℕ := 3
  let same_grade_B : ℕ := 6
  let same_grade_C : ℕ := 4
  let same_grade_D : ℕ := 2
  let total_same_grade := same_grade_A + same_grade_B + same_grade_C + same_grade_D
  (total_same_grade : ℚ) / total_students * 100 = 37.5 := by
  sorry

end same_grade_percentage_l1748_174882


namespace exists_self_appended_perfect_square_l1748_174862

theorem exists_self_appended_perfect_square :
  ∃ (A : ℕ), ∃ (n : ℕ), ∃ (B : ℕ),
    A > 0 ∧ 
    10^n ≤ A ∧ A < 10^(n+1) ∧
    A * (10^n + 1) = B^2 :=
sorry

end exists_self_appended_perfect_square_l1748_174862


namespace people_in_line_l1748_174852

theorem people_in_line (people_in_front : ℕ) (people_behind : ℕ) : 
  people_in_front = 11 → people_behind = 12 → people_in_front + people_behind + 1 = 24 := by
  sorry

end people_in_line_l1748_174852


namespace derivative_of_y_l1748_174894

noncomputable def y (x : ℝ) : ℝ := x / (1 - Real.cos x)

theorem derivative_of_y (x : ℝ) (h : x ≠ 0) :
  deriv y x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
by sorry

end derivative_of_y_l1748_174894


namespace john_shopping_cost_l1748_174875

/-- The total cost of buying shirts and ties -/
def total_cost (num_shirts : ℕ) (shirt_price : ℚ) (num_ties : ℕ) (tie_price : ℚ) : ℚ :=
  num_shirts * shirt_price + num_ties * tie_price

/-- Theorem: The total cost of 3 shirts at $15.75 each and 2 ties at $9.40 each is $66.05 -/
theorem john_shopping_cost : 
  total_cost 3 (15.75 : ℚ) 2 (9.40 : ℚ) = (66.05 : ℚ) := by
  sorry

end john_shopping_cost_l1748_174875


namespace xyz_maximum_l1748_174827

theorem xyz_maximum (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_one : x + y + z = 1) (sum_inv_eq_ten : 1/x + 1/y + 1/z = 10) :
  xyz ≤ 4/125 ∧ ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ 1/x + 1/y + 1/z = 10 ∧ x*y*z = 4/125 :=
by sorry

end xyz_maximum_l1748_174827


namespace largest_expression_l1748_174881

theorem largest_expression (x : ℝ) : 
  (x + 1/4) * (x - 1/4) ≥ (x + 1) * (x - 1) ∧
  (x + 1/4) * (x - 1/4) ≥ (x + 1/2) * (x - 1/2) ∧
  (x + 1/4) * (x - 1/4) ≥ (x + 1/3) * (x - 1/3) := by
  sorry

end largest_expression_l1748_174881


namespace sin_cube_identity_l1748_174847

theorem sin_cube_identity (θ : Real) : 
  Real.sin θ ^ 3 = (-1/4) * Real.sin (3*θ) + (3/4) * Real.sin θ := by
  sorry

end sin_cube_identity_l1748_174847


namespace max_points_for_28_lines_l1748_174849

/-- The maximum number of lines that can be determined by n distinct points on a plane -/
def maxLines (n : ℕ) : ℕ :=
  if n ≤ 1 then 0
  else (n * (n - 1)) / 2

/-- Theorem: 8 is the maximum number of distinct points on a plane that can determine at most 28 lines -/
theorem max_points_for_28_lines :
  (∀ n : ℕ, n ≤ 8 → maxLines n ≤ 28) ∧
  (maxLines 8 = 28) :=
sorry

end max_points_for_28_lines_l1748_174849


namespace x_calculation_l1748_174803

theorem x_calculation (m n p q x : ℝ) :
  x^2 + (2*m*p + 2*n*q)^2 + (2*m*q - 2*n*p)^2 = (m^2 + n^2 + p^2 + q^2)^2 →
  x = m^2 + n^2 - p^2 - q^2 ∨ x = -(m^2 + n^2 - p^2 - q^2) := by
sorry

end x_calculation_l1748_174803


namespace elisa_books_problem_l1748_174810

theorem elisa_books_problem :
  ∀ (total science math lit : ℕ),
  science = 24 →
  total = science + math + lit →
  total < 100 →
  (math + 1) * 9 = total + 1 →
  lit * 4 = total + 1 →
  math = 7 :=
by sorry

end elisa_books_problem_l1748_174810


namespace sin_90_degrees_l1748_174883

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by sorry

end sin_90_degrees_l1748_174883


namespace original_average_calculation_l1748_174869

theorem original_average_calculation (S : Finset ℝ) (A : ℝ) :
  Finset.card S = 10 →
  (Finset.sum S id + 8) / 10 = 7 →
  Finset.sum S id = 10 * A →
  A = 6.2 := by
  sorry

end original_average_calculation_l1748_174869


namespace james_annual_training_hours_l1748_174807

/-- Calculates the annual training hours for an athlete with a specific schedule -/
def annualTrainingHours (sessionsPerDay : ℕ) (hoursPerSession : ℕ) (daysPerWeek : ℕ) : ℕ :=
  sessionsPerDay * hoursPerSession * daysPerWeek * 52

/-- Proves that James' annual training hours equal 2080 -/
theorem james_annual_training_hours :
  annualTrainingHours 2 4 5 = 2080 := by
  sorry

#eval annualTrainingHours 2 4 5

end james_annual_training_hours_l1748_174807


namespace ellipse_theorem_l1748_174868

/-- Ellipse C with given properties -/
structure Ellipse :=
  (center : ℝ × ℝ)
  (major_axis : ℝ)
  (point_on_ellipse : ℝ × ℝ)
  (h_center : center = (0, 0))
  (h_major_axis : major_axis = 4)
  (h_point : point_on_ellipse = (1, Real.sqrt 3 / 2))

/-- Line with slope 1/2 passing through a point -/
structure Line (P : ℝ × ℝ) :=
  (slope : ℝ)
  (h_slope : slope = 1/2)

/-- Theorem about the ellipse C and intersecting lines -/
theorem ellipse_theorem (C : Ellipse) :
  (∃ (eq : ℝ × ℝ → Prop), ∀ (x y : ℝ), eq (x, y) ↔ x^2/4 + y^2 = 1) ∧
  (∀ (P : ℝ × ℝ), P.2 = 0 → P.1 ∈ Set.Icc (-2 : ℝ) 2 →
    ∀ (l : Line P) (A B : ℝ × ℝ),
      (∃ (t : ℝ), A = (t, (t - P.1)/2) ∧ A.1^2/4 + A.2^2 = 1) →
      (∃ (t : ℝ), B = (t, (t - P.1)/2) ∧ B.1^2/4 + B.2^2 = 1) →
      (A.1 - P.1)^2 + A.2^2 + (B.1 - P.1)^2 + B.2^2 = 5) :=
by sorry

end ellipse_theorem_l1748_174868


namespace triangle_angle_measure_l1748_174893

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  A = Real.pi / 6 := by sorry

end triangle_angle_measure_l1748_174893


namespace average_problem_l1748_174878

theorem average_problem (y : ℝ) : (15 + 25 + y) / 3 = 23 → y = 29 := by
  sorry

end average_problem_l1748_174878


namespace one_third_of_seven_times_nine_l1748_174812

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l1748_174812


namespace stationery_ratio_is_three_to_one_l1748_174877

/-- The number of pieces of stationery Georgia has -/
def georgia_stationery : ℕ := 25

/-- The number of pieces of stationery Lorene has -/
def lorene_stationery : ℕ := georgia_stationery + 50

/-- The ratio of Lorene's stationery to Georgia's stationery -/
def stationery_ratio : ℚ := lorene_stationery / georgia_stationery

theorem stationery_ratio_is_three_to_one :
  stationery_ratio = 3 / 1 := by
  sorry

end stationery_ratio_is_three_to_one_l1748_174877


namespace prob_C_gets_10000_equal_expected_values_l1748_174848

/-- Represents the bonus distribution problem for a work group of three people. -/
structure BonusDistribution where
  total_bonus : ℝ
  p₁ : ℝ  -- Probability of taking 10,000 yuan
  p₂ : ℝ  -- Probability of taking 20,000 yuan

/-- The total bonus is 40,000 yuan -/
def bonus_amount : ℝ := 40000

/-- The probability of A or B taking 10,000 yuan plus the probability of taking 20,000 yuan equals 1 -/
axiom prob_sum (bd : BonusDistribution) : bd.p₁ + bd.p₂ = 1

/-- Expected bonus for A or B -/
def expected_bonus_AB (bd : BonusDistribution) : ℝ := 10000 * bd.p₁ + 20000 * bd.p₂

/-- Expected bonus for C -/
def expected_bonus_C (bd : BonusDistribution) : ℝ := 
  20000 * bd.p₁^2 + 10000 * 2 * bd.p₁ * bd.p₂

/-- Theorem: When p₁ = p₂ = 1/2, the probability that C gets 10,000 yuan is 1/2 -/
theorem prob_C_gets_10000 (bd : BonusDistribution) 
  (h₁ : bd.p₁ = 1/2) (h₂ : bd.p₂ = 1/2) : 
  bd.p₁ * bd.p₂ + bd.p₁ * bd.p₂ = 1/2 := by sorry

/-- Theorem: When expected values are equal, p₁ = 2/3 and p₂ = 1/3 -/
theorem equal_expected_values (bd : BonusDistribution) 
  (h : expected_bonus_AB bd = expected_bonus_C bd) : 
  bd.p₁ = 2/3 ∧ bd.p₂ = 1/3 := by sorry

end prob_C_gets_10000_equal_expected_values_l1748_174848


namespace original_average_proof_l1748_174835

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℚ) : 
  n = 10 → new_avg = 160 → new_avg = 2 * original_avg → original_avg = 80 := by
  sorry

end original_average_proof_l1748_174835


namespace equal_squares_of_equal_products_l1748_174845

theorem equal_squares_of_equal_products (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * (b + c + d) = b * (a + c + d))
  (h2 : a * (b + c + d) = c * (a + b + d))
  (h3 : a * (b + c + d) = d * (a + b + c)) :
  a^2 = b^2 ∧ a^2 = c^2 ∧ a^2 = d^2 := by
sorry

end equal_squares_of_equal_products_l1748_174845


namespace smallest_winning_number_sum_of_digits_34_l1748_174897

def game_condition (M : ℕ) : Prop :=
  M ≤ 1999 ∧
  3 * M < 2000 ∧
  3 * M + 80 < 2000 ∧
  3 * (3 * M + 80) < 2000 ∧
  3 * (3 * M + 80) + 80 < 2000 ∧
  3 * (3 * (3 * M + 80) + 80) ≥ 2000

theorem smallest_winning_number :
  ∀ n : ℕ, n < 34 → ¬(game_condition n) ∧ game_condition 34 :=
by sorry

theorem sum_of_digits_34 : (3 : ℕ) + 4 = 7 :=
by sorry

end smallest_winning_number_sum_of_digits_34_l1748_174897


namespace final_sum_after_transformation_l1748_174895

theorem final_sum_after_transformation (a b c S : ℝ) (h : a + b + c = S) :
  3 * (a - 4) + 3 * (b - 4) + 3 * (c - 4) = 3 * S - 36 := by
  sorry

end final_sum_after_transformation_l1748_174895


namespace ellipse_standard_equation_l1748_174838

/-- An ellipse with one focus at (0,1) and eccentricity 1/2 has the standard equation x²/3 + y²/4 = 1 -/
theorem ellipse_standard_equation (x y : ℝ) : 
  let e : ℝ := 1/2
  let f : ℝ × ℝ := (0, 1)
  x^2/3 + y^2/4 = 1 ↔ 
    ∃ (a b c : ℝ), 
      a > 0 ∧ b > 0 ∧
      c = 1 ∧
      e = c/a ∧
      a^2 = b^2 + c^2 ∧
      x^2/a^2 + y^2/b^2 = 1 :=
by sorry


end ellipse_standard_equation_l1748_174838


namespace trajectory_of_moving_circle_l1748_174886

/-- The trajectory of the center of a moving circle -/
def trajectory_equation (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 7)^2 = 25

/-- The equation of the stationary circle -/
def stationary_circle (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 7)^2 = 16

/-- The radius of the moving circle -/
def moving_circle_radius : ℝ := 1

theorem trajectory_of_moving_circle :
  ∀ x y : ℝ,
  (∃ x₀ y₀ : ℝ, stationary_circle x₀ y₀ ∧ 
    ((x - x₀)^2 + (y - y₀)^2 = (moving_circle_radius + 4)^2)) →
  trajectory_equation x y :=
by sorry

end trajectory_of_moving_circle_l1748_174886


namespace circle_triangle_area_relation_l1748_174891

theorem circle_triangle_area_relation :
  ∀ (A B C : ℝ),
  -- The triangle has sides 20, 21, and 29
  20^2 + 21^2 = 29^2 →
  -- A circle is circumscribed about the triangle
  -- A, B, and C are areas of non-triangular regions
  -- C is the largest area
  C ≥ A ∧ C ≥ B →
  -- The area of the triangle is 210
  (20 * 21) / 2 = 210 →
  -- The diameter of the circle is 29
  -- C is half the area of the circle
  C = (29^2 * π) / 8 →
  -- Prove that A + B + 210 = C
  A + B + 210 = C :=
by
  sorry

end circle_triangle_area_relation_l1748_174891


namespace unique_solution_l1748_174873

/-- A polynomial that satisfies the given functional equation -/
def functional_equation (p : ℝ → ℝ) : Prop :=
  ∀ x, p (p x) = 2 * x * p x + 3 * x^2

/-- The theorem stating that p(x) = 3x is the unique solution to the functional equation -/
theorem unique_solution :
  ∃! p : ℝ → ℝ, functional_equation p ∧ ∀ x, p x = 3 * x :=
sorry

end unique_solution_l1748_174873


namespace power_of_power_l1748_174854

theorem power_of_power : (3^2)^4 = 6561 := by sorry

end power_of_power_l1748_174854


namespace cylinder_height_ratio_l1748_174888

theorem cylinder_height_ratio (h : ℝ) (h_pos : h > 0) : 
  ∃ (H : ℝ), H = (14 / 15) * h ∧ 
  (7 / 8) * π * h = (3 / 5) * π * ((5 / 4) ^ 2) * H :=
by sorry

end cylinder_height_ratio_l1748_174888


namespace sufficient_not_necessary_condition_l1748_174880

theorem sufficient_not_necessary_condition (x : ℝ) :
  {x | 1 / x > 1} ⊂ {x | Real.exp (x - 1) < 1} ∧ {x | 1 / x > 1} ≠ {x | Real.exp (x - 1) < 1} :=
by sorry

end sufficient_not_necessary_condition_l1748_174880


namespace distance_between_first_two_points_l1748_174840

theorem distance_between_first_two_points
  (n : ℕ)
  (sum_first : ℝ)
  (sum_second : ℝ)
  (h_n : n = 11)
  (h_sum_first : sum_first = 2018)
  (h_sum_second : sum_second = 2000) :
  ∃ (x : ℝ),
    x = 2 ∧
    x * (n - 2) = sum_first - sum_second :=
by sorry

end distance_between_first_two_points_l1748_174840


namespace necessary_condition_range_l1748_174863

theorem necessary_condition_range (a : ℝ) : 
  (∀ x : ℝ, x < a + 2 → x ≤ 2) → a ≤ 0 := by
  sorry

end necessary_condition_range_l1748_174863


namespace bank_through_window_l1748_174832

/-- Represents a letter as seen through a clear glass window from the inside --/
inductive MirroredLetter
  | Normal (c : Char)
  | Inverted (c : Char)

/-- Represents a word as seen through a clear glass window from the inside --/
def MirroredWord := List MirroredLetter

/-- Converts a character to its mirrored version --/
def mirrorChar (c : Char) : MirroredLetter :=
  match c with
  | 'B' => MirroredLetter.Inverted 'В'
  | 'A' => MirroredLetter.Normal 'A'
  | 'N' => MirroredLetter.Inverted 'И'
  | 'K' => MirroredLetter.Inverted 'И'
  | _ => MirroredLetter.Normal c

/-- Converts a string to its mirrored version --/
def mirrorWord (s : String) : MirroredWord :=
  s.toList.reverse.map mirrorChar

/-- Converts a MirroredWord to a string --/
def mirroredWordToString (w : MirroredWord) : String :=
  w.map (fun l => match l with
    | MirroredLetter.Normal c => c
    | MirroredLetter.Inverted c => c
  ) |>.asString

theorem bank_through_window :
  mirroredWordToString (mirrorWord "BANK") = "ИAИВ" := by
  sorry

#eval mirroredWordToString (mirrorWord "BANK")

end bank_through_window_l1748_174832


namespace range_of_a_l1748_174898

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 4*x + a = 0

theorem range_of_a (a : ℝ) (hp : prop_p a) (hq : prop_q a) :
  a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end range_of_a_l1748_174898


namespace seating_arrangements_count_l1748_174818

/-- Represents a seating arrangement of adults and children -/
def SeatingArrangement := Fin 6 → Bool

/-- Checks if a seating arrangement is valid (no two children sit next to each other) -/
def is_valid (arrangement : SeatingArrangement) : Prop :=
  ∀ i : Fin 6, arrangement i → arrangement ((i + 1) % 6) → False

/-- The number of valid seating arrangements -/
def num_valid_arrangements : ℕ := sorry

/-- The main theorem: there are 72 valid seating arrangements -/
theorem seating_arrangements_count :
  num_valid_arrangements = 72 := by sorry

end seating_arrangements_count_l1748_174818


namespace inequality_range_l1748_174899

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 - m*x - 1) < 0) ↔ 
  (-4 < m ∧ m ≤ 0) :=
sorry

end inequality_range_l1748_174899


namespace same_heads_probability_l1748_174800

def fair_coin_prob : ℚ := 1/2
def coin_prob_1 : ℚ := 3/5
def coin_prob_2 : ℚ := 2/3

def same_heads_prob : ℚ := 29/90

theorem same_heads_probability :
  let outcomes := (1 + 1) * (2 + 3) * (1 + 2)
  let squared_sum := (2^2 + 9^2 + 13^2 + 6^2 : ℚ)
  same_heads_prob = squared_sum / (outcomes^2 : ℚ) := by
sorry

end same_heads_probability_l1748_174800


namespace molecular_weight_one_mole_l1748_174861

/-- The molecular weight of Aluminium hydroxide for a given number of moles. -/
def molecular_weight (moles : ℝ) : ℝ := sorry

/-- The number of moles for which we know the molecular weight. -/
def known_moles : ℝ := 4

/-- The known molecular weight for the given number of moles. -/
def known_weight : ℝ := 312

/-- Theorem stating that the molecular weight of one mole of Aluminium hydroxide is 78 g/mol. -/
theorem molecular_weight_one_mole :
  molecular_weight 1 = 78 :=
sorry

end molecular_weight_one_mole_l1748_174861


namespace shop_owner_profit_l1748_174824

/-- Represents the profit calculation for a shop owner using false weights -/
theorem shop_owner_profit (buying_cheat : ℝ) (selling_cheat : ℝ) : 
  buying_cheat = 0.12 →
  selling_cheat = 0.30 →
  let actual_buy_amount := 1 + buying_cheat
  let actual_sell_amount := 1 - selling_cheat
  let sell_portions := actual_buy_amount / actual_sell_amount
  let revenue := sell_portions * 100
  let profit := revenue - 100
  let percentage_profit := (profit / 100) * 100
  percentage_profit = 60 := by
sorry


end shop_owner_profit_l1748_174824


namespace min_value_expression_l1748_174808

theorem min_value_expression (a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 2) 
  (heq : 2 * a + b - 6 = 0) : 
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 1 → y > 2 → 2 * x + y - 6 = 0 → 
    1 / (x - 1) + 2 / (y - 2) ≥ min :=
by sorry

end min_value_expression_l1748_174808


namespace train_length_l1748_174833

/-- Given a train traveling at 72 km/hr that crosses a pole in 9 seconds, prove its length is 180 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 72 → time = 9 → speed * time * (1000 / 3600) = 180 :=
by sorry

end train_length_l1748_174833


namespace multiple_choice_questions_l1748_174806

theorem multiple_choice_questions (total : ℕ) (problem_solving_percent : ℚ) 
  (h1 : total = 50)
  (h2 : problem_solving_percent = 80 / 100) :
  (total : ℚ) * (1 - problem_solving_percent) = 10 := by
  sorry

end multiple_choice_questions_l1748_174806


namespace largest_prime_factor_57_largest_prime_factor_57_is_19_l1748_174872

def numbers : List Nat := [57, 75, 91, 143, 169]

def largest_prime_factor (n : Nat) : Nat :=
  sorry

theorem largest_prime_factor_57 :
  ∀ n ∈ numbers, n ≠ 57 → largest_prime_factor n < largest_prime_factor 57 :=
  sorry

theorem largest_prime_factor_57_is_19 :
  largest_prime_factor 57 = 19 :=
  sorry

end largest_prime_factor_57_largest_prime_factor_57_is_19_l1748_174872


namespace charity_raffle_proof_l1748_174826

/-- Calculates the total money raised from a charity raffle and donations. -/
def total_money_raised (num_tickets : ℕ) (ticket_price : ℚ) (donation1 : ℚ) (num_donation1 : ℕ) (donation2 : ℚ) : ℚ :=
  (num_tickets : ℚ) * ticket_price + (num_donation1 : ℚ) * donation1 + donation2

/-- Proves that the total money raised is $100.00 given the specific conditions. -/
theorem charity_raffle_proof :
  let num_tickets : ℕ := 25
  let ticket_price : ℚ := 2
  let donation1 : ℚ := 15
  let num_donation1 : ℕ := 2
  let donation2 : ℚ := 20
  total_money_raised num_tickets ticket_price donation1 num_donation1 donation2 = 100 :=
by
  sorry


end charity_raffle_proof_l1748_174826


namespace yellow_red_ball_arrangements_l1748_174822

theorem yellow_red_ball_arrangements :
  let total_balls : ℕ := 7
  let yellow_balls : ℕ := 4
  let red_balls : ℕ := 3
  Nat.choose total_balls yellow_balls = 35 := by sorry

end yellow_red_ball_arrangements_l1748_174822


namespace arctan_sum_equals_arctan_29_22_l1748_174839

theorem arctan_sum_equals_arctan_29_22 (a b : ℝ) : 
  a = 3/4 → (a + 1) * (b + 1) = 9/4 → Real.arctan a + Real.arctan b = Real.arctan (29/22) := by
  sorry

end arctan_sum_equals_arctan_29_22_l1748_174839


namespace crazy_silly_school_movies_l1748_174866

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 9

/-- The number of different books in the series -/
def num_books : ℕ := 10

/-- The number of books read -/
def books_read : ℕ := 14

theorem crazy_silly_school_movies :
  (books_read = num_movies + 5) →
  (books_read ≤ num_books) →
  (num_movies = 9) := by
  sorry

end crazy_silly_school_movies_l1748_174866


namespace binomial_10_3_l1748_174870

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l1748_174870


namespace fish_catch_difference_l1748_174841

/-- Given the number of fish caught by various birds and a fisherman, prove the difference in catch between the fisherman and pelican. -/
theorem fish_catch_difference (pelican kingfisher osprey fisherman : ℕ) : 
  pelican = 13 →
  kingfisher = pelican + 7 →
  osprey = 2 * kingfisher →
  fisherman = 4 * (pelican + kingfisher + osprey) →
  fisherman - pelican = 279 := by
  sorry

end fish_catch_difference_l1748_174841


namespace sum_of_digits_A_squared_l1748_174801

/-- For a number with n digits, all being 9 -/
def A (n : ℕ) : ℕ := 10^n - 1

/-- Sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sum_of_digits (m / 10)

theorem sum_of_digits_A_squared :
  sum_of_digits ((A 221)^2) = 1989 := by
  sorry

end sum_of_digits_A_squared_l1748_174801
