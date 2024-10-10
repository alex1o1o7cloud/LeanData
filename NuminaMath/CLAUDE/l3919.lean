import Mathlib

namespace root_sum_equation_l3919_391997

theorem root_sum_equation (a b : ℝ) : 
  (Complex.I + 1) ^ 2 * a + (Complex.I + 1) * b + 2 = 0 → a + b = -1 := by
  sorry

end root_sum_equation_l3919_391997


namespace die_roll_events_l3919_391929

-- Define the sample space for a six-sided die roll
def Ω : Type := Fin 6

-- Define the events A_k
def A (k : Fin 6) : Set Ω := {ω : Ω | ω.val + 1 = k.val + 1}

-- Define event A: rolling an even number of points
def event_A : Set Ω := A 1 ∪ A 3 ∪ A 5

-- Define event B: rolling an odd number of points
def event_B : Set Ω := A 0 ∪ A 2 ∪ A 4

-- Define event C: rolling a multiple of three
def event_C : Set Ω := A 2 ∪ A 5

-- Define event D: rolling a number greater than three
def event_D : Set Ω := A 3 ∪ A 4 ∪ A 5

theorem die_roll_events :
  (event_A = A 1 ∪ A 3 ∪ A 5) ∧
  (event_B = A 0 ∪ A 2 ∪ A 4) ∧
  (event_C = A 2 ∪ A 5) ∧
  (event_D = A 3 ∪ A 4 ∪ A 5) := by sorry

end die_roll_events_l3919_391929


namespace x_zero_necessary_not_sufficient_l3919_391937

def a : ℝ × ℝ := (1, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem x_zero_necessary_not_sufficient :
  ∃ (x : ℝ), x ≠ 0 ∧ (a + b x) • (b x) = 0 ∧
  ∀ (y : ℝ), (a + b y) • (b y) = 0 → y = 0 ∨ y = -1 :=
by sorry

end x_zero_necessary_not_sufficient_l3919_391937


namespace arctan_sum_of_cubic_roots_l3919_391956

theorem arctan_sum_of_cubic_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 10*x₁ + 11 = 0 →
  x₂^3 - 10*x₂ + 11 = 0 →
  x₃^3 - 10*x₃ + 11 = 0 →
  -5 < x₁ ∧ x₁ < 5 →
  -5 < x₂ ∧ x₂ < 5 →
  -5 < x₃ ∧ x₃ < 5 →
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π/4 := by
sorry

end arctan_sum_of_cubic_roots_l3919_391956


namespace perpendicular_length_l3919_391973

/-- Given two oblique lines and their projections, find the perpendicular length -/
theorem perpendicular_length
  (oblique1 oblique2 : ℝ)
  (projection_ratio : ℚ)
  (h1 : oblique1 = 41)
  (h2 : oblique2 = 50)
  (h3 : projection_ratio = 3 / 10) :
  ∃ (perpendicular : ℝ), perpendicular = 40 :=
by
  sorry

end perpendicular_length_l3919_391973


namespace expression_value_l3919_391931

theorem expression_value (α : Real) (h : Real.tan α = -3/4) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11*π/12 - α) * Real.sin (9*π/2 + α)) = 3/4 := by
  sorry

end expression_value_l3919_391931


namespace remainder_theorem_l3919_391944

-- Define the polynomial Q
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom Q_at_20 : Q 20 = 100
axiom Q_at_100 : Q 100 = 20

-- Define the remainder function
def remainder (f : ℝ → ℝ) (x : ℝ) : ℝ := -x + 120

-- State the theorem
theorem remainder_theorem :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 20) * (x - 100) * R x + remainder Q x :=
sorry

end remainder_theorem_l3919_391944


namespace nine_seats_six_people_arrangement_l3919_391984

/-- The number of ways to arrange people and empty seats in a row -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  (Nat.factorial people) * (Nat.choose (people - 1) (total_seats - people))

/-- Theorem: There are 7200 ways to arrange 6 people and 3 empty seats in a row of 9 seats,
    where every empty seat is flanked by people on both sides -/
theorem nine_seats_six_people_arrangement :
  seating_arrangements 9 6 = 7200 := by
  sorry

end nine_seats_six_people_arrangement_l3919_391984


namespace largest_band_size_l3919_391930

theorem largest_band_size :
  ∀ m r x : ℕ,
  m = r * x + 3 →
  m = (r - 1) * (x + 2) →
  m < 100 →
  ∃ m_max : ℕ,
  m_max = 69 ∧
  ∀ m' : ℕ,
  (∃ r' x' : ℕ, m' = r' * x' + 3 ∧ m' = (r' - 1) * (x' + 2) ∧ m' < 100) →
  m' ≤ m_max :=
sorry

end largest_band_size_l3919_391930


namespace digit_sum_puzzle_l3919_391994

def digit_set : Finset Nat := {0, 2, 3, 4, 5, 7, 8, 9}

theorem digit_sum_puzzle :
  ∃ (a b c d e f : Nat),
    a ∈ digit_set ∧ b ∈ digit_set ∧ c ∈ digit_set ∧
    d ∈ digit_set ∧ e ∈ digit_set ∧ f ∈ digit_set ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a + b + c = 24 ∧
    b + d + e + f = 14 ∧
    a + b + c + d + e + f = 31 :=
by sorry

end digit_sum_puzzle_l3919_391994


namespace m_plus_2n_equals_neg_one_l3919_391998

theorem m_plus_2n_equals_neg_one (m n : ℝ) (h : |m - 3| + (n + 2)^2 = 0) : m + 2*n = -1 := by
  sorry

end m_plus_2n_equals_neg_one_l3919_391998


namespace wendy_trip_miles_l3919_391932

theorem wendy_trip_miles : 
  let day1_miles : ℕ := 125
  let day2_miles : ℕ := 223
  let day3_miles : ℕ := 145
  day1_miles + day2_miles + day3_miles = 493 := by sorry

end wendy_trip_miles_l3919_391932


namespace cookie_difference_l3919_391952

/-- Proves that Cristian had 50 more black cookies than white cookies initially -/
theorem cookie_difference (black_cookies white_cookies : ℕ) : 
  white_cookies = 80 →
  black_cookies > white_cookies →
  black_cookies / 2 + white_cookies / 4 = 85 →
  black_cookies - white_cookies = 50 :=
by
  sorry

#check cookie_difference

end cookie_difference_l3919_391952


namespace congested_sections_probability_l3919_391934

/-- The probability of selecting exactly 4 congested sections out of 10 randomly selected sections,
    given that there are 7 congested sections out of 16 total sections. -/
theorem congested_sections_probability :
  let total_sections : ℕ := 16
  let congested_sections : ℕ := 7
  let selected_sections : ℕ := 10
  let target_congested : ℕ := 4
  
  (Nat.choose congested_sections target_congested *
   Nat.choose (total_sections - congested_sections) (selected_sections - target_congested)) /
  Nat.choose total_sections selected_sections =
  (Nat.choose congested_sections target_congested *
   Nat.choose (total_sections - congested_sections) (selected_sections - target_congested)) /
  Nat.choose total_sections selected_sections :=
by
  sorry

end congested_sections_probability_l3919_391934


namespace ravenswood_remaining_gnomes_l3919_391941

/-- The number of gnomes in Westerville woods -/
def westerville_gnomes : ℕ := 20

/-- The ratio of gnomes in Ravenswood forest compared to Westerville woods -/
def ravenswood_ratio : ℕ := 4

/-- The percentage of gnomes taken from Ravenswood forest -/
def taken_percentage : ℚ := 40 / 100

/-- The number of gnomes remaining in Ravenswood forest after some are taken -/
def remaining_gnomes : ℕ := 48

theorem ravenswood_remaining_gnomes :
  (ravenswood_ratio * westerville_gnomes : ℚ) * (1 - taken_percentage) = remaining_gnomes := by
  sorry

end ravenswood_remaining_gnomes_l3919_391941


namespace division_with_remainder_l3919_391902

theorem division_with_remainder (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = divisor * quotient + remainder →
  remainder < divisor →
  dividend = 11 →
  divisor = 3 →
  remainder = 2 →
  quotient = 3 := by
sorry

end division_with_remainder_l3919_391902


namespace probability_sum_six_l3919_391915

def cards : Finset ℕ := {1, 2, 3, 4, 5, 6}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)}

def total_outcomes : Finset (ℕ × ℕ) :=
  cards.product cards

theorem probability_sum_six :
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 5 / 36 := by
  sorry

end probability_sum_six_l3919_391915


namespace inverse_value_equivalence_l3919_391943

-- Define the function f
def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9

-- Theorem stating that finding f⁻¹(-3.5) is equivalent to solving 7x³ - 2x² + 5x - 5.5 = 0
theorem inverse_value_equivalence :
  ∀ x : ℝ, f x = -3.5 ↔ 7 * x^3 - 2 * x^2 + 5 * x - 5.5 = 0 :=
by
  sorry

-- Note: The actual inverse function is not defined as it's not expressible in elementary functions

end inverse_value_equivalence_l3919_391943


namespace playground_area_l3919_391933

/-- 
A rectangular playground has a perimeter of 100 meters and its length is twice its width. 
This theorem proves that the area of such a playground is 5000/9 square meters.
-/
theorem playground_area (width : ℝ) (length : ℝ) : 
  (2 * length + 2 * width = 100) →  -- Perimeter condition
  (length = 2 * width) →            -- Length-width relation
  (length * width = 5000 / 9) :=    -- Area calculation
by sorry

end playground_area_l3919_391933


namespace permutations_of_seven_distinct_objects_l3919_391912

theorem permutations_of_seven_distinct_objects : Nat.factorial 7 = 5040 := by
  sorry

end permutations_of_seven_distinct_objects_l3919_391912


namespace A_intersect_B_eq_expected_result_l3919_391916

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | (1 - x) / x < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the expected result
def expected_result : Set ℝ := {x | -1 < x ∧ x < 0} ∪ {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem A_intersect_B_eq_expected_result : A_intersect_B = expected_result := by
  sorry

end A_intersect_B_eq_expected_result_l3919_391916


namespace lava_lamp_probability_l3919_391947

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 2
def num_lamps_on : ℕ := 3

def total_arrangements : ℕ := (Nat.choose (num_red_lamps + num_blue_lamps) num_blue_lamps) * 
                               (Nat.choose (num_red_lamps + num_blue_lamps) num_lamps_on)

def constrained_arrangements : ℕ := (Nat.choose (num_red_lamps + num_blue_lamps - 1) (num_blue_lamps - 1)) * 
                                    (Nat.choose (num_red_lamps + num_blue_lamps - 2) (num_lamps_on - 1))

theorem lava_lamp_probability : 
  (constrained_arrangements : ℚ) / total_arrangements = 1 / 10 := by sorry

end lava_lamp_probability_l3919_391947


namespace jacks_walking_speed_l3919_391978

/-- The problem of determining Jack's walking speed -/
theorem jacks_walking_speed
  (initial_distance : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (lindy_distance : ℝ)
  (h1 : initial_distance = 270)
  (h2 : christina_speed = 5)
  (h3 : lindy_speed = 8)
  (h4 : lindy_distance = 240) :
  ∃ (jack_speed : ℝ),
    jack_speed = 4 ∧
    jack_speed * (lindy_distance / lindy_speed) +
    christina_speed * (lindy_distance / lindy_speed) =
    initial_distance :=
by sorry

end jacks_walking_speed_l3919_391978


namespace problem_statement_l3919_391913

theorem problem_statement (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : Real.exp a + a = Real.log (b * Real.exp b) ∧ Real.log (b * Real.exp b) = 2) :
  (b * Real.exp b = Real.exp 2) ∧
  (a + b = 2) ∧
  (Real.exp a + Real.log b = 2) := by
  sorry

end problem_statement_l3919_391913


namespace arithmetic_series_sum_l3919_391985

theorem arithmetic_series_sum : 
  ∀ (a₁ aₙ d : ℚ) (n : ℕ),
    a₁ = 25 →
    aₙ = 50 →
    d = 2/5 →
    aₙ = a₁ + (n - 1) * d →
    (n : ℚ) * (a₁ + aₙ) / 2 = 2400 :=
by
  sorry

end arithmetic_series_sum_l3919_391985


namespace fraction_of_married_women_l3919_391924

theorem fraction_of_married_women (total : ℕ) (total_pos : total > 0) :
  let women := (58 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  (married_women / women) = 23 / 29 := by
  sorry

end fraction_of_married_women_l3919_391924


namespace hyperbola_properties_l3919_391925

-- Define the hyperbola
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passing_point : ℝ × ℝ

-- Define the point M
structure PointM where
  on_right_branch : Bool
  dot_product_zero : Bool

-- Theorem statement
theorem hyperbola_properties (h : Hyperbola) (m : PointM) 
    (h_center : h.center = (0, 0))
    (h_foci : h.foci_on_x_axis = true)
    (h_eccentricity : h.eccentricity = Real.sqrt 2)
    (h_passing_point : h.passing_point = (4, -2 * Real.sqrt 2))
    (h_m_right : m.on_right_branch = true)
    (h_m_dot : m.dot_product_zero = true) :
    (∃ (x y : ℝ), x^2 - y^2 = 8) ∧ 
    (∃ (area : ℝ), area = 8) := by
  sorry

end hyperbola_properties_l3919_391925


namespace cubic_root_sum_squares_l3919_391963

/-- Given that a, b, and c are the roots of x^3 - 3x - 2 = 0,
    prove that a(b+c)^2 + b(c+a)^2 + c(a+b)^2 = -6 -/
theorem cubic_root_sum_squares (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 3*x - 2 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a*(b+c)^2 + b*(c+a)^2 + c*(a+b)^2 = -6 := by
sorry

end cubic_root_sum_squares_l3919_391963


namespace product_equals_square_l3919_391955

theorem product_equals_square : 
  1000 * 2.998 * 2.998 * 100 = (29980 : ℝ)^2 := by sorry

end product_equals_square_l3919_391955


namespace cookie_production_l3919_391954

def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def efficiency_improvement : ℚ := 1/10
def new_flour : ℕ := 4

def improved_cookies : ℕ := 35

theorem cookie_production : 
  let initial_efficiency : ℚ := initial_cookies / initial_flour
  let improved_efficiency : ℚ := initial_efficiency * (1 + efficiency_improvement)
  let theoretical_cookies : ℚ := improved_efficiency * new_flour
  ⌊theoretical_cookies⌋ = improved_cookies := by sorry

end cookie_production_l3919_391954


namespace non_union_women_percentage_is_75_l3919_391923

/-- Represents the composition of employees in a company -/
structure CompanyEmployees where
  total : ℕ
  men : ℕ
  unionized : ℕ
  unionized_men : ℕ

/-- The percentage of non-union employees who are women -/
def non_union_women_percentage (c : CompanyEmployees) : ℚ :=
  let non_union := c.total - c.unionized
  let non_union_men := c.men - c.unionized_men
  let non_union_women := non_union - non_union_men
  (non_union_women : ℚ) / non_union * 100

/-- Theorem stating the percentage of non-union women employees -/
theorem non_union_women_percentage_is_75 (c : CompanyEmployees) : 
  c.total > 0 →
  c.men = (52 * c.total) / 100 →
  c.unionized = (60 * c.total) / 100 →
  c.unionized_men = (70 * c.unionized) / 100 →
  non_union_women_percentage c = 75 := by
sorry

end non_union_women_percentage_is_75_l3919_391923


namespace probability_is_24_1107_l3919_391939

/-- Represents a 5x5x5 cube with one face painted red and an internal diagonal painted green -/
structure PaintedCube where
  size : Nat
  size_eq : size = 5

/-- The number of unit cubes with exactly three painted faces -/
def three_painted_faces (cube : PaintedCube) : Nat := 8

/-- The number of unit cubes with exactly one painted face -/
def one_painted_face (cube : PaintedCube) : Nat := 21

/-- The total number of unit cubes in the larger cube -/
def total_cubes (cube : PaintedCube) : Nat := cube.size ^ 3

/-- The probability of selecting one cube with exactly three painted faces
    and one cube with exactly one painted face when choosing two cubes uniformly at random -/
def probability (cube : PaintedCube) : Rat :=
  (three_painted_faces cube * one_painted_face cube : Rat) / (total_cubes cube).choose 2

/-- The main theorem stating the probability is 24/1107 -/
theorem probability_is_24_1107 (cube : PaintedCube) : probability cube = 24 / 1107 := by
  sorry

end probability_is_24_1107_l3919_391939


namespace evening_campers_l3919_391918

theorem evening_campers (afternoon_campers : ℕ) (difference : ℕ) : 
  afternoon_campers = 34 → difference = 24 → afternoon_campers - difference = 10 := by
  sorry

end evening_campers_l3919_391918


namespace no_simultaneous_squares_l3919_391999

theorem no_simultaneous_squares (x y : ℕ) : ¬(∃ (a b : ℕ), x^2 + y = a^2 ∧ y^2 + x = b^2) := by
  sorry

end no_simultaneous_squares_l3919_391999


namespace school_committee_formation_l3919_391953

theorem school_committee_formation (n_children : ℕ) (n_teachers : ℕ) (committee_size : ℕ) :
  n_children = 12 →
  n_teachers = 3 →
  committee_size = 9 →
  (Nat.choose (n_children + n_teachers) committee_size) - (Nat.choose n_children committee_size) = 4785 :=
by sorry

end school_committee_formation_l3919_391953


namespace sqrt_real_condition_l3919_391911

theorem sqrt_real_condition (x : ℝ) : (∃ y : ℝ, y ^ 2 = (x - 1) / 9) ↔ x ≥ 1 := by sorry

end sqrt_real_condition_l3919_391911


namespace simplify_and_evaluate_l3919_391979

/-- Given a = 2 and b = -1/2, prove that a - 2(a - b^2) + 3(-a + b^2) = -27/4 -/
theorem simplify_and_evaluate (a b : ℚ) (ha : a = 2) (hb : b = -1/2) :
  a - 2 * (a - b^2) + 3 * (-a + b^2) = -27/4 := by
  sorry

end simplify_and_evaluate_l3919_391979


namespace min_value_theorem_l3919_391904

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 9 * Real.sqrt 5 / 5 := by
  sorry

end min_value_theorem_l3919_391904


namespace inequality_always_holds_l3919_391986

theorem inequality_always_holds (a b c : ℝ) (h : b > c) : a^2 + b > a^2 + c := by
  sorry

end inequality_always_holds_l3919_391986


namespace original_not_imply_converse_converse_implies_negation_l3919_391980

-- Define a proposition P and Q
variable (P Q : Prop)

-- Statement 1: The truth of an original statement does not necessarily imply the truth of its converse
theorem original_not_imply_converse : ∃ P Q, (P → Q) ∧ ¬(Q → P) := by sorry

-- Statement 2: If the converse of a statement is true, then its negation is also true
theorem converse_implies_negation : ∀ P Q, (Q → P) → (¬P → ¬Q) := by sorry

end original_not_imply_converse_converse_implies_negation_l3919_391980


namespace circle_equation_l3919_391975

theorem circle_equation (center : ℝ × ℝ) (p1 p2 : ℝ × ℝ) :
  center.1 + center.2 = 0 →
  p1 = (0, 2) →
  p2 = (-4, 0) →
  ∀ x y : ℝ, ((x - center.1)^2 + (y - center.2)^2 = (p1.1 - center.1)^2 + (p1.2 - center.2)^2) ↔
              ((x - center.1)^2 + (y - center.2)^2 = (p2.1 - center.1)^2 + (p2.2 - center.2)^2) →
  ∃ a b r : ℝ, (x + a)^2 + (y - b)^2 = r ∧ a = 3 ∧ b = 3 ∧ r = 10 :=
sorry

end circle_equation_l3919_391975


namespace fraction_simplification_l3919_391957

theorem fraction_simplification :
  (3 / 7 - 2 / 9) / (5 / 12 + 1 / 4) = 13 / 42 := by
  sorry

end fraction_simplification_l3919_391957


namespace constant_solution_implies_product_l3919_391959

/-- 
Given constants a and b, if the equation (2kx+a)/3 = 2 + (x-bk)/6 
always has a solution of x = 1 for any k, then ab = -26
-/
theorem constant_solution_implies_product (a b : ℚ) : 
  (∀ k : ℚ, ∃ x : ℚ, x = 1 ∧ (2*k*x + a) / 3 = 2 + (x - b*k) / 6) → 
  a * b = -26 := by
sorry

end constant_solution_implies_product_l3919_391959


namespace root_of_quadratic_l3919_391971

theorem root_of_quadratic (x v : ℝ) : 
  x = (-15 - Real.sqrt 409) / 12 →
  v = -23 / 3 →
  6 * x^2 + 15 * x + v = 0 := by sorry

end root_of_quadratic_l3919_391971


namespace five_circle_five_num_five_circle_seven_num_l3919_391906

-- Define the structure of the diagram
structure Diagram :=
  (n : ℕ)  -- number of circles
  (m : ℕ)  -- maximum number to be used

-- Define a valid filling of the diagram
def ValidFilling (d : Diagram) := Fin d.m → Fin d.n

-- Define the number of valid fillings
def NumValidFillings (d : Diagram) : ℕ := sorry

-- Theorem for the case with 5 circles and numbers 1 to 5
theorem five_circle_five_num :
  ∀ d : Diagram, d.n = 5 ∧ d.m = 5 → NumValidFillings d = 8 := by sorry

-- Theorem for the case with 5 circles and numbers 1 to 7
theorem five_circle_seven_num :
  ∀ d : Diagram, d.n = 5 ∧ d.m = 7 → NumValidFillings d = 48 := by sorry

end five_circle_five_num_five_circle_seven_num_l3919_391906


namespace curve_is_two_lines_l3919_391914

-- Define the equation of the curve
def curve_equation (x y : ℝ) : Prop := x^2 + x*y = x

-- Theorem stating that the curve equation represents two lines
theorem curve_is_two_lines :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, curve_equation x y ↔ (y = m₁ * x + b₁ ∨ y = m₂ * x + b₂)) :=
sorry

end curve_is_two_lines_l3919_391914


namespace jenny_project_hours_l3919_391982

/-- The total hours Jenny has to work on her school project -/
def total_project_hours (research_hours proposal_hours report_hours : ℕ) : ℕ :=
  research_hours + proposal_hours + report_hours

/-- Theorem stating that Jenny's total project hours is 20 -/
theorem jenny_project_hours :
  total_project_hours 10 2 8 = 20 := by
  sorry

end jenny_project_hours_l3919_391982


namespace doug_fires_count_l3919_391967

theorem doug_fires_count (doug kai eli total : ℕ) 
  (h1 : kai = 3 * doug)
  (h2 : eli = kai / 2)
  (h3 : doug + kai + eli = total)
  (h4 : total = 110) : doug = 20 := by
  sorry

end doug_fires_count_l3919_391967


namespace max_square_plots_l3919_391942

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℕ
  width : ℕ

/-- Represents the available internal fencing -/
def availableFence : ℕ := 1500

/-- Calculates the number of square plots given the side length of a square -/
def numPlots (d : FieldDimensions) (side : ℕ) : ℕ :=
  (d.length / side) * (d.width / side)

/-- Calculates the amount of internal fencing needed for a given number of squares per side -/
def fencingNeeded (d : FieldDimensions) (squaresPerSide : ℕ) : ℕ :=
  (d.length * (squaresPerSide - 1)) + (d.width * (squaresPerSide - 1))

/-- Theorem stating that 576 is the maximum number of square plots -/
theorem max_square_plots (d : FieldDimensions) (h1 : d.length = 30) (h2 : d.width = 45) :
  ∀ n : ℕ, numPlots d n ≤ 576 ∧ fencingNeeded d (d.width / (d.width / 24)) ≤ availableFence :=
sorry

end max_square_plots_l3919_391942


namespace polygon_sides_count_l3919_391977

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 → n = 8 :=
by
  sorry

end polygon_sides_count_l3919_391977


namespace nesbitt_inequality_l3919_391908

theorem nesbitt_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) :=
by sorry

end nesbitt_inequality_l3919_391908


namespace insurance_coverage_percentage_l3919_391949

def xray_cost : ℝ := 250
def mri_cost : ℝ := 3 * xray_cost
def total_cost : ℝ := xray_cost + mri_cost
def mike_payment : ℝ := 200
def insurance_coverage : ℝ := total_cost - mike_payment

theorem insurance_coverage_percentage : (insurance_coverage / total_cost) * 100 = 80 :=
by sorry

end insurance_coverage_percentage_l3919_391949


namespace expression_evaluation_l3919_391989

theorem expression_evaluation :
  let f (x : ℚ) := (3 * x + 2) / (2 * x - 1)
  let g (x : ℚ) := (3 * f x + 2) / (2 * f x - 1)
  g (1/3) = 113/31 := by
  sorry

end expression_evaluation_l3919_391989


namespace parabola_latus_rectum_l3919_391968

/-- A parabola passing through a specific point has a specific latus rectum equation -/
theorem parabola_latus_rectum (p : ℝ) (h1 : p > 0) :
  (∀ x y, y^2 = 2*p*x → x = 1 ∧ y = 1/2) →
  (∃ x, x = -1/16 ∧ ∀ y, y^2 = 2*p*x) := by
  sorry

end parabola_latus_rectum_l3919_391968


namespace rectangle_diagonal_shorter_percentage_rectangle_diagonal_shorter_approx_25_percent_l3919_391936

/-- The percentage difference between the sum of two sides of a 2x1 rectangle
    and its diagonal, relative to the sum of the sides. -/
theorem rectangle_diagonal_shorter_percentage : ℝ :=
  let side_sum := 2 + 1
  let diagonal := Real.sqrt (2^2 + 1^2)
  (side_sum - diagonal) / side_sum * 100

/-- The percentage difference is approximately 25%. -/
theorem rectangle_diagonal_shorter_approx_25_percent :
  ∃ ε > 0, abs (rectangle_diagonal_shorter_percentage - 25) < ε :=
sorry

end rectangle_diagonal_shorter_percentage_rectangle_diagonal_shorter_approx_25_percent_l3919_391936


namespace total_highlighters_l3919_391992

theorem total_highlighters (yellow : ℕ) (pink : ℕ) (blue : ℕ) 
  (h1 : yellow = 7)
  (h2 : pink = yellow + 7)
  (h3 : blue = pink + 5) :
  yellow + pink + blue = 40 := by
  sorry

end total_highlighters_l3919_391992


namespace min_product_of_prime_sum_l3919_391922

theorem min_product_of_prime_sum (m n p : ℕ) : 
  Prime m → Prime n → Prime p → 
  m ≠ n → m ≠ p → n ≠ p →
  m + n = p →
  (∀ m' n' p' : ℕ, Prime m' → Prime n' → Prime p' → 
    m' ≠ n' → m' ≠ p' → n' ≠ p' →
    m' + n' = p' → m' * n' * p' ≥ m * n * p) →
  m * n * p = 30 := by
sorry

end min_product_of_prime_sum_l3919_391922


namespace f_iteration_result_l3919_391907

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 - 1

-- State the theorem
theorem f_iteration_result :
  f (f (f (f (2 + I)))) = 1042434 - 131072 * I :=
by sorry

end f_iteration_result_l3919_391907


namespace smallest_book_count_l3919_391926

theorem smallest_book_count (b : ℕ) : 
  (b % 6 = 5) ∧ (b % 8 = 7) ∧ (b % 9 = 2) → 
  (∀ n : ℕ, n < b → ¬((n % 6 = 5) ∧ (n % 8 = 7) ∧ (n % 9 = 2))) → 
  b = 119 := by
sorry

end smallest_book_count_l3919_391926


namespace five_items_four_boxes_l3919_391948

/-- The number of ways to distribute n distinct items into k identical boxes --/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 46 ways to distribute 5 distinct items into 4 identical boxes --/
theorem five_items_four_boxes : distribute 5 4 = 46 := by sorry

end five_items_four_boxes_l3919_391948


namespace john_needs_thirteen_l3919_391921

/-- The amount of additional money John needs to buy a pogo stick -/
def additional_money_needed (saturday_earnings sunday_earnings previous_weekend_earnings pogo_stick_cost : ℕ) : ℕ :=
  pogo_stick_cost - (saturday_earnings + sunday_earnings + previous_weekend_earnings)

/-- Theorem stating how much additional money John needs -/
theorem john_needs_thirteen : 
  ∀ (saturday_earnings sunday_earnings previous_weekend_earnings pogo_stick_cost : ℕ),
    saturday_earnings = 18 →
    sunday_earnings = saturday_earnings / 2 →
    previous_weekend_earnings = 20 →
    pogo_stick_cost = 60 →
    additional_money_needed saturday_earnings sunday_earnings previous_weekend_earnings pogo_stick_cost = 13 := by
  sorry

end john_needs_thirteen_l3919_391921


namespace sequence_sum_property_l3919_391966

theorem sequence_sum_property (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) :
  (∀ n : ℕ+, S n = 1 - n * a n) →
  (∀ n : ℕ+, a n = 1 / (n * (n + 1))) :=
by
  sorry

end sequence_sum_property_l3919_391966


namespace opposite_of_negative_2011_l3919_391964

theorem opposite_of_negative_2011 : 
  -((-2011) : ℤ) = (2011 : ℤ) := by sorry

end opposite_of_negative_2011_l3919_391964


namespace smallest_n_for_sqrt_difference_smallest_n_is_626_l3919_391900

theorem smallest_n_for_sqrt_difference (n : ℕ) : n ≥ 626 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.02 := by
  sorry

theorem smallest_n_is_626 : ∀ k : ℕ, k < 626 → Real.sqrt k - Real.sqrt (k - 1) ≥ 0.02 := by
  sorry

end smallest_n_for_sqrt_difference_smallest_n_is_626_l3919_391900


namespace correct_ratio_achieved_l3919_391935

/-- Represents the ratio of diesel to water in the final mixture -/
def diesel_water_ratio : ℚ := 3 / 5

/-- The initial amount of diesel in quarts -/
def initial_diesel : ℚ := 4

/-- The initial amount of petrol in quarts -/
def initial_petrol : ℚ := 4

/-- The amount of water to be added in quarts -/
def water_to_add : ℚ := 20 / 3

/-- Theorem stating that adding the calculated amount of water results in the desired ratio -/
theorem correct_ratio_achieved :
  diesel_water_ratio = initial_diesel / water_to_add := by
  sorry

#check correct_ratio_achieved

end correct_ratio_achieved_l3919_391935


namespace toy_poodle_height_l3919_391961

/-- Proves that the height of a toy poodle is 14 inches given the heights of standard and miniature poodles -/
theorem toy_poodle_height 
  (standard_height : ℕ) 
  (standard_miniature_diff : ℕ) 
  (miniature_toy_diff : ℕ) 
  (h1 : standard_height = 28)
  (h2 : standard_miniature_diff = 8)
  (h3 : miniature_toy_diff = 6) : 
  standard_height - standard_miniature_diff - miniature_toy_diff = 14 := by
  sorry

#check toy_poodle_height

end toy_poodle_height_l3919_391961


namespace problem_solution_l3919_391996

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem problem_solution :
  (B (-1/3) ⊆ A) ∧
  (∀ a : ℝ, A ∪ B a = A ↔ a = 0 ∨ a = -1/3 ∨ a = -1/5) := by
  sorry

end problem_solution_l3919_391996


namespace solve_a_and_b_l3919_391981

theorem solve_a_and_b : ∃ (a b : ℝ), 
  (b^2 - 2*b = 24) ∧ 
  (4*(1:ℝ)^2 + a = 2) ∧ 
  (4*b^2 - 2*b = 72) ∧ 
  (a = -2) ∧ 
  (b = -4) := by
  sorry

end solve_a_and_b_l3919_391981


namespace number_selection_theorem_l3919_391960

def number_pairs : List (ℕ × ℕ) := [
  (1, 36), (2, 35), (3, 34), (4, 33),
  (5, 32), (6, 31), (7, 30), (8, 29),
  (9, 28), (10, 27), (11, 26), (12, 25)
]

def number_pairs_reduced : List (ℕ × ℕ) := [
  (1, 36), (2, 35), (3, 34), (4, 33),
  (5, 32), (6, 31), (7, 30), (8, 29),
  (9, 28), (10, 27)
]

def is_valid_selection (pairs : List (ℕ × ℕ)) (selection : List Bool) : Prop :=
  selection.length = pairs.length ∧
  (selection.zip pairs).foldl (λ sum (b, (x, y)) => sum + if b then x else y) 0 =
  (selection.zip pairs).foldl (λ sum (b, (x, y)) => sum + if b then y else x) 0

theorem number_selection_theorem :
  (∃ selection, is_valid_selection number_pairs selection) ∧
  (¬ ∃ selection, is_valid_selection number_pairs_reduced selection) := by sorry

end number_selection_theorem_l3919_391960


namespace quadratic_always_positive_implies_a_range_l3919_391995

theorem quadratic_always_positive_implies_a_range (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end quadratic_always_positive_implies_a_range_l3919_391995


namespace line_segment_endpoint_l3919_391945

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  (x - 2)^2 + 2^2 = 10^2 → 
  x = 2 + 4 * Real.sqrt 6 := by
sorry

end line_segment_endpoint_l3919_391945


namespace kite_area_is_18_l3919_391917

/-- The area of a kite with width 6 units and height 7 units, where each unit is one inch. -/
def kite_area : ℝ := 18

/-- The width of the kite in units. -/
def kite_width : ℕ := 6

/-- The height of the kite in units. -/
def kite_height : ℕ := 7

/-- Theorem stating that the area of the kite is 18 square inches. -/
theorem kite_area_is_18 : kite_area = 18 := by sorry

end kite_area_is_18_l3919_391917


namespace product_of_numbers_l3919_391976

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 72) (h2 : x - y = 12) (h3 : x / y = 3/2) :
  x * y = 1244.16 := by
  sorry

end product_of_numbers_l3919_391976


namespace longest_side_length_l3919_391901

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y ≤ 4 ∧ 3*x + y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the vertices of the quadrilateral
def Vertices : Set (ℝ × ℝ) :=
  {(0, 3), (0.4, 1.8), (4, 0), (0, 0)}

-- State the theorem
theorem longest_side_length :
  ∃ (a b : ℝ × ℝ), a ∈ Vertices ∧ b ∈ Vertices ∧
    (∀ (c d : ℝ × ℝ), c ∈ Vertices → d ∈ Vertices →
      Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) ≤ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)) ∧
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by sorry

end longest_side_length_l3919_391901


namespace linear_function_proof_l3919_391909

/-- A linear function passing through (-2, -1) and parallel to y = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x + 3

/-- The slope of the line y = 2x - 3 -/
def slope_parallel : ℝ := 2

theorem linear_function_proof :
  (∀ x, f x = 2 * x + 3) ∧
  f (-2) = -1 ∧
  (∀ x y, f y - f x = slope_parallel * (y - x)) :=
sorry

end linear_function_proof_l3919_391909


namespace root_sum_squares_l3919_391969

/-- The polynomial p(x) = 4x^3 - 2x^2 - 15x + 9 -/
def p (x : ℝ) : ℝ := 4 * x^3 - 2 * x^2 - 15 * x + 9

/-- The polynomial q(x) = 12x^3 + 6x^2 - 7x + 1 -/
def q (x : ℝ) : ℝ := 12 * x^3 + 6 * x^2 - 7 * x + 1

/-- A is the largest root of p(x) -/
def A : ℝ := sorry

/-- B is the largest root of q(x) -/
def B : ℝ := sorry

/-- p(x) has exactly three distinct real roots -/
axiom p_has_three_roots : ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (∀ (w : ℝ), p w = 0 ↔ w = x ∨ w = y ∨ w = z)

/-- q(x) has exactly three distinct real roots -/
axiom q_has_three_roots : ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (∀ (w : ℝ), q w = 0 ↔ w = x ∨ w = y ∨ w = z)

/-- A is a root of p(x) -/
axiom A_is_root_of_p : p A = 0

/-- B is a root of q(x) -/
axiom B_is_root_of_q : q B = 0

/-- A is the largest root of p(x) -/
axiom A_is_largest_root_of_p : ∀ (x : ℝ), p x = 0 → x ≤ A

/-- B is the largest root of q(x) -/
axiom B_is_largest_root_of_q : ∀ (x : ℝ), q x = 0 → x ≤ B

theorem root_sum_squares : A^2 + 3 * B^2 = 4 := by sorry

end root_sum_squares_l3919_391969


namespace no_three_rational_solutions_l3919_391938

theorem no_three_rational_solutions :
  ¬ ∃ (r : ℝ), ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (x^3 - 2023*x^2 - 2023*x + r = 0) ∧
    (y^3 - 2023*y^2 - 2023*y + r = 0) ∧
    (z^3 - 2023*z^2 - 2023*z + r = 0) := by
  sorry

end no_three_rational_solutions_l3919_391938


namespace rectangle_width_decrease_l3919_391965

theorem rectangle_width_decrease (L W : ℝ) (L_new W_new A_new : ℝ) 
  (h1 : L_new = 1.6 * L)
  (h2 : A_new = 1.36 * (L * W))
  (h3 : A_new = L_new * W_new) :
  W_new = 0.85 * W := by
sorry

end rectangle_width_decrease_l3919_391965


namespace polynomial_simplification_l3919_391940

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 :=
by sorry

end polynomial_simplification_l3919_391940


namespace quadratic_inequality_equivalence_l3919_391920

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < 15 ↔ x ∈ Set.Ioo (-5/2) 3 := by
  sorry

end quadratic_inequality_equivalence_l3919_391920


namespace product_relation_l3919_391990

theorem product_relation (x y z : ℝ) (h : x^2 + y^2 = x*y*(z + 1/z)) :
  x = y*z ∨ y = x*z := by sorry

end product_relation_l3919_391990


namespace whitewashing_cost_l3919_391958

-- Define the room dimensions
def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

-- Define the door dimensions
def door_height : ℝ := 6
def door_width : ℝ := 3

-- Define the window dimensions
def window_height : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℕ := 3

-- Define the cost per square foot
def cost_per_sqft : ℝ := 8

-- Theorem statement
theorem whitewashing_cost :
  let total_wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_height * door_width
  let window_area := window_height * window_width * num_windows
  let effective_area := total_wall_area - door_area - window_area
  effective_area * cost_per_sqft = 7248 := by
sorry


end whitewashing_cost_l3919_391958


namespace inequality_solutions_count_l3919_391951

theorem inequality_solutions_count : 
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ 5*x^2 + 19*x + 12 ≤ 20) ∧ Finset.card S = 8 := by
  sorry

end inequality_solutions_count_l3919_391951


namespace expression_value_l3919_391928

theorem expression_value (x y : ℝ) (h : y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 1) :
  (Real.sqrt (48 * y) + Real.sqrt (8 * x)) * (4 * Real.sqrt (3 * y) - 2 * Real.sqrt (2 * x)) - x * y = 30 := by
  sorry

end expression_value_l3919_391928


namespace worker_payment_problem_l3919_391905

/-- Proves that the total number of days is 60 given the conditions of the worker payment problem. -/
theorem worker_payment_problem (daily_pay : ℕ) (daily_deduction : ℕ) (total_payment : ℕ) (idle_days : ℕ) :
  daily_pay = 20 →
  daily_deduction = 3 →
  total_payment = 280 →
  idle_days = 40 →
  ∃ (work_days : ℕ), daily_pay * work_days - daily_deduction * idle_days = total_payment ∧
                      work_days + idle_days = 60 :=
by sorry

end worker_payment_problem_l3919_391905


namespace two_identical_objects_five_recipients_l3919_391962

theorem two_identical_objects_five_recipients : ∀ n : ℕ, n = 5 →
  (Nat.choose n 2) + (Nat.choose n 1) = 15 :=
by
  sorry

end two_identical_objects_five_recipients_l3919_391962


namespace absolute_difference_l3919_391927

theorem absolute_difference (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 10) : |m - n| = 2 * Real.sqrt 19 := by
  sorry

end absolute_difference_l3919_391927


namespace positive_sum_squares_bound_l3919_391974

theorem positive_sum_squares_bound (x y z a : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (ha : a > 0)
  (sum_eq : x + y + z = a)
  (sum_squares_eq : x^2 + y^2 + z^2 = a^2 / 2) :
  x ≤ 2*a/3 ∧ y ≤ 2*a/3 ∧ z ≤ 2*a/3 :=
by sorry

end positive_sum_squares_bound_l3919_391974


namespace min_a_for_quadratic_inequality_l3919_391993

theorem min_a_for_quadratic_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 ∧ x ≤ 1/2 → x^2 + a*x + 1 ≥ 0) ∧
  (∀ (b : ℝ), (∀ (x : ℝ), x > 0 ∧ x ≤ 1/2 → x^2 + b*x + 1 ≥ 0) → b ≥ -5/2) :=
by sorry

end min_a_for_quadratic_inequality_l3919_391993


namespace no_additional_painters_needed_l3919_391950

/-- Represents the painting job scenario -/
structure PaintingJob where
  initialPainters : ℕ
  initialDays : ℚ
  initialRate : ℚ
  newDays : ℕ
  newRate : ℚ

/-- Calculates the total work required for the job -/
def totalWork (job : PaintingJob) : ℚ :=
  job.initialPainters * job.initialDays * job.initialRate

/-- Calculates the number of painters needed for the new conditions -/
def paintersNeeded (job : PaintingJob) : ℚ :=
  (totalWork job) / (job.newDays * job.newRate)

/-- Theorem stating that no additional painters are needed -/
theorem no_additional_painters_needed (job : PaintingJob) 
  (h1 : job.initialPainters = 6)
  (h2 : job.initialDays = 5/2)
  (h3 : job.initialRate = 2)
  (h4 : job.newDays = 2)
  (h5 : job.newRate = 5/2) :
  paintersNeeded job = job.initialPainters :=
by sorry

#check no_additional_painters_needed

end no_additional_painters_needed_l3919_391950


namespace duty_roster_arrangements_l3919_391991

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def adjacent_arrangements (n : ℕ) : ℕ := 2 * (n - 1).factorial

def double_adjacent_arrangements (n : ℕ) : ℕ := 2 * 2 * (n - 2).factorial

theorem duty_roster_arrangements :
  let total := number_of_arrangements 6
  let adjacent_ab := adjacent_arrangements 6
  let adjacent_cd := adjacent_arrangements 6
  let both_adjacent := double_adjacent_arrangements 6
  total - adjacent_ab - adjacent_cd + both_adjacent = 336 := by sorry

end duty_roster_arrangements_l3919_391991


namespace complex_calculation_l3919_391946

theorem complex_calculation : 
  (((3.242^2 * (16 + 8)) / (100 - (3 * 25))) + (32 - 10)^2) = 494.09014144 := by
  sorry

end complex_calculation_l3919_391946


namespace interest_rate_calculation_l3919_391987

/-- Given a principal sum and conditions on simple interest, prove the annual interest rate -/
theorem interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  (P * (20 / 7) * 7) / 100 = P / 5 → (20 / 7 : ℝ) = 20 / 7 := by
  sorry

end interest_rate_calculation_l3919_391987


namespace three_digit_divisible_by_26_l3919_391988

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def first_digit (n : ℕ) : ℕ := n / 100

def second_digit (n : ℕ) : ℕ := (n / 10) % 10

def third_digit (n : ℕ) : ℕ := n % 10

def sum_of_squared_digits (n : ℕ) : ℕ :=
  (first_digit n)^2 + (second_digit n)^2 + (third_digit n)^2

def valid_number (n : ℕ) : Prop :=
  is_three_digit n ∧ 
  first_digit n ≠ 0 ∧
  26 % (sum_of_squared_digits n) = 0

theorem three_digit_divisible_by_26 :
  {n : ℕ | valid_number n} = 
  {100, 110, 101, 302, 320, 230, 203, 431, 413, 314, 341, 134, 143, 510, 501, 150, 105} :=
by sorry

end three_digit_divisible_by_26_l3919_391988


namespace expression_simplification_l3919_391972

theorem expression_simplification (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  a^3 * (-b^3)^2 + (-1/2 * a * b^2)^3 = -7 := by
  sorry

end expression_simplification_l3919_391972


namespace rhombus_perimeter_l3919_391983

/-- A rhombus with given diagonal lengths has a specific perimeter. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 52 := by
  sorry

end rhombus_perimeter_l3919_391983


namespace abcd_sum_l3919_391903

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 9)
  (eq3 : a + c + d = 20)
  (eq4 : b + c + d = 13) :
  a * b + c * d = 72 := by
  sorry

end abcd_sum_l3919_391903


namespace inequality_condition_l3919_391970

theorem inequality_condition : 
  (∀ x : ℝ, -3 < x ∧ x < 0 → (x + 3) * (x - 2) < 0) ∧ 
  (∃ x : ℝ, (x + 3) * (x - 2) < 0 ∧ ¬(-3 < x ∧ x < 0)) :=
sorry

end inequality_condition_l3919_391970


namespace triangle_cosine_ratio_l3919_391910

/-- In any triangle ABC, (b * cos C + c * cos B) / a = 1 -/
theorem triangle_cosine_ratio (A B C a b c : ℝ) : 
  0 < a → 0 < b → 0 < c →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  (b * Real.cos C + c * Real.cos B) / a = 1 := by
sorry

end triangle_cosine_ratio_l3919_391910


namespace tangent_line_problem_l3919_391919

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (k + 1) * x^3 - 3 * (k + 2) * x^2 - k^2 - 2 * k

-- Define the derivative of f
def f' (k : ℝ) (x : ℝ) : ℝ := 3 * (k + 1) * x^2 - 6 * (k + 2) * x

theorem tangent_line_problem (k : ℝ) (h1 : k > -1) :
  (∀ x ∈ Set.Ioo 0 4, f' k x < 0) →
  (k = 0 ∧ 
   ∃ t : ℝ, t = f' 0 1 ∧ 9 * 1 + (-5) + 4 = 0 ∧ 
   ∀ x y : ℝ, y = t * (x - 1) + (-5) ↔ 9 * x + y + 4 = 0) :=
by sorry

end tangent_line_problem_l3919_391919
