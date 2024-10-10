import Mathlib

namespace b_share_yearly_profit_l2890_289052

/-- Investment proportions and profit distribution for partners A, B, C, and D --/
structure Partnership where
  b_invest : ℝ  -- B's investment (base unit)
  a_invest : ℝ := 2.5 * b_invest  -- A's investment
  c_invest : ℝ := 1.5 * b_invest  -- C's investment
  d_invest : ℝ := 1.25 * b_invest  -- D's investment
  total_invest : ℝ := a_invest + b_invest + c_invest  -- Total investment of A, B, and C
  profit_6months : ℝ := 6000  -- Profit for 6 months
  d_fixed_amount : ℝ := 500  -- D's fixed amount per 6 months
  profit_year : ℝ := 16900  -- Total profit for the year

/-- Theorem stating B's share of the yearly profit --/
theorem b_share_yearly_profit (p : Partnership) :
  (p.b_invest / p.total_invest) * (p.profit_year - 2 * p.d_fixed_amount) = 3180 := by
  sorry

end b_share_yearly_profit_l2890_289052


namespace complement_A_in_U_l2890_289039

def U : Set ℕ := {x | (x + 1) / (x - 5) ≤ 0}

def A : Set ℕ := {1, 2, 4}

theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {0, 3} := by sorry

end complement_A_in_U_l2890_289039


namespace dans_car_efficiency_l2890_289028

/-- Represents the fuel efficiency of Dan's car in miles per gallon. -/
def miles_per_gallon : ℝ := 32

/-- Represents the cost of gas in dollars per gallon. -/
def gas_cost_per_gallon : ℝ := 4

/-- Represents the distance Dan's car can travel in miles. -/
def distance_traveled : ℝ := 368

/-- Represents the total cost of gas in dollars. -/
def total_gas_cost : ℝ := 46

/-- Proves that Dan's car gets 32 miles per gallon given the conditions. -/
theorem dans_car_efficiency :
  miles_per_gallon = distance_traveled / (total_gas_cost / gas_cost_per_gallon) := by
  sorry


end dans_car_efficiency_l2890_289028


namespace sin_cos_fourth_power_range_l2890_289031

theorem sin_cos_fourth_power_range (x : ℝ) : 
  0.5 ≤ Real.sin x ^ 4 + Real.cos x ^ 4 ∧ Real.sin x ^ 4 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end sin_cos_fourth_power_range_l2890_289031


namespace charity_fundraiser_revenue_l2890_289035

theorem charity_fundraiser_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 170)
  (h_total_revenue : total_revenue = 2917) :
  ∃ (full_price : ℕ) (full_count : ℕ) (quarter_count : ℕ),
    full_count + quarter_count = total_tickets ∧
    full_count * full_price + quarter_count * (full_price / 4) = total_revenue ∧
    full_count * full_price = 1748 :=
by sorry

end charity_fundraiser_revenue_l2890_289035


namespace max_ratio_squared_l2890_289012

theorem max_ratio_squared (a b x y : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_geq_b : a ≥ b)
  (h_eq : a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = Real.sqrt ((a - x)^2 + (b - y)^2))
  (h_bounds : 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b)
  (h_right_triangle : (a - b + x)^2 + (b - a + y)^2 = a^2 + b^2) :
  (∀ ρ : ℝ, a ≤ ρ * b → ρ^2 ≤ 1) :=
sorry

end max_ratio_squared_l2890_289012


namespace wayne_shrimp_cost_l2890_289048

/-- Calculates the cost of shrimp for an appetizer given the number of shrimp per guest,
    number of guests, cost per pound, and number of shrimp per pound. -/
def shrimpAppetizer (shrimpPerGuest : ℕ) (numGuests : ℕ) (costPerPound : ℚ) (shrimpPerPound : ℕ) : ℚ :=
  (shrimpPerGuest * numGuests : ℚ) / shrimpPerPound * costPerPound

/-- Proves that Wayne will spend $170.00 on the shrimp appetizer given the specified conditions. -/
theorem wayne_shrimp_cost :
  shrimpAppetizer 5 40 17 20 = 170 := by
  sorry

end wayne_shrimp_cost_l2890_289048


namespace negation_of_proposition_l2890_289063

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x * Real.exp x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ * Real.exp x₀ ≤ 0) :=
by sorry

end negation_of_proposition_l2890_289063


namespace least_divisible_by_first_ten_l2890_289088

def first_ten_integers : Finset ℕ := Finset.range 10

theorem least_divisible_by_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ i ∈ first_ten_integers, i ∣ n) ∧ 
  (∀ m : ℕ, m > 0 ∧ (∀ i ∈ first_ten_integers, i ∣ m) → n ≤ m) ∧ n = 2520 :=
sorry

end least_divisible_by_first_ten_l2890_289088


namespace trapezoid_side_length_l2890_289079

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- The sum of AB and CD is 300
  sum_sides : ab + cd = 300
  -- The ratio of areas is 5:4
  ratio_condition : area_ratio = 5 / 4

/-- 
Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC 
to the area of triangle ADC is 5:4, and AB + CD = 300 cm, 
then AB = 500/3 cm.
-/
theorem trapezoid_side_length (t : Trapezoid) : t.ab = 500 / 3 := by
  sorry

end trapezoid_side_length_l2890_289079


namespace molecular_weight_independent_of_moles_l2890_289006

/-- The molecular weight of an acid in g/mol -/
def molecular_weight : ℝ := 408

/-- The number of moles of the acid -/
def moles : ℝ := 6

/-- Theorem stating that the molecular weight is independent of the number of moles -/
theorem molecular_weight_independent_of_moles :
  molecular_weight = molecular_weight := by sorry

end molecular_weight_independent_of_moles_l2890_289006


namespace battle_gathering_count_l2890_289037

theorem battle_gathering_count :
  -- Define the number of cannoneers
  ∀ (cannoneers : ℕ),
  -- Define the number of women as double the number of cannoneers
  ∀ (women : ℕ),
  women = 2 * cannoneers →
  -- Define the number of men as twice the number of women
  ∀ (men : ℕ),
  men = 2 * women →
  -- Given condition: there are 63 cannoneers
  cannoneers = 63 →
  -- Prove that the total number of people is 378
  men + women = 378 := by
sorry

end battle_gathering_count_l2890_289037


namespace f_derivative_sum_l2890_289078

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- State the theorem
theorem f_derivative_sum (h1 : ∀ x, HasDerivAt f (f' x) x)
                         (h2 : ∀ x, f x = 3 * x^2 + 2 * x * f' 2) :
  f' 5 + f' 2 = -6 := by
  sorry

end f_derivative_sum_l2890_289078


namespace same_distance_different_speeds_l2890_289072

/-- Proves that given Joann's average speed and time, Fran needs to ride at a specific speed to cover the same distance in her given time -/
theorem same_distance_different_speeds (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4)
  (h3 : fran_time = 5) :
  joann_speed * joann_time = (60 / fran_time) * fran_time :=
by sorry

end same_distance_different_speeds_l2890_289072


namespace geometric_sequence_general_term_l2890_289050

/-- A geometric sequence with positive terms satisfying a certain relation -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = q * a n) ∧
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))

/-- The general term of the geometric sequence -/
def GeneralTerm (a : ℕ → ℝ) : Prop :=
  ∃ a₁ : ℝ, ∀ n, a n = a₁ * 2^(n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  GeometricSequence a → GeneralTerm a := by
  sorry

end geometric_sequence_general_term_l2890_289050


namespace unique_solution_for_abc_l2890_289053

theorem unique_solution_for_abc : ∃! (a b c : ℝ),
  a < b ∧ b < c ∧
  a + b + c = 21 / 4 ∧
  1 / a + 1 / b + 1 / c = 21 / 4 ∧
  a * b * c = 1 ∧
  a = 1 / 4 ∧ b = 1 ∧ c = 4 := by
  sorry

end unique_solution_for_abc_l2890_289053


namespace range_of_m_l2890_289007

/-- The function f(x) = x^2 - 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

/-- Set A: range of a for which f(x) has no real roots -/
def A : Set ℝ := {a | ∀ x, f a x ≠ 0}

/-- Set B: range of a for which f(x) is not monotonic on (m, m+3) -/
def B (m : ℝ) : Set ℝ := {a | ∃ x y, m < x ∧ x < y ∧ y < m + 3 ∧ (f a x - f a y) * (x - y) < 0}

/-- Theorem: If x ∈ A is a sufficient but not necessary condition for x ∈ B, 
    then -2 ≤ m ≤ -1 -/
theorem range_of_m (m : ℝ) : 
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → -2 ≤ m ∧ m ≤ -1 := by
  sorry

end range_of_m_l2890_289007


namespace calzone_time_is_124_l2890_289096

/-- The total time spent on making calzones -/
def total_calzone_time (onion_time garlic_pepper_time knead_time rest_time assemble_time : ℕ) : ℕ :=
  onion_time + garlic_pepper_time + knead_time + rest_time + assemble_time

/-- Theorem stating the total time spent on making calzones is 124 minutes -/
theorem calzone_time_is_124 : 
  ∀ (onion_time garlic_pepper_time knead_time rest_time assemble_time : ℕ),
    onion_time = 20 →
    garlic_pepper_time = onion_time / 4 →
    knead_time = 30 →
    rest_time = 2 * knead_time →
    assemble_time = (knead_time + rest_time) / 10 →
    total_calzone_time onion_time garlic_pepper_time knead_time rest_time assemble_time = 124 :=
by
  sorry


end calzone_time_is_124_l2890_289096


namespace patricks_pencil_loss_percentage_l2890_289036

/-- Calculates the overall loss percentage for Patrick's pencil sales -/
theorem patricks_pencil_loss_percentage : 
  let type_a_count : ℕ := 30
  let type_b_count : ℕ := 40
  let type_c_count : ℕ := 10
  let type_a_cost : ℚ := 1
  let type_b_cost : ℚ := 2
  let type_c_cost : ℚ := 3
  let type_a_discount : ℚ := 0.5
  let type_b_discount : ℚ := 1
  let type_c_discount : ℚ := 1.5
  let total_cost : ℚ := type_a_count * type_a_cost + type_b_count * type_b_cost + type_c_count * type_c_cost
  let total_revenue : ℚ := type_a_count * (type_a_cost - type_a_discount) + 
                           type_b_count * (type_b_cost - type_b_discount) + 
                           type_c_count * (type_c_cost - type_c_discount)
  let additional_loss : ℚ := type_a_count * (type_a_cost - type_a_discount)
  let total_loss : ℚ := total_cost - total_revenue + additional_loss
  let loss_percentage : ℚ := (total_loss / total_cost) * 100
  ∃ ε > 0, |loss_percentage - 60.71| < ε :=
by sorry

end patricks_pencil_loss_percentage_l2890_289036


namespace inversion_property_l2890_289091

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point in 2D plane -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Inversion of a point with respect to a circle -/
def inversion (c : Circle) (p : Point) : Point := sorry

/-- Theorem: Inversion property -/
theorem inversion_property (c : Circle) (p p' : Point) : 
  p' = inversion c p → 
  distance c.center p * distance c.center p' = c.radius ^ 2 := by
  sorry

end inversion_property_l2890_289091


namespace min_value_theorem_l2890_289057

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 4/y ≥ 1/a + 4/b) →
  1/a + 4/b = 9 :=
sorry

end min_value_theorem_l2890_289057


namespace crypto_puzzle_solution_l2890_289010

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem crypto_puzzle_solution :
  ∀ (A B C : ℕ),
    is_digit A →
    is_digit B →
    is_digit C →
    A + B + 1 = C + 10 →
    B = A + 2 →
    A ≠ B ∧ A ≠ C ∧ B ≠ C →
    C = 1 :=
by sorry

end crypto_puzzle_solution_l2890_289010


namespace tetrahedral_toys_probability_l2890_289090

-- Define the face values of the tetrahedral toys
def face_values : Finset ℕ := {1, 2, 3, 5}

-- Define the sample space of all possible outcomes
def sample_space : Finset (ℕ × ℕ) := face_values.product face_values

-- Define m as the sum of the two face values
def m (outcome : ℕ × ℕ) : ℕ := outcome.1 + outcome.2

-- Define the event where m is not less than 6
def event_m_ge_6 : Finset (ℕ × ℕ) := sample_space.filter (λ x => m x ≥ 6)

-- Define the event where m is odd
def event_m_odd : Finset (ℕ × ℕ) := sample_space.filter (λ x => m x % 2 = 1)

-- Define the event where m is even
def event_m_even : Finset (ℕ × ℕ) := sample_space.filter (λ x => m x % 2 = 0)

theorem tetrahedral_toys_probability :
  (event_m_ge_6.card : ℚ) / sample_space.card = 1/2 ∧
  (event_m_odd.card : ℚ) / sample_space.card = 3/8 ∧
  (event_m_even.card : ℚ) / sample_space.card = 5/8 :=
sorry

end tetrahedral_toys_probability_l2890_289090


namespace product_sequence_sum_l2890_289003

theorem product_sequence_sum (a b : ℕ) : 
  (a : ℚ) / 4 = 42 → b = a - 1 → a + b = 335 := by sorry

end product_sequence_sum_l2890_289003


namespace value_of_expression_l2890_289025

theorem value_of_expression (x : ℝ) (h : x = 4) : 3 * x + 5 = 17 := by
  sorry

end value_of_expression_l2890_289025


namespace apples_given_to_teachers_l2890_289030

/-- Given Sarah's apple distribution, prove the number given to teachers. -/
theorem apples_given_to_teachers 
  (initial_apples : Nat) 
  (final_apples : Nat) 
  (friends_given_apples : Nat) 
  (apples_eaten : Nat) 
  (h1 : initial_apples = 25)
  (h2 : final_apples = 3)
  (h3 : friends_given_apples = 5)
  (h4 : apples_eaten = 1) :
  initial_apples - final_apples - friends_given_apples - apples_eaten = 16 := by
  sorry

#check apples_given_to_teachers

end apples_given_to_teachers_l2890_289030


namespace josiah_hans_age_ratio_l2890_289059

theorem josiah_hans_age_ratio :
  ∀ (hans_age josiah_age : ℕ),
    hans_age = 15 →
    josiah_age + 3 + hans_age + 3 = 66 →
    josiah_age / hans_age = 3 :=
by
  sorry

end josiah_hans_age_ratio_l2890_289059


namespace meeting_percentage_is_37_5_l2890_289089

/-- Represents the duration of a workday in minutes -/
def workday_duration : ℕ := 8 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_duration : ℕ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_duration : ℕ := 2 * first_meeting_duration

/-- Represents the total duration of both meetings in minutes -/
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration

/-- Represents the percentage of the workday spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_duration : ℚ) / (workday_duration : ℚ) * 100

theorem meeting_percentage_is_37_5 : meeting_percentage = 37.5 := by
  sorry

end meeting_percentage_is_37_5_l2890_289089


namespace problem_1_problem_2_l2890_289092

-- Problem 1
theorem problem_1 : (1 * (1/6 - 5/7 + 2/3)) * (-42) = -5 := by sorry

-- Problem 2
theorem problem_2 : -(2^2) + (-3)^2 * (-2/3) - 4^2 / |(-4)| = -14 := by sorry

end problem_1_problem_2_l2890_289092


namespace problem_solution_l2890_289086

-- Define the equation
def equation (m x : ℝ) : ℝ := x^2 + m*x + 2*m + 5

-- Define the set A
def set_A : Set ℝ := {m : ℝ | ∀ x : ℝ, equation m x ≠ 0 ∨ ∃ y : ℝ, y ≠ x ∧ equation m x = 0 ∧ equation m y = 0}

-- Define the set B
def set_B (a : ℝ) : Set ℝ := {x : ℝ | 1 - 2*a ≤ x ∧ x ≤ a - 1}

theorem problem_solution :
  (∀ m : ℝ, m ∈ set_A ↔ -2 ≤ m ∧ m ≤ 10) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ set_A → x ∈ set_B a) ∧ (∃ x : ℝ, x ∈ set_B a ∧ x ∉ set_A) ↔ 11 ≤ a) :=
by sorry

end problem_solution_l2890_289086


namespace minimal_sum_distances_l2890_289043

noncomputable section

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point in 2D plane -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Inverse point with respect to a circle -/
def inverse_point (c : Circle) (p : Point) : Point := sorry

/-- Line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Perpendicular bisector of two points -/
def perpendicular_bisector (p q : Point) : Line := sorry

/-- Intersection point of a line and a circle -/
def line_circle_intersection (l : Line) (c : Circle) : Option Point := sorry

/-- Theorem: Minimal sum of distances from two fixed points to a point on a circle -/
theorem minimal_sum_distances (c : Circle) (p q : Point) 
  (h1 : distance c.center p = distance c.center q) 
  (h2 : distance c.center p < c.radius ∧ distance c.center q < c.radius) :
  ∃ z : Point, 
    (distance c.center z = c.radius) ∧ 
    (∀ w : Point, distance c.center w = c.radius → 
      distance p z + distance q z ≤ distance p w + distance q w) :=
sorry

end

end minimal_sum_distances_l2890_289043


namespace jean_calories_consumption_l2890_289066

/-- Calculates the total calories consumed by Jean while writing her paper. -/
def total_calories (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

/-- Proves that Jean consumes 900 calories while writing her paper. -/
theorem jean_calories_consumption :
  total_calories 12 2 150 = 900 := by
  sorry

#eval total_calories 12 2 150

end jean_calories_consumption_l2890_289066


namespace unique_solution_l2890_289004

/-- Represents the pictures in the table --/
inductive Picture : Type
| Cat : Picture
| Chicken : Picture
| Crab : Picture
| Bear : Picture
| Goat : Picture

/-- Assignment of digits to pictures --/
def PictureAssignment := Picture → Fin 10

/-- Checks if all pictures are assigned different digits --/
def is_valid_assignment (assignment : PictureAssignment) : Prop :=
  ∀ p q : Picture, p ≠ q → assignment p ≠ assignment q

/-- Checks if the assignment satisfies the row and column sums --/
def satisfies_sums (assignment : PictureAssignment) : Prop :=
  assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab = 10 ∧
  assignment Picture.Goat + assignment Picture.Goat + assignment Picture.Crab + assignment Picture.Bear + assignment Picture.Bear = 16 ∧
  assignment Picture.Cat + assignment Picture.Bear + assignment Picture.Goat + assignment Picture.Goat + assignment Picture.Crab = 13 ∧
  assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Chicken + assignment Picture.Chicken + assignment Picture.Goat = 17 ∧
  assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Goat = 11

/-- The theorem to be proved --/
theorem unique_solution :
  ∃! assignment : PictureAssignment,
    is_valid_assignment assignment ∧
    satisfies_sums assignment ∧
    assignment Picture.Cat = 1 ∧
    assignment Picture.Chicken = 5 ∧
    assignment Picture.Crab = 2 ∧
    assignment Picture.Bear = 4 ∧
    assignment Picture.Goat = 3 :=
  sorry

end unique_solution_l2890_289004


namespace shaded_area_is_16_l2890_289082

/-- Represents the shaded area of a 6x6 grid with triangles and trapezoids -/
def shadedArea (gridSize : Nat) (triangleCount : Nat) (trapezoidCount : Nat) 
  (triangleSquares : Nat) (trapezoidSquares : Nat) : Nat :=
  triangleCount * triangleSquares + trapezoidCount * trapezoidSquares

/-- Theorem stating that the shaded area of the described grid is 16 square units -/
theorem shaded_area_is_16 : 
  shadedArea 6 2 4 2 3 = 16 := by
  sorry

end shaded_area_is_16_l2890_289082


namespace range_of_a_l2890_289051

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (h : ∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ ∃ x, ¬(p x) ∧ (q x a)) :
  ∀ a : ℝ, a ≥ 1 :=
sorry

end range_of_a_l2890_289051


namespace toll_constant_value_l2890_289015

/-- Represents the toll formula for a bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ := constant + 0.50 * (x - 2)

/-- Calculates the number of axles for a truck given its wheel configuration -/
def calculate_axles (front_wheels : ℕ) (other_wheels : ℕ) : ℕ :=
  1 + (other_wheels / 4)

theorem toll_constant_value :
  ∃ (constant : ℝ),
    let x := calculate_axles 2 16
    toll_formula constant x = 4 ∧ constant = 2.50 := by
  sorry

end toll_constant_value_l2890_289015


namespace stripe_length_on_cylinder_l2890_289080

/-- Proves that the length of a diagonal line on a rectangle with sides 30 inches and 16 inches is 34 inches. -/
theorem stripe_length_on_cylinder (circumference height : ℝ) (h1 : circumference = 30) (h2 : height = 16) :
  Real.sqrt (circumference^2 + height^2) = 34 :=
by sorry

end stripe_length_on_cylinder_l2890_289080


namespace prop_analysis_l2890_289022

-- Define the original proposition
def original_prop (x y : ℝ) : Prop := (x + y = 5) → (x = 3 ∧ y = 2)

-- Define the converse
def converse (x y : ℝ) : Prop := (x = 3 ∧ y = 2) → (x + y = 5)

-- Define the inverse
def inverse (x y : ℝ) : Prop := (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2)

-- Define the contrapositive
def contrapositive (x y : ℝ) : Prop := (x ≠ 3 ∨ y ≠ 2) → (x + y ≠ 5)

-- Theorem stating the truth values of converse, inverse, and contrapositive
theorem prop_analysis :
  (∀ x y : ℝ, converse x y) ∧
  (¬ ∀ x y : ℝ, inverse x y) ∧
  (∀ x y : ℝ, contrapositive x y) :=
by sorry

end prop_analysis_l2890_289022


namespace cube_surface_area_l2890_289000

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1000 →
  volume = side^3 →
  surface_area = 6 * side^2 →
  surface_area = 600 :=
by
  sorry

end cube_surface_area_l2890_289000


namespace min_cost_50_percent_alloy_l2890_289040

/-- Represents a gold alloy with its gold percentage and cost per ounce -/
structure GoldAlloy where
  percentage : Rat
  cost : Rat

/-- Theorem stating the minimum cost per ounce to create a 50% gold alloy -/
theorem min_cost_50_percent_alloy 
  (alloy40 : GoldAlloy) 
  (alloy60 : GoldAlloy)
  (alloy90 : GoldAlloy)
  (h1 : alloy40.percentage = 40/100)
  (h2 : alloy60.percentage = 60/100)
  (h3 : alloy90.percentage = 90/100)
  (h4 : alloy40.cost = 200)
  (h5 : alloy60.cost = 300)
  (h6 : alloy90.cost = 400) :
  ∃ (x y z : Rat),
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    (x * alloy40.percentage + y * alloy60.percentage + z * alloy90.percentage) / (x + y + z) = 1/2 ∧
    (x * alloy40.cost + y * alloy60.cost + z * alloy90.cost) / (x + y + z) = 240 ∧
    ∀ (a b c : Rat),
      a ≥ 0 → b ≥ 0 → c ≥ 0 →
      (a * alloy40.percentage + b * alloy60.percentage + c * alloy90.percentage) / (a + b + c) = 1/2 →
      (a * alloy40.cost + b * alloy60.cost + c * alloy90.cost) / (a + b + c) ≥ 240 := by
  sorry

end min_cost_50_percent_alloy_l2890_289040


namespace P_on_x_axis_P_parallel_y_axis_P_second_quadrant_distance_l2890_289019

-- Define point P
def P (a : ℝ) := (a - 1, 6 + 2*a)

-- Question 1
theorem P_on_x_axis (a : ℝ) : 
  P a = (-4, 0) ↔ (P a).2 = 0 := by sorry

-- Question 2
def Q : ℝ × ℝ := (5, 8)

theorem P_parallel_y_axis (a : ℝ) : 
  P a = (5, 18) ↔ (P a).1 = Q.1 := by sorry

-- Question 3
theorem P_second_quadrant_distance (a : ℝ) : 
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).2| = 2 * |(P a).1| → 
  a^2023 + 2024 = 2023 := by sorry

end P_on_x_axis_P_parallel_y_axis_P_second_quadrant_distance_l2890_289019


namespace tomato_seeds_problem_l2890_289070

/-- Represents the number of tomato seeds planted by Mike in the morning -/
def mike_morning : ℕ := sorry

/-- Represents the number of tomato seeds planted by Ted in the morning -/
def ted_morning : ℕ := sorry

/-- Represents the number of tomato seeds planted by Mike in the afternoon -/
def mike_afternoon : ℕ := 60

/-- Represents the number of tomato seeds planted by Ted in the afternoon -/
def ted_afternoon : ℕ := sorry

theorem tomato_seeds_problem :
  ted_morning = 2 * mike_morning ∧
  ted_afternoon = mike_afternoon - 20 ∧
  mike_morning + ted_morning + mike_afternoon + ted_afternoon = 250 →
  mike_morning = 50 := by sorry

end tomato_seeds_problem_l2890_289070


namespace sphere_surface_area_l2890_289062

theorem sphere_surface_area (V : ℝ) (h : V = 72 * Real.pi) :
  ∃ (r : ℝ), V = (4 / 3) * Real.pi * r^3 ∧
              4 * Real.pi * r^2 = 36 * Real.pi * 2^(2/3) := by
  sorry

end sphere_surface_area_l2890_289062


namespace max_hands_for_54_coincidences_l2890_289083

/-- Represents a clock with minute hands moving in opposite directions -/
structure Clock where
  coincidences : ℕ  -- Number of coincidences in an hour
  handsForward : ℕ  -- Number of hands moving forward
  handsBackward : ℕ -- Number of hands moving backward

/-- The total number of hands on the clock -/
def Clock.totalHands (c : Clock) : ℕ := c.handsForward + c.handsBackward

/-- Predicate to check if the clock configuration is valid -/
def Clock.isValid (c : Clock) : Prop :=
  c.handsForward * c.handsBackward * 2 = c.coincidences

/-- Theorem stating the maximum number of hands for a clock with 54 coincidences -/
theorem max_hands_for_54_coincidences :
  ∃ (c : Clock), c.coincidences = 54 ∧ c.isValid ∧
  ∀ (d : Clock), d.coincidences = 54 → d.isValid → d.totalHands ≤ c.totalHands :=
by
  sorry

end max_hands_for_54_coincidences_l2890_289083


namespace james_soda_consumption_l2890_289001

/-- Calculates the number of sodas James drinks per day given the following conditions:
  * James buys 5 packs of sodas
  * Each pack contains 12 sodas
  * James already had 10 sodas
  * He finishes all the sodas in 1 week (7 days)
-/
theorem james_soda_consumption 
  (packs : ℕ) 
  (sodas_per_pack : ℕ) 
  (initial_sodas : ℕ) 
  (days_to_finish : ℕ) 
  (h1 : packs = 5)
  (h2 : sodas_per_pack = 12)
  (h3 : initial_sodas = 10)
  (h4 : days_to_finish = 7) :
  (packs * sodas_per_pack + initial_sodas) / days_to_finish = 10 := by
  sorry

end james_soda_consumption_l2890_289001


namespace even_function_properties_l2890_289041

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define what it means for a function to be symmetric about a vertical line
def symmetric_about_vertical_line (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- Define what it means for a function to be symmetric about the y-axis
def symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem even_function_properties (h : is_even (fun x => f (x + 1))) :
  (symmetric_about_vertical_line f 1) ∧
  (symmetric_about_y_axis (fun x => f (x + 1))) ∧
  (∀ x, f (1 + x) = f (1 - x)) :=
by sorry

end even_function_properties_l2890_289041


namespace no_parallel_axes_in_bounded_figures_parallel_axes_in_unbounded_figures_intersecting_axes_in_all_figures_l2890_289098

-- Define a spatial geometric figure
structure SpatialFigure where
  isBounded : Bool

-- Define an axis of symmetry
structure SymmetryAxis where
  figure : SpatialFigure

-- Define a relation for parallel axes
def areParallel (a1 a2 : SymmetryAxis) : Prop :=
  sorry

-- Define a relation for intersecting axes
def areIntersecting (a1 a2 : SymmetryAxis) : Prop :=
  sorry

-- Theorem 1: Bounded figures cannot have parallel axes of symmetry
theorem no_parallel_axes_in_bounded_figures (f : SpatialFigure) (h : f.isBounded) :
  ¬∃ (a1 a2 : SymmetryAxis), a1.figure = f ∧ a2.figure = f ∧ areParallel a1 a2 :=
sorry

-- Theorem 2: Unbounded figures can have parallel axes of symmetry
theorem parallel_axes_in_unbounded_figures :
  ∃ (f : SpatialFigure) (a1 a2 : SymmetryAxis), 
    ¬f.isBounded ∧ a1.figure = f ∧ a2.figure = f ∧ areParallel a1 a2 :=
sorry

-- Theorem 3: Both bounded and unbounded figures can have intersecting axes of symmetry
theorem intersecting_axes_in_all_figures :
  (∃ (f : SpatialFigure) (a1 a2 : SymmetryAxis), 
    f.isBounded ∧ a1.figure = f ∧ a2.figure = f ∧ areIntersecting a1 a2) ∧
  (∃ (f : SpatialFigure) (a1 a2 : SymmetryAxis), 
    ¬f.isBounded ∧ a1.figure = f ∧ a2.figure = f ∧ areIntersecting a1 a2) :=
sorry

end no_parallel_axes_in_bounded_figures_parallel_axes_in_unbounded_figures_intersecting_axes_in_all_figures_l2890_289098


namespace fifteenth_even_multiple_of_four_l2890_289024

-- Define a function that represents the nth positive integer that is both even and a multiple of 4
def evenMultipleOfFour (n : ℕ) : ℕ := 4 * n

-- State the theorem
theorem fifteenth_even_multiple_of_four : evenMultipleOfFour 15 = 60 := by
  sorry

end fifteenth_even_multiple_of_four_l2890_289024


namespace runner_speed_proof_l2890_289033

def total_distance : ℝ := 1000
def total_time : ℝ := 380
def first_segment_distance : ℝ := 720
def first_segment_speed : ℝ := 3

def second_segment_speed : ℝ := 2

theorem runner_speed_proof :
  let first_segment_time := first_segment_distance / first_segment_speed
  let second_segment_distance := total_distance - first_segment_distance
  let second_segment_time := total_time - first_segment_time
  second_segment_speed = second_segment_distance / second_segment_time :=
by
  sorry

end runner_speed_proof_l2890_289033


namespace inscribed_circle_diameter_l2890_289045

theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 8) (h3 : EF = 9) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let diameter := 2 * area / s
  diameter = 4 * Real.sqrt 35 / 5 := by sorry

end inscribed_circle_diameter_l2890_289045


namespace election_votes_l2890_289008

theorem election_votes (total_votes : ℕ) 
  (h1 : ∃ (winner loser : ℕ), winner + loser = total_votes) 
  (h2 : ∃ (winner : ℕ), winner = (70 * total_votes) / 100) 
  (h3 : ∃ (winner loser : ℕ), winner - loser = 188) : 
  total_votes = 470 := by
sorry

end election_votes_l2890_289008


namespace inequality_proof_l2890_289038

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_proof_l2890_289038


namespace negation_of_universal_proposition_l2890_289042

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) :=
by sorry

end negation_of_universal_proposition_l2890_289042


namespace price_decrease_percentage_l2890_289023

def original_soup_price : ℚ := 7.50 / 3
def original_bread_price : ℚ := 5 / 2
def new_soup_price : ℚ := 8 / 4
def new_bread_price : ℚ := 6 / 3

def original_bundle_avg : ℚ := (original_soup_price + original_bread_price) / 2
def new_bundle_avg : ℚ := (new_soup_price + new_bread_price) / 2

theorem price_decrease_percentage :
  (original_bundle_avg - new_bundle_avg) / original_bundle_avg * 100 = 20 := by
  sorry

end price_decrease_percentage_l2890_289023


namespace cylinder_radius_in_cone_l2890_289087

/-- 
Given a right circular cone with diameter 14 and altitude 16, and an inscribed right circular 
cylinder whose diameter equals its height, prove that the radius of the cylinder is 56/15.
-/
theorem cylinder_radius_in_cone (r : ℚ) : 
  (16 : ℚ) - 2 * r = (16 : ℚ) / 7 * r → r = 56 / 15 := by
  sorry

end cylinder_radius_in_cone_l2890_289087


namespace roots_of_quadratic_sum_of_eighth_powers_l2890_289099

theorem roots_of_quadratic_sum_of_eighth_powers (a b : ℂ) : 
  (a^2 - 2*a + 5 = 0) → (b^2 - 2*b + 5 = 0) → Complex.abs (a^8 + b^8) = 1054 := by
  sorry

end roots_of_quadratic_sum_of_eighth_powers_l2890_289099


namespace cooking_cleaning_combinations_l2890_289046

-- Define the number of friends
def total_friends : ℕ := 5

-- Define the number of cooks
def num_cooks : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem cooking_cleaning_combinations :
  combination total_friends num_cooks = 10 := by
  sorry

end cooking_cleaning_combinations_l2890_289046


namespace triangle_area_theorem_l2890_289047

theorem triangle_area_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 4*a*x + 4*b*y = 48) →
  ((1/2) * (12/a) * (12/b) = 48) →
  a * b = 3/2 := by sorry

end triangle_area_theorem_l2890_289047


namespace solution_product_l2890_289017

theorem solution_product (p q : ℝ) : 
  (p - 6) * (3 * p + 12) = p^2 + 2 * p - 72 →
  (q - 6) * (3 * q + 12) = q^2 + 2 * q - 72 →
  p ≠ q →
  (p + 2) * (q + 2) = -1 := by
  sorry

end solution_product_l2890_289017


namespace function_periodicity_l2890_289011

def periodic_function (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (x + c) = f x

theorem function_periodicity 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  ∃ c > 0, periodic_function f c ∧ c = 1 :=
sorry

end function_periodicity_l2890_289011


namespace sweets_neither_red_nor_green_l2890_289095

theorem sweets_neither_red_nor_green 
  (total : ℕ) 
  (red : ℕ) 
  (green : ℕ) 
  (h_total : total = 285) 
  (h_red : red = 49) 
  (h_green : green = 59) : 
  total - (red + green) = 177 := by
sorry

end sweets_neither_red_nor_green_l2890_289095


namespace function_properties_l2890_289067

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) : ℝ := x + a * x^2 + b * Real.log x

theorem function_properties :
  (f a b 1 = 0 ∧ (deriv (f a b)) 1 = 2) →
  (a = -1 ∧ b = 3 ∧ ∀ x > 0, f a b x ≤ 2 * x - 2) :=
by sorry

end function_properties_l2890_289067


namespace rosie_pies_from_27_apples_l2890_289074

/-- Represents the number of pies Rosie can make given a certain number of apples -/
def pies_from_apples (apples : ℕ) : ℚ :=
  (2 : ℚ) * apples / 9

theorem rosie_pies_from_27_apples :
  pies_from_apples 27 = 6 := by
  sorry

end rosie_pies_from_27_apples_l2890_289074


namespace tuesday_lesson_duration_is_one_hour_l2890_289020

/-- Represents the duration of each lesson on Tuesday in hours -/
def tuesday_lesson_duration : ℝ := 1

/-- The total number of hours Adam spent at school over the three days -/
def total_hours : ℝ := 12

/-- The number of lessons Adam had on Monday -/
def monday_lessons : ℕ := 6

/-- The duration of each lesson on Monday in hours -/
def monday_lesson_duration : ℝ := 0.5

/-- The number of lessons Adam had on Tuesday -/
def tuesday_lessons : ℕ := 3

/-- Theorem stating that the duration of each lesson on Tuesday is 1 hour -/
theorem tuesday_lesson_duration_is_one_hour :
  tuesday_lesson_duration = 1 ∧
  total_hours = (monday_lessons : ℝ) * monday_lesson_duration +
                (tuesday_lessons : ℝ) * tuesday_lesson_duration +
                2 * (tuesday_lessons : ℝ) * tuesday_lesson_duration :=
by sorry

end tuesday_lesson_duration_is_one_hour_l2890_289020


namespace early_arrival_l2890_289044

/-- Given a boy who usually takes 14 minutes to reach school, if he walks at 7/6 of his usual rate, he will arrive 2 minutes early. -/
theorem early_arrival (usual_time : ℝ) (new_rate : ℝ) : 
  usual_time = 14 → new_rate = 7/6 → usual_time - (usual_time / new_rate) = 2 :=
by sorry

end early_arrival_l2890_289044


namespace infinitely_many_squarefree_n_squared_plus_one_l2890_289075

/-- A natural number is squarefree if it has no repeated prime factors -/
def IsSquarefree (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ^ 2 ∣ n → p = 2)

/-- The set of positive integers n for which n^2 + 1 is squarefree -/
def SquarefreeSet : Set ℕ := {n : ℕ | n > 0 ∧ IsSquarefree (n^2 + 1)}

theorem infinitely_many_squarefree_n_squared_plus_one : Set.Infinite SquarefreeSet := by
  sorry

end infinitely_many_squarefree_n_squared_plus_one_l2890_289075


namespace thanksgiving_to_christmas_l2890_289094

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by a given number of days
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDays d m)

theorem thanksgiving_to_christmas (thanksgiving : DayOfWeek) :
  thanksgiving = DayOfWeek.Thursday →
  advanceDays thanksgiving 29 = DayOfWeek.Friday :=
by sorry

#check thanksgiving_to_christmas

end thanksgiving_to_christmas_l2890_289094


namespace linear_function_not_in_second_quadrant_l2890_289032

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear function y = kx + b -/
structure LinearFunction where
  k : ℝ
  b : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: A linear function with k > 0 and b < 0 does not pass through the second quadrant -/
theorem linear_function_not_in_second_quadrant (f : LinearFunction) 
    (h1 : f.k > 0) (h2 : f.b < 0) : 
    ∀ p : Point, p.y = f.k * p.x + f.b → ¬(isInSecondQuadrant p) := by
  sorry

end linear_function_not_in_second_quadrant_l2890_289032


namespace congruence_problem_l2890_289005

theorem congruence_problem (N : ℕ) (h1 : N > 1) 
  (h2 : 69 ≡ 90 [MOD N]) (h3 : 90 ≡ 125 [MOD N]) : 
  81 ≡ 4 [MOD N] := by
  sorry

end congruence_problem_l2890_289005


namespace mexica_numbers_less_than_2019_l2890_289002

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- A natural number is mexica if it's of the form n^(d(n)) -/
def is_mexica (m : ℕ) : Prop :=
  ∃ n : ℕ+, m = n.val ^ (d n)

/-- The set of mexica numbers less than 2019 -/
def mexica_set : Finset ℕ :=
  {1, 4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849, 64, 1296}

theorem mexica_numbers_less_than_2019 :
  {m : ℕ | is_mexica m ∧ m < 2019} = mexica_set := by sorry

end mexica_numbers_less_than_2019_l2890_289002


namespace min_y_value_l2890_289073

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 26*y) : 
  ∃ (y_min : ℝ), y_min = 13 - Real.sqrt 269 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 20*x' + 26*y' → y' ≥ y_min := by
sorry

end min_y_value_l2890_289073


namespace total_weight_lifted_l2890_289068

def weight_per_hand : ℕ := 8
def number_of_hands : ℕ := 2

theorem total_weight_lifted : weight_per_hand * number_of_hands = 16 := by
  sorry

end total_weight_lifted_l2890_289068


namespace alex_score_l2890_289009

-- Define the total number of shots
def total_shots : ℕ := 40

-- Define the success rates
def three_point_success_rate : ℚ := 1/4
def two_point_success_rate : ℚ := 1/5

-- Define the point values
def three_point_value : ℕ := 3
def two_point_value : ℕ := 2

-- Theorem statement
theorem alex_score :
  ∀ x y : ℕ,
  x + y = total_shots →
  (x : ℚ) * three_point_success_rate * three_point_value +
  (y : ℚ) * two_point_success_rate * two_point_value = 30 :=
by
  sorry


end alex_score_l2890_289009


namespace mary_hourly_wage_l2890_289077

def mary_long_day_hours : ℕ := 9
def mary_short_day_hours : ℕ := 5
def mary_long_days_per_week : ℕ := 3
def mary_short_days_per_week : ℕ := 2
def mary_weekly_earnings : ℕ := 407

def mary_total_weekly_hours : ℕ :=
  mary_long_day_hours * mary_long_days_per_week +
  mary_short_day_hours * mary_short_days_per_week

theorem mary_hourly_wage :
  mary_weekly_earnings / mary_total_weekly_hours = 11 := by
  sorry

end mary_hourly_wage_l2890_289077


namespace octahedron_construction_count_l2890_289097

/-- The number of faces in a regular octahedron -/
def octahedron_faces : ℕ := 8

/-- The number of distinct colored triangles available -/
def available_colors : ℕ := 9

/-- The number of rotational symmetries around a fixed face of an octahedron -/
def rotational_symmetries : ℕ := 3

/-- The number of distinguishable ways to construct a regular octahedron -/
def distinguishable_constructions : ℕ := 13440

theorem octahedron_construction_count :
  (Nat.choose available_colors (octahedron_faces - 1)) * 
  (Nat.factorial (octahedron_faces - 1)) / 
  rotational_symmetries = distinguishable_constructions := by
  sorry

end octahedron_construction_count_l2890_289097


namespace tv_cost_l2890_289093

theorem tv_cost (original_savings : ℝ) (furniture_fraction : ℚ) (tv_cost : ℝ) : 
  original_savings = 3000.0000000000005 →
  furniture_fraction = 5/6 →
  tv_cost = original_savings * (1 - furniture_fraction) →
  tv_cost = 500.0000000000001 := by
sorry

end tv_cost_l2890_289093


namespace union_of_sets_l2890_289071

theorem union_of_sets (a : ℝ) : 
  let A : Set ℝ := {x | x ≥ 0}
  let B : Set ℝ := {x | x ≤ a}
  (A ∪ B = Set.univ) → a ≥ 0 := by
  sorry

end union_of_sets_l2890_289071


namespace range_of_expression_l2890_289065

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem range_of_expression (a b : ℕ) 
  (ha : isPrime a ∧ 49 < a ∧ a < 61) 
  (hb : isPrime b ∧ 59 < b ∧ b < 71) : 
  -297954 ≤ (a^2 : ℤ) - b^3 ∧ (a^2 : ℤ) - b^3 ≤ -223500 :=
sorry

end range_of_expression_l2890_289065


namespace max_value_of_f_l2890_289081

def f (x : ℝ) : ℝ := -x^2 + 2*x + 8

theorem max_value_of_f :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l2890_289081


namespace range_of_x_range_of_a_l2890_289054

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 ≥ 0

-- Part 1
theorem range_of_x (x : ℝ) (h : p 1 x ∧ q x) : 2 ≤ x ∧ x < 3 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, ¬(p a x) → ¬(q x)) 
  (h3 : ∃ x, q x ∧ p a x) : 
  1 < a ∧ a < 2 :=
sorry

end range_of_x_range_of_a_l2890_289054


namespace max_value_constraint_l2890_289027

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 3 + 2 * a * c ≤ 8/5 := by
  sorry

end max_value_constraint_l2890_289027


namespace not_all_equilateral_triangles_congruent_l2890_289064

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for equilateral triangles
def congruent (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side_length = t2.side_length

-- Theorem statement
theorem not_all_equilateral_triangles_congruent :
  ∃ (t1 t2 : EquilateralTriangle), ¬(congruent t1 t2) :=
sorry

end not_all_equilateral_triangles_congruent_l2890_289064


namespace fred_cantelopes_count_l2890_289058

/-- The number of cantelopes grown by Fred and Tim together -/
def total_cantelopes : ℕ := 82

/-- The number of cantelopes grown by Tim -/
def tim_cantelopes : ℕ := 44

/-- The number of cantelopes grown by Fred -/
def fred_cantelopes : ℕ := total_cantelopes - tim_cantelopes

theorem fred_cantelopes_count : fred_cantelopes = 38 := by
  sorry

end fred_cantelopes_count_l2890_289058


namespace preimage_of_20_l2890_289085

def f (n : ℕ) : ℕ := 2^n + n

theorem preimage_of_20 : ∃! n : ℕ, f n = 20 ∧ n = 4 := by sorry

end preimage_of_20_l2890_289085


namespace vector_decomposition_l2890_289014

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![(-2), 4, 7]
def p : Fin 3 → ℝ := ![0, 1, 2]
def q : Fin 3 → ℝ := ![1, 0, 1]
def r : Fin 3 → ℝ := ![(-1), 2, 4]

/-- Theorem: x can be decomposed as 2p - q + r -/
theorem vector_decomposition :
  x = fun i => 2 * p i - q i + r i := by sorry

end vector_decomposition_l2890_289014


namespace reflection_over_x_axis_l2890_289049

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]

def reflects_over_x_axis (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ (x y : ℝ), M.mulVec ![x, y] = ![x, -y]

theorem reflection_over_x_axis :
  reflects_over_x_axis reflection_matrix := by sorry

end reflection_over_x_axis_l2890_289049


namespace quadratic_function_property_l2890_289076

/-- Given a quadratic function f(x) = x^2 + ax + b, if f(1) = 0 and f(2) = 0, then f(-1) = 6 -/
theorem quadratic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  (f 1 = 0) → (f 2 = 0) → (f (-1) = 6) := by
sorry

end quadratic_function_property_l2890_289076


namespace positive_sum_and_product_iff_both_positive_l2890_289013

theorem positive_sum_and_product_iff_both_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end positive_sum_and_product_iff_both_positive_l2890_289013


namespace solution_value_l2890_289016

-- Define the function E
def E (a b c : ℝ) : ℝ := a * b^2 + c

-- State the theorem
theorem solution_value : ∃ a : ℝ, 2*a + E a 3 2 = 4 + E a 5 3 ∧ a = -5/14 := by
  sorry

end solution_value_l2890_289016


namespace sqrt_15_bounds_l2890_289055

theorem sqrt_15_bounds : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 := by sorry

end sqrt_15_bounds_l2890_289055


namespace joan_found_six_shells_l2890_289060

/-- The number of seashells Jessica found -/
def jessica_shells : ℕ := 8

/-- The total number of seashells Joan and Jessica found together -/
def total_shells : ℕ := 14

/-- The number of seashells Joan found -/
def joan_shells : ℕ := total_shells - jessica_shells

theorem joan_found_six_shells : joan_shells = 6 := by
  sorry

end joan_found_six_shells_l2890_289060


namespace smallest_rational_l2890_289018

theorem smallest_rational (a b c d : ℚ) (ha : a = 1) (hb : b = 0) (hc : c = -1/2) (hd : d = -3) :
  d < c ∧ c < b ∧ b < a :=
by sorry

end smallest_rational_l2890_289018


namespace point_inside_circle_range_l2890_289026

theorem point_inside_circle_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end point_inside_circle_range_l2890_289026


namespace equal_prob_without_mult_higher_prob_even_with_mult_l2890_289069

/-- Represents a calculator with basic operations -/
structure Calculator where
  /-- The current display value -/
  display : ℕ
  /-- Whether multiplication is available -/
  mult_available : Bool

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- Get the parity of a natural number -/
def getParity (n : ℕ) : Parity :=
  if n % 2 = 0 then Parity.Even else Parity.Odd

/-- The probability of getting an odd result after a sequence of operations -/
def probOddResult (c : Calculator) : ℝ :=
  sorry

theorem equal_prob_without_mult (c : Calculator) (h : c.mult_available = false) :
  probOddResult c = 1 / 2 :=
sorry

theorem higher_prob_even_with_mult (c : Calculator) (h : c.mult_available = true) :
  probOddResult c < 1 / 2 :=
sorry

end equal_prob_without_mult_higher_prob_even_with_mult_l2890_289069


namespace units_digit_of_power_of_six_l2890_289034

theorem units_digit_of_power_of_six (n : ℕ) : (6^n) % 10 = 6 := by
  sorry

end units_digit_of_power_of_six_l2890_289034


namespace pen_price_relationship_l2890_289061

/-- Represents the relationship between the number of pens and their selling price. -/
theorem pen_price_relationship (x y : ℝ) : 
  (∀ (box_pens : ℝ) (box_price : ℝ), box_pens = 10 ∧ box_price = 16 → 
    y = (box_price / box_pens) * x) → 
  y = 1.6 * x := by
  sorry

end pen_price_relationship_l2890_289061


namespace polynomial_coefficient_sum_l2890_289084

theorem polynomial_coefficient_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) : 
  (∀ x : ℝ, x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 3*x^2 - 2*x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -2 :=
by sorry

end polynomial_coefficient_sum_l2890_289084


namespace bronze_to_silver_ratio_l2890_289029

def total_watches : ℕ := 88
def silver_watches : ℕ := 20
def gold_watches : ℕ := 9

def bronze_watches : ℕ := total_watches - silver_watches - gold_watches

theorem bronze_to_silver_ratio :
  bronze_watches * 20 = silver_watches * 59 := by sorry

end bronze_to_silver_ratio_l2890_289029


namespace boat_distribution_problem_l2890_289021

/-- Represents the boat distribution problem from "Nine Chapters on the Mathematical Art" --/
theorem boat_distribution_problem (x : ℕ) : 
  (∀ (total_boats : ℕ) (large_boat_capacity : ℕ) (small_boat_capacity : ℕ) (total_students : ℕ),
    total_boats = 8 ∧ 
    large_boat_capacity = 6 ∧ 
    small_boat_capacity = 4 ∧ 
    total_students = 38 ∧ 
    x ≤ total_boats ∧
    x * small_boat_capacity + (total_boats - x) * large_boat_capacity = total_students) →
  4 * x + 6 * (8 - x) = 38 :=
by sorry

end boat_distribution_problem_l2890_289021


namespace square_division_theorem_l2890_289056

theorem square_division_theorem (x : ℝ) (h1 : x > 0) :
  (∃ l : ℝ, l > 0 ∧ 2 * l = x^2 / 5) →
  (∃ a : ℝ, a > 0 ∧ x * a = x^2 / 5) →
  x = 8 ∧ x^2 = 64 := by
  sorry

end square_division_theorem_l2890_289056
