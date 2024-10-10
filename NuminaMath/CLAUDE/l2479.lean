import Mathlib

namespace sin_inequality_implies_angle_inequality_sin_positive_in_first_and_second_quadrant_l2479_247921

-- Define the first and second quadrants
def first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2
def second_quadrant (θ : ℝ) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi

theorem sin_inequality_implies_angle_inequality (α β : ℝ) :
  Real.sin α ≠ Real.sin β → α ≠ β :=
sorry

theorem sin_positive_in_first_and_second_quadrant (θ : ℝ) :
  (first_quadrant θ ∨ second_quadrant θ) → Real.sin θ > 0 :=
sorry

end sin_inequality_implies_angle_inequality_sin_positive_in_first_and_second_quadrant_l2479_247921


namespace P_is_ellipse_l2479_247990

-- Define the set of points P(x,y) satisfying the given equation
def P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; Real.sqrt ((x + 4)^2 + y^2) + Real.sqrt ((x - 4)^2 + y^2) = 10}

-- Define an ellipse with foci at (-4, 0) and (4, 0), and sum of distances equal to 10
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; Real.sqrt ((x + 4)^2 + y^2) + Real.sqrt ((x - 4)^2 + y^2) = 10}

-- Theorem stating that the set P is equivalent to the Ellipse
theorem P_is_ellipse : P = Ellipse := by sorry

end P_is_ellipse_l2479_247990


namespace mean_proportional_segment_l2479_247944

theorem mean_proportional_segment (a b c : ℝ) : 
  a = 1 → b = 2 → c^2 = a * b → c > 0 → c = Real.sqrt 2 := by
  sorry

#check mean_proportional_segment

end mean_proportional_segment_l2479_247944


namespace inequality_proof_l2479_247938

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c ≤ 3) :
  a / (1 + a^2) + b / (1 + b^2) + c / (1 + c^2) ≤ 3/2 ∧ 3/2 ≤ 1/(1+a) + 1/(1+b) + 1/(1+c) := by
  sorry

end inequality_proof_l2479_247938


namespace m_range_l2479_247956

/-- Proposition p: The quadratic equation with real coefficients x^2 + mx + 2 = 0 has imaginary roots -/
def prop_p (m : ℝ) : Prop := m^2 - 8 < 0

/-- Proposition q: For the equation 2x^2 - 4(m-1)x + m^2 + 7 = 0 (m ∈ ℝ), 
    the sum of the moduli of its two imaginary roots does not exceed 4√2 -/
def prop_q (m : ℝ) : Prop := 16*(m-1)^2 - 8*(m^2 + 7) < 0

/-- The range of m when both propositions p and q are true -/
theorem m_range (m : ℝ) : prop_p m ∧ prop_q m ↔ -1 < m ∧ m < 2 * Real.sqrt 2 := by
  sorry

end m_range_l2479_247956


namespace third_face_area_l2479_247920

-- Define the properties of the cuboidal box
def cuboidal_box (l w h : ℝ) : Prop :=
  l > 0 ∧ w > 0 ∧ h > 0 ∧
  l * w = 72 ∧
  w * h = 60 ∧
  l * w * h = 720

-- Theorem statement
theorem third_face_area (l w h : ℝ) :
  cuboidal_box l w h → l * h = 120 := by
  sorry

end third_face_area_l2479_247920


namespace quadrilateral_area_l2479_247914

/-- The area of a quadrilateral with given sides and one angle -/
theorem quadrilateral_area (a b c d : Real) (α : Real) : 
  a = 52 →
  b = 56 →
  c = 33 →
  d = 39 →
  α = 112 + 37 / 60 + 12 / 3600 →
  ∃ (area : Real), abs (area - 1774) < 1 ∧ 
  area = (1/2) * a * d * Real.sin α + 
          Real.sqrt ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) * 
                     ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) - b) * 
                     ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) - c) * 
                     ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) - Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α))) :=
by
  sorry

end quadrilateral_area_l2479_247914


namespace water_ice_mixture_theorem_l2479_247935

/-- Represents the properties of water and ice mixture -/
structure WaterIceMixture where
  total_mass : ℝ
  water_mass : ℝ
  ice_mass : ℝ
  water_mass_added : ℝ
  initial_temp : ℝ
  final_temp : ℝ
  latent_heat_fusion : ℝ

/-- Calculates the heat balance for the water-ice mixture -/
def heat_balance (m : WaterIceMixture) : ℝ :=
  m.water_mass_added * (m.initial_temp - m.final_temp) -
  (m.ice_mass * m.latent_heat_fusion + m.total_mass * (m.final_temp - 0))

/-- Theorem stating that the original water mass in the mixture is 90.625g -/
theorem water_ice_mixture_theorem (m : WaterIceMixture) 
  (h1 : m.total_mass = 250)
  (h2 : m.water_mass_added = 1000)
  (h3 : m.initial_temp = 20)
  (h4 : m.final_temp = 5)
  (h5 : m.latent_heat_fusion = 80)
  (h6 : m.water_mass + m.ice_mass = m.total_mass)
  (h7 : heat_balance m = 0) :
  m.water_mass = 90.625 :=
sorry

end water_ice_mixture_theorem_l2479_247935


namespace combined_age_is_23_l2479_247985

/-- Represents the ages and relationships in the problem -/
structure AgeRelationship where
  person_age : ℕ
  dog_age : ℕ
  cat_age : ℕ
  sister_age : ℕ

/-- The conditions of the problem -/
def problem_conditions (ar : AgeRelationship) : Prop :=
  ar.person_age = ar.dog_age + 15 ∧
  ar.cat_age = ar.dog_age + 3 ∧
  ar.dog_age + 2 = 4 ∧
  ar.sister_age + 2 = 2 * (ar.dog_age + 2)

/-- The theorem to prove -/
theorem combined_age_is_23 (ar : AgeRelationship) 
  (h : problem_conditions ar) : 
  ar.person_age + ar.sister_age = 23 := by
  sorry

end combined_age_is_23_l2479_247985


namespace original_number_theorem_l2479_247916

theorem original_number_theorem (x : ℝ) : 
  12 * ((x * 0.5 - 10) / 6) = 15 → x = 35 := by
sorry

end original_number_theorem_l2479_247916


namespace max_value_of_sum_of_squares_l2479_247968

theorem max_value_of_sum_of_squares (x y : ℝ) :
  x^2 + y^2 = 3*x + 8*y → x^2 + y^2 ≤ 73 := by
  sorry

end max_value_of_sum_of_squares_l2479_247968


namespace rectangle_length_l2479_247918

theorem rectangle_length (l w : ℝ) (h1 : l = 4 * w) (h2 : l * w = 100) : l = 20 := by
  sorry

end rectangle_length_l2479_247918


namespace cuboid_surface_area_l2479_247909

/-- The surface area of a cuboid created by three cubes of side length 8 cm -/
theorem cuboid_surface_area : 
  let cube_side : ℝ := 8
  let cuboid_length : ℝ := 3 * cube_side
  let cuboid_width : ℝ := cube_side
  let cuboid_height : ℝ := cube_side
  let surface_area : ℝ := 2 * (cuboid_length * cuboid_width + 
                               cuboid_length * cuboid_height + 
                               cuboid_width * cuboid_height)
  surface_area = 896 := by
sorry


end cuboid_surface_area_l2479_247909


namespace translation_left_2_units_l2479_247912

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateLeft (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x - units, y := p.y }

/-- The problem statement -/
theorem translation_left_2_units :
  let P : Point2D := { x := -2, y := -1 }
  let A' : Point2D := translateLeft P 2
  A' = { x := -4, y := -1 } := by
  sorry

end translation_left_2_units_l2479_247912


namespace dividend_problem_l2479_247917

theorem dividend_problem (M D Q : ℕ) (h1 : M = 6 * D) (h2 : D = 4 * Q) : M = 144 := by
  sorry

end dividend_problem_l2479_247917


namespace collinear_points_on_cubic_curve_l2479_247945

/-- Three points on a cubic curve that are collinear satisfy a specific relation -/
theorem collinear_points_on_cubic_curve
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h_curve₁ : y₁^2 = x₁^3)
  (h_curve₂ : y₂^2 = x₂^3)
  (h_curve₃ : y₃^2 = x₃^3)
  (h_collinear : (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁))
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁)
  (h_nonzero : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0) :
  x₁ / y₁ + x₂ / y₂ + x₃ / y₃ = 0 := by
sorry

end collinear_points_on_cubic_curve_l2479_247945


namespace aiNK_probability_l2479_247960

/-- The number of distinct cards labeled with the letters of "NanKai" -/
def total_cards : ℕ := 6

/-- The number of cards drawn -/
def drawn_cards : ℕ := 4

/-- The number of ways to form "aiNK" from the drawn cards -/
def successful_outcomes : ℕ := 1

/-- The total number of ways to draw 4 cards from 6 -/
def total_outcomes : ℕ := Nat.choose total_cards drawn_cards

/-- The probability of drawing four cards that can form "aiNK" -/
def probability : ℚ := successful_outcomes / total_outcomes

theorem aiNK_probability : probability = 1 / 15 := by
  sorry

end aiNK_probability_l2479_247960


namespace factor_implies_c_value_l2479_247901

def f (c x : ℝ) : ℝ := c*x^4 + 15*x^3 - 5*c*x^2 - 45*x + 55

theorem factor_implies_c_value (c : ℝ) :
  (∀ x : ℝ, (x + 5) ∣ f c x) → c = 319 / 100 := by
  sorry

end factor_implies_c_value_l2479_247901


namespace complex_equality_proof_l2479_247965

theorem complex_equality_proof (n : ℤ) (h : 0 ≤ n ∧ n ≤ 13) : 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.cos (2 * n * π / 14) + Complex.I * Complex.sin (2 * n * π / 14) → n = 5 :=
by sorry

end complex_equality_proof_l2479_247965


namespace angle_from_terminal_point_l2479_247998

/-- Given an angle α in degrees where 0 ≤ α < 360, if a point on its terminal side
    has coordinates (sin 150°, cos 150°), then α = 300°. -/
theorem angle_from_terminal_point : ∀ α : ℝ,
  0 ≤ α → α < 360 →
  (∃ (x y : ℝ), x = Real.sin (150 * π / 180) ∧ y = Real.cos (150 * π / 180) ∧
    x = Real.sin (α * π / 180) ∧ y = Real.cos (α * π / 180)) →
  α = 300 := by
  sorry

end angle_from_terminal_point_l2479_247998


namespace toy_store_spending_l2479_247949

/-- Proof of student's spending at toy store -/
theorem toy_store_spending (total_allowance : ℚ) 
  (arcade_fraction : ℚ) (candy_spending : ℚ) :
  total_allowance = 4.5 →
  arcade_fraction = 3/5 →
  candy_spending = 1.2 →
  let remaining_after_arcade := total_allowance - (arcade_fraction * total_allowance)
  let toy_store_spending := remaining_after_arcade - candy_spending
  toy_store_spending / remaining_after_arcade = 1/3 := by
  sorry

end toy_store_spending_l2479_247949


namespace factorization_1_l2479_247963

theorem factorization_1 (a : ℝ) : 3*a^3 - 6*a^2 + 3*a = 3*a*(a - 1)^2 := by
  sorry

end factorization_1_l2479_247963


namespace triangle_height_inradius_inequality_l2479_247983

theorem triangle_height_inradius_inequality 
  (h₁ h₂ h₃ r : ℝ) (α : ℝ) 
  (h_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ r > 0)
  (h_heights : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    h₁ = 2 * (r * (a + b + c) / 2) / a ∧
    h₂ = 2 * (r * (a + b + c) / 2) / b ∧
    h₃ = 2 * (r * (a + b + c) / 2) / c)
  (h_alpha : α ≥ 1) :
  h₁^α + h₂^α + h₃^α ≥ 3 * (3 * r)^α := by
  sorry

end triangle_height_inradius_inequality_l2479_247983


namespace coffee_price_increase_percentage_l2479_247913

def first_quarter_price : ℝ := 40
def fourth_quarter_price : ℝ := 60

theorem coffee_price_increase_percentage : 
  (fourth_quarter_price - first_quarter_price) / first_quarter_price * 100 = 50 := by
  sorry

end coffee_price_increase_percentage_l2479_247913


namespace range_of_a_minus_abs_b_l2479_247971

-- Define the conditions
def a_condition (a : ℝ) : Prop := 1 < a ∧ a < 3
def b_condition (b : ℝ) : Prop := -4 < b ∧ b < 2

-- Define the range of a - |b|
def range_a_minus_abs_b (x : ℝ) : Prop :=
  ∃ (a b : ℝ), a_condition a ∧ b_condition b ∧ x = a - |b|

-- Theorem statement
theorem range_of_a_minus_abs_b :
  ∀ x, range_a_minus_abs_b x ↔ -3 < x ∧ x < 3 :=
sorry

end range_of_a_minus_abs_b_l2479_247971


namespace sphere_surface_area_l2479_247930

/-- Given a sphere with an inscribed triangle on its section, prove its surface area --/
theorem sphere_surface_area (a b c : ℝ) (r R : ℝ) : 
  a = 6 → b = 8 → c = 10 →  -- Triangle side lengths
  r = 5 →  -- Radius of section's circle
  R^2 - (R/2)^2 = r^2 →  -- Relation between sphere radius and section
  4 * π * R^2 = 400 * π / 3 := by
  sorry

#check sphere_surface_area

end sphere_surface_area_l2479_247930


namespace all_acute_triangle_count_l2479_247982

/-- A function that checks if a triangle with sides a, b, c has all acute angles -/
def isAllAcuteTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧
  a * a + b * b > c * c ∧
  a * a + c * c > b * b ∧
  b * b + c * c > a * a

/-- The theorem stating that there are exactly 5 integer values of y that form an all-acute triangle with sides 15 and 8 -/
theorem all_acute_triangle_count :
  ∃! (s : Finset ℕ), s.card = 5 ∧ ∀ y ∈ s, isAllAcuteTriangle 15 8 y :=
sorry

end all_acute_triangle_count_l2479_247982


namespace batsman_average_increase_l2479_247951

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalScore : Nat
  notOutCount : Nat

/-- Calculate the batting average -/
def battingAverage (b : Batsman) : Rat :=
  b.totalScore / (b.innings - b.notOutCount)

/-- The increase in average after a new innings -/
def averageIncrease (b : Batsman) (newScore : Nat) : Rat :=
  battingAverage { innings := b.innings + 1, totalScore := b.totalScore + newScore, notOutCount := b.notOutCount } -
  battingAverage b

/-- Theorem: The batsman's average increase is 2 runs -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 11 →
    b.notOutCount = 0 →
    (battingAverage { innings := b.innings + 1, totalScore := b.totalScore + 80, notOutCount := b.notOutCount } = 58) →
    averageIncrease b 80 = 2 :=
by sorry

end batsman_average_increase_l2479_247951


namespace arithmetic_sequence_sum_l2479_247936

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: For an arithmetic sequence with first term a₁ and common difference d,
    if S₈ - S₃ = 20, then S₁₁ = 44 -/
theorem arithmetic_sequence_sum (a₁ d : ℚ) :
  S 8 a₁ d - S 3 a₁ d = 20 → S 11 a₁ d = 44 := by
  sorry

end arithmetic_sequence_sum_l2479_247936


namespace james_music_beats_l2479_247964

/-- Calculate the number of beats heard in a week given the beats per minute,
    hours of listening per day, and days in a week. -/
def beats_per_week (beats_per_minute : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  beats_per_minute * 60 * hours_per_day * days_per_week

/-- Theorem stating that listening to 200 beats per minute music for 2 hours a day
    for 7 days results in hearing 168,000 beats in a week. -/
theorem james_music_beats :
  beats_per_week 200 2 7 = 168000 := by
  sorry

end james_music_beats_l2479_247964


namespace product_cost_change_l2479_247902

theorem product_cost_change (initial_cost : ℝ) (h : initial_cost > 0) : 
  initial_cost * (1 + 0.2)^2 * (1 - 0.2)^2 < initial_cost := by
  sorry

end product_cost_change_l2479_247902


namespace diamond_three_four_l2479_247953

def diamond (a b : ℝ) : ℝ := 4 * a + 3 * b - 2 * a * b

theorem diamond_three_four : diamond 3 4 = 0 := by
  sorry

end diamond_three_four_l2479_247953


namespace bank_interest_rate_problem_l2479_247970

theorem bank_interest_rate_problem
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 2200)
  (h2 : additional_investment = 1099.9999999999998)
  (h3 : additional_rate = 0.08)
  (h4 : total_rate = 0.06)
  (h5 : total_rate * (initial_investment + additional_investment) =
        initial_investment * x + additional_investment * additional_rate) :
  x = 0.05 :=
by sorry

end bank_interest_rate_problem_l2479_247970


namespace school_emblem_estimate_l2479_247969

/-- Estimates the number of students who like a design in the entire school population
    based on a sample survey. -/
def estimate_liking (total_students : ℕ) (sample_size : ℕ) (sample_liking : ℕ) : ℕ :=
  (sample_liking * total_students) / sample_size

/-- Theorem stating that the estimated number of students liking design A
    in a school of 2000 students is 1200, given a survey where 60 out of 100
    students liked design A. -/
theorem school_emblem_estimate :
  let total_students : ℕ := 2000
  let sample_size : ℕ := 100
  let sample_liking : ℕ := 60
  estimate_liking total_students sample_size sample_liking = 1200 := by
sorry

end school_emblem_estimate_l2479_247969


namespace planks_per_tree_value_l2479_247989

/-- The number of planks John can make from each tree -/
def planks_per_tree : ℕ := sorry

/-- The number of trees John chops down -/
def num_trees : ℕ := 30

/-- The number of planks needed to make one table -/
def planks_per_table : ℕ := 15

/-- The selling price of one table in dollars -/
def table_price : ℕ := 300

/-- The total labor cost in dollars -/
def labor_cost : ℕ := 3000

/-- The total profit in dollars -/
def total_profit : ℕ := 12000

/-- Theorem stating the number of planks John can make from each tree -/
theorem planks_per_tree_value : planks_per_tree = 25 := by sorry

end planks_per_tree_value_l2479_247989


namespace largest_number_l2479_247919

def a : ℚ := 24680 + 1 / 1357
def b : ℚ := 24680 - 1 / 1357
def c : ℚ := 24680 * (1 / 1357)
def d : ℚ := 24680 / (1 / 1357)
def e : ℚ := 24680.1357

theorem largest_number : 
  d > a ∧ d > b ∧ d > c ∧ d > e :=
sorry

end largest_number_l2479_247919


namespace factorial_loop_condition_l2479_247900

/-- A function that calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

/-- The theorem stating that if a factorial program outputs 720,
    then the loop condition must be i <= 6 -/
theorem factorial_loop_condition (output : ℕ) (loop_condition : ℕ → Bool) :
  output = 720 →
  (∀ n : ℕ, factorial n = output → loop_condition = fun i => i ≤ n) →
  loop_condition = fun i => i ≤ 6 :=
sorry

end factorial_loop_condition_l2479_247900


namespace largest_perfect_square_factor_of_1980_l2479_247992

theorem largest_perfect_square_factor_of_1980 : 
  ∃ (n : ℕ), n^2 = 36 ∧ n^2 ∣ 1980 ∧ ∀ (m : ℕ), m^2 ∣ 1980 → m^2 ≤ 36 := by
  sorry

end largest_perfect_square_factor_of_1980_l2479_247992


namespace semicircles_to_circle_area_ratio_l2479_247995

theorem semicircles_to_circle_area_ratio (r : ℝ) (hr : r > 0) : 
  (2 * (π * r^2 / 2)) / (π * r^2) = 1 := by sorry

end semicircles_to_circle_area_ratio_l2479_247995


namespace cone_vertex_angle_l2479_247937

theorem cone_vertex_angle (α β αf : Real) : 
  β = 2 * Real.arcsin (1/4) →
  2 * α = αf →
  2 * α = Real.pi/6 + Real.arcsin (1/4) :=
by sorry

end cone_vertex_angle_l2479_247937


namespace quadratic_solution_implies_a_greater_than_one_l2479_247924

/-- Represents a quadratic function of the form f(x) = 2ax^2 - x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

/-- Condition for exactly one solution in (0, 1) -/
def has_exactly_one_solution_in_interval (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo 0 1 ∧ f a x = 0

theorem quadratic_solution_implies_a_greater_than_one :
  ∀ a : ℝ, has_exactly_one_solution_in_interval a → a > 1 := by
  sorry

end quadratic_solution_implies_a_greater_than_one_l2479_247924


namespace dealer_profit_percentage_l2479_247981

/-- The profit percentage of a dealer who sells 900 grams of goods for the price of 1000 grams -/
theorem dealer_profit_percentage : 
  let actual_weight : ℝ := 900
  let claimed_weight : ℝ := 1000
  let profit_percentage := (claimed_weight / actual_weight - 1) * 100
  profit_percentage = (1 / 9) * 100 := by sorry

end dealer_profit_percentage_l2479_247981


namespace parabola_parameter_l2479_247943

/-- Given a circle C₁ and a parabola C₂ intersecting at two points with a specific chord length,
    prove that the parameter of the parabola has a specific value. -/
theorem parabola_parameter (p : ℝ) (h_p : p > 0) : 
  ∃ A B : ℝ × ℝ,
    (A.1^2 + (A.2 - 2)^2 = 4) ∧ 
    (B.1^2 + (B.2 - 2)^2 = 4) ∧
    (A.2^2 = 2*p*A.1) ∧ 
    (B.2^2 = 2*p*B.1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (8*Real.sqrt 5/5)^2) →
    p = 16/5 := by
  sorry

end parabola_parameter_l2479_247943


namespace total_pins_used_l2479_247987

/-- The number of sides of a rectangle -/
def rectangle_sides : ℕ := 4

/-- The number of pins used on each side of the cardboard -/
def pins_per_side : ℕ := 35

/-- Theorem: The total number of pins used to attach a rectangular cardboard to a box -/
theorem total_pins_used : rectangle_sides * pins_per_side = 140 := by
  sorry

end total_pins_used_l2479_247987


namespace polyhedron_edges_existence_l2479_247928

/-- The number of edges in the initial polyhedra we can start with -/
def initial_edges : List Nat := [8, 9, 10]

/-- The number of edges added when slicing off a triangular angle -/
def edges_per_slice : Nat := 3

/-- Proposition: For any natural number n ≥ 8, there exists a polyhedron with exactly n edges -/
theorem polyhedron_edges_existence (n : Nat) (h : n ≥ 8) :
  ∃ (k : Nat) (m : Nat), k ∈ initial_edges ∧ n = k + m * edges_per_slice :=
sorry

end polyhedron_edges_existence_l2479_247928


namespace second_bucket_capacity_l2479_247962

/-- Proves that given a tank of 48 liters and two buckets, where one bucket has a capacity of 4 liters
    and is used 4 times less than the other bucket to fill the tank, the capacity of the second bucket is 3 liters. -/
theorem second_bucket_capacity
  (tank_capacity : ℕ)
  (first_bucket_capacity : ℕ)
  (usage_difference : ℕ)
  (h1 : tank_capacity = 48)
  (h2 : first_bucket_capacity = 4)
  (h3 : usage_difference = 4)
  (h4 : ∃ (second_bucket_capacity : ℕ),
    tank_capacity / first_bucket_capacity = tank_capacity / second_bucket_capacity - usage_difference) :
  ∃ (second_bucket_capacity : ℕ), second_bucket_capacity = 3 :=
by sorry

end second_bucket_capacity_l2479_247962


namespace symbol_set_has_14_plus_l2479_247933

/-- A set of symbols consisting of plus and minus signs -/
structure SymbolSet where
  total : ℕ
  plus : ℕ
  minus : ℕ
  sum_eq : total = plus + minus
  plus_constraint : ∀ (n : ℕ), n ≤ total - plus → n < 10
  minus_constraint : ∀ (n : ℕ), n ≤ total - minus → n < 15

/-- The theorem stating that a SymbolSet with 23 total symbols has 14 plus signs -/
theorem symbol_set_has_14_plus (s : SymbolSet) (h : s.total = 23) : s.plus = 14 := by
  sorry

end symbol_set_has_14_plus_l2479_247933


namespace sin_negative_330_degrees_l2479_247903

theorem sin_negative_330_degrees : Real.sin ((-330 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end sin_negative_330_degrees_l2479_247903


namespace shirts_to_wash_l2479_247977

def washing_machine_capacity : ℕ := 7
def number_of_sweaters : ℕ := 33
def number_of_loads : ℕ := 5

theorem shirts_to_wash (shirts : ℕ) : 
  shirts = number_of_loads * washing_machine_capacity - number_of_sweaters :=
by sorry

end shirts_to_wash_l2479_247977


namespace unimodal_peak_interval_peak_interval_length_specific_peak_interval_l2479_247973

/-- A unimodal function on [0,1] is a function that is monotonically increasing
    on [0,x*] and monotonically decreasing on [x*,1] for some x* in (0,1) -/
def UnimodalFunction (f : ℝ → ℝ) : Prop := 
  ∃ x_star : ℝ, 0 < x_star ∧ x_star < 1 ∧ 
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ x_star → f x ≤ f y) ∧
  (∀ x y : ℝ, x_star ≤ x ∧ x < y ∧ y ≤ 1 → f x ≥ f y)

/-- The peak interval of a unimodal function contains the peak point -/
def PeakInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  UnimodalFunction f ∧ 0 ≤ a ∧ b ≤ 1 ∧
  ∃ x_star : ℝ, a < x_star ∧ x_star < b ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ x_star → f x ≤ f y) ∧
  (∀ x y : ℝ, x_star ≤ x ∧ x < y ∧ y ≤ 1 → f x ≥ f y)

theorem unimodal_peak_interval 
  (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h_unimodal : UnimodalFunction f)
  (h_x₁ : 0 < x₁) (h_x₂ : x₂ < 1) (h_order : x₁ < x₂) :
  (f x₁ ≥ f x₂ → PeakInterval f 0 x₂) ∧
  (f x₁ ≤ f x₂ → PeakInterval f x₁ 1) := by sorry

theorem peak_interval_length 
  (f : ℝ → ℝ) (r : ℝ) 
  (h_unimodal : UnimodalFunction f)
  (h_r : 0 < r ∧ r < 0.5) :
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₂ < 1 ∧ x₁ < x₂ ∧ x₂ - x₁ ≥ 2*r ∧
  ((PeakInterval f 0 x₂ ∧ x₂ ≤ 0.5 + r) ∨
   (PeakInterval f x₁ 1 ∧ 1 - x₁ ≤ 0.5 + r)) := by sorry

theorem specific_peak_interval 
  (f : ℝ → ℝ) 
  (h_unimodal : UnimodalFunction f) :
  ∃ x₁ x₂ x₃ : ℝ, 
    x₁ = 0.34 ∧ x₂ = 0.66 ∧ x₃ = 0.32 ∧
    PeakInterval f 0 x₂ ∧
    PeakInterval f 0 x₁ ∧
    |x₁ - x₂| ≥ 0.02 ∧ |x₁ - x₃| ≥ 0.02 ∧ |x₂ - x₃| ≥ 0.02 := by sorry

end unimodal_peak_interval_peak_interval_length_specific_peak_interval_l2479_247973


namespace symmetric_equation_example_symmetric_equation_values_quadratic_equation_solutions_l2479_247999

-- Definition of symmetric equations
def is_symmetric (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₁ + a₂ = 0 ∧ b₁ = b₂ ∧ c₁ + c₂ = 0

-- Theorem 1: Symmetric equation of x² - 4x + 3 = 0
theorem symmetric_equation_example : 
  is_symmetric 1 (-4) 3 (-1) (-4) (-3) :=
sorry

-- Theorem 2: Finding m and n for symmetric equations
theorem symmetric_equation_values (m n : ℝ) :
  is_symmetric 3 (m - 1) (-n) (-3) (-1) 1 → m = 0 ∧ n = 1 :=
sorry

-- Theorem 3: Solutions of the quadratic equation
theorem quadratic_equation_solutions :
  let x₁ := (1 + Real.sqrt 13) / 6
  let x₂ := (1 - Real.sqrt 13) / 6
  3 * x₁^2 - x₁ - 1 = 0 ∧ 3 * x₂^2 - x₂ - 1 = 0 :=
sorry

end symmetric_equation_example_symmetric_equation_values_quadratic_equation_solutions_l2479_247999


namespace problem_statement_l2479_247996

theorem problem_statement (x y : ℝ) (h : 3 * y - x^2 = -5) :
  6 * y - 2 * x^2 - 6 = -16 := by
sorry

end problem_statement_l2479_247996


namespace parabola_opens_upwards_l2479_247908

/-- For a parabola y = (2-m)x^2 + 1 to open upwards, m must be less than 2 -/
theorem parabola_opens_upwards (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (2 - m) * x^2 + 1) → 
  (∀ a b : ℝ, a < b → ((2 - m) * a^2 + 1) < ((2 - m) * b^2 + 1)) →
  m < 2 := by
sorry

end parabola_opens_upwards_l2479_247908


namespace product_zero_l2479_247927

theorem product_zero (b : ℤ) (h : b = 4) : 
  (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end product_zero_l2479_247927


namespace car_insurance_present_value_l2479_247904

/-- Calculate the present value of a series of payments with annual growth and inflation --/
theorem car_insurance_present_value
  (initial_payment : ℝ)
  (insurance_growth_rate : ℝ)
  (inflation_rate : ℝ)
  (years : ℕ)
  (h1 : initial_payment = 3000)
  (h2 : insurance_growth_rate = 0.05)
  (h3 : inflation_rate = 0.02)
  (h4 : years = 10) :
  ∃ (pv : ℝ), abs (pv - ((initial_payment * ((1 + insurance_growth_rate) ^ years - 1) / insurance_growth_rate) / (1 + inflation_rate) ^ years)) < 0.01 ∧ 
  30954.87 < pv ∧ pv < 30954.89 :=
by
  sorry

end car_insurance_present_value_l2479_247904


namespace smallest_valid_m_l2479_247961

def is_valid (m : ℕ+) : Prop :=
  ∃ k₁ k₂ : ℕ, k₁ ≤ m ∧ k₂ ≤ m ∧ 
  (m^2 + m) % k₁ = 0 ∧ 
  (m^2 + m) % k₂ ≠ 0

theorem smallest_valid_m :
  (∀ m : ℕ+, m < 4 → ¬(is_valid m)) ∧ 
  is_valid 4 := by sorry

end smallest_valid_m_l2479_247961


namespace function_range_l2479_247994

theorem function_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*m*x + m + 2 = 0 ∧ y^2 - 2*m*y + m + 2 = 0) ∧ 
  (∀ x ≥ 1, ∀ y ≥ x, (y^2 - 2*m*y + m + 2) ≥ (x^2 - 2*m*x + m + 2)) →
  m < -1 :=
by sorry

end function_range_l2479_247994


namespace field_trip_students_l2479_247907

theorem field_trip_students (adult_chaperones : ℕ) (student_fee adult_fee total_cost : ℚ) : 
  adult_chaperones = 4 →
  student_fee = 5 →
  adult_fee = 6 →
  total_cost = 199 →
  ∃ (num_students : ℕ), (num_students : ℚ) * student_fee + (adult_chaperones : ℚ) * adult_fee = total_cost ∧ 
    num_students = 35 := by
  sorry

end field_trip_students_l2479_247907


namespace factorization_proof_l2479_247984

theorem factorization_proof (x y : ℝ) : x * y^2 - x = x * (y + 1) * (y - 1) := by
  sorry

end factorization_proof_l2479_247984


namespace solve_for_A_l2479_247942

theorem solve_for_A (x : ℝ) (A : ℝ) (h : (5 : ℝ) / (x + 1) = A - ((2 * x - 3) / (x + 1))) :
  A = 2 := by
  sorry

end solve_for_A_l2479_247942


namespace parabola_c_value_l2479_247925

/-- A parabola with equation x = ay² + by + c, vertex at (5, -3), and passing through (7, 1) has c = 49/8 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ y : ℝ, 5 = a * (-3)^2 + b * (-3) + c) →  -- vertex condition
  (7 = a * 1^2 + b * 1 + c) →                 -- point condition
  (c = 49/8) := by
  sorry

end parabola_c_value_l2479_247925


namespace group_size_proof_l2479_247905

/-- The number of men in a group where replacing one man increases the average weight by 2.5 kg, 
    and the difference between the new man's weight and the replaced man's weight is 25 kg. -/
def number_of_men : ℕ := 10

/-- The increase in average weight when one man is replaced. -/
def average_weight_increase : ℚ := 5/2

/-- The difference in weight between the new man and the replaced man. -/
def weight_difference : ℕ := 25

theorem group_size_proof : 
  number_of_men * average_weight_increase = weight_difference := by
  sorry

end group_size_proof_l2479_247905


namespace jacoby_lottery_ticket_cost_l2479_247947

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def sisters_count : ℕ := 2
def remaining_needed : ℕ := 3214

theorem jacoby_lottery_ticket_cost :
  let job_earnings := hourly_wage * hours_worked
  let cookie_earnings := cookie_price * cookies_sold
  let gifts := sister_gift * sisters_count
  let total_earned := job_earnings + cookie_earnings + lottery_winnings + gifts
  let actual_total := trip_cost - remaining_needed
  total_earned - actual_total = 10
  := by sorry

end jacoby_lottery_ticket_cost_l2479_247947


namespace expression_equality_l2479_247978

theorem expression_equality : 
  (2^3 ≠ 3^2) ∧ 
  ((-2)^3 ≠ (-3)^2) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  ((-2)^3 = (-2^3)) := by
  sorry

end expression_equality_l2479_247978


namespace quadratic_roots_conditions_l2479_247950

variable (m : ℝ)

def quadratic_equation (x : ℝ) := m * x^2 + (m - 3) * x + 1

theorem quadratic_roots_conditions :
  (∀ x, quadratic_equation m x ≠ 0 → m > 1) ∧
  ((∃ x y, x ≠ y ∧ x > 0 ∧ y > 0 ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) ↔ 0 < m ∧ m < 1) ∧
  ((∃ x y, x > 0 ∧ y < 0 ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) ↔ m < 0) := by
  sorry

end quadratic_roots_conditions_l2479_247950


namespace negative_rational_function_interval_l2479_247967

theorem negative_rational_function_interval (x : ℝ) :
  x ≠ 3 →
  ((x - 5) / ((x - 3)^2) < 0) ↔ (3 < x ∧ x < 5) :=
by sorry

end negative_rational_function_interval_l2479_247967


namespace cubic_expansion_property_l2479_247975

theorem cubic_expansion_property (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x, (Real.sqrt 3 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -8 := by
  sorry

end cubic_expansion_property_l2479_247975


namespace halloween_candy_duration_l2479_247946

/-- Calculates the number of full days candy will last given initial amounts, trades, losses, and daily consumption. -/
def candy_duration (neighbors : ℕ) (sister : ℕ) (traded : ℕ) (lost : ℕ) (daily_consumption : ℕ) : ℕ :=
  ((neighbors + sister - traded - lost) / daily_consumption : ℕ)

/-- Theorem stating that under the given conditions, the candy will last for 23 full days. -/
theorem halloween_candy_duration :
  candy_duration 75 130 25 15 7 = 23 := by
  sorry

end halloween_candy_duration_l2479_247946


namespace floor_sum_opposite_l2479_247948

theorem floor_sum_opposite (x : ℝ) (h : x = 15.8) : 
  ⌊x⌋ + ⌊-x⌋ = -1 := by sorry

end floor_sum_opposite_l2479_247948


namespace hyperbola_ellipse_foci_l2479_247929

-- Define the hyperbola equation
def hyperbola (m : ℝ) (x y : ℝ) : Prop := m * y^2 - x^2 = 1

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := y^2 / 5 + x^2 = 1

-- Define that the hyperbola and ellipse share the same foci
def same_foci (m : ℝ) : Prop := ∃ (a b : ℝ), (hyperbola m a b ∧ ellipse a b)

-- Theorem statement
theorem hyperbola_ellipse_foci (m : ℝ) (h : same_foci m) : m = 1/3 :=
sorry

end hyperbola_ellipse_foci_l2479_247929


namespace average_cost_calculation_l2479_247931

/-- Calculates the average cost of products sold given the quantities and prices of different product types -/
theorem average_cost_calculation
  (iphone_quantity : ℕ) (iphone_price : ℕ)
  (ipad_quantity : ℕ) (ipad_price : ℕ)
  (appletv_quantity : ℕ) (appletv_price : ℕ)
  (h1 : iphone_quantity = 100)
  (h2 : iphone_price = 1000)
  (h3 : ipad_quantity = 20)
  (h4 : ipad_price = 900)
  (h5 : appletv_quantity = 80)
  (h6 : appletv_price = 200) :
  (iphone_quantity * iphone_price + ipad_quantity * ipad_price + appletv_quantity * appletv_price) /
  (iphone_quantity + ipad_quantity + appletv_quantity) = 670 :=
by sorry

end average_cost_calculation_l2479_247931


namespace sum_of_integers_l2479_247979

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 130) (h2 : x * y = 27) : 
  x + y = 14 := by
sorry

end sum_of_integers_l2479_247979


namespace inequality_proof_l2479_247988

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hsum : a + b + c = 1) :
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧ 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1+a)*(1+b)*(1+c) := by
  sorry

end inequality_proof_l2479_247988


namespace power_function_through_point_l2479_247955

/-- A power function passing through (2, √2/2) has f(9) = 1/3 -/
theorem power_function_through_point (f : ℝ → ℝ) :
  (∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) →  -- f is a power function
  f 2 = Real.sqrt 2 / 2 →            -- f passes through (2, √2/2)
  f 9 = 1 / 3 :=                     -- f(9) = 1/3
by sorry

end power_function_through_point_l2479_247955


namespace blended_tea_selling_price_l2479_247934

/-- Calculates the selling price of a blended tea variety -/
theorem blended_tea_selling_price
  (cost1 : ℝ) (cost2 : ℝ) (ratio1 : ℝ) (ratio2 : ℝ) (gain_percent : ℝ)
  (h1 : cost1 = 18)
  (h2 : cost2 = 20)
  (h3 : ratio1 = 5)
  (h4 : ratio2 = 3)
  (h5 : gain_percent = 12)
  : (cost1 * ratio1 + cost2 * ratio2) / (ratio1 + ratio2) * (1 + gain_percent / 100) = 21 := by
  sorry

#check blended_tea_selling_price

end blended_tea_selling_price_l2479_247934


namespace yoga_studio_men_count_l2479_247957

theorem yoga_studio_men_count :
  ∀ (num_men : ℕ) (avg_weight_men avg_weight_women avg_weight_all : ℝ),
    avg_weight_men = 190 →
    avg_weight_women = 120 →
    num_men + 6 = 14 →
    (num_men * avg_weight_men + 6 * avg_weight_women) / 14 = avg_weight_all →
    avg_weight_all = 160 →
    num_men = 8 := by
  sorry

end yoga_studio_men_count_l2479_247957


namespace quadratic_inequality_bounds_l2479_247980

theorem quadratic_inequality_bounds (x : ℝ) (h : x^2 - 6*x + 8 < 0) :
  25 < x^2 + 6*x + 9 ∧ x^2 + 6*x + 9 < 49 := by
  sorry

end quadratic_inequality_bounds_l2479_247980


namespace platform_length_l2479_247915

/-- Calculates the length of a platform given train speed and crossing times -/
theorem platform_length
  (train_speed : ℝ)
  (platform_crossing_time : ℝ)
  (man_crossing_time : ℝ)
  (h1 : train_speed = 72)  -- Train speed in kmph
  (h2 : platform_crossing_time = 30)  -- Time to cross platform in seconds
  (h3 : man_crossing_time = 15)  -- Time to cross man in seconds
  : ∃ (platform_length : ℝ), platform_length = 300 := by
  sorry

#check platform_length

end platform_length_l2479_247915


namespace gcd_count_for_product_360_l2479_247976

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ+), (∀ d ∈ S, ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) ∧ 
                       (∀ d : ℕ+, (∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) → d ∈ S) ∧ 
                       S.card = 9) := by
  sorry

end gcd_count_for_product_360_l2479_247976


namespace no_prime_generating_pair_l2479_247974

theorem no_prime_generating_pair : ∀ a b : ℕ+, ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p > 1000 ∧ q > 1000 ∧ p ≠ q ∧ ¬(Prime (a * p + b * q)) := by
  sorry

end no_prime_generating_pair_l2479_247974


namespace cyclist_distance_difference_l2479_247923

/-- Represents a cyclist with a constant speed --/
structure Cyclist where
  speed : ℝ

/-- Calculates the distance traveled by a cyclist in a given time --/
def distance (c : Cyclist) (time : ℝ) : ℝ := c.speed * time

theorem cyclist_distance_difference 
  (clara : Cyclist) 
  (david : Cyclist) 
  (h1 : clara.speed = 14.4) 
  (h2 : david.speed = 10.8) 
  (time : ℝ) 
  (h3 : time = 5) : 
  distance clara time - distance david time = 18 := by
sorry

end cyclist_distance_difference_l2479_247923


namespace square_diff_theorem_l2479_247939

theorem square_diff_theorem : (25 + 9)^2 - (25^2 + 9^2) = 450 := by
  sorry

end square_diff_theorem_l2479_247939


namespace hiring_theorem_l2479_247997

/-- Given probabilities for hiring three students A, B, and C --/
structure HiringProbabilities where
  probA : ℝ
  probNeitherANorB : ℝ
  probBothBAndC : ℝ

/-- The hiring probabilities satisfy the given conditions --/
def ValidHiringProbabilities (h : HiringProbabilities) : Prop :=
  h.probA = 2/3 ∧ h.probNeitherANorB = 1/12 ∧ h.probBothBAndC = 3/8

/-- Individual probabilities for B and C, and the probability of at least two being hired --/
structure HiringResults where
  probB : ℝ
  probC : ℝ
  probAtLeastTwo : ℝ

/-- The main theorem: given the conditions, prove the results --/
theorem hiring_theorem (h : HiringProbabilities) 
  (hvalid : ValidHiringProbabilities h) : 
  ∃ (r : HiringResults), r.probB = 3/4 ∧ r.probC = 1/2 ∧ r.probAtLeastTwo = 2/3 := by
  sorry

end hiring_theorem_l2479_247997


namespace quadratic_function_properties_l2479_247922

theorem quadratic_function_properties (b : ℝ) : 
  (∃ x : ℝ, x^2 - 2*b*x + b^2 + b - 5 = 0) →
  (∀ x < 3.5, (2*x - 2*b) < 0) →
  3.5 ≤ b ∧ b ≤ 5 := by
sorry

end quadratic_function_properties_l2479_247922


namespace base_five_digits_of_1837_l2479_247959

theorem base_five_digits_of_1837 (n : Nat) (h : n = 1837) :
  (Nat.log 5 n + 1 : Nat) = 5 := by
  sorry

end base_five_digits_of_1837_l2479_247959


namespace infinitely_many_benelux_couples_l2479_247906

/-- Definition of a Benelux couple -/
def is_benelux_couple (m n : ℕ) : Prop :=
  1 < m ∧ m < n ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)) ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ (m + 1) ↔ p ∣ (n + 1)))

/-- Theorem: There exist infinitely many Benelux couples -/
theorem infinitely_many_benelux_couples :
  ∀ N : ℕ, ∃ m n : ℕ, N < m ∧ is_benelux_couple m n :=
sorry

end infinitely_many_benelux_couples_l2479_247906


namespace rationalize_denominator_l2479_247954

theorem rationalize_denominator :
  (Real.sqrt 2) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5) = 
  (3 + Real.sqrt 6 + Real.sqrt 15) / 6 := by
  sorry

end rationalize_denominator_l2479_247954


namespace n_over_8_equals_2_pow_3997_l2479_247911

theorem n_over_8_equals_2_pow_3997 (n : ℕ) : n = 16^1000 → n/8 = 2^3997 := by
  sorry

end n_over_8_equals_2_pow_3997_l2479_247911


namespace power_sum_of_i_l2479_247972

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23456 + i^23457 + i^23458 + i^23459 = 0 := by
  sorry

end power_sum_of_i_l2479_247972


namespace tims_change_theorem_l2479_247991

/-- Calculates the change received after a purchase --/
def calculate_change (initial_amount : ℕ) (purchase_amount : ℕ) : ℕ :=
  initial_amount - purchase_amount

/-- Proves that the change received is correct for Tim's candy bar purchase --/
theorem tims_change_theorem :
  let initial_amount : ℕ := 50
  let purchase_amount : ℕ := 45
  calculate_change initial_amount purchase_amount = 5 := by
  sorry

end tims_change_theorem_l2479_247991


namespace min_value_when_a_2_range_of_a_l2479_247941

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 1|

-- Theorem for the minimum value when a = 2
theorem min_value_when_a_2 :
  ∃ (min : ℝ), min = 3 ∧ ∀ x, f 2 x ≥ min :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∃ x, f a x < 2) ↔ -3 < a ∧ a < 1 :=
sorry

end min_value_when_a_2_range_of_a_l2479_247941


namespace servant_cash_received_l2479_247952

/-- Calculates the cash received by a servant after working for a partial year --/
theorem servant_cash_received
  (annual_cash : ℕ)
  (turban_price : ℕ)
  (months_worked : ℕ)
  (h1 : annual_cash = 90)
  (h2 : turban_price = 50)
  (h3 : months_worked = 9) :
  (months_worked * (annual_cash + turban_price) / 12) - turban_price = 55 :=
sorry

end servant_cash_received_l2479_247952


namespace evaluate_expression_l2479_247932

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 2 * y^x = 533 := by
  sorry

end evaluate_expression_l2479_247932


namespace quadratic_inequality_l2479_247986

theorem quadratic_inequality (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (ha₁ : 0 < a₁) (hb₁ : 0 < b₁) (hc₁ : 0 < c₁)
  (ha₂ : 0 < a₂) (hb₂ : 0 < b₂) (hc₂ : 0 < c₂)
  (h₁ : b₁^2 ≤ a₁*c₁) (h₂ : b₂^2 ≤ a₂*c₂) :
  (a₁ + a₂ + 5) * (c₁ + c₂ + 2) > (b₁ + b₂ + 3)^2 := by
  sorry

end quadratic_inequality_l2479_247986


namespace inequality_proof_l2479_247958

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : 
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 
    2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end inequality_proof_l2479_247958


namespace angle_four_times_complement_l2479_247926

theorem angle_four_times_complement (x : ℝ) : 
  (x = 4 * (90 - x)) → x = 72 := by
  sorry

end angle_four_times_complement_l2479_247926


namespace game_ends_in_25_rounds_l2479_247966

/-- Represents a player in the game -/
inductive Player : Type
| A | B | C | D

/-- The state of the game at any given round -/
structure GameState :=
  (tokens : Player → ℕ)
  (round : ℕ)

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.A => 16
    | Player.B => 15
    | Player.C => 14
    | Player.D => 13,
    round := 0 }

/-- Determines if the game has ended (i.e., if any player has 0 tokens) -/
def gameEnded (state : GameState) : Prop :=
  ∃ p : Player, state.tokens p = 0

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- The theorem to prove -/
theorem game_ends_in_25_rounds :
  ∃ finalState : GameState,
    finalState.round = 25 ∧
    gameEnded finalState ∧
    (∀ prevState : GameState, prevState.round < 25 → ¬gameEnded prevState) :=
  sorry

end game_ends_in_25_rounds_l2479_247966


namespace number_solution_l2479_247993

theorem number_solution : ∃ x : ℝ, (3034 - (1002 / x) = 2984) ∧ x = 20.04 := by
  sorry

end number_solution_l2479_247993


namespace consecutive_page_numbers_sum_l2479_247910

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20250 → n + (n + 1) = 285 := by
  sorry

end consecutive_page_numbers_sum_l2479_247910


namespace regularity_lemma_l2479_247940

/-- A graph represented as a set of vertices and a set of edges -/
structure Graph (V : Type) where
  vertices : Set V
  edges : Set (V × V)

/-- The maximum degree of a graph -/
def max_degree (G : Graph V) : ℕ := sorry

/-- A regularity graph with parameters ε, ℓ, and d -/
structure RegularityGraph (V : Type) extends Graph V where
  ε : ℝ
  ℓ : ℕ
  d : ℝ

/-- The s-closure of a regularity graph -/
def s_closure (R : RegularityGraph V) (s : ℕ) : Graph V := sorry

/-- Subgraph relation -/
def is_subgraph (H G : Graph V) : Prop := sorry

theorem regularity_lemma {V : Type} (d : ℝ) (Δ : ℕ) 
  (hd : d ∈ Set.Icc 0 1) (hΔ : Δ ≥ 1) :
  ∃ ε₀ > 0, ∀ (G H : Graph V) (s : ℕ) (R : RegularityGraph V),
    max_degree H ≤ Δ →
    R.ε ≤ ε₀ →
    R.ℓ ≥ 2 * s / d^Δ →
    R.d = d →
    is_subgraph H (s_closure R s) →
    is_subgraph H G :=
sorry

end regularity_lemma_l2479_247940
