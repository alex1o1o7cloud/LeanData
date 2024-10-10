import Mathlib

namespace solve_equation_and_calculate_l3119_311955

theorem solve_equation_and_calculate (x : ℝ) :
  Real.sqrt ((3 / x) + 1) = 4 / 3 →
  x = 27 / 7 ∧ x + 6 = 69 / 7 := by
  sorry

end solve_equation_and_calculate_l3119_311955


namespace max_value_quadratic_l3119_311914

theorem max_value_quadratic (s t : ℝ) (h : t = 4) :
  ∃ (max : ℝ), max = 46 ∧ ∀ s, -2 * s^2 + 24 * s + 3 * t - 38 ≤ max :=
by sorry

end max_value_quadratic_l3119_311914


namespace tangent_circles_count_l3119_311984

-- Define a type for lines in a plane
structure Line where
  -- Add necessary properties for a line

-- Define a type for circles in a plane
structure Circle where
  -- Add necessary properties for a circle

-- Define a function to check if a circle is tangent to a line
def is_tangent (c : Circle) (l : Line) : Prop :=
  sorry

-- Define a function to count the number of circles tangent to three lines
def count_tangent_circles (l1 l2 l3 : Line) : ℕ :=
  sorry

-- Define predicates for different line configurations
def general_position (l1 l2 l3 : Line) : Prop :=
  sorry

def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  sorry

def all_parallel (l1 l2 l3 : Line) : Prop :=
  sorry

def two_parallel_one_intersecting (l1 l2 l3 : Line) : Prop :=
  sorry

theorem tangent_circles_count 
  (l1 l2 l3 : Line) : 
  (general_position l1 l2 l3 → count_tangent_circles l1 l2 l3 = 4) ∧
  (intersect_at_point l1 l2 l3 → count_tangent_circles l1 l2 l3 = 0) ∧
  (all_parallel l1 l2 l3 → count_tangent_circles l1 l2 l3 = 0) ∧
  (two_parallel_one_intersecting l1 l2 l3 → count_tangent_circles l1 l2 l3 = 2) :=
by sorry

end tangent_circles_count_l3119_311984


namespace product_of_roots_l3119_311972

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 20 → ∃ y : ℝ, (x + 3) * (x - 5) = 20 ∧ (y + 3) * (y - 5) = 20 ∧ x * y = -35 := by
  sorry

end product_of_roots_l3119_311972


namespace zac_strawberries_l3119_311974

def strawberry_problem (total : ℕ) (jonathan_matthew : ℕ) (matthew_zac : ℕ) : Prop :=
  ∃ (jonathan matthew zac : ℕ),
    jonathan + matthew + zac = total ∧
    jonathan + matthew = jonathan_matthew ∧
    matthew + zac = matthew_zac ∧
    zac = 200

theorem zac_strawberries :
  strawberry_problem 550 350 250 :=
sorry

end zac_strawberries_l3119_311974


namespace log_10_50_between_consecutive_integers_sum_l3119_311959

theorem log_10_50_between_consecutive_integers_sum :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < b ∧ a + b = 3 := by
sorry

end log_10_50_between_consecutive_integers_sum_l3119_311959


namespace only_villages_comprehensive_villages_only_comprehensive_option_l3119_311943

/-- Represents a survey option -/
inductive SurveyOption
  | VillagesPollution
  | DrugQuality
  | PublicOpinion
  | RiverWaterQuality

/-- Defines what makes a survey comprehensive -/
def is_comprehensive (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.VillagesPollution => true
  | _ => false

/-- Theorem stating that only the villages pollution survey is comprehensive -/
theorem only_villages_comprehensive :
  ∀ (option : SurveyOption),
    is_comprehensive option ↔ option = SurveyOption.VillagesPollution :=
by sorry

/-- Main theorem proving that investigating five villages is the only suitable option -/
theorem villages_only_comprehensive_option :
  ∃! (option : SurveyOption), is_comprehensive option :=
by sorry

end only_villages_comprehensive_villages_only_comprehensive_option_l3119_311943


namespace polar_to_cartesian_line_l3119_311936

/-- Given a curve in polar coordinates defined by r = 2 / (2sin θ - cos θ),
    prove that it represents a line in Cartesian coordinates. -/
theorem polar_to_cartesian_line :
  ∀ θ r : ℝ,
  r = 2 / (2 * Real.sin θ - Real.cos θ) →
  ∃ m c : ℝ, ∀ x y : ℝ,
  x = r * Real.cos θ ∧ y = r * Real.sin θ →
  y = m * x + c :=
sorry

end polar_to_cartesian_line_l3119_311936


namespace correct_calculation_l3119_311901

theorem correct_calculation (x : ℤ) (h : x - 48 = 52) : x + 48 = 148 := by
  sorry

end correct_calculation_l3119_311901


namespace min_sum_squares_l3119_311977

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 6) :
  x^2 + y^2 + z^2 ≥ 18/7 := by
  sorry

end min_sum_squares_l3119_311977


namespace white_ball_mutually_exclusive_l3119_311996

-- Define the set of balls
inductive Ball : Type
  | Red : Ball
  | Black : Ball
  | White : Ball

-- Define the set of people
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

-- Define a distribution as a function from Person to Ball
def Distribution := Person → Ball

-- Define the event "person receives the white ball"
def receives_white_ball (p : Person) (d : Distribution) : Prop :=
  d p = Ball.White

-- State the theorem
theorem white_ball_mutually_exclusive :
  ∀ (d : Distribution),
    (∀ (p1 p2 : Person), p1 ≠ p2 → d p1 ≠ d p2) →
    ¬(receives_white_ball Person.A d ∧ receives_white_ball Person.B d) :=
by sorry

end white_ball_mutually_exclusive_l3119_311996


namespace max_value_quadratic_l3119_311980

theorem max_value_quadratic :
  (∀ x : ℝ, -3 * x^2 + 15 * x + 9 ≤ 111/4) ∧
  (∃ x : ℝ, -3 * x^2 + 15 * x + 9 = 111/4) := by
  sorry

end max_value_quadratic_l3119_311980


namespace ratio_of_a_over_3_to_b_over_2_l3119_311966

theorem ratio_of_a_over_3_to_b_over_2 (a b c : ℝ) 
  (h1 : 2 * a = 3 * b) 
  (h2 : c ≠ 0) 
  (h3 : 3 * a + 2 * b = c) : 
  (a / 3) / (b / 2) = 1 := by
sorry

end ratio_of_a_over_3_to_b_over_2_l3119_311966


namespace solve_linear_equation_l3119_311953

theorem solve_linear_equation (x y : ℝ) :
  4 * x - y = 3 → y = 4 * x - 3 := by
  sorry

end solve_linear_equation_l3119_311953


namespace circle_symmetry_l3119_311994

def circle_equation (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem circle_symmetry (h : Set (ℝ × ℝ)) :
  h = circle_equation (-2, 1) 1 →
  (λ (x, y) => circle_equation (2, -1) 1 (x, y)) =
  (λ (x, y) => (x - 2)^2 + (y + 1)^2 = 1) := by
  sorry

end circle_symmetry_l3119_311994


namespace commuting_days_businessman_commute_l3119_311983

/-- Represents the commuting options for a businessman over a period of days. -/
structure CommutingData where
  /-- Total number of days -/
  total_days : ℕ
  /-- Number of times taking bus to work in the morning -/
  morning_bus : ℕ
  /-- Number of times coming home by bus in the afternoon -/
  afternoon_bus : ℕ
  /-- Number of train commuting segments (either morning or afternoon) -/
  train_segments : ℕ

/-- Theorem stating that given the commuting conditions, the total number of days is 32 -/
theorem commuting_days (data : CommutingData) : 
  data.morning_bus = 12 ∧ 
  data.afternoon_bus = 20 ∧ 
  data.train_segments = 15 →
  data.total_days = 32 := by
  sorry

/-- Main theorem proving the specific case -/
theorem businessman_commute : ∃ (data : CommutingData), 
  data.morning_bus = 12 ∧ 
  data.afternoon_bus = 20 ∧ 
  data.train_segments = 15 ∧
  data.total_days = 32 := by
  sorry

end commuting_days_businessman_commute_l3119_311983


namespace units_digit_a_2017_l3119_311989

/-- Represents the units digit of an integer -/
def M (x : ℤ) : ℕ :=
  x.natAbs % 10

/-- Sequence defined by the recurrence relation -/
def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sorry  -- Definition to be filled based on the recurrence relation

/-- The main theorem to be proved -/
theorem units_digit_a_2017 :
  M (Int.floor (a 2016)) = 1 := by sorry

end units_digit_a_2017_l3119_311989


namespace probability_of_favorable_outcome_l3119_311939

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ a ≠ b

def is_favorable_pair (a b : ℕ) : Prop :=
  is_valid_pair a b ∧ ∃ k : ℕ, a * b + a + b = 7 * k - 1

def total_pairs : ℕ := Nat.choose 60 2

def favorable_pairs : ℕ := 444

theorem probability_of_favorable_outcome :
  (favorable_pairs : ℚ) / total_pairs = 74 / 295 := by sorry

end probability_of_favorable_outcome_l3119_311939


namespace product_of_absolute_sum_l3119_311982

theorem product_of_absolute_sum (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^2 / (b*c) + b^2 / (c*a) + c^2 / (a*b) = 1) : 
  (Complex.abs (a + b + c) - 1) * 
  (Complex.abs (a + b + c) - (1 + Real.sqrt 3)) * 
  (Complex.abs (a + b + c) - (1 - Real.sqrt 3)) = 2 := by
sorry

end product_of_absolute_sum_l3119_311982


namespace square_area_ratio_l3119_311927

theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : a^2 / b^2 = 16 := by
  sorry

end square_area_ratio_l3119_311927


namespace fourth_degree_polynomial_roots_l3119_311909

theorem fourth_degree_polynomial_roots : ∃ (a b c d : ℝ),
  (a = 1 - Real.sqrt 3) ∧
  (b = 1 + Real.sqrt 3) ∧
  (c = (1 - Real.sqrt 13) / 2) ∧
  (d = (1 + Real.sqrt 13) / 2) ∧
  (∀ x : ℝ, x^4 - 3*x^3 + 3*x^2 - x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end fourth_degree_polynomial_roots_l3119_311909


namespace polynomial_remainder_l3119_311917

theorem polynomial_remainder (x : ℝ) : (x^15 + 1) % (x + 1) = 0 := by
  sorry

end polynomial_remainder_l3119_311917


namespace cube_side_area_l3119_311999

theorem cube_side_area (V : ℝ) (s : ℝ) (h : V = 125) :
  V = s^3 → s^2 = 25 := by
  sorry

end cube_side_area_l3119_311999


namespace symmetric_point_about_origin_l3119_311947

/-- Given a point P (-2, -3), prove that (2, 3) is its symmetric point about the origin -/
theorem symmetric_point_about_origin :
  let P : ℝ × ℝ := (-2, -3)
  let Q : ℝ × ℝ := (2, 3)
  (∀ (x y : ℝ), (x, y) = P → (-x, -y) = Q) :=
by sorry

end symmetric_point_about_origin_l3119_311947


namespace unique_solution_l3119_311941

/-- Represents the number of correct answers for each friend -/
structure ExamResults :=
  (A B C D : Nat)

/-- Checks if the given results satisfy the conditions of the problem -/
def satisfiesConditions (results : ExamResults) : Prop :=
  -- Total correct answers is 6
  results.A + results.B + results.C + results.D = 6 ∧
  -- Each result is between 0 and 3
  results.A ≤ 3 ∧ results.B ≤ 3 ∧ results.C ≤ 3 ∧ results.D ≤ 3 ∧
  -- Number of true statements matches correct answers
  (results.A = 1 ∨ results.A = 2) ∧
  (results.B = 0 ∨ results.B = 2) ∧
  (results.C = 0 ∨ results.C = 1) ∧
  (results.D = 0 ∨ results.D = 3) ∧
  -- Relative performance statements
  (results.A > results.B → results.A = 2) ∧
  (results.C < results.D → results.A = 2) ∧
  (results.C = 0 → results.B = 3) ∧
  (results.A < results.D → results.B = 3) ∧
  (results.D = 2 → results.C = 1) ∧
  (results.B < results.A → results.C = 1) ∧
  (results.C < results.D → results.D = 3) ∧
  (results.A < results.B → results.D = 3)

theorem unique_solution :
  ∃! results : ExamResults, satisfiesConditions results ∧ 
    results.A = 1 ∧ results.B = 2 ∧ results.C = 0 ∧ results.D = 3 :=
sorry

end unique_solution_l3119_311941


namespace octagon_perimeter_96cm_l3119_311923

/-- A regular octagon is a polygon with 8 equal sides -/
structure RegularOctagon where
  side_length : ℝ
  
/-- The perimeter of a regular octagon -/
def perimeter (octagon : RegularOctagon) : ℝ :=
  8 * octagon.side_length

theorem octagon_perimeter_96cm :
  ∀ (octagon : RegularOctagon),
    octagon.side_length = 12 →
    perimeter octagon = 96 := by
  sorry

end octagon_perimeter_96cm_l3119_311923


namespace unique_function_characterization_l3119_311908

theorem unique_function_characterization :
  ∀ f : ℕ+ → ℕ+,
  (∀ x y : ℕ+, x < y → f x < f y) →
  (∀ x y : ℕ+, f (y * f x) = x^2 * f (x * y)) →
  ∀ x : ℕ+, f x = x^2 := by
sorry

end unique_function_characterization_l3119_311908


namespace quadratic_polynomial_satisfies_conditions_l3119_311921

-- Define the quadratic polynomial
def p (x : ℚ) : ℚ := (13/6) * x^2 - (7/6) * x + 2

-- State the theorem
theorem quadratic_polynomial_satisfies_conditions :
  p 1 = 3 ∧ p 0 = 2 ∧ p 3 = 18 := by
  sorry

end quadratic_polynomial_satisfies_conditions_l3119_311921


namespace floor_tiles_theorem_l3119_311961

/-- A rectangular floor covered with congruent square tiles. -/
structure TiledFloor where
  width : ℕ
  length : ℕ
  perimeterTiles : ℕ
  lengthTwiceWidth : length = 2 * width
  tilesAlongPerimeter : perimeterTiles = 2 * (width + length)

/-- The total number of tiles covering the floor. -/
def totalTiles (floor : TiledFloor) : ℕ :=
  floor.width * floor.length

/-- Theorem stating that a rectangular floor with 88 tiles along the perimeter
    and length twice the width has 430 tiles in total. -/
theorem floor_tiles_theorem (floor : TiledFloor) 
    (h : floor.perimeterTiles = 88) : totalTiles floor = 430 := by
  sorry

end floor_tiles_theorem_l3119_311961


namespace kaydence_age_l3119_311907

/-- The age of Kaydence given the ages of her family members and the total family age -/
theorem kaydence_age (total_age father_age mother_age brother_age sister_age : ℕ)
  (h_total : total_age = 200)
  (h_father : father_age = 60)
  (h_mother : mother_age = father_age - 2)
  (h_brother : brother_age = father_age / 2)
  (h_sister : sister_age = 40) :
  total_age - (father_age + mother_age + brother_age + sister_age) = 12 := by
  sorry

end kaydence_age_l3119_311907


namespace continuous_of_strictly_increasing_and_continuous_compose_l3119_311913

/-- Given a strictly increasing function f: ℝ → ℝ where f ∘ f is continuous, f is continuous. -/
theorem continuous_of_strictly_increasing_and_continuous_compose (f : ℝ → ℝ)
  (h_increasing : StrictMono f) (h_continuous_compose : Continuous (f ∘ f)) :
  Continuous f := by
  sorry

end continuous_of_strictly_increasing_and_continuous_compose_l3119_311913


namespace total_spent_pinball_l3119_311932

/-- The amount of money in dollars represented by a half-dollar coin -/
def half_dollar_value : ℚ := 0.5

/-- The number of half-dollars Joan spent on Wednesday -/
def wednesday_spent : ℕ := 4

/-- The number of half-dollars Joan spent on Thursday -/
def thursday_spent : ℕ := 14

/-- The number of half-dollars Joan spent on Friday -/
def friday_spent : ℕ := 8

/-- Theorem: The total amount Joan spent playing pinball over three days is $13.00 -/
theorem total_spent_pinball :
  (wednesday_spent + thursday_spent + friday_spent : ℚ) * half_dollar_value = 13 :=
by sorry

end total_spent_pinball_l3119_311932


namespace division_multiplication_equality_l3119_311924

theorem division_multiplication_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / (b / a) * (a / b) = a^2 / b^2 := by
  sorry

end division_multiplication_equality_l3119_311924


namespace percentage_to_original_number_l3119_311988

theorem percentage_to_original_number :
  let percentage : Float := 501.99999999999994
  let original_number : Float := percentage / 100
  original_number = 5.0199999999999994 := by
sorry

end percentage_to_original_number_l3119_311988


namespace solution_set_implies_a_equals_one_l3119_311993

def f (a x : ℝ) : ℝ := |x - a| - 2

theorem solution_set_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, |f a x| < 1 ↔ x ∈ Set.union (Set.Ioo (-2) 0) (Set.Ioo 2 4)) →
  a = 1 := by
  sorry

end solution_set_implies_a_equals_one_l3119_311993


namespace problem_statement_l3119_311918

theorem problem_statement :
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ 4) ∧
  ¬(∃ x₀ : ℝ, x₀ > 0 ∧ 2 * x₀ = 1 / 2) := by
  sorry

end problem_statement_l3119_311918


namespace tomato_production_l3119_311944

theorem tomato_production (plant1 plant2 plant3 total : ℕ) : 
  plant1 = 24 →
  plant2 = (plant1 / 2) + 5 →
  plant3 = plant2 + 2 →
  total = plant1 + plant2 + plant3 →
  total = 60 :=
by sorry

end tomato_production_l3119_311944


namespace sum_53_to_100_l3119_311991

def sum_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_53_to_100 (h : sum_range 51 100 = 3775) : sum_range 53 100 = 3672 := by
  sorry

end sum_53_to_100_l3119_311991


namespace factorial_10_trailing_zeros_base_15_l3119_311976

/-- The number of trailing zeros in the base 15 representation of a natural number -/
def trailingZerosBase15 (n : ℕ) : ℕ := sorry

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeros in the base 15 representation of 10! is 2 -/
theorem factorial_10_trailing_zeros_base_15 : 
  trailingZerosBase15 (factorial 10) = 2 := by sorry

end factorial_10_trailing_zeros_base_15_l3119_311976


namespace baxter_peanut_purchase_l3119_311946

/-- Represents the peanut purchase scenario at the Peanut Emporium -/
structure PeanutPurchase where
  pricePerPound : ℝ
  minimumPurchase : ℝ
  bulkDiscountThreshold : ℝ
  bulkDiscountRate : ℝ
  earlyBirdDiscountRate : ℝ
  salesTaxRate : ℝ
  totalSpent : ℝ

/-- Calculates the pounds of peanuts purchased given the purchase scenario -/
def calculatePoundsPurchased (p : PeanutPurchase) : ℝ :=
  sorry

/-- Theorem: Given the purchase conditions, Baxter bought 28 pounds over the minimum -/
theorem baxter_peanut_purchase :
  let p := PeanutPurchase.mk 3 15 25 0.1 0.05 0.08 119.88
  calculatePoundsPurchased p - p.minimumPurchase = 28 := by
  sorry

end baxter_peanut_purchase_l3119_311946


namespace inequality_implies_a_bound_l3119_311975

theorem inequality_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x > 0 → 2 * x * Real.log x ≥ -x^2 + a*x - 3) → 
  a ≤ 4 := by
sorry

end inequality_implies_a_bound_l3119_311975


namespace complement_union_eq_inter_complements_l3119_311973

variable {Ω : Type*} [MeasurableSpace Ω]
variable (A B : Set Ω)

theorem complement_union_eq_inter_complements :
  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ := by sorry

end complement_union_eq_inter_complements_l3119_311973


namespace twentieth_term_of_arithmetic_sequence_l3119_311963

/-- Given an arithmetic sequence with first term 2 and common difference 3,
    prove that the 20th term is 59. -/
theorem twentieth_term_of_arithmetic_sequence :
  let a : ℕ → ℤ := λ n => 2 + 3 * (n - 1)
  a 20 = 59 := by sorry

end twentieth_term_of_arithmetic_sequence_l3119_311963


namespace m_equality_l3119_311933

theorem m_equality (M : ℕ) (h : M^2 = 16^81 * 81^16) : M = 6^64 * 2^260 := by
  sorry

end m_equality_l3119_311933


namespace calculator_change_l3119_311928

/-- Calculates the change received after buying three types of calculators. -/
theorem calculator_change (total_money : ℕ) (basic_cost : ℕ) : 
  total_money = 100 →
  basic_cost = 8 →
  total_money - (basic_cost + 2 * basic_cost + 3 * (2 * basic_cost)) = 28 := by
  sorry

#check calculator_change

end calculator_change_l3119_311928


namespace little_krish_money_distribution_l3119_311967

-- Define the problem parameters
def initial_amount : ℚ := 200.50
def spent_on_sweets : ℚ := 35.25
def amount_left : ℚ := 114.85
def num_friends : ℕ := 2

-- Define the theorem
theorem little_krish_money_distribution :
  ∃ (amount_per_friend : ℚ),
    amount_per_friend = 25.20 ∧
    initial_amount - spent_on_sweets - (num_friends : ℚ) * amount_per_friend = amount_left :=
by
  sorry


end little_krish_money_distribution_l3119_311967


namespace butanoic_acid_nine_moles_weight_l3119_311930

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in Butanoic acid -/
def carbon_count : ℕ := 4

/-- The number of Hydrogen atoms in Butanoic acid -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in Butanoic acid -/
def oxygen_count : ℕ := 2

/-- The number of moles of Butanoic acid -/
def moles : ℝ := 9

/-- The molecular weight of Butanoic acid in g/mol -/
def butanoic_acid_weight : ℝ := 
  carbon_weight * carbon_count + 
  hydrogen_weight * hydrogen_count + 
  oxygen_weight * oxygen_count

/-- Theorem: The molecular weight of 9 moles of Butanoic acid is 792.936 grams -/
theorem butanoic_acid_nine_moles_weight : 
  butanoic_acid_weight * moles = 792.936 := by sorry

end butanoic_acid_nine_moles_weight_l3119_311930


namespace correct_pies_left_l3119_311951

/-- Calculates the number of pies left after baking and dropping some -/
def pies_left (oven_capacity : ℕ) (num_batches : ℕ) (dropped_pies : ℕ) : ℕ :=
  oven_capacity * num_batches - dropped_pies

theorem correct_pies_left :
  let oven_capacity : ℕ := 5
  let num_batches : ℕ := 7
  let dropped_pies : ℕ := 8
  pies_left oven_capacity num_batches dropped_pies = 27 := by
  sorry

end correct_pies_left_l3119_311951


namespace nested_sqrt_problem_l3119_311978

theorem nested_sqrt_problem (n : ℕ) :
  (∃ k : ℕ, (n * (n * n^(1/2))^(1/2))^(1/2) = k) ∧ 
  (n * (n * n^(1/2))^(1/2))^(1/2) < 2217 →
  n = 256 := by
sorry

end nested_sqrt_problem_l3119_311978


namespace system_equation_solution_l3119_311916

theorem system_equation_solution (x y c d : ℝ) (h1 : 4 * x + 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end system_equation_solution_l3119_311916


namespace calculation_proof_l3119_311992

theorem calculation_proof : (-3)^3 + 5^2 - (-2)^2 = -6 := by
  sorry

end calculation_proof_l3119_311992


namespace arccos_negative_half_equals_two_pi_thirds_l3119_311957

theorem arccos_negative_half_equals_two_pi_thirds : 
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end arccos_negative_half_equals_two_pi_thirds_l3119_311957


namespace smallest_value_l3119_311990

theorem smallest_value (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 ≤ x ∧ x^3 ≤ 3*x ∧ x^3 ≤ x^(1/3) ∧ x^3 ≤ 1/x^2 := by
  sorry

end smallest_value_l3119_311990


namespace largest_and_smallest_results_l3119_311931

/-- The type representing our expression with parentheses -/
inductive Expr
  | num : ℕ → Expr
  | op : Expr → Expr → Expr

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.num n => n
  | Expr.op e₁ e₂ => (eval e₁) / (eval e₂)

/-- Check if a rational number is an integer -/
def isInteger (q : ℚ) : Prop := ∃ n : ℤ, q = n

/-- The set of all possible expressions using numbers 1 to 10 -/
def validExpr : Set Expr := sorry

/-- The theorem stating the largest and smallest possible integer results -/
theorem largest_and_smallest_results :
  (∃ e ∈ validExpr, eval e = 44800 ∧ isInteger (eval e)) ∧
  (∃ e ∈ validExpr, eval e = 7 ∧ isInteger (eval e)) ∧
  (∀ e ∈ validExpr, isInteger (eval e) → 7 ≤ eval e ∧ eval e ≤ 44800) :=
sorry

end largest_and_smallest_results_l3119_311931


namespace set_a_range_l3119_311948

theorem set_a_range (a : ℝ) : 
  let A : Set ℝ := {x | 6 * x + a > 0}
  1 ∉ A → a ∈ Set.Iic (-6) :=
by sorry

end set_a_range_l3119_311948


namespace middle_digit_is_zero_l3119_311935

/-- Represents a three-digit number in base 8 -/
structure Base8Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 8 ∧ tens < 8 ∧ ones < 8

/-- Converts a Base8Number to its decimal (base 10) representation -/
def toDecimal (n : Base8Number) : Nat :=
  64 * n.hundreds + 8 * n.tens + n.ones

/-- Represents a three-digit number in base 10 -/
structure Base10Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Checks if a Base8Number has its digits reversed in base 10 representation -/
def hasReversedDigits (n : Base8Number) : Prop :=
  ∃ (m : Base10Number), 
    toDecimal n = 100 * m.hundreds + 10 * m.tens + m.ones ∧
    n.hundreds = m.ones ∧
    n.tens = m.tens ∧
    n.ones = m.hundreds

theorem middle_digit_is_zero (n : Base8Number) 
  (h : hasReversedDigits n) : n.tens = 0 := by
  sorry

end middle_digit_is_zero_l3119_311935


namespace john_star_wars_toys_cost_l3119_311954

/-- The total cost of John's Star Wars toys, including the lightsaber -/
def total_cost (other_toys_cost lightsaber_cost : ℕ) : ℕ :=
  other_toys_cost + lightsaber_cost

/-- The cost of the lightsaber -/
def lightsaber_cost (other_toys_cost : ℕ) : ℕ :=
  2 * other_toys_cost

theorem john_star_wars_toys_cost (other_toys_cost : ℕ) 
  (h : other_toys_cost = 1000) : 
  total_cost other_toys_cost (lightsaber_cost other_toys_cost) = 3000 := by
  sorry

end john_star_wars_toys_cost_l3119_311954


namespace tomato_plants_problem_l3119_311934

theorem tomato_plants_problem (plant1 plant2 plant3 plant4 : ℕ) : 
  plant1 = 8 →
  plant2 = plant1 + 4 →
  plant3 = 3 * (plant1 + plant2) →
  plant4 = 3 * (plant1 + plant2) →
  plant1 + plant2 + plant3 + plant4 = 140 →
  plant2 - plant1 = 4 := by
sorry

end tomato_plants_problem_l3119_311934


namespace number_of_hens_l3119_311956

theorem number_of_hens (total_animals : ℕ) (total_feet : ℕ) (hen_feet cow_feet : ℕ) :
  total_animals = 48 →
  total_feet = 140 →
  hen_feet = 2 →
  cow_feet = 4 →
  ∃ (num_hens num_cows : ℕ),
    num_hens + num_cows = total_animals ∧
    num_hens * hen_feet + num_cows * cow_feet = total_feet ∧
    num_hens = 26 :=
by sorry

end number_of_hens_l3119_311956


namespace intersection_of_sets_l3119_311962

/-- The intersection of {x | x ≥ -1} and {x | -2 < x < 2} is [-1, 2) -/
theorem intersection_of_sets : 
  let M : Set ℝ := {x | x ≥ -1}
  let N : Set ℝ := {x | -2 < x ∧ x < 2}
  M ∩ N = Set.Icc (-1) 2 := by sorry

end intersection_of_sets_l3119_311962


namespace first_restaurant_meals_first_restaurant_meals_proof_l3119_311970

theorem first_restaurant_meals (total_restaurants : Nat) 
  (second_restaurant_meals : Nat) (third_restaurant_meals : Nat) 
  (total_weekly_meals : Nat) (days_per_week : Nat) : Nat :=
  let first_restaurant_daily_meals := 
    (total_weekly_meals - (second_restaurant_meals + third_restaurant_meals) * days_per_week) / days_per_week
  first_restaurant_daily_meals

#check @first_restaurant_meals

theorem first_restaurant_meals_proof 
  (h1 : total_restaurants = 3)
  (h2 : second_restaurant_meals = 40)
  (h3 : third_restaurant_meals = 50)
  (h4 : total_weekly_meals = 770)
  (h5 : days_per_week = 7) :
  first_restaurant_meals total_restaurants second_restaurant_meals third_restaurant_meals total_weekly_meals days_per_week = 20 := by
  sorry

end first_restaurant_meals_first_restaurant_meals_proof_l3119_311970


namespace rectangular_solid_diagonal_cubes_l3119_311903

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed (l w h : ℕ) : ℕ :=
  l + w + h
  - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l)
  + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating the number of cubes passed through by the internal diagonal -/
theorem rectangular_solid_diagonal_cubes :
  cubes_passed 150 324 375 = 768 := by
  sorry

end rectangular_solid_diagonal_cubes_l3119_311903


namespace vector_basis_range_l3119_311987

/-- Two vectors form a basis of a 2D plane if they are linearly independent -/
def is_basis (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 ≠ a.2 * b.1

/-- The range of m for which (1,2) and (m,3m-2) form a basis -/
theorem vector_basis_range :
  ∀ m : ℝ, is_basis (1, 2) (m, 3*m-2) ↔ m ≠ 2 :=
by sorry

end vector_basis_range_l3119_311987


namespace fortiethSelectedNumber_l3119_311958

/-- Calculates the nth selected number in a sequence -/
def nthSelectedNumber (totalParticipants : ℕ) (numSelected : ℕ) (firstNumber : ℕ) (n : ℕ) : ℕ :=
  let spacing := totalParticipants / numSelected
  (n - 1) * spacing + firstNumber

theorem fortiethSelectedNumber :
  nthSelectedNumber 1000 50 15 40 = 795 := by
  sorry

end fortiethSelectedNumber_l3119_311958


namespace place_digit_two_equals_formula_l3119_311995

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_bound : hundreds < 10
  tens_bound : tens < 10
  units_bound : units < 10

/-- Converts a ThreeDigitNumber to its integer value -/
def ThreeDigitNumber.toInt (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Places the digit 2 before a three-digit number -/
def placeDigitTwo (n : ThreeDigitNumber) : ℕ :=
  2000 + 10 * n.toInt

theorem place_digit_two_equals_formula (n : ThreeDigitNumber) :
  placeDigitTwo n = 1000 * (n.hundreds + 2) + 100 * n.tens + 10 * n.units := by
  sorry

end place_digit_two_equals_formula_l3119_311995


namespace two_parts_divisibility_l3119_311925

theorem two_parts_divisibility (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ 13 * x + 17 * y = 283 → 
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b = 283 ∧ 13 ∣ a ∧ 17 ∣ b :=
by sorry

end two_parts_divisibility_l3119_311925


namespace welders_problem_l3119_311945

/-- The number of welders initially working on the order -/
def initial_welders : ℕ := 16

/-- The number of days needed to complete the order with the initial number of welders -/
def total_days : ℕ := 8

/-- The number of welders that leave after the first day -/
def leaving_welders : ℕ := 9

/-- The number of additional days needed by the remaining welders to complete the order -/
def additional_days : ℕ := 16

/-- The work completed in one day by all initial welders -/
def daily_work : ℚ := 1 / initial_welders

theorem welders_problem :
  (1 : ℚ) + additional_days * ((initial_welders - leaving_welders : ℚ) * daily_work) = total_days := by
  sorry

#eval initial_welders

end welders_problem_l3119_311945


namespace quadratic_inequality_solution_l3119_311920

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x < 12 ↔ -4 < x ∧ x < -3 := by
  sorry

end quadratic_inequality_solution_l3119_311920


namespace kimberly_peanuts_l3119_311985

/-- The number of times Kimberly went to the store last month -/
def store_visits : ℕ := 3

/-- The number of peanuts Kimberly buys each time she goes to the store -/
def peanuts_per_visit : ℕ := 7

/-- The total number of peanuts Kimberly bought last month -/
def total_peanuts : ℕ := store_visits * peanuts_per_visit

theorem kimberly_peanuts : total_peanuts = 21 := by
  sorry

end kimberly_peanuts_l3119_311985


namespace range_of_a_l3119_311969

def M (a : ℝ) : Set ℝ := { x | -1 < x - a ∧ x - a < 2 }
def N : Set ℝ := { x | x^2 ≥ x }

theorem range_of_a (a : ℝ) : M a ∪ N = Set.univ → a ∈ Set.Icc (-1) 1 := by
  sorry

end range_of_a_l3119_311969


namespace function_equation_solution_l3119_311929

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x * y * f y) : 
  ∀ x : ℝ, f x = 0 ∨ f x = x^2 := by
  sorry

end function_equation_solution_l3119_311929


namespace don_bottles_from_shop_c_l3119_311997

/-- The number of bottles Don can buy in total -/
def total_bottles : ℕ := 550

/-- The number of bottles Don buys from Shop A -/
def shop_a_bottles : ℕ := 150

/-- The number of bottles Don buys from Shop B -/
def shop_b_bottles : ℕ := 180

/-- The number of bottles Don buys from Shop C -/
def shop_c_bottles : ℕ := total_bottles - (shop_a_bottles + shop_b_bottles)

theorem don_bottles_from_shop_c : 
  shop_c_bottles = 550 - (150 + 180) := by sorry

end don_bottles_from_shop_c_l3119_311997


namespace ladder_problem_l3119_311900

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base : ℝ, base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end ladder_problem_l3119_311900


namespace fraction_sum_equals_decimal_l3119_311911

theorem fraction_sum_equals_decimal : (1 : ℚ) / 20 + 2 / 10 + 4 / 40 = (35 : ℚ) / 100 := by
  sorry

end fraction_sum_equals_decimal_l3119_311911


namespace triangle_sine_theorem_l3119_311915

/-- Given a triangle with area 30, a side of length 12, and a median to that side of length 8,
    the sine of the angle between the side and the median is 5/8. -/
theorem triangle_sine_theorem (A : ℝ) (a m θ : ℝ) 
    (h_area : A = 30)
    (h_side : a = 12)
    (h_median : m = 8)
    (h_angle : 0 < θ ∧ θ < π / 2)
    (h_triangle_area : A = 1/2 * a * m * Real.sin θ) : 
  Real.sin θ = 5/8 := by
sorry

end triangle_sine_theorem_l3119_311915


namespace arithmetic_geometric_sequence_ratio_l3119_311919

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (d q : ℝ) :
  a 1 = 2 →
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 3 = a 1 * q →
  a 11 = a 1 * q^2 →
  q = 4 := by
  sorry

end arithmetic_geometric_sequence_ratio_l3119_311919


namespace unique_solution_condition_l3119_311926

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 5) * (x - 6) = -34 + k * x) ↔ 
  (k = -13 + 4 * Real.sqrt 3 ∨ k = -13 - 4 * Real.sqrt 3) := by
sorry

end unique_solution_condition_l3119_311926


namespace binomial_sum_l3119_311902

theorem binomial_sum : (7 : ℕ).choose 2 + (6 : ℕ).choose 4 = 36 := by sorry

end binomial_sum_l3119_311902


namespace square_root_equation_l3119_311949

theorem square_root_equation (n : ℕ+) :
  Real.sqrt (1 + 1 / (n : ℝ)^2 + 1 / ((n + 1) : ℝ)^2) = 1 + 1 / ((n : ℝ) * (n + 1)) := by
  sorry

end square_root_equation_l3119_311949


namespace arithmetic_sequence_eighth_term_l3119_311971

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 6 = 22)
  (h_third : a 3 = 7) :
  a 8 = 15 := by
  sorry

end arithmetic_sequence_eighth_term_l3119_311971


namespace shirt_cost_calculation_l3119_311904

/-- The amount Sandy spent on clothes -/
def total_spent : ℚ := 33.56

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℚ := 13.99

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℚ := 7.43

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℚ := total_spent - shorts_cost - jacket_cost

theorem shirt_cost_calculation :
  shirt_cost = 12.14 := by
  sorry

end shirt_cost_calculation_l3119_311904


namespace tan_pi_third_plus_cos_nineteen_sixths_pi_l3119_311940

theorem tan_pi_third_plus_cos_nineteen_sixths_pi :
  Real.tan (π / 3) + Real.cos (19 * π / 6) = Real.sqrt 3 / 2 := by
  sorry

end tan_pi_third_plus_cos_nineteen_sixths_pi_l3119_311940


namespace max_daily_sales_l3119_311964

def price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def daily_sales (t : ℕ) : ℝ := price t * sales_volume t

theorem max_daily_sales :
  ∃ (max_sales : ℝ) (max_day : ℕ),
    max_sales = 1125 ∧
    max_day = 25 ∧
    ∀ t : ℕ, 0 < t ∧ t ≤ 30 → daily_sales t ≤ max_sales ∧
    daily_sales max_day = max_sales :=
  sorry

end max_daily_sales_l3119_311964


namespace hiking_rate_ratio_l3119_311922

/-- Proves that the ratio of the hiking rate down to the rate up is 1.5 -/
theorem hiking_rate_ratio : 
  let rate_up : ℝ := 7 -- miles per day
  let days_up : ℝ := 2
  let distance_down : ℝ := 21 -- miles
  let days_down : ℝ := days_up -- same time for both routes
  let rate_down : ℝ := distance_down / days_down
  rate_down / rate_up = 1.5 := by
sorry


end hiking_rate_ratio_l3119_311922


namespace smallest_result_l3119_311905

def S : Set Int := {-10, -4, 0, 2, 7}

theorem smallest_result (x y : Int) (hx : x ∈ S) (hy : y ∈ S) :
  (x * y ≥ -70 ∧ x + y ≥ -70) ∧ ∃ a b : Int, a ∈ S ∧ b ∈ S ∧ (a * b = -70 ∨ a + b = -70) :=
by sorry

end smallest_result_l3119_311905


namespace yang_hui_problem_l3119_311998

theorem yang_hui_problem : ∃ (x : ℕ), 
  (x % 2 = 1) ∧ 
  (x % 5 = 2) ∧ 
  (x % 7 = 3) ∧ 
  (x % 9 = 4) ∧ 
  (∀ y : ℕ, y < x → ¬((y % 2 = 1) ∧ (y % 5 = 2) ∧ (y % 7 = 3) ∧ (y % 9 = 4))) ∧
  x = 157 :=
by sorry

end yang_hui_problem_l3119_311998


namespace b_investment_is_correct_l3119_311938

-- Define the partnership
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  duration : ℕ
  a_share : ℕ
  b_share : ℕ

-- Define the problem
def partnership_problem : Partnership :=
  { a_investment := 7000
  , b_investment := 11000  -- This is what we want to prove
  , c_investment := 18000
  , duration := 8
  , a_share := 1400
  , b_share := 2200 }

-- Theorem statement
theorem b_investment_is_correct (p : Partnership) 
  (h1 : p.a_investment = 7000)
  (h2 : p.c_investment = 18000)
  (h3 : p.duration = 8)
  (h4 : p.a_share = 1400)
  (h5 : p.b_share = 2200) :
  p.b_investment = 11000 := by
  sorry

end b_investment_is_correct_l3119_311938


namespace min_value_expression_l3119_311910

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ≥ 6 ∧ 
  (m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) = 6 ↔ m - 2 * n = 3) :=
by sorry

end min_value_expression_l3119_311910


namespace smallest_number_divisible_by_11_after_change_all_replacements_divisible_by_11_l3119_311960

/-- A function that replaces a digit at a given position in a number with a new digit. -/
def replaceDigit (n : ℕ) (pos : ℕ) (newDigit : ℕ) : ℕ :=
  sorry

/-- A function that checks if a number is divisible by 11. -/
def isDivisibleBy11 (n : ℕ) : Prop :=
  n % 11 = 0

/-- A function that generates all possible numbers after replacing one digit. -/
def allPossibleReplacements (n : ℕ) : List ℕ :=
  sorry

theorem smallest_number_divisible_by_11_after_change : 
  ∀ n : ℕ, n < 909090909 → 
  ∃ m ∈ allPossibleReplacements n, ¬isDivisibleBy11 m :=
by sorry

theorem all_replacements_divisible_by_11 : 
  ∀ m ∈ allPossibleReplacements 909090909, isDivisibleBy11 m :=
by sorry

end smallest_number_divisible_by_11_after_change_all_replacements_divisible_by_11_l3119_311960


namespace log_relation_l3119_311952

theorem log_relation (c b : ℝ) (hc : c = Real.log 625 / Real.log 16) (hb : b = Real.log 25 / Real.log 2) : 
  c = b / 2 := by
sorry

end log_relation_l3119_311952


namespace show_episodes_l3119_311937

/-- Calculates the number of episodes in a show given the watching conditions -/
def num_episodes (days : ℕ) (episode_length : ℕ) (hours_per_day : ℕ) : ℕ :=
  (days * hours_per_day * 60) / episode_length

/-- Proves that the number of episodes in the show is 20 -/
theorem show_episodes : num_episodes 5 30 2 = 20 := by
  sorry

end show_episodes_l3119_311937


namespace beach_ball_surface_area_l3119_311906

theorem beach_ball_surface_area (d : ℝ) (h : d = 15) :
  4 * Real.pi * (d / 2)^2 = 225 * Real.pi := by
  sorry

end beach_ball_surface_area_l3119_311906


namespace max_value_of_sum_products_l3119_311965

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 →
  a * b + b * c + c * d + d * a ≤ 10000 ∧ 
  ∃ (a' b' c' d' : ℝ), a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ d' ≥ 0 ∧ 
    a' + b' + c' + d' = 200 ∧
    a' * b' + b' * c' + c' * d' + d' * a' = 10000 := by
  sorry

end max_value_of_sum_products_l3119_311965


namespace monkeys_for_three_bananas_l3119_311950

/-- The number of monkeys needed to eat a given number of bananas in 8 minutes -/
def monkeys_needed (bananas : ℕ) : ℕ :=
  bananas

theorem monkeys_for_three_bananas :
  monkeys_needed 3 = 3 :=
by
  sorry

/-- Given condition: 8 monkeys take 8 minutes to eat 8 bananas -/
axiom eight_monkeys_eight_bananas : monkeys_needed 8 = 8

end monkeys_for_three_bananas_l3119_311950


namespace tom_catch_l3119_311942

/-- The number of trout Melanie caught -/
def melanie_catch : ℕ := 8

/-- The factor by which Tom's catch exceeds Melanie's -/
def tom_factor : ℕ := 2

/-- Tom's catch is equal to the product of Melanie's catch and Tom's factor -/
theorem tom_catch : melanie_catch * tom_factor = 16 := by
  sorry

end tom_catch_l3119_311942


namespace greatest_integer_difference_l3119_311968

theorem greatest_integer_difference (x y : ℝ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) :
  ∃ (n : ℕ), n = 2 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 5 < a ∧ a < 8 ∧ 8 < b ∧ b < 13 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end greatest_integer_difference_l3119_311968


namespace modulus_of_z_l3119_311986

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end modulus_of_z_l3119_311986


namespace no_integer_solution_l3119_311981

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), x * (x - y) + y * (y - z) + z * (z - x) = 3 ∧ x > y ∧ y > z :=
by sorry

end no_integer_solution_l3119_311981


namespace equation_equivalence_l3119_311979

theorem equation_equivalence (x y : ℝ) : 
  (2 * x - y = 3) ↔ (y = 2 * x - 3) := by
  sorry

end equation_equivalence_l3119_311979


namespace parabola_focus_and_directrix_l3119_311912

/-- A parabola is defined by the equation x² = 8y -/
def is_parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (f : ℝ × ℝ) (x y : ℝ) : Prop :=
  is_parabola x y → f = (0, 2)

/-- The directrix of a parabola is a line perpendicular to its axis of symmetry -/
def is_directrix (y : ℝ) : Prop :=
  ∀ x, is_parabola x y → y = -2

/-- Theorem: For the parabola x² = 8y, the focus is at (0, 2) and the directrix is y = -2 -/
theorem parabola_focus_and_directrix :
  (∀ x y, is_focus (0, 2) x y) ∧ is_directrix (-2) := by
  sorry

end parabola_focus_and_directrix_l3119_311912
