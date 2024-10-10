import Mathlib

namespace mock_exam_participants_l3234_323453

/-- The number of students who took a mock exam -/
def total_students : ℕ := 400

/-- The number of girls who took the exam -/
def num_girls : ℕ := 100

/-- The proportion of boys who cleared the cut off -/
def boys_cleared_ratio : ℚ := 3/5

/-- The proportion of girls who cleared the cut off -/
def girls_cleared_ratio : ℚ := 4/5

/-- The total proportion of students who qualified -/
def total_qualified_ratio : ℚ := 13/20

theorem mock_exam_participants :
  ∃ (num_boys : ℕ),
    (boys_cleared_ratio * num_boys + girls_cleared_ratio * num_girls : ℚ) = 
    total_qualified_ratio * (num_boys + num_girls) ∧
    total_students = num_boys + num_girls :=
by sorry

end mock_exam_participants_l3234_323453


namespace smallest_composite_no_small_factors_l3234_323485

def is_composite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p < 20 → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 529 ∧ has_no_small_prime_factors 529) ∧
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end smallest_composite_no_small_factors_l3234_323485


namespace sampling_suitability_l3234_323449

/-- Represents a sampling scenario --/
structure SamplingScenario where
  population : ℕ
  sample_size : ℕ
  (valid : sample_size ≤ population)

/-- Determines if a sampling scenario is suitable for simple random sampling --/
def suitable_for_simple_random_sampling (scenario : SamplingScenario) : Prop :=
  scenario.sample_size ≤ 10 ∧ scenario.population ≤ 100

/-- Determines if a sampling scenario is suitable for systematic sampling --/
def suitable_for_systematic_sampling (scenario : SamplingScenario) : Prop :=
  scenario.sample_size > 10 ∧ scenario.population > 100

/-- The first sampling scenario --/
def scenario1 : SamplingScenario where
  population := 10
  sample_size := 2
  valid := by norm_num

/-- The second sampling scenario --/
def scenario2 : SamplingScenario where
  population := 1000
  sample_size := 50
  valid := by norm_num

/-- Theorem stating that the first scenario is suitable for simple random sampling
    and the second scenario is suitable for systematic sampling --/
theorem sampling_suitability :
  suitable_for_simple_random_sampling scenario1 ∧
  suitable_for_systematic_sampling scenario2 := by
  sorry


end sampling_suitability_l3234_323449


namespace inequality_and_equality_condition_l3234_323497

theorem inequality_and_equality_condition (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + y + z = 12) : 
  (x / y + y / z + z / x + 3 ≥ Real.sqrt x + Real.sqrt y + Real.sqrt z) ∧
  (x / y + y / z + z / x + 3 = Real.sqrt x + Real.sqrt y + Real.sqrt z ↔ x = 4 ∧ y = 4 ∧ z = 4) :=
by sorry

end inequality_and_equality_condition_l3234_323497


namespace cloth_sale_meters_l3234_323431

/-- Proves that the number of meters of cloth sold is 60, given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 8400)
    (h2 : profit_per_meter = 12)
    (h3 : cost_price_per_meter = 128) :
    total_selling_price / (cost_price_per_meter + profit_per_meter) = 60 := by
  sorry

#check cloth_sale_meters

end cloth_sale_meters_l3234_323431


namespace carnival_days_l3234_323471

theorem carnival_days (daily_income total_income : ℕ) 
  (h1 : daily_income = 144)
  (h2 : total_income = 3168) :
  total_income / daily_income = 22 := by
  sorry

end carnival_days_l3234_323471


namespace rhombus_shorter_diagonal_l3234_323410

/-- A rhombus with perimeter 9.6 and adjacent angles in ratio 1:2 has a shorter diagonal of length 2.4 -/
theorem rhombus_shorter_diagonal (p : ℝ) (r : ℚ) (d : ℝ) : 
  p = 9.6 → -- perimeter is 9.6
  r = 1/2 → -- ratio of adjacent angles is 1:2
  d = 2.4 -- shorter diagonal is 2.4
  := by sorry

end rhombus_shorter_diagonal_l3234_323410


namespace complex_fraction_sum_l3234_323473

theorem complex_fraction_sum (a b : ℝ) : 
  (2 + 3 * Complex.I) / Complex.I = Complex.mk a b → a + b = 1 := by
  sorry

end complex_fraction_sum_l3234_323473


namespace sanitizer_sprays_effectiveness_l3234_323498

theorem sanitizer_sprays_effectiveness (spray1_kill_rate spray2_kill_rate overlap_rate remaining_rate : Real) :
  spray1_kill_rate = 0.5 →
  overlap_rate = 0.05 →
  remaining_rate = 0.3 →
  1 - (spray1_kill_rate + spray2_kill_rate - overlap_rate) = remaining_rate →
  spray2_kill_rate = 0.15 := by
  sorry

end sanitizer_sprays_effectiveness_l3234_323498


namespace system_solution_exists_l3234_323408

theorem system_solution_exists (m : ℝ) : 
  (∃ x y : ℝ, y = (3 * m + 2) * x + 1 ∧ y = (5 * m - 4) * x + 5) ↔ m ≠ 3 :=
by sorry

end system_solution_exists_l3234_323408


namespace factory_temporary_stats_l3234_323481

/-- Represents the different employee categories in the factory -/
inductive EmployeeCategory
  | Technician
  | SkilledLaborer
  | Manager
  | Administrative

/-- Represents the employment status of an employee -/
inductive EmploymentStatus
  | Permanent
  | Temporary

/-- Structure to hold information about each employee category -/
structure CategoryInfo where
  category : EmployeeCategory
  percentage : Float
  permanentPercentage : Float
  weeklyHours : Nat

def factory : List CategoryInfo := [
  { category := EmployeeCategory.Technician, percentage := 0.4, permanentPercentage := 0.6, weeklyHours := 45 },
  { category := EmployeeCategory.SkilledLaborer, percentage := 0.3, permanentPercentage := 0.5, weeklyHours := 40 },
  { category := EmployeeCategory.Manager, percentage := 0.2, permanentPercentage := 0.8, weeklyHours := 50 },
  { category := EmployeeCategory.Administrative, percentage := 0.1, permanentPercentage := 0.9, weeklyHours := 35 }
]

def totalEmployees : Nat := 100

/-- Calculate the percentage of temporary employees -/
def calculateTemporaryPercentage (factoryInfo : List CategoryInfo) : Float :=
  factoryInfo.foldl (fun acc info => 
    acc + info.percentage * (1 - info.permanentPercentage)) 0

/-- Calculate the total weekly hours worked by temporary employees -/
def calculateTemporaryHours (factoryInfo : List CategoryInfo) (totalEmp : Nat) : Float :=
  factoryInfo.foldl (fun acc info => 
    acc + (info.percentage * totalEmp.toFloat * (1 - info.permanentPercentage) * info.weeklyHours.toFloat)) 0

theorem factory_temporary_stats :
  calculateTemporaryPercentage factory = 0.36 ∧ 
  calculateTemporaryHours factory totalEmployees = 1555 := by
  sorry


end factory_temporary_stats_l3234_323481


namespace stopped_clock_more_accurate_l3234_323415

/-- Represents the frequency of showing correct time for a clock --/
structure ClockAccuracy where
  correct_times_per_day : ℚ

/-- A clock that is one minute slow --/
def slow_clock : ClockAccuracy where
  correct_times_per_day := 1 / 720

/-- A stopped clock --/
def stopped_clock : ClockAccuracy where
  correct_times_per_day := 2

theorem stopped_clock_more_accurate : 
  stopped_clock.correct_times_per_day > slow_clock.correct_times_per_day := by
  sorry

#check stopped_clock_more_accurate

end stopped_clock_more_accurate_l3234_323415


namespace linear_function_property_l3234_323480

theorem linear_function_property (x : ℝ) : ∃ x > -1, -2 * x + 2 ≥ 4 := by
  sorry

end linear_function_property_l3234_323480


namespace sugar_solution_percentage_l3234_323479

theorem sugar_solution_percentage (original_percentage : ℝ) : 
  (3/4 : ℝ) * original_percentage + (1/4 : ℝ) * 28 = 16 → 
  original_percentage = 12 := by
sorry

end sugar_solution_percentage_l3234_323479


namespace power_of_three_simplification_l3234_323468

theorem power_of_three_simplification :
  3^2012 - 6 * 3^2013 + 2 * 3^2014 = 3^2012 := by
  sorry

end power_of_three_simplification_l3234_323468


namespace rabbit_can_cross_tracks_l3234_323437

/-- The distance from the rabbit (point A) to the railway track -/
def rabbit_distance : ℝ := 160

/-- The speed of the train -/
def train_speed : ℝ := 30

/-- The initial distance of the train from point T -/
def train_initial_distance : ℝ := 300

/-- The speed of the rabbit -/
def rabbit_speed : ℝ := 15

/-- The lower bound of the safe crossing distance -/
def lower_bound : ℝ := 23.21

/-- The upper bound of the safe crossing distance -/
def upper_bound : ℝ := 176.79

theorem rabbit_can_cross_tracks :
  ∃ x : ℝ, lower_bound < x ∧ x < upper_bound ∧
  (((rabbit_distance ^ 2 + x ^ 2).sqrt / rabbit_speed) < ((train_initial_distance + x) / train_speed)) :=
by sorry

end rabbit_can_cross_tracks_l3234_323437


namespace symmetric_lines_l3234_323469

/-- Given two lines l and k symmetric with respect to y = x, prove that if l has equation y = ax + b, then k has equation y = (1/a)x - (b/a) -/
theorem symmetric_lines (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let l := {p : ℝ × ℝ | p.2 = a * p.1 + b}
  let k := {p : ℝ × ℝ | p.2 = (1/a) * p.1 - b/a}
  let symmetry := {p : ℝ × ℝ | p.1 = p.2}
  (∀ p, p ∈ l ↔ (p.2, p.1) ∈ k) ∧ (∀ p, p ∈ k ↔ (p.2, p.1) ∈ l) :=
by sorry

end symmetric_lines_l3234_323469


namespace body_lotion_cost_is_60_l3234_323412

/-- Represents the cost of items and total spent at Target --/
structure TargetPurchase where
  tanya_face_moisturizer_cost : ℕ
  tanya_face_moisturizer_count : ℕ
  tanya_body_lotion_count : ℕ
  total_spent : ℕ

/-- Calculates the cost of each body lotion based on the given conditions --/
def body_lotion_cost (p : TargetPurchase) : ℕ :=
  let tanya_total := p.tanya_face_moisturizer_cost * p.tanya_face_moisturizer_count + 
                     p.tanya_body_lotion_count * (p.total_spent / 3)
  (p.total_spent / 3 - p.tanya_face_moisturizer_cost * p.tanya_face_moisturizer_count) / p.tanya_body_lotion_count

/-- Theorem stating that the cost of each body lotion is $60 --/
theorem body_lotion_cost_is_60 (p : TargetPurchase) 
  (h1 : p.tanya_face_moisturizer_cost = 50)
  (h2 : p.tanya_face_moisturizer_count = 2)
  (h3 : p.tanya_body_lotion_count = 4)
  (h4 : p.total_spent = 1020) :
  body_lotion_cost p = 60 := by
  sorry


end body_lotion_cost_is_60_l3234_323412


namespace total_money_is_102_l3234_323493

def jack_money : ℕ := 26

def ben_money (jack : ℕ) : ℕ := jack - 9

def eric_money (ben : ℕ) : ℕ := ben - 10

def anna_money (jack : ℕ) : ℕ := jack * 2

def total_money (eric ben jack anna : ℕ) : ℕ := eric + ben + jack + anna

theorem total_money_is_102 :
  total_money (eric_money (ben_money jack_money)) (ben_money jack_money) jack_money (anna_money jack_money) = 102 := by
  sorry

end total_money_is_102_l3234_323493


namespace trigonometric_equation_solution_l3234_323416

theorem trigonometric_equation_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  ∀ x : ℝ, 0 < x → x < 2 * Real.pi →
    (Real.sin (3 * x) + a * Real.sin (2 * x) + 2 * Real.sin x = 0) →
    (x = 0 ∨ x = Real.pi) :=
by sorry

end trigonometric_equation_solution_l3234_323416


namespace mean_equality_implies_y_value_l3234_323426

theorem mean_equality_implies_y_value :
  let mean1 := (7 + 10 + 15 + 23) / 4
  let mean2 := (18 + y + 30) / 3
  mean1 = mean2 → y = -6.75 := by
sorry

end mean_equality_implies_y_value_l3234_323426


namespace min_brownies_is_36_l3234_323459

/-- Represents the dimensions of a rectangular pan of brownies -/
structure BrowniePan where
  length : ℕ
  width : ℕ

/-- Calculates the total number of brownies in the pan -/
def total_brownies (pan : BrowniePan) : ℕ := pan.length * pan.width

/-- Calculates the number of brownies on the perimeter of the pan -/
def perimeter_brownies (pan : BrowniePan) : ℕ := 2 * (pan.length + pan.width) - 4

/-- Calculates the number of brownies in the interior of the pan -/
def interior_brownies (pan : BrowniePan) : ℕ := (pan.length - 2) * (pan.width - 2)

/-- Checks if the pan satisfies the perimeter-to-interior ratio condition -/
def satisfies_ratio (pan : BrowniePan) : Prop :=
  perimeter_brownies pan = 2 * interior_brownies pan

/-- The main theorem stating that 36 is the smallest number of brownies satisfying all conditions -/
theorem min_brownies_is_36 :
  ∃ (pan : BrowniePan), satisfies_ratio pan ∧
    total_brownies pan = 36 ∧
    (∀ (other_pan : BrowniePan), satisfies_ratio other_pan →
      total_brownies other_pan ≥ 36) :=
  sorry

end min_brownies_is_36_l3234_323459


namespace prop_2_prop_4_l3234_323496

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Axiom: m and n are different lines
axiom different_lines : m ≠ n

-- Axiom: α and β are different planes
axiom different_planes : α ≠ β

-- Theorem 1 (Proposition ②)
theorem prop_2 : 
  perpendicular_line_plane m α → 
  perpendicular_line_plane n β → 
  perpendicular m n → 
  perpendicular_plane α β :=
sorry

-- Theorem 2 (Proposition ④)
theorem prop_4 : 
  perpendicular_line_plane m α → 
  parallel_line_plane n β → 
  parallel_plane α β → 
  perpendicular m n :=
sorry

end prop_2_prop_4_l3234_323496


namespace arithmetic_sequence_general_term_l3234_323455

theorem arithmetic_sequence_general_term (a : ℕ → ℝ) :
  a 1 = 1 ∧ (∀ n : ℕ, n ≥ 2 → a n - a (n - 1) = 2) →
  ∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_general_term_l3234_323455


namespace fewer_correct_answers_l3234_323462

-- Define the number of correct answers for each person
def cherry_correct : ℕ := 17
def nicole_correct : ℕ := 22
def kim_correct : ℕ := cherry_correct + 8

-- State the theorem
theorem fewer_correct_answers :
  kim_correct - nicole_correct = 3 ∧
  nicole_correct < kim_correct :=
by sorry

end fewer_correct_answers_l3234_323462


namespace blake_guarantee_four_ruby_prevent_more_than_four_largest_guaranteed_score_l3234_323482

/-- Represents a cell on the infinite grid --/
structure Cell :=
  (x : Int) (y : Int)

/-- Represents the color of a cell --/
inductive Color
  | White
  | Blue
  | Red

/-- Represents the game state --/
structure GameState :=
  (grid : Cell → Color)

/-- Blake's score is the size of the largest blue simple polygon --/
def blakeScore (state : GameState) : Nat :=
  sorry

/-- Blake's strategy to color adjacent cells --/
def blakeStrategy (state : GameState) : Cell :=
  sorry

/-- Ruby's strategy to block Blake --/
def rubyStrategy (state : GameState) : Cell × Cell :=
  sorry

/-- The game play function --/
def playGame (initialState : GameState) : Nat :=
  sorry

theorem blake_guarantee_four :
  ∀ (initialState : GameState),
    (∀ c, initialState.grid c = Color.White) →
    ∃ (finalState : GameState),
      blakeScore finalState ≥ 4 :=
sorry

theorem ruby_prevent_more_than_four :
  ∀ (initialState : GameState),
    (∀ c, initialState.grid c = Color.White) →
    ¬∃ (finalState : GameState),
      blakeScore finalState > 4 :=
sorry

theorem largest_guaranteed_score :
  ∀ (initialState : GameState),
    (∀ c, initialState.grid c = Color.White) →
    (∃ (finalState : GameState), blakeScore finalState = 4) ∧
    (¬∃ (finalState : GameState), blakeScore finalState > 4) :=
sorry

end blake_guarantee_four_ruby_prevent_more_than_four_largest_guaranteed_score_l3234_323482


namespace equation_is_linear_l3234_323446

/-- An equation is linear with one variable if it can be written in the form ax + b = 0,
    where a and b are constants and a ≠ 0. --/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation 7x + 5 = 6(x - 1) --/
def f (x : ℝ) : ℝ := 7 * x + 5 - (6 * (x - 1))

theorem equation_is_linear : is_linear_equation_one_var f := by
  sorry

end equation_is_linear_l3234_323446


namespace total_watermelons_l3234_323400

def jason_watermelons : ℕ := 37
def sandy_watermelons : ℕ := 11

theorem total_watermelons : jason_watermelons + sandy_watermelons = 48 := by
  sorry

end total_watermelons_l3234_323400


namespace line_through_points_l3234_323495

/-- A structure representing a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a line passing through two points -/
def Line (p1 p2 : Point) :=
  {p : Point | (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)}

/-- The statement of the problem -/
theorem line_through_points :
  ∃ (s : Finset ℤ), s.card = 4 ∧
    (∀ m ∈ s, m > 0) ∧
    (∀ m ∈ s, ∃ k : ℤ, k > 0 ∧
      Line (Point.mk (-m) 0) (Point.mk 0 2) (Point.mk 7 k)) ∧
    (∀ m : ℤ, m > 0 →
      (∃ k : ℤ, k > 0 ∧
        Line (Point.mk (-m) 0) (Point.mk 0 2) (Point.mk 7 k)) →
      m ∈ s) := by
  sorry

end line_through_points_l3234_323495


namespace equality_multiplication_l3234_323476

theorem equality_multiplication (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end equality_multiplication_l3234_323476


namespace g_equivalence_l3234_323458

theorem g_equivalence (x : Real) : 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) - 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2) = 
  -Real.cos (2 * x) := by sorry

end g_equivalence_l3234_323458


namespace fraction_simplification_l3234_323478

theorem fraction_simplification : 
  (3 - 6 + 12 - 24 + 48 - 96) / (6 - 12 + 24 - 48 + 96 - 192) = 1 / 2 := by
  sorry

end fraction_simplification_l3234_323478


namespace stationery_purchase_l3234_323444

theorem stationery_purchase (brother_money sister_money : ℕ) : 
  brother_money = 2 * sister_money →
  brother_money - 180 = sister_money - 30 →
  brother_money = 300 ∧ sister_money = 150 := by
  sorry

end stationery_purchase_l3234_323444


namespace smallest_tetrahedron_volume_ellipsoid_l3234_323440

/-- The smallest volume of a tetrahedron bounded by a tangent plane to an ellipsoid and coordinate planes -/
theorem smallest_tetrahedron_volume_ellipsoid (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ (V : ℝ), V ≥ (Real.sqrt 3 * a * b * c) / 2 → 
  ∃ (x y z : ℝ), x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 ∧ 
    V = (1/6) * (a^2/x) * (b^2/y) * (c^2/z) :=
by sorry

end smallest_tetrahedron_volume_ellipsoid_l3234_323440


namespace vegetarian_eaters_count_l3234_323411

/-- Given a family where some members eat vegetarian, some eat non-vegetarian, and some eat both,
    this theorem proves that the total number of people who eat vegetarian food is 28. -/
theorem vegetarian_eaters_count (only_veg : ℕ) (both : ℕ) 
    (h1 : only_veg = 16) (h2 : both = 12) : only_veg + both = 28 := by
  sorry

end vegetarian_eaters_count_l3234_323411


namespace average_weight_b_c_l3234_323477

theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 42 →
  b = 35 →
  (b + c) / 2 = 43 :=
by
  sorry

end average_weight_b_c_l3234_323477


namespace cos_180_degrees_l3234_323413

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end cos_180_degrees_l3234_323413


namespace garden_perimeter_l3234_323450

/-- The total perimeter of a rectangular garden with an attached triangular flower bed -/
theorem garden_perimeter (garden_length garden_width triangle_height : ℝ) 
  (hl : garden_length = 15)
  (hw : garden_width = 10)
  (ht : triangle_height = 6) :
  2 * (garden_length + garden_width) + 
  (Real.sqrt (garden_length^2 + triangle_height^2) + triangle_height) - 
  garden_length = 41 + Real.sqrt 261 := by
  sorry

end garden_perimeter_l3234_323450


namespace mall_sale_plate_cost_l3234_323418

theorem mall_sale_plate_cost 
  (treadmill_price : ℝ)
  (discount_rate : ℝ)
  (num_plates : ℕ)
  (plate_price : ℝ)
  (total_paid : ℝ)
  (h1 : treadmill_price = 1350)
  (h2 : discount_rate = 0.3)
  (h3 : num_plates = 2)
  (h4 : plate_price = 50)
  (h5 : total_paid = 1045) :
  treadmill_price * (1 - discount_rate) + num_plates * plate_price = total_paid ∧
  num_plates * plate_price = 100 := by
sorry

end mall_sale_plate_cost_l3234_323418


namespace sqrt_of_square_positive_l3234_323488

theorem sqrt_of_square_positive (a : ℝ) (h : a > 0) : Real.sqrt (a^2) = a := by
  sorry

end sqrt_of_square_positive_l3234_323488


namespace house_population_total_l3234_323441

/-- Represents the number of people on each floor of a three-story house. -/
structure HousePopulation where
  ground : ℕ
  first : ℕ
  second : ℕ

/-- Proves that given the conditions, the total number of people in the house is 60. -/
theorem house_population_total (h : HousePopulation) :
  (h.ground + h.first + h.second = 60) ∧
  (h.first + h.second = 35) ∧
  (h.ground + h.first = 45) ∧
  (h.first = (h.ground + h.first + h.second) / 3) :=
by sorry

end house_population_total_l3234_323441


namespace fraction_equality_l3234_323439

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 12) :
  m / q = 3 / 4 := by
  sorry

end fraction_equality_l3234_323439


namespace linda_spent_680_l3234_323423

/-- The total amount Linda spent on school supplies -/
def linda_total_spent : ℚ :=
  let notebook_price : ℚ := 1.20
  let notebook_quantity : ℕ := 3
  let pencil_box_price : ℚ := 1.50
  let pen_box_price : ℚ := 1.70
  notebook_price * notebook_quantity + pencil_box_price + pen_box_price

theorem linda_spent_680 : linda_total_spent = 6.80 := by
  sorry

end linda_spent_680_l3234_323423


namespace point_to_line_distance_l3234_323428

theorem point_to_line_distance (M : ℝ) : 
  (|(3 : ℝ) + Real.sqrt 3 * M - 4| / Real.sqrt (1 + 3) = 1) ↔ 
  (M = Real.sqrt 3 ∨ M = -(Real.sqrt 3) / 3) := by
sorry

end point_to_line_distance_l3234_323428


namespace product_sum_theorem_l3234_323438

theorem product_sum_theorem (a b : ℕ) : 
  10 ≤ a ∧ a ≤ 99 ∧ 
  10 ≤ b ∧ b ≤ 99 ∧ 
  a * b = 1656 ∧ 
  (a % 10) * b < 1000 →
  a + b = 110 := by
sorry

end product_sum_theorem_l3234_323438


namespace soccer_players_count_l3234_323465

theorem soccer_players_count (total_socks : ℕ) (socks_per_player : ℕ) : total_socks = 16 → socks_per_player = 2 → total_socks / socks_per_player = 8 := by
  sorry

end soccer_players_count_l3234_323465


namespace isosceles_triangle_from_quadratic_perimeter_l3234_323454

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Checks if a quadratic equation has two real roots -/
def has_two_real_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq > 0

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.base + 2*t.leg

theorem isosceles_triangle_from_quadratic_perimeter
  (k : ℝ)
  (eq : QuadraticEquation)
  (t : IsoscelesTriangle)
  (h1 : eq = { a := 1, b := -4, c := 2*k })
  (h2 : has_two_real_roots eq)
  (h3 : t.base = 1)
  (h4 : t.leg = 2) :
  perimeter t = 5 :=
sorry

end isosceles_triangle_from_quadratic_perimeter_l3234_323454


namespace abc_value_l3234_323494

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30) (hbc : b * c = 54) (hca : c * a = 45) :
  a * b * c = 270 := by
sorry

end abc_value_l3234_323494


namespace tournament_ranking_sequences_l3234_323457

/-- Represents a team in the tournament -/
inductive Team
| E | F | G | H | I | J | K

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  preliminary_matches : List Match
  semifinal_matches : List Match
  final_match : Match
  third_place_match : Match

/-- Represents a possible ranking sequence of four teams -/
structure RankingSequence where
  first : Team
  second : Team
  third : Team
  fourth : Team

/-- The main theorem to prove -/
theorem tournament_ranking_sequences (t : Tournament) :
  (t.preliminary_matches.length = 3) →
  (t.semifinal_matches.length = 2) →
  (∃ m ∈ t.preliminary_matches, m.team1 = Team.E ∧ m.team2 = Team.F) →
  (∃ m ∈ t.preliminary_matches, m.team1 = Team.G ∧ m.team2 = Team.H) →
  (∃ m ∈ t.preliminary_matches, m.team1 = Team.I ∧ m.team2 = Team.J) →
  (∃ m ∈ t.semifinal_matches, m.team2 = Team.K) →
  (∃ ranking_sequences : List RankingSequence,
    ranking_sequences.length = 16 ∧
    ∀ rs ∈ ranking_sequences,
      (rs.first ∈ [t.final_match.team1, t.final_match.team2]) ∧
      (rs.second ∈ [t.final_match.team1, t.final_match.team2]) ∧
      (rs.third ∈ [t.third_place_match.team1, t.third_place_match.team2]) ∧
      (rs.fourth ∈ [t.third_place_match.team1, t.third_place_match.team2])) :=
by
  sorry

end tournament_ranking_sequences_l3234_323457


namespace one_cow_one_bag_days_l3234_323466

/-- Given that 40 cows eat 40 bags of husk in 40 days, prove that one cow will eat one bag of husk in 40 days. -/
theorem one_cow_one_bag_days (cows bags days : ℕ) (h : cows = 40 ∧ bags = 40 ∧ days = 40) : 
  (cows * bags) / (cows * days) = 1 := by
  sorry

end one_cow_one_bag_days_l3234_323466


namespace problem_statement_l3234_323456

theorem problem_statement :
  -- Statement 1
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) ∧
  -- Statement 2
  (∀ p q : Prop, (p ∧ q) → (p ∨ q)) ∧
  (∃ p q : Prop, (p ∨ q) ∧ ¬(p ∧ q)) ∧
  -- Statement 4 (negation)
  ¬(∀ A B C D : Set α, (A ∪ B = A ∧ C ∩ D = C) → (A ⊆ B ∧ C ⊆ D)) :=
by sorry

end problem_statement_l3234_323456


namespace tray_height_is_five_l3234_323421

/-- The height of a tray formed by cutting and folding a square piece of paper -/
def trayHeight (sideLength : ℝ) (cutDistance : ℝ) (cutAngle : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the height of the tray is 5 under given conditions -/
theorem tray_height_is_five :
  trayHeight 120 (Real.sqrt 25) (π / 4) = 5 := by
  sorry

end tray_height_is_five_l3234_323421


namespace sticker_distribution_l3234_323425

theorem sticker_distribution (total_stickers : ℕ) (ratio_sum : ℕ) (sam_ratio : ℕ) (andrew_ratio : ℕ) :
  total_stickers = 1500 →
  ratio_sum = 1 + 1 + sam_ratio →
  sam_ratio = 3 →
  andrew_ratio = 1 →
  (total_stickers / ratio_sum * andrew_ratio) + (total_stickers / ratio_sum * sam_ratio * 2 / 3) = 900 :=
by sorry

end sticker_distribution_l3234_323425


namespace original_number_l3234_323429

theorem original_number (w : ℝ) : (1.125 * w) - (0.75 * w) = 30 → w = 80 := by
  sorry

end original_number_l3234_323429


namespace sum_inequality_l3234_323420

theorem sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 ≥ 3) :
  (x^2 + y^2 + z^2) / (x^5 + y^2 + z^2) +
  (x^2 + y^2 + z^2) / (y^5 + x^2 + z^2) +
  (x^2 + y^2 + z^2) / (z^5 + x^2 + y^2) ≤ 3 := by
  sorry

end sum_inequality_l3234_323420


namespace complex_equation_solution_l3234_323475

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : (a - 2*i) * i = b - i) : 
  a + b*i = -1 + 2*i := by sorry

end complex_equation_solution_l3234_323475


namespace surface_area_unchanged_l3234_323442

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℝ
  positive : side > 0

/-- Calculates the surface area of a cube -/
def surfaceArea (c : CubeDimensions) : ℝ := 6 * c.side^2

/-- Represents the original cube -/
def originalCube : CubeDimensions := ⟨4, by norm_num⟩

/-- Represents the cube to be removed -/
def removedCube : CubeDimensions := ⟨2, by norm_num⟩

/-- The number of faces of the removed cube that were initially exposed -/
def initiallyExposedFaces : ℕ := 3

theorem surface_area_unchanged :
  surfaceArea originalCube = 
  surfaceArea originalCube - 
  (initiallyExposedFaces : ℝ) * surfaceArea removedCube + 
  (initiallyExposedFaces : ℝ) * surfaceArea removedCube :=
sorry

end surface_area_unchanged_l3234_323442


namespace technician_journey_percentage_l3234_323434

theorem technician_journey_percentage (D : ℝ) (h : D > 0) : 
  let total_distance := 2 * D
  let completed_distance := 0.65 * total_distance
  let outbound_distance := D
  let return_distance_completed := completed_distance - outbound_distance
  (return_distance_completed / D) * 100 = 30 := by
  sorry

end technician_journey_percentage_l3234_323434


namespace special_function_inequality_l3234_323430

/-- A function satisfying the given properties in the problem -/
structure SpecialFunction where
  f : ℝ → ℝ
  odd : ∀ x, f (-x) = -f x
  special_property : ∀ x, f (1 + x) + f (1 - x) = f 1
  decreasing_on_unit_interval : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f y < f x

/-- The main theorem to be proved -/
theorem special_function_inequality (sf : SpecialFunction) :
  sf.f (-2 + Real.sqrt 2 / 2) < -sf.f (10 / 3) ∧ -sf.f (10 / 3) < sf.f (9 / 2) := by
  sorry


end special_function_inequality_l3234_323430


namespace negation_of_existence_equivalence_l3234_323407

theorem negation_of_existence_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end negation_of_existence_equivalence_l3234_323407


namespace tiffany_sunscreen_cost_l3234_323432

/-- Calculates the total cost of sunscreen for a beach trip -/
def sunscreenCost (reapplyTime hours applicationAmount bottleAmount bottlePrice : ℕ) : ℕ :=
  let applications := hours / reapplyTime
  let totalAmount := applications * applicationAmount
  let bottlesNeeded := (totalAmount + bottleAmount - 1) / bottleAmount  -- Ceiling division
  bottlesNeeded * bottlePrice

/-- Theorem: The total cost of sunscreen for Tiffany's beach trip is $14 -/
theorem tiffany_sunscreen_cost :
  sunscreenCost 2 16 3 12 7 = 14 := by
  sorry

end tiffany_sunscreen_cost_l3234_323432


namespace sequence_is_arithmetic_sum_of_4th_and_6th_is_zero_l3234_323404

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := -n^2 + 9*n

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℤ := S n - S (n-1)

/-- The sequence {a_n} is arithmetic -/
theorem sequence_is_arithmetic : ∀ n : ℕ+, a (n+1) - a n = a (n+2) - a (n+1) :=
sorry

/-- The sum of the 4th and 6th terms is zero -/
theorem sum_of_4th_and_6th_is_zero : a 4 + a 6 = 0 :=
sorry

end sequence_is_arithmetic_sum_of_4th_and_6th_is_zero_l3234_323404


namespace handshake_problem_l3234_323414

theorem handshake_problem (n : ℕ) : n * (n - 1) / 2 = 78 → n = 13 := by
  sorry

end handshake_problem_l3234_323414


namespace odot_count_53_l3234_323461

/-- Represents a sequence of four symbols -/
structure SymbolSequence :=
  (symbols : Fin 4 → Char)
  (odot_count : (symbols 2 = '⊙' ∧ symbols 3 = '⊙') ∨ (symbols 1 = '⊙' ∧ symbols 2 = '⊙') ∨ (symbols 0 = '⊙' ∧ symbols 3 = '⊙'))

/-- Counts the occurrences of a symbol in the repeated pattern up to a given position -/
def count_symbol (seq : SymbolSequence) (symbol : Char) (n : Nat) : Nat :=
  (n / 4) * 2 + if n % 4 ≥ 3 then 2 else if n % 4 ≥ 2 then 1 else 0

/-- The main theorem stating that the count of ⊙ in the first 53 positions is 26 -/
theorem odot_count_53 (seq : SymbolSequence) : count_symbol seq '⊙' 53 = 26 := by
  sorry


end odot_count_53_l3234_323461


namespace lateral_surface_area_of_pyramid_l3234_323472

theorem lateral_surface_area_of_pyramid (sin_alpha : ℝ) (diagonal_section_area : ℝ) :
  sin_alpha = 15 / 17 →
  diagonal_section_area = 3 * Real.sqrt 34 →
  (4 * diagonal_section_area) / (2 * Real.sqrt ((1 + (-Real.sqrt (1 - sin_alpha^2))) / 2)) = 68 :=
by sorry

end lateral_surface_area_of_pyramid_l3234_323472


namespace linear_function_decreasing_l3234_323419

/-- A linear function y = (2m + 2)x + 5 is decreasing if and only if m < -1 -/
theorem linear_function_decreasing (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((2*m + 2)*x₁ + 5) > ((2*m + 2)*x₂ + 5)) ↔ m < -1 :=
by sorry

end linear_function_decreasing_l3234_323419


namespace mother_notebooks_l3234_323443

/-- The number of notebooks the mother initially had -/
def initial_notebooks : ℕ := sorry

/-- The number of children -/
def num_children : ℕ := sorry

/-- If each child gets 13 notebooks, the mother has 8 notebooks left -/
axiom condition1 : initial_notebooks = 13 * num_children + 8

/-- If each child gets 15 notebooks, all notebooks are distributed -/
axiom condition2 : initial_notebooks = 15 * num_children

theorem mother_notebooks : initial_notebooks = 60 := by sorry

end mother_notebooks_l3234_323443


namespace min_surface_area_to_volume_ratio_cylinder_in_sphere_l3234_323460

/-- For a right circular cylinder inscribed in a sphere of radius R,
    the minimum ratio of surface area to volume is (((4^(1/3)) + 1)^(3/2)) / R. -/
theorem min_surface_area_to_volume_ratio_cylinder_in_sphere (R : ℝ) (h : R > 0) :
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧ r^2 + (h/2)^2 = R^2 ∧
    ∀ (r' h' : ℝ), r' > 0 → h' > 0 → r'^2 + (h'/2)^2 = R^2 →
      (2 * π * r * (r + h)) / (π * r^2 * h) ≥ (((4^(1/3) : ℝ) + 1)^(3/2)) / R :=
by sorry

end min_surface_area_to_volume_ratio_cylinder_in_sphere_l3234_323460


namespace inequality_product_l3234_323467

theorem inequality_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end inequality_product_l3234_323467


namespace product_closest_to_2400_l3234_323427

def options : List ℝ := [210, 240, 2100, 2400, 24000]

theorem product_closest_to_2400 : 
  let product := 0.000315 * 7928564
  ∀ x ∈ options, x ≠ 2400 → |product - 2400| < |product - x| := by
  sorry

end product_closest_to_2400_l3234_323427


namespace circus_performers_standing_time_l3234_323486

/-- The combined time that Pulsar, Polly, and Petra stand on their back legs -/
theorem circus_performers_standing_time :
  let pulsar_time : ℕ := 10
  let polly_time : ℕ := 3 * pulsar_time
  let petra_time : ℕ := polly_time / 6
  pulsar_time + polly_time + petra_time = 45 := by
sorry

end circus_performers_standing_time_l3234_323486


namespace sum_of_coefficients_l3234_323451

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -31 := by
sorry

end sum_of_coefficients_l3234_323451


namespace matthews_cows_l3234_323422

/-- Proves that Matthews has 60 cows given the problem conditions -/
theorem matthews_cows :
  ∀ (matthews aaron marovich : ℕ),
  aaron = 4 * matthews →
  aaron + matthews = marovich + 30 →
  matthews + aaron + marovich = 570 →
  matthews = 60 := by
sorry

end matthews_cows_l3234_323422


namespace bicycle_price_l3234_323417

theorem bicycle_price (upfront_payment : ℝ) (upfront_percentage : ℝ) 
  (h1 : upfront_payment = 120)
  (h2 : upfront_percentage = 0.2) :
  upfront_payment / upfront_percentage = 600 := by
  sorry

end bicycle_price_l3234_323417


namespace parabola_y_values_l3234_323470

def f (x : ℝ) := -(x - 2)^2

theorem parabola_y_values :
  ∀ (y₁ y₂ y₃ : ℝ),
  f (-1) = y₁ → f 1 = y₂ → f 4 = y₃ →
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end parabola_y_values_l3234_323470


namespace triangle_area_l3234_323463

/-- Given a triangle ABC with side a = 2, angle A = 30°, and angle C = 45°, 
    prove that its area S is equal to √3 + 1 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = 2 → 
  A = π / 6 → 
  C = π / 4 → 
  A + B + C = π → 
  a / Real.sin A = c / Real.sin C → 
  S = (1 / 2) * a * c * Real.sin B → 
  S = Real.sqrt 3 + 1 := by
  sorry


end triangle_area_l3234_323463


namespace union_M_N_equals_real_l3234_323409

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 > 4}
def N : Set ℝ := {x : ℝ | x < 3}

-- Statement to prove
theorem union_M_N_equals_real : M ∪ N = Set.univ := by sorry

end union_M_N_equals_real_l3234_323409


namespace no_integer_solution_l3234_323402

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 + z^2 = x*y*z - 1 := by
sorry

end no_integer_solution_l3234_323402


namespace range_of_m_l3234_323424

def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < m^2}

def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m)

theorem range_of_m : 
  {m : ℝ | necessary_not_sufficient m} = {m : ℝ | -1/2 ≤ m ∧ m ≤ 2} := by sorry

end range_of_m_l3234_323424


namespace club_group_size_l3234_323491

theorem club_group_size (N : ℕ) (x : ℕ) 
  (h1 : 10 < N ∧ N < 40)
  (h2 : (N - 3) % 5 = 0 ∧ (N - 3) % 6 = 0)
  (h3 : N % x = 5)
  : x = 7 := by
  sorry

end club_group_size_l3234_323491


namespace inequality_implies_linear_form_l3234_323490

/-- A function satisfying the given inequality for all real x and y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + y ≤ f (f (f x))

/-- The theorem stating that any function satisfying the inequality must be of the form f(x) = C - x -/
theorem inequality_implies_linear_form {f : ℝ → ℝ} (h : SatisfiesInequality f) :
  ∃ C : ℝ, ∀ x : ℝ, f x = C - x :=
sorry

end inequality_implies_linear_form_l3234_323490


namespace expand_product_l3234_323436

theorem expand_product (x : ℝ) : 3 * (x^2 - 5*x + 6) * (x^2 + 8*x - 10) = 3*x^4 + 9*x^3 - 132*x^2 + 294*x - 180 := by
  sorry

end expand_product_l3234_323436


namespace odd_periodic_function_property_l3234_323452

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 5) 
  (h1 : f 1 = 1) 
  (h2 : f 2 = 2) : 
  f 23 + f (-14) = -1 := by
  sorry

end odd_periodic_function_property_l3234_323452


namespace power_product_specific_calculation_l3234_323483

-- Define the power function for rational numbers
def rat_pow (a : ℚ) (n : ℕ) : ℚ := a ^ n

-- Theorem 1: For any rational numbers a and b, and positive integer n, (ab)^n = a^n * b^n
theorem power_product (a b : ℚ) (n : ℕ+) : rat_pow (a * b) n = rat_pow a n * rat_pow b n := by
  sorry

-- Theorem 2: (3/2)^2019 * (-2/3)^2019 = -1
theorem specific_calculation : rat_pow (3/2) 2019 * rat_pow (-2/3) 2019 = -1 := by
  sorry

end power_product_specific_calculation_l3234_323483


namespace fencing_cost_per_metre_l3234_323433

-- Define the ratio of the sides
def ratio_length : ℚ := 3
def ratio_width : ℚ := 4

-- Define the area of the field
def area : ℚ := 9408

-- Define the total cost of fencing
def total_cost : ℚ := 98

-- Statement to prove
theorem fencing_cost_per_metre :
  let length := (ratio_length * Real.sqrt (area / (ratio_length * ratio_width)))
  let width := (ratio_width * Real.sqrt (area / (ratio_length * ratio_width)))
  let perimeter := 2 * (length + width)
  total_cost / perimeter = 0.25 := by sorry

end fencing_cost_per_metre_l3234_323433


namespace maria_white_towels_l3234_323405

/-- The number of green towels Maria bought -/
def green_towels : ℕ := 40

/-- The number of towels Maria gave to her mother -/
def towels_given : ℕ := 65

/-- The number of towels Maria ended up with -/
def towels_left : ℕ := 19

/-- The number of white towels Maria bought -/
def white_towels : ℕ := green_towels + towels_given - towels_left

theorem maria_white_towels : white_towels = 44 := by
  sorry

end maria_white_towels_l3234_323405


namespace sqrt_product_equation_l3234_323445

theorem sqrt_product_equation (y : ℝ) (h_pos : y > 0) 
  (h_eq : Real.sqrt (12 * y) * Real.sqrt (25 * y) * Real.sqrt (5 * y) * Real.sqrt (20 * y) = 40) :
  y = (Real.sqrt 30 * Real.rpow 3 (1/4)) / 15 := by
sorry

end sqrt_product_equation_l3234_323445


namespace angle_ABC_less_than_60_degrees_l3234_323474

/-- A triangle with vertices A, B, and C -/
structure Triangle (V : Type*) where
  A : V
  B : V
  C : V

/-- The angle at vertex B in a triangle -/
def angle_at_B {V : Type*} (t : Triangle V) : ℝ := sorry

/-- The altitude from vertex A in a triangle -/
def altitude_from_A {V : Type*} (t : Triangle V) : ℝ := sorry

/-- The median from vertex B in a triangle -/
def median_from_B {V : Type*} (t : Triangle V) : ℝ := sorry

/-- Predicate to check if a triangle is acute-angled -/
def is_acute_angled {V : Type*} (t : Triangle V) : Prop := sorry

/-- Predicate to check if the altitude from A is the longest -/
def altitude_A_is_longest {V : Type*} (t : Triangle V) : Prop := sorry

theorem angle_ABC_less_than_60_degrees {V : Type*} (t : Triangle V) :
  is_acute_angled t →
  altitude_A_is_longest t →
  altitude_from_A t = median_from_B t →
  angle_at_B t < 60 := by sorry

end angle_ABC_less_than_60_degrees_l3234_323474


namespace quadratic_inequality_condition_l3234_323487

theorem quadratic_inequality_condition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → (-1 ≤ a ∧ a ≤ 3) := by sorry

end quadratic_inequality_condition_l3234_323487


namespace cone_volume_l3234_323406

/-- The volume of a cone with slant height 15 cm and height 9 cm is 432π cubic centimeters. -/
theorem cone_volume (π : ℝ) : 
  let l : ℝ := 15  -- slant height
  let h : ℝ := 9   -- height
  let r : ℝ := Real.sqrt (l^2 - h^2)  -- radius of the base
  (1/3 : ℝ) * π * r^2 * h = 432 * π := by
  sorry

end cone_volume_l3234_323406


namespace monday_messages_l3234_323447

/-- Proves that given the specified message sending pattern and average, 
    the number of messages sent on Monday must be 220. -/
theorem monday_messages (x : ℝ) : 
  (x + x/2 + 50 + 50 + 50) / 5 = 96 → x = 220 := by
  sorry

end monday_messages_l3234_323447


namespace triangle_function_sign_l3234_323464

/-- Triangle with ordered sides -/
structure OrderedTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a ≤ b
  hbc : b ≤ c

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : OrderedTriangle) : ℝ := sorry

/-- The inradius of a triangle -/
noncomputable def inradius (t : OrderedTriangle) : ℝ := sorry

/-- The angle C of a triangle -/
noncomputable def angle_C (t : OrderedTriangle) : ℝ := sorry

theorem triangle_function_sign (t : OrderedTriangle) :
  let f := t.a + t.b - 2 * circumradius t - 2 * inradius t
  let C := angle_C t
  (π / 3 ≤ C ∧ C < π / 2 → f > 0) ∧
  (C = π / 2 → f = 0) ∧
  (π / 2 < C ∧ C < π → f < 0) :=
sorry

end triangle_function_sign_l3234_323464


namespace combined_solid_sum_l3234_323401

/-- A right rectangular prism -/
structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- A pyramid added to a rectangular prism -/
structure PrismWithPyramid :=
  (prism : RectangularPrism)
  (pyramid_base_face : ℕ)

/-- The combined solid (prism and pyramid) -/
def CombinedSolid (pw : PrismWithPyramid) : ℕ × ℕ × ℕ :=
  let new_faces := pw.prism.faces - pw.pyramid_base_face + 4
  let new_edges := pw.prism.edges + 4
  let new_vertices := pw.prism.vertices + 1
  (new_faces, new_edges, new_vertices)

theorem combined_solid_sum (pw : PrismWithPyramid) 
  (h1 : pw.prism.faces = 6)
  (h2 : pw.prism.edges = 12)
  (h3 : pw.prism.vertices = 8)
  (h4 : pw.pyramid_base_face = 1) :
  let (f, e, v) := CombinedSolid pw
  f + e + v = 34 := by sorry

end combined_solid_sum_l3234_323401


namespace cards_left_l3234_323484

/-- The number of basketball card boxes Ben has -/
def basketball_boxes : ℕ := 4

/-- The number of cards in each basketball box -/
def basketball_cards_per_box : ℕ := 10

/-- The number of baseball card boxes Ben's mother gave him -/
def baseball_boxes : ℕ := 5

/-- The number of cards in each baseball box -/
def baseball_cards_per_box : ℕ := 8

/-- The number of cards Ben gave to his classmates -/
def cards_given_away : ℕ := 58

/-- Theorem stating the number of cards Ben has left -/
theorem cards_left : 
  basketball_boxes * basketball_cards_per_box + 
  baseball_boxes * baseball_cards_per_box - 
  cards_given_away = 22 := by sorry

end cards_left_l3234_323484


namespace chord_equation_l3234_323448

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define a chord passing through P with P as its midpoint
def is_chord_midpoint (x y : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    (x1 + x2) / 2 = P.1 ∧ (y1 + y2) / 2 = P.2 ∧
    x = (x2 - x1) ∧ y = (y2 - y1)

-- Theorem statement
theorem chord_equation :
  ∀ (x y : ℝ), is_chord_midpoint x y → 2 * x + 3 * y = 12 :=
by sorry

end chord_equation_l3234_323448


namespace quadratic_perfect_square_l3234_323492

theorem quadratic_perfect_square (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 12*x + k = (x + a)^2) ↔ k = 36 :=
sorry

end quadratic_perfect_square_l3234_323492


namespace video_views_equation_l3234_323435

/-- Represents the number of views on the first day -/
def initial_views : ℕ := 4400

/-- Represents the increase in views after 4 days -/
def increase_factor : ℕ := 10

/-- Represents the additional views after 2 more days -/
def additional_views : ℕ := 50000

/-- Represents the total views at the end -/
def total_views : ℕ := 94000

/-- Proves that the initial number of views satisfies the given equation -/
theorem video_views_equation : 
  increase_factor * initial_views + additional_views = total_views := by
  sorry

end video_views_equation_l3234_323435


namespace similar_triangle_longest_side_l3234_323489

/-- Given a triangle with sides 8, 10, and 12, and a similar triangle with perimeter 150,
    prove that the longest side of the similar triangle is 60. -/
theorem similar_triangle_longest_side
  (a b c : ℝ)
  (h_original : a = 8 ∧ b = 10 ∧ c = 12)
  (h_similar_perimeter : ∃ k : ℝ, k * (a + b + c) = 150)
  : ∃ x y z : ℝ, x = k * a ∧ y = k * b ∧ z = k * c ∧ max x (max y z) = 60 :=
sorry

end similar_triangle_longest_side_l3234_323489


namespace meaningful_fraction_l3234_323403

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (2*x + 3).sqrt / (x - 1)) ↔ x ≥ -3/2 ∧ x ≠ 1 := by
  sorry

end meaningful_fraction_l3234_323403


namespace equality_of_reciprocals_l3234_323499

theorem equality_of_reciprocals (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (3 : ℝ) ^ a = (4 : ℝ) ^ b ∧ (4 : ℝ) ^ b = (6 : ℝ) ^ c) : 
  2 / c = 2 / a + 1 / b :=
sorry

end equality_of_reciprocals_l3234_323499
