import Mathlib

namespace NUMINAMATH_CALUDE_train_speed_calculation_l766_76696

/-- Calculates the speed of a train given the parameters of a passing goods train -/
theorem train_speed_calculation (goods_train_speed : ℝ) (goods_train_length : ℝ) (passing_time : ℝ) : 
  goods_train_speed = 108 →
  goods_train_length = 340 →
  passing_time = 8 →
  ∃ (man_train_speed : ℝ), man_train_speed = 45 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l766_76696


namespace NUMINAMATH_CALUDE_concert_songs_theorem_l766_76685

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (sc : SongCounts) : Prop :=
  sc.hanna = 4 ∧
  sc.mary = 7 ∧
  sc.alina > sc.hanna ∧
  sc.alina < sc.mary ∧
  sc.tina > sc.hanna ∧
  sc.tina < sc.mary

/-- The total number of songs sung by the trios -/
def total_songs (sc : SongCounts) : ℕ :=
  (sc.mary + sc.alina + sc.tina + sc.hanna) / 3

/-- The main theorem to be proved -/
theorem concert_songs_theorem (sc : SongCounts) :
  satisfies_conditions sc → total_songs sc = 7 := by
  sorry

end NUMINAMATH_CALUDE_concert_songs_theorem_l766_76685


namespace NUMINAMATH_CALUDE_peaches_left_l766_76625

/-- Given baskets of peaches with specific initial conditions, proves the number of peaches left after removal. -/
theorem peaches_left (initial_baskets : Nat) (initial_peaches : Nat) (added_baskets : Nat) (added_peaches : Nat) (removed_peaches : Nat) : 
  initial_baskets = 5 →
  initial_peaches = 20 →
  added_baskets = 4 →
  added_peaches = 25 →
  removed_peaches = 10 →
  (initial_baskets * initial_peaches + added_baskets * added_peaches) - 
  ((initial_baskets + added_baskets) * removed_peaches) = 110 := by
  sorry

end NUMINAMATH_CALUDE_peaches_left_l766_76625


namespace NUMINAMATH_CALUDE_translation_sum_l766_76612

/-- A translation that moves a point 5 units right and 3 units up -/
def translation (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 5, p.2 + 3)

/-- Apply a translation n times to a point -/
def apply_translation (p : ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  Nat.recOn n p (fun _ q => translation q)

theorem translation_sum (initial : ℝ × ℝ) :
  let final := apply_translation initial 6
  final.1 + final.2 = 47 :=
sorry

end NUMINAMATH_CALUDE_translation_sum_l766_76612


namespace NUMINAMATH_CALUDE_spade_calculation_l766_76689

def spade (k : ℕ) (x y : ℝ) : ℝ := (x + y + k) * (x - y + k)

theorem spade_calculation : 
  let k : ℕ := 2
  spade k 5 (spade k 3 2) = -392 := by
sorry

end NUMINAMATH_CALUDE_spade_calculation_l766_76689


namespace NUMINAMATH_CALUDE_square_difference_theorem_l766_76683

theorem square_difference_theorem (a b A : ℝ) : 
  (5*a + 3*b)^2 = (5*a - 3*b)^2 + A → A = 60*a*b := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l766_76683


namespace NUMINAMATH_CALUDE_count_integer_solutions_l766_76622

theorem count_integer_solutions : 
  ∃! (S : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ S ↔ m > 0 ∧ n > 0 ∧ 5 / m + 3 / n = 1) ∧ 
    Finset.card S = 4 :=
by sorry

end NUMINAMATH_CALUDE_count_integer_solutions_l766_76622


namespace NUMINAMATH_CALUDE_cupcakes_problem_l766_76663

/-- Calculates the number of cupcakes per package given the initial number of cupcakes,
    the number of cupcakes eaten, and the number of packages. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Proves that given 50 initial cupcakes, 5 cupcakes eaten, and 9 equal packages,
    the number of cupcakes in each package is 5. -/
theorem cupcakes_problem :
  cupcakes_per_package 50 5 9 = 5 := by
  sorry

#eval cupcakes_per_package 50 5 9

end NUMINAMATH_CALUDE_cupcakes_problem_l766_76663


namespace NUMINAMATH_CALUDE_percentage_problem_l766_76600

theorem percentage_problem (P : ℝ) : (P / 100) * 150 - 40 = 50 → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l766_76600


namespace NUMINAMATH_CALUDE_bracelet_sales_average_l766_76652

theorem bracelet_sales_average (bike_cost : ℕ) (bracelet_price : ℕ) (selling_days : ℕ) 
  (h1 : bike_cost = 112)
  (h2 : bracelet_price = 1)
  (h3 : selling_days = 14) :
  (bike_cost / bracelet_price) / selling_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_sales_average_l766_76652


namespace NUMINAMATH_CALUDE_child_ticket_price_l766_76645

/-- Given the following information about a movie theater's ticket sales:
  - Total tickets sold is 900
  - Total revenue is $5,100
  - Adult ticket price is $7
  - Number of adult tickets sold is 500
  Prove that the price of a child's ticket is $4. -/
theorem child_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (adult_price : ℕ) 
  (adult_tickets : ℕ) 
  (h1 : total_tickets = 900) 
  (h2 : total_revenue = 5100) 
  (h3 : adult_price = 7) 
  (h4 : adult_tickets = 500) : 
  (total_revenue - adult_price * adult_tickets) / (total_tickets - adult_tickets) = 4 := by
sorry


end NUMINAMATH_CALUDE_child_ticket_price_l766_76645


namespace NUMINAMATH_CALUDE_barbell_cost_l766_76605

def number_of_barbells : ℕ := 3
def amount_given : ℕ := 850
def change_received : ℕ := 40

theorem barbell_cost :
  (amount_given - change_received) / number_of_barbells = 270 :=
by sorry

end NUMINAMATH_CALUDE_barbell_cost_l766_76605


namespace NUMINAMATH_CALUDE_no_nines_in_product_l766_76682

def first_number : Nat := 123456789
def second_number : Nat := 999999999

theorem no_nines_in_product : 
  ∀ d : Nat, d ∈ (first_number * second_number).digits 10 → d ≠ 9 := by
  sorry

end NUMINAMATH_CALUDE_no_nines_in_product_l766_76682


namespace NUMINAMATH_CALUDE_triangle_tangent_l766_76699

theorem triangle_tangent (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : Real.tan A = 1/2) (h3 : Real.cos B = (3 * Real.sqrt 10) / 10) : 
  Real.tan C = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_l766_76699


namespace NUMINAMATH_CALUDE_height_survey_is_census_l766_76631

/-- Represents a survey method --/
inductive SurveyMethod
| HeightOfStudents
| CarCrashResistance
| TVViewership
| ShoeSoleDurability

/-- Defines the properties of a census --/
structure Census where
  collectsAllData : Bool
  isFeasible : Bool

/-- Determines if a survey method is suitable for a census --/
def isSuitableForCensus (method : SurveyMethod) : Prop :=
  ∃ (c : Census), c.collectsAllData ∧ c.isFeasible

/-- The main theorem stating that measuring the height of all students is suitable for a census --/
theorem height_survey_is_census : isSuitableForCensus SurveyMethod.HeightOfStudents :=
  sorry

end NUMINAMATH_CALUDE_height_survey_is_census_l766_76631


namespace NUMINAMATH_CALUDE_division_of_decimals_l766_76648

theorem division_of_decimals : (0.2 : ℚ) / (0.005 : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l766_76648


namespace NUMINAMATH_CALUDE_sum_interior_angles_octagon_l766_76632

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The sum of the interior angles of an octagon is 1080 degrees -/
theorem sum_interior_angles_octagon :
  sum_interior_angles octagon_sides = 1080 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_octagon_l766_76632


namespace NUMINAMATH_CALUDE_point_translation_to_origin_l766_76637

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point by a given vector -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem point_translation_to_origin (A : Point) :
  translate A 3 2 = ⟨0, 0⟩ → A = ⟨-3, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_translation_to_origin_l766_76637


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l766_76688

/-- A step in the linear regression analysis process -/
inductive RegressionStep
  | predict : RegressionStep
  | collectData : RegressionStep
  | deriveEquation : RegressionStep
  | plotScatter : RegressionStep

/-- The correct sequence of steps in linear regression analysis -/
def correctSequence : List RegressionStep :=
  [RegressionStep.collectData, RegressionStep.plotScatter, 
   RegressionStep.deriveEquation, RegressionStep.predict]

/-- Theorem stating that the given sequence is the correct order of steps -/
theorem correct_regression_sequence :
  correctSequence = [RegressionStep.collectData, RegressionStep.plotScatter, 
                     RegressionStep.deriveEquation, RegressionStep.predict] := by
  sorry

end NUMINAMATH_CALUDE_correct_regression_sequence_l766_76688


namespace NUMINAMATH_CALUDE_quadratic_even_iff_a_eq_zero_l766_76659

/-- A quadratic function f(x) = x^2 + ax + b is even if and only if a = 0 -/
theorem quadratic_even_iff_a_eq_zero (a b : ℝ) :
  (∀ x : ℝ, x^2 + a*x + b = (-x)^2 + a*(-x) + b) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_a_eq_zero_l766_76659


namespace NUMINAMATH_CALUDE_smallest_y_coordinate_on_ellipse_l766_76669

/-- The ellipse is defined by the equation (x^2/49) + ((y-3)^2/25) = 1 -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2/49 + (y-3)^2/25 = 1

/-- The smallest y-coordinate of any point on the ellipse -/
def smallest_y_coordinate : ℝ := -2

/-- Theorem stating that the smallest y-coordinate of any point on the ellipse is -2 -/
theorem smallest_y_coordinate_on_ellipse :
  ∀ x y : ℝ, is_on_ellipse x y → y ≥ smallest_y_coordinate :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_coordinate_on_ellipse_l766_76669


namespace NUMINAMATH_CALUDE_paul_penny_count_l766_76647

theorem paul_penny_count (k m : ℕ+) : ∃! k, ∃ m, 1 + 3 * (k - 1) = 2017 - 5 * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_paul_penny_count_l766_76647


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l766_76634

theorem exponential_equation_solution :
  ∀ x : ℝ, (10 : ℝ)^x * (1000 : ℝ)^(2*x) = (100 : ℝ)^6 → x = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l766_76634


namespace NUMINAMATH_CALUDE_equation_solution_l766_76698

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → ((4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 5) → x = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l766_76698


namespace NUMINAMATH_CALUDE_max_prob_second_highest_l766_76654

variable (p₁ p₂ p₃ : ℝ)

-- Define the conditions
axiom prob_order : 0 < p₁ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ ≤ 1

-- Define the probability of winning two consecutive games for each scenario
def P_A := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

-- Theorem statement
theorem max_prob_second_highest :
  P_C p₁ p₂ p₃ > P_A p₁ p₂ p₃ ∧ P_C p₁ p₂ p₃ > P_B p₁ p₂ p₃ :=
sorry

end NUMINAMATH_CALUDE_max_prob_second_highest_l766_76654


namespace NUMINAMATH_CALUDE_two_intersection_points_l766_76677

/-- A line in the plane represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of intersection points between at least two of three given lines -/
def intersection_count (l1 l2 l3 : Line) : ℕ :=
  sorry

/-- The three lines from the problem -/
def line1 : Line := { a := -2, b := 3, c := 1 }
def line2 : Line := { a := 1, b := 2, c := 2 }
def line3 : Line := { a := 4, b := -6, c := 5 }

theorem two_intersection_points : intersection_count line1 line2 line3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_intersection_points_l766_76677


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l766_76681

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 60 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l766_76681


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l766_76697

/-- Given a function f(x) = ax³ + b sin(x) + 1 where f(1) = 5, prove that f(-1) = -3 -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * Real.sin x + 1) 
  (h2 : f 1 = 5) : 
  f (-1) = -3 :=
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l766_76697


namespace NUMINAMATH_CALUDE_cheerful_not_green_l766_76673

-- Define the universe of birds
variable (Bird : Type)

-- Define properties of birds
variable (green : Bird → Prop)
variable (cheerful : Bird → Prop)
variable (can_sing : Bird → Prop)
variable (can_dance : Bird → Prop)

-- Define Jen's collection of birds
variable (jen_birds : Set Bird)

-- State the theorem
theorem cheerful_not_green 
  (h1 : ∀ b ∈ jen_birds, cheerful b → can_sing b)
  (h2 : ∀ b ∈ jen_birds, green b → ¬can_dance b)
  (h3 : ∀ b ∈ jen_birds, ¬can_dance b → ¬can_sing b)
  : ∀ b ∈ jen_birds, cheerful b → ¬green b :=
by
  sorry


end NUMINAMATH_CALUDE_cheerful_not_green_l766_76673


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l766_76627

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ (z : ℂ), z = Complex.mk (a * (a - 1)) (a) ∧ z.re = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l766_76627


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l766_76664

theorem smallest_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (∀ m : ℕ, m > 0 → m % 4 = 3 → m % 3 = 2 → m ≥ n) ∧
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l766_76664


namespace NUMINAMATH_CALUDE_road_network_impossibility_l766_76617

/-- Represents an intersection in the road network -/
structure Intersection where
  branches : ℕ
  (branch_count : branches ≥ 2)

/-- Represents the road network -/
structure RoadNetwork where
  A : Intersection
  B : Intersection
  C : Intersection
  k_A : ℕ
  k_B : ℕ
  k_C : ℕ
  (k_A_def : A.branches = k_A)
  (k_B_def : B.branches = k_B)
  (k_C_def : C.branches = k_C)

/-- Total number of toll stations in the network -/
def total_toll_stations (rn : RoadNetwork) : ℕ :=
  4 + 4 * (rn.k_A + rn.k_B + rn.k_C)

/-- Theorem stating the impossibility of the road network design -/
theorem road_network_impossibility (rn : RoadNetwork) :
  ¬ ∃ (distances : Finset ℕ), 
    distances.card = (total_toll_stations rn).choose 2 ∧ 
    (∀ i ∈ distances, i ≤ distances.card) ∧
    (∀ i ≤ distances.card, i ∈ distances) :=
sorry

end NUMINAMATH_CALUDE_road_network_impossibility_l766_76617


namespace NUMINAMATH_CALUDE_amusement_park_earnings_l766_76633

/-- Calculates the total earnings of an amusement park for a week --/
theorem amusement_park_earnings 
  (ticket_price : ℕ)
  (weekday_visitors : ℕ)
  (saturday_visitors : ℕ)
  (sunday_visitors : ℕ) :
  ticket_price = 3 →
  weekday_visitors = 100 →
  saturday_visitors = 200 →
  sunday_visitors = 300 →
  (5 * weekday_visitors + saturday_visitors + sunday_visitors) * ticket_price = 3000 := by
  sorry

#check amusement_park_earnings

end NUMINAMATH_CALUDE_amusement_park_earnings_l766_76633


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l766_76684

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + 2 * b₀ = 1 ∧ 1 / a₀ + 1 / b₀ = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l766_76684


namespace NUMINAMATH_CALUDE_frank_has_twelve_cookies_l766_76676

-- Define the number of cookies each person has
def lucy_cookies : ℕ := 5
def millie_cookies : ℕ := 2 * lucy_cookies
def mike_cookies : ℕ := 3 * millie_cookies
def frank_cookies : ℕ := mike_cookies / 2 - 3

-- Theorem to prove
theorem frank_has_twelve_cookies : frank_cookies = 12 := by
  sorry

end NUMINAMATH_CALUDE_frank_has_twelve_cookies_l766_76676


namespace NUMINAMATH_CALUDE_fern_purchase_cost_l766_76649

/-- The total cost of purchasing high heels and ballet slippers -/
def total_cost (high_heel_price : ℝ) (ballet_slipper_ratio : ℝ) (ballet_slipper_count : ℕ) : ℝ :=
  high_heel_price + (ballet_slipper_ratio * high_heel_price * ballet_slipper_count)

/-- Theorem stating the total cost of Fern's purchase -/
theorem fern_purchase_cost :
  total_cost 60 (2/3) 5 = 260 := by
  sorry

end NUMINAMATH_CALUDE_fern_purchase_cost_l766_76649


namespace NUMINAMATH_CALUDE_inequality_iff_solution_set_l766_76601

def inequality (x : ℝ) : Prop :=
  (2 / (x + 2)) + (8 / (x + 6)) ≥ 2

theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ -6 < x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_iff_solution_set_l766_76601


namespace NUMINAMATH_CALUDE_rectangle_ratio_l766_76687

theorem rectangle_ratio (s w h : ℝ) (h1 : w > 0) (h2 : h > 0) (h3 : s > 0) : 
  (s + 2*w) * (s + h) = 3 * s^2 → h / w = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l766_76687


namespace NUMINAMATH_CALUDE_valid_star_arrangement_exists_l766_76692

/-- Represents a domino piece with two sides -/
structure Domino :=
  (side1 : Nat)
  (side2 : Nat)

/-- Represents a ray in the star arrangement -/
structure Ray :=
  (pieces : List Domino)
  (length : Nat)
  (sum : Nat)

/-- Represents the center of the star -/
structure Center :=
  (tiles : List Nat)

/-- Represents the entire star arrangement -/
structure StarArrangement :=
  (rays : List Ray)
  (center : Center)

/-- Checks if a domino arrangement is valid according to domino rules -/
def isValidDominoArrangement (arrangement : List Domino) : Prop :=
  sorry

/-- Checks if a ray is valid (correct length and sum) -/
def isValidRay (ray : Ray) : Prop :=
  ray.length ∈ [3, 4] ∧ ray.sum = 21 ∧ isValidDominoArrangement ray.pieces

/-- Checks if the center is valid -/
def isValidCenter (center : Center) : Prop :=
  center.tiles.length = 8 ∧
  (∀ n, n ∈ [1, 2, 3, 4, 5, 6] → n ∈ center.tiles) ∧
  (center.tiles.filter (· = 0)).length = 2

/-- Checks if the entire star arrangement is valid -/
def isValidStarArrangement (star : StarArrangement) : Prop :=
  star.rays.length = 8 ∧
  (∀ ray ∈ star.rays, isValidRay ray) ∧
  isValidCenter star.center

/-- The main theorem stating that a valid star arrangement exists -/
theorem valid_star_arrangement_exists : ∃ (star : StarArrangement), isValidStarArrangement star :=
  sorry

end NUMINAMATH_CALUDE_valid_star_arrangement_exists_l766_76692


namespace NUMINAMATH_CALUDE_product_of_roots_l766_76657

theorem product_of_roots (x z : ℝ) (h1 : x - z = 6) (h2 : x^3 - z^3 = 108) : x * z = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l766_76657


namespace NUMINAMATH_CALUDE_cosine_value_proof_l766_76674

theorem cosine_value_proof (α : ℝ) (h : Real.sin (π/6 - α) = 4/5) : 
  Real.cos (π/3 + α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_proof_l766_76674


namespace NUMINAMATH_CALUDE_complex_equation_solution_l766_76671

theorem complex_equation_solution : ∃ (z : ℂ), z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) ∧ z = (1/2 : ℂ) - Complex.I * ((Real.sqrt 3)/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l766_76671


namespace NUMINAMATH_CALUDE_expression_value_l766_76629

theorem expression_value (x : ℝ) (h : x^2 - x - 3 = 0) :
  (x + 2) * (x - 2) - x * (2 - x) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l766_76629


namespace NUMINAMATH_CALUDE_stating_transportation_equation_correct_l766_76643

/-- Represents the rate at which Vehicle A transports goods per day -/
def vehicle_a_rate : ℚ := 1/4

/-- Represents the time Vehicle A works alone -/
def vehicle_a_solo_time : ℚ := 1

/-- Represents the time both vehicles work together -/
def combined_work_time : ℚ := 1/2

/-- Represents the total amount of goods (100%) -/
def total_goods : ℚ := 1

/-- 
Theorem stating that the equation correctly represents the transportation situation
given the conditions of the problem
-/
theorem transportation_equation_correct (x : ℚ) : 
  vehicle_a_rate * vehicle_a_solo_time + 
  combined_work_time * (vehicle_a_rate + 1/x) = total_goods := by
  sorry

end NUMINAMATH_CALUDE_stating_transportation_equation_correct_l766_76643


namespace NUMINAMATH_CALUDE_hilt_fountain_distance_l766_76608

/-- The total distance Mrs. Hilt walks to the water fountain -/
def total_distance (desk_to_fountain : ℕ) (num_trips : ℕ) : ℕ :=
  2 * desk_to_fountain * num_trips

/-- Theorem: Mrs. Hilt walks 240 feet given the problem conditions -/
theorem hilt_fountain_distance :
  total_distance 30 4 = 240 :=
by sorry

end NUMINAMATH_CALUDE_hilt_fountain_distance_l766_76608


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l766_76614

/-- Given a quadratic inequality ax² + bx + 2 > 0 with solution set {x | -1/2 < x < 1/3},
    prove that a + b = -14 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l766_76614


namespace NUMINAMATH_CALUDE_seventh_flip_probability_l766_76658

/-- A fair coin is a coin where the probability of getting heads is 1/2. -/
def fair_coin (p : ℝ → ℝ) : Prop := p 1 = 1/2

/-- A sequence of coin flips is independent if the probability of any outcome
    is not affected by the previous flips. -/
def independent_flips (p : ℕ → ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∀ x : ℝ, p n x = p 0 x

/-- The probability of getting heads on the seventh flip of a fair coin is 1/2,
    regardless of the outcomes of the previous six flips. -/
theorem seventh_flip_probability (p : ℕ → ℝ → ℝ) :
  fair_coin (p 0) →
  independent_flips p →
  p 6 1 = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_flip_probability_l766_76658


namespace NUMINAMATH_CALUDE_equal_weight_partition_l766_76610

theorem equal_weight_partition : ∃ (A B C : Finset Nat), 
  (A ∪ B ∪ C = Finset.range 556 \ {0}) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (A.sum id = B.sum id) ∧ (B.sum id = C.sum id) := by
  sorry

#check equal_weight_partition

end NUMINAMATH_CALUDE_equal_weight_partition_l766_76610


namespace NUMINAMATH_CALUDE_opposite_numbers_l766_76678

theorem opposite_numbers : -(-(3 : ℤ)) = -(-3) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l766_76678


namespace NUMINAMATH_CALUDE_permutations_of_377353752_div_by_5_l766_76630

def original_number : ℕ := 377353752

-- Function to count occurrences of a digit in a number
def count_digit (n : ℕ) (d : ℕ) : ℕ := sorry

-- Function to calculate factorial
def factorial (n : ℕ) : ℕ := sorry

-- Function to calculate permutations of multiset
def permutations_multiset (n : ℕ) (counts : List ℕ) : ℕ := sorry

theorem permutations_of_377353752_div_by_5 :
  let digits := [3, 3, 3, 7, 7, 7, 5, 2]
  let n := digits.length
  let counts := [
    count_digit original_number 3,
    count_digit original_number 7,
    count_digit original_number 5,
    count_digit original_number 2
  ]
  permutations_multiset n counts = 1120 :=
by sorry

end NUMINAMATH_CALUDE_permutations_of_377353752_div_by_5_l766_76630


namespace NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l766_76616

theorem polygon_sides_from_exterior_angle (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → 
  (exterior_angle > 0) → 
  (exterior_angle < 180) → 
  (n * exterior_angle = 360) → 
  (exterior_angle = 30) → 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l766_76616


namespace NUMINAMATH_CALUDE_regular_octagon_perimeter_l766_76611

/-- A regular octagon is a polygon with 8 equal sides -/
def RegularOctagon : Type := Unit

/-- The side length of the regular octagon -/
def side_length : ℝ := 3

/-- The number of sides in an octagon -/
def num_sides : ℕ := 8

/-- The perimeter of a regular octagon is the product of its number of sides and side length -/
def perimeter (o : RegularOctagon) : ℝ := num_sides * side_length

theorem regular_octagon_perimeter : 
  ∀ (o : RegularOctagon), perimeter o = 24 := by sorry

end NUMINAMATH_CALUDE_regular_octagon_perimeter_l766_76611


namespace NUMINAMATH_CALUDE_x_in_M_l766_76670

def M : Set ℝ := {x | x ≤ 7}

theorem x_in_M : 4 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_x_in_M_l766_76670


namespace NUMINAMATH_CALUDE_unique_solution_l766_76672

theorem unique_solution : ∃! x : ℝ, 3 * x + 3 * 12 + 3 * 16 + 11 = 134 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l766_76672


namespace NUMINAMATH_CALUDE_jeremy_jerseys_l766_76656

def jerseyProblem (initialAmount basketballCost shortsCost jerseyCost remainingAmount : ℕ) : Prop :=
  let totalSpent := initialAmount - remainingAmount
  let nonJerseyCost := basketballCost + shortsCost
  let jerseyTotalCost := totalSpent - nonJerseyCost
  jerseyTotalCost / jerseyCost = 5

theorem jeremy_jerseys :
  jerseyProblem 50 18 8 2 14 := by sorry

end NUMINAMATH_CALUDE_jeremy_jerseys_l766_76656


namespace NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l766_76628

/-- The distance from a point on a parabola to its directrix -/
theorem parabola_point_to_directrix_distance 
  (p : ℝ) -- Parameter of the parabola
  (A : ℝ × ℝ) -- Point A
  (h1 : A.1 = 1) -- x-coordinate of A is 1
  (h2 : A.2 = Real.sqrt 5) -- y-coordinate of A is √5
  (h3 : A.2^2 = 2 * p * A.1) -- A lies on the parabola y² = 2px
  : |A.1 - (-p/2)| = 9/4 := by
sorry


end NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l766_76628


namespace NUMINAMATH_CALUDE_ap_has_ten_terms_l766_76675

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  n : ℕ                  -- number of terms
  a : ℝ                  -- first term
  d : ℤ                  -- common difference
  n_even : Even n
  sum_odd : (n / 2) * (a + (a + (n - 2) * d)) = 56
  sum_even : (n / 2) * (a + d + (a + (n - 1) * d)) = 80
  last_minus_first : a + (n - 1) * d - a = 18

/-- The theorem stating that an arithmetic progression with the given properties has 10 terms -/
theorem ap_has_ten_terms (ap : ArithmeticProgression) : ap.n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ap_has_ten_terms_l766_76675


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l766_76646

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2 * Real.sqrt 5 - 1
  (1 / (x^2 + 2*x + 1)) * (1 + 3 / (x - 1)) / ((x + 2) / (x^2 - 1)) = Real.sqrt 5 / 10 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l766_76646


namespace NUMINAMATH_CALUDE_machine_value_is_35000_l766_76623

/-- Represents the denomination of a bill in dollars -/
inductive BillType
  | five
  | ten
  | twenty

/-- Returns the value of a bill in dollars -/
def billValue : BillType → Nat
  | BillType.five => 5
  | BillType.ten => 10
  | BillType.twenty => 20

/-- Represents a bundle of bills -/
structure Bundle where
  billType : BillType
  count : Nat

/-- Represents a cash machine -/
structure CashMachine where
  bundles : List Bundle

/-- The number of bills in each bundle -/
def billsPerBundle : Nat := 100

/-- The number of bundles for each bill type -/
def bundlesPerType : Nat := 10

/-- Calculates the total value of a bundle -/
def bundleValue (b : Bundle) : Nat :=
  billValue b.billType * b.count

/-- Calculates the total value of all bundles in the machine -/
def machineValue (m : CashMachine) : Nat :=
  m.bundles.map bundleValue |>.sum

/-- The cash machine configuration -/
def filledMachine : CashMachine :=
  { bundles := [
    { billType := BillType.five, count := billsPerBundle },
    { billType := BillType.ten, count := billsPerBundle },
    { billType := BillType.twenty, count := billsPerBundle }
  ] }

/-- Theorem: The total amount of money required to fill the machine is $35,000 -/
theorem machine_value_is_35000 : 
  machineValue filledMachine = 35000 := by sorry

end NUMINAMATH_CALUDE_machine_value_is_35000_l766_76623


namespace NUMINAMATH_CALUDE_rockham_soccer_league_members_l766_76606

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 4

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 5

/-- The total cost for all members in dollars -/
def total_cost : ℕ := 2366

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 2

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- Theorem: The number of members in the Rockham Soccer League is 91 -/
theorem rockham_soccer_league_members : 
  (total_cost / (socks_per_member * sock_cost + 
                 tshirts_per_member * (sock_cost + tshirt_additional_cost))) = 91 := by
  sorry

end NUMINAMATH_CALUDE_rockham_soccer_league_members_l766_76606


namespace NUMINAMATH_CALUDE_investment_average_interest_rate_l766_76619

/-- Proves that given a total investment split into two parts with different interest rates
    and equal annual returns, the average interest rate is as calculated. -/
theorem investment_average_interest_rate 
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (h_total : total_investment = 4500)
  (h_rates : rate1 = 0.04 ∧ rate2 = 0.06)
  (h_equal_returns : ∃ (x : ℝ), 
    x > 0 ∧ x < total_investment ∧
    rate1 * (total_investment - x) = rate2 * x) :
  (rate1 * (total_investment - x) + rate2 * x) / total_investment = 0.048 := by
  sorry

#check investment_average_interest_rate

end NUMINAMATH_CALUDE_investment_average_interest_rate_l766_76619


namespace NUMINAMATH_CALUDE_common_factor_extraction_l766_76691

def polynomial (a b c : ℤ) : ℤ := 8 * a^3 * b^2 + 12 * a^3 * b * c - 4 * a^2 * b

theorem common_factor_extraction (a b c : ℤ) :
  ∃ (k : ℤ), polynomial a b c = (4 * a^2 * b) * k ∧ 
  (∀ (d : ℤ), (∃ (l : ℤ), polynomial a b c = d * l) → d ≤ 4 * a^2 * b) :=
sorry

end NUMINAMATH_CALUDE_common_factor_extraction_l766_76691


namespace NUMINAMATH_CALUDE_alan_collected_48_shells_l766_76693

/-- Given the number of shells collected by Laurie, calculate the number of shells collected by Alan. -/
def alan_shells (laurie_shells : ℕ) : ℕ :=
  let ben_shells := laurie_shells / 3
  4 * ben_shells

/-- Theorem stating that if Laurie collected 36 shells, Alan collected 48 shells. -/
theorem alan_collected_48_shells :
  alan_shells 36 = 48 := by
  sorry

end NUMINAMATH_CALUDE_alan_collected_48_shells_l766_76693


namespace NUMINAMATH_CALUDE_hot_dog_problem_l766_76653

theorem hot_dog_problem :
  let hot_dogs := 12
  let hot_dog_buns := 9
  let mustard := 18
  let ketchup := 24
  Nat.lcm (Nat.lcm (Nat.lcm hot_dogs hot_dog_buns) mustard) ketchup = 72 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_problem_l766_76653


namespace NUMINAMATH_CALUDE_distinct_digit_count_is_4032_l766_76642

/-- The number of integers between 2000 and 9999 with four distinct digits -/
def distinct_digit_count : ℕ := sorry

/-- The range of possible first digits -/
def first_digit_range : List ℕ := List.range 8

/-- The range of possible second digits (including 0) -/
def second_digit_range : List ℕ := List.range 10

/-- The range of possible third digits -/
def third_digit_range : List ℕ := List.range 10

/-- The range of possible fourth digits -/
def fourth_digit_range : List ℕ := List.range 10

theorem distinct_digit_count_is_4032 :
  distinct_digit_count = first_digit_range.length *
                         (second_digit_range.length - 1) *
                         (third_digit_range.length - 2) *
                         (fourth_digit_range.length - 3) :=
by sorry

end NUMINAMATH_CALUDE_distinct_digit_count_is_4032_l766_76642


namespace NUMINAMATH_CALUDE_james_age_l766_76650

theorem james_age (dan_age james_age : ℕ) : 
  (dan_age : ℚ) / james_age = 6 / 5 →
  dan_age + 4 = 28 →
  james_age = 20 := by
sorry

end NUMINAMATH_CALUDE_james_age_l766_76650


namespace NUMINAMATH_CALUDE_no_wobbly_multiple_iff_div_10_or_25_l766_76690

/-- A wobbly number is a positive integer whose digits in base 10 are alternatively non-zero and zero, with the units digit being non-zero. -/
def IsWobbly (n : ℕ) : Prop := sorry

/-- Theorem: A positive integer n does not divide any wobbly number if and only if n is divisible by 10 or 25. -/
theorem no_wobbly_multiple_iff_div_10_or_25 (n : ℕ) (hn : n > 0) :
  (∀ w : ℕ, IsWobbly w → ¬(w % n = 0)) ↔ (n % 10 = 0 ∨ n % 25 = 0) := by sorry

end NUMINAMATH_CALUDE_no_wobbly_multiple_iff_div_10_or_25_l766_76690


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_six_l766_76613

theorem infinite_solutions_iff_b_eq_neg_six :
  ∀ b : ℝ, (∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 10)) ↔ b = -6 := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_six_l766_76613


namespace NUMINAMATH_CALUDE_count_valid_numbers_l766_76644

/-- A function that generates all valid numbers from digits 1, 2, and 3 without repetition -/
def validNumbers : List ℕ :=
  [1, 2, 3, 12, 13, 21, 23, 31, 32, 123, 132, 213, 231, 312, 321]

/-- The count of natural numbers composed of digits 1, 2, and 3 without repetition -/
theorem count_valid_numbers : validNumbers.length = 15 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l766_76644


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l766_76694

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l766_76694


namespace NUMINAMATH_CALUDE_min_value_theorem_l766_76603

/-- The function f(x) = x|x - a| has a minimum value of 2 on the interval [1, 2] when a = 3 -/
theorem min_value_theorem (a : ℝ) (h1 : a > 0) :
  (∀ x ∈ Set.Icc 1 2, x * |x - a| ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 1 2, x * |x - a| = 2) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l766_76603


namespace NUMINAMATH_CALUDE_jori_water_remaining_l766_76655

/-- The amount of water remaining after usage -/
def water_remaining (initial : ℚ) (usage1 : ℚ) (usage2 : ℚ) : ℚ :=
  initial - usage1 - usage2

/-- Theorem stating the remaining water after Jori's usage -/
theorem jori_water_remaining :
  water_remaining 3 (5/4) (1/2) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_jori_water_remaining_l766_76655


namespace NUMINAMATH_CALUDE_syrup_box_cost_l766_76668

/-- Represents the cost of syrup boxes for a convenience store -/
def SyrupCost (total_soda : ℕ) (soda_per_box : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / (total_soda / soda_per_box)

/-- Theorem: The cost per box of syrup is $40 -/
theorem syrup_box_cost :
  SyrupCost 180 30 240 = 40 := by
  sorry

end NUMINAMATH_CALUDE_syrup_box_cost_l766_76668


namespace NUMINAMATH_CALUDE_equivalent_angle_exists_l766_76680

-- Define the angle in degrees
def angle : ℝ := -463

-- Theorem stating that there exists an equivalent angle in the form k·360° + 257°
theorem equivalent_angle_exists :
  ∃ (k : ℤ), (k : ℝ) * 360 + 257 = angle + 360 * ⌊angle / 360⌋ := by
  sorry

end NUMINAMATH_CALUDE_equivalent_angle_exists_l766_76680


namespace NUMINAMATH_CALUDE_mod_37_5_l766_76666

theorem mod_37_5 : 37 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_37_5_l766_76666


namespace NUMINAMATH_CALUDE_mr_callen_wooden_toys_solution_is_eight_l766_76620

/-- Proves that the number of wooden toys bought is 8, given the conditions of Mr. Callen's purchase and sale. -/
theorem mr_callen_wooden_toys : ℕ :=
  let num_paintings : ℕ := 10
  let painting_cost : ℚ := 40
  let toy_cost : ℚ := 20
  let painting_discount : ℚ := 0.1
  let toy_discount : ℚ := 0.15
  let total_loss : ℚ := 64

  let painting_revenue := num_paintings * (painting_cost * (1 - painting_discount))
  let toy_revenue (num_toys : ℕ) := num_toys * (toy_cost * (1 - toy_discount))
  let total_cost (num_toys : ℕ) := num_paintings * painting_cost + num_toys * toy_cost
  let total_revenue (num_toys : ℕ) := painting_revenue + toy_revenue num_toys

  have h : ∃ (num_toys : ℕ), total_cost num_toys - total_revenue num_toys = total_loss :=
    sorry

  Classical.choose h

/-- The solution to the problem is 8 wooden toys. -/
theorem solution_is_eight : mr_callen_wooden_toys = 8 := by
  sorry

end NUMINAMATH_CALUDE_mr_callen_wooden_toys_solution_is_eight_l766_76620


namespace NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l766_76651

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem specific_fraction_division :
  (3 : ℚ) / 7 / ((2 : ℚ) / 5) = 15 / 14 := by sorry

end NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l766_76651


namespace NUMINAMATH_CALUDE_bus_departure_interval_l766_76660

/-- Represents the scenario of Xiao Wang and the No. 18 buses -/
structure BusScenario where
  /-- Speed of Xiao Wang in meters per minute -/
  wang_speed : ℝ
  /-- Speed of the No. 18 buses in meters per minute -/
  bus_speed : ℝ
  /-- Distance between two adjacent buses traveling in the same direction in meters -/
  bus_distance : ℝ
  /-- Xiao Wang walks at a constant speed -/
  wang_constant_speed : wang_speed > 0
  /-- Buses travel at a constant speed -/
  bus_constant_speed : bus_speed > 0
  /-- A bus passes Xiao Wang from behind every 6 minutes -/
  overtake_condition : 6 * bus_speed - 6 * wang_speed = bus_distance
  /-- A bus comes towards Xiao Wang every 3 minutes -/
  approach_condition : 3 * bus_speed + 3 * wang_speed = bus_distance

/-- The interval between bus departures is 4 minutes -/
theorem bus_departure_interval (scenario : BusScenario) : 
  scenario.bus_distance = 4 * scenario.bus_speed := by
  sorry

#check bus_departure_interval

end NUMINAMATH_CALUDE_bus_departure_interval_l766_76660


namespace NUMINAMATH_CALUDE_sandwich_bread_count_l766_76624

/-- The number of pieces of bread needed for one double meat sandwich -/
def double_meat_bread : ℕ := 3

/-- The number of regular sandwiches -/
def regular_sandwiches : ℕ := 14

/-- The number of double meat sandwiches -/
def double_meat_sandwiches : ℕ := 12

/-- The number of pieces of bread needed for one regular sandwich -/
def regular_bread : ℕ := 2

/-- The total number of pieces of bread used -/
def total_bread : ℕ := 64

theorem sandwich_bread_count : 
  regular_sandwiches * regular_bread + double_meat_sandwiches * double_meat_bread = total_bread := by
  sorry

end NUMINAMATH_CALUDE_sandwich_bread_count_l766_76624


namespace NUMINAMATH_CALUDE_nickel_chocolates_l766_76665

/-- Given that Robert ate 7 chocolates and 4 more than Nickel, prove that Nickel ate 3 chocolates. -/
theorem nickel_chocolates (robert : ℕ) (nickel : ℕ) 
  (h1 : robert = 7)
  (h2 : robert = nickel + 4) : 
  nickel = 3 := by
  sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l766_76665


namespace NUMINAMATH_CALUDE_cycle_original_price_l766_76661

/-- Proves that the original price of a cycle is 1600 when sold at a 10% loss for 1440 --/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1440)
  (h2 : loss_percentage = 10) : 
  selling_price / (1 - loss_percentage / 100) = 1600 := by
sorry

end NUMINAMATH_CALUDE_cycle_original_price_l766_76661


namespace NUMINAMATH_CALUDE_book_price_change_l766_76641

theorem book_price_change (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 + 20 / 100) = P * (1 + 16 / 100) → 
  x = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_book_price_change_l766_76641


namespace NUMINAMATH_CALUDE_F_opposite_A_l766_76662

/-- Represents a face of a cube --/
inductive Face : Type
| A | B | C | D | E | F

/-- Represents a cube net that can be folded into a cube --/
structure CubeNet where
  faces : List Face
  can_fold : Bool

/-- Represents a folded cube --/
structure Cube where
  net : CubeNet
  bottom : Face

/-- Defines the opposite face relation in a cube --/
def opposite_face (c : Cube) (f1 f2 : Face) : Prop :=
  f1 ≠ f2 ∧ ∀ (f : Face), f ≠ f1 → f ≠ f2 → (f ∈ c.net.faces)

/-- Theorem: In a cube formed from a net where face F is the bottom, face F is opposite to face A --/
theorem F_opposite_A (c : Cube) (h : c.bottom = Face.F) : opposite_face c Face.A Face.F :=
sorry

end NUMINAMATH_CALUDE_F_opposite_A_l766_76662


namespace NUMINAMATH_CALUDE_fred_marbles_count_l766_76604

/-- Represents the number of marbles Fred has of each color -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  dark_blue : ℕ

/-- Calculates the total number of marbles -/
def total_marbles (m : MarbleCount) : ℕ :=
  m.red + m.green + m.dark_blue

/-- Theorem stating the total number of marbles Fred has -/
theorem fred_marbles_count :
  ∃ (m : MarbleCount),
    m.red = 38 ∧
    m.green = m.red / 2 ∧
    m.dark_blue = 6 ∧
    total_marbles m = 63 := by
  sorry

end NUMINAMATH_CALUDE_fred_marbles_count_l766_76604


namespace NUMINAMATH_CALUDE_cube_root_of_a_plus_b_l766_76621

theorem cube_root_of_a_plus_b (a b : ℝ) : 
  a > 0 → (2*b - 1)^2 = a → (b + 4)^2 = a → (a + b)^(1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_plus_b_l766_76621


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l766_76639

theorem initial_markup_percentage (C : ℝ) (M : ℝ) : 
  (C * (1 + M) * 1.25 * 0.92 = C * 1.38) → M = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l766_76639


namespace NUMINAMATH_CALUDE_jakes_weight_ratio_l766_76607

/-- Proves that the ratio of Jake's weight after losing 20 pounds to his sister's weight is 2:1 -/
theorem jakes_weight_ratio (jake_weight sister_weight : ℕ) : 
  jake_weight = 156 →
  jake_weight + sister_weight = 224 →
  (jake_weight - 20) / sister_weight = 2 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_ratio_l766_76607


namespace NUMINAMATH_CALUDE_product_of_red_is_red_l766_76635

-- Define the color type
inductive Color : Type
  | Red : Color
  | Blue : Color

-- Define the coloring function
def coloring : ℕ+ → Color := sorry

-- Define the conditions
axiom all_colored : ∀ n : ℕ+, (coloring n = Color.Red) ∨ (coloring n = Color.Blue)
axiom sum_different_colors : ∀ m n : ℕ+, coloring m ≠ coloring n → coloring (m + n) = Color.Blue
axiom product_different_colors : ∀ m n : ℕ+, coloring m ≠ coloring n → coloring (m * n) = Color.Red

-- State the theorem
theorem product_of_red_is_red :
  ∀ m n : ℕ+, coloring m = Color.Red → coloring n = Color.Red → coloring (m * n) = Color.Red :=
sorry

end NUMINAMATH_CALUDE_product_of_red_is_red_l766_76635


namespace NUMINAMATH_CALUDE_sum_of_roots_l766_76626

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1716 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l766_76626


namespace NUMINAMATH_CALUDE_alpha_value_l766_76609

theorem alpha_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) 
  (h_min : ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → 1/m + 16/n ≤ 1/x + 16/y) 
  (h_curve : (m/5)^α = m/4) : α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l766_76609


namespace NUMINAMATH_CALUDE_picking_black_is_random_event_l766_76638

/-- Represents a ball in the box -/
inductive Ball
| White
| Black

/-- Represents the box containing the balls -/
structure Box where
  white_balls : ℕ
  black_balls : ℕ

/-- Defines what a random event is -/
def is_random_event (box : Box) (pick : Ball → Prop) : Prop :=
  (∃ b : Ball, pick b) ∧ 
  (∃ b : Ball, ¬ pick b) ∧ 
  (box.white_balls + box.black_balls > 0)

/-- The main theorem to prove -/
theorem picking_black_is_random_event (box : Box) 
  (h1 : box.white_balls = 1) 
  (h2 : box.black_balls = 200) : 
  is_random_event box (λ b => b = Ball.Black) := by
  sorry


end NUMINAMATH_CALUDE_picking_black_is_random_event_l766_76638


namespace NUMINAMATH_CALUDE_problem_odometer_miles_l766_76640

/-- Represents a faulty odometer that skips certain digits --/
structure FaultyOdometer where
  skipped_digits : List Nat
  display : Nat

/-- Converts a faulty odometer reading to actual miles traveled --/
def actualMiles (o : FaultyOdometer) : Nat :=
  sorry

/-- The specific faulty odometer in the problem --/
def problemOdometer : FaultyOdometer :=
  { skipped_digits := [4, 7], display := 5006 }

/-- Theorem stating that the problemOdometer has traveled 1721 miles --/
theorem problem_odometer_miles :
  actualMiles problemOdometer = 1721 := by
  sorry

end NUMINAMATH_CALUDE_problem_odometer_miles_l766_76640


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l766_76636

/-- Represents the price and composition of orangeade on two consecutive days -/
structure Orangeade where
  orange_juice : ℝ
  water_day1 : ℝ
  water_day2 : ℝ
  price_day1 : ℝ
  price_day2 : ℝ

/-- Theorem stating the conditions and the result to be proven -/
theorem orangeade_price_day2 (o : Orangeade) 
  (h1 : o.orange_juice = o.water_day1)
  (h2 : o.water_day2 = 2 * o.water_day1)
  (h3 : o.price_day1 = 0.3)
  (h4 : (o.orange_juice + o.water_day1) * o.price_day1 = 
        (o.orange_juice + o.water_day2) * o.price_day2) :
  o.price_day2 = 0.2 := by
  sorry

#check orangeade_price_day2

end NUMINAMATH_CALUDE_orangeade_price_day2_l766_76636


namespace NUMINAMATH_CALUDE_polynomial_value_l766_76618

def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

theorem polynomial_value : f 4 = 1559 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l766_76618


namespace NUMINAMATH_CALUDE_g_of_4_l766_76667

/-- Given a function g: ℝ → ℝ satisfying g(x) + 3*g(2 - x) = 2*x^2 + x - 1 for all real x,
    prove that g(4) = -5/2 -/
theorem g_of_4 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 2 * x^2 + x - 1) : 
  g 4 = -5/2 := by
sorry

end NUMINAMATH_CALUDE_g_of_4_l766_76667


namespace NUMINAMATH_CALUDE_loop_iterations_count_l766_76695

theorem loop_iterations_count (i : ℕ) : 
  i = 1 → (∀ j, j ≥ 1 ∧ j < 21 → i + j = j + 1) → i + 20 = 21 :=
by sorry

end NUMINAMATH_CALUDE_loop_iterations_count_l766_76695


namespace NUMINAMATH_CALUDE_range_of_m_l766_76686

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 5*x - 6 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 6*x + 9 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, p x ∧ ¬(q x m)) →
  m ∈ Set.Ioo 0 3 := by
sorry

-- Note: Set.Ioo 0 3 represents the open interval (0, 3)

end NUMINAMATH_CALUDE_range_of_m_l766_76686


namespace NUMINAMATH_CALUDE_calculate_expression_largest_integer_solution_three_is_largest_integer_solution_l766_76615

-- Part 1
theorem calculate_expression : 4 * Real.sin (π / 3) - |-1| + (Real.sqrt 3 - 1)^0 + Real.sqrt 48 = 6 * Real.sqrt 3 := by
  sorry

-- Part 2
theorem largest_integer_solution (x : ℝ) :
  (1/2 * (x - 1) ≤ 1 ∧ 1 - x < 2) → x ≤ 3 := by
  sorry

theorem three_is_largest_integer_solution :
  ∃ (x : ℤ), x = 3 ∧ (1/2 * (x - 1) ≤ 1 ∧ 1 - x < 2) ∧
  ∀ (y : ℤ), y > 3 → ¬(1/2 * (y - 1) ≤ 1 ∧ 1 - y < 2) := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_largest_integer_solution_three_is_largest_integer_solution_l766_76615


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l766_76602

/-- Theorem: Relationship between heights of two cylinders with equal volumes and different radii -/
theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l766_76602


namespace NUMINAMATH_CALUDE_car_distance_proof_l766_76679

theorem car_distance_proof (speed1 speed2 speed3 : ℝ) 
  (h1 : speed1 = 180)
  (h2 : speed2 = 160)
  (h3 : speed3 = 220) :
  speed1 + speed2 + speed3 = 560 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l766_76679
