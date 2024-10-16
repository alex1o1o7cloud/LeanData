import Mathlib

namespace NUMINAMATH_CALUDE_marcus_pies_l3549_354977

def pies_left (batch_size : ℕ) (num_batches : ℕ) (dropped_pies : ℕ) : ℕ :=
  batch_size * num_batches - dropped_pies

theorem marcus_pies :
  pies_left 5 7 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pies_l3549_354977


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l3549_354984

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) = 485 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l3549_354984


namespace NUMINAMATH_CALUDE_min_area_AOB_l3549_354901

noncomputable section

-- Define the hyperbola C₁
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (2 * a^2) = 1 ∧ a > 0

-- Define the parabola C₂
def C₂ (a : ℝ) (x y : ℝ) : Prop := y^2 = -4 * Real.sqrt 3 * a * x

-- Define the focus F₁
def F₁ (a : ℝ) : ℝ × ℝ := (-Real.sqrt 3 * a, 0)

-- Define a chord AB of C₂ passing through F₁
def chord_AB (a k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + Real.sqrt 3 * a) ∧ C₂ a x y

-- Define the area of triangle AOB
def area_AOB (a k : ℝ) : ℝ := 6 * a^2 * Real.sqrt (1 + 1 / k^2)

-- Main theorem
theorem min_area_AOB (a : ℝ) :
  (∃ k : ℝ, ∀ k' : ℝ, area_AOB a k ≤ area_AOB a k') ∧
  (∃ x : ℝ, x = -Real.sqrt 3 * a ∧ 
    ∀ k : ℝ, area_AOB a k ≥ 6 * a^2) :=
sorry

end

end NUMINAMATH_CALUDE_min_area_AOB_l3549_354901


namespace NUMINAMATH_CALUDE_cube_face_sum_l3549_354964

/-- Represents the six positive integers on the faces of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- The sum of the products of the three numbers adjacent to each vertex -/
def vertexSum (faces : CubeFaces) : ℕ :=
  faces.a * faces.b * faces.c +
  faces.a * faces.e * faces.c +
  faces.a * faces.b * faces.f +
  faces.a * faces.e * faces.f +
  faces.d * faces.b * faces.c +
  faces.d * faces.e * faces.c +
  faces.d * faces.b * faces.f +
  faces.d * faces.e * faces.f

/-- The sum of the numbers on the faces -/
def faceSum (faces : CubeFaces) : ℕ :=
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f

theorem cube_face_sum (faces : CubeFaces) :
  vertexSum faces = 1386 → faceSum faces = 38 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_sum_l3549_354964


namespace NUMINAMATH_CALUDE_sum_inequality_l3549_354990

theorem sum_inequality {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : c + a > d + b := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3549_354990


namespace NUMINAMATH_CALUDE_last_two_digits_product_l3549_354931

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 0) →
  (n % 8 = 0) → 
  ((n % 100) / 10 + n % 10 = 12) → 
  ((n % 100) / 10) * (n % 10) = 32 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l3549_354931


namespace NUMINAMATH_CALUDE_complex_cube_theorem_l3549_354979

theorem complex_cube_theorem (z : ℂ) (h : z = 1 - I) :
  ((1 + I) / z) ^ 3 = -I := by sorry

end NUMINAMATH_CALUDE_complex_cube_theorem_l3549_354979


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l3549_354994

/-- Given a triangle ABC with ∠A = 60°, ∠C = 45°, and side b = 4,
    prove that the smallest side of the triangle is 4√3 - 4. -/
theorem smallest_side_of_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = 60 * π / 180 →
  C = 45 * π / 180 →
  b = 4 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  A + B + C = π →
  c / Real.sin C = b / Real.sin B →
  c = 4 * Real.sqrt 3 - 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l3549_354994


namespace NUMINAMATH_CALUDE_rowing_conference_votes_l3549_354903

theorem rowing_conference_votes 
  (num_coaches : ℕ) 
  (num_rowers : ℕ) 
  (votes_per_coach : ℕ) 
  (h1 : num_coaches = 36) 
  (h2 : num_rowers = 60) 
  (h3 : votes_per_coach = 5) : 
  (num_coaches * votes_per_coach) / num_rowers = 3 :=
by sorry

end NUMINAMATH_CALUDE_rowing_conference_votes_l3549_354903


namespace NUMINAMATH_CALUDE_intersection_line_slope_l3549_354900

/-- Given two circles in the xy-plane, this theorem proves that the slope of the line 
    passing through their intersection points is -2/3. -/
theorem intersection_line_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 12 = 0) →
  (x^2 + y^2 - 10*x - 2*y + 22 = 0) →
  ∃ (m : ℝ), m = -2/3 ∧ 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 12 = 0) →
    (x₁^2 + y₁^2 - 10*x₁ - 2*y₁ + 22 = 0) →
    (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 12 = 0) →
    (x₂^2 + y₂^2 - 10*x₂ - 2*y₂ + 22 = 0) →
    x₁ ≠ x₂ →
    (y₂ - y₁) / (x₂ - x₁) = m :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l3549_354900


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l3549_354945

theorem trigonometric_expression_value (α : ℝ) 
  (h : Real.sin (3 * Real.pi - α) = 2 * Real.sin (Real.pi / 2 + α)) :
  (Real.sin (Real.pi - α) ^ 3 - Real.sin (Real.pi / 2 - α)) / 
  (3 * Real.cos (Real.pi / 2 + α) + 2 * Real.cos (Real.pi + α)) = -3 / 40 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l3549_354945


namespace NUMINAMATH_CALUDE_cube_root_of_a_plus_b_l3549_354925

theorem cube_root_of_a_plus_b (a b : ℝ) : 
  a > 0 → (2*b - 1)^2 = a → (b + 4)^2 = a → (a + b)^(1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_plus_b_l3549_354925


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3549_354954

theorem arithmetic_sequence_inequality (a b c : ℝ) (h1 : b - a = c - b) (h2 : b - a ≠ 0) :
  ¬ (∀ a b c : ℝ, a^3*b + b^3*c + c^3*a ≥ a^4 + b^4 + c^4) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l3549_354954


namespace NUMINAMATH_CALUDE_total_class_time_l3549_354987

def hours_per_day : ℝ := 8
def days_per_week : ℝ := 5
def math_percentage : ℝ := 0.25
def language_percentage : ℝ := 0.30
def science_percentage : ℝ := 0.20
def history_percentage : ℝ := 0.10

theorem total_class_time :
  let total_hours := hours_per_day * days_per_week
  let math_hours := total_hours * math_percentage
  let language_hours := total_hours * language_percentage
  let science_hours := total_hours * science_percentage
  let history_hours := total_hours * history_percentage
  math_hours + language_hours + science_hours + history_hours = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_class_time_l3549_354987


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l3549_354996

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetric_x (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

theorem symmetry_coordinates :
  let A : Point := ⟨1, 2⟩
  let A' : Point := ⟨1, -2⟩
  symmetric_x A A' :=
by sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l3549_354996


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_odd_integers_with_product_110895_l3549_354981

-- Define a function that generates five consecutive odd integers
def fiveConsecutiveOddIntegers (x : ℤ) : List ℤ :=
  [x - 4, x - 2, x, x + 2, x + 4]

-- Theorem statement
theorem largest_of_five_consecutive_odd_integers_with_product_110895 :
  ∃ x : ℤ, 
    (fiveConsecutiveOddIntegers x).prod = 110895 ∧
    (fiveConsecutiveOddIntegers x).all (λ i => i % 2 ≠ 0) ∧
    (fiveConsecutiveOddIntegers x).maximum? = some 17 :=
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_odd_integers_with_product_110895_l3549_354981


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l3549_354929

/-- Given a man's rowing speed against the stream and in still water, 
    calculate his speed with the stream. -/
theorem mans_speed_with_stream 
  (speed_against_stream : ℝ) 
  (speed_still_water : ℝ) 
  (h1 : speed_against_stream = 4) 
  (h2 : speed_still_water = 11) : 
  speed_still_water + (speed_still_water - speed_against_stream) = 18 := by
  sorry

#check mans_speed_with_stream

end NUMINAMATH_CALUDE_mans_speed_with_stream_l3549_354929


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3549_354939

theorem inequality_solution_set (x : ℝ) : 
  (3/16 : ℝ) + |x - 5/32| < 7/32 ↔ x ∈ Set.Ioo (1/8 : ℝ) (3/16 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3549_354939


namespace NUMINAMATH_CALUDE_card_distribution_events_l3549_354960

structure Card where
  color : String
  deriving Repr

structure Person where
  name : String
  deriving Repr

def distribute_cards (cards : List Card) (people : List Person) : List (Person × Card) :=
  sorry

def event_A_red (distribution : List (Person × Card)) : Prop :=
  sorry

def event_B_red (distribution : List (Person × Card)) : Prop :=
  sorry

def mutually_exclusive (event1 event2 : List (Person × Card) → Prop) : Prop :=
  sorry

def opposite_events (event1 event2 : List (Person × Card) → Prop) : Prop :=
  sorry

theorem card_distribution_events :
  let cards := [Card.mk "red", Card.mk "black", Card.mk "blue", Card.mk "white"]
  let people := [Person.mk "A", Person.mk "B", Person.mk "C", Person.mk "D"]
  let distributions := distribute_cards cards people
  mutually_exclusive event_A_red event_B_red ∧
  ¬(opposite_events event_A_red event_B_red) :=
by
  sorry

end NUMINAMATH_CALUDE_card_distribution_events_l3549_354960


namespace NUMINAMATH_CALUDE_games_to_sell_l3549_354933

def playstation_cost : ℝ := 500
def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def game_price : ℝ := 7.5

theorem games_to_sell : 
  ⌈(playstation_cost - (birthday_money + christmas_money)) / game_price⌉ = 20 := by sorry

end NUMINAMATH_CALUDE_games_to_sell_l3549_354933


namespace NUMINAMATH_CALUDE_hilt_fountain_distance_l3549_354921

/-- The total distance Mrs. Hilt walks to the water fountain -/
def total_distance (desk_to_fountain : ℕ) (num_trips : ℕ) : ℕ :=
  2 * desk_to_fountain * num_trips

/-- Theorem: Mrs. Hilt walks 240 feet given the problem conditions -/
theorem hilt_fountain_distance :
  total_distance 30 4 = 240 :=
by sorry

end NUMINAMATH_CALUDE_hilt_fountain_distance_l3549_354921


namespace NUMINAMATH_CALUDE_circle_square_area_l3549_354971

theorem circle_square_area (r : ℝ) (s : ℝ) (hr : r = 1) (hs : s = 2) :
  let circle_area := π * r^2
  let square_area := s^2
  let square_diagonal := s * Real.sqrt 2
  circle_area - square_area = 0 := by sorry

end NUMINAMATH_CALUDE_circle_square_area_l3549_354971


namespace NUMINAMATH_CALUDE_binomial_18_4_l3549_354978

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_4_l3549_354978


namespace NUMINAMATH_CALUDE_trig_identity_l3549_354989

theorem trig_identity (α : Real) :
  (∃ P : Real × Real, P.1 = Real.sin 2 ∧ P.2 = Real.cos 2 ∧ 
    P.1^2 + P.2^2 = 1 ∧ Real.sin α = P.2) →
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3549_354989


namespace NUMINAMATH_CALUDE_g_4_equals_10_l3549_354915

/-- A function g satisfying xg(y) = yg(x) for all real x and y, and g(12) = 30 -/
def g : ℝ → ℝ :=
  sorry

/-- The property that xg(y) = yg(x) for all real x and y -/
axiom g_property : ∀ x y : ℝ, x * g y = y * g x

/-- The given condition that g(12) = 30 -/
axiom g_12 : g 12 = 30

/-- Theorem stating that g(4) = 10 -/
theorem g_4_equals_10 : g 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_g_4_equals_10_l3549_354915


namespace NUMINAMATH_CALUDE_ap_has_ten_terms_l3549_354914

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

end NUMINAMATH_CALUDE_ap_has_ten_terms_l3549_354914


namespace NUMINAMATH_CALUDE_prob_both_blue_l3549_354942

/-- Represents a jar containing buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- The probability of selecting a blue button from a jar -/
def prob_blue (j : Jar) : ℚ :=
  j.blue / (j.red + j.blue)

/-- The initial state of Jar C -/
def initial_jar_c : Jar :=
  ⟨6, 12⟩

/-- The number of buttons moved from Jar C to Jar D -/
def buttons_moved : ℕ := 6

/-- The final state of Jar C after moving buttons -/
def final_jar_c : Jar :=
  ⟨initial_jar_c.red - buttons_moved / 2, initial_jar_c.blue - buttons_moved / 2⟩

/-- The state of Jar D after receiving buttons -/
def jar_d : Jar :=
  ⟨buttons_moved / 2, buttons_moved / 2⟩

theorem prob_both_blue :
  prob_blue final_jar_c * prob_blue jar_d = 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_prob_both_blue_l3549_354942


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3549_354946

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- This value doesn't matter as g is not defined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3549_354946


namespace NUMINAMATH_CALUDE_share_difference_l3549_354956

/-- Represents the distribution of money among three people -/
structure MoneyDistribution where
  total : ℕ
  ratio_faruk : ℕ
  ratio_vasim : ℕ
  ratio_ranjith : ℕ

/-- Calculates the share of a person given their ratio and the total amount -/
def calculate_share (dist : MoneyDistribution) (ratio : ℕ) : ℕ :=
  dist.total * ratio / (dist.ratio_faruk + dist.ratio_vasim + dist.ratio_ranjith)

theorem share_difference (dist : MoneyDistribution) 
  (h1 : dist.ratio_faruk = 3)
  (h2 : dist.ratio_vasim = 5)
  (h3 : dist.ratio_ranjith = 9)
  (h4 : calculate_share dist dist.ratio_vasim = 1500) :
  calculate_share dist dist.ratio_ranjith - calculate_share dist dist.ratio_faruk = 1800 :=
by sorry

end NUMINAMATH_CALUDE_share_difference_l3549_354956


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3549_354907

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt (3 * x - 4)) = 4 → x = 173 / 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3549_354907


namespace NUMINAMATH_CALUDE_money_percentage_difference_l3549_354982

/-- The problem statement about Kim, Sal, and Phil's money --/
theorem money_percentage_difference 
  (sal_phil_total : ℝ)
  (kim_money : ℝ)
  (sal_percent_less : ℝ)
  (h1 : sal_phil_total = 1.80)
  (h2 : kim_money = 1.12)
  (h3 : sal_percent_less = 20) :
  let phil_money := sal_phil_total / (2 - sal_percent_less / 100)
  let sal_money := phil_money * (1 - sal_percent_less / 100)
  let percentage_difference := (kim_money - sal_money) / sal_money * 100
  percentage_difference = 40 := by
sorry

end NUMINAMATH_CALUDE_money_percentage_difference_l3549_354982


namespace NUMINAMATH_CALUDE_road_network_impossibility_l3549_354922

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

end NUMINAMATH_CALUDE_road_network_impossibility_l3549_354922


namespace NUMINAMATH_CALUDE_opposite_numbers_l3549_354928

theorem opposite_numbers : -(-(3 : ℤ)) = -(-3) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l3549_354928


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l3549_354948

theorem rectangle_ratio_theorem (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∃ (k l m n : ℕ), k * a + l * b = a * Real.sqrt 30 ∧
                     m * a + n * b = b * Real.sqrt 30 ∧
                     k * n = l * m ∧ l * m = 30) →
  (a / b = Real.sqrt 30 ∨
   a / b = Real.sqrt 30 / 2 ∨
   a / b = Real.sqrt 30 / 3 ∨
   a / b = Real.sqrt 30 / 5 ∨
   a / b = Real.sqrt 30 / 6 ∨
   a / b = Real.sqrt 30 / 10 ∨
   a / b = Real.sqrt 30 / 15 ∨
   a / b = Real.sqrt 30 / 30) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l3549_354948


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l3549_354958

theorem triangle_side_and_area 
  (A B C : Real) -- Angles
  (a b c : Real) -- Sides
  (h1 : b = Real.sqrt 7)
  (h2 : c = 1)
  (h3 : B = 2 * π / 3) -- 120° in radians
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c) -- Triangle inequality
  (h5 : b^2 = a^2 + c^2 - 2*a*c*Real.cos B) -- Cosine rule
  : a = 2 ∧ (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l3549_354958


namespace NUMINAMATH_CALUDE_vertical_bisecting_line_of_circles_l3549_354973

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y - 4 = 0

-- Define the vertical bisecting line
def bisecting_line (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- Theorem statement
theorem vertical_bisecting_line_of_circles :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → bisecting_line x y :=
by sorry

end NUMINAMATH_CALUDE_vertical_bisecting_line_of_circles_l3549_354973


namespace NUMINAMATH_CALUDE_f_negative_a_eq_zero_l3549_354934

/-- The function f(x) = x^3 + x + 1 -/
def f (x : ℝ) : ℝ := x^3 + x + 1

/-- Theorem: If f(a) = 2, then f(-a) = 0 -/
theorem f_negative_a_eq_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_eq_zero_l3549_354934


namespace NUMINAMATH_CALUDE_store_breaks_even_l3549_354965

/-- Represents the financial outcome of selling two items -/
def break_even (selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : Prop :=
  let cost_price_1 := selling_price / (1 + profit_percent / 100)
  let cost_price_2 := selling_price / (1 - loss_percent / 100)
  cost_price_1 + cost_price_2 = selling_price * 2

/-- Theorem: A store breaks even when selling two items at $150 each, 
    with one making 50% profit and the other incurring 25% loss -/
theorem store_breaks_even : break_even 150 50 25 := by
  sorry

end NUMINAMATH_CALUDE_store_breaks_even_l3549_354965


namespace NUMINAMATH_CALUDE_local_min_implies_a_eq_neg_three_l3549_354935

/-- The function f(x) defined as x(x-a)² --/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - a)^2

/-- The first derivative of f(x) --/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + a^2

/-- The second derivative of f(x) --/
def f'' (a : ℝ) (x : ℝ) : ℝ := 6*x - 4*a

/-- Theorem: If x = -1 is a point of local minimum for f(x) = x(x-a)², then a = -3 --/
theorem local_min_implies_a_eq_neg_three (a : ℝ) :
  (f' a (-1) = 0 ∧ f'' a (-1) > 0) → a = -3 := by
  sorry

#check local_min_implies_a_eq_neg_three

end NUMINAMATH_CALUDE_local_min_implies_a_eq_neg_three_l3549_354935


namespace NUMINAMATH_CALUDE_wharf_length_l3549_354932

/-- The length of the wharf in meters -/
def L_wharf : ℝ := 64

/-- The average speed in meters per second -/
def V_avg : ℝ := 2

/-- The travel time in seconds -/
def T_travel : ℝ := 16

/-- Theorem: The length of the wharf is 64 meters -/
theorem wharf_length : L_wharf = 2 * V_avg * T_travel := by
  sorry

end NUMINAMATH_CALUDE_wharf_length_l3549_354932


namespace NUMINAMATH_CALUDE_yulgi_allowance_l3549_354927

theorem yulgi_allowance (Y G : ℕ) 
  (sum : Y + G = 6000)
  (sum_minus_diff : Y + G - (Y - G) = 4800)
  (Y_greater : Y > G) : Y = 3600 := by
sorry

end NUMINAMATH_CALUDE_yulgi_allowance_l3549_354927


namespace NUMINAMATH_CALUDE_jakes_weight_ratio_l3549_354920

/-- Proves that the ratio of Jake's weight after losing 20 pounds to his sister's weight is 2:1 -/
theorem jakes_weight_ratio (jake_weight sister_weight : ℕ) : 
  jake_weight = 156 →
  jake_weight + sister_weight = 224 →
  (jake_weight - 20) / sister_weight = 2 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_ratio_l3549_354920


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3549_354916

/-- Given two pipes A and B that can fill a tank, this theorem proves the time
    it takes for pipe B to fill the tank alone, given the filling times for
    pipe A alone and both pipes together. -/
theorem pipe_filling_time (fill_time_A fill_time_both : ℝ) 
  (h1 : fill_time_A = 30) 
  (h2 : fill_time_both = 18) : 
  (1 / fill_time_A + 1 / (1 / (1 / fill_time_both - 1 / fill_time_A)))⁻¹ = 45 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l3549_354916


namespace NUMINAMATH_CALUDE_simplify_expression_l3549_354967

theorem simplify_expression (x : ℝ) : 105*x - 57*x + 8 = 48*x + 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3549_354967


namespace NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_jack_is_9_221_l3549_354937

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_jacks : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of the given event -/
def probability_two_queens_or_at_least_one_jack (d : Deck) : ℚ :=
  sorry

/-- Theorem stating the probability of drawing either two queens or at least one jack -/
theorem prob_two_queens_or_at_least_one_jack_is_9_221 :
  let standard_deck : Deck := ⟨52, 1, 3⟩
  probability_two_queens_or_at_least_one_jack standard_deck = 9/221 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_jack_is_9_221_l3549_354937


namespace NUMINAMATH_CALUDE_equation_solution_l3549_354906

theorem equation_solution :
  let x : ℝ := -Real.sqrt 3
  let y : ℝ := 4
  x^2 + 2 * Real.sqrt 3 * x + y - 4 * Real.sqrt y + 7 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3549_354906


namespace NUMINAMATH_CALUDE_division_chain_l3549_354995

theorem division_chain : (132 / 6) / 2 = 11 := by sorry

end NUMINAMATH_CALUDE_division_chain_l3549_354995


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3549_354991

theorem least_positive_integer_with_remainders : ∃! N : ℕ,
  N > 0 ∧
  N % 4 = 3 ∧
  N % 5 = 4 ∧
  N % 6 = 5 ∧
  N % 7 = 6 ∧
  ∀ M : ℕ, (M > 0 ∧ M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6) → N ≤ M :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3549_354991


namespace NUMINAMATH_CALUDE_number_plus_five_equals_500_l3549_354911

theorem number_plus_five_equals_500 : ∃ x : ℤ, x + 5 = 500 ∧ x = 495 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_five_equals_500_l3549_354911


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l3549_354949

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → x % 4 = 0 → x^3 < 8000 → x ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l3549_354949


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_one_l3549_354988

theorem trig_expression_equals_negative_one :
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) - Real.sin (6 * π / 180) * Real.sin (66 * π / 180)) /
  (Real.sin (21 * π / 180) * Real.cos (39 * π / 180) - Real.sin (39 * π / 180) * Real.cos (21 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_one_l3549_354988


namespace NUMINAMATH_CALUDE_monotonicity_criterion_other_statements_incorrect_l3549_354957

/-- A function f is monotonically decreasing on ℝ if for all x₁ < x₂, f(x₁) > f(x₂) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem monotonicity_criterion (f : ℝ → ℝ) :
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ ≤ f x₂) → ¬(MonotonicallyDecreasing f) :=
by sorry

theorem other_statements_incorrect (f : ℝ → ℝ) :
  ¬(∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ → (∀ y₁ y₂, y₁ < y₂ → f y₁ < f y₂)) ∧
  ¬(∀ x₂ > 0, (∀ x₁, f x₁ < f (x₁ + x₂)) → (∀ y₁ y₂, y₁ < y₂ → f y₁ < f y₂)) ∧
  ¬(∀ x₁ x₂, x₁ < x₂ → f x₁ ≥ f x₂ → MonotonicallyDecreasing f) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_criterion_other_statements_incorrect_l3549_354957


namespace NUMINAMATH_CALUDE_theater_admission_revenue_l3549_354980

/-- Calculates the total amount collected from theater admissions --/
def total_amount_collected (adult_price child_price : ℚ) (total_attendance children_attendance : ℕ) : ℚ :=
  let adults_attendance := total_attendance - children_attendance
  let adult_revenue := adult_price * adults_attendance
  let child_revenue := child_price * children_attendance
  adult_revenue + child_revenue

/-- Theorem stating that the total amount collected is $140 given the specified conditions --/
theorem theater_admission_revenue :
  total_amount_collected (60/100) (25/100) 280 80 = 140 := by
  sorry

end NUMINAMATH_CALUDE_theater_admission_revenue_l3549_354980


namespace NUMINAMATH_CALUDE_equation_solution_l3549_354923

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → ((4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 5) → x = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3549_354923


namespace NUMINAMATH_CALUDE_max_value_of_function_l3549_354938

theorem max_value_of_function (x : ℝ) : 
  (3 * Real.sin x + 2 * Real.sqrt (2 + 2 * Real.cos (2 * x))) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3549_354938


namespace NUMINAMATH_CALUDE_gcd_12345_6789_l3549_354904

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_6789_l3549_354904


namespace NUMINAMATH_CALUDE_material_left_proof_l3549_354940

theorem material_left_proof (material1 material2 used_material : ℚ) : 
  material1 = 5/11 →
  material2 = 2/3 →
  used_material = 2/3 →
  material1 + material2 - used_material = 5/11 := by
sorry

end NUMINAMATH_CALUDE_material_left_proof_l3549_354940


namespace NUMINAMATH_CALUDE_smallest_positive_angle_neg_1050_l3549_354974

/-- The smallest positive angle (in degrees) with the same terminal side as a given angle -/
def smallestPositiveEquivalentAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

/-- Theorem: The smallest positive angle with the same terminal side as -1050° is 30° -/
theorem smallest_positive_angle_neg_1050 :
  smallestPositiveEquivalentAngle (-1050) = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_neg_1050_l3549_354974


namespace NUMINAMATH_CALUDE_fraction_value_l3549_354908

theorem fraction_value : (2222 - 2123)^2 / 121 = 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3549_354908


namespace NUMINAMATH_CALUDE_count_integer_solutions_l3549_354926

theorem count_integer_solutions : 
  ∃! (S : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ S ↔ m > 0 ∧ n > 0 ∧ 5 / m + 3 / n = 1) ∧ 
    Finset.card S = 4 :=
by sorry

end NUMINAMATH_CALUDE_count_integer_solutions_l3549_354926


namespace NUMINAMATH_CALUDE_line_arrangements_with_restriction_l3549_354968

def number_of_students : ℕ := 5

def number_of_restricted_students : ℕ := 2

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def arrangements_with_restricted_together (n : ℕ) (r : ℕ) : ℕ :=
  (Nat.factorial (n - r + 1)) * (Nat.factorial r)

theorem line_arrangements_with_restriction :
  total_arrangements number_of_students - 
  arrangements_with_restricted_together number_of_students number_of_restricted_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_arrangements_with_restriction_l3549_354968


namespace NUMINAMATH_CALUDE_count_even_factors_div_by_5_l3549_354951

/-- The number of even natural-number factors divisible by 5 of 2^3 * 5^2 * 11^1 -/
def num_even_factors_div_by_5 : ℕ :=
  let n : ℕ := 2^3 * 5^2 * 11^1
  -- Define the function here
  12

theorem count_even_factors_div_by_5 :
  num_even_factors_div_by_5 = 12 := by sorry

end NUMINAMATH_CALUDE_count_even_factors_div_by_5_l3549_354951


namespace NUMINAMATH_CALUDE_inequality_solution_l3549_354930

theorem inequality_solution : 
  {x : ℕ | 2 * x - 1 < 5} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3549_354930


namespace NUMINAMATH_CALUDE_cosine_value_proof_l3549_354913

theorem cosine_value_proof (α : ℝ) (h : Real.sin (π/6 - α) = 4/5) : 
  Real.cos (π/3 + α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_proof_l3549_354913


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3549_354944

theorem sum_of_three_numbers : 4.75 + 0.303 + 0.432 = 5.485 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3549_354944


namespace NUMINAMATH_CALUDE_book_weight_is_205_l3549_354998

/-- Calculates the weight of a single book given the following conditions:
  * 6 books in each small box
  * Small box weighs 220 grams
  * 9 small boxes in a large box
  * Large box weighs 250 grams
  * Total weight is 13.3 kilograms
  * All books weigh the same
-/
def bookWeight (booksPerSmallBox : ℕ) (smallBoxWeight : ℕ) (smallBoxCount : ℕ) 
                (largeBoxWeight : ℕ) (totalWeightKg : ℚ) : ℚ :=
  let totalWeightG : ℚ := totalWeightKg * 1000
  let smallBoxesWeight : ℚ := smallBoxWeight * smallBoxCount
  let booksWeight : ℚ := totalWeightG - largeBoxWeight - smallBoxesWeight
  let totalBooks : ℕ := booksPerSmallBox * smallBoxCount
  booksWeight / totalBooks

theorem book_weight_is_205 :
  bookWeight 6 220 9 250 (13.3 : ℚ) = 205 := by
  sorry

#eval bookWeight 6 220 9 250 (13.3 : ℚ)

end NUMINAMATH_CALUDE_book_weight_is_205_l3549_354998


namespace NUMINAMATH_CALUDE_absolute_value_calculation_l3549_354950

theorem absolute_value_calculation : |-3| * 2 - (-1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_l3549_354950


namespace NUMINAMATH_CALUDE_painting_problem_l3549_354976

/-- The fraction of a wall that can be painted by two people working together in a given time -/
def combined_painting_fraction (rate1 rate2 time : ℚ) : ℚ :=
  (rate1 + rate2) * time

theorem painting_problem :
  let heidi_rate : ℚ := 1 / 60
  let linda_rate : ℚ := 1 / 40
  let work_time : ℚ := 12
  combined_painting_fraction heidi_rate linda_rate work_time = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_painting_problem_l3549_354976


namespace NUMINAMATH_CALUDE_simplify_cube_root_exponent_sum_l3549_354969

theorem simplify_cube_root_exponent_sum (a b c : ℝ) : 
  ∃ (k : ℝ) (m n p : ℕ), 
    (54 * a^6 * b^8 * c^14)^(1/3) = k * a^m * b^n * c^p ∧ m + n + p = 8 :=
by sorry

end NUMINAMATH_CALUDE_simplify_cube_root_exponent_sum_l3549_354969


namespace NUMINAMATH_CALUDE_x_power_minus_reciprocal_l3549_354910

theorem x_power_minus_reciprocal (θ : Real) (x : Real) (n : Nat) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x - 1/x = 2 * Real.sin θ) (h4 : n > 0) : 
  x^n - 1/(x^n) = 2 * Real.sinh (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_x_power_minus_reciprocal_l3549_354910


namespace NUMINAMATH_CALUDE_base7_521_equals_260_l3549_354947

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem base7_521_equals_260 :
  base7ToBase10 [1, 2, 5] = 260 := by
  sorry

end NUMINAMATH_CALUDE_base7_521_equals_260_l3549_354947


namespace NUMINAMATH_CALUDE_digits_of_2_15_times_5_6_l3549_354993

/-- The number of digits in 2^15 * 5^6 is 9 -/
theorem digits_of_2_15_times_5_6 : (Nat.digits 10 (2^15 * 5^6)).length = 9 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_2_15_times_5_6_l3549_354993


namespace NUMINAMATH_CALUDE_survey_result_l3549_354983

theorem survey_result : ∀ (total : ℕ) (dangerous : ℕ) (fire : ℕ),
  (dangerous : ℚ) / total = 825 / 1000 →
  (fire : ℚ) / dangerous = 524 / 1000 →
  fire = 27 →
  total = 63 := by
sorry

end NUMINAMATH_CALUDE_survey_result_l3549_354983


namespace NUMINAMATH_CALUDE_function_derivative_at_two_l3549_354961

/-- Given a function f(x) = a*ln(x) + b/x where f(1) = -2 and f'(1) = 0, prove that f'(2) = -1/2 -/
theorem function_derivative_at_two 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x > 0, f x = a * Real.log x + b / x)
  (h2 : f 1 = -2)
  (h3 : deriv f 1 = 0) :
  deriv f 2 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_function_derivative_at_two_l3549_354961


namespace NUMINAMATH_CALUDE_cheerful_not_green_l3549_354912

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


end NUMINAMATH_CALUDE_cheerful_not_green_l3549_354912


namespace NUMINAMATH_CALUDE_microtron_stock_price_l3549_354917

/-- Represents the stock market scenario with Microtron and Dynaco stocks -/
structure StockMarket where
  microtron_price : ℝ
  dynaco_price : ℝ
  total_shares_sold : ℕ
  average_price : ℝ
  dynaco_shares_sold : ℕ

/-- Theorem stating the price of Microtron stock given the market conditions -/
theorem microtron_stock_price (market : StockMarket) 
  (h1 : market.dynaco_price = 44)
  (h2 : market.total_shares_sold = 300)
  (h3 : market.average_price = 40)
  (h4 : market.dynaco_shares_sold = 150) :
  market.microtron_price = 36 := by
  sorry

end NUMINAMATH_CALUDE_microtron_stock_price_l3549_354917


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3549_354962

-- Define the universal set U
def U : Set Char := {'a', 'b', 'c', 'd', 'e'}

-- Define set A
def A : Set Char := {'a', 'b', 'c', 'd'}

-- Define set B
def B : Set Char := {'d', 'e'}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {'d'} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3549_354962


namespace NUMINAMATH_CALUDE_forgotten_angles_sum_l3549_354972

theorem forgotten_angles_sum (n : ℕ) (measured_sum : ℝ) : 
  n > 2 → 
  measured_sum = 2873 → 
  ∃ (missing_sum : ℝ), 
    missing_sum = ((n - 2) * 180 : ℝ) - measured_sum ∧ 
    missing_sum = 7 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_angles_sum_l3549_354972


namespace NUMINAMATH_CALUDE_intersection_point_on_fixed_line_l3549_354986

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  eccentricity : ℝ
  left_vertex : ℝ × ℝ
  right_vertex : ℝ × ℝ

-- Define the intersection points M and N
structure IntersectionPoints where
  M : ℝ × ℝ
  N : ℝ × ℝ

-- Define the point P
def P (h : Hyperbola) (i : IntersectionPoints) : ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_point_on_fixed_line 
  (h : Hyperbola) 
  (i : IntersectionPoints) :
  h.center = (0, 0) →
  h.left_focus = (-2 * Real.sqrt 5, 0) →
  h.eccentricity = Real.sqrt 5 →
  h.left_vertex = (-2, 0) →
  h.right_vertex = (2, 0) →
  (∃ (m : ℝ), i.M.1 = m * i.M.2 - 4 ∧ i.N.1 = m * i.N.2 - 4) →
  i.M.2 > 0 →
  (P h i).1 = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_on_fixed_line_l3549_354986


namespace NUMINAMATH_CALUDE_ellipse_k_range_l3549_354966

theorem ellipse_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2 / (k - 4) + y^2 / (10 - k) = 1) ∧ 
  (∀ x y : ℝ, x^2 / (k - 4) + y^2 / (10 - k) = 1 → 
    ∃ a b c : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ c^2 = a^2 - b^2 ∧ c ≠ 0) →
  k > 7 ∧ k < 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l3549_354966


namespace NUMINAMATH_CALUDE_function_composition_identity_l3549_354985

/-- Given two functions f and g defined as f(x) = Ax² + B and g(x) = Bx² + A,
    where A ≠ B, if f(g(x)) - g(f(x)) = 2(B - A) for all x, then A + B = 0 -/
theorem function_composition_identity (A B : ℝ) (h : A ≠ B) :
  (∀ x : ℝ, (A * (B * x^2 + A)^2 + B) - (B * (A * x^2 + B)^2 + A) = 2 * (B - A)) →
  A + B = 0 := by
sorry


end NUMINAMATH_CALUDE_function_composition_identity_l3549_354985


namespace NUMINAMATH_CALUDE_mass_ratio_simplification_l3549_354999

-- Define the units
def kg : ℚ → ℚ := id
def ton : ℚ → ℚ := (· * 1000)

-- Define the ratio
def ratio (a b : ℚ) : ℚ × ℚ := (a, b)

-- Define the problem
theorem mass_ratio_simplification :
  let mass1 := kg 250
  let mass2 := ton 0.5
  let simplified_ratio := ratio 1 2
  let decimal_value := 0.5
  (mass1 / mass2 = decimal_value) ∧
  (ratio (mass1 / gcd mass1 mass2) (mass2 / gcd mass1 mass2) = simplified_ratio) := by
  sorry


end NUMINAMATH_CALUDE_mass_ratio_simplification_l3549_354999


namespace NUMINAMATH_CALUDE_mini_van_capacity_correct_l3549_354918

/-- Represents the capacity of a mini-van's tank in liters -/
def mini_van_capacity : ℝ := 65

/-- Represents the service cost per vehicle in dollars -/
def service_cost : ℝ := 2.10

/-- Represents the fuel cost per liter in dollars -/
def fuel_cost : ℝ := 0.60

/-- Represents the number of mini-vans -/
def num_mini_vans : ℕ := 3

/-- Represents the number of trucks -/
def num_trucks : ℕ := 2

/-- Represents the total cost in dollars -/
def total_cost : ℝ := 299.1

/-- Represents the ratio of truck tank capacity to mini-van tank capacity -/
def truck_capacity_ratio : ℝ := 2.2

theorem mini_van_capacity_correct :
  service_cost * (num_mini_vans + num_trucks) +
  fuel_cost * (num_mini_vans * mini_van_capacity + num_trucks * (truck_capacity_ratio * mini_van_capacity)) =
  total_cost := by sorry

end NUMINAMATH_CALUDE_mini_van_capacity_correct_l3549_354918


namespace NUMINAMATH_CALUDE_divisibility_by_9_52B7_l3549_354943

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def number_52B7 (B : ℕ) : ℕ := 5000 + 200 + B * 10 + 7

theorem divisibility_by_9_52B7 :
  ∀ B : ℕ, B < 10 → (is_divisible_by_9 (number_52B7 B) ↔ B = 4) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_9_52B7_l3549_354943


namespace NUMINAMATH_CALUDE_jellybean_problem_l3549_354953

theorem jellybean_problem (x : ℚ) 
  (caleb_jellybeans : x > 0)
  (sophie_jellybeans : ℚ → ℚ)
  (sophie_half_caleb : sophie_jellybeans x = x / 2)
  (total_jellybeans : 12 * x + 12 * (sophie_jellybeans x) = 54) :
  x = 3 := by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3549_354953


namespace NUMINAMATH_CALUDE_num_possible_lists_eq_50625_l3549_354902

/-- The number of balls in the bin -/
def num_balls : ℕ := 15

/-- The number of draws -/
def num_draws : ℕ := 4

/-- The number of possible lists when drawing 'num_draws' times from 'num_balls' with replacement -/
def num_possible_lists : ℕ := num_balls ^ num_draws

/-- Theorem stating that the number of possible lists is 50625 -/
theorem num_possible_lists_eq_50625 : num_possible_lists = 50625 := by
  sorry

end NUMINAMATH_CALUDE_num_possible_lists_eq_50625_l3549_354902


namespace NUMINAMATH_CALUDE_tissue_cost_theorem_l3549_354941

/-- Calculates the total cost of tissues given the number of boxes, packs per box,
    tissues per pack, and price per tissue. -/
def totalCost (boxes : ℕ) (packsPerBox : ℕ) (tissuesPerPack : ℕ) (pricePerTissue : ℚ) : ℚ :=
  boxes * packsPerBox * tissuesPerPack * pricePerTissue

/-- Proves that the total cost of 10 boxes of tissues is $1,000 given the specified conditions. -/
theorem tissue_cost_theorem :
  totalCost 10 20 100 (5 / 100) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_tissue_cost_theorem_l3549_354941


namespace NUMINAMATH_CALUDE_pentagon_area_l3549_354952

-- Define the pentagon
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  angle : ℝ

-- Define the given pentagon
def given_pentagon : Pentagon :=
  { side1 := 18
  , side2 := 25
  , side3 := 30
  , side4 := 28
  , side5 := 22
  , angle := 110 }

-- Define the area calculation function
noncomputable def calculate_area (p : Pentagon) : ℝ := sorry

-- Theorem stating the area of the given pentagon
theorem pentagon_area :
  ∃ ε > 0, |calculate_area given_pentagon - 738| < ε := by sorry

end NUMINAMATH_CALUDE_pentagon_area_l3549_354952


namespace NUMINAMATH_CALUDE_go_stones_count_l3549_354959

/-- Calculates the total number of go stones given the number of stones per bundle,
    the number of bundles of black stones, and the number of white stones. -/
def total_go_stones (stones_per_bundle : ℕ) (black_bundles : ℕ) (white_stones : ℕ) : ℕ :=
  stones_per_bundle * black_bundles + white_stones

/-- Proves that the total number of go stones is 46 given the specified conditions. -/
theorem go_stones_count : total_go_stones 10 3 16 = 46 := by
  sorry

end NUMINAMATH_CALUDE_go_stones_count_l3549_354959


namespace NUMINAMATH_CALUDE_divide_by_reciprocal_twelve_divided_by_one_twelfth_l3549_354905

theorem divide_by_reciprocal (a b : ℚ) (h : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_twelfth : 12 / (1 / 12) = 144 := by sorry

end NUMINAMATH_CALUDE_divide_by_reciprocal_twelve_divided_by_one_twelfth_l3549_354905


namespace NUMINAMATH_CALUDE_rockham_soccer_league_members_l3549_354919

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

end NUMINAMATH_CALUDE_rockham_soccer_league_members_l3549_354919


namespace NUMINAMATH_CALUDE_min_movements_for_ten_l3549_354997

/-- Represents a circular arrangement of n distinct elements -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- A single movement in the circular arrangement -/
def Movement (n : ℕ) (arr : CircularArrangement n) (i j : Fin n) : CircularArrangement n :=
  sorry

/-- Checks if the circular arrangement is sorted in ascending order clockwise -/
def IsSorted (n : ℕ) (arr : CircularArrangement n) : Prop :=
  sorry

/-- The minimum number of movements required to sort the arrangement -/
def MinMovements (n : ℕ) (arr : CircularArrangement n) : ℕ :=
  sorry

/-- Theorem: For 10 distinct elements, 8 movements are always sufficient and necessary -/
theorem min_movements_for_ten :
  ∀ (arr : CircularArrangement 10),
    (∀ i j : Fin 10, i ≠ j → arr i ≠ arr j) →
    MinMovements 10 arr = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_movements_for_ten_l3549_354997


namespace NUMINAMATH_CALUDE_triangle_tangent_l3549_354924

theorem triangle_tangent (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : Real.tan A = 1/2) (h3 : Real.cos B = (3 * Real.sqrt 10) / 10) : 
  Real.tan C = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_l3549_354924


namespace NUMINAMATH_CALUDE_enclosure_posts_count_l3549_354975

/-- Calculates the number of posts needed for a rectangular enclosure with a stone wall --/
def calculate_posts (length width wall_length post_spacing : ℕ) : ℕ :=
  let long_side := max length width
  let short_side := min length width
  let long_side_posts := long_side / post_spacing + 1
  let short_side_posts := (short_side / post_spacing + 1) - 1
  long_side_posts + 2 * short_side_posts

/-- The number of posts required for the given enclosure is 19 --/
theorem enclosure_posts_count :
  calculate_posts 50 80 120 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_enclosure_posts_count_l3549_354975


namespace NUMINAMATH_CALUDE_equal_area_triangles_l3549_354955

noncomputable def triangle_area (a b : ℝ) (θ : ℝ) : ℝ := (1/2) * a * b * Real.sin θ

theorem equal_area_triangles (AB AC AD : ℝ) (θ : ℝ) (AE : ℝ) : 
  AB = 4 →
  AC = 5 →
  AD = 2.5 →
  θ = Real.pi / 3 →
  triangle_area AB AC θ = triangle_area AD AE θ →
  AE = 8 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l3549_354955


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_half_l3549_354909

/-- Two vectors are parallel if their cross product is zero -/
def IsParallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_imply_x_half :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (1, x)
  IsParallel (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_half_l3549_354909


namespace NUMINAMATH_CALUDE_smaug_silver_coins_l3549_354970

/-- Represents the number of coins of each type in Smaug's hoard -/
structure DragonHoard where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Calculates the total value of the hoard in copper coins -/
def hoardValue (h : DragonHoard) : ℕ :=
  h.gold * 3 * 8 + h.silver * 8 + h.copper

/-- Theorem stating that Smaug has 60 silver coins -/
theorem smaug_silver_coins :
  ∃ h : DragonHoard,
    h.gold = 100 ∧
    h.copper = 33 ∧
    hoardValue h = 2913 ∧
    h.silver = 60 := by
  sorry

end NUMINAMATH_CALUDE_smaug_silver_coins_l3549_354970


namespace NUMINAMATH_CALUDE_equation_condition_l3549_354963

theorem equation_condition (a d e : ℕ) : 
  (0 < a ∧ a < 10) → (0 < d ∧ d < 10) → (0 < e ∧ e < 10) →
  ((10 * a + d) * (10 * a + e) = 100 * a^2 + 110 * a + d * e ↔ d + e = 11) :=
by sorry

end NUMINAMATH_CALUDE_equation_condition_l3549_354963


namespace NUMINAMATH_CALUDE_segment_ratio_l3549_354936

/-- Given a line segment GH with points E and F on it, where GE is 3 times EH and GF is 4 times FH,
    prove that EF is 1/20 of GH. -/
theorem segment_ratio (G E F H : Real) (GH EF : Real) : 
  E ∈ Set.Icc G H → 
  F ∈ Set.Icc G H → 
  G - E = 3 * (H - E) → 
  G - F = 4 * (H - F) → 
  GH = G - H → 
  EF = E - F → 
  EF = (1 / 20) * GH := by
sorry

end NUMINAMATH_CALUDE_segment_ratio_l3549_354936


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3549_354992

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3549_354992
