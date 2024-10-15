import Mathlib

namespace NUMINAMATH_CALUDE_money_distribution_l569_56973

/-- Given three people A, B, and C with the following conditions:
  - The total amount between A, B, and C is 900
  - A and C together have 400
  - B and C together have 750
  Prove that C has 250. -/
theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 900)
  (h2 : A + C = 400)
  (h3 : B + C = 750) : 
  C = 250 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l569_56973


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l569_56912

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) ↔ a < -1 ∨ a > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l569_56912


namespace NUMINAMATH_CALUDE_shaded_area_sum_l569_56960

theorem shaded_area_sum (r₁ : ℝ) (r₂ : ℝ) : 
  r₁ > 0 → 
  r₂ > 0 → 
  r₁ = 8 → 
  r₂ = r₁ / 2 → 
  (π * r₁^2) / 2 + (π * r₂^2) / 2 = 40 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_sum_l569_56960


namespace NUMINAMATH_CALUDE_probability_two_green_balls_l569_56979

def total_balls : ℕ := 12
def red_balls : ℕ := 3
def yellow_balls : ℕ := 5
def green_balls : ℕ := 4
def drawn_balls : ℕ := 3

theorem probability_two_green_balls :
  (Nat.choose green_balls 2 * Nat.choose (total_balls - green_balls) 1) /
  Nat.choose total_balls drawn_balls = 12 / 55 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_green_balls_l569_56979


namespace NUMINAMATH_CALUDE_number_division_theorem_l569_56953

theorem number_division_theorem : 
  ∃ (n : ℕ), (n : ℝ) / 189 = 18.444444444444443 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_number_division_theorem_l569_56953


namespace NUMINAMATH_CALUDE_birds_flew_away_l569_56950

theorem birds_flew_away (original : ℝ) (remaining : ℝ) (flew_away : ℝ) : 
  original = 21.0 → remaining = 7 → flew_away = original - remaining → flew_away = 14.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_flew_away_l569_56950


namespace NUMINAMATH_CALUDE_max_vertex_sum_l569_56919

def parabola (a b c : ℤ) (x : ℚ) : ℚ := a * x^2 + b * x + c

theorem max_vertex_sum (a T : ℤ) (h : T ≠ 0) :
  ∃ b c : ℤ,
    (parabola a b c 0 = 0) ∧
    (parabola a b c (3 * T) = 0) ∧
    (parabola a b c (3 * T + 1) = 36) →
    ∃ x y : ℚ,
      (∀ t : ℚ, parabola a b c t ≤ parabola a b c x) ∧
      y = parabola a b c x ∧
      x + y ≤ 62 :=
by sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l569_56919


namespace NUMINAMATH_CALUDE_rotate_vector_2_3_l569_56939

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Rotates a 2D vector 90 degrees clockwise -/
def rotate90Clockwise (v : Vector2D) : Vector2D :=
  { x := v.y, y := -v.x }

/-- The theorem stating that rotating (2, 3) by 90 degrees clockwise results in (3, -2) -/
theorem rotate_vector_2_3 :
  rotate90Clockwise { x := 2, y := 3 } = { x := 3, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_rotate_vector_2_3_l569_56939


namespace NUMINAMATH_CALUDE_fundraising_event_l569_56925

theorem fundraising_event (p : ℝ) (initial_boys : ℕ) :
  -- Initial conditions
  initial_boys = Int.floor (0.35 * p) →
  -- Changes in group composition
  (initial_boys - 3 + 2) / (p + 3) = 0.3 →
  -- Conclusion
  initial_boys = 13 := by
sorry

end NUMINAMATH_CALUDE_fundraising_event_l569_56925


namespace NUMINAMATH_CALUDE_jerrie_situp_minutes_l569_56900

/-- The number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- The number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The number of minutes Barney does sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie does sit-ups -/
def carrie_minutes : ℕ := 2

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := 510

/-- Theorem stating that Jerrie did sit-ups for 3 minutes -/
theorem jerrie_situp_minutes :
  ∃ (j : ℕ), j * jerrie_situps + barney_minutes * barney_situps + carrie_minutes * carrie_situps = total_situps ∧ j = 3 :=
by sorry

end NUMINAMATH_CALUDE_jerrie_situp_minutes_l569_56900


namespace NUMINAMATH_CALUDE_triangle_property_l569_56994

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with given conditions -/
theorem triangle_property (t : Triangle) 
  (h1 : Real.sin t.A + Real.sin t.B = 5/4 * Real.sin t.C)
  (h2 : t.a + t.b + t.c = 9)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sin t.C) :
  t.c = 4 ∧ Real.cos t.C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l569_56994


namespace NUMINAMATH_CALUDE_distribution_count_is_18_l569_56999

/-- The number of ways to distribute 6 numbered balls into 3 boxes -/
def distributionCount : ℕ :=
  let totalBalls : ℕ := 6
  let numBoxes : ℕ := 3
  let ballsPerBox : ℕ := 2
  let fixedPair : Fin totalBalls := 2  -- Represents balls 1 and 2 as a fixed pair
  18

/-- Theorem stating that the number of distributions is 18 -/
theorem distribution_count_is_18 : distributionCount = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_is_18_l569_56999


namespace NUMINAMATH_CALUDE_polynomial_value_at_root_l569_56940

theorem polynomial_value_at_root (p : ℝ) : 
  p^3 - 5*p + 1 = 0 → p^4 - 3*p^3 - 5*p^2 + 16*p + 2015 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_root_l569_56940


namespace NUMINAMATH_CALUDE_sport_water_amount_l569_56906

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  { flavoring := standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup / 3,
    water := standard_ratio.water * 2 }

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 4

/-- Theorem stating the amount of water in the sport formulation -/
theorem sport_water_amount :
  (sport_corn_syrup * sport_ratio.water) / sport_ratio.corn_syrup = 15 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l569_56906


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_964807_div_8_l569_56976

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem least_subtraction_964807_div_8 :
  ∃ (k : Nat), k < 8 ∧ (964807 - k) % 8 = 0 ∧ ∀ (m : Nat), m < k → (964807 - m) % 8 ≠ 0 ∧ k = 7 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_964807_div_8_l569_56976


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l569_56954

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l569_56954


namespace NUMINAMATH_CALUDE_fraction_equality_l569_56933

theorem fraction_equality : (1 : ℝ) / (2 - Real.sqrt 3) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l569_56933


namespace NUMINAMATH_CALUDE_korona_division_l569_56946

theorem korona_division (total : ℕ) (a b c d : ℝ) :
  total = 9246 →
  (2 * a = 3 * b) →
  (5 * b = 6 * c) →
  (3 * c = 4 * d) →
  (a + b + c + d = total) →
  ∃ (k : ℝ), k > 0 ∧ a = 1380 * k ∧ b = 2070 * k ∧ c = 2484 * k ∧ d = 3312 * k :=
by sorry

end NUMINAMATH_CALUDE_korona_division_l569_56946


namespace NUMINAMATH_CALUDE_cattle_transport_time_l569_56932

/-- Calculates the total driving time required to transport cattle to higher ground -/
theorem cattle_transport_time 
  (total_cattle : ℕ) 
  (distance : ℕ) 
  (truck_capacity : ℕ) 
  (speed : ℕ) 
  (h1 : total_cattle = 400)
  (h2 : distance = 60)
  (h3 : truck_capacity = 20)
  (h4 : speed = 60)
  : (total_cattle / truck_capacity) * (2 * distance) / speed = 40 := by
  sorry

end NUMINAMATH_CALUDE_cattle_transport_time_l569_56932


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l569_56959

def initial_average : ℝ := 40
def initial_count : ℕ := 8
def leaving_age : ℝ := 25
def final_count : ℕ := 7

theorem average_age_after_leaving :
  let initial_total_age := initial_average * initial_count
  let remaining_total_age := initial_total_age - leaving_age
  let final_average := remaining_total_age / final_count
  final_average = 42 := by sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l569_56959


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l569_56980

theorem largest_n_for_equation : 
  (∀ n : ℕ, n > 4 → ¬∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) ∧
  (∃ x y z : ℕ+, 4^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l569_56980


namespace NUMINAMATH_CALUDE_max_wrong_questions_l569_56951

theorem max_wrong_questions (total_questions : Nat) (success_percentage : Rat) 
  (h1 : total_questions = 50)
  (h2 : success_percentage = 75 / 100) :
  ∃ (max_wrong : Nat), 
    (max_wrong ≤ total_questions) ∧ 
    ((total_questions - max_wrong : Rat) / total_questions ≥ success_percentage) ∧
    (∀ (n : Nat), n > max_wrong → (total_questions - n : Rat) / total_questions < success_percentage) ∧
    max_wrong = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_wrong_questions_l569_56951


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l569_56931

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with given side length -/
structure Cube where
  sideLength : ℝ

/-- Defines the position of points P, Q, R on the cube edges -/
structure PointsOnCube where
  cube : Cube
  P : Point3D
  Q : Point3D
  R : Point3D

/-- Calculates the area of the intersection polygon -/
def intersectionArea (c : Cube) (pts : PointsOnCube) : ℝ :=
  sorry

/-- Theorem stating the area of the intersection polygon -/
theorem intersection_area_theorem (c : Cube) (pts : PointsOnCube) :
  c.sideLength = 30 ∧
  pts.P.x = 10 ∧ pts.P.y = 0 ∧ pts.P.z = 0 ∧
  pts.Q.x = 30 ∧ pts.Q.y = 0 ∧ pts.Q.z = 20 ∧
  pts.R.x = 30 ∧ pts.R.y = 5 ∧ pts.R.z = 30 →
  intersectionArea c pts = 450 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l569_56931


namespace NUMINAMATH_CALUDE_age_of_other_man_is_21_l569_56974

/-- The age of the other replaced man in a group replacement scenario -/
def age_of_other_replaced_man (initial_count : ℕ) (age_increase : ℝ) (age_of_one_replaced : ℕ) (avg_age_new_men : ℝ) : ℝ :=
  let total_age_increase := initial_count * age_increase
  let total_age_new_men := 2 * avg_age_new_men
  total_age_new_men - total_age_increase - age_of_one_replaced

/-- Theorem: The age of the other replaced man is 21 years -/
theorem age_of_other_man_is_21 :
  age_of_other_replaced_man 10 2 23 32 = 21 := by
  sorry

#eval age_of_other_replaced_man 10 2 23 32

end NUMINAMATH_CALUDE_age_of_other_man_is_21_l569_56974


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l569_56984

theorem polynomial_division_quotient :
  ∀ x : ℝ, x ≠ 1 →
  (x^6 + 5) / (x - 1) = x^5 + x^4 + x^3 + x^2 + x + 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l569_56984


namespace NUMINAMATH_CALUDE_conjunction_false_implication_l569_56941

theorem conjunction_false_implication : ∃ (p q : Prop), (p ∧ q → False) ∧ ¬(p → False ∧ q → False) := by sorry

end NUMINAMATH_CALUDE_conjunction_false_implication_l569_56941


namespace NUMINAMATH_CALUDE_investment_rate_problem_l569_56935

theorem investment_rate_problem (total_investment remaining_investment : ℚ)
  (rate1 rate2 required_rate : ℚ) (investment1 investment2 : ℚ) (desired_income : ℚ)
  (h1 : total_investment = 12000)
  (h2 : investment1 = 5000)
  (h3 : investment2 = 4000)
  (h4 : rate1 = 3 / 100)
  (h5 : rate2 = 9 / 200)
  (h6 : desired_income = 600)
  (h7 : remaining_investment = total_investment - investment1 - investment2)
  (h8 : desired_income = investment1 * rate1 + investment2 * rate2 + remaining_investment * required_rate) :
  required_rate = 9 / 100 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l569_56935


namespace NUMINAMATH_CALUDE_percentage_problem_l569_56952

theorem percentage_problem (P : ℝ) : P = 30 :=
by
  -- Define the condition from the problem
  have h1 : P / 100 * 100 = 50 / 100 * 40 + 10 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l569_56952


namespace NUMINAMATH_CALUDE_alice_bob_number_sum_l569_56958

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
    A ∈ Finset.range 50 →
    B ∈ Finset.range 50 →
    (∀ x ∈ Finset.range 50, x ≠ A → ¬(A > x ↔ B > x)) →
    (∀ y ∈ Finset.range 50, y ≠ B → (B > y ↔ A < y)) →
    is_prime B →
    B % 2 = 0 →
    is_perfect_square (90 * B + A) →
    A + B = 18 :=
by sorry

end NUMINAMATH_CALUDE_alice_bob_number_sum_l569_56958


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l569_56911

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 3 + a 9 = 16 → a 5 + a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l569_56911


namespace NUMINAMATH_CALUDE_max_tulips_is_15_l569_56920

/-- Represents the cost of yellow and red tulips in rubles -/
structure TulipCosts where
  yellow : ℕ
  red : ℕ

/-- Represents the number of yellow and red tulips in the bouquet -/
structure Bouquet where
  yellow : ℕ
  red : ℕ

/-- Calculates the total cost of a bouquet given the costs of tulips -/
def totalCost (b : Bouquet) (c : TulipCosts) : ℕ :=
  b.yellow * c.yellow + b.red * c.red

/-- Checks if a bouquet satisfies the conditions -/
def isValidBouquet (b : Bouquet) : Prop :=
  (b.yellow + b.red) % 2 = 1 ∧ 
  (b.yellow = b.red + 1 ∨ b.red = b.yellow + 1)

/-- The maximum number of tulips in the bouquet -/
def maxTulips : ℕ := 15

/-- The theorem stating that 15 is the maximum number of tulips -/
theorem max_tulips_is_15 (c : TulipCosts) 
    (h1 : c.yellow = 50) 
    (h2 : c.red = 31) : 
    (∀ b : Bouquet, isValidBouquet b → totalCost b c ≤ 600 → b.yellow + b.red ≤ maxTulips) ∧
    (∃ b : Bouquet, isValidBouquet b ∧ totalCost b c ≤ 600 ∧ b.yellow + b.red = maxTulips) :=
  sorry

end NUMINAMATH_CALUDE_max_tulips_is_15_l569_56920


namespace NUMINAMATH_CALUDE_inequality_implication_l569_56985

theorem inequality_implication (a b c : ℝ) (h1 : a / c^2 > b / c^2) (h2 : c ≠ 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l569_56985


namespace NUMINAMATH_CALUDE_max_condition_l569_56969

/-- Given a function f with derivative f' and a parameter a, 
    proves that if f has a maximum at x = a and a < 0, then -1 < a < 0 -/
theorem max_condition (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, HasDerivAt f (a * (x + 1) * (x - a)) x) →
  a < 0 →
  (∀ x, f x ≤ f a) →
  -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_max_condition_l569_56969


namespace NUMINAMATH_CALUDE_alice_initial_cookies_count_l569_56930

/-- The number of chocolate chip cookies Alice initially baked -/
def alices_initial_cookies : ℕ := 91

/-- The number of peanut butter cookies Bob initially baked -/
def bobs_initial_cookies : ℕ := 7

/-- The number of cookies thrown on the floor -/
def thrown_cookies : ℕ := 29

/-- The number of additional cookies Alice baked after the accident -/
def alices_additional_cookies : ℕ := 5

/-- The number of additional cookies Bob baked after the accident -/
def bobs_additional_cookies : ℕ := 36

/-- The total number of edible cookies at the end -/
def total_edible_cookies : ℕ := 93

theorem alice_initial_cookies_count :
  alices_initial_cookies = 91 :=
by
  sorry

#check alice_initial_cookies_count

end NUMINAMATH_CALUDE_alice_initial_cookies_count_l569_56930


namespace NUMINAMATH_CALUDE_P_zero_for_floor_l569_56996

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The polynomial P(x,y) -/
def P (x y : ℤ) : ℤ :=
  (y - 2*x) * (y - 2*x - 1)

/-- Theorem stating that P(⌊a⌋, ⌊2a⌋) = 0 for all real numbers a -/
theorem P_zero_for_floor (a : ℝ) : P (floor a) (floor (2*a)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_P_zero_for_floor_l569_56996


namespace NUMINAMATH_CALUDE_notebook_cost_l569_56918

theorem notebook_cost (notebook_cost pen_cost : ℝ) 
  (total_cost : notebook_cost + pen_cost = 4.50)
  (cost_difference : notebook_cost = pen_cost + 3) : 
  notebook_cost = 3.75 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l569_56918


namespace NUMINAMATH_CALUDE_like_terms_sum_exponents_l569_56982

/-- Two monomials are like terms if they have the same variables raised to the same powers. -/
def are_like_terms (a b : ℕ) (m n : ℤ) : Prop :=
  m + 1 = 1 ∧ 3 = n

/-- If 5x^(m+1)y^3 and -3xy^n are like terms, then m + n = 3. -/
theorem like_terms_sum_exponents (m n : ℤ) :
  are_like_terms 5 3 m n → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_exponents_l569_56982


namespace NUMINAMATH_CALUDE_die_roll_invariant_l569_56914

/-- Represents the faces of a tetrahedral die -/
inductive DieFace
  | one
  | two
  | three
  | four

/-- Represents a position in the triangular grid -/
structure GridPosition where
  x : ℕ
  y : ℕ

/-- Represents the state of the die on the grid -/
structure DieState where
  position : GridPosition
  faceDown : DieFace

/-- Represents a single roll of the die -/
inductive DieRoll
  | rollLeft
  | rollRight
  | rollUp
  | rollDown

/-- Defines the starting corner of the grid -/
def startCorner : GridPosition :=
  { x := 0, y := 0 }

/-- Defines the opposite corner of the grid -/
def endCorner : GridPosition :=
  { x := 1, y := 1 }  -- Simplified for demonstration; actual values depend on grid size

/-- Function to perform a single roll -/
def performRoll (state : DieState) (roll : DieRoll) : DieState :=
  sorry  -- Implementation details omitted

/-- Theorem stating that regardless of the path taken, the die will end with face 1 down -/
theorem die_roll_invariant (path : List DieRoll) :
  let initialState : DieState := { position := startCorner, faceDown := DieFace.four }
  let finalState := path.foldl performRoll initialState
  finalState.position = endCorner → finalState.faceDown = DieFace.one :=
by sorry

end NUMINAMATH_CALUDE_die_roll_invariant_l569_56914


namespace NUMINAMATH_CALUDE_partial_sum_base_7_l569_56944

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem partial_sum_base_7 :
  let a := [2, 3, 4, 5, 1]
  let b := [1, 5, 6, 4, 2]
  let sum := [4, 2, 4, 2, 3]
  let base := 7
  (to_decimal a base + to_decimal b base = to_decimal sum base) ∧
  (∀ d ∈ (a ++ b ++ sum), d < base) :=
by sorry

end NUMINAMATH_CALUDE_partial_sum_base_7_l569_56944


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l569_56967

theorem fractional_equation_positive_root (m : ℝ) : 
  (∃ x : ℝ, x > 2 ∧ (3 / (x - 2) + (x + m) / (2 - x) = 1)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l569_56967


namespace NUMINAMATH_CALUDE_smallest_school_size_l569_56902

theorem smallest_school_size : ∃ n : ℕ, n > 0 ∧ n % 4 = 0 ∧ (n / 4) % 10 = 0 ∧
  (∃ y z : ℕ, y > 0 ∧ z > 0 ∧ 2 * y = 3 * z ∧ y + z - (n / 40) = n / 4) ∧
  ∀ m : ℕ, m > 0 → m % 4 = 0 → (m / 4) % 10 = 0 →
    (∃ y z : ℕ, y > 0 ∧ z > 0 ∧ 2 * y = 3 * z ∧ y + z - (m / 40) = m / 4) →
    m ≥ 200 :=
by sorry

end NUMINAMATH_CALUDE_smallest_school_size_l569_56902


namespace NUMINAMATH_CALUDE_triangle_theorem_l569_56987

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.c * Real.sin t.C = (2 * t.b + t.a) * Real.sin t.B + (2 * t.a - 3 * t.b) * Real.sin t.A) :
  t.C = π / 3 ∧ (t.c = 4 → 4 < t.a + t.b ∧ t.a + t.b ≤ 8) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l569_56987


namespace NUMINAMATH_CALUDE_n_pointed_star_interior_angle_sum_l569_56947

/-- An n-pointed star where n is a multiple of 3 and n ≥ 6 -/
structure NPointedStar where
  n : ℕ
  n_multiple_of_3 : 3 ∣ n
  n_ge_6 : n ≥ 6

/-- The sum of interior angles of an n-pointed star -/
def interior_angle_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 4)

/-- Theorem: The sum of interior angles of an n-pointed star is 180° (n-4) -/
theorem n_pointed_star_interior_angle_sum (star : NPointedStar) :
  interior_angle_sum star = 180 * (star.n - 4) := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_interior_angle_sum_l569_56947


namespace NUMINAMATH_CALUDE_complex_cube_root_l569_56972

theorem complex_cube_root (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑a + ↑b * Complex.I) ^ 3 = (2 : ℂ) + 11 * Complex.I →
  ↑a + ↑b * Complex.I = (2 : ℂ) + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l569_56972


namespace NUMINAMATH_CALUDE_bus_trip_distance_l569_56938

/-- Given a bus trip with specific conditions, prove that the distance traveled is 210 miles. -/
theorem bus_trip_distance (actual_speed : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) 
  (h1 : actual_speed = 30)
  (h2 : speed_increase = 5)
  (h3 : time_reduction = 1)
  (h4 : ∀ (distance : ℝ), distance / actual_speed = distance / (actual_speed + speed_increase) + time_reduction) :
  ∃ (distance : ℝ), distance = 210 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l569_56938


namespace NUMINAMATH_CALUDE_rain_received_calculation_l569_56929

/-- The number of days in a year -/
def daysInYear : ℕ := 365

/-- The normal average daily rainfall in inches -/
def normalDailyRainfall : ℚ := 2

/-- The number of days left in the year -/
def daysLeft : ℕ := 100

/-- The required average daily rainfall for the remaining days, in inches -/
def requiredDailyRainfall : ℚ := 3

/-- The amount of rain received so far this year, in inches -/
def rainReceivedSoFar : ℚ := 430

theorem rain_received_calculation :
  rainReceivedSoFar = 
    normalDailyRainfall * daysInYear - requiredDailyRainfall * daysLeft :=
by sorry

end NUMINAMATH_CALUDE_rain_received_calculation_l569_56929


namespace NUMINAMATH_CALUDE_telephone_probability_l569_56989

theorem telephone_probability (p1 p2 : ℝ) (h1 : p1 = 0.2) (h2 : p2 = 0.3) :
  p1 + p2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_telephone_probability_l569_56989


namespace NUMINAMATH_CALUDE_unique_solution_equation_l569_56937

/-- There exists exactly one ordered pair of real numbers (x, y) satisfying the given equation -/
theorem unique_solution_equation :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = Real.sqrt 2 ∧ x = -1/2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l569_56937


namespace NUMINAMATH_CALUDE_confidence_95_error_5_l569_56965

/-- Represents the confidence level as a real number between 0 and 1 -/
def ConfidenceLevel : Type := {r : ℝ // 0 < r ∧ r < 1}

/-- Represents the probability of making an incorrect inference -/
def ErrorProbability : Type := {r : ℝ // 0 ≤ r ∧ r ≤ 1}

/-- Given a confidence level, calculates the probability of making an incorrect inference -/
def calculateErrorProbability (cl : ConfidenceLevel) : ErrorProbability :=
  sorry

/-- The theorem states that for a 95% confidence level, the error probability is 5% -/
theorem confidence_95_error_5 :
  let cl95 : ConfidenceLevel := ⟨0.95, by sorry⟩
  calculateErrorProbability cl95 = ⟨0.05, by sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_confidence_95_error_5_l569_56965


namespace NUMINAMATH_CALUDE_area_covered_by_strips_l569_56968

/-- The area covered by four overlapping rectangular strips on a table -/
def area_covered (length width : ℝ) : ℝ :=
  4 * length * width - 4 * width * width

/-- Theorem stating that the area covered by four overlapping rectangular strips,
    each 16 cm long and 2 cm wide, is 112 cm² -/
theorem area_covered_by_strips :
  area_covered 16 2 = 112 := by sorry

end NUMINAMATH_CALUDE_area_covered_by_strips_l569_56968


namespace NUMINAMATH_CALUDE_sum_of_factors_l569_56924

theorem sum_of_factors (p q r s t : ℝ) : 
  (∀ y : ℝ, 512 * y^3 + 27 = (p * y + q) * (r * y^2 + s * y + t)) →
  p + q + r + s + t = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l569_56924


namespace NUMINAMATH_CALUDE_ab2_minus_41_equals_591_l569_56998

/-- Given two single-digit numbers A and B, where AB2 is a three-digit number,
    prove that when A = 6 and B = 2, the equation AB2 - 41 = 591 is valid. -/
theorem ab2_minus_41_equals_591 (A B : Nat) : 
  A < 10 → B < 10 → 100 ≤ A * 100 + B * 10 + 2 → A * 100 + B * 10 + 2 < 1000 →
  A = 6 → B = 2 → A * 100 + B * 10 + 2 - 41 = 591 := by
sorry

end NUMINAMATH_CALUDE_ab2_minus_41_equals_591_l569_56998


namespace NUMINAMATH_CALUDE_linear_equation_solution_l569_56901

theorem linear_equation_solution : 
  ∃ x : ℝ, (2 / 3 : ℝ) * x - 2 = 4 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l569_56901


namespace NUMINAMATH_CALUDE_worker_production_theorem_l569_56934

/-- Represents the production of two workers before and after a productivity increase -/
structure WorkerProduction where
  initial_total : ℕ
  increase1 : ℚ
  increase2 : ℚ
  final_total : ℕ

/-- Calculates the individual production of two workers after a productivity increase -/
def calculate_production (w : WorkerProduction) : ℕ × ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the workers produce 46 and 40 parts after the increase -/
theorem worker_production_theorem (w : WorkerProduction) 
  (h1 : w.initial_total = 72)
  (h2 : w.increase1 = 15 / 100)
  (h3 : w.increase2 = 25 / 100)
  (h4 : w.final_total = 86) :
  calculate_production w = (46, 40) :=
sorry

end NUMINAMATH_CALUDE_worker_production_theorem_l569_56934


namespace NUMINAMATH_CALUDE_other_soap_bubble_ratio_l569_56977

/- Define the number of bubbles Dawn can make per ounce -/
def dawn_bubbles_per_oz : ℕ := 200000

/- Define the number of bubbles made by half ounce of mixed soap -/
def mixed_bubbles_half_oz : ℕ := 150000

/- Define the ratio of bubbles made by the other soap to Dawn soap -/
def other_soap_ratio : ℚ := 1 / 2

/- Theorem statement -/
theorem other_soap_bubble_ratio :
  ∀ (other_bubbles_per_oz : ℕ),
    2 * mixed_bubbles_half_oz = dawn_bubbles_per_oz + other_bubbles_per_oz →
    other_bubbles_per_oz / dawn_bubbles_per_oz = other_soap_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_other_soap_bubble_ratio_l569_56977


namespace NUMINAMATH_CALUDE_speed_time_relationship_l569_56983

-- Define the initial speed and time
variable (x y : ℝ)

-- Define the percentage increases/decreases
variable (a b : ℝ)

-- Condition: x and y are positive (speed and time can't be negative or zero)
variable (hx : x > 0)
variable (hy : y > 0)

-- Condition: a and b are percentages (between 0 and 100)
variable (ha : 0 ≤ a ∧ a ≤ 100)
variable (hb : 0 ≤ b ∧ b ≤ 100)

-- Theorem stating the relationship between a and b
theorem speed_time_relationship : 
  x * y = x * (1 + a / 100) * (y * (1 - b / 100)) → 
  b = (100 * a) / (100 + a) := by
sorry

end NUMINAMATH_CALUDE_speed_time_relationship_l569_56983


namespace NUMINAMATH_CALUDE_workshop_A_more_stable_l569_56945

def workshop_A : List ℕ := [102, 101, 99, 98, 103, 98, 99]
def workshop_B : List ℕ := [110, 115, 90, 85, 75, 115, 110]

def variance (data : List ℕ) : ℚ :=
  let mean := (data.sum : ℚ) / data.length
  (data.map (fun x => ((x : ℚ) - mean) ^ 2)).sum / data.length

theorem workshop_A_more_stable :
  variance workshop_A < variance workshop_B :=
sorry

end NUMINAMATH_CALUDE_workshop_A_more_stable_l569_56945


namespace NUMINAMATH_CALUDE_third_segment_less_than_quarter_l569_56921

open Real

/-- Given a triangle ABC with angles A, B, C, and side lengths a, b, c, 
    where angle B is divided into four equal parts, prove that the third segment 
    on AC (counting from A) is less than |AC| / 4 -/
theorem third_segment_less_than_quarter (A B C : ℝ) (a b c : ℝ) : 
  A > 0 → B > 0 → C > 0 → 
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  3 * A - C < π →
  ∃ (K L M : ℝ), 0 < K ∧ K < L ∧ L < M ∧ M < b ∧
    (L - K = M - L) ∧ (M - L = b - M) ∧
    (L - K < b / 4) :=
by sorry

end NUMINAMATH_CALUDE_third_segment_less_than_quarter_l569_56921


namespace NUMINAMATH_CALUDE_min_value_of_p_l569_56943

-- Define the polynomial p
def p (a b : ℝ) : ℝ := a^2 + 2*b^2 + 2*a + 4*b + 2008

-- Theorem stating the minimum value of p
theorem min_value_of_p :
  ∃ (min : ℝ), min = 2005 ∧ ∀ (a b : ℝ), p a b ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_p_l569_56943


namespace NUMINAMATH_CALUDE_line_divides_polygon_equally_l569_56907

/-- Polygon type representing a closed shape with vertices --/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Line type representing a line in slope-intercept form --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculate the area of a polygon using the shoelace formula --/
def polygonArea (p : Polygon) : ℝ := sorry

/-- Check if a point lies on a line --/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop := sorry

/-- Check if a line divides a polygon into two equal areas --/
def dividesEqualArea (l : Line) (p : Polygon) : Prop := sorry

/-- The main theorem --/
theorem line_divides_polygon_equally (polygon : Polygon) (line : Line) :
  polygon.vertices = [(0, 0), (0, 6), (4, 6), (4, 4), (6, 4), (6, 0)] →
  line.slope = -1/3 →
  line.intercept = 11/3 →
  pointOnLine line (2, 3) →
  dividesEqualArea line polygon := by
  sorry

end NUMINAMATH_CALUDE_line_divides_polygon_equally_l569_56907


namespace NUMINAMATH_CALUDE_total_goals_is_fifteen_l569_56913

/-- The total number of goals scored in a soccer match -/
def total_goals (kickers_first : ℕ) : ℕ :=
  let kickers_second := 2 * kickers_first
  let spiders_first := kickers_first / 2
  let spiders_second := 2 * kickers_second
  kickers_first + kickers_second + spiders_first + spiders_second

/-- Theorem stating that the total goals scored is 15 when The Kickers score 2 goals in the first period -/
theorem total_goals_is_fifteen : total_goals 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_goals_is_fifteen_l569_56913


namespace NUMINAMATH_CALUDE_M_equals_six_eight_C_U_A_inter_C_U_B_equals_five_nine_ten_l569_56923

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6, 8}

-- Define set B
def B : Set ℕ := {x ∈ U | x < 5}

-- Define set M
def M : Set ℕ := {x ∈ A | x ∉ B}

-- Define the complement of A in U
def C_U_A : Set ℕ := U \ A

-- Define the complement of B in U
def C_U_B : Set ℕ := U \ B

-- Theorem for part (1)
theorem M_equals_six_eight : M = {6, 8} := by sorry

-- Theorem for part (2)
theorem C_U_A_inter_C_U_B_equals_five_nine_ten : C_U_A ∩ C_U_B = {5, 9, 10} := by sorry

end NUMINAMATH_CALUDE_M_equals_six_eight_C_U_A_inter_C_U_B_equals_five_nine_ten_l569_56923


namespace NUMINAMATH_CALUDE_pauls_erasers_l569_56986

/-- The number of erasers Paul got for his birthday -/
def erasers : ℕ := 0  -- We'll prove this is actually 457

/-- The number of crayons Paul got for his birthday -/
def initial_crayons : ℕ := 617

/-- The number of crayons Paul had left at the end of the school year -/
def remaining_crayons : ℕ := 523

/-- The difference between the number of crayons and erasers left -/
def crayon_eraser_difference : ℕ := 66

theorem pauls_erasers : 
  erasers = 457 ∧ 
  initial_crayons = 617 ∧
  remaining_crayons = 523 ∧
  crayon_eraser_difference = 66 ∧
  remaining_crayons = erasers + crayon_eraser_difference :=
sorry

end NUMINAMATH_CALUDE_pauls_erasers_l569_56986


namespace NUMINAMATH_CALUDE_sum_of_fractions_l569_56971

theorem sum_of_fractions : 
  (1/10 : ℚ) + (2/10 : ℚ) + (3/10 : ℚ) + (4/10 : ℚ) + (5/10 : ℚ) + 
  (6/10 : ℚ) + (7/10 : ℚ) + (8/10 : ℚ) + (9/10 : ℚ) + (90/10 : ℚ) = 
  (27/2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l569_56971


namespace NUMINAMATH_CALUDE_p_subset_q_condition_l569_56928

def P : Set ℝ := {x : ℝ | |x + 2| ≤ 3}
def Q : Set ℝ := {x : ℝ | x ≥ -8}

theorem p_subset_q_condition : P ⊂ Q ∧ 
  (∀ x : ℝ, x ∈ P → x ∈ Q) ∧ 
  (∃ x : ℝ, x ∈ Q ∧ x ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_p_subset_q_condition_l569_56928


namespace NUMINAMATH_CALUDE_ratio_equality_l569_56963

theorem ratio_equality (n m : ℚ) (h1 : 3 * n = 4 * m) (h2 : m ≠ 0) (h3 : n ≠ 0) : n / m = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l569_56963


namespace NUMINAMATH_CALUDE_modulus_of_complex_power_l569_56908

theorem modulus_of_complex_power : 
  Complex.abs ((2 - 3 * Complex.I * Real.sqrt 3) ^ 4) = 961 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_power_l569_56908


namespace NUMINAMATH_CALUDE_uniform_color_subgrid_l569_56961

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A point in the grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A function that assigns a color to each point in the grid -/
def ColoringFunction := GridPoint → Color

/-- A theorem stating that in any two-color infinite grid, there exist two horizontal
    and two vertical lines forming a subgrid with uniformly colored intersection points -/
theorem uniform_color_subgrid
  (coloring : ColoringFunction) :
  ∃ (x₁ x₂ y₁ y₂ : ℤ) (c : Color),
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    coloring ⟨x₁, y₁⟩ = c ∧
    coloring ⟨x₁, y₂⟩ = c ∧
    coloring ⟨x₂, y₁⟩ = c ∧
    coloring ⟨x₂, y₂⟩ = c :=
by sorry


end NUMINAMATH_CALUDE_uniform_color_subgrid_l569_56961


namespace NUMINAMATH_CALUDE_M_union_N_eq_l569_56966

def M : Set ℤ := {x | |x| < 2}
def N : Set ℤ := {-2, -1, 0}

theorem M_union_N_eq : M ∪ N = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_M_union_N_eq_l569_56966


namespace NUMINAMATH_CALUDE_tea_blend_cost_l569_56910

theorem tea_blend_cost (blend_ratio : ℚ) (second_tea_cost : ℚ) (blend_sell_price : ℚ) (gain_percent : ℚ) :
  blend_ratio = 5 / 3 →
  second_tea_cost = 20 →
  blend_sell_price = 21 →
  gain_percent = 12 →
  ∃ first_tea_cost : ℚ,
    first_tea_cost = 18 ∧
    (1 + gain_percent / 100) * ((blend_ratio * first_tea_cost + second_tea_cost) / (blend_ratio + 1)) = blend_sell_price :=
by sorry

end NUMINAMATH_CALUDE_tea_blend_cost_l569_56910


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l569_56993

theorem complex_fraction_sum (a b : ℂ) (h1 : a = 5 + 7*I) (h2 : b = 5 - 7*I) : 
  a / b + b / a = -23 / 37 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l569_56993


namespace NUMINAMATH_CALUDE_rabbits_distance_specific_rabbits_distance_l569_56990

/-- The distance between two rabbits' homes given their resting patterns --/
theorem rabbits_distance (white_rest_interval : ℕ) (gray_rest_interval : ℕ) 
  (rest_difference : ℕ) : ℕ :=
  let meeting_point := white_rest_interval * gray_rest_interval * rest_difference / 
    (white_rest_interval - gray_rest_interval)
  2 * meeting_point

/-- Proof of the specific rabbit problem --/
theorem specific_rabbits_distance : 
  rabbits_distance 30 20 15 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_distance_specific_rabbits_distance_l569_56990


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l569_56970

theorem arithmetic_geometric_sequence (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- distinct real numbers
  2 * b = a + c →  -- arithmetic sequence
  (c * a) * (b * c) = (a * b) * (a * b) →  -- geometric sequence
  a + b + c = 15 →
  a = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l569_56970


namespace NUMINAMATH_CALUDE_student_mistake_difference_l569_56957

theorem student_mistake_difference (n : ℚ) (h : n = 480) : 5/6 * n - 5/16 * n = 250 := by
  sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l569_56957


namespace NUMINAMATH_CALUDE_prob_good_friends_is_one_fourth_l569_56903

/-- The number of balls in the pocket -/
def num_balls : ℕ := 4

/-- The set of possible ball numbers -/
def ball_numbers : Finset ℕ := Finset.range num_balls

/-- The probability space of drawing two balls with replacement -/
def draw_space : Finset (ℕ × ℕ) := ball_numbers.product ball_numbers

/-- The event of drawing the same number (becoming "good friends") -/
def good_friends : Finset (ℕ × ℕ) := 
  draw_space.filter (fun p => p.1 = p.2)

/-- The probability of becoming "good friends" -/
def prob_good_friends : ℚ :=
  good_friends.card / draw_space.card

theorem prob_good_friends_is_one_fourth : 
  prob_good_friends = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_good_friends_is_one_fourth_l569_56903


namespace NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l569_56956

/-- The decimal representation of 5/17 has a 6-digit repetend equal to 294117 -/
theorem repetend_of_five_seventeenths :
  ∃ (a b : ℕ), (5 : ℚ) / 17 = (a : ℚ) + (b : ℚ) / 999999 ∧ b = 294117 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l569_56956


namespace NUMINAMATH_CALUDE_tree_branches_count_l569_56997

/-- Proves that a tree with the given characteristics has 30 branches -/
theorem tree_branches_count : 
  ∀ (total_leaves : ℕ) (twigs_per_branch : ℕ) 
    (four_leaf_twig_percent : ℚ) (five_leaf_twig_percent : ℚ),
  total_leaves = 12690 →
  twigs_per_branch = 90 →
  four_leaf_twig_percent = 30 / 100 →
  five_leaf_twig_percent = 70 / 100 →
  four_leaf_twig_percent + five_leaf_twig_percent = 1 →
  ∃ (branches : ℕ),
    branches * (four_leaf_twig_percent * twigs_per_branch * 4 + 
                five_leaf_twig_percent * twigs_per_branch * 5) = total_leaves ∧
    branches = 30 := by
  sorry

end NUMINAMATH_CALUDE_tree_branches_count_l569_56997


namespace NUMINAMATH_CALUDE_exam_scores_sum_l569_56905

theorem exam_scores_sum (scores : List ℝ) :
  scores.length = 6 ∧
  65 ∈ scores ∧ 75 ∈ scores ∧ 85 ∈ scores ∧ 95 ∈ scores ∧
  scores.sum / scores.length = 80 →
  ∃ x y, x ∈ scores ∧ y ∈ scores ∧ x + y = 160 :=
by sorry

end NUMINAMATH_CALUDE_exam_scores_sum_l569_56905


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l569_56904

theorem functional_equation_solutions (f : ℤ → ℤ) :
  (∀ a b c : ℤ, a + b + c = 0 →
    f a ^ 2 + f b ^ 2 + f c ^ 2 = 2 * f a * f b + 2 * f b * f c + 2 * f c * f a) →
  (∀ x, f x = 0) ∨
  (∃ k : ℤ, ∀ x, f x = if x % 2 = 0 then 0 else k) ∨
  (∃ k : ℤ, ∀ x, f x = if x % 4 = 0 then 0 else if x % 4 = 2 then k else k) ∨
  (∃ k : ℤ, ∀ x, f x = k * x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l569_56904


namespace NUMINAMATH_CALUDE_zeros_of_f_product_inequality_l569_56936

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x

def g (x : ℝ) : ℝ := (1/3) * x^3 + x + 1

theorem zeros_of_f_product_inequality (x₁ x₂ : ℝ) 
  (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) (h₃ : x₁ ≠ x₂) :
  g (x₁ * x₂) > g (Real.exp 2) :=
sorry

end

end NUMINAMATH_CALUDE_zeros_of_f_product_inequality_l569_56936


namespace NUMINAMATH_CALUDE_algebraic_identities_l569_56942

theorem algebraic_identities (x y : ℝ) : 
  (3 * x^2 * y * (-2 * x * y)^3 = -24 * x^5 * y^4) ∧ 
  ((5 * x + 2 * y) * (3 * x - 2 * y) = 15 * x^2 - 4 * x * y - 4 * y^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l569_56942


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_364_l569_56975

/-- Triangle with positive integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ+
  base : ℕ+

/-- Angle bisector intersection point -/
structure AngleBisectorIntersection (t : IsoscelesTriangle) where
  distance_to_vertex : ℕ+

/-- The smallest possible perimeter of an isosceles triangle with given angle bisector intersection -/
def smallest_perimeter (t : IsoscelesTriangle) (j : AngleBisectorIntersection t) : ℕ :=
  2 * (t.side.val + t.base.val)

/-- Theorem stating the smallest possible perimeter of the triangle -/
theorem smallest_perimeter_is_364 :
  ∃ (t : IsoscelesTriangle) (j : AngleBisectorIntersection t),
    j.distance_to_vertex = 10 ∧
    (∀ (t' : IsoscelesTriangle) (j' : AngleBisectorIntersection t'),
      j'.distance_to_vertex = 10 →
      smallest_perimeter t j ≤ smallest_perimeter t' j') ∧
    smallest_perimeter t j = 364 :=
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_364_l569_56975


namespace NUMINAMATH_CALUDE_george_monthly_income_l569_56916

def monthly_income : ℝ := 240

theorem george_monthly_income :
  let half_income := monthly_income / 2
  let remaining_after_groceries := half_income - 20
  remaining_after_groceries = 100 → monthly_income = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_george_monthly_income_l569_56916


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l569_56915

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- Number of cubic yards we're converting -/
def cubic_yards : ℝ := 4

/-- Theorem stating that 4 cubic yards equals 108 cubic feet -/
theorem cubic_yards_to_cubic_feet : 
  cubic_yards * (yards_to_feet ^ 3) = 108 := by sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l569_56915


namespace NUMINAMATH_CALUDE_candy_average_l569_56995

theorem candy_average (eunji_candies : ℕ) (jimin_diff : ℕ) (jihyun_diff : ℕ) : 
  eunji_candies = 35 →
  jimin_diff = 6 →
  jihyun_diff = 3 →
  (eunji_candies + (eunji_candies + jimin_diff) + (eunji_candies - jihyun_diff)) / 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_candy_average_l569_56995


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l569_56978

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define symmetry with respect to x-axis
def symmetricToXAxis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_correct :
  let P : Point := (-2, 3)
  symmetricToXAxis P = (-2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l569_56978


namespace NUMINAMATH_CALUDE_complement_of_union_equals_four_l569_56991

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_equals_four :
  (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_four_l569_56991


namespace NUMINAMATH_CALUDE_range_of_a_l569_56981

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4*x + a ≤ 0

theorem range_of_a (a : ℝ) (hp : prop_p a) (hq : prop_q a) :
  Real.exp 1 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l569_56981


namespace NUMINAMATH_CALUDE_mark_soup_donation_l569_56962

/-- The number of homeless shelters -/
def num_shelters : ℕ := 6

/-- The number of people served by each shelter -/
def people_per_shelter : ℕ := 30

/-- The number of cans of soup bought per person -/
def cans_per_person : ℕ := 10

/-- The total number of cans of soup Mark donates -/
def total_cans : ℕ := num_shelters * people_per_shelter * cans_per_person

theorem mark_soup_donation : total_cans = 1800 := by
  sorry

end NUMINAMATH_CALUDE_mark_soup_donation_l569_56962


namespace NUMINAMATH_CALUDE_ferry_problem_l569_56949

/-- Ferry problem -/
theorem ferry_problem (v_p v_q : ℝ) (d_p d_q : ℝ) (t_p t_q : ℝ) :
  v_p = 8 →
  d_q = 3 * d_p →
  v_q = v_p + 1 →
  t_q = t_p + 5 →
  d_p = v_p * t_p →
  d_q = v_q * t_q →
  t_p = 3 := by
  sorry

#check ferry_problem

end NUMINAMATH_CALUDE_ferry_problem_l569_56949


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l569_56992

/-- Given that x and y are inversely proportional, prove that y = -27 when x = -9,
    given that x = 3y when x + y = 36. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
    (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 36 ∧ x₀ = 3 * y₀ ∧ x₀ * y₀ = k) : 
    x = -9 → y = -27 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l569_56992


namespace NUMINAMATH_CALUDE_adam_basswood_blocks_l569_56988

/-- The number of figurines that can be created from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be created from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be created from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of butternut wood blocks Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 245

/-- Theorem stating that Adam owns 15 blocks of basswood -/
theorem adam_basswood_blocks : 
  ∃ (basswood_blocks : ℕ), 
    basswood_blocks * basswood_figurines + 
    butternut_blocks * butternut_figurines + 
    aspen_blocks * aspen_figurines = total_figurines ∧ 
    basswood_blocks = 15 := by
  sorry

end NUMINAMATH_CALUDE_adam_basswood_blocks_l569_56988


namespace NUMINAMATH_CALUDE_container_volume_ratio_l569_56927

theorem container_volume_ratio (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : 2/3 * A = 5/8 * B) : A / B = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l569_56927


namespace NUMINAMATH_CALUDE_infinitely_many_primes_of_year_2022_l569_56964

/-- A prime p is a prime of the year 2022 if there exists a positive integer n 
    such that p^2022 divides n^2022 + 2022 -/
def IsPrimeOfYear2022 (p : Nat) : Prop :=
  Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ (p^2022 ∣ n^2022 + 2022)

/-- There are infinitely many primes of the year 2022 -/
theorem infinitely_many_primes_of_year_2022 :
  ∀ N : Nat, ∃ p : Nat, p > N ∧ IsPrimeOfYear2022 p := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_of_year_2022_l569_56964


namespace NUMINAMATH_CALUDE_find_other_number_l569_56926

theorem find_other_number (a b : ℤ) : 
  3 * a + 2 * b = 105 → (a = 15 ∨ b = 15) → (a = 30 ∨ b = 30) := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l569_56926


namespace NUMINAMATH_CALUDE_line_circle_intersection_l569_56955

/-- The line y = kx + 1 always intersects with the circle x^2 + y^2 - 2ax + a^2 - 2a - 4 = 0 
    for any real k if and only if -1 ≤ a ≤ 3 -/
theorem line_circle_intersection (a : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 + y^2 - 2*a*x + a^2 - 2*a - 4 = 0) ↔ 
  -1 ≤ a ∧ a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l569_56955


namespace NUMINAMATH_CALUDE_smaller_circle_area_l569_56948

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  -- Centers of the smaller and larger circles
  S : ℝ × ℝ
  L : ℝ × ℝ
  -- Radii of the smaller and larger circles
  r_small : ℝ
  r_large : ℝ
  -- Point P from which tangents are drawn
  P : ℝ × ℝ
  -- Points of tangency on the circles
  A : ℝ × ℝ
  B : ℝ × ℝ
  -- The circles are externally tangent
  externally_tangent : dist S L = r_small + r_large
  -- PAB is a common tangent
  tangent_line : dist P A = dist A B
  -- A is on the smaller circle, B is on the larger circle
  on_circles : dist S A = r_small ∧ dist L B = r_large
  -- Length condition
  length_condition : dist P A = 4 ∧ dist A B = 4

/-- The area of the smaller circle in the TangentCircles configuration is 2π -/
theorem smaller_circle_area (tc : TangentCircles) : Real.pi * tc.r_small ^ 2 = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_smaller_circle_area_l569_56948


namespace NUMINAMATH_CALUDE_square_root_problem_l569_56922

theorem square_root_problem (a : ℝ) (n : ℝ) (hn : n > 0) :
  (2 * a - 3)^2 = n ∧ (3 * a - 22)^2 = n → n = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l569_56922


namespace NUMINAMATH_CALUDE_tangent_slope_point_M_l569_56917

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

theorem tangent_slope_point_M :
  ∀ x y : ℝ, f y = f x → f' x = -4 → x = -1 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_point_M_l569_56917


namespace NUMINAMATH_CALUDE_volleyball_handshakes_l569_56909

theorem volleyball_handshakes (total_handshakes : ℕ) (h : total_handshakes = 496) :
  ∃ (n : ℕ), 
    n * (n - 1) / 2 = total_handshakes ∧
    ∀ (coach_handshakes : ℕ), 
      n * (n - 1) / 2 + coach_handshakes = total_handshakes → 
      coach_handshakes ≥ 0 ∧
      (coach_handshakes = 0 → 
        ∀ (other_coach_handshakes : ℕ), 
          n * (n - 1) / 2 + other_coach_handshakes = total_handshakes → 
          other_coach_handshakes ≥ coach_handshakes) :=
by sorry

end NUMINAMATH_CALUDE_volleyball_handshakes_l569_56909
