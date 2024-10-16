import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_sequence_existence_l1089_108965

theorem quadratic_sequence_existence (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧
  ∀ i : ℕ, i ≤ n → i ≠ 0 → |a i - a (i - 1)| = i^2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_existence_l1089_108965


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1089_108917

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 1)^5 = a₅*(x + 1)^5 + a₄*(x + 1)^4 + a₃*(x + 1)^3 + a₂*(x + 1)^2 + a₁*(x + 1) + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1089_108917


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l1089_108993

/-- In a cyclic quadrilateral ABCD where ∠A : ∠B : ∠C = 1 : 2 : 3, ∠D = 90° -/
theorem cyclic_quadrilateral_angle (A B C D : Real) (h1 : A + C = 180) (h2 : B + D = 180)
  (h3 : A / B = 1 / 2) (h4 : B / C = 2 / 3) : D = 90 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l1089_108993


namespace NUMINAMATH_CALUDE_expression_evaluation_l1089_108928

theorem expression_evaluation : 
  let f (x : ℝ) := (x + 1) / (x - 1)
  let g (x : ℝ) := (f x + 1) / (f x - 1)
  g (1/2) = -3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1089_108928


namespace NUMINAMATH_CALUDE_christine_speed_l1089_108915

/-- Given a distance of 80 miles traveled in 4 hours, prove that the speed is 20 miles per hour. -/
theorem christine_speed (distance : ℝ) (time : ℝ) (h1 : distance = 80) (h2 : time = 4) :
  distance / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_christine_speed_l1089_108915


namespace NUMINAMATH_CALUDE_parabola_specific_point_l1089_108978

def parabola_point (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = (y + 2)^2

theorem parabola_specific_point :
  let x : ℝ := Real.sqrt 704
  let y : ℝ := 88
  parabola_point x y ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  Real.sqrt (x^2 + (y - 2)^2) = 90 := by sorry

end NUMINAMATH_CALUDE_parabola_specific_point_l1089_108978


namespace NUMINAMATH_CALUDE_flower_count_proof_l1089_108933

theorem flower_count_proof (total : ℕ) (red green blue yellow purple orange : ℕ) : 
  total = 180 →
  red = (30 * total) / 100 →
  green = (10 * total) / 100 →
  blue = green / 2 →
  yellow = red + 5 →
  3 * purple = 7 * orange →
  red + green + blue + yellow + purple + orange = total →
  red = 54 ∧ green = 18 ∧ blue = 9 ∧ yellow = 59 ∧ purple = 12 ∧ orange = 28 :=
by sorry

end NUMINAMATH_CALUDE_flower_count_proof_l1089_108933


namespace NUMINAMATH_CALUDE_explicit_formula_l1089_108938

noncomputable section

variable (f : ℝ → ℝ)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = (deriv f 1) * Real.exp (x - 1) - (f 0) * x + (1/2) * x^2

theorem explicit_formula (h : satisfies_condition f) :
  ∀ x, f x = Real.exp x - x + (1/2) * x^2 := by
  sorry

end

end NUMINAMATH_CALUDE_explicit_formula_l1089_108938


namespace NUMINAMATH_CALUDE_smallest_S_for_equal_probability_l1089_108947

-- Define the number of sides on a standard die
def standardDieSides : ℕ := 6

-- Define the target sum
def targetSum : ℕ := 2000

-- Define the function to calculate the minimum number of dice needed to reach the target sum
def minDiceNeeded (target : ℕ) (sides : ℕ) : ℕ :=
  (target + sides - 1) / sides

-- Define the function to calculate S given n dice
def calculateS (n : ℕ) (target : ℕ) : ℕ :=
  7 * n - target

-- Theorem statement
theorem smallest_S_for_equal_probability :
  let n := minDiceNeeded targetSum standardDieSides
  calculateS n targetSum = 338 := by sorry

end NUMINAMATH_CALUDE_smallest_S_for_equal_probability_l1089_108947


namespace NUMINAMATH_CALUDE_mike_owes_jennifer_l1089_108980

theorem mike_owes_jennifer (payment_per_room : ℚ) (rooms_cleaned : ℚ) 
  (h1 : payment_per_room = 13 / 3)
  (h2 : rooms_cleaned = 8 / 5) :
  payment_per_room * rooms_cleaned = 104 / 15 := by
sorry

end NUMINAMATH_CALUDE_mike_owes_jennifer_l1089_108980


namespace NUMINAMATH_CALUDE_special_ap_sums_l1089_108956

/-- An arithmetic progression with special properties -/
structure SpecialAP where
  m : ℕ
  n : ℕ
  sum_m_terms : ℕ
  sum_n_terms : ℕ
  h1 : sum_m_terms = n
  h2 : sum_n_terms = m

/-- The sum of (m+n) terms and (m-n) terms for a SpecialAP -/
def special_sums (ap : SpecialAP) : ℤ × ℚ :=
  (-(ap.m + ap.n : ℤ), (ap.m - ap.n : ℚ) * (2 * ap.n + ap.m) / ap.m)

/-- Theorem stating the sums of (m+n) and (m-n) terms for a SpecialAP -/
theorem special_ap_sums (ap : SpecialAP) :
  special_sums ap = (-(ap.m + ap.n : ℤ), (ap.m - ap.n : ℚ) * (2 * ap.n + ap.m) / ap.m) := by
  sorry

#check special_ap_sums

end NUMINAMATH_CALUDE_special_ap_sums_l1089_108956


namespace NUMINAMATH_CALUDE_triangle_problem_l1089_108985

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * Real.sin t.B = Real.sin t.A + Real.cos t.A * Real.tan t.C)
  (h2 : t.b = 4)
  (h3 : (t.a + t.b + t.c) / 2 * (Real.sqrt 3 / 2) = t.a * t.b * Real.sin t.C / 2) :
  t.C = Real.pi / 3 ∧ t.a - t.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1089_108985


namespace NUMINAMATH_CALUDE_tickets_sold_second_week_l1089_108911

/-- The number of tickets sold in the second week of a fair, given the total number of tickets,
    tickets sold in the first week, and tickets left to sell. -/
theorem tickets_sold_second_week
  (total_tickets : ℕ)
  (first_week_sales : ℕ)
  (tickets_left : ℕ)
  (h1 : total_tickets = 90)
  (h2 : first_week_sales = 38)
  (h3 : tickets_left = 35) :
  total_tickets - (first_week_sales + tickets_left) = 17 :=
by sorry

end NUMINAMATH_CALUDE_tickets_sold_second_week_l1089_108911


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1089_108902

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem linear_function_not_in_third_quadrant 
  (k b : ℝ) 
  (h : ∀ x y : ℝ, y = linear_function k b x → ¬in_third_quadrant x y) : 
  k < 0 ∧ b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1089_108902


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l1089_108988

theorem binomial_coefficient_equation_solution (x : ℕ) : 
  Nat.choose 11 x = Nat.choose 11 (2*x - 4) ↔ x = 4 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l1089_108988


namespace NUMINAMATH_CALUDE_jellybean_purchase_min_jellybean_purchase_l1089_108934

theorem jellybean_purchase (n : ℕ) : n ≥ 150 ∧ n % 17 = 15 → n ≥ 151 :=
by
  sorry

theorem min_jellybean_purchase : ∃ (n : ℕ), n ≥ 150 ∧ n % 17 = 15 ∧ ∀ (m : ℕ), m ≥ 150 ∧ m % 17 = 15 → m ≥ n :=
by
  sorry

#check jellybean_purchase
#check min_jellybean_purchase

end NUMINAMATH_CALUDE_jellybean_purchase_min_jellybean_purchase_l1089_108934


namespace NUMINAMATH_CALUDE_number_equation_solution_l1089_108981

theorem number_equation_solution : ∃! x : ℚ, x = (x - 5) * 4 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1089_108981


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1089_108969

theorem smallest_fraction_between (a b c d : ℕ) (h1 : a < b) (h2 : c < d) :
  ∃ (x y : ℕ), 
    (x : ℚ) / y > (a : ℚ) / b ∧ 
    (x : ℚ) / y < (c : ℚ) / d ∧ 
    (∀ (p q : ℕ), (p : ℚ) / q > (a : ℚ) / b ∧ (p : ℚ) / q < (c : ℚ) / d → y ≤ q) ∧
    x = 2 ∧ y = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1089_108969


namespace NUMINAMATH_CALUDE_beluga_breath_interval_proof_l1089_108962

/-- The average time (in minutes) between a bottle-nosed dolphin's air breaths -/
def dolphin_breath_interval : ℝ := 3

/-- The number of minutes in a 24-hour period -/
def minutes_per_day : ℝ := 24 * 60

/-- The ratio of dolphin breaths to beluga whale breaths in a 24-hour period -/
def breath_ratio : ℝ := 2.5

/-- The average time (in minutes) between a beluga whale's air breaths -/
def beluga_breath_interval : ℝ := 7.5

theorem beluga_breath_interval_proof :
  (minutes_per_day / dolphin_breath_interval) = breath_ratio * (minutes_per_day / beluga_breath_interval) :=
by sorry

end NUMINAMATH_CALUDE_beluga_breath_interval_proof_l1089_108962


namespace NUMINAMATH_CALUDE_graph_intersection_sum_l1089_108998

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop :=
  (x^2 + y^2 - 2*x)^2 = 2*(x^2 + y^2)^2

/-- The number of points where the graph meets the x-axis -/
def p : ℕ :=
  3  -- This is given as a fact from the problem, not derived from the solution

/-- The number of points where the graph meets the y-axis -/
def q : ℕ :=
  1  -- This is given as a fact from the problem, not derived from the solution

/-- The theorem to be proved -/
theorem graph_intersection_sum : 100 * p + 100 * q = 400 := by
  sorry

end NUMINAMATH_CALUDE_graph_intersection_sum_l1089_108998


namespace NUMINAMATH_CALUDE_parabola_equation_l1089_108955

/-- A parabola in the Cartesian coordinate system with focus at (-2, 0) has the standard equation y^2 = -8x -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ y^2 = -2*p*x ∧ p/2 = 2) → 
  y^2 = -8*x := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1089_108955


namespace NUMINAMATH_CALUDE_f_composition_proof_l1089_108900

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_proof : f (f (f (-1))) = Real.pi + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_proof_l1089_108900


namespace NUMINAMATH_CALUDE_product_absolute_value_one_l1089_108992

theorem product_absolute_value_one 
  (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
  (h1 : a + 1/b = b + 1/c)
  (h2 : b + 1/c = c + 1/d)
  (h3 : c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 := by
sorry

end NUMINAMATH_CALUDE_product_absolute_value_one_l1089_108992


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1089_108914

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (25^3 + 15^4 - 5^6 + 20^3) ∧ 
  ∀ (q : ℕ), q.Prime → q ∣ (25^3 + 15^4 - 5^6 + 20^3) → q ≤ p ∧ p = 97 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1089_108914


namespace NUMINAMATH_CALUDE_sin_equals_cos_810_deg_l1089_108931

theorem sin_equals_cos_810_deg (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * Real.pi / 180) = Real.cos (810 * Real.pi / 180) ↔ n = -180 ∨ n = 0 ∨ n = 180) :=
by sorry

end NUMINAMATH_CALUDE_sin_equals_cos_810_deg_l1089_108931


namespace NUMINAMATH_CALUDE_remainder_of_n_l1089_108983

theorem remainder_of_n (n : ℕ) (h1 : n^3 % 7 = 3) (h2 : n^4 % 7 = 2) : n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l1089_108983


namespace NUMINAMATH_CALUDE_train_length_l1089_108979

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 27 → ∃ (length_m : ℝ), abs (length_m - 450.09) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1089_108979


namespace NUMINAMATH_CALUDE_dinner_cakes_l1089_108972

def total_cakes : ℕ := 15
def lunch_cakes : ℕ := 6

theorem dinner_cakes : total_cakes - lunch_cakes = 9 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cakes_l1089_108972


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1089_108919

/-- Acme T-Shirt Company's pricing function -/
def acme_price (x : ℕ) : ℕ := 80 + 10 * x

/-- Beta T-Shirt Company's pricing function -/
def beta_price (x : ℕ) : ℕ := 20 + 15 * x

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_acme_cheaper : ℕ := 13

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_acme_cheaper < beta_price min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper → acme_price n ≥ beta_price n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1089_108919


namespace NUMINAMATH_CALUDE_banana_kiwi_equivalence_l1089_108961

-- Define the cost relationship between fruits
def cost_relation (banana pear kiwi : ℕ) : Prop :=
  4 * banana = 3 * pear ∧ 9 * pear = 6 * kiwi

-- Theorem statement
theorem banana_kiwi_equivalence :
  ∀ (banana pear kiwi : ℕ), cost_relation banana pear kiwi → 24 * banana = 12 * kiwi :=
by
  sorry

end NUMINAMATH_CALUDE_banana_kiwi_equivalence_l1089_108961


namespace NUMINAMATH_CALUDE_literate_female_percentage_approx_81_percent_l1089_108909

/-- Represents the demographics and literacy rates of a town -/
structure TownDemographics where
  total_inhabitants : ℕ
  adult_male_percent : ℚ
  adult_female_percent : ℚ
  children_percent : ℚ
  adult_male_literacy : ℚ
  adult_female_literacy : ℚ
  children_literacy : ℚ

/-- Calculates the percentage of literate females in the town -/
def literate_female_percentage (town : TownDemographics) : ℚ :=
  let adult_females := town.total_inhabitants * town.adult_female_percent
  let female_children := town.total_inhabitants * town.children_percent / 2
  let literate_adult_females := adult_females * town.adult_female_literacy
  let literate_female_children := female_children * town.children_literacy
  let total_literate_females := literate_adult_females + literate_female_children
  let total_females := adult_females + female_children
  total_literate_females / total_females

/-- Theorem stating that the percentage of literate females in the town is approximately 81% -/
theorem literate_female_percentage_approx_81_percent 
  (town : TownDemographics)
  (h1 : town.total_inhabitants = 3500)
  (h2 : town.adult_male_percent = 60 / 100)
  (h3 : town.adult_female_percent = 35 / 100)
  (h4 : town.children_percent = 5 / 100)
  (h5 : town.adult_male_literacy = 55 / 100)
  (h6 : town.adult_female_literacy = 80 / 100)
  (h7 : town.children_literacy = 95 / 100) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 / 100 ∧ 
  |literate_female_percentage town - 81 / 100| < ε :=
sorry

end NUMINAMATH_CALUDE_literate_female_percentage_approx_81_percent_l1089_108909


namespace NUMINAMATH_CALUDE_product_of_polynomials_l1089_108930

theorem product_of_polynomials (p q : ℝ) : 
  (∀ k : ℝ, (5 * k^2 - 2 * k + p) * (4 * k^2 + q * k - 6) = 20 * k^4 - 18 * k^3 - 31 * k^2 + 12 * k + 18) →
  p + q = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l1089_108930


namespace NUMINAMATH_CALUDE_negation_of_universal_inequality_l1089_108936

theorem negation_of_universal_inequality :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_inequality_l1089_108936


namespace NUMINAMATH_CALUDE_angle_bisectors_concurrent_l1089_108901

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrilateral ABCD
def Quadrilateral (A B C D : Point2D) : Prop := sorry

-- Define that P is an interior point of ABCD
def InteriorPoint (P : Point2D) (A B C D : Point2D) : Prop := sorry

-- Define the angle between three points
def Angle (P Q R : Point2D) : ℝ := sorry

-- Define the angle bisector
def AngleBisector (A B C : Point2D) : Point2D → Point2D → Prop := sorry

-- Define the perpendicular bisector of a line segment
def PerpendicularBisector (A B : Point2D) : Point2D → Point2D → Prop := sorry

-- Define when three lines are concurrent
def Concurrent (L1 L2 L3 : Point2D → Point2D → Prop) : Prop := sorry

theorem angle_bisectors_concurrent 
  (A B C D P : Point2D) 
  (h1 : Quadrilateral A B C D)
  (h2 : InteriorPoint P A B C D)
  (h3 : Angle P A D / Angle P B A / Angle D P A = 1 / 2 / 3)
  (h4 : Angle C B P / Angle B A P / Angle B P C = 1 / 2 / 3) :
  Concurrent 
    (AngleBisector A D P) 
    (AngleBisector P C B) 
    (PerpendicularBisector A B) := by sorry

end NUMINAMATH_CALUDE_angle_bisectors_concurrent_l1089_108901


namespace NUMINAMATH_CALUDE_quadratic_function_value_l1089_108976

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- Define the derivative of f(x)
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 8 * x - m

theorem quadratic_function_value : 
  ∀ (m : ℝ), 
  (∀ x : ℝ, x ≥ -2 → (f_derivative m) x ≥ 0) →  -- f(x) is increasing on [−2, +∞)
  (∀ x : ℝ, x < -2 → (f_derivative m) x < 0) →  -- f(x) is decreasing on (-∞, −2)
  f m 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l1089_108976


namespace NUMINAMATH_CALUDE_cars_cannot_meet_between_intersections_l1089_108952

/-- Represents a point in the triangular grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the triangular grid --/
inductive Direction
  | Up
  | UpRight
  | DownRight

/-- Represents a car's state --/
structure CarState where
  position : GridPoint
  direction : Direction

/-- Represents the possible moves a car can make --/
inductive Move
  | Straight
  | Left
  | Right

/-- Function to update a car's state based on a move --/
def updateCarState (state : CarState) (move : Move) : CarState :=
  sorry

/-- Predicate to check if two cars are at the same position --/
def samePosition (car1 : CarState) (car2 : CarState) : Prop :=
  car1.position = car2.position

/-- Predicate to check if a point is an intersection --/
def isIntersection (point : GridPoint) : Prop :=
  sorry

/-- Theorem stating that two cars cannot meet between intersections --/
theorem cars_cannot_meet_between_intersections 
  (initialState : CarState) 
  (moves1 moves2 : List Move) : 
  let finalState1 := moves1.foldl updateCarState initialState
  let finalState2 := moves2.foldl updateCarState initialState
  samePosition finalState1 finalState2 → isIntersection finalState1.position :=
sorry

end NUMINAMATH_CALUDE_cars_cannot_meet_between_intersections_l1089_108952


namespace NUMINAMATH_CALUDE_fraction_division_and_addition_l1089_108924

theorem fraction_division_and_addition : 
  (5 / 6 : ℚ) / (9 / 10 : ℚ) + 1 / 15 = 402 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_and_addition_l1089_108924


namespace NUMINAMATH_CALUDE_tea_box_duration_l1089_108943

-- Define the daily tea usage in ounces
def daily_usage : ℚ := 1 / 5

-- Define the box size in ounces
def box_size : ℚ := 28

-- Define the number of days in a week
def days_per_week : ℕ := 7

-- Theorem to prove
theorem tea_box_duration : 
  (box_size / daily_usage) / days_per_week = 20 := by
  sorry

end NUMINAMATH_CALUDE_tea_box_duration_l1089_108943


namespace NUMINAMATH_CALUDE_exercise_gender_relation_l1089_108957

/-- Represents the contingency table data -/
structure ContingencyTable where
  male_regular : ℕ
  female_regular : ℕ
  male_not_regular : ℕ
  female_not_regular : ℕ

/-- Calculates the chi-square value -/
def chi_square (table : ContingencyTable) : ℚ :=
  let n := table.male_regular + table.female_regular + table.male_not_regular + table.female_not_regular
  let a := table.male_regular
  let b := table.female_regular
  let c := table.male_not_regular
  let d := table.female_not_regular
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The theorem to be proved -/
theorem exercise_gender_relation (total_students : ℕ) (prob_regular : ℚ) (male_regular : ℕ) (female_not_regular : ℕ) 
    (h_total : total_students = 100)
    (h_prob : prob_regular = 1/2)
    (h_male_regular : male_regular = 35)
    (h_female_not_regular : female_not_regular = 25)
    (h_critical_value : (2706 : ℚ)/1000 < (3841 : ℚ)/1000) :
  let table := ContingencyTable.mk 
    male_regular
    (total_students / 2 - male_regular)
    (total_students / 2 - female_not_regular)
    female_not_regular
  chi_square table > (2706 : ℚ)/1000 := by
  sorry

end NUMINAMATH_CALUDE_exercise_gender_relation_l1089_108957


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l1089_108975

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_point_one :
  f' 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l1089_108975


namespace NUMINAMATH_CALUDE_periodic_points_measure_l1089_108970

open MeasureTheory

theorem periodic_points_measure (f : ℝ → ℝ) (hf : Continuous f) (hf0 : f 0 = 0) (hf1 : f 1 = 0) :
  let A := {h ∈ Set.Icc 0 1 | ∃ x ∈ Set.Icc 0 1, f (x + h) = f x}
  Measurable A ∧ volume A ≥ 1/2 := by
sorry

end NUMINAMATH_CALUDE_periodic_points_measure_l1089_108970


namespace NUMINAMATH_CALUDE_aloks_age_l1089_108941

theorem aloks_age (alok_age bipin_age chandan_age : ℕ) : 
  bipin_age = 6 * alok_age →
  bipin_age + 10 = 2 * (chandan_age + 10) →
  chandan_age = 10 →
  alok_age = 5 := by
sorry

end NUMINAMATH_CALUDE_aloks_age_l1089_108941


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1089_108920

theorem trigonometric_equation_solution (n : ℤ) :
  let f (x : ℝ) := (Real.sin x) ^ (Real.arctan (Real.sin x + Real.cos x))
  let g (x : ℝ) := (1 / Real.sin x) ^ (Real.arctan (Real.sin (2 * x)) + π / 4)
  let x₁ := 2 * n * π + π / 2
  let x₂ := 2 * n * π + 3 * π / 4
  ∀ x ∈ Set.Ioo (2 * n * π) ((2 * n + 1) * π),
    f x = g x ↔ (x = x₁ ∨ x = x₂) := by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1089_108920


namespace NUMINAMATH_CALUDE_max_silver_tokens_l1089_108967

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules --/
inductive ExchangeRule
  | RedToSilver : ExchangeRule  -- 3 red → 2 silver + 1 blue
  | BlueToSilver : ExchangeRule -- 4 blue → 1 silver + 2 red

/-- Applies an exchange rule to a token count --/
def applyExchange (tc : TokenCount) (rule : ExchangeRule) : Option TokenCount :=
  match rule with
  | ExchangeRule.RedToSilver =>
      if tc.red ≥ 3 then
        some ⟨tc.red - 3, tc.blue + 1, tc.silver + 2⟩
      else
        none
  | ExchangeRule.BlueToSilver =>
      if tc.blue ≥ 4 then
        some ⟨tc.red + 2, tc.blue - 4, tc.silver + 1⟩
      else
        none

/-- Determines if any exchange is possible --/
def canExchange (tc : TokenCount) : Bool :=
  tc.red ≥ 3 ∨ tc.blue ≥ 4

/-- The main theorem to prove --/
theorem max_silver_tokens :
  ∃ (final : TokenCount),
    final.silver = 113 ∧
    ¬(canExchange final) ∧
    (∀ (tc : TokenCount),
      tc.red = 100 ∧ tc.blue = 50 ∧ tc.silver = 0 →
      (∃ (exchanges : List ExchangeRule),
        (exchanges.foldl (λ acc rule => (applyExchange acc rule).getD acc) tc) = final)) :=
  sorry


end NUMINAMATH_CALUDE_max_silver_tokens_l1089_108967


namespace NUMINAMATH_CALUDE_problem_solution_l1089_108996

theorem problem_solution (x : ℝ) : x * 120 = 346 → x = 346 / 120 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1089_108996


namespace NUMINAMATH_CALUDE_eunji_confetti_l1089_108932

theorem eunji_confetti (red : ℕ) (green : ℕ) (given : ℕ) : 
  red = 1 → green = 9 → given = 4 → red + green - given = 6 := by sorry

end NUMINAMATH_CALUDE_eunji_confetti_l1089_108932


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_1_l1089_108987

theorem simplify_and_evaluate_1 (y : ℝ) (h : y = 2) :
  -3 * y^2 - 6 * y + 2 * y^2 + 5 * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_1_l1089_108987


namespace NUMINAMATH_CALUDE_train_speeds_equal_l1089_108923

-- Define the speeds and times
def speed_A : ℝ := 110
def time_A_after_meeting : ℝ := 9
def time_B_after_meeting : ℝ := 4

-- Define the theorem
theorem train_speeds_equal :
  ∀ (speed_B : ℝ) (time_before_meeting : ℝ),
    speed_B > 0 →
    time_before_meeting > 0 →
    speed_A * time_before_meeting + speed_A * time_A_after_meeting =
    speed_B * time_before_meeting + speed_B * time_B_after_meeting →
    speed_A * time_before_meeting = speed_B * time_before_meeting →
    speed_B = speed_A :=
by
  sorry

#check train_speeds_equal

end NUMINAMATH_CALUDE_train_speeds_equal_l1089_108923


namespace NUMINAMATH_CALUDE_katie_total_marbles_l1089_108903

/-- The number of marbles Katie has -/
def total_marbles (pink orange purple : ℕ) : ℕ := pink + orange + purple

/-- The properties of Katie's marble collection -/
def katie_marbles (pink orange purple : ℕ) : Prop :=
  pink = 13 ∧ orange = pink - 9 ∧ purple = 4 * orange

theorem katie_total_marbles :
  ∀ pink orange purple : ℕ,
    katie_marbles pink orange purple →
    total_marbles pink orange purple = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_katie_total_marbles_l1089_108903


namespace NUMINAMATH_CALUDE_plywood_perimeter_l1089_108997

theorem plywood_perimeter (length width perimeter : ℝ) : 
  length = 6 → width = 5 → perimeter = 2 * (length + width) → perimeter = 22 := by
  sorry

end NUMINAMATH_CALUDE_plywood_perimeter_l1089_108997


namespace NUMINAMATH_CALUDE_square_area_relation_l1089_108916

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I : ℝ := 2*a + 3*b
  let area_I : ℝ := (diagonal_I^2) / 2
  let area_II : ℝ := area_I^3
  area_II = (diagonal_I^6) / 8 := by
sorry

end NUMINAMATH_CALUDE_square_area_relation_l1089_108916


namespace NUMINAMATH_CALUDE_max_value_x3y2z_l1089_108944

theorem max_value_x3y2z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^3 * y^2 * z ≤ 1/432 := by
sorry

end NUMINAMATH_CALUDE_max_value_x3y2z_l1089_108944


namespace NUMINAMATH_CALUDE_rectangle_to_square_l1089_108971

theorem rectangle_to_square (area : ℝ) (reduction : ℝ) (side : ℝ) : 
  area = 600 →
  reduction = 10 →
  (side + reduction) * side = area →
  side * side = area →
  side = 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l1089_108971


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1089_108925

/-- The sum of the coefficients of the expanded expression -(2x - 5)(4x + 3(2x - 5)) is -15 -/
theorem sum_of_coefficients : ∃ a b c : ℚ,
  -(2 * X - 5) * (4 * X + 3 * (2 * X - 5)) = a * X^2 + b * X + c ∧ a + b + c = -15 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1089_108925


namespace NUMINAMATH_CALUDE_quadratic_properties_l1089_108995

/-- A quadratic function passing through specific points -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  f 0 = -7/2 ∧ f 1 = 1/2 ∧ f (3/2) = 1 ∧ f 2 = 1/2

theorem quadratic_properties (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧  -- f is quadratic
  f 0 = -7/2 ∧  -- y-axis intersection
  (∀ x, f (3/2 - x) = f (3/2 + x)) ∧  -- axis of symmetry
  (∀ x, f x ≤ f (3/2)) ∧  -- vertex
  (∀ x, f x = -2 * (x - 3/2)^2 + 1)  -- analytical expression
  := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1089_108995


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l1089_108950

theorem walking_rate_ratio (usual_time new_time usual_rate new_rate : ℝ) 
  (h1 : usual_time = 49)
  (h2 : new_time = usual_time - 7)
  (h3 : usual_rate * usual_time = new_rate * new_time) :
  new_rate / usual_rate = 7 / 6 := by
sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l1089_108950


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l1089_108953

theorem triangle_side_ratio (a b c q : ℝ) : 
  c = b * q ∧ c = a * q^2 → ((Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l1089_108953


namespace NUMINAMATH_CALUDE_max_y_value_max_y_value_achievable_l1089_108977

theorem max_y_value (x y : ℤ) (h : x * y + 5 * x + 4 * y = -5) : y ≤ 10 := by
  sorry

theorem max_y_value_achievable : ∃ x y : ℤ, x * y + 5 * x + 4 * y = -5 ∧ y = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_max_y_value_achievable_l1089_108977


namespace NUMINAMATH_CALUDE_money_distribution_l1089_108959

def total_proportion : ℕ := 5 + 2 + 4 + 3

theorem money_distribution (S : ℚ) (A_share B_share C_share D_share : ℚ) : 
  A_share = 2500 ∧ 
  A_share = (5 : ℚ) / total_proportion * S ∧
  B_share = (2 : ℚ) / total_proportion * S ∧
  C_share = (4 : ℚ) / total_proportion * S ∧
  D_share = (3 : ℚ) / total_proportion * S →
  C_share - D_share = 500 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1089_108959


namespace NUMINAMATH_CALUDE_friends_team_assignment_l1089_108973

theorem friends_team_assignment : 
  let n : ℕ := 8  -- number of friends
  let k : ℕ := 4  -- number of teams
  k ^ n = 65536 := by sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l1089_108973


namespace NUMINAMATH_CALUDE_polygon_angles_l1089_108942

theorem polygon_angles (n : ℕ) (sum_interior : ℝ) (sum_exterior : ℝ) : 
  sum_exterior = 180 → 
  sum_interior = 4 * sum_exterior → 
  sum_interior = (n - 2) * 180 → 
  n = 11 ∧ sum_interior = 1620 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l1089_108942


namespace NUMINAMATH_CALUDE_roots_shifted_polynomial_l1089_108905

theorem roots_shifted_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 4*x - 8 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 23*x + 7 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_shifted_polynomial_l1089_108905


namespace NUMINAMATH_CALUDE_tangent_lines_condition_l1089_108929

/-- The function f(x) = 4x + ax² has two tangent lines passing through (1,1) iff a ∈ (-∞, -3) ∪ (0, +∞) -/
theorem tangent_lines_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    (4 * x₁ + a * x₁^2 - (4 + 2*a*x₁) * x₁ + (4 + 2*a*x₁) = 1) ∧
    (4 * x₂ + a * x₂^2 - (4 + 2*a*x₂) * x₂ + (4 + 2*a*x₂) = 1)) ↔
  (a < -3 ∨ a > 0) :=
by sorry


end NUMINAMATH_CALUDE_tangent_lines_condition_l1089_108929


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1089_108935

theorem square_plus_reciprocal_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1089_108935


namespace NUMINAMATH_CALUDE_elephant_to_big_cat_ratio_l1089_108937

/-- Represents the population of animals in a park -/
structure ParkPopulation where
  lions : ℕ
  leopards : ℕ
  elephants : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem about the ratio of elephants to lions and leopards in a park -/
theorem elephant_to_big_cat_ratio 
  (park : ParkPopulation) 
  (h1 : park.lions = 2 * park.leopards) 
  (h2 : park.lions = 200) 
  (h3 : park.lions + park.leopards + park.elephants = 450) : 
  Ratio.mk park.elephants (park.lions + park.leopards) = Ratio.mk 1 2 := by
  sorry

end NUMINAMATH_CALUDE_elephant_to_big_cat_ratio_l1089_108937


namespace NUMINAMATH_CALUDE_safari_park_acrobats_l1089_108912

theorem safari_park_acrobats :
  ∀ (acrobats giraffes : ℕ),
    2 * acrobats + 4 * giraffes = 32 →
    acrobats + giraffes = 10 →
    acrobats = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_safari_park_acrobats_l1089_108912


namespace NUMINAMATH_CALUDE_shopping_total_l1089_108913

def tuesday_discount : ℝ := 0.1
def jimmy_shorts_count : ℕ := 3
def jimmy_shorts_price : ℝ := 15
def irene_shirts_count : ℕ := 5
def irene_shirts_price : ℝ := 17

theorem shopping_total : 
  let total_before_discount := jimmy_shorts_count * jimmy_shorts_price + 
                               irene_shirts_count * irene_shirts_price
  let discount := total_before_discount * tuesday_discount
  let final_amount := total_before_discount - discount
  final_amount = 117 := by sorry

end NUMINAMATH_CALUDE_shopping_total_l1089_108913


namespace NUMINAMATH_CALUDE_johns_tour_program_l1089_108949

theorem johns_tour_program (total_budget : ℕ) (budget_reduction : ℕ) (extra_days : ℕ) :
  total_budget = 360 ∧ budget_reduction = 3 ∧ extra_days = 4 →
  ∃ (days : ℕ) (daily_expense : ℕ),
    total_budget = days * daily_expense ∧
    total_budget = (days + extra_days) * (daily_expense - budget_reduction) ∧
    days = 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_tour_program_l1089_108949


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l1089_108906

/-- The ratio of cylinder volumes formed from a 5x8 rectangle -/
theorem cylinder_volume_ratio : 
  ∀ (h₁ h₂ r₁ r₂ : ℝ), 
    h₁ = 8 ∧ h₂ = 5 ∧ 
    2 * Real.pi * r₁ = 5 ∧ 
    2 * Real.pi * r₂ = 8 →
    max (Real.pi * r₁^2 * h₁) (Real.pi * r₂^2 * h₂) / 
    min (Real.pi * r₁^2 * h₁) (Real.pi * r₂^2 * h₂) = 8/5 := by
  sorry


end NUMINAMATH_CALUDE_cylinder_volume_ratio_l1089_108906


namespace NUMINAMATH_CALUDE_volume_Q₃_l1089_108986

/-- Represents a polyhedron in the sequence -/
structure Polyhedron where
  volume : ℚ

/-- Generates the next polyhedron in the sequence -/
def next_polyhedron (Q : Polyhedron) : Polyhedron :=
  { volume := Q.volume + 4 * (27/64) * Q.volume }

/-- The initial tetrahedron Q₀ -/
def Q₀ : Polyhedron :=
  { volume := 2 }

/-- The sequence of polyhedra -/
def Q : ℕ → Polyhedron
  | 0 => Q₀
  | n + 1 => next_polyhedron (Q n)

theorem volume_Q₃ : (Q 3).volume = 156035 / 65536 := by sorry

end NUMINAMATH_CALUDE_volume_Q₃_l1089_108986


namespace NUMINAMATH_CALUDE_business_trip_distance_l1089_108974

/-- Calculates the total distance traveled during a business trip -/
theorem business_trip_distance (total_duration : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_duration = 8 →
  speed1 = 70 →
  speed2 = 85 →
  (total_duration / 2 * speed1) + (total_duration / 2 * speed2) = 620 := by
  sorry

#check business_trip_distance

end NUMINAMATH_CALUDE_business_trip_distance_l1089_108974


namespace NUMINAMATH_CALUDE_buses_needed_for_trip_l1089_108964

/-- Calculates the number of buses needed for a school trip -/
theorem buses_needed_for_trip (total_students : ℕ) (van_students : ℕ) (bus_capacity : ℕ) 
  (h1 : total_students = 500)
  (h2 : van_students = 56)
  (h3 : bus_capacity = 45) :
  Nat.ceil ((total_students - van_students) / bus_capacity) = 10 := by
  sorry

#check buses_needed_for_trip

end NUMINAMATH_CALUDE_buses_needed_for_trip_l1089_108964


namespace NUMINAMATH_CALUDE_parallel_vectors_k_l1089_108946

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![0, 1]
def vector_c (k : ℝ) : Fin 2 → ℝ := ![-2, k]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, ∀ i, v i = t * w i

theorem parallel_vectors_k (k : ℝ) :
  parallel (λ i => vector_a i + 2 * vector_b i) (vector_c k) → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_l1089_108946


namespace NUMINAMATH_CALUDE_problem_solution_l1089_108910

theorem problem_solution (x : ℝ) (h1 : x < 0) (h2 : 1 / (x + 1 / (x + 2)) = 2) : x + 7/2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1089_108910


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1089_108994

theorem smaller_number_problem (x y : ℕ) : 
  x * y = 40 → x + y = 14 → min x y = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1089_108994


namespace NUMINAMATH_CALUDE_one_ball_selection_l1089_108990

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 2

/-- The number of blue balls in the bag -/
def num_blue_balls : ℕ := 4

/-- Each ball has a different number -/
axiom balls_are_distinct : True

/-- The number of ways to select one ball from the bag -/
def ways_to_select_one_ball : ℕ := num_red_balls + num_blue_balls

theorem one_ball_selection :
  ways_to_select_one_ball = 6 :=
by sorry

end NUMINAMATH_CALUDE_one_ball_selection_l1089_108990


namespace NUMINAMATH_CALUDE_line_through_point_l1089_108966

/-- Given a line described by the equation 2 - kx = -4y that contains the point (3, 1),
    prove that k = 2. -/
theorem line_through_point (k : ℝ) : 
  (2 - k * 3 = -4 * 1) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1089_108966


namespace NUMINAMATH_CALUDE_smallest_AAB_value_l1089_108922

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem smallest_AAB_value :
  ∀ A B : ℕ,
  is_digit A →
  is_digit B →
  two_digit (10 * A + B) →
  three_digit (100 * A + 10 * A + B) →
  (10 * A + B : ℚ) = (1 / 7) * (100 * A + 10 * A + B) →
  ∀ A' B' : ℕ,
  is_digit A' →
  is_digit B' →
  two_digit (10 * A' + B') →
  three_digit (100 * A' + 10 * A' + B') →
  (10 * A' + B' : ℚ) = (1 / 7) * (100 * A' + 10 * A' + B') →
  100 * A + 10 * A + B ≤ 100 * A' + 10 * A' + B' →
  100 * A + 10 * A + B = 332 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAB_value_l1089_108922


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1089_108954

theorem polynomial_divisibility : ∃ q : Polynomial ℝ, 
  (X^3 * 6 + X^2 * 1 + -1) = (X * 2 + -1) * q :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1089_108954


namespace NUMINAMATH_CALUDE_max_value_theorem_l1089_108951

theorem max_value_theorem (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h : (a₂ - a₁)^2 + (a₃ - a₂)^2 + (a₄ - a₃)^2 + (a₅ - a₄)^2 + (a₆ - a₅)^2 = 1) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ 
  ∀ (x : ℝ), x = (a₅ + a₆) - (a₁ + a₄) → x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1089_108951


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l1089_108991

def M : ℕ := 36 * 36 * 65 * 272

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M) * 510 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l1089_108991


namespace NUMINAMATH_CALUDE_local_minimum_implies_b_range_l1089_108926

def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

theorem local_minimum_implies_b_range (b : ℝ) :
  (∃ x₀ ∈ Set.Ioo 0 1, IsLocalMin (f b) x₀) →
  0 < b ∧ b < 1 := by
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_b_range_l1089_108926


namespace NUMINAMATH_CALUDE_peanuts_in_jar_l1089_108907

theorem peanuts_in_jar (initial_peanuts : ℕ) (brock_fraction : ℚ) (bonita_fraction : ℚ) (carlos_peanuts : ℕ) : 
  initial_peanuts = 220 →
  brock_fraction = 1/4 →
  bonita_fraction = 2/5 →
  carlos_peanuts = 17 →
  initial_peanuts - 
    (initial_peanuts * brock_fraction).floor - 
    ((initial_peanuts - (initial_peanuts * brock_fraction).floor) * bonita_fraction).floor - 
    carlos_peanuts = 82 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_jar_l1089_108907


namespace NUMINAMATH_CALUDE_purple_pants_count_l1089_108960

/-- Represents the number of shirts Teairra has -/
def total_shirts : ℕ := 5

/-- Represents the number of pants Teairra has -/
def total_pants : ℕ := 24

/-- Represents the number of plaid shirts Teairra has -/
def plaid_shirts : ℕ := 3

/-- Represents the number of items that are neither plaid nor purple -/
def neither_plaid_nor_purple : ℕ := 21

/-- Represents the number of purple pants Teairra has -/
def purple_pants : ℕ := total_pants - (neither_plaid_nor_purple - (total_shirts - plaid_shirts))

theorem purple_pants_count : purple_pants = 5 := by
  sorry

end NUMINAMATH_CALUDE_purple_pants_count_l1089_108960


namespace NUMINAMATH_CALUDE_find_unknown_number_l1089_108904

/-- Given two positive integers with known HCF and LCM, find the unknown number -/
theorem find_unknown_number (A B : ℕ+) (h1 : A = 24) 
  (h2 : Nat.gcd A B = 12) (h3 : Nat.lcm A B = 312) : B = 156 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l1089_108904


namespace NUMINAMATH_CALUDE_max_value_of_f_sum_of_powers_gt_one_l1089_108948

-- Part 1
theorem max_value_of_f (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ M : ℝ, M = 1 ∧ ∀ x > -1, (1 + x)^a - a * x ≤ M := by sorry

-- Part 2
theorem sum_of_powers_gt_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^b + b^a > 1 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_sum_of_powers_gt_one_l1089_108948


namespace NUMINAMATH_CALUDE_option_A_not_sufficient_l1089_108968

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def perp_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem option_A_not_sufficient
  (a b : Line3D)
  (α β : Plane3D)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : line_parallel_plane a α)
  (h4 : line_parallel_plane b β)
  (h5 : line_perp_plane a β) :
  ¬ (perp_lines a b) :=
sorry

end NUMINAMATH_CALUDE_option_A_not_sufficient_l1089_108968


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1089_108982

/-- Given that the solution set of the inequality system {x + 1 > 2x - 2, x < a} is x < 3,
    prove that the range of values for a is a ≥ 3. -/
theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 1 > 2*x - 2 ∧ x < a) ↔ x < 3) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1089_108982


namespace NUMINAMATH_CALUDE_min_non_parallel_lines_l1089_108945

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A type representing a line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Predicate to check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Function to create a line passing through two points -/
def line_through_points (p q : Point) : Line :=
  { a := q.y - p.y,
    b := p.x - q.x,
    c := p.x * q.y - q.x * p.y }

/-- The main theorem -/
theorem min_non_parallel_lines (n : ℕ) (points : Fin n → Point) 
  (h_n : n ≥ 3)
  (h_not_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ (lines : Fin n → Line),
    (∀ i j, i ≠ j → ¬parallel (lines i) (lines j)) ∧
    (∀ lines' : Fin n' → Line, n' < n →
      ¬(∀ i j, i ≠ j → ¬parallel (lines' i) (lines' j))) :=
sorry

end NUMINAMATH_CALUDE_min_non_parallel_lines_l1089_108945


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1089_108963

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := a^2 - 4 + (a^2 - 3*a + 2)*I
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1089_108963


namespace NUMINAMATH_CALUDE_max_c_value_l1089_108921

theorem max_c_value (a b c x y z : ℝ) : 
  a ≥ 1 → b ≥ 1 → c ≥ 1 → x > 0 → y > 0 → z > 0 →
  a^x + b^y + c^z = 4 →
  x * a^x + y * b^y + z * c^z = 6 →
  x^2 * a^x + y^2 * b^y + z^2 * c^z = 9 →
  c ≤ Real.rpow 4 (1/3) :=
sorry

end NUMINAMATH_CALUDE_max_c_value_l1089_108921


namespace NUMINAMATH_CALUDE_employee_hire_year_l1089_108940

/-- Represents the rule of 70 provision for retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Calculates the year an employee was hired given their retirement eligibility year and years of employment -/
def hire_year (retirement_eligibility_year : ℕ) (years_employed : ℕ) : ℕ :=
  retirement_eligibility_year - years_employed

theorem employee_hire_year :
  ∀ (retirement_eligibility_year : ℕ) (hire_age : ℕ),
    hire_age = 32 →
    retirement_eligibility_year = 2009 →
    (∃ (years_employed : ℕ), rule_of_70 (hire_age + years_employed) years_employed) →
    hire_year retirement_eligibility_year (retirement_eligibility_year - (hire_age + 32)) = 1971 :=
by sorry

end NUMINAMATH_CALUDE_employee_hire_year_l1089_108940


namespace NUMINAMATH_CALUDE_unique_decimal_base7_number_l1089_108908

theorem unique_decimal_base7_number : ∃! n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a b c d : ℕ),
    n = 1000*a + 100*b + 10*c + d ∧
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    n = 343*d + 49*c + 7*b + a ∧
    n = 2116 :=
by sorry

#check unique_decimal_base7_number

end NUMINAMATH_CALUDE_unique_decimal_base7_number_l1089_108908


namespace NUMINAMATH_CALUDE_smallest_five_digit_number_with_conditions_l1089_108939

theorem smallest_five_digit_number_with_conditions : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- 5-digit number
  (n % 32 = 0) ∧              -- divisible by 32
  (n % 45 = 0) ∧              -- divisible by 45
  (n % 54 = 0) ∧              -- divisible by 54
  (30 % n = 0) ∧              -- factor of 30
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m % 32 = 0) ∧ (m % 45 = 0) ∧ (m % 54 = 0) ∧ (30 % m = 0) → n ≤ m) ∧
  n = 12960 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_number_with_conditions_l1089_108939


namespace NUMINAMATH_CALUDE_bicycle_business_loss_percentage_l1089_108958

/-- Calculates the overall loss percentage for a bicycle business -/
def overall_loss_percentage (cp1 sp1 cp2 sp2 cp3 sp3 : ℚ) : ℚ :=
  let tcp := cp1 + cp2 + cp3
  let tsp := sp1 + sp2 + sp3
  let loss := tcp - tsp
  (loss / tcp) * 100

/-- Theorem stating the overall loss percentage for the given bicycle business -/
theorem bicycle_business_loss_percentage :
  let cp1 := 1000
  let sp1 := 1080
  let cp2 := 1500
  let sp2 := 1100
  let cp3 := 2000
  let sp3 := 2200
  overall_loss_percentage cp1 sp1 cp2 sp2 cp3 sp3 = 2.67 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_business_loss_percentage_l1089_108958


namespace NUMINAMATH_CALUDE_odd_integer_pairs_theorem_l1089_108989

def phi : ℕ → ℕ := sorry  -- Euler's totient function

theorem odd_integer_pairs_theorem (a b : ℕ) (ha : Odd a) (hb : Odd b) (ha_gt_1 : a > 1) (hb_gt_1 : b > 1) :
  7 * (phi a)^2 - phi (a * b) + 11 * (phi b)^2 = 2 * (a^2 + b^2) →
  ∃ x : ℕ, a = 15 * 3^x ∧ b = 3 * 3^x :=
sorry

end NUMINAMATH_CALUDE_odd_integer_pairs_theorem_l1089_108989


namespace NUMINAMATH_CALUDE_total_hours_worked_l1089_108999

def ordinary_rate : ℚ := 60 / 100
def overtime_rate : ℚ := 90 / 100
def total_earnings : ℚ := 3240 / 100
def overtime_hours : ℕ := 8

theorem total_hours_worked : ℕ := by
  sorry

#check total_hours_worked = 50

end NUMINAMATH_CALUDE_total_hours_worked_l1089_108999


namespace NUMINAMATH_CALUDE_k_range_l1089_108984

theorem k_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) → 
  -1 < k ∧ k ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_k_range_l1089_108984


namespace NUMINAMATH_CALUDE_letter_cost_l1089_108927

/-- The cost to mail each letter, given the total cost, package cost, and number of letters and packages. -/
theorem letter_cost (total_cost package_cost : ℚ) (num_letters num_packages : ℕ) : 
  total_cost = 4.49 →
  package_cost = 0.88 →
  num_letters = 5 →
  num_packages = 3 →
  (num_letters : ℚ) * ((total_cost - (package_cost * (num_packages : ℚ))) / (num_letters : ℚ)) = 0.37 := by
  sorry

end NUMINAMATH_CALUDE_letter_cost_l1089_108927


namespace NUMINAMATH_CALUDE_correct_arrangements_l1089_108918

def num_seats : ℕ := 5
def num_teachers : ℕ := 4

/-- The number of arrangements where Teacher A is to the left of Teacher B -/
def arrangements_a_left_of_b : ℕ := 60

theorem correct_arrangements :
  arrangements_a_left_of_b = (num_seats.factorial / (num_seats - num_teachers).factorial) / 2 :=
sorry

end NUMINAMATH_CALUDE_correct_arrangements_l1089_108918
