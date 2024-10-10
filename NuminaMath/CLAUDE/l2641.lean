import Mathlib

namespace number_of_groups_l2641_264184

def lunch_times : List ℕ := [10, 12, 15, 8, 16, 18, 19, 18, 20, 18, 18, 20, 28, 22, 25, 20, 15, 16, 21, 16]

def class_interval : ℕ := 4

theorem number_of_groups : 
  let min_time := lunch_times.minimum?
  let max_time := lunch_times.maximum?
  match min_time, max_time with
  | some min, some max => 
    (max - min) / class_interval + 1 = 6
  | _, _ => False
  := by sorry

end number_of_groups_l2641_264184


namespace existence_of_x_and_y_l2641_264186

theorem existence_of_x_and_y (f : ℝ → ℝ) : ∃ x y : ℝ, f (x - f y) > y * f x + x := by
  sorry

end existence_of_x_and_y_l2641_264186


namespace f_not_monotonic_exists_even_f_exists_three_zeros_l2641_264159

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - abs (x + a)

-- Theorem 1: f is not monotonic for any a
theorem f_not_monotonic : ∀ a : ℝ, ¬(Monotone (f a)) := by sorry

-- Theorem 2: There exists an 'a' for which f is even
theorem exists_even_f : ∃ a : ℝ, ∀ x : ℝ, f a x = f a (-x) := by sorry

-- Theorem 3: There exists a negative 'a' for which f has three zeros
theorem exists_three_zeros : ∃ a : ℝ, a < 0 ∧ (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) := by sorry

end f_not_monotonic_exists_even_f_exists_three_zeros_l2641_264159


namespace evaluate_expression_l2641_264134

theorem evaluate_expression (c d : ℝ) (h1 : c = 3) (h2 : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 := by
  sorry

end evaluate_expression_l2641_264134


namespace greatest_n_value_l2641_264162

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 ∧ ∃ m : ℤ, m = 10 ∧ 101 * m^2 ≤ 12100 := by
  sorry

end greatest_n_value_l2641_264162


namespace triangle_area_l2641_264133

theorem triangle_area (A B C : EuclideanSpace ℝ (Fin 2)) :
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3
  let angle_A : ℝ := π / 4
  let area : ℝ := (1 / 2) * b * c * Real.sin angle_A
  area = Real.sqrt 6 / 4 := by
sorry

end triangle_area_l2641_264133


namespace suji_age_l2641_264187

theorem suji_age (abi_age suji_age : ℕ) : 
  (abi_age : ℚ) / suji_age = 5 / 4 →
  ((abi_age + 3) : ℚ) / (suji_age + 3) = 11 / 9 →
  suji_age = 24 := by
sorry

end suji_age_l2641_264187


namespace friends_chicken_pieces_l2641_264111

/-- Given a total number of chicken pieces, the number eaten by Lyndee, and the number of friends,
    calculate the number of pieces each friend ate. -/
def chicken_per_friend (total : ℕ) (lyndee_ate : ℕ) (num_friends : ℕ) : ℕ :=
  (total - lyndee_ate) / num_friends

theorem friends_chicken_pieces :
  chicken_per_friend 11 1 5 = 2 := by
sorry

end friends_chicken_pieces_l2641_264111


namespace bake_sale_money_raised_l2641_264120

/-- Represents the number of items in a dozen -/
def dozenSize : Nat := 12

/-- Calculates the total number of items given the number of dozens -/
def totalItems (dozens : Nat) : Nat := dozens * dozenSize

/-- Represents the price of a cookie in cents -/
def cookiePrice : Nat := 100

/-- Represents the price of a brownie or blondie in cents -/
def browniePrice : Nat := 200

/-- Calculates the total money raised from the bake sale -/
def totalMoneyRaised : Nat :=
  let bettyChocolateChip := totalItems 4
  let bettyOatmealRaisin := totalItems 6
  let bettyBrownies := totalItems 2
  let paigeSugar := totalItems 6
  let paigeBlondies := totalItems 3
  let paigeCreamCheese := totalItems 5

  let totalCookies := bettyChocolateChip + bettyOatmealRaisin + paigeSugar
  let totalBrowniesBlondies := bettyBrownies + paigeBlondies + paigeCreamCheese

  totalCookies * cookiePrice + totalBrowniesBlondies * browniePrice

theorem bake_sale_money_raised :
  totalMoneyRaised = 43200 := by sorry

end bake_sale_money_raised_l2641_264120


namespace largest_number_with_sum_19_l2641_264174

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 10) ((m % 10) :: acc)
  aux n []

def sum_digits (n : ℕ) : ℕ :=
  (digits n).sum

def all_digits_different (n : ℕ) : Prop :=
  (digits n).Nodup

theorem largest_number_with_sum_19 :
  ∀ n : ℕ, 
    sum_digits n = 19 → 
    all_digits_different n → 
    n ≤ 65431 :=
by sorry

end largest_number_with_sum_19_l2641_264174


namespace tommy_pencil_case_erasers_l2641_264106

/-- Represents the contents of Tommy's pencil case -/
structure PencilCase where
  total_items : ℕ
  pencils : ℕ
  pens : ℕ
  erasers : ℕ

/-- Theorem stating the number of erasers in Tommy's pencil case -/
theorem tommy_pencil_case_erasers (pc : PencilCase)
    (h1 : pc.total_items = 13)
    (h2 : pc.pens = 2 * pc.pencils)
    (h3 : pc.pencils = 4)
    (h4 : pc.total_items = pc.pencils + pc.pens + pc.erasers) :
    pc.erasers = 1 := by
  sorry

end tommy_pencil_case_erasers_l2641_264106


namespace math_club_payment_l2641_264112

theorem math_club_payment (B : ℕ) : 
  B < 10 →  -- Ensure B is a single digit
  (100 + 10 * B + 8) % 9 = 0 →  -- The number 1B8 is divisible by 9
  B = 0 := by
sorry

end math_club_payment_l2641_264112


namespace incircle_iff_reciprocal_heights_sum_l2641_264177

/-- A quadrilateral with heights h₁, h₂, h₃, h₄ -/
structure Quadrilateral where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  h₁_pos : 0 < h₁
  h₂_pos : 0 < h₂
  h₃_pos : 0 < h₃
  h₄_pos : 0 < h₄

/-- The property of having an incircle -/
def has_incircle (q : Quadrilateral) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∃ (center : ℝ × ℝ), True  -- We don't specify the exact conditions for an incircle

/-- The main theorem: a quadrilateral has an incircle iff the sum of reciprocals of opposite heights are equal -/
theorem incircle_iff_reciprocal_heights_sum (q : Quadrilateral) :
  has_incircle q ↔ 1 / q.h₁ + 1 / q.h₃ = 1 / q.h₂ + 1 / q.h₄ := by
  sorry

end incircle_iff_reciprocal_heights_sum_l2641_264177


namespace consecutive_squares_not_perfect_square_l2641_264176

theorem consecutive_squares_not_perfect_square (n : ℕ) : 
  ∃ k : ℕ, (n - 1)^2 + n^2 + (n + 1)^2 ≠ k^2 := by
  sorry

end consecutive_squares_not_perfect_square_l2641_264176


namespace problem_1_l2641_264154

theorem problem_1 : Real.sqrt 12 + (-2024)^(0 : ℕ) - 4 * Real.sin (π / 3) = 1 := by
  sorry

end problem_1_l2641_264154


namespace victors_friend_bought_two_decks_l2641_264129

/-- The number of decks Victor's friend bought given the conditions of the problem -/
def victors_friend_decks (deck_cost : ℕ) (victors_decks : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent - deck_cost * victors_decks) / deck_cost

/-- Theorem stating that Victor's friend bought 2 decks -/
theorem victors_friend_bought_two_decks :
  victors_friend_decks 8 6 64 = 2 := by
  sorry

end victors_friend_bought_two_decks_l2641_264129


namespace problem_statement_l2641_264116

theorem problem_statement : 2 * ((7 + 5)^2 + (7^2 + 5^2)) = 436 := by
  sorry

end problem_statement_l2641_264116


namespace chord_relations_l2641_264191

theorem chord_relations (d s : ℝ) : 
  0 < s ∧ s < d ∧ d < 2 →  -- Conditions for chords in a unit circle
  (d - s = 1 ∧ d * s = 1 ∧ d^2 - s^2 = Real.sqrt 5) ↔
  (d = (1 + Real.sqrt 5) / 2 ∧ s = (-1 + Real.sqrt 5) / 2) :=
by sorry

end chord_relations_l2641_264191


namespace f_monotonicity_when_k_zero_f_nonnegative_condition_exponential_fraction_inequality_l2641_264104

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - 1 - 2 * x - k * x^2

theorem f_monotonicity_when_k_zero :
  ∀ x : ℝ, x > 0 → (deriv (f 0)) x > 0 ∧
  ∀ x : ℝ, x < 0 → (deriv (f 0)) x < 0 := by sorry

theorem f_nonnegative_condition :
  ∀ k : ℝ, (∀ x : ℝ, x ≥ 0 → f k x ≥ 0) ↔ k ≤ 2 := by sorry

theorem exponential_fraction_inequality :
  ∀ n : ℕ+, (Real.exp (2 * ↑n) - 1) / (Real.exp 2 - 1) ≥ (2 * ↑n^3 + ↑n) / 3 := by sorry

end f_monotonicity_when_k_zero_f_nonnegative_condition_exponential_fraction_inequality_l2641_264104


namespace distance_inequality_l2641_264170

theorem distance_inequality (x y : ℝ) :
  Real.sqrt ((x + 4)^2 + (y + 2)^2) + Real.sqrt ((x - 5)^2 + (y + 4)^2) ≤
  Real.sqrt ((x - 2)^2 + (y - 6)^2) + Real.sqrt ((x - 5)^2 + (y - 6)^2) + 20 := by
  sorry

end distance_inequality_l2641_264170


namespace cupcakes_sold_l2641_264198

/-- Proves that Carol sold 9 cupcakes given the initial and final conditions -/
theorem cupcakes_sold (initial : ℕ) (made_after : ℕ) (final : ℕ) : 
  initial = 30 → made_after = 28 → final = 49 → 
  ∃ (sold : ℕ), sold = 9 ∧ initial - sold + made_after = final := by
  sorry

end cupcakes_sold_l2641_264198


namespace ceiling_sqrt_200_l2641_264151

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end ceiling_sqrt_200_l2641_264151


namespace smallest_right_triangle_area_l2641_264145

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area := (1 / 2) * a * b
  ∀ c : ℝ, (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → area ≤ (1 / 2) * a * c ∧ area ≤ (1 / 2) * b * c :=
by sorry

end smallest_right_triangle_area_l2641_264145


namespace calculation_proof_l2641_264171

theorem calculation_proof : (-1)^2023 + 6 * Real.cos (π / 3) + (Real.pi - 3.14)^0 - Real.sqrt 16 = -1 := by
  sorry

end calculation_proof_l2641_264171


namespace square_root_fraction_simplification_l2641_264142

theorem square_root_fraction_simplification :
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = 5 * Real.sqrt 65 / 13 := by
  sorry

end square_root_fraction_simplification_l2641_264142


namespace collinear_points_x_value_l2641_264128

/-- Given three collinear points A(-1, 2), B(2, 4), and C(x, 3), prove that x = 1/2 --/
theorem collinear_points_x_value :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (2, 4)
  let C : ℝ × ℝ := (x, 3)
  (∀ t : ℝ, ∃ u v : ℝ, u * (B.1 - A.1) + v * (C.1 - A.1) = t * (B.1 - A.1) ∧
                       u * (B.2 - A.2) + v * (C.2 - A.2) = t * (B.2 - A.2)) →
  x = 1/2 := by
sorry

end collinear_points_x_value_l2641_264128


namespace probability_theorem_l2641_264153

/-- The probability of having a child with younger brother, older brother, younger sister, and older sister
    given n > 4 children and equal probability of male and female births -/
def probability (n : ℕ) : ℚ :=
  1 - (n - 2 : ℚ) / 2^(n - 3)

/-- Theorem stating the probability for the given conditions -/
theorem probability_theorem (n : ℕ) (h : n > 4) :
  probability n = 1 - (n - 2 : ℚ) / 2^(n - 3) :=
by sorry

end probability_theorem_l2641_264153


namespace maximum_value_implies_ratio_l2641_264196

/-- The function f(x) = x³ + ax² + bx - a² - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem maximum_value_implies_ratio (a b : ℝ) :
  (∀ x, f a b x ≤ f a b 1) ∧  -- f(x) reaches maximum at x = 1
  (f a b 1 = 10) ∧            -- The maximum value is 10
  (f_deriv a b 1 = 0)         -- Derivative is zero at x = 1
  → a / b = -2 / 3 := by sorry

end maximum_value_implies_ratio_l2641_264196


namespace isosceles_triangle_with_80_degree_angle_l2641_264180

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We represent angles in degrees as natural numbers
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ
  -- Sum of angles is 180°
  sum_180 : angle1 + angle2 + angle3 = 180
  -- Two angles are equal (property of isosceles triangle)
  two_equal : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3)

-- Theorem statement
theorem isosceles_triangle_with_80_degree_angle 
  (t : IsoscelesTriangle) 
  (h : t.angle1 = 80 ∨ t.angle2 = 80 ∨ t.angle3 = 80) :
  (t.angle1 = 80 ∧ t.angle2 = 80 ∧ t.angle3 = 20) ∨
  (t.angle1 = 80 ∧ t.angle2 = 20 ∧ t.angle3 = 80) ∨
  (t.angle1 = 20 ∧ t.angle2 = 80 ∧ t.angle3 = 80) ∨
  (t.angle1 = 50 ∧ t.angle2 = 50 ∧ t.angle3 = 80) :=
by sorry

end isosceles_triangle_with_80_degree_angle_l2641_264180


namespace cube_solid_surface_area_l2641_264192

/-- A solid composed of 7 identical cubes -/
structure CubeSolid where
  -- The volume of each individual cube
  cube_volume : ℝ
  -- The side length of each cube
  cube_side : ℝ
  -- The total volume of the solid
  total_volume : ℝ
  -- The surface area of the solid
  surface_area : ℝ
  -- Conditions
  cube_volume_def : cube_volume = total_volume / 7
  cube_side_def : cube_side ^ 3 = cube_volume
  surface_area_def : surface_area = 30 * (cube_side ^ 2)

/-- Theorem: If the total volume of the CubeSolid is 875 cm³, then its surface area is 750 cm² -/
theorem cube_solid_surface_area (s : CubeSolid) (h : s.total_volume = 875) : 
  s.surface_area = 750 := by
  sorry

#check cube_solid_surface_area

end cube_solid_surface_area_l2641_264192


namespace art_fair_sales_l2641_264123

/-- The total number of paintings sold at Tracy's art fair booth -/
def total_paintings_sold (first_group : Nat) (second_group : Nat) (third_group : Nat)
  (first_group_purchase : Nat) (second_group_purchase : Nat) (third_group_purchase : Nat) : Nat :=
  first_group * first_group_purchase +
  second_group * second_group_purchase +
  third_group * third_group_purchase

/-- Theorem stating the total number of paintings sold at Tracy's art fair booth -/
theorem art_fair_sales :
  total_paintings_sold 4 12 4 2 1 4 = 36 := by
  sorry

end art_fair_sales_l2641_264123


namespace jackie_daily_distance_l2641_264190

/-- Prove that Jackie walks 2 miles per day -/
theorem jackie_daily_distance (jessie_daily : Real) (days : Nat) (extra_distance : Real) :
  jessie_daily = 1.5 →
  days = 6 →
  extra_distance = 3 →
  ∃ (jackie_daily : Real),
    jackie_daily * days = jessie_daily * days + extra_distance ∧
    jackie_daily = 2 := by
  sorry

end jackie_daily_distance_l2641_264190


namespace complement_of_37_12_l2641_264157

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_37_12 :
  let a : Angle := ⟨37, 12⟩
  complement a = ⟨52, 48⟩ := by
  sorry

end complement_of_37_12_l2641_264157


namespace initial_donuts_l2641_264155

theorem initial_donuts (remaining : ℕ) (missing_percent : ℚ) : 
  remaining = 9 → missing_percent = 70/100 → 
  (1 - missing_percent) * 30 = remaining :=
by sorry

end initial_donuts_l2641_264155


namespace sum_of_absolute_values_zero_l2641_264188

theorem sum_of_absolute_values_zero (a b : ℝ) : 
  |a + 3| + |2*b - 4| = 0 → a + b = -1 := by sorry

end sum_of_absolute_values_zero_l2641_264188


namespace exam_day_percentage_l2641_264136

/-- Proves that 70% of students took the exam on the assigned day given the conditions of the problem -/
theorem exam_day_percentage :
  ∀ (x : ℝ),
  (x ≥ 0) →
  (x ≤ 100) →
  (0.6 * x + 0.9 * (100 - x) = 69) →
  x = 70 := by
  sorry

end exam_day_percentage_l2641_264136


namespace solution_set_l2641_264138

theorem solution_set (x y z : ℝ) : 
  x^2 = y^2 + z^2 ∧ 
  x^2024 = y^2024 + z^2024 ∧ 
  x^2025 = y^2025 + z^2025 →
  ((y = x ∧ z = 0) ∨ (y = -x ∧ z = 0) ∨ (y = 0 ∧ z = x) ∨ (y = 0 ∧ z = -x)) := by
sorry

end solution_set_l2641_264138


namespace x_times_one_minus_f_equals_one_l2641_264169

/-- Given x = (3 + √8)^1000, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 1000
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end x_times_one_minus_f_equals_one_l2641_264169


namespace evaluate_expression_l2641_264125

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  (3 + x * (3 + x) - 3^2 + x) / (x - 3 + x^2 - x) = -3/2 := by
  sorry

end evaluate_expression_l2641_264125


namespace locus_of_circumscribed_rectangles_centers_l2641_264101

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  center : Point2D
  width : ℝ
  height : ℝ

/-- Represents a curvilinear triangle formed by semicircles -/
structure CurvilinearTriangle where
  midpoints : Triangle  -- Represents the triangle formed by midpoints of the original triangle

/-- Checks if a triangle is acute-angled -/
def isAcuteTriangle (t : Triangle) : Prop :=
  sorry  -- Definition of acute triangle

/-- Checks if a rectangle is circumscribed around a triangle -/
def isCircumscribed (r : Rectangle) (t : Triangle) : Prop :=
  sorry  -- Definition of circumscribed rectangle

/-- Computes the midpoints of a triangle -/
def midpoints (t : Triangle) : Triangle :=
  sorry  -- Computation of midpoints

/-- Checks if a point is on the locus (curvilinear triangle) -/
def isOnLocus (p : Point2D) (ct : CurvilinearTriangle) : Prop :=
  sorry  -- Definition of being on the locus

/-- The main theorem -/
theorem locus_of_circumscribed_rectangles_centers 
  (t : Triangle) (h : isAcuteTriangle t) :
  ∀ (r : Rectangle), isCircumscribed r t → 
    isOnLocus r.center (CurvilinearTriangle.mk (midpoints t)) :=
  sorry

end locus_of_circumscribed_rectangles_centers_l2641_264101


namespace arithmetic_sequence_general_term_l2641_264109

/-- Given an arithmetic sequence {a_n} where n ∈ ℕ+, if a_n + a_{n+2} = 4n + 6,
    then a_n = 2n + 1 for all n ∈ ℕ+ -/
theorem arithmetic_sequence_general_term
  (a : ℕ+ → ℝ)  -- a is a function from positive naturals to reals
  (h : ∀ n : ℕ+, a n + a (n + 2) = 4 * n + 6) :  -- given condition
  ∀ n : ℕ+, a n = 2 * n + 1 :=  -- conclusion to prove
by sorry

end arithmetic_sequence_general_term_l2641_264109


namespace trig_expressions_l2641_264119

theorem trig_expressions (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6/11 ∧
  (1/4) * Real.sin α ^ 2 + (1/3) * Real.sin α * Real.cos α + (1/2) * Real.cos α ^ 2 = 13/30 := by
  sorry

end trig_expressions_l2641_264119


namespace discount_percentage_calculation_l2641_264103

/-- Given an article with a marked price and cost price, calculate the discount percentage. -/
theorem discount_percentage_calculation
  (marked_price : ℝ)
  (cost_price : ℝ)
  (h1 : cost_price = 0.64 * marked_price)
  (h2 : (cost_price * 1.375 - cost_price) / cost_price = 0.375) :
  (marked_price - cost_price * 1.375) / marked_price = 0.12 := by
  sorry

#check discount_percentage_calculation

end discount_percentage_calculation_l2641_264103


namespace animals_to_shore_l2641_264131

theorem animals_to_shore (initial_sheep initial_cows initial_dogs : ℕ) 
  (drowned_sheep : ℕ) (h1 : initial_sheep = 20) (h2 : initial_cows = 10) 
  (h3 : initial_dogs = 14) (h4 : drowned_sheep = 3) 
  (h5 : 2 * drowned_sheep = initial_cows - (initial_cows - 2 * drowned_sheep)) :
  initial_sheep - drowned_sheep + (initial_cows - 2 * drowned_sheep) + initial_dogs = 35 := by
  sorry

end animals_to_shore_l2641_264131


namespace basketball_competition_probabilities_l2641_264189

/-- Represents a team in the basketball competition -/
inductive Team : Type
| A
| B
| C

/-- The probability of one team winning against another -/
def win_probability (winner loser : Team) : ℚ :=
  match winner, loser with
  | Team.A, Team.B => 2/3
  | Team.A, Team.C => 2/3
  | Team.B, Team.C => 1/2
  | Team.B, Team.A => 1/3
  | Team.C, Team.A => 1/3
  | Team.C, Team.B => 1/2
  | _, _ => 0

/-- Team A gets a bye in the first match -/
def first_match_bye : Team := Team.A

/-- The probability that Team B is eliminated after the first three matches -/
def prob_b_eliminated_three_matches : ℚ := 11/36

/-- The probability that Team A wins the championship in only four matches -/
def prob_a_wins_four_matches : ℚ := 8/27

/-- The probability that a fifth match is needed -/
def prob_fifth_match_needed : ℚ := 35/54

theorem basketball_competition_probabilities :
  (prob_b_eliminated_three_matches = 11/36) ∧
  (prob_a_wins_four_matches = 8/27) ∧
  (prob_fifth_match_needed = 35/54) :=
by sorry

end basketball_competition_probabilities_l2641_264189


namespace apples_picked_l2641_264173

theorem apples_picked (benny_apples dan_apples : ℕ) 
  (h1 : benny_apples = 2) 
  (h2 : dan_apples = 9) : 
  benny_apples + dan_apples = 11 := by
sorry

end apples_picked_l2641_264173


namespace hexagon_angles_arithmetic_progression_l2641_264114

theorem hexagon_angles_arithmetic_progression :
  ∃ (a d : ℝ), 
    (∀ i : Fin 6, 0 ≤ a + i * d ∧ a + i * d ≤ 180) ∧ 
    (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 720) ∧
    (∃ i : Fin 6, a + i * d = 120) := by
  sorry

end hexagon_angles_arithmetic_progression_l2641_264114


namespace subtraction_difference_l2641_264107

theorem subtraction_difference (x : ℝ) : (x - 2152) - (x - 1264) = 888 := by
  sorry

end subtraction_difference_l2641_264107


namespace point_B_coordinates_l2641_264141

-- Define the coordinate system
def Point := ℝ × ℝ

-- Define point A
def A : Point := (2, 1)

-- Define the length of AB
def AB_length : ℝ := 4

-- Define a function to check if a point is on the line parallel to x-axis passing through A
def on_parallel_line (p : Point) : Prop :=
  p.2 = A.2

-- Define a function to check if the distance between two points is correct
def correct_distance (p : Point) : Prop :=
  (p.1 - A.1)^2 + (p.2 - A.2)^2 = AB_length^2

-- Theorem statement
theorem point_B_coordinates :
  ∃ (B : Point), on_parallel_line B ∧ correct_distance B ∧
  (B = (6, 1) ∨ B = (-2, 1)) :=
sorry

end point_B_coordinates_l2641_264141


namespace orange_problem_l2641_264195

theorem orange_problem (total : ℕ) (ripe_fraction : ℚ) (eaten_ripe_fraction : ℚ) (eaten_unripe_fraction : ℚ) :
  total = 96 →
  ripe_fraction = 1/2 →
  eaten_ripe_fraction = 1/4 →
  eaten_unripe_fraction = 1/8 →
  (total : ℚ) * ripe_fraction * (1 - eaten_ripe_fraction) +
  (total : ℚ) * (1 - ripe_fraction) * (1 - eaten_unripe_fraction) = 78 :=
by sorry

end orange_problem_l2641_264195


namespace faster_train_speed_faster_train_speed_is_10_l2641_264181

theorem faster_train_speed 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (speed_ratio : ℝ) : ℝ :=
  let slower_speed := (2 * train_length) / (crossing_time * (1 + speed_ratio))
  let faster_speed := speed_ratio * slower_speed
  faster_speed

theorem faster_train_speed_is_10 :
  faster_train_speed 200 30 3 = 10 := by sorry

end faster_train_speed_faster_train_speed_is_10_l2641_264181


namespace slower_speed_calculation_l2641_264199

/-- Given a person who walks 30 km at a slower speed and could have walked 45 km at 15 km/hr 
    in the same amount of time, prove that the slower speed is 10 km/hr. -/
theorem slower_speed_calculation (x : ℝ) (h1 : x > 0) : 
  (30 / x = 45 / 15) → x = 10 := by
  sorry

end slower_speed_calculation_l2641_264199


namespace train_distance_l2641_264183

/-- Proves that a train traveling at 10 m/s for 8 seconds covers 80 meters. -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 10)
  (h2 : time = 8)
  (h3 : distance = speed * time) : 
  distance = 80 := by
  sorry

end train_distance_l2641_264183


namespace joel_contributed_22_toys_l2641_264152

/-- The number of toys Joel contributed to the donation -/
def joels_toys (toys_from_friends : ℕ) (total_toys : ℕ) : ℕ :=
  2 * ((total_toys - toys_from_friends) / 3)

/-- Theorem stating that Joel contributed 22 toys -/
theorem joel_contributed_22_toys : 
  joels_toys 75 108 = 22 := by
  sorry

end joel_contributed_22_toys_l2641_264152


namespace total_cost_after_discount_l2641_264143

def child_ticket_price : ℚ := 4.25
def adult_ticket_price : ℚ := child_ticket_price + 3.5
def senior_ticket_price : ℚ := adult_ticket_price - 1.75
def discount_per_5_tickets : ℚ := 3
def num_adult_tickets : ℕ := 2
def num_child_tickets : ℕ := 4
def num_senior_tickets : ℕ := 1

def total_ticket_cost : ℚ :=
  num_adult_tickets * adult_ticket_price +
  num_child_tickets * child_ticket_price +
  num_senior_tickets * senior_ticket_price

def total_tickets : ℕ := num_adult_tickets + num_child_tickets + num_senior_tickets

def discount_amount : ℚ := (total_tickets / 5 : ℚ) * discount_per_5_tickets

theorem total_cost_after_discount :
  total_ticket_cost - discount_amount = 35.5 := by sorry

end total_cost_after_discount_l2641_264143


namespace solutions_for_15_l2641_264148

/-- The number of different integer solutions for |x| + |y| = n -/
def numSolutions (n : ℕ) : ℕ :=
  4 * n

theorem solutions_for_15 : numSolutions 15 = 60 := by
  sorry

end solutions_for_15_l2641_264148


namespace max_area_triangle_l2641_264124

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3)

theorem max_area_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + B + C = Real.pi →
  a > 0 ∧ b > 0 ∧ c > 0 →
  g (A / 2) = 1 / 2 →
  a = 1 →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  (1 / 2 * b * c * Real.sin A) ≤ (2 + Real.sqrt 3) / 4 :=
by sorry

end max_area_triangle_l2641_264124


namespace principal_is_20000_l2641_264122

/-- Calculates the principal amount given the interest rate, time, and total interest -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 20000 -/
theorem principal_is_20000 : 
  let rate : ℚ := 12
  let time : ℕ := 3
  let interest : ℚ := 7200
  calculate_principal rate time interest = 20000 := by
  sorry

end principal_is_20000_l2641_264122


namespace unique_solution_l2641_264135

/-- Represents the denomination of a coin -/
inductive Coin : Type
  | One : Coin
  | Two : Coin
  | Five : Coin
  | Ten : Coin
  | Twenty : Coin

/-- The value of a coin in forints -/
def coin_value : Coin → Nat
  | Coin.One => 1
  | Coin.Two => 2
  | Coin.Five => 5
  | Coin.Ten => 10
  | Coin.Twenty => 20

/-- Represents the count of each coin type -/
structure CoinCount where
  one : Nat
  two : Nat
  five : Nat
  ten : Nat
  twenty : Nat

/-- The given coin count from the problem -/
def problem_coin_count : CoinCount :=
  { one := 3, two := 9, five := 5, ten := 6, twenty := 3 }

/-- Check if a number can be represented by the given coin count -/
def can_represent (n : Nat) (cc : CoinCount) : Prop :=
  ∃ (a b c d e : Nat),
    a ≤ cc.twenty ∧ b ≤ cc.ten ∧ c ≤ cc.five ∧ d ≤ cc.two ∧ e ≤ cc.one ∧
    n = a * 20 + b * 10 + c * 5 + d * 2 + e * 1

/-- The set of drawn numbers -/
def drawn_numbers : Finset Nat :=
  {34, 33, 29, 19, 18, 17, 16}

/-- The theorem to be proved -/
theorem unique_solution :
  (∀ n ∈ drawn_numbers, n ≤ 35) ∧
  (drawn_numbers.card = 7) ∧
  (∀ n ∈ drawn_numbers, can_represent n problem_coin_count) ∧
  (∀ s : Finset Nat, s ≠ drawn_numbers →
    s.card = 7 →
    (∀ n ∈ s, n ≤ 35) →
    (∀ n ∈ s, can_represent n problem_coin_count) →
    False) :=
by sorry

end unique_solution_l2641_264135


namespace face_value_in_product_l2641_264193

/-- Given a number with specific local values for its digits and their product, 
    prove the face value of a digit with a given local value in the product. -/
theorem face_value_in_product (n : ℕ) (product : ℕ) 
  (local_value_6 : ℕ) (local_value_8 : ℕ) :
  n = 7098060 →
  local_value_6 = 6000 →
  local_value_8 = 80 →
  product = local_value_6 * local_value_8 →
  (product / 1000) % 10 = 6 →
  (product / 1000) % 10 = 6 := by
  sorry

end face_value_in_product_l2641_264193


namespace erics_chickens_l2641_264117

/-- The number of chickens on Eric's farm -/
def num_chickens : ℕ := 4

/-- The number of eggs each chicken lays per day -/
def eggs_per_chicken_per_day : ℕ := 3

/-- The number of days Eric collected eggs -/
def days_collected : ℕ := 3

/-- The total number of eggs collected -/
def total_eggs_collected : ℕ := 36

theorem erics_chickens :
  num_chickens * eggs_per_chicken_per_day * days_collected = total_eggs_collected :=
by sorry

end erics_chickens_l2641_264117


namespace empty_solution_set_implies_a_range_l2641_264178

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(abs (x + 2) + abs (x - 1) < a)) → a ∈ Set.Iic 3 := by
  sorry

end empty_solution_set_implies_a_range_l2641_264178


namespace cubic_expression_value_l2641_264160

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : 
  m^3 + 2*m^2 - 2001 = -2000 := by
  sorry

end cubic_expression_value_l2641_264160


namespace rectangular_prism_volume_l2641_264150

theorem rectangular_prism_volume (l w h : ℝ) 
  (face1 : l * w = 10)
  (face2 : w * h = 14)
  (face3 : l * h = 35) :
  l * w * h = 70 := by
  sorry

end rectangular_prism_volume_l2641_264150


namespace f_increasing_when_a_zero_f_odd_when_a_one_f_domain_iff_a_lt_two_l2641_264100

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4^x - 1) / (4^x + 1 - a * 2^x)

-- Theorem 1: When a = 0, f is increasing
theorem f_increasing_when_a_zero : 
  ∀ x y : ℝ, x < y → f 0 x < f 0 y :=
sorry

-- Theorem 2: When a = 1, f is odd
theorem f_odd_when_a_one :
  ∀ x : ℝ, f 1 (-x) = -(f 1 x) :=
sorry

-- Theorem 3: Domain of f is R iff a < 2
theorem f_domain_iff_a_lt_two :
  ∀ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a < 2 :=
sorry

end f_increasing_when_a_zero_f_odd_when_a_one_f_domain_iff_a_lt_two_l2641_264100


namespace brody_calculator_theorem_l2641_264102

def calculator_problem (total_battery : ℝ) (used_fraction : ℝ) (exam_duration : ℝ) : Prop :=
  let remaining_before_exam := total_battery * (1 - used_fraction)
  let remaining_after_exam := remaining_before_exam - exam_duration
  remaining_after_exam = 13

theorem brody_calculator_theorem :
  calculator_problem 60 (3/4) 2 := by
  sorry

end brody_calculator_theorem_l2641_264102


namespace max_value_of_2x_plus_y_l2641_264156

theorem max_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y/2 + 1/x + 8/y = 10) : 
  ∃ (z : ℝ), z = 2*x + y ∧ ∀ (w : ℝ), (∃ (a b : ℝ) (ha : a > 0) (hb : b > 0), 
    w = 2*a + b ∧ a + b/2 + 1/a + 8/b = 10) → w ≤ z :=
by sorry

end max_value_of_2x_plus_y_l2641_264156


namespace largest_integer_with_remainder_l2641_264144

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
  by sorry

end largest_integer_with_remainder_l2641_264144


namespace christmas_tree_ornaments_l2641_264197

/-- The number of ornaments Pilyulkin hung on the tree -/
def pilyulkin_ornaments : ℕ := 3

/-- The number of ornaments Guslya hung on the tree -/
def guslya_ornaments : ℕ := 2 * pilyulkin_ornaments

/-- The number of ornaments Toropyzhka hung on the tree -/
def toropyzhka_ornaments : ℕ := pilyulkin_ornaments + 15

theorem christmas_tree_ornaments :
  guslya_ornaments = 2 * pilyulkin_ornaments ∧
  toropyzhka_ornaments = pilyulkin_ornaments + 15 ∧
  toropyzhka_ornaments = 2 * (guslya_ornaments + pilyulkin_ornaments) ∧
  pilyulkin_ornaments + guslya_ornaments + toropyzhka_ornaments = 27 := by
  sorry

end christmas_tree_ornaments_l2641_264197


namespace ships_converge_l2641_264158

/-- Represents a ship with a given round trip duration -/
structure Ship where
  roundTripDays : ℕ

/-- Represents the fleet of ships -/
def Fleet : List Ship := [
  { roundTripDays := 2 },
  { roundTripDays := 3 },
  { roundTripDays := 5 }
]

/-- The number of days after which all ships converge -/
def convergenceDays : ℕ := 30

/-- Theorem stating that the ships converge after the specified number of days -/
theorem ships_converge :
  ∀ (ship : Ship), ship ∈ Fleet → convergenceDays % ship.roundTripDays = 0 := by
  sorry

#check ships_converge

end ships_converge_l2641_264158


namespace tomatoes_sold_on_saturday_l2641_264163

theorem tomatoes_sold_on_saturday (initial_shipment : ℕ) (rotted_amount : ℕ) (final_amount : ℕ) :
  initial_shipment = 1000 →
  rotted_amount = 200 →
  final_amount = 2500 →
  ∃ (sold_amount : ℕ),
    sold_amount = 300 ∧
    final_amount = initial_shipment - sold_amount - rotted_amount + 2 * initial_shipment :=
by sorry

end tomatoes_sold_on_saturday_l2641_264163


namespace ceiling_of_negative_two_point_four_l2641_264113

theorem ceiling_of_negative_two_point_four :
  ⌈(-2.4 : ℝ)⌉ = -2 := by sorry

end ceiling_of_negative_two_point_four_l2641_264113


namespace sphere_volume_from_surface_area_l2641_264182

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), r > 0 → 4 * π * r^2 = 400 * π → (4 / 3) * π * r^3 = (4000 / 3) * π :=
by
  sorry

end sphere_volume_from_surface_area_l2641_264182


namespace equation_solution_l2641_264168

theorem equation_solution :
  ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end equation_solution_l2641_264168


namespace got_percentage_is_fifty_percent_l2641_264105

/-- Represents the vote counts for three books -/
structure VoteCounts where
  got : ℕ  -- Game of Thrones
  twi : ℕ  -- Twilight
  aotd : ℕ  -- The Art of the Deal

/-- Calculates the altered vote counts after throwing away votes -/
def alterVotes (vc : VoteCounts) : VoteCounts :=
  { got := vc.got,
    twi := vc.twi / 2,
    aotd := vc.aotd - (vc.aotd * 4 / 5) }

/-- Calculates the percentage of votes for Game of Thrones after alteration -/
def gotPercentage (vc : VoteCounts) : ℚ :=
  let altered := alterVotes vc
  altered.got * 100 / (altered.got + altered.twi + altered.aotd)

/-- Theorem stating that for the given vote counts, the percentage of
    altered votes for Game of Thrones is 50% -/
theorem got_percentage_is_fifty_percent :
  let original := VoteCounts.mk 10 12 20
  gotPercentage original = 50 := by sorry

end got_percentage_is_fifty_percent_l2641_264105


namespace number_of_zeros_equal_l2641_264175

/-- f(n) denotes the number of 0's in the binary representation of a positive integer n -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem stating that the number of 0's in the binary representation of 8n + 7 
    is equal to the number of 0's in the binary representation of 4n + 3 -/
theorem number_of_zeros_equal (n : ℕ+) : f (8 * n + 7) = f (4 * n + 3) := by
  sorry

end number_of_zeros_equal_l2641_264175


namespace greatest_consecutive_sum_48_l2641_264194

/-- The sum of N consecutive integers starting from a -/
def sum_consecutive (a : ℤ) (N : ℕ) : ℤ := N * (2 * a + N - 1) / 2

/-- The proposition that 96 is the greatest number of consecutive integers whose sum is 48 -/
theorem greatest_consecutive_sum_48 :
  ∀ N : ℕ, (∃ a : ℤ, sum_consecutive a N = 48) → N ≤ 96 :=
by sorry

end greatest_consecutive_sum_48_l2641_264194


namespace exists_common_tiling_l2641_264139

/-- Represents a domino type with integer dimensions -/
structure Domino where
  length : ℤ
  width : ℤ

/-- Checks if a rectangle can be tiled by a given domino type -/
def canTile (d : Domino) (rectLength rectWidth : ℤ) : Prop :=
  rectLength ≥ max 1 (2 * d.length) ∧ rectWidth % (2 * d.width) = 0

/-- Proves the existence of a rectangle that can be tiled by either of two domino types -/
theorem exists_common_tiling (d1 d2 : Domino) : 
  ∃ (rectLength rectWidth : ℤ), 
    canTile d1 rectLength rectWidth ∧ canTile d2 rectLength rectWidth :=
by
  sorry

end exists_common_tiling_l2641_264139


namespace yard_length_l2641_264137

theorem yard_length (n : ℕ) (d : ℝ) (h1 : n = 26) (h2 : d = 24) : 
  (n - 1) * d = 600 := by
  sorry

end yard_length_l2641_264137


namespace water_usage_calculation_l2641_264167

/-- Water pricing policy and usage calculation -/
theorem water_usage_calculation (m : ℝ) (usage : ℝ) (payment : ℝ) : 
  (m > 0) →
  (usage > 0) →
  (payment = if usage ≤ 10 then m * usage else 10 * m + 2 * m * (usage - 10)) →
  (payment = 16 * m) →
  (usage = 13) :=
by sorry

end water_usage_calculation_l2641_264167


namespace f_digit_sum_properties_l2641_264165

/-- The function f(n) = 3n^2 + n + 1 -/
def f (n : ℕ+) : ℕ := 3 * n.val^2 + n.val + 1

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the smallest sum of digits and existence of 1999 sum -/
theorem f_digit_sum_properties :
  (∃ (n : ℕ+), sum_of_digits (f n) = 3) ∧ 
  (∀ (n : ℕ+), sum_of_digits (f n) ≥ 3) ∧
  (∃ (n : ℕ+), sum_of_digits (f n) = 1999) :=
sorry

end f_digit_sum_properties_l2641_264165


namespace tom_apple_slices_l2641_264121

theorem tom_apple_slices (total_apples : ℕ) (slices_per_apple : ℕ) (slices_left : ℕ) :
  total_apples = 2 →
  slices_per_apple = 8 →
  slices_left = 5 →
  (∃ (slices_given : ℕ),
    slices_given + 2 * slices_left = total_apples * slices_per_apple ∧
    slices_given = (3 : ℚ) / 8 * (total_apples * slices_per_apple : ℚ)) :=
by sorry

end tom_apple_slices_l2641_264121


namespace sandwich_fraction_l2641_264126

theorem sandwich_fraction (total : ℝ) (ticket_fraction : ℝ) (book_fraction : ℝ) (leftover : ℝ) 
  (h1 : total = 90)
  (h2 : ticket_fraction = 1/6)
  (h3 : book_fraction = 1/2)
  (h4 : leftover = 12) :
  ∃ (sandwich_fraction : ℝ), 
    sandwich_fraction * total + ticket_fraction * total + book_fraction * total + leftover = total ∧ 
    sandwich_fraction = 1/5 := by
  sorry

end sandwich_fraction_l2641_264126


namespace remainder_property_l2641_264115

theorem remainder_property (x y u v : ℕ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x = u * y + v) (h4 : 0 ≤ v) (h5 : v < y) : 
  (x + 3 * u * y) % y = v := by
sorry

end remainder_property_l2641_264115


namespace problem_1_problem_2_l2641_264166

-- Problem 1
theorem problem_1 : Real.sqrt 8 - 2 * Real.sin (π / 4) + |1 - Real.sqrt 2| + (1 / 2)⁻¹ = 2 * Real.sqrt 2 + 1 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 1 ∧ ∀ x : ℝ, x^2 + 4*x - 5 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end problem_1_problem_2_l2641_264166


namespace student_multiplication_problem_l2641_264172

theorem student_multiplication_problem (initial_number : ℕ) (final_result : ℕ) : 
  initial_number = 48 → final_result = 102 → ∃ (x : ℕ), initial_number * x - 138 = final_result ∧ x = 5 := by
  sorry

end student_multiplication_problem_l2641_264172


namespace difference_of_squares_l2641_264179

theorem difference_of_squares (x : ℝ) : x^2 - 25 = (x + 5) * (x - 5) := by
  sorry

end difference_of_squares_l2641_264179


namespace gas_price_difference_l2641_264146

/-- The difference between actual and expected gas prices -/
theorem gas_price_difference (actual_gallons : ℕ) (actual_price : ℕ) (expected_gallons : ℕ) :
  actual_gallons = 10 →
  actual_price = 150 →
  expected_gallons = 12 →
  actual_price - (actual_gallons * actual_price / expected_gallons) = 25 := by
sorry

end gas_price_difference_l2641_264146


namespace camping_activities_count_l2641_264130

/-- The number of campers who went rowing and hiking in total, considering both morning and afternoon sessions -/
def total_rowing_and_hiking (total_campers : ℕ)
  (morning_rowing : ℕ) (morning_hiking : ℕ) (morning_swimming : ℕ)
  (afternoon_rowing : ℕ) (afternoon_hiking : ℕ) : ℕ :=
  morning_rowing + afternoon_rowing + morning_hiking + afternoon_hiking

/-- Theorem stating that the total number of campers who went rowing and hiking is 79 -/
theorem camping_activities_count
  (total_campers : ℕ)
  (morning_rowing : ℕ) (morning_hiking : ℕ) (morning_swimming : ℕ)
  (afternoon_rowing : ℕ) (afternoon_hiking : ℕ)
  (h1 : total_campers = 80)
  (h2 : morning_rowing = 41)
  (h3 : morning_hiking = 4)
  (h4 : morning_swimming = 15)
  (h5 : afternoon_rowing = 26)
  (h6 : afternoon_hiking = 8) :
  total_rowing_and_hiking total_campers morning_rowing morning_hiking morning_swimming afternoon_rowing afternoon_hiking = 79 := by
  sorry

#check camping_activities_count

end camping_activities_count_l2641_264130


namespace union_complement_theorem_l2641_264149

def I : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {2}

theorem union_complement_theorem :
  B ∪ (I \ A) = {2, 4, 5} := by sorry

end union_complement_theorem_l2641_264149


namespace amy_bought_21_tickets_l2641_264118

/-- Calculates the number of tickets Amy bought at the fair -/
def tickets_bought (initial_tickets total_tickets : ℕ) : ℕ :=
  total_tickets - initial_tickets

/-- Proves that Amy bought 21 tickets at the fair -/
theorem amy_bought_21_tickets :
  tickets_bought 33 54 = 21 := by
  sorry

end amy_bought_21_tickets_l2641_264118


namespace area_ratio_hexagons_l2641_264140

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A regular hexagon -/
structure RegularHexagon :=
  (center : Point)
  (sideLength : ℝ)

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (center : Point)
  (sideLength : ℝ)

/-- The hexagon ABCD -/
def hexagonABCD : RegularHexagon := sorry

/-- The equilateral triangles constructed on the sides of ABCD -/
def trianglesOnABCD : List EquilateralTriangle := sorry

/-- The hexagon EFGHIJ formed by the centers of the equilateral triangles -/
def hexagonEFGHIJ : RegularHexagon := sorry

/-- The area of a regular hexagon -/
def areaRegularHexagon (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The ratio of the area of hexagon EFGHIJ to the area of hexagon ABCD is 4/3 -/
theorem area_ratio_hexagons :
  (areaRegularHexagon hexagonEFGHIJ) / (areaRegularHexagon hexagonABCD) = 4/3 := by
  sorry

end area_ratio_hexagons_l2641_264140


namespace factorization_1_factorization_2_l2641_264147

/-- Proves the factorization of x²y - 4xy + 4y -/
theorem factorization_1 (x y : ℝ) : x^2*y - 4*x*y + 4*y = y*(x-2)^2 := by
  sorry

/-- Proves the factorization of x² - 4y² -/
theorem factorization_2 (x y : ℝ) : x^2 - 4*y^2 = (x+2*y)*(x-2*y) := by
  sorry

end factorization_1_factorization_2_l2641_264147


namespace student_failed_marks_l2641_264127

def total_marks : ℕ := 500
def passing_percentage : ℚ := 33 / 100
def student_marks : ℕ := 125

theorem student_failed_marks :
  (total_marks * passing_percentage).floor - student_marks = 40 :=
by sorry

end student_failed_marks_l2641_264127


namespace solution_set_when_m_is_2_range_of_m_for_all_real_solution_l2641_264164

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Part I
theorem solution_set_when_m_is_2 :
  ∀ x : ℝ, f 2 x < 0 ↔ -2 < x ∧ x < 1 := by sorry

-- Part II
theorem range_of_m_for_all_real_solution :
  ∀ m : ℝ, (∀ x : ℝ, f m x ≥ -1) ↔ -3 ≤ m ∧ m ≤ 1 := by sorry

end solution_set_when_m_is_2_range_of_m_for_all_real_solution_l2641_264164


namespace inequality_range_l2641_264132

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) → 
  a ≥ -6 := by
  sorry

end inequality_range_l2641_264132


namespace average_divisible_by_seven_l2641_264110

theorem average_divisible_by_seven : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 12 < n ∧ n < 150 ∧ n % 7 = 0) ∧ 
  (∀ n, 12 < n → n < 150 → n % 7 = 0 → n ∈ S) ∧
  (S.sum id / S.card : ℚ) = 161/2 := by
  sorry

end average_divisible_by_seven_l2641_264110


namespace graveyard_bones_count_l2641_264161

/-- Represents the number of bones in a skeleton based on its type -/
def bonesInSkeleton (type : String) : ℕ :=
  match type with
  | "woman" => 20
  | "man" => 25
  | "child" => 10
  | _ => 0

/-- Calculates the total number of bones in the graveyard -/
def totalBonesInGraveyard : ℕ :=
  let totalSkeletons : ℕ := 20
  let womenSkeletons : ℕ := totalSkeletons / 2
  let menSkeletons : ℕ := (totalSkeletons - womenSkeletons) / 2
  let childrenSkeletons : ℕ := totalSkeletons - womenSkeletons - menSkeletons
  
  womenSkeletons * bonesInSkeleton "woman" +
  menSkeletons * bonesInSkeleton "man" +
  childrenSkeletons * bonesInSkeleton "child"

theorem graveyard_bones_count :
  totalBonesInGraveyard = 375 := by
  sorry

#eval totalBonesInGraveyard

end graveyard_bones_count_l2641_264161


namespace cube_sum_is_90_l2641_264108

-- Define the type for cube face numbers
def CubeFaces := Fin 6 → ℝ

-- Define the property of consecutive numbers
def IsConsecutive (faces : CubeFaces) : Prop :=
  ∀ i j : Fin 6, i.val < j.val → faces j - faces i = j.val - i.val

-- Define the property of opposite faces summing to 30
def OppositeFacesSum30 (faces : CubeFaces) : Prop :=
  faces 0 + faces 5 = 30 ∧ faces 1 + faces 4 = 30 ∧ faces 2 + faces 3 = 30

-- Theorem statement
theorem cube_sum_is_90 (faces : CubeFaces) 
  (h1 : IsConsecutive faces) (h2 : OppositeFacesSum30 faces) : 
  (Finset.univ.sum faces) = 90 := by sorry

end cube_sum_is_90_l2641_264108


namespace yearly_dumpling_production_l2641_264185

/-- The monthly production of dumplings in kilograms -/
def monthly_production : ℝ := 182.88

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The yearly production of dumplings in kilograms -/
def yearly_production : ℝ := monthly_production * months_in_year

/-- Theorem stating that the yearly production of dumplings is 2194.56 kg -/
theorem yearly_dumpling_production :
  yearly_production = 2194.56 := by
  sorry

end yearly_dumpling_production_l2641_264185
