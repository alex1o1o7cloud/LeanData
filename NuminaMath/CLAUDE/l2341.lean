import Mathlib

namespace cylinder_j_value_l2341_234157

/-- The value of J for a cylinder with specific properties -/
theorem cylinder_j_value (h d r : ℝ) (j : ℝ) : 
  h > 0 → d > 0 → r > 0 →
  h = d →  -- Cylinder height equals diameter
  r = d / 2 →  -- Radius is half the diameter
  6 * 3^2 = 2 * π * r^2 + π * d * h →  -- Surface area of cylinder equals surface area of cube
  j * π / 6 = π * r^2 * h →  -- Volume of cylinder
  j = 324 * Real.sqrt π := by
sorry

end cylinder_j_value_l2341_234157


namespace three_zeros_condition_l2341_234132

-- Define the function f(x) = x^3 + ax + 2
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

-- Theorem statement
theorem three_zeros_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ a < -3 :=
sorry

end three_zeros_condition_l2341_234132


namespace average_girls_per_grade_l2341_234194

/-- Represents a grade with its student composition -/
structure Grade where
  girls : ℕ
  boys : ℕ
  clubGirls : ℕ
  clubBoys : ℕ

/-- The total number of grades -/
def totalGrades : ℕ := 3

/-- List of grades with their student composition -/
def grades : List Grade := [
  { girls := 28, boys := 35, clubGirls := 6, clubBoys := 6 },
  { girls := 45, boys := 42, clubGirls := 7, clubBoys := 8 },
  { girls := 38, boys := 51, clubGirls := 3, clubBoys := 7 }
]

/-- Calculate the total number of girls across all grades -/
def totalGirls : ℕ := (grades.map (·.girls)).sum

/-- Theorem: The average number of girls per grade is 37 -/
theorem average_girls_per_grade :
  totalGirls / totalGrades = 37 := by sorry

end average_girls_per_grade_l2341_234194


namespace next_event_occurrence_l2341_234105

/-- Represents the periodic event occurrence pattern -/
structure EventPattern where
  x : ℕ  -- Number of consecutive years the event occurs
  y : ℕ  -- Number of consecutive years of break

/-- Checks if the event occurs in a given year based on the pattern and a reference year -/
def eventOccurs (pattern : EventPattern) (referenceYear : ℕ) (year : ℕ) : Prop :=
  (year - referenceYear) % (pattern.x + pattern.y) < pattern.x

/-- The main theorem stating the next occurrence of the event after 2013 -/
theorem next_event_occurrence (pattern : EventPattern) : 
  (eventOccurs pattern 1964 1964) ∧
  (eventOccurs pattern 1964 1986) ∧
  (eventOccurs pattern 1964 1996) ∧
  (eventOccurs pattern 1964 2008) ∧
  (¬ eventOccurs pattern 1964 1976) ∧
  (¬ eventOccurs pattern 1964 1993) ∧
  (¬ eventOccurs pattern 1964 2006) ∧
  (¬ eventOccurs pattern 1964 2013) →
  ∀ year : ℕ, year > 2013 → eventOccurs pattern 1964 year → year ≥ 2018 :=
by
  sorry

#check next_event_occurrence

end next_event_occurrence_l2341_234105


namespace nonagon_diagonal_intersection_probability_l2341_234186

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- The probability that two randomly chosen diagonals of a regular nonagon intersect inside the nonagon -/
def intersectionProbability (n : RegularNonagon) : ℚ :=
  14 / 39

/-- Theorem: The probability that two randomly chosen diagonals of a regular nonagon intersect inside the nonagon is 14/39 -/
theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersectionProbability n = 14 / 39 := by
  sorry

end nonagon_diagonal_intersection_probability_l2341_234186


namespace profit_per_meter_l2341_234145

/-- Given the selling price and cost price of cloth, calculate the profit per meter -/
theorem profit_per_meter (total_meters : ℕ) (selling_price cost_per_meter : ℚ) :
  total_meters = 85 →
  selling_price = 8925 →
  cost_per_meter = 85 →
  (selling_price - total_meters * cost_per_meter) / total_meters = 20 := by
sorry

end profit_per_meter_l2341_234145


namespace triangle_existence_condition_l2341_234138

theorem triangle_existence_condition 
  (k : ℝ) (α : ℝ) (m_a : ℝ) 
  (h_k : k > 0) 
  (h_α : 0 < α ∧ α < π) 
  (h_m_a : m_a > 0) : 
  (∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = k ∧
    ∃ (β γ : ℝ), 
      0 < β ∧ 0 < γ ∧
      α + β + γ = π ∧
      m_a = (b * c * Real.sin α) / (b + c)) ↔ 
  m_a ≤ (k / 2) * ((1 - Real.sin (α / 2)) / Real.cos (α / 2)) :=
sorry

end triangle_existence_condition_l2341_234138


namespace range_of_x_range_of_a_l2341_234106

-- Part 1
theorem range_of_x (x : ℝ) :
  (x^2 - 4*x + 3 < 0) → ((x - 3)^2 < 1) → (2 < x ∧ x < 3) := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 ≥ 0 → (x - 3)^2 ≥ 1)) →
  (∃ x : ℝ, (x - 3)^2 ≥ 1 ∧ x^2 - 4*a*x + 3*a^2 < 0) →
  (4/3 ≤ a ∧ a ≤ 2) := by sorry

end range_of_x_range_of_a_l2341_234106


namespace inequality_proof_l2341_234161

theorem inequality_proof (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ x*y*z + 2 := by
  sorry

end inequality_proof_l2341_234161


namespace sin_70_degrees_l2341_234199

theorem sin_70_degrees (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by sorry

end sin_70_degrees_l2341_234199


namespace tan_beta_value_l2341_234153

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
  sorry

end tan_beta_value_l2341_234153


namespace victoria_work_hours_l2341_234143

/-- Calculates the number of hours worked per day given the total hours and number of weeks worked. -/
def hours_per_day (total_hours : ℕ) (weeks : ℕ) : ℚ :=
  total_hours / (weeks * 7)

/-- Theorem: Given 315 total hours worked over 5 weeks, the number of hours worked per day is 9. -/
theorem victoria_work_hours :
  hours_per_day 315 5 = 9 := by
  sorry

end victoria_work_hours_l2341_234143


namespace star_example_l2341_234152

-- Define the star operation
def star (a b c d : ℚ) : ℚ := a * c * (d / b)

-- Theorem statement
theorem star_example : star (5/9) (4/6) = 40/3 := by sorry

end star_example_l2341_234152


namespace inequality_system_solution_set_l2341_234156

theorem inequality_system_solution_set :
  let S := {x : ℝ | -3 * (x - 2) ≥ 4 - x ∧ (1 + 2 * x) / 3 > x - 1}
  S = {x : ℝ | x ≤ 1} := by
  sorry

end inequality_system_solution_set_l2341_234156


namespace average_score_is_8_1_l2341_234116

theorem average_score_is_8_1 (shooters_7 shooters_8 shooters_9 shooters_10 : ℕ)
  (h1 : shooters_7 = 4)
  (h2 : shooters_8 = 2)
  (h3 : shooters_9 = 3)
  (h4 : shooters_10 = 1) :
  let total_points := 7 * shooters_7 + 8 * shooters_8 + 9 * shooters_9 + 10 * shooters_10
  let total_shooters := shooters_7 + shooters_8 + shooters_9 + shooters_10
  (total_points : ℚ) / total_shooters = 81 / 10 :=
by sorry

end average_score_is_8_1_l2341_234116


namespace parabola_point_distance_to_x_axis_l2341_234109

/-- Prove that for a point on a specific parabola with a given distance to the focus,
    its distance to the x-axis is 15/16 -/
theorem parabola_point_distance_to_x_axis 
  (x₀ y₀ : ℝ) -- Coordinates of point M
  (h_parabola : x₀^2 = (1/4) * y₀) -- M is on the parabola
  (h_focus_dist : (x₀^2 + (y₀ - 1/16)^2) = 1) -- Distance from M to focus is 1
  : |y₀| = 15/16 := by
  sorry

end parabola_point_distance_to_x_axis_l2341_234109


namespace temporary_employee_percentage_is_32_l2341_234197

/-- Represents the composition of workers in a factory -/
structure WorkforceComposition where
  technician_ratio : ℝ
  non_technician_ratio : ℝ
  technician_permanent_ratio : ℝ
  non_technician_permanent_ratio : ℝ

/-- Calculates the percentage of temporary employees given a workforce composition -/
def temporary_employee_percentage (wc : WorkforceComposition) : ℝ :=
  100 - (wc.technician_ratio * wc.technician_permanent_ratio + 
         wc.non_technician_ratio * wc.non_technician_permanent_ratio)

/-- The main theorem stating the percentage of temporary employees -/
theorem temporary_employee_percentage_is_32 (wc : WorkforceComposition) 
  (h1 : wc.technician_ratio = 80)
  (h2 : wc.non_technician_ratio = 20)
  (h3 : wc.technician_permanent_ratio = 80)
  (h4 : wc.non_technician_permanent_ratio = 20)
  (h5 : wc.technician_ratio + wc.non_technician_ratio = 100) :
  temporary_employee_percentage wc = 32 := by
  sorry

#eval temporary_employee_percentage ⟨80, 20, 80, 20⟩

end temporary_employee_percentage_is_32_l2341_234197


namespace pascal_interior_sum_l2341_234140

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The problem statement -/
theorem pascal_interior_sum :
  interior_sum 5 = 14 →
  interior_sum 6 = 30 →
  interior_sum 8 = 126 := by
sorry

end pascal_interior_sum_l2341_234140


namespace smallest_winning_number_l2341_234177

theorem smallest_winning_number : ∃ N : ℕ,
  N ≥ 0 ∧ N ≤ 999 ∧
  (∀ m : ℕ, m ≥ 0 ∧ m < N →
    (3*m < 1000 ∧
     3*m - 30 < 1000 ∧
     9*m - 90 < 1000 ∧
     9*m - 120 < 1000 ∧
     27*m - 360 < 1000 ∧
     27*m - 390 < 1000 ∧
     81*m - 1170 < 1000 ∧
     81*(m-1) - 1170 ≥ 1000)) ∧
  3*N < 1000 ∧
  3*N - 30 < 1000 ∧
  9*N - 90 < 1000 ∧
  9*N - 120 < 1000 ∧
  27*N - 360 < 1000 ∧
  27*N - 390 < 1000 ∧
  81*N - 1170 < 1000 ∧
  81*(N-1) - 1170 ≥ 1000 :=
by sorry

end smallest_winning_number_l2341_234177


namespace sphere_only_circular_views_l2341_234169

-- Define the geometric shapes
inductive Shape
  | Cuboid
  | Cylinder
  | Cone
  | Sphere

-- Define the views
inductive View
  | Front
  | Left
  | Top

-- Define a function to check if a view is circular for a given shape
def isCircularView (s : Shape) (v : View) : Prop :=
  match s, v with
  | Shape.Sphere, _ => True
  | Shape.Cylinder, View.Top => True
  | Shape.Cone, View.Top => True
  | _, _ => False

-- Define a function to check if all views are circular for a given shape
def allViewsCircular (s : Shape) : Prop :=
  isCircularView s View.Front ∧ isCircularView s View.Left ∧ isCircularView s View.Top

-- Theorem statement
theorem sphere_only_circular_views :
  ∀ s : Shape, allViewsCircular s ↔ s = Shape.Sphere :=
by sorry

end sphere_only_circular_views_l2341_234169


namespace quadratic_inequality_solution_set_l2341_234180

theorem quadratic_inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - 1 < 0) ↔ (-1 < k ∧ k ≤ 0) :=
by sorry

end quadratic_inequality_solution_set_l2341_234180


namespace quadratic_inequality_solution_l2341_234126

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 4*x > 12 ↔ x ∈ Set.Iio (-2) ∪ Set.Ioi 6 := by
  sorry

end quadratic_inequality_solution_l2341_234126


namespace cubic_polynomial_theorem_l2341_234119

-- Define the cubic polynomial whose roots are a, b, c
def cubic (x : ℝ) : ℝ := x^3 + 4*x^2 + 6*x + 9

-- Define the properties of P
def P_properties (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
  cubic a = 0 ∧ cubic b = 0 ∧ cubic c = 0 ∧
  P a = b + c ∧ P b = a + c ∧ P c = a + b ∧
  P (a + b + c) = -20

-- Theorem statement
theorem cubic_polynomial_theorem :
  ∀ (P : ℝ → ℝ) (a b c : ℝ),
  P_properties P a b c →
  (∀ x, P x = 16*x^3 + 64*x^2 + 90*x + 140) :=
sorry

end cubic_polynomial_theorem_l2341_234119


namespace basketball_handshakes_l2341_234121

/-- The number of handshakes in a basketball game scenario --/
def total_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let player_handshakes := team_size * team_size
  let referee_handshakes := (team_size * num_teams) * num_referees
  player_handshakes + referee_handshakes

/-- Theorem stating the total number of handshakes in the given scenario --/
theorem basketball_handshakes :
  total_handshakes 6 2 3 = 72 := by
  sorry

end basketball_handshakes_l2341_234121


namespace pet_shop_guinea_pigs_l2341_234174

/-- Given a pet shop with rabbits and guinea pigs, prove the number of guinea pigs. -/
theorem pet_shop_guinea_pigs (rabbit_count : ℕ) (ratio_rabbits : ℕ) (ratio_guinea_pigs : ℕ) 
  (h_ratio : ratio_rabbits = 5 ∧ ratio_guinea_pigs = 4)
  (h_rabbits : rabbit_count = 25) :
  (rabbit_count * ratio_guinea_pigs) / ratio_rabbits = 20 := by
sorry

end pet_shop_guinea_pigs_l2341_234174


namespace polar_to_cartesian_l2341_234139

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ + ρ - 3 * ρ * Real.cos θ - 3 = 0

-- Define the Cartesian equations
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

def line_equation (x : ℝ) : Prop :=
  x = -1

-- Theorem statement
theorem polar_to_cartesian :
  ∀ ρ θ x y : ℝ, 
    polar_equation ρ θ ↔ 
    (circle_equation x y ∨ line_equation x) ∧
    x = ρ * Real.cos θ ∧
    y = ρ * Real.sin θ :=
sorry

end polar_to_cartesian_l2341_234139


namespace four_Y_three_equals_negative_eleven_l2341_234123

/-- The Y operation defined for any two real numbers -/
def Y (x y : ℝ) : ℝ := x^2 - 3*x*y + y^2

/-- Theorem stating that 4 Y 3 equals -11 -/
theorem four_Y_three_equals_negative_eleven : Y 4 3 = -11 := by
  sorry

end four_Y_three_equals_negative_eleven_l2341_234123


namespace largest_prime_factors_difference_l2341_234188

theorem largest_prime_factors_difference (n : Nat) (h : n = 171689) : 
  ∃ (p q : Nat), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > q ∧ 
    (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p) ∧
    p ∣ n ∧ 
    q ∣ n ∧ 
    p - q = 282 :=
by sorry

end largest_prime_factors_difference_l2341_234188


namespace unique_solution_l2341_234183

theorem unique_solution : ∃! (x p : ℕ), 
  Prime p ∧ 
  x * (x + 1) * (x + 2) * (x + 3) = 1679^(p - 1) + 1680^(p - 1) + 1681^(p - 1) ∧
  x = 4 ∧ p = 2 := by
sorry

end unique_solution_l2341_234183


namespace max_value_implies_ratio_l2341_234187

/-- Given a cubic function f(x) with a maximum at x=1, prove that a/b = -2/3 --/
theorem max_value_implies_ratio (a b : ℝ) :
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x - a^2 - 7*a
  (∀ x, f x ≤ f 1) ∧ (f 1 = 10) →
  a / b = -2 / 3 :=
by sorry

end max_value_implies_ratio_l2341_234187


namespace total_students_calculation_l2341_234179

/-- Represents a high school with three years of students -/
structure HighSchool where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Represents a sample taken from the high school -/
structure Sample where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- The theorem stating the conditions and the conclusion about the total number of students -/
theorem total_students_calculation (school : HighSchool) (sample : Sample) :
  school.second_year = 300 →
  sample.first_year = 20 →
  sample.third_year = 10 →
  sample.first_year + sample.second_year + sample.third_year = 45 →
  (sample.first_year : ℚ) / sample.third_year = 2 →
  (sample.first_year : ℚ) / school.first_year = 
    (sample.second_year : ℚ) / school.second_year →
  (sample.second_year : ℚ) / school.second_year = 
    (sample.third_year : ℚ) / school.third_year →
  school.first_year + school.second_year + school.third_year = 900 := by
  sorry


end total_students_calculation_l2341_234179


namespace eighth_group_sample_l2341_234167

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (m : Nat) (k : Nat) : Nat :=
  (k - 1) * 10 + (m + k) % 10

/-- The problem statement as a theorem -/
theorem eighth_group_sample :
  ∀ m : Nat,
  m = 8 →
  systematicSample m 8 = 76 := by
  sorry

end eighth_group_sample_l2341_234167


namespace base_h_equation_l2341_234101

/-- Converts a base-h number to decimal --/
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

/-- Checks if a list of digits is valid in base h --/
def valid_digits (digits : List Nat) (h : Nat) : Prop :=
  ∀ d ∈ digits, d < h

theorem base_h_equation (h : Nat) : 
  h > 8 → 
  valid_digits [8, 6, 7, 4] h → 
  valid_digits [4, 3, 2, 9] h → 
  valid_digits [1, 3, 0, 0, 3] h → 
  to_decimal [8, 6, 7, 4] h + to_decimal [4, 3, 2, 9] h = to_decimal [1, 3, 0, 0, 3] h → 
  h = 10 :=
sorry

end base_h_equation_l2341_234101


namespace geometric_sequence_fourth_term_l2341_234196

/-- A geometric sequence with the given first three terms has its fourth term equal to -24 -/
theorem geometric_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (x : ℝ) 
  (h1 : a 1 = x)
  (h2 : a 2 = 3*x + 3)
  (h3 : a 3 = 6*x + 6)
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n+1) / a n = a 2 / a 1) :
  a 4 = -24 := by
  sorry

end geometric_sequence_fourth_term_l2341_234196


namespace age_ratio_this_year_l2341_234137

def yoongi_age_last_year : ℕ := 6
def grandfather_age_last_year : ℕ := 62

def yoongi_age_this_year : ℕ := yoongi_age_last_year + 1
def grandfather_age_this_year : ℕ := grandfather_age_last_year + 1

theorem age_ratio_this_year :
  grandfather_age_this_year / yoongi_age_this_year = 9 := by
  sorry

end age_ratio_this_year_l2341_234137


namespace not_p_sufficient_for_not_q_l2341_234155

/-- Definition of proposition p -/
def p (x : ℝ) : Prop := |3*x - 4| > 2

/-- Definition of proposition q -/
def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0

/-- Theorem stating that not p implies not q, but not q does not necessarily imply not p -/
theorem not_p_sufficient_for_not_q :
  (∃ x : ℝ, ¬(p x) ∧ ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x) :=
sorry

end not_p_sufficient_for_not_q_l2341_234155


namespace earnings_calculation_l2341_234192

/-- If a person spends 10% of their earnings and is left with $405, prove their total earnings were $450. -/
theorem earnings_calculation (spent_percentage : Real) (remaining_amount : Real) (total_earnings : Real) : 
  spent_percentage = 0.1 →
  remaining_amount = 405 →
  remaining_amount = (1 - spent_percentage) * total_earnings →
  total_earnings = 450 := by
sorry

end earnings_calculation_l2341_234192


namespace three_pairs_probability_l2341_234104

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (cards_per_rank : Nat)
  (h1 : cards = ranks * cards_per_rank)

/-- A poker hand -/
structure PokerHand :=
  (size : Nat)

/-- The probability of drawing a specific hand -/
def probability (deck : Deck) (hand : PokerHand) (valid_hands : Nat) : Rat :=
  valid_hands / (Nat.choose deck.cards hand.size)

/-- Theorem: Probability of drawing exactly three pairs in a 6-card hand -/
theorem three_pairs_probability (d : Deck) (h : PokerHand) : 
  d.cards = 52 → d.ranks = 13 → d.cards_per_rank = 4 → h.size = 6 → 
  probability d h ((Nat.choose d.ranks 3) * (Nat.choose d.cards_per_rank 2)^3) = 154/51845 := by
  sorry


end three_pairs_probability_l2341_234104


namespace power_of_product_l2341_234168

theorem power_of_product (a b : ℝ) : ((-3 * a^2 * b^3)^2) = 9 * a^4 * b^6 := by
  sorry

end power_of_product_l2341_234168


namespace binomial_coefficient_two_l2341_234166

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l2341_234166


namespace min_value_of_product_l2341_234120

/-- Given positive real numbers x₁, x₂, x₃, x₄ such that their sum is π,
    the product of (2sin²xᵢ + 1/sin²xᵢ) for i = 1 to 4 has a minimum value of 81. -/
theorem min_value_of_product (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0)
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) *
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) *
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) *
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 := by
  sorry

end min_value_of_product_l2341_234120


namespace min_initial_coins_l2341_234178

/-- Represents the game state at each round -/
structure GameState where
  huanhuan : ℕ
  lele : ℕ

/-- Represents the game with initial state and two rounds -/
structure Game where
  initial : GameState
  first_round : ℕ
  second_round : ℕ

/-- Checks if the game satisfies all the given conditions -/
def valid_game (g : Game) : Prop :=
  g.initial.huanhuan = 7 * g.initial.lele ∧
  g.initial.huanhuan + g.first_round = 6 * (g.initial.lele + g.first_round) ∧
  g.initial.huanhuan + g.first_round + g.second_round = 
    5 * (g.initial.lele + g.first_round + g.second_round)

/-- Theorem stating the minimum number of gold coins Huanhuan had at the beginning -/
theorem min_initial_coins (g : Game) (h : valid_game g) : g.initial.huanhuan ≥ 70 := by
  sorry

#check min_initial_coins

end min_initial_coins_l2341_234178


namespace arithmetic_sequence_common_difference_l2341_234117

/-- 
An arithmetic sequence is a sequence where the difference between 
any two consecutive terms is constant.
-/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/--
Given an arithmetic sequence a_n where a_5 = 10 and a_12 = 31,
the common difference d is equal to 3.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a5 : a 5 = 10) 
  (h_a12 : a 12 = 31) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 :=
sorry

end arithmetic_sequence_common_difference_l2341_234117


namespace investment_period_proof_l2341_234113

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_period_proof (principal : ℝ) (rate1 rate2 : ℝ) (time : ℝ) 
    (h1 : principal = 900)
    (h2 : rate1 = 0.04)
    (h3 : rate2 = 0.045)
    (h4 : simpleInterest principal rate2 time - simpleInterest principal rate1 time = 31.5) :
  time = 7 := by
  sorry

end investment_period_proof_l2341_234113


namespace bisecting_line_value_l2341_234176

/-- The equation of a line that bisects the circumference of a circle. -/
def bisecting_line (b : ℝ) (x y : ℝ) : Prop :=
  y = x + b

/-- The equation of the circle. -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 2*y + 8 = 0

/-- Theorem stating that if the line y = x + b bisects the circumference of the given circle,
    then b = -5. -/
theorem bisecting_line_value (b : ℝ) :
  (∀ x y : ℝ, bisecting_line b x y ∧ circle_equation x y → 
    ∃ c_x c_y : ℝ, c_x^2 + c_y^2 - 8*c_x + 2*c_y + 8 = 0 ∧ bisecting_line b c_x c_y) →
  b = -5 :=
sorry

end bisecting_line_value_l2341_234176


namespace surface_points_is_75_l2341_234100

/-- Represents a cube with faces marked with points -/
structure Cube where
  faces : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the assembled shape of cubes -/
structure AssembledShape where
  cubes : Fin 7 → Cube
  glued_pairs : Fin 9 → Fin 7 × Fin 6 × Fin 7 × Fin 6
  glued_pairs_same_points : ∀ i : Fin 9,
    let (c1, f1, c2, f2) := glued_pairs i
    (cubes c1).faces f1 = (cubes c2).faces f2

/-- The total number of points on the surface of the assembled shape -/
def surface_points (shape : AssembledShape) : Nat :=
  sorry

/-- Theorem stating that the total number of points on the surface is 75 -/
theorem surface_points_is_75 (shape : AssembledShape) :
  surface_points shape = 75 := by
  sorry

end surface_points_is_75_l2341_234100


namespace camilla_original_strawberry_l2341_234125

/-- Represents the number of strawberry jelly beans Camilla originally had. -/
def original_strawberry : ℕ := sorry

/-- Represents the number of grape jelly beans Camilla originally had. -/
def original_grape : ℕ := sorry

/-- States that Camilla originally had three times as many strawberry jelly beans as grape jelly beans. -/
axiom initial_ratio : original_strawberry = 3 * original_grape

/-- States that after eating 12 strawberry jelly beans and 8 grape jelly beans, 
    Camilla now has four times as many strawberry jelly beans as grape jelly beans. -/
axiom final_ratio : original_strawberry - 12 = 4 * (original_grape - 8)

/-- Theorem stating that Camilla originally had 60 strawberry jelly beans. -/
theorem camilla_original_strawberry : original_strawberry = 60 := by sorry

end camilla_original_strawberry_l2341_234125


namespace arithmetic_sequence_formula_and_sum_l2341_234172

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula_and_sum 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 = 11 →
  a 2 + a 6 = 18 →
  (∀ n : ℕ, b n = a n + 3^n) →
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, S n = n^2 + 2*n - 3/2 + 3^(n+1)/2) :=
by sorry

end arithmetic_sequence_formula_and_sum_l2341_234172


namespace sum_of_max_min_g_l2341_234127

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |3*x - 9|

-- Define the domain
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem sum_of_max_min_g :
  ∃ (max_g min_g : ℝ),
    (∀ x, domain x → g x ≤ max_g) ∧
    (∃ x, domain x ∧ g x = max_g) ∧
    (∀ x, domain x → min_g ≤ g x) ∧
    (∃ x, domain x ∧ g x = min_g) ∧
    max_g + min_g = 1 := by
  sorry

end sum_of_max_min_g_l2341_234127


namespace sarah_final_toads_l2341_234107

-- Define the number of toads each person has
def tim_toads : ℕ := 30
def jim_toads : ℕ := tim_toads + 20
def sarah_initial_toads : ℕ := 2 * jim_toads

-- Define the number of toads Sarah gives away
def sarah_gives_away : ℕ := sarah_initial_toads / 4

-- Define the number of toads Sarah buys
def sarah_buys : ℕ := 15

-- Theorem to prove
theorem sarah_final_toads :
  sarah_initial_toads - sarah_gives_away + sarah_buys = 90 := by
  sorry

end sarah_final_toads_l2341_234107


namespace circle_area_relationship_l2341_234134

theorem circle_area_relationship (A B : ℝ → ℝ → Prop) : 
  (∃ r : ℝ, (∀ x y : ℝ, A x y ↔ (x - r)^2 + (y - r)^2 = r^2) ∧ 
             (∀ x y : ℝ, B x y ↔ (x - 2*r)^2 + (y - 2*r)^2 = (2*r)^2)) →
  (π * r^2 = 16 * π) →
  (π * (2*r)^2 = 64 * π) :=
by sorry

end circle_area_relationship_l2341_234134


namespace xy_value_l2341_234115

theorem xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x / y = 81) (h4 : y = 0.2222222222222222) :
  x * y = 4 := by
  sorry

end xy_value_l2341_234115


namespace tangent_line_2sinx_at_pi_l2341_234150

/-- The equation of the tangent line to y = 2sin(x) at (π, 0) is y = -2x + 2π -/
theorem tangent_line_2sinx_at_pi (x y : ℝ) : 
  (y = 2 * Real.sin x) → -- curve equation
  (y = -2 * (x - Real.pi) + 0) → -- point-slope form of tangent line
  (y = -2 * x + 2 * Real.pi) -- final equation of tangent line
  := by sorry

end tangent_line_2sinx_at_pi_l2341_234150


namespace vector_relations_l2341_234136

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define parallelism
def parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

-- Define collinearity (same as parallelism for vectors)
def collinear (v w : V) : Prop := parallel v w

-- Theorem: Only "Equal vectors must be collinear" is true
theorem vector_relations :
  (∀ v w : V, parallel v w → v = w) = false ∧ 
  (∀ v w : V, v ≠ w → ¬(parallel v w)) = false ∧
  (∀ v w : V, collinear v w → v = w) = false ∧
  (∀ v w : V, v = w → collinear v w) = true :=
sorry

end vector_relations_l2341_234136


namespace polynomial_simplification_l2341_234135

/-- Simplification of polynomial expression -/
theorem polynomial_simplification (x : ℝ) :
  (15 * x^12 + 8 * x^9 + 5 * x^7) + (3 * x^13 + 2 * x^12 + x^11 + 6 * x^9 + 3 * x^7 + 4 * x^4 + 6 * x^2 + 9) =
  3 * x^13 + 17 * x^12 + x^11 + 14 * x^9 + 8 * x^7 + 4 * x^4 + 6 * x^2 + 9 := by
  sorry

end polynomial_simplification_l2341_234135


namespace H_upper_bound_l2341_234110

open Real

noncomputable def f (x : ℝ) : ℝ := x + log x

noncomputable def H (x m : ℝ) : ℝ := f x - log (exp x - 1)

theorem H_upper_bound {m : ℝ} (hm : m > 0) :
  ∀ x, 0 < x → x < m → H x m < m / 2 := by sorry

end H_upper_bound_l2341_234110


namespace book_price_change_l2341_234164

theorem book_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.3)
  let final_price := price_after_decrease * (1 + 0.2)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = -16 := by
sorry

end book_price_change_l2341_234164


namespace negation_of_cubic_inequality_l2341_234124

theorem negation_of_cubic_inequality :
  (¬ (∀ x : ℝ, x^3 - x ≥ 0)) ↔ (∃ x : ℝ, x^3 - x < 0) := by sorry

end negation_of_cubic_inequality_l2341_234124


namespace remainder_2007_div_81_l2341_234173

theorem remainder_2007_div_81 : 2007 % 81 = 63 := by
  sorry

end remainder_2007_div_81_l2341_234173


namespace philip_banana_count_l2341_234111

/-- The number of banana groups in Philip's collection -/
def banana_groups : ℕ := 2

/-- The number of bananas in each group -/
def bananas_per_group : ℕ := 145

/-- The total number of bananas in Philip's collection -/
def total_bananas : ℕ := banana_groups * bananas_per_group

theorem philip_banana_count : total_bananas = 290 := by
  sorry

end philip_banana_count_l2341_234111


namespace max_profit_l2341_234114

noncomputable def fixed_cost : ℝ := 14000
noncomputable def variable_cost : ℝ := 210

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then (1 / 625) * x^2
  else 256

noncomputable def g (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then -(5 / 8) * x + 750
  else 500

noncomputable def c (x : ℝ) : ℝ := fixed_cost + variable_cost * x

noncomputable def Q (x : ℝ) : ℝ := f x * g x - c x

theorem max_profit (x : ℝ) : Q x ≤ 30000 ∧ Q 400 = 30000 := by sorry

end max_profit_l2341_234114


namespace simplify_expression_l2341_234193

theorem simplify_expression (x : ℝ) : 2 * x^5 * (3 * x^9) = 6 * x^14 := by
  sorry

end simplify_expression_l2341_234193


namespace f_of_g_of_three_l2341_234184

def f (x : ℝ) : ℝ := 2 * x - 5

def g (x : ℝ) : ℝ := x + 2

theorem f_of_g_of_three : f (1 + g 3) = 7 := by
  sorry

end f_of_g_of_three_l2341_234184


namespace five_consecutive_integers_product_not_square_l2341_234198

theorem five_consecutive_integers_product_not_square (a : ℕ+) :
  ∃ (n : ℕ), (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) : ℕ) ≠ n ^ 2 := by
  sorry

end five_consecutive_integers_product_not_square_l2341_234198


namespace weekend_weather_probability_l2341_234142

/-- The probability of rain on each day -/
def rain_prob : ℝ := 0.75

/-- The number of days in the weekend -/
def num_days : ℕ := 3

/-- The number of desired sunny days -/
def desired_sunny_days : ℕ := 2

/-- Theorem: The probability of having exactly two sunny days and one rainy day
    during a three-day period, where the probability of rain each day is 0.75,
    is equal to 27/64 -/
theorem weekend_weather_probability :
  (Nat.choose num_days desired_sunny_days : ℝ) *
  (1 - rain_prob) ^ desired_sunny_days *
  rain_prob ^ (num_days - desired_sunny_days) =
  27 / 64 := by sorry

end weekend_weather_probability_l2341_234142


namespace odd_function_properties_l2341_234122

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_shift : ∀ x, f (x - 2) = -f x) : 
  f 2 = 0 ∧ has_period f 4 ∧ ∀ x, f (x + 2) = f (-x) := by
  sorry

end odd_function_properties_l2341_234122


namespace permutation_100_2_l2341_234112

/-- The number of permutations of n distinct objects taken k at a time -/
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

/-- The permutation A₁₀₀² equals 9900 -/
theorem permutation_100_2 : permutation 100 2 = 9900 := by sorry

end permutation_100_2_l2341_234112


namespace function_inequality_l2341_234163

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) : 3 * f (log 2) < 2 * f (log 3) := by
  sorry

end function_inequality_l2341_234163


namespace cheryl_same_color_probability_l2341_234182

def total_marbles : ℕ := 12
def marbles_per_color : ℕ := 3
def num_colors : ℕ := 4
def marbles_taken_each_turn : ℕ := 3

def probability_cheryl_same_color : ℚ := 2 / 55

theorem cheryl_same_color_probability :
  let total_outcomes := Nat.choose total_marbles marbles_taken_each_turn *
                        Nat.choose (total_marbles - marbles_taken_each_turn) marbles_taken_each_turn *
                        Nat.choose (total_marbles - 2 * marbles_taken_each_turn) marbles_taken_each_turn
  let favorable_outcomes := num_colors * Nat.choose (total_marbles - marbles_taken_each_turn) marbles_taken_each_turn *
                            Nat.choose (total_marbles - 2 * marbles_taken_each_turn) marbles_taken_each_turn
  (favorable_outcomes : ℚ) / total_outcomes = probability_cheryl_same_color := by
  sorry

end cheryl_same_color_probability_l2341_234182


namespace sum_in_base6_l2341_234165

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem sum_in_base6 :
  let a := base6ToDecimal [5, 5, 5, 1]
  let b := base6ToDecimal [5, 5, 1]
  let c := base6ToDecimal [5, 1]
  decimalToBase6 (a + b + c) = [2, 2, 0, 3] := by
  sorry

end sum_in_base6_l2341_234165


namespace perpendicular_lines_l2341_234148

theorem perpendicular_lines (b : ℝ) : 
  (∀ x y, 2*x - 3*y + 6 = 0 → bx - 3*y - 4 = 0 → 
    (2/3) * (b/3) = -1) → 
  b = -9/2 := by
sorry

end perpendicular_lines_l2341_234148


namespace shop_owner_gain_percentage_l2341_234154

/-- Calculates the shop owner's total gain percentage --/
theorem shop_owner_gain_percentage
  (cost_A cost_B cost_C : ℝ)
  (markup_A markup_B markup_C : ℝ)
  (quantity_A quantity_B quantity_C : ℝ)
  (discount tax : ℝ)
  (h1 : cost_A = 4)
  (h2 : cost_B = 6)
  (h3 : cost_C = 8)
  (h4 : markup_A = 0.25)
  (h5 : markup_B = 0.30)
  (h6 : markup_C = 0.20)
  (h7 : quantity_A = 25)
  (h8 : quantity_B = 15)
  (h9 : quantity_C = 10)
  (h10 : discount = 0.05)
  (h11 : tax = 0.05) :
  ∃ (gain_percentage : ℝ), abs (gain_percentage - 0.2487) < 0.0001 := by
  sorry


end shop_owner_gain_percentage_l2341_234154


namespace arc_length_theorem_l2341_234195

-- Define the curve
def curve (x y : ℝ) : Prop := Real.exp (2 * y) * (Real.exp (2 * x) - 1) = Real.exp (2 * x) + 1

-- Define the arc length function
noncomputable def arcLength (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (1 + (deriv f x) ^ 2)

-- State the theorem
theorem arc_length_theorem :
  ∃ f : ℝ → ℝ,
    (∀ x, curve x (f x)) ∧
    arcLength f 1 2 = (1 / 2) * Real.log (Real.exp 4 + 1) - 1 := by sorry

end arc_length_theorem_l2341_234195


namespace money_distribution_l2341_234189

theorem money_distribution (x y z : ℝ) : 
  x + (y/2 + z/2) = 90 →
  y + (x/2 + z/2) = 70 →
  z + (x/2 + y/2) = 56 →
  y = 32 := by
  sorry

end money_distribution_l2341_234189


namespace f_three_zeros_c_range_l2341_234133

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + c

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 12

-- Theorem statement
theorem f_three_zeros_c_range (c : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f c x₁ = 0 ∧ f c x₂ = 0 ∧ f c x₃ = 0) →
  -16 < c ∧ c < 16 := by sorry

end f_three_zeros_c_range_l2341_234133


namespace rearrangement_count_correct_l2341_234191

/-- The number of ways to rearrange 3 out of 8 people in a row, 
    while keeping the other 5 in their original positions. -/
def rearrangement_count : ℕ := Nat.choose 8 3 * 2

/-- Theorem stating that the number of rearrangements is correct. -/
theorem rearrangement_count_correct : 
  rearrangement_count = Nat.choose 8 3 * 2 := by sorry

end rearrangement_count_correct_l2341_234191


namespace xy_2yz_3zx_value_l2341_234147

theorem xy_2yz_3zx_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2/3 = 25)
  (eq2 : y^2/3 + z^2 = 9)
  (eq3 : z^2 + z*x + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24 * Real.sqrt 3 := by
sorry

end xy_2yz_3zx_value_l2341_234147


namespace base9_734_equals_base10_598_l2341_234175

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₀ * 9^0 + d₁ * 9^1 + d₂ * 9^2

/-- Theorem: The base-9 number 734 is equal to 598 in base-10 --/
theorem base9_734_equals_base10_598 : base9ToBase10 7 3 4 = 598 := by
  sorry

#eval base9ToBase10 7 3 4

end base9_734_equals_base10_598_l2341_234175


namespace unique_solution_diophantine_system_l2341_234103

theorem unique_solution_diophantine_system :
  ∀ a b c : ℕ,
  a^3 - b^3 - c^3 = 3*a*b*c →
  a^2 = 2*(b + c) →
  a = 2 ∧ b = 1 ∧ c = 1 := by
sorry

end unique_solution_diophantine_system_l2341_234103


namespace line_slope_is_one_l2341_234181

/-- Given a line in the xy-plane with y-intercept -2 and passing through the midpoint
    of the line segment with endpoints (2, 8) and (14, 4), its slope is 1. -/
theorem line_slope_is_one (m : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ m ↔ y = x - 2) →  -- y-intercept is -2
  ((8 : ℝ), 6) ∈ m →  -- passes through midpoint ((2+14)/2, (8+4)/2) = (8, 6)
  (∃ (k b : ℝ), ∀ (x y : ℝ), (x, y) ∈ m ↔ y = k * x + b) →  -- m is a line
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ m ↔ y = k * x - 2) →  -- combine line equation with y-intercept
  ∀ (x y : ℝ), (x, y) ∈ m ↔ y = x - 2 :=
by sorry

end line_slope_is_one_l2341_234181


namespace quadratic_trinomial_condition_l2341_234159

theorem quadratic_trinomial_condition (m : ℤ) : 
  (|m| = 2 ∧ m ≠ 2) ↔ m = -2 := by sorry

end quadratic_trinomial_condition_l2341_234159


namespace sqrt_121_equals_plus_minus_11_l2341_234141

theorem sqrt_121_equals_plus_minus_11 : ∀ (x : ℝ), x^2 = 121 ↔ x = 11 ∨ x = -11 := by
  sorry

end sqrt_121_equals_plus_minus_11_l2341_234141


namespace smallest_n_for_reducible_fraction_l2341_234170

theorem smallest_n_for_reducible_fraction : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 13) ∧ k ∣ (5*m + 6))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 13) ∧ k ∣ (5*n + 6)) ∧
  n = 84 := by
  sorry

#check smallest_n_for_reducible_fraction

end smallest_n_for_reducible_fraction_l2341_234170


namespace complex_equation_solution_l2341_234130

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 3 - 4 * Complex.I) → z = -4 - 3 * Complex.I := by
  sorry

end complex_equation_solution_l2341_234130


namespace inequality_proof_l2341_234171

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * (a + b) + b * c * (b + c) + a * c * (a + c) ≥ 6 * a * b * c := by
  sorry

end inequality_proof_l2341_234171


namespace fruit_drink_volume_l2341_234158

theorem fruit_drink_volume (grapefruit_percent : ℝ) (lemon_percent : ℝ) (orange_volume : ℝ) :
  grapefruit_percent = 0.25 →
  lemon_percent = 0.35 →
  orange_volume = 20 →
  ∃ total_volume : ℝ,
    total_volume = 50 ∧
    grapefruit_percent * total_volume + lemon_percent * total_volume + orange_volume = total_volume :=
by sorry

end fruit_drink_volume_l2341_234158


namespace g_of_3_eq_6_l2341_234129

/-- A function satisfying the given conditions -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 1 = 2 ∧ ∀ x y : ℝ, g (x^2 + y^2) = (x + y) * (g x + g y)

/-- Theorem stating that g(3) = 6 for any function satisfying the conditions -/
theorem g_of_3_eq_6 (g : ℝ → ℝ) (h : special_function g) : g 3 = 6 := by
  sorry

end g_of_3_eq_6_l2341_234129


namespace boxes_with_neither_pens_nor_pencils_l2341_234118

/-- Given a set of boxes with pens and pencils, this theorem proves
    the number of boxes containing neither pens nor pencils. -/
theorem boxes_with_neither_pens_nor_pencils
  (total_boxes : ℕ)
  (pencil_boxes : ℕ)
  (pen_boxes : ℕ)
  (both_boxes : ℕ)
  (h1 : total_boxes = 10)
  (h2 : pencil_boxes = 6)
  (h3 : pen_boxes = 3)
  (h4 : both_boxes = 2)
  : total_boxes - (pencil_boxes + pen_boxes - both_boxes) = 3 := by
  sorry

#check boxes_with_neither_pens_nor_pencils

end boxes_with_neither_pens_nor_pencils_l2341_234118


namespace rotation_180_maps_points_l2341_234144

/-- Rotation of 180° clockwise about the origin in 2D plane -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotation_180_maps_points :
  let C : ℝ × ℝ := (-3, 2)
  let D : ℝ × ℝ := (-2, 5)
  let C' : ℝ × ℝ := (3, -2)
  let D' : ℝ × ℝ := (2, -5)
  rotate180 C = C' ∧ rotate180 D = D' :=
by sorry

end rotation_180_maps_points_l2341_234144


namespace josie_shortage_l2341_234185

def gift_amount : ℝ := 150
def cassette_count : ℕ := 5
def cassette_price : ℝ := 18
def headphone_count : ℕ := 2
def headphone_price : ℝ := 45
def vinyl_count : ℕ := 3
def vinyl_price : ℝ := 22
def magazine_count : ℕ := 4
def magazine_price : ℝ := 7

def total_cost : ℝ :=
  cassette_count * cassette_price +
  headphone_count * headphone_price +
  vinyl_count * vinyl_price +
  magazine_count * magazine_price

theorem josie_shortage : gift_amount - total_cost = -124 := by
  sorry

end josie_shortage_l2341_234185


namespace sum_equals_three_l2341_234128

/-- The largest proper fraction with denominator 9 -/
def largest_proper_fraction : ℚ := 8/9

/-- The smallest improper fraction with denominator 9 -/
def smallest_improper_fraction : ℚ := 9/9

/-- The smallest mixed number with fractional part having denominator 9 -/
def smallest_mixed_number : ℚ := 1 + 1/9

/-- The sum of the largest proper fraction, smallest improper fraction, and smallest mixed number -/
def sum_of_fractions : ℚ := largest_proper_fraction + smallest_improper_fraction + smallest_mixed_number

theorem sum_equals_three : sum_of_fractions = 3 := by
  sorry

end sum_equals_three_l2341_234128


namespace no_linear_term_implies_a_equals_six_l2341_234108

theorem no_linear_term_implies_a_equals_six (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (2*x + a) * (3 - x) = b * x^2 + c) → a = 6 := by
  sorry

end no_linear_term_implies_a_equals_six_l2341_234108


namespace decagon_perimeter_30_l2341_234146

/-- A regular decagon is a polygon with 10 sides of equal length. -/
structure RegularDecagon where
  side_length : ℝ
  sides : Nat
  sides_eq : sides = 10

/-- The perimeter of a polygon is the sum of the lengths of its sides. -/
def perimeter (d : RegularDecagon) : ℝ := d.side_length * d.sides

theorem decagon_perimeter_30 (d : RegularDecagon) (h : d.side_length = 3) : perimeter d = 30 := by
  sorry

end decagon_perimeter_30_l2341_234146


namespace d_equals_square_iff_l2341_234151

/-- Move the last digit of a number to the first position -/
def moveLastToFirst (a : ℕ) : ℕ :=
  sorry

/-- Square a number -/
def square (b : ℕ) : ℕ :=
  sorry

/-- Move the first digit of a number to the end -/
def moveFirstToLast (c : ℕ) : ℕ :=
  sorry

/-- The d(a) function as described in the problem -/
def d (a : ℕ) : ℕ :=
  moveFirstToLast (square (moveLastToFirst a))

/-- Check if a number is of the form 222...21 -/
def is222_21 (a : ℕ) : Prop :=
  sorry

/-- The main theorem -/
theorem d_equals_square_iff (a : ℕ) :
  d a = a^2 ↔ a = 1 ∨ a = 2 ∨ a = 3 ∨ is222_21 a :=
sorry

end d_equals_square_iff_l2341_234151


namespace matrix_power_difference_l2341_234162

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 4, 2; 0, 2, 3; 0, 0, 1]

theorem matrix_power_difference :
  A^5 - 3 * A^4 = !![0, 4, 2; 0, -1, 3; 0, 0, -2] := by sorry

end matrix_power_difference_l2341_234162


namespace average_speed_round_trip_l2341_234149

theorem average_speed_round_trip (outbound_speed inbound_speed : ℝ) 
  (h1 : outbound_speed = 130)
  (h2 : inbound_speed = 88)
  (h3 : outbound_speed > 0)
  (h4 : inbound_speed > 0) :
  (2 * outbound_speed * inbound_speed) / (outbound_speed + inbound_speed) = 105 := by
sorry

end average_speed_round_trip_l2341_234149


namespace min_value_squared_sum_l2341_234131

theorem min_value_squared_sum (x y z a : ℝ) (h : x + 2*y + 3*z = a) :
  x^2 + y^2 + z^2 ≥ a^2 / 14 := by
  sorry

end min_value_squared_sum_l2341_234131


namespace nine_bounces_on_12x10_table_l2341_234160

/-- Represents a rectangular pool table -/
structure PoolTable where
  width : ℕ
  height : ℕ

/-- Represents a ball's path on the pool table -/
structure BallPath where
  start_x : ℕ
  start_y : ℕ
  slope : ℚ

/-- Calculates the number of wall bounces for a ball's path on a pool table -/
def count_wall_bounces (table : PoolTable) (path : BallPath) : ℕ :=
  sorry

/-- Theorem stating that a ball hit from (0,0) along y=x on a 12x10 table bounces 9 times -/
theorem nine_bounces_on_12x10_table :
  let table : PoolTable := { width := 12, height := 10 }
  let path : BallPath := { start_x := 0, start_y := 0, slope := 1 }
  count_wall_bounces table path = 9 :=
sorry

end nine_bounces_on_12x10_table_l2341_234160


namespace min_value_theorem_l2341_234190

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a) + (4 / b) ≥ 9 := by
sorry

end min_value_theorem_l2341_234190


namespace unique_determinable_score_l2341_234102

/-- The AHSME scoring system and constraints -/
structure AHSME where
  total_questions : ℕ
  score : ℕ
  correct : ℕ
  wrong : ℕ
  score_formula : score = 30 + 4 * correct - wrong
  total_answered : correct + wrong ≤ total_questions

/-- The uniqueness of the score for determining correct answers -/
def is_unique_determinable_score (s : ℕ) : Prop :=
  s > 80 ∧
  ∃! (exam : AHSME),
    exam.total_questions = 30 ∧
    exam.score = s ∧
    ∀ (s' : ℕ), 80 < s' ∧ s' < s →
      ¬∃! (exam' : AHSME),
        exam'.total_questions = 30 ∧
        exam'.score = s'

/-- The theorem stating that 119 is the unique score that satisfies the conditions -/
theorem unique_determinable_score :
  is_unique_determinable_score 119 :=
sorry

end unique_determinable_score_l2341_234102
