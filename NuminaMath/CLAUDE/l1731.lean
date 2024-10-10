import Mathlib

namespace quadratic_solution_difference_l1731_173144

theorem quadratic_solution_difference : 
  let f : ℝ → ℝ := λ x => x^2 - 5*x + 7 - (x + 35)
  let s₁ := (6 + Real.sqrt 148) / 2
  let s₂ := (6 - Real.sqrt 148) / 2
  f s₁ = 0 ∧ f s₂ = 0 ∧ s₁ - s₂ = 2 * Real.sqrt 37 := by sorry

end quadratic_solution_difference_l1731_173144


namespace soccer_team_points_l1731_173161

/-- Calculates the total points for a soccer team given their game results -/
def calculate_points (total_games : ℕ) (wins : ℕ) (losses : ℕ) (win_points : ℕ) (draw_points : ℕ) (loss_points : ℕ) : ℕ :=
  let draws := total_games - wins - losses
  wins * win_points + draws * draw_points + losses * loss_points

/-- Theorem stating that the soccer team's total points is 46 -/
theorem soccer_team_points :
  calculate_points 20 14 2 3 1 0 = 46 := by
  sorry

end soccer_team_points_l1731_173161


namespace set_intersection_complement_l1731_173143

def U : Set Int := Set.univ
def M : Set Int := {1, 2}
def P : Set Int := {-2, -1, 0, 1, 2}

theorem set_intersection_complement : P ∩ (U \ M) = {-2, -1, 0} := by
  sorry

end set_intersection_complement_l1731_173143


namespace third_group_size_l1731_173133

/-- The number of students in each group and the total number of students -/
structure StudentGroups where
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ
  group4 : ℕ
  total : ℕ

/-- The theorem stating that the third group has 7 students -/
theorem third_group_size (sg : StudentGroups) 
  (h1 : sg.group1 = 5)
  (h2 : sg.group2 = 8)
  (h3 : sg.group4 = 4)
  (h4 : sg.total = 24)
  (h5 : sg.total = sg.group1 + sg.group2 + sg.group3 + sg.group4) :
  sg.group3 = 7 := by
  sorry

end third_group_size_l1731_173133


namespace cos_squared_minus_sin_squared_pi_over_twelve_l1731_173101

theorem cos_squared_minus_sin_squared_pi_over_twelve (π : Real) :
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end cos_squared_minus_sin_squared_pi_over_twelve_l1731_173101


namespace min_sum_x_y_l1731_173181

theorem min_sum_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + x*y - 7 = 0) :
  ∃ (m : ℝ), m = 3 ∧ x + y ≥ m ∧ ∀ (z : ℝ), x + y > z → z < m :=
sorry

end min_sum_x_y_l1731_173181


namespace customers_added_l1731_173111

theorem customers_added (initial : ℕ) (no_tip : ℕ) (tip : ℕ) : 
  initial = 39 → no_tip = 49 → tip = 2 → 
  (no_tip + tip) - initial = 12 := by
  sorry

end customers_added_l1731_173111


namespace coach_sunscreen_fraction_is_correct_l1731_173152

/-- The fraction of sunscreen transferred to a person's forehead when heading the ball -/
def transfer_fraction : ℚ := 1 / 10

/-- The fraction of sunscreen remaining on the ball after a header -/
def remaining_fraction : ℚ := 1 - transfer_fraction

/-- The sequence of headers -/
inductive Header
| C : Header  -- Coach
| A : Header  -- Player A
| B : Header  -- Player B

/-- The repeating sequence of headers -/
def header_sequence : List Header := [Header.C, Header.A, Header.C, Header.B]

/-- The fraction of original sunscreen on Coach C's forehead after infinite headers -/
def coach_sunscreen_fraction : ℚ := 10 / 19

/-- Theorem stating that the fraction of original sunscreen on Coach C's forehead
    after infinite headers is 10/19 -/
theorem coach_sunscreen_fraction_is_correct :
  coach_sunscreen_fraction = 
    (transfer_fraction * (1 / (1 - remaining_fraction^2))) := by sorry

end coach_sunscreen_fraction_is_correct_l1731_173152


namespace arccos_cos_ten_equals_two_l1731_173167

theorem arccos_cos_ten_equals_two : Real.arccos (Real.cos 10) = 2 := by
  sorry

end arccos_cos_ten_equals_two_l1731_173167


namespace two_digit_number_problem_l1731_173188

theorem two_digit_number_problem (x y : ℕ) :
  x < 10 ∧ y < 10 ∧ 
  (10 * x + y) - (10 * y + x) = 36 ∧
  x + y = 8 →
  10 * x + y = 62 := by
  sorry

end two_digit_number_problem_l1731_173188


namespace line_slope_l1731_173171

theorem line_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : 
  (y - 4) / x = -4 / 7 := by
  sorry

end line_slope_l1731_173171


namespace smaller_number_problem_l1731_173182

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 := by
  sorry

end smaller_number_problem_l1731_173182


namespace river_speed_l1731_173160

/-- The speed of a river given certain rowing conditions -/
theorem river_speed (man_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) : 
  man_speed = 8 → 
  total_time = 1 → 
  total_distance = 7.5 → 
  ∃ (river_speed : ℝ), 
    river_speed = 2 ∧ 
    total_distance / (man_speed - river_speed) + total_distance / (man_speed + river_speed) = total_time :=
by sorry

end river_speed_l1731_173160


namespace right_triangle_arithmetic_progression_l1731_173136

theorem right_triangle_arithmetic_progression :
  ∃ (a d c : ℕ), 
    a > 0 ∧ d > 0 ∧ c > 0 ∧
    a * a + (a + d) * (a + d) = c * c ∧
    c = a + 2 * d ∧
    (a = 120 ∨ a + d = 120 ∨ c = 120) :=
by sorry

end right_triangle_arithmetic_progression_l1731_173136


namespace shekhar_shobha_age_ratio_l1731_173122

/-- The ratio of Shekhar's age to Shobha's age -/
def age_ratio (shekhar_age shobha_age : ℕ) : ℚ :=
  shekhar_age / shobha_age

/-- Theorem stating the ratio of Shekhar's age to Shobha's age -/
theorem shekhar_shobha_age_ratio :
  ∃ (shekhar_age : ℕ),
    shekhar_age + 6 = 26 ∧
    age_ratio shekhar_age 15 = 4/3 := by
  sorry

end shekhar_shobha_age_ratio_l1731_173122


namespace sum_of_integers_ending_in_2_l1731_173118

def sumOfIntegersEndingIn2 (lower upper : ℕ) : ℕ :=
  let firstTerm := (lower + 2 - lower % 10)
  let lastTerm := (upper - upper % 10 + 2)
  let numTerms := (lastTerm - firstTerm) / 10 + 1
  numTerms * (firstTerm + lastTerm) / 2

theorem sum_of_integers_ending_in_2 :
  sumOfIntegersEndingIn2 60 460 = 10280 := by
  sorry

end sum_of_integers_ending_in_2_l1731_173118


namespace min_values_theorem_l1731_173134

theorem min_values_theorem (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h : (r + s - r * s) * (r + s + r * s) = r * s) : 
  (∃ (r' s' : ℝ), r' > 0 ∧ s' > 0 ∧ 
    (r' + s' - r' * s') * (r' + s' + r' * s') = r' * s' ∧
    r + s - r * s ≥ -3 + 2 * Real.sqrt 3 ∧
    r + s + r * s ≥ 3 + 2 * Real.sqrt 3) ∧
  (∀ (r' s' : ℝ), r' > 0 → s' > 0 → 
    (r' + s' - r' * s') * (r' + s' + r' * s') = r' * s' →
    r' + s' - r' * s' ≥ -3 + 2 * Real.sqrt 3 ∧
    r' + s' + r' * s' ≥ 3 + 2 * Real.sqrt 3) :=
by sorry


end min_values_theorem_l1731_173134


namespace flowchart_output_l1731_173178

def swap_operation (a b c : ℕ) : ℕ × ℕ × ℕ := 
  let (a', c') := (c, a)
  let (b', c'') := (c', b)
  (a', b', c'')

theorem flowchart_output (a b c : ℕ) (h1 : a = 21) (h2 : b = 32) (h3 : c = 75) :
  swap_operation a b c = (75, 21, 32) := by
  sorry

end flowchart_output_l1731_173178


namespace two_digit_numbers_satisfying_condition_l1731_173168

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  (sum_of_digits n)^2 = sum_of_digits (n^2)

theorem two_digit_numbers_satisfying_condition :
  {n : ℕ | is_two_digit n ∧ satisfies_condition n} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} := by sorry

end two_digit_numbers_satisfying_condition_l1731_173168


namespace total_laundry_count_l1731_173121

/-- Represents the number of items for each person and shared items -/
structure LaundryItems where
  cally : Nat
  danny : Nat
  emily : Nat
  cally_danny : Nat
  emily_danny : Nat
  cally_emily : Nat

/-- Calculates the total number of laundry items -/
def total_laundry (items : LaundryItems) : Nat :=
  items.cally + items.danny + items.emily + items.cally_danny + items.emily_danny + items.cally_emily

/-- Theorem: The total number of clothes and accessories washed is 141 -/
theorem total_laundry_count :
  ∃ (items : LaundryItems),
    items.cally = 40 ∧
    items.danny = 39 ∧
    items.emily = 39 ∧
    items.cally_danny = 8 ∧
    items.emily_danny = 6 ∧
    items.cally_emily = 9 ∧
    total_laundry items = 141 := by
  sorry

end total_laundry_count_l1731_173121


namespace max_square_sum_max_square_sum_achievable_l1731_173127

/-- The maximum value of x^2 + y^2 given 0 ≤ x ≤ 1 and 0 ≤ y ≤ 1 -/
theorem max_square_sum : 
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → x^2 + y^2 ≤ 2 := by
  sorry

/-- The maximum value 2 is achievable -/
theorem max_square_sum_achievable : 
  ∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x^2 + y^2 = 2 := by
  sorry

end max_square_sum_max_square_sum_achievable_l1731_173127


namespace graces_initial_fruits_l1731_173103

/-- The number of Graces --/
def num_graces : ℕ := 3

/-- The number of Muses --/
def num_muses : ℕ := 9

/-- Represents the distribution of fruits --/
structure FruitDistribution where
  initial_grace : ℕ  -- Initial number of fruits each Grace had
  given_to_muse : ℕ  -- Number of fruits each Grace gave to each Muse

/-- Theorem stating the conditions and the result to be proved --/
theorem graces_initial_fruits (fd : FruitDistribution) : 
  -- Each Grace gives fruits to each Muse
  (fd.initial_grace ≥ num_muses * fd.given_to_muse) →
  -- After exchange, Graces and Muses have the same number of fruits
  (fd.initial_grace - num_muses * fd.given_to_muse = num_graces * fd.given_to_muse) →
  -- Initial number of fruits each Grace had is 12
  fd.initial_grace = 12 :=
by
  sorry

end graces_initial_fruits_l1731_173103


namespace converse_square_sum_zero_contrapositive_subset_intersection_l1731_173142

-- Define the propositions
def P (x y : ℝ) : Prop := x^2 + y^2 = 0
def Q (x y : ℝ) : Prop := x = 0 ∧ y = 0

def R (A B : Set α) : Prop := A ∩ B = A
def S (A B : Set α) : Prop := A ⊆ B

-- Theorem for the converse of statement ①
theorem converse_square_sum_zero :
  ∀ x y : ℝ, Q x y → P x y :=
sorry

-- Theorem for the contrapositive of statement ③
theorem contrapositive_subset_intersection :
  ∀ A B : Set α, ¬(S A B) → ¬(R A B) :=
sorry

end converse_square_sum_zero_contrapositive_subset_intersection_l1731_173142


namespace right_triangle_perimeter_right_triangle_perimeter_proof_l1731_173140

/-- The perimeter of a right triangle with legs 8 and 6 is 24. -/
theorem right_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun PQ QR PR =>
    QR = 8 ∧ PR = 6 ∧ PQ ^ 2 = QR ^ 2 + PR ^ 2 →
    PQ + QR + PR = 24

/-- Proof of the theorem -/
theorem right_triangle_perimeter_proof : right_triangle_perimeter 10 8 6 := by
  sorry

end right_triangle_perimeter_right_triangle_perimeter_proof_l1731_173140


namespace two_digit_number_sum_l1731_173194

theorem two_digit_number_sum (n : ℕ) : 
  10 ≤ n ∧ n < 100 →  -- n is a two-digit number
  (n : ℚ) / 2 = n / 4 + 3 →  -- one half of n exceeds its one fourth by 3
  (n / 10 + n % 10 : ℕ) = 12  -- sum of digits is 12
  := by sorry

end two_digit_number_sum_l1731_173194


namespace endpoint_coordinate_sum_l1731_173113

/-- Given a line segment with one endpoint (6, -2) and midpoint (3, 5),
    the sum of coordinates of the other endpoint is 12. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (6 + x) / 2 = 3 ∧ (-2 + y) / 2 = 5 → x + y = 12 := by
  sorry

end endpoint_coordinate_sum_l1731_173113


namespace ant_meeting_probability_l1731_173145

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8)

/-- Represents an ant on a vertex of the cube -/
structure Ant :=
  (position : Fin 8)

/-- Represents a movement of an ant along an edge -/
def AntMovement := Fin 8 → Fin 8

/-- The total number of possible movement combinations for 8 ants -/
def totalMovements : ℕ := 3^8

/-- The number of non-colliding movement configurations -/
def nonCollidingMovements : ℕ := 24

/-- The probability of ants meeting -/
def probabilityOfMeeting : ℚ := 1 - (nonCollidingMovements : ℚ) / totalMovements

theorem ant_meeting_probability (c : Cube) (ants : Fin 8 → Ant) 
  (movements : Fin 8 → AntMovement) : 
  probabilityOfMeeting = 2381/2387 :=
sorry

end ant_meeting_probability_l1731_173145


namespace sqrt_difference_inequality_l1731_173170

theorem sqrt_difference_inequality (m : ℝ) (h : m > 1) :
  Real.sqrt (m + 1) - Real.sqrt m < Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end sqrt_difference_inequality_l1731_173170


namespace ellipse_chord_slope_l1731_173176

/-- The slope of a chord in an ellipse --/
theorem ellipse_chord_slope (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁^2 / 8 + y₁^2 / 6 = 1) →
  (x₂^2 / 8 + y₂^2 / 6 = 1) →
  ((x₁ + x₂) / 2 = 2) →
  ((y₁ + y₂) / 2 = 1) →
  (y₂ - y₁) / (x₂ - x₁) = -3/2 := by
sorry

end ellipse_chord_slope_l1731_173176


namespace minibus_students_l1731_173164

theorem minibus_students (boys : ℕ) (girls : ℕ) : 
  boys = 8 →
  girls = boys + 2 →
  boys + girls = 18 :=
by
  sorry

end minibus_students_l1731_173164


namespace inverse_proportion_y_comparison_l1731_173123

/-- Given two points on the inverse proportion function y = -5/x,
    where the x-coordinate of the first point is positive and
    the x-coordinate of the second point is negative,
    prove that the y-coordinate of the first point is less than
    the y-coordinate of the second point. -/
theorem inverse_proportion_y_comparison
  (x₁ x₂ y₁ y₂ : ℝ)
  (h1 : y₁ = -5 / x₁)
  (h2 : y₂ = -5 / x₂)
  (h3 : x₁ > 0)
  (h4 : x₂ < 0) :
  y₁ < y₂ :=
sorry

end inverse_proportion_y_comparison_l1731_173123


namespace barycentric_geometry_l1731_173196

/-- Barycentric coordinates in a triangle --/
structure BarycentricCoord where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Definition of a line in barycentric coordinates --/
def is_line (f : BarycentricCoord → ℝ) : Prop :=
  ∃ u v w : ℝ, ∀ p : BarycentricCoord, f p = u * p.α + v * p.β + w * p.γ

/-- Definition of a circle in barycentric coordinates --/
def is_circle (f : BarycentricCoord → ℝ) : Prop :=
  ∃ a b c u v w : ℝ, ∀ p : BarycentricCoord,
    f p = -a^2 * p.β * p.γ - b^2 * p.γ * p.α - c^2 * p.α * p.β + 
          (u * p.α + v * p.β + w * p.γ) * (p.α + p.β + p.γ)

theorem barycentric_geometry :
  ∀ A : BarycentricCoord,
  (∃ f : BarycentricCoord → ℝ, is_line f ∧ ∀ p : BarycentricCoord, f p = p.β * w - p.γ * v) ∧
  (∃ g : BarycentricCoord → ℝ, is_line g) ∧
  (∃ h : BarycentricCoord → ℝ, is_circle h) :=
by sorry

end barycentric_geometry_l1731_173196


namespace sufficient_not_necessary_l1731_173149

theorem sufficient_not_necessary :
  (∀ x : ℝ, x < Real.sqrt 2 → 2 * x < 3) ∧
  ¬(∀ x : ℝ, 2 * x < 3 → x < Real.sqrt 2) :=
by sorry

end sufficient_not_necessary_l1731_173149


namespace binomial_expansion_theorem_l1731_173174

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) : 
  n ≥ 2 → 
  a * b ≠ 0 → 
  k ≥ 1 → 
  a = 2 * k * b → 
  (Nat.choose n 2 * (2 * b)^(n - 2) * (k - 1)^2 + 
   Nat.choose n 3 * (2 * b)^(n - 3) * (k - 1)^3 = 0) → 
  n = 3 * k - 1 := by
sorry

end binomial_expansion_theorem_l1731_173174


namespace prob_different_topics_correct_l1731_173147

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    out of num_topics is equal to prob_different_topics -/
theorem prob_different_topics_correct :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics := by
  sorry

end prob_different_topics_correct_l1731_173147


namespace complex_equation_solution_l1731_173153

theorem complex_equation_solution (x : ℂ) : 5 - 2 * Complex.I * x = 7 - 5 * Complex.I * x ↔ x = (2 * Complex.I) / 3 := by
  sorry

end complex_equation_solution_l1731_173153


namespace probability_at_least_one_woman_l1731_173138

def num_men : ℕ := 10
def num_women : ℕ := 5
def num_chosen : ℕ := 4

theorem probability_at_least_one_woman :
  let total := num_men + num_women
  (1 - (Nat.choose num_men num_chosen : ℚ) / (Nat.choose total num_chosen : ℚ)) = 77 / 91 := by
  sorry

end probability_at_least_one_woman_l1731_173138


namespace part_one_part_two_l1731_173156

/-- The absolute value function -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

/-- Part I: Range of a such that f(x) ≤ 3 for all x in [-1, 3] -/
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, x ∈ [-1, 3] → f a x ≤ 3) ↔ a ∈ Set.Icc 0 2 := by sorry

/-- Part II: Minimum value of a such that f(x-a) + f(x+a) ≥ 1-2a for all x -/
theorem part_two : 
  (∃ a : ℝ, (∀ x : ℝ, f a (x-a) + f a (x+a) ≥ 1-2*a) ∧ 
   (∀ b : ℝ, (∀ x : ℝ, f b (x-b) + f b (x+b) ≥ 1-2*b) → a ≤ b)) ∧
  (let a := (1/4 : ℝ); ∀ x : ℝ, f a (x-a) + f a (x+a) ≥ 1-2*a) := by sorry

end part_one_part_two_l1731_173156


namespace four_good_points_l1731_173125

/-- A point (x, y) is a "good point" if x is an integer, y is a perfect square,
    and y = (x - 90)^2 - 4907 -/
def is_good_point (x y : ℤ) : Prop :=
  ∃ (m : ℤ), y = m^2 ∧ y = (x - 90)^2 - 4907

/-- The set of all "good points" -/
def good_points : Set (ℤ × ℤ) :=
  {p | is_good_point p.1 p.2}

/-- The theorem stating that there are exactly four "good points" -/
theorem four_good_points :
  good_points = {(444, 120409), (-264, 120409), (2544, 6017209), (-2364, 6017209)} := by
  sorry

#check four_good_points

end four_good_points_l1731_173125


namespace always_separable_l1731_173148

/-- Represents a cell in the square -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents a square of size 2n × 2n -/
structure Square (n : Nat) where
  size : Nat := 2 * n

/-- Represents a cut in the square -/
inductive Cut
  | Vertical : Nat → Cut
  | Horizontal : Nat → Cut

/-- Checks if two cells are separated by a cut -/
def separatedByCut (c1 c2 : Cell) (cut : Cut) : Prop :=
  match cut with
  | Cut.Vertical x => (c1.x ≤ x ∧ c2.x > x) ∨ (c1.x > x ∧ c2.x ≤ x)
  | Cut.Horizontal y => (c1.y ≤ y ∧ c2.y > y) ∨ (c1.y > y ∧ c2.y ≤ y)

/-- Main theorem: There always exists a cut that separates any two colored cells -/
theorem always_separable (n : Nat) (c1 c2 : Cell) 
    (h1 : c1.x < 2 * n ∧ c1.y < 2 * n)
    (h2 : c2.x < 2 * n ∧ c2.y < 2 * n)
    (h3 : c1 ≠ c2) :
    ∃ (cut : Cut), separatedByCut c1 c2 cut :=
  sorry


end always_separable_l1731_173148


namespace polynomial_fixed_point_l1731_173139

theorem polynomial_fixed_point (P : ℤ → ℤ) (h_poly : ∃ (coeffs : List ℤ), ∀ x, P x = (coeffs.map (λ (c : ℤ) (i : ℕ) => c * x ^ i)).sum) :
  P 1 = 2013 → P 2013 = 1 → ∃ k : ℤ, P k = k → k = 1007 := by
  sorry

end polynomial_fixed_point_l1731_173139


namespace x_cubed_coefficient_l1731_173132

theorem x_cubed_coefficient (p q : Polynomial ℤ) : 
  p = 3 * X^3 + 2 * X^2 + 5 * X + 6 →
  q = 4 * X^3 + 7 * X^2 + 9 * X + 8 →
  (p * q).coeff 3 = 83 := by
  sorry

end x_cubed_coefficient_l1731_173132


namespace february_to_january_ratio_l1731_173108

def january_bill : ℚ := 180

def february_bill : ℚ := 270

theorem february_to_january_ratio :
  (february_bill / january_bill) = 3 / 2 ∧
  ((february_bill + 30) / january_bill) = 5 / 3 := by
  sorry

end february_to_january_ratio_l1731_173108


namespace parallelogram_area_theorem_l1731_173114

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Calculates the area of a triangle -/
def areaTriangle (t : Triangle) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (M A B : Point) : Prop := sorry

/-- Checks if three points are collinear -/
def collinear (A B C : Point) : Prop := sorry

theorem parallelogram_area_theorem (ABCD : Parallelogram) (E F : Point) :
  collinear C E F →
  isMidpoint F ABCD.A ABCD.B →
  areaTriangle ⟨ABCD.B, E, C⟩ = 100 →
  areaQuadrilateral ⟨ABCD.A, F, E, ABCD.D⟩ = 250 := by
  sorry

end parallelogram_area_theorem_l1731_173114


namespace ball_ratio_proof_l1731_173104

/-- Given that Robert initially had 25 balls, Tim initially had 40 balls,
    and Robert ended up with 45 balls after Tim gave him some balls,
    prove that the ratio of the number of balls Tim gave to Robert
    to the number of balls Tim had initially is 1:2. -/
theorem ball_ratio_proof (robert_initial : ℕ) (tim_initial : ℕ) (robert_final : ℕ)
    (h1 : robert_initial = 25)
    (h2 : tim_initial = 40)
    (h3 : robert_final = 45) :
    (robert_final - robert_initial) * 2 = tim_initial := by
  sorry

end ball_ratio_proof_l1731_173104


namespace parabola_equation_l1731_173155

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define points and vectors
variable (F A B C : ℝ × ℝ)  -- Points as pairs of real numbers
variable (AF FB BA BC : ℝ × ℝ)  -- Vectors as pairs of real numbers

-- Define vector operations
def vector_equal (v w : ℝ × ℝ) : Prop := v.1 = w.1 ∧ v.2 = w.2
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem parabola_equation (p : ℝ) :
  parabola p A.1 A.2 →  -- A is on the parabola
  vector_equal AF FB →  -- AF = FB
  dot_product BA BC = 48 →  -- BA · BC = 48
  p = 2 ∧ parabola 2 A.1 A.2  -- The parabola equation is y² = 4x
  := by sorry

end parabola_equation_l1731_173155


namespace area_of_connected_paper_l1731_173189

/-- The area of connected colored paper sheets -/
theorem area_of_connected_paper (n : ℕ) (side_length overlap : ℝ) :
  n > 0 →
  side_length > 0 →
  overlap ≥ 0 →
  overlap < side_length →
  let total_length := side_length + (n - 1 : ℝ) * (side_length - overlap)
  let area := total_length * side_length
  n = 6 ∧ side_length = 30 ∧ overlap = 7 →
  area = 4350 := by
  sorry

end area_of_connected_paper_l1731_173189


namespace sharons_journey_l1731_173135

/-- The distance between Sharon's house and her mother's house -/
def total_distance : ℝ := 200

/-- The time Sharon usually takes to complete the journey -/
def usual_time : ℝ := 200

/-- The time taken on the day with traffic -/
def traffic_time : ℝ := 300

/-- The speed reduction due to traffic in miles per hour -/
def speed_reduction : ℝ := 30

theorem sharons_journey :
  ∃ (initial_speed : ℝ),
    initial_speed > 0 ∧
    initial_speed * usual_time = total_distance ∧
    (total_distance / 2) / initial_speed +
    (total_distance / 2) / (initial_speed - speed_reduction / 60) = traffic_time :=
  sorry

end sharons_journey_l1731_173135


namespace ordering_abc_l1731_173124

theorem ordering_abc (a b c : ℝ) : 
  a = 6 - Real.log 2 - Real.log 3 →
  b = Real.exp 1 - Real.log 3 →
  c = Real.exp 2 - 2 →
  c > a ∧ a > b := by sorry

end ordering_abc_l1731_173124


namespace geometric_proportion_proof_l1731_173187

theorem geometric_proportion_proof :
  let a : ℝ := 21
  let b : ℝ := 7
  let c : ℝ := 9
  let d : ℝ := 3
  (a / b = c / d) ∧
  (a + d = 24) ∧
  (b + c = 16) ∧
  (a^2 + b^2 + c^2 + d^2 = 580) := by
  sorry

end geometric_proportion_proof_l1731_173187


namespace assignment_schemes_proof_l1731_173158

/-- The number of ways to assign 3 out of 5 volunteers to 3 distinct tasks -/
def assignment_schemes : ℕ := 60

/-- The total number of volunteers -/
def total_volunteers : ℕ := 5

/-- The number of volunteers to be selected -/
def selected_volunteers : ℕ := 3

/-- The number of tasks -/
def num_tasks : ℕ := 3

theorem assignment_schemes_proof :
  assignment_schemes = (total_volunteers.factorial) / ((total_volunteers - selected_volunteers).factorial) :=
by sorry

end assignment_schemes_proof_l1731_173158


namespace inequality_solution_l1731_173117

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≤ 5/4) ↔ (x < -8 ∨ (-2 < x ∧ x ≤ -8/5)) :=
by sorry

end inequality_solution_l1731_173117


namespace periodic_decimal_is_rational_l1731_173130

/-- Represents a periodic decimal fraction -/
structure PeriodicDecimal where
  nonRepeatingPart : List Nat
  repeatingPart : List Nat
  nonEmpty : repeatingPart.length > 0

/-- Converts a PeriodicDecimal to a real number -/
noncomputable def toReal (x : PeriodicDecimal) : Real := sorry

/-- Theorem: Every periodic decimal fraction is a rational number -/
theorem periodic_decimal_is_rational (x : PeriodicDecimal) :
  ∃ (p q : Int), q ≠ 0 ∧ toReal x = p / q := by sorry

end periodic_decimal_is_rational_l1731_173130


namespace equation_solvability_l1731_173199

theorem equation_solvability (a : ℝ) : 
  (∃ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x = 2 * a - 1) ↔ 
  (-1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ) :=
sorry

end equation_solvability_l1731_173199


namespace open_box_volume_l1731_173193

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_square_side : ℝ) 
  (h1 : sheet_length = 50) 
  (h2 : sheet_width = 36) 
  (h3 : cut_square_side = 8) : 
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 5440 := by
  sorry

#check open_box_volume

end open_box_volume_l1731_173193


namespace log_579_between_consecutive_integers_l1731_173191

theorem log_579_between_consecutive_integers : 
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 579 / Real.log 10 ∧ Real.log 579 / Real.log 10 < b ∧ a + b = 5 := by
  sorry

end log_579_between_consecutive_integers_l1731_173191


namespace count_five_or_six_base_eight_l1731_173192

/-- 
Given a positive integer n and a base b, returns true if n (when expressed in base b)
contains at least one digit that is either 5 or 6.
-/
def contains_five_or_six (n : ℕ+) (b : ℕ) : Prop := sorry

/-- 
Counts the number of positive integers up to n (inclusive) that contain
at least one 5 or 6 when expressed in base b.
-/
def count_with_five_or_six (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 
Theorem: The number of integers from 1 to 256 (inclusive) in base 8
that contain at least one 5 or 6 digit is equal to 220.
-/
theorem count_five_or_six_base_eight : 
  count_with_five_or_six 256 8 = 220 := by sorry

end count_five_or_six_base_eight_l1731_173192


namespace like_terms_sum_l1731_173116

theorem like_terms_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^2 * y^4 = 3 * y - x^n * y^(2*m)) → m + n = 4 := by
  sorry

end like_terms_sum_l1731_173116


namespace problem_statement_l1731_173162

theorem problem_statement : (π - 3.14) ^ 0 + (-0.125) ^ 2008 * 8 ^ 2008 = 2 := by
  sorry

end problem_statement_l1731_173162


namespace intersection_A_complement_B_range_of_a_l1731_173175

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | |x - 4| ≤ 2}
def B : Set ℝ := {x : ℝ | (5 - x) / (x + 1) > 0}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem 1: A ∩ (Uᶜ B) = [5,6]
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = Set.Icc 5 6 := by sorry

-- Theorem 2: If A ∩ C ≠ ∅, then a ∈ (2, +∞)
theorem range_of_a (a : ℝ) (h : (A ∩ C a).Nonempty) :
  a ∈ Set.Ioi 2 := by sorry

end intersection_A_complement_B_range_of_a_l1731_173175


namespace three_number_sum_l1731_173163

theorem three_number_sum : ∀ (a b c : ℝ),
  b = 150 →
  a = 2 * b →
  c = a / 3 →
  a + b + c = 550 := by
sorry

end three_number_sum_l1731_173163


namespace colored_pencils_erasers_difference_l1731_173105

/-- Proves that the difference between colored pencils and erasers left is 22 --/
theorem colored_pencils_erasers_difference :
  let initial_crayons : ℕ := 531
  let initial_erasers : ℕ := 38
  let initial_colored_pencils : ℕ := 67
  let final_crayons : ℕ := 391
  let final_erasers : ℕ := 28
  let final_colored_pencils : ℕ := 50
  final_colored_pencils - final_erasers = 22 := by
  sorry

end colored_pencils_erasers_difference_l1731_173105


namespace committee_formation_count_l1731_173159

def total_members : ℕ := 25
def male_members : ℕ := 15
def female_members : ℕ := 10
def committee_size : ℕ := 5
def min_females : ℕ := 2

theorem committee_formation_count : 
  (Finset.sum (Finset.range (committee_size - min_females + 1))
    (fun k => Nat.choose female_members (k + min_females) * 
              Nat.choose male_members (committee_size - k - min_females))) = 36477 := by
  sorry

end committee_formation_count_l1731_173159


namespace min_value_of_some_expression_l1731_173141

-- Define the expression
def expression (x : ℝ) (some_expression : ℝ) : ℝ :=
  |x - 4| + |x + 5| + |some_expression|

-- State the theorem
theorem min_value_of_some_expression :
  ∃ (some_expression : ℝ), 
    (∀ x : ℝ, expression x some_expression ≥ 10) ∧ 
    (∃ x : ℝ, expression x some_expression = 10) ∧
    |some_expression| = 1 := by
  sorry

end min_value_of_some_expression_l1731_173141


namespace bathtub_fill_time_l1731_173185

/-- Represents the filling and draining rates of a bathtub -/
structure BathtubRates where
  cold_fill_time : ℚ
  hot_fill_time : ℚ
  drain_time : ℚ

/-- Calculates the time to fill the bathtub with both taps open and drain unplugged -/
def fill_time (rates : BathtubRates) : ℚ :=
  1 / ((1 / rates.cold_fill_time) + (1 / rates.hot_fill_time) - (1 / rates.drain_time))

/-- Theorem: Given the specified filling and draining rates, the bathtub will fill in 5 minutes -/
theorem bathtub_fill_time (rates : BathtubRates) 
  (h1 : rates.cold_fill_time = 20 / 3)
  (h2 : rates.hot_fill_time = 8)
  (h3 : rates.drain_time = 40 / 3) :
  fill_time rates = 5 := by
  sorry

#eval fill_time { cold_fill_time := 20 / 3, hot_fill_time := 8, drain_time := 40 / 3 }

end bathtub_fill_time_l1731_173185


namespace system_solution_l1731_173173

theorem system_solution :
  ∃ (x y : ℝ), x = -1 ∧ y = -2 ∧ x - 3*y = 5 ∧ 4*x - 3*y = 2 := by
  sorry

end system_solution_l1731_173173


namespace treasure_points_l1731_173154

theorem treasure_points (total_treasures : ℕ) (total_score : ℕ) 
  (h1 : total_treasures = 7) (h2 : total_score = 35) : 
  (total_score / total_treasures : ℚ) = 5 := by
  sorry

end treasure_points_l1731_173154


namespace f_simplification_and_result_l1731_173179

noncomputable def f (α : ℝ) : ℝ :=
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi) ^ 2) /
  (Real.sin (α - Real.pi / 2) * Real.cos (Real.pi / 2 + α) * Real.tan (Real.pi - α))

theorem f_simplification_and_result (α : ℝ) :
  f α = Real.tan α ∧
  (f α = 2 → (3 * Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 7/3) :=
by sorry

end f_simplification_and_result_l1731_173179


namespace nested_squares_perimeter_difference_l1731_173109

/-- The difference between the perimeters of two nested squares -/
theorem nested_squares_perimeter_difference :
  ∀ (x : ℝ),
  x > 0 →
  let small_square_side : ℝ := x
  let large_square_side : ℝ := x + 8
  let small_perimeter : ℝ := 4 * small_square_side
  let large_perimeter : ℝ := 4 * large_square_side
  large_perimeter - small_perimeter = 32 :=
by
  sorry

#check nested_squares_perimeter_difference

end nested_squares_perimeter_difference_l1731_173109


namespace perfect_square_trinomial_l1731_173169

theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + (a - 1)*x + 25 = (x + b)^2) → (a = 11 ∨ a = -9) :=
by sorry

end perfect_square_trinomial_l1731_173169


namespace recurring_decimal_subtraction_l1731_173165

theorem recurring_decimal_subtraction : 
  (1 : ℚ) / 3 - (2 : ℚ) / 99 = (31 : ℚ) / 99 := by sorry

end recurring_decimal_subtraction_l1731_173165


namespace exists_m_not_greater_l1731_173115

theorem exists_m_not_greater (a b : ℝ) (h : a < b) : ∃ m : ℝ, m * a ≤ m * b := by
  sorry

end exists_m_not_greater_l1731_173115


namespace pharmacy_tubs_l1731_173128

def tubs_needed : ℕ := 100
def tubs_in_storage : ℕ := 20

def tubs_to_buy : ℕ := tubs_needed - tubs_in_storage

def tubs_from_new_vendor : ℕ := tubs_to_buy / 4

def tubs_from_usual_vendor : ℕ := tubs_needed - (tubs_in_storage + tubs_from_new_vendor)

theorem pharmacy_tubs :
  tubs_from_usual_vendor = 60 := by sorry

end pharmacy_tubs_l1731_173128


namespace inequality_proof_l1731_173137

theorem inequality_proof (a b c d x y u v : ℝ) (h : a * b * c * d > 0) :
  (a * x + b * u) * (a * v + b * y) * (c * x + d * v) * (c * u + d * y) ≥ 
  (a * c * u * v * x + b * c * u * x * y + a * d * v * x * y + b * d * u * v * y) * 
  (a * c * x + b * c * u + a * d * v + b * d * y) := by
  sorry

end inequality_proof_l1731_173137


namespace expression_value_l1731_173110

theorem expression_value : 
  (2023^3 - 3 * 2023^2 * 2024 + 4 * 2023 * 2024^2 - 2024^3 + 2) / (2023 * 2024) = 2023 := by
  sorry

end expression_value_l1731_173110


namespace janet_hourly_earnings_l1731_173102

/-- Calculates Janet's hourly earnings for moderating social media posts -/
theorem janet_hourly_earnings (cents_per_post : ℚ) (seconds_per_post : ℕ) : 
  cents_per_post = 25 → seconds_per_post = 10 → 
  (3600 / seconds_per_post) * cents_per_post = 9000 := by
  sorry

#check janet_hourly_earnings

end janet_hourly_earnings_l1731_173102


namespace fourth_student_is_18_l1731_173126

/-- Represents a systematic sampling of students -/
structure SystematicSample where
  total_students : ℕ
  sample_size : ℕ
  first_student : ℕ
  h_total_positive : 0 < total_students
  h_sample_positive : 0 < sample_size
  h_sample_size : sample_size ≤ total_students
  h_first_valid : first_student ≤ total_students

/-- The sampling interval for a systematic sample -/
def sampling_interval (s : SystematicSample) : ℕ :=
  s.total_students / s.sample_size

/-- The nth student in the sample -/
def nth_student (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_student + (n - 1) * sampling_interval s

/-- Theorem: In a systematic sample of 4 from 52, if 5, 31, and 44 are sampled, then 18 is the fourth -/
theorem fourth_student_is_18 (s : SystematicSample) 
    (h_total : s.total_students = 52)
    (h_sample : s.sample_size = 4)
    (h_first : s.first_student = 5)
    (h_third : nth_student s 3 = 31)
    (h_fourth : nth_student s 4 = 44) :
    nth_student s 2 = 18 := by
  sorry

end fourth_student_is_18_l1731_173126


namespace f_minimum_value_l1731_173100

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- State the theorem
theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ 3) ∧ (∃ x : ℝ, f x = 3) := by
  sorry

end f_minimum_value_l1731_173100


namespace unique_solutions_l1731_173183

/-- A triple of strictly positive integers (a, b, p) satisfies the equation if a^p = b! + p and p is prime. -/
def SatisfiesEquation (a b p : ℕ+) : Prop :=
  a ^ p.val = Nat.factorial b.val + p.val ∧ Nat.Prime p.val

theorem unique_solutions :
  ∀ a b p : ℕ+, SatisfiesEquation a b p →
    ((a = 2 ∧ b = 2 ∧ p = 2) ∨ (a = 3 ∧ b = 4 ∧ p = 3)) :=
by sorry

end unique_solutions_l1731_173183


namespace range_of_m_l1731_173112

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| ≤ 2
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the condition that ¬p is sufficient but not necessary for ¬q
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ¬(∀ x, ¬(q x m) → ¬(p x))

-- Theorem statement
theorem range_of_m :
  ∀ m, sufficient_not_necessary m ↔ (2 < m ∧ m < 4) :=
sorry

end range_of_m_l1731_173112


namespace divisibility_condition_l1731_173119

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔
    (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) := by
  sorry

end divisibility_condition_l1731_173119


namespace point_A_in_transformed_plane_l1731_173186

/-- The similarity transformation coefficient -/
def k : ℝ := -2

/-- The original plane equation -/
def plane_a (x y z : ℝ) : Prop := x - 2*y + z + 1 = 0

/-- The transformed plane equation -/
def plane_a' (x y z : ℝ) : Prop := x - 2*y + z - 2 = 0

/-- Point A -/
def point_A : ℝ × ℝ × ℝ := (2, 1, 2)

/-- Theorem stating that point A belongs to the image of plane a -/
theorem point_A_in_transformed_plane : 
  let (x, y, z) := point_A
  plane_a' x y z := by sorry

end point_A_in_transformed_plane_l1731_173186


namespace union_of_P_and_Q_l1731_173157

-- Define the sets P and Q
def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | -2 < x ∧ x < 0}

-- Define the open interval (-2, 1)
def openInterval : Set ℝ := {x | -2 < x ∧ x < 1}

-- Theorem statement
theorem union_of_P_and_Q : P ∪ Q = openInterval := by sorry

end union_of_P_and_Q_l1731_173157


namespace garden_length_is_32_l1731_173120

/-- Calculates the length of a garden with mango trees -/
def garden_length (num_columns : ℕ) (tree_distance : ℝ) (boundary : ℝ) : ℝ :=
  (num_columns - 1 : ℝ) * tree_distance + 2 * boundary

/-- Theorem: The length of the garden is 32 meters -/
theorem garden_length_is_32 :
  garden_length 12 2 5 = 32 := by
  sorry

end garden_length_is_32_l1731_173120


namespace tom_books_count_l1731_173177

/-- Given that Joan has 10 books and the total number of books is 48,
    prove that Tom has 38 books. -/
theorem tom_books_count (joan_books : ℕ) (total_books : ℕ) (tom_books : ℕ) 
    (h1 : joan_books = 10)
    (h2 : total_books = 48)
    (h3 : tom_books + joan_books = total_books) : 
  tom_books = 38 := by
sorry

end tom_books_count_l1731_173177


namespace single_color_subgraph_exists_l1731_173190

/-- A graph where each pair of vertices is connected by exactly one of two types of edges -/
structure TwoColorGraph (α : Type*) where
  vertices : Set α
  edge_type1 : α → α → Prop
  edge_type2 : α → α → Prop
  edge_exists : ∀ (v w : α), v ∈ vertices → w ∈ vertices → v ≠ w → 
    (edge_type1 v w ∧ ¬edge_type2 v w) ∨ (edge_type2 v w ∧ ¬edge_type1 v w)

/-- A subgraph that includes all vertices and uses only one type of edge -/
def SingleColorSubgraph {α : Type*} (G : TwoColorGraph α) :=
  {H : Set (α × α) // 
    (∀ v ∈ G.vertices, ∃ w, (v, w) ∈ H ∨ (w, v) ∈ H) ∧
    (∀ (v w : α), (v, w) ∈ H → G.edge_type1 v w) ∨
    (∀ (v w : α), (v, w) ∈ H → G.edge_type2 v w)}

/-- The main theorem: there always exists a single-color subgraph -/
theorem single_color_subgraph_exists {α : Type*} (G : TwoColorGraph α) :
  Nonempty (SingleColorSubgraph G) := by
  sorry

end single_color_subgraph_exists_l1731_173190


namespace complex_modulus_l1731_173195

theorem complex_modulus (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : 
  Complex.abs z = 2 * Real.sqrt 313 / 13 := by sorry

end complex_modulus_l1731_173195


namespace geometric_sequence_sum_6_l1731_173106

/-- A geometric sequence with its partial sums -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1
  sum_formula : ∀ n : ℕ, n > 0 → S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

/-- The main theorem -/
theorem geometric_sequence_sum_6 (seq : GeometricSequence) 
    (h2 : seq.S 2 = 3) (h4 : seq.S 4 = 15) : seq.S 6 = 63 := by
  sorry

end geometric_sequence_sum_6_l1731_173106


namespace unique_prime_ending_l1731_173151

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def number (A : ℕ) : ℕ := 130400 + A

theorem unique_prime_ending :
  ∃! A : ℕ, A < 10 ∧ is_prime (number A) :=
sorry

end unique_prime_ending_l1731_173151


namespace intersection_equals_set_iff_complement_subset_l1731_173129

universe u

theorem intersection_equals_set_iff_complement_subset {U : Type u} (A B : Set U) :
  A ∩ B = A ↔ (Bᶜ : Set U) ⊆ (Aᶜ : Set U) := by sorry

end intersection_equals_set_iff_complement_subset_l1731_173129


namespace single_burger_cost_l1731_173180

/-- Proves that the cost of a single burger is $1.00 given the specified conditions -/
theorem single_burger_cost
  (total_spent : ℝ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (double_burger_cost : ℝ)
  (h1 : total_spent = 74.50)
  (h2 : total_hamburgers = 50)
  (h3 : double_burgers = 49)
  (h4 : double_burger_cost = 1.50) :
  total_spent - (double_burgers * double_burger_cost) = 1.00 := by
  sorry

end single_burger_cost_l1731_173180


namespace min_value_on_interval_l1731_173166

def f (x : ℝ) := x^2 - x

theorem min_value_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ f c = -1/4 ∧ ∀ x ∈ Set.Icc 0 1, f x ≥ f c := by
  sorry

end min_value_on_interval_l1731_173166


namespace evaluate_expression_l1731_173146

theorem evaluate_expression : -25 + 5 * (4^2 / 2) = 15 := by
  sorry

end evaluate_expression_l1731_173146


namespace imaginary_part_of_complex_fraction_l1731_173150

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + 2*i) / (3 - 4*i)) = 2/5 := by
  sorry

end imaginary_part_of_complex_fraction_l1731_173150


namespace mass_of_cao_l1731_173107

/-- Calculates the mass of a given number of moles of a compound -/
def calculate_mass (moles : ℝ) (atomic_mass_ca : ℝ) (atomic_mass_o : ℝ) : ℝ :=
  moles * (atomic_mass_ca + atomic_mass_o)

/-- Theorem: The mass of 8 moles of CaO containing only 42Ca is 464 grams -/
theorem mass_of_cao : calculate_mass 8 42 16 = 464 := by
  sorry

end mass_of_cao_l1731_173107


namespace triangle_angle_expression_range_l1731_173131

theorem triangle_angle_expression_range (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -25/16 < 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) ∧ 
  3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) < 6 :=
sorry

end triangle_angle_expression_range_l1731_173131


namespace fraction_order_l1731_173197

theorem fraction_order : 
  let f1 := 21 / 16
  let f2 := 25 / 19
  let f3 := 23 / 17
  let f4 := 27 / 20
  f1 < f2 ∧ f2 < f4 ∧ f4 < f3 := by
  sorry

end fraction_order_l1731_173197


namespace salary_increase_l1731_173198

/-- Regression equation for monthly salary based on labor productivity -/
def salary_equation (x : ℝ) : ℝ := 50 + 80 * x

/-- Theorem stating that an increase of 1000 yuan in labor productivity
    results in an increase of 80 yuan in salary -/
theorem salary_increase (x : ℝ) :
  salary_equation (x + 1) - salary_equation x = 80 := by
  sorry

#check salary_increase

end salary_increase_l1731_173198


namespace cubic_equation_solution_l1731_173172

theorem cubic_equation_solution :
  ∀ x : ℝ, x^3 + (x + 1)^3 + (x + 2)^3 = (x + 3)^3 ↔ x = 3 := by sorry

end cubic_equation_solution_l1731_173172


namespace sum_squares_inequality_l1731_173184

theorem sum_squares_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) :
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := by
  sorry

end sum_squares_inequality_l1731_173184
