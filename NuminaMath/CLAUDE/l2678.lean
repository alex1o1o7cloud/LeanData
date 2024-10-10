import Mathlib

namespace geometric_increasing_iff_second_greater_first_l2678_267895

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem: For a geometric sequence with positive first term,
    the second term being greater than the first is equivalent to
    the sequence being increasing -/
theorem geometric_increasing_iff_second_greater_first
    (a : ℕ → ℝ) (h_geom : GeometricSequence a) (h_pos : a 1 > 0) :
    a 1 < a 2 ↔ IncreasingSequence a :=
  sorry

end geometric_increasing_iff_second_greater_first_l2678_267895


namespace remaining_milk_calculation_l2678_267800

/-- The amount of milk arranged by the shop owner in liters -/
def total_milk : ℝ := 21.52

/-- The amount of milk sold in liters -/
def sold_milk : ℝ := 12.64

/-- The amount of remaining milk in liters -/
def remaining_milk : ℝ := total_milk - sold_milk

theorem remaining_milk_calculation : remaining_milk = 8.88 := by
  sorry

end remaining_milk_calculation_l2678_267800


namespace triangle_inequality_l2678_267884

/-- Given a triangle ABC with sides a, b, c, heights h_a, h_b, h_c, area Δ, and a positive real number n,
    the inequality (ah_b)^n + (bh_c)^n + (ch_a)^n ≥ 3 * 2^n * Δ^n holds. -/
theorem triangle_inequality (a b c h_a h_b h_c Δ : ℝ) (n : ℝ) 
    (h_pos : n > 0)
    (h_heights : h_a = 2 * Δ / a ∧ h_b = 2 * Δ / b ∧ h_c = 2 * Δ / c)
    (h_area : Δ = a * h_a / 2 ∧ Δ = b * h_b / 2 ∧ Δ = c * h_c / 2) :
  (a * h_b)^n + (b * h_c)^n + (c * h_a)^n ≥ 3 * 2^n * Δ^n := by
  sorry

end triangle_inequality_l2678_267884


namespace train_journey_time_l2678_267807

/-- Proves that if a train moving at 6/7 of its usual speed is 10 minutes late, then its usual journey time is 1 hour -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → usual_time > 0 →
  (6 / 7 * usual_speed) * (usual_time + 1 / 6) = usual_speed * usual_time →
  usual_time = 1 := by
sorry

end train_journey_time_l2678_267807


namespace solution_difference_l2678_267841

theorem solution_difference (p q : ℝ) : 
  (p - 2) * (p + 4) = 26 * p - 100 →
  (q - 2) * (q + 4) = 26 * q - 100 →
  p ≠ q →
  p > q →
  p - q = 4 * Real.sqrt 13 := by sorry

end solution_difference_l2678_267841


namespace anna_savings_account_l2678_267802

def geometricSeriesSum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem anna_savings_account (a : ℕ) (r : ℕ) (target : ℕ) :
  a = 2 → r = 2 → target = 500 →
  (∀ k < 8, geometricSeriesSum a r k < target) ∧
  geometricSeriesSum a r 8 ≥ target :=
by sorry

end anna_savings_account_l2678_267802


namespace min_k_value_l2678_267814

def is_valid (n k : ℕ) : Prop :=
  ∀ i ∈ Finset.range (k - 1), n % (i + 2) = i + 1

theorem min_k_value :
  ∃ (k : ℕ), k > 0 ∧
  (∃ (n : ℕ), n > 2000 ∧ n < 3000 ∧ is_valid n k ∧
    ∀ (m : ℕ), m < n → ¬(is_valid m k)) ∧
  ∀ (j : ℕ), j < k →
    ¬(∃ (n : ℕ), n > 2000 ∧ n < 3000 ∧ is_valid n j ∧
      ∀ (m : ℕ), m < n → ¬(is_valid m j)) ∧
  k = 9 :=
sorry

end min_k_value_l2678_267814


namespace evening_customers_is_40_l2678_267896

/-- Represents the revenue and customer data for a movie theater on a Friday night. -/
structure TheaterData where
  matineePrice : ℕ
  eveningPrice : ℕ
  openingNightPrice : ℕ
  popcornPrice : ℕ
  matineeCustomers : ℕ
  openingNightCustomers : ℕ
  totalRevenue : ℕ

/-- Calculates the number of evening customers based on the theater data. -/
def eveningCustomers (data : TheaterData) : ℕ :=
  let totalCustomers := data.matineeCustomers + data.openingNightCustomers + (data.totalRevenue - 
    (data.matineePrice * data.matineeCustomers + 
     data.openingNightPrice * data.openingNightCustomers + 
     data.popcornPrice * (data.matineeCustomers + data.openingNightCustomers) / 2)) / data.eveningPrice
  (totalCustomers - data.matineeCustomers - data.openingNightCustomers)

/-- Theorem stating that the number of evening customers is 40 given the specific theater data. -/
theorem evening_customers_is_40 (data : TheaterData) 
  (h1 : data.matineePrice = 5)
  (h2 : data.eveningPrice = 7)
  (h3 : data.openingNightPrice = 10)
  (h4 : data.popcornPrice = 10)
  (h5 : data.matineeCustomers = 32)
  (h6 : data.openingNightCustomers = 58)
  (h7 : data.totalRevenue = 1670) :
  eveningCustomers data = 40 := by
  sorry

end evening_customers_is_40_l2678_267896


namespace ratio_of_percentages_l2678_267803

theorem ratio_of_percentages (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hR : R = 0.6 * P)
  (hN : N = 0.75 * R)
  (hP : P ≠ 0) :
  M / N = 2 / 9 := by
  sorry

end ratio_of_percentages_l2678_267803


namespace inequality_always_true_l2678_267804

theorem inequality_always_true (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end inequality_always_true_l2678_267804


namespace arithmetic_mean_of_first_four_primes_reciprocals_l2678_267858

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => 1 / x)
  (reciprocals.sum / 4 : ℚ) = 247 / 840 := by
  sorry

end arithmetic_mean_of_first_four_primes_reciprocals_l2678_267858


namespace triangle_max_area_l2678_267837

/-- Given a triangle ABC with sides a, b, c, where S = a² - (b-c)² and b + c = 8,
    the maximum value of S is 64/17 -/
theorem triangle_max_area (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
    (h2 : b + c = 8) (h3 : ∀ S : ℝ, S = a^2 - (b-c)^2) : 
    ∃ (S : ℝ), S ≤ 64/17 ∧ ∃ (a' b' c' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ b' + c' = 8 ∧ 
    64/17 = a'^2 - (b'-c')^2 := by
  sorry

end triangle_max_area_l2678_267837


namespace solve_for_a_l2678_267886

theorem solve_for_a : ∃ a : ℝ, (2 : ℝ) - a * (1 : ℝ) = 3 ∧ a = -1 := by
  sorry

end solve_for_a_l2678_267886


namespace sqrt_x_minus_2_meaningful_l2678_267824

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ (y : ℝ), y^2 = x - 2) ↔ x ≥ 2 :=
by sorry

end sqrt_x_minus_2_meaningful_l2678_267824


namespace perpendicular_line_equation_l2678_267823

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_line_equation :
  ∃ (x0 y0 : ℝ), intersection_point x0 y0 ∧
  ∃ (m : ℝ), perpendicular m (-2) ∧
  ∀ (x y : ℝ), y - y0 = m * (x - x0) ↔ x - 2 * y + 7 = 0 :=
sorry

end perpendicular_line_equation_l2678_267823


namespace quadratic_discriminant_zero_implies_geometric_progression_l2678_267892

/-- Given a quadratic equation ax^2 + 2bx + c = 0 with discriminant zero,
    prove that a, b, and c form a geometric progression -/
theorem quadratic_discriminant_zero_implies_geometric_progression
  (a b c : ℝ) (h : a ≠ 0) :
  (2 * b)^2 - 4 * a * c = 0 →
  ∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r :=
by sorry

end quadratic_discriminant_zero_implies_geometric_progression_l2678_267892


namespace find_c_l2678_267843

theorem find_c (a b c : ℝ) 
  (eq1 : a + b = 3) 
  (eq2 : a * c + b = 18) 
  (eq3 : b * c + a = 6) : 
  c = 7 := by sorry

end find_c_l2678_267843


namespace eye_color_hair_color_proportions_l2678_267893

/-- Represents the population characteristics of a kingdom -/
structure Kingdom where
  total : ℕ
  blondes : ℕ
  blue_eyes : ℕ
  blonde_blue_eyes : ℕ
  blonde_blue_eyes_le_blondes : blonde_blue_eyes ≤ blondes
  blonde_blue_eyes_le_blue_eyes : blonde_blue_eyes ≤ blue_eyes
  blondes_le_total : blondes ≤ total
  blue_eyes_le_total : blue_eyes ≤ total

/-- The main theorem about eye color and hair color proportions in the kingdom -/
theorem eye_color_hair_color_proportions (k : Kingdom) :
  (k.blonde_blue_eyes : ℚ) / k.blue_eyes > (k.blondes : ℚ) / k.total →
  (k.blonde_blue_eyes : ℚ) / k.blondes > (k.blue_eyes : ℚ) / k.total :=
by
  sorry

end eye_color_hair_color_proportions_l2678_267893


namespace intersection_points_form_convex_polygon_l2678_267871

/-- Represents a point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an L-shaped figure -/
structure LShape where
  A : Point
  longSegment : List Point
  shortSegment : Point

/-- Represents the problem setup -/
structure ProblemSetup where
  L1 : LShape
  L2 : LShape
  n : ℕ
  intersectionPoints : List Point

/-- Predicate to check if a list of points forms a convex polygon -/
def IsConvexPolygon (points : List Point) : Prop := sorry

/-- Main theorem statement -/
theorem intersection_points_form_convex_polygon (setup : ProblemSetup) :
  IsConvexPolygon setup.intersectionPoints :=
by sorry

end intersection_points_form_convex_polygon_l2678_267871


namespace parrot_initial_phrases_l2678_267816

/-- The number of phrases a parrot initially knew, given the current number of phrases,
    the rate of learning, and the duration of ownership. -/
theorem parrot_initial_phrases (current_phrases : ℕ) (phrases_per_week : ℕ) (days_owned : ℕ) 
    (h1 : current_phrases = 17)
    (h2 : phrases_per_week = 2)
    (h3 : days_owned = 49) :
  current_phrases - (days_owned / 7 * phrases_per_week) = 3 := by
  sorry

#check parrot_initial_phrases

end parrot_initial_phrases_l2678_267816


namespace ripe_oranges_harvested_per_day_l2678_267835

/-- The number of days of harvest -/
def harvest_days : ℕ := 25

/-- The total number of sacks of ripe oranges after the harvest period -/
def total_ripe_oranges : ℕ := 2050

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges_per_day : ℕ := total_ripe_oranges / harvest_days

theorem ripe_oranges_harvested_per_day :
  ripe_oranges_per_day = 82 := by
  sorry

end ripe_oranges_harvested_per_day_l2678_267835


namespace sequence_max_term_l2678_267846

def a (n : ℕ) : ℤ := -2 * n^2 + 29 * n + 3

theorem sequence_max_term :
  ∃ (k : ℕ), k = 7 ∧ a k = 108 ∧ ∀ (n : ℕ), a n ≤ a k :=
sorry

end sequence_max_term_l2678_267846


namespace prize_distribution_l2678_267898

/-- The number of ways to distribute prizes in a class --/
theorem prize_distribution (n m : ℕ) (hn : n = 28) (hm : m = 4) :
  /- Identical prizes, at most one per student -/
  (n.choose m = 20475) ∧ 
  /- Identical prizes, more than one per student allowed -/
  ((n + m - 1).choose m = 31465) ∧ 
  /- Distinct prizes, at most one per student -/
  (n * (n - 1) * (n - 2) * (n - 3) = 491400) ∧ 
  /- Distinct prizes, more than one per student allowed -/
  (n ^ m = 614656) :=
by sorry

end prize_distribution_l2678_267898


namespace prob_pass_exactly_once_l2678_267854

/-- The probability of passing a single computer test -/
def p : ℚ := 1 / 3

/-- The number of times the test is taken -/
def n : ℕ := 3

/-- The number of times we want the event to occur -/
def k : ℕ := 1

/-- The probability of passing exactly k times in n independent trials -/
def prob_exactly_k (p : ℚ) (n k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem prob_pass_exactly_once :
  prob_exactly_k p n k = 4 / 9 := by
  sorry

end prob_pass_exactly_once_l2678_267854


namespace smallest_x_absolute_value_equation_l2678_267880

theorem smallest_x_absolute_value_equation : 
  ∃ x : ℝ, (∀ y : ℝ, |4*y + 9| = 37 → x ≤ y) ∧ |4*x + 9| = 37 := by
  sorry

end smallest_x_absolute_value_equation_l2678_267880


namespace train_speed_calculation_l2678_267869

theorem train_speed_calculation (train_length platform_length crossing_time : ℝ) 
  (h1 : train_length = 120)
  (h2 : platform_length = 380.04)
  (h3 : crossing_time = 25) :
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * 3.6
  ∃ ε > 0, abs (speed_kmh - 72.01) < ε :=
by sorry

end train_speed_calculation_l2678_267869


namespace star_property_l2678_267872

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := 
  fun (a, b) (c, d) => (a + d, b - c)

theorem star_property : 
  ∃ (y : ℤ), star (3, y) (4, 2) = star (4, 5) (1, 1) := by
  sorry

end star_property_l2678_267872


namespace twelve_pointed_stars_count_l2678_267830

/-- Counts the number of non-similar regular n-pointed stars -/
def count_non_similar_stars (n : ℕ) : ℕ :=
  let valid_m := (Finset.range (n - 1)).filter (λ m => m > 1 ∧ m < n - 1 ∧ Nat.gcd m n = 1)
  (valid_m.card + 1) / 2

/-- The number of non-similar regular 12-pointed stars is 1 -/
theorem twelve_pointed_stars_count :
  count_non_similar_stars 12 = 1 := by
  sorry

end twelve_pointed_stars_count_l2678_267830


namespace sqrt_3_times_sqrt_12_l2678_267885

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l2678_267885


namespace perp_planes_necessary_not_sufficient_l2678_267838

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the containment relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- Main theorem
theorem perp_planes_necessary_not_sufficient 
  (α β : Plane) (m : Line) 
  (h_different : α ≠ β)
  (h_contained : contained_in m α) :
  (∀ m, contained_in m α → perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m, contained_in m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end perp_planes_necessary_not_sufficient_l2678_267838


namespace roots_sum_greater_than_2a_l2678_267897

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.log x + a / x

theorem roots_sum_greater_than_2a
  (h₁ : x₁ > 0)
  (h₂ : x₂ > 0)
  (h₃ : x₁ ≠ x₂)
  (h₄ : f a x₁ = a / 2)
  (h₅ : f a x₂ = a / 2) :
  x₁ + x₂ > 2 * a :=
sorry

end roots_sum_greater_than_2a_l2678_267897


namespace inverse_proportion_ratio_l2678_267853

/-- Given that x is inversely proportional to y, prove that y₁/y₂ = 5/3 when x₁/x₂ = 3/5 -/
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx : x₁ ≠ 0 ∧ x₂ ≠ 0) (hy : y₁ ≠ 0 ∧ y₂ ≠ 0)
  (h_prop : ∃ (k : ℝ), ∀ (x y : ℝ), x * y = k)
  (h_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 := by
  sorry

end inverse_proportion_ratio_l2678_267853


namespace function_equation_implies_identity_l2678_267831

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^4 + 4*y^4) = (f (x^2))^2 + 4*y^3 * f y) : 
  ∀ x : ℝ, f x = x := by
  sorry

end function_equation_implies_identity_l2678_267831


namespace albert_cabbage_count_l2678_267825

-- Define the number of rows in Albert's cabbage patch
def num_rows : ℕ := 12

-- Define the number of cabbage heads in each row
def heads_per_row : ℕ := 15

-- Define the total number of cabbage heads
def total_heads : ℕ := num_rows * heads_per_row

-- Theorem statement
theorem albert_cabbage_count : total_heads = 180 := by
  sorry

end albert_cabbage_count_l2678_267825


namespace three_similar_points_l2678_267801

/-- Right trapezoid ABCD with given side lengths -/
structure RightTrapezoid where
  AB : ℝ
  AD : ℝ
  BC : ℝ
  ab_positive : AB > 0
  ad_positive : AD > 0
  bc_positive : BC > 0

/-- Point P on side AB of the trapezoid -/
def PointP (t : RightTrapezoid) := { x : ℝ // 0 ≤ x ∧ x ≤ t.AB }

/-- Condition for triangle PAD to be similar to triangle PBC -/
def IsSimilar (t : RightTrapezoid) (p : PointP t) : Prop :=
  p.val / (t.AB - p.val) = t.AD / t.BC ∨ p.val / t.BC = t.AD / (t.AB - p.val)

/-- The main theorem stating that there are exactly 3 points P satisfying the similarity condition -/
theorem three_similar_points (t : RightTrapezoid) 
  (h1 : t.AB = 7) (h2 : t.AD = 2) (h3 : t.BC = 3) : 
  ∃! (s : Finset (PointP t)), s.card = 3 ∧ ∀ p ∈ s, IsSimilar t p := by
  sorry

end three_similar_points_l2678_267801


namespace remainder_problem_l2678_267850

theorem remainder_problem (N : ℤ) (h : N % 350 = 37) : (2 * N) % 13 = 9 := by
  sorry

end remainder_problem_l2678_267850


namespace games_within_division_is_48_l2678_267844

/-- Represents a baseball league with two divisions -/
structure BaseballLeague where
  /-- Number of games played against each team in the same division -/
  N : ℕ
  /-- Number of games played against each team in the other division -/
  M : ℕ
  /-- N is greater than 2M -/
  h1 : N > 2 * M
  /-- M is greater than 4 -/
  h2 : M > 4
  /-- Total number of games in the schedule is 76 -/
  h3 : 3 * N + 4 * M = 76

/-- The number of games a team plays within its own division -/
def gamesWithinDivision (league : BaseballLeague) : ℕ := 3 * league.N

/-- Theorem stating that the number of games within division is 48 -/
theorem games_within_division_is_48 (league : BaseballLeague) :
  gamesWithinDivision league = 48 := by
  sorry


end games_within_division_is_48_l2678_267844


namespace otimes_nested_l2678_267811

/-- Definition of the ⊗ operation -/
def otimes (g y : ℝ) : ℝ := g^2 + 2*y

/-- Theorem stating the result of g ⊗ (g ⊗ g) -/
theorem otimes_nested (g : ℝ) : otimes g (otimes g g) = g^4 + 4*g^3 + 6*g^2 + 4*g := by
  sorry

end otimes_nested_l2678_267811


namespace quadratic_polynomial_property_l2678_267808

theorem quadratic_polynomial_property (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let p := fun x => (a^2 + a*b + b^2 + a*c + b*c + c^2) * x^2 - 
                    (a + b) * (b + c) * (a + c) * x + 
                    a * b * c * (a + b + c)
  p a = a^4 ∧ p b = b^4 ∧ p c = c^4 := by
  sorry

end quadratic_polynomial_property_l2678_267808


namespace handmade_sweater_cost_l2678_267852

/-- The cost of a handmade sweater given Maria's shopping scenario -/
theorem handmade_sweater_cost 
  (num_sweaters num_scarves : ℕ)
  (scarf_cost : ℚ)
  (initial_savings remaining_savings : ℚ)
  (h1 : num_sweaters = 6)
  (h2 : num_scarves = 6)
  (h3 : scarf_cost = 20)
  (h4 : initial_savings = 500)
  (h5 : remaining_savings = 200) :
  (initial_savings - remaining_savings - num_scarves * scarf_cost) / num_sweaters = 30 := by
  sorry

end handmade_sweater_cost_l2678_267852


namespace triangle_angle_measure_l2678_267859

-- Define the triangle ABC
theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  -- Conditions
  (a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = (Real.sqrt 3 / 2) * b) →
  (c > b) →
  -- Conclusion
  B = π / 3 := by
  sorry

end triangle_angle_measure_l2678_267859


namespace area_of_three_sectors_l2678_267815

/-- The area of a figure formed by three sectors of a circle,
    where each sector subtends an angle of 40° at the center
    and the circle has a radius of 15. -/
theorem area_of_three_sectors (r : ℝ) (angle : ℝ) (n : ℕ) :
  r = 15 →
  angle = 40 * π / 180 →
  n = 3 →
  n * (angle / (2 * π) * π * r^2) = 75 * π := by
  sorry

end area_of_three_sectors_l2678_267815


namespace sine_cosine_inequality_l2678_267855

theorem sine_cosine_inequality (b a : ℝ) (hb : 0 < b ∧ b < 1) (ha : 0 < a ∧ a < Real.pi / 2) :
  Real.rpow b (Real.sin a) < Real.rpow b (Real.sin a) ∧ Real.rpow b (Real.sin a) < Real.rpow b (Real.cos a) := by
  sorry

end sine_cosine_inequality_l2678_267855


namespace simplify_expression_l2678_267817

theorem simplify_expression (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) + 1 = x^4 := by
  sorry

end simplify_expression_l2678_267817


namespace simplified_irrational_expression_l2678_267891

theorem simplified_irrational_expression :
  ∃ (a b c : ℤ), 
    (c > 0) ∧ 
    (∀ (a' b' c' : ℤ), c' > 0 → 
      Real.sqrt 11 + 2 / Real.sqrt 11 + Real.sqrt 2 + 3 / Real.sqrt 2 = (a' * Real.sqrt 11 + b' * Real.sqrt 2) / c' → 
      c ≤ c') ∧
    Real.sqrt 11 + 2 / Real.sqrt 11 + Real.sqrt 2 + 3 / Real.sqrt 2 = (a * Real.sqrt 11 + b * Real.sqrt 2) / c ∧
    a = 11 ∧ b = 44 ∧ c = 22 := by
  sorry

end simplified_irrational_expression_l2678_267891


namespace lucas_150_mod_9_l2678_267874

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

theorem lucas_150_mod_9 : lucas 149 % 9 = 3 := by sorry

end lucas_150_mod_9_l2678_267874


namespace inequality_proof_l2678_267829

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^3 / ((1+y)*(1+z))) + (y^3 / ((1+z)*(1+x))) + (z^3 / ((1+x)*(1+y))) ≥ 3/4 := by
  sorry

end inequality_proof_l2678_267829


namespace factor_4t_squared_minus_100_l2678_267888

theorem factor_4t_squared_minus_100 (t : ℝ) : 4 * t^2 - 100 = (2*t - 10) * (2*t + 10) := by
  sorry

end factor_4t_squared_minus_100_l2678_267888


namespace least_multiple_72_112_l2678_267827

theorem least_multiple_72_112 : 
  (∀ k : ℕ, k > 0 ∧ k < 14 → ¬(112 ∣ 72 * k)) ∧ (112 ∣ 72 * 14) :=
sorry

end least_multiple_72_112_l2678_267827


namespace square_of_real_not_always_positive_l2678_267882

theorem square_of_real_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end square_of_real_not_always_positive_l2678_267882


namespace sin_alpha_value_l2678_267834

/-- 
Given an angle α with vertex at the origin, initial side on the positive x-axis,
and terminal side on the ray 3x + 4y = 0 with x > 0, prove that sin α = -3/5.
-/
theorem sin_alpha_value (α : Real) : 
  (∃ (x y : Real), x > 0 ∧ 3 * x + 4 * y = 0 ∧ 
   x = Real.cos α ∧ y = Real.sin α) → 
  Real.sin α = -3/5 := by
  sorry

end sin_alpha_value_l2678_267834


namespace average_age_decrease_l2678_267848

theorem average_age_decrease (initial_avg : ℝ) : 
  let initial_total := 10 * initial_avg
  let new_total := initial_total - 45 + 15
  let new_avg := new_total / 10
  initial_avg - new_avg = 3 := by sorry

end average_age_decrease_l2678_267848


namespace remainder_91_92_mod_100_l2678_267833

theorem remainder_91_92_mod_100 : 91^92 % 100 = 81 := by
  sorry

end remainder_91_92_mod_100_l2678_267833


namespace square_sum_reciprocal_l2678_267857

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) :
  x^2 + (1 / x)^2 = 23 := by
sorry

end square_sum_reciprocal_l2678_267857


namespace rectangle_perimeter_is_128_l2678_267810

/-- Represents a rectangle with an inscribed ellipse -/
structure RectangleWithEllipse where
  rect_area : ℝ
  ellipse_area : ℝ
  major_axis : ℝ

/-- The perimeter of the rectangle given the specified conditions -/
def rectangle_perimeter (r : RectangleWithEllipse) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the rectangle under given conditions -/
theorem rectangle_perimeter_is_128 (r : RectangleWithEllipse) 
  (h1 : r.rect_area = 4032)
  (h2 : r.ellipse_area = 4032 * Real.pi)
  (h3 : r.major_axis = 2 * rectangle_perimeter r / 2) : 
  rectangle_perimeter r = 128 := by
  sorry

end rectangle_perimeter_is_128_l2678_267810


namespace equation1_solution_equation2_no_solution_l2678_267873

-- Define the first equation
def equation1 (x : ℝ) : Prop := 2 / x = 3 / (x + 2)

-- Define the second equation
def equation2 (x : ℝ) : Prop := 5 / (x - 2) + 1 = (x - 7) / (2 - x)

-- Theorem for the first equation
theorem equation1_solution :
  ∃! x : ℝ, equation1 x ∧ x ≠ 0 ∧ x + 2 ≠ 0 := by sorry

-- Theorem for the second equation
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x ∧ x ≠ 2 := by sorry

end equation1_solution_equation2_no_solution_l2678_267873


namespace coffee_package_size_l2678_267887

/-- Proves that the size of the larger coffee package is 10 ounces given the conditions -/
theorem coffee_package_size (total_coffee : ℕ) (larger_package_count : ℕ) 
  (small_package_size : ℕ) (small_package_count : ℕ) (larger_package_size : ℕ) :
  total_coffee = 115 ∧ 
  larger_package_count = 7 ∧
  small_package_size = 5 ∧
  small_package_count = larger_package_count + 2 ∧
  total_coffee = larger_package_count * larger_package_size + small_package_count * small_package_size →
  larger_package_size = 10 := by
  sorry

#check coffee_package_size

end coffee_package_size_l2678_267887


namespace percentage_passed_both_subjects_l2678_267864

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 32)
  (h2 : failed_english = 56)
  (h3 : failed_both = 12) :
  100 - (failed_hindi + failed_english - failed_both) = 24 := by
sorry

end percentage_passed_both_subjects_l2678_267864


namespace mason_savings_l2678_267840

theorem mason_savings (total_savings : ℚ) (days : ℕ) (dime_value : ℚ) : 
  total_savings = 3 → days = 30 → dime_value = 0.1 → 
  (total_savings / days) * dime_value = 0.01 := by
sorry

end mason_savings_l2678_267840


namespace january_bill_is_120_l2678_267806

/-- Represents the oil bill for a month -/
structure OilBill where
  amount : ℚ

/-- Represents the oil bills for three months -/
structure ThreeMonthBills where
  january : OilBill
  february : OilBill
  march : OilBill

/-- The conditions given in the problem -/
def satisfiesConditions (bills : ThreeMonthBills) : Prop :=
  let j := bills.january.amount
  let f := bills.february.amount
  let m := bills.march.amount
  f / j = 3 / 2 ∧
  f / m = 4 / 5 ∧
  (f + 20) / j = 5 / 3 ∧
  (f + 20) / m = 2 / 3

/-- The theorem to be proved -/
theorem january_bill_is_120 (bills : ThreeMonthBills) :
  satisfiesConditions bills → bills.january.amount = 120 := by
  sorry

end january_bill_is_120_l2678_267806


namespace opposite_numbers_pairs_l2678_267851

theorem opposite_numbers_pairs (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ 0) :
  ((-a) + b ≠ 0) ∧
  ((-a) + (-b) = 0) ∧
  (|a| + |b| ≠ 0) ∧
  (a^2 + b^2 ≠ 0) :=
by sorry

end opposite_numbers_pairs_l2678_267851


namespace sum_of_negatives_l2678_267828

theorem sum_of_negatives : (-4) + (-6) = -10 := by
  sorry

end sum_of_negatives_l2678_267828


namespace sum_of_a_and_c_l2678_267847

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48) 
  (h2 : b + d = 6) : 
  a + c = 8 := by
sorry

end sum_of_a_and_c_l2678_267847


namespace unique_solution_to_equation_l2678_267867

theorem unique_solution_to_equation :
  ∀ x y z : ℝ,
  x^2 + 2*x + y^2 + 4*y + z^2 + 6*z = -14 →
  x = -1 ∧ y = -2 ∧ z = -3 :=
by sorry

end unique_solution_to_equation_l2678_267867


namespace M_elements_l2678_267822

def M : Set (ℕ × ℕ) := {p | p.1 + p.2 ≤ 1}

theorem M_elements : M = {(0, 0), (0, 1), (1, 0)} := by
  sorry

end M_elements_l2678_267822


namespace system_solution_l2678_267862

theorem system_solution :
  ∃ (x y : ℝ), x + y = 5 ∧ 2 * x - y = 1 ∧ x = 2 ∧ y = 3 := by
  sorry

end system_solution_l2678_267862


namespace complex_number_real_imag_equal_l2678_267812

theorem complex_number_real_imag_equal (a : ℝ) : 
  let z : ℂ := a + (Complex.I - 1) / (1 + Complex.I)
  (z.re = z.im) → a = 1 := by
  sorry

end complex_number_real_imag_equal_l2678_267812


namespace special_sequence_representation_l2678_267868

/-- A sequence of natural numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, a n < 2 * n)

/-- The main theorem statement -/
theorem special_sequence_representation (a : ℕ → ℕ) (h : SpecialSequence a) :
  ∀ m : ℕ, (∃ n, a n = m) ∨ (∃ k l, a k - a l = m) :=
sorry

end special_sequence_representation_l2678_267868


namespace speed_increase_time_reduction_l2678_267860

theorem speed_increase_time_reduction 
  (initial_speed : ℝ) 
  (speed_increase : ℝ) 
  (distance : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : speed_increase = 10)
  (h3 : distance > 0) :
  let final_speed := initial_speed + speed_increase
  let initial_time := distance / initial_speed
  let final_time := distance / final_speed
  final_time / initial_time = 3/4 :=
by sorry

end speed_increase_time_reduction_l2678_267860


namespace vector_inequality_l2678_267842

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors
variable (A B C D : V)

-- Define the theorem
theorem vector_inequality (h : C - B = -(B - C)) :
  C - B + (A - D) - (B - C) ≠ A - D :=
sorry

end vector_inequality_l2678_267842


namespace used_computer_lifespan_l2678_267809

/-- Proves the lifespan of used computers given certain conditions -/
theorem used_computer_lifespan 
  (new_computer_cost : ℕ)
  (new_computer_lifespan : ℕ)
  (used_computer_cost : ℕ)
  (num_used_computers : ℕ)
  (savings : ℕ)
  (h1 : new_computer_cost = 600)
  (h2 : new_computer_lifespan = 6)
  (h3 : used_computer_cost = 200)
  (h4 : num_used_computers = 2)
  (h5 : savings = 200)
  (h6 : new_computer_cost - savings = num_used_computers * used_computer_cost) :
  ∃ (used_computer_lifespan : ℕ), 
    used_computer_lifespan * num_used_computers = new_computer_lifespan ∧ 
    used_computer_lifespan = 3 := by
  sorry


end used_computer_lifespan_l2678_267809


namespace larger_number_problem_l2678_267879

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 1375)
  (h2 : L = 6 * S + 15) : 
  L = 1647 := by
sorry

end larger_number_problem_l2678_267879


namespace sum_of_squares_l2678_267813

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end sum_of_squares_l2678_267813


namespace inequality_proof_l2678_267826

theorem inequality_proof (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

end inequality_proof_l2678_267826


namespace candy_distribution_proof_l2678_267877

def distribute_candies (n : ℕ) (k : ℕ) (min_counts : List ℕ) : ℕ :=
  sorry

theorem candy_distribution_proof :
  distribute_candies 10 4 [1, 1, 1, 0] = 3176 := by
  sorry

end candy_distribution_proof_l2678_267877


namespace polynomial_remainder_theorem_l2678_267875

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g (x : ℚ) := c * x^3 - 8 * x^2 + d * x - 7
  (g 2 = -7) → (g (-3) = -80) → (c = -47/15 ∧ d = 428/15) :=
by
  sorry

end polynomial_remainder_theorem_l2678_267875


namespace max_y_coordinate_sin_3theta_l2678_267863

/-- The maximum y-coordinate of a point on the graph of r = sin 3θ is 9/8 -/
theorem max_y_coordinate_sin_3theta (θ : Real) :
  let r := Real.sin (3 * θ)
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  ∀ y', y' = r * Real.sin θ → y' ≤ 9/8 :=
by sorry

end max_y_coordinate_sin_3theta_l2678_267863


namespace original_sweets_per_child_l2678_267849

/-- Proves that the original number of sweets per child is 15 --/
theorem original_sweets_per_child (total_children : ℕ) (absent_children : ℕ) (extra_sweets : ℕ) : 
  total_children = 112 → 
  absent_children = 32 → 
  extra_sweets = 6 → 
  ∃ (total_sweets : ℕ), 
    total_sweets = total_children * 15 ∧ 
    total_sweets = (total_children - absent_children) * (15 + extra_sweets) := by
  sorry


end original_sweets_per_child_l2678_267849


namespace lcm_gcf_problem_l2678_267883

theorem lcm_gcf_problem (n : ℕ+) :
  Nat.lcm n 16 = 52 →
  Nat.gcd n 16 = 8 →
  n = 26 := by
  sorry

end lcm_gcf_problem_l2678_267883


namespace problem_solution_l2678_267866

theorem problem_solution : 
  ((-3 : ℝ)^0 + (1/3)^2 + (-2)^3 = -62/9) ∧ 
  (∀ x : ℝ, (x + 1)^2 - (1 - 2*x)*(1 + 2*x) = 5*x^2 + 2*x) := by
  sorry

end problem_solution_l2678_267866


namespace right_triangle_side_length_l2678_267820

theorem right_triangle_side_length 
  (X Y Z : ℝ) 
  (hypotenuse : ℝ) 
  (right_angle : X = 90) 
  (hyp_length : Y - Z = hypotenuse) 
  (hyp_value : hypotenuse = 13) 
  (tan_cos_relation : Real.tan Z = 3 * Real.cos Y) : 
  X - Y = (2 * Real.sqrt 338) / 3 := by
  sorry

end right_triangle_side_length_l2678_267820


namespace sequence_properties_l2678_267821

def S (n : ℕ) : ℤ := n^2 - 9*n

def a (n : ℕ) : ℤ := 2*n - 10

theorem sequence_properties :
  (∀ n, S (n+1) - S n = a (n+1)) ∧
  (∃! k : ℕ, k > 0 ∧ 5 < a k ∧ a k < 8) ∧
  (∀ k : ℕ, k > 0 ∧ 5 < a k ∧ a k < 8 → k = 8) :=
sorry

end sequence_properties_l2678_267821


namespace class_mean_score_l2678_267839

theorem class_mean_score (total_students : ℕ) (first_day_students : ℕ) (second_day_students : ℕ)
  (first_day_mean : ℚ) (second_day_mean : ℚ) :
  total_students = first_day_students + second_day_students →
  first_day_students = 54 →
  second_day_students = 6 →
  first_day_mean = 76 / 100 →
  second_day_mean = 82 / 100 →
  let new_class_mean := (first_day_students * first_day_mean + second_day_students * second_day_mean) / total_students
  new_class_mean = 766 / 1000 :=
by sorry

end class_mean_score_l2678_267839


namespace no_polynomial_satisfies_conditions_exists_polynomial_satisfies_modified_conditions_l2678_267819

-- Part 1
theorem no_polynomial_satisfies_conditions :
  ¬(∃ P : ℝ → ℝ, (∀ x : ℝ, Differentiable ℝ P ∧ Differentiable ℝ (deriv P)) ∧
    (∀ x : ℝ, (deriv P) x > (deriv (deriv P)) x ∧ P x > (deriv (deriv P)) x)) :=
sorry

-- Part 2
theorem exists_polynomial_satisfies_modified_conditions :
  ∃ P : ℝ → ℝ, (∀ x : ℝ, Differentiable ℝ P ∧ Differentiable ℝ (deriv P)) ∧
    (∀ x : ℝ, P x > (deriv P) x ∧ P x > (deriv (deriv P)) x) :=
sorry

end no_polynomial_satisfies_conditions_exists_polynomial_satisfies_modified_conditions_l2678_267819


namespace more_24_than_32_placements_l2678_267836

/-- Represents a chessboard configuration --/
structure Chessboard :=
  (size : Nat)
  (dominoes : Nat)

/-- Represents the number of ways to place dominoes on a chessboard --/
def PlacementCount (board : Chessboard) : Nat := sorry

/-- The 8x8 chessboard with 32 dominoes --/
def board32 : Chessboard :=
  { size := 8, dominoes := 32 }

/-- The 8x8 chessboard with 24 dominoes --/
def board24 : Chessboard :=
  { size := 8, dominoes := 24 }

/-- Theorem stating that there are more ways to place 24 dominoes than 32 dominoes --/
theorem more_24_than_32_placements : PlacementCount board24 > PlacementCount board32 := by
  sorry

end more_24_than_32_placements_l2678_267836


namespace tan_product_squared_l2678_267876

theorem tan_product_squared (a b : ℝ) :
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 6 / 13 := by
  sorry

end tan_product_squared_l2678_267876


namespace problem_1_problem_2_l2678_267878

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f (1 + x) + 2 * f (1 - x) = 6 - 1 / x) :
  f (Real.sqrt 2) = 3 + Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (g : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ x > 0, g x = (m^2 - 2*m - 2) * x^(m^2 + 3*m + 2))
  (h2 : StrictMono g)
  (h3 : ∀ x, g (2*x - 1) ≥ 1) :
  ∀ x, x ≤ 0 ∨ x ≥ 1 := by
  sorry

end problem_1_problem_2_l2678_267878


namespace unique_tangent_circle_existence_l2678_267889

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given elements
variable (M : Point) -- Given point
variable (O : Point) -- Center of the given circle
variable (r : ℝ) -- Radius of the given circle
variable (N : Point) -- Point on the given circle

-- Define the condition that N is on the given circle
def is_on_circle (P : Point) (C : Circle) : Prop :=
  (P.x - C.center.x)^2 + (P.y - C.center.y)^2 = C.radius^2

-- Define tangency between two circles
def are_tangent (C1 C2 : Circle) : Prop :=
  (C1.center.x - C2.center.x)^2 + (C1.center.y - C2.center.y)^2 = (C1.radius + C2.radius)^2

-- State the theorem
theorem unique_tangent_circle_existence 
  (h_N_on_circle : is_on_circle N { center := O, radius := r }) :
  ∃! C : Circle, (is_on_circle M C) ∧ 
                 (are_tangent C { center := O, radius := r }) ∧ 
                 (is_on_circle N C) := by
  sorry

end unique_tangent_circle_existence_l2678_267889


namespace project_work_time_l2678_267870

/-- Calculates the time spent working on a project given the total days and nap information -/
def timeSpentWorking (totalDays : ℕ) (numberOfNaps : ℕ) (hoursPerNap : ℕ) : ℕ :=
  totalDays * 24 - numberOfNaps * hoursPerNap

/-- Theorem: Given 4 days and 6 seven-hour naps, the time spent working is 54 hours -/
theorem project_work_time :
  timeSpentWorking 4 6 7 = 54 := by
  sorry

end project_work_time_l2678_267870


namespace solution_to_equation_l2678_267890

theorem solution_to_equation : ∃ (x y : ℤ), x + 3 * y = 7 ∧ x = -2 ∧ y = 3 := by
  sorry

end solution_to_equation_l2678_267890


namespace balloon_arrangements_count_l2678_267805

/-- The number of distinct arrangements of the letters in "BALLOON" -/
def balloon_arrangements : ℕ := 1260

/-- The total number of letters in "BALLOON" -/
def total_letters : ℕ := 7

/-- The number of times 'L' appears in "BALLOON" -/
def l_count : ℕ := 2

/-- The number of times 'O' appears in "BALLOON" -/
def o_count : ℕ := 2

/-- Theorem stating that the number of distinct arrangements of the letters in "BALLOON" is 1260 -/
theorem balloon_arrangements_count :
  balloon_arrangements = (Nat.factorial total_letters) / (Nat.factorial l_count * Nat.factorial o_count) :=
sorry

end balloon_arrangements_count_l2678_267805


namespace floss_leftover_result_l2678_267865

/-- Calculates the amount of floss left over after distributing to students --/
def floss_leftover (class1_size class2_size class3_size : ℕ) 
                   (floss_per_student1 floss_per_student2 floss_per_student3 : ℚ) 
                   (yards_per_packet : ℚ) : ℚ :=
  let total_floss_needed := class1_size * floss_per_student1 + 
                            class2_size * floss_per_student2 + 
                            class3_size * floss_per_student3
  let packets_needed := (total_floss_needed / yards_per_packet).ceil
  let total_floss_bought := packets_needed * yards_per_packet
  total_floss_bought - total_floss_needed

/-- Theorem stating the amount of floss left over --/
theorem floss_leftover_result : 
  floss_leftover 20 25 30 (3/2) (7/4) 2 35 = 25/4 := by
  sorry

end floss_leftover_result_l2678_267865


namespace simplify_fraction_and_multiply_l2678_267861

theorem simplify_fraction_and_multiply :
  (144 : ℚ) / 1296 * 36 = 4 := by sorry

end simplify_fraction_and_multiply_l2678_267861


namespace book_pages_introduction_l2678_267856

theorem book_pages_introduction (total_pages : ℕ) (text_pages : ℕ) : 
  total_pages = 98 →
  text_pages = 19 →
  (total_pages - total_pages / 2 - text_pages * 2 = 11) :=
by
  sorry

end book_pages_introduction_l2678_267856


namespace intersection_point_of_lines_l2678_267845

theorem intersection_point_of_lines (x y : ℝ) :
  (x - 2*y + 7 = 0) ∧ (2*x + y - 1 = 0) ↔ (x = -1 ∧ y = 3) :=
by sorry

end intersection_point_of_lines_l2678_267845


namespace g_zero_at_seven_fifths_l2678_267881

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 7

-- Theorem statement
theorem g_zero_at_seven_fifths : g (7 / 5) = 0 := by
  sorry

end g_zero_at_seven_fifths_l2678_267881


namespace cos_45_minus_cos_90_l2678_267894

theorem cos_45_minus_cos_90 : Real.cos (π/4) - Real.cos (π/2) = Real.sqrt 2 / 2 := by
  sorry

end cos_45_minus_cos_90_l2678_267894


namespace point_translation_l2678_267818

def translate_point (x y dx dy : Int) : (Int × Int) := (x + dx, y + dy)

theorem point_translation :
  let P : (Int × Int) := (-5, 1)
  let P1 := translate_point P.1 P.2 2 0
  let P2 := translate_point P1.1 P1.2 0 (-4)
  P2 = (-3, -3) := by
  sorry

end point_translation_l2678_267818


namespace book_arrangement_theorem_l2678_267832

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def num_history_books : ℕ := 4
def num_science_books : ℕ := 6

theorem book_arrangement_theorem :
  factorial 2 * factorial num_history_books * factorial num_science_books = 34560 := by
  sorry

end book_arrangement_theorem_l2678_267832


namespace prime_divisibility_l2678_267899

theorem prime_divisibility (p a b : ℤ) (hp : Prime p) (hp_not_3 : p ≠ 3)
  (h_sum : p ∣ (a + b)) (h_cube_sum : p^2 ∣ (a^3 + b^3)) :
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) := by
  sorry

end prime_divisibility_l2678_267899
