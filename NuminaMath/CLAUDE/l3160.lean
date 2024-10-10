import Mathlib

namespace particle_position_after_1989_minutes_l3160_316058

/-- Represents the position of a particle in 2D space -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Calculates the time taken to enclose n squares -/
def timeForSquares (n : ℕ) : ℕ :=
  (n + 1)^2 - 1

/-- Calculates the position of the particle after a given number of minutes -/
def particlePosition (minutes : ℕ) : Position :=
  sorry

/-- The theorem stating the position of the particle after 1989 minutes -/
theorem particle_position_after_1989_minutes :
  particlePosition 1989 = Position.mk 44 35 := by sorry

end particle_position_after_1989_minutes_l3160_316058


namespace zero_shaded_area_l3160_316072

/-- Represents a square tile with a pattern of triangles -/
structure Tile where
  sideLength : ℝ
  triangleArea : ℝ

/-- Represents a rectangular floor tiled with square tiles -/
structure Floor where
  length : ℝ
  width : ℝ
  tile : Tile

/-- Calculates the total shaded area of the floor -/
def totalShadedArea (floor : Floor) : ℝ :=
  let totalTiles := floor.length * floor.width
  let tileArea := floor.tile.sideLength ^ 2
  let shadedAreaPerTile := tileArea - 4 * floor.tile.triangleArea
  totalTiles * shadedAreaPerTile

/-- Theorem stating that the total shaded area of the specific floor is 0 -/
theorem zero_shaded_area :
  let tile : Tile := {
    sideLength := 1,
    triangleArea := 1/4
  }
  let floor : Floor := {
    length := 12,
    width := 9,
    tile := tile
  }
  totalShadedArea floor = 0 := by sorry

end zero_shaded_area_l3160_316072


namespace hyperbola_asymptote_l3160_316013

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ hyperbola x' y' ∧ asymptote x' y') :=
sorry

end hyperbola_asymptote_l3160_316013


namespace compare_a_and_b_l3160_316027

theorem compare_a_and_b (a b : ℝ) (h : 5 * (a - 1) = b + a^2) : a > b := by
  sorry

end compare_a_and_b_l3160_316027


namespace closest_vector_to_origin_l3160_316035

/-- The vector v is closest to the origin when t = 1/13 -/
theorem closest_vector_to_origin (t : ℝ) : 
  let v : ℝ × ℝ × ℝ := (1 + 3*t, 2 - 4*t, 3 + t)
  let a : ℝ × ℝ × ℝ := (0, 0, 0)
  let direction : ℝ × ℝ × ℝ := (3, -4, 1)
  (∀ s : ℝ, ‖v - a‖ ≤ ‖(1 + 3*s, 2 - 4*s, 3 + s) - a‖) ↔ t = 1/13 :=
by sorry


end closest_vector_to_origin_l3160_316035


namespace rope_length_problem_l3160_316021

theorem rope_length_problem (total_ropes : ℕ) (avg_length_all : ℝ) (avg_length_third : ℝ) :
  total_ropes = 6 →
  avg_length_all = 80 →
  avg_length_third = 70 →
  let third_ropes := total_ropes / 3
  let remaining_ropes := total_ropes - third_ropes
  let total_length := total_ropes * avg_length_all
  let third_length := third_ropes * avg_length_third
  let remaining_length := total_length - third_length
  remaining_length / remaining_ropes = 85 := by
sorry

end rope_length_problem_l3160_316021


namespace bird_percentage_problem_l3160_316041

theorem bird_percentage_problem (total : ℝ) (pigeons sparrows crows parakeets : ℝ) :
  pigeons = 0.4 * total →
  sparrows = 0.2 * total →
  crows = 0.15 * total →
  parakeets = total - (pigeons + sparrows + crows) →
  crows / (total - sparrows) = 0.1875 := by
sorry

end bird_percentage_problem_l3160_316041


namespace median_interval_is_65_to_69_l3160_316006

/-- Represents a score interval with its lower and upper bounds -/
structure ScoreInterval where
  lower : ℕ
  upper : ℕ

/-- Represents the distribution of scores -/
structure ScoreDistribution where
  intervals : List ScoreInterval
  counts : List ℕ

/-- Finds the interval containing the median score -/
def findMedianInterval (dist : ScoreDistribution) : Option ScoreInterval :=
  sorry

/-- The given score distribution -/
def testScoreDistribution : ScoreDistribution :=
  { intervals := [
      { lower := 50, upper := 54 },
      { lower := 55, upper := 59 },
      { lower := 60, upper := 64 },
      { lower := 65, upper := 69 },
      { lower := 70, upper := 74 }
    ],
    counts := [10, 15, 25, 30, 20]
  }

/-- Theorem: The median score interval for the given distribution is 65-69 -/
theorem median_interval_is_65_to_69 :
  findMedianInterval testScoreDistribution = some { lower := 65, upper := 69 } :=
  sorry

end median_interval_is_65_to_69_l3160_316006


namespace evaluate_expression_l3160_316096

theorem evaluate_expression : 3002^3 - 3001 * 3002^2 - 3001^2 * 3002 + 3001^3 + 1 = 6004 := by
  sorry

end evaluate_expression_l3160_316096


namespace sum_of_reciprocal_equation_l3160_316020

theorem sum_of_reciprocal_equation (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (eq1 : 1 / x + 1 / y = 4)
  (eq2 : 1 / x - 1 / y = -8) :
  x + y = -1/3 := by
  sorry

end sum_of_reciprocal_equation_l3160_316020


namespace committee_formation_l3160_316094

theorem committee_formation (total : ℕ) (mathematicians : ℕ) (economists : ℕ) (committee_size : ℕ) :
  total = mathematicians + economists →
  mathematicians = 3 →
  economists = 10 →
  committee_size = 7 →
  (Nat.choose total committee_size) - (Nat.choose economists committee_size) = 1596 :=
by sorry

end committee_formation_l3160_316094


namespace sin_equals_cos_810_deg_l3160_316034

theorem sin_equals_cos_810_deg (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * Real.pi / 180) = Real.cos (810 * Real.pi / 180) ↔ n = -180 ∨ n = 0 ∨ n = 180) :=
by sorry

end sin_equals_cos_810_deg_l3160_316034


namespace max_value_and_x_l3160_316009

theorem max_value_and_x (x : ℝ) (y : ℝ) (h : x < 0) (h1 : y = 3*x + 4/x) :
  (∀ z, z < 0 → 3*z + 4/z ≤ y) → y = -4*Real.sqrt 3 ∧ x = -2*Real.sqrt 3/3 :=
sorry

end max_value_and_x_l3160_316009


namespace fence_decoration_combinations_l3160_316010

def num_colors : ℕ := 6
def num_techniques : ℕ := 5

theorem fence_decoration_combinations :
  num_colors * num_techniques = 30 := by
  sorry

end fence_decoration_combinations_l3160_316010


namespace union_with_complement_l3160_316076

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem union_with_complement : A ∪ (U \ B) = {1, 2, 4} := by
  sorry

end union_with_complement_l3160_316076


namespace consecutive_odd_integers_sum_l3160_316067

theorem consecutive_odd_integers_sum (x : ℤ) : 
  x > 0 ∧ 
  Odd x ∧ 
  Odd (x + 2) ∧ 
  x * (x + 2) = 945 → 
  x + (x + 2) = 60 := by
sorry

end consecutive_odd_integers_sum_l3160_316067


namespace family_probability_l3160_316012

/-- The probability of having a boy or a girl -/
def child_probability : ℚ := 1 / 2

/-- The number of children in the family -/
def family_size : ℕ := 4

/-- The probability of having at least one boy and one girl in a family of four children -/
def prob_at_least_one_boy_and_girl : ℚ := 7 / 8

theorem family_probability : 
  1 - (child_probability ^ family_size + child_probability ^ family_size) = prob_at_least_one_boy_and_girl :=
sorry

end family_probability_l3160_316012


namespace ab_range_l3160_316059

def f (x : ℝ) : ℝ := |2 - x^2|

theorem ab_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  ∃ (l u : ℝ), l = 0 ∧ u = 2 ∧ ∀ x, a * b = x → l < x ∧ x < u :=
sorry

end ab_range_l3160_316059


namespace min_pizzas_break_even_l3160_316003

/-- The minimum number of whole pizzas John must deliver to break even -/
def min_pizzas : ℕ := 1000

/-- The cost of the car -/
def car_cost : ℕ := 8000

/-- The earning per pizza -/
def earning_per_pizza : ℕ := 12

/-- The gas cost per pizza -/
def gas_cost_per_pizza : ℕ := 4

/-- Theorem stating that min_pizzas is the minimum number of whole pizzas
    John must deliver to at least break even on his car purchase -/
theorem min_pizzas_break_even :
  min_pizzas = (car_cost + gas_cost_per_pizza - 1) / (earning_per_pizza - gas_cost_per_pizza) :=
by sorry

end min_pizzas_break_even_l3160_316003


namespace largest_c_value_l3160_316093

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 3*x + 1

-- State the theorem
theorem largest_c_value (d : ℝ) (hd : d > 0) :
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (x : ℝ), |x - 1| ≤ d → |g x - 1| ≤ c) ∧
    (∀ (c' : ℝ), c' > c → ∃ (x : ℝ), |x - 1| ≤ d ∧ |g x - 1| > c')) ∧
  (∀ (c : ℝ), 
    (c > 0 ∧ 
     (∀ (x : ℝ), |x - 1| ≤ d → |g x - 1| ≤ c) ∧
     (∀ (c' : ℝ), c' > c → ∃ (x : ℝ), |x - 1| ≤ d ∧ |g x - 1| > c'))
    → c = 4) :=
by sorry

end largest_c_value_l3160_316093


namespace prob_at_least_7_heads_theorem_l3160_316033

/-- A fair coin is flipped 10 times. -/
def total_flips : ℕ := 10

/-- The number of heads required for the event. -/
def min_heads : ℕ := 7

/-- The number of fixed heads at the end. -/
def fixed_heads : ℕ := 2

/-- The probability of getting heads on a single flip of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The probability of getting at least 7 heads in 10 flips, given that the last two are heads. -/
def prob_at_least_7_heads_given_last_2_heads : ℚ := 93/256

/-- 
Theorem: The probability of getting at least 7 heads in 10 flips of a fair coin, 
given that the last two flips are heads, is equal to 93/256.
-/
theorem prob_at_least_7_heads_theorem : 
  prob_at_least_7_heads_given_last_2_heads = 93/256 :=
sorry

end prob_at_least_7_heads_theorem_l3160_316033


namespace greatest_m_value_l3160_316050

def reverse_number (n : ℕ) : ℕ := sorry

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_m_value (m : ℕ) 
  (h1 : is_four_digit m)
  (h2 : is_four_digit (reverse_number m))
  (h3 : m % 63 = 0)
  (h4 : (reverse_number m) % 63 = 0)
  (h5 : m % 11 = 0) :
  m ≤ 9811 ∧ ∃ (m : ℕ), m = 9811 ∧ 
    is_four_digit m ∧
    is_four_digit (reverse_number m) ∧
    m % 63 = 0 ∧
    (reverse_number m) % 63 = 0 ∧
    m % 11 = 0 :=
sorry

end greatest_m_value_l3160_316050


namespace ellipse_and_line_problem_l3160_316098

structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  e : ℝ

def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + 1}

theorem ellipse_and_line_problem (C : Ellipse) (L : ℝ → Set (ℝ × ℝ)) :
  C.c = 4 * Real.sqrt 3 →
  C.e = Real.sqrt 3 / 2 →
  (∀ x y, (x, y) ∈ {p : ℝ × ℝ | x^2 / C.a^2 + y^2 / C.b^2 = 1} ↔ (x, y) ∈ {p : ℝ × ℝ | x^2 / 16 + y^2 / 4 = 1}) →
  (∃ A B : ℝ × ℝ, A ∈ L k ∧ B ∈ L k ∧ A ∈ {p : ℝ × ℝ | x^2 / 16 + y^2 / 4 = 1} ∧ B ∈ {p : ℝ × ℝ | x^2 / 16 + y^2 / 4 = 1}) →
  (∀ A B : ℝ × ℝ, A ∈ L k ∧ B ∈ L k → A.1 = -2 * B.1) →
  (k = Real.sqrt 15 / 10 ∨ k = -Real.sqrt 15 / 10) :=
by sorry

end ellipse_and_line_problem_l3160_316098


namespace election_percentage_l3160_316023

theorem election_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) 
  (h1 : winner_votes = 744)
  (h2 : margin = 288)
  (h3 : total_votes = winner_votes + (winner_votes - margin)) :
  (winner_votes : ℚ) / total_votes * 100 = 62 := by
  sorry

end election_percentage_l3160_316023


namespace share_calculation_l3160_316087

theorem share_calculation (total : ℝ) (a b c : ℝ) : 
  total = 500 →
  a = (2/3) * (b + c) →
  b = (2/3) * (a + c) →
  a + b + c = total →
  a = 200 := by
sorry

end share_calculation_l3160_316087


namespace harmonic_mean_closest_to_ten_l3160_316085

theorem harmonic_mean_closest_to_ten :
  let a := 5
  let b := 2023
  let harmonic_mean := 2 * a * b / (a + b)
  ∀ n : ℤ, n ≠ 10 → |harmonic_mean - 10| < |harmonic_mean - n| :=
by sorry

end harmonic_mean_closest_to_ten_l3160_316085


namespace specific_hexagon_area_l3160_316071

/-- Regular hexagon with vertices A and C -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a regular hexagon -/
def hexagon_area (h : RegularHexagon) : ℝ := sorry

/-- Theorem: Area of the specific regular hexagon -/
theorem specific_hexagon_area :
  let h : RegularHexagon := { A := (0, 0), C := (8, 2) }
  hexagon_area h = 34 * Real.sqrt 3 := by sorry

end specific_hexagon_area_l3160_316071


namespace werewolf_identity_l3160_316055

/-- Represents a forest dweller -/
inductive Dweller
| A
| B
| C

/-- Represents the status of a dweller -/
structure Status where
  is_werewolf : Bool
  is_knight : Bool

/-- The statement made by B -/
def b_statement (status : Dweller → Status) : Prop :=
  (status Dweller.C).is_werewolf

theorem werewolf_identity (status : Dweller → Status) :
  (∃! d : Dweller, (status d).is_werewolf ∧ (status d).is_knight) →
  (∀ d : Dweller, d ≠ Dweller.A → d ≠ Dweller.B → ¬(status d).is_knight) →
  b_statement status →
  (status Dweller.A).is_werewolf := by
  sorry

end werewolf_identity_l3160_316055


namespace value_of_a_l3160_316082

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = 2 * c - 1)
  (eq2 : b + c = 7)
  (eq3 : c = 4) : 
  a = 4 := by
  sorry

end value_of_a_l3160_316082


namespace common_root_of_quadratic_equations_l3160_316097

theorem common_root_of_quadratic_equations (x : ℚ) :
  (6 * x^2 + 5 * x - 1 = 0) ∧ (18 * x^2 + 41 * x - 7 = 0) → x = 1/3 := by
  sorry

end common_root_of_quadratic_equations_l3160_316097


namespace product_abcd_l3160_316000

theorem product_abcd (a b c d : ℚ) : 
  3*a + 2*b + 4*c + 6*d = 48 →
  4*(d+c) = b →
  4*b + 2*c = a →
  2*c - 2 = d →
  a * b * c * d = -58735360 / 81450625 :=
by sorry

end product_abcd_l3160_316000


namespace aquarium_visit_cost_difference_l3160_316028

/-- Represents the cost structure and family composition for an aquarium visit -/
structure AquariumVisit where
  family_pass_cost : ℚ
  adult_ticket_cost : ℚ
  child_ticket_cost : ℚ
  num_adults : ℕ
  num_children : ℕ

/-- Calculates the cost of separate tickets with the special offer applied -/
def separate_tickets_cost (visit : AquariumVisit) : ℚ :=
  let discounted_adults := visit.num_children / 3
  let full_price_adults := visit.num_adults - discounted_adults
  let discounted_adult_cost := visit.adult_ticket_cost * (1/2)
  discounted_adults * discounted_adult_cost +
  full_price_adults * visit.adult_ticket_cost +
  visit.num_children * visit.child_ticket_cost

/-- Theorem stating the difference between separate tickets and family pass -/
theorem aquarium_visit_cost_difference (visit : AquariumVisit) 
  (h1 : visit.family_pass_cost = 150)
  (h2 : visit.adult_ticket_cost = 35)
  (h3 : visit.child_ticket_cost = 20)
  (h4 : visit.num_adults = 2)
  (h5 : visit.num_children = 5) :
  separate_tickets_cost visit - visit.family_pass_cost = 5/2 := by
  sorry

end aquarium_visit_cost_difference_l3160_316028


namespace fraction_equality_l3160_316079

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 4 / 3) 
  (h2 : r / t = 9 / 14) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end fraction_equality_l3160_316079


namespace equilateral_is_peculiar_specific_right_triangle_is_peculiar_right_angled_peculiar_triangle_ratio_l3160_316024

-- Definition of a peculiar triangle
def is_peculiar_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2*c^2 ∨ a^2 + c^2 = 2*b^2 ∨ b^2 + c^2 = 2*a^2

-- Definition of an equilateral triangle
def is_equilateral_triangle (a b c : ℝ) : Prop :=
  a = b ∧ b = c

-- Definition of a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Theorem 1: An equilateral triangle is a peculiar triangle
theorem equilateral_is_peculiar (a b c : ℝ) :
  is_equilateral_triangle a b c → is_peculiar_triangle a b c :=
sorry

-- Theorem 2: A right triangle with sides 5√2, 10, and 5√6 is a peculiar triangle
theorem specific_right_triangle_is_peculiar :
  let a : ℝ := 5 * Real.sqrt 2
  let b : ℝ := 5 * Real.sqrt 6
  let c : ℝ := 10
  is_right_triangle a b c ∧ is_peculiar_triangle a b c :=
sorry

-- Theorem 3: In a right-angled peculiar triangle, the ratio of sides is 1:√2:√3
theorem right_angled_peculiar_triangle_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > a) :
  is_right_triangle a b c ∧ is_peculiar_triangle a b c →
  ∃ (k : ℝ), a = k ∧ b = k * Real.sqrt 2 ∧ c = k * Real.sqrt 3 :=
sorry

end equilateral_is_peculiar_specific_right_triangle_is_peculiar_right_angled_peculiar_triangle_ratio_l3160_316024


namespace chocolate_chip_calculation_l3160_316019

/-- The number of cups of chocolate chips needed for one recipe -/
def cups_per_recipe : ℕ := 2

/-- The number of recipes to be made -/
def number_of_recipes : ℕ := 23

/-- The total number of cups of chocolate chips needed -/
def total_cups : ℕ := cups_per_recipe * number_of_recipes

theorem chocolate_chip_calculation : total_cups = 46 := by
  sorry

end chocolate_chip_calculation_l3160_316019


namespace profit_percentage_calculation_l3160_316089

theorem profit_percentage_calculation (C S : ℝ) (h1 : C > 0) (h2 : S > 0) :
  20 * C = 16 * S →
  (S - C) / C * 100 = 25 := by
  sorry

end profit_percentage_calculation_l3160_316089


namespace compare_negative_roots_l3160_316037

theorem compare_negative_roots : -6 * Real.sqrt 5 < -5 * Real.sqrt 6 := by
  sorry

end compare_negative_roots_l3160_316037


namespace parabola_point_relation_l3160_316064

theorem parabola_point_relation (a : ℝ) (y₁ y₂ : ℝ) 
  (h1 : a > 0) 
  (h2 : y₁ = a * (-2)^2) 
  (h3 : y₂ = a * 1^2) : 
  y₁ > y₂ := by
  sorry

end parabola_point_relation_l3160_316064


namespace max_area_30_60_90_triangle_in_rectangle_l3160_316043

/-- The maximum area of a 30-60-90 triangle inscribed in a 12x15 rectangle --/
theorem max_area_30_60_90_triangle_in_rectangle : 
  ∃ (A : ℝ), 
    (∀ t : ℝ, t ≥ 0 ∧ t ≤ 12 → t^2 * Real.sqrt 3 / 2 ≤ A) ∧ 
    A = 72 * Real.sqrt 3 := by
  sorry

end max_area_30_60_90_triangle_in_rectangle_l3160_316043


namespace unique_w_exists_l3160_316061

theorem unique_w_exists : ∃! w : ℝ, w > 0 ∧ 
  (Real.sqrt 1.5) / (Real.sqrt 0.81) + (Real.sqrt 1.44) / (Real.sqrt w) = 3.0751133491652576 := by
  sorry

end unique_w_exists_l3160_316061


namespace committee_formations_count_l3160_316053

/-- The number of ways to form a committee of 5 members from a club of 15 people,
    where the committee must include exactly 2 designated roles and 3 additional members. -/
def committeeFormations (clubSize : ℕ) (committeeSize : ℕ) (designatedRoles : ℕ) (additionalMembers : ℕ) : ℕ :=
  (clubSize * (clubSize - 1)) * Nat.choose (clubSize - designatedRoles) additionalMembers

/-- Theorem stating that the number of committee formations
    for the given conditions is 60060. -/
theorem committee_formations_count :
  committeeFormations 15 5 2 3 = 60060 := by
  sorry

end committee_formations_count_l3160_316053


namespace minimum_value_inequality_l3160_316042

theorem minimum_value_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : 2 * m + n = 1) :
  1 / m + 2 / n ≥ 8 ∧ (1 / m + 2 / n = 8 ↔ n = 2 * m ∧ n = 1 / 3) := by
  sorry

end minimum_value_inequality_l3160_316042


namespace six_selected_in_interval_l3160_316004

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ
  (population_positive : population > 0)
  (sample_size_positive : sample_size > 0)
  (sample_size_le_population : sample_size ≤ population)
  (interval_valid : interval_start ≤ interval_end)
  (interval_in_range : interval_end ≤ population)

/-- Calculates the number of selected individuals within a given interval -/
def selected_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) + (s.population / s.sample_size - 1)) / (s.population / s.sample_size)

/-- Theorem stating that for the given parameters, 6 individuals are selected within the interval -/
theorem six_selected_in_interval (s : SystematicSample) 
  (h_pop : s.population = 420)
  (h_sample : s.sample_size = 21)
  (h_start : s.interval_start = 241)
  (h_end : s.interval_end = 360) :
  selected_in_interval s = 6 := by
  sorry

#eval selected_in_interval {
  population := 420,
  sample_size := 21,
  interval_start := 241,
  interval_end := 360,
  population_positive := by norm_num,
  sample_size_positive := by norm_num,
  sample_size_le_population := by norm_num,
  interval_valid := by norm_num,
  interval_in_range := by norm_num
}

end six_selected_in_interval_l3160_316004


namespace elliptical_orbit_distance_l3160_316015

/-- Given an elliptical orbit with perigee 3 AU and apogee 15 AU, 
    the distance from the sun (at one focus) to a point on the minor axis is 3√5 + 6 AU -/
theorem elliptical_orbit_distance (perigee apogee : ℝ) (h1 : perigee = 3) (h2 : apogee = 15) :
  let semi_major_axis := (apogee + perigee) / 2
  let focal_distance := semi_major_axis - perigee
  let semi_minor_axis := Real.sqrt (semi_major_axis^2 - focal_distance^2)
  semi_minor_axis + focal_distance = 3 * Real.sqrt 5 + 6 := by
  sorry

end elliptical_orbit_distance_l3160_316015


namespace sequence_sum_l3160_316090

/-- Given a sequence {a_n} where a_1 = 1 and S_n = n^2 * a_n for all positive integers n,
    prove that the sum of the first n terms (S_n) is equal to 2n / (n + 1). -/
theorem sequence_sum (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) (h1 : a 1 = 1)
    (h2 : ∀ n : ℕ+, S n = n^2 * a n) :
    ∀ n : ℕ+, S n = 2 * n / (n + 1) := by
  sorry

end sequence_sum_l3160_316090


namespace ram_krish_work_time_l3160_316011

/-- Represents the efficiency of a worker -/
structure Efficiency : Type :=
  (value : ℝ)

/-- Represents the time taken to complete a task -/
structure Time : Type :=
  (days : ℝ)

/-- Represents the amount of work in a task -/
structure Work : Type :=
  (amount : ℝ)

/-- The theorem stating the relationship between Ram and Krish's efficiency and their combined work time -/
theorem ram_krish_work_time 
  (ram_efficiency : Efficiency)
  (krish_efficiency : Efficiency)
  (ram_alone_time : Time)
  (task : Work)
  (h1 : ram_efficiency.value = (1 / 2) * krish_efficiency.value)
  (h2 : ram_alone_time.days = 30)
  (h3 : task.amount = ram_efficiency.value * ram_alone_time.days) :
  ∃ (combined_time : Time),
    combined_time.days = 10 ∧
    task.amount = (ram_efficiency.value + krish_efficiency.value) * combined_time.days :=
sorry

end ram_krish_work_time_l3160_316011


namespace arithmetic_sequence_problem_l3160_316001

/-- Given an arithmetic sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := sorry

/-- a_n represents the nth term of the arithmetic sequence -/
def a : ℕ → ℝ := sorry

theorem arithmetic_sequence_problem (h : S 9 a = 45) : a 5 = 5 := by sorry

end arithmetic_sequence_problem_l3160_316001


namespace cyclic_sum_inequality_l3160_316002

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b + b / c + c / a ≥ 3 := by
  sorry

end cyclic_sum_inequality_l3160_316002


namespace least_subtraction_for_divisibility_subtraction_for_509_divisible_by_9_least_subtraction_509_divisible_by_9_l3160_316069

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem subtraction_for_509_divisible_by_9 :
  ∃ (k : ℕ), k < 9 ∧ (509 - k) % 9 = 0 ∧ ∀ (m : ℕ), m < k → (509 - m) % 9 ≠ 0 :=
by
  sorry

#eval (509 - 5) % 9  -- Should output 0

theorem least_subtraction_509_divisible_by_9 :
  5 < 9 ∧ (509 - 5) % 9 = 0 ∧ ∀ (m : ℕ), m < 5 → (509 - m) % 9 ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_subtraction_for_509_divisible_by_9_least_subtraction_509_divisible_by_9_l3160_316069


namespace polar_to_parabola_l3160_316066

/-- The polar equation r = 1 / (1 - sin θ) represents a parabola -/
theorem polar_to_parabola :
  ∃ (x y : ℝ), (∃ (r θ : ℝ), r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  x^2 = 1 + 2*y := by
  sorry

end polar_to_parabola_l3160_316066


namespace arcsin_arccos_range_l3160_316039

theorem arcsin_arccos_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = 2 * Real.arcsin x - Real.arccos y ∧ -3 * π / 2 ≤ z ∧ z ≤ π / 2 := by
  sorry

end arcsin_arccos_range_l3160_316039


namespace square_difference_hundred_ninetynine_l3160_316046

theorem square_difference_hundred_ninetynine : 100^2 - 2*100*99 + 99^2 = 1 := by
  sorry

end square_difference_hundred_ninetynine_l3160_316046


namespace gcd_of_136_and_1275_l3160_316057

theorem gcd_of_136_and_1275 : Nat.gcd 136 1275 = 17 := by
  sorry

end gcd_of_136_and_1275_l3160_316057


namespace intersection_midpoint_theorem_l3160_316026

-- Define the curve
def curve (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 2

-- Define the line
def line (x : ℝ) : ℝ := -2 * x

-- Theorem statement
theorem intersection_midpoint_theorem (s : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    curve x₁ = line x₁ ∧ 
    curve x₂ = line x₂ ∧ 
    (curve x₁ + curve x₂) / 2 = 7 / s) →
  s = -2 :=
by sorry

end intersection_midpoint_theorem_l3160_316026


namespace motor_lifespan_probability_l3160_316047

variable (X : Real → Real)  -- Random variable representing motor lifespan

-- Define the expected value of X
def expected_value : Real := 4

-- Define the theorem
theorem motor_lifespan_probability :
  (∫ x, X x) = expected_value →  -- The expected value of X is 4
  (∫ x in {x | x < 20}, X x) ≥ 0.8 := by
  sorry

end motor_lifespan_probability_l3160_316047


namespace percentage_calculation_l3160_316040

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 66 = 235.42 := by
  sorry

end percentage_calculation_l3160_316040


namespace thirty_five_million_scientific_notation_l3160_316068

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem thirty_five_million_scientific_notation :
  scientific_notation 35000000 = (3.5, 7) :=
sorry

end thirty_five_million_scientific_notation_l3160_316068


namespace rectangle_areas_sum_l3160_316080

theorem rectangle_areas_sum : 
  let base_width : ℕ := 2
  let lengths : List ℕ := [1, 4, 9, 16, 25, 36, 49]
  let areas : List ℕ := lengths.map (λ l => base_width * l)
  areas.sum = 280 := by sorry

end rectangle_areas_sum_l3160_316080


namespace solution_set_inequality1_solution_set_inequality2_a_eq_0_solution_set_inequality2_a_pos_solution_set_inequality2_a_neg_l3160_316014

-- Define the inequalities
def inequality1 (x : ℝ) := -x^2 + 3*x + 4 ≥ 0
def inequality2 (x a : ℝ) := x^2 + 2*x + (1-a)*(1+a) ≥ 0

-- Theorem for the first inequality
theorem solution_set_inequality1 :
  {x : ℝ | inequality1 x} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Theorems for the second inequality
theorem solution_set_inequality2_a_eq_0 :
  ∀ x : ℝ, inequality2 x 0 := by sorry

theorem solution_set_inequality2_a_pos (a : ℝ) (h : a > 0) :
  {x : ℝ | inequality2 x a} = {x : ℝ | x ≥ a - 1 ∨ x ≤ -a - 1} := by sorry

theorem solution_set_inequality2_a_neg (a : ℝ) (h : a < 0) :
  {x : ℝ | inequality2 x a} = {x : ℝ | x ≥ -a - 1 ∨ x ≤ a - 1} := by sorry

end solution_set_inequality1_solution_set_inequality2_a_eq_0_solution_set_inequality2_a_pos_solution_set_inequality2_a_neg_l3160_316014


namespace nyusha_ate_28_candies_l3160_316077

-- Define the number of candies eaten by each person
variable (K E N B : ℕ)

-- Define the conditions
axiom total_candies : K + E + N + B = 86
axiom minimum_candies : K ≥ 5 ∧ E ≥ 5 ∧ N ≥ 5 ∧ B ≥ 5
axiom nyusha_ate_most : N > K ∧ N > E ∧ N > B
axiom kros_yozhik_total : K + E = 53

-- Theorem to prove
theorem nyusha_ate_28_candies : N = 28 := by
  sorry


end nyusha_ate_28_candies_l3160_316077


namespace hyperbola_eccentricity_sqrt_6_l3160_316084

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Hyperbola type -/
structure Hyperbola where
  equation : ℝ → ℝ → ℝ → Prop
  a : ℝ

/-- Theorem: Eccentricity of hyperbola given specific conditions -/
theorem hyperbola_eccentricity_sqrt_6
  (p : Parabola)
  (h : Hyperbola)
  (A B : ℝ × ℝ)
  (h_parabola : p.equation = fun x y ↦ y^2 = 4*x)
  (h_hyperbola : h.equation = fun x y a ↦ x^2/a^2 - y^2 = 1)
  (h_a_pos : h.a > 0)
  (h_intersection : p.equation A.1 A.2 ∧ h.equation A.1 A.2 h.a ∧
                    p.equation B.1 B.2 ∧ h.equation B.1 B.2 h.a)
  (h_right_angle : (A.1 - p.focus.1) * (B.1 - p.focus.1) +
                   (A.2 - p.focus.2) * (B.2 - p.focus.2) = 0) :
  ∃ (e : ℝ), e^2 = 6 ∧ e = (Real.sqrt ((h.a^2 + 1) / h.a^2)) :=
sorry

end hyperbola_eccentricity_sqrt_6_l3160_316084


namespace max_product_constraint_l3160_316052

theorem max_product_constraint (a b : ℝ) : 
  a > 0 → b > 0 → 3 * a + 8 * b = 72 → ab ≤ 54 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 8 * b₀ = 72 ∧ a₀ * b₀ = 54 := by
  sorry

end max_product_constraint_l3160_316052


namespace total_treats_eq_155_l3160_316095

/-- The number of chewing gums -/
def chewing_gums : ℕ := 60

/-- The number of chocolate bars -/
def chocolate_bars : ℕ := 55

/-- The number of candies of different flavors -/
def candies : ℕ := 40

/-- The total number of treats -/
def total_treats : ℕ := chewing_gums + chocolate_bars + candies

theorem total_treats_eq_155 : total_treats = 155 := by sorry

end total_treats_eq_155_l3160_316095


namespace angle_terminal_side_l3160_316074

theorem angle_terminal_side (α : Real) :
  let P : ℝ × ℝ := (Real.tan α, Real.cos α)
  (P.1 < 0 ∧ P.2 < 0) →  -- Point P is in the third quadrant
  (Real.cos α < 0 ∧ Real.sin α > 0) -- Terminal side of α is in the second quadrant
:= by sorry

end angle_terminal_side_l3160_316074


namespace min_shots_is_60_l3160_316044

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : Nat
  shots_taken : Nat
  nora_lead : Nat
  nora_min_score : Nat

/-- Calculates the minimum number of consecutive 10-point shots needed for Nora to guarantee victory -/
def min_shots_for_victory (comp : ArcheryCompetition) : Nat :=
  let remaining_shots := comp.total_shots - comp.shots_taken
  let max_opponent_score := remaining_shots * 10
  let n := (max_opponent_score - comp.nora_lead + comp.nora_min_score * remaining_shots - 1) / (10 - comp.nora_min_score) + 1
  n

/-- Theorem stating that for the given competition scenario, the minimum number of 10-point shots needed is 60 -/
theorem min_shots_is_60 (comp : ArcheryCompetition) 
    (h1 : comp.total_shots = 150)
    (h2 : comp.shots_taken = 75)
    (h3 : comp.nora_lead = 80)
    (h4 : comp.nora_min_score = 5) : 
  min_shots_for_victory comp = 60 := by
  sorry

end min_shots_is_60_l3160_316044


namespace unknown_number_problem_l3160_316022

theorem unknown_number_problem (x : ℝ) : 
  (50 : ℝ) / 100 * 100 = (20 : ℝ) / 100 * x + 47 → x = 15 := by
sorry

end unknown_number_problem_l3160_316022


namespace harry_worked_36_hours_l3160_316036

/-- Payment structure for Harry and James -/
structure PaymentStructure where
  base_rate : ℝ
  harry_base_hours : ℕ := 30
  harry_overtime_rate : ℝ := 2
  james_base_hours : ℕ := 40
  james_overtime_rate : ℝ := 1.5

/-- Calculate pay for a given number of hours worked -/
def calculate_pay (ps : PaymentStructure) (base_hours : ℕ) (overtime_rate : ℝ) (hours_worked : ℕ) : ℝ :=
  if hours_worked ≤ base_hours then
    ps.base_rate * hours_worked
  else
    ps.base_rate * base_hours + ps.base_rate * overtime_rate * (hours_worked - base_hours)

/-- Theorem stating that Harry worked 36 hours if paid the same as James who worked 41 hours -/
theorem harry_worked_36_hours (ps : PaymentStructure) :
  calculate_pay ps ps.james_base_hours ps.james_overtime_rate 41 =
  calculate_pay ps ps.harry_base_hours ps.harry_overtime_rate 36 :=
sorry

end harry_worked_36_hours_l3160_316036


namespace compound_interest_rate_interest_rate_calculation_l3160_316031

/-- Compound interest calculation -/
theorem compound_interest_rate (P : ℝ) (A : ℝ) (n : ℝ) (t : ℝ) (h1 : P > 0) (h2 : A > P) (h3 : n > 0) (h4 : t > 0) :
  A = P * (1 + 0.2 / n) ^ (n * t) → 
  A - P = 240 ∧ P = 1200 ∧ n = 1 ∧ t = 1 :=
by sorry

/-- Main theorem: Interest rate calculation -/
theorem interest_rate_calculation (P : ℝ) (A : ℝ) (n : ℝ) (t : ℝ) 
  (h1 : P > 0) (h2 : A > P) (h3 : n > 0) (h4 : t > 0) 
  (h5 : A - P = 240) (h6 : P = 1200) (h7 : n = 1) (h8 : t = 1) :
  ∃ r : ℝ, A = P * (1 + r) ∧ r = 0.2 :=
by sorry

end compound_interest_rate_interest_rate_calculation_l3160_316031


namespace equal_variables_l3160_316045

theorem equal_variables (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + 1/y = y + 1/x)
  (h2 : y + 1/z = z + 1/y)
  (h3 : z + 1/x = x + 1/z) :
  x = y ∨ y = z ∨ x = z := by
  sorry

end equal_variables_l3160_316045


namespace solution_difference_l3160_316051

theorem solution_difference (r s : ℝ) : 
  (∀ x, x ≠ 3 → (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) →
  r ≠ s →
  r > s →
  r - s = 3 := by
sorry

end solution_difference_l3160_316051


namespace floor_ceiling_sum_l3160_316088

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by
  sorry

end floor_ceiling_sum_l3160_316088


namespace motel_room_rate_problem_l3160_316005

theorem motel_room_rate_problem (total_rent : ℕ) (lower_rate : ℕ) (num_rooms_changed : ℕ) (rent_decrease_percent : ℚ) (higher_rate : ℕ) : 
  total_rent = 400 →
  lower_rate = 50 →
  num_rooms_changed = 10 →
  rent_decrease_percent = 1/4 →
  (total_rent : ℚ) * rent_decrease_percent = (num_rooms_changed : ℚ) * (higher_rate - lower_rate) →
  higher_rate = 60 := by
sorry

end motel_room_rate_problem_l3160_316005


namespace triangle_implies_s_range_l3160_316008

-- Define the system of inequalities
def SystemOfInequalities : Type := Unit  -- Placeholder, as we don't have specific inequalities

-- Define what it means for a region to be a triangle
def IsTriangle (region : SystemOfInequalities) : Prop := sorry

-- Define the range of s
def SRange (s : ℝ) : Prop := (0 < s ∧ s ≤ 2) ∨ s ≥ 4

-- Theorem statement
theorem triangle_implies_s_range (region : SystemOfInequalities) :
  IsTriangle region → ∀ s, SRange s :=
sorry

end triangle_implies_s_range_l3160_316008


namespace fundamental_theorem_of_calculus_l3160_316032

open Set
open Interval
open MeasureTheory
open Real

-- Define the theorem
theorem fundamental_theorem_of_calculus 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) (a b : ℝ) 
  (h1 : ContinuousOn f (Icc a b))
  (h2 : DifferentiableOn ℝ f (Ioc a b))
  (h3 : ∀ x ∈ Ioc a b, deriv f x = f' x) :
  ∫ x in a..b, f' x = f b - f a :=
sorry

end fundamental_theorem_of_calculus_l3160_316032


namespace circle_radius_with_modified_area_formula_l3160_316038

/-- Given a circle with a modified area formula, prove that its radius is 10√2 units. -/
theorem circle_radius_with_modified_area_formula 
  (area : ℝ) 
  (k : ℝ) 
  (h1 : area = 100 * Real.pi)
  (h2 : k = 0.5)
  (h3 : ∀ r, Real.pi * k * r^2 = area) :
  ∃ r, r = 10 * Real.sqrt 2 ∧ Real.pi * k * r^2 = area := by
  sorry

end circle_radius_with_modified_area_formula_l3160_316038


namespace aj_has_370_stamps_l3160_316065

/-- The number of stamps each person has -/
structure Stamps where
  aj : ℕ
  kj : ℕ
  cj : ℕ

/-- The conditions of the stamp collection problem -/
def stamp_problem (s : Stamps) : Prop :=
  s.cj = 5 + 2 * s.kj ∧
  s.kj = s.aj / 2 ∧
  s.aj + s.kj + s.cj = 930

/-- The theorem stating that AJ has 370 stamps -/
theorem aj_has_370_stamps :
  ∃ (s : Stamps), stamp_problem s ∧ s.aj = 370 :=
by
  sorry


end aj_has_370_stamps_l3160_316065


namespace infinite_series_sum_l3160_316025

/-- The sum of the infinite series Σ(k^2 / 3^k) from k=1 to infinity is equal to 4 -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k : ℝ)^2 / 3^k) = 4 := by sorry

end infinite_series_sum_l3160_316025


namespace x_squared_minus_two_is_quadratic_l3160_316062

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 2 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- Theorem: x^2 - 2 = 0 is a quadratic equation -/
theorem x_squared_minus_two_is_quadratic : is_quadratic_equation f := by
  sorry

end x_squared_minus_two_is_quadratic_l3160_316062


namespace davids_trip_spending_l3160_316007

theorem davids_trip_spending (initial_amount spent_amount remaining_amount : ℕ) : 
  initial_amount = 1800 →
  remaining_amount = spent_amount - 800 →
  initial_amount = spent_amount + remaining_amount →
  remaining_amount = 500 := by
  sorry

end davids_trip_spending_l3160_316007


namespace rhombus_min_rotation_l3160_316083

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  sides : Fin 4 → ℝ
  sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The minimum rotation angle for a rhombus to coincide with its original position -/
def min_rotation_angle (r : Rhombus) : ℝ := 180

/-- Theorem: The minimum rotation angle for a rhombus with a 60° angle to coincide with its original position is 180° -/
theorem rhombus_min_rotation (r : Rhombus) (angle : ℝ) (h : angle = 60) :
  min_rotation_angle r = 180 := by
  sorry

end rhombus_min_rotation_l3160_316083


namespace xiao_ying_score_l3160_316063

/-- Given an average score and a student's score relative to the average,
    calculate the student's actual score. -/
def calculate_score (average : ℕ) (relative_score : ℤ) : ℕ :=
  (average : ℤ) + relative_score |>.toNat

/-- The problem statement -/
theorem xiao_ying_score :
  let average_score : ℕ := 83
  let xiao_ying_relative_score : ℤ := -3
  calculate_score average_score xiao_ying_relative_score = 80 := by
  sorry

#eval calculate_score 83 (-3)

end xiao_ying_score_l3160_316063


namespace santos_salvadore_earnings_ratio_l3160_316056

/-- Proves that the ratio of Santo's earnings to Salvadore's earnings is 1:2 -/
theorem santos_salvadore_earnings_ratio :
  let salvadore_earnings : ℚ := 1956
  let total_earnings : ℚ := 2934
  let santo_earnings : ℚ := total_earnings - salvadore_earnings
  santo_earnings / salvadore_earnings = 1 / 2 := by
  sorry

end santos_salvadore_earnings_ratio_l3160_316056


namespace inequality_solution_l3160_316091

theorem inequality_solution :
  {x : ℝ | (x^2 - 9) / (x^2 - 16) > 0} = {x : ℝ | x < -4 ∨ x > 4} := by
  sorry

end inequality_solution_l3160_316091


namespace cube_volume_from_surface_area_l3160_316081

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 1734 → s^3 = 4913 := by
  sorry

end cube_volume_from_surface_area_l3160_316081


namespace squirrel_nut_difference_squirrel_nut_difference_example_l3160_316016

theorem squirrel_nut_difference : ℕ → ℕ → ℕ
  | num_squirrels, num_nuts =>
    num_squirrels - num_nuts

theorem squirrel_nut_difference_example : squirrel_nut_difference 4 2 = 2 := by
  sorry

end squirrel_nut_difference_squirrel_nut_difference_example_l3160_316016


namespace log_problem_l3160_316073

theorem log_problem (x : ℝ) (h : x = (Real.log 2 / Real.log 4) ^ ((Real.log 16 / Real.log 2) ^ 2)) :
  Real.log x / Real.log 5 = -16 / Real.log 5 * Real.log 2 := by
  sorry

end log_problem_l3160_316073


namespace orphanage_donation_percentage_l3160_316078

def total_income : ℝ := 1000000
def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def wife_percentage : ℝ := 0.3
def final_amount : ℝ := 50000

theorem orphanage_donation_percentage :
  let family_distribution := children_percentage * num_children + wife_percentage
  let remaining_before_donation := total_income * (1 - family_distribution)
  let donation_amount := remaining_before_donation - final_amount
  (donation_amount / remaining_before_donation) * 100 = 50 := by sorry

end orphanage_donation_percentage_l3160_316078


namespace equilateral_triangle_point_distance_l3160_316060

/-- Given an equilateral triangle ABC with side length a and a point P inside the triangle
    such that PA = u, PB = v, PC = w, and u^2 + v^2 = w^2, prove that w^2 + √3uv = a^2. -/
theorem equilateral_triangle_point_distance (a u v w : ℝ) :
  a > 0 →  -- Ensure positive side length
  u > 0 ∧ v > 0 ∧ w > 0 →  -- Ensure positive distances
  u^2 + v^2 = w^2 →  -- Given condition
  w^2 + Real.sqrt 3 * u * v = a^2 := by
  sorry

end equilateral_triangle_point_distance_l3160_316060


namespace sqrt_product_plus_one_l3160_316048

theorem sqrt_product_plus_one : 
  Real.sqrt ((21 : ℝ) * 20 * 19 * 18 + 1) = 379 := by sorry

end sqrt_product_plus_one_l3160_316048


namespace land_area_scientific_notation_l3160_316092

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The land area in square kilometers -/
def land_area : ℝ := 9600000

/-- The scientific notation of the land area -/
def land_area_scientific : ScientificNotation :=
  { coefficient := 9.6
  , exponent := 6
  , norm_coeff := by sorry }

theorem land_area_scientific_notation :
  land_area = land_area_scientific.coefficient * (10 : ℝ) ^ land_area_scientific.exponent :=
by sorry

end land_area_scientific_notation_l3160_316092


namespace cube_volume_from_surface_area_l3160_316099

/-- Given a cube with surface area 864 square centimeters, its volume is 1728 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (a : ℝ), 
  (6 * a^2 = 864) →  -- Surface area of cube is 864 sq cm
  (a^3 = 1728)       -- Volume of cube is 1728 cubic cm
:= by sorry

end cube_volume_from_surface_area_l3160_316099


namespace symmetry_line_equation_l3160_316017

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := y = -x^2 + 4*x - 2
def C2 (x y : ℝ) : Prop := y^2 = x

-- Define symmetry about a line
def symmetric_about_line (l : ℝ → ℝ → Prop) (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), C1 x1 y1 → C2 x2 y2 → 
    ∃ (x' y' : ℝ), l x' y' ∧ 
      x' = (x1 + x2) / 2 ∧ 
      y' = (y1 + y2) / 2

-- Theorem statement
theorem symmetry_line_equation :
  ∀ (l : ℝ → ℝ → Prop),
  symmetric_about_line l C1 C2 →
  (∀ x y, l x y ↔ x + y - 2 = 0) :=
sorry

end symmetry_line_equation_l3160_316017


namespace parallelogram_base_l3160_316029

theorem parallelogram_base (area height base : ℝ) : 
  area = 336 ∧ height = 24 ∧ area = base * height → base = 14 := by
  sorry

end parallelogram_base_l3160_316029


namespace undefined_function_roots_sum_l3160_316086

theorem undefined_function_roots_sum : 
  let f (x : ℝ) := 3 * x^2 - 9 * x + 6
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂ ∧ r₁ + r₂ = 3 := by
  sorry

end undefined_function_roots_sum_l3160_316086


namespace range_of_a_l3160_316075

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) 3, 2 * x > x^2 + a) → a < -8 := by
  sorry

end range_of_a_l3160_316075


namespace luke_game_points_per_round_l3160_316070

/-- Given a total score and number of rounds in a game where equal points are gained in each round,
    calculate the points gained per round. -/
def points_per_round (total_score : ℕ) (num_rounds : ℕ) : ℚ :=
  total_score / num_rounds

/-- Theorem stating that for Luke's game with 154 total points over 14 rounds,
    the points gained per round is 11. -/
theorem luke_game_points_per_round :
  points_per_round 154 14 = 11 := by
  sorry

end luke_game_points_per_round_l3160_316070


namespace other_roots_of_polynomial_l3160_316054

def f (a b x : ℝ) : ℝ := x^3 + 4*x^2 + a*x + b

theorem other_roots_of_polynomial (a b : ℚ) :
  (f a b (2 + Real.sqrt 3) = 0) →
  (f a b (2 - Real.sqrt 3) = 0) ∧ (f a b (-8) = 0) :=
by sorry

end other_roots_of_polynomial_l3160_316054


namespace concentric_circles_chord_count_l3160_316018

theorem concentric_circles_chord_count
  (angle_ABC : ℝ)
  (is_tangent : Bool)
  (h1 : angle_ABC = 60)
  (h2 : is_tangent = true) :
  ∃ n : ℕ, n * angle_ABC = 180 ∧ n = 3 := by
  sorry

end concentric_circles_chord_count_l3160_316018


namespace ordering_abc_l3160_316049

theorem ordering_abc : ∃ (a b c : ℝ), 
  a = Real.sqrt 1.2 ∧ 
  b = Real.exp 0.1 ∧ 
  c = 1 + Real.log 1.1 ∧ 
  b > a ∧ a > c := by
  sorry

end ordering_abc_l3160_316049


namespace percentage_calculation_l3160_316030

theorem percentage_calculation (x : ℝ) : 
  x = 0.18 * 4750 → 1.5 * x = 1282.5 := by
  sorry

end percentage_calculation_l3160_316030
