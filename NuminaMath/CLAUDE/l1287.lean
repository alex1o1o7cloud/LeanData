import Mathlib

namespace NUMINAMATH_CALUDE_log_equation_solution_l1287_128725

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 2 + Real.log x / Real.log 4 = 6 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1287_128725


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l1287_128723

theorem cube_less_than_triple : ∃! (x : ℤ), x^3 < 3*x :=
by sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l1287_128723


namespace NUMINAMATH_CALUDE_kitchen_upgrade_cost_l1287_128764

/-- The cost of a kitchen upgrade with cabinet knobs and drawer pulls -/
theorem kitchen_upgrade_cost (num_knobs : ℕ) (num_pulls : ℕ) (pull_cost : ℚ) (total_cost : ℚ) 
  (h1 : num_knobs = 18)
  (h2 : num_pulls = 8)
  (h3 : pull_cost = 4)
  (h4 : total_cost = 77) :
  (total_cost - num_pulls * pull_cost) / num_knobs = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_upgrade_cost_l1287_128764


namespace NUMINAMATH_CALUDE_mailbox_distribution_l1287_128739

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of mailboxes -/
def num_mailboxes : ℕ := 4

/-- The number of letters -/
def num_letters : ℕ := 3

theorem mailbox_distribution :
  distribute num_letters num_mailboxes = 64 := by
  sorry

end NUMINAMATH_CALUDE_mailbox_distribution_l1287_128739


namespace NUMINAMATH_CALUDE_count_cubes_with_at_most_two_shared_vertices_l1287_128719

/-- Given a cube with edge length n divided into n^3 unit cubes, 
    this function calculates the number of unit cubes that share 
    no more than 2 vertices with any other unit cube. -/
def cubes_with_at_most_two_shared_vertices (n : ℕ) : ℕ :=
  (n^2 * (n^4 - 7*n + 6)) / 2

/-- Theorem stating that the number of unit cubes sharing no more than 2 vertices 
    in a cube of edge length n is given by the formula (1/2) * n^2 * (n^4 - 7n + 6). -/
theorem count_cubes_with_at_most_two_shared_vertices (n : ℕ) :
  cubes_with_at_most_two_shared_vertices n = (n^2 * (n^4 - 7*n + 6)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_count_cubes_with_at_most_two_shared_vertices_l1287_128719


namespace NUMINAMATH_CALUDE_missy_capacity_l1287_128758

/-- The number of insurance claims each agent can handle -/
structure AgentCapacity where
  jan : ℕ
  john : ℕ
  missy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (c : AgentCapacity) : Prop :=
  c.jan = 20 ∧
  c.john = c.jan + (c.jan * 30 / 100) ∧
  c.missy = c.john + 15

/-- The theorem to prove -/
theorem missy_capacity (c : AgentCapacity) :
  problem_conditions c → c.missy = 41 := by
  sorry

end NUMINAMATH_CALUDE_missy_capacity_l1287_128758


namespace NUMINAMATH_CALUDE_reading_time_difference_l1287_128797

/-- Proves that the difference in reading time between Molly and Xanthia is 150 minutes -/
theorem reading_time_difference 
  (xanthia_speed : ℝ) 
  (molly_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 60)
  (h3 : book_pages = 300) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 150 := by
sorry

end NUMINAMATH_CALUDE_reading_time_difference_l1287_128797


namespace NUMINAMATH_CALUDE_equation_solution_l1287_128793

theorem equation_solution : 
  ∃! x : ℝ, (Real.sqrt (x + 20) - 4 / Real.sqrt (x + 20) = 7) ∧ 
  (x = (114 + 14 * Real.sqrt 65) / 4 - 20) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1287_128793


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1287_128733

theorem isosceles_triangle_base_angle (α β γ : ℝ) : 
  -- The triangle is isosceles
  (α = β ∨ β = γ ∨ α = γ) →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- One angle is 70°
  (α = 70 ∨ β = 70 ∨ γ = 70) →
  -- The base angle is either 70° or 55°
  (α = 70 ∨ α = 55 ∨ β = 70 ∨ β = 55 ∨ γ = 70 ∨ γ = 55) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1287_128733


namespace NUMINAMATH_CALUDE_max_remainder_and_dividend_l1287_128759

theorem max_remainder_and_dividend (star : ℕ) (triangle : ℕ) :
  star / 7 = 102 ∧ star % 7 = triangle →
  triangle ≤ 6 ∧
  (triangle = 6 → star = 720) :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_and_dividend_l1287_128759


namespace NUMINAMATH_CALUDE_geometric_sequence_tenth_term_l1287_128720

theorem geometric_sequence_tenth_term
  (a₁ : ℚ)
  (a₂ : ℚ)
  (h₁ : a₁ = 4)
  (h₂ : a₂ = -2) :
  let r := a₂ / a₁
  let a_k (k : ℕ) := a₁ * r^(k - 1)
  a_k 10 = -1/128 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_tenth_term_l1287_128720


namespace NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l1287_128728

theorem proposition_p_sufficient_not_necessary_for_q (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 1| ≥ m →
   ∃ x₀ : ℝ, x₀^2 - 2*m*x₀ + m^2 + m - 3 = 0) ∧
  (∃ m : ℝ, (∃ x₀ : ℝ, x₀^2 - 2*m*x₀ + m^2 + m - 3 = 0) ∧
   ¬(∀ x : ℝ, |x + 1| + |x - 1| ≥ m)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l1287_128728


namespace NUMINAMATH_CALUDE_equation_roots_imply_m_range_l1287_128794

theorem equation_roots_imply_m_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    4^x₁ - m * 2^(x₁ + 1) + 2 - m = 0 ∧
    4^x₂ - m * 2^(x₂ + 1) + 2 - m = 0) →
  1 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_imply_m_range_l1287_128794


namespace NUMINAMATH_CALUDE_inequality_holds_l1287_128746

theorem inequality_holds (a b c : ℝ) (h : a > b) : a * |c| ≥ b * |c| := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1287_128746


namespace NUMINAMATH_CALUDE_sector_properties_l1287_128750

/-- Given a sector with radius 2 cm and central angle 2 radians, prove that its arc length is 4 cm and its area is 4 cm². -/
theorem sector_properties :
  let r : ℝ := 2  -- radius in cm
  let α : ℝ := 2  -- central angle in radians
  let arc_length : ℝ := r * α
  let sector_area : ℝ := (1/2) * r^2 * α
  (arc_length = 4 ∧ sector_area = 4) :=
by sorry

end NUMINAMATH_CALUDE_sector_properties_l1287_128750


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l1287_128711

theorem least_five_digit_congruent_to_8_mod_17 :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    (n % 17 = 8) ∧
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 8 → m ≥ n) ∧
    n = 10004 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l1287_128711


namespace NUMINAMATH_CALUDE_smallest_base_perfect_cube_l1287_128763

/-- Given a base b > 5, returns the value of 12_b in base 10 -/
def baseB_to_base10 (b : ℕ) : ℕ := b + 2

/-- Checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem smallest_base_perfect_cube :
  (∀ b : ℕ, b > 5 ∧ b < 6 → ¬ is_perfect_cube (baseB_to_base10 b)) ∧
  is_perfect_cube (baseB_to_base10 6) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_cube_l1287_128763


namespace NUMINAMATH_CALUDE_remainder_calculation_l1287_128731

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem remainder_calculation :
  rem (5/9 : ℚ) (-3/7 : ℚ) = -19/63 := by
  sorry

end NUMINAMATH_CALUDE_remainder_calculation_l1287_128731


namespace NUMINAMATH_CALUDE_kitten_growth_l1287_128747

/-- The length of a kitten after doubling twice from an initial length of 4 inches. -/
def kitten_length : ℕ := 16

/-- The initial length of the kitten in inches. -/
def initial_length : ℕ := 4

/-- Doubling function -/
def double (n : ℕ) : ℕ := 2 * n

theorem kitten_growth : kitten_length = double (double initial_length) := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_l1287_128747


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1287_128768

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Defines a line in 2D space using the equation y = mx + b -/
structure Line where
  m : ℚ  -- slope
  b : ℚ  -- y-intercept

def line1 : Line := { m := 3, b := 0 }
def line2 : Line := { m := -7, b := 5 }

theorem intersection_of_lines (l1 l2 : Line) : 
  ∃! p : IntersectionPoint, 
    p.y = l1.m * p.x + l1.b ∧ 
    p.y = l2.m * p.x + l2.b := by
  sorry

#check intersection_of_lines line1 line2

end NUMINAMATH_CALUDE_intersection_of_lines_l1287_128768


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_multiples_l1287_128775

theorem largest_of_three_consecutive_multiples (a b c : ℕ) : 
  (∃ n : ℕ, a = 3 * n ∧ b = 3 * n + 3 ∧ c = 3 * n + 6) →  -- Consecutive multiples of 3
  a + b + c = 117 →                                      -- Sum is 117
  c = 42 ∧ c ≥ a ∧ c ≥ b                                 -- c is the largest and equals 42
  := by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_multiples_l1287_128775


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l1287_128776

theorem gcd_of_polynomial_and_multiple (y : ℤ) : 
  (∃ k : ℤ, y = 30492 * k) →
  Int.gcd ((3*y+4)*(8*y+3)*(11*y+5)*(y+11)) y = 660 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l1287_128776


namespace NUMINAMATH_CALUDE_witch_cake_votes_l1287_128718

/-- The number of votes for the witch cake -/
def witch_votes : ℕ := sorry

/-- The number of votes for the unicorn cake -/
def unicorn_votes : ℕ := 3 * witch_votes

/-- The number of votes for the dragon cake -/
def dragon_votes : ℕ := witch_votes + 25

/-- The total number of votes cast -/
def total_votes : ℕ := 60

theorem witch_cake_votes :
  witch_votes = 7 ∧
  unicorn_votes = 3 * witch_votes ∧
  dragon_votes = witch_votes + 25 ∧
  witch_votes + unicorn_votes + dragon_votes = total_votes :=
sorry

end NUMINAMATH_CALUDE_witch_cake_votes_l1287_128718


namespace NUMINAMATH_CALUDE_sophie_joe_marbles_l1287_128770

theorem sophie_joe_marbles (sophie_initial : ℕ) (joe_initial : ℕ) (marbles_given : ℕ) :
  sophie_initial = 120 →
  joe_initial = 19 →
  marbles_given = 16 →
  sophie_initial - marbles_given = 3 * (joe_initial + marbles_given) :=
by
  sorry

end NUMINAMATH_CALUDE_sophie_joe_marbles_l1287_128770


namespace NUMINAMATH_CALUDE_three_digit_numbers_property_l1287_128710

theorem three_digit_numbers_property : 
  (∃! (l : List Nat), 
    (∀ n ∈ l, 100 ≤ n ∧ n < 1000) ∧ 
    (∀ n ∈ l, let a := n / 100
              let b := (n / 10) % 10
              let c := n % 10
              10 * a + c = (100 * a + 10 * b + c) / 9) ∧
    l.length = 4) := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_property_l1287_128710


namespace NUMINAMATH_CALUDE_isabel_photo_distribution_l1287_128736

/-- Given a total number of pictures and a number of albums, 
    calculate the number of pictures in each album assuming equal distribution. -/
def picturesPerAlbum (totalPictures : ℕ) (numAlbums : ℕ) : ℕ :=
  totalPictures / numAlbums

/-- Theorem stating that given 6 pictures divided into 3 albums, 
    each album contains 2 pictures. -/
theorem isabel_photo_distribution :
  let phonePhotos := 2
  let cameraPhotos := 4
  let totalPhotos := phonePhotos + cameraPhotos
  let numAlbums := 3
  picturesPerAlbum totalPhotos numAlbums = 2 := by
  sorry

end NUMINAMATH_CALUDE_isabel_photo_distribution_l1287_128736


namespace NUMINAMATH_CALUDE_steven_owes_jeremy_l1287_128737

/-- The amount Steven owes Jeremy for cleaning rooms -/
theorem steven_owes_jeremy (rate : ℚ) (rooms : ℚ) : rate = 13/3 → rooms = 5/2 → rate * rooms = 65/6 := by
  sorry

end NUMINAMATH_CALUDE_steven_owes_jeremy_l1287_128737


namespace NUMINAMATH_CALUDE_rats_to_chihuahuas_ratio_l1287_128748

theorem rats_to_chihuahuas_ratio : 
  ∀ (total : ℕ) (rats : ℕ) (chihuahuas : ℕ),
  total = 70 →
  rats = 60 →
  chihuahuas = total - rats →
  ∃ (k : ℕ), rats = k * chihuahuas →
  (rats : ℚ) / chihuahuas = 6 / 1 := by
sorry

end NUMINAMATH_CALUDE_rats_to_chihuahuas_ratio_l1287_128748


namespace NUMINAMATH_CALUDE_closest_point_on_line_l1287_128756

/-- The line y = -2x + 3 --/
def line (x : ℝ) : ℝ := -2 * x + 3

/-- The point we're finding the closest point to --/
def point : ℝ × ℝ := (2, -1)

/-- The squared distance between two points --/
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem closest_point_on_line :
  ∀ x : ℝ, squared_distance (x, line x) point ≥ squared_distance (2, line 2) point :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l1287_128756


namespace NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l1287_128799

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the center and focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- State the theorem
theorem min_dot_product_on_ellipse :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
    ∃ min_value : ℝ, min_value = 6 ∧
      ∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
        dot_product (Q.1 - O.1, Q.2 - O.2) (Q.1 - F.1, Q.2 - F.2) ≥ min_value :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l1287_128799


namespace NUMINAMATH_CALUDE_soccer_team_matches_l1287_128734

theorem soccer_team_matches :
  ∀ (initial_matches : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_matches / 5) →
    ∀ (total_matches : ℕ),
      total_matches = initial_matches + 12 →
      (initial_wins + 8 : ℚ) / total_matches = 11 / 20 →
      total_matches = 21 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_matches_l1287_128734


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1287_128777

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define parallelism between two lines
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define when a point lies on a line
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem parallel_line_through_point :
  ∃ (l : Line2D),
    parallel l (Line2D.mk 2 1 (-1)) ∧
    point_on_line (Point2D.mk 1 2) l ∧
    l = Line2D.mk 2 1 (-4) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1287_128777


namespace NUMINAMATH_CALUDE_water_distribution_l1287_128786

structure Bottle where
  volume : ℝ
  h_volume : volume > 0 ∧ volume < 1

def total_volume (bottles : List Bottle) : ℝ :=
  bottles.foldl (fun acc b => acc + b.volume) 0

theorem water_distribution (n : ℕ) (h_n : n ≥ 1) (bottles : List Bottle) 
  (h_total : total_volume bottles = n / 2) :
  ∃ (distribution : List ℝ), 
    distribution.length = n ∧ 
    (∀ v ∈ distribution, v ≤ 1) ∧
    (total_volume bottles = distribution.foldl (· + ·) 0) := by
  sorry

end NUMINAMATH_CALUDE_water_distribution_l1287_128786


namespace NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l1287_128771

theorem pythagorean_triple_6_8_10 : 
  ∃ (a b c : ℕ+), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 := by
  sorry

#check pythagorean_triple_6_8_10

end NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l1287_128771


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_on_unit_interval_l1287_128735

/-- The minimum value of the maximum absolute value of a quadratic function on [-1, 1] -/
theorem min_max_abs_quadratic_on_unit_interval :
  ∃ (F : ℝ), F = 1/2 ∧ 
  (∀ (a b : ℝ) (f : ℝ → ℝ), 
    (∀ x, f x = x^2 + a*x + b) → 
    (∀ x, |x| ≤ 1 → |f x| ≤ F) ∧
    (∃ a b : ℝ, ∃ x, |x| ≤ 1 ∧ |f x| = F)) :=
sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_on_unit_interval_l1287_128735


namespace NUMINAMATH_CALUDE_min_area_triangle_abc_l1287_128714

/-- The minimum area of a triangle ABC where A = (0, 0), B = (30, 16), and C has integer coordinates --/
theorem min_area_triangle_abc : 
  ∀ (p q : ℤ), 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 16)
  let C : ℝ × ℝ := (p, q)
  let area := (1/2 : ℝ) * |16 * p - 30 * q|
  1 ≤ area ∧ (∃ (p' q' : ℤ), (1/2 : ℝ) * |16 * p' - 30 * q'| = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_area_triangle_abc_l1287_128714


namespace NUMINAMATH_CALUDE_stating_count_initial_sets_eq_720_l1287_128791

/-- The number of letters available (A through J) -/
def num_letters : ℕ := 10

/-- The length of each set of initials -/
def set_length : ℕ := 3

/-- 
Calculates the number of different three-letter sets of initials 
using letters A through J, where no letter can be used more than once in each set.
-/
def count_initial_sets : ℕ :=
  (num_letters) * (num_letters - 1) * (num_letters - 2)

/-- 
Theorem stating that the number of different three-letter sets of initials 
using letters A through J, where no letter can be used more than once in each set, 
is equal to 720.
-/
theorem count_initial_sets_eq_720 : count_initial_sets = 720 := by
  sorry

end NUMINAMATH_CALUDE_stating_count_initial_sets_eq_720_l1287_128791


namespace NUMINAMATH_CALUDE_birds_in_trees_l1287_128741

theorem birds_in_trees (stones : ℕ) (trees : ℕ) (birds : ℕ) : 
  stones = 40 →
  trees = 3 * stones →
  birds = 2 * (trees + stones) →
  birds = 400 := by
sorry

end NUMINAMATH_CALUDE_birds_in_trees_l1287_128741


namespace NUMINAMATH_CALUDE_bus_passengers_l1287_128757

theorem bus_passengers (total : ℕ) (women_fraction : ℚ) (standing_men_fraction : ℚ) 
  (h1 : total = 48)
  (h2 : women_fraction = 2/3)
  (h3 : standing_men_fraction = 1/8) : 
  ↑total * (1 - women_fraction) * (1 - standing_men_fraction) = 14 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l1287_128757


namespace NUMINAMATH_CALUDE_sum_cube_over_power_of_three_l1287_128713

open Real BigOperators

/-- The sum of the infinite series $\sum_{k=1}^\infty \frac{k^3}{3^k}$ is equal to $\frac{39}{16}$. -/
theorem sum_cube_over_power_of_three :
  ∑' k : ℕ+, (k : ℝ)^3 / 3^(k : ℝ) = 39 / 16 := by sorry

end NUMINAMATH_CALUDE_sum_cube_over_power_of_three_l1287_128713


namespace NUMINAMATH_CALUDE_triangle_double_angle_sine_sum_l1287_128712

/-- For angles α, β, and γ of a triangle, sin 2α + sin 2β + sin 2γ = 4 sin α sin β sin γ -/
theorem triangle_double_angle_sine_sum (α β γ : ℝ) 
  (h : α + β + γ = Real.pi) : 
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 
  4 * Real.sin α * Real.sin β * Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_double_angle_sine_sum_l1287_128712


namespace NUMINAMATH_CALUDE_smallest_b_value_l1287_128778

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ c : ℕ+, c.val < b.val → 
    ¬(∃ d : ℕ+, d.val - c.val = 8 ∧ 
      Nat.gcd ((d.val^3 + c.val^3) / (d.val + c.val)) (d.val * c.val) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1287_128778


namespace NUMINAMATH_CALUDE_system_solution_l1287_128702

theorem system_solution (x y z : ℝ) : 
  (x^2 - y*z = |y - z| + 1 ∧
   y^2 - z*x = |z - x| + 1 ∧
   z^2 - x*y = |x - y| + 1) ↔ 
  ((x = 5/3 ∧ y = -4/3 ∧ z = -4/3) ∨
   (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
   (x = 5/3 ∧ y = -4/3 ∧ z = -4/3) ∨
   (x = -4/3 ∧ y = 5/3 ∧ z = -4/3) ∨
   (x = -4/3 ∧ y = -4/3 ∧ z = 5/3) ∨
   (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
   (x = 4/3 ∧ y = -5/3 ∧ z = 4/3) ∨
   (x = -5/3 ∧ y = 4/3 ∧ z = 4/3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1287_128702


namespace NUMINAMATH_CALUDE_tan_theta_values_l1287_128716

theorem tan_theta_values (θ : Real) (h : 2 * Real.sin θ = 1 + Real.cos θ) : 
  Real.tan θ = 4/3 ∨ Real.tan θ = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_values_l1287_128716


namespace NUMINAMATH_CALUDE_non_working_video_games_l1287_128796

theorem non_working_video_games (total : ℕ) (price : ℕ) (earned : ℕ) :
  total = 15 →
  price = 7 →
  earned = 63 →
  total - (earned / price) = 6 := by
sorry

end NUMINAMATH_CALUDE_non_working_video_games_l1287_128796


namespace NUMINAMATH_CALUDE_sunday_cost_theorem_l1287_128717

-- Define the constants
def weekday_discount : ℝ := 0.1
def weekend_increase : ℝ := 0.5
def shaving_cost : ℝ := 10
def styling_cost : ℝ := 15
def monday_total : ℝ := 18

-- Define the theorem
theorem sunday_cost_theorem :
  let weekday_haircut_cost := (monday_total - shaving_cost) / (1 - weekday_discount)
  let weekend_haircut_cost := weekday_haircut_cost * (1 + weekend_increase)
  let sunday_total := weekend_haircut_cost + styling_cost
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |sunday_total - 28.34| < ε :=
sorry

end NUMINAMATH_CALUDE_sunday_cost_theorem_l1287_128717


namespace NUMINAMATH_CALUDE_baron_munchausen_claim_l1287_128701

-- Define a function to calculate the sum of squares of digits
def sumOfSquaresOfDigits (n : ℕ) : ℕ := sorry

-- Define the property for the numbers we're looking for
def satisfiesProperty (a b : ℕ) : Prop :=
  (a ≠ b) ∧ 
  (a ≥ 10^9) ∧ (a < 10^10) ∧ 
  (b ≥ 10^9) ∧ (b < 10^10) ∧ 
  (a % 10 ≠ 0) ∧ (b % 10 ≠ 0) ∧
  (a - sumOfSquaresOfDigits a = b - sumOfSquaresOfDigits b)

-- Theorem statement
theorem baron_munchausen_claim : ∃ a b : ℕ, satisfiesProperty a b := by sorry

end NUMINAMATH_CALUDE_baron_munchausen_claim_l1287_128701


namespace NUMINAMATH_CALUDE_money_division_l1287_128707

/-- Given a division of money among three people a, b, and c, where b's share is 65% of a's
    and c's share is 40% of a's, and c's share is 64 rupees, prove that the total sum is 328 rupees. -/
theorem money_division (a b c : ℝ) : 
  (b = 0.65 * a) →  -- b's share is 65% of a's
  (c = 0.40 * a) →  -- c's share is 40% of a's
  (c = 64) →        -- c's share is 64 rupees
  (a + b + c = 328) -- total sum is 328 rupees
:= by sorry

end NUMINAMATH_CALUDE_money_division_l1287_128707


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l1287_128730

/-- Proves the number of girls in a school given specific conditions --/
theorem number_of_girls_in_school (total_students : ℕ) 
  (avg_age_boys avg_age_girls avg_age_school : ℚ) :
  total_students = 604 →
  avg_age_boys = 12 →
  avg_age_girls = 11 →
  avg_age_school = 47/4 →
  ∃ (num_girls : ℕ), num_girls = 151 ∧ 
    (num_girls : ℚ) * avg_age_girls + (total_students - num_girls : ℚ) * avg_age_boys = 
      total_students * avg_age_school :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l1287_128730


namespace NUMINAMATH_CALUDE_translate_f_to_g_l1287_128703

def f (x : ℝ) : ℝ := 2 * x^2

def g (x : ℝ) : ℝ := 2 * (x + 1)^2 + 3

theorem translate_f_to_g : 
  ∀ x : ℝ, g x = f (x + 1) + 3 := by sorry

end NUMINAMATH_CALUDE_translate_f_to_g_l1287_128703


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_product_of_c_values_l1287_128754

theorem quadratic_equation_rational_solutions (c : ℕ+) : 
  (∃ x : ℚ, 3 * x^2 + 17 * x + c.val = 0) ↔ (c.val = 14 ∨ c.val = 24) :=
sorry

theorem product_of_c_values : 
  (∃ c₁ c₂ : ℕ+, c₁ ≠ c₂ ∧ 
    (∃ x : ℚ, 3 * x^2 + 17 * x + c₁.val = 0) ∧ 
    (∃ x : ℚ, 3 * x^2 + 17 * x + c₂.val = 0) ∧
    c₁.val * c₂.val = 336) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_product_of_c_values_l1287_128754


namespace NUMINAMATH_CALUDE_triangle_angle_determination_l1287_128789

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that if a = 2√3, b = 6, and A = 30°, then B = 60° or B = 120°. -/
theorem triangle_angle_determination (a b c : ℝ) (A B C : ℝ) :
  a = 2 * Real.sqrt 3 →
  b = 6 →
  A = π / 6 →
  (B = π / 3 ∨ B = 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_determination_l1287_128789


namespace NUMINAMATH_CALUDE_statement_holds_for_given_numbers_l1287_128773

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def given_numbers : List ℕ := [45, 54, 63, 81]

theorem statement_holds_for_given_numbers :
  ∀ n ∈ given_numbers, (sum_of_digits n) % 9 = 0 → n % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_statement_holds_for_given_numbers_l1287_128773


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l1287_128743

/-- The function f(x) = 2 - x^2 -/
def f (x : ℝ) : ℝ := 2 - x^2

/-- The monotonic decreasing interval of f(x) = 2 - x^2 is (0, +∞) -/
theorem monotonic_decreasing_interval_of_f :
  ∀ x y, 0 < x → x < y → f y < f x :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l1287_128743


namespace NUMINAMATH_CALUDE_all_items_used_as_money_l1287_128732

structure MoneyItem where
  name : String
  used_as_money : Bool

def gold : MoneyItem := { name := "gold", used_as_money := true }
def stones : MoneyItem := { name := "stones", used_as_money := true }
def horses : MoneyItem := { name := "horses", used_as_money := true }
def dried_fish : MoneyItem := { name := "dried fish", used_as_money := true }
def mollusk_shells : MoneyItem := { name := "mollusk shells", used_as_money := true }

def money_items : List MoneyItem := [gold, stones, horses, dried_fish, mollusk_shells]

theorem all_items_used_as_money :
  (∀ item ∈ money_items, item.used_as_money = true) →
  (¬ ∃ item ∈ money_items, item.used_as_money = false) := by
  sorry

end NUMINAMATH_CALUDE_all_items_used_as_money_l1287_128732


namespace NUMINAMATH_CALUDE_alice_fruit_consumption_impossible_l1287_128708

/-- Represents the number of each type of fruit in the basket -/
structure FruitBasket :=
  (apples : ℕ)
  (pears : ℕ)
  (oranges : ℕ)

/-- Represents Alice's fruit consumption for a day -/
inductive DailyConsumption
  | AP  -- Apple and Pear
  | AO  -- Apple and Orange
  | PO  -- Pear and Orange

def initial_basket : FruitBasket :=
  { apples := 5, pears := 8, oranges := 11 }

def consume_fruits (basket : FruitBasket) (consumption : DailyConsumption) : FruitBasket :=
  match consumption with
  | DailyConsumption.AP => { apples := basket.apples - 1, pears := basket.pears - 1, oranges := basket.oranges }
  | DailyConsumption.AO => { apples := basket.apples - 1, pears := basket.pears, oranges := basket.oranges - 1 }
  | DailyConsumption.PO => { apples := basket.apples, pears := basket.pears - 1, oranges := basket.oranges - 1 }

def fruits_equal (basket : FruitBasket) : Prop :=
  basket.apples = basket.pears ∧ basket.pears = basket.oranges

theorem alice_fruit_consumption_impossible :
  ∀ (days : ℕ) (consumptions : List DailyConsumption),
    days = consumptions.length →
    ¬(fruits_equal (consumptions.foldl consume_fruits initial_basket)) :=
  sorry


end NUMINAMATH_CALUDE_alice_fruit_consumption_impossible_l1287_128708


namespace NUMINAMATH_CALUDE_sqrt_of_square_positive_l1287_128727

theorem sqrt_of_square_positive (a : ℝ) (h : a > 0) : Real.sqrt (a^2) = a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_positive_l1287_128727


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l1287_128762

/-- Proves that given a person who misses a bus by 6 minutes when walking at 4/5 of their usual speed, their usual time to catch the bus is 24 minutes. -/
theorem usual_time_to_catch_bus (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0)
  (h3 : (4/5 * usual_speed) * (usual_time + 6) = usual_speed * usual_time) : 
  usual_time = 24 := by
sorry


end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l1287_128762


namespace NUMINAMATH_CALUDE_blood_donation_selection_count_l1287_128782

def total_teachers : ℕ := 9
def male_teachers : ℕ := 3
def female_teachers : ℕ := 6
def selected_teachers : ℕ := 5

theorem blood_donation_selection_count :
  (Nat.choose total_teachers selected_teachers) - (Nat.choose female_teachers selected_teachers) = 120 := by
  sorry

end NUMINAMATH_CALUDE_blood_donation_selection_count_l1287_128782


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1287_128751

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - k*x + 1 = (x - a)^2) → (k = 2 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1287_128751


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_1000_l1287_128790

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of all numbers from 1 to 1000 is 13501 -/
theorem sum_of_digits_up_to_1000 : sumOfDigitsUpTo 1000 = 13501 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_1000_l1287_128790


namespace NUMINAMATH_CALUDE_sequence_first_element_l1287_128755

def sequence_property (a b c d e : ℚ) : Prop :=
  c = a * b ∧ d = b * c ∧ e = c * d

theorem sequence_first_element :
  ∀ a b c d e : ℚ,
    sequence_property a b c d e →
    c = 3 →
    e = 18 →
    a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_first_element_l1287_128755


namespace NUMINAMATH_CALUDE_four_numbers_problem_l1287_128745

theorem four_numbers_problem (A B C D : ℤ) : 
  A + B + C + D = 43 ∧ 
  2 * A + 8 = 3 * B ∧ 
  3 * B = 4 * C ∧ 
  4 * C = 5 * D - 4 →
  A = 14 ∧ B = 12 ∧ C = 9 ∧ D = 8 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_problem_l1287_128745


namespace NUMINAMATH_CALUDE_equation_solution_l1287_128752

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (2 / (x - 2) = 3 / (x + 2)) ↔ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1287_128752


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1287_128729

-- Define a geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geo : isGeometric a)
  (h_pos : ∀ n, a n > 0)
  (h_sum : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1287_128729


namespace NUMINAMATH_CALUDE_class_mean_calculation_l1287_128726

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students group2_students group3_students : ℕ)
  (group1_mean group2_mean group3_mean : ℚ) :
  total_students = group1_students + group2_students + group3_students →
  group1_students = 50 →
  group2_students = 8 →
  group3_students = 2 →
  group1_mean = 68 / 100 →
  group2_mean = 75 / 100 →
  group3_mean = 82 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean + group3_students * group3_mean) / total_students = 694 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l1287_128726


namespace NUMINAMATH_CALUDE_toy_store_shelves_l1287_128706

def number_of_shelves (initial_stock new_shipment bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

theorem toy_store_shelves : 
  number_of_shelves 4 10 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l1287_128706


namespace NUMINAMATH_CALUDE_remainder_difference_l1287_128721

theorem remainder_difference (d r : ℤ) : d > 1 →
  1134 % d = r →
  1583 % d = r →
  2660 % d = r →
  d - r = 213 := by sorry

end NUMINAMATH_CALUDE_remainder_difference_l1287_128721


namespace NUMINAMATH_CALUDE_specific_female_selection_probability_l1287_128722

def total_students : ℕ := 50
def male_students : ℕ := 30
def selected_students : ℕ := 5

theorem specific_female_selection_probability :
  (selected_students : ℚ) / total_students = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_specific_female_selection_probability_l1287_128722


namespace NUMINAMATH_CALUDE_converse_of_square_angles_is_false_l1287_128787

-- Define a quadrilateral
structure Quadrilateral where
  angles : Fin 4 → ℝ

-- Define a property for right angles
def has_right_angles (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, q.angles i = 90

-- Define a property for equal sides
def has_equal_sides (q : Quadrilateral) : Prop :=
  -- This is a placeholder definition, as we don't have side lengths in our structure
  True

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  has_right_angles q ∧ has_equal_sides q

-- The theorem to prove
theorem converse_of_square_angles_is_false : 
  ¬(∀ q : Quadrilateral, has_right_angles q → is_square q) := by
  sorry

end NUMINAMATH_CALUDE_converse_of_square_angles_is_false_l1287_128787


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l1287_128785

/-- A quadrilateral with an inscribed circle -/
structure InscribedCircleQuadrilateral where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- Length of AP -/
  ap : ℝ
  /-- Length of PB -/
  pb : ℝ
  /-- Length of CQ -/
  cq : ℝ
  /-- Length of QD -/
  qd : ℝ
  /-- The circle is tangent to AB at P and to CD at Q -/
  tangent_condition : True

/-- The theorem stating that for the given quadrilateral, the square of the radius is 13325 -/
theorem inscribed_circle_radius_squared
  (quad : InscribedCircleQuadrilateral)
  (h1 : quad.ap = 25)
  (h2 : quad.pb = 35)
  (h3 : quad.cq = 30)
  (h4 : quad.qd = 40) :
  quad.r ^ 2 = 13325 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l1287_128785


namespace NUMINAMATH_CALUDE_power_division_equality_l1287_128709

theorem power_division_equality : (3^3)^2 / 3^2 = 81 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l1287_128709


namespace NUMINAMATH_CALUDE_matthews_crackers_l1287_128788

/-- The number of crackers Matthew gave to each friend -/
def crackers_per_friend : ℕ := 6

/-- The number of friends Matthew gave crackers to -/
def number_of_friends : ℕ := 6

/-- The total number of crackers Matthew had -/
def total_crackers : ℕ := crackers_per_friend * number_of_friends

theorem matthews_crackers : total_crackers = 36 := by
  sorry

end NUMINAMATH_CALUDE_matthews_crackers_l1287_128788


namespace NUMINAMATH_CALUDE_abs_S_eq_1024_l1287_128780

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the expression S
def S : ℂ := (1 + i)^18 - (1 - i)^18

-- Theorem statement
theorem abs_S_eq_1024 : Complex.abs S = 1024 := by
  sorry

end NUMINAMATH_CALUDE_abs_S_eq_1024_l1287_128780


namespace NUMINAMATH_CALUDE_f_odd_and_monotonic_l1287_128798

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else -x^2 + 2*x

theorem f_odd_and_monotonic :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_monotonic_l1287_128798


namespace NUMINAMATH_CALUDE_ln_product_eq_sum_of_ln_l1287_128784

-- Define the formal power series type
def FormalPowerSeries (α : Type*) := ℕ → α

-- Define the logarithm operation for formal power series
noncomputable def Ln (f : FormalPowerSeries ℝ) : FormalPowerSeries ℝ := sorry

-- Define the multiplication operation for formal power series
def mul (f g : FormalPowerSeries ℝ) : FormalPowerSeries ℝ := sorry

-- Theorem statement
theorem ln_product_eq_sum_of_ln 
  (f h : FormalPowerSeries ℝ) 
  (hf : f 0 = 1) 
  (hh : h 0 = 1) : 
  Ln (mul f h) = λ n => (Ln f n) + (Ln h n) := by sorry

end NUMINAMATH_CALUDE_ln_product_eq_sum_of_ln_l1287_128784


namespace NUMINAMATH_CALUDE_book_selection_l1287_128767

theorem book_selection (n m k : ℕ) (h1 : n = 7) (h2 : m = 5) (h3 : k = 3) :
  (Nat.choose (n - 2) k) = (Nat.choose m k) :=
by sorry

end NUMINAMATH_CALUDE_book_selection_l1287_128767


namespace NUMINAMATH_CALUDE_positions_after_307_moves_l1287_128772

/-- Represents the positions of the cat -/
inductive CatPosition
  | Top
  | TopRight
  | BottomRight
  | Bottom
  | BottomLeft
  | TopLeft

/-- Represents the positions of the mouse -/
inductive MousePosition
  | Top
  | TopRight
  | BetweenTopRightAndBottomRight
  | BottomRight
  | BetweenBottomRightAndBottom
  | Bottom
  | BottomLeft
  | BetweenBottomLeftAndTopLeft
  | TopLeft
  | BetweenTopLeftAndTop

/-- The number of hexagons in the larger hexagon -/
def numHexagons : Nat := 6

/-- The number of segments the mouse moves through -/
def numMouseSegments : Nat := 12

/-- Calculates the cat's position after a given number of moves -/
def catPositionAfterMoves (moves : Nat) : CatPosition :=
  match moves % numHexagons with
  | 0 => CatPosition.TopLeft
  | 1 => CatPosition.Top
  | 2 => CatPosition.TopRight
  | 3 => CatPosition.BottomRight
  | 4 => CatPosition.Bottom
  | 5 => CatPosition.BottomLeft
  | _ => CatPosition.Top  -- This case should never occur

/-- Calculates the mouse's position after a given number of moves -/
def mousePositionAfterMoves (moves : Nat) : MousePosition :=
  match moves % numMouseSegments with
  | 0 => MousePosition.Top
  | 1 => MousePosition.BetweenTopLeftAndTop
  | 2 => MousePosition.TopLeft
  | 3 => MousePosition.BetweenBottomLeftAndTopLeft
  | 4 => MousePosition.BottomLeft
  | 5 => MousePosition.Bottom
  | 6 => MousePosition.BetweenBottomRightAndBottom
  | 7 => MousePosition.BottomRight
  | 8 => MousePosition.BetweenTopRightAndBottomRight
  | 9 => MousePosition.TopRight
  | 10 => MousePosition.Top
  | 11 => MousePosition.BetweenTopLeftAndTop
  | _ => MousePosition.Top  -- This case should never occur

theorem positions_after_307_moves :
  catPositionAfterMoves 307 = CatPosition.Top ∧
  mousePositionAfterMoves 307 = MousePosition.BetweenBottomRightAndBottom :=
by sorry

end NUMINAMATH_CALUDE_positions_after_307_moves_l1287_128772


namespace NUMINAMATH_CALUDE_simplify_expression_find_value_evaluate_composite_l1287_128781

-- Part 1
theorem simplify_expression (x y : ℝ) :
  3 * (x - y)^2 - 6 * (x - y)^2 + 2 * (x - y)^2 = -(x - y)^2 := by
  sorry

-- Part 2
theorem find_value (a b : ℝ) (h : a^2 - 2*b = 4) :
  3*a^2 - 6*b - 21 = -9 := by
  sorry

-- Part 3
theorem evaluate_composite (a b c d : ℝ) 
  (h1 : a - 5*b = 3) 
  (h2 : 5*b - 3*c = -5) 
  (h3 : 3*c - d = 10) :
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_find_value_evaluate_composite_l1287_128781


namespace NUMINAMATH_CALUDE_tropicenglish_word_count_l1287_128724

/-- Represents a letter in Tropicenglish -/
inductive TropicLetter
| A | M | O | P | T

/-- Represents whether a letter is a vowel or consonant -/
def isVowel : TropicLetter → Bool
  | TropicLetter.A => true
  | TropicLetter.O => true
  | _ => false

/-- A Tropicenglish word is a list of TropicLetters -/
def TropicWord := List TropicLetter

/-- Checks if a word is valid in Tropicenglish -/
def isValidWord (word : TropicWord) : Bool :=
  let consonantsBetweenVowels (w : TropicWord) : Bool :=
    -- Implementation details omitted
    sorry
  word.length == 6 && consonantsBetweenVowels word

/-- Counts the number of valid 6-letter Tropicenglish words -/
def countValidWords : Nat :=
  -- Implementation details omitted
  sorry

/-- The main theorem to prove -/
theorem tropicenglish_word_count : 
  ∃ (n : Nat), n < 1000 ∧ countValidWords % 1000 = n :=
sorry

end NUMINAMATH_CALUDE_tropicenglish_word_count_l1287_128724


namespace NUMINAMATH_CALUDE_charity_race_fundraising_l1287_128766

theorem charity_race_fundraising (total_students : ℕ) (group1_students : ℕ) (group1_amount : ℕ) (group2_amount : ℕ) :
  total_students = 30 →
  group1_students = 10 →
  group1_amount = 20 →
  group2_amount = 30 →
  (group1_students * group1_amount) + ((total_students - group1_students) * group2_amount) = 800 :=
by sorry

end NUMINAMATH_CALUDE_charity_race_fundraising_l1287_128766


namespace NUMINAMATH_CALUDE_area_of_rectangle_with_three_squares_l1287_128742

/-- Given three non-overlapping squares where one square has twice the side length of the other two,
    and the larger square has an area of 4 square inches, the area of the rectangle encompassing
    all three squares is 6 square inches. -/
theorem area_of_rectangle_with_three_squares (s : ℝ) : 
  s > 0 → (2 * s)^2 = 4 → 3 * s * 2 * s = 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rectangle_with_three_squares_l1287_128742


namespace NUMINAMATH_CALUDE_money_distribution_l1287_128744

theorem money_distribution (P Q R S : ℕ) : 
  P = 2 * Q →  -- P gets twice as that of Q
  S = 4 * R →  -- S gets 4 times as that of R
  Q = R →      -- Q and R are to receive equal amounts
  S - P = 250 →  -- The difference between S and P is 250
  P + Q + R + S = 1000 :=  -- Total amount to be distributed
by sorry

end NUMINAMATH_CALUDE_money_distribution_l1287_128744


namespace NUMINAMATH_CALUDE_right_triangle_tan_y_l1287_128783

theorem right_triangle_tan_y (XY YZ : ℝ) (h_right_angle : XY ^ 2 + XZ ^ 2 = YZ ^ 2) 
  (h_XY : XY = 24) (h_YZ : YZ = 26) : Real.tan Y = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tan_y_l1287_128783


namespace NUMINAMATH_CALUDE_symmetry_implies_difference_l1287_128738

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_implies_difference (a b : ℝ) :
  symmetric_wrt_y_axis (a, 3) (4, b) → a - b = -7 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_implies_difference_l1287_128738


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l1287_128700

/-- Given three points A, B, and C in a plane, where C divides AB in a 1:3 ratio,
    prove that the sum of coordinates of A is 21 when B and C are known. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  B = (2, 10) →
  C = (5, 4) →
  A.1 + A.2 = 21 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l1287_128700


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l1287_128753

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l1287_128753


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l1287_128774

theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) > 0) 
  (h2 : Real.sin α + Real.cos α < 0) : 
  α ∈ Set.Icc (Real.pi) (3 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l1287_128774


namespace NUMINAMATH_CALUDE_power_of_two_plus_three_l1287_128704

/-- Definition of the sequences a_i and b_i -/
def sequence_step (a b : ℤ) : ℤ × ℤ :=
  if a < b then (2*a + 1, b - a - 1)
  else if a > b then (a - b - 1, 2*b + 1)
  else (a, b)

/-- Theorem statement -/
theorem power_of_two_plus_three (n : ℕ) :
  (∃ k : ℕ, ∃ a b : ℕ → ℤ,
    a 0 = 1 ∧ b 0 = n ∧
    (∀ i : ℕ, i > 0 → (a i, b i) = sequence_step (a (i-1)) (b (i-1))) ∧
    a k = b k) →
  ∃ m : ℕ, n + 3 = 2^m := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_three_l1287_128704


namespace NUMINAMATH_CALUDE_card_sending_probability_l1287_128769

def num_senders : ℕ := 3
def num_recipients : ℕ := 2

theorem card_sending_probability :
  let total_outcomes := num_recipients ^ num_senders
  let favorable_outcomes := num_recipients
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_card_sending_probability_l1287_128769


namespace NUMINAMATH_CALUDE_car_repair_cost_l1287_128715

/-- Calculates the total cost for a car repair given the hourly rate, hours worked per day,
    number of days worked, and cost of parts. -/
def total_cost (hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (parts_cost : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_worked + parts_cost

/-- Proves that the total cost for the car repair is $9220 given the specified conditions. -/
theorem car_repair_cost :
  total_cost 60 8 14 2500 = 9220 := by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_l1287_128715


namespace NUMINAMATH_CALUDE_multiple_problem_l1287_128779

theorem multiple_problem (n : ℝ) (m : ℝ) (h1 : n = 25.0) (h2 : m * n = 3 * n - 25) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l1287_128779


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1287_128792

/-- A geometric sequence with positive terms, a₁ = 1, and a₁ + a₂ + a₃ = 7 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (a 1 + a 2 + a 3 = 7) ∧
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n)

/-- The general formula for the geometric sequence -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1287_128792


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1287_128795

theorem fraction_sum_equality : ∃ (a b c d e f : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (a ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   b ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   c ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   d ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   e ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   f ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ)) ∧
  (Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1) ∧
  (a * d * f + c * b * f = e * b * d) ∧
  (b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1287_128795


namespace NUMINAMATH_CALUDE_purple_to_seafoam_ratio_is_one_fourth_l1287_128749

/-- The number of skirts in Azure Valley -/
def azure_skirts : ℕ := 60

/-- The number of skirts in Purple Valley -/
def purple_skirts : ℕ := 10

/-- The number of skirts in Seafoam Valley -/
def seafoam_skirts : ℕ := (2 * azure_skirts) / 3

/-- The ratio of skirts in Purple Valley to Seafoam Valley -/
def purple_to_seafoam_ratio : ℚ := purple_skirts / seafoam_skirts

theorem purple_to_seafoam_ratio_is_one_fourth :
  purple_to_seafoam_ratio = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_purple_to_seafoam_ratio_is_one_fourth_l1287_128749


namespace NUMINAMATH_CALUDE_sin_theta_value_l1287_128765

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < π) : 
  Real.sin θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1287_128765


namespace NUMINAMATH_CALUDE_numbers_five_units_from_negative_one_l1287_128740

theorem numbers_five_units_from_negative_one :
  ∀ x : ℝ, |x - (-1)| = 5 ↔ x = 4 ∨ x = -6 := by sorry

end NUMINAMATH_CALUDE_numbers_five_units_from_negative_one_l1287_128740


namespace NUMINAMATH_CALUDE_max_min_difference_c_l1287_128761

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 18) : 
  ∃ (c_max c_min : ℝ), 
    (∀ c', a + b + c' = 3 ∧ a^2 + b^2 + c'^2 = 18 → c' ≤ c_max ∧ c' ≥ c_min) ∧
    c_max - c_min = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l1287_128761


namespace NUMINAMATH_CALUDE_problem_solution_l1287_128705

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x/y + y/x = 8) :
  (x + y)/(x - y) = Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1287_128705


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1287_128760

theorem linear_equation_solution :
  ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 ∧ x = -30 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1287_128760
