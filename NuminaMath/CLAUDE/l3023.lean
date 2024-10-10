import Mathlib

namespace central_cell_value_l3023_302358

theorem central_cell_value (a b c d e f g h i : ℝ) 
  (row_prod : a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10)
  (col_prod : a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10)
  (square_prod : a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3) :
  e = 0.00081 := by
  sorry

end central_cell_value_l3023_302358


namespace simple_interest_problem_l3023_302309

/-- Given a principal P and an interest rate R, if increasing the rate by 15%
    results in $300 more interest over 10 years, then P must equal $200. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) : 
  (P * (R + 15) * 10 / 100 = P * R * 10 / 100 + 300) → P = 200 := by
  sorry

end simple_interest_problem_l3023_302309


namespace concentric_circles_radii_difference_l3023_302371

theorem concentric_circles_radii_difference
  (r R : ℝ) -- r and R are real numbers representing radii
  (h_positive : r > 0) -- r is positive
  (h_ratio : π * R^2 = 4 * π * r^2) -- area ratio is 1:4
  : R - r = r := by
sorry

end concentric_circles_radii_difference_l3023_302371


namespace rectangle_24_60_parts_l3023_302386

/-- The number of parts a rectangle is divided into when split into unit squares and its diagonal is drawn -/
def rectangle_parts (width : ℕ) (length : ℕ) : ℕ :=
  width * length + width + length - Nat.gcd width length

/-- Theorem stating that a 24 × 60 rectangle divided into unit squares and with its diagonal drawn is divided into 1512 parts -/
theorem rectangle_24_60_parts :
  rectangle_parts 24 60 = 1512 := by
  sorry

#eval rectangle_parts 24 60

end rectangle_24_60_parts_l3023_302386


namespace parabola_translation_l3023_302303

/-- The equation of a parabola translated upwards by 1 unit from y = x^2 -/
theorem parabola_translation (x y : ℝ) : 
  (y = x^2) → (∃ y', y' = y + 1 ∧ y' = x^2 + 1) :=
by sorry

end parabola_translation_l3023_302303


namespace rectangle_area_l3023_302350

theorem rectangle_area (p : ℝ) (p_small : ℝ) (h1 : p = 30) (h2 : p_small = 16) :
  let w := (p - p_small) / 2
  let l := p_small / 2 - w + w
  w * l = 56 := by sorry

end rectangle_area_l3023_302350


namespace mix_alcohol_solutions_l3023_302332

/-- Represents an alcohol solution with a given volume and concentration -/
structure AlcoholSolution where
  volume : ℝ
  concentration : ℝ

/-- Proves that mixing two alcohol solutions results in the desired solution -/
theorem mix_alcohol_solutions
  (solution_a : AlcoholSolution)
  (solution_b : AlcoholSolution)
  (mixed_solution : AlcoholSolution)
  (h1 : solution_a.volume = 10.5)
  (h2 : solution_a.concentration = 0.75)
  (h3 : solution_b.volume = 7.5)
  (h4 : solution_b.concentration = 0.15)
  (h5 : mixed_solution.volume = 18)
  (h6 : mixed_solution.concentration = 0.5)
  : solution_a.volume * solution_a.concentration + solution_b.volume * solution_b.concentration
    = mixed_solution.volume * mixed_solution.concentration :=
by
  sorry

#check mix_alcohol_solutions

end mix_alcohol_solutions_l3023_302332


namespace max_log_sum_l3023_302374

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 4*y = 40) :
  ∃ (max : ℝ), max = 8 * Real.log 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 4*b = 40 → Real.log a + Real.log b ≤ max :=
by sorry

end max_log_sum_l3023_302374


namespace thousand_pow_seven_div_ten_pow_seventeen_l3023_302325

theorem thousand_pow_seven_div_ten_pow_seventeen :
  (1000 : ℕ)^7 / (10 : ℕ)^17 = (10 : ℕ)^4 := by
  sorry

end thousand_pow_seven_div_ten_pow_seventeen_l3023_302325


namespace arithmetic_sequence_sum_mod_15_l3023_302348

theorem arithmetic_sequence_sum_mod_15 : 
  let first_term := 1
  let last_term := 101
  let common_diff := 5
  let num_terms := (last_term - first_term) / common_diff + 1
  ∃ (sum : ℕ), sum = (num_terms * (first_term + last_term)) / 2 ∧ sum ≡ 6 [MOD 15] :=
by sorry

end arithmetic_sequence_sum_mod_15_l3023_302348


namespace exterior_angle_decreases_l3023_302338

theorem exterior_angle_decreases (n : ℕ) (h : n > 2) :
  (360 : ℝ) / (n + 1) < 360 / n := by
sorry

end exterior_angle_decreases_l3023_302338


namespace problem_statement_l3023_302395

/-- Given real numbers a and b satisfying a + 2b = 9, prove:
    1. If |9 - 2b| + |a + 1| < 3, then -2 < a < 1.
    2. If a > 0, b > 0, and z = ab^2, then the maximum value of z is 27. -/
theorem problem_statement (a b : ℝ) (h1 : a + 2*b = 9) :
  (|9 - 2*b| + |a + 1| < 3 → -2 < a ∧ a < 1) ∧
  (a > 0 → b > 0 → ∃ z : ℝ, z = a*b^2 ∧ ∀ w : ℝ, w = a*b^2 → w ≤ 27) :=
by sorry

end problem_statement_l3023_302395


namespace range_of_a_for_nonempty_solution_set_l3023_302360

theorem range_of_a_for_nonempty_solution_set (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) ↔ a ∈ Set.Ici 2 ∪ Set.Iic (-6) :=
sorry

end range_of_a_for_nonempty_solution_set_l3023_302360


namespace simplify_expression_l3023_302364

theorem simplify_expression (x : ℝ) : 125 * x - 57 * x = 68 * x := by
  sorry

end simplify_expression_l3023_302364


namespace arithmetic_series_sum_problem_solution_l3023_302337

theorem arithmetic_series_sum (a₁ d n : ℕ) (h : n > 0) : 
  (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d) = (n : ℝ) / 2 * (a₁ + (a₁ + (n - 1) * d)) :=
by sorry

theorem problem_solution : 
  let a₁ : ℕ := 9
  let d : ℕ := 4
  let n : ℕ := 50
  (n : ℝ) / 2 * (a₁ + (a₁ + (n - 1) * d)) = 5350 :=
by sorry

end arithmetic_series_sum_problem_solution_l3023_302337


namespace line_plane_perpendicularity_l3023_302375

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Two lines are distinct -/
def distinct_lines (l1 l2 : Line3D) : Prop := sorry

/-- Two planes are distinct -/
def distinct_planes (p1 p2 : Plane3D) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Two lines are parallel -/
def lines_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perpendicular (p1 p2 : Plane3D) : Prop := sorry

theorem line_plane_perpendicularity 
  (m n : Line3D) (α β : Plane3D) 
  (h1 : distinct_lines m n)
  (h2 : distinct_planes α β)
  (h3 : line_parallel_to_plane m α)
  (h4 : line_perpendicular_to_plane n β)
  (h5 : lines_parallel m n) :
  planes_perpendicular α β := by sorry

end line_plane_perpendicularity_l3023_302375


namespace milk_production_l3023_302390

/-- Given x cows with efficiency α producing y gallons in z days,
    calculate the milk production of w cows with efficiency β in v days -/
theorem milk_production
  (x y z w v : ℝ) (α β : ℝ) (hx : x > 0) (hz : z > 0) (hα : α > 0) :
  let production := (β * y * w * v) / (α^2 * x * z)
  production = (β * y * w * v) / (α^2 * x * z) := by
  sorry

end milk_production_l3023_302390


namespace line_inclination_angle_l3023_302319

theorem line_inclination_angle (x y : ℝ) :
  x - Real.sqrt 3 * y + 2 = 0 →
  Real.arctan (1 / Real.sqrt 3) = π / 6 :=
by sorry

end line_inclination_angle_l3023_302319


namespace complex_vector_magnitude_l3023_302382

/-- Given two complex numbers, prove that the magnitude of their difference is √29 -/
theorem complex_vector_magnitude (z1 z2 : ℂ) : 
  z1 = 1 - 2*I ∧ z2 = -1 + 3*I → Complex.abs (z2 - z1) = Real.sqrt 29 := by
  sorry

end complex_vector_magnitude_l3023_302382


namespace abs_sum_inequality_solution_set_l3023_302308

theorem abs_sum_inequality_solution_set :
  {x : ℝ | |x - 1| + |x| < 3} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end abs_sum_inequality_solution_set_l3023_302308


namespace equal_integers_from_ratio_l3023_302361

theorem equal_integers_from_ratio (a b : ℕ+) 
  (hK : K = Real.sqrt ((a.val ^ 2 + b.val ^ 2) / 2))
  (hA : A = (a.val + b.val) / 2)
  (hKA : ∃ (n : ℕ+), K / A = n.val) :
  a = b := by
  sorry

end equal_integers_from_ratio_l3023_302361


namespace nested_radical_value_l3023_302357

/-- The value of the infinite nested radical sqrt(3 - sqrt(3 - sqrt(3 - ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3))))))

/-- Theorem stating that the nested radical equals (-1 + √13) / 2 -/
theorem nested_radical_value : nestedRadical = (-1 + Real.sqrt 13) / 2 := by
  sorry

end nested_radical_value_l3023_302357


namespace parallel_line_through_point_l3023_302349

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) is on a line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point (x₀ y₀ : ℝ) :
  ∃ (l : Line), l.isParallel ⟨1, -2, -2⟩ ∧ l.containsPoint 1 0 ∧ l = ⟨1, -2, -1⟩ := by
  sorry

end parallel_line_through_point_l3023_302349


namespace sin_eq_sin_sin_unique_solution_l3023_302333

noncomputable def arcsin_099 : ℝ := Real.arcsin 0.99

theorem sin_eq_sin_sin_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ arcsin_099 ∧ Real.sin x = Real.sin (Real.sin x) :=
sorry

end sin_eq_sin_sin_unique_solution_l3023_302333


namespace absolute_value_inequality_l3023_302330

theorem absolute_value_inequality (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := by
  sorry

end absolute_value_inequality_l3023_302330


namespace cosine_equation_roots_l3023_302365

theorem cosine_equation_roots :
  ∃ (a b c : ℝ), (∀ x : ℝ, 4 * Real.cos (2007 * x) = 2007 * x ↔ x = a ∨ x = b ∨ x = c) ∧
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end cosine_equation_roots_l3023_302365


namespace gold_coins_in_urn_l3023_302329

-- Define the total percentage
def total_percentage : ℝ := 100

-- Define the percentage of beads
def bead_percentage : ℝ := 30

-- Define the percentage of silver coins among all coins
def silver_coin_percentage : ℝ := 50

-- Define the percentage of coins
def coin_percentage : ℝ := total_percentage - bead_percentage

-- Define the percentage of gold coins among all coins
def gold_coin_percentage : ℝ := total_percentage - silver_coin_percentage

-- Theorem to prove
theorem gold_coins_in_urn : 
  (coin_percentage * gold_coin_percentage) / total_percentage = 35 := by
  sorry

end gold_coins_in_urn_l3023_302329


namespace exactly_two_statements_true_l3023_302320

theorem exactly_two_statements_true :
  let statement1 := (¬∀ x : ℝ, x^2 - 3*x - 2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 3*x₀ - 2 ≤ 0)
  let statement2 := ∀ P Q : Prop, (P ∨ Q → P ∧ Q) ∧ ¬(P ∧ Q → P ∨ Q)
  let statement3 := ∃ m : ℝ, ∀ x : ℝ, x > 0 → (
    (∃ α : ℝ, ∀ x : ℝ, x > 0 → m * x^(m^2 + 2*m) = x^α) ∧
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → m * x₁^(m^2 + 2*m) < m * x₂^(m^2 + 2*m))
  )
  let statement4 := ∀ a b : ℝ, a ≠ 0 ∧ b ≠ 0 →
    (∀ x y : ℝ, x/a + y/b = 1 ↔ ∃ k : ℝ, k ≠ 0 ∧ x = k*a ∧ y = k*b)
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) :=
by sorry

end exactly_two_statements_true_l3023_302320


namespace journey_final_distance_l3023_302372

-- Define the directions
inductive Direction
| NorthEast
| SouthEast
| SouthWest
| NorthWest

-- Define a leg of the journey
structure Leg where
  distance : ℝ
  direction : Direction

-- Define the journey
def journey : List Leg := [
  { distance := 5, direction := Direction.NorthEast },
  { distance := 15, direction := Direction.SouthEast },
  { distance := 25, direction := Direction.SouthWest },
  { distance := 35, direction := Direction.NorthWest },
  { distance := 20, direction := Direction.NorthEast }
]

-- Function to calculate the final distance
def finalDistance (j : List Leg) : ℝ := sorry

-- Theorem stating that the final distance is 20 miles
theorem journey_final_distance : finalDistance journey = 20 := by sorry

end journey_final_distance_l3023_302372


namespace sphere_box_height_l3023_302356

/-- A rectangular box with a large sphere and eight smaller spheres -/
structure SphereBox where
  h : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  box_width : ℝ
  box_length : ℝ
  num_small_spheres : ℕ

/-- The configuration of spheres in the box satisfies the given conditions -/
def valid_configuration (box : SphereBox) : Prop :=
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1.5 ∧
  box.box_width = 6 ∧
  box.box_length = 6 ∧
  box.num_small_spheres = 8 ∧
  ∀ (small_sphere : Fin box.num_small_spheres),
    (∃ (side1 side2 side3 : ℝ), 
      side1 + side2 + side3 = box.box_width + box.box_length + box.h) ∧
    (box.large_sphere_radius + box.small_sphere_radius = 
      box.box_width / 2 - box.small_sphere_radius)

/-- The height of the box is 9 given the valid configuration -/
theorem sphere_box_height (box : SphereBox) :
  valid_configuration box → box.h = 9 :=
by
  sorry

end sphere_box_height_l3023_302356


namespace lara_today_cans_l3023_302396

/-- The number of cans collected by Sarah and Lara over two days -/
structure CanCollection where
  sarah_yesterday : ℕ
  lara_yesterday : ℕ
  sarah_today : ℕ
  lara_today : ℕ

/-- The conditions of the can collection problem -/
def can_collection_problem (c : CanCollection) : Prop :=
  c.sarah_yesterday = 50 ∧
  c.lara_yesterday = c.sarah_yesterday + 30 ∧
  c.sarah_today = 40 ∧
  c.sarah_today + c.lara_today = c.sarah_yesterday + c.lara_yesterday - 20

/-- The theorem stating Lara collected 70 cans today -/
theorem lara_today_cans (c : CanCollection) :
  can_collection_problem c → c.lara_today = 70 := by
  sorry

end lara_today_cans_l3023_302396


namespace square_difference_identity_l3023_302387

theorem square_difference_identity : 287 * 287 + 269 * 269 - 2 * 287 * 269 = 324 := by
  sorry

end square_difference_identity_l3023_302387


namespace max_gold_coins_l3023_302301

theorem max_gold_coins (n : ℕ) : n < 150 ∧ ∃ k : ℕ, n = 13 * k + 3 → n ≤ 146 :=
by
  sorry

end max_gold_coins_l3023_302301


namespace new_person_age_l3023_302344

/-- Given a group of 10 people, prove that if replacing a 44-year-old person
    with a new person decreases the average age by 3 years, then the age of
    the new person is 14 years. -/
theorem new_person_age (group_size : ℕ) (old_person_age : ℕ) (avg_decrease : ℕ) :
  group_size = 10 →
  old_person_age = 44 →
  avg_decrease = 3 →
  ∃ (new_person_age : ℕ),
    (group_size * (avg_decrease + new_person_age) : ℤ) = old_person_age - new_person_age ∧
    new_person_age = 14 :=
by
  sorry

end new_person_age_l3023_302344


namespace sum_and_equality_condition_l3023_302321

/-- Given three real numbers x, y, and z satisfying the conditions:
    1. x + y + z = 150
    2. (x + 10) = (y - 10) = 3z
    Prove that x = 380/7 -/
theorem sum_and_equality_condition (x y z : ℝ) 
  (sum_eq : x + y + z = 150)
  (equality_cond : (x + 10) = (y - 10) ∧ (x + 10) = 3*z) :
  x = 380/7 := by
  sorry

end sum_and_equality_condition_l3023_302321


namespace function_upper_bound_l3023_302392

theorem function_upper_bound
  (f : ℝ → ℝ)
  (h1 : ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2))
  (h2 : ∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f x| ≤ M)
  : ∀ (x : ℝ), x ≥ 0 → f x ≤ x^2 :=
by sorry

end function_upper_bound_l3023_302392


namespace trapezoid_gh_length_l3023_302363

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  side_ef : ℝ
  side_gh : ℝ

/-- The area of a trapezoid is equal to the average of its parallel sides multiplied by its altitude -/
axiom trapezoid_area (t : Trapezoid) : t.area = (t.side_ef + t.side_gh) / 2 * t.altitude

theorem trapezoid_gh_length (t : Trapezoid) 
    (h_area : t.area = 250)
    (h_altitude : t.altitude = 10)
    (h_ef : t.side_ef = 15) :
    t.side_gh = 35 := by
  sorry

end trapezoid_gh_length_l3023_302363


namespace football_practice_missed_days_l3023_302373

/-- Calculates the number of days a football team missed practice due to rain. -/
theorem football_practice_missed_days
  (daily_hours : ℕ)
  (total_hours : ℕ)
  (days_in_week : ℕ)
  (h1 : daily_hours = 5)
  (h2 : total_hours = 30)
  (h3 : days_in_week = 7) :
  days_in_week - (total_hours / daily_hours) = 1 :=
by sorry

end football_practice_missed_days_l3023_302373


namespace cosine_inequality_solution_l3023_302385

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 (Real.pi / 2)) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.cos (x + y) ≥ Real.cos x + Real.cos y - 1) ↔ 
  y = 0 := by
sorry

end cosine_inequality_solution_l3023_302385


namespace even_odd_sum_difference_l3023_302339

def sumEven (n : ℕ) : ℕ := 
  (n / 2) * (2 + n)

def sumOdd (n : ℕ) : ℕ := 
  (n / 2) * (1 + (n - 1))

theorem even_odd_sum_difference : 
  sumEven 100 - sumOdd 100 = 50 := by
  sorry

end even_odd_sum_difference_l3023_302339


namespace bills_age_l3023_302383

/-- Bill's current age -/
def b : ℕ := 24

/-- Tracy's current age -/
def t : ℕ := 18

/-- Bill's age is one third larger than Tracy's age -/
axiom bill_tracy_relation : b = (4 * t) / 3

/-- In 30 years, Bill's age will be one eighth larger than Tracy's age -/
axiom future_relation : b + 30 = (9 * (t + 30)) / 8

/-- Theorem: Given the age relations between Bill and Tracy, Bill's current age is 24 -/
theorem bills_age : b = 24 := by sorry

end bills_age_l3023_302383


namespace earliest_year_500_mismatched_l3023_302352

/-- Number of shoe pairs in Moor's room in a given year -/
def shoe_pairs (year : ℕ) : ℕ := 2^(year - 2013)

/-- Number of mismatched shoe pairs possible with a given number of shoe pairs -/
def mismatched_pairs (pairs : ℕ) : ℕ := pairs * (pairs - 1)

/-- Predicate for whether a year allows at least 500 mismatched pairs -/
def can_wear_500_mismatched (year : ℕ) : Prop :=
  mismatched_pairs (shoe_pairs year) ≥ 500

theorem earliest_year_500_mismatched :
  (∀ y < 2018, ¬ can_wear_500_mismatched y) ∧ can_wear_500_mismatched 2018 := by
  sorry

end earliest_year_500_mismatched_l3023_302352


namespace rayden_vs_lily_birds_l3023_302398

theorem rayden_vs_lily_birds (lily_ducks lily_geese lily_chickens lily_pigeons : ℕ)
  (rayden_ducks rayden_geese rayden_chickens rayden_pigeons : ℕ)
  (h1 : lily_ducks = 20)
  (h2 : lily_geese = 10)
  (h3 : lily_chickens = 5)
  (h4 : lily_pigeons = 30)
  (h5 : rayden_ducks = 3 * lily_ducks)
  (h6 : rayden_geese = 4 * lily_geese)
  (h7 : rayden_chickens = 5 * lily_chickens)
  (h8 : lily_pigeons = 2 * rayden_pigeons) :
  (rayden_ducks + rayden_geese + rayden_chickens + rayden_pigeons) -
  (lily_ducks + lily_geese + lily_chickens + lily_pigeons) = 75 :=
by sorry

end rayden_vs_lily_birds_l3023_302398


namespace sufficient_but_not_necessary_l3023_302322

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the subset relation for a line being contained in a plane
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicularLines : Line → Line → Prop)

theorem sufficient_but_not_necessary 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : subset m β) :
  (∃ (h : parallel α β), ∀ (l m : Line), perpendicular l α → subset m β → perpendicularLines l m) ∧
  (∃ (l m : Line) (α β : Plane), perpendicular l α ∧ subset m β ∧ perpendicularLines l m ∧ ¬parallel α β) :=
sorry

end sufficient_but_not_necessary_l3023_302322


namespace william_journey_time_l3023_302393

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_minutes : minutes < 60

/-- Calculates the difference between two times in hours -/
def timeDifferenceInHours (t1 t2 : Time) : ℚ :=
  (t2.hours - t1.hours : ℚ) + (t2.minutes - t1.minutes : ℚ) / 60

/-- Represents a journey with stops and delays -/
structure Journey where
  departureTime : Time
  arrivalTime : Time
  timeZoneDifference : ℕ
  stops : List ℕ
  trafficDelay : ℕ

theorem william_journey_time (j : Journey) 
  (h1 : j.departureTime = ⟨7, 0, by norm_num⟩)
  (h2 : j.arrivalTime = ⟨20, 0, by norm_num⟩)
  (h3 : j.timeZoneDifference = 2)
  (h4 : j.stops = [25, 10, 25])
  (h5 : j.trafficDelay = 45) :
  timeDifferenceInHours j.departureTime ⟨18, 0, by norm_num⟩ + 
  (j.stops.sum / 60 : ℚ) + (j.trafficDelay / 60 : ℚ) = 12.75 := by
  sorry

#check william_journey_time

end william_journey_time_l3023_302393


namespace no_common_solutions_l3023_302315

theorem no_common_solutions : 
  ¬∃ x : ℝ, (|x - 10| = |x + 3| ∧ 2 * x + 6 = 18) := by
  sorry

end no_common_solutions_l3023_302315


namespace probability_one_doctor_one_nurse_l3023_302394

/-- The probability of selecting exactly 1 doctor and 1 nurse from a group of 3 doctors and 2 nurses, when choosing 2 people randomly. -/
theorem probability_one_doctor_one_nurse :
  let total_people : ℕ := 5
  let doctors : ℕ := 3
  let nurses : ℕ := 2
  let selection : ℕ := 2
  Nat.choose total_people selection ≠ 0 →
  (Nat.choose doctors 1 * Nat.choose nurses 1 : ℚ) / Nat.choose total_people selection = 3/5 :=
by sorry

end probability_one_doctor_one_nurse_l3023_302394


namespace tenth_even_term_is_92_l3023_302331

def arithmetic_sequence (n : ℕ) : ℤ := 2 + (n - 1) * 5

def is_even (z : ℤ) : Prop := ∃ k : ℤ, z = 2 * k

def nth_even_term (n : ℕ) : ℕ := 2 * n - 1

theorem tenth_even_term_is_92 :
  arithmetic_sequence (nth_even_term 10) = 92 :=
sorry

end tenth_even_term_is_92_l3023_302331


namespace girls_boys_difference_l3023_302362

theorem girls_boys_difference (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 36 → 
  5 * boys = 4 * girls → 
  total = girls + boys → 
  girls - boys = 4 := by
sorry

end girls_boys_difference_l3023_302362


namespace violinists_count_l3023_302314

/-- Represents the number of people playing each instrument in an orchestra -/
structure Orchestra where
  total : ℕ
  drums : ℕ
  trombone : ℕ
  trumpet : ℕ
  frenchHorn : ℕ
  cello : ℕ
  contrabass : ℕ
  clarinet : ℕ
  flute : ℕ
  maestro : ℕ

/-- Calculates the number of violinists in the orchestra -/
def violinists (o : Orchestra) : ℕ :=
  o.total - (o.drums + o.trombone + o.trumpet + o.frenchHorn + o.cello + o.contrabass + o.clarinet + o.flute + o.maestro)

/-- Theorem stating that the number of violinists in the given orchestra is 3 -/
theorem violinists_count (o : Orchestra) 
  (h1 : o.total = 21)
  (h2 : o.drums = 1)
  (h3 : o.trombone = 4)
  (h4 : o.trumpet = 2)
  (h5 : o.frenchHorn = 1)
  (h6 : o.cello = 1)
  (h7 : o.contrabass = 1)
  (h8 : o.clarinet = 3)
  (h9 : o.flute = 4)
  (h10 : o.maestro = 1) :
  violinists o = 3 := by
  sorry


end violinists_count_l3023_302314


namespace equal_volume_equivalent_by_decomposition_l3023_302307

/-- A type representing geometric shapes (either rectangular parallelepipeds or prisms) -/
structure GeometricShape where
  volume : ℝ

/-- A type representing a decomposition of a geometric shape -/
structure Decomposition (α : Type) where
  parts : List α

/-- A function that checks if two decompositions are equivalent -/
def equivalent_decompositions {α : Type} (d1 d2 : Decomposition α) : Prop :=
  sorry

/-- A function that transforms one shape into another using a decomposition -/
def transform (s1 s2 : GeometricShape) (d : Decomposition GeometricShape) : Prop :=
  sorry

/-- The main theorem stating that equal-volume shapes are equivalent by decomposition -/
theorem equal_volume_equivalent_by_decomposition (s1 s2 : GeometricShape) :
  s1.volume = s2.volume →
  ∃ (d : Decomposition GeometricShape), transform s1 s2 d ∧ transform s2 s1 d :=
sorry

end equal_volume_equivalent_by_decomposition_l3023_302307


namespace lindsey_remaining_money_l3023_302342

/-- Calculates Lindsey's remaining money after saving and spending --/
theorem lindsey_remaining_money 
  (september_savings : ℕ) 
  (october_savings : ℕ) 
  (november_savings : ℕ) 
  (mom_bonus_threshold : ℕ) 
  (mom_bonus : ℕ) 
  (video_game_cost : ℕ) : 
  (september_savings = 50 ∧ 
   october_savings = 37 ∧ 
   november_savings = 11 ∧ 
   mom_bonus_threshold = 75 ∧ 
   mom_bonus = 25 ∧ 
   video_game_cost = 87) → 
  (let total_savings := september_savings + october_savings + november_savings
   let total_with_bonus := total_savings + (if total_savings > mom_bonus_threshold then mom_bonus else 0)
   let remaining_money := total_with_bonus - video_game_cost
   remaining_money = 36) := by
sorry

end lindsey_remaining_money_l3023_302342


namespace expected_total_audience_l3023_302341

theorem expected_total_audience (saturday_attendance : ℕ) 
  (monday_attendance : ℕ) (wednesday_attendance : ℕ) (friday_attendance : ℕ) 
  (actual_total : ℕ) (expected_total : ℕ) : 
  saturday_attendance = 80 →
  monday_attendance = saturday_attendance - 20 →
  wednesday_attendance = monday_attendance + 50 →
  friday_attendance = saturday_attendance + monday_attendance →
  actual_total = saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance →
  actual_total = expected_total + 40 →
  expected_total = 350 := by
sorry

end expected_total_audience_l3023_302341


namespace original_triangle_area_l3023_302334

/-- Given a triangle whose dimensions are quintupled to form a new triangle with an area of 100 square feet,
    the area of the original triangle is 4 square feet. -/
theorem original_triangle_area (original : ℝ) (new : ℝ) : 
  new = 100 → 
  new = original * 25 → 
  original = 4 := by sorry

end original_triangle_area_l3023_302334


namespace six_fold_application_of_f_on_four_l3023_302381

noncomputable def f (x : ℝ) : ℝ := -1 / x

theorem six_fold_application_of_f_on_four (h : ∀ (x : ℝ), x ≠ 0 → f x = -1 / x) :
  f (f (f (f (f (f 4))))) = 4 :=
by sorry

end six_fold_application_of_f_on_four_l3023_302381


namespace certain_number_problem_l3023_302368

theorem certain_number_problem :
  ∃ x : ℝ, (1/10 : ℝ) * x - (1/1000 : ℝ) * x = 693 ∧ x = 7000 := by
  sorry

end certain_number_problem_l3023_302368


namespace oatmeal_cookies_given_away_l3023_302346

/-- Represents the number of cookies in a dozen. -/
def dozen : ℕ := 12

/-- Represents the total number of cookies Ann baked. -/
def totalBaked : ℕ := 3 * dozen + 2 * dozen + 4 * dozen

/-- Represents the number of sugar cookies Ann gave away. -/
def sugarGivenAway : ℕ := (3 * dozen) / 2

/-- Represents the number of chocolate chip cookies Ann gave away. -/
def chocolateGivenAway : ℕ := (5 * dozen) / 2

/-- Represents the number of cookies Ann kept. -/
def cookiesKept : ℕ := 36

/-- Proves that Ann gave away 2 dozen oatmeal raisin cookies. -/
theorem oatmeal_cookies_given_away :
  ∃ (x : ℕ), x * dozen + sugarGivenAway + chocolateGivenAway + cookiesKept = totalBaked ∧ x = 2 := by
  sorry

end oatmeal_cookies_given_away_l3023_302346


namespace perpendicular_vector_scalar_l3023_302305

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to (a + mb), then m = 5. -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (2, -1))
  (h2 : b = (1, 3))
  (h3 : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) :
  m = 5 := by
  sorry

#check perpendicular_vector_scalar

end perpendicular_vector_scalar_l3023_302305


namespace odd_product_remainder_l3023_302388

def odd_product : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then odd_product n else (2 * n + 1) * odd_product n

theorem odd_product_remainder :
  odd_product 1002 % 1000 = 875 :=
sorry

end odd_product_remainder_l3023_302388


namespace cos_sin_inequality_range_l3023_302304

theorem cos_sin_inequality_range (θ : Real) :
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  Real.cos θ ^ 5 - Real.sin θ ^ 5 < 7 * (Real.sin θ ^ 3 - Real.cos θ ^ 3) →
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) :=
by sorry

end cos_sin_inequality_range_l3023_302304


namespace geometric_sequence_special_case_l3023_302310

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → b (n + 1) - b n = d

theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, n ≥ 1 → a n > 0) →
  a 1 = 2 →
  arithmetic_sequence (λ n => match n with
    | 1 => 2 * a 1
    | 2 => a 3
    | 3 => 3 * a 2
    | _ => 0
  ) →
  ∀ n : ℕ, n ≥ 1 → a n = 2^n :=
by sorry

end geometric_sequence_special_case_l3023_302310


namespace triangle_side_c_l3023_302316

theorem triangle_side_c (a b c : ℝ) (S : ℝ) (B : ℝ) :
  B = π / 4 →  -- 45° in radians
  a = 4 →
  S = 16 * Real.sqrt 2 →
  S = 1 / 2 * a * c * Real.sin B →
  c = 16 :=
by sorry

end triangle_side_c_l3023_302316


namespace prob_objects_meet_l3023_302340

/-- The number of steps required for objects to meet -/
def stepsToMeet : ℕ := 9

/-- The possible x-coordinates of meeting points -/
def meetingPoints : List ℕ := [0, 2, 4, 6, 8]

/-- Calculate binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Calculate the number of paths to a meeting point for each object -/
def pathsToPoint (i : ℕ) : ℕ × ℕ :=
  (binomial stepsToMeet i, binomial stepsToMeet (i + 1))

/-- Calculate the probability of meeting at a specific point -/
def probMeetAtPoint (i : ℕ) : ℚ :=
  let (a, b) := pathsToPoint i
  (a * b : ℚ) / (2^(2 * stepsToMeet) : ℚ)

/-- The main theorem: probability of objects meeting -/
theorem prob_objects_meet :
  (meetingPoints.map probMeetAtPoint).sum = 207 / 262144 := by sorry

end prob_objects_meet_l3023_302340


namespace functional_equation_solution_l3023_302313

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * (1 + y)) = f x * (1 + f y)

/-- The main theorem stating that any function satisfying the functional equation
    is either the identity function or the zero function -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 0) := by
  sorry

end functional_equation_solution_l3023_302313


namespace four_inch_cube_three_painted_faces_l3023_302367

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Represents a smaller cube resulting from cutting a larger cube -/
structure SmallCube where
  paintedFaces : ℕ

/-- The number of small cubes with at least three painted faces in a painted cube -/
def numCubesWithThreePaintedFaces (c : Cube) : ℕ :=
  8

/-- Theorem stating that a 4-inch cube cut into 1-inch cubes has 8 cubes with at least three painted faces -/
theorem four_inch_cube_three_painted_faces :
  ∀ (c : Cube), c.sideLength = 4 → numCubesWithThreePaintedFaces c = 8 := by
  sorry

end four_inch_cube_three_painted_faces_l3023_302367


namespace major_axis_length_is_13_l3023_302369

/-- A configuration of a cylinder and two spheres -/
structure CylinderSphereConfig where
  cylinder_radius : ℝ
  sphere_radius : ℝ
  sphere_centers_distance : ℝ

/-- The length of the major axis of the ellipse formed by the intersection of a plane 
    touching both spheres and the cylinder surface -/
def major_axis_length (config : CylinderSphereConfig) : ℝ :=
  config.sphere_centers_distance

/-- Theorem stating that for the given configuration, the major axis length is 13 -/
theorem major_axis_length_is_13 :
  let config := CylinderSphereConfig.mk 6 6 13
  major_axis_length config = 13 := by
  sorry

#eval major_axis_length (CylinderSphereConfig.mk 6 6 13)

end major_axis_length_is_13_l3023_302369


namespace sequence_inequality_l3023_302336

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a (n + 1) = 0)
  (h : ∀ k : ℕ, k ≥ 1 → k ≤ n → |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : ℕ, k ≤ n + 1 → |a k| ≤ k * (n + 1 - k) / 2 :=
by sorry

end sequence_inequality_l3023_302336


namespace relationship_abc_l3023_302306

theorem relationship_abc (a b c : ℝ) : 
  a = (1.01 : ℝ) ^ (0.5 : ℝ) →
  b = (1.01 : ℝ) ^ (0.6 : ℝ) →
  c = (0.6 : ℝ) ^ (0.5 : ℝ) →
  b > a ∧ a > c :=
by sorry

end relationship_abc_l3023_302306


namespace pentagon_rectangle_ratio_l3023_302302

/-- Given a regular pentagon and a rectangle with perimeters of 75 inches,
    where the rectangle's length is twice its width, prove that the ratio of
    the side length of the pentagon to the width of the rectangle is 6/5. -/
theorem pentagon_rectangle_ratio :
  ∀ (pentagon_side : ℚ) (rect_width : ℚ),
    -- Pentagon perimeter is 75 inches
    5 * pentagon_side = 75 →
    -- Rectangle perimeter is 75 inches, and length is twice the width
    2 * (rect_width + 2 * rect_width) = 75 →
    -- The ratio of pentagon side to rectangle width is 6/5
    pentagon_side / rect_width = 6 / 5 := by
  sorry

end pentagon_rectangle_ratio_l3023_302302


namespace largest_house_number_l3023_302376

def phone_number : List Nat := [4, 3, 1, 7, 8, 2]

def digit_sum (num : List Nat) : Nat :=
  num.sum

def is_distinct (num : List Nat) : Prop :=
  num.length = num.toFinset.card

theorem largest_house_number :
  ∃ (house : List Nat),
    house.length = 5 ∧
    is_distinct house ∧
    digit_sum house = digit_sum phone_number ∧
    (∀ other : List Nat,
      other.length = 5 →
      is_distinct other →
      digit_sum other = digit_sum phone_number →
      house.foldl (fun acc d => acc * 10 + d) 0 ≥
      other.foldl (fun acc d => acc * 10 + d) 0) ∧
    house = [9, 8, 7, 1, 0] :=
sorry

end largest_house_number_l3023_302376


namespace bird_sanctuary_theorem_l3023_302335

def bird_sanctuary_problem (initial_storks initial_herons initial_sparrows : ℕ)
  (storks_left herons_left sparrows_arrived hummingbirds_arrived : ℕ) : ℤ :=
  let final_storks : ℕ := initial_storks - storks_left
  let final_herons : ℕ := initial_herons - herons_left
  let final_sparrows : ℕ := initial_sparrows + sparrows_arrived
  let final_hummingbirds : ℕ := hummingbirds_arrived
  let total_other_birds : ℕ := final_herons + final_sparrows + final_hummingbirds
  (final_storks : ℤ) - (total_other_birds : ℤ)

theorem bird_sanctuary_theorem :
  bird_sanctuary_problem 8 4 5 3 2 4 2 = -8 := by
  sorry

end bird_sanctuary_theorem_l3023_302335


namespace fraction_of_25_comparison_l3023_302384

theorem fraction_of_25_comparison : ∃ x : ℚ, 
  (x * 25 = 80 / 100 * 60 - 28) ∧ 
  (x = 4 / 5) := by
sorry

end fraction_of_25_comparison_l3023_302384


namespace triangle_type_l3023_302317

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.B = Real.pi / 6 ∧  -- 30 degrees in radians
  t.c = 15 ∧
  t.b = 5 * Real.sqrt 3

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Define right triangle
def is_right (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

-- Theorem statement
theorem triangle_type (t : Triangle) :
  triangle_conditions t → (is_isosceles t ∨ is_right t) :=
sorry

end triangle_type_l3023_302317


namespace inequalities_hold_l3023_302328

theorem inequalities_hold (a b : ℝ) (h : a * b > 0) :
  (2 * (a^2 + b^2) ≥ (a + b)^2) ∧
  (b / a + a / b ≥ 2) ∧
  ((a + 1 / a) * (b + 1 / b) ≥ 4) := by
  sorry

end inequalities_hold_l3023_302328


namespace compute_expression_l3023_302312

theorem compute_expression : 9 * (2/3)^4 = 16/9 := by
  sorry

end compute_expression_l3023_302312


namespace inequality_proof_l3023_302347

theorem inequality_proof (x : ℝ) (n : ℕ) (h : x > 0) :
  x + n^n / x^n ≥ n + 1 := by
  sorry

end inequality_proof_l3023_302347


namespace fraction_zero_implies_x_negative_two_l3023_302351

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  ((x - 1) * (x + 2)) / (x^2 - 1) = 0 → x = -2 :=
by sorry

end fraction_zero_implies_x_negative_two_l3023_302351


namespace melanie_plums_count_l3023_302377

/-- The number of plums Melanie picked initially -/
def melanie_picked : ℝ := 7.0

/-- The number of plums Sam gave to Melanie -/
def sam_gave : ℝ := 3.0

/-- The total number of plums Melanie has now -/
def total_plums : ℝ := melanie_picked + sam_gave

theorem melanie_plums_count : total_plums = 10.0 := by
  sorry

end melanie_plums_count_l3023_302377


namespace sqrt_x_plus_inverse_l3023_302324

theorem sqrt_x_plus_inverse (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_x_plus_inverse_l3023_302324


namespace f_inequality_l3023_302379

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_increasing : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y
axiom f_even_shifted : ∀ x, f (x + 2) = f (-x + 2)

-- State the theorem to be proved
theorem f_inequality : f (5/2) > f 1 ∧ f 1 > f (7/2) := by
  sorry

end f_inequality_l3023_302379


namespace problem_solution_l3023_302318

/-- The function g(x) = -|x+m| -/
def g (m : ℝ) (x : ℝ) : ℝ := -|x + m|

/-- The function f(x) = 2|x-1| - a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*|x - 1| - a

theorem problem_solution :
  (∃! (n : ℤ), g m n > -1) ∧ (∀ (x : ℤ), g m x > -1 → x = -3) →
  m = 3 ∧
  (∀ x, f a x > g 3 x) →
  a < 4 :=
sorry

end problem_solution_l3023_302318


namespace maximize_sum_with_constraint_l3023_302378

theorem maximize_sum_with_constraint (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_constraint : 9 * a^2 + 4 * b^2 + c^2 = 91) :
  a + 2*b + 3*c ≤ 91/3 :=
by sorry

end maximize_sum_with_constraint_l3023_302378


namespace locus_of_concyclic_points_l3023_302366

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the special points of the triangle
def H : ℝ × ℝ := sorry  -- Orthocenter
def I : ℝ × ℝ := sorry  -- Incenter
def G : ℝ × ℝ := sorry  -- Centroid

-- Define points E and F that divide AB into three equal parts
def E : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- Define the angle at vertex C
def angle_C : ℝ := sorry

-- Define a predicate for points being concyclic
def are_concyclic (p q r s : ℝ × ℝ) : Prop := sorry

-- Define a predicate for a point being on a circular arc
def on_circular_arc (p center : ℝ × ℝ) (start_point end_point : ℝ × ℝ) (arc_angle : ℝ) : Prop := sorry

theorem locus_of_concyclic_points :
  are_concyclic A B H I →
  (angle_C = 60) ∧ 
  (∃ (center : ℝ × ℝ), on_circular_arc G center E F 120) :=
sorry

end locus_of_concyclic_points_l3023_302366


namespace wax_remaining_l3023_302327

/-- The amount of wax remaining after detailing vehicles -/
def remaining_wax (initial : ℕ) (spilled : ℕ) (car : ℕ) (suv : ℕ) : ℕ :=
  initial - spilled - car - suv

/-- Theorem stating the remaining wax after detailing vehicles -/
theorem wax_remaining :
  remaining_wax 11 2 3 4 = 2 := by
  sorry

end wax_remaining_l3023_302327


namespace surrounding_circles_radius_l3023_302399

theorem surrounding_circles_radius (r : ℝ) : r = 2 * (Real.sqrt 2 + 1) :=
  let central_radius := 2
  let square_side := 2 * r
  let square_diagonal := square_side * Real.sqrt 2
  let total_diagonal := 2 * central_radius + 2 * r
by
  sorry

end surrounding_circles_radius_l3023_302399


namespace base_subtraction_l3023_302389

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The theorem statement --/
theorem base_subtraction :
  to_base_10 [3, 2, 5] 9 - to_base_10 [2, 3, 1] 6 = 175 := by
  sorry

end base_subtraction_l3023_302389


namespace evaluate_expression_l3023_302323

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^(y-1) + 2 * y^(x+1) = 647 := by
  sorry

end evaluate_expression_l3023_302323


namespace smaller_number_is_35_l3023_302326

theorem smaller_number_is_35 (x y : ℝ) : 
  x + y = 77 ∧ 
  (x = 42 ∨ y = 42) ∧ 
  (5 * x = 6 * y ∨ 5 * y = 6 * x) →
  min x y = 35 := by
sorry

end smaller_number_is_35_l3023_302326


namespace min_distance_to_line_l3023_302391

-- Define the right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ c > a ∧ c > b ∧ a^2 + b^2 = c^2

-- Define the point (m, n) on the line ax + by + c = 0
def point_on_line (a b c m n : ℝ) : Prop :=
  a * m + b * n + c = 0

-- Theorem statement
theorem min_distance_to_line (a b c m n : ℝ) 
  (h1 : is_right_triangle a b c) 
  (h2 : point_on_line a b c m n) : 
  m^2 + n^2 ≥ 1 :=
sorry

end min_distance_to_line_l3023_302391


namespace tuesday_kids_l3023_302300

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 24

/-- The difference in the number of kids Julia played with between Monday and Tuesday -/
def difference : ℕ := 18

/-- Theorem: The number of kids Julia played with on Tuesday is 6 -/
theorem tuesday_kids : monday_kids - difference = 6 := by
  sorry

end tuesday_kids_l3023_302300


namespace point_outside_circle_l3023_302370

/-- The line ax + by = 1 intersects with the circle x^2 + y^2 = 1 -/
def line_intersects_circle (a b : ℝ) : Prop :=
  ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1

theorem point_outside_circle (a b : ℝ) :
  line_intersects_circle a b → a^2 + b^2 > 1 := by
  sorry

end point_outside_circle_l3023_302370


namespace nissan_cars_sold_l3023_302345

theorem nissan_cars_sold (total_cars : ℕ) (audi_percent : ℚ) (toyota_percent : ℚ) (acura_percent : ℚ) (bmw_percent : ℚ) 
  (h1 : total_cars = 250)
  (h2 : audi_percent = 10 / 100)
  (h3 : toyota_percent = 25 / 100)
  (h4 : acura_percent = 15 / 100)
  (h5 : bmw_percent = 18 / 100)
  : ℕ :=
by
  sorry

#check nissan_cars_sold

end nissan_cars_sold_l3023_302345


namespace sphere_volume_ratio_l3023_302380

/-- Given a sphere O with radius R and a plane perpendicular to a radius OP at its midpoint M,
    intersecting the sphere to form a circle O₁, the volume ratio of the sphere with O₁ as its
    great circle to sphere O is 3/8 * √3. -/
theorem sphere_volume_ratio (R : ℝ) (h : R > 0) : 
  let r := R * (Real.sqrt 3 / 2)
  (4 / 3 * Real.pi * r^3) / (4 / 3 * Real.pi * R^3) = 3 / 8 * Real.sqrt 3 := by
  sorry

end sphere_volume_ratio_l3023_302380


namespace trains_passing_time_l3023_302311

/-- Given two trains with specified characteristics, prove that they will completely pass each other in 11 seconds. -/
theorem trains_passing_time (tunnel_length : ℝ) (tunnel_time : ℝ) (bridge_length : ℝ) (bridge_time : ℝ)
  (freight_train_length : ℝ) (freight_train_speed : ℝ) :
  tunnel_length = 285 →
  tunnel_time = 24 →
  bridge_length = 245 →
  bridge_time = 22 →
  freight_train_length = 135 →
  freight_train_speed = 10 →
  ∃ (train_speed : ℝ) (train_length : ℝ),
    train_speed = (tunnel_length - bridge_length) / (tunnel_time - bridge_time) ∧
    train_length = train_speed * tunnel_time - tunnel_length ∧
    (train_length + freight_train_length) / (train_speed + freight_train_speed) = 11 :=
by sorry

end trains_passing_time_l3023_302311


namespace euclidean_division_l3023_302359

theorem euclidean_division (a b : ℕ) (hb : b > 0) :
  ∃ q r : ℤ, 0 ≤ r ∧ r < b ∧ (a : ℤ) = b * q + r :=
sorry

end euclidean_division_l3023_302359


namespace min_omega_for_cosine_function_l3023_302355

theorem min_omega_for_cosine_function (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.cos (ω * x - π / 6)) →
  (ω > 0) →
  (∀ x, f x ≤ f (π / 4)) →
  (∀ ω' > 0, (∀ x, Real.cos (ω' * x - π / 6) ≤ Real.cos (ω' * π / 4 - π / 6)) → ω' ≥ 2 / 3) →
  ω = 2 / 3 := by
sorry

end min_omega_for_cosine_function_l3023_302355


namespace smallest_set_with_both_progressions_l3023_302343

/-- A sequence of integers forms a geometric progression of length 5 -/
def IsGeometricProgression (s : Finset ℤ) : Prop :=
  ∃ (a q : ℤ), q ≠ 0 ∧ s = {a, a*q, a*q^2, a*q^3, a*q^4}

/-- A sequence of integers forms an arithmetic progression of length 5 -/
def IsArithmeticProgression (s : Finset ℤ) : Prop :=
  ∃ (a d : ℤ), s = {a, a+d, a+2*d, a+3*d, a+4*d}

/-- The main theorem stating the smallest number of distinct integers -/
theorem smallest_set_with_both_progressions :
  ∀ (s : Finset ℤ), (∃ (s1 s2 : Finset ℤ), s1 ⊆ s ∧ s2 ⊆ s ∧ 
    IsGeometricProgression s1 ∧ IsArithmeticProgression s2) →
  s.card ≥ 6 :=
sorry

end smallest_set_with_both_progressions_l3023_302343


namespace third_row_sum_is_226_l3023_302354

/-- Represents a position in the grid -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the spiral grid -/
def SpiralGrid :=
  Position → Nat

/-- The size of the grid -/
def gridSize : Nat := 13

/-- The starting number -/
def startNum : Nat := 100

/-- The ending number -/
def endNum : Nat := 268

/-- The center position of the grid -/
def centerPos : Position :=
  { row := 6, col := 6 }  -- 0-based index

/-- Generates the spiral grid -/
def generateSpiralGrid : SpiralGrid :=
  sorry

/-- Gets the numbers in the third row -/
def getThirdRowNumbers (grid : SpiralGrid) : List Nat :=
  sorry

/-- Theorem: The sum of the greatest and least numbers in the third row is 226 -/
theorem third_row_sum_is_226 (grid : SpiralGrid) :
  grid = generateSpiralGrid →
  let thirdRowNums := getThirdRowNumbers grid
  (List.maximum thirdRowNums).getD 0 + (List.minimum thirdRowNums).getD 0 = 226 :=
sorry

end third_row_sum_is_226_l3023_302354


namespace sequence_is_arithmetic_l3023_302353

theorem sequence_is_arithmetic (a : ℕ+ → ℝ)
  (h : ∀ p q : ℕ+, a p = a q + 2003 * (p - q)) :
  ∃ d : ℝ, ∀ n m : ℕ+, a n = a m + d * (n - m) := by
  sorry

end sequence_is_arithmetic_l3023_302353


namespace side_x_must_be_green_l3023_302397

-- Define the possible colors
inductive Color
  | Red
  | Green
  | Blue

-- Define a triangle with three sides
structure Triangle where
  side1 : Color
  side2 : Color
  side3 : Color

-- Define the condition that each triangle must have one of each color
def validTriangle (t : Triangle) : Prop :=
  t.side1 ≠ t.side2 ∧ t.side2 ≠ t.side3 ∧ t.side1 ≠ t.side3

-- Define the configuration of five triangles
structure Configuration where
  t1 : Triangle
  t2 : Triangle
  t3 : Triangle
  t4 : Triangle
  t5 : Triangle

-- Define the given colored sides
def givenColoring (c : Configuration) : Prop :=
  c.t1.side1 = Color.Green ∧
  c.t2.side1 = Color.Blue ∧
  c.t3.side3 = Color.Green ∧
  c.t5.side2 = Color.Blue

-- Define the shared sides
def sharedSides (c : Configuration) : Prop :=
  c.t1.side2 = c.t2.side3 ∧
  c.t1.side3 = c.t3.side1 ∧
  c.t2.side2 = c.t3.side2 ∧
  c.t3.side3 = c.t4.side1 ∧
  c.t4.side2 = c.t5.side1 ∧
  c.t4.side3 = c.t5.side3

-- Theorem statement
theorem side_x_must_be_green (c : Configuration) 
  (h1 : givenColoring c)
  (h2 : sharedSides c)
  (h3 : ∀ t, t ∈ [c.t1, c.t2, c.t3, c.t4, c.t5] → validTriangle t) :
  c.t4.side3 = Color.Green :=
sorry

end side_x_must_be_green_l3023_302397
