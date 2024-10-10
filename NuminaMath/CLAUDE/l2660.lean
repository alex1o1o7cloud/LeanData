import Mathlib

namespace line_tangent_to_parabola_l2660_266054

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, 4*x + 6*y + k = 0 → y^2 = 32*x) ↔ k = 72 := by
  sorry

end line_tangent_to_parabola_l2660_266054


namespace alices_total_distance_l2660_266082

/-- Alice's weekly walking distance to school and back home --/
def alices_weekly_walking_distance (days_per_week : ℕ) (distance_to_school : ℕ) (distance_from_school : ℕ) : ℕ :=
  (days_per_week * distance_to_school) + (days_per_week * distance_from_school)

/-- Theorem: Alice walks 110 miles in a week --/
theorem alices_total_distance :
  alices_weekly_walking_distance 5 10 12 = 110 := by
  sorry

end alices_total_distance_l2660_266082


namespace inequality_proof_l2660_266097

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end inequality_proof_l2660_266097


namespace sqrt_abs_sum_zero_implies_power_sum_zero_l2660_266089

theorem sqrt_abs_sum_zero_implies_power_sum_zero (a b : ℝ) :
  Real.sqrt (a + 1) + |b - 1| = 0 → a^2023 + b^2024 = 0 := by
  sorry

end sqrt_abs_sum_zero_implies_power_sum_zero_l2660_266089


namespace rachel_plant_arrangement_l2660_266011

/-- Represents the number of ways to arrange plants under lamps -/
def arrangement_count (cactus_count : ℕ) (orchid_count : ℕ) (yellow_lamp_count : ℕ) (blue_lamp_count : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating the number of arrangements for the given problem -/
theorem rachel_plant_arrangement :
  arrangement_count 3 1 3 2 = 13 := by
  sorry

end rachel_plant_arrangement_l2660_266011


namespace line_ellipse_intersection_l2660_266084

/-- Given a line and a circle with no common points, prove that a line through a point
    on that line intersects a specific ellipse at exactly two points. -/
theorem line_ellipse_intersection (m n : ℝ) : 
  (∀ x y : ℝ, m*x + n*y - 3 = 0 → x^2 + y^2 ≠ 3) →
  0 < m^2 + n^2 →
  m^2 + n^2 < 3 →
  ∃! (p : ℕ), p = 2 ∧ 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
      (∃ (k : ℝ), x₁ = m*k ∧ y₁ = n*k) ∧
      (∃ (k : ℝ), x₂ = m*k ∧ y₂ = n*k) ∧
      x₁^2/7 + y₁^2/3 = 1 ∧
      x₂^2/7 + y₂^2/3 = 1 ∧
      (∀ x y : ℝ, (∃ k : ℝ, x = m*k ∧ y = n*k) → 
        x^2/7 + y^2/3 = 1 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end line_ellipse_intersection_l2660_266084


namespace solve_rock_problem_l2660_266096

def rock_problem (joshua_rocks : ℕ) : Prop :=
  let jose_rocks := joshua_rocks - 14
  let albert_rocks := jose_rocks + 28
  let clara_rocks := jose_rocks / 2
  let maria_rocks := clara_rocks + 18
  albert_rocks - joshua_rocks = 14

theorem solve_rock_problem :
  rock_problem 80 := by sorry

end solve_rock_problem_l2660_266096


namespace property_P_for_given_numbers_l2660_266076

-- Define property P
def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3*x*y*z

-- Theorem statement
theorem property_P_for_given_numbers :
  (has_property_P 1) ∧
  (has_property_P 5) ∧
  (has_property_P 2014) ∧
  (¬ has_property_P 2013) :=
by sorry

end property_P_for_given_numbers_l2660_266076


namespace arithmetic_sequence_sum_11_l2660_266061

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
structure ArithmeticSequence (α : Type*) [AddCommGroup α] where
  a : ℕ → α
  d : α
  h : ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_11 (a : ArithmeticSequence ℝ) 
  (h : a.a 4 + a.a 8 = 16) : 
  (Finset.range 11).sum a.a = 88 := by
  sorry

end arithmetic_sequence_sum_11_l2660_266061


namespace longest_segment_in_cylinder_l2660_266030

/-- The longest segment in a cylinder. -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 3) (hh : h = 8) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 := by
  sorry

end longest_segment_in_cylinder_l2660_266030


namespace smallest_n_multiple_of_eight_l2660_266044

theorem smallest_n_multiple_of_eight (x y : ℤ) 
  (h1 : ∃ k : ℤ, x + 2 = 8 * k) 
  (h2 : ∃ m : ℤ, y - 2 = 8 * m) : 
  (∀ n : ℕ, n > 0 → n < 4 → ¬(∃ p : ℤ, x^2 - x*y + y^2 + n = 8 * p)) ∧ 
  (∃ q : ℤ, x^2 - x*y + y^2 + 4 = 8 * q) :=
sorry

end smallest_n_multiple_of_eight_l2660_266044


namespace vector_collinearity_l2660_266040

/-- Two vectors in R² -/
def PA : Fin 2 → ℝ := ![(-1 : ℝ), 2]
def PB (x : ℝ) : Fin 2 → ℝ := ![2, x]

/-- Collinearity condition for three points in R² -/
def collinear (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 - v 1 * w 0 = 0

theorem vector_collinearity (x : ℝ) : 
  collinear PA (PB x) → x = -4 := by sorry

end vector_collinearity_l2660_266040


namespace garden_furniture_cost_l2660_266069

def bench_cost : ℝ := 150

def table_cost (bench_cost : ℝ) : ℝ := 2 * bench_cost

def combined_cost (bench_cost table_cost : ℝ) : ℝ := bench_cost + table_cost

theorem garden_furniture_cost : combined_cost bench_cost (table_cost bench_cost) = 450 := by
  sorry

end garden_furniture_cost_l2660_266069


namespace linden_birch_problem_l2660_266081

theorem linden_birch_problem :
  ∃ (x y : ℕ), 
    x + y > 14 ∧ 
    y + 18 > 2 * x ∧ 
    x > 2 * y ∧ 
    x = 11 ∧ 
    y = 5 := by
  sorry

end linden_birch_problem_l2660_266081


namespace problem_statement_l2660_266060

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 7)
  (h_b_a : b + 1 / a = 16) :
  c + 1 / b = 25 / 111 := by
sorry

end problem_statement_l2660_266060


namespace divisibility_of_sum_l2660_266016

theorem divisibility_of_sum (a b c d x : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (x^2 - (a+b)*x + a*b) * (x^2 - (c+d)*x + c*d) = 9 →
  ∃ k : ℤ, a + b + c + d = 4 * k :=
by sorry

end divisibility_of_sum_l2660_266016


namespace cone_volume_from_cylinder_l2660_266023

theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let cylinder_volume := π * r^2 * h
  let cone_volume := (1/3) * π * r^2 * h
  cylinder_volume = 72 * π → cone_volume = 24 * π := by
  sorry

end cone_volume_from_cylinder_l2660_266023


namespace larger_number_proof_l2660_266019

theorem larger_number_proof (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 23)
  (lcm_eq : Nat.lcm a b = 23 * 14 * 15) :
  max a b = 345 := by
  sorry

end larger_number_proof_l2660_266019


namespace symmetric_points_sum_l2660_266003

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- Given that point A(a,1) and point B(5,b) are symmetric with respect to the origin O, prove that a + b = -6 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin a 1 5 b) : a + b = -6 := by
  sorry

end symmetric_points_sum_l2660_266003


namespace class_average_mark_l2660_266070

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_avg : ℚ) (remaining_avg : ℚ) :
  total_students = 33 →
  excluded_students = 3 →
  excluded_avg = 40 →
  remaining_avg = 95 →
  (total_students * (total_students - excluded_students) * remaining_avg +
   total_students * excluded_students * excluded_avg) /
  (total_students * total_students) = 90 := by
  sorry

end class_average_mark_l2660_266070


namespace expression_value_l2660_266009

theorem expression_value (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 3*x^2 + 9*x - 2 = 4 := by
  sorry

end expression_value_l2660_266009


namespace no_odd_sided_cross_section_polyhedron_l2660_266002

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  convex : Bool

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields here

/-- A polygon -/
structure Polygon where
  sides : ℕ

/-- Represents a cross-section of a polyhedron with a plane -/
def cross_section (p : ConvexPolyhedron) (plane : Plane) : Polygon :=
  sorry

/-- Predicate to check if a plane passes through a vertex of the polyhedron -/
def passes_through_vertex (p : ConvexPolyhedron) (plane : Plane) : Prop :=
  sorry

/-- Main theorem: No such convex polyhedron exists -/
theorem no_odd_sided_cross_section_polyhedron :
  ¬ ∃ (p : ConvexPolyhedron),
    (∀ (plane : Plane),
      ¬passes_through_vertex p plane →
      (cross_section p plane).sides % 2 = 1) :=
sorry

end no_odd_sided_cross_section_polyhedron_l2660_266002


namespace quadratic_inequality_solution_set_l2660_266056

/-- The solution set of the quadratic inequality -x^2 + 4x + 12 > 0 is (-2, 6) -/
theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 4*x + 12 > 0} = Set.Ioo (-2 : ℝ) 6 := by
  sorry

end quadratic_inequality_solution_set_l2660_266056


namespace classroom_wall_paint_area_l2660_266077

/-- Calculates the area to be painted on a wall with two windows. -/
def areaToBePainted (wallHeight wallWidth window1Height window1Width window2Height window2Width : ℕ) : ℕ :=
  let wallArea := wallHeight * wallWidth
  let window1Area := window1Height * window1Width
  let window2Area := window2Height * window2Width
  wallArea - window1Area - window2Area

/-- Proves that the area to be painted on the classroom wall is 243 square feet. -/
theorem classroom_wall_paint_area :
  areaToBePainted 15 18 3 5 2 6 = 243 := by
  sorry

#eval areaToBePainted 15 18 3 5 2 6

end classroom_wall_paint_area_l2660_266077


namespace hajar_score_is_24_l2660_266031

def guessing_game (hajar_score farah_score : ℕ) : Prop :=
  farah_score - hajar_score = 21 ∧
  farah_score + hajar_score = 69 ∧
  farah_score > hajar_score

theorem hajar_score_is_24 :
  ∃ (hajar_score farah_score : ℕ), guessing_game hajar_score farah_score ∧ hajar_score = 24 :=
by sorry

end hajar_score_is_24_l2660_266031


namespace smallest_angle_is_27_l2660_266001

/-- Represents the properties of a circle divided into sectors --/
structure CircleSectors where
  num_sectors : ℕ
  angle_sum : ℕ
  is_arithmetic_sequence : Bool
  all_angles_integer : Bool

/-- Finds the smallest possible sector angle given the circle properties --/
def smallest_sector_angle (circle : CircleSectors) : ℕ :=
  sorry

/-- Theorem stating that for a circle divided into 10 sectors with the given properties,
    the smallest possible sector angle is 27 degrees --/
theorem smallest_angle_is_27 :
  ∀ (circle : CircleSectors),
    circle.num_sectors = 10 ∧
    circle.angle_sum = 360 ∧
    circle.is_arithmetic_sequence = true ∧
    circle.all_angles_integer = true →
    smallest_sector_angle circle = 27 :=
  sorry

end smallest_angle_is_27_l2660_266001


namespace sqrt_sum_equals_sqrt_l2660_266050

theorem sqrt_sum_equals_sqrt (n : ℕ+) :
  (∃ x y : ℕ+, Real.sqrt x + Real.sqrt y = Real.sqrt n) ↔
  (∃ p q : ℕ, p > 1 ∧ n = p^2 * q) :=
sorry

end sqrt_sum_equals_sqrt_l2660_266050


namespace sqrt_power_eight_equals_390625_l2660_266034

theorem sqrt_power_eight_equals_390625 :
  (Real.sqrt ((Real.sqrt 5) ^ 4)) ^ 8 = 390625 := by sorry

end sqrt_power_eight_equals_390625_l2660_266034


namespace rectangle_ratio_l2660_266038

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The configuration of squares and rectangle -/
structure Configuration where
  square : Square
  rectangle : Rectangle
  square_count : ℕ

/-- The theorem statement -/
theorem rectangle_ratio (config : Configuration) :
  config.square_count = 3 →
  config.rectangle.length = config.square_count * config.square.side →
  config.rectangle.width = config.square.side →
  config.rectangle.length / config.rectangle.width = 3 := by
  sorry


end rectangle_ratio_l2660_266038


namespace distinct_scores_is_nineteen_l2660_266057

/-- Represents the number of distinct possible scores for a basketball player -/
def distinctScores : ℕ :=
  let shotTypes := 3  -- free throw, 2-point basket, 3-point basket
  let totalShots := 8
  let pointValues := [1, 2, 3]
  19  -- The actual count of distinct scores

/-- Theorem stating that the number of distinct possible scores is 19 -/
theorem distinct_scores_is_nineteen :
  distinctScores = 19 := by sorry

end distinct_scores_is_nineteen_l2660_266057


namespace sufficient_not_necessary_l2660_266093

/-- A hyperbola with parameter b > 0 -/
structure Hyperbola (b : ℝ) : Prop where
  pos : b > 0

/-- A line with equation x + 3y - 1 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 3 * p.2 - 1 = 0}

/-- The left branch of the hyperbola -/
def LeftBranch (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.1^2 / 4 - p.2^2 / b^2 = 1}

/-- Predicate for line intersecting the left branch of hyperbola -/
def Intersects (b : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ Line ∩ LeftBranch b

/-- Theorem stating that b > 1 is sufficient but not necessary for intersection -/
theorem sufficient_not_necessary (h : Hyperbola b) :
    (b > 1 → Intersects b) ∧ ¬(Intersects b → b > 1) := by
  sorry

end sufficient_not_necessary_l2660_266093


namespace even_and_increasing_order_l2660_266026

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem even_and_increasing_order (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : increasing_on_nonneg f) : 
  f (-2) < f 3 ∧ f 3 < f (-π) := by
  sorry

end even_and_increasing_order_l2660_266026


namespace bead_probability_l2660_266066

/-- The probability that a point on a line segment of length 3 is more than 1 unit away from both endpoints is 1/3 -/
theorem bead_probability : 
  let segment_length : ℝ := 3
  let min_distance : ℝ := 1
  let favorable_length : ℝ := segment_length - 2 * min_distance
  favorable_length / segment_length = 1 / 3 := by
sorry

end bead_probability_l2660_266066


namespace katarina_miles_l2660_266014

/-- The total miles run by all four runners -/
def total_miles : ℕ := 195

/-- The number of miles run by Harriet -/
def harriet_miles : ℕ := 48

/-- The number of runners who ran the same distance as Harriet -/
def same_distance_runners : ℕ := 3

theorem katarina_miles : 
  total_miles - harriet_miles * same_distance_runners = 51 := by sorry

end katarina_miles_l2660_266014


namespace road_paving_length_l2660_266094

/-- The length of road paved in April, in meters -/
def april_length : ℕ := 480

/-- The difference between March and April paving lengths, in meters -/
def length_difference : ℕ := 160

/-- The total length of road paved in March and April -/
def total_length : ℕ := april_length + (april_length + length_difference)

theorem road_paving_length : total_length = 1120 := by sorry

end road_paving_length_l2660_266094


namespace project_completion_time_l2660_266039

theorem project_completion_time (a b total_time quit_time : ℝ) 
  (hb : b = 30)
  (htotal : total_time = 15)
  (hquit : quit_time = 10)
  (h_completion : 5 * (1/a + 1/b) + 10 * (1/b) = 1) :
  a = 10 := by
sorry

end project_completion_time_l2660_266039


namespace events_mutually_exclusive_but_not_complementary_l2660_266091

/-- A box containing white and red balls -/
structure Box where
  white : Nat
  red : Nat

/-- The number of balls drawn from the box -/
def drawn : Nat := 3

/-- Event A: Exactly one red ball is drawn -/
def eventA (box : Box) : Prop :=
  ∃ (r w : Nat), r = 1 ∧ w = drawn - r ∧ r ≤ box.red ∧ w ≤ box.white

/-- Event B: Exactly one white ball is drawn -/
def eventB (box : Box) : Prop :=
  ∃ (w r : Nat), w = 1 ∧ r = drawn - w ∧ w ≤ box.white ∧ r ≤ box.red

/-- The box in the problem -/
def problemBox : Box := ⟨4, 3⟩

theorem events_mutually_exclusive_but_not_complementary :
  (¬ ∃ (outcome : Nat × Nat), eventA problemBox ∧ eventB problemBox) ∧
  (∃ (outcome : Nat × Nat), ¬(eventA problemBox ∨ eventB problemBox)) :=
by sorry


end events_mutually_exclusive_but_not_complementary_l2660_266091


namespace root_implies_u_value_l2660_266067

theorem root_implies_u_value (u : ℝ) : 
  (6 * ((-25 - Real.sqrt 421) / 12)^2 + 25 * ((-25 - Real.sqrt 421) / 12) + u = 0) → 
  u = 8.5 := by
sorry

end root_implies_u_value_l2660_266067


namespace remainder_of_2543_base12_div_9_l2660_266025

/-- Converts a base-12 digit to its decimal value -/
def base12ToDecimal (digit : Nat) : Nat :=
  if digit < 12 then digit else 0

/-- Converts a base-12 number to its decimal equivalent -/
def convertBase12ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun digit acc => acc * 12 + base12ToDecimal digit) 0

/-- The base-12 representation of 2543 -/
def base12Number : List Nat := [2, 5, 4, 3]

theorem remainder_of_2543_base12_div_9 :
  (convertBase12ToDecimal base12Number) % 9 = 8 := by
  sorry

end remainder_of_2543_base12_div_9_l2660_266025


namespace no_charming_seven_digit_number_l2660_266024

/-- A function that checks if a list of digits forms a charming number -/
def is_charming (digits : List Nat) : Prop :=
  digits.length = 7 ∧
  digits.toFinset = Finset.range 7 ∧
  (∀ k : Nat, k ∈ Finset.range 7 → 
    (digits.take k).foldl (fun acc d => acc * 10 + d) 0 % k = 0) ∧
  digits.getLast? = some 7

/-- Theorem stating that no charming 7-digit number exists -/
theorem no_charming_seven_digit_number : 
  ¬ ∃ (digits : List Nat), is_charming digits := by
  sorry

end no_charming_seven_digit_number_l2660_266024


namespace cindy_hourly_rate_l2660_266032

/-- Represents Cindy's teaching situation -/
structure TeachingSituation where
  num_courses : ℕ
  total_weekly_hours : ℕ
  weeks_in_month : ℕ
  monthly_earnings_per_course : ℕ

/-- Calculates the hourly rate given a teaching situation -/
def hourly_rate (s : TeachingSituation) : ℚ :=
  s.monthly_earnings_per_course / (s.total_weekly_hours / s.num_courses * s.weeks_in_month)

/-- Theorem stating that Cindy's hourly rate is $25 given the specified conditions -/
theorem cindy_hourly_rate :
  let s : TeachingSituation := {
    num_courses := 4,
    total_weekly_hours := 48,
    weeks_in_month := 4,
    monthly_earnings_per_course := 1200
  }
  hourly_rate s = 25 := by sorry

end cindy_hourly_rate_l2660_266032


namespace shirt_probabilities_l2660_266051

structure ShirtPack :=
  (m_shirts : ℕ)
  (count : ℕ)

def total_packs : ℕ := 50

def shirt_distribution : List ShirtPack := [
  ⟨0, 7⟩, ⟨1, 3⟩, ⟨4, 10⟩, ⟨5, 15⟩, ⟨7, 5⟩, ⟨9, 4⟩, ⟨10, 3⟩, ⟨11, 3⟩
]

def count_packs (pred : ShirtPack → Bool) : ℕ :=
  (shirt_distribution.filter pred).foldl (λ acc pack => acc + pack.count) 0

theorem shirt_probabilities :
  (count_packs (λ pack => pack.m_shirts = 0) : ℚ) / total_packs = 7 / 50 ∧
  (count_packs (λ pack => pack.m_shirts < 7) : ℚ) / total_packs = 7 / 10 ∧
  (count_packs (λ pack => pack.m_shirts > 9) : ℚ) / total_packs = 3 / 25 := by
  sorry

end shirt_probabilities_l2660_266051


namespace strongroom_keys_l2660_266022

theorem strongroom_keys (n : ℕ) : n > 0 → (
  (∃ (key_distribution : Fin 5 → Finset (Fin 10)),
    (∀ d : Fin 5, (key_distribution d).card = n) ∧
    (∀ majority : Finset (Fin 5), majority.card ≥ 3 →
      (majority.biUnion key_distribution).card = 10) ∧
    (∀ minority : Finset (Fin 5), minority.card ≤ 2 →
      (minority.biUnion key_distribution).card < 10))
  ↔ n = 6) :=
by sorry

end strongroom_keys_l2660_266022


namespace min_value_m_plus_n_l2660_266086

theorem min_value_m_plus_n (a b m n : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : (a + b) / 2 = 1 / 2) (hm : m = a + 1 / a) (hn : n = b + 1 / b) :
  ∀ x y, x > 0 → y > 0 → (x + y) / 2 = 1 / 2 → 
  (x + 1 / x) + (y + 1 / y) ≥ m + n ∧ m + n ≥ 5 := by
  sorry

end min_value_m_plus_n_l2660_266086


namespace complex_number_problem_l2660_266017

theorem complex_number_problem (α β : ℂ) :
  (α + β).re > 0 →
  (Complex.I * (α - 3 * β)).re > 0 →
  β = 2 + 3 * Complex.I →
  α = 6 - 3 * Complex.I :=
by sorry

end complex_number_problem_l2660_266017


namespace min_value_theorem_l2660_266098

theorem min_value_theorem (a b c : ℝ) (h : a + 2*b + 3*c = 2) :
  (∀ x y z : ℝ, x + 2*y + 3*z = 2 → a^2 + 2*b^2 + 3*c^2 ≤ x^2 + 2*y^2 + 3*z^2) →
  2*a + 4*b + 9*c = 5 := by
  sorry

end min_value_theorem_l2660_266098


namespace jake_sausage_spending_l2660_266046

/-- Represents a type of sausage package -/
structure SausagePackage where
  weight : Real
  price_per_pound : Real

/-- Calculates the total cost for a given number of packages of a specific type -/
def total_cost_for_type (package : SausagePackage) (num_packages : Nat) : Real :=
  package.weight * package.price_per_pound * num_packages

/-- Theorem: Jake spends $52 on sausages -/
theorem jake_sausage_spending :
  let type1 : SausagePackage := { weight := 2, price_per_pound := 4 }
  let type2 : SausagePackage := { weight := 1.5, price_per_pound := 5 }
  let type3 : SausagePackage := { weight := 3, price_per_pound := 3.5 }
  let num_packages : Nat := 2
  total_cost_for_type type1 num_packages +
  total_cost_for_type type2 num_packages +
  total_cost_for_type type3 num_packages = 52 := by
  sorry

end jake_sausage_spending_l2660_266046


namespace modular_congruence_solution_l2660_266033

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end modular_congruence_solution_l2660_266033


namespace intersection_S_T_l2660_266085

def S : Set ℝ := {x | x + 1 ≥ 2}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} := by sorry

end intersection_S_T_l2660_266085


namespace power_of_two_in_product_l2660_266080

theorem power_of_two_in_product (w : ℕ+) : 
  (∃ k : ℕ, 936 * w = k * 3^3 * 11^2) →  -- The product has 3^3 and 11^2 as factors
  (∀ x : ℕ+, x < 132 → ¬∃ k : ℕ, 936 * x = k * 3^3 * 11^2) →  -- 132 is the smallest possible w
  (∃ m : ℕ, 936 * w = 2^5 * m ∧ m % 2 ≠ 0) :=  -- The highest power of 2 dividing the product is 2^5
by sorry

end power_of_two_in_product_l2660_266080


namespace expression_simplification_l2660_266045

theorem expression_simplification (x : ℝ) (h : x^2 + x - 5 = 0) :
  (x - 2) / (x^2 - 4*x + 4) / (x + 2 - (x^2 + x - 4) / (x - 2)) + 1 / (x + 1) = -1/5 := by
  sorry

end expression_simplification_l2660_266045


namespace count_solutions_x_plus_y_plus_z_eq_10_l2660_266036

def positive_integer_solutions (n : ℕ) : ℕ :=
  (n - 1) * (n - 2) / 2

theorem count_solutions_x_plus_y_plus_z_eq_10 :
  positive_integer_solutions 10 = 36 := by
  sorry

end count_solutions_x_plus_y_plus_z_eq_10_l2660_266036


namespace paper_strip_length_l2660_266074

theorem paper_strip_length (strip_length : ℝ) : 
  strip_length > 0 →
  strip_length + strip_length - 6 = 30 →
  strip_length = 18 := by
sorry

end paper_strip_length_l2660_266074


namespace sum_of_roots_quadratic_l2660_266053

theorem sum_of_roots_quadratic (b : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ + b = 0 → x₂^2 - 2*x₂ + b = 0 → x₁ + x₂ = 2 := by
sorry

end sum_of_roots_quadratic_l2660_266053


namespace x_minus_y_equals_nine_l2660_266005

theorem x_minus_y_equals_nine
  (x y : ℕ)
  (h1 : 3^x * 4^y = 19683)
  (h2 : x = 9) :
  x - y = 9 := by
sorry

end x_minus_y_equals_nine_l2660_266005


namespace min_value_of_f_l2660_266015

/-- The function f(x) = 5x^2 + 10x + 20 -/
def f (x : ℝ) : ℝ := 5 * x^2 + 10 * x + 20

/-- The minimum value of f(x) is 15 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 15 ∧ ∀ x, f x ≥ min :=
by sorry

end min_value_of_f_l2660_266015


namespace quadrilateral_comparison_l2660_266028

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral :=
  (a b c d : Point)

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Quadrilateral I defined by its vertices -/
def quadI : Quadrilateral :=
  { a := {x := 0, y := 0},
    b := {x := 3, y := 0},
    c := {x := 3, y := 3},
    d := {x := 0, y := 2} }

/-- Quadrilateral II defined by its vertices -/
def quadII : Quadrilateral :=
  { a := {x := 0, y := 0},
    b := {x := 3, y := 0},
    c := {x := 3, y := 2},
    d := {x := 0, y := 3} }

theorem quadrilateral_comparison :
  (area quadI = 7.5 ∧ area quadII = 7.5) ∧
  perimeter quadI > perimeter quadII := by sorry

end quadrilateral_comparison_l2660_266028


namespace smallest_three_digit_multiple_of_17_l2660_266072

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n := by
  sorry

end smallest_three_digit_multiple_of_17_l2660_266072


namespace shirt_price_shirt_price_is_33_l2660_266083

theorem shirt_price (pants_price : ℝ) (num_pants : ℕ) (num_shirts : ℕ) (total_payment : ℝ) (change : ℝ) : ℝ :=
  let total_spent := total_payment - change
  let pants_total := pants_price * num_pants
  let shirts_total := total_spent - pants_total
  shirts_total / num_shirts

#check shirt_price 54 2 4 250 10 = 33

-- The proof
theorem shirt_price_is_33 :
  shirt_price 54 2 4 250 10 = 33 := by
  sorry

end shirt_price_shirt_price_is_33_l2660_266083


namespace f_x_plus_3_odd_l2660_266065

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_x_plus_3_odd (f : ℝ → ℝ) 
  (h1 : is_odd (fun x ↦ f (x + 1))) 
  (h2 : is_odd (fun x ↦ f (x - 1))) : 
  is_odd (fun x ↦ f (x + 3)) := by
  sorry

end f_x_plus_3_odd_l2660_266065


namespace daniel_animals_legs_l2660_266071

/-- The number of legs an animal has --/
def legs (animal : String) : ℕ :=
  match animal with
  | "horse" => 4
  | "dog" => 4
  | "cat" => 4
  | "turtle" => 4
  | "goat" => 4
  | "snake" => 0
  | "spider" => 8
  | "bird" => 2
  | _ => 0

/-- The number of each type of animal Daniel has --/
def animals : List (String × ℕ) := [
  ("horse", 2),
  ("dog", 5),
  ("cat", 7),
  ("turtle", 3),
  ("goat", 1),
  ("snake", 4),
  ("spider", 2),
  ("bird", 3)
]

/-- The total number of legs of all animals --/
def totalLegs : ℕ := (animals.map (fun (a, n) => n * legs a)).sum

theorem daniel_animals_legs :
  totalLegs = 94 := by sorry

end daniel_animals_legs_l2660_266071


namespace problem_statement_l2660_266079

theorem problem_statement (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) :
  x^2004 + y^2004 = 2^2004 := by
  sorry

end problem_statement_l2660_266079


namespace endpoint_coordinate_sum_l2660_266041

/-- Given a line segment with midpoint (6, -10) and one endpoint (8, 0),
    the sum of the coordinates of the other endpoint is -16. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (x + 8) / 2 = 6 ∧ (y + 0) / 2 = -10 → 
  x + y = -16 := by
sorry

end endpoint_coordinate_sum_l2660_266041


namespace number_equation_solution_l2660_266042

theorem number_equation_solution : 
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by sorry

end number_equation_solution_l2660_266042


namespace real_root_of_cubic_l2660_266073

def cubic_polynomial (c d x : ℝ) : ℝ := c * x^3 + 4 * x^2 + d * x - 78

theorem real_root_of_cubic (c d : ℝ) :
  (∃ (z : ℂ), z = -3 - 4*I ∧ cubic_polynomial c d z.re = 0) →
  ∃ (x : ℝ), cubic_polynomial c d x = 0 ∧ x = -3 :=
sorry

end real_root_of_cubic_l2660_266073


namespace function_inequality_l2660_266063

/-- Given a function f(x) = ln x - 3x defined on (0, +∞), and for all x ∈ (0, +∞),
    f(x) ≤ x(ae^x - 4) + b, prove that a + b ≥ 0. -/
theorem function_inequality (a b : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.log x - 3 * x ≤ x * (a * Real.exp x - 4) + b) → 
  a + b ≥ 0 := by
  sorry

end function_inequality_l2660_266063


namespace f_monotonicity_f_shifted_even_f_positive_domain_l2660_266048

-- Define the function f(x) = lg|x-1|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 1)) / Real.log 10

-- Statement 1: f(x) is monotonically decreasing on (-∞, 1) and increasing on (1, +∞)
theorem f_monotonicity :
  (∀ x y, x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) := by sorry

-- Statement 2: f(x+1) is an even function
theorem f_shifted_even :
  ∀ x, f (x + 1) = f (-x + 1) := by sorry

-- Statement 3: If f(a) > 0, then a < 0 or a > 2
theorem f_positive_domain :
  ∀ a, f a > 0 → a < 0 ∨ a > 2 := by sorry

end f_monotonicity_f_shifted_even_f_positive_domain_l2660_266048


namespace quadratic_inequality_solution_l2660_266047

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + 5*x - 14 > 0} = {x : ℝ | x < -7 ∨ x > 2} := by
  sorry

end quadratic_inequality_solution_l2660_266047


namespace ratio_equality_product_l2660_266099

theorem ratio_equality_product (x : ℝ) : 
  (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4) → 
  ∃ y : ℝ, (x = 0 ∨ x = 5) ∧ x * y = 0 := by sorry

end ratio_equality_product_l2660_266099


namespace framed_painting_ratio_l2660_266006

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_height : ℝ
  painting_width : ℝ
  frame_side_width : ℝ

/-- Calculates the framed dimensions of the painting -/
def framed_dimensions (fp : FramedPainting) : ℝ × ℝ :=
  (fp.painting_width + 2 * fp.frame_side_width, 
   fp.painting_height + 6 * fp.frame_side_width)

/-- Calculates the area of the framed painting -/
def framed_area (fp : FramedPainting) : ℝ :=
  let (w, h) := framed_dimensions fp
  w * h

/-- Theorem stating the ratio of smaller to larger dimension of the framed painting -/
theorem framed_painting_ratio 
  (fp : FramedPainting)
  (h1 : fp.painting_height = 30)
  (h2 : fp.painting_width = 20)
  (h3 : framed_area fp = fp.painting_height * fp.painting_width) :
  let (w, h) := framed_dimensions fp
  min w h / max w h = 4 / 7 := by
    sorry

end framed_painting_ratio_l2660_266006


namespace archie_red_coins_l2660_266087

/-- Represents the number of coins collected for each color --/
structure CoinCount where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Calculates the total number of coins --/
def total_coins (c : CoinCount) : ℕ := c.yellow + c.red + c.blue

/-- Calculates the total money earned --/
def total_money (c : CoinCount) : ℕ := c.yellow + 3 * c.red + 5 * c.blue

/-- Theorem stating that Archie collected 700 red coins --/
theorem archie_red_coins :
  ∃ (c : CoinCount),
    total_coins c = 2800 ∧
    total_money c = 7800 ∧
    c.blue = c.red + 200 ∧
    c.red = 700 := by
  sorry


end archie_red_coins_l2660_266087


namespace min_reciprocal_sum_l2660_266043

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
sorry

end min_reciprocal_sum_l2660_266043


namespace student_number_problem_l2660_266027

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 102 → x = 120 := by
  sorry

end student_number_problem_l2660_266027


namespace cut_cube_edges_l2660_266012

/-- A cube with one corner cut off, creating a new triangular face -/
structure CutCube where
  /-- The number of edges in the original cube -/
  original_edges : ℕ
  /-- The number of new edges created by the cut -/
  new_edges : ℕ
  /-- The cut creates a triangular face -/
  triangular_face : Bool

/-- The total number of edges in the cut cube -/
def CutCube.total_edges (c : CutCube) : ℕ := c.original_edges + c.new_edges

/-- Theorem stating that a cube with one corner cut off has 15 edges -/
theorem cut_cube_edges :
  ∀ (c : CutCube),
  c.original_edges = 12 ∧ c.new_edges = 3 ∧ c.triangular_face = true →
  c.total_edges = 15 := by
  sorry

end cut_cube_edges_l2660_266012


namespace square_difference_equality_l2660_266078

theorem square_difference_equality : (15 + 12)^2 - (12^2 + 15^2) = 360 := by
  sorry

end square_difference_equality_l2660_266078


namespace fibonacci_inequality_l2660_266000

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- State the theorem
theorem fibonacci_inequality (n : ℕ) (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) 
  (h_ineq : min (fib n / fib (n-1)) (fib (n+1) / fib n) < a / b ∧ 
            a / b < max (fib n / fib (n-1)) (fib (n+1) / fib n)) : 
  b ≥ fib (n+1) := by
  sorry


end fibonacci_inequality_l2660_266000


namespace min_sum_given_log_sum_l2660_266004

theorem min_sum_given_log_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : Real.log m / Real.log 3 + Real.log n / Real.log 3 = 4) : 
  m + n ≥ 18 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 
    Real.log m₀ / Real.log 3 + Real.log n₀ / Real.log 3 = 4 ∧ m₀ + n₀ = 18 :=
by sorry

end min_sum_given_log_sum_l2660_266004


namespace cannot_complete_in_five_trips_l2660_266064

def truck_capacity : ℕ := 2000
def rice_sacks : ℕ := 150
def corn_sacks : ℕ := 100
def rice_weight : ℕ := 60
def corn_weight : ℕ := 25
def num_trips : ℕ := 5

theorem cannot_complete_in_five_trips :
  rice_sacks * rice_weight + corn_sacks * corn_weight > num_trips * truck_capacity :=
by sorry

end cannot_complete_in_five_trips_l2660_266064


namespace valid_plate_count_l2660_266068

/-- Represents a license plate with 4 characters -/
structure LicensePlate :=
  (first : Char) (second : Char) (third : Char) (fourth : Char)

/-- Checks if a character is a letter -/
def isLetter (c : Char) : Bool := c.isAlpha

/-- Checks if a character is a digit -/
def isDigit (c : Char) : Bool := c.isDigit

/-- Checks if a license plate is valid according to the given conditions -/
def isValidPlate (plate : LicensePlate) : Bool :=
  (isLetter plate.first) &&
  (isDigit plate.second) &&
  (isDigit plate.third) &&
  (isLetter plate.fourth) &&
  (plate.first == plate.fourth || plate.second == plate.third)

/-- The total number of possible characters for a letter position -/
def numLetters : Nat := 26

/-- The total number of possible characters for a digit position -/
def numDigits : Nat := 10

/-- Counts the number of valid license plates -/
def countValidPlates : Nat :=
  (numLetters * numDigits * 1 * numLetters) +  -- Same digits
  (numLetters * numDigits * numDigits * 1) -   -- Same letters
  (numLetters * numDigits * 1 * 1)             -- Both pairs same

theorem valid_plate_count :
  countValidPlates = 9100 := by
  sorry

#eval countValidPlates  -- Should output 9100

end valid_plate_count_l2660_266068


namespace find_M_l2660_266058

theorem find_M : ∃ (M : ℕ+), (36 : ℕ)^2 * 81^2 = 18^2 * M^2 ∧ M = 162 := by
  sorry

end find_M_l2660_266058


namespace base_7_divisibility_l2660_266010

def base_7_to_decimal (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7 + d

def is_divisible_by_9 (n : ℕ) : Prop :=
  ∃ k, n = 9 * k

theorem base_7_divisibility (x : ℕ) :
  (x < 7) →
  (is_divisible_by_9 (base_7_to_decimal 4 5 x 2)) ↔ x = 4 :=
by sorry

end base_7_divisibility_l2660_266010


namespace cubic_three_distinct_roots_in_ap_l2660_266018

/-- A cubic polynomial with coefficients a and b -/
def cubic_polynomial (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- Predicate for a cubic polynomial having three distinct roots in arithmetic progression -/
def has_three_distinct_roots_in_ap (a b : ℝ) : Prop :=
  ∃ (r d : ℝ), d ≠ 0 ∧
    cubic_polynomial a b (-d) = 0 ∧
    cubic_polynomial a b 0 = 0 ∧
    cubic_polynomial a b d = 0

/-- Theorem stating the condition for a cubic polynomial to have three distinct roots in arithmetic progression -/
theorem cubic_three_distinct_roots_in_ap (a b : ℝ) :
  has_three_distinct_roots_in_ap a b ↔ b = 0 ∧ a < 0 :=
sorry

end cubic_three_distinct_roots_in_ap_l2660_266018


namespace mary_chopped_six_tables_l2660_266037

/-- Represents the number of sticks of wood produced by different furniture items -/
structure FurnitureWood where
  chair : Nat
  table : Nat
  stool : Nat

/-- Represents the chopping and burning scenario -/
structure WoodScenario where
  furniture : FurnitureWood
  chopped_chairs : Nat
  chopped_stools : Nat
  burn_rate : Nat
  warm_hours : Nat

/-- Calculates the number of tables chopped given a wood scenario -/
def tables_chopped (scenario : WoodScenario) : Nat :=
  let total_wood := scenario.warm_hours * scenario.burn_rate
  let wood_from_chairs := scenario.chopped_chairs * scenario.furniture.chair
  let wood_from_stools := scenario.chopped_stools * scenario.furniture.stool
  let wood_from_tables := total_wood - wood_from_chairs - wood_from_stools
  wood_from_tables / scenario.furniture.table

theorem mary_chopped_six_tables :
  let mary_scenario : WoodScenario := {
    furniture := { chair := 6, table := 9, stool := 2 },
    chopped_chairs := 18,
    chopped_stools := 4,
    burn_rate := 5,
    warm_hours := 34
  }
  tables_chopped mary_scenario = 6 := by
  sorry

end mary_chopped_six_tables_l2660_266037


namespace min_value_expression_l2660_266095

theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) (heq : 2 * m + n = 4) :
  1 / m + 2 / n ≥ 2 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 4 ∧ 1 / m₀ + 2 / n₀ = 2 :=
sorry

end min_value_expression_l2660_266095


namespace beta_highest_success_ratio_l2660_266035

/-- Represents a participant's scores in a two-day challenge -/
structure ParticipantScores where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ

def ParticipantScores.total_score (p : ParticipantScores) : ℕ :=
  p.day1_score + p.day2_score

def ParticipantScores.total_attempted (p : ParticipantScores) : ℕ :=
  p.day1_attempted + p.day2_attempted

def ParticipantScores.success_ratio (p : ParticipantScores) : ℚ :=
  (p.total_score : ℚ) / p.total_attempted

def ParticipantScores.daily_success_ratio (p : ParticipantScores) (day : Fin 2) : ℚ :=
  match day with
  | 0 => (p.day1_score : ℚ) / p.day1_attempted
  | 1 => (p.day2_score : ℚ) / p.day2_attempted

theorem beta_highest_success_ratio
  (alpha : ParticipantScores)
  (beta : ParticipantScores)
  (h_total_points : alpha.total_attempted = 500)
  (h_alpha_scores : alpha.day1_score = 200 ∧ alpha.day1_attempted = 300 ∧
                    alpha.day2_score = 100 ∧ alpha.day2_attempted = 200)
  (h_beta_fewer : beta.day1_attempted < alpha.day1_attempted ∧
                  beta.day2_attempted < alpha.day2_attempted)
  (h_beta_nonzero : beta.day1_score > 0 ∧ beta.day2_score > 0)
  (h_beta_lower_ratio : ∀ day, beta.daily_success_ratio day < alpha.daily_success_ratio day)
  (h_alpha_ratio : alpha.success_ratio = 3/5)
  (h_beta_day1 : beta.day1_attempted = 220) :
  beta.success_ratio ≤ 248/500 :=
sorry

end beta_highest_success_ratio_l2660_266035


namespace LL₁_length_is_20_over_17_l2660_266088

/-- Right triangle XYZ with hypotenuse XZ = 13 and leg XY = 5 -/
structure TriangleXYZ where
  XZ : ℝ
  XY : ℝ
  is_right : XZ = 13 ∧ XY = 5

/-- Point X₁ on YZ where the angle bisector of ∠X meets YZ -/
def X₁ (t : TriangleXYZ) : ℝ × ℝ := sorry

/-- Right triangle LMN with hypotenuse LM = X₁Z and leg LN = X₁Y -/
structure TriangleLMN (t : TriangleXYZ) where
  LM : ℝ
  LN : ℝ
  is_right : LM = (X₁ t).2 ∧ LN = (X₁ t).1

/-- Point L₁ on MN where the angle bisector of ∠L meets MN -/
def L₁ (t : TriangleXYZ) (u : TriangleLMN t) : ℝ × ℝ := sorry

/-- The length of LL₁ -/
def LL₁_length (t : TriangleXYZ) (u : TriangleLMN t) : ℝ := sorry

/-- Theorem: The length of LL₁ is 20/17 -/
theorem LL₁_length_is_20_over_17 (t : TriangleXYZ) (u : TriangleLMN t) :
  LL₁_length t u = 20 / 17 := by sorry

end LL₁_length_is_20_over_17_l2660_266088


namespace correlation_theorem_l2660_266020

/-- A function to represent the relationship between x and y -/
def f (x : ℝ) : ℝ := 0.1 * x - 10

/-- Definition of positive correlation -/
def positively_correlated (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- Definition of negative correlation -/
def negatively_correlated (f g : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → g x₁ > g x₂

/-- The main theorem -/
theorem correlation_theorem (z : ℝ → ℝ) 
  (h : negatively_correlated f z) :
  positively_correlated f ∧ negatively_correlated id z := by
  sorry

end correlation_theorem_l2660_266020


namespace incorrect_average_calculation_l2660_266008

theorem incorrect_average_calculation (n : Nat) (incorrect_num correct_num : ℚ) (correct_avg : ℚ) :
  n = 10 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 75 ∧ 
  correct_avg = 51 →
  ∃ (S : ℚ), 
    (S + correct_num) / n = correct_avg ∧
    (S + incorrect_num) / n = 46 :=
by sorry

end incorrect_average_calculation_l2660_266008


namespace base8_addition_problem_l2660_266049

-- Define the base
def base : ℕ := 8

-- Define the addition operation in base 8
def add_base8 (a b : ℕ) : ℕ := (a + b) % base

-- Define the carry operation in base 8
def carry_base8 (a b : ℕ) : ℕ := (a + b) / base

-- The theorem to prove
theorem base8_addition_problem (square : ℕ) :
  square < base →
  add_base8 (add_base8 square square) 4 = 6 →
  add_base8 (add_base8 3 5) square = square →
  add_base8 (add_base8 4 square) (carry_base8 3 5) = 3 →
  square = 1 := by
sorry

end base8_addition_problem_l2660_266049


namespace divisors_of_cube_l2660_266062

/-- 
Given a natural number n with exactly two prime divisors,
if n^2 has 81 divisors, then n^3 has either 160 or 169 divisors.
-/
theorem divisors_of_cube (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ ∃ α β : ℕ, n = p^α * q^β) →
  (Finset.card (Nat.divisors (n^2)) = 81) →
  (Finset.card (Nat.divisors (n^3)) = 160 ∨ Finset.card (Nat.divisors (n^3)) = 169) :=
by sorry

end divisors_of_cube_l2660_266062


namespace arithmetic_sequence_common_difference_l2660_266055

/-- Given an arithmetic sequence {a_n} where a_2 = 1 and a_3 + a_5 = 4,
    the common difference of the sequence is 1/2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ) -- The sequence as a function from natural numbers to rationals
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Arithmetic sequence condition
  (h_a2 : a 2 = 1) -- Given: a_2 = 1
  (h_sum : a 3 + a 5 = 4) -- Given: a_3 + a_5 = 4
  : ∃ d : ℚ, d = 1/2 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end arithmetic_sequence_common_difference_l2660_266055


namespace coefficient_x3_is_correct_l2660_266007

/-- The coefficient of x^3 in the expansion of (x^2 - 2x)(1 + x)^6 -/
def coefficient_x3 : ℤ := -24

/-- The expansion of (x^2 - 2x)(1 + x)^6 -/
def expansion (x : ℚ) : ℚ := (x^2 - 2*x) * (1 + x)^6

theorem coefficient_x3_is_correct :
  (∃ f : ℚ → ℚ, ∀ x, expansion x = f x + coefficient_x3 * x^3 + x^4 * f x) :=
sorry

end coefficient_x3_is_correct_l2660_266007


namespace store_purchase_cost_l2660_266021

/-- Given the prices of pens, notebooks, and pencils satisfying certain conditions,
    prove that 4 pens, 5 notebooks, and 5 pencils cost 73 rubles. -/
theorem store_purchase_cost (pen_price notebook_price pencil_price : ℚ) :
  (2 * pen_price + 3 * notebook_price + pencil_price = 33) →
  (pen_price + notebook_price + 2 * pencil_price = 20) →
  (4 * pen_price + 5 * notebook_price + 5 * pencil_price = 73) :=
by sorry

end store_purchase_cost_l2660_266021


namespace otimes_inequality_system_l2660_266059

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a - 2 * b

-- Theorem statement
theorem otimes_inequality_system (a : ℝ) :
  (∀ x : ℝ, x > 6 ↔ (otimes x 3 > 0 ∧ otimes x a > a)) →
  a ≤ 2 := by
  sorry

end otimes_inequality_system_l2660_266059


namespace ladder_problem_l2660_266029

theorem ladder_problem (ladder_length height base : ℝ) :
  ladder_length = 15 ∧ height = 12 ∧ ladder_length^2 = height^2 + base^2 → base = 9 := by
  sorry

end ladder_problem_l2660_266029


namespace inequality_proof_l2660_266092

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Real.sqrt (a * b) < (a - b) / (Real.log a - Real.log b) ∧ 
  (a - b) / (Real.log a - Real.log b) < (a + b) / 2 := by
  sorry

end inequality_proof_l2660_266092


namespace max_area_wire_rectangle_or_square_l2660_266013

/-- The maximum area enclosed by a rectangle or square formed from a wire of length 2 meters -/
theorem max_area_wire_rectangle_or_square : 
  let wire_length : ℝ := 2
  let max_area : ℝ := (1 : ℝ) / 4
  ∀ l w : ℝ, 
    0 < l ∧ 0 < w →  -- positive length and width
    2 * (l + w) ≤ wire_length →  -- perimeter constraint
    l * w ≤ max_area :=
by sorry

end max_area_wire_rectangle_or_square_l2660_266013


namespace pencil_users_count_l2660_266075

/-- The number of attendants who used a pen -/
def pen_users : ℕ := 15

/-- The number of attendants who used only one type of writing tool -/
def single_tool_users : ℕ := 20

/-- The number of attendants who used both types of writing tools -/
def both_tool_users : ℕ := 10

/-- The number of attendants who used a pencil -/
def pencil_users : ℕ := single_tool_users + both_tool_users - (pen_users - both_tool_users)

theorem pencil_users_count : pencil_users = 25 := by sorry

end pencil_users_count_l2660_266075


namespace ratio_odd_even_divisors_l2660_266052

def M : ℕ := 18 * 18 * 56 * 165

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 62 = sum_even_divisors M := by sorry

end ratio_odd_even_divisors_l2660_266052


namespace limit_exponential_arcsin_ratio_l2660_266090

open Real

theorem limit_exponential_arcsin_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → 
    |((exp (3 * x) - exp (-2 * x)) / (2 * arcsin x - sin x)) - 5| < ε := by
  sorry

end limit_exponential_arcsin_ratio_l2660_266090
