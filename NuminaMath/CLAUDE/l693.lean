import Mathlib

namespace not_p_false_sufficient_not_necessary_for_p_or_q_true_l693_69312

theorem not_p_false_sufficient_not_necessary_for_p_or_q_true (p q : Prop) :
  (¬¬p → p ∨ q) ∧ ∃ (p q : Prop), (p ∨ q) ∧ ¬(¬¬p) :=
sorry

end not_p_false_sufficient_not_necessary_for_p_or_q_true_l693_69312


namespace diamond_equation_solution_l693_69324

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem diamond_equation_solution :
  ∀ x : ℝ, diamond 5 x = 12 → x = 42 := by
  sorry

end diamond_equation_solution_l693_69324


namespace arithmetic_geometric_sequence_ratio_l693_69326

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (d S : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  (∀ n, a n > 0) →  -- positivity condition
  (∀ n, S * n = (n / 2) * (2 * a 1 + (n - 1) * d)) →  -- sum formula
  (a 2) * (a 2 + S * 5) = (S * 3) ^ 2 →  -- geometric sequence condition
  d / a 1 = 3 / 2 := by
sorry

end arithmetic_geometric_sequence_ratio_l693_69326


namespace symmetry_implies_m_equals_one_l693_69303

/-- Two points are symmetric about the origin if their coordinates are negations of each other -/
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- The theorem stating that if P(2, -1) and Q(-2, m) are symmetric about the origin, then m = 1 -/
theorem symmetry_implies_m_equals_one :
  ∀ m : ℝ, symmetric_about_origin (2, -1) (-2, m) → m = 1 :=
by
  sorry

end symmetry_implies_m_equals_one_l693_69303


namespace floor_sqrt_80_l693_69323

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end floor_sqrt_80_l693_69323


namespace cuboid_volume_l693_69330

/-- The volume of a cuboid with dimensions 4 cm, 6 cm, and 15 cm is 360 cubic centimeters. -/
theorem cuboid_volume : 
  let length : ℝ := 4
  let width : ℝ := 6
  let height : ℝ := 15
  length * width * height = 360 := by sorry

end cuboid_volume_l693_69330


namespace problem_shape_surface_area_l693_69354

/-- Represents a solid shape made of unit cubes -/
structure CubeShape where
  base_length : ℕ
  base_width : ℕ
  top_length : ℕ
  top_width : ℕ
  total_cubes : ℕ

/-- Calculates the surface area of the CubeShape -/
def surface_area (shape : CubeShape) : ℕ :=
  sorry

/-- The specific cube shape described in the problem -/
def problem_shape : CubeShape :=
  { base_length := 4
  , base_width := 3
  , top_length := 3
  , top_width := 1
  , total_cubes := 15
  }

/-- Theorem stating that the surface area of the problem_shape is 36 square units -/
theorem problem_shape_surface_area :
  surface_area problem_shape = 36 := by
  sorry

end problem_shape_surface_area_l693_69354


namespace max_intersections_is_19_l693_69339

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between a line and three circles -/
def max_line_circle_intersections : ℕ := 6

/-- The number of lines -/
def num_lines : ℕ := 2

/-- The number of intersection points between two lines -/
def line_line_intersections : ℕ := 1

/-- The maximum number of intersection points between 3 circles and 2 straight lines on a plane -/
def max_total_intersections : ℕ :=
  max_circle_intersections + 
  (num_lines * max_line_circle_intersections) + 
  line_line_intersections

theorem max_intersections_is_19 : 
  max_total_intersections = 19 := by sorry

end max_intersections_is_19_l693_69339


namespace kelly_chickens_count_l693_69320

/-- The number of chickens Kelly has -/
def number_of_chickens : ℕ := 8

/-- The number of eggs each chicken lays per day -/
def eggs_per_chicken_per_day : ℕ := 3

/-- The price of a dozen eggs in dollars -/
def price_per_dozen : ℕ := 5

/-- The total amount Kelly makes in 4 weeks in dollars -/
def total_earnings : ℕ := 280

/-- The number of days in 4 weeks -/
def days_in_four_weeks : ℕ := 28

theorem kelly_chickens_count :
  number_of_chickens * eggs_per_chicken_per_day * days_in_four_weeks / 12 * price_per_dozen = total_earnings :=
sorry

end kelly_chickens_count_l693_69320


namespace library_book_count_l693_69385

/-- The number of books in the library after taking out and bringing back some books -/
def final_book_count (initial : ℕ) (taken_out : ℕ) (brought_back : ℕ) : ℕ :=
  initial - taken_out + brought_back

/-- Theorem: Given 336 initial books, 124 taken out, and 22 brought back, there are 234 books now -/
theorem library_book_count : final_book_count 336 124 22 = 234 := by
  sorry

end library_book_count_l693_69385


namespace fiona_id_is_17_l693_69379

/-- A structure representing a math club member with an ID number -/
structure MathClubMember where
  name : String
  id : Nat

/-- A predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

/-- A predicate to check if a number is a two-digit number -/
def isTwoDigit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

theorem fiona_id_is_17 
  (dan emily fiona : MathClubMember)
  (h1 : isPrime dan.id ∧ isPrime emily.id ∧ isPrime fiona.id)
  (h2 : isTwoDigit dan.id ∧ isTwoDigit emily.id ∧ isTwoDigit fiona.id)
  (h3 : ∃ p q : Nat, dan.id < p ∧ p < q ∧ 
    (emily.id = p ∨ emily.id = q) ∧ 
    (fiona.id = p ∨ fiona.id = q) ∧
    isPrime p ∧ isPrime q)
  (h4 : ∃ today : Nat, emily.id + fiona.id = today ∧ today ≤ 31)
  (h5 : ∃ emilys_birthday : Nat, dan.id + fiona.id = emilys_birthday - 1 ∧ emilys_birthday ≤ 31)
  (h6 : dan.id + emily.id = (emily.id + fiona.id) + 1)
  : fiona.id = 17 := by
  sorry

end fiona_id_is_17_l693_69379


namespace one_way_cost_proof_l693_69377

/-- Represents the cost of one-way travel from home to office -/
def one_way_cost : ℝ := 16

/-- Represents the total number of trips in 9 working days -/
def total_trips : ℕ := 18

/-- Represents the total cost of travel for 9 working days -/
def total_cost : ℝ := 288

/-- Theorem stating that the one-way cost multiplied by the total number of trips
    equals the total cost for 9 working days -/
theorem one_way_cost_proof :
  one_way_cost * (total_trips : ℝ) = total_cost := by sorry

end one_way_cost_proof_l693_69377


namespace mrs_hilt_apples_l693_69394

/-- Calculates the total number of apples eaten given a rate and time period. -/
def applesEaten (rate : ℕ) (hours : ℕ) : ℕ := rate * hours

/-- Theorem stating that eating 5 apples per hour for 3 hours results in 15 apples eaten. -/
theorem mrs_hilt_apples : applesEaten 5 3 = 15 := by
  sorry

end mrs_hilt_apples_l693_69394


namespace part_one_part_two_part_three_l693_69311

-- Define the main equation
def main_equation (x a : ℝ) : Prop :=
  Real.arctan (x / 2) + Real.arctan (2 - x) = a

-- Part 1
theorem part_one :
  ∀ x : ℝ, main_equation x (π / 4) →
    Real.arccos (x / 2) = 2 * π / 3 ∨ Real.arccos (x / 2) = 0 := by sorry

-- Part 2
theorem part_two :
  ∃ x a : ℝ, main_equation x a →
    a ∈ Set.Icc (Real.arctan (1 / (-2 * Real.sqrt 10 - 6))) (Real.arctan (1 / (2 * Real.sqrt 10 - 6))) := by sorry

-- Part 3
theorem part_three :
  ∀ α β a : ℝ, 
    α ∈ Set.Icc 5 15 → β ∈ Set.Icc 5 15 →
    α ≠ β →
    main_equation α a → main_equation β a →
    α + β ≤ 19 := by sorry

end part_one_part_two_part_three_l693_69311


namespace line_b_production_l693_69333

/-- Represents the production of a factory with three production lines -/
structure FactoryProduction where
  total : ℕ
  lineA : ℕ
  lineB : ℕ
  lineC : ℕ

/-- 
Given a factory production with three lines where:
1. The total production is 24,000 units
2. The number of units sampled from each line forms an arithmetic sequence
3. The sum of production from all lines equals the total production

Then the production of line B is 8,000 units
-/
theorem line_b_production (prod : FactoryProduction) 
  (h_total : prod.total = 24000)
  (h_arithmetic : prod.lineB * 2 = prod.lineA + prod.lineC)
  (h_sum : prod.lineA + prod.lineB + prod.lineC = prod.total) :
  prod.lineB = 8000 := by
  sorry

end line_b_production_l693_69333


namespace f_plus_g_is_non_horizontal_line_l693_69313

/-- Represents a parabola in vertex form -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ
  a_nonzero : a ≠ 0

/-- The function resulting from translating the original parabola 7 units right -/
def f (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h + 7)^2 + p.k

/-- The function resulting from reflecting the parabola and translating 7 units left -/
def g (p : Parabola) (x : ℝ) : ℝ :=
  -p.a * (x - p.h - 7)^2 - p.k

/-- The sum of f and g -/
def f_plus_g (p : Parabola) (x : ℝ) : ℝ :=
  f p x + g p x

/-- Theorem stating that f_plus_g is a non-horizontal line -/
theorem f_plus_g_is_non_horizontal_line (p : Parabola) :
  ∃ m b, m ≠ 0 ∧ ∀ x, f_plus_g p x = m * x + b := by
  sorry

end f_plus_g_is_non_horizontal_line_l693_69313


namespace product_of_reals_l693_69305

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 10) (sum_cubes_eq : a^3 + b^3 = 172) :
  a * b = 27.6 := by
  sorry

end product_of_reals_l693_69305


namespace total_children_l693_69344

/-- Given a group of children where:
    k children are initially selected and given an apple,
    m children are selected later,
    n of the m children had previously received an apple,
    prove that the total number of children is k * (m/n) -/
theorem total_children (k m n : ℕ) (h : n ≤ m) (h' : n > 0) :
  ∃ (total : ℚ), total = k * (m / n) := by
  sorry

end total_children_l693_69344


namespace expression_behavior_l693_69350

theorem expression_behavior (x : ℝ) (h : -3 < x ∧ x < 2) :
  (x^2 + 4*x + 5) / (2*x + 6) ≥ 3/4 ∧
  ((x^2 + 4*x + 5) / (2*x + 6) = 3/4 ↔ x = -1) := by
  sorry

end expression_behavior_l693_69350


namespace set_intersection_range_l693_69375

theorem set_intersection_range (a : ℝ) : 
  let A : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B : Set ℝ := {x | x < -1 ∨ x > 16}
  A ∩ B = A → a < 6 ∨ a > 7.5 := by
sorry

end set_intersection_range_l693_69375


namespace polynomial_divisibility_l693_69392

theorem polynomial_divisibility (a : ℤ) : 
  (∃ q : Polynomial ℤ, X^6 - 33•X + 20 = (X^2 - X + a•1) * q) → a = 4 := by
  sorry

end polynomial_divisibility_l693_69392


namespace least_distinct_values_l693_69353

theorem least_distinct_values (list : List ℕ+) : 
  list.length = 2023 →
  ∃! m : ℕ+, (list.count m = 11 ∧ ∀ n : ℕ+, n ≠ m → list.count n < 11) →
  (∀ k : ℕ+, k < 203 → ∃ x : ℕ+, list.count x > list.count k) →
  ∃ x : ℕ+, list.count x = 203 :=
by sorry

end least_distinct_values_l693_69353


namespace triangle_side_calculation_l693_69390

theorem triangle_side_calculation (a b : ℝ) (A B : Real) (hpos : 0 < a) :
  a = Real.sqrt 2 →
  B = 60 * π / 180 →
  A = 45 * π / 180 →
  b = a * Real.sin B / Real.sin A →
  b = Real.sqrt 3 := by
sorry

end triangle_side_calculation_l693_69390


namespace initial_distance_between_trains_l693_69337

def train_length_1 : ℝ := 120
def train_length_2 : ℝ := 210
def speed_1 : ℝ := 69
def speed_2 : ℝ := 82
def meeting_time : ℝ := 1.9071321976361095

theorem initial_distance_between_trains : 
  let relative_speed := (speed_1 + speed_2) * 1000 / 3600
  let distance_covered := relative_speed * (meeting_time * 3600)
  distance_covered - (train_length_1 + train_length_2) = 287670 := by
  sorry

end initial_distance_between_trains_l693_69337


namespace food_distribution_l693_69376

/-- The initial number of men for whom the food lasts 50 days -/
def initial_men : ℕ := sorry

/-- The number of days the food lasts for the initial group -/
def initial_days : ℕ := 50

/-- The number of additional men who join -/
def additional_men : ℕ := 20

/-- The number of days the food lasts after additional men join -/
def new_days : ℕ := 25

/-- Theorem stating that the initial number of men is 20 -/
theorem food_distribution :
  initial_men * initial_days = (initial_men + additional_men) * new_days ∧
  initial_men = 20 := by sorry

end food_distribution_l693_69376


namespace books_loaned_out_l693_69393

theorem books_loaned_out (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 65 / 100)
  (h3 : final_books = 68) : 
  ∃ (loaned_books : ℕ), loaned_books = 20 ∧ 
    final_books = initial_books - (1 - return_rate) * loaned_books :=
by sorry

end books_loaned_out_l693_69393


namespace surface_area_rotated_sector_l693_69349

/-- The surface area of a solid formed by rotating a circular sector about one of its radii. -/
theorem surface_area_rotated_sector (R θ : ℝ) (h_R : R > 0) (h_θ : 0 < θ ∧ θ < 2 * π) :
  let surface_area := 2 * π * R^2 * Real.sin (θ/2) * (Real.cos (θ/2) + 2 * Real.sin (θ/2))
  ∃ (S : ℝ), S = surface_area ∧ 
    S = (π * (R * Real.sin θ)^2) +  -- Area of the circular base
        (π * R * (R * Real.sin θ)) +  -- Curved surface area of the cone
        (2 * π * R * (R * (1 - Real.cos θ)))  -- Surface area of the spherical cap
  := by sorry

end surface_area_rotated_sector_l693_69349


namespace species_x_count_l693_69302

def ant_farm (x y : ℕ) : Prop :=
  -- Initial total number of ants
  x + y = 50 ∧
  -- Total number of ants on Day 4
  81 * x + 16 * y = 2914

theorem species_x_count : ∃ x y : ℕ, ant_farm x y ∧ 81 * x = 2754 := by
  sorry

end species_x_count_l693_69302


namespace stevens_grapes_l693_69384

def apple_seeds : ℕ := 6
def pear_seeds : ℕ := 2
def grape_seeds : ℕ := 3
def total_seeds_needed : ℕ := 60
def apples_set_aside : ℕ := 4
def pears_set_aside : ℕ := 3
def additional_seeds_needed : ℕ := 3

theorem stevens_grapes (grapes_set_aside : ℕ) : grapes_set_aside = 9 := by
  sorry

#check stevens_grapes

end stevens_grapes_l693_69384


namespace intersection_P_Q_l693_69362

/-- The set P -/
def P : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}

/-- The set Q -/
def Q : Set ℝ := {x : ℝ | |x - 5| < 3}

/-- The open interval (2, 5) -/
def open_interval_2_5 : Set ℝ := {x : ℝ | 2 < x ∧ x < 5}

theorem intersection_P_Q : P ∩ Q = open_interval_2_5 := by sorry

end intersection_P_Q_l693_69362


namespace circle_condition_l693_69328

/-- The equation x^2 + y^2 + 4x - 2y + 5m = 0 represents a circle if and only if m < 1 -/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*x - 2*y + 5*m = 0 ∧ 
   ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 + 4*x - 2*y + 5*m = 0) 
  ↔ m < 1 := by
sorry


end circle_condition_l693_69328


namespace positive_real_floor_product_48_l693_69387

theorem positive_real_floor_product_48 (x : ℝ) :
  x > 0 ∧ x * ⌊x⌋ = 48 → x = 8 :=
by sorry

end positive_real_floor_product_48_l693_69387


namespace games_lost_l693_69301

theorem games_lost (total_games won_games : ℕ) 
  (h1 : total_games = 18) 
  (h2 : won_games = 15) : 
  total_games - won_games = 3 := by
sorry

end games_lost_l693_69301


namespace tangent_line_problem_l693_69358

/-- Given two functions f and g, where f is the natural logarithm and g is a quadratic function with parameter m,
    and a line l tangent to both f and g at the point (1, 0), prove that m = -2. -/
theorem tangent_line_problem (m : ℝ) :
  (m < 0) →
  let f : ℝ → ℝ := λ x ↦ Real.log x
  let g : ℝ → ℝ := λ x ↦ (1/2) * x^2 + m * x + 7/2
  let l : ℝ → ℝ := λ x ↦ x - 1
  (∀ x, deriv f x = 1/x) →
  (∀ x, deriv g x = x + m) →
  (f 1 = 0) →
  (g 1 = l 1) →
  (deriv f 1 = deriv l 1) →
  (∃ x, g x = l x ∧ deriv g x = deriv l x) →
  m = -2 := by
  sorry

end tangent_line_problem_l693_69358


namespace sum_consecutive_products_l693_69347

/-- The sum of products of three consecutive integers from 19 to 2001 -/
def S : ℕ → ℕ
  | 0 => 0
  | n + 1 => (18 + n) * (19 + n) * (20 + n) + S n

/-- The main theorem stating the closed form of the sum -/
theorem sum_consecutive_products (n : ℕ) :
  S (1981) = 6 * (Nat.choose 2002 4 - Nat.choose 21 4) :=
by sorry

end sum_consecutive_products_l693_69347


namespace polynomial_identity_sum_l693_69351

theorem polynomial_identity_sum (d₁ d₂ d₃ e₁ e₂ e₃ : ℝ) : 
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + d₁*x + e₁)*(x^2 + d₂*x + e₂)*(x^2 + d₃*x + e₃)*(x^2 + 1)) →
  d₁*e₁ + d₂*e₂ + d₃*e₃ = -1 := by
sorry

end polynomial_identity_sum_l693_69351


namespace abs_neg_half_eq_half_l693_69398

theorem abs_neg_half_eq_half : |(-1/2 : ℚ)| = 1/2 := by
  sorry

end abs_neg_half_eq_half_l693_69398


namespace samantha_routes_count_l693_69325

/-- Represents a location on a grid -/
structure Location :=
  (x : Int) (y : Int)

/-- Represents Central Park -/
structure CentralPark :=
  (sw : Location) (ne : Location)

/-- Calculates the number of shortest paths between two locations on a grid -/
def gridPaths (start finish : Location) : Nat :=
  let dx := (finish.x - start.x).natAbs
  let dy := (finish.y - start.y).natAbs
  Nat.choose (dx + dy) dx

/-- The number of diagonal paths through Central Park -/
def parkPaths : Nat := 2

/-- Theorem stating the number of shortest routes from Samantha's house to her school -/
theorem samantha_routes_count (park : CentralPark) 
  (home : Location) 
  (school : Location) 
  (home_to_sw : home.x = park.sw.x - 3 ∧ home.y = park.sw.y - 2)
  (school_to_ne : school.x = park.ne.x + 3 ∧ school.y = park.ne.y + 3) :
  gridPaths home park.sw * parkPaths * gridPaths park.ne school = 400 := by
  sorry

end samantha_routes_count_l693_69325


namespace sum_odd_500_to_800_l693_69373

def first_odd_after (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 1 else n + 2

def last_odd_before (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 1 else n - 2

def sum_odd_between (a b : ℕ) : ℕ :=
  let first := first_odd_after a
  let last := last_odd_before b
  let count := (last - first) / 2 + 1
  count * (first + last) / 2

theorem sum_odd_500_to_800 :
  sum_odd_between 500 800 = 97500 := by
  sorry

end sum_odd_500_to_800_l693_69373


namespace factor_of_polynomial_l693_69308

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 + 4*x^2 + 16 : ℝ) = (x^2 + 4) * q x := by
  sorry

end factor_of_polynomial_l693_69308


namespace pear_distribution_count_l693_69371

def family_size : ℕ := 7
def elder_count : ℕ := 4

theorem pear_distribution_count : 
  (elder_count : ℕ) * (Nat.factorial (family_size - 2)) = 480 :=
sorry

end pear_distribution_count_l693_69371


namespace problem_l693_69388

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sin x / Real.cos x + 3

theorem problem (a : ℝ) (h : f (Real.log a) = 4) : f (Real.log (1 / a)) = 2 := by
  sorry

end problem_l693_69388


namespace negation_of_proposition_l693_69382

theorem negation_of_proposition :
  (¬ (∀ x y : ℝ, x^2 + y^2 ≥ 0)) ↔ (∃ x y : ℝ, x^2 + y^2 < 0) := by sorry

end negation_of_proposition_l693_69382


namespace juice_distribution_l693_69342

theorem juice_distribution (container_capacity : ℝ) : 
  container_capacity > 0 → 
  let total_juice := (3 / 4) * container_capacity
  let num_cups := 5
  let juice_per_cup := total_juice / num_cups
  (juice_per_cup / container_capacity) * 100 = 15 := by sorry

end juice_distribution_l693_69342


namespace all_statements_false_l693_69307

theorem all_statements_false :
  (¬ ∀ a b : ℝ, a > b → a^2 > b^2) ∧
  (¬ ∀ a b : ℝ, a^2 > b^2 → a > b) ∧
  (¬ ∀ a b c : ℝ, a > b → a*c^2 > b*c^2) ∧
  (¬ ∀ a b : ℝ, (a > b ↔ |a| > |b|)) :=
by sorry

end all_statements_false_l693_69307


namespace halloween_candy_l693_69380

theorem halloween_candy (debby_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : 
  debby_candy = 32 → sister_candy = 42 → eaten_candy = 35 →
  debby_candy + sister_candy - eaten_candy = 39 := by
sorry

end halloween_candy_l693_69380


namespace abs_m_minus_n_l693_69399

theorem abs_m_minus_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := by
  sorry

end abs_m_minus_n_l693_69399


namespace concatenated_numbers_divisible_by_45_l693_69335

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Definition of concatenating numbers from 1 to n
  sorry

theorem concatenated_numbers_divisible_by_45 :
  ∃ k : ℕ, concatenate_numbers 50 = 45 * k := by
  sorry

end concatenated_numbers_divisible_by_45_l693_69335


namespace deck_size_l693_69334

theorem deck_size (r b : ℕ) : 
  r > 0 → b > 0 →
  r / (r + b) = 1 / 4 →
  r / (r + b + 6) = 1 / 6 →
  r + b = 12 := by
sorry

end deck_size_l693_69334


namespace triangle_area_with_cosine_root_l693_69365

/-- The area of a triangle with two sides of length 3 and 5, where the cosine of the angle between them is a root of 5x^2 - 7x - 6 = 0, is equal to 6. -/
theorem triangle_area_with_cosine_root : ∃ (θ : ℝ), 
  (5 * (Real.cos θ)^2 - 7 * (Real.cos θ) - 6 = 0) →
  (1/2 * 3 * 5 * Real.sin θ = 6) := by
  sorry

end triangle_area_with_cosine_root_l693_69365


namespace triangle_perimeter_l693_69348

noncomputable def line_through_origin (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1}

def vertical_line (x : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = x}

def sloped_line (m : ℝ) (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + b}

def is_equilateral_triangle (t : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ × ℝ), a ∈ t ∧ b ∈ t ∧ c ∈ t ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
    (b.1 - c.1)^2 + (b.2 - c.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2

def perimeter (t : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem triangle_perimeter :
  ∃ (m : ℝ),
    let l1 := line_through_origin m
    let l2 := vertical_line 1
    let l3 := sloped_line (Real.sqrt 3 / 3) 1
    let t := l1 ∪ l2 ∪ l3
    is_equilateral_triangle t ∧ perimeter t = 3 + 2 * Real.sqrt 3 :=
  sorry

end triangle_perimeter_l693_69348


namespace ending_number_of_range_l693_69372

theorem ending_number_of_range : ∃ n : ℕ, 
  (n ≥ 100) ∧ 
  ((200 + 400) / 2 = ((100 + n) / 2) + 150) ∧ 
  (n = 200) := by
  sorry

end ending_number_of_range_l693_69372


namespace intersection_complement_theorem_l693_69306

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {1, 4}

theorem intersection_complement_theorem : A ∩ (U \ B) = {2} := by
  sorry

end intersection_complement_theorem_l693_69306


namespace log_inequality_l693_69309

theorem log_inequality (k : ℝ) (h : k ≥ 3) :
  Real.log k / Real.log (k - 1) > Real.log (k + 1) / Real.log k := by
  sorry

end log_inequality_l693_69309


namespace decreasing_function_implies_a_bound_l693_69383

/-- A function f: ℝ → ℝ is decreasing if for all x, y ∈ ℝ, x < y implies f(x) > f(y) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

/-- The function f(x) = -x³ + x² + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + a*x

theorem decreasing_function_implies_a_bound :
  ∀ a : ℝ, DecreasingFunction (f a) → a ≤ -1/3 := by sorry

end decreasing_function_implies_a_bound_l693_69383


namespace min_distance_to_line_l693_69343

/-- The minimum distance from the origin (0, 0) to the line 2x + y + 5 = 0 is √5 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | 2 * p.1 + p.2 + 5 = 0}
  ∃ d : ℝ, d = Real.sqrt 5 ∧ ∀ p ∈ line, d ≤ Real.sqrt (p.1^2 + p.2^2) := by
  sorry

end min_distance_to_line_l693_69343


namespace non_shaded_area_of_square_with_semicircles_l693_69318

/-- The area of the non-shaded part of a square with side length 4 and eight congruent semicircles --/
theorem non_shaded_area_of_square_with_semicircles :
  let square_side : ℝ := 4
  let num_semicircles : ℕ := 8
  let square_area : ℝ := square_side ^ 2
  let semicircle_radius : ℝ := square_side / 2
  let semicircle_area : ℝ := π * semicircle_radius ^ 2 / 2
  let total_shaded_area : ℝ := num_semicircles * semicircle_area
  let non_shaded_area : ℝ := square_area - total_shaded_area
  non_shaded_area = 8 := by sorry

end non_shaded_area_of_square_with_semicircles_l693_69318


namespace inverse_variation_cube_root_l693_69397

theorem inverse_variation_cube_root (y x : ℝ) (k : ℝ) (h1 : y * x^(1/3) = k) (h2 : 2 * 8^(1/3) = k) :
  8 * x^(1/3) = k → x = 1/8 := by
sorry

end inverse_variation_cube_root_l693_69397


namespace isosceles_triangle_perimeter_l693_69300

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = b ∧ (a = 2 ∧ c = 5 ∨ a = 5 ∧ c = 2) ∨
   a = c ∧ (a = 2 ∧ b = 5 ∨ a = 5 ∧ b = 2) ∨
   b = c ∧ (b = 2 ∧ a = 5 ∨ b = 5 ∧ a = 2)) →  -- isosceles with sides 2 and 5
  a + b + c = 12  -- perimeter is 12
:= by sorry

end isosceles_triangle_perimeter_l693_69300


namespace greatest_value_2q_minus_r_l693_69378

theorem greatest_value_2q_minus_r : 
  ∃ (q r : ℕ+), 
    1027 = 21 * q + r ∧ 
    ∀ (q' r' : ℕ+), 1027 = 21 * q' + r' → 2 * q - r ≥ 2 * q' - r' ∧
    2 * q - r = 77 := by
  sorry

end greatest_value_2q_minus_r_l693_69378


namespace spinner_probability_l693_69369

theorem spinner_probability (p_A p_B p_C : ℚ) : 
  p_A = 1/3 → p_B = 1/2 → p_A + p_B + p_C = 1 → p_C = 1/6 := by
  sorry

end spinner_probability_l693_69369


namespace unread_messages_proof_l693_69367

/-- The number of days it takes to read all messages -/
def days : ℕ := 7

/-- The number of messages read per day -/
def messages_read_per_day : ℕ := 20

/-- The number of new messages received per day -/
def new_messages_per_day : ℕ := 6

/-- The initial number of unread messages -/
def initial_messages : ℕ := days * (messages_read_per_day - new_messages_per_day)

theorem unread_messages_proof :
  initial_messages = 98 := by sorry

end unread_messages_proof_l693_69367


namespace dereks_initial_money_l693_69352

/-- Proves that Derek's initial amount of money was $40 -/
theorem dereks_initial_money :
  ∀ (derek_initial : ℕ) (derek_spent dave_initial dave_spent : ℕ),
  derek_spent = 30 →
  dave_initial = 50 →
  dave_spent = 7 →
  dave_initial - dave_spent = (derek_initial - derek_spent) + 33 →
  derek_initial = 40 := by
  sorry

end dereks_initial_money_l693_69352


namespace cab_journey_time_l693_69315

/-- Given a cab walking at 5/6 of its usual speed and arriving 15 minutes late,
    prove that its usual time to cover the journey is 1.25 hours. -/
theorem cab_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (usual_speed * usual_time = (5/6 * usual_speed) * (usual_time + 1/4)) → 
  usual_time = 5/4 := by
  sorry

end cab_journey_time_l693_69315


namespace initial_cookies_l693_69356

/-- The number of basketball team members -/
def team_members : ℕ := 8

/-- The number of cookies Andy ate -/
def andy_ate : ℕ := 3

/-- The number of cookies Andy gave to his brother -/
def brother_got : ℕ := 5

/-- The number of cookies the first player took -/
def first_player_cookies : ℕ := 1

/-- The increase in cookies taken by each subsequent player -/
def cookie_increase : ℕ := 2

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a₁ aₙ : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- The total number of cookies taken by the basketball team -/
def team_cookies : ℕ :=
  arithmetic_sum team_members first_player_cookies (first_player_cookies + cookie_increase * (team_members - 1))

/-- The theorem stating the initial number of cookies -/
theorem initial_cookies : 
  andy_ate + brother_got + team_cookies = 72 := by sorry

end initial_cookies_l693_69356


namespace complex_fraction_simplification_l693_69357

theorem complex_fraction_simplification :
  (3 + 8 * Complex.I) / (1 - 4 * Complex.I) = -29/17 + 20/17 * Complex.I :=
by sorry

end complex_fraction_simplification_l693_69357


namespace incorrect_fraction_transformation_l693_69355

theorem incorrect_fraction_transformation (a b : ℝ) (hb : b ≠ 0) :
  ¬(∀ (a b : ℝ), b ≠ 0 → |(-a)| / b = a / (-b)) :=
sorry

end incorrect_fraction_transformation_l693_69355


namespace particle_probability_l693_69332

/-- Probability of reaching (0, 0) from (x, y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

/-- The probability of reaching (0, 0) from (6, 6) is 855/3^12 -/
theorem particle_probability : P 6 6 = 855 / 3^12 := by
  sorry

end particle_probability_l693_69332


namespace newton_albert_game_l693_69359

theorem newton_albert_game (a n : ℂ) : 
  a * n = 40 - 24 * I ∧ a = 8 - 4 * I → n = 2.8 - 0.4 * I :=
by sorry

end newton_albert_game_l693_69359


namespace greatest_prime_factor_of_sum_l693_69364

def double_factorial (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

def braced_notation (x : ℕ) : ℕ := double_factorial x

theorem greatest_prime_factor_of_sum (n : ℕ) (h : n = 22) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (braced_notation n + braced_notation (n - 2)) ∧
    ∀ q : ℕ, Nat.Prime q → q ∣ (braced_notation n + braced_notation (n - 2)) → q ≤ p :=
by sorry

end greatest_prime_factor_of_sum_l693_69364


namespace unique_positive_solution_l693_69389

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x^12 + 8*x^11 + 18*x^10 + 2048*x^9 - 1638*x^8 = 0 :=
by sorry

end unique_positive_solution_l693_69389


namespace wanda_walking_distance_l693_69338

/-- The distance Wanda walks to school (in miles) -/
def distance_to_school : ℝ := 0.5

/-- The number of times Wanda walks to and from school per day -/
def trips_per_day : ℕ := 2

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 4

/-- The total distance Wanda walks in miles after the given number of weeks -/
def total_distance : ℝ := 
  distance_to_school * 2 * trips_per_day * school_days_per_week * num_weeks

theorem wanda_walking_distance : total_distance = 40 := by
  sorry

end wanda_walking_distance_l693_69338


namespace gcd_of_90_and_450_l693_69374

theorem gcd_of_90_and_450 : Nat.gcd 90 450 = 90 := by
  sorry

end gcd_of_90_and_450_l693_69374


namespace arithmetic_sequence_first_term_and_difference_l693_69322

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_first_term_and_difference
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ArithmeticSequence a d)
  (h_fifth : a 5 = 10)
  (h_sum : a 1 + a 2 + a 3 = 3) :
  a 1 = -2 ∧ d = 3 := by
  sorry


end arithmetic_sequence_first_term_and_difference_l693_69322


namespace count_special_numbers_l693_69314

/-- Counts the number of four-digit numbers with digit sum 12 that are divisible by 9 -/
def countSpecialNumbers : ℕ :=
  (Finset.range 9).sum fun a =>
    Nat.choose (14 - (a + 1)) 2

/-- The count of four-digit numbers with digit sum 12 that are divisible by 9 is 354 -/
theorem count_special_numbers : countSpecialNumbers = 354 := by
  sorry

end count_special_numbers_l693_69314


namespace tromino_coverage_l693_69368

/-- Represents a tromino (L-shaped piece formed from three squares) --/
structure Tromino

/-- Represents a chessboard --/
structure Chessboard (n : ℕ) where
  size : n ≥ 7
  odd : Odd n

/-- Counts the number of black squares on the chessboard --/
def black_squares (n : ℕ) : ℕ := (n^2 + 1) / 2

/-- Counts the minimum number of trominoes required to cover all black squares --/
def min_trominoes (n : ℕ) : ℕ := (n + 1)^2 / 4

/-- Theorem stating the minimum number of trominoes required to cover all black squares --/
theorem tromino_coverage (n : ℕ) (board : Chessboard n) :
  min_trominoes n = black_squares n := by sorry

end tromino_coverage_l693_69368


namespace geometric_progression_ratio_l693_69346

theorem geometric_progression_ratio (x y z r : ℂ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ∃ (a : ℂ), x * (y - z) = a ∧ y * (z - x) = a * r ∧ z * (x - y) = a * r^2 →
  x + y + z = 0 →
  r^2 + r + 1 = 0 :=
by sorry

end geometric_progression_ratio_l693_69346


namespace scooter_cost_calculation_l693_69329

theorem scooter_cost_calculation (original_cost : ℝ) 
  (repair1_percent repair2_percent repair3_percent tax_percent : ℝ)
  (discount_percent profit_percent : ℝ) (profit : ℝ) :
  repair1_percent = 0.05 →
  repair2_percent = 0.10 →
  repair3_percent = 0.07 →
  tax_percent = 0.12 →
  discount_percent = 0.15 →
  profit_percent = 0.30 →
  profit = 2000 →
  profit = profit_percent * original_cost →
  let total_spent := original_cost * (1 + repair1_percent + repair2_percent + repair3_percent + tax_percent)
  total_spent = 1.34 * original_cost :=
by sorry

end scooter_cost_calculation_l693_69329


namespace grant_baseball_gear_sale_l693_69361

/-- The total money Grant made from selling his baseball gear -/
def total_money (card_price bat_price glove_original_price glove_discount cleats_price cleats_count : ℝ) : ℝ :=
  card_price + bat_price + (glove_original_price * (1 - glove_discount)) + (cleats_price * cleats_count)

/-- Theorem stating that Grant made $79 from selling his baseball gear -/
theorem grant_baseball_gear_sale :
  total_money 25 10 30 0.2 10 2 = 79 := by
  sorry

end grant_baseball_gear_sale_l693_69361


namespace scientific_notation_of_1_35_billion_l693_69381

theorem scientific_notation_of_1_35_billion :
  ∃ (a : ℝ) (n : ℤ), 1.35e9 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 :=
by
  sorry

end scientific_notation_of_1_35_billion_l693_69381


namespace complex_fraction_evaluation_l693_69363

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 - a*b + b^2 = 0) : 
  (a^7 + b^7) / (a - b)^7 = 2 := by
  sorry

end complex_fraction_evaluation_l693_69363


namespace complement_union_theorem_l693_69319

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 3, 5}
def Q : Set Nat := {1, 2, 4}

theorem complement_union_theorem :
  (U \ P) ∪ Q = {1, 2, 4, 6} := by
  sorry

end complement_union_theorem_l693_69319


namespace intersection_complement_equality_l693_69386

-- Define the universal set U
def U : Finset ℕ := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset ℕ := {1, 4}

-- Define set N
def N : Finset ℕ := {1, 3, 5}

-- Theorem statement
theorem intersection_complement_equality :
  N ∩ (U \ M) = {3, 5} := by sorry

end intersection_complement_equality_l693_69386


namespace soda_ratio_l693_69360

/-- Proves that the ratio of regular sodas to diet sodas is 9:7 -/
theorem soda_ratio (total_sodas : ℕ) (diet_sodas : ℕ) : 
  total_sodas = 64 → diet_sodas = 28 → 
  (total_sodas - diet_sodas : ℚ) / diet_sodas = 9 / 7 := by
  sorry

end soda_ratio_l693_69360


namespace sin_plus_two_cos_alpha_l693_69366

theorem sin_plus_two_cos_alpha (α : Real) :
  (∃ x y : Real, x = -3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end sin_plus_two_cos_alpha_l693_69366


namespace ticket_price_possibilities_l693_69370

theorem ticket_price_possibilities : ∃ (divisors : Finset ℕ), 
  (∀ x ∈ divisors, x ∣ 60 ∧ x ∣ 90) ∧ 
  (∀ x : ℕ, x ∣ 60 ∧ x ∣ 90 → x ∈ divisors) ∧
  Finset.card divisors = 8 :=
sorry

end ticket_price_possibilities_l693_69370


namespace solution_set_f_gt_7_minus_x_range_of_m_for_solution_existence_l693_69391

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| * |x - 3|

-- Part 1: Solution set of f(x) > 7-x
theorem solution_set_f_gt_7_minus_x :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} := by sorry

-- Part 2: Range of m for which f(x) ≤ |3m-2| has a solution
theorem range_of_m_for_solution_existence :
  {m : ℝ | ∃ x, f x ≤ |3*m - 2|} = {m : ℝ | m ≤ -1 ∨ m ≥ 7/3} := by sorry

end solution_set_f_gt_7_minus_x_range_of_m_for_solution_existence_l693_69391


namespace quadratic_monotonicity_l693_69310

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2*a - 1)*x + a + 1

-- Define monotonicity in an interval
def monotonic_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ ∀ z, a < z ∧ z < b → f z = f x)

-- State the theorem
theorem quadratic_monotonicity (a : ℝ) :
  monotonic_in (f a) 1 2 → (a ≥ 5/2 ∨ a ≤ 3/2) :=
by sorry

end quadratic_monotonicity_l693_69310


namespace fraction_inequality_l693_69341

theorem fraction_inequality (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end fraction_inequality_l693_69341


namespace quadratic_equation_solutions_l693_69345

theorem quadratic_equation_solutions : ∀ x : ℝ,
  (2 * x^2 + 7 * x - 1 = 4 * x + 1 ↔ x = -2 ∨ x = 1/2) ∧
  (2 * x^2 + 7 * x - 1 = -(x^2 - 19) ↔ x = -4 ∨ x = 5/3) := by
  sorry

end quadratic_equation_solutions_l693_69345


namespace shortest_diagonal_probability_l693_69396

/-- The number of sides in the regular polygon -/
def n : ℕ := 20

/-- The total number of diagonals in the polygon -/
def total_diagonals : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in the polygon -/
def shortest_diagonals : ℕ := n

/-- The probability of selecting a shortest diagonal -/
def probability : ℚ := shortest_diagonals / total_diagonals

theorem shortest_diagonal_probability :
  probability = 2 / 17 := by sorry

end shortest_diagonal_probability_l693_69396


namespace mod_fifteen_equivalence_l693_69327

theorem mod_fifteen_equivalence (n : ℤ) : 
  0 ≤ n ∧ n ≤ 14 ∧ n ≡ 15827 [ZMOD 15] → n = 2 := by
  sorry

end mod_fifteen_equivalence_l693_69327


namespace square_of_integer_l693_69331

theorem square_of_integer (n : ℕ+) (h : ∃ (m : ℤ), m = 2 + 2 * Int.sqrt (28 * n.val^2 + 1)) :
  ∃ (k : ℤ), (2 + 2 * Int.sqrt (28 * n.val^2 + 1)) = k^2 := by
  sorry

end square_of_integer_l693_69331


namespace reflection_across_y_axis_l693_69340

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that the reflection of (1, 6) across the y-axis is (-1, 6) -/
theorem reflection_across_y_axis :
  let original := Point.mk 1 6
  reflect_y original = Point.mk (-1) 6 := by
  sorry

end reflection_across_y_axis_l693_69340


namespace unique_distribution_function_decomposition_l693_69304

/-- A distribution function -/
class DistributionFunction (F : ℝ → ℝ) : Prop where
  -- Add necessary axioms for a distribution function

/-- A discrete distribution function -/
class DiscreteDistributionFunction (F : ℝ → ℝ) extends DistributionFunction F : Prop where
  -- Add necessary axioms for a discrete distribution function

/-- An absolutely continuous distribution function -/
class AbsContinuousDistributionFunction (F : ℝ → ℝ) extends DistributionFunction F : Prop where
  -- Add necessary axioms for an absolutely continuous distribution function

/-- A singular distribution function -/
class SingularDistributionFunction (F : ℝ → ℝ) extends DistributionFunction F : Prop where
  -- Add necessary axioms for a singular distribution function

/-- The uniqueness of distribution function decomposition -/
theorem unique_distribution_function_decomposition
  (F : ℝ → ℝ) [DistributionFunction F] :
  ∃! (α₁ α₂ α₃ : ℝ) (Fₐ Fₐbc Fsc : ℝ → ℝ),
    α₁ ≥ 0 ∧ α₂ ≥ 0 ∧ α₃ ≥ 0 ∧
    α₁ + α₂ + α₃ = 1 ∧
    DiscreteDistributionFunction Fₐ ∧
    AbsContinuousDistributionFunction Fₐbc ∧
    SingularDistributionFunction Fsc ∧
    F = λ x => α₁ * Fₐ x + α₂ * Fₐbc x + α₃ * Fsc x :=
by sorry

end unique_distribution_function_decomposition_l693_69304


namespace rectangular_prism_diagonals_l693_69316

/-- A rectangular prism with its properties -/
structure RectangularPrism where
  vertices : Nat
  edges : Nat
  dimensions : Nat
  has_face_diagonals : Bool
  has_space_diagonals : Bool

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (prism : RectangularPrism) : Nat :=
  sorry

/-- Theorem stating that the total number of diagonals in a rectangular prism is 16 -/
theorem rectangular_prism_diagonals :
  ∀ (prism : RectangularPrism),
    prism.vertices = 8 ∧
    prism.edges = 12 ∧
    prism.dimensions = 3 ∧
    prism.has_face_diagonals = true ∧
    prism.has_space_diagonals = true →
    total_diagonals prism = 16 :=
  sorry

end rectangular_prism_diagonals_l693_69316


namespace nine_knights_in_room_l693_69321

/-- Represents a person on the island, either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- The total number of people in the room -/
def totalPeople : Nat := 15

/-- Represents the statements made by each person -/
structure Statements where
  sixLiars : Bool  -- "Among my acquaintances in this room, there are exactly six liars"
  noMoreThanSevenKnights : Bool  -- "Among my acquaintances in this room, there are no more than seven knights"

/-- Returns true if the statements are consistent with the person's type and the room's composition -/
def statementsAreConsistent (p : Person) (statements : Statements) (knightCount : Nat) : Bool :=
  match p with
  | Person.Knight => statements.sixLiars = (totalPeople - knightCount - 1 = 6) ∧ 
                     statements.noMoreThanSevenKnights = (knightCount - 1 ≤ 7)
  | Person.Liar => statements.sixLiars ≠ (totalPeople - knightCount - 1 = 6) ∧ 
                   statements.noMoreThanSevenKnights ≠ (knightCount - 1 ≤ 7)

/-- The main theorem: there are exactly 9 knights in the room -/
theorem nine_knights_in_room : 
  ∃ (knightCount : Nat), knightCount = 9 ∧ 
  (∀ (p : Person) (s : Statements), statementsAreConsistent p s knightCount) ∧
  knightCount + (totalPeople - knightCount) = totalPeople :=
sorry

end nine_knights_in_room_l693_69321


namespace min_value_theorem_l693_69317

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  (x + 4) / Real.sqrt (x - 2) ≥ 2 * Real.sqrt 6 ∧
  ∃ y : ℝ, y > 2 ∧ (y + 4) / Real.sqrt (y - 2) = 2 * Real.sqrt 6 :=
sorry

end min_value_theorem_l693_69317


namespace min_value_expression_l693_69395

open Real

theorem min_value_expression (α β : ℝ) (h : α + β = π / 2) :
  (∀ x y : ℝ, (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 65) ∧
  (∃ x y : ℝ, (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 12)^2 = 65) :=
by sorry

end min_value_expression_l693_69395


namespace smallest_n_multiple_of_seven_l693_69336

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (hx : (x + 2) % 7 = 0) 
  (hy : (y - 2) % 7 = 0) : 
  ∃ n : ℕ+, (∀ m : ℕ+, (x^2 + x*y + y^2 + m) % 7 = 0 → n ≤ m) ∧ 
            (x^2 + x*y + y^2 + n) % 7 = 0 ∧ 
            n = 3 := by
  sorry

end smallest_n_multiple_of_seven_l693_69336
