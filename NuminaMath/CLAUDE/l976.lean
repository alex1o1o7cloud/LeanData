import Mathlib

namespace NUMINAMATH_CALUDE_parabola_contradiction_l976_97664

theorem parabola_contradiction (a b c : ℝ) : 
  ¬(((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c > 0)) ∧
    ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0))) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_contradiction_l976_97664


namespace NUMINAMATH_CALUDE_zero_sequence_arithmetic_not_geometric_l976_97690

-- Define the sequence
def a : ℕ → ℝ
  | _ => 0

-- Theorem statement
theorem zero_sequence_arithmetic_not_geometric :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧ 
  ¬(∀ n m : ℕ, a n ≠ 0 → a (n + 1) / a n = a (m + 1) / a m) :=
by sorry

end NUMINAMATH_CALUDE_zero_sequence_arithmetic_not_geometric_l976_97690


namespace NUMINAMATH_CALUDE_alissa_earring_ratio_l976_97640

/-- The ratio of Alissa's total earrings to the number of earrings she was given -/
def earring_ratio (barbie_pairs : ℕ) (alissa_total : ℕ) : ℚ :=
  let barbie_total := 2 * barbie_pairs
  let alissa_given := barbie_total / 2
  alissa_total / alissa_given

/-- Theorem stating the ratio of Alissa's total earrings to the number of earrings she was given -/
theorem alissa_earring_ratio :
  let barbie_pairs := 12
  let alissa_total := 36
  earring_ratio barbie_pairs alissa_total = 3 := by
  sorry

end NUMINAMATH_CALUDE_alissa_earring_ratio_l976_97640


namespace NUMINAMATH_CALUDE_high_school_total_students_l976_97637

/-- Represents a high school with three grades -/
structure HighSchool :=
  (freshman_count : ℕ)
  (sophomore_count : ℕ)
  (senior_count : ℕ)

/-- Represents a stratified sample from the high school -/
structure StratifiedSample :=
  (freshman_sample : ℕ)
  (sophomore_sample : ℕ)
  (senior_sample : ℕ)

/-- The total number of students in the high school -/
def total_students (hs : HighSchool) : ℕ :=
  hs.freshman_count + hs.sophomore_count + hs.senior_count

/-- The total number of students in the sample -/
def total_sample (s : StratifiedSample) : ℕ :=
  s.freshman_sample + s.sophomore_sample + s.senior_sample

theorem high_school_total_students 
  (hs : HighSchool) 
  (sample : StratifiedSample) 
  (h1 : hs.freshman_count = 400)
  (h2 : sample.sophomore_sample = 15)
  (h3 : sample.senior_sample = 10)
  (h4 : total_sample sample = 45) :
  total_students hs = 900 :=
sorry

end NUMINAMATH_CALUDE_high_school_total_students_l976_97637


namespace NUMINAMATH_CALUDE_cauchy_functional_equation_l976_97615

theorem cauchy_functional_equation 
  (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + y) = f x + f y) : 
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_cauchy_functional_equation_l976_97615


namespace NUMINAMATH_CALUDE_lenkas_numbers_l976_97623

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def both_digits_even (n : ℕ) : Prop :=
  is_two_digit n ∧ n % 2 = 0 ∧ (n / 10) % 2 = 0

def both_digits_odd (n : ℕ) : Prop :=
  is_two_digit n ∧ n % 2 = 1 ∧ (n / 10) % 2 = 1

def sum_has_even_odd_digits (n : ℕ) : Prop :=
  is_two_digit n ∧ (n / 10) % 2 = 0 ∧ n % 2 = 1

theorem lenkas_numbers :
  ∀ a b : ℕ,
    both_digits_even a →
    both_digits_odd b →
    sum_has_even_odd_digits (a + b) →
    a % 3 = 0 →
    b % 3 = 0 →
    (a % 10 = 9 ∨ b % 10 = 9 ∨ (a + b) % 10 = 9) →
    ((a = 24 ∧ b = 39) ∨ (a = 42 ∧ b = 39) ∨ (a = 48 ∧ b = 39)) :=
by sorry

end NUMINAMATH_CALUDE_lenkas_numbers_l976_97623


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l976_97667

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (prod_sum_eq : a * b + a * c + b * c = 5) 
  (prod_eq : a * b * c = -12) : 
  a^3 + b^3 + c^3 = 90 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l976_97667


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l976_97646

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 5*x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l976_97646


namespace NUMINAMATH_CALUDE_solution_sum_l976_97642

-- Define the solution set
def SolutionSet : Set ℝ := Set.union (Set.Iio 1) (Set.Ioi 4)

-- Define the theorem
theorem solution_sum (a b : ℝ) 
  (h : ∀ x, x ∈ SolutionSet ↔ (x - a) / (x - b) > 0) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_solution_sum_l976_97642


namespace NUMINAMATH_CALUDE_turtle_problem_l976_97662

theorem turtle_problem (initial_turtles : ℕ) (h1 : initial_turtles = 25) :
  let additional_turtles := 5 * initial_turtles - 4
  let total_turtles := initial_turtles + additional_turtles
  let remaining_turtles := total_turtles - (total_turtles / 3)
  remaining_turtles = 98 := by
sorry

end NUMINAMATH_CALUDE_turtle_problem_l976_97662


namespace NUMINAMATH_CALUDE_triangle_problem_l976_97655

theorem triangle_problem (a b c A B C : Real) (h1 : b * (Real.sin B - Real.sin C) = a * Real.sin A - c * Real.sin C)
  (h2 : a = 2 * Real.sqrt 3) (h3 : (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3) :
  A = π/3 ∧ a + b + c = 2 * Real.sqrt 3 + 6 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l976_97655


namespace NUMINAMATH_CALUDE_teena_speed_calculation_l976_97601

/-- Teena's speed in miles per hour -/
def teena_speed : ℝ := 55

/-- Roe's speed in miles per hour -/
def roe_speed : ℝ := 40

/-- Initial distance Teena is behind Roe in miles -/
def initial_distance_behind : ℝ := 7.5

/-- Time elapsed in hours -/
def time_elapsed : ℝ := 1.5

/-- Final distance Teena is ahead of Roe in miles -/
def final_distance_ahead : ℝ := 15

theorem teena_speed_calculation :
  teena_speed * time_elapsed = 
    roe_speed * time_elapsed + initial_distance_behind + final_distance_ahead := by
  sorry

#check teena_speed_calculation

end NUMINAMATH_CALUDE_teena_speed_calculation_l976_97601


namespace NUMINAMATH_CALUDE_smallest_t_for_complete_circle_l976_97686

/-- The smallest value of t such that when r = sin θ is plotted for 0 ≤ θ ≤ t,
    the resulting graph represents the entire circle is π. -/
theorem smallest_t_for_complete_circle : 
  ∃ t : ℝ, t > 0 ∧ 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = Real.sin θ) ∧
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = Real.sin θ) ∧
  (∀ t' : ℝ, t' < t → 
    ∃ x y : ℝ, x^2 + y^2 ≤ 1 ∧ 
    ∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t' → x ≠ (Real.sin θ) * (Real.cos θ) ∨ y ≠ Real.sin θ)) ∧
  t = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_smallest_t_for_complete_circle_l976_97686


namespace NUMINAMATH_CALUDE_parallel_line_distance_in_circle_l976_97600

/-- Given a circle intersected by four equally spaced parallel lines creating chords of lengths 44, 44, 40, and 40, the distance between two adjacent parallel lines is 8/√23. -/
theorem parallel_line_distance_in_circle : ∀ (r : ℝ) (d : ℝ),
  (44 + (1/4) * d^2 = r^2) →
  (40 + (27/16) * d^2 = r^2) →
  d = 8 / Real.sqrt 23 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_distance_in_circle_l976_97600


namespace NUMINAMATH_CALUDE_cans_difference_l976_97628

/-- The number of cans Sarah collected yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of additional cans Lara collected compared to Sarah yesterday -/
def lara_extra_yesterday : ℕ := 30

/-- The number of cans Sarah collected today -/
def sarah_today : ℕ := 40

/-- The number of cans Lara collected today -/
def lara_today : ℕ := 70

/-- Theorem stating the difference in total cans collected between yesterday and today -/
theorem cans_difference : 
  (sarah_yesterday + (sarah_yesterday + lara_extra_yesterday)) - (sarah_today + lara_today) = 20 :=
by sorry

end NUMINAMATH_CALUDE_cans_difference_l976_97628


namespace NUMINAMATH_CALUDE_equation_solution_l976_97611

theorem equation_solution : ∃ x : ℚ, x ≠ 0 ∧ (3 / x - (3 / x) / (9 / x) = 1 / 2) ∧ x = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l976_97611


namespace NUMINAMATH_CALUDE_max_red_bulbs_in_garland_red_bulbs_33_possible_l976_97632

/-- Represents a Christmas garland with red and blue bulbs. -/
structure Garland where
  total_bulbs : ℕ
  red_bulbs : ℕ
  blue_bulbs : ℕ
  adjacent_blue : red_bulbs > 0 → blue_bulbs > 0
  sum_bulbs : red_bulbs + blue_bulbs = total_bulbs

/-- Theorem stating the maximum number of red bulbs in a 50-bulb garland. -/
theorem max_red_bulbs_in_garland :
  ∀ g : Garland, g.total_bulbs = 50 → g.red_bulbs ≤ 33 := by
  sorry

/-- Theorem stating that 33 red bulbs is achievable in a 50-bulb garland. -/
theorem red_bulbs_33_possible :
  ∃ g : Garland, g.total_bulbs = 50 ∧ g.red_bulbs = 33 := by
  sorry

end NUMINAMATH_CALUDE_max_red_bulbs_in_garland_red_bulbs_33_possible_l976_97632


namespace NUMINAMATH_CALUDE_vector_parallel_problem_l976_97634

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, 2]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ v = fun i => c * w i

theorem vector_parallel_problem :
  ∃ (k : ℝ), parallel
    (fun i => 2 * (a i) + b i)
    (fun i => (1/2) * (a i) + k * (b i)) ∧
    k = 1/4 := by sorry

end NUMINAMATH_CALUDE_vector_parallel_problem_l976_97634


namespace NUMINAMATH_CALUDE_brian_tennis_balls_l976_97607

/-- Given the number of tennis balls for Lily, Frodo, and Brian, prove that Brian has 22 tennis balls. -/
theorem brian_tennis_balls (lily frodo brian : ℕ) 
  (h1 : lily = 3)
  (h2 : frodo = lily + 8)
  (h3 : brian = 2 * frodo) :
  brian = 22 := by
  sorry

end NUMINAMATH_CALUDE_brian_tennis_balls_l976_97607


namespace NUMINAMATH_CALUDE_solve_kitchen_supplies_l976_97663

def kitchen_supplies_problem (angela_pots : ℕ) (angela_plates : ℕ) (angela_cutlery : ℕ) 
  (sharon_total : ℕ) : Prop :=
  angela_pots = 20 ∧
  angela_plates > 3 * angela_pots ∧
  angela_cutlery = angela_plates / 2 ∧
  sharon_total = 254 ∧
  sharon_total = angela_pots / 2 + (3 * angela_plates - 20) + 2 * angela_cutlery ∧
  angela_plates - 3 * angela_pots = 6

theorem solve_kitchen_supplies : 
  ∃ (angela_pots angela_plates angela_cutlery : ℕ),
    kitchen_supplies_problem angela_pots angela_plates angela_cutlery 254 :=
sorry

end NUMINAMATH_CALUDE_solve_kitchen_supplies_l976_97663


namespace NUMINAMATH_CALUDE_seating_arrangements_equals_60_l976_97674

/-- The number of ways to arrange 3 people in a row of 9 seats,
    with empty seats on both sides of each person. -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  let gaps := total_seats - people - people
  let combinations := Nat.choose gaps people
  combinations * Nat.factorial people

/-- Theorem stating that the number of seating arrangements
    for 3 people in 9 seats with required spacing is 60. -/
theorem seating_arrangements_equals_60 :
  seating_arrangements 9 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_equals_60_l976_97674


namespace NUMINAMATH_CALUDE_circle_reflection_minimum_l976_97618

/-- Given a circle and a line, if reflection about the line keeps points on the circle,
    then there's a minimum value for a certain expression involving the line's parameters. -/
theorem circle_reflection_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 → 
   ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 1 = 0 ∧ 
              ((x + x')/2, (y + y')/2) ∈ {(x, y) | 2*a*x - b*y + 2 = 0}) →
  (∃ m : ℝ, m = 1/a + 2/b ∧ ∀ k : ℝ, k = 1/a + 2/b → m ≤ k) →
  1/a + 2/b = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_circle_reflection_minimum_l976_97618


namespace NUMINAMATH_CALUDE_root_in_interval_l976_97636

def f (x : ℝ) := 3*x^2 + 3*x - 8

theorem root_in_interval :
  (∃ x ∈ Set.Ioo 1 2, f x = 0) →
  (f 1 < 0) →
  (f 1.5 > 0) →
  (f 1.25 < 0) →
  ∃ x ∈ Set.Ioo 1.25 1.5, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l976_97636


namespace NUMINAMATH_CALUDE_fraction_comparison_l976_97683

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a / b < c / d) 
  (h2 : b > d) 
  (h3 : d > 0) : 
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l976_97683


namespace NUMINAMATH_CALUDE_smallest_number_l976_97670

theorem smallest_number (a b c d : ℝ) :
  a = 1 ∧ b = 0 ∧ c = -Real.sqrt 3 ∧ d = -Real.sqrt 2 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l976_97670


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l976_97687

theorem number_puzzle_solution : 
  ∃ x : ℚ, 3 * (2 * x + 7) = 99 ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l976_97687


namespace NUMINAMATH_CALUDE_product_pattern_l976_97650

theorem product_pattern (n : ℕ) : (10 * n + 3) * (10 * n + 7) = 100 * n * (n + 1) + 21 := by
  sorry

end NUMINAMATH_CALUDE_product_pattern_l976_97650


namespace NUMINAMATH_CALUDE_watch_time_loss_l976_97610

/-- Represents the number of minutes lost by a watch per day -/
def minutes_lost_per_day : ℚ := 13/4

/-- Represents the number of hours between 1 P.M. on March 15 and 3 P.M. on March 22 -/
def hours_passed : ℕ := 7 * 24 + 2

/-- Theorem stating that the watch loses 221/96 minutes over the given period -/
theorem watch_time_loss : 
  (minutes_lost_per_day * (hours_passed : ℚ) / 24) = 221/96 := by sorry

end NUMINAMATH_CALUDE_watch_time_loss_l976_97610


namespace NUMINAMATH_CALUDE_marks_speed_l976_97697

/-- Given a distance of 24 miles and a time of 4 hours, the speed is 6 miles per hour. -/
theorem marks_speed (distance : ℝ) (time : ℝ) (h1 : distance = 24) (h2 : time = 4) :
  distance / time = 6 := by
  sorry

end NUMINAMATH_CALUDE_marks_speed_l976_97697


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l976_97658

theorem complex_fraction_equality : (5 * Complex.I) / (1 - 2 * Complex.I) = -2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l976_97658


namespace NUMINAMATH_CALUDE_kevin_cards_l976_97633

/-- The number of cards Kevin ends with given his initial cards and found cards -/
def total_cards (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Kevin ends with 54 cards -/
theorem kevin_cards : total_cards 7 47 = 54 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l976_97633


namespace NUMINAMATH_CALUDE_triangle_area_and_square_coverage_l976_97699

/-- Given a triangle with side lengths 9, 40, and 41, prove its area and the fraction it covers of a square with side length 41. -/
theorem triangle_area_and_square_coverage :
  ∃ (triangle_area : ℝ) (square_area : ℝ) (coverage_fraction : ℚ),
    triangle_area = 180 ∧
    square_area = 41 ^ 2 ∧
    coverage_fraction = 180 / 1681 ∧
    (9 : ℝ) ^ 2 + 40 ^ 2 = 41 ^ 2 ∧
    triangle_area = (1 / 2 : ℝ) * 9 * 40 ∧
    coverage_fraction = triangle_area / square_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_and_square_coverage_l976_97699


namespace NUMINAMATH_CALUDE_sally_quarters_l976_97681

theorem sally_quarters (x : ℕ) : 
  (x + 418 = 1178) → (x = 760) := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l976_97681


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l976_97665

theorem largest_solution_of_equation (y : ℝ) :
  (3 * y^2 + 18 * y - 90 = y * (y + 17)) →
  y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l976_97665


namespace NUMINAMATH_CALUDE_part_I_part_II_l976_97673

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- Part I
theorem part_I (a : ℝ) (h : a = 3) :
  (A ∪ B a = {x | 1 ≤ x ∧ x ≤ 5}) ∧
  (B a ∩ (Set.univ \ A) = {x | 4 < x ∧ x ≤ 5}) := by
  sorry

-- Part II
theorem part_II (a : ℝ) :
  B a ⊆ A ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_part_I_part_II_l976_97673


namespace NUMINAMATH_CALUDE_superinverse_value_l976_97639

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 9*x^2 + 27*x + 81

-- State that g is bijective
axiom g_bijective : Function.Bijective g

-- Define the superinverse property
def is_superinverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, (f ∘ g) x = Function.invFun g x

-- State that f is the superinverse of g
axiom f_is_superinverse : ∃ f : ℝ → ℝ, is_superinverse f g

-- The theorem to prove
theorem superinverse_value :
  ∃ f : ℝ → ℝ, is_superinverse f g ∧ |f (-289)| = 10 := by
  sorry

end NUMINAMATH_CALUDE_superinverse_value_l976_97639


namespace NUMINAMATH_CALUDE_josh_candy_count_l976_97612

def candy_problem (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) (shared_candies : ℕ) : ℕ :=
  let remaining_after_siblings := initial_candies - siblings * candies_per_sibling
  let remaining_after_friend := remaining_after_siblings / 2
  remaining_after_friend - shared_candies

theorem josh_candy_count : candy_problem 100 3 10 19 = 16 := by
  sorry

end NUMINAMATH_CALUDE_josh_candy_count_l976_97612


namespace NUMINAMATH_CALUDE_triplets_equal_sum_l976_97631

/-- The number of ordered triplets (m, n, p) of nonnegative integers satisfying m + 3n + 5p ≤ 600 -/
def countTriplets : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + 3 * t.2.1 + 5 * t.2.2 ≤ 600) (Finset.product (Finset.range 601) (Finset.product (Finset.range 201) (Finset.range 121)))).card

/-- The sum of (i+1) for all nonnegative integer solutions of i + 3j + 5k = 600 -/
def sumSolutions : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + 3 * t.2.1 + 5 * t.2.2 = 600) (Finset.product (Finset.range 601) (Finset.product (Finset.range 201) (Finset.range 121)))).sum (fun t => t.1 + 1)

theorem triplets_equal_sum : countTriplets = sumSolutions := by
  sorry

end NUMINAMATH_CALUDE_triplets_equal_sum_l976_97631


namespace NUMINAMATH_CALUDE_circle_area_tripled_l976_97660

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (1 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l976_97660


namespace NUMINAMATH_CALUDE_swordfish_pufferfish_ratio_l976_97659

/-- The ratio of swordfish to pufferfish in an aquarium -/
theorem swordfish_pufferfish_ratio 
  (total_fish : ℕ) 
  (pufferfish : ℕ) 
  (n : ℕ) 
  (h1 : total_fish = 90)
  (h2 : pufferfish = 15)
  (h3 : total_fish = n * pufferfish + pufferfish) :
  (n * pufferfish) / pufferfish = 5 := by
sorry

end NUMINAMATH_CALUDE_swordfish_pufferfish_ratio_l976_97659


namespace NUMINAMATH_CALUDE_range_of_c_l976_97695

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 1 / a + 1 / b = 1) (h2 : 1 / (a + b) + 1 / c = 1) :
  1 < c ∧ c ≤ 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l976_97695


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_always_positive_range_l976_97677

def f (k : ℝ) (x : ℝ) : ℝ := x^2 + 2*k*x + 4

-- Part 1
theorem monotonic_increasing_range (k : ℝ) :
  (∀ x ∈ Set.Icc 1 4, Monotone (f k)) ↔ k ≥ -1 :=
sorry

-- Part 2
theorem always_positive_range (k : ℝ) :
  (∀ x : ℝ, f k x > 0) ↔ -2 < k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_always_positive_range_l976_97677


namespace NUMINAMATH_CALUDE_tuesday_poodles_count_l976_97624

/-- Represents the number of hours Charlotte can walk dogs on a weekday -/
def weekday_hours : ℕ := 8

/-- Represents the number of hours Charlotte can walk dogs on a weekend day -/
def weekend_hours : ℕ := 4

/-- Represents the number of weekdays in a week -/
def weekdays : ℕ := 5

/-- Represents the number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- Represents the time it takes to walk a poodle -/
def poodle_time : ℕ := 2

/-- Represents the time it takes to walk a Chihuahua -/
def chihuahua_time : ℕ := 1

/-- Represents the time it takes to walk a Labrador -/
def labrador_time : ℕ := 3

/-- Represents the time it takes to walk a Golden Retriever -/
def golden_retriever_time : ℕ := 4

/-- Represents the number of poodles walked on Monday -/
def monday_poodles : ℕ := 4

/-- Represents the number of Chihuahuas walked on Monday and Tuesday -/
def monday_tuesday_chihuahuas : ℕ := 2

/-- Represents the number of Golden Retrievers walked on Monday -/
def monday_golden_retrievers : ℕ := 1

/-- Represents the number of Labradors walked on Wednesday -/
def wednesday_labradors : ℕ := 4

/-- Represents the number of Golden Retrievers walked on Tuesday -/
def tuesday_golden_retrievers : ℕ := 1

theorem tuesday_poodles_count :
  ∃ (tuesday_poodles : ℕ),
    tuesday_poodles = 1 ∧
    weekday_hours * weekdays + weekend_hours * weekend_days ≥
      (monday_poodles * poodle_time +
       monday_tuesday_chihuahuas * chihuahua_time +
       monday_golden_retrievers * golden_retriever_time) +
      (tuesday_poodles * poodle_time +
       monday_tuesday_chihuahuas * chihuahua_time +
       tuesday_golden_retrievers * golden_retriever_time) +
      (wednesday_labradors * labrador_time) :=
by sorry

end NUMINAMATH_CALUDE_tuesday_poodles_count_l976_97624


namespace NUMINAMATH_CALUDE_smallest_four_digit_solution_l976_97619

theorem smallest_four_digit_solution (x : ℕ) : x = 1053 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧ 
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (9 * y ≡ 27 [ZMOD 15] ∧
     3 * y + 15 ≡ 21 [ZMOD 8] ∧
     -3 * y + 4 ≡ 2 * y + 5 [ZMOD 16]) →
    x ≤ y) ∧
  (9 * x ≡ 27 [ZMOD 15]) ∧
  (3 * x + 15 ≡ 21 [ZMOD 8]) ∧
  (-3 * x + 4 ≡ 2 * x + 5 [ZMOD 16]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_solution_l976_97619


namespace NUMINAMATH_CALUDE_cameron_wins_probability_l976_97679

-- Define the faces of each cube
def cameron_cube : Finset Nat := {6}
def dean_cube : Finset Nat := {1, 2, 3}
def olivia_cube : Finset Nat := {3, 6}

-- Define the number of faces for each number on each cube
def cameron_faces (n : Nat) : Nat := if n = 6 then 6 else 0
def dean_faces (n : Nat) : Nat := if n ∈ dean_cube then 2 else 0
def olivia_faces (n : Nat) : Nat := if n = 3 then 4 else if n = 6 then 2 else 0

-- Define the probability of rolling less than 6 for each player
def dean_prob_less_than_6 : ℚ :=
  (dean_faces 1 + dean_faces 2 + dean_faces 3) / 6

def olivia_prob_less_than_6 : ℚ :=
  olivia_faces 3 / 6

-- Theorem statement
theorem cameron_wins_probability :
  dean_prob_less_than_6 * olivia_prob_less_than_6 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cameron_wins_probability_l976_97679


namespace NUMINAMATH_CALUDE_equalize_piles_in_three_moves_l976_97654

/-- Represents a configuration of pin piles -/
structure PinPiles :=
  (pile1 : Nat) (pile2 : Nat) (pile3 : Nat)

/-- Represents a move between two piles -/
inductive Move
  | one_to_two
  | one_to_three
  | two_to_one
  | two_to_three
  | three_to_one
  | three_to_two

/-- Applies a move to a given configuration -/
def apply_move (piles : PinPiles) (move : Move) : PinPiles :=
  match move with
  | Move.one_to_two => PinPiles.mk (piles.pile1 - piles.pile2) (piles.pile2 * 2) piles.pile3
  | Move.one_to_three => PinPiles.mk (piles.pile1 - piles.pile3) piles.pile2 (piles.pile3 * 2)
  | Move.two_to_one => PinPiles.mk (piles.pile1 * 2) (piles.pile2 - piles.pile1) piles.pile3
  | Move.two_to_three => PinPiles.mk piles.pile1 (piles.pile2 - piles.pile3) (piles.pile3 * 2)
  | Move.three_to_one => PinPiles.mk (piles.pile1 * 2) piles.pile2 (piles.pile3 - piles.pile1)
  | Move.three_to_two => PinPiles.mk piles.pile1 (piles.pile2 * 2) (piles.pile3 - piles.pile2)

/-- The main theorem to be proved -/
theorem equalize_piles_in_three_moves :
  ∃ (m1 m2 m3 : Move),
    let initial := PinPiles.mk 11 7 6
    let step1 := apply_move initial m1
    let step2 := apply_move step1 m2
    let step3 := apply_move step2 m3
    step3 = PinPiles.mk 8 8 8 :=
by
  sorry

end NUMINAMATH_CALUDE_equalize_piles_in_three_moves_l976_97654


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l976_97603

/-- Given a point with rectangular coordinates (2, -3, 6) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, π + θ, φ) has rectangular coordinates (-2, 3, 6). -/
theorem spherical_coordinate_transformation (ρ θ φ : Real) :
  (2 : Real) = ρ * Real.sin φ * Real.cos θ ∧
  (-3 : Real) = ρ * Real.sin φ * Real.sin θ ∧
  (6 : Real) = ρ * Real.cos φ →
  (-2 : Real) = ρ * Real.sin φ * Real.cos (Real.pi + θ) ∧
  (3 : Real) = ρ * Real.sin φ * Real.sin (Real.pi + θ) ∧
  (6 : Real) = ρ * Real.cos φ := by
  sorry


end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l976_97603


namespace NUMINAMATH_CALUDE_octagon_area_l976_97630

/-- The area of a regular octagon inscribed in a circle with area 400π -/
theorem octagon_area (circle_area : ℝ) (h : circle_area = 400 * Real.pi) :
  let r := (circle_area / Real.pi).sqrt
  let triangle_area := (1 / 2) * r^2 * Real.sin (Real.pi / 4)
  8 * triangle_area = 800 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l976_97630


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l976_97685

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem sum_of_coefficients_equals_one (a b : ℝ) : 
  (i^2 + a * i + b = 0) → (a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l976_97685


namespace NUMINAMATH_CALUDE_total_distance_walked_l976_97678

theorem total_distance_walked (first_part second_part : Real) 
  (h1 : first_part = 0.75)
  (h2 : second_part = 0.25) : 
  first_part + second_part = 1 := by
sorry

end NUMINAMATH_CALUDE_total_distance_walked_l976_97678


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l976_97606

theorem sufficient_but_not_necessary (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (¬q → p) ∧ ¬(p → ¬q) := by
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l976_97606


namespace NUMINAMATH_CALUDE_doraemon_toys_count_l976_97645

theorem doraemon_toys_count : ∃! n : ℕ, 40 ≤ n ∧ n ≤ 55 ∧ (n - 3) % 5 = 0 ∧ (n + 2) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_doraemon_toys_count_l976_97645


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l976_97621

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + 3 * a 8 + a 13 = 120) : 
  a 3 + a 13 - a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l976_97621


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l976_97652

theorem divisibility_implies_equality (a b : ℕ+) :
  (4 * a * b - 1) ∣ (4 * a^2 - 1)^2 → a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l976_97652


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l976_97692

/-- The area of a regular octagon inscribed in a circle with area 400π square units is 800√2 square units. -/
theorem inscribed_octagon_area (circle_area : ℝ) (octagon_area : ℝ) :
  circle_area = 400 * Real.pi →
  octagon_area = 8 * (1 / 2 * (20^2) * Real.sin (π / 4)) →
  octagon_area = 800 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l976_97692


namespace NUMINAMATH_CALUDE_unique_base_number_l976_97693

theorem unique_base_number : ∃! (x : ℕ), x < 6 ∧ x^23 % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_number_l976_97693


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l976_97627

theorem simplified_fraction_ratio (k : ℤ) : 
  ∃ (a b : ℤ), (6 * k + 18) / 3 = a * k + b ∧ a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l976_97627


namespace NUMINAMATH_CALUDE_x_completion_time_l976_97622

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℝ
  is_positive : days > 0

/-- Represents a worker who can complete a work in a given time -/
structure Worker where
  time_to_complete : WorkTime

/-- The work scenario -/
structure WorkScenario where
  x : Worker
  y : Worker
  x_partial_work : WorkTime
  y_completion_after_x : WorkTime
  y_solo_completion : WorkTime
  work_continuity : x_partial_work.days + y_completion_after_x.days = y_solo_completion.days

/-- The theorem stating that x takes 40 days to complete the work -/
theorem x_completion_time (scenario : WorkScenario) 
  (h1 : scenario.x_partial_work.days = 8)
  (h2 : scenario.y_completion_after_x.days = 16)
  (h3 : scenario.y_solo_completion.days = 20) :
  scenario.x.time_to_complete.days = 40 := by
  sorry


end NUMINAMATH_CALUDE_x_completion_time_l976_97622


namespace NUMINAMATH_CALUDE_remainder_relationship_l976_97691

theorem remainder_relationship (M M' N D S S' s s' : ℕ) : 
  M > M' →
  M % D = S →
  M' % D = S' →
  (M^2 * M') % D = s →
  N^2 % D = s' →
  (∃ M M' N D S S' s s' : ℕ, s = s') ∧
  (∃ M M' N D S S' s s' : ℕ, s < s') :=
by sorry

end NUMINAMATH_CALUDE_remainder_relationship_l976_97691


namespace NUMINAMATH_CALUDE_no_solution_iff_a_in_range_l976_97698

/-- The equation has no solutions if and only if a is in the specified range -/
theorem no_solution_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, 5*|x - 4*a| + |x - a^2| + 4*x - 4*a ≠ 0) ↔ 
  (a < -8 ∨ a > 0) := by
sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_in_range_l976_97698


namespace NUMINAMATH_CALUDE_marbles_lost_l976_97651

theorem marbles_lost (initial : ℕ) (current : ℕ) (lost : ℕ) : 
  initial = 16 → current = 9 → lost = initial - current → lost = 7 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l976_97651


namespace NUMINAMATH_CALUDE_common_tangents_O₁_O₂_l976_97671

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Number of common tangents between two circles -/
def num_common_tangents (c1 c2 : Circle) : ℕ := sorry

/-- Circle O₁: x² + y² - 2x = 0 -/
def O₁ : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*x = 0 }

/-- Circle O₂: x² + y² - 4x = 0 -/
def O₂ : Circle :=
  { equation := λ x y => x^2 + y^2 - 4*x = 0 }

theorem common_tangents_O₁_O₂ :
  num_common_tangents O₁ O₂ = 1 := by sorry

end NUMINAMATH_CALUDE_common_tangents_O₁_O₂_l976_97671


namespace NUMINAMATH_CALUDE_rachel_painting_time_l976_97608

/-- Prove that Rachel's painting time is 13 hours -/
theorem rachel_painting_time :
  let matt_time : ℕ := 12
  let patty_time : ℕ := matt_time / 3
  let rachel_time : ℕ := 2 * patty_time + 5
  rachel_time = 13 := by
  sorry

end NUMINAMATH_CALUDE_rachel_painting_time_l976_97608


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l976_97696

/-- The number of heartbeats during a race -/
def heartbeats_during_race (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proof that the athlete's heart beats 19200 times during the race -/
theorem athlete_heartbeats :
  heartbeats_during_race 160 6 20 = 19200 := by
  sorry

#eval heartbeats_during_race 160 6 20

end NUMINAMATH_CALUDE_athlete_heartbeats_l976_97696


namespace NUMINAMATH_CALUDE_neds_weekly_sales_l976_97675

def normal_mouse_price : ℝ := 120
def normal_keyboard_price : ℝ := 80
def normal_scissors_price : ℝ := 30

def left_handed_mouse_price : ℝ := normal_mouse_price * 1.3
def left_handed_keyboard_price : ℝ := normal_keyboard_price * 1.2
def left_handed_scissors_price : ℝ := normal_scissors_price * 1.5

def daily_mouse_sales : ℝ := 25
def daily_keyboard_sales : ℝ := 10
def daily_scissors_sales : ℝ := 15
def daily_bundle_sales : ℝ := 5

def bundle_price : ℝ := (left_handed_mouse_price + left_handed_keyboard_price + left_handed_scissors_price) * 0.9

def regular_open_days : ℕ := 3
def extended_open_days : ℕ := 1
def extended_day_multiplier : ℝ := 1.5

def total_weekly_sales : ℝ :=
  (daily_mouse_sales * left_handed_mouse_price +
   daily_keyboard_sales * left_handed_keyboard_price +
   daily_scissors_sales * left_handed_scissors_price +
   daily_bundle_sales * bundle_price) *
  (regular_open_days + extended_open_days * extended_day_multiplier)

theorem neds_weekly_sales :
  total_weekly_sales = 29922.25 := by sorry

end NUMINAMATH_CALUDE_neds_weekly_sales_l976_97675


namespace NUMINAMATH_CALUDE_matrix_power_identity_l976_97617

/-- Given a 2x2 matrix B, prove that B^4 = 51*B + 52*I --/
theorem matrix_power_identity (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B = !![1, 2; 3, 1]) : 
  B^4 = 51 • B + 52 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_identity_l976_97617


namespace NUMINAMATH_CALUDE_percentage_of_300_is_66_l976_97649

theorem percentage_of_300_is_66 : 
  (66 : ℝ) / 300 * 100 = 22 := by sorry

end NUMINAMATH_CALUDE_percentage_of_300_is_66_l976_97649


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l976_97625

theorem largest_solution_of_equation : 
  let f : ℝ → ℝ := λ b => (3*b + 7)*(b - 2) - 9*b
  let largest_solution : ℝ := (4 + Real.sqrt 58) / 3
  (f largest_solution = 0) ∧ 
  (∀ b : ℝ, f b = 0 → b ≤ largest_solution) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l976_97625


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l976_97682

/-- Given a group of people, their average age, and the age of the youngest person,
    calculate the average age of the group when the youngest was born. -/
theorem average_age_when_youngest_born 
  (n : ℕ) -- Total number of people
  (avg : ℝ) -- Current average age
  (youngest : ℝ) -- Age of the youngest person
  (h1 : n = 7) -- There are 7 people
  (h2 : avg = 30) -- The current average age is 30
  (h3 : youngest = 3) -- The youngest person is 3 years old
  : (n * avg - youngest) / (n - 1) = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l976_97682


namespace NUMINAMATH_CALUDE_min_value_theorem_l976_97644

-- Define the optimization problem
def optimization_problem (x y : ℝ) : Prop :=
  x - y ≥ 0 ∧ x + y - 2 ≥ 0 ∧ x ≤ 2

-- Define the objective function
def objective_function (x y : ℝ) : ℝ :=
  x^2 + y^2 - 2*x

-- Theorem statement
theorem min_value_theorem :
  ∃ (min_val : ℝ), min_val = -1/2 ∧
  (∀ (x y : ℝ), optimization_problem x y → objective_function x y ≥ min_val) ∧
  (∃ (x y : ℝ), optimization_problem x y ∧ objective_function x y = min_val) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l976_97644


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l976_97635

theorem polynomial_division_theorem (x : ℝ) :
  (x^5 - 2*x^4 + 4*x^3 - 8*x^2 + 16*x - 32) * (x + 2) + 76 = x^6 + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l976_97635


namespace NUMINAMATH_CALUDE_triangle_properties_l976_97605

/-- Triangle ABC with sides AB = 8, BC = 2a+2, AC = 22 -/
structure Triangle (a : ℝ) where
  AB : ℝ := 8
  BC : ℝ := 2*a + 2
  AC : ℝ := 22

/-- The range of a for a valid triangle -/
def valid_a_range (a : ℝ) : Prop :=
  6 < a ∧ a < 14

/-- The triangle is isosceles -/
def is_isosceles (t : Triangle a) : Prop :=
  t.AB = t.BC ∨ t.AB = t.AC ∨ t.BC = t.AC

/-- The perimeter of the triangle -/
def perimeter (t : Triangle a) : ℝ :=
  t.AB + t.BC + t.AC

theorem triangle_properties (a : ℝ) :
  (∀ t : Triangle a, valid_a_range a) ∧
  (∀ t : Triangle a, is_isosceles t → perimeter t = 52) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l976_97605


namespace NUMINAMATH_CALUDE_factorization_equality_l976_97680

theorem factorization_equality (y : ℝ) : 3 * y * (y - 5) + 4 * (y - 5) = (3 * y + 4) * (y - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l976_97680


namespace NUMINAMATH_CALUDE_skips_mode_is_165_l976_97653

def skips : List ℕ := [165, 165, 165, 165, 165, 170, 170, 145, 150, 150]

def mode (l : List ℕ) : ℕ := 
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem skips_mode_is_165 : mode skips = 165 := by sorry

end NUMINAMATH_CALUDE_skips_mode_is_165_l976_97653


namespace NUMINAMATH_CALUDE_inequality_proof_l976_97620

open Real BigOperators Finset

theorem inequality_proof (n : ℕ) (x : Fin n → ℝ) (σ : Equiv.Perm (Fin n)) 
  (h : ∀ i, 0 < x i ∧ x i < 1) : 
  ∑ i, (1 / (1 - x i)) ≥ 
  (1 + (1 / n) * ∑ i, x i) * ∑ i, (1 / (1 - x i * x (σ i))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l976_97620


namespace NUMINAMATH_CALUDE_max_value_of_f_l976_97613

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x)

theorem max_value_of_f :
  ∃ (M : ℝ), M = (16 * Real.sqrt 3) / 9 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l976_97613


namespace NUMINAMATH_CALUDE_specific_right_triangle_with_square_l976_97638

/-- Represents a right triangle with a square inscribed on its hypotenuse -/
structure RightTriangleWithSquare where
  /-- Length of one leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the other leg of the right triangle -/
  leg2 : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- Distance from the right angle vertex to the side of the square on the hypotenuse -/
  distance_to_square : ℝ

/-- Theorem stating the properties of the specific right triangle with inscribed square -/
theorem specific_right_triangle_with_square :
  ∃ (t : RightTriangleWithSquare),
    t.leg1 = 9 ∧
    t.leg2 = 12 ∧
    t.square_side = 75 / 7 ∧
    t.distance_to_square = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_right_triangle_with_square_l976_97638


namespace NUMINAMATH_CALUDE_total_amount_l976_97672

theorem total_amount (z : ℚ) (y : ℚ) (x : ℚ) 
  (hz : z = 200)
  (hy : y = 1.2 * z)
  (hx : x = 1.25 * y) :
  x + y + z = 740 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_l976_97672


namespace NUMINAMATH_CALUDE_expression_evaluation_l976_97614

/-- Given x = 3, y = 2, and z = 4, prove that 3 * x - 2 * y + 4 * z = 21 -/
theorem expression_evaluation (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 2 * y + 4 * z = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l976_97614


namespace NUMINAMATH_CALUDE_lottery_win_probability_l976_97676

/-- A lottery event with two prize categories -/
structure LotteryEvent where
  firstPrizeProb : ℝ
  secondPrizeProb : ℝ

/-- The probability of winning a prize in the lottery event -/
def winPrizeProb (event : LotteryEvent) : ℝ :=
  event.firstPrizeProb + event.secondPrizeProb

/-- Theorem stating the probability of winning a prize in the given lottery event -/
theorem lottery_win_probability :
  ∃ (event : LotteryEvent), 
    event.firstPrizeProb = 0.1 ∧ 
    event.secondPrizeProb = 0.1 ∧ 
    winPrizeProb event = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_lottery_win_probability_l976_97676


namespace NUMINAMATH_CALUDE_employee_age_at_hiring_l976_97669

/-- Rule of 70 provision: An employee can retire when their age plus years of employment total at least 70. -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1987

/-- The year the employee became eligible to retire -/
def retirement_eligibility_year : ℕ := 2006

/-- The age of the employee when hired -/
def age_when_hired : ℕ := 51

theorem employee_age_at_hiring :
  rule_of_70 (age_when_hired + (retirement_eligibility_year - hire_year)) (retirement_eligibility_year - hire_year) ∧
  age_when_hired = 51 := by
  sorry

end NUMINAMATH_CALUDE_employee_age_at_hiring_l976_97669


namespace NUMINAMATH_CALUDE_complex_equation_solution_l976_97616

theorem complex_equation_solution (z : ℂ) (h : z * (1 + Complex.I) = 2 - 4 * Complex.I) :
  z = -1 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l976_97616


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_max_sum_achievable_l976_97688

theorem max_sum_with_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 :=
by sorry

theorem max_sum_achievable :
  ∃ (x y z : ℚ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  16 * x * y * z = (x + y)^2 * (x + z)^2 ∧
  x + y + z = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_max_sum_achievable_l976_97688


namespace NUMINAMATH_CALUDE_broken_line_rectangle_ratio_l976_97648

/-- A rectangle with a broken line inside it -/
structure BrokenLineRectangle where
  /-- The shorter side of the rectangle -/
  short_side : ℝ
  /-- The longer side of the rectangle -/
  long_side : ℝ
  /-- The broken line consists of segments equal to the shorter side -/
  segment_length : ℝ
  /-- The short side is positive -/
  short_positive : 0 < short_side
  /-- The long side is longer than the short side -/
  long_longer : short_side < long_side
  /-- The segment length is equal to the shorter side -/
  segment_eq_short : segment_length = short_side
  /-- Adjacent segments of the broken line are perpendicular -/
  segments_perpendicular : True

/-- The ratio of the shorter side to the longer side is 1:2 -/
theorem broken_line_rectangle_ratio (r : BrokenLineRectangle) :
  r.short_side / r.long_side = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_rectangle_ratio_l976_97648


namespace NUMINAMATH_CALUDE_salary_adjustment_proof_l976_97647

def initial_salary : ℝ := 2500
def june_raise_percentage : ℝ := 0.15
def june_bonus : ℝ := 300
def july_cut_percentage : ℝ := 0.25

def final_salary : ℝ :=
  (initial_salary * (1 + june_raise_percentage) + june_bonus) * (1 - july_cut_percentage)

theorem salary_adjustment_proof :
  final_salary = 2381.25 := by sorry

end NUMINAMATH_CALUDE_salary_adjustment_proof_l976_97647


namespace NUMINAMATH_CALUDE_orange_groups_l976_97689

theorem orange_groups (total_oranges : ℕ) (num_groups : ℕ) 
  (h1 : total_oranges = 384) (h2 : num_groups = 16) :
  total_oranges / num_groups = 24 := by
sorry

end NUMINAMATH_CALUDE_orange_groups_l976_97689


namespace NUMINAMATH_CALUDE_square_difference_equals_one_l976_97602

theorem square_difference_equals_one (x y : ℝ) 
  (sum_eq : x + y = 5) 
  (product_eq : x * y = 6) : 
  (x - y)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_square_difference_equals_one_l976_97602


namespace NUMINAMATH_CALUDE_athlete_shots_l976_97668

theorem athlete_shots (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 →  -- Each point value scored at least once
  x + y + z > 11 →         -- More than 11 shots
  8*x + 9*y + 10*z = 100 → -- Total score is 100
  x = 9                    -- Number of 8-point shots is 9
  := by sorry

end NUMINAMATH_CALUDE_athlete_shots_l976_97668


namespace NUMINAMATH_CALUDE_supplier_payment_proof_l976_97604

/-- Calculates the amount paid to a supplier given initial funds, received payment, expenses, and final amount -/
def amount_paid_to_supplier (initial_funds : ℤ) (received_payment : ℤ) (expenses : ℤ) (final_amount : ℤ) : ℤ :=
  initial_funds + received_payment - expenses - final_amount

/-- Proves that the amount paid to the supplier is 600 given the problem conditions -/
theorem supplier_payment_proof (initial_funds : ℤ) (received_payment : ℤ) (expenses : ℤ) (final_amount : ℤ)
  (h1 : initial_funds = 2000)
  (h2 : received_payment = 800)
  (h3 : expenses = 1200)
  (h4 : final_amount = 1000) :
  amount_paid_to_supplier initial_funds received_payment expenses final_amount = 600 := by
  sorry

#eval amount_paid_to_supplier 2000 800 1200 1000

end NUMINAMATH_CALUDE_supplier_payment_proof_l976_97604


namespace NUMINAMATH_CALUDE_flowchart_connection_is_flow_line_l976_97629

-- Define the basic elements of a flowchart
inductive FlowchartElement
  | ConnectionPoint
  | DecisionBox
  | FlowLine
  | ProcessBox

-- Define a property for connecting steps in a flowchart
def connects_steps (element : FlowchartElement) : Prop :=
  element = FlowchartElement.FlowLine

-- Theorem statement
theorem flowchart_connection_is_flow_line :
  ∃ (element : FlowchartElement), connects_steps element :=
sorry

end NUMINAMATH_CALUDE_flowchart_connection_is_flow_line_l976_97629


namespace NUMINAMATH_CALUDE_complex_and_imaginary_solution_l976_97641

-- Define z as a complex number
variable (z : ℂ)

-- Define the conditions
def condition1 : Prop := (z + Complex.I).im = 0
def condition2 : Prop := (z / (1 - Complex.I)).im = 0

-- Define m as a purely imaginary number
def m : ℂ → ℂ := fun c => Complex.I * c

-- Define the equation with real roots
def has_real_roots (z m : ℂ) : Prop :=
  ∃ x : ℝ, x^2 + x * (1 + z) - (3 * m - 1) * Complex.I = 0

-- State the theorem
theorem complex_and_imaginary_solution :
  condition1 z → condition2 z → has_real_roots z (m 1) →
  z = 1 - Complex.I ∧ m 1 = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_and_imaginary_solution_l976_97641


namespace NUMINAMATH_CALUDE_apple_trees_count_l976_97694

/-- The number of apple trees in the orchard -/
def num_apple_trees : ℕ := 30

/-- The yield of apples per apple tree in kg -/
def apple_yield : ℕ := 150

/-- The number of peach trees in the orchard -/
def num_peach_trees : ℕ := 45

/-- The average yield of peaches per peach tree in kg -/
def peach_yield : ℕ := 65

/-- The total mass of fruit harvested in kg -/
def total_harvest : ℕ := 7425

/-- Theorem stating that the number of apple trees is correct given the conditions -/
theorem apple_trees_count :
  num_apple_trees * apple_yield + num_peach_trees * peach_yield = total_harvest :=
by sorry


end NUMINAMATH_CALUDE_apple_trees_count_l976_97694


namespace NUMINAMATH_CALUDE_cube_side_length_l976_97626

theorem cube_side_length (surface_area : ℝ) (side_length : ℝ) : 
  surface_area = 600 → 
  6 * side_length^2 = surface_area → 
  side_length = 10 := by
sorry

end NUMINAMATH_CALUDE_cube_side_length_l976_97626


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sixths_l976_97684

theorem smallest_fraction_greater_than_five_sixths :
  ∀ a b : ℕ, 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 5 / 6 →
    81 / 97 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sixths_l976_97684


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l976_97643

/-- Given an ellipse with semi-major axis a and semi-minor axis b, where a > b > 0,
    and foci F₁ and F₂, a line passing through F₁ intersects the ellipse at points A and B.
    If AB ⟂ AF₂ and |AB| = |AF₂|, then the eccentricity of the ellipse is √6 - √3. -/
theorem ellipse_eccentricity (a b : ℝ) (F₁ F₂ A B : ℝ × ℝ) :
  a > b ∧ b > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ ({F₁, F₂, A, B} : Set (ℝ × ℝ))) →
  (A.1 - B.1) * (A.1 - F₂.1) + (A.2 - B.2) * (A.2 - F₂.2) = 0 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - F₂.1)^2 + (A.2 - F₂.2)^2 →
  let e := Real.sqrt ((a^2 - b^2) / a^2)
  e = Real.sqrt 6 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l976_97643


namespace NUMINAMATH_CALUDE_height_range_selection_probability_overall_avg_height_l976_97666

-- Define the track and field team
def num_male : ℕ := 12
def num_female : ℕ := 8
def total_athletes : ℕ := num_male + num_female
def max_height : ℕ := 190
def min_height : ℕ := 160
def avg_height_male : ℝ := 175
def avg_height_female : ℝ := 165

-- Theorem 1: The range of heights is 30cm
theorem height_range : max_height - min_height = 30 := by sorry

-- Theorem 2: The probability of an athlete being selected in a random sample of 10 is 1/2
theorem selection_probability : (10 : ℝ) / total_athletes = (1 : ℝ) / 2 := by sorry

-- Theorem 3: The overall average height of the team is 171cm
theorem overall_avg_height :
  (num_male : ℝ) / total_athletes * avg_height_male +
  (num_female : ℝ) / total_athletes * avg_height_female = 171 := by sorry

end NUMINAMATH_CALUDE_height_range_selection_probability_overall_avg_height_l976_97666


namespace NUMINAMATH_CALUDE_unique_solution_l976_97656

theorem unique_solution (x y z : ℝ) 
  (h1 : x + y^2 + z^3 = 3)
  (h2 : y + z^2 + x^3 = 3)
  (h3 : z + x^2 + y^3 = 3)
  (px : x > 0)
  (py : y > 0)
  (pz : z > 0) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l976_97656


namespace NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l976_97609

theorem matrix_inverse_scalar_multiple (d k : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 6, d]
  (A⁻¹ = k • A) → d = -1 ∧ k = (1 : ℝ) / 25 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l976_97609


namespace NUMINAMATH_CALUDE_probability_of_losing_l976_97657

theorem probability_of_losing (odds_win odds_lose : ℕ) 
  (h_odds : odds_win = 5 ∧ odds_lose = 3) : 
  (odds_lose : ℚ) / (odds_win + odds_lose) = 3 / 8 :=
by
  sorry

#check probability_of_losing

end NUMINAMATH_CALUDE_probability_of_losing_l976_97657


namespace NUMINAMATH_CALUDE_sweep_probability_l976_97661

/-- Represents a clock with four equally spaced points -/
structure Clock :=
  (points : Fin 4 → ℕ)
  (equally_spaced : ∀ i : Fin 4, points i = i.val * 3)

/-- Represents a 20-minute period on the clock -/
def Period : ℕ := 20

/-- Calculates the number of favorable intervals in a 60-minute period -/
def favorable_intervals (c : Clock) (p : ℕ) : ℕ :=
  4 * 5  -- 4 intervals of 5 minutes each

/-- The probability of sweeping exactly two points in the given period -/
def probability (c : Clock) (p : ℕ) : ℚ :=
  (favorable_intervals c p : ℚ) / 60

/-- The main theorem stating the probability is 1/3 -/
theorem sweep_probability (c : Clock) :
  probability c Period = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sweep_probability_l976_97661
