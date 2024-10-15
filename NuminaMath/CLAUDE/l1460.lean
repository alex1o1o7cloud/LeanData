import Mathlib

namespace NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l1460_146034

theorem sin_thirteen_pi_fourths : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l1460_146034


namespace NUMINAMATH_CALUDE_area_not_perfect_square_l1460_146029

/-- A primitive Pythagorean triple -/
structure PrimitivePythagoreanTriple where
  a : ℕ
  b : ℕ
  c : ℕ
  isPrimitive : Nat.gcd a b = 1
  isPythagorean : a^2 + b^2 = c^2

/-- The area of a right triangle with legs a and b is not a perfect square -/
theorem area_not_perfect_square (t : PrimitivePythagoreanTriple) :
  ¬ ∃ (n : ℕ), (t.a * t.b) / 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_area_not_perfect_square_l1460_146029


namespace NUMINAMATH_CALUDE_min_value_expression_l1460_146003

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 2 + 24 / (n : ℝ) ≥ 7 ∧ 
  ∃ m : ℕ+, (m : ℝ) / 2 + 24 / (m : ℝ) = 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1460_146003


namespace NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_546_l1460_146036

theorem sum_largest_smallest_prime_factors_546 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧
    largest.Prime ∧
    (smallest ∣ 546) ∧
    (largest ∣ 546) ∧
    (∀ p : ℕ, p.Prime → p ∣ 546 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 546 → p ≥ smallest) ∧
    smallest + largest = 15 :=
by sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_546_l1460_146036


namespace NUMINAMATH_CALUDE_course_combinations_l1460_146041

def type_A_courses : ℕ := 3
def type_B_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

def combinations_with_both_types (a b k : ℕ) : ℕ :=
  Nat.choose a (k - 1) * Nat.choose b 1 + Nat.choose a 1 * Nat.choose b (k - 1)

theorem course_combinations :
  combinations_with_both_types type_A_courses type_B_courses total_courses_to_choose = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_combinations_l1460_146041


namespace NUMINAMATH_CALUDE_orchid_bushes_after_planting_l1460_146066

/-- The number of orchid bushes in the park after planting -/
def total_orchid_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: Given 22 initial orchid bushes and 13 newly planted orchid bushes,
    the total number of orchid bushes after planting will be 35. -/
theorem orchid_bushes_after_planting :
  total_orchid_bushes 22 13 = 35 := by
  sorry

end NUMINAMATH_CALUDE_orchid_bushes_after_planting_l1460_146066


namespace NUMINAMATH_CALUDE_right_triangle_side_difference_l1460_146054

theorem right_triangle_side_difference (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧ 
  C = π / 2 ∧ 
  a = 6 ∧ 
  B = π / 6 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  c / Real.sin C = a / Real.sin A
  → c - b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_difference_l1460_146054


namespace NUMINAMATH_CALUDE_angle_position_l1460_146010

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An angle in the 2D plane -/
structure Angle where
  -- We don't need to define the internal structure of an angle for this problem

/-- The terminal side of an angle -/
def terminal_side (α : Angle) : Set Point :=
  sorry -- Definition not needed for the statement

/-- Predicate to check if a point is on the non-negative side of the y-axis -/
def on_nonnegative_y_side (p : Point) : Prop :=
  p.x = 0 ∧ p.y ≥ 0

theorem angle_position (α : Angle) (P : Point) :
  P ∈ terminal_side α →
  P = ⟨0, 3⟩ →
  ∃ (p : Point), p ∈ terminal_side α ∧ on_nonnegative_y_side p :=
sorry

end NUMINAMATH_CALUDE_angle_position_l1460_146010


namespace NUMINAMATH_CALUDE_bus_stoppage_time_l1460_146030

/-- Given a bus with speeds excluding and including stoppages, 
    calculate the number of minutes the bus stops per hour -/
theorem bus_stoppage_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 48)
  (h2 : speed_with_stops = 12) :
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bus_stoppage_time_l1460_146030


namespace NUMINAMATH_CALUDE_chocolate_problem_l1460_146019

/-- The number of chocolates in the cost price -/
def cost_chocolates : ℕ := 24

/-- The gain percentage -/
def gain_percent : ℚ := 1/2

/-- The number of chocolates in the selling price -/
def selling_chocolates : ℕ := 16

theorem chocolate_problem (C S : ℚ) (n : ℕ) 
  (h1 : C > 0) 
  (h2 : S > 0) 
  (h3 : n > 0) 
  (h4 : cost_chocolates * C = n * S) 
  (h5 : gain_percent = (S - C) / C) : 
  n = selling_chocolates := by
  sorry

#check chocolate_problem

end NUMINAMATH_CALUDE_chocolate_problem_l1460_146019


namespace NUMINAMATH_CALUDE_cubic_polynomial_distinct_roots_condition_l1460_146009

theorem cubic_polynomial_distinct_roots_condition (p q : ℝ) : 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 + p*x + q = (x - a) * (x - b) * (x - c))) →
  p < 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_distinct_roots_condition_l1460_146009


namespace NUMINAMATH_CALUDE_petpals_center_total_cats_l1460_146017

/-- Represents the PetPals Training Center -/
structure PetPalsCenter where
  jump : ℕ
  fetch : ℕ
  spin : ℕ
  jump_fetch : ℕ
  fetch_spin : ℕ
  jump_spin : ℕ
  all_three : ℕ
  none : ℕ

/-- The number of cats in the PetPals Training Center -/
def total_cats (center : PetPalsCenter) : ℕ :=
  center.all_three +
  (center.jump_fetch - center.all_three) +
  (center.fetch_spin - center.all_three) +
  (center.jump_spin - center.all_three) +
  (center.jump - center.jump_fetch - center.jump_spin + center.all_three) +
  (center.fetch - center.jump_fetch - center.fetch_spin + center.all_three) +
  (center.spin - center.jump_spin - center.fetch_spin + center.all_three) +
  center.none

/-- Theorem stating the total number of cats in the PetPals Training Center -/
theorem petpals_center_total_cats :
  ∀ (center : PetPalsCenter),
    center.jump = 60 →
    center.fetch = 40 →
    center.spin = 50 →
    center.jump_fetch = 25 →
    center.fetch_spin = 20 →
    center.jump_spin = 30 →
    center.all_three = 15 →
    center.none = 5 →
    total_cats center = 95 := by
  sorry

end NUMINAMATH_CALUDE_petpals_center_total_cats_l1460_146017


namespace NUMINAMATH_CALUDE_max_backyard_area_l1460_146082

/-- Represents a rectangular backyard with given constraints -/
structure Backyard where
  length : ℝ
  width : ℝ
  fencing : ℝ
  length_min : ℝ
  fence_constraint : fencing = length + 2 * width
  length_constraint : length ≥ length_min
  proportion_constraint : length ≤ 2 * width

/-- The area of a backyard -/
def area (b : Backyard) : ℝ := b.length * b.width

/-- Theorem stating the maximum area of a backyard with given constraints -/
theorem max_backyard_area (b : Backyard) (h1 : b.fencing = 400) (h2 : b.length_min = 100) :
  ∃ (max_area : ℝ), max_area = 20000 ∧ ∀ (other : Backyard), 
    other.fencing = 400 → other.length_min = 100 → area other ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_backyard_area_l1460_146082


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruent_to_one_mod_37_l1460_146089

theorem smallest_three_digit_congruent_to_one_mod_37 : 
  ∃ n : ℕ, 
    (100 ≤ n ∧ n ≤ 999) ∧ 
    n % 37 = 1 ∧ 
    (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) ∧ m % 37 = 1 → n ≤ m) ∧
    n = 112 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruent_to_one_mod_37_l1460_146089


namespace NUMINAMATH_CALUDE_prob_selected_twice_correct_most_likely_selected_correct_l1460_146087

/-- Represents the total number of students --/
def total_students : ℕ := 60

/-- Represents the number of students selected in each round --/
def selected_per_round : ℕ := 45

/-- Probability of a student being selected in both rounds --/
def prob_selected_twice : ℚ := 9 / 16

/-- The most likely number of students selected in both rounds --/
def most_likely_selected : ℕ := 34

/-- Function to calculate the probability of being selected in both rounds --/
def calculate_prob_selected_twice : ℚ :=
  (Nat.choose (total_students - 1) (selected_per_round - 1) / Nat.choose total_students selected_per_round) ^ 2

/-- Function to calculate the probability of exactly n students being selected in both rounds --/
def prob_n_selected (n : ℕ) : ℚ :=
  (Nat.choose total_students n * Nat.choose (total_students - n) (selected_per_round - n) * Nat.choose (selected_per_round - n) (selected_per_round - n)) /
  (Nat.choose total_students selected_per_round * Nat.choose total_students selected_per_round)

theorem prob_selected_twice_correct :
  calculate_prob_selected_twice = prob_selected_twice :=
sorry

theorem most_likely_selected_correct :
  ∀ n, 30 ≤ n ∧ n ≤ 45 → prob_n_selected most_likely_selected ≥ prob_n_selected n :=
sorry

end NUMINAMATH_CALUDE_prob_selected_twice_correct_most_likely_selected_correct_l1460_146087


namespace NUMINAMATH_CALUDE_total_chips_is_135_l1460_146050

/-- Calculates the total number of chips for Viviana, Susana, and Manuel --/
def total_chips (viviana_vanilla : ℕ) (susana_chocolate : ℕ) : ℕ :=
  let viviana_chocolate := susana_chocolate + 5
  let susana_vanilla := (3 * viviana_vanilla) / 4
  let manuel_vanilla := 2 * susana_vanilla
  let manuel_chocolate := viviana_chocolate / 2
  (viviana_chocolate + viviana_vanilla) + 
  (susana_chocolate + susana_vanilla) + 
  (manuel_chocolate + manuel_vanilla)

/-- Theorem stating the total number of chips is 135 --/
theorem total_chips_is_135 : total_chips 20 25 = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_chips_is_135_l1460_146050


namespace NUMINAMATH_CALUDE_stone_piles_total_l1460_146038

theorem stone_piles_total (pile1 pile2 pile3 pile4 pile5 : ℕ) : 
  pile5 = 6 * pile3 →
  pile2 = 2 * (pile3 + pile5) →
  pile1 * 3 = pile5 →
  pile1 + 10 = pile4 →
  2 * pile4 = pile2 →
  pile1 + pile2 + pile3 + pile4 + pile5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stone_piles_total_l1460_146038


namespace NUMINAMATH_CALUDE_union_and_complement_find_a_l1460_146081

-- Part 1
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_and_complement : 
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧ 
  ((Set.univ \ A) ∩ B = {x | 2 < x ∧ x < 3} ∪ {x | 7 ≤ x ∧ x < 10}) := by sorry

-- Part 2
def A' (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B' : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C' : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem find_a : 
  ∃ a : ℝ, (A' a ∩ B' ≠ ∅) ∧ (A' a ∩ C' = ∅) ∧ (a = -2) := by sorry

end NUMINAMATH_CALUDE_union_and_complement_find_a_l1460_146081


namespace NUMINAMATH_CALUDE_base_digit_conversion_l1460_146049

theorem base_digit_conversion (N : ℕ+) :
  (9^19 ≤ N ∧ N < 9^20) ∧ (27^12 ≤ N ∧ N < 27^13) →
  3^38 ≤ N ∧ N < 3^39 :=
by sorry

end NUMINAMATH_CALUDE_base_digit_conversion_l1460_146049


namespace NUMINAMATH_CALUDE_minimum_trees_l1460_146027

theorem minimum_trees (L : ℕ) (X : ℕ) : 
  (∀ n < L, ¬ ∃ m : ℕ, (0.13 : ℝ) * n < m ∧ m < (0.14 : ℝ) * n) →
  ((0.13 : ℝ) * L < X ∧ X < (0.14 : ℝ) * L) →
  L = 15 := by
sorry

end NUMINAMATH_CALUDE_minimum_trees_l1460_146027


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1460_146068

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x : ℝ, x^2 - (2*m - 1)*x + m^2 = 0) →  -- Equation has real roots
  (x₁^2 - (2*m - 1)*x₁ + m^2 = 0) →         -- x₁ is a root
  (x₂^2 - (2*m - 1)*x₂ + m^2 = 0) →         -- x₂ is a root
  ((x₁ + 1) * (x₂ + 1) = 3) →               -- Given condition
  (m = -3) :=                               -- Conclusion
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1460_146068


namespace NUMINAMATH_CALUDE_negative_quadratic_symmetry_implies_inequality_l1460_146072

/-- A quadratic function with a negative leading coefficient -/
structure NegativeQuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  negative_leading_coeff : ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ a < 0

/-- The theorem statement -/
theorem negative_quadratic_symmetry_implies_inequality
  (f : NegativeQuadraticFunction)
  (h_symmetry : ∀ x : ℝ, f.f (2 - x) = f.f (2 + x)) :
  ∀ x : ℝ, -2 < x → x < 0 → f.f (1 + 2*x - x^2) < f.f (1 - 2*x^2) :=
sorry

end NUMINAMATH_CALUDE_negative_quadratic_symmetry_implies_inequality_l1460_146072


namespace NUMINAMATH_CALUDE_ending_number_is_989_l1460_146013

/-- A function that counts the number of integers between 0 and n (inclusive) 
    that do not contain the digit 1 in their decimal representation -/
def count_no_one (n : ℕ) : ℕ := sorry

/-- The theorem stating that 989 is the smallest positive integer n such that 
    there are exactly 728 integers between 0 and n (inclusive) that do not 
    contain the digit 1 -/
theorem ending_number_is_989 : 
  (∀ m : ℕ, m < 989 → count_no_one m < 728) ∧ count_no_one 989 = 728 :=
sorry

end NUMINAMATH_CALUDE_ending_number_is_989_l1460_146013


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_seven_l1460_146077

theorem least_three_digit_multiple_of_seven : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 7 ∣ n → 105 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_seven_l1460_146077


namespace NUMINAMATH_CALUDE_ring_arrangements_value_l1460_146023

/-- The number of possible 6-ring arrangements on 4 fingers, given 10 distinguishable rings,
    with no more than 2 rings per finger. -/
def ring_arrangements : ℕ :=
  let total_rings : ℕ := 10
  let fingers : ℕ := 4
  let rings_to_arrange : ℕ := 6
  let max_rings_per_finger : ℕ := 2
  
  let ways_to_choose_rings : ℕ := Nat.choose total_rings rings_to_arrange
  let ways_to_distribute_rings : ℕ := Nat.choose (rings_to_arrange + fingers - 1) (fingers - 1) -
    fingers * Nat.choose (rings_to_arrange - max_rings_per_finger - 1 + fingers - 1) (fingers - 1)
  let ways_to_order_rings : ℕ := Nat.factorial rings_to_arrange

  ways_to_choose_rings * ways_to_distribute_rings * ways_to_order_rings

theorem ring_arrangements_value : ring_arrangements = 604800 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangements_value_l1460_146023


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l1460_146069

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 9 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 3147 ∧ ∀ m < 3147, ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l1460_146069


namespace NUMINAMATH_CALUDE_divisibility_problem_l1460_146033

theorem divisibility_problem (n : ℕ) (h : ∀ a : ℕ, a < 60 → ¬(n ∣ a^3)) : n = 216000 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1460_146033


namespace NUMINAMATH_CALUDE_sun_overhead_locations_sun_angle_locations_l1460_146053

/-- Represents a location on Earth by its latitude and longitude -/
structure Location :=
  (lat : Real)
  (lon : Real)

/-- Budapest's location -/
def budapest : Location := ⟨47.5, 19.1⟩

/-- Calculates the location where the Sun is directly overhead given the latitude -/
def overheadLocation (lat : Real) : Location × Location :=
  sorry

/-- Calculates the location where the Sun's rays hit Budapest at a given angle -/
def angleLocation (angle : Real) : Location × Location :=
  sorry

theorem sun_overhead_locations :
  (overheadLocation (-23.5) = (⟨-23.5, 80.8⟩, ⟨-23.5, -42.6⟩)) ∧
  (overheadLocation 0 = (⟨0, 109.1⟩, ⟨0, -70.9⟩)) ∧
  (overheadLocation 23.5 = (⟨23.5, 137.4⟩, ⟨23.5, 99.2⟩)) :=
sorry

theorem sun_angle_locations :
  (angleLocation 60 = (⟨17.5, 129.2⟩, ⟨17.5, -91.0⟩)) ∧
  (angleLocation 30 = (⟨-12.5, 95.1⟩, ⟨-12.5, -56.9⟩)) :=
sorry

end NUMINAMATH_CALUDE_sun_overhead_locations_sun_angle_locations_l1460_146053


namespace NUMINAMATH_CALUDE_adult_ticket_price_l1460_146079

/-- Represents the cost of movie tickets for different age groups --/
structure TicketPrices where
  adult : ℕ
  child : ℕ
  senior : ℕ

/-- Represents the composition of Mrs. Lopez's family --/
structure Family where
  adults : ℕ
  children : ℕ
  seniors : ℕ

/-- The theorem states that given the family composition and ticket prices,
    the adult ticket price is 10 when the total cost is 64 --/
theorem adult_ticket_price 
  (prices : TicketPrices) 
  (family : Family) 
  (h1 : prices.child = 8)
  (h2 : prices.senior = 9)
  (h3 : family.adults = 3)
  (h4 : family.children = 2)
  (h5 : family.seniors = 2)
  (h6 : family.adults * prices.adult + family.children * prices.child + family.seniors * prices.senior = 64) :
  prices.adult = 10 := by
  sorry

#check adult_ticket_price

end NUMINAMATH_CALUDE_adult_ticket_price_l1460_146079


namespace NUMINAMATH_CALUDE_two_pow_2016_days_from_thursday_is_friday_l1460_146044

-- Define the days of the week
inductive Day : Type
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day
  | sunday : Day

def next_day (d : Day) : Day :=
  match d with
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday
  | Day.sunday => Day.monday

def days_from_now (start : Day) (n : ℕ) : Day :=
  match n with
  | 0 => start
  | n + 1 => next_day (days_from_now start n)

theorem two_pow_2016_days_from_thursday_is_friday :
  days_from_now Day.thursday (2^2016) = Day.friday :=
sorry

end NUMINAMATH_CALUDE_two_pow_2016_days_from_thursday_is_friday_l1460_146044


namespace NUMINAMATH_CALUDE_workshop_probability_l1460_146039

def total_students : ℕ := 30
def painting_students : ℕ := 22
def sculpting_students : ℕ := 24

theorem workshop_probability : 
  let both_workshops := painting_students + sculpting_students - total_students
  let painting_only := painting_students - both_workshops
  let sculpting_only := sculpting_students - both_workshops
  let total_combinations := total_students.choose 2
  let not_both_workshops := (painting_only.choose 2) + (sculpting_only.choose 2)
  (total_combinations - not_both_workshops : ℚ) / total_combinations = 56 / 62 :=
by sorry

end NUMINAMATH_CALUDE_workshop_probability_l1460_146039


namespace NUMINAMATH_CALUDE_sector_angle_l1460_146051

/-- Theorem: For a circular sector with perimeter 4 cm and area 1 cm², 
    the radian measure of its central angle is 2 radians. -/
theorem sector_angle (r : ℝ) (α : ℝ) 
  (h_perimeter : 2 * r + r * α = 4)
  (h_area : 1/2 * α * r^2 = 1) : 
  α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1460_146051


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l1460_146074

/-- 
Given two cyclists traveling in opposite directions for 2 hours,
with one traveling at 15 km/h and ending up 50 km apart,
prove that the speed of the other cyclist is 10 km/h.
-/
theorem cyclist_speed_problem (time : ℝ) (distance : ℝ) (speed_south : ℝ) (speed_north : ℝ) :
  time = 2 →
  distance = 50 →
  speed_south = 15 →
  (speed_north + speed_south) * time = distance →
  speed_north = 10 := by
sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l1460_146074


namespace NUMINAMATH_CALUDE_distribute_5_4_l1460_146088

/-- The number of ways to distribute n distinct objects into k identical containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 4 identical containers,
    allowing empty containers, is 37. -/
theorem distribute_5_4 : distribute 5 4 = 37 := by sorry

end NUMINAMATH_CALUDE_distribute_5_4_l1460_146088


namespace NUMINAMATH_CALUDE_expand_expression_l1460_146073

theorem expand_expression (x : ℝ) : (7*x - 3) * 5*x^2 = 35*x^3 - 15*x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1460_146073


namespace NUMINAMATH_CALUDE_largest_prime_factor_133_l1460_146042

def numbers : List Nat := [45, 65, 91, 85, 133]

def largest_prime_factor (n : Nat) : Nat :=
  sorry

theorem largest_prime_factor_133 :
  ∀ m ∈ numbers, m ≠ 133 → largest_prime_factor 133 > largest_prime_factor m :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_133_l1460_146042


namespace NUMINAMATH_CALUDE_fraction_simplification_l1460_146060

theorem fraction_simplification :
  1 / (1 / (1/2)^2 + 1 / (1/2)^3 + 1 / (1/2)^4 + 1 / (1/2)^5) = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1460_146060


namespace NUMINAMATH_CALUDE_farm_feet_count_l1460_146020

/-- Represents a farm with hens and cows -/
structure Farm where
  total_heads : ℕ
  num_hens : ℕ
  hen_feet : ℕ
  cow_feet : ℕ

/-- Calculates the total number of feet on the farm -/
def total_feet (f : Farm) : ℕ :=
  f.num_hens * f.hen_feet + (f.total_heads - f.num_hens) * f.cow_feet

/-- Theorem: Given a farm with 48 total animals, 24 hens, 2 feet per hen, and 4 feet per cow, 
    the total number of feet is 144 -/
theorem farm_feet_count : 
  ∀ (f : Farm), f.total_heads = 48 → f.num_hens = 24 → f.hen_feet = 2 → f.cow_feet = 4 
  → total_feet f = 144 := by
  sorry


end NUMINAMATH_CALUDE_farm_feet_count_l1460_146020


namespace NUMINAMATH_CALUDE_difference_of_squares_l1460_146005

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1460_146005


namespace NUMINAMATH_CALUDE_model_c_sample_size_l1460_146091

/-- Represents the total number of units produced -/
def total_population : ℕ := 1000

/-- Represents the number of units of model C -/
def model_c_population : ℕ := 300

/-- Represents the total sample size -/
def total_sample_size : ℕ := 60

/-- Calculates the number of units to be sampled from model C using stratified sampling -/
def stratified_sample_size (total_pop : ℕ) (model_pop : ℕ) (sample_size : ℕ) : ℕ :=
  (model_pop * sample_size) / total_pop

/-- Theorem stating that the stratified sample size for model C is 18 -/
theorem model_c_sample_size :
  stratified_sample_size total_population model_c_population total_sample_size = 18 := by
  sorry


end NUMINAMATH_CALUDE_model_c_sample_size_l1460_146091


namespace NUMINAMATH_CALUDE_parabola_and_line_theorem_l1460_146058

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the directrix
def directrix (p : ℝ) (x : ℝ) : Prop := x = -1

-- Define a point on the directrix
def directrix_point (x y : ℝ) : Prop := x = -1 ∧ y = 1

-- Define a line passing through the focus
def focus_line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1) ∧ k ≠ 0

-- Define the length of AB
def length_AB : ℝ := 5

-- Main theorem
theorem parabola_and_line_theorem (p : ℝ) :
  (∃ x y : ℝ, parabola p x y ∧ directrix p x ∧ directrix_point x y) →
  (∃ k : ℝ, ∀ x y : ℝ, parabola p x y ∧ focus_line k x y →
    (y^2 = 4*x ∧ (2*x - y - 2 = 0 ∨ 2*x + y - 2 = 0))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_theorem_l1460_146058


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1460_146076

-- Problem 1
theorem problem_one (x n : ℝ) (h : x^n = 2) : 
  (3*x^n)^2 - 4*(x^2)^n = 20 := by sorry

-- Problem 2
theorem problem_two (x y n : ℝ) (h1 : x = 2^n - 1) (h2 : y = 3 + 8^n) :
  y = 3 + (x + 1)^3 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1460_146076


namespace NUMINAMATH_CALUDE_a_investment_value_l1460_146000

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  c_profit_share : ℝ

/-- Theorem stating that given the conditions of the problem, a's investment is 30000. -/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 45000)
  (hc : p.c_investment = 50000)
  (htotal : p.total_profit = 90000)
  (hc_share : p.c_profit_share = 36000) :
  p.a_investment = 30000 := by
  sorry

#check a_investment_value

end NUMINAMATH_CALUDE_a_investment_value_l1460_146000


namespace NUMINAMATH_CALUDE_worker_completion_time_l1460_146090

/-- Given that two workers a and b can complete a job together in 8 days,
    and worker a alone can complete the job in 12 days,
    prove that worker b alone can complete the job in 24 days. -/
theorem worker_completion_time (a b : ℝ) 
  (h1 : a + b = 1 / 8)  -- a and b together complete 1/8 of the work per day
  (h2 : a = 1 / 12)     -- a alone completes 1/12 of the work per day
  : b = 1 / 24 :=       -- b alone completes 1/24 of the work per day
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l1460_146090


namespace NUMINAMATH_CALUDE_daves_initial_files_l1460_146055

theorem daves_initial_files (initial_apps : ℕ) (final_apps : ℕ) (final_files : ℕ) :
  initial_apps = 24 →
  final_apps = 12 →
  final_files = 5 →
  final_apps = final_files + 7 →
  initial_apps - final_apps + final_files = 17 := by
  sorry

end NUMINAMATH_CALUDE_daves_initial_files_l1460_146055


namespace NUMINAMATH_CALUDE_least_product_of_two_primes_above_50_l1460_146011

theorem least_product_of_two_primes_above_50 (p q : ℕ) : 
  p.Prime → q.Prime → p > 50 → q > 50 → p ≠ q → 
  ∃ (min_product : ℕ), min_product = 3127 ∧ 
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 50 → s > 50 → r ≠ s → 
      p * q ≤ r * s := by
  sorry

end NUMINAMATH_CALUDE_least_product_of_two_primes_above_50_l1460_146011


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l1460_146047

/-- 
Given a mixture of 1 liter of pure water and x liters of 30% salt solution,
resulting in a 15% salt solution, prove that x = 1.
-/
theorem salt_solution_mixture (x : ℝ) : 
  (0.30 * x = 0.15 * (1 + x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l1460_146047


namespace NUMINAMATH_CALUDE_divisibility_by_a_squared_l1460_146002

theorem divisibility_by_a_squared (a : ℤ) (n : ℕ) :
  ∃ k : ℤ, (a * n - 1) * (a + 1)^n + 1 = a^2 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_a_squared_l1460_146002


namespace NUMINAMATH_CALUDE_original_price_calculation_l1460_146004

theorem original_price_calculation (P : ℝ) : 
  (P * (1 - 0.3) * (1 - 0.2) = 1120) → P = 2000 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1460_146004


namespace NUMINAMATH_CALUDE_same_color_probability_l1460_146015

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 6

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selected_shoes : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
theorem same_color_probability :
  (num_pairs : ℚ) / (total_shoes.choose selected_shoes) = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1460_146015


namespace NUMINAMATH_CALUDE_nonagon_diagonal_count_l1460_146059

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonal_count : nonagon_diagonals = (nonagon_sides * (nonagon_sides - 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_count_l1460_146059


namespace NUMINAMATH_CALUDE_smores_marshmallows_needed_l1460_146075

def graham_crackers : ℕ := 48
def marshmallows : ℕ := 6
def crackers_per_smore : ℕ := 2
def marshmallows_per_smore : ℕ := 1

theorem smores_marshmallows_needed : 
  (graham_crackers / crackers_per_smore) - marshmallows = 18 :=
by sorry

end NUMINAMATH_CALUDE_smores_marshmallows_needed_l1460_146075


namespace NUMINAMATH_CALUDE_max_two_greater_than_half_l1460_146043

theorem max_two_greater_than_half (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  let values := [Real.sin α * Real.cos β, Real.sin β * Real.cos γ, Real.sin γ * Real.cos α]
  (values.filter (λ x => x > 1/2)).length ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_max_two_greater_than_half_l1460_146043


namespace NUMINAMATH_CALUDE_no_solution_system_l1460_146056

theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 18) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l1460_146056


namespace NUMINAMATH_CALUDE_probability_three_tails_one_head_l1460_146037

theorem probability_three_tails_one_head : 
  let n : ℕ := 4 -- number of coins
  let k : ℕ := 3 -- number of tails (or heads, whichever is larger)
  let p : ℚ := 1/2 -- probability of getting tails (or heads) for a single coin
  (n.choose k) * p^k * (1 - p)^(n - k) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_tails_one_head_l1460_146037


namespace NUMINAMATH_CALUDE_equation_roots_l1460_146067

theorem equation_roots : ∃ (x₁ x₂ : ℝ), 
  x₁ ≠ x₂ ∧ 
  x₁ = (29 + Real.sqrt 457) / 24 ∧ 
  x₂ = (29 - Real.sqrt 457) / 24 ∧ 
  ∀ x : ℝ, x ≠ 2 → 
    3 * x^2 / (x - 2) - (x + 4) / 4 + (7 - 9 * x) / (x - 2) + 2 = 0 ↔ 
    (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l1460_146067


namespace NUMINAMATH_CALUDE_lunks_for_two_dozen_bananas_l1460_146024

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℕ) : ℕ := l / 2

/-- Exchange rate between kunks and bananas -/
def kunks_to_bananas (k : ℕ) : ℕ := 2 * k

/-- Number of lunks needed to buy a given number of bananas -/
def lunks_for_bananas (b : ℕ) : ℕ :=
  let kunks_needed := (b + 5) / 6 * 3  -- Round up division
  2 * kunks_needed

theorem lunks_for_two_dozen_bananas :
  lunks_for_bananas 24 = 24 := by
  sorry

end NUMINAMATH_CALUDE_lunks_for_two_dozen_bananas_l1460_146024


namespace NUMINAMATH_CALUDE_father_child_ages_l1460_146065

theorem father_child_ages : ∃ (f b : ℕ), 
  13 ≤ b ∧ b ≤ 19 ∧ 
  100 * f + b - (f - b) = 4289 ∧ 
  f + b = 59 := by
sorry

end NUMINAMATH_CALUDE_father_child_ages_l1460_146065


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1460_146096

theorem complex_equation_solution (x : ℂ) : x / Complex.I = 1 - Complex.I → x = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1460_146096


namespace NUMINAMATH_CALUDE_white_marbles_in_bag_a_l1460_146040

/-- Represents the number of marbles of each color in Bag A -/
structure BagA where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Represents the ratios of marbles in Bag A -/
structure BagARatios where
  red_to_white : ℚ
  white_to_blue : ℚ

/-- Theorem stating that if Bag A contains 5 red marbles, it must contain 15 white marbles -/
theorem white_marbles_in_bag_a 
  (bag : BagA) 
  (ratios : BagARatios) 
  (h1 : ratios.red_to_white = 1 / 3) 
  (h2 : ratios.white_to_blue = 2 / 3) 
  (h3 : bag.red = 5) : 
  bag.white = 15 := by
  sorry

#check white_marbles_in_bag_a

end NUMINAMATH_CALUDE_white_marbles_in_bag_a_l1460_146040


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l1460_146012

/-- The perimeter of figure ABFCDE formed by cutting a right isosceles triangle from a square and translating it -/
theorem perimeter_of_modified_square (side_length : ℝ) : 
  side_length > 0 →
  4 * side_length = 64 →
  let perimeter_ABFCDE := 4 * side_length + 2 * side_length * Real.sqrt 2
  perimeter_ABFCDE = 64 + 32 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l1460_146012


namespace NUMINAMATH_CALUDE_johns_phd_time_l1460_146098

/-- Represents the duration of John's PhD journey in years -/
def total_phd_time (
  acclimation_period : ℝ)
  (basics_period : ℝ)
  (research_ratio : ℝ)
  (sabbatical1 : ℝ)
  (sabbatical2 : ℝ)
  (conference1 : ℝ)
  (conference2 : ℝ)
  (dissertation_ratio : ℝ)
  (dissertation_conference : ℝ) : ℝ :=
  acclimation_period +
  basics_period +
  (basics_period * (1 + research_ratio) + sabbatical1 + sabbatical2 + conference1 + conference2) +
  (acclimation_period * dissertation_ratio + dissertation_conference)

/-- Theorem stating that John's total PhD time is 8.75 years -/
theorem johns_phd_time :
  total_phd_time 1 2 0.75 0.5 0.25 (4/12) (5/12) 0.5 0.25 = 8.75 := by
  sorry


end NUMINAMATH_CALUDE_johns_phd_time_l1460_146098


namespace NUMINAMATH_CALUDE_nth_equation_pattern_l1460_146025

theorem nth_equation_pattern (n : ℕ+) : n^2 - n = n * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_pattern_l1460_146025


namespace NUMINAMATH_CALUDE_two_digit_number_with_specific_division_properties_l1460_146032

theorem two_digit_number_with_specific_division_properties :
  ∀ n : ℕ,
  (n ≥ 10 ∧ n ≤ 99) →
  (n % 6 = n / 10) →
  (n / 10 = 3 ∧ n % 10 = n % 10) →
  (n = 33 ∨ n = 39) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_with_specific_division_properties_l1460_146032


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1460_146016

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ - 2*m + 5 = 0 ∧ 
    x₂^2 - 4*x₂ - 2*m + 5 = 0 ∧
    x₁*x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1460_146016


namespace NUMINAMATH_CALUDE_min_value_and_inequality_solution_l1460_146031

def f (x a : ℝ) : ℝ := |x - a| + |x - 1|

theorem min_value_and_inequality_solution 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : ∀ x, f x a ≥ 2) 
  (h3 : ∃ x, f x a = 2) :
  (a = 3) ∧ 
  (∀ x, f x a ≥ 4 ↔ x ∈ Set.Iic 0 ∪ Set.Ici 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_solution_l1460_146031


namespace NUMINAMATH_CALUDE_two_item_combinations_l1460_146099

theorem two_item_combinations (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_item_combinations_l1460_146099


namespace NUMINAMATH_CALUDE_perpendicular_segments_in_cube_l1460_146094

/-- Represents a cube in 3D space -/
structure Cube where
  -- We don't need to define the specifics of a cube for this statement

/-- Represents a line segment in the cube (edge, face diagonal, or space diagonal) -/
structure LineSegment where
  -- We don't need to define the specifics of a line segment for this statement

/-- Checks if a line segment is perpendicular to a given edge of the cube -/
def is_perpendicular (c : Cube) (l : LineSegment) (edge : LineSegment) : Prop :=
  sorry -- Definition not needed for the statement

/-- Counts the number of line segments perpendicular to a given edge -/
def count_perpendicular_segments (c : Cube) (edge : LineSegment) : Nat :=
  sorry -- Definition not needed for the statement

/-- Theorem: The number of line segments perpendicular to any edge in a cube is 12 -/
theorem perpendicular_segments_in_cube (c : Cube) (edge : LineSegment) :
  count_perpendicular_segments c edge = 12 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_segments_in_cube_l1460_146094


namespace NUMINAMATH_CALUDE_show_revenue_l1460_146062

def tickets_first_showing : ℕ := 200
def ticket_price : ℕ := 25

theorem show_revenue : 
  (tickets_first_showing + 3 * tickets_first_showing) * ticket_price = 20000 := by
  sorry

end NUMINAMATH_CALUDE_show_revenue_l1460_146062


namespace NUMINAMATH_CALUDE_min_value_h_negative_reals_l1460_146021

-- Define odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the theorem
theorem min_value_h_negative_reals 
  (f g : ℝ → ℝ) 
  (hf : IsOdd f) 
  (hg : IsOdd g) 
  (h : ℝ → ℝ) 
  (h_def : ∀ x, h x = f x + g x - 2) 
  (h_max : ∃ M, M = 6 ∧ ∀ x > 0, h x ≤ M) :
  ∃ m, m = -10 ∧ ∀ x < 0, h x ≥ m := by
sorry

end NUMINAMATH_CALUDE_min_value_h_negative_reals_l1460_146021


namespace NUMINAMATH_CALUDE_odot_specific_values_odot_power_relation_l1460_146064

/-- Definition of the ⊙ operation for rational numbers -/
def odot (m n : ℚ) : ℚ := m * n * (m - n)

/-- Theorem for part 1 of the problem -/
theorem odot_specific_values :
  let a : ℚ := 1/2
  let b : ℚ := -1
  odot (a + b) (a - b) = 3/2 := by sorry

/-- Theorem for part 2 of the problem -/
theorem odot_power_relation (x y : ℚ) :
  odot (x^2 * y) (odot x y) = x^5 * y^4 - x^4 * y^5 := by sorry

end NUMINAMATH_CALUDE_odot_specific_values_odot_power_relation_l1460_146064


namespace NUMINAMATH_CALUDE_shelby_buys_three_posters_l1460_146093

/-- Calculates the number of posters Shelby can buy after her initial purchases and taxes --/
def posters_shelby_can_buy (initial_amount : ℚ) (book1_price : ℚ) (book2_price : ℚ) 
  (bookmark_price : ℚ) (pencils_price : ℚ) (tax_rate : ℚ) (poster_price : ℚ) : ℕ :=
  let total_before_tax := book1_price + book2_price + bookmark_price + pencils_price
  let total_with_tax := total_before_tax * (1 + tax_rate)
  let money_left := initial_amount - total_with_tax
  (money_left / poster_price).floor.toNat

/-- Theorem stating that Shelby can buy exactly 3 posters --/
theorem shelby_buys_three_posters : 
  posters_shelby_can_buy 50 12.50 7.25 2.75 3.80 0.07 5.50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_shelby_buys_three_posters_l1460_146093


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_powers_l1460_146001

theorem min_value_of_sum_of_powers (a b : ℝ) (h : a + b = 2) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x y : ℝ), x + y = 2 → 3^x + 3^y ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_powers_l1460_146001


namespace NUMINAMATH_CALUDE_f_decreasing_on_positive_reals_l1460_146022

/-- The function f(x) = -x^2 + 3 is decreasing on the interval (0, +∞) -/
theorem f_decreasing_on_positive_reals :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ → x₁ < x₂ → (-x₁^2 + 3) > (-x₂^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_positive_reals_l1460_146022


namespace NUMINAMATH_CALUDE_algebraic_operation_proof_l1460_146061

theorem algebraic_operation_proof (a b : ℝ) : 5 * a * b - 6 * a * b = -a * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_operation_proof_l1460_146061


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_eq_neg_eight_l1460_146070

-- Define the polynomial expression
def poly (x m : ℝ) : ℝ := (x^2 - x + m) * (x - 8)

-- Theorem statement
theorem no_linear_term_implies_m_eq_neg_eight :
  (∀ x : ℝ, ∃ a b c : ℝ, poly x m = a * x^3 + b * x^2 + c) → m = -8 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_eq_neg_eight_l1460_146070


namespace NUMINAMATH_CALUDE_smallest_n_for_fraction_l1460_146092

def fraction (n : ℕ) : ℚ :=
  (5^(n+1) + 2^(n+1)) / (5^n + 2^n)

theorem smallest_n_for_fraction :
  (∀ k : ℕ, k < 7 → fraction k ≤ 4.99) ∧
  fraction 7 > 4.99 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_fraction_l1460_146092


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l1460_146095

/-- A geometric sequence with integer common ratio -/
def GeometricSequence (a : ℕ → ℤ) (q : ℤ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a10 
  (a : ℕ → ℤ) 
  (q : ℤ) 
  (h_geom : GeometricSequence a q)
  (h_prod : a 4 * a 7 = -512)
  (h_sum : a 3 + a 8 = 124) :
  a 10 = 512 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l1460_146095


namespace NUMINAMATH_CALUDE_science_fiction_books_l1460_146078

/-- Represents the number of books in the science fiction section of a library. -/
def num_books : ℕ := 3824 / 478

/-- Theorem stating that the number of books in the science fiction section is 8. -/
theorem science_fiction_books : num_books = 8 := by
  sorry

end NUMINAMATH_CALUDE_science_fiction_books_l1460_146078


namespace NUMINAMATH_CALUDE_binomial_60_2_l1460_146026

theorem binomial_60_2 : Nat.choose 60 2 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_2_l1460_146026


namespace NUMINAMATH_CALUDE_parabolas_coefficient_sum_zero_l1460_146048

/-- Two distinct parabolas with leading coefficients p and q, where the vertex of each parabola lies on the other parabola -/
structure DistinctParabolas (p q : ℝ) : Prop where
  distinct : p ≠ q
  vertex_on_other : ∃ (a b : ℝ), a ≠ 0 ∧ b = p * a^2 ∧ 0 = q * a^2 + b

/-- The sum of leading coefficients of two distinct parabolas with vertices on each other is zero -/
theorem parabolas_coefficient_sum_zero {p q : ℝ} (h : DistinctParabolas p q) : p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_coefficient_sum_zero_l1460_146048


namespace NUMINAMATH_CALUDE_fraction_chain_l1460_146018

theorem fraction_chain (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by
sorry

end NUMINAMATH_CALUDE_fraction_chain_l1460_146018


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_11_mod_14_l1460_146045

theorem least_five_digit_congruent_to_11_mod_14 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  n % 14 = 11 ∧               -- congruent to 11 (mod 14)
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 14 = 11 → m ≥ n) ∧  -- least such number
  n = 10007 :=                -- the answer is 10007
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_11_mod_14_l1460_146045


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1460_146028

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a * r^5 / (1 - r)) / (a / (1 - r)) = 1 / 81 → r = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1460_146028


namespace NUMINAMATH_CALUDE_three_heads_probability_l1460_146097

-- Define a fair coin
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of three heads in a row
def three_heads_prob : ℚ := fair_coin_prob * fair_coin_prob * fair_coin_prob

-- Theorem statement
theorem three_heads_probability :
  three_heads_prob = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_three_heads_probability_l1460_146097


namespace NUMINAMATH_CALUDE_earth_orbit_radius_scientific_notation_l1460_146008

theorem earth_orbit_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 149000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.49 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_earth_orbit_radius_scientific_notation_l1460_146008


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l1460_146084

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- 
If the points (4, -6), (3b - 1, 5), and (b + 4, 4) are collinear, then b = 50/19.
-/
theorem collinear_points_b_value :
  ∀ b : ℝ, collinear 4 (-6) (3*b - 1) 5 (b + 4) 4 → b = 50/19 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l1460_146084


namespace NUMINAMATH_CALUDE_base8_subtraction_l1460_146007

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def base8ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def natToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The subtraction operation in base 8. -/
def base8Sub (a b : List Nat) : List Nat :=
  natToBase8 (base8ToNat a - base8ToNat b)

theorem base8_subtraction :
  base8Sub [4, 5, 3] [3, 2, 6] = [1, 2, 5] := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_l1460_146007


namespace NUMINAMATH_CALUDE_tire_price_proof_l1460_146085

theorem tire_price_proof : 
  let fourth_tire_discount : ℝ := 0.75
  let total_cost : ℝ := 270
  let regular_price : ℝ := 72
  (3 * regular_price + fourth_tire_discount * regular_price = total_cost) →
  regular_price = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_tire_price_proof_l1460_146085


namespace NUMINAMATH_CALUDE_angle_between_polar_lines_eq_arctan_half_l1460_146006

/-- The angle between two lines in polar coordinates -/
def angle_between_polar_lines (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- First line equation in polar coordinates -/
def line1 (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ + 2 * Real.sin θ) = 1

/-- Second line equation in polar coordinates -/
def line2 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ = 1

/-- Theorem stating the angle between the two given lines -/
theorem angle_between_polar_lines_eq_arctan_half :
  angle_between_polar_lines line1 line2 = Real.arctan (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_angle_between_polar_lines_eq_arctan_half_l1460_146006


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_360_l1460_146057

/-- A rectangular table with 6 seats -/
structure RectangularTable :=
  (total_seats : ℕ)
  (longer_side_seats : ℕ)
  (shorter_side_seats : ℕ)
  (h_total : total_seats = 2 * longer_side_seats + 2 * shorter_side_seats)
  (h_longer : longer_side_seats = 2)
  (h_shorter : shorter_side_seats = 1)

/-- The number of ways to seat 5 persons at a rectangular table with 6 seats -/
def seating_arrangements (table : RectangularTable) (persons : ℕ) : ℕ := 
  3 * Nat.factorial (table.total_seats - 1)

/-- Theorem stating that the number of seating arrangements for 5 persons
    at the specified rectangular table is 360 -/
theorem seating_arrangements_eq_360 (table : RectangularTable) :
  seating_arrangements table 5 = 360 := by
  sorry

#eval seating_arrangements ⟨6, 2, 1, rfl, rfl, rfl⟩ 5

end NUMINAMATH_CALUDE_seating_arrangements_eq_360_l1460_146057


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1460_146071

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (7^14 + 11^15) ∧ ∀ (q : ℕ), q.Prime → q ∣ (7^14 + 11^15) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1460_146071


namespace NUMINAMATH_CALUDE_prank_week_combinations_l1460_146014

/-- The number of available helpers for each day of the week -/
def helpers_per_day : List Nat := [1, 2, 3, 4, 1]

/-- The total number of possible combinations of helpers throughout the week -/
def total_combinations : Nat := List.prod helpers_per_day

/-- Theorem stating that the total number of combinations is 24 -/
theorem prank_week_combinations :
  total_combinations = 24 := by
  sorry

end NUMINAMATH_CALUDE_prank_week_combinations_l1460_146014


namespace NUMINAMATH_CALUDE_lesser_fraction_l1460_146086

theorem lesser_fraction (x y : ℚ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x + y = 13/14) 
  (h4 : x * y = 1/8) : 
  min x y = 163/625 := by
sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1460_146086


namespace NUMINAMATH_CALUDE_complex_real_part_twice_imaginary_l1460_146063

theorem complex_real_part_twice_imaginary (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (Complex.re z = 2 * Complex.im z) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_real_part_twice_imaginary_l1460_146063


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l1460_146046

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The specific problem of a ball dropped from 120 feet with 1/3 rebound ratio -/
theorem ball_bounce_distance :
  totalDistance 120 (1/3) 5 = 248 + 26/27 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l1460_146046


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1460_146080

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  (∀ y, -49 ≤ y ∧ y ≤ 49 → Real.sqrt (49 + y) + Real.sqrt (49 - y) ≤ Real.sqrt (49 + x) + Real.sqrt (49 - x)) →
  Real.sqrt (49 + x) + Real.sqrt (49 - x) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1460_146080


namespace NUMINAMATH_CALUDE_lionel_distance_walked_l1460_146035

/-- The distance between Lionel's and Walt's houses -/
def total_distance : ℝ := 48

/-- Lionel's walking speed in miles per hour -/
def lionel_speed : ℝ := 2

/-- Walt's running speed in miles per hour -/
def walt_speed : ℝ := 6

/-- The time Walt waits before starting to run, in hours -/
def walt_wait_time : ℝ := 2

/-- The theorem stating that Lionel walked 15 miles when he met Walt -/
theorem lionel_distance_walked : ℝ := by
  sorry

end NUMINAMATH_CALUDE_lionel_distance_walked_l1460_146035


namespace NUMINAMATH_CALUDE_solution_set_x_squared_leq_one_l1460_146083

theorem solution_set_x_squared_leq_one :
  ∀ x : ℝ, x^2 ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_leq_one_l1460_146083


namespace NUMINAMATH_CALUDE_max_a_value_l1460_146052

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := (x - t) * abs x

-- State the theorem
theorem max_a_value (t : ℝ) (h : t ∈ Set.Ioo 0 2) :
  (∃ a : ℝ, ∀ x ∈ Set.Icc (-1) 2, f t x > x + a) →
  (∃ a : ℝ, (∀ x ∈ Set.Icc (-1) 2, f t x > x + a) ∧ a = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l1460_146052
