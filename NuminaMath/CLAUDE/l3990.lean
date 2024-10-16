import Mathlib

namespace NUMINAMATH_CALUDE_zhang_fei_probabilities_l3990_399007

/-- The set of events Zhang Fei can participate in -/
inductive Event : Type
  | LongJump : Event
  | Meters100 : Event
  | Meters200 : Event
  | Meters400 : Event

/-- The probability of selecting an event -/
def selectProbability (e : Event) : ℚ :=
  1 / 4

/-- The probability of selecting two specific events when choosing at most two events -/
def selectTwoEventsProbability (e1 e2 : Event) : ℚ :=
  2 / 12

theorem zhang_fei_probabilities :
  (selectProbability Event.LongJump = 1 / 4) ∧
  (selectTwoEventsProbability Event.LongJump Event.Meters100 = 1 / 6) := by
  sorry

end NUMINAMATH_CALUDE_zhang_fei_probabilities_l3990_399007


namespace NUMINAMATH_CALUDE_problem_1_l3990_399047

theorem problem_1 : (-2)^2 + Real.sqrt 12 - 2 * Real.sin (π / 3) = 4 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3990_399047


namespace NUMINAMATH_CALUDE_union_condition_intersection_empty_l3990_399038

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 7}
def B (t : ℝ) : Set ℝ := {x : ℝ | t + 1 ≤ x ∧ x ≤ 2*t - 2}

-- Statement 1: A ∪ B = A if and only if t ∈ (-∞, 9/2]
theorem union_condition (t : ℝ) : A ∪ B t = A ↔ t ≤ 9/2 := by sorry

-- Statement 2: A ∩ B = ∅ if and only if t ∈ (-∞, 3) ∪ (6, +∞)
theorem intersection_empty (t : ℝ) : A ∩ B t = ∅ ↔ t < 3 ∨ t > 6 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_empty_l3990_399038


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l3990_399008

theorem square_difference_fourth_power : (6^2 - 3^2)^4 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l3990_399008


namespace NUMINAMATH_CALUDE_restaurant_cooks_count_l3990_399034

/-- Proves that the number of cooks is 9 given the initial and final ratios of cooks to waiters -/
theorem restaurant_cooks_count : ∀ (C W : ℕ),
  C / W = 3 / 11 →
  C / (W + 12) = 1 / 5 →
  C = 9 := by
sorry

end NUMINAMATH_CALUDE_restaurant_cooks_count_l3990_399034


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l3990_399040

def bag_weights : List Nat := [13, 15, 16, 17, 21, 24]

def total_weight : Nat := bag_weights.sum

structure BagDistribution where
  turnip_weight : Nat
  onion_weights : List Nat
  carrot_weights : List Nat

def is_valid_distribution (d : BagDistribution) : Prop :=
  d.turnip_weight ∈ bag_weights ∧
  (d.onion_weights ++ d.carrot_weights).sum = total_weight - d.turnip_weight ∧
  d.carrot_weights.sum = 2 * d.onion_weights.sum ∧
  (d.onion_weights ++ d.carrot_weights).toFinset ⊆ bag_weights.toFinset.erase d.turnip_weight

theorem turnip_bag_weights :
  ∀ d : BagDistribution, is_valid_distribution d → d.turnip_weight = 13 ∨ d.turnip_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l3990_399040


namespace NUMINAMATH_CALUDE_no_consecutive_squares_l3990_399090

def t (n : ℕ) : ℕ := (Nat.divisors n).card

def a : ℕ → ℕ
  | 0 => 1  -- Arbitrary starting value
  | n + 1 => a n + 2 * t n

theorem no_consecutive_squares (n k : ℕ) :
  a n = k^2 → ¬∃ m : ℕ, a (n + 1) = (k + m)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_consecutive_squares_l3990_399090


namespace NUMINAMATH_CALUDE_enclosed_area_equals_four_l3990_399028

-- Define the functions for the line and curve
def f (x : ℝ) : ℝ := 4 * x
def g (x : ℝ) : ℝ := x^3

-- Define the intersection points
def x₁ : ℝ := 0
def x₂ : ℝ := 2

-- State the theorem
theorem enclosed_area_equals_four :
  (∫ x in x₁..x₂, f x - g x) = 4 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_equals_four_l3990_399028


namespace NUMINAMATH_CALUDE_intersection_dot_product_l3990_399021

/-- Given a line and a parabola that intersect at points A and B, and a point M,
    prove that if the dot product of MA and MB is zero, then the y-coordinate of M is √2/2. -/
theorem intersection_dot_product (A B M : ℝ × ℝ) : 
  (∃ x y, y = 2 * Real.sqrt 2 * (x - 1) ∧ y^2 = 4 * x ∧ A = (x, y)) →  -- Line and parabola intersection for A
  (∃ x y, y = 2 * Real.sqrt 2 * (x - 1) ∧ y^2 = 4 * x ∧ B = (x, y)) →  -- Line and parabola intersection for B
  M.1 = -1 →  -- x-coordinate of M is -1
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0 →  -- Dot product of MA and MB is zero
  M.2 = Real.sqrt 2 / 2 := by  -- y-coordinate of M is √2/2
sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l3990_399021


namespace NUMINAMATH_CALUDE_billy_horses_count_l3990_399080

/-- The number of horses Billy has -/
def num_horses : ℕ := 4

/-- The amount of oats (in pounds) each horse eats per feeding -/
def oats_per_feeding : ℕ := 4

/-- The number of feedings per day -/
def feedings_per_day : ℕ := 2

/-- The number of days Billy needs to feed his horses -/
def days_to_feed : ℕ := 3

/-- The total amount of oats (in pounds) Billy needs for all his horses for the given days -/
def total_oats_needed : ℕ := 96

theorem billy_horses_count : 
  num_horses * oats_per_feeding * feedings_per_day * days_to_feed = total_oats_needed :=
sorry

end NUMINAMATH_CALUDE_billy_horses_count_l3990_399080


namespace NUMINAMATH_CALUDE_smallest_divisible_power_l3990_399063

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisible_power : 
  ∃! k : ℕ+, (∀ z : ℂ, f z ∣ (z^k.val - 1)) ∧ 
  (∀ m : ℕ+, m < k → ∃ z : ℂ, ¬(f z ∣ (z^m.val - 1))) ∧ 
  k.val = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_power_l3990_399063


namespace NUMINAMATH_CALUDE_ratio_proof_l3990_399052

theorem ratio_proof (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 2) : (a + b) / (b + c) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l3990_399052


namespace NUMINAMATH_CALUDE_work_days_solution_l3990_399071

/-- The number of days worked by person a -/
def days_a : ℕ := 6

/-- The number of days worked by person b -/
def days_b : ℕ := 9

/-- The number of days worked by person c -/
def days_c : ℕ := 4

/-- The daily wage of person c -/
def wage_c : ℕ := 100

/-- The total earnings of all three workers -/
def total_earnings : ℕ := 1480

/-- The ratio of daily wages for a, b, and c respectively -/
def wage_ratio : Fin 3 → ℕ
| 0 => 3
| 1 => 4
| 2 => 5

theorem work_days_solution :
  ∃ (wage_a wage_b : ℕ),
    wage_a = wage_ratio 0 * (wage_c / wage_ratio 2) ∧
    wage_b = wage_ratio 1 * (wage_c / wage_ratio 2) ∧
    wage_a * days_a + wage_b * days_b + wage_c * days_c = total_earnings :=
by sorry


end NUMINAMATH_CALUDE_work_days_solution_l3990_399071


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3990_399070

theorem root_sum_reciprocal (p q r : ℝ) : 
  (p^3 - p - 6 = 0) → 
  (q^3 - q - 6 = 0) → 
  (r^3 - r - 6 = 0) → 
  (1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 11 / 12) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3990_399070


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3990_399088

/-- A circle in the 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle. -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ x y : ℝ, c.equation x y ↔ (x - 1)^2 + y^2 = 1) ∧
                  c.center = (1, 0) ∧
                  c.radius = 1 := by
  sorry


end NUMINAMATH_CALUDE_circle_center_and_radius_l3990_399088


namespace NUMINAMATH_CALUDE_angle_sum_in_triangle_l3990_399041

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- State the theorem
theorem angle_sum_in_triangle (t : Triangle) 
  (h1 : t.A = 65)
  (h2 : t.B = 40) : 
  t.C = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_triangle_l3990_399041


namespace NUMINAMATH_CALUDE_largest_constant_divisor_l3990_399064

theorem largest_constant_divisor (n : ℤ) : 
  let x : ℤ := 4 * n - 1
  ∃ (k : ℤ), (12 * x + 2) * (8 * x + 6) * (6 * x + 3) = 60 * k ∧ 
  ∀ (m : ℤ), m > 60 → 
    ∃ (l : ℤ), (12 * x + 2) * (8 * x + 6) * (6 * x + 3) ≠ m * l :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_divisor_l3990_399064


namespace NUMINAMATH_CALUDE_matrix_eigenpair_l3990_399012

/-- Given a 2x2 matrix M with eigenvector [1, 1] for eigenvalue 1, 
    prove it has another eigenvalue 2 with eigenvector [1, 0] -/
theorem matrix_eigenpair (a b : ℝ) : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, a; 0, b]
  (M.mulVec ![1, 1] = ![1, 1]) →
  (M.mulVec ![1, 0] = 2 • ![1, 0]) := by
sorry

end NUMINAMATH_CALUDE_matrix_eigenpair_l3990_399012


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l3990_399077

/-- Given three squares A, B, and C with specific perimeters, 
    this theorem proves the ratio of areas of A to C. -/
theorem area_ratio_of_squares (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (pa : 4 * a = 16) -- perimeter of A is 16
  (pb : 4 * b = 40) -- perimeter of B is 40
  (pc : 4 * c = 120) -- perimeter of C is 120 (3 times B's perimeter)
  : (a * a) / (c * c) = 4 / 225 := by
  sorry

#check area_ratio_of_squares

end NUMINAMATH_CALUDE_area_ratio_of_squares_l3990_399077


namespace NUMINAMATH_CALUDE_f_min_at_neg_seven_l3990_399042

/-- The quadratic function f(x) = x^2 + 14x + 6 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 6

/-- Theorem: The minimum value of f(x) occurs when x = -7 -/
theorem f_min_at_neg_seven :
  ∀ x : ℝ, f x ≥ f (-7) := by sorry

end NUMINAMATH_CALUDE_f_min_at_neg_seven_l3990_399042


namespace NUMINAMATH_CALUDE_city_population_l3990_399092

/-- Represents the population distribution of a city -/
structure CityPopulation where
  total : ℕ
  under18 : ℕ
  between18and65 : ℕ
  over65 : ℕ
  belowPovertyLine : ℕ
  middleClass : ℕ
  wealthy : ℕ
  menUnder18 : ℕ
  womenUnder18 : ℕ

/-- Theorem stating the total population of the city given the conditions -/
theorem city_population (c : CityPopulation) : c.total = 500000 :=
  by
  have h1 : c.under18 = c.total / 4 := sorry
  have h2 : c.between18and65 = c.total * 11 / 20 := sorry
  have h3 : c.over65 = c.total / 5 := sorry
  have h4 : c.belowPovertyLine = c.total * 3 / 20 := sorry
  have h5 : c.middleClass = c.total * 13 / 20 := sorry
  have h6 : c.wealthy = c.total / 5 := sorry
  have h7 : c.menUnder18 = c.under18 * 3 / 5 := sorry
  have h8 : c.womenUnder18 = c.under18 * 2 / 5 := sorry
  have h9 : c.wealthy * 1 / 5 = 20000 := sorry
  sorry

#check city_population

end NUMINAMATH_CALUDE_city_population_l3990_399092


namespace NUMINAMATH_CALUDE_mrs_hilt_bug_count_l3990_399099

theorem mrs_hilt_bug_count (flowers_per_bug : ℕ) (total_flowers : ℕ) (num_bugs : ℕ) : 
  flowers_per_bug = 2 →
  total_flowers = 6 →
  num_bugs * flowers_per_bug = total_flowers →
  num_bugs = 3 := by
sorry

end NUMINAMATH_CALUDE_mrs_hilt_bug_count_l3990_399099


namespace NUMINAMATH_CALUDE_area_FYH_specific_l3990_399014

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Calculates the area of triangle FYH in a trapezoid -/
def area_FYH (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of triangle FYH in the specific trapezoid -/
theorem area_FYH_specific : 
  let t : Trapezoid := { base1 := 24, base2 := 36, area := 360 }
  area_FYH t = 86.4 := by
  sorry

end NUMINAMATH_CALUDE_area_FYH_specific_l3990_399014


namespace NUMINAMATH_CALUDE_second_rewind_time_l3990_399060

theorem second_rewind_time (total_time first_segment first_rewind second_segment third_segment : ℕ) : 
  total_time = 120 ∧ 
  first_segment = 35 ∧ 
  first_rewind = 5 ∧ 
  second_segment = 45 ∧ 
  third_segment = 20 → 
  total_time - (first_segment + first_rewind + second_segment + third_segment) = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_rewind_time_l3990_399060


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3990_399086

theorem imaginary_part_of_z (m : ℝ) : 
  let z : ℂ := 1 - m * Complex.I
  (z ^ 2 = -2 * Complex.I) → (z.im = -1) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3990_399086


namespace NUMINAMATH_CALUDE_playlist_song_length_l3990_399094

theorem playlist_song_length 
  (n_short_songs : ℕ) 
  (short_song_length : ℕ) 
  (n_long_songs : ℕ) 
  (total_duration : ℕ) 
  (additional_time_needed : ℕ) 
  (h1 : n_short_songs = 10)
  (h2 : short_song_length = 3)
  (h3 : n_long_songs = 15)
  (h4 : total_duration = 100)
  (h5 : additional_time_needed = 40) :
  ∃ (long_song_length : ℚ),
    long_song_length = 14/3 ∧ 
    n_short_songs * short_song_length + n_long_songs * long_song_length = total_duration := by
  sorry

end NUMINAMATH_CALUDE_playlist_song_length_l3990_399094


namespace NUMINAMATH_CALUDE_square_side_length_l3990_399029

theorem square_side_length (area : Real) (side : Real) : 
  area = 25 → side * side = area → side = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3990_399029


namespace NUMINAMATH_CALUDE_total_muffins_for_sale_l3990_399024

theorem total_muffins_for_sale : 
  let num_boys : ℕ := 3
  let num_girls : ℕ := 2
  let muffins_per_boy : ℕ := 12
  let muffins_per_girl : ℕ := 20
  let total_muffins : ℕ := num_boys * muffins_per_boy + num_girls * muffins_per_girl
  total_muffins = 76 :=
by
  sorry

end NUMINAMATH_CALUDE_total_muffins_for_sale_l3990_399024


namespace NUMINAMATH_CALUDE_stickers_needed_for_both_prizes_l3990_399046

def current_stickers : ℕ := 250
def small_prize_requirement : ℕ := 800
def big_prize_requirement : ℕ := 1500

theorem stickers_needed_for_both_prizes :
  (small_prize_requirement - current_stickers) + (big_prize_requirement - current_stickers) = 1800 :=
by sorry

end NUMINAMATH_CALUDE_stickers_needed_for_both_prizes_l3990_399046


namespace NUMINAMATH_CALUDE_inverse_proportion_change_l3990_399010

theorem inverse_proportion_change (x y k q : ℝ) :
  x > 0 → y > 0 → q > 0 → q < 100 →
  x * y = k →
  let x' := x * (1 - q / 100)
  let y' := k / x'
  (y' - y) / y * 100 = 100 * q / (100 - q) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_change_l3990_399010


namespace NUMINAMATH_CALUDE_simplify_expression_l3990_399054

theorem simplify_expression (n : ℕ) : (2^(n+4) - 3*(2^n)) / (2*(2^(n+3))) = 13/16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3990_399054


namespace NUMINAMATH_CALUDE_birthday_crayons_count_l3990_399057

/-- The number of crayons Paul got for his birthday -/
def birthday_crayons : ℕ := sorry

/-- The number of crayons Paul got at the end of the school year -/
def school_year_crayons : ℕ := 134

/-- The total number of crayons Paul has now -/
def total_crayons : ℕ := 613

/-- Theorem stating that the number of crayons Paul got for his birthday is 479 -/
theorem birthday_crayons_count : birthday_crayons = 479 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_count_l3990_399057


namespace NUMINAMATH_CALUDE_a_range_for_increasing_f_l3990_399039

/-- A cubic function f(x) that is increasing on the entire real line. -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + 7*a*x

/-- The property that f is increasing on the entire real line. -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The theorem stating the range of a for which f is increasing. -/
theorem a_range_for_increasing_f :
  ∀ a : ℝ, is_increasing (f a) ↔ 0 ≤ a ∧ a ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_a_range_for_increasing_f_l3990_399039


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3990_399002

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distributeBalls 5 3 = 21 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3990_399002


namespace NUMINAMATH_CALUDE_box_weight_is_42_l3990_399066

/-- The weight of a box of books -/
def box_weight (book_weight : ℕ) (num_books : ℕ) : ℕ :=
  book_weight * num_books

/-- Theorem: The weight of a box of books is 42 pounds -/
theorem box_weight_is_42 : box_weight 3 14 = 42 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_is_42_l3990_399066


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3990_399026

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧
  (∀ n : ℕ, a n < a (n + 1)) ∧
  (a 3)^2 - 10 * (a 3) + 16 = 0 ∧
  (a 6)^2 - 10 * (a 6) + 16 = 0

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  (∀ n : ℕ, a n = 2 * n - 4) ∧ (a 136 = 268) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3990_399026


namespace NUMINAMATH_CALUDE_total_snakes_l3990_399072

/-- Given information about pet ownership, prove the total number of snakes. -/
theorem total_snakes (total_people : ℕ) (only_dogs : ℕ) (only_cats : ℕ) (only_snakes : ℕ)
  (dogs_and_cats : ℕ) (cats_and_snakes : ℕ) (dogs_and_snakes : ℕ) (all_three : ℕ)
  (h1 : total_people = 120)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_snakes = 12)
  (h5 : dogs_and_cats = 15)
  (h6 : cats_and_snakes = 10)
  (h7 : dogs_and_snakes = 8)
  (h8 : all_three = 5) :
  only_snakes + cats_and_snakes + dogs_and_snakes + all_three = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_snakes_l3990_399072


namespace NUMINAMATH_CALUDE_stream_speed_l3990_399015

/-- Given a boat's travel times and distances, calculate the stream speed -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 90) 
  (h2 : upstream_distance = 72)
  (h3 : time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3990_399015


namespace NUMINAMATH_CALUDE_ticket_price_difference_l3990_399032

def total_cost : ℝ := 77
def adult_ticket_cost : ℝ := 19
def num_adults : ℕ := 2
def num_children : ℕ := 3

theorem ticket_price_difference : 
  ∃ (child_ticket_cost : ℝ),
    total_cost = num_adults * adult_ticket_cost + num_children * child_ticket_cost ∧
    adult_ticket_cost - child_ticket_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_ticket_price_difference_l3990_399032


namespace NUMINAMATH_CALUDE_bus_stoppage_time_l3990_399087

theorem bus_stoppage_time (s1 s2 s3 v1 v2 v3 : ℝ) 
  (h1 : s1 = 54) (h2 : s2 = 60) (h3 : s3 = 72)
  (h4 : v1 = 36) (h5 : v2 = 40) (h6 : v3 = 48) :
  (1 - v1 / s1) + (1 - v2 / s2) + (1 - v3 / s3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_bus_stoppage_time_l3990_399087


namespace NUMINAMATH_CALUDE_standing_arrangements_eq_48_l3990_399033

/-- The number of different standing arrangements for 5 students in a row,
    given the specified conditions. -/
def standing_arrangements : ℕ :=
  let total_students : ℕ := 5
  let positions_for_A : ℕ := total_students - 1
  let remaining_positions : ℕ := total_students - 1
  let arrangements_for_D_and_E : ℕ := remaining_positions * (remaining_positions - 1) / 2
  positions_for_A * arrangements_for_D_and_E

/-- Theorem stating that the number of standing arrangements is 48. -/
theorem standing_arrangements_eq_48 : standing_arrangements = 48 := by
  sorry

#eval standing_arrangements  -- This should output 48

end NUMINAMATH_CALUDE_standing_arrangements_eq_48_l3990_399033


namespace NUMINAMATH_CALUDE_project_assignment_l3990_399061

theorem project_assignment (total_people : ℕ) (total_months : ℕ) (n : ℕ) : 
  total_people = 60 → 
  total_months = 10 → 
  (∀ assignment : ℕ → Finset ℕ, 
    (∀ month, month ∈ Finset.range total_months → (assignment month).card ≤ total_people) →
    (∀ person, person ∈ Finset.range total_people → ∃ month, month ∈ Finset.range total_months ∧ person ∈ assignment month) →
    ∃ month, month ∈ Finset.range total_months ∧ (assignment month).card ≥ n) →
  n ≤ 6 ∧ 
  ∃ assignment : ℕ → Finset ℕ,
    (∀ month, month ∈ Finset.range total_months → (assignment month).card ≤ total_people) ∧
    (∀ person, person ∈ Finset.range total_people → ∃ month, month ∈ Finset.range total_months ∧ person ∈ assignment month) ∧
    (∀ month, month ∈ Finset.range total_months → (assignment month).card < 7) :=
by
  sorry

end NUMINAMATH_CALUDE_project_assignment_l3990_399061


namespace NUMINAMATH_CALUDE_three_cubes_of_27_equals_3_to_10_l3990_399013

theorem three_cubes_of_27_equals_3_to_10 : ∃ x : ℕ, 27^3 + 27^3 + 27^3 = 3^x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_three_cubes_of_27_equals_3_to_10_l3990_399013


namespace NUMINAMATH_CALUDE_common_remainder_difference_l3990_399025

theorem common_remainder_difference (d r : ℕ) : 
  d.Prime → 
  d > 1 → 
  r < d → 
  1274 % d = r → 
  1841 % d = r → 
  2866 % d = r → 
  d - r = 6 := by sorry

end NUMINAMATH_CALUDE_common_remainder_difference_l3990_399025


namespace NUMINAMATH_CALUDE_divisibility_condition_l3990_399075

theorem divisibility_condition (n : ℤ) : (3 * n + 7) ∣ (5 * n + 13) ↔ n ∈ ({-3, -2, -1} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3990_399075


namespace NUMINAMATH_CALUDE_evaluate_expression_l3990_399048

theorem evaluate_expression : (3^2)^2 - (2^3)^3 = -431 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3990_399048


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_f_l3990_399009

-- Define the function
def f (x : ℝ) : ℝ := |x - 2| - 1

-- State the theorem
theorem monotonic_increasing_interval_f :
  ∀ a b : ℝ, a ≥ 2 → b ≥ 2 → a ≤ b → f a ≤ f b :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_f_l3990_399009


namespace NUMINAMATH_CALUDE_vidyas_age_l3990_399006

theorem vidyas_age (vidya_age : ℕ) (mother_age : ℕ) : 
  mother_age = 3 * vidya_age + 5 →
  mother_age = 44 →
  vidya_age = 13 := by
sorry

end NUMINAMATH_CALUDE_vidyas_age_l3990_399006


namespace NUMINAMATH_CALUDE_ten_possible_values_for_d_l3990_399096

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct_digits (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts a five-digit number represented by individual digits to a natural number -/
def to_nat (a b c d e : Digit) : ℕ :=
  10000 * a.val + 1000 * b.val + 100 * c.val + 10 * d.val + e.val

/-- The main theorem stating that there are 10 possible values for D -/
theorem ten_possible_values_for_d :
  ∃ (possible_d_values : Finset Digit),
    possible_d_values.card = 10 ∧
    ∀ (a b c d : Digit),
      distinct_digits a b c d →
      (to_nat a b c b c) + (to_nat c b a d b) = (to_nat d b d d d) →
      d ∈ possible_d_values :=
sorry

end NUMINAMATH_CALUDE_ten_possible_values_for_d_l3990_399096


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3990_399037

/-- Given a line L1 with equation 4x + 5y = 10 and a perpendicular line L2 with y-intercept -3,
    the x-intercept of L2 is 12/5 -/
theorem perpendicular_line_x_intercept :
  ∀ (L1 L2 : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ L1 ↔ 4 * x + 5 * y = 10) →
  (∃ m : ℝ, ∀ x y, (x, y) ∈ L2 ↔ y = m * x - 3) →
  (∀ x y₁ y₂, (x, y₁) ∈ L1 ∧ (x, y₂) ∈ L2 → (y₂ - y₁) * (4 * (x + 1) + 5 * y₁ - 10) = 0) →
  (0, -3) ∈ L2 →
  (12/5, 0) ∈ L2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3990_399037


namespace NUMINAMATH_CALUDE_ratio_equality_l3990_399018

theorem ratio_equality (x y z : ℝ) 
  (h1 : x * y * z ≠ 0) 
  (h2 : 2 * x * y = 3 * y * z) 
  (h3 : 3 * y * z = 5 * x * z) : 
  (x + 3 * y - 3 * z) / (x + 3 * y - 6 * z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3990_399018


namespace NUMINAMATH_CALUDE_simplest_square_root_l3990_399050

theorem simplest_square_root :
  let options := [Real.sqrt 8, (Real.sqrt 2)⁻¹, Real.sqrt 2, Real.sqrt (1/2)]
  ∃ (x : ℝ), x ∈ options ∧ 
    (∀ y ∈ options, x ≠ y → (∃ z : ℝ, z ≠ 1 ∧ y = z * x ∨ y = x / z ∨ y = Real.sqrt (z * x^2))) ∧
    x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplest_square_root_l3990_399050


namespace NUMINAMATH_CALUDE_magician_trick_minimum_digits_l3990_399062

theorem magician_trick_minimum_digits : ∃ (N : ℕ), N = 101 ∧ 
  (∀ (k : ℕ), k < N → (k - 1) * 10^(k - 2) < 10^k) ∧
  ((N - 1) * 10^(N - 2) ≥ 10^N) := by
  sorry

end NUMINAMATH_CALUDE_magician_trick_minimum_digits_l3990_399062


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3990_399056

theorem simplify_and_evaluate (a : ℕ) (ha : a = 2030) :
  (a + 1 : ℚ) / a - a / (a + 1) = (2 * a + 1 : ℚ) / (a * (a + 1)) ∧
  2 * a + 1 = 4061 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3990_399056


namespace NUMINAMATH_CALUDE_book_pages_total_l3990_399083

/-- A book with 5 chapters, each containing 111 pages, has a total of 555 pages. -/
theorem book_pages_total (num_chapters : ℕ) (pages_per_chapter : ℕ) :
  num_chapters = 5 → pages_per_chapter = 111 → num_chapters * pages_per_chapter = 555 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_total_l3990_399083


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3990_399079

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 5| = 3 * x + 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3990_399079


namespace NUMINAMATH_CALUDE_c_share_of_rent_l3990_399004

/-- Represents the usage of the pasture by a person -/
structure Usage where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a given usage -/
def calculateShare (u : Usage) (totalRent : ℕ) (totalUsage : ℕ) : ℚ :=
  (u.oxen * u.months : ℚ) / totalUsage * totalRent

/-- The main theorem stating C's share of the rent -/
theorem c_share_of_rent :
  let a := Usage.mk 10 7
  let b := Usage.mk 12 5
  let c := Usage.mk 15 3
  let totalRent := 210
  let totalUsage := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  calculateShare c totalRent totalUsage = 54 := by
  sorry


end NUMINAMATH_CALUDE_c_share_of_rent_l3990_399004


namespace NUMINAMATH_CALUDE_sons_age_l3990_399053

/-- Proves that the son's current age is 16 years given the specified conditions -/
theorem sons_age (son_age father_age : ℕ) : 
  father_age = 4 * son_age →
  (son_age - 10) + (father_age - 10) = 60 →
  son_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l3990_399053


namespace NUMINAMATH_CALUDE_acute_angle_specific_circles_l3990_399098

/-- The acute angle formed by two lines intersecting three concentric circles -/
def acute_angle_concentric_circles (r1 r2 r3 : ℝ) (shaded_ratio : ℝ) : ℝ :=
  sorry

/-- The theorem stating the acute angle for the given problem -/
theorem acute_angle_specific_circles :
  acute_angle_concentric_circles 5 3 1 (10/17) = 107/459 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_specific_circles_l3990_399098


namespace NUMINAMATH_CALUDE_expression_evaluation_l3990_399031

theorem expression_evaluation :
  (∀ a : ℤ, a = -3 → (a + 3)^2 + (2 + a) * (2 - a) = -5) ∧
  (∀ x : ℤ, x = -3 → 2 * x * (3 * x^2 - 4 * x + 1) - 3 * x^2 * (x - 3) = -78) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3990_399031


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_divisible_by_five_l3990_399017

theorem count_four_digit_numbers_divisible_by_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_divisible_by_five_l3990_399017


namespace NUMINAMATH_CALUDE_total_employee_purchase_price_l3990_399058

/-- Represents an item in the store -/
structure Item where
  name : String
  wholesale_cost : ℝ
  markup : ℝ
  employee_discount : ℝ

/-- Calculates the final price for an employee -/
def employee_price (item : Item) : ℝ :=
  item.wholesale_cost * (1 + item.markup) * (1 - item.employee_discount)

/-- The three items in the store -/
def video_recorder : Item :=
  { name := "Video Recorder", wholesale_cost := 200, markup := 0.20, employee_discount := 0.30 }

def digital_camera : Item :=
  { name := "Digital Camera", wholesale_cost := 150, markup := 0.25, employee_discount := 0.20 }

def smart_tv : Item :=
  { name := "Smart TV", wholesale_cost := 800, markup := 0.15, employee_discount := 0.25 }

/-- Theorem: The total amount paid by an employee for all three items is $1008 -/
theorem total_employee_purchase_price :
  employee_price video_recorder + employee_price digital_camera + employee_price smart_tv = 1008 := by
  sorry

end NUMINAMATH_CALUDE_total_employee_purchase_price_l3990_399058


namespace NUMINAMATH_CALUDE_budget_projection_l3990_399082

/-- Given the equation fp - w = 15000, where f = 7 and w = 70 + 210i, prove that p = 2153 + 30i -/
theorem budget_projection (f : ℝ) (w p : ℂ) 
  (eq : f * p - w = 15000)
  (hf : f = 7)
  (hw : w = 70 + 210 * Complex.I) : 
  p = 2153 + 30 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_budget_projection_l3990_399082


namespace NUMINAMATH_CALUDE_power_of_three_mod_eleven_l3990_399078

theorem power_of_three_mod_eleven : 3^1234 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eleven_l3990_399078


namespace NUMINAMATH_CALUDE_path_area_and_cost_calculation_l3990_399068

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem path_area_and_cost_calculation 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_unit : ℝ)
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.8)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 759.36 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1518.72 := by
  sorry

#eval path_area 75 55 2.8
#eval construction_cost (path_area 75 55 2.8) 2

end NUMINAMATH_CALUDE_path_area_and_cost_calculation_l3990_399068


namespace NUMINAMATH_CALUDE_middle_number_values_l3990_399089

/-- Represents a three-layer product pyramid --/
structure ProductPyramid where
  bottom_left : ℕ+
  bottom_middle : ℕ+
  bottom_right : ℕ+

/-- Calculates the top number of the pyramid --/
def top_number (p : ProductPyramid) : ℕ :=
  (p.bottom_left * p.bottom_middle) * (p.bottom_middle * p.bottom_right)

/-- Theorem stating the possible values for the middle number --/
theorem middle_number_values (p : ProductPyramid) :
  top_number p = 90 → p.bottom_middle = 1 ∨ p.bottom_middle = 3 := by
  sorry

#check middle_number_values

end NUMINAMATH_CALUDE_middle_number_values_l3990_399089


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3990_399019

theorem trigonometric_simplification (x : ℝ) :
  (2 * Real.cos x ^ 4 - 2 * Real.cos x ^ 2 + 1/2) / 
  (2 * Real.tan (π/4 - x) * Real.sin (π/4 + x) ^ 2) = 
  (1/2) * Real.cos (2*x) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3990_399019


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3990_399073

theorem min_value_sum_reciprocals (x y : ℝ) 
  (h1 : x^2 + y^2 = 2) 
  (h2 : |x| ≠ |y|) : 
  (∀ a b : ℝ, a^2 + b^2 = 2 → |a| ≠ |b| → 
    1 / (x + y)^2 + 1 / (x - y)^2 ≤ 1 / (a + b)^2 + 1 / (a - b)^2) ∧ 
  (∃ x y : ℝ, x^2 + y^2 = 2 ∧ |x| ≠ |y| ∧ 1 / (x + y)^2 + 1 / (x - y)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3990_399073


namespace NUMINAMATH_CALUDE_exists_n_sum_digits_greater_l3990_399011

/-- Sum of digits of a natural number in a given base -/
def sumOfDigits (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers m, there exists a natural number N such that
    for all natural numbers b where 2 ≤ b ≤ 1389, the sum of digits of N in base b
    is greater than m. -/
theorem exists_n_sum_digits_greater (m : ℕ) : 
  ∃ N : ℕ, ∀ b : ℕ, 2 ≤ b → b ≤ 1389 → sumOfDigits N b > m := by sorry

end NUMINAMATH_CALUDE_exists_n_sum_digits_greater_l3990_399011


namespace NUMINAMATH_CALUDE_fifth_dog_weight_l3990_399000

def dog_weights : List ℝ := [25, 31, 35, 33]

theorem fifth_dog_weight (w : ℝ) :
  (dog_weights.sum + w) / 5 = dog_weights.sum / 4 →
  w = 31 :=
by sorry

end NUMINAMATH_CALUDE_fifth_dog_weight_l3990_399000


namespace NUMINAMATH_CALUDE_derivative_parity_l3990_399043

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem derivative_parity (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (IsEven f → IsOdd f') ∧ (IsOdd f → IsEven f') := by sorry

end NUMINAMATH_CALUDE_derivative_parity_l3990_399043


namespace NUMINAMATH_CALUDE_xy_and_x3y_plus_x2_l3990_399044

theorem xy_and_x3y_plus_x2 (x y : ℝ) 
  (hx : x = 2 + Real.sqrt 3) 
  (hy : y = 2 - Real.sqrt 3) : 
  x * y = 1 ∧ x^3 * y + x^2 = 14 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_and_x3y_plus_x2_l3990_399044


namespace NUMINAMATH_CALUDE_conference_arrangement_count_l3990_399055

/-- Represents the number of teachers from each school -/
structure SchoolTeachers :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)

/-- Calculates the number of ways to arrange teachers from different schools -/
def arrangementCount (teachers : SchoolTeachers) : ℕ :=
  sorry

/-- The specific arrangement of teachers from the problem -/
def conferenceTeachers : SchoolTeachers :=
  { A := 2, B := 2, C := 1 }

/-- Theorem stating that the number of valid arrangements is 48 -/
theorem conference_arrangement_count :
  arrangementCount conferenceTeachers = 48 :=
sorry

end NUMINAMATH_CALUDE_conference_arrangement_count_l3990_399055


namespace NUMINAMATH_CALUDE_three_by_five_uncoverable_l3990_399035

/-- Represents a chessboard --/
structure Chessboard where
  rows : Nat
  cols : Nat

/-- Represents a domino --/
structure Domino where
  black : Unit
  white : Unit

/-- Defines a complete covering of a chessboard by dominoes --/
def CompleteCovering (board : Chessboard) (dominoes : List Domino) : Prop :=
  dominoes.length * 2 = board.rows * board.cols

/-- Theorem: A 3x5 chessboard cannot be completely covered by dominoes --/
theorem three_by_five_uncoverable :
  ¬ ∃ (dominoes : List Domino), CompleteCovering { rows := 3, cols := 5 } dominoes := by
  sorry

end NUMINAMATH_CALUDE_three_by_five_uncoverable_l3990_399035


namespace NUMINAMATH_CALUDE_minimum_total_balls_l3990_399022

/-- Given a set of balls with red, blue, and green colors, prove that there are at least 23 balls in total -/
theorem minimum_total_balls (red green blue : ℕ) : 
  green = 12 → red + green < 24 → red + green + blue ≥ 23 := by
  sorry

end NUMINAMATH_CALUDE_minimum_total_balls_l3990_399022


namespace NUMINAMATH_CALUDE_craft_store_sales_l3990_399084

theorem craft_store_sales (total_sales : ℕ) : 
  (total_sales / 3 : ℕ) + (total_sales / 4 : ℕ) + 15 = total_sales → 
  total_sales = 36 := by
  sorry

end NUMINAMATH_CALUDE_craft_store_sales_l3990_399084


namespace NUMINAMATH_CALUDE_preimage_of_4_3_l3990_399076

/-- The mapping f from R² to R² -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2*p.2, 2*p.1 - p.2)

/-- Theorem: The pre-image of (4,3) under the mapping f is (2,1) -/
theorem preimage_of_4_3 :
  f (2, 1) = (4, 3) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_4_3_l3990_399076


namespace NUMINAMATH_CALUDE_expression_simplification_l3990_399016

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x + a)^2 / ((a - b) * (a - c)) + (x + b)^2 / ((b - a) * (b - c)) + (x + c)^2 / ((c - a) * (c - b)) =
  a * x + b * x + c * x - a - b - c :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3990_399016


namespace NUMINAMATH_CALUDE_sector_area_l3990_399093

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 16) (h2 : central_angle = 2) : 
  let radius := perimeter / (2 + central_angle)
  (1/2) * central_angle * radius^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3990_399093


namespace NUMINAMATH_CALUDE_class_size_proof_l3990_399027

theorem class_size_proof :
  ∃ (n : ℕ), 
    n > 0 ∧
    (n / 2 : ℕ) > 0 ∧
    (n / 4 : ℕ) > 0 ∧
    (n / 7 : ℕ) > 0 ∧
    n - (n / 2) - (n / 4) - (n / 7) < 6 ∧
    n = 28 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l3990_399027


namespace NUMINAMATH_CALUDE_infinitely_many_unreachable_integers_l3990_399065

/-- Sum of digits in base b -/
def sum_of_digits (b : ℕ) (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem infinitely_many_unreachable_integers (b : ℕ) (h : b ≥ 2) :
  ∀ M : ℕ, ∃ S : Finset ℕ, (Finset.card S = M) ∧ 
  (∀ k ∈ S, ∀ n : ℕ, n + sum_of_digits b n ≠ k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_unreachable_integers_l3990_399065


namespace NUMINAMATH_CALUDE_locus_of_M_l3990_399069

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- The setup of the problem -/
structure Configuration :=
  (A B C : Point)
  (D : Point)
  (l : Line)
  (P Q N M L : Point)

/-- Condition that A, B, and C are collinear -/
def collinear (A B C : Point) : Prop := sorry

/-- Condition that a point is not on a line -/
def not_on_line (P : Point) (l : Line) : Prop := sorry

/-- Condition that two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Condition that a line passes through a point -/
def passes_through (l : Line) (P : Point) : Prop := sorry

/-- Condition that a point is the foot of the perpendicular from another point to a line -/
def is_foot_of_perpendicular (M C : Point) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem locus_of_M (config : Configuration) :
  collinear config.A config.B config.C →
  not_on_line config.D config.l →
  parallel (Line.mk 0 0 0) (Line.mk 0 0 0) →  -- CP parallel to AD
  parallel (Line.mk 0 0 0) (Line.mk 0 0 0) →  -- CQ parallel to BD
  is_foot_of_perpendicular config.M config.C (Line.mk 0 0 0) →  -- PQ line
  (config.C.x - config.N.x) / (config.A.x - config.N.x) = (config.C.x - config.B.x) / (config.A.x - config.C.x) →
  ∃ (l_M : Line),
    passes_through l_M config.L ∧
    parallel l_M (Line.mk 0 0 0) ∧  -- MN line
    ∀ (M : Point), passes_through l_M M ↔ 
      ∃ (D : Point), is_foot_of_perpendicular M config.C (Line.mk 0 0 0) :=
sorry

end NUMINAMATH_CALUDE_locus_of_M_l3990_399069


namespace NUMINAMATH_CALUDE_power_equation_solution_l3990_399074

theorem power_equation_solution (x y : ℕ) :
  (3 : ℝ) ^ x * (4 : ℝ) ^ y = 59049 ∧ x = 10 → x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3990_399074


namespace NUMINAMATH_CALUDE_horner_method_for_f_l3990_399095

def f (x : ℝ) : ℝ := x^6 + 2*x^5 + 4*x^3 + 5*x^2 + 6*x + 12

theorem horner_method_for_f :
  f 3 = 588 := by sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l3990_399095


namespace NUMINAMATH_CALUDE_valid_pairs_characterization_l3990_399097

/-- A function that checks if a given pair (m, n) of natural numbers
    satisfies the condition that 2^m * 3^n + 1 is a perfect square. -/
def is_valid_pair (m n : ℕ) : Prop :=
  ∃ x : ℕ, 2^m * 3^n + 1 = x^2

/-- The set of all valid pairs (m, n) that satisfy the condition. -/
def valid_pairs : Set (ℕ × ℕ) :=
  {p | is_valid_pair p.1 p.2}

/-- The theorem stating that the only valid pairs are (3, 1), (4, 1), and (5, 2). -/
theorem valid_pairs_characterization :
  valid_pairs = {(3, 1), (4, 1), (5, 2)} := by
  sorry


end NUMINAMATH_CALUDE_valid_pairs_characterization_l3990_399097


namespace NUMINAMATH_CALUDE_brendan_remaining_money_l3990_399020

/-- Brendan's remaining money calculation -/
theorem brendan_remaining_money 
  (june_earnings : ℕ) 
  (car_cost : ℕ) 
  (h1 : june_earnings = 5000)
  (h2 : car_cost = 1500) :
  (june_earnings / 2) - car_cost = 1000 :=
by sorry

end NUMINAMATH_CALUDE_brendan_remaining_money_l3990_399020


namespace NUMINAMATH_CALUDE_fifth_term_zero_l3990_399036

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fifth_term_zero
  (a : ℕ → ℚ)
  (x y : ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 0 = x + 2*y)
  (h_second : a 1 = x - 2*y)
  (h_third : a 2 = x^2 - 4*y^2)
  (h_fourth : a 3 = x / (2*y))
  (h_x : x = 1/2)
  (h_y : y = 1/4)
  : a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_zero_l3990_399036


namespace NUMINAMATH_CALUDE_theater_seats_l3990_399045

/-- The number of people watching the movie -/
def people_watching : ℕ := 532

/-- The number of empty seats -/
def empty_seats : ℕ := 218

/-- The total number of seats in the theater -/
def total_seats : ℕ := people_watching + empty_seats

theorem theater_seats : total_seats = 750 := by sorry

end NUMINAMATH_CALUDE_theater_seats_l3990_399045


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l3990_399085

-- Define the probability of having a boy or a girl
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_one_girl :
  1 - (prob_boy_or_girl ^ num_children + prob_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l3990_399085


namespace NUMINAMATH_CALUDE_chandler_bike_savings_l3990_399049

/-- The number of weeks Chandler needs to save to buy a mountain bike -/
def weeks_to_save (bike_cost : ℕ) (birthday_money : ℕ) (weekly_earnings : ℕ) : ℕ :=
  (bike_cost - birthday_money) / weekly_earnings

theorem chandler_bike_savings : 
  let bike_cost : ℕ := 600
  let grandparents_gift : ℕ := 60
  let aunt_gift : ℕ := 40
  let cousin_gift : ℕ := 20
  let weekly_earnings : ℕ := 20
  let total_birthday_money : ℕ := grandparents_gift + aunt_gift + cousin_gift
  weeks_to_save bike_cost total_birthday_money weekly_earnings = 24 := by
  sorry

#eval weeks_to_save 600 (60 + 40 + 20) 20

end NUMINAMATH_CALUDE_chandler_bike_savings_l3990_399049


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3990_399003

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3990_399003


namespace NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l3990_399030

/-- The percentage of chromium in the new alloy formed by mixing two alloys -/
theorem chromium_percentage_in_new_alloy 
  (chromium_percentage1 : Real) 
  (chromium_percentage2 : Real)
  (weight1 : Real) 
  (weight2 : Real) 
  (h1 : chromium_percentage1 = 12 / 100)
  (h2 : chromium_percentage2 = 8 / 100)
  (h3 : weight1 = 15)
  (h4 : weight2 = 40) : 
  (chromium_percentage1 * weight1 + chromium_percentage2 * weight2) / (weight1 + weight2) = 1 / 11 := by
sorry

#eval (1 / 11 : Float) * 100 -- To show the approximate percentage

end NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l3990_399030


namespace NUMINAMATH_CALUDE_no_real_solutions_cubic_equation_l3990_399091

theorem no_real_solutions_cubic_equation :
  ∀ x : ℝ, x^3 + 2*(x+1)^3 + 3*(x+2)^3 ≠ 6*(x+4)^3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_cubic_equation_l3990_399091


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3990_399023

theorem fourteenth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * π * n / 14)) := by
  sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3990_399023


namespace NUMINAMATH_CALUDE_percentage_of_female_dogs_l3990_399051

theorem percentage_of_female_dogs (total_dogs : ℕ) (birth_ratio : ℚ) (puppies_per_birth : ℕ) (total_puppies : ℕ) :
  total_dogs = 40 →
  birth_ratio = 3 / 4 →
  puppies_per_birth = 10 →
  total_puppies = 180 →
  (↑total_puppies : ℚ) = (birth_ratio * puppies_per_birth * (60 / 100 * total_dogs)) →
  60 = (100 * (total_puppies / (birth_ratio * puppies_per_birth * total_dogs))) := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_female_dogs_l3990_399051


namespace NUMINAMATH_CALUDE_game_result_l3990_399001

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 4 = 0 then 8
  else if n % 3 = 0 then 3
  else if n % 4 = 0 then 1
  else 0

def allie_rolls : List ℕ := [6, 3, 4, 1]
def betty_rolls : List ℕ := [12, 9, 4, 2]

def total_points (rolls : List ℕ) : ℕ :=
  (rolls.map g).sum

theorem game_result : 
  total_points allie_rolls * total_points betty_rolls = 84 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l3990_399001


namespace NUMINAMATH_CALUDE_aquarium_dolphins_l3990_399005

/-- The number of hours each dolphin requires for daily training -/
def training_hours_per_dolphin : ℕ := 3

/-- The number of trainers in the aquarium -/
def number_of_trainers : ℕ := 2

/-- The number of hours each trainer spends training dolphins -/
def hours_per_trainer : ℕ := 6

/-- The total number of training hours available -/
def total_training_hours : ℕ := number_of_trainers * hours_per_trainer

/-- The number of dolphins in the aquarium -/
def number_of_dolphins : ℕ := total_training_hours / training_hours_per_dolphin

theorem aquarium_dolphins :
  number_of_dolphins = 4 := by sorry

end NUMINAMATH_CALUDE_aquarium_dolphins_l3990_399005


namespace NUMINAMATH_CALUDE_triangle_area_l3990_399081

/-- Given a triangle with perimeter 32 cm and inradius 3.5 cm, its area is 56 cm² -/
theorem triangle_area (p : ℝ) (r : ℝ) (h1 : p = 32) (h2 : r = 3.5) :
  r * p / 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3990_399081


namespace NUMINAMATH_CALUDE_probability_second_genuine_given_first_genuine_l3990_399067

def total_items : ℕ := 10
def genuine_items : ℕ := 6
def defective_items : ℕ := 4

theorem probability_second_genuine_given_first_genuine :
  let first_genuine : ℝ := genuine_items / total_items
  let second_genuine : ℝ := (genuine_items - 1) / (total_items - 1)
  let both_genuine : ℝ := first_genuine * second_genuine
  both_genuine / first_genuine = 5 / 9 :=
by sorry

end NUMINAMATH_CALUDE_probability_second_genuine_given_first_genuine_l3990_399067


namespace NUMINAMATH_CALUDE_expression_equality_l3990_399059

theorem expression_equality : 
  (Real.sqrt (4/3) + Real.sqrt 3) * Real.sqrt 6 - (Real.sqrt 20 - Real.sqrt 5) / Real.sqrt 5 = 5 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3990_399059
