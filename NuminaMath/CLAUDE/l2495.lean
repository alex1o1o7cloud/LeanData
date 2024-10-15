import Mathlib

namespace NUMINAMATH_CALUDE_nursery_school_age_distribution_l2495_249590

theorem nursery_school_age_distribution (total : ℕ) (four_and_older : ℕ) (not_between_three_and_four : ℕ) :
  total = 50 →
  four_and_older = total / 10 →
  not_between_three_and_four = 25 →
  four_and_older + (total - four_and_older - (total - not_between_three_and_four)) = not_between_three_and_four →
  total - four_and_older - (total - not_between_three_and_four) = 20 := by
sorry

end NUMINAMATH_CALUDE_nursery_school_age_distribution_l2495_249590


namespace NUMINAMATH_CALUDE_sum_not_divisible_l2495_249597

theorem sum_not_divisible : ∃ (y : ℤ), 
  y = 42 + 98 + 210 + 333 + 175 + 28 ∧ 
  ¬(∃ (k : ℤ), y = 7 * k) ∧ 
  ¬(∃ (k : ℤ), y = 14 * k) ∧ 
  ¬(∃ (k : ℤ), y = 28 * k) ∧ 
  ¬(∃ (k : ℤ), y = 21 * k) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_divisible_l2495_249597


namespace NUMINAMATH_CALUDE_stating_min_gloves_for_matching_pair_l2495_249532

/-- Represents the number of different glove patterns -/
def num_patterns : ℕ := 4

/-- Represents the number of pairs for each pattern -/
def pairs_per_pattern : ℕ := 3

/-- Represents the total number of gloves in the wardrobe -/
def total_gloves : ℕ := num_patterns * pairs_per_pattern * 2

/-- 
Theorem stating the minimum number of gloves needed to ensure a matching pair
-/
theorem min_gloves_for_matching_pair : 
  ∃ (n : ℕ), n = num_patterns * pairs_per_pattern + 1 ∧ 
  (∀ (m : ℕ), m < n → ∃ (pattern : Fin num_patterns), 
    (m.choose 2 : ℕ) < pairs_per_pattern) ∧
  n ≤ total_gloves := by
  sorry

end NUMINAMATH_CALUDE_stating_min_gloves_for_matching_pair_l2495_249532


namespace NUMINAMATH_CALUDE_abhay_speed_l2495_249502

theorem abhay_speed (distance : ℝ) (a s : ℝ) : 
  distance = 18 →
  distance / a = distance / s + 2 →
  distance / (2 * a) = distance / s - 1 →
  a = 81 / 10 := by
  sorry

end NUMINAMATH_CALUDE_abhay_speed_l2495_249502


namespace NUMINAMATH_CALUDE_equation_solutions_l2495_249534

theorem equation_solutions :
  (∀ x : ℝ, 3 * (x - 1)^2 = 27 ↔ x = 4 ∨ x = -2) ∧
  (∀ x : ℝ, x^3 / 8 + 2 = 3 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2495_249534


namespace NUMINAMATH_CALUDE_decimal_51_to_binary_l2495_249574

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec to_binary_aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else to_binary_aux (m / 2) ((m % 2) :: acc)
    to_binary_aux n []

theorem decimal_51_to_binary :
  decimal_to_binary 51 = [1, 1, 0, 0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_51_to_binary_l2495_249574


namespace NUMINAMATH_CALUDE_solve_amusement_park_problem_l2495_249598

def amusement_park_problem (adult_price child_price total_tickets child_tickets : ℕ) : Prop :=
  adult_price = 8 ∧
  child_price = 5 ∧
  total_tickets = 33 ∧
  child_tickets = 21 ∧
  (total_tickets - child_tickets) * adult_price + child_tickets * child_price = 201

theorem solve_amusement_park_problem :
  ∃ (adult_price child_price total_tickets child_tickets : ℕ),
    amusement_park_problem adult_price child_price total_tickets child_tickets :=
by
  sorry

end NUMINAMATH_CALUDE_solve_amusement_park_problem_l2495_249598


namespace NUMINAMATH_CALUDE_angle_problem_l2495_249580

theorem angle_problem (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (h3 : angle1 = 80)
  (h4 : angle2 = 100) :
  angle4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l2495_249580


namespace NUMINAMATH_CALUDE_male_salmon_count_l2495_249563

theorem male_salmon_count (total : ℕ) (female : ℕ) (h1 : total = 971639) (h2 : female = 259378) :
  total - female = 712261 := by
  sorry

end NUMINAMATH_CALUDE_male_salmon_count_l2495_249563


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2495_249512

theorem sum_of_squares_and_square_of_sum : (3 + 7)^2 + (3^2 + 7^2) = 158 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2495_249512


namespace NUMINAMATH_CALUDE_jeff_probability_multiple_of_four_l2495_249507

/-- The number of cards --/
def num_cards : ℕ := 12

/-- The probability of moving left on a single spin --/
def prob_left : ℚ := 1/2

/-- The probability of moving right on a single spin --/
def prob_right : ℚ := 1/2

/-- The number of spaces moved left --/
def spaces_left : ℕ := 1

/-- The number of spaces moved right --/
def spaces_right : ℕ := 2

/-- The probability of ending up at a multiple of 4 --/
def prob_multiple_of_four : ℚ := 5/32

theorem jeff_probability_multiple_of_four :
  let start_at_multiple_of_four := (num_cards / 4 : ℚ) / num_cards
  let start_two_more_than_multiple_of_four := (num_cards / 4 : ℚ) / num_cards
  let start_two_less_than_multiple_of_four := (num_cards / 4 : ℚ) / num_cards
  let end_at_multiple_of_four_from_multiple_of_four := prob_left * prob_right + prob_right * prob_left
  let end_at_multiple_of_four_from_two_more := prob_right * prob_right
  let end_at_multiple_of_four_from_two_less := prob_left * prob_left
  start_at_multiple_of_four * end_at_multiple_of_four_from_multiple_of_four +
  start_two_more_than_multiple_of_four * end_at_multiple_of_four_from_two_more +
  start_two_less_than_multiple_of_four * end_at_multiple_of_four_from_two_less =
  prob_multiple_of_four := by
  sorry

end NUMINAMATH_CALUDE_jeff_probability_multiple_of_four_l2495_249507


namespace NUMINAMATH_CALUDE_total_pizza_slices_l2495_249578

theorem total_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 36) 
  (h2 : slices_per_pizza = 12) : 
  num_pizzas * slices_per_pizza = 432 :=
by sorry

end NUMINAMATH_CALUDE_total_pizza_slices_l2495_249578


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2495_249570

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2495_249570


namespace NUMINAMATH_CALUDE_rachel_songs_total_l2495_249569

theorem rachel_songs_total (albums : ℕ) (songs_per_album : ℕ) (h1 : albums = 8) (h2 : songs_per_album = 2) :
  albums * songs_per_album = 16 := by
  sorry

end NUMINAMATH_CALUDE_rachel_songs_total_l2495_249569


namespace NUMINAMATH_CALUDE_circle_equation_l2495_249587

theorem circle_equation 
  (a b : ℝ) 
  (h1 : a > 0 ∧ b > 0)  -- Center in first quadrant
  (h2 : 2 * a - b + 1 = 0)  -- Center on the line 2x - y + 1 = 0
  (h3 : (a + 4)^2 + (b - 3)^2 = 5^2)  -- Passes through (-4, 3) with radius 5
  : ∀ x y : ℝ, (x - 1)^2 + (y - 3)^2 = 25 ↔ (x - a)^2 + (y - b)^2 = 5^2 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l2495_249587


namespace NUMINAMATH_CALUDE_chef_leftover_potatoes_l2495_249514

/-- Given a chef's potato and fry situation, calculate the number of leftover potatoes. -/
def leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ) : ℕ :=
  total_potatoes - (required_fries / fries_per_potato)

/-- Prove that the chef will have 7 potatoes leftover. -/
theorem chef_leftover_potatoes :
  leftover_potatoes 25 15 200 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chef_leftover_potatoes_l2495_249514


namespace NUMINAMATH_CALUDE_allison_craft_items_l2495_249541

/-- Represents the number of craft items bought by a person -/
structure CraftItems where
  glueSticks : ℕ
  constructionPaper : ℕ

/-- Calculates the total number of craft items -/
def totalItems (items : CraftItems) : ℕ :=
  items.glueSticks + items.constructionPaper

theorem allison_craft_items (marie : CraftItems) 
    (marie_glue : marie.glueSticks = 15)
    (marie_paper : marie.constructionPaper = 30)
    (allison : CraftItems)
    (glue_diff : allison.glueSticks = marie.glueSticks + 8)
    (paper_ratio : marie.constructionPaper = 6 * allison.constructionPaper) :
    totalItems allison = 28 := by
  sorry

end NUMINAMATH_CALUDE_allison_craft_items_l2495_249541


namespace NUMINAMATH_CALUDE_hexagon_count_l2495_249573

/-- Represents a regular hexagon divided into smaller equilateral triangles -/
structure DividedHexagon where
  side_length : ℕ
  num_small_triangles : ℕ
  small_triangle_side : ℕ

/-- Counts the number of regular hexagons that can be formed in a divided hexagon -/
def count_hexagons (h : DividedHexagon) : ℕ :=
  sorry

/-- Theorem stating the number of hexagons in the specific configuration -/
theorem hexagon_count (h : DividedHexagon) 
  (h_side : h.side_length = 3)
  (h_triangles : h.num_small_triangles = 54)
  (h_small_side : h.small_triangle_side = 1) :
  count_hexagons h = 36 :=
sorry

end NUMINAMATH_CALUDE_hexagon_count_l2495_249573


namespace NUMINAMATH_CALUDE_time_at_6_oclock_l2495_249576

/-- Represents a clock with ticks at each hour -/
structure Clock where
  /-- The time between each tick (in seconds) -/
  tick_interval : ℝ
  /-- The total time for all ticks at 12 o'clock (in seconds) -/
  total_time_at_12 : ℝ

/-- Calculates the time between first and last ticks for a given hour -/
def time_between_ticks (c : Clock) (hour : ℕ) : ℝ :=
  c.tick_interval * (hour - 1)

/-- Theorem stating the time between first and last ticks at 6 o'clock -/
theorem time_at_6_oclock (c : Clock) 
  (h1 : c.total_time_at_12 = 66)
  (h2 : c.tick_interval = c.total_time_at_12 / 11) :
  time_between_ticks c 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_time_at_6_oclock_l2495_249576


namespace NUMINAMATH_CALUDE_smallest_b_value_b_equals_one_l2495_249538

def gcd_notation (a b : ℕ) : ℕ := Nat.gcd a b

theorem smallest_b_value (b : ℕ) : b > 0 → (gcd_notation (gcd_notation 16 20) (gcd_notation 18 b) = 2) → b ≥ 1 :=
by
  sorry

theorem b_equals_one : ∃ (b : ℕ), b > 0 ∧ (gcd_notation (gcd_notation 16 20) (gcd_notation 18 b) = 2) ∧ 
  ∀ (c : ℕ), c > 0 → (gcd_notation (gcd_notation 16 20) (gcd_notation 18 c) = 2) → b ≤ c :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_b_equals_one_l2495_249538


namespace NUMINAMATH_CALUDE_road_length_probability_l2495_249568

/-- The probability of a road from A to B being at least 5 miles long -/
def prob_ab : ℚ := 2/3

/-- The probability of a road from B to C being at least 5 miles long -/
def prob_bc : ℚ := 3/4

/-- The probability that at least one of two randomly picked roads
    (one from A to B, one from B to C) is at least 5 miles long -/
def prob_at_least_one : ℚ := 1 - (1 - prob_ab) * (1 - prob_bc)

theorem road_length_probability : prob_at_least_one = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_road_length_probability_l2495_249568


namespace NUMINAMATH_CALUDE_gcd_problem_l2495_249539

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 1632 * k) : 
  Int.gcd (a^2 + 13*a + 36) (a + 6) = 6 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l2495_249539


namespace NUMINAMATH_CALUDE_sum_of_four_integers_with_product_5_4_l2495_249530

theorem sum_of_four_integers_with_product_5_4 (a b c d : ℕ+) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 5^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) = 156 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_with_product_5_4_l2495_249530


namespace NUMINAMATH_CALUDE_terms_are_not_like_l2495_249519

/-- Two algebraic terms are considered like terms if they have the same variables raised to the same powers. -/
def are_like_terms (term1 term2 : Type) : Prop := sorry

/-- The first term in the problem -/
def term1 : Type := sorry

/-- The second term in the problem -/
def term2 : Type := sorry

/-- Theorem stating that the two terms are not like terms -/
theorem terms_are_not_like : ¬(are_like_terms term1 term2) := by sorry

end NUMINAMATH_CALUDE_terms_are_not_like_l2495_249519


namespace NUMINAMATH_CALUDE_symmetric_complex_sum_third_quadrant_l2495_249546

/-- Given two complex numbers symmetric with respect to the imaginary axis,
    prove that their sum with one divided by its modulus squared is in the third quadrant -/
theorem symmetric_complex_sum_third_quadrant (z₁ z : ℂ) : 
  z₁ = 2 - I →
  z = -Complex.re z₁ + Complex.im z₁ * I → 
  let w := z₁ / Complex.normSq z₁ + z
  Complex.re w < 0 ∧ Complex.im w < 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_sum_third_quadrant_l2495_249546


namespace NUMINAMATH_CALUDE_smallest_overlap_percentage_l2495_249564

theorem smallest_overlap_percentage (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 60)
  (h2 : tea_drinkers = 90) :
  coffee_drinkers + tea_drinkers - 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_overlap_percentage_l2495_249564


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l2495_249571

/-- 
Given a two-digit number n = 10a + b, where a and b are single digits,
if 1000a + 100b = 37(100a + 10b + 1), then n = 27.
-/
theorem two_digit_number_problem (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : 1000 * a + 100 * b = 37 * (100 * a + 10 * b + 1)) :
  10 * a + b = 27 := by
  sorry

#check two_digit_number_problem

end NUMINAMATH_CALUDE_two_digit_number_problem_l2495_249571


namespace NUMINAMATH_CALUDE_prime_power_sum_l2495_249560

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 588 → 2*w + 3*x + 5*y + 7*z = 21 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l2495_249560


namespace NUMINAMATH_CALUDE_determinant_property_l2495_249591

theorem determinant_property (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 4 →
  Matrix.det ![![a + 2*c, b + 2*d], ![c, d]] = 4 := by
  sorry

end NUMINAMATH_CALUDE_determinant_property_l2495_249591


namespace NUMINAMATH_CALUDE_ball_max_height_l2495_249503

/-- The height of the ball as a function of time -/
def f (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 45 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l2495_249503


namespace NUMINAMATH_CALUDE_triangle_theorem_l2495_249535

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = Real.sqrt 3 * t.a * t.c)
  (h2 : 2 * t.b * Real.cos t.A = Real.sqrt 3 * (t.c * Real.cos t.A + t.a * Real.cos t.C))
  (h3 : (t.a^2 + t.b^2 + t.c^2 - (t.b^2 + t.c^2 - t.a^2) / 2) / 4 = 7) :
  t.B = π / 6 ∧ t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2495_249535


namespace NUMINAMATH_CALUDE_distance_covered_proof_l2495_249583

/-- Calculates the distance covered given a fuel-to-distance ratio and fuel consumption -/
def distance_covered (fuel_ratio : ℚ) (distance_ratio : ℚ) (fuel_consumed : ℚ) : ℚ :=
  (distance_ratio / fuel_ratio) * fuel_consumed

/-- Proves that given a fuel-to-distance ratio of 4:7 and fuel consumption of 44 gallons, 
    the distance covered is 77 miles -/
theorem distance_covered_proof :
  let fuel_ratio : ℚ := 4
  let distance_ratio : ℚ := 7
  let fuel_consumed : ℚ := 44
  distance_covered fuel_ratio distance_ratio fuel_consumed = 77 := by
sorry

end NUMINAMATH_CALUDE_distance_covered_proof_l2495_249583


namespace NUMINAMATH_CALUDE_largest_four_digit_product_l2495_249572

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_four_digit_product (m x y z : ℕ) : 
  m > 0 →
  m = x * y * (10 * x + y) * z →
  is_prime x →
  is_prime y →
  is_prime (10 * x + y) →
  is_prime z →
  x < 20 →
  y < 20 →
  z < 20 →
  x ≠ y →
  x ≠ 10 * x + y →
  y ≠ 10 * x + y →
  x ≠ z →
  y ≠ z →
  (10 * x + y) ≠ z →
  1000 ≤ m →
  m < 10000 →
  m ≤ 7478 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_product_l2495_249572


namespace NUMINAMATH_CALUDE_polynomial_inequality_l2495_249525

theorem polynomial_inequality (x : ℝ) : 
  x^6 + 4*x^5 + 2*x^4 - 6*x^3 - 2*x^2 + 4*x - 1 ≥ 0 ↔ 
  x ≤ -1 - Real.sqrt 2 ∨ x = (-1 - Real.sqrt 5) / 2 ∨ x ≥ -1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l2495_249525


namespace NUMINAMATH_CALUDE_students_not_reading_l2495_249593

theorem students_not_reading (total : ℕ) (three_or_more : ℚ) (two : ℚ) (one : ℚ) :
  total = 240 →
  three_or_more = 1 / 6 →
  two = 35 / 100 →
  one = 5 / 12 →
  ↑total - (↑total * (three_or_more + two + one)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_not_reading_l2495_249593


namespace NUMINAMATH_CALUDE_p_true_and_q_false_l2495_249566

-- Define proposition P
def P : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 1/x₀ > 3

-- Define proposition q
def q : Prop := ∀ x : ℝ, x > 2 → x^2 > 2^x

-- Theorem statement
theorem p_true_and_q_false : P ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_l2495_249566


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l2495_249521

open Real

theorem parallel_vectors_tan_theta (θ : ℝ) : 
  let a : Fin 2 → ℝ := ![2, sin θ]
  let b : Fin 2 → ℝ := ![1, cos θ]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → tan θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l2495_249521


namespace NUMINAMATH_CALUDE_circle_radius_l2495_249504

theorem circle_radius (A C : ℝ) (h : A / C = 25) : 
  ∃ r : ℝ, r > 0 ∧ A = π * r^2 ∧ C = 2 * π * r ∧ r = 50 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2495_249504


namespace NUMINAMATH_CALUDE_pyramid_has_one_base_l2495_249588

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a point (apex) --/
structure Pyramid where
  base : Set Point
  apex : Point
  faces : Set (Set Point)

/-- Any pyramid has only one base --/
theorem pyramid_has_one_base (p : Pyramid) : ∃! b : Set Point, b = p.base := by
  sorry

end NUMINAMATH_CALUDE_pyramid_has_one_base_l2495_249588


namespace NUMINAMATH_CALUDE_domain_of_f_l2495_249526

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 2*x + 1)^(1/3) + (9 - x^2)^(1/3)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l2495_249526


namespace NUMINAMATH_CALUDE_range_of_a_l2495_249529

-- Define the propositions p and q
def p (x : ℝ) : Prop := |4*x - 3| ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the theorem
theorem range_of_a :
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(q x a) ∧ p x) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2495_249529


namespace NUMINAMATH_CALUDE_average_percentage_increase_l2495_249515

/-- Given an item with original price of 100 yuan, increased first by 40% and then by 10%,
    prove that the average percentage increase x per time satisfies (1 + 40%)(1 + 10%) = (1 + x)² -/
theorem average_percentage_increase (original_price : ℝ) (first_increase second_increase : ℝ) 
  (x : ℝ) (h1 : original_price = 100) (h2 : first_increase = 0.4) (h3 : second_increase = 0.1) :
  (1 + first_increase) * (1 + second_increase) = (1 + x)^2 := by
  sorry

end NUMINAMATH_CALUDE_average_percentage_increase_l2495_249515


namespace NUMINAMATH_CALUDE_max_blocks_fit_l2495_249544

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the problem of fitting smaller blocks into a larger box -/
structure BlockFittingProblem where
  largeBox : BoxDimensions
  smallBlock : BoxDimensions

/-- Calculates the maximum number of blocks that can fit based on volume -/
def maxBlocksByVolume (p : BlockFittingProblem) : ℕ :=
  (boxVolume p.largeBox) / (boxVolume p.smallBlock)

/-- Calculates the maximum number of blocks that can fit based on physical arrangement -/
def maxBlocksByArrangement (p : BlockFittingProblem) : ℕ :=
  (p.largeBox.length / p.smallBlock.length) *
  (p.largeBox.width / p.smallBlock.width) *
  (p.largeBox.height / p.smallBlock.height)

/-- The main theorem stating that the maximum number of blocks that can fit is 6 -/
theorem max_blocks_fit (p : BlockFittingProblem) 
    (h1 : p.largeBox = ⟨4, 3, 2⟩) 
    (h2 : p.smallBlock = ⟨3, 1, 1⟩) : 
    min (maxBlocksByVolume p) (maxBlocksByArrangement p) = 6 := by
  sorry


end NUMINAMATH_CALUDE_max_blocks_fit_l2495_249544


namespace NUMINAMATH_CALUDE_smallest_constant_for_sum_squares_inequality_l2495_249551

theorem smallest_constant_for_sum_squares_inequality :
  ∃ k : ℝ, k > 0 ∧
  (∀ y₁ y₂ y₃ A : ℝ,
    y₁ + y₂ + y₃ = 0 →
    A = max (abs y₁) (max (abs y₂) (abs y₃)) →
    y₁^2 + y₂^2 + y₃^2 ≥ k * A^2) ∧
  (∀ k' : ℝ, k' < k →
    ∃ y₁ y₂ y₃ A : ℝ,
      y₁ + y₂ + y₃ = 0 ∧
      A = max (abs y₁) (max (abs y₂) (abs y₃)) ∧
      y₁^2 + y₂^2 + y₃^2 < k' * A^2) ∧
  k = 1.5 := by
sorry

end NUMINAMATH_CALUDE_smallest_constant_for_sum_squares_inequality_l2495_249551


namespace NUMINAMATH_CALUDE_swap_values_l2495_249554

theorem swap_values (a b : ℕ) : 
  let c := b
  let b' := a
  let a' := c
  (a' = b ∧ b' = a) :=
by
  sorry

end NUMINAMATH_CALUDE_swap_values_l2495_249554


namespace NUMINAMATH_CALUDE_sine_integral_negative_l2495_249518

theorem sine_integral_negative : ∫ x in -Real.pi..0, Real.sin x < 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_integral_negative_l2495_249518


namespace NUMINAMATH_CALUDE_complex_expressions_equality_l2495_249561

theorem complex_expressions_equality : 
  let z₁ : ℂ := (-2 * Real.sqrt 3 * I + 1) / (1 + 2 * Real.sqrt 3 * I) + ((Real.sqrt 2) / (1 + I)) ^ 2000 + (1 + I) / (3 - I)
  let z₂ : ℂ := (5 * (4 + I)^2) / (I * (2 + I)) + 2 / (1 - I)^2
  z₁ = 6/65 + (39/65) * I ∧ z₂ = -1 + 39 * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expressions_equality_l2495_249561


namespace NUMINAMATH_CALUDE_all_sides_equal_not_imply_rectangle_l2495_249531

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : ℝ)  -- Side lengths
  (α β γ δ : ℝ)  -- Internal angles

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  q.α = q.β ∧ q.β = q.γ ∧ q.γ = q.δ ∧ q.δ = 90

-- Define a quadrilateral with all sides equal
def all_sides_equal (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

-- Theorem statement
theorem all_sides_equal_not_imply_rectangle :
  ∃ q : Quadrilateral, all_sides_equal q ∧ ¬(is_rectangle q) := by
  sorry


end NUMINAMATH_CALUDE_all_sides_equal_not_imply_rectangle_l2495_249531


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l2495_249528

/-- Calculates the repair cost of a machine given its purchase price, transportation charges, profit percentage, and selling price. -/
theorem repair_cost_calculation (purchase_price : ℕ) (transportation_charges : ℕ) (profit_percentage : ℕ) (selling_price : ℕ) : 
  purchase_price = 11000 →
  transportation_charges = 1000 →
  profit_percentage = 50 →
  selling_price = 25500 →
  ∃ (repair_cost : ℕ), 
    repair_cost = 5000 ∧
    selling_price = (purchase_price + repair_cost + transportation_charges) * (100 + profit_percentage) / 100 :=
by sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l2495_249528


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2495_249559

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt (a - 2) - Real.sqrt (a - 3) > Real.sqrt a - Real.sqrt (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2495_249559


namespace NUMINAMATH_CALUDE_nickel_difference_is_zero_l2495_249522

/-- Represents the number of coins of each type -/
structure CoinCollection where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins -/
def total_coins : ℕ := 150

/-- The total value of the coins in cents -/
def total_value : ℕ := 2000

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Checks if a coin collection is valid -/
def is_valid_collection (c : CoinCollection) : Prop :=
  c.nickels + c.dimes + c.quarters = total_coins ∧
  c.nickels * nickel_value + c.dimes * dime_value + c.quarters * quarter_value = total_value ∧
  c.nickels > 0 ∧ c.dimes > 0 ∧ c.quarters > 0

/-- The theorem to be proved -/
theorem nickel_difference_is_zero :
  ∃ (min_nickels max_nickels : ℕ),
    (∀ c : CoinCollection, is_valid_collection c → c.nickels ≥ min_nickels) ∧
    (∀ c : CoinCollection, is_valid_collection c → c.nickels ≤ max_nickels) ∧
    max_nickels - min_nickels = 0 := by
  sorry

end NUMINAMATH_CALUDE_nickel_difference_is_zero_l2495_249522


namespace NUMINAMATH_CALUDE_one_fourth_divided_by_one_eighth_l2495_249584

theorem one_fourth_divided_by_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_divided_by_one_eighth_l2495_249584


namespace NUMINAMATH_CALUDE_friends_recycled_sixteen_pounds_l2495_249533

/-- Represents the recycling scenario -/
structure RecyclingScenario where
  pounds_per_point : ℕ
  vanessa_pounds : ℕ
  total_points : ℕ

/-- Calculates the amount of paper recycled by Vanessa's friends -/
def friends_recycled_pounds (scenario : RecyclingScenario) : ℕ :=
  scenario.total_points * scenario.pounds_per_point - scenario.vanessa_pounds

/-- Theorem stating that Vanessa's friends recycled 16 pounds -/
theorem friends_recycled_sixteen_pounds :
  ∃ (scenario : RecyclingScenario),
    scenario.pounds_per_point = 9 ∧
    scenario.vanessa_pounds = 20 ∧
    scenario.total_points = 4 ∧
    friends_recycled_pounds scenario = 16 := by
  sorry


end NUMINAMATH_CALUDE_friends_recycled_sixteen_pounds_l2495_249533


namespace NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_l2495_249586

theorem sin_cos_sum_fifteen_seventyfive : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (75 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_l2495_249586


namespace NUMINAMATH_CALUDE_number_equality_l2495_249513

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (25/216) * (1/x)) : x = 144/25 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2495_249513


namespace NUMINAMATH_CALUDE_min_value_theorem_range_theorem_l2495_249575

-- Define the variables and conditions
variable (a b : ℝ) (hsum : a + b = 1) (ha : a > 0) (hb : b > 0)

-- Part I: Minimum value theorem
theorem min_value_theorem : 
  ∃ (min : ℝ), min = 9 ∧ ∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → 1/a + 4/b ≥ min :=
sorry

-- Part II: Range theorem
theorem range_theorem :
  ∀ (x : ℝ), (∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → 1/a + 4/b ≥ |2*x - 1| - |x + 1|) ↔ x ∈ Set.Icc (-7) 11 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_range_theorem_l2495_249575


namespace NUMINAMATH_CALUDE_infinite_image_is_infinite_l2495_249585

-- Define the concept of an infinite set
def IsInfinite (α : Type*) : Prop := ∃ f : α → α, Function.Injective f ∧ ¬Function.Surjective f

-- State the theorem
theorem infinite_image_is_infinite {A B : Type*} (f : A → B) (h : IsInfinite A) : IsInfinite B := by
  sorry

end NUMINAMATH_CALUDE_infinite_image_is_infinite_l2495_249585


namespace NUMINAMATH_CALUDE_jason_age_2004_l2495_249557

/-- Jason's age at the end of 1997 -/
def jason_age_1997 : ℝ := 35.5

/-- Jason's grandmother's age at the end of 1997 -/
def grandmother_age_1997 : ℝ := 3 * jason_age_1997

/-- The sum of Jason's and his grandmother's birth years -/
def birth_years_sum : ℕ := 3852

/-- The year we're considering for Jason's age -/
def target_year : ℕ := 2004

/-- The reference year for ages -/
def reference_year : ℕ := 1997

theorem jason_age_2004 :
  jason_age_1997 + (target_year - reference_year) = 42.5 ∧
  jason_age_1997 = grandmother_age_1997 / 3 ∧
  (reference_year - jason_age_1997) + (reference_year - grandmother_age_1997) = birth_years_sum :=
by sorry

end NUMINAMATH_CALUDE_jason_age_2004_l2495_249557


namespace NUMINAMATH_CALUDE_arrangement_count_eq_960_l2495_249556

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a row,
    where the elderly people must be adjacent but not at the ends. -/
def arrangement_count : ℕ :=
  let n_volunteers : ℕ := 5
  let n_elderly : ℕ := 2
  let n_total : ℕ := n_volunteers + n_elderly
  let n_ends : ℕ := 2
  let n_remaining_volunteers : ℕ := n_volunteers - n_ends
  let elderly_group : ℕ := 1  -- Treat adjacent elderly as one group

  (n_volunteers.choose n_ends) *    -- Ways to choose volunteers for the ends
  ((n_remaining_volunteers + elderly_group).factorial) *  -- Ways to arrange middle positions
  (n_elderly.factorial)             -- Ways to arrange within elderly group

theorem arrangement_count_eq_960 : arrangement_count = 960 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_eq_960_l2495_249556


namespace NUMINAMATH_CALUDE_sandwich_problem_solution_l2495_249536

/-- Represents the sandwich making problem --/
def sandwich_problem (bread_packages : ℕ) (bread_slices_per_package : ℕ) 
  (ham_packages : ℕ) (ham_slices_per_package : ℕ)
  (turkey_packages : ℕ) (turkey_slices_per_package : ℕ)
  (roast_beef_packages : ℕ) (roast_beef_slices_per_package : ℕ)
  (ham_proportion : ℚ) (turkey_proportion : ℚ) (roast_beef_proportion : ℚ) : Prop :=
  let total_bread := bread_packages * bread_slices_per_package
  let total_ham := ham_packages * ham_slices_per_package
  let total_turkey := turkey_packages * turkey_slices_per_package
  let total_roast_beef := roast_beef_packages * roast_beef_slices_per_package
  let total_sandwiches := min (total_ham / ham_proportion) 
                              (min (total_turkey / turkey_proportion) 
                                   (total_roast_beef / roast_beef_proportion))
  let bread_used := 2 * total_sandwiches
  let leftover_bread := total_bread - bread_used
  leftover_bread = 16

/-- The sandwich problem theorem --/
theorem sandwich_problem_solution : 
  sandwich_problem 4 24 3 14 2 18 1 10 (2/5) (7/20) (1/4) := by
  sorry

end NUMINAMATH_CALUDE_sandwich_problem_solution_l2495_249536


namespace NUMINAMATH_CALUDE_bob_always_has_valid_move_l2495_249508

-- Define the game board
def GameBoard (n : ℕ) := ℤ × ℤ

-- Define the possible moves for Bob and Alice
def BobMove (p : ℤ × ℤ) : Set (ℤ × ℤ) :=
  {(p.1 + 2, p.2 + 1), (p.1 + 2, p.2 - 1), (p.1 - 2, p.2 + 1), (p.1 - 2, p.2 - 1)}

def AliceMove (p : ℤ × ℤ) : Set (ℤ × ℤ) :=
  {(p.1 + 1, p.2 + 2), (p.1 + 1, p.2 - 2), (p.1 - 1, p.2 + 2), (p.1 - 1, p.2 - 2)}

-- Define the modulo condition
def ModuloCondition (n : ℕ) (a b c d : ℤ) : Prop :=
  c % n = a % n ∧ d % n = b % n

-- Define a valid move
def ValidMove (n : ℕ) (occupied : Set (ℤ × ℤ)) (p : ℤ × ℤ) : Prop :=
  ∀ (a b : ℤ), (a, b) ∈ occupied → ¬(ModuloCondition n a b p.1 p.2)

-- Theorem: Bob always has a valid move
theorem bob_always_has_valid_move (n : ℕ) (h : n = 2018 ∨ n = 2019) 
  (occupied : Set (ℤ × ℤ)) (last_move : ℤ × ℤ) :
  ∃ (next_move : ℤ × ℤ), next_move ∈ BobMove last_move ∧ ValidMove n occupied next_move :=
sorry

end NUMINAMATH_CALUDE_bob_always_has_valid_move_l2495_249508


namespace NUMINAMATH_CALUDE_jasons_cousins_l2495_249517

theorem jasons_cousins (cupcakes_bought : ℕ) (cupcakes_per_cousin : ℕ) : 
  cupcakes_bought = 4 * 12 → cupcakes_per_cousin = 3 → 
  cupcakes_bought / cupcakes_per_cousin = 16 := by
  sorry

end NUMINAMATH_CALUDE_jasons_cousins_l2495_249517


namespace NUMINAMATH_CALUDE_product_inequality_l2495_249548

theorem product_inequality : 
  (190 * 80 = 19 * 800) → 
  (190 * 80 = 19 * 8 * 100) → 
  (19 * 8 * 10 ≠ 190 * 80) := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l2495_249548


namespace NUMINAMATH_CALUDE_chocolate_doughnut_cost_l2495_249550

/-- The cost of a chocolate doughnut given the number of students wanting each type,
    the cost of glazed doughnuts, and the total cost. -/
theorem chocolate_doughnut_cost
  (chocolate_students : ℕ)
  (glazed_students : ℕ)
  (glazed_cost : ℚ)
  (total_cost : ℚ)
  (h1 : chocolate_students = 10)
  (h2 : glazed_students = 15)
  (h3 : glazed_cost = 1)
  (h4 : total_cost = 35) :
  ∃ (chocolate_cost : ℚ),
    chocolate_cost * chocolate_students + glazed_cost * glazed_students = total_cost ∧
    chocolate_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_doughnut_cost_l2495_249550


namespace NUMINAMATH_CALUDE_disputed_food_weight_l2495_249543

/-- 
Given a piece of food disputed by a dog and a cat:
- x is the total weight of the piece
- d is the difference in the amount the dog wants to take compared to the cat
- The cat takes (x - d) grams
- The dog takes (x + d) grams
- We know that (x - d) = 300 and (x + d) = 500

This theorem proves that the total weight of the disputed piece is 400 grams.
-/
theorem disputed_food_weight (x d : ℝ) 
  (h1 : x - d = 300) 
  (h2 : x + d = 500) : 
  x = 400 := by
sorry


end NUMINAMATH_CALUDE_disputed_food_weight_l2495_249543


namespace NUMINAMATH_CALUDE_expression_simplification_l2495_249510

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := ((a * (b^(1/3))) / (b * (a^3)^(1/2)))^(3/2) + ((a^(1/2)) / (a * (b^3)^(1/8)))^2
  x / (a^(1/4) + b^(1/4)) = 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2495_249510


namespace NUMINAMATH_CALUDE_nested_radical_sixteen_l2495_249558

theorem nested_radical_sixteen (x : ℝ) : x = Real.sqrt (16 + x) → x = (1 + Real.sqrt 65) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_sixteen_l2495_249558


namespace NUMINAMATH_CALUDE_subset_of_A_l2495_249506

def A : Set ℝ := {x | x > -1}

theorem subset_of_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_subset_of_A_l2495_249506


namespace NUMINAMATH_CALUDE_min_benches_for_equal_seating_l2495_249542

/-- Represents the seating capacity of a bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Defines the standard bench capacity -/
def standard_bench : BenchCapacity := ⟨8, 12⟩

/-- Defines the extended bench capacity -/
def extended_bench : BenchCapacity := ⟨8, 16⟩

/-- Theorem stating the minimum number of benches required -/
theorem min_benches_for_equal_seating :
  ∃ (n : Nat), n > 0 ∧
    n * standard_bench.adults + n * extended_bench.adults =
    n * standard_bench.children + n * extended_bench.children ∧
    ∀ (m : Nat), m > 0 →
      m * standard_bench.adults + m * extended_bench.adults =
      m * standard_bench.children + m * extended_bench.children →
      m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_benches_for_equal_seating_l2495_249542


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l2495_249592

/-- The surface area of a sphere that circumscribes a cube with edge length 1 is 3π. -/
theorem sphere_surface_area_circumscribing_unit_cube (π : ℝ) : 
  (∃ (S : ℝ), S = 3 * π ∧ 
    S = 4 * π * (((1 : ℝ)^2 + (1 : ℝ)^2 + (1 : ℝ)^2).sqrt / 2)^2) :=
by sorry


end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l2495_249592


namespace NUMINAMATH_CALUDE_grand_hall_expenditure_l2495_249516

/-- Calculates the total expenditure for covering a rectangular floor with a mat -/
def total_expenditure (length width cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Proves that the total expenditure for covering a 50m × 30m floor with a mat 
    costing Rs. 100 per square meter is Rs. 150,000 -/
theorem grand_hall_expenditure :
  total_expenditure 50 30 100 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_grand_hall_expenditure_l2495_249516


namespace NUMINAMATH_CALUDE_end_time_calculation_l2495_249567

-- Define the structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define the problem parameters
def glowInterval : Nat := 17
def startTime : Time := { hours := 1, minutes := 57, seconds := 58 }
def glowCount : Float := 292.29411764705884

-- Define the function to calculate the ending time
def calculateEndTime (start : Time) (interval : Nat) (count : Float) : Time :=
  sorry

-- Theorem statement
theorem end_time_calculation :
  calculateEndTime startTime glowInterval glowCount = { hours := 3, minutes := 20, seconds := 42 } :=
sorry

end NUMINAMATH_CALUDE_end_time_calculation_l2495_249567


namespace NUMINAMATH_CALUDE_complex_cube_equation_l2495_249582

def complex (x y : ℤ) := x + y * Complex.I

theorem complex_cube_equation (x y d : ℤ) (hx : x > 0) (hy : y > 0) :
  (complex x y)^3 = complex (-26) d → complex x y = complex 1 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_equation_l2495_249582


namespace NUMINAMATH_CALUDE_temperature_conversion_l2495_249565

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 50 → k = 122 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2495_249565


namespace NUMINAMATH_CALUDE_tunnel_length_l2495_249553

/-- The length of a tunnel given train and time information --/
theorem tunnel_length (train_length : ℝ) (train_speed : ℝ) (total_time : ℝ) (front_exit_time : ℝ) :
  train_length = 1 ∧ 
  train_speed = 30 ∧ 
  total_time = 5 ∧ 
  front_exit_time = 3 →
  1 = train_speed * front_exit_time - train_length / 2 := by
  sorry

#check tunnel_length

end NUMINAMATH_CALUDE_tunnel_length_l2495_249553


namespace NUMINAMATH_CALUDE_expression_value_l2495_249549

theorem expression_value (a b c d x y : ℤ) :
  (a + b = 0) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (abs x = 3) →  -- absolute value of x is 3
  (y = -1) →     -- y is the largest negative integer
  (2*x - c*d + 6*(a + b) - abs y = 4) ∨ (2*x - c*d + 6*(a + b) - abs y = -8) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2495_249549


namespace NUMINAMATH_CALUDE_set_A_is_open_interval_zero_two_l2495_249520

-- Define the function f(x) = x^3 - x
def f (x : ℝ) : ℝ := x^3 - x

-- Define the set A
def A : Set ℝ := {a : ℝ | a > 0 ∧ ∃ x : ℝ, f (x + a) = f x}

-- Theorem statement
theorem set_A_is_open_interval_zero_two :
  A = Set.Ioo 0 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_set_A_is_open_interval_zero_two_l2495_249520


namespace NUMINAMATH_CALUDE_triangle_area_l2495_249589

/-- The area of a triangle with vertices A(0,0), B(1424233,2848467), and C(1424234,2848469) is 1/2 -/
theorem triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1424233, 2848467)
  let C : ℝ × ℝ := (1424234, 2848469)
  let triangle_area := abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2
  triangle_area = 1/2 := by
sorry

#eval (1424233 * 2848469 - 1424234 * 2848467) / 2

end NUMINAMATH_CALUDE_triangle_area_l2495_249589


namespace NUMINAMATH_CALUDE_cost_of_tomato_seeds_l2495_249500

theorem cost_of_tomato_seeds :
  let pumpkin_cost : ℚ := 5/2
  let chili_cost : ℚ := 9/10
  let pumpkin_packets : ℕ := 3
  let tomato_packets : ℕ := 4
  let chili_packets : ℕ := 5
  let total_spent : ℚ := 18
  ∃ tomato_cost : ℚ, 
    tomato_cost = 3/2 ∧
    pumpkin_cost * pumpkin_packets + tomato_cost * tomato_packets + chili_cost * chili_packets = total_spent :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_tomato_seeds_l2495_249500


namespace NUMINAMATH_CALUDE_parallelepiped_base_sides_l2495_249562

/-- Given a rectangular parallelepiped with a cross-section having diagonals of 20 and 8 units
    intersecting at a 60° angle, the lengths of the sides of its base are 2√5 and √30. -/
theorem parallelepiped_base_sides (d₁ d₂ : ℝ) (θ : ℝ) 
  (h₁ : d₁ = 20) (h₂ : d₂ = 8) (h₃ : θ = Real.pi / 3) :
  ∃ (a b : ℝ), a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 30 ∧ 
  (a * a + b * b = d₁ * d₁) ∧ 
  (d₂ * d₂ = 2 * a * b * Real.cos θ) := by
sorry


end NUMINAMATH_CALUDE_parallelepiped_base_sides_l2495_249562


namespace NUMINAMATH_CALUDE_day_in_consecutive_years_l2495_249509

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℕ
  is_leap : Bool

/-- Function to get the day of the week for a given day number in a year -/
def day_of_week (y : Year) (day_number : ℕ) : DayOfWeek :=
  sorry

/-- Function to check if a given day number is a Friday -/
def is_friday (y : Year) (day_number : ℕ) : Prop :=
  day_of_week y day_number = DayOfWeek.Friday

/-- Theorem stating the relationship between the days in consecutive years -/
theorem day_in_consecutive_years 
  (n : ℕ) 
  (year_n : Year)
  (year_n_plus_1 : Year)
  (year_n_minus_1 : Year)
  (h1 : year_n.number = n)
  (h2 : year_n_plus_1.number = n + 1)
  (h3 : year_n_minus_1.number = n - 1)
  (h4 : is_friday year_n 250)
  (h5 : is_friday year_n_plus_1 150) :
  day_of_week year_n_minus_1 50 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_day_in_consecutive_years_l2495_249509


namespace NUMINAMATH_CALUDE_triangle_properties_l2495_249527

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (c * Real.sin B = Real.sqrt 3 * Real.cos C) →
  (a + b = 6) →
  (C = π / 3 ∧ a + b + c ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2495_249527


namespace NUMINAMATH_CALUDE_sam_remaining_money_l2495_249511

/-- Given an initial amount, number of books, and cost per book, 
    calculate the remaining amount after purchase. -/
def remaining_amount (initial : ℕ) (num_books : ℕ) (cost_per_book : ℕ) : ℕ :=
  initial - (num_books * cost_per_book)

/-- Theorem stating that given the specific conditions of Sam's purchase,
    the remaining amount is 16 dollars. -/
theorem sam_remaining_money :
  remaining_amount 79 9 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_remaining_money_l2495_249511


namespace NUMINAMATH_CALUDE_orange_price_is_60_l2495_249577

/-- The price of an orange in cents, given the conditions of the fruit stand problem -/
def orange_price : ℕ :=
  let apple_price : ℕ := 40
  let total_fruits : ℕ := 15
  let initial_avg_price : ℕ := 48
  let final_avg_price : ℕ := 45
  let removed_oranges : ℕ := 3
  60

/-- Theorem stating that the price of an orange is 60 cents -/
theorem orange_price_is_60 :
  orange_price = 60 := by sorry

end NUMINAMATH_CALUDE_orange_price_is_60_l2495_249577


namespace NUMINAMATH_CALUDE_robin_water_bottles_l2495_249599

/-- The number of additional bottles needed on the last day -/
def additional_bottles (total_bottles : ℕ) (daily_consumption : ℕ) : ℕ :=
  daily_consumption - (total_bottles % daily_consumption)

/-- Theorem stating that given 617 bottles and a daily consumption of 9 bottles, 
    4 additional bottles are needed on the last day -/
theorem robin_water_bottles : additional_bottles 617 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_robin_water_bottles_l2495_249599


namespace NUMINAMATH_CALUDE_distance_A_to_B_is_300_l2495_249581

/-- The distance between two points A and B, given the following conditions:
    - Monkeys travel from A to B
    - A monkey departs from A every 3 minutes
    - It takes a monkey 12 minutes to travel from A to B
    - A rabbit runs from B to A
    - When the rabbit starts, a monkey has just arrived at B
    - The rabbit encounters 5 monkeys on its way to A
    - The rabbit arrives at A just as another monkey leaves A
    - The rabbit's speed is 3 km/h
-/
def distance_A_to_B : ℝ :=
  let monkey_departure_interval : ℝ := 3 -- minutes
  let monkey_travel_time : ℝ := 12 -- minutes
  let encountered_monkeys : ℕ := 5
  let rabbit_speed : ℝ := 3 * 1000 / 60 -- convert 3 km/h to m/min

  -- Define the distance based on the given conditions
  300 -- meters

theorem distance_A_to_B_is_300 :
  distance_A_to_B = 300 := by sorry

end NUMINAMATH_CALUDE_distance_A_to_B_is_300_l2495_249581


namespace NUMINAMATH_CALUDE_binomial_60_3_l2495_249547

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l2495_249547


namespace NUMINAMATH_CALUDE_inequality_proof_l2495_249552

theorem inequality_proof (x : ℝ) : x^2 + 1 + 1/(x^2 + 1) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2495_249552


namespace NUMINAMATH_CALUDE_probability_third_is_three_l2495_249524

-- Define the set of permutations
def T : Finset (Fin 6 → Fin 6) :=
  (Finset.univ.filter (λ σ : Fin 6 → Fin 6 => Function.Injective σ ∧ σ 0 ≠ 1))

-- Define the probability of the event
def prob_third_is_three (T : Finset (Fin 6 → Fin 6)) : ℚ :=
  (T.filter (λ σ : Fin 6 → Fin 6 => σ 2 = 2)).card / T.card

-- Theorem statement
theorem probability_third_is_three :
  prob_third_is_three T = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_third_is_three_l2495_249524


namespace NUMINAMATH_CALUDE_monotonicity_f_range_of_a_l2495_249505

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -a * x^2 + Real.log x

-- State the theorems
theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < 1 / Real.sqrt (2 * a) → f a x₁ < f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, 1 / Real.sqrt (2 * a) < x₁ → x₁ < x₂ → f a x₁ > f a x₂) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, x > 1 ∧ f a x > -a) ↔ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_f_range_of_a_l2495_249505


namespace NUMINAMATH_CALUDE_matrix_commute_result_l2495_249595

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]

def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]

theorem matrix_commute_result (a b c d : ℝ) (h1 : A * B a b c d = B a b c d * A) 
  (h2 : 4 * b ≠ c) : (a - d) / (c - 4 * b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_commute_result_l2495_249595


namespace NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l2495_249545

theorem fourth_root_256_times_cube_root_64_times_sqrt_16 :
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 64 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l2495_249545


namespace NUMINAMATH_CALUDE_square_root_cube_root_relation_l2495_249594

theorem square_root_cube_root_relation (x : ℝ) : 
  (∃ y : ℝ, y^2 = x ∧ (y = 8 ∨ y = -8)) → x^(1/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_cube_root_relation_l2495_249594


namespace NUMINAMATH_CALUDE_gravel_path_cost_is_360_l2495_249596

/-- Calculates the cost of gravelling a path around a rectangular plot -/
def gravel_path_cost (plot_length plot_width path_width : ℝ) (cost_per_sqm : ℝ) : ℝ :=
  let outer_length := plot_length + 2 * path_width
  let outer_width := plot_width + 2 * path_width
  let path_area := outer_length * outer_width - plot_length * plot_width
  path_area * cost_per_sqm

/-- Theorem: The cost of gravelling the path is 360 rupees -/
theorem gravel_path_cost_is_360 :
  gravel_path_cost 110 65 2.5 0.4 = 360 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_cost_is_360_l2495_249596


namespace NUMINAMATH_CALUDE_randy_money_problem_l2495_249555

def randy_initial_money (randy_received : ℕ) (randy_gave : ℕ) (randy_left : ℕ) : Prop :=
  ∃ (initial : ℕ), initial + randy_received - randy_gave = randy_left

theorem randy_money_problem :
  randy_initial_money 200 1200 2000 → ∃ (initial : ℕ), initial = 3000 := by
  sorry

end NUMINAMATH_CALUDE_randy_money_problem_l2495_249555


namespace NUMINAMATH_CALUDE_divisible_by_four_or_seven_count_divisible_by_four_or_seven_l2495_249540

theorem divisible_by_four_or_seven (n : Nat) : 
  (∃ k : Nat, n = 4 * k ∨ n = 7 * k) ↔ n ∈ Finset.filter (λ x : Nat => x % 4 = 0 ∨ x % 7 = 0) (Finset.range 61) :=
sorry

theorem count_divisible_by_four_or_seven : 
  (Finset.filter (λ x : Nat => x % 4 = 0 ∨ x % 7 = 0) (Finset.range 61)).card = 21 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_four_or_seven_count_divisible_by_four_or_seven_l2495_249540


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l2495_249579

theorem isosceles_triangle_condition (a b c A B C : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  (a * Real.cos C + c * Real.cos B = b) →
  (a = b ∧ A = B) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l2495_249579


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l2495_249537

theorem inscribed_octagon_area (r : ℝ) (h : r^2 * Real.pi = 400 * Real.pi) :
  2 * r^2 * (1 + Real.sqrt 2) = 800 + 800 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l2495_249537


namespace NUMINAMATH_CALUDE_robotics_camp_age_problem_l2495_249523

theorem robotics_camp_age_problem (total_members : ℕ) (overall_avg_age : ℕ) 
  (num_girls num_boys num_adults : ℕ) (avg_age_girls avg_age_boys : ℕ) :
  total_members = 60 →
  overall_avg_age = 20 →
  num_girls = 30 →
  num_boys = 20 →
  num_adults = 10 →
  avg_age_girls = 18 →
  avg_age_boys = 22 →
  num_girls + num_boys + num_adults = total_members →
  (avg_age_girls * num_girls + avg_age_boys * num_boys + 
   22 * num_adults : ℕ) / total_members = overall_avg_age :=
by sorry

end NUMINAMATH_CALUDE_robotics_camp_age_problem_l2495_249523


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_l2495_249501

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the perpendicular relation between planes and between lines and planes
variable (perp : Plane → Plane → Prop)
variable (perpL : Line → Plane → Prop)

-- Define the planes and lines
variable (α β γ : Plane) (m n : Line)

-- State the theorem
theorem perpendicular_line_to_plane
  (h1 : intersect α γ = m)
  (h2 : perp β α)
  (h3 : perp β γ)
  (h4 : perpL n α)
  (h5 : perpL n β)
  (h6 : perpL m α) :
  perpL m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_l2495_249501
