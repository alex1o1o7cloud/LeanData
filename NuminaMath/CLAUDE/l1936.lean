import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_regions_l1936_193660

/-- 
Given n ellipses in a plane where:
- Any two ellipses intersect at exactly two points
- No three ellipses intersect at the same point

The number of regions these ellipses divide the plane into is n(n-1) + 2.
-/
theorem ellipse_regions (n : ℕ) : ℕ := by
  sorry

#check ellipse_regions

end NUMINAMATH_CALUDE_ellipse_regions_l1936_193660


namespace NUMINAMATH_CALUDE_eight_lines_divide_plane_into_37_regions_l1936_193628

/-- The number of regions created by n lines in a plane, where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + n.choose 2

/-- Theorem stating that 8 lines divide the plane into 37 regions -/
theorem eight_lines_divide_plane_into_37_regions :
  num_regions 8 = 37 := by sorry

end NUMINAMATH_CALUDE_eight_lines_divide_plane_into_37_regions_l1936_193628


namespace NUMINAMATH_CALUDE_fred_total_games_l1936_193640

/-- The number of basketball games Fred attended this year -/
def games_this_year : ℕ := 36

/-- The number of basketball games Fred attended last year -/
def games_last_year : ℕ := 11

/-- The total number of basketball games Fred attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem fred_total_games : total_games = 47 := by
  sorry

end NUMINAMATH_CALUDE_fred_total_games_l1936_193640


namespace NUMINAMATH_CALUDE_correct_product_l1936_193662

def reverse_digits (n : Nat) : Nat :=
  (n % 10) * 10 + (n / 10)

theorem correct_product (a b : Nat) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (0 < b) →             -- b is positive
  ((reverse_digits a) * b = 143) →  -- erroneous product
  (a * b = 341) :=
by sorry

end NUMINAMATH_CALUDE_correct_product_l1936_193662


namespace NUMINAMATH_CALUDE_no_valid_ratio_l1936_193615

/-- Represents an isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  b : ℝ  -- Length of the larger base
  a : ℝ  -- Length of the smaller base
  h : ℝ  -- Height of the trapezoid
  is_positive : 0 < b
  smaller_base_eq_diagonal : a = h
  altitude_eq_larger_base : h = b

/-- Theorem stating that no valid ratio exists between the bases of the described trapezoid -/
theorem no_valid_ratio (t : IsoscelesTrapezoid) : False :=
sorry

end NUMINAMATH_CALUDE_no_valid_ratio_l1936_193615


namespace NUMINAMATH_CALUDE_square_sum_triples_l1936_193673

theorem square_sum_triples :
  ∀ a b c : ℝ,
  (a = (b + c)^2 ∧ b = (a + c)^2 ∧ c = (a + b)^2) →
  ((a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = 1/4 ∧ b = 1/4 ∧ c = 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_triples_l1936_193673


namespace NUMINAMATH_CALUDE_dividend_calculation_l1936_193654

/-- Calculate the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (share_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : share_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.05) :
  let actual_share_price := share_value * (1 + premium_rate)
  let num_shares := investment / actual_share_price
  let dividend_per_share := share_value * dividend_rate
  num_shares * dividend_per_share = 600 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1936_193654


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1936_193619

theorem candy_bar_cost (initial_amount : ℝ) (num_candy_bars : ℕ) (remaining_amount : ℝ) :
  initial_amount = 20 →
  num_candy_bars = 4 →
  remaining_amount = 12 →
  (initial_amount - remaining_amount) / num_candy_bars = 2 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1936_193619


namespace NUMINAMATH_CALUDE_bobbit_worm_consumption_l1936_193633

/-- Represents the number of fish eaten by the Bobbit worm each day -/
def fish_eaten_per_day : ℕ := 2

/-- The initial number of fish in the aquarium -/
def initial_fish : ℕ := 60

/-- The number of fish added after two weeks -/
def fish_added : ℕ := 8

/-- The number of fish remaining after three weeks -/
def remaining_fish : ℕ := 26

/-- The total number of days -/
def total_days : ℕ := 21

theorem bobbit_worm_consumption :
  initial_fish + fish_added - (fish_eaten_per_day * total_days) = remaining_fish := by
  sorry

end NUMINAMATH_CALUDE_bobbit_worm_consumption_l1936_193633


namespace NUMINAMATH_CALUDE_fifth_term_is_123_40_l1936_193679

-- Define the arithmetic sequence
def arithmeticSequence (x y : ℚ) : ℕ → ℚ
  | 0 => x + y
  | 1 => x - y
  | 2 => x * y
  | 3 => x / y
  | n + 4 => arithmeticSequence x y 3 - (n + 1) * (2 * y)

-- Theorem statement
theorem fifth_term_is_123_40 (x y : ℚ) :
  x - y - (x + y) = -2 * y →
  x - 3 * y = x * y →
  x - 5 * y = x / y →
  y ≠ 0 →
  arithmeticSequence x y 4 = 123 / 40 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_is_123_40_l1936_193679


namespace NUMINAMATH_CALUDE_min_vertices_blue_triangle_or_red_K4_l1936_193643

/-- A type representing a 2-coloring of edges in a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate for the existence of a blue triangle in a 2-coloring -/
def has_blue_triangle (n : ℕ) (c : TwoColoring n) : Prop :=
  ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    c i j = false ∧ c j k = false ∧ c i k = false

/-- Predicate for the existence of a red K4 in a 2-coloring -/
def has_red_K4 (n : ℕ) (c : TwoColoring n) : Prop :=
  ∃ i j k l : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    c i j = true ∧ c i k = true ∧ c i l = true ∧
    c j k = true ∧ c j l = true ∧ c k l = true

/-- The main theorem -/
theorem min_vertices_blue_triangle_or_red_K4 :
  (∀ n < 9, ∃ c : TwoColoring n, ¬has_blue_triangle n c ∧ ¬has_red_K4 n c) ∧
  (∀ c : TwoColoring 9, has_blue_triangle 9 c ∨ has_red_K4 9 c) :=
sorry

end NUMINAMATH_CALUDE_min_vertices_blue_triangle_or_red_K4_l1936_193643


namespace NUMINAMATH_CALUDE_triangle_inequality_specific_l1936_193693

/-- Triangle inequality theorem for a specific triangle --/
theorem triangle_inequality_specific (a b c : ℝ) (ha : a = 5) (hb : b = 8) (hc : c = 6) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_specific_l1936_193693


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_exists_and_unique_l1936_193614

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Checks if a number is part of the systematic sample -/
def isInSample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

theorem systematic_sample_fourth_element_exists_and_unique
  (total_students : ℕ)
  (sample_size : ℕ)
  (n1 n2 n3 : ℕ)
  (h_total : total_students = 52)
  (h_sample : sample_size = 4)
  (h_n1 : n1 = 6)
  (h_n2 : n2 = 32)
  (h_n3 : n3 = 45)
  (h_distinct : n1 < n2 ∧ n2 < n3)
  (h_valid : n1 ≤ total_students ∧ n2 ≤ total_students ∧ n3 ≤ total_students) :
  ∃! n4 : ℕ,
    ∃ s : SystematicSample,
      s.population_size = total_students ∧
      s.sample_size = sample_size ∧
      isInSample s n1 ∧
      isInSample s n2 ∧
      isInSample s n3 ∧
      isInSample s n4 ∧
      n4 ≠ n1 ∧ n4 ≠ n2 ∧ n4 ≠ n3 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_exists_and_unique_l1936_193614


namespace NUMINAMATH_CALUDE_fraction_removal_sum_one_l1936_193686

theorem fraction_removal_sum_one :
  let fractions : List ℚ := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let removed : List ℚ := [1/8, 1/10]
  let remaining : List ℚ := fractions.filter (fun x => x ∉ removed)
  remaining.sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_removal_sum_one_l1936_193686


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l1936_193625

theorem cubic_sum_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z = 0) (hprod : x * y + x * z + y * z ≠ 0) :
  (x^3 + y^3 + z^3) / (x * y * z * (x * y + x * z + y * z)) = -3 :=
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l1936_193625


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l1936_193670

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x + y + x * y = 500 ↔ 
    ((x = 0 ∧ y = 500) ∨ 
     (x = -2 ∧ y = -502) ∨ 
     (x = 2 ∧ y = 166) ∨ 
     (x = -4 ∧ y = -168)) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l1936_193670


namespace NUMINAMATH_CALUDE_vacation_cost_equality_l1936_193649

/-- Proves that t - d + s = 20 given the vacation cost conditions --/
theorem vacation_cost_equality (tom_paid dorothy_paid sammy_paid t d s : ℚ) :
  tom_paid = 150 →
  dorothy_paid = 160 →
  sammy_paid = 210 →
  let total := tom_paid + dorothy_paid + sammy_paid
  let per_person := total / 3
  t = per_person - tom_paid →
  d = per_person - dorothy_paid →
  s = sammy_paid - per_person →
  t - d + s = 20 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_equality_l1936_193649


namespace NUMINAMATH_CALUDE_chennys_friends_l1936_193682

theorem chennys_friends (initial_candies : ℕ) (additional_candies : ℕ) (candies_per_friend : ℕ) : 
  initial_candies = 10 →
  additional_candies = 4 →
  candies_per_friend = 2 →
  (initial_candies + additional_candies) / candies_per_friend = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_chennys_friends_l1936_193682


namespace NUMINAMATH_CALUDE_no_common_points_implies_b_range_l1936_193681

theorem no_common_points_implies_b_range 
  (f g : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x = x^2 - a*x) 
  (k : ∀ x : ℝ, g x = b + a * Real.log (x - 1)) 
  (a_ge_one : a ≥ 1) 
  (no_common_points : ∀ x : ℝ, f x ≠ g x) : 
  b < 3/4 + Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_no_common_points_implies_b_range_l1936_193681


namespace NUMINAMATH_CALUDE_smallest_four_digit_perfect_square_multiple_smallest_four_digit_perfect_square_multiple_exists_l1936_193687

theorem smallest_four_digit_perfect_square_multiple :
  ∀ m : ℕ, m ≥ 1000 → m < 1029 → ¬∃ n : ℕ, 21 * m = n * n :=
by
  sorry

theorem smallest_four_digit_perfect_square_multiple_exists :
  ∃ n : ℕ, 21 * 1029 = n * n :=
by
  sorry

#check smallest_four_digit_perfect_square_multiple
#check smallest_four_digit_perfect_square_multiple_exists

end NUMINAMATH_CALUDE_smallest_four_digit_perfect_square_multiple_smallest_four_digit_perfect_square_multiple_exists_l1936_193687


namespace NUMINAMATH_CALUDE_two_books_into_five_l1936_193602

/-- The number of ways to insert new books into a shelf while maintaining the order of existing books -/
def insert_books (original : ℕ) (new : ℕ) : ℕ :=
  (original + 1) * (original + 2) / 2

/-- Theorem stating that inserting 2 books into a shelf with 5 books results in 42 different arrangements -/
theorem two_books_into_five : insert_books 5 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_two_books_into_five_l1936_193602


namespace NUMINAMATH_CALUDE_kolakoski_next_eight_terms_l1936_193626

/-- The Kolakoski sequence -/
def kolakoski : ℕ → Fin 2
  | 0 => 0  -- represents 1
  | 1 => 1  -- represents 2
  | 2 => 1  -- represents 2
  | n + 3 => sorry

/-- The run-length encoding of the Kolakoski sequence -/
def kolakoski_rle : ℕ → Fin 2
  | n => kolakoski n

theorem kolakoski_next_eight_terms :
  (List.range 8).map (fun i => kolakoski (i + 12)) = [1, 1, 0, 0, 1, 0, 0, 1] := by
  sorry

#check kolakoski_next_eight_terms

end NUMINAMATH_CALUDE_kolakoski_next_eight_terms_l1936_193626


namespace NUMINAMATH_CALUDE_reading_time_calculation_l1936_193623

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of paragraphs per page -/
def paragraphs_per_page : ℕ := 20

/-- Represents the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ := 10

/-- Represents the total number of pages in the book -/
def total_pages : ℕ := 50

/-- Calculates the total reading time in hours -/
def total_reading_time : ℚ :=
  (total_pages * paragraphs_per_page * sentences_per_paragraph) / reading_speed

theorem reading_time_calculation :
  total_reading_time = 50 := by sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l1936_193623


namespace NUMINAMATH_CALUDE_smallest_x_is_correct_smallest_x_works_l1936_193688

/-- The smallest positive integer x such that 1800x is a perfect cube -/
def smallest_x : ℕ := 15

theorem smallest_x_is_correct :
  ∀ y : ℕ, y > 0 → (∃ m : ℕ, 1800 * y = m^3) → y ≥ smallest_x :=
by sorry

theorem smallest_x_works :
  ∃ m : ℕ, 1800 * smallest_x = m^3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_is_correct_smallest_x_works_l1936_193688


namespace NUMINAMATH_CALUDE_first_non_divisor_is_seven_l1936_193685

def is_valid_integer (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 300 ∧ n % 3 ≠ 0 ∧ n % 5 ≠ 0

theorem first_non_divisor_is_seven :
  ∃ (S : Finset ℕ), 
    Finset.card S = 26 ∧ 
    (∀ n ∈ S, is_valid_integer n) ∧
    (∀ k > 5, k < 7 → ∃ n ∈ S, n % k = 0) ∧
    (∀ n ∈ S, n % 7 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_first_non_divisor_is_seven_l1936_193685


namespace NUMINAMATH_CALUDE_bruce_payment_l1936_193600

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grapes_quantity grapes_rate mangoes_quantity mangoes_rate : ℕ) : ℕ :=
  grapes_quantity * grapes_rate + mangoes_quantity * mangoes_rate

/-- Theorem stating that Bruce paid 1055 to the shopkeeper -/
theorem bruce_payment : total_amount 8 70 9 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l1936_193600


namespace NUMINAMATH_CALUDE_distribution_count_l1936_193674

def number_of_women : ℕ := 2
def number_of_men : ℕ := 10
def number_of_magazines : ℕ := 8
def number_of_newspapers : ℕ := 4

theorem distribution_count :
  (Nat.choose number_of_men (number_of_newspapers - 1)) +
  (Nat.choose number_of_men number_of_newspapers) = 255 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_l1936_193674


namespace NUMINAMATH_CALUDE_math_books_count_l1936_193613

/-- Proves that the number of math books bought is 27 given the conditions of the problem -/
theorem math_books_count (total_books : ℕ) (math_book_price history_book_price total_price : ℕ) 
  (h1 : total_books = 80)
  (h2 : math_book_price = 4)
  (h3 : history_book_price = 5)
  (h4 : total_price = 373) :
  ∃ (math_books : ℕ), 
    math_books + (total_books - math_books) = total_books ∧ 
    math_books * math_book_price + (total_books - math_books) * history_book_price = total_price ∧
    math_books = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l1936_193613


namespace NUMINAMATH_CALUDE_spherical_distance_for_pi_over_six_l1936_193620

/-- The spherical distance between two points on a sphere's surface -/
def spherical_distance (R : ℝ) (angle : ℝ) : ℝ := R * angle

/-- Theorem: The spherical distance between two points A and B on a sphere with radius R,
    where the angle AOB is π/6, is equal to (π/6)R -/
theorem spherical_distance_for_pi_over_six (R : ℝ) (h : R > 0) :
  spherical_distance R (π/6) = (π/6) * R := by sorry

end NUMINAMATH_CALUDE_spherical_distance_for_pi_over_six_l1936_193620


namespace NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_120_l1936_193689

theorem greatest_common_multiple_10_15_under_120 : 
  ∃ (n : ℕ), n = Nat.lcm 10 15 ∧ n < 120 ∧ ∀ m : ℕ, (m = Nat.lcm 10 15 ∧ m < 120) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_120_l1936_193689


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_fifth_l1936_193605

theorem reciprocal_of_negative_one_fifth : 
  ∀ x : ℚ, x = -1/5 → (∃ y : ℚ, y * x = 1 ∧ y = -5) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_fifth_l1936_193605


namespace NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l1936_193667

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (A B : ℝ × ℝ) 
  (h_A : point_on_parabola A) 
  (h_B : point_on_parabola B) 
  (h_dist : distance A focus + distance B focus = 12) :
  (A.1 + B.1) / 2 = 5 := by sorry

end

end NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l1936_193667


namespace NUMINAMATH_CALUDE_product_of_roots_l1936_193695

theorem product_of_roots (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ - 1 = 0) → (x₂^2 + x₂ - 1 = 0) → x₁ * x₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1936_193695


namespace NUMINAMATH_CALUDE_correct_equation_l1936_193671

theorem correct_equation (a b : ℝ) : 3 * a^2 * b - 4 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l1936_193671


namespace NUMINAMATH_CALUDE_triangle_properties_l1936_193669

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B = 3 * t.C ∧
  2 * Real.sin (t.A - t.C) = Real.sin t.B ∧
  t.AB = 5

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = 3 * Real.sqrt 10 / 10 ∧
  ∃ (height : Real), height = 6 ∧ 
    height * t.AB / 2 = Real.sin t.C * (t.AB * Real.sin t.B / Real.sin t.C) * (t.AB * Real.sin t.A / Real.sin t.C) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1936_193669


namespace NUMINAMATH_CALUDE_inequality_proof_l1936_193697

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1936_193697


namespace NUMINAMATH_CALUDE_tims_age_l1936_193612

theorem tims_age (james_age john_age tim_age : ℕ) : 
  james_age = 23 → 
  john_age = 35 → 
  tim_age = 2 * john_age - 5 → 
  tim_age = 65 := by
sorry

end NUMINAMATH_CALUDE_tims_age_l1936_193612


namespace NUMINAMATH_CALUDE_randys_initial_amount_l1936_193666

/-- Proves that Randy's initial amount was $3000 given the described transactions --/
theorem randys_initial_amount (initial final smith_gave sally_received : ℕ) :
  final = initial + smith_gave - sally_received →
  smith_gave = 200 →
  sally_received = 1200 →
  final = 2000 →
  initial = 3000 := by
  sorry

end NUMINAMATH_CALUDE_randys_initial_amount_l1936_193666


namespace NUMINAMATH_CALUDE_expand_product_l1936_193629

theorem expand_product (x : ℝ) : (x^2 - 3*x + 4) * (x^2 + 3*x + 1) = x^4 - 4*x^2 + 9*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1936_193629


namespace NUMINAMATH_CALUDE_anoop_investment_l1936_193606

/-- Calculates the investment amount of the second partner in a business partnership --/
def calculate_second_partner_investment (first_partner_investment : ℕ) (first_partner_months : ℕ) (second_partner_months : ℕ) : ℕ :=
  (first_partner_investment * first_partner_months) / second_partner_months

/-- Proves that Anoop's investment is 40,000 given the problem conditions --/
theorem anoop_investment :
  let arjun_investment : ℕ := 20000
  let total_months : ℕ := 12
  let anoop_months : ℕ := 6
  calculate_second_partner_investment arjun_investment total_months anoop_months = 40000 := by
  sorry

#eval calculate_second_partner_investment 20000 12 6

end NUMINAMATH_CALUDE_anoop_investment_l1936_193606


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1936_193683

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ((-2 : ℝ) * m + 2 * m + 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1936_193683


namespace NUMINAMATH_CALUDE_unique_abc_solution_l1936_193678

theorem unique_abc_solution (a b c : ℕ+) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_abc_solution_l1936_193678


namespace NUMINAMATH_CALUDE_inequality_proof_l1936_193655

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (2*b + a + c)^2 / (2*b^2 + (c + a)^2) +
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1936_193655


namespace NUMINAMATH_CALUDE_functional_equation_solution_functional_equation_continuous_solution_l1936_193632

def functional_equation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f 0 = 0 ∧ f 1 = 1 ∧ ∀ x y, x ≤ y → f ((x + y) / 2) = (1 - a) * f x + a * f y

theorem functional_equation_solution (a : ℝ) :
  (∃ f : ℝ → ℝ, functional_equation f a) ↔ (a = 0 ∨ a = 1/2 ∨ a = 1) :=
sorry

theorem functional_equation_continuous_solution (a : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ functional_equation f a) ↔ a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_functional_equation_continuous_solution_l1936_193632


namespace NUMINAMATH_CALUDE_lost_money_proof_l1936_193668

def money_lost (initial_amount spent_amount remaining_amount : ℕ) : ℕ :=
  (initial_amount - spent_amount) - remaining_amount

theorem lost_money_proof (initial_amount spent_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  money_lost initial_amount spent_amount remaining_amount = 6 := by
  sorry

end NUMINAMATH_CALUDE_lost_money_proof_l1936_193668


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_sum_l1936_193692

theorem smallest_integer_fraction_sum (A B C D : ℕ) : 
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  (A + B) % (C + D) = 0 →
  (∀ E F G H : ℕ, E ≤ 9 ∧ F ≤ 9 ∧ G ≤ 9 ∧ H ≤ 9 →
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H →
    (E + F) % (G + H) = 0 →
    (A + B) / (C + D) ≤ (E + F) / (G + H)) →
  C + D = 17 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_sum_l1936_193692


namespace NUMINAMATH_CALUDE_line_parametric_to_standard_l1936_193642

/-- Given a line with parametric equations x = 1 + t and y = -1 + t,
    prove that its standard equation is x - y - 2 = 0 -/
theorem line_parametric_to_standard :
  ∀ (x y t : ℝ), x = 1 + t ∧ y = -1 + t → x - y - 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_parametric_to_standard_l1936_193642


namespace NUMINAMATH_CALUDE_lucille_weeding_ratio_l1936_193663

def weed_value : ℕ := 6
def flower_bed_weeds : ℕ := 11
def vegetable_patch_weeds : ℕ := 14
def total_grass_weeds : ℕ := 32
def soda_cost : ℕ := 99
def remaining_money : ℕ := 147

theorem lucille_weeding_ratio :
  let total_earned := remaining_money + soda_cost
  let flower_veg_earnings := (flower_bed_weeds + vegetable_patch_weeds) * weed_value
  let grass_earnings := total_earned - flower_veg_earnings
  let grass_weeds_pulled := grass_earnings / weed_value
  (grass_weeds_pulled : ℚ) / total_grass_weeds = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_lucille_weeding_ratio_l1936_193663


namespace NUMINAMATH_CALUDE_total_teaching_years_is_70_l1936_193610

/-- The total number of years Tom and Devin have been teaching -/
def total_teaching_years (tom_years devin_years : ℕ) : ℕ := tom_years + devin_years

/-- Tom's teaching years -/
def tom_years : ℕ := 50

/-- Devin's teaching years in terms of Tom's -/
def devin_years : ℕ := tom_years / 2 - 5

theorem total_teaching_years_is_70 : 
  total_teaching_years tom_years devin_years = 70 := by sorry

end NUMINAMATH_CALUDE_total_teaching_years_is_70_l1936_193610


namespace NUMINAMATH_CALUDE_min_chess_pieces_chess_pieces_solution_l1936_193690

theorem min_chess_pieces (n : ℕ) : 
  (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) → n ≥ 103 :=
by sorry

theorem chess_pieces_solution : 
  ∃ (n : ℕ), (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) ∧ n = 103 :=
by sorry

end NUMINAMATH_CALUDE_min_chess_pieces_chess_pieces_solution_l1936_193690


namespace NUMINAMATH_CALUDE_french_books_count_l1936_193698

/-- The number of English books -/
def num_english_books : ℕ := 11

/-- The total number of arrangement ways -/
def total_arrangements : ℕ := 220

/-- The number of French books -/
def num_french_books : ℕ := 3

/-- The number of slots for French books -/
def num_slots : ℕ := num_english_books + 1

theorem french_books_count :
  (Nat.choose num_slots num_french_books = total_arrangements) ∧
  (∀ k : ℕ, k ≠ num_french_books → Nat.choose num_slots k ≠ total_arrangements) :=
sorry

end NUMINAMATH_CALUDE_french_books_count_l1936_193698


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1936_193624

theorem line_passes_through_fixed_point (a b c : ℝ) (h : a - b + c = 0) :
  a * 1 + b * (-1) + c = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1936_193624


namespace NUMINAMATH_CALUDE_f_has_at_most_two_zeros_l1936_193627

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + a

-- State the theorem
theorem f_has_at_most_two_zeros (a : ℝ) (h : a ≥ 16) :
  ∃ (z₁ z₂ : ℝ), ∀ x : ℝ, f a x = 0 → x = z₁ ∨ x = z₂ :=
sorry

end NUMINAMATH_CALUDE_f_has_at_most_two_zeros_l1936_193627


namespace NUMINAMATH_CALUDE_division_result_l1936_193611

theorem division_result : ∃ (q : ℕ), 1254 = 6 * q → q = 209 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1936_193611


namespace NUMINAMATH_CALUDE_smallest_prime_factors_sum_of_286_l1936_193699

theorem smallest_prime_factors_sum_of_286 : 
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p < q ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ 286 → r = p ∨ r = q ∨ r > q) ∧ 
    p + q = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factors_sum_of_286_l1936_193699


namespace NUMINAMATH_CALUDE_reading_time_difference_l1936_193607

def xanthia_speed : ℝ := 120
def molly_speed : ℝ := 60
def book_pages : ℝ := 300

theorem reading_time_difference : 
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 150 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l1936_193607


namespace NUMINAMATH_CALUDE_roots_properties_l1936_193657

theorem roots_properties (x : ℝ) : 
  (x^2 - 7 * |x| + 6 = 0) → 
  (∃ (roots : Finset ℝ), 
    (∀ r ∈ roots, r^2 - 7 * |r| + 6 = 0) ∧ 
    (Finset.sum roots id = 0) ∧ 
    (Finset.prod roots id = 36)) :=
by sorry

end NUMINAMATH_CALUDE_roots_properties_l1936_193657


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l1936_193641

theorem sum_remainder_mod_nine (n : ℤ) : ((9 - n) + (n + 5)) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l1936_193641


namespace NUMINAMATH_CALUDE_functional_equation_iff_forms_l1936_193609

/-- The functional equation that f and g must satisfy for all real x and y -/
def functional_equation (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, Real.sin x + Real.cos y = f x + f y + g x - g y

/-- The proposed form of function f -/
def f_form (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = (Real.sin x + Real.cos x) / 2

/-- The proposed form of function g, with an arbitrary constant C -/
def g_form (g : ℝ → ℝ) : Prop :=
  ∃ C : ℝ, ∀ x : ℝ, g x = (Real.sin x - Real.cos x) / 2 + C

/-- The main theorem stating the equivalence between the functional equation and the proposed forms of f and g -/
theorem functional_equation_iff_forms (f g : ℝ → ℝ) :
  functional_equation f g ↔ (f_form f ∧ g_form g) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_iff_forms_l1936_193609


namespace NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l1936_193638

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (a b : ℝ) : Prop := a * b = -1

/-- The condition p: ax + y + 1 = 0 is perpendicular to ax - y + 2 = 0 -/
def p (a : ℝ) : Prop := perpendicular a (-a)

/-- The condition q: a = 1 -/
def q : ℝ → Prop := (· = 1)

/-- p is neither sufficient nor necessary for q -/
theorem p_neither_sufficient_nor_necessary_for_q :
  (¬∀ a, p a → q a) ∧ (¬∀ a, q a → p a) := by sorry

end NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l1936_193638


namespace NUMINAMATH_CALUDE_max_intersections_ellipse_cosine_l1936_193618

-- Define the ellipse equation
def ellipse (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

-- Define the cosine function
def cosine_graph (x y : ℝ) : Prop :=
  y = Real.cos x

-- Theorem statement
theorem max_intersections_ellipse_cosine :
  ∃ (h k a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∃ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points → ellipse p.1 p.2 h k a b ∧ cosine_graph p.1 p.2) ∧
    points.card = 8) ∧
  (∀ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points → ellipse p.1 p.2 h k a b ∧ cosine_graph p.1 p.2) →
    points.card ≤ 8) :=
by sorry


end NUMINAMATH_CALUDE_max_intersections_ellipse_cosine_l1936_193618


namespace NUMINAMATH_CALUDE_peters_pumpkin_profit_l1936_193677

/-- Represents the total amount of money collected from selling pumpkins -/
def total_money (jumbo_price regular_price : ℝ) (total_pumpkins regular_pumpkins : ℕ) : ℝ :=
  regular_price * regular_pumpkins + jumbo_price * (total_pumpkins - regular_pumpkins)

/-- Theorem stating that Peter's total money collected is $395.00 -/
theorem peters_pumpkin_profit :
  total_money 9 4 80 65 = 395 := by
  sorry

end NUMINAMATH_CALUDE_peters_pumpkin_profit_l1936_193677


namespace NUMINAMATH_CALUDE_fraction_addition_l1936_193684

theorem fraction_addition : (7 : ℚ) / 12 + (3 : ℚ) / 8 = (23 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1936_193684


namespace NUMINAMATH_CALUDE_sheets_per_box_l1936_193675

theorem sheets_per_box (total_sheets : ℕ) (num_boxes : ℕ) (h1 : total_sheets = 700) (h2 : num_boxes = 7) :
  total_sheets / num_boxes = 100 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_box_l1936_193675


namespace NUMINAMATH_CALUDE_certain_point_on_circle_l1936_193653

/-- A point on the parabola y^2 = 8x -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 8*x

/-- A circle with center on the parabola y^2 = 8x and tangent to x = -2 -/
structure TangentCircle where
  center : ParabolaPoint
  radius : ℝ
  is_tangent : center.x + radius = 2  -- Distance from center to x = -2 is equal to radius

theorem certain_point_on_circle (c : TangentCircle) : 
  (c.center.x - 2)^2 + c.center.y^2 = c.radius^2 := by
  sorry

#check certain_point_on_circle

end NUMINAMATH_CALUDE_certain_point_on_circle_l1936_193653


namespace NUMINAMATH_CALUDE_solution_range_l1936_193603

theorem solution_range (b : ℝ) : 
  let f := fun x : ℝ => x^2 - b*x - 5
  (f (-2) = 5) → 
  (f (-1) = -1) → 
  (f 4 = -1) → 
  (f 5 = 5) → 
  ∀ x : ℝ, f x = 0 → ((-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l1936_193603


namespace NUMINAMATH_CALUDE_car_speed_comparison_l1936_193608

/-- Given two cars A and B that travel the same distance, where:
    - Car A travels 1/3 of the distance at u mph, 1/3 at v mph, and 1/3 at w mph
    - Car B travels 1/3 of the time at u mph, 1/3 at v mph, and 1/3 at w mph
    - Average speed of Car A is x mph
    - Average speed of Car B is y mph
    This theorem proves that the average speed of Car A is less than or equal to the average speed of Car B. -/
theorem car_speed_comparison 
  (u v w : ℝ) 
  (hu : u > 0) (hv : v > 0) (hw : w > 0) 
  (x y : ℝ) 
  (hx : x = 3 / (1/u + 1/v + 1/w)) 
  (hy : y = (u + v + w) / 3) : 
  x ≤ y := by
sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l1936_193608


namespace NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l1936_193635

def alice_number : ℕ := 60

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n → p ∣ m)

theorem smallest_number_with_same_prime_factors :
  ∃ (bob_number : ℕ), 
    has_all_prime_factors alice_number bob_number ∧
    ∀ (m : ℕ), has_all_prime_factors alice_number m → bob_number ≤ m ∧
    bob_number = 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l1936_193635


namespace NUMINAMATH_CALUDE_last_released_theorem_l1936_193639

/-- The position of the last released captive's servant -/
def last_released_position (N : ℕ) (total_purses : ℕ) : Set ℕ :=
  if total_purses = N + (N - 1) * N / 2
  then {N}
  else if total_purses = N + (N - 1) * N / 2 - 1
  then {N - 1, N}
  else ∅

/-- The main theorem about the position of the last released captive's servant -/
theorem last_released_theorem (N : ℕ) (total_purses : ℕ) 
  (h1 : N > 0) 
  (h2 : total_purses ≥ N) 
  (h3 : total_purses ≤ N + (N - 1) * N / 2) :
  (last_released_position N total_purses).Nonempty := by
  sorry

end NUMINAMATH_CALUDE_last_released_theorem_l1936_193639


namespace NUMINAMATH_CALUDE_train_speed_problem_l1936_193656

theorem train_speed_problem (distance : ℝ) (time : ℝ) (speed_A : ℝ) : 
  distance = 480 →
  time = 2.5 →
  speed_A = 102 →
  (distance / time) - speed_A = 90 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1936_193656


namespace NUMINAMATH_CALUDE_power_function_symmetry_l1936_193646

/-- A function f is a power function if it can be written as f(x) = kx^n for some constant k and real number n. -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k n : ℝ), ∀ x, f x = k * x ^ n

/-- A function f is symmetric about the y-axis if f(x) = f(-x) for all x in the domain of f. -/
def isSymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The main theorem stating the properties of the function f(x) = (2m^2 - m)x^(2m+3) -/
theorem power_function_symmetry (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (2 * m^2 - m) * x^(2*m + 3)
  isPowerFunction f ∧ isSymmetricAboutYAxis f →
  (m = -1/2) ∧
  (∀ a : ℝ, 3/2 < a ∧ a < 2 ↔ (a - 1)^m < (2*a - 3)^m) :=
by sorry

end NUMINAMATH_CALUDE_power_function_symmetry_l1936_193646


namespace NUMINAMATH_CALUDE_remainder_55_pow_55_plus_15_mod_8_l1936_193617

theorem remainder_55_pow_55_plus_15_mod_8 : (55^55 + 15) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_55_pow_55_plus_15_mod_8_l1936_193617


namespace NUMINAMATH_CALUDE_initial_salt_concentration_l1936_193676

/-- Given a salt solution that is diluted, proves the initial salt concentration --/
theorem initial_salt_concentration
  (initial_volume : ℝ)
  (water_added : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 90)
  (h2 : water_added = 30)
  (h3 : final_concentration = 0.15)
  : ∃ (initial_concentration : ℝ),
    initial_concentration * initial_volume = 
    final_concentration * (initial_volume + water_added) ∧
    initial_concentration = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_initial_salt_concentration_l1936_193676


namespace NUMINAMATH_CALUDE_apple_orange_pricing_l1936_193601

/-- The price of an orange in dollars -/
def orange_price : ℝ := 2

/-- The price of an apple in dollars -/
def apple_price : ℝ := 3 * orange_price

theorem apple_orange_pricing :
  (4 * apple_price + 7 * orange_price = 38) →
  (orange_price = 2 ∧ 5 * apple_price = 30) := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_pricing_l1936_193601


namespace NUMINAMATH_CALUDE_joan_sofa_cost_l1936_193616

theorem joan_sofa_cost (joan_cost karl_cost : ℝ) 
  (sum_condition : joan_cost + karl_cost = 600)
  (price_relation : 2 * joan_cost = karl_cost + 90) : 
  joan_cost = 230 := by
sorry

end NUMINAMATH_CALUDE_joan_sofa_cost_l1936_193616


namespace NUMINAMATH_CALUDE_division_problem_l1936_193647

theorem division_problem (n : ℚ) : n / 4 = 12 → n / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1936_193647


namespace NUMINAMATH_CALUDE_sine_inequality_range_l1936_193634

theorem sine_inequality_range (a : ℝ) : 
  (∃ x : ℝ, Real.sin x < a) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_range_l1936_193634


namespace NUMINAMATH_CALUDE_max_sum_product_l1936_193664

theorem max_sum_product (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 → 
  a ≥ 50 → 
  a * b + b * c + c * d + d * a ≤ 5000 := by
sorry

end NUMINAMATH_CALUDE_max_sum_product_l1936_193664


namespace NUMINAMATH_CALUDE_chocolate_eating_impossibility_l1936_193637

/-- Proves that it's impossible to eat enough of the remaining chocolates to reach 3/2 of all chocolates eaten --/
theorem chocolate_eating_impossibility (total : ℕ) (initial_percent : ℚ) : 
  total = 10000 →
  initial_percent = 1/5 →
  ¬∃ (remaining_percent : ℚ), 
    0 ≤ remaining_percent ∧ 
    remaining_percent ≤ 1 ∧
    (initial_percent * total + remaining_percent * (total - initial_percent * total) : ℚ) = 3/2 * total := by
  sorry


end NUMINAMATH_CALUDE_chocolate_eating_impossibility_l1936_193637


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l1936_193621

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 5 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Theorem for part 1
theorem intersection_and_union_when_a_is_neg_one :
  (A ∩ B (-1)) = {x | -2 ≤ x ∧ x ≤ -1} ∧
  (A ∪ B (-1)) = {x | x ≤ 1 ∨ x ≥ 5} := by sorry

-- Theorem for part 2
theorem intersection_equals_B_iff :
  ∀ a : ℝ, (A ∩ B a = B a) ↔ (a > 2 ∨ a ≤ -3) := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l1936_193621


namespace NUMINAMATH_CALUDE_campaign_donation_proof_l1936_193691

theorem campaign_donation_proof (max_donors : ℕ) (half_donors : ℕ) (total_raised : ℚ) 
  (h1 : max_donors = 500)
  (h2 : half_donors = 3 * max_donors)
  (h3 : total_raised = 3750000)
  (h4 : (max_donors * x + half_donors * (x / 2)) / total_raised = 2 / 5) :
  x = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_campaign_donation_proof_l1936_193691


namespace NUMINAMATH_CALUDE_product_scaled_down_l1936_193672

theorem product_scaled_down (h : 213 * 16 = 3408) : 0.016 * 2.13 = 0.03408 := by
  sorry

end NUMINAMATH_CALUDE_product_scaled_down_l1936_193672


namespace NUMINAMATH_CALUDE_distance_to_work_l1936_193680

-- Define the problem parameters
def speed_to_work : ℝ := 45
def speed_from_work : ℝ := 30
def total_commute_time : ℝ := 1

-- Define the theorem
theorem distance_to_work :
  ∃ (d : ℝ), d / speed_to_work + d / speed_from_work = total_commute_time ∧ d = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_distance_to_work_l1936_193680


namespace NUMINAMATH_CALUDE_smallest_integer_m_l1936_193622

theorem smallest_integer_m (x y m : ℝ) : 
  (3 * x + y = m + 8) → 
  (2 * x + 2 * y = 2 * m + 5) → 
  (x - y < 1) → 
  (∀ k : ℤ, k ≥ m → k ≥ 3) ∧ (3 : ℝ) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_m_l1936_193622


namespace NUMINAMATH_CALUDE_remaining_students_average_l1936_193661

theorem remaining_students_average (total_students : ℕ) (class_average : ℚ)
  (group1_fraction : ℚ) (group1_average : ℚ)
  (group2_fraction : ℚ) (group2_average : ℚ)
  (group3_fraction : ℚ) (group3_average : ℚ)
  (h1 : total_students = 120)
  (h2 : class_average = 84)
  (h3 : group1_fraction = 1/4)
  (h4 : group1_average = 96)
  (h5 : group2_fraction = 1/5)
  (h6 : group2_average = 75)
  (h7 : group3_fraction = 1/8)
  (h8 : group3_average = 90) :
  let remaining_students := total_students - (group1_fraction * total_students + group2_fraction * total_students + group3_fraction * total_students)
  let remaining_average := (total_students * class_average - (group1_fraction * total_students * group1_average + group2_fraction * total_students * group2_average + group3_fraction * total_students * group3_average)) / remaining_students
  remaining_average = 4050 / 51 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_average_l1936_193661


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1936_193694

/-- Given a point P with coordinates (-2, 3), prove that the coordinates of the point symmetric to the origin with respect to P are (2, -3). -/
theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := (-2, 3)
  let symmetric_point := (-P.1, -P.2)
  symmetric_point = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1936_193694


namespace NUMINAMATH_CALUDE_line_equation_l1936_193644

/-- A line passing through a point with given intercepts -/
structure Line where
  point : ℝ × ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfies_equation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Check if an equation represents the given line -/
def is_equation_of_line (l : Line) (eq : LineEquation) : Prop :=
  satisfies_equation l.point eq ∧
  (eq.a ≠ 0 → satisfies_equation (l.x_intercept, 0) eq) ∧
  (eq.b ≠ 0 → satisfies_equation (0, l.y_intercept) eq)

/-- The main theorem -/
theorem line_equation (l : Line) 
    (h1 : l.point = (1, 2))
    (h2 : l.x_intercept = 2 * l.y_intercept) :
  (is_equation_of_line l ⟨2, -1, 0⟩) ∨ 
  (is_equation_of_line l ⟨1, 2, -5⟩) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l1936_193644


namespace NUMINAMATH_CALUDE_lowest_possible_score_l1936_193636

def exam_count : ℕ := 4
def max_score : ℕ := 100
def first_exam_score : ℕ := 84
def second_exam_score : ℕ := 67
def target_average : ℕ := 75

theorem lowest_possible_score :
  ∃ (third_exam_score fourth_exam_score : ℕ),
    third_exam_score ≤ max_score ∧
    fourth_exam_score ≤ max_score ∧
    (first_exam_score + second_exam_score + third_exam_score + fourth_exam_score) / exam_count ≥ target_average ∧
    (third_exam_score = 49 ∨ fourth_exam_score = 49) ∧
    ∀ (x y : ℕ),
      x ≤ max_score →
      y ≤ max_score →
      (first_exam_score + second_exam_score + x + y) / exam_count ≥ target_average →
      x ≥ 49 ∧ y ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l1936_193636


namespace NUMINAMATH_CALUDE_real_roots_condition_l1936_193658

theorem real_roots_condition (a b : ℝ) : 
  (∃ x : ℝ, (1 - a * x) / (1 + a * x) * Real.sqrt ((1 + b * x) / (1 - b * x)) = 1) ↔ 
  (1 / 2 < a / b ∧ a / b < 1) :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_l1936_193658


namespace NUMINAMATH_CALUDE_cosine_min_phase_l1936_193650

/-- Given a cosine function y = a cos(bx + c) where a, b, and c are positive constants,
    if the function reaches its first minimum at x = π/(2b), then c = π/2. -/
theorem cosine_min_phase (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x : ℝ, x ≥ 0 → a * Real.cos (b * x + c) ≥ a * Real.cos (b * (Real.pi / (2 * b)) + c)) →
  c = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_min_phase_l1936_193650


namespace NUMINAMATH_CALUDE_diophantine_approximation_l1936_193604

theorem diophantine_approximation (x : ℝ) (h_irr : Irrational x) (h_pos : x > 0) :
  ∀ n : ℕ, ∃ p q : ℤ, q > n ∧ q > 0 ∧ |x - (p : ℝ) / q| ≤ 1 / q^2 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_approximation_l1936_193604


namespace NUMINAMATH_CALUDE_w_plus_reciprocal_w_traces_ellipse_l1936_193645

theorem w_plus_reciprocal_w_traces_ellipse :
  ∀ (w : ℂ) (x y : ℝ),
  (Complex.abs w = 3) →
  (w + w⁻¹ = x + y * Complex.I) →
  ∃ (a b : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ∧ a ≠ b ∧ a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_w_plus_reciprocal_w_traces_ellipse_l1936_193645


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_in_U_l1936_193696

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_of_A_union_B_in_U :
  (U \ (A ∪ B)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_in_U_l1936_193696


namespace NUMINAMATH_CALUDE_unique_eventually_one_l1936_193631

def f (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else 3 * n

def eventually_one (n : ℕ) : Prop :=
  ∃ k : ℕ, (f^[k] n) = 1

theorem unique_eventually_one :
  ∃! n : ℕ, n ∈ Finset.range 200 ∧ eventually_one n :=
sorry

end NUMINAMATH_CALUDE_unique_eventually_one_l1936_193631


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l1936_193659

theorem triangle_sine_inequality (A B C : Real) (h_triangle : A + B + C = π) :
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l1936_193659


namespace NUMINAMATH_CALUDE_unique_reverse_difference_l1936_193651

/-- Reverses the digits of a 4-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

/-- Checks if a number is a 4-digit number -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_reverse_difference :
  ∃! n : ℕ, isFourDigit n ∧ reverseDigits n = n + 8802 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_reverse_difference_l1936_193651


namespace NUMINAMATH_CALUDE_parallel_sides_implies_parallelogram_l1936_193648

/-- A quadrilateral is defined as a polygon with four sides -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- Parallel sides in a quadrilateral -/
def parallel_sides (q : Quadrilateral) (side1 side2 : Fin 4) : Prop :=
  -- Definition of parallel sides omitted for brevity
  sorry

/-- A parallelogram is a quadrilateral with both pairs of opposite sides parallel -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  parallel_sides q 0 2 ∧ parallel_sides q 1 3

/-- Theorem: If both pairs of opposite sides of a quadrilateral are parallel, then it is a parallelogram -/
theorem parallel_sides_implies_parallelogram (q : Quadrilateral) :
  (parallel_sides q 0 2 ∧ parallel_sides q 1 3) → is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_parallel_sides_implies_parallelogram_l1936_193648


namespace NUMINAMATH_CALUDE_volunteers_needed_l1936_193665

/-- Represents the number of volunteers needed for the school Christmas play --/
def total_volunteers_needed : ℕ := 100

/-- Represents the number of math classes --/
def math_classes : ℕ := 5

/-- Represents the number of students volunteering from each math class --/
def students_per_class : ℕ := 4

/-- Represents the total number of teachers volunteering --/
def teachers_volunteering : ℕ := 10

/-- Represents the number of teachers skilled in carpentry --/
def teachers_carpentry : ℕ := 3

/-- Represents the total number of parents volunteering --/
def parents_volunteering : ℕ := 15

/-- Represents the number of parents experienced with lighting and sound --/
def parents_lighting_sound : ℕ := 6

/-- Represents the additional number of volunteers needed with carpentry skills --/
def additional_carpentry_needed : ℕ := 8

/-- Represents the additional number of volunteers needed with lighting and sound experience --/
def additional_lighting_sound_needed : ℕ := 10

/-- Theorem stating that 9 more volunteers are needed to meet the requirements --/
theorem volunteers_needed : 
  (math_classes * students_per_class + teachers_volunteering + parents_volunteering) +
  (additional_carpentry_needed - teachers_carpentry) + 
  (additional_lighting_sound_needed - parents_lighting_sound) = 9 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_needed_l1936_193665


namespace NUMINAMATH_CALUDE_curve_C_not_centrally_symmetric_l1936_193652

-- Define the curve C
def C : ℝ → ℝ := fun x ↦ x^3 - x + 2

-- Theorem statement
theorem curve_C_not_centrally_symmetric :
  ∀ (a b : ℝ), ¬(∀ (x y : ℝ), C x = y → C (2*a - x) = 2*b - y) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_not_centrally_symmetric_l1936_193652


namespace NUMINAMATH_CALUDE_smallest_coin_set_l1936_193630

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin in cents --/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A function that checks if a given set of coins can make all amounts from 1 to 99 cents --/
def canMakeAllAmounts (coins : List Coin) : Prop :=
  ∀ (amount : Nat), 1 ≤ amount ∧ amount ≤ 99 →
    ∃ (subset : List Coin), subset ⊆ coins ∧ (subset.map coinValue).sum = amount

/-- The theorem stating that 6 is the smallest number of coins needed --/
theorem smallest_coin_set :
  ∃ (coins : List Coin),
    coins.length = 6 ∧
    canMakeAllAmounts coins ∧
    ∀ (other_coins : List Coin),
      canMakeAllAmounts other_coins →
      other_coins.length ≥ 6 :=
by sorry

#check smallest_coin_set

end NUMINAMATH_CALUDE_smallest_coin_set_l1936_193630
