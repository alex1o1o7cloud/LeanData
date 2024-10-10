import Mathlib

namespace no_solution_system_l640_64016

theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 18) := by
  sorry

end no_solution_system_l640_64016


namespace complex_equation_solution_l640_64040

theorem complex_equation_solution (x : ℂ) : x / Complex.I = 1 - Complex.I → x = 1 + Complex.I := by
  sorry

end complex_equation_solution_l640_64040


namespace distinct_words_count_l640_64087

def first_digit (n : ℕ) : ℕ := 
  -- Definition of the function that returns the first digit of 2^n
  sorry

def word_sequence (start : ℕ) : List ℕ := 
  -- Definition of a list of 13 consecutive terms starting from 'start'
  (List.range 13).map (λ i => first_digit (start + i))

def distinct_words : Finset (List ℕ) :=
  -- Set of all distinct words in the sequence
  sorry

theorem distinct_words_count : Finset.card distinct_words = 57 := by
  sorry

end distinct_words_count_l640_64087


namespace two_item_combinations_l640_64045

theorem two_item_combinations (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end two_item_combinations_l640_64045


namespace total_chips_is_135_l640_64082

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

end total_chips_is_135_l640_64082


namespace collinear_points_b_value_l640_64065

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- 
If the points (4, -6), (3b - 1, 5), and (b + 4, 4) are collinear, then b = 50/19.
-/
theorem collinear_points_b_value :
  ∀ b : ℝ, collinear 4 (-6) (3*b - 1) 5 (b + 4) 4 → b = 50/19 := by
sorry

end collinear_points_b_value_l640_64065


namespace area_not_perfect_square_l640_64053

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

end area_not_perfect_square_l640_64053


namespace prob_both_genders_selected_l640_64000

def total_students : ℕ := 8
def male_students : ℕ := 5
def female_students : ℕ := 3
def students_to_select : ℕ := 5

theorem prob_both_genders_selected :
  (Nat.choose total_students students_to_select - Nat.choose male_students students_to_select) /
  Nat.choose total_students students_to_select = 55 / 56 :=
by sorry

end prob_both_genders_selected_l640_64000


namespace pig_count_after_joining_l640_64001

theorem pig_count_after_joining (initial_pigs joining_pigs : ℕ) :
  initial_pigs = 64 →
  joining_pigs = 22 →
  initial_pigs + joining_pigs = 86 :=
by
  sorry

end pig_count_after_joining_l640_64001


namespace ball_bounce_distance_l640_64020

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The specific problem of a ball dropped from 120 feet with 1/3 rebound ratio -/
theorem ball_bounce_distance :
  totalDistance 120 (1/3) 5 = 248 + 26/27 := by
  sorry

end ball_bounce_distance_l640_64020


namespace orchid_bushes_after_planting_l640_64073

/-- The number of orchid bushes in the park after planting -/
def total_orchid_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: Given 22 initial orchid bushes and 13 newly planted orchid bushes,
    the total number of orchid bushes after planting will be 35. -/
theorem orchid_bushes_after_planting :
  total_orchid_bushes 22 13 = 35 := by
  sorry

end orchid_bushes_after_planting_l640_64073


namespace lionel_distance_walked_l640_64075

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

end lionel_distance_walked_l640_64075


namespace nth_equation_pattern_l640_64067

theorem nth_equation_pattern (n : ℕ+) : n^2 - n = n * (n - 1) := by
  sorry

end nth_equation_pattern_l640_64067


namespace negative_quadratic_symmetry_implies_inequality_l640_64008

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

end negative_quadratic_symmetry_implies_inequality_l640_64008


namespace digit_product_property_l640_64046

theorem digit_product_property :
  ∃! (count : Nat), count = (Finset.filter 
    (fun pair : Nat × Nat => 
      let x := pair.1
      let y := pair.2
      x ≠ y ∧ 
      x < 10 ∧ 
      y < 10 ∧ 
      1000 ≤ x * 1111 ∧ 
      x * 1111 < 10000 ∧
      1000 ≤ y * 1111 ∧ 
      y * 1111 < 10000 ∧
      1000000 ≤ (x * 1111) * (y * 1111) ∧ 
      (x * 1111) * (y * 1111) < 10000000 ∧
      (x * 1111) * (y * 1111) % 10 = x ∧
      ((x * 1111) * (y * 1111) / 1000000) % 10 = x)
    (Finset.product (Finset.range 10) (Finset.range 10))).card ∧
  count = 3 := by
sorry

end digit_product_property_l640_64046


namespace minimum_trees_l640_64051

theorem minimum_trees (L : ℕ) (X : ℕ) : 
  (∀ n < L, ¬ ∃ m : ℕ, (0.13 : ℝ) * n < m ∧ m < (0.14 : ℝ) * n) →
  ((0.13 : ℝ) * L < X ∧ X < (0.14 : ℝ) * L) →
  L = 15 := by
sorry

end minimum_trees_l640_64051


namespace perpendicular_segments_in_cube_l640_64038

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

end perpendicular_segments_in_cube_l640_64038


namespace sean_cricket_theorem_l640_64026

def sean_cricket_problem (total_days : ℕ) (total_minutes : ℕ) (indira_minutes : ℕ) : Prop :=
  let sean_total_minutes := total_minutes - indira_minutes
  sean_total_minutes / total_days = 50

theorem sean_cricket_theorem :
  sean_cricket_problem 14 1512 812 := by
  sorry

end sean_cricket_theorem_l640_64026


namespace science_fiction_books_l640_64013

/-- Represents the number of books in the science fiction section of a library. -/
def num_books : ℕ := 3824 / 478

/-- Theorem stating that the number of books in the science fiction section is 8. -/
theorem science_fiction_books : num_books = 8 := by
  sorry

end science_fiction_books_l640_64013


namespace nonagon_diagonal_count_l640_64035

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonal_count : nonagon_diagonals = (nonagon_sides * (nonagon_sides - 3)) / 2 := by
  sorry

end nonagon_diagonal_count_l640_64035


namespace complex_real_part_twice_imaginary_l640_64092

theorem complex_real_part_twice_imaginary (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (Complex.re z = 2 * Complex.im z) → a = 2 := by
sorry

end complex_real_part_twice_imaginary_l640_64092


namespace sun_overhead_locations_sun_angle_locations_l640_64025

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

end sun_overhead_locations_sun_angle_locations_l640_64025


namespace workshop_probability_l640_64062

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

end workshop_probability_l640_64062


namespace three_solutions_sum_and_m_value_m_range_for_positive_f_l640_64002

noncomputable section

def f (m : ℝ) (x : ℝ) := 4 - m * Real.sin x - 3 * (Real.cos x)^2

theorem three_solutions_sum_and_m_value 
  (m : ℝ) 
  (h₁ : ∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < π ∧ 
                       0 < x₂ ∧ x₂ < π ∧ 
                       0 < x₃ ∧ x₃ < π ∧ 
                       x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
                       f m x₁ = 0 ∧ f m x₂ = 0 ∧ f m x₃ = 0) : 
  m = 4 ∧ ∃ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = 3 * π / 2 :=
sorry

theorem m_range_for_positive_f 
  (m : ℝ) 
  (h : ∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π → f m x > 0) : 
  -7/2 < m ∧ m < 2 * Real.sqrt 3 :=
sorry

end three_solutions_sum_and_m_value_m_range_for_positive_f_l640_64002


namespace max_backyard_area_l640_64028

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

end max_backyard_area_l640_64028


namespace union_and_complement_find_a_l640_64027

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

end union_and_complement_find_a_l640_64027


namespace lunks_for_two_dozen_bananas_l640_64083

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

end lunks_for_two_dozen_bananas_l640_64083


namespace positive_expression_l640_64054

theorem positive_expression (x : ℝ) (h : x > 0) : x^2 + π*x + (15*π/2)*Real.sin x > 0 := by
  sorry

end positive_expression_l640_64054


namespace max_stores_visited_l640_64036

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ)
  (two_store_visitors : ℕ) (three_store_visitors : ℕ) (four_store_visitors : ℕ)
  (h1 : total_stores = 15)
  (h2 : total_visits = 60)
  (h3 : total_shoppers = 30)
  (h4 : two_store_visitors = 12)
  (h5 : three_store_visitors = 6)
  (h6 : four_store_visitors = 4)
  (h7 : two_store_visitors * 2 + three_store_visitors * 3 + four_store_visitors * 4 < total_visits)
  (h8 : ∀ n : ℕ, n ≤ total_shoppers → n > 0) :
  ∃ (max_visited : ℕ), max_visited = 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visited :=
sorry

end max_stores_visited_l640_64036


namespace max_a_value_l640_64024

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := (x - t) * abs x

-- State the theorem
theorem max_a_value (t : ℝ) (h : t ∈ Set.Ioo 0 2) :
  (∃ a : ℝ, ∀ x ∈ Set.Icc (-1) 2, f t x > x + a) →
  (∃ a : ℝ, (∀ x ∈ Set.Icc (-1) 2, f t x > x + a) ∧ a = -1/4) :=
by sorry

end max_a_value_l640_64024


namespace two_digit_number_with_specific_division_properties_l640_64084

theorem two_digit_number_with_specific_division_properties :
  ∀ n : ℕ,
  (n ≥ 10 ∧ n ≤ 99) →
  (n % 6 = n / 10) →
  (n / 10 = 3 ∧ n % 10 = n % 10) →
  (n = 33 ∨ n = 39) :=
by sorry

end two_digit_number_with_specific_division_properties_l640_64084


namespace target_number_is_294_l640_64042

/-- Represents the list of numbers starting with digit 2 in increasing order -/
def digit2List : List ℕ := sorry

/-- Returns the nth digit in the concatenated representation of digit2List -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1498th, 1499th, and 1500th digits -/
def targetNumber : ℕ := 100 * (nthDigit 1498) + 10 * (nthDigit 1499) + (nthDigit 1500)

theorem target_number_is_294 : targetNumber = 294 := by sorry

end target_number_is_294_l640_64042


namespace smallest_n_for_fraction_l640_64060

def fraction (n : ℕ) : ℚ :=
  (5^(n+1) + 2^(n+1)) / (5^n + 2^n)

theorem smallest_n_for_fraction :
  (∀ k : ℕ, k < 7 → fraction k ≤ 4.99) ∧
  fraction 7 > 4.99 :=
sorry

end smallest_n_for_fraction_l640_64060


namespace smallest_three_digit_congruent_to_one_mod_37_l640_64091

theorem smallest_three_digit_congruent_to_one_mod_37 : 
  ∃ n : ℕ, 
    (100 ≤ n ∧ n ≤ 999) ∧ 
    n % 37 = 1 ∧ 
    (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) ∧ m % 37 = 1 → n ≤ m) ∧
    n = 112 := by
  sorry

end smallest_three_digit_congruent_to_one_mod_37_l640_64091


namespace adult_ticket_price_l640_64063

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

end adult_ticket_price_l640_64063


namespace sector_angle_l640_64071

/-- Theorem: For a circular sector with perimeter 4 cm and area 1 cm², 
    the radian measure of its central angle is 2 radians. -/
theorem sector_angle (r : ℝ) (α : ℝ) 
  (h_perimeter : 2 * r + r * α = 4)
  (h_area : 1/2 * α * r^2 = 1) : 
  α = 2 := by
  sorry

end sector_angle_l640_64071


namespace salt_solution_mixture_l640_64007

/-- 
Given a mixture of 1 liter of pure water and x liters of 30% salt solution,
resulting in a 15% salt solution, prove that x = 1.
-/
theorem salt_solution_mixture (x : ℝ) : 
  (0.30 * x = 0.15 * (1 + x)) → x = 1 := by
  sorry

end salt_solution_mixture_l640_64007


namespace shelby_buys_three_posters_l640_64010

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

end shelby_buys_three_posters_l640_64010


namespace smallest_number_divisible_by_all_l640_64043

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 9 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 3147 ∧ ∀ m < 3147, ¬is_divisible_by_all m :=
by sorry

end smallest_number_divisible_by_all_l640_64043


namespace sin_thirteen_pi_fourths_l640_64086

theorem sin_thirteen_pi_fourths : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end sin_thirteen_pi_fourths_l640_64086


namespace johns_phd_time_l640_64070

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


end johns_phd_time_l640_64070


namespace gcd_12012_18018_l640_64004

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end gcd_12012_18018_l640_64004


namespace cyclist_speed_problem_l640_64080

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

end cyclist_speed_problem_l640_64080


namespace equation_roots_l640_64074

theorem equation_roots : ∃ (x₁ x₂ : ℝ), 
  x₁ ≠ x₂ ∧ 
  x₁ = (29 + Real.sqrt 457) / 24 ∧ 
  x₂ = (29 - Real.sqrt 457) / 24 ∧ 
  ∀ x : ℝ, x ≠ 2 → 
    3 * x^2 / (x - 2) - (x + 4) / 4 + (7 - 9 * x) / (x - 2) + 2 = 0 ↔ 
    (x = x₁ ∨ x = x₂) := by
  sorry

end equation_roots_l640_64074


namespace number_with_special_average_l640_64012

theorem number_with_special_average (x : ℝ) (h1 : x ≠ 0) 
  (h2 : (x + x^2) / 2 = 5 * x) : x = 9 := by
  sorry

end number_with_special_average_l640_64012


namespace bus_stoppage_time_l640_64005

/-- Given a bus with speeds excluding and including stoppages, 
    calculate the number of minutes the bus stops per hour -/
theorem bus_stoppage_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 48)
  (h2 : speed_with_stops = 12) :
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 45 := by
  sorry

end bus_stoppage_time_l640_64005


namespace geometric_series_ratio_l640_64052

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a * r^5 / (1 - r)) / (a / (1 - r)) = 1 / 81 → r = 1 / 3 := by
  sorry

end geometric_series_ratio_l640_64052


namespace probability_three_tails_one_head_l640_64066

theorem probability_three_tails_one_head : 
  let n : ℕ := 4 -- number of coins
  let k : ℕ := 3 -- number of tails (or heads, whichever is larger)
  let p : ℚ := 1/2 -- probability of getting tails (or heads) for a single coin
  (n.choose k) * p^k * (1 - p)^(n - k) = 1/4 :=
by sorry

end probability_three_tails_one_head_l640_64066


namespace expand_expression_l640_64009

theorem expand_expression (x : ℝ) : (7*x - 3) * 5*x^2 = 35*x^3 - 15*x^2 := by
  sorry

end expand_expression_l640_64009


namespace model_c_sample_size_l640_64059

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


end model_c_sample_size_l640_64059


namespace fraction_simplification_l640_64077

theorem fraction_simplification :
  1 / (1 / (1/2)^2 + 1 / (1/2)^3 + 1 / (1/2)^4 + 1 / (1/2)^5) = 1 / 60 := by
  sorry

end fraction_simplification_l640_64077


namespace solution_set_of_inequality_l640_64041

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_of_inequality
  (h_even : ∀ x, f (-x) = f x)  -- f is even
  (h_derivative : ∀ x, HasDerivAt f (f' x) x)  -- f' is the derivative of f
  (h_condition : ∀ x, x < 0 → x * f' x - f x > 0)  -- condition for x < 0
  (h_f_1 : f 1 = 0)  -- f(1) = 0
  : {x : ℝ | f x / x < 0} = {x | x < -1 ∨ (0 < x ∧ x < 1)} :=
sorry

end solution_set_of_inequality_l640_64041


namespace min_value_zero_implies_t_l640_64094

/-- The function f(x) defined in the problem -/
def f (t : ℝ) (x : ℝ) : ℝ := 4 * x^4 - 6 * t * x^3 + (2 * t + 6) * x^2 - 3 * t * x + 1

/-- The theorem statement -/
theorem min_value_zero_implies_t (t : ℝ) :
  (∀ x > 0, f t x ≥ 0) ∧ 
  (∃ x > 0, f t x = 0) →
  t = 2 * Real.sqrt 2 :=
sorry

end min_value_zero_implies_t_l640_64094


namespace triangle_area_l640_64056

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is √3/2 when c = √2, b = √6, and B = 120°. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  b = Real.sqrt 6 →
  B = 2 * π / 3 →  -- 120° in radians
  (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
by sorry

end triangle_area_l640_64056


namespace three_heads_probability_l640_64069

-- Define a fair coin
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of three heads in a row
def three_heads_prob : ℚ := fair_coin_prob * fair_coin_prob * fair_coin_prob

-- Theorem statement
theorem three_heads_probability :
  three_heads_prob = 1 / 8 := by sorry

end three_heads_probability_l640_64069


namespace unique_number_with_nine_divisors_and_special_property_l640_64095

def has_exactly_nine_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 9

theorem unique_number_with_nine_divisors_and_special_property :
  ∃! n : ℕ, has_exactly_nine_divisors n ∧
  ∃ (a b c : ℕ), a ∣ n ∧ b ∣ n ∧ c ∣ n ∧
  a + b + c = 79 ∧ a * a = b * c :=
by
  use 441
  sorry

end unique_number_with_nine_divisors_and_special_property_l640_64095


namespace reading_ratio_is_two_l640_64011

/-- The minimum number of pages assigned for reading -/
def min_assigned : ℕ := 25

/-- The number of extra pages Harrison read -/
def harrison_extra : ℕ := 10

/-- The number of extra pages Pam read compared to Harrison -/
def pam_extra : ℕ := 15

/-- The number of pages Sam read -/
def sam_pages : ℕ := 100

/-- Calculate the number of pages Harrison read -/
def harrison_pages : ℕ := min_assigned + harrison_extra

/-- Calculate the number of pages Pam read -/
def pam_pages : ℕ := harrison_pages + pam_extra

/-- The ratio of pages Sam read to pages Pam read -/
def reading_ratio : ℚ := sam_pages / pam_pages

theorem reading_ratio_is_two : reading_ratio = 2 := by
  sorry

end reading_ratio_is_two_l640_64011


namespace distribute_5_4_l640_64090

/-- The number of ways to distribute n distinct objects into k identical containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 4 identical containers,
    allowing empty containers, is 37. -/
theorem distribute_5_4 : distribute 5 4 = 37 := by sorry

end distribute_5_4_l640_64090


namespace max_value_of_a_minus_b_squared_l640_64048

theorem max_value_of_a_minus_b_squared (a b : ℝ) (h : a^2 + b^2 = 4) :
  (∀ x y : ℝ, x^2 + y^2 = 4 → (x - y)^2 ≤ 8) ∧ 
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ (x - y)^2 = 8) :=
by sorry

end max_value_of_a_minus_b_squared_l640_64048


namespace range_of_f_l640_64003

def f (x : ℝ) : ℝ := 2 * x - 1

theorem range_of_f : 
  ∀ y ∈ Set.Icc (-1 : ℝ) 3, ∃ x ∈ Set.Icc 0 2, f x = y ∧
  ∀ x ∈ Set.Icc 0 2, f x ∈ Set.Icc (-1 : ℝ) 3 := by
sorry

end range_of_f_l640_64003


namespace divisibility_problem_l640_64085

theorem divisibility_problem (n : ℕ) (h : ∀ a : ℕ, a < 60 → ¬(n ∣ a^3)) : n = 216000 := by
  sorry

end divisibility_problem_l640_64085


namespace max_value_sqrt_sum_l640_64064

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  (∀ y, -49 ≤ y ∧ y ≤ 49 → Real.sqrt (49 + y) + Real.sqrt (49 - y) ≤ Real.sqrt (49 + x) + Real.sqrt (49 - x)) →
  Real.sqrt (49 + x) + Real.sqrt (49 - x) = 14 :=
by sorry

end max_value_sqrt_sum_l640_64064


namespace no_linear_term_implies_m_eq_neg_eight_l640_64044

-- Define the polynomial expression
def poly (x m : ℝ) : ℝ := (x^2 - x + m) * (x - 8)

-- Theorem statement
theorem no_linear_term_implies_m_eq_neg_eight :
  (∀ x : ℝ, ∃ a b c : ℝ, poly x m = a * x^3 + b * x^2 + c) → m = -8 :=
by sorry

end no_linear_term_implies_m_eq_neg_eight_l640_64044


namespace klinker_daughter_age_l640_64055

/-- Proves that given Mr. Klinker is 35 years old and in 15 years he will be twice as old as his daughter, his daughter's current age is 10 years. -/
theorem klinker_daughter_age (klinker_age : ℕ) (daughter_age : ℕ) : 
  klinker_age = 35 →
  klinker_age + 15 = 2 * (daughter_age + 15) →
  daughter_age = 10 := by
sorry

end klinker_daughter_age_l640_64055


namespace algebraic_operation_proof_l640_64078

theorem algebraic_operation_proof (a b : ℝ) : 5 * a * b - 6 * a * b = -a * b := by
  sorry

end algebraic_operation_proof_l640_64078


namespace binomial_60_2_l640_64068

theorem binomial_60_2 : Nat.choose 60 2 = 1770 := by
  sorry

end binomial_60_2_l640_64068


namespace man_son_age_difference_l640_64047

/-- Represents the age difference between a man and his son -/
def ageDifference (manAge sonAge : ℕ) : ℕ := manAge - sonAge

/-- Theorem stating the age difference between a man and his son -/
theorem man_son_age_difference :
  ∀ (manAge sonAge : ℕ),
    sonAge = 18 →
    manAge + 2 = 2 * (sonAge + 2) →
    ageDifference manAge sonAge = 20 := by
  sorry


end man_son_age_difference_l640_64047


namespace cosine_period_l640_64088

theorem cosine_period (ω : ℝ) (h1 : ω > 0) : 
  (∃ y : ℝ → ℝ, y = λ x => Real.cos (ω * x - π / 6)) →
  (π / 5 = 2 * π / ω) →
  ω = 10 := by
sorry

end cosine_period_l640_64088


namespace sin_angle_DAE_sin_angle_DAE_value_l640_64099

/-- An equilateral triangle with side length 9 -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Points D and E on side BC -/
structure PointsOnBC where
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Main theorem: sin ∠DAE in the given configuration -/
theorem sin_angle_DAE (triangle : EquilateralTriangle) (points : PointsOnBC) : ℝ :=
  sorry

/-- The value of sin ∠DAE is √3/2 -/
theorem sin_angle_DAE_value (triangle : EquilateralTriangle) (points : PointsOnBC) :
    sin_angle_DAE triangle points = Real.sqrt 3 / 2 := by
  sorry

end sin_angle_DAE_sin_angle_DAE_value_l640_64099


namespace intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l640_64019

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem 1
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ∈ Set.Icc 0 1 :=
sorry

-- Theorem 2
theorem union_equals_B_iff_a_in_range (a : ℝ) :
  A a ∪ B = B ↔ a ∈ Set.Iic (-2) ∪ Set.Ici 3 :=
sorry

end intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l640_64019


namespace daves_initial_files_l640_64015

theorem daves_initial_files (initial_apps : ℕ) (final_apps : ℕ) (final_files : ℕ) :
  initial_apps = 24 →
  final_apps = 12 →
  final_files = 5 →
  final_apps = final_files + 7 →
  initial_apps - final_apps + final_files = 17 := by
  sorry

end daves_initial_files_l640_64015


namespace sum_largest_smallest_prime_factors_546_l640_64076

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

end sum_largest_smallest_prime_factors_546_l640_64076


namespace ampersand_composition_l640_64031

-- Define the & operation for the case &=9-x
def ampersand1 (x : ℤ) : ℤ := 9 - x

-- Define the & operation for the case &x = x - 9
def ampersand2 (x : ℤ) : ℤ := x - 9

-- Theorem to prove
theorem ampersand_composition : ampersand1 (ampersand2 15) = 3 := by
  sorry

end ampersand_composition_l640_64031


namespace derivative_f_at_zero_l640_64032

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x + 1

theorem derivative_f_at_zero :
  deriv f 0 = 2 := by sorry

end derivative_f_at_zero_l640_64032


namespace reflect_F_final_coords_l640_64050

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def F : ℝ × ℝ := (1, 3)

theorem reflect_F_final_coords :
  (reflect_y_eq_x ∘ reflect_y ∘ reflect_x) F = (-3, -1) := by
  sorry

end reflect_F_final_coords_l640_64050


namespace officers_selection_count_l640_64037

/-- The number of ways to select 3 distinct individuals from a group of 6 people to fill 3 distinct positions -/
def selectOfficers (n : ℕ) : ℕ :=
  if n < 3 then 0
  else n * (n - 1) * (n - 2)

/-- Theorem stating that selecting 3 officers from 6 people results in 120 possibilities -/
theorem officers_selection_count : selectOfficers 6 = 120 := by
  sorry

end officers_selection_count_l640_64037


namespace dice_sum_probability_l640_64022

theorem dice_sum_probability : 
  let die := Finset.range 6
  let outcomes := die.product die
  let favorable_outcomes := outcomes.filter (fun (x, y) => x + y + 2 ≥ 10)
  (favorable_outcomes.card : ℚ) / outcomes.card = 1 / 6 := by
sorry

end dice_sum_probability_l640_64022


namespace range_of_a_l640_64057

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f y < f x

-- Define the theorem
theorem range_of_a (f : ℝ → ℝ) (h_decreasing : DecreasingFunction f) :
  (∀ a, f (2 * a - 1) < f (1 - a)) → 
  (∀ a, (2/3 : ℝ) < a ∧ a < 1) :=
sorry

end range_of_a_l640_64057


namespace ring_arrangements_value_l640_64021

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

end ring_arrangements_value_l640_64021


namespace largest_prime_factor_133_l640_64034

def numbers : List Nat := [45, 65, 91, 85, 133]

def largest_prime_factor (n : Nat) : Nat :=
  sorry

theorem largest_prime_factor_133 :
  ∀ m ∈ numbers, m ≠ 133 → largest_prime_factor 133 > largest_prime_factor m :=
by sorry

end largest_prime_factor_133_l640_64034


namespace show_revenue_l640_64079

def tickets_first_showing : ℕ := 200
def ticket_price : ℕ := 25

theorem show_revenue : 
  (tickets_first_showing + 3 * tickets_first_showing) * ticket_price = 20000 := by
  sorry

end show_revenue_l640_64079


namespace odot_specific_values_odot_power_relation_l640_64093

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

end odot_specific_values_odot_power_relation_l640_64093


namespace stone_piles_total_l640_64061

theorem stone_piles_total (pile1 pile2 pile3 pile4 pile5 : ℕ) : 
  pile5 = 6 * pile3 →
  pile2 = 2 * (pile3 + pile5) →
  pile1 * 3 = pile5 →
  pile1 + 10 = pile4 →
  2 * pile4 = pile2 →
  pile1 + pile2 + pile3 + pile4 + pile5 = 60 := by
  sorry

end stone_piles_total_l640_64061


namespace least_five_digit_congruent_to_11_mod_14_l640_64089

theorem least_five_digit_congruent_to_11_mod_14 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  n % 14 = 11 ∧               -- congruent to 11 (mod 14)
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 14 = 11 → m ≥ n) ∧  -- least such number
  n = 10007 :=                -- the answer is 10007
by sorry

end least_five_digit_congruent_to_11_mod_14_l640_64089


namespace solution_set_x_squared_leq_one_l640_64029

theorem solution_set_x_squared_leq_one :
  ∀ x : ℝ, x^2 ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 := by sorry

end solution_set_x_squared_leq_one_l640_64029


namespace parabola_and_line_theorem_l640_64096

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

end parabola_and_line_theorem_l640_64096


namespace course_combinations_l640_64033

def type_A_courses : ℕ := 3
def type_B_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

def combinations_with_both_types (a b k : ℕ) : ℕ :=
  Nat.choose a (k - 1) * Nat.choose b 1 + Nat.choose a 1 * Nat.choose b (k - 1)

theorem course_combinations :
  combinations_with_both_types type_A_courses type_B_courses total_courses_to_choose = 30 := by
  sorry

end course_combinations_l640_64033


namespace quadratic_equation_solution_l640_64097

theorem quadratic_equation_solution (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 7 * x - 20 = 0) : x = 4/3 := by
  sorry

end quadratic_equation_solution_l640_64097


namespace calculation_proof_l640_64023

theorem calculation_proof : (42 / (12 - 10 + 3)) ^ 2 * 7 = 493.92 := by
  sorry

end calculation_proof_l640_64023


namespace right_triangle_side_difference_l640_64014

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

end right_triangle_side_difference_l640_64014


namespace distance_from_focus_to_line_l640_64049

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the right focus
def right_focus : ℝ × ℝ := (3, 0)

-- State the theorem
theorem distance_from_focus_to_line :
  let (x₀, y₀) := right_focus
  ∃ d : ℝ, d = |x₀ + 2*y₀ - 8| / Real.sqrt (1^2 + 2^2) ∧ d = Real.sqrt 5 :=
sorry

end distance_from_focus_to_line_l640_64049


namespace base_digit_conversion_l640_64081

theorem base_digit_conversion (N : ℕ+) :
  (9^19 ≤ N ∧ N < 9^20) ∧ (27^12 ≤ N ∧ N < 27^13) →
  3^38 ≤ N ∧ N < 3^39 :=
by sorry

end base_digit_conversion_l640_64081


namespace quadratic_roots_problem_l640_64017

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x : ℝ, x^2 - (2*m - 1)*x + m^2 = 0) →  -- Equation has real roots
  (x₁^2 - (2*m - 1)*x₁ + m^2 = 0) →         -- x₁ is a root
  (x₂^2 - (2*m - 1)*x₂ + m^2 = 0) →         -- x₂ is a root
  ((x₁ + 1) * (x₂ + 1) = 3) →               -- Given condition
  (m = -3) :=                               -- Conclusion
by sorry

end quadratic_roots_problem_l640_64017


namespace geometric_sequence_a10_l640_64039

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

end geometric_sequence_a10_l640_64039


namespace arithmetic_sequence_common_difference_l640_64018

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (h1 : a 2 = 1)  -- given: a2 = 1
  (h2 : a 6 = 13)  -- given: a6 = 13
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
sorry

end arithmetic_sequence_common_difference_l640_64018


namespace b_over_a_range_l640_64030

-- Define an acute triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

-- Define the function f
def f (x a b c : ℝ) : ℝ := x^2 + c^2 - a^2 - a*b

-- State the theorem
theorem b_over_a_range (t : AcuteTriangle) 
  (h : ∃! x, f x t.a t.b t.c = 0) : 
  1 < t.b / t.a ∧ t.b / t.a < 2 := by
  sorry

end b_over_a_range_l640_64030


namespace min_value_and_inequality_solution_l640_64006

def f (x a : ℝ) : ℝ := |x - a| + |x - 1|

theorem min_value_and_inequality_solution 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : ∀ x, f x a ≥ 2) 
  (h3 : ∃ x, f x a = 2) :
  (a = 3) ∧ 
  (∀ x, f x a ≥ 4 ↔ x ∈ Set.Iic 0 ∪ Set.Ici 4) :=
sorry

end min_value_and_inequality_solution_l640_64006


namespace father_child_ages_l640_64072

theorem father_child_ages : ∃ (f b : ℕ), 
  13 ≤ b ∧ b ≤ 19 ∧ 
  100 * f + b - (f - b) = 4289 ∧ 
  f + b = 59 := by
sorry

end father_child_ages_l640_64072


namespace bakery_order_cost_is_54_l640_64098

/-- Calculates the final cost of a bakery order with a possible discount --/
def bakery_order_cost (quiche_price croissant_price biscuit_price : ℚ) 
  (quiche_quantity croissant_quantity biscuit_quantity : ℕ)
  (discount_rate : ℚ) (discount_threshold : ℚ) : ℚ :=
  let total_before_discount := 
    quiche_price * quiche_quantity + 
    croissant_price * croissant_quantity + 
    biscuit_price * biscuit_quantity
  let discount := 
    if total_before_discount > discount_threshold 
    then total_before_discount * discount_rate 
    else 0
  total_before_discount - discount

/-- Theorem stating that the bakery order cost is $54.00 given the specified conditions --/
theorem bakery_order_cost_is_54 : 
  bakery_order_cost 15 3 2 2 6 6 (1/10) 50 = 54 := by
  sorry

end bakery_order_cost_is_54_l640_64098


namespace worker_completion_time_l640_64058

/-- Given that two workers a and b can complete a job together in 8 days,
    and worker a alone can complete the job in 12 days,
    prove that worker b alone can complete the job in 24 days. -/
theorem worker_completion_time (a b : ℝ) 
  (h1 : a + b = 1 / 8)  -- a and b together complete 1/8 of the work per day
  (h2 : a = 1 / 12)     -- a alone completes 1/12 of the work per day
  : b = 1 / 24 :=       -- b alone completes 1/24 of the work per day
by sorry

end worker_completion_time_l640_64058
