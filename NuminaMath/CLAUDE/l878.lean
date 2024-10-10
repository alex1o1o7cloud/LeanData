import Mathlib

namespace tens_digit_of_17_to_1993_l878_87823

theorem tens_digit_of_17_to_1993 : ∃ n : ℕ, 17^1993 ≡ 30 + n [ZMOD 100] :=
sorry

end tens_digit_of_17_to_1993_l878_87823


namespace cube_difference_implies_sum_of_squares_l878_87851

theorem cube_difference_implies_sum_of_squares (n : ℕ) (hn : n > 0) :
  (∃ x : ℕ, x > 0 ∧ (x + 1)^3 - x^3 = n^2) →
  ∃ a b : ℕ, n = a^2 + b^2 := by
sorry

end cube_difference_implies_sum_of_squares_l878_87851


namespace last_digit_of_322_power_111569_last_digit_is_two_l878_87807

theorem last_digit_of_322_power_111569 : ℕ → Prop :=
  fun n => (322^111569 : ℕ) % 10 = n

theorem last_digit_is_two : last_digit_of_322_power_111569 2 := by
  sorry

end last_digit_of_322_power_111569_last_digit_is_two_l878_87807


namespace apple_cost_is_75_cents_l878_87844

/-- The cost of an apple given the amount paid and change received -/
def appleCost (amountPaid change : ℚ) : ℚ :=
  amountPaid - change

/-- Proof that the apple costs $0.75 given the conditions -/
theorem apple_cost_is_75_cents (amountPaid change : ℚ) 
  (h1 : amountPaid = 5)
  (h2 : change = 4.25) : 
  appleCost amountPaid change = 0.75 := by
  sorry

end apple_cost_is_75_cents_l878_87844


namespace house_number_problem_l878_87817

theorem house_number_problem (numbers : List Nat) 
  (h_numbers : numbers = [1, 3, 4, 6, 8, 9, 11, 12, 16]) 
  (h_total : numbers.sum = 70) 
  (vova_sum dima_sum : Nat) 
  (h_vova_dima : vova_sum = 3 * dima_sum) 
  (h_sum_relation : vova_sum + dima_sum + house_number = 70) 
  (h_house_mod : house_number % 4 = 2) : 
  house_number = 6 := by
  sorry

end house_number_problem_l878_87817


namespace paint_mixture_ratio_l878_87883

/-- Given a paint mixture with ratio blue:green:white as 3:3:5, 
    prove that using 15 quarts of white paint requires 9 quarts of green paint -/
theorem paint_mixture_ratio (blue green white : ℚ) : 
  blue / green = 1 →
  green / white = 3 / 5 →
  white = 15 →
  green = 9 :=
by sorry

end paint_mixture_ratio_l878_87883


namespace total_salmon_count_l878_87824

/-- Represents the count of male and female salmon for a species -/
structure SalmonCount where
  males : Nat
  females : Nat

/-- Calculates the total number of salmon for a given species -/
def totalForSpecies (count : SalmonCount) : Nat :=
  count.males + count.females

/-- The counts for each salmon species -/
def chinookCount : SalmonCount := { males := 451228, females := 164225 }
def sockeyeCount : SalmonCount := { males := 212001, females := 76914 }
def cohoCount : SalmonCount := { males := 301008, females := 111873 }
def pinkCount : SalmonCount := { males := 518001, females := 182945 }
def chumCount : SalmonCount := { males := 230023, females := 81321 }

/-- Theorem stating that the total number of salmon across all species is 2,329,539 -/
theorem total_salmon_count : 
  totalForSpecies chinookCount + 
  totalForSpecies sockeyeCount + 
  totalForSpecies cohoCount + 
  totalForSpecies pinkCount + 
  totalForSpecies chumCount = 2329539 := by
  sorry

end total_salmon_count_l878_87824


namespace at_most_one_prime_between_factorial_and_factorial_plus_n_plus_one_l878_87835

theorem at_most_one_prime_between_factorial_and_factorial_plus_n_plus_one (n : ℕ) (hn : n > 1) :
  ∃! p : ℕ, Prime p ∧ n! < p ∧ p < n! + n + 1 :=
by sorry

end at_most_one_prime_between_factorial_and_factorial_plus_n_plus_one_l878_87835


namespace amount_with_r_l878_87828

theorem amount_with_r (total : ℝ) (amount_r : ℝ) : 
  total = 7000 →
  amount_r = (2/3) * (total - amount_r) →
  amount_r = 2800 := by
sorry

end amount_with_r_l878_87828


namespace hyperbola_asymptote_l878_87815

/-- Given a hyperbola with equation x²-y²/b²=1 and b > 0, 
    prove that if one of its asymptote lines is 2x-y=0, then b = 2 -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 - y^2/b^2 = 1 → (2*x - y = 0 → b = 2)) :=
by sorry

end hyperbola_asymptote_l878_87815


namespace goldbach_nine_l878_87897

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

-- Define the theorem
theorem goldbach_nine : 
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ p + q + r = 9 :=
sorry

end goldbach_nine_l878_87897


namespace even_integer_solution_l878_87893

-- Define the function h for even integers
def h (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n ≥ 2 then
    (n / 2) * (2 + n) / 2
  else
    0

-- Theorem statement
theorem even_integer_solution :
  ∃ x : ℕ, x % 2 = 0 ∧ x ≥ 2 ∧ h 18 / h x = 3 → x = 10 := by
  sorry

end even_integer_solution_l878_87893


namespace inequality_solution_set_l878_87857

theorem inequality_solution_set (x : ℝ) : 
  (6 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 4) ↔ (2 + Real.sqrt 2 < x ∧ x < 4) :=
sorry

end inequality_solution_set_l878_87857


namespace inscribed_hexagon_radius_equation_l878_87888

/-- A hexagon inscribed in a circle with specific side lengths -/
structure InscribedHexagon where
  r : ℝ  -- radius of the circumscribed circle
  side1 : ℝ  -- length of two sides
  side2 : ℝ  -- length of two other sides
  side3 : ℝ  -- length of the remaining two sides
  h1 : side1 = 1
  h2 : side2 = 2
  h3 : side3 = 3

/-- The radius of the circumscribed circle satisfies a specific equation -/
theorem inscribed_hexagon_radius_equation (h : InscribedHexagon) : 
  2 * h.r^3 - 7 * h.r - 3 = 0 := by
  sorry

end inscribed_hexagon_radius_equation_l878_87888


namespace no_integer_points_on_circle_l878_87879

theorem no_integer_points_on_circle : 
  ¬ ∃ (x : ℤ), (x - 3)^2 + (x + 1 + 2)^2 ≤ 8^2 :=
by sorry

end no_integer_points_on_circle_l878_87879


namespace hexagon_angle_measure_l878_87822

/-- In a convex hexagon ABCDEF, prove that the measure of angle D is 145 degrees
    given the following conditions:
    - Angles A, B, and C are congruent
    - Angles D, E, and F are congruent
    - Angle A is 50 degrees less than angle D -/
theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  A = B ∧ B = C ∧  -- Angles A, B, and C are congruent
  D = E ∧ E = F ∧  -- Angles D, E, and F are congruent
  A + 50 = D ∧     -- Angle A is 50 degrees less than angle D
  A + B + C + D + E + F = 720  -- Sum of angles in a hexagon
  → D = 145 := by
  sorry

end hexagon_angle_measure_l878_87822


namespace number_of_boys_l878_87819

theorem number_of_boys (num_vans : ℕ) (students_per_van : ℕ) (num_girls : ℕ) : 
  num_vans = 5 → students_per_van = 28 → num_girls = 80 → 
  num_vans * students_per_van - num_girls = 60 := by
  sorry

end number_of_boys_l878_87819


namespace union_A_B_intersection_A_complement_B_range_of_a_l878_87855

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem 1: A ∪ B = {x | -1 ≤ x}
theorem union_A_B : A ∪ B = {x : ℝ | -1 ≤ x} := by sorry

-- Theorem 2: A ∩ (Cᴿ B) = {x | -1 ≤ x < 2}
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

-- Theorem 3: If B ∪ C = C, then a ≤ 3
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end union_A_B_intersection_A_complement_B_range_of_a_l878_87855


namespace percentage_problem_l878_87834

theorem percentage_problem (x : ℝ) : (0.15 * 0.30 * 0.50 * x = 90) → x = 4000 := by
  sorry

end percentage_problem_l878_87834


namespace apples_eaten_by_two_children_l878_87801

/-- Proves that given 5 children who each collected 15 apples, if one child sold 7 apples
    and they had 60 apples left when they got home, then two children ate a total of 8 apples. -/
theorem apples_eaten_by_two_children
  (num_children : Nat)
  (apples_per_child : Nat)
  (apples_sold : Nat)
  (apples_left : Nat)
  (h1 : num_children = 5)
  (h2 : apples_per_child = 15)
  (h3 : apples_sold = 7)
  (h4 : apples_left = 60) :
  ∃ (eaten_by_two : Nat), eaten_by_two = 8 ∧
    num_children * apples_per_child = apples_left + apples_sold + eaten_by_two :=
by sorry


end apples_eaten_by_two_children_l878_87801


namespace min_sum_of_squares_l878_87869

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 3*b + 5*c + 7*d = 14) :
  a^2 + b^2 + c^2 + d^2 ≥ 7/3 ∧
  (a^2 + b^2 + c^2 + d^2 = 7/3 ↔ a = 1/6 ∧ b = 1/2 ∧ c = 5/6 ∧ d = 7/6) :=
by sorry

end min_sum_of_squares_l878_87869


namespace journey_distance_proof_l878_87809

def total_journey_time : Real := 2.5
def first_segment_time : Real := 0.5
def first_segment_speed : Real := 20
def break_time : Real := 0.25
def second_segment_time : Real := 1
def second_segment_speed : Real := 30
def third_segment_speed : Real := 15

theorem journey_distance_proof :
  let first_segment_distance := first_segment_time * first_segment_speed
  let second_segment_distance := second_segment_time * second_segment_speed
  let third_segment_time := total_journey_time - (first_segment_time + break_time + second_segment_time)
  let third_segment_distance := third_segment_time * third_segment_speed
  let total_distance := first_segment_distance + second_segment_distance + third_segment_distance
  total_distance = 51.25 := by
  sorry

end journey_distance_proof_l878_87809


namespace quadratic_polynomial_with_complex_root_l878_87833

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (x : ℂ), a * x^2 + b * x + c = 0 ↔ x = 4 + 2*I ∨ x = 4 - 2*I) ∧
    (a * (4 + 2*I)^2 + b * (4 + 2*I) + c = 0) ∧
    (a = 3 ∧ b = -24 ∧ c = 36) :=
by
  sorry

end quadratic_polynomial_with_complex_root_l878_87833


namespace expression_value_l878_87881

theorem expression_value (a b : ℝ) (h : 3 * (a - 2) = 2 * (2 * b - 4)) :
  9 * a^2 - 24 * a * b + 16 * b^2 + 25 = 29 := by
sorry

end expression_value_l878_87881


namespace intersection_complement_equals_set_l878_87895

theorem intersection_complement_equals_set (U A B : Set Nat) : 
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {2, 3, 4, 5} →
  B = {2, 3, 6, 7} →
  B ∩ (U \ A) = {6, 7} := by
  sorry

end intersection_complement_equals_set_l878_87895


namespace village_foods_lettuce_price_l878_87887

/-- The price of a head of lettuce at Village Foods -/
def lettuce_price : ℝ := 1

/-- The number of customers per month -/
def customers_per_month : ℕ := 500

/-- The number of lettuce heads each customer buys -/
def lettuce_per_customer : ℕ := 2

/-- The number of tomatoes each customer buys -/
def tomatoes_per_customer : ℕ := 4

/-- The price of each tomato -/
def tomato_price : ℝ := 0.5

/-- The total monthly sales of lettuce and tomatoes -/
def total_monthly_sales : ℝ := 2000

theorem village_foods_lettuce_price :
  lettuce_price = 1 ∧
  customers_per_month * lettuce_per_customer * lettuce_price +
  customers_per_month * tomatoes_per_customer * tomato_price = total_monthly_sales :=
by sorry

end village_foods_lettuce_price_l878_87887


namespace square_perimeter_equal_area_rectangle_l878_87863

theorem square_perimeter_equal_area_rectangle (l w : ℝ) (h1 : l = 1024) (h2 : w = 1) :
  let rectangle_area := l * w
  let square_side := (rectangle_area).sqrt
  let square_perimeter := 4 * square_side
  square_perimeter = 128 := by
sorry

end square_perimeter_equal_area_rectangle_l878_87863


namespace problem_statement_l878_87836

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- State the theorem
theorem problem_statement (m : ℝ) (a b c : ℝ) :
  (∀ x, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1) 1) →
  a > 0 → b > 0 → c > 0 →
  1 / a + 1 / (2 * b) + 1 / (3 * c) = m →
  (m = 1 ∧ a + 2 * b + 3 * c ≥ 9) :=
by sorry

end problem_statement_l878_87836


namespace divisors_of_expression_l878_87811

theorem divisors_of_expression (n : ℕ+) : 
  ∃ (d : Finset ℕ+), 
    (∀ k : ℕ+, k ∈ d ↔ ∀ m : ℕ+, k ∣ (m * (m^2 - 1) * (m^2 + 3) * (m^2 + 5))) ∧
    d.card = 16 :=
sorry

end divisors_of_expression_l878_87811


namespace inscribed_square_area_l878_87816

/-- A triangle with an inscribed square -/
structure TriangleWithInscribedSquare where
  /-- The base of the triangle -/
  base : ℝ
  /-- The altitude of the triangle -/
  altitude : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- The square's side is parallel to and lies on the triangle's base -/
  square_on_base : square_side ≤ base

/-- The area of the inscribed square -/
def square_area (t : TriangleWithInscribedSquare) : ℝ :=
  t.square_side ^ 2

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area
  (t : TriangleWithInscribedSquare)
  (h1 : t.base = 12)
  (h2 : t.altitude = 7) :
  square_area t = 36 := by
  sorry

end inscribed_square_area_l878_87816


namespace equation_solution_l878_87827

theorem equation_solution : 
  let f (x : ℝ) := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1)
  let g (x : ℝ) := (x - 2) * (x - 4) * (x - 5) * (x - 2)
  ∀ x : ℝ, (g x ≠ 0 ∧ f x / g x = 1) ↔ (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end equation_solution_l878_87827


namespace geometric_sequence_property_l878_87872

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 * a 19 = 16) →
  (a 1 + a 19 = 10) →
  a 8 * a 10 * a 12 = 64 := by
  sorry

end geometric_sequence_property_l878_87872


namespace muffin_boxes_l878_87843

theorem muffin_boxes (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) : 
  total_muffins = 95 →
  muffins_per_box = 5 →
  available_boxes = 10 →
  (total_muffins - available_boxes * muffins_per_box + muffins_per_box - 1) / muffins_per_box = 9 :=
by sorry

end muffin_boxes_l878_87843


namespace remainder_theorem_l878_87832

theorem remainder_theorem (y : ℤ) 
  (h1 : (2 + y) % (3^3) = 3^2 % (3^3))
  (h2 : (4 + y) % (5^3) = 2^3 % (5^3))
  (h3 : (6 + y) % (7^3) = 7^2 % (7^3)) :
  y % 105 = 1 := by
  sorry

end remainder_theorem_l878_87832


namespace extremum_and_tangent_imply_max_min_difference_l878_87841

def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

theorem extremum_and_tangent_imply_max_min_difference
  (a b c : ℝ) :
  (∃ x, deriv (f a b c) x = 0 ∧ x = 2) →
  (deriv (f a b c) 1 = -3) →
  ∃ max min : ℝ, 
    (∀ x, f a b c x ≤ max) ∧
    (∀ x, f a b c x ≥ min) ∧
    (max - min = 4) :=
sorry

end extremum_and_tangent_imply_max_min_difference_l878_87841


namespace gcf_lcm_sum_l878_87868

def A : ℕ := Nat.gcd 9 (Nat.gcd 12 18)
def B : ℕ := Nat.lcm 9 (Nat.lcm 12 18)

theorem gcf_lcm_sum : A + B = 39 := by
  sorry

end gcf_lcm_sum_l878_87868


namespace spherical_to_rectangular_conversion_l878_87804

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 3
  let θ : ℝ := 5 * π / 12
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (3 * (Real.sqrt 12 + 2) / 8, 3 * (Real.sqrt 12 - 2) / 8, 3 * Real.sqrt 2 / 2) :=
by sorry

end spherical_to_rectangular_conversion_l878_87804


namespace expression_in_terms_of_k_l878_87859

theorem expression_in_terms_of_k (x y k : ℝ) (h : x ≠ y) 
  (hk : (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k) :
  (x^8 + y^8) / (x^8 - y^8) - (x^8 - y^8) / (x^8 + y^8) = 
    (k - 2)^2 * (k + 2)^2 / (4 * k * (k^2 + 4)) :=
by sorry

end expression_in_terms_of_k_l878_87859


namespace imaginary_part_of_complex_fraction_l878_87876

theorem imaginary_part_of_complex_fraction :
  Complex.im ((5 : ℂ) + Complex.I) / ((1 : ℂ) + Complex.I) = -2 := by
  sorry

end imaginary_part_of_complex_fraction_l878_87876


namespace line_passes_through_fixed_point_l878_87882

/-- A line in the form y - 2 = mx + m passes through the point (-1, 2) for all values of m -/
theorem line_passes_through_fixed_point (m : ℝ) :
  let line := fun (x y : ℝ) => y - 2 = m * x + m
  line (-1) 2 := by sorry

end line_passes_through_fixed_point_l878_87882


namespace room_height_proof_l878_87818

theorem room_height_proof (length breadth diagonal : ℝ) (h : ℝ) :
  length = 12 →
  breadth = 8 →
  diagonal = 17 →
  diagonal^2 = length^2 + breadth^2 + h^2 →
  h = 9 := by
sorry

end room_height_proof_l878_87818


namespace nonAttackingRooksPlacementCount_l878_87839

/-- The size of the chessboard -/
def boardSize : Nat := 8

/-- The total number of squares on the chessboard -/
def totalSquares : Nat := boardSize * boardSize

/-- The number of squares a rook attacks in its row and column, excluding itself -/
def attackedSquares : Nat := 2 * (boardSize - 1)

/-- The number of ways to place two rooks on a chessboard so they don't attack each other -/
def nonAttackingRooksPlacement : Nat := totalSquares * (totalSquares - 1 - attackedSquares)

theorem nonAttackingRooksPlacementCount : nonAttackingRooksPlacement = 3136 := by
  sorry

end nonAttackingRooksPlacementCount_l878_87839


namespace inscribed_square_area_l878_87831

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A square inscribed in the region bounded by the parabola and the x-axis -/
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  side : ℝ    -- length of the square's side
  h1 : center - side/2 ≥ 0  -- left side of square is non-negative
  h2 : center + side/2 ≤ 10 -- right side of square is at most the x-intercept
  h3 : parabola (center + side/2) = side -- top-right corner lies on the parabola

theorem inscribed_square_area :
  ∃ (s : InscribedSquare), s.side^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end inscribed_square_area_l878_87831


namespace parabola_point_ordering_l878_87800

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 5

-- Define the theorem
theorem parabola_point_ordering :
  ∀ (y₁ y₂ y₃ : ℝ),
  f (-4) = y₁ →
  f (-1) = y₂ →
  f 2 = y₃ →
  y₂ > y₃ ∧ y₃ > y₁ :=
by
  sorry

end parabola_point_ordering_l878_87800


namespace can_transport_machines_l878_87862

/-- Given three machines with masses in kg and a truck's capacity in kg,
    prove that the truck can transport all machines at once. -/
theorem can_transport_machines (m1 m2 m3 truck_capacity : ℕ) 
  (h1 : m1 = 800)
  (h2 : m2 = 500)
  (h3 : m3 = 600)
  (h4 : truck_capacity = 2000) :
  m1 + m2 + m3 ≤ truck_capacity := by
  sorry

#check can_transport_machines

end can_transport_machines_l878_87862


namespace candidate_vote_percentage_l878_87898

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (loss_margin : ℕ) 
  (candidate_percentage : ℚ) :
  total_votes = 2000 →
  loss_margin = 640 →
  candidate_percentage * total_votes + (candidate_percentage * total_votes + loss_margin) = total_votes →
  candidate_percentage = 34 / 100 := by
sorry

end candidate_vote_percentage_l878_87898


namespace vector_properties_l878_87866

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_properties :
  (∀ (m : ℝ) (a b : V), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : V), (m - n) • a = m • a - n • a) ∧
  (∃ (m : ℝ) (a b : V), m • a = m • b ∧ a ≠ b) ∧
  (∀ (m n : ℝ) (a : V), a ≠ 0 → m • a = n • a → m = n) :=
sorry

end vector_properties_l878_87866


namespace laborer_income_l878_87810

/-- Represents the financial situation of a laborer over a 10-month period -/
structure LaborerFinances where
  monthly_income : ℝ
  initial_expenditure : ℝ
  initial_months : ℕ
  reduced_expenditure : ℝ
  reduced_months : ℕ
  savings : ℝ

/-- The theorem stating the laborer's monthly income given the conditions -/
theorem laborer_income (lf : LaborerFinances) 
  (h1 : lf.initial_expenditure = 85)
  (h2 : lf.initial_months = 6)
  (h3 : lf.reduced_expenditure = 60)
  (h4 : lf.reduced_months = 4)
  (h5 : lf.savings = 30)
  (h6 : ∃ d : ℝ, d > 0 ∧ 
        lf.monthly_income * lf.initial_months = lf.initial_expenditure * lf.initial_months - d ∧
        lf.monthly_income * lf.reduced_months = lf.reduced_expenditure * lf.reduced_months + d + lf.savings) :
  lf.monthly_income = 78 := by
sorry

end laborer_income_l878_87810


namespace max_square_pen_area_l878_87891

/-- Given 36 feet of fencing, the maximum area of a square pen is 81 square feet. -/
theorem max_square_pen_area (fencing : ℝ) (h : fencing = 36) : 
  (fencing / 4) ^ 2 = 81 :=
by sorry

end max_square_pen_area_l878_87891


namespace cylinder_section_volume_l878_87864

/-- Represents a cylinder -/
structure Cylinder where
  base_area : ℝ
  height : ℝ

/-- Represents a plane cutting the cylinder -/
structure CuttingPlane where
  not_parallel_to_base : Bool
  not_intersect_base : Bool

/-- Represents the section of the cylinder cut by the plane -/
structure CylinderSection where
  cylinder : Cylinder
  cutting_plane : CuttingPlane
  segment_height : ℝ

/-- The volume of a cylinder section -/
def section_volume (s : CylinderSection) : ℝ :=
  s.cylinder.base_area * s.segment_height

theorem cylinder_section_volume 
  (s : CylinderSection) 
  (h1 : s.cutting_plane.not_parallel_to_base = true) 
  (h2 : s.cutting_plane.not_intersect_base = true) : 
  ∃ (v : ℝ), v = section_volume s :=
sorry

end cylinder_section_volume_l878_87864


namespace february_greatest_difference_l878_87852

/-- Sales data for trumpet and trombone players -/
structure SalesData where
  trumpet : ℕ
  trombone : ℕ

/-- Calculate percentage difference between two numbers -/
def percentDifference (a b : ℕ) : ℚ :=
  (max a b - min a b : ℚ) / (min a b : ℚ) * 100

/-- Months of the year -/
inductive Month
  | Jan | Feb | Mar | Apr | May

/-- Sales data for each month -/
def monthlySales : Month → SalesData
  | Month.Jan => ⟨6, 4⟩
  | Month.Feb => ⟨27, 5⟩  -- Trumpet sales tripled
  | Month.Mar => ⟨8, 5⟩
  | Month.Apr => ⟨7, 8⟩
  | Month.May => ⟨5, 6⟩

/-- February has the greatest percent difference in sales -/
theorem february_greatest_difference :
  ∀ m : Month, m ≠ Month.Feb →
    percentDifference (monthlySales Month.Feb).trumpet (monthlySales Month.Feb).trombone >
    percentDifference (monthlySales m).trumpet (monthlySales m).trombone :=
by sorry

end february_greatest_difference_l878_87852


namespace pension_formula_l878_87830

/-- Represents the annual pension function based on years of service -/
def annual_pension (k : ℝ) (x : ℝ) : ℝ := k * x^2

/-- The pension increase after 4 additional years of service -/
def increase_4_years (k : ℝ) (x : ℝ) : ℝ := annual_pension k (x + 4) - annual_pension k x

/-- The pension increase after 9 additional years of service -/
def increase_9_years (k : ℝ) (x : ℝ) : ℝ := annual_pension k (x + 9) - annual_pension k x

theorem pension_formula (k : ℝ) (x : ℝ) :
  (increase_4_years k x = 144) ∧ 
  (increase_9_years k x = 324) →
  annual_pension k x = (Real.sqrt 171 / 5) * x^2 := by
  sorry

end pension_formula_l878_87830


namespace total_wheels_is_25_l878_87892

/-- Calculates the total number of wheels in Jordan's driveway -/
def total_wheels : ℕ :=
  let num_cars : ℕ := 2
  let wheels_per_car : ℕ := 4
  let num_bikes : ℕ := 2
  let wheels_per_bike : ℕ := 2
  let num_trash_cans : ℕ := 1
  let wheels_per_trash_can : ℕ := 2
  let num_tricycles : ℕ := 1
  let wheels_per_tricycle : ℕ := 3
  let num_roller_skate_pairs : ℕ := 1
  let wheels_per_roller_skate : ℕ := 4
  let wheels_per_roller_skate_pair : ℕ := 2 * wheels_per_roller_skate

  num_cars * wheels_per_car +
  num_bikes * wheels_per_bike +
  num_trash_cans * wheels_per_trash_can +
  num_tricycles * wheels_per_tricycle +
  num_roller_skate_pairs * wheels_per_roller_skate_pair

theorem total_wheels_is_25 : total_wheels = 25 := by
  sorry

end total_wheels_is_25_l878_87892


namespace odd_function_derivative_l878_87858

theorem odd_function_derivative (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →
  (∀ x, HasDerivAt f (g x) x) →
  ∀ x, g (-x) = -g x := by
sorry

end odd_function_derivative_l878_87858


namespace cylinder_volume_increase_l878_87894

/-- Proves that tripling the height and increasing the radius by 150% results in a volume increase by a factor of 18.75 -/
theorem cylinder_volume_increase (r h : ℝ) (r_pos : 0 < r) (h_pos : 0 < h) : 
  let new_r := 2.5 * r
  let new_h := 3 * h
  π * new_r^2 * new_h = 18.75 * (π * r^2 * h) :=
by sorry

end cylinder_volume_increase_l878_87894


namespace intercepts_count_l878_87837

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 3*x - 2

-- Define x-intercepts
def is_x_intercept (x : ℝ) : Prop := f x = 0

-- Define y-intercepts
def is_y_intercept (y : ℝ) : Prop := ∃ x, f x = y ∧ x = 0

-- Theorem statement
theorem intercepts_count :
  (∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, is_x_intercept x) ∧
  (∃! y, is_y_intercept y) :=
sorry

end intercepts_count_l878_87837


namespace total_dumbbell_weight_l878_87865

/-- Represents the weight of a single dumbbell in a pair --/
def dumbbell_weights : List ℕ := [3, 5, 8, 12, 18, 27]

/-- Theorem: The total weight of the dumbbell system is 146 lb --/
theorem total_dumbbell_weight : 
  (dumbbell_weights.map (·*2)).sum = 146 := by sorry

end total_dumbbell_weight_l878_87865


namespace diamond_calculation_l878_87877

-- Define the diamond operation
def diamond (X Y : ℚ) : ℚ := (2 * X + 3 * Y) / 5

-- Theorem statement
theorem diamond_calculation : diamond (diamond 3 15) 6 = 192 / 25 := by
  sorry

end diamond_calculation_l878_87877


namespace pizza_eaten_fraction_l878_87856

theorem pizza_eaten_fraction (n : Nat) : 
  let r : ℚ := 1/3
  let sum : ℚ := (1 - r^n) / (1 - r)
  n = 6 → sum = 364/729 := by
sorry

end pizza_eaten_fraction_l878_87856


namespace shelby_rainy_driving_time_l878_87821

/-- Represents the driving scenario of Shelby --/
structure DrivingScenario where
  sunny_speed : ℝ  -- Speed in sunny conditions (mph)
  rainy_speed : ℝ  -- Speed in rainy conditions (mph)
  total_distance : ℝ  -- Total distance covered (miles)
  total_time : ℝ  -- Total time of travel (minutes)

/-- Calculates the time spent driving in rainy conditions --/
def rainy_time (scenario : DrivingScenario) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that given the specific conditions, the rainy driving time is 40 minutes --/
theorem shelby_rainy_driving_time :
  let scenario : DrivingScenario := {
    sunny_speed := 35,
    rainy_speed := 25,
    total_distance := 22.5,
    total_time := 50
  }
  rainy_time scenario = 40 := by
  sorry

end shelby_rainy_driving_time_l878_87821


namespace negative_product_inequality_l878_87820

theorem negative_product_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end negative_product_inequality_l878_87820


namespace logarithm_expression_equals_negative_one_l878_87899

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_negative_one :
  log10 (5/2) + 2 * log10 2 - (1/2)⁻¹ = -1 :=
by
  -- Assume the given condition
  have h : log10 2 + log10 5 = 1 := by sorry
  -- The proof would go here
  sorry

end logarithm_expression_equals_negative_one_l878_87899


namespace four_point_triangles_l878_87849

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A set of four points in a plane -/
structure FourPoints :=
  (a b c d : Point)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if no three points in a set of four points are collinear -/
def no_three_collinear (points : FourPoints) : Prop :=
  ¬(collinear points.a points.b points.c) ∧
  ¬(collinear points.a points.b points.d) ∧
  ¬(collinear points.a points.c points.d) ∧
  ¬(collinear points.b points.c points.d)

/-- The number of distinct triangles that can be formed from four points -/
def num_triangles (points : FourPoints) : ℕ := sorry

/-- Theorem: Given four points on a plane where no three points are collinear,
    the number of distinct triangles that can be formed is 4 -/
theorem four_point_triangles (points : FourPoints) 
  (h : no_three_collinear points) : num_triangles points = 4 := by
  sorry

end four_point_triangles_l878_87849


namespace unique_two_digit_number_l878_87842

/-- A function that returns the tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- A function that returns the ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

/-- A predicate that checks if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- A predicate that checks if a natural number is even -/
def is_even (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

/-- A predicate that checks if a natural number is a multiple of 9 -/
def is_multiple_of_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

/-- A predicate that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem unique_two_digit_number : 
  ∀ n : ℕ, 
    is_two_digit n ∧ 
    is_even n ∧ 
    is_multiple_of_9 n ∧ 
    is_perfect_square (tens_digit n * ones_digit n) → 
    n = 90 :=
sorry

end unique_two_digit_number_l878_87842


namespace ellipse_foci_product_range_l878_87845

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def leftFocus : ℝ × ℝ := sorry
def rightFocus : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_foci_product_range (p : ℝ × ℝ) :
  ellipse p.1 p.2 →
  3 ≤ (distance p leftFocus) * (distance p rightFocus) ∧
  (distance p leftFocus) * (distance p rightFocus) ≤ 4 :=
sorry

end ellipse_foci_product_range_l878_87845


namespace least_five_digit_square_cube_l878_87889

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, m < n →
    ¬((m ≥ 10000 ∧ m < 100000) ∧
      (∃ x : ℕ, m = x^2) ∧
      (∃ y : ℕ, m = y^3))) ∧
  n = 15625 :=
by sorry

end least_five_digit_square_cube_l878_87889


namespace mangoes_sold_to_market_proof_l878_87847

/-- Calculates the amount of mangoes sold to market given total harvest, mangoes per kilogram, and remaining mangoes -/
def mangoes_sold_to_market (total_harvest : ℕ) (mangoes_per_kg : ℕ) (remaining_mangoes : ℕ) : ℕ :=
  let total_mangoes := total_harvest * mangoes_per_kg
  let sold_mangoes := total_mangoes - remaining_mangoes
  sold_mangoes / 2 / mangoes_per_kg

/-- Theorem stating that given the problem conditions, 20 kilograms of mangoes were sold to market -/
theorem mangoes_sold_to_market_proof :
  mangoes_sold_to_market 60 8 160 = 20 := by
  sorry

end mangoes_sold_to_market_proof_l878_87847


namespace max_value_x2_plus_y2_l878_87884

theorem max_value_x2_plus_y2 (x y : ℝ) :
  5 * x^2 - 10 * x + 4 * y^2 = 0 →
  x^2 + y^2 ≤ 4 :=
by sorry

end max_value_x2_plus_y2_l878_87884


namespace Z_in_third_quadrant_implies_a_range_l878_87846

def Z (a : ℝ) : ℂ := Complex.mk (a^2 - 2*a) (a^2 - a - 2)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem Z_in_third_quadrant_implies_a_range (a : ℝ) :
  in_third_quadrant (Z a) → 0 < a ∧ a < 2 := by sorry

end Z_in_third_quadrant_implies_a_range_l878_87846


namespace inequality_implies_k_range_l878_87886

/-- The natural logarithm function -/
noncomputable def f (x : ℝ) : ℝ := Real.log x

/-- The exponential function -/
noncomputable def g (x : ℝ) : ℝ := Real.exp x

/-- The main theorem -/
theorem inequality_implies_k_range (k : ℝ) :
  (∀ x : ℝ, x ≥ 1 → x * f x - k * (x + 1) * f (g (x - 1)) ≤ 0) →
  k ≥ 1/2 := by sorry

end inequality_implies_k_range_l878_87886


namespace travel_allowance_percentage_l878_87870

theorem travel_allowance_percentage
  (total_employees : ℕ)
  (salary_increase_percentage : ℚ)
  (no_increase : ℕ)
  (h1 : total_employees = 480)
  (h2 : salary_increase_percentage = 1/10)
  (h3 : no_increase = 336) :
  (total_employees - (salary_increase_percentage * total_employees + no_increase : ℚ)) / total_employees = 1/5 :=
by sorry

end travel_allowance_percentage_l878_87870


namespace symmetric_trig_function_property_l878_87812

/-- Given a function f(x) = a*sin(2x) + b*cos(2x) where a and b are real numbers,
    ab ≠ 0, and f is symmetric about x = π/6, prove that a = √3 * b. -/
theorem symmetric_trig_function_property (a b : ℝ) (h1 : a * b ≠ 0) :
  (∀ x, a * Real.sin (2 * x) + b * Real.cos (2 * x) = 
        a * Real.sin (2 * (Real.pi / 6 - x)) + b * Real.cos (2 * (Real.pi / 6 - x))) →
  a = Real.sqrt 3 * b := by
  sorry

end symmetric_trig_function_property_l878_87812


namespace f_range_implies_a_value_l878_87890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then
    -x + 3
  else if 2 ≤ x ∧ x ≤ 8 then
    1 + Real.log (2 * x) / Real.log (a^2 - 1)
  else
    0  -- undefined for other x values

theorem f_range_implies_a_value (a : ℝ) :
  (∀ y ∈ Set.range (f a), 2 ≤ y ∧ y ≤ 5) →
  (a = Real.sqrt 3 ∨ a = -Real.sqrt 3) :=
by sorry

end f_range_implies_a_value_l878_87890


namespace base5_product_l878_87878

/-- Converts a list of digits in base 5 to a natural number -/
def fromBase5 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 5 * acc + d) 0

/-- Converts a natural number to a list of digits in base 5 -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The main theorem stating that the product of 1324₅ and 23₅ in base 5 is 42112₅ -/
theorem base5_product :
  toBase5 (fromBase5 [1, 3, 2, 4] * fromBase5 [2, 3]) = [4, 2, 1, 1, 2] := by
  sorry

end base5_product_l878_87878


namespace equal_integers_from_cyclic_equation_l878_87853

theorem equal_integers_from_cyclic_equation 
  (n : ℕ+) (p : ℕ) (a b c : ℤ) 
  (h_prime : Nat.Prime p)
  (h_eq1 : a^(n : ℕ) + p * b = b^(n : ℕ) + p * c)
  (h_eq2 : b^(n : ℕ) + p * c = c^(n : ℕ) + p * a) :
  a = b ∧ b = c := by
sorry

end equal_integers_from_cyclic_equation_l878_87853


namespace b_payment_l878_87854

/-- Calculate the amount b should pay for renting a pasture -/
theorem b_payment (total_rent : ℕ) 
  (a_horses a_months a_rate : ℕ) 
  (b_horses b_months b_rate : ℕ)
  (c_horses c_months c_rate : ℕ)
  (d_horses d_months d_rate : ℕ)
  (h_total_rent : total_rent = 725)
  (h_a : a_horses = 12 ∧ a_months = 8 ∧ a_rate = 5)
  (h_b : b_horses = 16 ∧ b_months = 9 ∧ b_rate = 6)
  (h_c : c_horses = 18 ∧ c_months = 6 ∧ c_rate = 7)
  (h_d : d_horses = 20 ∧ d_months = 4 ∧ d_rate = 4) :
  ∃ (b_payment : ℕ), b_payment = 259 ∧ 
  b_payment = round ((b_horses * b_months * b_rate : ℚ) / 
    ((a_horses * a_months * a_rate + b_horses * b_months * b_rate + 
      c_horses * c_months * c_rate + d_horses * d_months * d_rate) : ℚ) * total_rent) :=
by sorry

#check b_payment

end b_payment_l878_87854


namespace projection_theorem_l878_87829

def vector_a : Fin 2 → ℝ := ![2, 3]
def vector_b : Fin 2 → ℝ := ![-1, 2]

theorem projection_theorem :
  let dot_product := (vector_a 0) * (vector_b 0) + (vector_a 1) * (vector_b 1)
  let magnitude_b := Real.sqrt ((vector_b 0)^2 + (vector_b 1)^2)
  dot_product / magnitude_b = 4 * Real.sqrt 5 / 5 := by
  sorry

end projection_theorem_l878_87829


namespace tall_mirror_passes_l878_87840

/-- The number of times Sarah and Ellie passed through the room with tall mirrors -/
def T : ℕ := sorry

/-- Sarah's reflections in tall mirrors -/
def sarah_tall : ℕ := 10

/-- Sarah's reflections in wide mirrors -/
def sarah_wide : ℕ := 5

/-- Ellie's reflections in tall mirrors -/
def ellie_tall : ℕ := 6

/-- Ellie's reflections in wide mirrors -/
def ellie_wide : ℕ := 3

/-- Number of times they passed through the wide mirrors room -/
def wide_passes : ℕ := 5

/-- Total number of reflections seen by Sarah and Ellie -/
def total_reflections : ℕ := 88

theorem tall_mirror_passes :
  T * (sarah_tall + ellie_tall) + wide_passes * (sarah_wide + ellie_wide) = total_reflections ∧
  T = 3 := by sorry

end tall_mirror_passes_l878_87840


namespace power_of_product_l878_87873

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end power_of_product_l878_87873


namespace probability_two_common_books_is_36_105_l878_87880

def total_books : ℕ := 12
def books_to_choose : ℕ := 4

def probability_two_common_books : ℚ :=
  (Nat.choose total_books 2 * Nat.choose (total_books - 2) 2 * Nat.choose (total_books - 4) 2) /
  (Nat.choose total_books books_to_choose * Nat.choose total_books books_to_choose)

theorem probability_two_common_books_is_36_105 :
  probability_two_common_books = 36 / 105 := by
  sorry

end probability_two_common_books_is_36_105_l878_87880


namespace vectors_collinear_l878_87874

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b in ℝ², prove they are collinear -/
theorem vectors_collinear :
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (1, -2)
  collinear a b := by
  sorry

end vectors_collinear_l878_87874


namespace number_ratio_l878_87803

theorem number_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y = 8 * (x - y)) : x / y = 9 / 7 := by
  sorry

end number_ratio_l878_87803


namespace arrangement_count_l878_87861

theorem arrangement_count :
  let teachers : ℕ := 3
  let students : ℕ := 6
  let groups : ℕ := 3
  let teachers_per_group : ℕ := 1
  let students_per_group : ℕ := 2
  
  (teachers.factorial * (students.choose students_per_group) * 
   ((students - students_per_group).choose students_per_group) * 
   ((students - 2 * students_per_group).choose students_per_group)) = 540 :=
by sorry

end arrangement_count_l878_87861


namespace prime_pairs_congruence_l878_87871

theorem prime_pairs_congruence (p : ℕ) (hp : Nat.Prime p) : 
  (∃ S : Finset (ℕ × ℕ), S.card = p ∧ 
    (∀ (x y : ℕ), (x, y) ∈ S ↔ 
      (x ≤ p ∧ y ≤ p ∧ (y^2 : ZMod p) = (x^3 - x : ZMod p))))
  ↔ (p = 2 ∨ p % 4 = 3) :=
sorry

end prime_pairs_congruence_l878_87871


namespace coin_toss_total_l878_87813

theorem coin_toss_total (head_count tail_count : ℕ) :
  let total_tosses := head_count + tail_count
  total_tosses = head_count + tail_count := by
  sorry

#check coin_toss_total 3 7

end coin_toss_total_l878_87813


namespace union_of_sets_l878_87875

def A (a : ℕ) : Set ℕ := {3, 2^a}
def B (a b : ℕ) : Set ℕ := {a, b}

theorem union_of_sets (a b : ℕ) (h : A a ∩ B a b = {2}) : A a ∪ B a b = {1, 2, 3} := by
  sorry

end union_of_sets_l878_87875


namespace line_through_point_equal_intercepts_l878_87814

/-- A line passing through (1,2) with equal intercepts has equation 2x - y = 0 or x + y - 3 = 0 -/
theorem line_through_point_equal_intercepts :
  ∀ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 → (x = 1 ∧ y = 2)) →  -- Line passes through (1,2)
    (∃ k : ℝ, k ≠ 0 ∧ a = k ∧ b = k) →                      -- Equal intercepts condition
    ((a = 2 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -3)) := by
  sorry


end line_through_point_equal_intercepts_l878_87814


namespace unique_p_q_sum_l878_87885

theorem unique_p_q_sum (p q : ℤ) : 
  p > 1 → q > 1 → 
  ∃ (k₁ k₂ : ℤ), (2*p - 1 = k₁ * q) ∧ (2*q - 1 = k₂ * p) →
  p + q = 8 := by
sorry

end unique_p_q_sum_l878_87885


namespace dartboard_region_angle_l878_87896

/-- Given a circular dartboard with a region where the probability of a dart landing is 1/4,
    prove that the central angle of this region is 90°. -/
theorem dartboard_region_angle (probability : ℝ) (angle : ℝ) :
  probability = 1/4 →
  angle = probability * 360 →
  angle = 90 :=
by sorry

end dartboard_region_angle_l878_87896


namespace height_on_hypotenuse_l878_87826

theorem height_on_hypotenuse (a b h : ℝ) : 
  a = 3 → b = 6 → a^2 + b^2 = (a*b/h)^2 → h = (6 * Real.sqrt 5) / 5 := by
  sorry

end height_on_hypotenuse_l878_87826


namespace parabola_equation_l878_87860

/-- A parabola perpendicular to the x-axis passing through (1, -√2) has the equation y² = 2x -/
theorem parabola_equation : ∃ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ y^2 = 2*x) ∧ 
  (f 1 = -Real.sqrt 2) ∧
  (∀ x y : ℝ, f x = y → (x, y) ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1}) := by
  sorry

end parabola_equation_l878_87860


namespace total_earrings_l878_87802

/-- Proves that the total number of earrings for Bella, Monica, and Rachel is 70 -/
theorem total_earrings (bella_earrings : ℕ) (monica_earrings : ℕ) (rachel_earrings : ℕ)
  (h1 : bella_earrings = 10)
  (h2 : bella_earrings = monica_earrings / 4)
  (h3 : monica_earrings = 2 * rachel_earrings) :
  bella_earrings + monica_earrings + rachel_earrings = 70 := by
  sorry

#check total_earrings

end total_earrings_l878_87802


namespace zinc_copper_mixture_weight_l878_87848

/-- Given a mixture of zinc and copper in the ratio 9:11, where 27 kg of zinc is used,
    the total weight of the mixture is 60 kg. -/
theorem zinc_copper_mixture_weight : 
  ∀ (zinc copper total : ℝ),
  zinc = 27 →
  zinc / copper = 9 / 11 →
  total = zinc + copper →
  total = 60 :=
by
  sorry

end zinc_copper_mixture_weight_l878_87848


namespace magic_box_result_l878_87805

def magic_box (a b : ℝ) : ℝ := a^2 + b + 1

theorem magic_box_result : 
  let m := magic_box (-2) 3
  magic_box m 1 = 66 := by sorry

end magic_box_result_l878_87805


namespace alpha_plus_beta_equals_two_l878_87850

theorem alpha_plus_beta_equals_two (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0)
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := by
sorry

end alpha_plus_beta_equals_two_l878_87850


namespace book_arrangement_proof_l878_87806

theorem book_arrangement_proof :
  let total_books : ℕ := 8
  let geometry_books : ℕ := 5
  let number_theory_books : ℕ := 3
  Nat.choose total_books geometry_books = 56 :=
by
  sorry

end book_arrangement_proof_l878_87806


namespace cubic_root_equation_solution_l878_87838

theorem cubic_root_equation_solution :
  ∀ x : ℝ, (x^(1/3) = 15 / (8 - x^(1/3))) ↔ (x = 27 ∨ x = 125) :=
by sorry

end cubic_root_equation_solution_l878_87838


namespace carries_profit_l878_87867

/-- Calculates the profit for a cake maker after taxes and expenses -/
def cake_profit (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℚ) 
                (supply_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_hours := hours_per_day * days_worked
  let gross_earnings := hourly_rate * total_hours
  let tax_amount := gross_earnings * tax_rate
  let after_tax_earnings := gross_earnings - tax_amount
  after_tax_earnings - supply_cost

/-- Theorem stating that Carrie's profit is $631.20 given the problem conditions -/
theorem carries_profit :
  cake_profit 4 6 35 150 (7/100) = 631.2 := by
  sorry

end carries_profit_l878_87867


namespace sin_five_pi_sixths_l878_87825

theorem sin_five_pi_sixths : Real.sin (5 * π / 6) = 1 / 2 := by
  sorry

end sin_five_pi_sixths_l878_87825


namespace water_bottles_total_l878_87808

/-- Represents the number of water bottles filled for each team --/
structure TeamBottles where
  football : ℕ
  soccer : ℕ
  lacrosse : ℕ
  rugby : ℕ

/-- Calculate the total number of water bottles filled for all teams --/
def total_bottles (t : TeamBottles) : ℕ :=
  t.football + t.soccer + t.lacrosse + t.rugby

/-- Theorem stating the total number of water bottles filled for the teams --/
theorem water_bottles_total :
  ∃ (t : TeamBottles),
    t.football = 11 * 6 ∧
    t.soccer = 53 ∧
    t.lacrosse = t.football + 12 ∧
    t.rugby = 49 ∧
    total_bottles t = 246 :=
by
  sorry


end water_bottles_total_l878_87808
