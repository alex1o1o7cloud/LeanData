import Mathlib

namespace opposite_player_no_aces_l3323_332306

/-- The number of cards in the deck -/
def deck_size : ℕ := 32

/-- The number of players -/
def num_players : ℕ := 4

/-- The number of cards each player receives -/
def cards_per_player : ℕ := deck_size / num_players

/-- The number of aces in the deck -/
def num_aces : ℕ := 4

/-- The probability that the opposite player has no aces given that one player has no aces -/
def opposite_player_no_aces_prob : ℚ := 130 / 759

theorem opposite_player_no_aces (h1 : deck_size = 32) 
                                (h2 : num_players = 4) 
                                (h3 : cards_per_player = deck_size / num_players) 
                                (h4 : num_aces = 4) : 
  opposite_player_no_aces_prob = 130 / 759 := by
  sorry

end opposite_player_no_aces_l3323_332306


namespace sum_sequence_square_l3323_332352

theorem sum_sequence_square (n : ℕ) : 
  (List.range n).sum + n + (List.range n).reverse.sum = n^2 := by
  sorry

end sum_sequence_square_l3323_332352


namespace min_sum_of_powers_with_same_last_four_digits_l3323_332383

theorem min_sum_of_powers_with_same_last_four_digits :
  ∀ m n : ℕ+,
    m ≠ n →
    (10000 : ℤ) ∣ (2019^(m.val) - 2019^(n.val)) →
    ∀ k l : ℕ+,
      k ≠ l →
      (10000 : ℤ) ∣ (2019^(k.val) - 2019^(l.val)) →
      m.val + n.val ≤ k.val + l.val →
      m.val + n.val = 22 :=
by sorry

end min_sum_of_powers_with_same_last_four_digits_l3323_332383


namespace cottage_rental_cost_per_hour_l3323_332327

/-- Represents the cost of renting a cottage -/
structure CottageRental where
  hours : ℕ
  jack_payment : ℕ
  jill_payment : ℕ

/-- Calculates the cost per hour of renting a cottage -/
def cost_per_hour (rental : CottageRental) : ℚ :=
  (rental.jack_payment + rental.jill_payment : ℚ) / rental.hours

/-- Theorem: The cost per hour of the cottage rental is $5 -/
theorem cottage_rental_cost_per_hour :
  let rental := CottageRental.mk 8 20 20
  cost_per_hour rental = 5 := by
  sorry

end cottage_rental_cost_per_hour_l3323_332327


namespace pyramid_height_l3323_332387

theorem pyramid_height (perimeter : ℝ) (apex_to_vertex : ℝ) (h_perimeter : perimeter = 32) (h_apex : apex_to_vertex = 12) :
  let side := perimeter / 4
  let half_diagonal := side * Real.sqrt 2 / 2
  let height := Real.sqrt (apex_to_vertex ^ 2 - half_diagonal ^ 2)
  height = 4 * Real.sqrt 7 := by
  sorry

end pyramid_height_l3323_332387


namespace square_fraction_below_line_l3323_332337

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line passing through two points -/
structure Line :=
  (p1 : Point)
  (p2 : Point)

/-- Represents a square defined by its corners -/
structure Square :=
  (bottomLeft : Point)
  (topRight : Point)

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ :=
  sorry

/-- Finds the intersection of a line with the right edge of a square -/
def rightEdgeIntersection (l : Line) (s : Square) : Point :=
  sorry

/-- Main theorem: The fraction of the square's area below the line is 1/18 -/
theorem square_fraction_below_line :
  let s := Square.mk (Point.mk 2 0) (Point.mk 5 3)
  let l := Line.mk (Point.mk 2 3) (Point.mk 5 1)
  let intersection := rightEdgeIntersection l s
  let belowArea := triangleArea (Point.mk 2 0) (Point.mk 5 0) intersection
  let totalArea := squareArea s
  belowArea / totalArea = 1 / 18 := by
  sorry

end square_fraction_below_line_l3323_332337


namespace exponential_graph_condition_l3323_332316

/-- A function f : ℝ → ℝ does not pass through the first quadrant if
    for all x > 0, f(x) ≤ 0 or for all x ≥ 0, f(x) < 0 -/
def not_pass_first_quadrant (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x ≤ 0) ∨ (∀ x ≥ 0, f x < 0)

theorem exponential_graph_condition
  (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1)
  (h : not_pass_first_quadrant (fun x ↦ a^x + b - 1)) :
  0 < a ∧ a < 1 ∧ b ≤ 0 := by
  sorry

end exponential_graph_condition_l3323_332316


namespace equation_solutions_l3323_332369

theorem equation_solutions :
  (∀ x : ℝ, (x - 5)^2 = 16 ↔ x = 1 ∨ x = 9) ∧
  (∀ x : ℝ, 2*x^2 - 1 = -4*x ↔ x = -1 + Real.sqrt 6 / 2 ∨ x = -1 - Real.sqrt 6 / 2) ∧
  (∀ x : ℝ, 5*x*(x+1) = 2*(x+1) ↔ x = -1 ∨ x = 2/5) ∧
  (∀ x : ℝ, 2*x^2 - x - 1 = 0 ↔ x = -1/2 ∨ x = 1) :=
by sorry

end equation_solutions_l3323_332369


namespace vertex_at_max_min_l3323_332328

/-- The quadratic function f parameterized by k -/
def f (x k : ℝ) : ℝ := x^2 - 2*(2*k - 1)*x + 3*k^2 - 2*k + 6

/-- The x-coordinate of the vertex of f for a given k -/
def vertex_x (k : ℝ) : ℝ := 2*k - 1

/-- The minimum value of f for a given k -/
def min_value (k : ℝ) : ℝ := f (vertex_x k) k

/-- The theorem stating that the x-coordinate of the vertex when the minimum value is maximized is 1 -/
theorem vertex_at_max_min : 
  ∃ (k : ℝ), ∀ (k' : ℝ), min_value k ≥ min_value k' ∧ vertex_x k = 1 := by sorry

end vertex_at_max_min_l3323_332328


namespace range_of_a_l3323_332385

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  (9 - 5*x) / 4 > 1 ∧ x < a

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 1

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x, inequality_system x a ↔ solution_set x) → a ≥ 1 :=
by
  sorry


end range_of_a_l3323_332385


namespace triangle_count_l3323_332340

theorem triangle_count : ∃ (n : ℕ), n = 36 ∧ 
  n = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 ≤ p.2 ∧ p.1 + p.2 > 11) 
    (Finset.product (Finset.range 12) (Finset.range 12))).card :=
by sorry

end triangle_count_l3323_332340


namespace parallel_vectors_acute_angle_l3323_332373

def parallelVectors (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_acute_angle (x : ℝ) 
  (h1 : 0 < x) (h2 : x < π/2) 
  (h3 : parallelVectors (Real.sin x, 1) (1/2, Real.cos x)) : 
  x = π/4 := by
  sorry

end parallel_vectors_acute_angle_l3323_332373


namespace function_property_l3323_332339

def is_valid_function (f : ℕ+ → ℝ) : Prop :=
  ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2

theorem function_property (f : ℕ+ → ℝ) (h : is_valid_function f) (h4 : f 4 ≥ 25) :
  ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 := by
  sorry

end function_property_l3323_332339


namespace abs_rational_nonnegative_l3323_332356

theorem abs_rational_nonnegative (x : ℚ) : 0 ≤ |x| := by
  sorry

end abs_rational_nonnegative_l3323_332356


namespace special_triangle_properties_l3323_332313

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
def SpecialTriangle (t : Triangle) : Prop :=
  3 * t.a * Real.cos t.C = 2 * t.c * Real.cos t.A ∧ 
  Real.tan t.C = 1/2 ∧
  t.b = 5

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.B = 3 * Real.pi / 4 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C = 5/2) := by
  sorry


end special_triangle_properties_l3323_332313


namespace largest_inscribed_circle_radius_l3323_332366

/-- The radius of the quarter circle -/
def R : ℝ := 12

/-- The radius of the largest inscribed circle -/
def r : ℝ := 3

/-- Theorem stating that r is the radius of the largest inscribed circle -/
theorem largest_inscribed_circle_radius : 
  (R - r)^2 - r^2 = (R/2 + r)^2 - (R/2 - r)^2 := by sorry

end largest_inscribed_circle_radius_l3323_332366


namespace expression_satisfies_conditions_l3323_332392

def original_expression : ℕ := 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1

def transformed_expression : ℕ := 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1

theorem expression_satisfies_conditions :
  (original_expression = 11) ∧
  (transformed_expression = 11) :=
by
  sorry

#eval original_expression
#eval transformed_expression

end expression_satisfies_conditions_l3323_332392


namespace systematic_sampling_probability_l3323_332314

theorem systematic_sampling_probability
  (total_parts : Nat)
  (first_grade : Nat)
  (second_grade : Nat)
  (third_grade : Nat)
  (sample_size : Nat)
  (h1 : total_parts = 120)
  (h2 : first_grade = 24)
  (h3 : second_grade = 36)
  (h4 : third_grade = 60)
  (h5 : sample_size = 20)
  (h6 : total_parts = first_grade + second_grade + third_grade) :
  (sample_size : ℚ) / (total_parts : ℚ) = 1 / 6 := by
  sorry

end systematic_sampling_probability_l3323_332314


namespace product_of_binary_numbers_l3323_332362

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

theorem product_of_binary_numbers :
  let a := [true, true, false, false, true, true]
  let b := [true, true, false, true]
  let result := [true, false, false, true, true, false, false, false, true, false, true]
  binary_to_nat a * binary_to_nat b = binary_to_nat result := by
  sorry

end product_of_binary_numbers_l3323_332362


namespace hyperbola_asymptote_slope_l3323_332374

/-- Given a hyperbola with equation x²/25 - y²/16 = 1, prove that the positive value m
    such that y = ±mx represents the asymptotes is 4/5 -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  x^2 / 25 - y^2 / 16 = 1 →
  ∃ (m : ℝ), m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 4/5 := by
sorry

end hyperbola_asymptote_slope_l3323_332374


namespace max_discount_rate_l3323_332344

/-- The maximum discount rate that can be applied without incurring a loss,
    given an initial markup of 25% -/
theorem max_discount_rate : ∀ (m : ℝ) (x : ℝ),
  m > 0 →  -- Assuming positive cost
  (1.25 * m * (1 - x) ≥ m) ↔ (x ≤ 0.2) :=
by sorry

end max_discount_rate_l3323_332344


namespace unique_quadratic_root_l3323_332304

theorem unique_quadratic_root (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) → a = 4 := by
  sorry

end unique_quadratic_root_l3323_332304


namespace add_inequality_preserves_order_l3323_332363

theorem add_inequality_preserves_order (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) : a + c > b + d := by sorry

end add_inequality_preserves_order_l3323_332363


namespace amount_left_after_spending_l3323_332348

def mildred_spent : ℕ := 25
def candice_spent : ℕ := 35
def total_given : ℕ := 100

theorem amount_left_after_spending :
  total_given - (mildred_spent + candice_spent) = 40 :=
by sorry

end amount_left_after_spending_l3323_332348


namespace two_x_is_equal_mean_value_function_l3323_332382

/-- A function is an "equal mean value function" if it satisfies two conditions:
    1) For any x in its domain, f(x) + f(-x) = 0
    2) For any x₁ in its domain, there exists x₂ such that (f(x₁) + f(x₂))/2 = (x₁ + x₂)/2 -/
def is_equal_mean_value_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x₁, ∃ x₂, (f x₁ + f x₂) / 2 = (x₁ + x₂) / 2)

/-- The function f(x) = 2x is an "equal mean value function" -/
theorem two_x_is_equal_mean_value_function :
  is_equal_mean_value_function (λ x ↦ 2 * x) := by
  sorry

end two_x_is_equal_mean_value_function_l3323_332382


namespace minimum_race_distance_l3323_332307

/-- The minimum distance a runner must travel in a race with given constraints -/
theorem minimum_race_distance (wall_length : ℝ) (a_to_wall : ℝ) (wall_to_b : ℝ) :
  wall_length = 1500 ∧ a_to_wall = 400 ∧ wall_to_b = 600 →
  ⌊Real.sqrt (wall_length ^ 2 + (a_to_wall + wall_to_b) ^ 2) + 0.5⌋ = 1803 := by
  sorry

end minimum_race_distance_l3323_332307


namespace chord_length_l3323_332309

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop := ρ - ρ * Real.cos (2 * θ) - 12 * Real.cos θ = 0

-- Define the line l in parametric form
def line_l (t x y : ℝ) : Prop := x = -4/5 * t + 2 ∧ y = 3/5 * t

-- Define the curve C in rectangular coordinates
def curve_C_rect (x y : ℝ) : Prop := y^2 = 6 * x

-- Define the line l in normal form
def line_l_normal (x y : ℝ) : Prop := 3 * x + 4 * y - 6 = 0

-- Theorem statement
theorem chord_length :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  curve_C_rect x₁ y₁ ∧ curve_C_rect x₂ y₂ ∧
  line_l_normal x₁ y₁ ∧ line_l_normal x₂ y₂ ∧
  x₁ ≠ x₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = (20 * Real.sqrt 7) / 3 :=
sorry

end chord_length_l3323_332309


namespace distance_between_locations_l3323_332386

/-- The distance between two locations given the speeds of two vehicles traveling towards each other and the time they take to meet. -/
theorem distance_between_locations (car_speed truck_speed : ℝ) (time : ℝ) : 
  car_speed > 0 → truck_speed > 0 → time > 0 →
  (car_speed + truck_speed) * time = 1925 → car_speed = 100 → truck_speed = 75 → time = 11 :=
by sorry

end distance_between_locations_l3323_332386


namespace lawn_area_20_l3323_332349

/-- The area of a rectangular lawn with given width and length -/
def lawn_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The area of a rectangular lawn with width 5 feet and length 4 feet is 20 square feet -/
theorem lawn_area_20 : lawn_area 5 4 = 20 := by
  sorry

end lawn_area_20_l3323_332349


namespace smallest_n_with_hcf_condition_l3323_332303

theorem smallest_n_with_hcf_condition : 
  ∃ (n : ℕ), n > 0 ∧ n ≠ 11 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n ∧ m ≠ 11 → Nat.gcd (m - 11) (3 * m + 20) = 1) ∧
  Nat.gcd (n - 11) (3 * n + 20) > 1 ∧
  n = 64 :=
by sorry

end smallest_n_with_hcf_condition_l3323_332303


namespace point_division_l3323_332350

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, and P
variable (A B P : V)

-- Define the condition that P is on the line segment AB
def on_line_segment (P A B : V) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the ratio condition
def ratio_condition (P A B : V) : Prop := ∃ (k : ℝ), k > 0 ∧ 2 • (P - A) = k • (B - P) ∧ 7 • (P - A) = k • (B - P)

-- Theorem statement
theorem point_division (h1 : on_line_segment P A B) (h2 : ratio_condition P A B) :
  P = (7/9 : ℝ) • A + (2/9 : ℝ) • B :=
sorry

end point_division_l3323_332350


namespace smallest_two_digit_prime_with_reversed_composite_l3323_332396

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to reverse the digits of a two-digit number
def reverseDigits (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

-- Define a function to check if a number is a two-digit number with 2 as the tens digit
def isTwoDigitWithTensTwo (n : ℕ) : Prop := n ≥ 20 ∧ n < 30

-- Main theorem
theorem smallest_two_digit_prime_with_reversed_composite :
  ∃ (n : ℕ), 
    isPrime n ∧ 
    isTwoDigitWithTensTwo n ∧ 
    ¬(isPrime (reverseDigits n)) ∧
    (∀ m : ℕ, m < n → ¬(isPrime m ∧ isTwoDigitWithTensTwo m ∧ ¬(isPrime (reverseDigits m)))) ∧
    n = 23 := by
  sorry

end smallest_two_digit_prime_with_reversed_composite_l3323_332396


namespace division_result_l3323_332342

theorem division_result : 
  (4 * 10^2011 - 1) / (4 * (3 * (10^2011 - 1) / 9) + 1) = 3 := by sorry

end division_result_l3323_332342


namespace large_glass_cost_l3323_332343

def cost_of_large_glass (initial_money : ℕ) (small_glass_cost : ℕ) (num_small_glasses : ℕ) (num_large_glasses : ℕ) (money_left : ℕ) : ℕ :=
  let money_after_small := initial_money - (small_glass_cost * num_small_glasses)
  let total_large_cost := money_after_small - money_left
  total_large_cost / num_large_glasses

theorem large_glass_cost :
  cost_of_large_glass 50 3 8 5 1 = 5 := by
  sorry

end large_glass_cost_l3323_332343


namespace last_number_is_25_l3323_332310

theorem last_number_is_25 (numbers : Fin 7 → ℝ) : 
  (((numbers 0) + (numbers 1) + (numbers 2) + (numbers 3)) / 4 = 13) →
  (((numbers 3) + (numbers 4) + (numbers 5) + (numbers 6)) / 4 = 15) →
  ((numbers 4) + (numbers 5) + (numbers 6) = 55) →
  ((numbers 3) ^ 2 = numbers 6) →
  (numbers 6 = 25) := by
  sorry

end last_number_is_25_l3323_332310


namespace family_ages_solution_l3323_332315

/-- Represents the ages of the family members -/
structure FamilyAges where
  father : ℕ
  person : ℕ
  sister : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ∃ u : ℕ,
    ages.father + 6 = 3 * (ages.person - u) ∧
    ages.father = ages.person + ages.sister - u ∧
    ages.person = ages.father - u ∧
    ages.father + 19 = 2 * ages.sister

/-- The theorem to be proved -/
theorem family_ages_solution :
  ∃ ages : FamilyAges,
    satisfiesConditions ages ∧
    ages.father = 69 ∧
    ages.person = 47 ∧
    ages.sister = 44 := by
  sorry

end family_ages_solution_l3323_332315


namespace h_function_iff_strictly_increasing_l3323_332332

/-- Definition of an H function -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function is strictly increasing -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- Theorem: A function is an H function if and only if it is strictly increasing -/
theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end h_function_iff_strictly_increasing_l3323_332332


namespace ceil_sum_sqrt_l3323_332371

theorem ceil_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 35⌉ + ⌈Real.sqrt 350⌉ = 27 := by
  sorry

end ceil_sum_sqrt_l3323_332371


namespace proportion_solution_l3323_332364

theorem proportion_solution : ∃ x : ℚ, (1 : ℚ) / 3 = (5 : ℚ) / (3 * x) → x = 5 := by
  sorry

end proportion_solution_l3323_332364


namespace imma_fraction_is_83_125_l3323_332379

/-- Represents the rose distribution problem --/
structure RoseDistribution where
  total_money : ℕ
  rose_price : ℕ
  roses_to_friends : ℕ
  jenna_fraction : ℚ

/-- Calculates the fraction of roses Imma receives --/
def imma_fraction (rd : RoseDistribution) : ℚ :=
  sorry

/-- Theorem stating the fraction of roses Imma receives --/
theorem imma_fraction_is_83_125 (rd : RoseDistribution) 
  (h1 : rd.total_money = 300)
  (h2 : rd.rose_price = 2)
  (h3 : rd.roses_to_friends = 125)
  (h4 : rd.jenna_fraction = 1/3) :
  imma_fraction rd = 83/125 :=
sorry

end imma_fraction_is_83_125_l3323_332379


namespace farm_animals_problem_l3323_332308

/-- Represents the farm animals problem --/
theorem farm_animals_problem (cows ducks pigs : ℕ) : 
  cows = 20 →
  ducks = (3 : ℕ) * cows / 2 →
  cows + ducks + pigs = 60 →
  pigs = (cows + ducks) / 5 :=
by sorry

end farm_animals_problem_l3323_332308


namespace age_problem_l3323_332378

/-- Given the ages of five people a, b, c, d, and e satisfying certain conditions,
    prove that b is 16 years old. -/
theorem age_problem (a b c d e : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = c / 2 →
  e = d - 3 →
  a + b + c + d + e = 52 →
  b = 16 := by
  sorry

end age_problem_l3323_332378


namespace distance_between_points_l3323_332380

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (-2, -3)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 34 := by
  sorry

end distance_between_points_l3323_332380


namespace parabola_vertex_l3323_332324

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex coordinates of the parabola y = 2(x-3)^2 + 1 are (3, 1) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end parabola_vertex_l3323_332324


namespace west_movement_notation_l3323_332312

/-- Represents the direction of movement -/
inductive Direction
  | East
  | West

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Converts a movement to its numerical representation -/
def toNumber (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.distance
  | Direction.West => -m.distance

theorem west_movement_notation :
  let eastMovement : Movement := ⟨5, Direction.East⟩
  let westMovement : Movement := ⟨3, Direction.West⟩
  toNumber eastMovement = 5 →
  toNumber westMovement = -3 := by
  sorry

end west_movement_notation_l3323_332312


namespace vector_angle_cosine_l3323_332341

theorem vector_angle_cosine (α β : Real) (a b : ℝ × ℝ) :
  -π/2 < α ∧ α < 0 ∧ 0 < β ∧ β < π/2 →
  a = (Real.cos α, Real.sin α) →
  b = (Real.cos β, Real.sin β) →
  ‖a - b‖ = Real.sqrt 10 / 5 →
  Real.cos α = 12/13 →
  Real.cos (α - β) = 4/5 ∧ Real.cos β = 63/65 := by
  sorry

end vector_angle_cosine_l3323_332341


namespace hyperbola_asymptotic_lines_l3323_332359

/-- The asymptotic lines of the hyperbola 3x^2 - y^2 = 3 are y = ± √3 x -/
theorem hyperbola_asymptotic_lines :
  let hyperbola := {(x, y) : ℝ × ℝ | 3 * x^2 - y^2 = 3}
  let asymptotic_lines := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x}
  (Set.range fun t : ℝ => (t, Real.sqrt 3 * t)) ∪ (Set.range fun t : ℝ => (t, -Real.sqrt 3 * t)) =
    {p | p ∈ asymptotic_lines ∧ p ∉ hyperbola ∧ ∀ ε > 0, ∃ q ∈ hyperbola, dist p q < ε} := by
  sorry


end hyperbola_asymptotic_lines_l3323_332359


namespace complex_fraction_simplification_l3323_332326

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7*I
  let z₂ : ℂ := 4 - 7*I
  (z₁ / z₂) - (z₂ / z₁) = 112 * I / 65 := by sorry

end complex_fraction_simplification_l3323_332326


namespace diophantine_equation_solution_l3323_332336

theorem diophantine_equation_solution :
  ∀ x y : ℕ+,
  let d := Nat.gcd x.val y.val
  x.val * y.val * d = x.val + y.val + d^2 →
  (x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2) :=
by sorry

end diophantine_equation_solution_l3323_332336


namespace marie_erasers_l3323_332301

/-- Given Marie's eraser situation, prove that she ends up with 755 erasers. -/
theorem marie_erasers (initial : ℕ) (lost : ℕ) (packs_bought : ℕ) (erasers_per_pack : ℕ) 
  (h1 : initial = 950)
  (h2 : lost = 420)
  (h3 : packs_bought = 3)
  (h4 : erasers_per_pack = 75) : 
  initial - lost + packs_bought * erasers_per_pack = 755 := by
  sorry

#check marie_erasers

end marie_erasers_l3323_332301


namespace existence_of_even_floor_l3323_332390

theorem existence_of_even_floor (n : ℕ) : ∃ k ∈ Finset.range (n + 1), Even (⌊(2 ^ (n + k) : ℝ) * Real.sqrt 2⌋) := by
  sorry

end existence_of_even_floor_l3323_332390


namespace abs_sum_inequality_sum_bound_from_square_sum_l3323_332388

-- Part I
theorem abs_sum_inequality (x a : ℝ) (ha : a > 0) :
  |x - 1/a| + |x + a| ≥ 2 := by sorry

-- Part II
theorem sum_bound_from_square_sum (x y z : ℝ) (h : x^2 + 4*y^2 + z^2 = 3) :
  |x + 2*y + z| ≤ 3 := by sorry

end abs_sum_inequality_sum_bound_from_square_sum_l3323_332388


namespace motel_flat_fee_calculation_l3323_332361

/-- A motel charging system with a flat fee for the first night and a fixed amount for additional nights. -/
structure MotelCharge where
  flatFee : ℕ  -- Flat fee for the first night
  nightlyRate : ℕ  -- Fixed amount for each additional night

/-- Calculates the total cost for a given number of nights -/
def totalCost (charge : MotelCharge) (nights : ℕ) : ℕ :=
  charge.flatFee + (nights - 1) * charge.nightlyRate

theorem motel_flat_fee_calculation (charge : MotelCharge) :
  totalCost charge 3 = 155 → totalCost charge 6 = 290 → charge.flatFee = 65 := by
  sorry

#check motel_flat_fee_calculation

end motel_flat_fee_calculation_l3323_332361


namespace integer_solutions_of_system_l3323_332372

theorem integer_solutions_of_system : 
  ∀ (x y z t : ℤ), 
    (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
    ((x, y, z, t) = (1, 0, 3, 1) ∨ 
     (x, y, z, t) = (-1, 0, -3, -1) ∨ 
     (x, y, z, t) = (3, 1, 1, 0) ∨ 
     (x, y, z, t) = (-3, -1, -1, 0)) :=
by sorry

end integer_solutions_of_system_l3323_332372


namespace fence_cost_square_plot_l3323_332399

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 58) :
  let side_length : ℝ := Real.sqrt area
  let perimeter : ℝ := 4 * side_length
  let total_cost : ℝ := perimeter * price_per_foot
  total_cost = 3944 := by
sorry

end fence_cost_square_plot_l3323_332399


namespace right_angles_in_two_days_l3323_332391

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour_hand : ℕ)
  (minute_hand : ℕ)

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Represents the number of right angles formed by clock hands in one day -/
def right_angles_per_day : ℕ := 44

/-- Checks if the clock hands form a right angle -/
def is_right_angle (c : Clock) : Prop :=
  (c.minute_hand - c.hour_hand) % 60 = 15 ∨ (c.hour_hand - c.minute_hand) % 60 = 15

/-- The main theorem: In 2 days, clock hands form a right angle 88 times -/
theorem right_angles_in_two_days :
  (2 * right_angles_per_day = 88) ∧
  (∀ t : ℕ, t < 2 * minutes_per_day →
    (∃ c : Clock, c.hour_hand = t % 720 ∧ c.minute_hand = t % 60 ∧
      is_right_angle c) ↔ t % (minutes_per_day / right_angles_per_day) = 0) :=
sorry

end right_angles_in_two_days_l3323_332391


namespace boys_not_adjacent_girls_adjacent_girls_not_at_ends_l3323_332397

/-- The number of boys in the group -/
def num_boys : Nat := 3

/-- The number of girls in the group -/
def num_girls : Nat := 2

/-- The total number of people in the group -/
def total_people : Nat := num_boys + num_girls

/-- Calculates the number of ways to arrange n distinct objects -/
def permutations (n : Nat) : Nat := Nat.factorial n

/-- Theorem stating the number of ways boys are not adjacent -/
theorem boys_not_adjacent : 
  permutations num_girls * permutations num_boys = 12 := by sorry

/-- Theorem stating the number of ways girls are adjacent -/
theorem girls_adjacent : 
  permutations (total_people - num_girls + 1) * permutations num_girls = 48 := by sorry

/-- Theorem stating the number of ways girls are not at the ends -/
theorem girls_not_at_ends : 
  (total_people - 2) * permutations num_boys = 36 := by sorry

end boys_not_adjacent_girls_adjacent_girls_not_at_ends_l3323_332397


namespace max_area_rectangular_pen_l3323_332334

/-- Given 60 feet of fencing for a rectangular pen, the maximum possible area is 225 square feet -/
theorem max_area_rectangular_pen (perimeter : ℝ) (area : ℝ → ℝ → ℝ) :
  perimeter = 60 →
  (∀ x y, x > 0 → y > 0 → x + y = perimeter / 2 → area x y = x * y) →
  (∃ x y, x > 0 ∧ y > 0 ∧ x + y = perimeter / 2 ∧ area x y = 225) ∧
  (∀ x y, x > 0 → y > 0 → x + y = perimeter / 2 → area x y ≤ 225) :=
by sorry

end max_area_rectangular_pen_l3323_332334


namespace absolute_value_inequality_l3323_332320

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x - 4| > 6 ↔ x < 0 ∨ x > 12 := by
  sorry

end absolute_value_inequality_l3323_332320


namespace banana_price_reduction_l3323_332360

/-- Given a 50% reduction in banana prices allows buying 80 more dozens for 60000.25 rupees,
    prove the reduced price per dozen is 375.0015625 rupees. -/
theorem banana_price_reduction (original_price : ℝ) : 
  (2 * 60000.25 / original_price - 60000.25 / original_price = 80) → 
  (original_price / 2 = 375.0015625) :=
by
  sorry

end banana_price_reduction_l3323_332360


namespace flowchart_result_for_6_l3323_332395

-- Define the function that represents the flowchart logic
def flowchart_program (n : ℕ) : ℕ :=
  -- The actual implementation is not provided, so we'll use a placeholder
  sorry

-- Theorem statement
theorem flowchart_result_for_6 : flowchart_program 6 = 2 := by
  sorry

end flowchart_result_for_6_l3323_332395


namespace intersection_M_N_l3323_332325

def M : Set ℝ := {x : ℝ | (x - 3) / (x + 1) ≤ 0}

def N : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_M_N : M ∩ N = {1, 3} := by
  sorry

end intersection_M_N_l3323_332325


namespace potatoes_remaining_l3323_332367

/-- Calculates the number of potatoes left after distribution -/
def potatoes_left (total : ℕ) (to_gina : ℕ) : ℕ :=
  let to_tom := 2 * to_gina
  let to_anne := to_tom / 3
  total - (to_gina + to_tom + to_anne)

/-- Theorem stating that 47 potatoes are left after distribution -/
theorem potatoes_remaining : potatoes_left 300 69 = 47 := by
  sorry

end potatoes_remaining_l3323_332367


namespace jill_and_emily_total_l3323_332329

/-- The number of peaches each person has -/
structure Peaches where
  steven : ℕ
  jake : ℕ
  jill : ℕ
  maria : ℕ
  emily : ℕ

/-- The conditions of the peach distribution problem -/
def peach_conditions (p : Peaches) : Prop :=
  p.steven = 14 ∧
  p.jake = p.steven - 6 ∧
  p.jake = p.jill + 3 ∧
  p.maria = 2 * p.jake ∧
  p.emily = p.maria - 9

/-- The theorem stating that Jill and Emily have 12 peaches in total -/
theorem jill_and_emily_total (p : Peaches) (h : peach_conditions p) : 
  p.jill + p.emily = 12 := by
  sorry

end jill_and_emily_total_l3323_332329


namespace jame_card_tearing_l3323_332357

/-- The number of cards in a new deck -/
def cards_per_deck : ℕ := 55

/-- The number of times Jame tears cards per week -/
def tear_times_per_week : ℕ := 3

/-- The number of decks Jame buys -/
def decks_bought : ℕ := 18

/-- The number of weeks Jame can go with the bought decks -/
def weeks_lasted : ℕ := 11

/-- The number of cards Jame can tear at a time -/
def cards_torn_at_once : ℕ := decks_bought * cards_per_deck / (weeks_lasted * tear_times_per_week)

theorem jame_card_tearing :
  cards_torn_at_once = 30 :=
sorry

end jame_card_tearing_l3323_332357


namespace unique_m_value_l3323_332389

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {m, 1}
  let B : Set ℝ := {m^2, -1}
  A = B → m = -1 := by
  sorry

end unique_m_value_l3323_332389


namespace group_dance_arrangements_l3323_332384

/-- The number of boys in the group dance -/
def num_boys : ℕ := 10

/-- The number of girls in the group dance -/
def num_girls : ℕ := 10

/-- The total number of people in the group dance -/
def total_people : ℕ := num_boys + num_girls

/-- The number of columns in the group dance -/
def num_columns : ℕ := 2

/-- The number of arrangements when boys and girls are in separate columns -/
def separate_columns_arrangements : ℕ := 2 * (Nat.factorial num_boys)^2

/-- The number of arrangements when boys and girls can stand in any column -/
def mixed_columns_arrangements : ℕ := Nat.factorial total_people

/-- The number of pairings when boys and girls are in separate columns and internal order doesn't matter -/
def pairings_separate_columns : ℕ := 2 * Nat.factorial num_boys

theorem group_dance_arrangements :
  (separate_columns_arrangements = 2 * (Nat.factorial num_boys)^2) ∧
  (mixed_columns_arrangements = Nat.factorial total_people) ∧
  (pairings_separate_columns = 2 * Nat.factorial num_boys) :=
sorry

end group_dance_arrangements_l3323_332384


namespace gdp_growth_problem_l3323_332351

/-- Calculates the GDP after compound growth -/
def gdp_growth (initial_gdp : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_gdp * (1 + growth_rate) ^ years

/-- The GDP growth problem -/
theorem gdp_growth_problem :
  let initial_gdp : ℝ := 9593.3
  let growth_rate : ℝ := 0.073
  let years : ℕ := 4
  let final_gdp : ℝ := gdp_growth initial_gdp growth_rate years
  ∃ ε > 0, |final_gdp - 127254| < ε :=
by
  sorry

end gdp_growth_problem_l3323_332351


namespace one_black_one_white_probability_l3323_332377

/-- The probability of picking one black ball and one white ball from a jar -/
theorem one_black_one_white_probability (black_balls white_balls : ℕ) : 
  black_balls = 5 → white_balls = 2 → 
  (black_balls * white_balls : ℚ) / ((black_balls + white_balls) * (black_balls + white_balls - 1) / 2) = 10/21 := by
sorry

end one_black_one_white_probability_l3323_332377


namespace sqrt_81_div_3_equals_3_l3323_332317

theorem sqrt_81_div_3_equals_3 : Real.sqrt 81 / 3 = 3 := by
  sorry

end sqrt_81_div_3_equals_3_l3323_332317


namespace sum_of_decimals_l3323_332368

theorem sum_of_decimals : 0.4 + 0.02 + 0.006 = 0.426 := by
  sorry

end sum_of_decimals_l3323_332368


namespace united_additional_charge_is_correct_l3323_332365

/-- The additional charge per minute for United Telephone -/
def united_additional_charge : ℚ := 1/4

/-- United Telephone's base rate -/
def united_base_rate : ℚ := 6

/-- Atlantic Call's base rate -/
def atlantic_base_rate : ℚ := 12

/-- Atlantic Call's additional charge per minute -/
def atlantic_additional_charge : ℚ := 1/5

/-- The number of minutes at which both companies' bills are equal -/
def equal_minutes : ℕ := 120

theorem united_additional_charge_is_correct : 
  united_base_rate + equal_minutes * united_additional_charge = 
  atlantic_base_rate + equal_minutes * atlantic_additional_charge :=
by sorry

end united_additional_charge_is_correct_l3323_332365


namespace interesting_factor_exists_l3323_332376

/-- A natural number is interesting if it can be represented both as the sum of two consecutive integers and as the sum of three consecutive integers. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ k m : ℤ, n = (k + (k + 1)) ∧ n = (m - 1 + m + (m + 1))

/-- The theorem states that if the product of five different natural numbers is interesting,
    then at least one of these natural numbers is interesting. -/
theorem interesting_factor_exists (a b c d e : ℕ) (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
    (h_interesting : is_interesting (a * b * c * d * e)) :
    is_interesting a ∨ is_interesting b ∨ is_interesting c ∨ is_interesting d ∨ is_interesting e :=
  sorry

end interesting_factor_exists_l3323_332376


namespace min_negations_for_zero_sum_l3323_332333

def clock_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def sum_list (l : List ℤ) : ℤ := l.foldl (· + ·) 0

def negate_elements (l : List ℕ) (indices : List ℕ) : List ℤ :=
  l.enum.map (fun (i, x) => if i + 1 ∈ indices then -x else x)

theorem min_negations_for_zero_sum :
  ∃ (indices : List ℕ),
    (indices.length = 4) ∧
    (sum_list (negate_elements clock_numbers indices) = 0) ∧
    (∀ (other_indices : List ℕ),
      sum_list (negate_elements clock_numbers other_indices) = 0 →
      other_indices.length ≥ 4) :=
sorry

end min_negations_for_zero_sum_l3323_332333


namespace wire_cutting_l3323_332358

theorem wire_cutting (total_length : ℝ) (ratio : ℚ) (shorter_piece : ℝ) :
  total_length = 60 →
  ratio = 2/4 →
  shorter_piece + ratio * shorter_piece = total_length →
  shorter_piece = 20 := by
  sorry

end wire_cutting_l3323_332358


namespace parallel_line_equation_l3323_332300

/-- Given a point and a line, find the equation of a parallel line passing through the point. -/
theorem parallel_line_equation (x₀ y₀ : ℝ) (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ k : ℝ, ∀ x y : ℝ, (x = x₀ ∧ y = y₀) ∨ (a * x + b * y + k = 0) ↔ 
  (x = -3 ∧ y = -1) ∨ (x - 3 * y = 0) :=
sorry

end parallel_line_equation_l3323_332300


namespace floor_sufficiency_not_necessity_l3323_332394

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem floor_sufficiency_not_necessity :
  (∀ x y : ℝ, floor x = floor y → |x - y| < 1) ∧
  (∃ x y : ℝ, |x - y| < 1 ∧ floor x ≠ floor y) :=
by sorry

end floor_sufficiency_not_necessity_l3323_332394


namespace right_triangle_inequality_l3323_332393

theorem right_triangle_inequality (a b c : ℝ) (n : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_n_ge_3 : n ≥ 3) : 
  a^n + b^n < c^n := by
sorry

end right_triangle_inequality_l3323_332393


namespace polygon_contains_integer_different_points_l3323_332346

/-- A polygon on the coordinate plane. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon,
  -- just its existence and area property
  area : ℝ

/-- Two points are integer-different if their coordinate differences are integers. -/
def integer_different (p₁ p₂ : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p₁.1 - p₂.1 = m ∧ p₁.2 - p₂.2 = n

/-- Main theorem: If a polygon has area greater than 1, it contains two integer-different points. -/
theorem polygon_contains_integer_different_points (P : Polygon) (h : P.area > 1) :
  ∃ (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ integer_different p₁ p₂ := by
  sorry

end polygon_contains_integer_different_points_l3323_332346


namespace trouser_original_price_l3323_332323

/-- 
Given a trouser with a sale price of $70 after a 30% decrease,
prove that its original price was $100.
-/
theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) : 
  sale_price = 70 → 
  discount_percentage = 30 → 
  sale_price = (1 - discount_percentage / 100) * 100 :=
by
  sorry

end trouser_original_price_l3323_332323


namespace unique_prime_sum_of_squares_l3323_332355

theorem unique_prime_sum_of_squares (p k x y a b : ℤ) : 
  Prime p → 
  p = 4 * k + 1 → 
  p = x^2 + y^2 → 
  p = a^2 + b^2 → 
  (x = a ∧ y = b) ∨ (x = -a ∧ y = -b) ∨ (x = b ∧ y = -a) ∨ (x = -b ∧ y = a) :=
sorry

end unique_prime_sum_of_squares_l3323_332355


namespace triangle_angle_determinant_l3323_332353

/-- Given angles α, β, γ of a triangle, the determinant of the matrix
    | tan α   sin α cos α   1 |
    | tan β   sin β cos β   1 |
    | tan γ   sin γ cos γ   1 |
    is equal to 0. -/
theorem triangle_angle_determinant (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.tan α, Real.sin α * Real.cos α, 1],
    ![Real.tan β, Real.sin β * Real.cos β, 1],
    ![Real.tan γ, Real.sin γ * Real.cos γ, 1]
  ]
  Matrix.det M = 0 := by sorry

end triangle_angle_determinant_l3323_332353


namespace arithmetic_sequence_properties_l3323_332338

/-- An arithmetic sequence with given first term and 17th term -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 17 = 66 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_properties (a : ℕ → ℤ) 
  (h : arithmetic_sequence a) : 
  (∀ n : ℕ, a n = 4 * n - 2) ∧ 
  (¬ ∃ n : ℕ, a n = 88) := by
  sorry

end arithmetic_sequence_properties_l3323_332338


namespace parentheses_removal_equivalence_l3323_332375

theorem parentheses_removal_equivalence (x : ℝ) : 
  (3 * x + 2) - 2 * (2 * x - 1) = 3 * x + 2 - 4 * x + 2 := by
  sorry

end parentheses_removal_equivalence_l3323_332375


namespace train_length_calculation_l3323_332302

/-- The length of two trains given their speeds and overtaking time -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) (overtake_time : ℝ) : 
  faster_speed = 46 →
  slower_speed = 36 →
  overtake_time = 45 →
  ∃ (train_length : ℝ), train_length = 62.5 := by
  sorry


end train_length_calculation_l3323_332302


namespace interview_pass_probability_l3323_332311

-- Define the number of questions
def num_questions : ℕ := 3

-- Define the probability of answering a question correctly
def prob_correct : ℝ := 0.7

-- Define the number of attempts per question
def num_attempts : ℕ := 3

-- Theorem statement
theorem interview_pass_probability :
  1 - (1 - prob_correct) ^ num_attempts = 0.973 := by
  sorry

end interview_pass_probability_l3323_332311


namespace bishop_white_invariant_l3323_332318

/-- Represents a position on a chessboard -/
structure Position where
  i : Nat
  j : Nat
  h_valid : i < 8 ∧ j < 8

/-- Checks if a position is on a white square -/
def isWhite (p : Position) : Prop :=
  (p.i + p.j) % 2 = 1

/-- Represents a valid bishop move -/
inductive BishopMove : Position → Position → Prop where
  | diag (p q : Position) (k : Int) :
      q.i = p.i + k ∧ q.j = p.j + k → BishopMove p q

theorem bishop_white_invariant (p q : Position) (h : BishopMove p q) :
  isWhite p → isWhite q := by
  sorry

end bishop_white_invariant_l3323_332318


namespace direct_variation_theorem_y_value_at_negative_ten_l3323_332381

/-- A function representing direct variation --/
def DirectVariation (k : ℝ) : ℝ → ℝ := fun x ↦ k * x

theorem direct_variation_theorem (k : ℝ) :
  (DirectVariation k 5 = 15) → (DirectVariation k (-10) = -30) := by
  sorry

/-- Main theorem proving the relationship between y and x --/
theorem y_value_at_negative_ten :
  ∃ k : ℝ, (DirectVariation k 5 = 15) ∧ (DirectVariation k (-10) = -30) := by
  sorry

end direct_variation_theorem_y_value_at_negative_ten_l3323_332381


namespace christine_wandering_time_l3323_332305

/-- Given a distance of 20 miles and a speed of 4 miles per hour, 
    the time taken is 5 hours. -/
theorem christine_wandering_time :
  let distance : ℝ := 20
  let speed : ℝ := 4
  let time := distance / speed
  time = 5 := by sorry

end christine_wandering_time_l3323_332305


namespace largest_multiple_18_with_9_0_is_correct_division_result_l3323_332335

/-- The largest multiple of 18 with digits 9 or 0 -/
def largest_multiple_18_with_9_0 : ℕ := 9990

/-- Check if a natural number consists only of digits 9 and 0 -/
def has_only_9_and_0_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 9 ∨ d = 0

theorem largest_multiple_18_with_9_0_is_correct :
  largest_multiple_18_with_9_0 % 18 = 0 ∧
  has_only_9_and_0_digits largest_multiple_18_with_9_0 ∧
  ∀ m : ℕ, m > largest_multiple_18_with_9_0 →
    m % 18 ≠ 0 ∨ ¬(has_only_9_and_0_digits m) :=
by sorry

theorem division_result :
  largest_multiple_18_with_9_0 / 18 = 555 :=
by sorry

end largest_multiple_18_with_9_0_is_correct_division_result_l3323_332335


namespace circle_properties_l3323_332354

/-- Circle with center (6,8) and radius 10 -/
def Circle := {p : ℝ × ℝ | (p.1 - 6)^2 + (p.2 - 8)^2 = 100}

/-- The circle passes through the origin -/
axiom origin_on_circle : (0, 0) ∈ Circle

/-- P is the point where the circle intersects the x-axis -/
def P : ℝ × ℝ := (12, 0)

/-- Q is the point on the circle with maximum y-coordinate -/
def Q : ℝ × ℝ := (6, 18)

/-- R is the point on the circle forming a right angle with P and Q -/
def R : ℝ × ℝ := (0, 16)

/-- S and T are the points on the circle forming 45-degree angles with P and Q -/
def S : ℝ × ℝ := (14, 14)
def T : ℝ × ℝ := (-2, 2)

theorem circle_properties :
  P ∈ Circle ∧
  Q ∈ Circle ∧
  R ∈ Circle ∧
  S ∈ Circle ∧
  T ∈ Circle ∧
  P.2 = 0 ∧
  ∀ p ∈ Circle, p.2 ≤ Q.2 ∧
  (R.1 - Q.1) * (P.1 - Q.1) + (R.2 - Q.2) * (P.2 - Q.2) = 0 ∧
  (S.1 - Q.1) * (P.1 - Q.1) + (S.2 - Q.2) * (P.2 - Q.2) =
    (T.1 - Q.1) * (P.1 - Q.1) + (T.2 - Q.2) * (P.2 - Q.2) :=
by sorry

end circle_properties_l3323_332354


namespace geometric_arithmetic_progression_existence_l3323_332319

theorem geometric_arithmetic_progression_existence :
  ∃ (q : ℝ) (i j k : ℕ), 
    1 < q ∧ 
    i < j ∧ j < k ∧ 
    q^j - q^i = q^k - q^j ∧
    1.9 < q :=
by sorry

end geometric_arithmetic_progression_existence_l3323_332319


namespace rectangle_length_l3323_332331

theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 9 →
  rect_width = 3 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 27 := by
sorry

end rectangle_length_l3323_332331


namespace work_earnings_equation_l3323_332322

theorem work_earnings_equation (t : ℚ) : 
  (t + 2) * (4 * t - 4) = (4 * t - 2) * (t + 3) + 3 → t = -14/9 := by
  sorry

end work_earnings_equation_l3323_332322


namespace alice_nike_sales_alice_nike_sales_proof_l3323_332321

/-- Proves that Alice sold 8 Nike shoes given the problem conditions -/
theorem alice_nike_sales : Int → Prop :=
  fun x =>
    let quota : Int := 1000
    let adidas_price : Int := 45
    let nike_price : Int := 60
    let reebok_price : Int := 35
    let adidas_sold : Int := 6
    let reebok_sold : Int := 9
    let over_goal : Int := 65
    (adidas_price * adidas_sold + nike_price * x + reebok_price * reebok_sold = quota + over_goal) →
    x = 8

/-- Proof of the theorem -/
theorem alice_nike_sales_proof : ∃ x, alice_nike_sales x :=
  sorry

end alice_nike_sales_alice_nike_sales_proof_l3323_332321


namespace prob_ace_ten_jack_standard_deck_l3323_332345

/-- Represents a standard deck of 52 playing cards. -/
structure Deck :=
  (cards : Nat)
  (aces : Nat)
  (tens : Nat)
  (jacks : Nat)

/-- The probability of drawing an Ace, then a 10, then a Jack from a standard 52-card deck without replacement. -/
def prob_ace_ten_jack (d : Deck) : ℚ :=
  (d.aces : ℚ) / d.cards *
  (d.tens : ℚ) / (d.cards - 1) *
  (d.jacks : ℚ) / (d.cards - 2)

/-- Theorem stating that the probability of drawing an Ace, then a 10, then a Jack
    from a standard 52-card deck without replacement is 8/16575. -/
theorem prob_ace_ten_jack_standard_deck :
  prob_ace_ten_jack ⟨52, 4, 4, 4⟩ = 8 / 16575 := by
  sorry

end prob_ace_ten_jack_standard_deck_l3323_332345


namespace circle_C_properties_l3323_332370

def circle_C (ρ θ : ℝ) : Prop :=
  ρ^2 = 4*ρ*(Real.cos θ + Real.sin θ) - 3

def point_on_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = 2 + Real.sqrt 5 * Real.cos θ ∧ y = 2 + Real.sqrt 5 * Real.sin θ

theorem circle_C_properties :
  -- 1. Parametric equations
  (∀ x y θ : ℝ, point_on_C x y ↔ 
    x = 2 + Real.sqrt 5 * Real.cos θ ∧ 
    y = 2 + Real.sqrt 5 * Real.sin θ) ∧
  -- 2. Maximum value of x + 2y
  (∀ x y : ℝ, point_on_C x y → x + 2*y ≤ 11) ∧
  -- 3. Coordinates at maximum
  (point_on_C 3 4 ∧ 3 + 2*4 = 11) :=
by sorry

end circle_C_properties_l3323_332370


namespace total_shoes_l3323_332347

def scott_shoes : ℕ := 7

def anthony_shoes : ℕ := 3 * scott_shoes

def jim_shoes : ℕ := anthony_shoes - 2

def melissa_shoes : ℕ := jim_shoes / 2

def tim_shoes : ℕ := (anthony_shoes + melissa_shoes) / 2

theorem total_shoes : scott_shoes + anthony_shoes + jim_shoes + melissa_shoes + tim_shoes = 71 := by
  sorry

end total_shoes_l3323_332347


namespace arctan_sum_equation_l3323_332330

theorem arctan_sum_equation (x : ℝ) : 
  3 * Real.arctan (1/4) + Real.arctan (1/20) + Real.arctan (1/x) = π/4 → x = 1985 := by
  sorry

end arctan_sum_equation_l3323_332330


namespace triangle_circumradius_l3323_332398

/-- Given a triangle with side lengths 8, 15, and 17, its circumradius is 8.5 -/
theorem triangle_circumradius : ∀ (a b c : ℝ), 
  a = 8 ∧ b = 15 ∧ c = 17 →
  (a^2 + b^2 = c^2) →
  (c / 2 = 8.5) := by
  sorry

#check triangle_circumradius

end triangle_circumradius_l3323_332398
