import Mathlib

namespace golden_ratio_percentage_l3886_388628

theorem golden_ratio_percentage (a b : ℝ) (h : a > 0) (h' : b > 0) :
  b / a = a / (a + b) → b / a = (Real.sqrt 5 - 1) / 2 := by
  sorry

end golden_ratio_percentage_l3886_388628


namespace count_primes_with_digit_sum_10_l3886_388610

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem count_primes_with_digit_sum_10 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_condition n) ∧ S.card = 3 :=
sorry

end count_primes_with_digit_sum_10_l3886_388610


namespace five_balls_four_boxes_l3886_388654

/-- The number of ways to place n distinct objects into k distinct containers -/
def placement_count (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to place 5 distinct balls into 4 distinct boxes -/
theorem five_balls_four_boxes : placement_count 5 4 = 1024 := by sorry

end five_balls_four_boxes_l3886_388654


namespace max_value_of_f_min_value_of_f_in_interval_range_of_a_l3886_388621

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 5) / Real.exp x

theorem max_value_of_f :
  ∃ (x : ℝ), f x = 5 ∧ ∀ (y : ℝ), f y ≤ 5 :=
sorry

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ≤ 0 ∧ f x = -Real.exp 3 ∧ ∀ (y : ℝ), y ≤ 0 → f y ≥ -Real.exp 3 :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x^2 + 5*x + 5 - a * Real.exp x ≥ 0) ↔ a ≤ -Real.exp 3 :=
sorry

end max_value_of_f_min_value_of_f_in_interval_range_of_a_l3886_388621


namespace library_shelf_count_l3886_388690

theorem library_shelf_count (notebooks : ℕ) (pen_difference : ℕ) : 
  notebooks = 30 → pen_difference = 50 → notebooks + (notebooks + pen_difference) = 110 :=
by sorry

end library_shelf_count_l3886_388690


namespace point_coordinates_wrt_origin_l3886_388634

/-- In a Cartesian coordinate system, the coordinates of the point (11, 9) with respect to the origin are (11, 9). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (11, 9)
  P = P :=
by sorry

end point_coordinates_wrt_origin_l3886_388634


namespace f_sum_negative_l3886_388696

def f (x : ℝ) : ℝ := 2 * x^3 + 4 * x

theorem f_sum_negative (a b c : ℝ) 
  (hab : a + b < 0) (hbc : b + c < 0) (hca : c + a < 0) : 
  f a + f b + f c < 0 := by
  sorry

end f_sum_negative_l3886_388696


namespace characterization_of_matrices_with_power_in_S_l3886_388624

-- Define the set S
def S : Set (Matrix (Fin 2) (Fin 2) ℝ) :=
  {M | ∃ (a r : ℝ), M = !![a, a+r; a+2*r, a+3*r]}

-- Define the property of M^k being in S for some k > 1
def has_power_in_S (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ (M ^ k) ∈ S

-- Main theorem
theorem characterization_of_matrices_with_power_in_S :
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ),
  M ∈ S → (has_power_in_S M ↔ 
    (∃ (c : ℝ), M = c • !![1, 1; 1, 1]) ∨
    (∃ (c : ℝ), M = c • !![-3, -1; 1, 3])) :=
by sorry

end characterization_of_matrices_with_power_in_S_l3886_388624


namespace line_through_circle_center_l3886_388682

/-- Given a line and a circle, if the line passes through the center of the circle,
    then the value of m in the line equation is 0. -/
theorem line_through_circle_center (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y = 0 → 
    ∃ h k : ℝ, (h - 1)^2 + (k + 2)^2 = 0 ∧ 2*h + k + m = 0) → 
  m = 0 := by
sorry

end line_through_circle_center_l3886_388682


namespace arc_length_calculation_l3886_388669

theorem arc_length_calculation (r α : Real) (h1 : r = π) (h2 : α = 2 * π / 3) :
  r * α = (2 / 3) * π^2 := by
  sorry

end arc_length_calculation_l3886_388669


namespace sports_activity_division_l3886_388662

theorem sports_activity_division :
  ∀ (a b c : ℕ),
    a + b + c = 48 →
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (∃ (x : ℕ), a = 10 * x + 6) →
    (∃ (y : ℕ), b = 10 * y + 6) →
    (∃ (z : ℕ), c = 10 * z + 6) →
    (a = 6 ∧ b = 16 ∧ c = 26) ∨ (a = 6 ∧ b = 26 ∧ c = 16) ∨
    (a = 16 ∧ b = 6 ∧ c = 26) ∨ (a = 16 ∧ b = 26 ∧ c = 6) ∨
    (a = 26 ∧ b = 6 ∧ c = 16) ∨ (a = 26 ∧ b = 16 ∧ c = 6) :=
by sorry


end sports_activity_division_l3886_388662


namespace sphere_volume_equals_area_l3886_388699

theorem sphere_volume_equals_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
  sorry

end sphere_volume_equals_area_l3886_388699


namespace activity_participation_l3886_388604

theorem activity_participation (total : ℕ) (books songs movies : ℕ) 
  (books_songs books_movies songs_movies : ℕ) (all_three : ℕ) : 
  total = 200 → 
  books = 80 → 
  songs = 60 → 
  movies = 30 → 
  books_songs = 25 → 
  books_movies = 15 → 
  songs_movies = 20 → 
  all_three = 10 → 
  books + songs + movies - books_songs - books_movies - songs_movies + all_three = 120 :=
by sorry

end activity_participation_l3886_388604


namespace inequality_systems_solution_l3886_388686

theorem inequality_systems_solution :
  (∀ x : ℝ, (2 * x ≥ x - 1 ∧ 4 * x + 10 > x + 1) ↔ x ≥ -1) ∧
  (∀ x : ℝ, (2 * x - 7 < 5 - 2 * x ∧ x / 4 - 1 ≤ (x - 1) / 2) ↔ -2 ≤ x ∧ x < 3) :=
by sorry

end inequality_systems_solution_l3886_388686


namespace tom_spending_l3886_388612

def apple_count : ℕ := 4
def egg_count : ℕ := 6
def bread_count : ℕ := 3
def cheese_count : ℕ := 2
def chicken_count : ℕ := 1

def apple_price : ℚ := 1
def egg_price : ℚ := 0.5
def bread_price : ℚ := 3
def cheese_price : ℚ := 6
def chicken_price : ℚ := 8

def coupon_threshold : ℚ := 40
def coupon_value : ℚ := 10

def total_cost : ℚ :=
  apple_count * apple_price +
  egg_count * egg_price +
  bread_count * bread_price +
  cheese_count * cheese_price +
  chicken_count * chicken_price

theorem tom_spending :
  (if total_cost ≥ coupon_threshold then total_cost - coupon_value else total_cost) = 36 := by
  sorry

end tom_spending_l3886_388612


namespace triangle_xz_interval_l3886_388609

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the point W on YZ
def W (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Define the angle bisector
def is_angle_bisector (t : Triangle) (w : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_xz_interval (t : Triangle) :
  length t.X t.Y = 8 →
  is_angle_bisector t (W t) →
  length (W t) t.Z = 5 →
  perimeter t = 24 →
  ∃ m n : ℝ, m < n ∧ 
    (∀ xz : ℝ, m < xz ∧ xz < n ↔ length t.X t.Z = xz) ∧
    m + n = 13 := by
  sorry

end triangle_xz_interval_l3886_388609


namespace triangle_cosA_value_l3886_388629

theorem triangle_cosA_value (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  b = Real.sqrt 2 * c →  -- Given condition
  Real.sin A + Real.sqrt 2 * Real.sin C = 2 * Real.sin B →  -- Given condition
  -- Triangle inequality (to ensure it's a valid triangle)
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Law of sines (to connect side lengths and angles)
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  Real.cos A = Real.sqrt 2 / 4 :=
by
  sorry


end triangle_cosA_value_l3886_388629


namespace purple_balls_count_l3886_388623

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (p : ℚ) :
  total = 100 ∧
  white = 10 ∧
  green = 30 ∧
  yellow = 10 ∧
  red = 47 ∧
  p = 1/2 ∧
  p = (white + green + yellow : ℚ) / total →
  ∃ purple : ℕ, purple = 3 ∧ total = white + green + yellow + red + purple :=
by sorry

end purple_balls_count_l3886_388623


namespace clothing_business_optimization_l3886_388614

/-- Represents the monthly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -3 * x + 900

/-- Represents the total monthly revenue as a function of selling price -/
def total_revenue (x : ℝ) : ℝ := (x - 80) * (sales_volume x)

/-- The cost price of clothing in yuan -/
def cost_price : ℝ := 100

/-- The government subsidy per piece in yuan -/
def subsidy_per_piece : ℝ := 20

theorem clothing_business_optimization :
  /- Part 1: Government subsidy when selling price is 160 yuan -/
  (sales_volume 160 * subsidy_per_piece = 8400) ∧
  /- Part 2: Optimal selling price and maximum revenue -/
  (∃ (x_max : ℝ), x_max = 190 ∧
    (∀ x, total_revenue x ≤ total_revenue x_max) ∧
    total_revenue x_max = 36300) :=
by sorry

end clothing_business_optimization_l3886_388614


namespace b_investment_is_200_l3886_388674

/-- Represents the investment scenario with two investors A and B --/
structure Investment where
  a_amount : ℝ  -- A's investment amount
  b_amount : ℝ  -- B's investment amount
  a_months : ℝ  -- Months A's money was invested
  b_months : ℝ  -- Months B's money was invested
  total_profit : ℝ  -- Total profit at the end of the year
  a_profit : ℝ  -- A's share of the profit

/-- The theorem stating that B's investment is $200 given the conditions --/
theorem b_investment_is_200 (inv : Investment) 
  (h1 : inv.a_amount = 150)
  (h2 : inv.a_months = 12)
  (h3 : inv.b_months = 6)
  (h4 : inv.total_profit = 100)
  (h5 : inv.a_profit = 60)
  (h6 : inv.a_profit / inv.total_profit = 
        (inv.a_amount * inv.a_months) / 
        (inv.a_amount * inv.a_months + inv.b_amount * inv.b_months)) :
  inv.b_amount = 200 := by
  sorry


end b_investment_is_200_l3886_388674


namespace twelve_times_minus_square_l3886_388697

theorem twelve_times_minus_square (x : ℕ) (h : x = 6) : 12 * x - x^2 = 36 := by
  sorry

end twelve_times_minus_square_l3886_388697


namespace power_tower_mod_500_l3886_388651

theorem power_tower_mod_500 : 2^(2^(2^2)) % 500 = 36 := by
  sorry

end power_tower_mod_500_l3886_388651


namespace fraction_product_is_three_fifths_l3886_388683

theorem fraction_product_is_three_fifths :
  (7 / 4 : ℚ) * (8 / 14 : ℚ) * (20 / 12 : ℚ) * (15 / 25 : ℚ) *
  (21 / 14 : ℚ) * (12 / 18 : ℚ) * (28 / 14 : ℚ) * (30 / 50 : ℚ) = 3 / 5 := by
  sorry

end fraction_product_is_three_fifths_l3886_388683


namespace unique_base_number_for_16_factorial_l3886_388671

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_base_number_for_16_factorial :
  ∃! b : ℕ+, b > 1 ∧ (factorial 16 % (b : ℕ)^6 = 0) ∧ (factorial 16 % (b : ℕ)^7 ≠ 0) :=
by sorry

end unique_base_number_for_16_factorial_l3886_388671


namespace product_of_fractions_l3886_388626

theorem product_of_fractions :
  (3 : ℚ) / 7 * (5 : ℚ) / 13 * (11 : ℚ) / 17 * (19 : ℚ) / 23 = 3135 / 35581 := by
  sorry

end product_of_fractions_l3886_388626


namespace division_multiplication_equality_l3886_388672

theorem division_multiplication_equality : (144 / 6) * 3 = 72 := by
  sorry

end division_multiplication_equality_l3886_388672


namespace profit_growth_equation_l3886_388607

/-- 
Given an initial profit of 250,000 yuan in May and an expected profit of 360,000 yuan in July,
with an average monthly growth rate of x over 2 months, prove that the equation 25(1+x)^2 = 36 holds true.
-/
theorem profit_growth_equation (x : ℝ) : 
  (250000 : ℝ) * (1 + x)^2 = 360000 → 25 * (1 + x)^2 = 36 := by
  sorry


end profit_growth_equation_l3886_388607


namespace stratified_sampling_distribution_l3886_388633

theorem stratified_sampling_distribution 
  (total : ℕ) (senior : ℕ) (intermediate : ℕ) (junior : ℕ) (sample_size : ℕ)
  (h_total : total = 150)
  (h_senior : senior = 45)
  (h_intermediate : intermediate = 90)
  (h_junior : junior = 15)
  (h_sum : senior + intermediate + junior = total)
  (h_sample : sample_size = 30) :
  ∃ (sample_senior sample_intermediate sample_junior : ℕ),
    sample_senior + sample_intermediate + sample_junior = sample_size ∧
    sample_senior * total = senior * sample_size ∧
    sample_intermediate * total = intermediate * sample_size ∧
    sample_junior * total = junior * sample_size ∧
    sample_senior = 3 ∧
    sample_intermediate = 18 ∧
    sample_junior = 3 :=
by
  sorry

end stratified_sampling_distribution_l3886_388633


namespace finance_equation_solution_l3886_388670

/-- Given the equation fp - w = 20000, where f = 4 and w = 10 + 200i, prove that p = 5002.5 + 50i. -/
theorem finance_equation_solution (f w p : ℂ) : 
  f = 4 → w = 10 + 200 * Complex.I → f * p - w = 20000 → p = 5002.5 + 50 * Complex.I := by
  sorry

end finance_equation_solution_l3886_388670


namespace basketball_team_probabilities_l3886_388631

/-- Represents a series of independent events -/
structure EventSeries where
  n : ℕ  -- number of events
  p : ℝ  -- probability of success for each event
  h1 : 0 ≤ p ∧ p ≤ 1  -- probability is between 0 and 1

/-- The probability of k failures before the first success -/
def prob_k_failures_before_success (es : EventSeries) (k : ℕ) : ℝ :=
  (1 - es.p)^k * es.p

/-- The probability of exactly k successes in n events -/
def prob_exactly_k_successes (es : EventSeries) (k : ℕ) : ℝ :=
  (Nat.choose es.n k : ℝ) * es.p^k * (1 - es.p)^(es.n - k)

/-- The expected number of successes in n events -/
def expected_successes (es : EventSeries) : ℝ :=
  es.n * es.p

theorem basketball_team_probabilities :
  ∀ es : EventSeries,
    es.n = 6 ∧ es.p = 1/3 →
    (prob_k_failures_before_success es 2 = 4/27) ∧
    (prob_exactly_k_successes es 3 = 160/729) ∧
    (expected_successes es = 2) :=
by sorry

end basketball_team_probabilities_l3886_388631


namespace sqrt_meaningful_range_l3886_388658

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 := by
sorry

end sqrt_meaningful_range_l3886_388658


namespace ticket_popcorn_difference_l3886_388655

/-- Represents the cost of items and the deal in a movie theater. -/
structure MovieTheaterCosts where
  deal : ℝ
  ticket : ℝ
  popcorn : ℝ
  drink : ℝ
  candy : ℝ

/-- The conditions of the movie theater deal problem. -/
def dealConditions (c : MovieTheaterCosts) : Prop :=
  c.deal = 20 ∧
  c.ticket = 8 ∧
  c.drink = c.popcorn + 1 ∧
  c.candy = c.drink / 2 ∧
  c.deal = c.ticket + c.popcorn + c.drink + c.candy - 2

/-- The theorem stating the difference between ticket and popcorn costs. -/
theorem ticket_popcorn_difference (c : MovieTheaterCosts) 
  (h : dealConditions c) : c.ticket - c.popcorn = 3 := by
  sorry


end ticket_popcorn_difference_l3886_388655


namespace y1_greater_than_y2_l3886_388688

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2

theorem y1_greater_than_y2 (y₁ y₂ : ℝ) 
  (h1 : y₁ = quadratic_function 3)
  (h2 : y₂ = quadratic_function 1) :
  y₁ > y₂ := by
  sorry

end y1_greater_than_y2_l3886_388688


namespace complex_absolute_value_l3886_388663

theorem complex_absolute_value (z : ℂ) (h : (z + 1) * Complex.I = 3 + 2 * Complex.I) :
  Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_absolute_value_l3886_388663


namespace two_color_no_monochromatic_ap_l3886_388673

theorem two_color_no_monochromatic_ap :
  ∃ f : ℕ+ → Bool, ∀ q r : ℕ+, ∃ n1 n2 : ℕ+, f (q * n1 + r) ≠ f (q * n2 + r) :=
by sorry

end two_color_no_monochromatic_ap_l3886_388673


namespace car_trip_distance_l3886_388637

theorem car_trip_distance (D : ℝ) : 
  (D / 2 : ℝ) + (D / 2 / 4 : ℝ) + 105 = D → D = 280 :=
by sorry

end car_trip_distance_l3886_388637


namespace complex_fraction_difference_l3886_388656

theorem complex_fraction_difference (i : ℂ) (h : i * i = -1) :
  (3 + 2*i) / (2 - 3*i) - (3 - 2*i) / (2 + 3*i) = 2*i :=
by sorry

end complex_fraction_difference_l3886_388656


namespace sum_of_powers_l3886_388678

theorem sum_of_powers : (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2 := by
  sorry

end sum_of_powers_l3886_388678


namespace baseball_cap_production_l3886_388659

theorem baseball_cap_production (caps_week1 caps_week2 caps_week3 total_4_weeks : ℕ) : 
  caps_week1 = 320 →
  caps_week3 = 300 →
  (caps_week1 + caps_week2 + caps_week3 + (caps_week1 + caps_week2 + caps_week3) / 3) = total_4_weeks →
  total_4_weeks = 1360 →
  caps_week2 = 400 := by
sorry

end baseball_cap_production_l3886_388659


namespace semicircle_perimeter_l3886_388687

/-- The perimeter of a semicircle with radius 3.1 cm is equal to π * 3.1 + 6.2 cm. -/
theorem semicircle_perimeter :
  let r : Real := 3.1
  let perimeter := π * r + 2 * r
  perimeter = π * 3.1 + 6.2 := by sorry

end semicircle_perimeter_l3886_388687


namespace geometric_sequence_common_ratio_l3886_388627

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (h1 : a 3 = 4) (h2 : a 6 = 1/2) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 1/2 := by
sorry

end geometric_sequence_common_ratio_l3886_388627


namespace complex_sum_equality_l3886_388625

theorem complex_sum_equality : 
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -5
  let R : ℂ := 1 - I
  let T : ℂ := 3 + 5*I
  B - Q + R + T = 2 + 6*I :=
by sorry

end complex_sum_equality_l3886_388625


namespace negation_of_universal_proposition_l3886_388642

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x < 0 → x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x < 0 ∧ x^3 - x^2 + 1 > 0) :=
by sorry

end negation_of_universal_proposition_l3886_388642


namespace factorization_of_quadratic_l3886_388646

theorem factorization_of_quadratic (a : ℝ) : a^2 - 2*a = a*(a - 2) := by
  sorry

end factorization_of_quadratic_l3886_388646


namespace maple_trees_after_planting_l3886_388602

/-- The number of maple trees in the park after planting is equal to the sum of 
    the initial number of trees and the number of newly planted trees. -/
theorem maple_trees_after_planting 
  (initial_trees : ℕ) 
  (planted_trees : ℕ) 
  (h1 : initial_trees = 53) 
  (h2 : planted_trees = 11) : 
  initial_trees + planted_trees = 64 := by
  sorry

#check maple_trees_after_planting

end maple_trees_after_planting_l3886_388602


namespace spheres_intersection_similar_triangles_l3886_388632

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Checks if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Checks if a point lies on an edge of a tetrahedron -/
def on_edge (p : Point3D) (t : Tetrahedron) : Prop := sorry

/-- Checks if a sphere passes through a point -/
def sphere_passes_through (s : Sphere) (p : Point3D) : Prop := sorry

/-- Checks if two triangles are similar -/
def triangles_similar (p1 p2 p3 q1 q2 q3 : Point3D) : Prop := sorry

/-- Main theorem -/
theorem spheres_intersection_similar_triangles 
  (ABCD : Tetrahedron) (G₁ G₂ : Sphere) 
  (K L M P Q R : Point3D) : 
  sphere_passes_through G₁ ABCD.A ∧ 
  sphere_passes_through G₁ ABCD.B ∧ 
  sphere_passes_through G₁ ABCD.C ∧
  sphere_passes_through G₂ ABCD.A ∧ 
  sphere_passes_through G₂ ABCD.B ∧ 
  sphere_passes_through G₂ ABCD.D ∧
  on_edge K ABCD ∧ collinear K ABCD.D ABCD.A ∧
  on_edge L ABCD ∧ collinear L ABCD.D ABCD.B ∧
  on_edge M ABCD ∧ collinear M ABCD.D ABCD.C ∧
  on_edge P ABCD ∧ collinear P ABCD.C ABCD.A ∧
  on_edge Q ABCD ∧ collinear Q ABCD.C ABCD.B ∧
  on_edge R ABCD ∧ collinear R ABCD.C ABCD.D
  →
  triangles_similar K L M P Q R := by
  sorry

end spheres_intersection_similar_triangles_l3886_388632


namespace angle_FDB_is_40_l3886_388681

-- Define the points
variable (A B C D E F : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- Define isosceles triangle
def isosceles (P Q R : Point) : Prop :=
  angle P Q R = angle P R Q

-- State the theorem
theorem angle_FDB_is_40 :
  isosceles A D E →
  isosceles A B C →
  angle D F C = 150 →
  angle F D B = 40 := by sorry

end angle_FDB_is_40_l3886_388681


namespace max_elements_sum_l3886_388650

/-- A shape formed by adding a pyramid to a rectangular prism -/
structure PrismPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_new_faces : Nat
  pyramid_new_edges : Nat
  pyramid_new_vertex : Nat

/-- The total number of exterior elements in the combined shape -/
def total_elements (shape : PrismPyramid) : Nat :=
  (shape.prism_faces - 1 + shape.pyramid_new_faces) +
  (shape.prism_edges + shape.pyramid_new_edges) +
  (shape.prism_vertices + shape.pyramid_new_vertex)

/-- Theorem stating the maximum sum of exterior elements -/
theorem max_elements_sum :
  ∀ shape : PrismPyramid,
  shape.prism_faces = 6 →
  shape.prism_edges = 12 →
  shape.prism_vertices = 8 →
  shape.pyramid_new_faces ≤ 4 →
  shape.pyramid_new_edges ≤ 4 →
  shape.pyramid_new_vertex ≤ 1 →
  total_elements shape ≤ 34 :=
sorry

end max_elements_sum_l3886_388650


namespace y1_less_than_y2_l3886_388692

/-- Given a linear function y = (m² + 1)x + 2n where m and n are constants,
    and two points A(2a - 1, y₁) and B(a² + 1, y₂) on this function,
    prove that y₁ < y₂ -/
theorem y1_less_than_y2 (m n a : ℝ) (y₁ y₂ : ℝ) 
  (h1 : y₁ = (m^2 + 1) * (2*a - 1) + 2*n) 
  (h2 : y₂ = (m^2 + 1) * (a^2 + 1) + 2*n) : 
  y₁ < y₂ := by
  sorry

end y1_less_than_y2_l3886_388692


namespace sum_lower_bound_l3886_388675

theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  a + b ≥ 4 := by
  sorry

end sum_lower_bound_l3886_388675


namespace division_22_by_8_l3886_388638

theorem division_22_by_8 : (22 : ℚ) / 8 = 2.75 := by sorry

end division_22_by_8_l3886_388638


namespace consecutive_integers_product_552_l3886_388660

theorem consecutive_integers_product_552 (x : ℕ) 
  (h1 : x > 0) 
  (h2 : x * (x + 1) = 552) : 
  x + (x + 1) = 47 ∧ (x + 1) - x = 1 := by
  sorry

end consecutive_integers_product_552_l3886_388660


namespace product_greater_than_sum_l3886_388684

theorem product_greater_than_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a - b = a / b) : a * b > a + b := by
  sorry

end product_greater_than_sum_l3886_388684


namespace f_3_eq_9_l3886_388666

/-- A function f that is monotonic on R and satisfies f(f(x) - 2^x) = 3 for all x ∈ R -/
def f : ℝ → ℝ :=
  sorry

/-- f is monotonic on R -/
axiom f_monotonic : Monotone f

/-- f satisfies f(f(x) - 2^x) = 3 for all x ∈ R -/
axiom f_property (x : ℝ) : f (f x - 2^x) = 3

/-- The main theorem: f(3) = 9 -/
theorem f_3_eq_9 : f 3 = 9 := by
  sorry

end f_3_eq_9_l3886_388666


namespace retail_price_approx_163_59_l3886_388606

/-- Calculates the retail price of a machine before discount -/
def retail_price_before_discount (
  num_machines : ℕ) 
  (wholesale_price : ℚ) 
  (bulk_discount_rate : ℚ) 
  (sales_tax_rate : ℚ) 
  (profit_rate : ℚ) 
  (customer_discount_rate : ℚ) : ℚ :=
  let total_wholesale := num_machines * wholesale_price
  let bulk_discount := bulk_discount_rate * total_wholesale
  let total_cost_after_discount := total_wholesale - bulk_discount
  let profit_per_machine := profit_rate * wholesale_price
  let total_profit := num_machines * profit_per_machine
  let sales_tax := sales_tax_rate * total_profit
  let total_amount_after_tax := total_cost_after_discount + total_profit - sales_tax
  let price_before_discount := total_amount_after_tax / (num_machines * (1 - customer_discount_rate))
  price_before_discount

/-- Theorem stating that the retail price before discount is approximately $163.59 -/
theorem retail_price_approx_163_59 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |retail_price_before_discount 15 126 0.06 0.08 0.22 0.12 - 163.59| < ε :=
sorry

end retail_price_approx_163_59_l3886_388606


namespace equation_solution_l3886_388667

theorem equation_solution :
  ∃ y : ℚ, (2 * y + 3 * y = 600 - (4 * y + 5 * y + 100)) ∧ y = 250 / 7 := by
  sorry

end equation_solution_l3886_388667


namespace complex_number_modulus_l3886_388620

theorem complex_number_modulus (a : ℝ) : a < 0 → Complex.abs (3 + a * Complex.I) = 5 → a = -4 := by
  sorry

end complex_number_modulus_l3886_388620


namespace parabola_hyperbola_equations_l3886_388677

/-- Given a parabola and a hyperbola satisfying certain conditions, 
    prove their equations. -/
theorem parabola_hyperbola_equations :
  ∀ (parabola : ℝ → ℝ → Prop) (hyperbola : ℝ → ℝ → Prop),
  (∀ x y, parabola x y → (x = 0 ∧ y = 0)) →  -- vertex at origin
  (∃ x₀, ∀ y, hyperbola x₀ y → parabola x₀ y) →  -- axis of symmetry passes through focus
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y, hyperbola x y ↔ x^2/a^2 - y^2/b^2 = 1) →  -- general form of hyperbola
  hyperbola (3/2) (Real.sqrt 6) →  -- intersection point
  (∀ x y, parabola x y ↔ y^2 = 4*x) ∧  -- equation of parabola
  (∀ x y, hyperbola x y ↔ 4*x^2 - 4*y^2/3 = 1) :=  -- equation of hyperbola
by sorry

end parabola_hyperbola_equations_l3886_388677


namespace simplify_and_evaluate_l3886_388618

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = -1) (h2 : y = 2) :
  ((x + y)^2 - (x + 2*y)*(x - 2*y)) / (2*y) = 4 := by
  sorry

end simplify_and_evaluate_l3886_388618


namespace brown_mms_second_bag_l3886_388647

theorem brown_mms_second_bag (bags : Nat) (first_bag third_bag fourth_bag fifth_bag average : Nat) : 
  bags = 5 → 
  first_bag = 9 → 
  third_bag = 8 → 
  fourth_bag = 8 → 
  fifth_bag = 3 → 
  average = 8 → 
  ∃ second_bag : Nat, 
    second_bag = 12 ∧ 
    (first_bag + second_bag + third_bag + fourth_bag + fifth_bag) / bags = average := by
  sorry


end brown_mms_second_bag_l3886_388647


namespace heaviest_person_l3886_388635

def weight_problem (A D T V M : ℕ) : Prop :=
  A + D = 82 ∧
  D + T = 74 ∧
  T + V = 75 ∧
  V + M = 65 ∧
  M + A = 62

theorem heaviest_person (A D T V M : ℕ) 
  (h : weight_problem A D T V M) : 
  V = 43 ∧ V ≥ A ∧ V ≥ D ∧ V ≥ T ∧ V ≥ M :=
by
  sorry

#check heaviest_person

end heaviest_person_l3886_388635


namespace square_of_modified_41_l3886_388668

theorem square_of_modified_41 (n : ℕ) :
  let modified_num := (5 * 10^n - 1) * 10^(n+1) + 1
  modified_num^2 = (10^(n+1) - 1)^2 := by
  sorry

end square_of_modified_41_l3886_388668


namespace washing_time_proof_l3886_388615

def shirts : ℕ := 18
def pants : ℕ := 12
def sweaters : ℕ := 17
def jeans : ℕ := 13
def max_items_per_cycle : ℕ := 15
def minutes_per_cycle : ℕ := 45

def total_items : ℕ := shirts + pants + sweaters + jeans

def cycles_needed : ℕ := (total_items + max_items_per_cycle - 1) / max_items_per_cycle

def total_minutes : ℕ := cycles_needed * minutes_per_cycle

theorem washing_time_proof : 
  total_minutes / 60 = 3 := by sorry

end washing_time_proof_l3886_388615


namespace lanas_initial_pages_l3886_388636

theorem lanas_initial_pages (x : ℕ) : 
  x + (42 / 2) = 29 → x = 8 := by sorry

end lanas_initial_pages_l3886_388636


namespace profit_achievement_l3886_388605

/-- The number of pens in a pack -/
def pens_per_pack : ℕ := 4

/-- The cost of a pack of pens in dollars -/
def pack_cost : ℚ := 7

/-- The number of pens sold at the given rate -/
def pens_sold_rate : ℕ := 5

/-- The price for the number of pens sold at the given rate in dollars -/
def price_sold_rate : ℚ := 12

/-- The target profit in dollars -/
def target_profit : ℚ := 50

/-- The minimum number of pens needed to be sold to achieve the target profit -/
def min_pens_to_sell : ℕ := 77

theorem profit_achievement :
  ∃ (n : ℕ), n ≥ min_pens_to_sell ∧
  (n : ℚ) * (price_sold_rate / pens_sold_rate) - 
  (n : ℚ) * (pack_cost / pens_per_pack) ≥ target_profit ∧
  ∀ (m : ℕ), m < min_pens_to_sell →
  (m : ℚ) * (price_sold_rate / pens_sold_rate) - 
  (m : ℚ) * (pack_cost / pens_per_pack) < target_profit :=
by sorry

end profit_achievement_l3886_388605


namespace range_of_a_l3886_388640

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x - a > 0
def q (x : ℝ) : Prop := x > 1

-- Define what it means for p to be a sufficient condition for q
def sufficient (a : ℝ) : Prop := ∀ x, p x a → q x

-- Define what it means for p to be not a necessary condition for q
def not_necessary (a : ℝ) : Prop := ∃ x, q x ∧ ¬(p x a)

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (sufficient a ∧ not_necessary a) → a > 1 := by sorry

end range_of_a_l3886_388640


namespace sector_central_angle_l3886_388645

/-- Given a sector with radius 1 and perimeter 4, its central angle in radians has an absolute value of 2. -/
theorem sector_central_angle (r : ℝ) (L : ℝ) (α : ℝ) : 
  r = 1 → L = 4 → L = r * α + 2 * r → |α| = 2 := by sorry

end sector_central_angle_l3886_388645


namespace distance_to_x_axis_l3886_388685

theorem distance_to_x_axis (P : ℝ × ℝ) : P = (3, -2) → |P.2| = 2 := by
  sorry

end distance_to_x_axis_l3886_388685


namespace slope_of_line_l3886_388619

/-- The slope of a line given by the equation (x/4) - (y/3) = -2 is -3/4 -/
theorem slope_of_line (x y : ℝ) : (x / 4 - y / 3 = -2) → (y = (-3 / 4) * x - 6) := by
  sorry

end slope_of_line_l3886_388619


namespace decomposition_675_l3886_388600

theorem decomposition_675 (n : Nat) (h : n = 675) :
  ∃ (num_stacks height : Nat),
    num_stacks > 1 ∧
    height > 1 ∧
    n = 3^3 * 5^2 ∧
    num_stacks = 3 ∧
    height = 3^2 * 5^2 ∧
    height^num_stacks = n := by
  sorry

end decomposition_675_l3886_388600


namespace square_root_divided_by_two_l3886_388622

theorem square_root_divided_by_two : Real.sqrt 16 / 2 = 2 := by sorry

end square_root_divided_by_two_l3886_388622


namespace car_pedestrian_speed_ratio_l3886_388657

/-- The ratio of a car's speed to a pedestrian's speed on a bridge -/
theorem car_pedestrian_speed_ratio :
  ∀ (L : ℝ) (v_p v_c : ℝ),
  L > 0 → v_p > 0 → v_c > 0 →
  (4/9 * L) / v_p = (5/9 * L) / v_c →
  v_c / v_p = 9 := by
  sorry

end car_pedestrian_speed_ratio_l3886_388657


namespace minimum_days_to_plant_trees_l3886_388695

def tree_sequence (n : ℕ) : ℕ := 2^(n + 1) - 2

theorem minimum_days_to_plant_trees :
  ∃ (n : ℕ), n > 0 ∧ tree_sequence n ≥ 100 ∧ ∀ m : ℕ, m > 0 → m < n → tree_sequence m < 100 :=
by sorry

end minimum_days_to_plant_trees_l3886_388695


namespace second_point_x_coordinate_l3886_388694

/-- Given two points on a line, prove that the x-coordinate of the second point is m + 5 -/
theorem second_point_x_coordinate 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m + 5 = m + 5 := by
  sorry

end second_point_x_coordinate_l3886_388694


namespace trumpet_cost_l3886_388691

/-- The cost of Mike's trumpet, given his total spending and the cost of a song book. -/
theorem trumpet_cost (total_spent song_book_cost : ℚ) 
  (h1 : total_spent = 151)
  (h2 : song_book_cost = 584 / 100) : 
  total_spent - song_book_cost = 14516 / 100 := by
  sorry

end trumpet_cost_l3886_388691


namespace expression_evaluation_l3886_388613

theorem expression_evaluation : 
  let x : ℚ := 1/2
  (x^2 * (x - 1) - x * (x^2 + x - 1)) = 0 := by sorry

end expression_evaluation_l3886_388613


namespace tinas_fourth_hour_coins_verify_final_coins_l3886_388652

/-- Represents the number of coins in Tina's jar at different stages -/
structure CoinJar where
  initial : ℕ := 0
  first_hour : ℕ
  second_third_hours : ℕ
  fourth_hour : ℕ
  fifth_hour : ℕ

/-- The coin jar problem setup -/
def tinas_jar : CoinJar :=
  { first_hour := 20
  , second_third_hours := 60
  , fourth_hour := 40  -- This is what we want to prove
  , fifth_hour := 100 }

/-- Theorem stating that the number of coins Tina put in during the fourth hour is 40 -/
theorem tinas_fourth_hour_coins :
  tinas_jar.fourth_hour = 40 :=
by
  -- The actual proof would go here
  sorry

/-- Verify that the final number of coins matches the problem statement -/
theorem verify_final_coins :
  tinas_jar.first_hour + tinas_jar.second_third_hours + tinas_jar.fourth_hour - 20 = tinas_jar.fifth_hour :=
by
  -- The actual proof would go here
  sorry

end tinas_fourth_hour_coins_verify_final_coins_l3886_388652


namespace weight_of_A_l3886_388653

theorem weight_of_A (a b c d : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  (b + c + d + (d + 6)) / 4 = 79 →
  a = 174 :=
by sorry

end weight_of_A_l3886_388653


namespace max_value_of_fraction_l3886_388643

theorem max_value_of_fraction (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ (max : ℝ), max = 7/5 ∧ ∀ x y, 
    x + y - 2 ≥ 0 → y - x - 1 ≤ 0 → x ≤ 1 → 
    (x + 2*y) / (2*x + y) ≤ max :=
by sorry

end max_value_of_fraction_l3886_388643


namespace perpendicular_vectors_m_value_l3886_388661

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (-2, 3) and b = (3, m) are perpendicular, prove that m = 2 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (-2, 3)
  let b : ℝ × ℝ := (3, m)
  perpendicular a b → m = 2 := by
  sorry

end perpendicular_vectors_m_value_l3886_388661


namespace two_thousand_five_power_l3886_388616

theorem two_thousand_five_power : ∃ a b : ℕ, (2005 : ℕ)^2005 = a^2 + b^2 ∧ ¬∃ c d : ℕ, (2005 : ℕ)^2005 = c^3 + d^3 := by
  sorry

end two_thousand_five_power_l3886_388616


namespace sphere_radius_from_cylinder_l3886_388693

/-- The radius of a sphere formed by recasting a cylindrical iron block -/
theorem sphere_radius_from_cylinder (cylinder_radius : ℝ) (cylinder_height : ℝ) (sphere_radius : ℝ) : 
  cylinder_radius = 2 →
  cylinder_height = 9 →
  (4 / 3) * Real.pi * sphere_radius ^ 3 = Real.pi * cylinder_radius ^ 2 * cylinder_height →
  sphere_radius = 3 := by
  sorry

#check sphere_radius_from_cylinder

end sphere_radius_from_cylinder_l3886_388693


namespace calculation_proof_l3886_388608

theorem calculation_proof : 
  (168 / 100 * ((1265^2) / 21)) / (6 - (3^2)) = -42646.26666666667 := by
  sorry

end calculation_proof_l3886_388608


namespace partnership_investment_ratio_l3886_388698

/-- A partnership business between A and B -/
structure Partnership where
  /-- A's investment as a multiple of B's investment -/
  a_investment_multiple : ℝ
  /-- B's profit -/
  b_profit : ℝ
  /-- Total profit -/
  total_profit : ℝ

/-- The ratio of A's investment to B's investment in the partnership -/
def investment_ratio (p : Partnership) : ℝ := p.a_investment_multiple

/-- Theorem stating the investment ratio in the given partnership scenario -/
theorem partnership_investment_ratio (p : Partnership) 
  (h1 : p.b_profit = 4000)
  (h2 : p.total_profit = 28000) : 
  investment_ratio p = 3 := by
  sorry

end partnership_investment_ratio_l3886_388698


namespace ratio_proof_l3886_388689

theorem ratio_proof (x y z : ℚ) :
  (5 * x + 4 * y - 6 * z) / (4 * x - 5 * y + 7 * z) = 1 / 27 ∧
  (5 * x + 4 * y - 6 * z) / (6 * x + 5 * y - 4 * z) = 1 / 18 →
  ∃ (k : ℚ), x = 3 * k ∧ y = 4 * k ∧ z = 5 * k :=
by sorry

end ratio_proof_l3886_388689


namespace probability_sum_10_l3886_388603

-- Define a die roll as a natural number between 1 and 6
def DieRoll : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define a function to check if the sum of three die rolls is 10
def sumIs10 (roll1 roll2 roll3 : DieRoll) : Prop :=
  roll1.val + roll2.val + roll3.val = 10

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := 216

-- Define the number of favorable outcomes (sum is 10)
def favorableOutcomes : ℕ := 27

-- Theorem statement
theorem probability_sum_10 :
  (favorableOutcomes : ℚ) / totalOutcomes = 27 / 216 :=
sorry

end probability_sum_10_l3886_388603


namespace cube_surface_area_l3886_388649

/-- The surface area of a cube that can be cut into 27 smaller cubes, each with an edge length of 4 cm, is 864 cm². -/
theorem cube_surface_area : 
  ∀ (original_cube_edge : ℝ) (small_cube_edge : ℝ),
  small_cube_edge = 4 →
  (original_cube_edge / small_cube_edge)^3 = 27 →
  6 * original_cube_edge^2 = 864 := by
sorry

end cube_surface_area_l3886_388649


namespace circle_C_properties_l3886_388601

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define the line that contains the center of circle C
def center_line (x y : ℝ) : Prop :=
  x + 2*y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  8*x - 15*y - 3 = 0 ∨ x = 6

-- Define the line l
def line_l (x y m : ℝ) : Prop :=
  y = x + m

theorem circle_C_properties :
  -- Circle C passes through M(0, -2) and N(3, 1)
  circle_C 0 (-2) ∧ circle_C 3 1 ∧
  -- The center of circle C lies on the line x + 2y + 1 = 0
  ∃ (cx cy : ℝ), center_line cx cy ∧
    ∀ (x y : ℝ), circle_C x y ↔ (x - cx)^2 + (y - cy)^2 = (cx^2 + cy^2 - 4) →
  -- The tangent line to circle C passing through (6, 3) is correct
  tangent_line 6 3 ∧
  -- The line l has the correct equations
  (line_l x y (-1) ∨ line_l x y (-4)) ∧
  -- Circle C₁ with diameter AB (intersection of l and C) passes through the origin
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    ((line_l x₁ y₁ (-1) ∧ line_l x₂ y₂ (-1)) ∨ (line_l x₁ y₁ (-4) ∧ line_l x₂ y₂ (-4))) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2 * ((x₁^2 + y₁^2) + (x₂^2 + y₂^2)) :=
sorry

end circle_C_properties_l3886_388601


namespace equivalent_discount_l3886_388679

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price > 0 →
  0 ≤ discount1 → discount1 < 1 →
  0 ≤ discount2 → discount2 < 1 →
  original_price * (1 - discount1) * (1 - discount2) = original_price * (1 - 0.4) :=
by
  sorry

#check equivalent_discount 50 0.25 0.2

end equivalent_discount_l3886_388679


namespace solution_set_implies_m_value_l3886_388639

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- State the theorem
theorem solution_set_implies_m_value (m : ℝ) :
  (∀ x : ℝ, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by
  sorry

end solution_set_implies_m_value_l3886_388639


namespace arithmetic_mean_proof_l3886_388617

theorem arithmetic_mean_proof (x a b : ℝ) (hx : x ≠ b ∧ x ≠ -b) :
  (1/2) * ((x + a + b)/(x + b) + (x - a - b)/(x - b)) = 1 - a*b/(x^2 - b^2) := by
  sorry

end arithmetic_mean_proof_l3886_388617


namespace equally_spaced_posts_l3886_388680

/-- Given a sequence of 8 equally spaced posts, if the distance between the first and fifth post
    is 100 meters, then the distance between the first and last post is 175 meters. -/
theorem equally_spaced_posts (posts : Fin 8 → ℝ) 
  (equally_spaced : ∀ i j k : Fin 8, i.val + 1 = j.val → j.val + 1 = k.val → 
    posts k - posts j = posts j - posts i)
  (first_to_fifth : posts 4 - posts 0 = 100) :
  posts 7 - posts 0 = 175 :=
sorry

end equally_spaced_posts_l3886_388680


namespace place_one_after_two_digit_number_l3886_388648

/-- Given a two-digit number with tens digit t and units digit u,
    prove that placing the digit 1 after this number results in 100t + 10u + 1 -/
theorem place_one_after_two_digit_number (t u : ℕ) :
  let original := 10 * t + u
  let new_number := original * 10 + 1
  new_number = 100 * t + 10 * u + 1 := by
sorry

end place_one_after_two_digit_number_l3886_388648


namespace lines_intersection_l3886_388665

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℚ) : Prop :=
  l.a * x + l.b * y = l.c

/-- The three lines given in the problem -/
def line1 : Line := ⟨-3, 2, 4⟩
def line2 : Line := ⟨1, 3, 3⟩
def line3 : Line := ⟨5, -3, 6⟩

/-- Theorem stating that the given lines intersect at the specified points -/
theorem lines_intersection :
  (line1.contains (10/11) (13/11) ∧
   line2.contains (10/11) (13/11) ∧
   line3.contains 24 38) ∧
  (line1.contains 24 38 ∧
   line2.contains 24 38 ∧
   line3.contains 24 38) := by
  sorry

end lines_intersection_l3886_388665


namespace wyatt_envelopes_l3886_388630

theorem wyatt_envelopes (blue : ℕ) (yellow : ℕ) : 
  yellow = blue - 4 →
  blue + yellow = 16 →
  blue = 10 := by
sorry

end wyatt_envelopes_l3886_388630


namespace one_in_range_of_f_l3886_388676

/-- The function f(x) = x^2 + bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 1

/-- Theorem: For all real numbers b, 1 is always in the range of f(x) = x^2 + bx - 1 -/
theorem one_in_range_of_f (b : ℝ) : ∃ x : ℝ, f b x = 1 := by
  sorry

end one_in_range_of_f_l3886_388676


namespace intersecting_lines_sum_of_intercepts_l3886_388644

/-- Two lines intersecting at (3, 3) have the sum of their y-intercepts equal to 4 -/
theorem intersecting_lines_sum_of_intercepts (c d : ℝ) : 
  (3 = (1/3) * 3 + c) ∧ (3 = (1/3) * 3 + d) → c + d = 4 := by
  sorry

#check intersecting_lines_sum_of_intercepts

end intersecting_lines_sum_of_intercepts_l3886_388644


namespace problem_solution_l3886_388641

theorem problem_solution (A B : ℝ) 
  (h1 : 30 - (4 * A + 5) = 3 * B) 
  (h2 : B = 2 * A) : 
  A = 2.5 ∧ B = 5 := by
sorry

end problem_solution_l3886_388641


namespace consecutive_odd_numbers_sum_l3886_388611

/-- Given 6 consecutive odd numbers whose product is 135135, prove their sum is 48 -/
theorem consecutive_odd_numbers_sum (a b c d e f : ℕ) : 
  (a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) →  -- consecutive
  (∃ k, a = 2*k + 1) →  -- a is odd
  (b = a + 2) → (c = b + 2) → (d = c + 2) → (e = d + 2) → (f = e + 2) →  -- consecutive odd numbers
  (a * b * c * d * e * f = 135135) →  -- product is 135135
  (a + b + c + d + e + f = 48) :=  -- sum is 48
by sorry

end consecutive_odd_numbers_sum_l3886_388611


namespace original_number_before_increase_l3886_388664

theorem original_number_before_increase (x : ℝ) : x * 1.5 = 165 → x = 110 := by
  sorry

end original_number_before_increase_l3886_388664
