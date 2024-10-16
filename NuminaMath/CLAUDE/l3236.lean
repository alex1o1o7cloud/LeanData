import Mathlib

namespace NUMINAMATH_CALUDE_key_dimension_in_polygon_division_l3236_323618

/-- Represents a polygon with a key dimension --/
structure Polygon where
  keyDimension : ℝ

/-- Represents a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Function to check if two polygons are congruent --/
def areCongruent (p1 p2 : Polygon) : Prop := sorry

/-- Function to check if polygons can form a square --/
def canFormSquare (p1 p2 : Polygon) (s : Square) : Prop := sorry

/-- Theorem stating the existence of a key dimension x = 4 in the polygons --/
theorem key_dimension_in_polygon_division (r : Rectangle) 
  (h1 : r.width = 12 ∧ r.height = 12) 
  (p1 p2 : Polygon) (s : Square)
  (h2 : areCongruent p1 p2)
  (h3 : canFormSquare p1 p2 s)
  (h4 : s.side^2 = r.width * r.height) :
  ∃ x : ℝ, x = 4 ∧ (p1.keyDimension = x ∨ p2.keyDimension = x) :=
sorry

end NUMINAMATH_CALUDE_key_dimension_in_polygon_division_l3236_323618


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3236_323626

/-- A geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a → a 1 = 3 → a 4 = 24 → a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3236_323626


namespace NUMINAMATH_CALUDE_vegan_soy_free_fraction_l3236_323698

theorem vegan_soy_free_fraction (total_dishes : ℕ) (vegan_dishes : ℕ) (soy_dishes : ℕ)
  (h1 : vegan_dishes = total_dishes / 4)
  (h2 : soy_dishes = 4 * vegan_dishes / 5)
  (h3 : vegan_dishes > 0)
  (h4 : total_dishes > 0) :
  (vegan_dishes - soy_dishes) / total_dishes = 1 / 20 :=
by sorry

end NUMINAMATH_CALUDE_vegan_soy_free_fraction_l3236_323698


namespace NUMINAMATH_CALUDE_projection_matrix_values_l3236_323685

def projection_matrix (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 10/21; c, 35/63]

theorem projection_matrix_values :
  ∀ a c : ℚ, projection_matrix a c ^ 2 = projection_matrix a c → a = 2/9 ∧ c = 7/6 :=
by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l3236_323685


namespace NUMINAMATH_CALUDE_coefficient_of_expansion_l3236_323679

theorem coefficient_of_expansion (x : ℝ) : 
  ∃ a b c d e : ℝ, (2*x + 1)^5 = a + b*(x+1) + c*(x+1)^2 + d*(x+1)^3 + e*(x+1)^4 + (-5)*(x+1)^5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_expansion_l3236_323679


namespace NUMINAMATH_CALUDE_binomial_inequality_l3236_323653

theorem binomial_inequality (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n) :
  (n^n : ℚ) / (m^m * (n-m)^(n-m)) > (n.factorial : ℚ) / (m.factorial * (n-m).factorial) ∧
  (n.factorial : ℚ) / (m.factorial * (n-m).factorial) > (n^n : ℚ) / (m^m * (n+1) * (n-m)^(n-m)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_inequality_l3236_323653


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3236_323651

theorem imaginary_part_of_z : Complex.im ((1 + Complex.I)^2 + Complex.I^2011) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3236_323651


namespace NUMINAMATH_CALUDE_bowling_team_weight_problem_l3236_323600

theorem bowling_team_weight_problem (original_players : ℕ) 
                                    (original_avg_weight : ℝ) 
                                    (new_players : ℕ) 
                                    (known_new_player_weight : ℝ) 
                                    (new_avg_weight : ℝ) :
  original_players = 7 →
  original_avg_weight = 76 →
  new_players = 2 →
  known_new_player_weight = 60 →
  new_avg_weight = 78 →
  ∃ (unknown_new_player_weight : ℝ),
    (original_players * original_avg_weight + known_new_player_weight + unknown_new_player_weight) / 
    (original_players + new_players) = new_avg_weight ∧
    unknown_new_player_weight = 110 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_weight_problem_l3236_323600


namespace NUMINAMATH_CALUDE_a_integer_not_multiple_of_five_l3236_323612

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 6*x + 1 = 0

-- Define the sequence aₙ
def a (n : ℕ) (x₁ x₂ : ℝ) : ℝ := x₁^n + x₂^n

-- State the theorem
theorem a_integer_not_multiple_of_five 
  (x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation x₁) 
  (h₂ : quadratic_equation x₂) :
  ∀ n : ℕ, ∃ k : ℤ, (a n x₁ x₂ = k) ∧ ¬(∃ m : ℤ, k = 5 * m) :=
by sorry

end NUMINAMATH_CALUDE_a_integer_not_multiple_of_five_l3236_323612


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l3236_323665

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 5*x^3 + 1

-- Define the interval
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Theorem statement
theorem f_extrema_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 153) ∧
  (∃ x ∈ interval, f x = -4) := by
sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l3236_323665


namespace NUMINAMATH_CALUDE_circle_equation_radius_8_l3236_323650

/-- The equation x^2 + 14x + y^2 + 10y - k = 0 represents a circle of radius 8 if and only if k = 10 -/
theorem circle_equation_radius_8 (x y k : ℝ) : 
  (∃ h₁ h₂ : ℝ, ∀ x y : ℝ, x^2 + 14*x + y^2 + 10*y - k = 0 ↔ (x - h₁)^2 + (y - h₂)^2 = 64) ↔ 
  k = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_8_l3236_323650


namespace NUMINAMATH_CALUDE_fraction_of_product_l3236_323662

theorem fraction_of_product (x : ℚ) : x * (1/2 * 2/5 * 5100) = 765.0000000000001 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_product_l3236_323662


namespace NUMINAMATH_CALUDE_mikes_remaining_nickels_l3236_323643

/-- Represents the number of nickels Mike has after his dad's borrowing. -/
def mikesRemainingNickels (initialNickels : ℕ) (borrowedNickels : ℕ) : ℕ :=
  initialNickels - borrowedNickels

/-- Represents the total number of nickels borrowed by Mike's dad. -/
def totalBorrowedNickels (mikesBorrowed : ℕ) (sistersBorrowed : ℕ) : ℕ :=
  mikesBorrowed + sistersBorrowed

/-- Represents the relationship between nickels borrowed from Mike and his sister. -/
def borrowingPattern (mikesBorrowed : ℕ) (sistersBorrowed : ℕ) : Prop :=
  3 * sistersBorrowed = 2 * mikesBorrowed

theorem mikes_remaining_nickels :
  ∀ (mikesInitialNickels : ℕ) (mikesBorrowed : ℕ) (sistersBorrowed : ℕ),
    mikesInitialNickels = 87 →
    totalBorrowedNickels mikesBorrowed sistersBorrowed = 75 →
    borrowingPattern mikesBorrowed sistersBorrowed →
    mikesRemainingNickels mikesInitialNickels mikesBorrowed = 42 :=
by sorry

end NUMINAMATH_CALUDE_mikes_remaining_nickels_l3236_323643


namespace NUMINAMATH_CALUDE_salary_percentage_increase_l3236_323634

theorem salary_percentage_increase 
  (original : ℝ) 
  (decrease_percent : ℝ) 
  (increase_percent : ℝ) 
  (overall_decrease_percent : ℝ) 
  (h1 : decrease_percent = 50) 
  (h2 : overall_decrease_percent = 35) 
  (h3 : original * (1 - decrease_percent / 100) * (1 + increase_percent / 100) = 
        original * (1 - overall_decrease_percent / 100)) : 
  increase_percent = 30 := by
sorry

end NUMINAMATH_CALUDE_salary_percentage_increase_l3236_323634


namespace NUMINAMATH_CALUDE_remainder_problem_l3236_323647

theorem remainder_problem (n : ℤ) (h : n % 22 = 12) : (2 * n) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3236_323647


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3236_323673

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ p x) ↔ (∀ x : ℝ, x ≥ 0 → ¬ p x) := by
  sorry

-- The specific proposition
def proposition (x : ℝ) : Prop := 2 * x = 3

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ proposition x) ↔ (∀ x : ℝ, x ≥ 0 → 2 * x ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3236_323673


namespace NUMINAMATH_CALUDE_swim_time_ratio_l3236_323630

/-- Proves that the ratio of time taken to swim upstream to downstream is 2:1 given specific speeds -/
theorem swim_time_ratio (man_speed stream_speed : ℝ) 
  (h1 : man_speed = 3)
  (h2 : stream_speed = 1) :
  (man_speed - stream_speed)⁻¹ / (man_speed + stream_speed)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_swim_time_ratio_l3236_323630


namespace NUMINAMATH_CALUDE_stickers_after_loss_l3236_323649

/-- Given a certain number of sticker pages and stickers per page, 
    calculate the number of stickers remaining after losing one page -/
theorem stickers_after_loss (stickers_per_page : ℕ) (initial_pages : ℕ) : 
  stickers_per_page = 20 →
  initial_pages = 12 →
  (initial_pages - 1) * stickers_per_page = 220 := by
  sorry

end NUMINAMATH_CALUDE_stickers_after_loss_l3236_323649


namespace NUMINAMATH_CALUDE_pies_sold_weekend_l3236_323663

/-- Represents the number of slices in each type of pie -/
structure PieSlices where
  apple : ℕ
  peach : ℕ
  cherry : ℕ

/-- Represents the number of customers who ordered each type of pie -/
structure PieOrders where
  apple : ℕ
  peach : ℕ
  cherry : ℕ

/-- Calculates the total number of pies sold given the number of slices per pie and the number of orders -/
def totalPiesSold (slices : PieSlices) (orders : PieOrders) : ℕ :=
  let applePies := (orders.apple + slices.apple - 1) / slices.apple
  let peachPies := (orders.peach + slices.peach - 1) / slices.peach
  let cherryPies := (orders.cherry + slices.cherry - 1) / slices.cherry
  applePies + peachPies + cherryPies

/-- Theorem stating that given the specific conditions, the total pies sold is 29 -/
theorem pies_sold_weekend (slices : PieSlices) (orders : PieOrders)
    (h1 : slices.apple = 8)
    (h2 : slices.peach = 6)
    (h3 : slices.cherry = 10)
    (h4 : orders.apple = 88)
    (h5 : orders.peach = 78)
    (h6 : orders.cherry = 45) :
    totalPiesSold slices orders = 29 := by
  sorry


end NUMINAMATH_CALUDE_pies_sold_weekend_l3236_323663


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l3236_323629

theorem lcm_gcf_problem (n m : ℕ+) : 
  Nat.lcm n m = 54 → Nat.gcd n m = 8 → n = 36 → m = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l3236_323629


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l3236_323681

theorem intersection_implies_m_equals_one :
  ∀ m : ℝ, ({3, 4, m^2 - 3*m - 1} ∩ {2*m, -3} : Set ℝ) = {-3} → m = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l3236_323681


namespace NUMINAMATH_CALUDE_bee_multiple_l3236_323686

theorem bee_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bee_multiple_l3236_323686


namespace NUMINAMATH_CALUDE_simplest_common_denominator_l3236_323602

-- Define the fractions
def fraction1 (x y : ℚ) : ℚ := 1 / (2 * x^2 * y)
def fraction2 (x y : ℚ) : ℚ := 1 / (6 * x * y^3)

-- Define the common denominator
def common_denominator (x y : ℚ) : ℚ := 6 * x^2 * y^3

-- Theorem statement
theorem simplest_common_denominator (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (a b : ℚ), 
    fraction1 x y = a / common_denominator x y ∧
    fraction2 x y = b / common_denominator x y ∧
    (∀ (c : ℚ), c > 0 → 
      (∃ (d e : ℚ), fraction1 x y = d / c ∧ fraction2 x y = e / c) →
      c ≥ common_denominator x y) :=
sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_l3236_323602


namespace NUMINAMATH_CALUDE_pizza_delivery_problem_l3236_323601

theorem pizza_delivery_problem (total_time : ℕ) (avg_time_per_stop : ℕ) 
  (two_pizza_stops : ℕ) (pizzas_per_two_pizza_stop : ℕ) :
  total_time = 40 →
  avg_time_per_stop = 4 →
  two_pizza_stops = 2 →
  pizzas_per_two_pizza_stop = 2 →
  ∃ (single_pizza_stops : ℕ),
    (single_pizza_stops + two_pizza_stops) * avg_time_per_stop = total_time ∧
    single_pizza_stops + two_pizza_stops * pizzas_per_two_pizza_stop = 12 :=
by sorry

end NUMINAMATH_CALUDE_pizza_delivery_problem_l3236_323601


namespace NUMINAMATH_CALUDE_six_students_three_groups_arrangements_l3236_323695

/-- The number of ways to divide n students into k equal groups -/
def divide_into_groups (n k : ℕ) : ℕ := sorry

/-- The number of ways to assign k groups to k topics -/
def assign_topics (k : ℕ) : ℕ := sorry

/-- The total number of arrangements for n students divided into k equal groups 
    and assigned to k different topics -/
def total_arrangements (n k : ℕ) : ℕ :=
  divide_into_groups n k * assign_topics k

theorem six_students_three_groups_arrangements :
  total_arrangements 6 3 = 540 := by sorry

end NUMINAMATH_CALUDE_six_students_three_groups_arrangements_l3236_323695


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l3236_323627

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (2 * Complex.I)
  (Complex.re z = Complex.im z) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l3236_323627


namespace NUMINAMATH_CALUDE_harry_sea_stars_harry_collected_34_sea_stars_l3236_323614

theorem harry_sea_stars : ℕ → Prop :=
  fun sea_stars =>
    sea_stars + 21 + 29 = 59 + 25 ∧ 
    sea_stars = 34

/-- Proof that Harry collected 34 sea stars initially -/
theorem harry_collected_34_sea_stars : ∃ (sea_stars : ℕ), harry_sea_stars sea_stars :=
by
  sorry

end NUMINAMATH_CALUDE_harry_sea_stars_harry_collected_34_sea_stars_l3236_323614


namespace NUMINAMATH_CALUDE_expression_value_l3236_323611

theorem expression_value (x y : ℝ) (h : x - 2 * y^2 = 1) :
  -2 * x + 4 * y^2 + 1 = -1 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3236_323611


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l3236_323631

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l3236_323631


namespace NUMINAMATH_CALUDE_arithmetic_sequence_26th_term_l3236_323646

/-- Given an arithmetic sequence with first term 3 and second term 13, 
    the 26th term is 253. -/
theorem arithmetic_sequence_26th_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
    a 0 = 3 →                            -- first term is 3
    a 1 = 13 →                           -- second term is 13
    a 25 = 253 :=                        -- 26th term (index 25) is 253
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_26th_term_l3236_323646


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l3236_323616

/-- Represents the number of ways to choose a starting lineup for a volleyball team -/
def volleyballLineupCount (totalPlayers : ℕ) (versatilePlayers : ℕ) (specializedPlayers : ℕ) : ℕ :=
  totalPlayers * (totalPlayers - 1) * versatilePlayers * (versatilePlayers - 1) * (versatilePlayers - 2)

/-- Theorem stating the number of ways to choose a starting lineup for a volleyball team with given conditions -/
theorem volleyball_lineup_count :
  volleyballLineupCount 10 8 2 = 30240 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l3236_323616


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3236_323683

/-- The parabola function --/
def f (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A point is on the parabola if its y-coordinate equals f(x) --/
def on_parabola (p : ℝ × ℝ) : Prop := p.2 = f p.1

/-- A point is on the x-axis if its y-coordinate is 0 --/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A square is defined by its center and side length --/
structure Square where
  center : ℝ × ℝ
  side : ℝ

/-- A square is inscribed if all its vertices are either on the parabola or on the x-axis --/
def inscribed (s : Square) : Prop :=
  let half_side := s.side / 2
  let left := s.center.1 - half_side
  let right := s.center.1 + half_side
  let top := s.center.2 + half_side
  let bottom := s.center.2 - half_side
  on_x_axis (left, bottom) ∧
  on_x_axis (right, bottom) ∧
  on_parabola (left, top) ∧
  on_parabola (right, top)

/-- The theorem to be proved --/
theorem inscribed_square_area :
  ∃ (s : Square), inscribed s ∧ s.side^2 = 24 - 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3236_323683


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3236_323676

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (x₀ : ℝ) : Prop := parabola x₀ 4

-- Define a line passing through a point and intersecting the parabola at two other points
def intersecting_line (x₀ m t : ℝ) : Prop :=
  point_on_parabola x₀ ∧ ∃ y₁ y₂ : ℝ, 
    y₁ ≠ y₂ ∧ 
    parabola (m*y₁ + t) y₁ ∧ 
    parabola (m*y₂ + t) y₂

-- Define perpendicularity condition
def perpendicular_condition (x₀ m t : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, 
    (m*y₁ + t - x₀) * (m*y₂ + t - x₀) + (y₁ - 4) * (y₂ - 4) = 0

-- Theorem statement
theorem line_passes_through_fixed_point (x₀ m t : ℝ) :
  intersecting_line x₀ m t ∧ perpendicular_condition x₀ m t →
  t = 4*m + 8 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3236_323676


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l3236_323687

/-- Calculates the area of a rectangular plot given its breadth and the fact that its length is thrice its breadth -/
def rectangle_area (breadth : ℝ) : ℝ :=
  3 * breadth * breadth

/-- Proves that the area of a rectangular plot with breadth 26 meters and length thrice its breadth is 2028 square meters -/
theorem rectangle_area_proof : rectangle_area 26 = 2028 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l3236_323687


namespace NUMINAMATH_CALUDE_consecutive_numbers_equation_l3236_323688

theorem consecutive_numbers_equation (x y z : ℤ) : 
  (x = y + 1) → 
  (z = y - 1) → 
  (x > y) → 
  (y > z) → 
  (z = 2) → 
  (2 * x + 3 * y + 3 * z = 8 * y - 1) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_equation_l3236_323688


namespace NUMINAMATH_CALUDE_ad_campaign_cost_l3236_323615

/-- Calculates the total cost of an ad campaign with given parameters and discount rules --/
theorem ad_campaign_cost 
  (page_width : ℝ) 
  (page_height : ℝ) 
  (full_page_rate : ℝ) 
  (half_page_rate : ℝ) 
  (quarter_page_rate : ℝ) 
  (eighth_page_rate : ℝ) 
  (half_page_count : ℕ) 
  (quarter_page_count : ℕ) 
  (eighth_page_count : ℕ) 
  (discount_rate_4_to_5 : ℝ) 
  (discount_rate_6_or_more : ℝ) : 
  page_width = 9 → 
  page_height = 12 → 
  full_page_rate = 6.5 → 
  half_page_rate = 8 → 
  quarter_page_rate = 10 → 
  eighth_page_rate = 12 → 
  half_page_count = 1 → 
  quarter_page_count = 3 → 
  eighth_page_count = 4 → 
  discount_rate_4_to_5 = 0.1 → 
  discount_rate_6_or_more = 0.15 → 
  ∃ (total_cost : ℝ), total_cost = 1606.5 := by
  sorry


end NUMINAMATH_CALUDE_ad_campaign_cost_l3236_323615


namespace NUMINAMATH_CALUDE_even_function_shift_l3236_323632

/-- Given a function f and a real number a, proves that if f(x) = 3sin(2x - π/3) 
    and y = f(x + a) is an even function where 0 < a < π/2, then a = 5π/12 -/
theorem even_function_shift (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = 3 * Real.sin (2 * x - π / 3)) →
  (∀ x, f (x + a) = f (-x + a)) →
  (0 < a) →
  (a < π / 2) →
  a = 5 * π / 12 := by
sorry

end NUMINAMATH_CALUDE_even_function_shift_l3236_323632


namespace NUMINAMATH_CALUDE_chessboard_tiling_l3236_323633

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a polyomino -/
structure Polyomino :=
  (width : ℕ)
  (height : ℕ)

/-- Represents an L-shaped polyomino -/
def LPolyomino : Polyomino :=
  ⟨2, 2⟩

/-- Checks if a chessboard can be tiled with a given polyomino -/
def can_tile (board : Chessboard) (tile : Polyomino) : Prop :=
  ∃ n : ℕ, board.size * board.size = n * (tile.width * tile.height)

theorem chessboard_tiling (board : Chessboard) :
  board.size = 9 →
  ¬(can_tile board ⟨2, 1⟩) ∧
  (can_tile board ⟨3, 1⟩) ∧
  (can_tile board LPolyomino) :=
sorry

end NUMINAMATH_CALUDE_chessboard_tiling_l3236_323633


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3236_323696

theorem sum_of_four_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3236_323696


namespace NUMINAMATH_CALUDE_underpaid_amount_is_correct_l3236_323680

/-- Represents the time it takes for the minute hand to coincide with the hour hand once on an accurate clock (in minutes) -/
def accurate_clock_time : ℚ := 60 + 60 / 11

/-- Represents the time it takes for the minute hand to coincide with the hour hand once on the inaccurate clock (in minutes) -/
def inaccurate_clock_time : ℚ := 69

/-- Represents the hourly wage rate (in yuan) -/
def hourly_wage : ℚ := 6

/-- Represents the nominal workday length (in hours) -/
def nominal_workday : ℚ := 8

/-- Calculates the actual working time in a day (in hours) -/
def actual_working_time : ℚ :=
  nominal_workday * (inaccurate_clock_time / accurate_clock_time)

/-- Calculates the excess time worked (in hours) -/
def excess_time : ℚ := actual_working_time - nominal_workday

/-- Calculates the amount underpaid to each worker per day (in yuan) -/
def underpaid_amount : ℚ := hourly_wage * excess_time

theorem underpaid_amount_is_correct :
  underpaid_amount = 26 / 10 := by sorry

end NUMINAMATH_CALUDE_underpaid_amount_is_correct_l3236_323680


namespace NUMINAMATH_CALUDE_brain_info_scientific_notation_l3236_323658

/-- The number of pieces of information the human brain can record per day -/
def brain_info_capacity : ℕ := 86000000

/-- Scientific notation representation of brain_info_capacity -/
def brain_info_scientific : ℝ := 8.6 * (10 ^ 7)

theorem brain_info_scientific_notation :
  (brain_info_capacity : ℝ) = brain_info_scientific := by
  sorry

end NUMINAMATH_CALUDE_brain_info_scientific_notation_l3236_323658


namespace NUMINAMATH_CALUDE_child_height_end_of_year_l3236_323613

/-- Calculates the child's height at the end of the school year given initial height and growth rates -/
def final_height (initial_height : ℝ) (rate1 : ℝ) (rate2 : ℝ) (rate3 : ℝ) : ℝ :=
  initial_height + (3 * rate1) + (3 * rate2) + (6 * rate3)

/-- Theorem stating that the child's height at the end of the school year is 43.3 inches -/
theorem child_height_end_of_year :
  final_height 38.5 0.5 0.3 0.4 = 43.3 := by
  sorry

#eval final_height 38.5 0.5 0.3 0.4

end NUMINAMATH_CALUDE_child_height_end_of_year_l3236_323613


namespace NUMINAMATH_CALUDE_slower_train_speed_l3236_323678

/-- Proves the speed of a slower train given specific conditions -/
theorem slower_train_speed
  (train_length : ℝ)
  (faster_speed : ℝ)
  (passing_time : ℝ)
  (h_length : train_length = 75)
  (h_faster_speed : faster_speed = 46)
  (h_passing_time : passing_time = 54)
  : ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (2 * train_length) = (faster_speed - slower_speed) * (5/18) * passing_time :=
by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l3236_323678


namespace NUMINAMATH_CALUDE_rodney_lifts_118_l3236_323667

/-- Represents the weight-lifting abilities of Rebecca, Rodney, Roger, and Ron. -/
structure WeightLifters where
  rebecca : ℕ
  rodney : ℕ
  roger : ℕ
  ron : ℕ

/-- The conditions of the weight-lifting problem. -/
def weightLiftingConditions (w : WeightLifters) : Prop :=
  w.rebecca + w.rodney + w.roger + w.ron = 375 ∧
  w.rodney = 2 * w.roger ∧
  w.roger = w.ron + 5 ∧
  w.rebecca = 3 * w.ron - 20

/-- Theorem stating that under the given conditions, Rodney can lift 118 pounds. -/
theorem rodney_lifts_118 (w : WeightLifters) :
  weightLiftingConditions w → w.rodney = 118 := by
  sorry


end NUMINAMATH_CALUDE_rodney_lifts_118_l3236_323667


namespace NUMINAMATH_CALUDE_sum_r_s_equals_48_l3236_323652

/-- Parabola P with equation y = x^2 + 4x + 4 -/
def P : ℝ → ℝ := λ x => x^2 + 4*x + 4

/-- Point Q (10, 24) -/
def Q : ℝ × ℝ := (10, 24)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : ℝ → ℝ := λ x => m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line_through_Q m x

/-- Theorem: Sum of r and s equals 48 -/
theorem sum_r_s_equals_48 (r s : ℝ) 
  (h : ∀ m, no_intersection m ↔ r < m ∧ m < s) : 
  r + s = 48 := by sorry

end NUMINAMATH_CALUDE_sum_r_s_equals_48_l3236_323652


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l3236_323689

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  3 * a + Real.sqrt (4 * b + Real.sqrt (4 * b + Real.sqrt (4 * b + Real.sqrt (4 * b))))

-- Theorem statement
theorem bowtie_equation_solution (y : ℝ) : bowtie 5 y = 20 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l3236_323689


namespace NUMINAMATH_CALUDE_sequence_a_correct_l3236_323606

def sequence_a (n : ℕ) : ℚ :=
  (2 * 3^n) / (3^n - 1)

theorem sequence_a_correct (n : ℕ) : 
  n ≥ 1 → 
  sequence_a (n + 1) = (3^(n + 1) * sequence_a n) / (sequence_a n + 3^(n + 1)) ∧
  sequence_a 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_a_correct_l3236_323606


namespace NUMINAMATH_CALUDE_tobias_apps_downloaded_l3236_323697

/-- The number of apps downloaded by Tobias -/
def m : ℕ := 24

/-- The base cost of each app in cents -/
def base_cost : ℕ := 200

/-- The tax rate as a percentage -/
def tax_rate : ℕ := 10

/-- The total amount spent in cents -/
def total_spent : ℕ := 5280

/-- Theorem stating that m is the correct number of apps downloaded -/
theorem tobias_apps_downloaded :
  m * (base_cost + base_cost * tax_rate / 100) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_tobias_apps_downloaded_l3236_323697


namespace NUMINAMATH_CALUDE_olivia_remaining_money_l3236_323692

/-- Calculates the remaining money after a purchase with sales tax -/
def remaining_money (initial_amount purchase_amount tax_rate : ℚ) : ℚ :=
  initial_amount - (purchase_amount * (1 + tax_rate))

/-- Proves that given the specific conditions, the remaining money is $86.96 -/
theorem olivia_remaining_money :
  remaining_money 128 38 (8/100) = 86.96 := by
  sorry

end NUMINAMATH_CALUDE_olivia_remaining_money_l3236_323692


namespace NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l3236_323693

def is_palindrome (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ ∃ a b : ℕ, n = 1000 * a + 100 * b + 10 * b + a

theorem all_four_digit_palindromes_divisible_by_11 :
  ∀ n : ℕ, is_palindrome n → n % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l3236_323693


namespace NUMINAMATH_CALUDE_roots_equation_l3236_323682

theorem roots_equation (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 4 = 0 → x₂^2 + x₂ - 4 = 0 → x₁^3 - 5*x₂^2 + 10 = -19 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l3236_323682


namespace NUMINAMATH_CALUDE_email_sample_not_representative_l3236_323609

/-- Represents the urban population --/
def UrbanPopulation : Type := Unit

/-- Represents a person in the urban population --/
def Person : Type := Unit

/-- Represents whether a person has an email address --/
def has_email (p : Person) : Prop := sorry

/-- Represents whether a person uses the internet for news --/
def uses_internet_for_news (p : Person) : Prop := sorry

/-- Represents a sample of the population --/
def Sample := Set Person

/-- Defines what it means for a sample to be representative --/
def is_representative (s : Sample) : Prop := sorry

/-- The sample of email address owners --/
def email_sample : Sample := sorry

/-- Theorem stating that the sample of email address owners is not representative --/
theorem email_sample_not_representative :
  (∀ p : Person, has_email p → uses_internet_for_news p) →
  ¬ (is_representative email_sample) := by sorry

end NUMINAMATH_CALUDE_email_sample_not_representative_l3236_323609


namespace NUMINAMATH_CALUDE_comic_reconstruction_l3236_323684

theorem comic_reconstruction (pages_per_comic : ℕ) (torn_pages : ℕ) (untorn_comics : ℕ) : 
  pages_per_comic = 45 →
  torn_pages = 2700 →
  untorn_comics = 15 →
  (torn_pages / pages_per_comic + untorn_comics : ℕ) = 75 :=
by sorry

end NUMINAMATH_CALUDE_comic_reconstruction_l3236_323684


namespace NUMINAMATH_CALUDE_marbles_per_friend_l3236_323628

theorem marbles_per_friend (total_marbles : ℕ) (num_friends : ℕ) 
  (h1 : total_marbles = 72) (h2 : num_friends = 9) :
  total_marbles / num_friends = 8 :=
by sorry

end NUMINAMATH_CALUDE_marbles_per_friend_l3236_323628


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l3236_323610

theorem complex_magnitude_theorem (s : ℝ) (w : ℂ) 
  (h1 : |s| < 3) 
  (h2 : w + 2 / w = s) : 
  Complex.abs w = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l3236_323610


namespace NUMINAMATH_CALUDE_dereks_initial_money_l3236_323690

/-- Proves that Derek's initial amount of money was $40 -/
theorem dereks_initial_money :
  ∀ (derek_initial : ℕ) (derek_spent dave_initial dave_spent : ℕ),
  derek_spent = 30 →
  dave_initial = 50 →
  dave_spent = 7 →
  dave_initial - dave_spent = (derek_initial - derek_spent) + 33 →
  derek_initial = 40 := by
  sorry

end NUMINAMATH_CALUDE_dereks_initial_money_l3236_323690


namespace NUMINAMATH_CALUDE_custom_bowling_ball_volume_l3236_323657

/-- The volume of a customized bowling ball -/
theorem custom_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 6
  let small_hole_diameter : ℝ := 2.5
  let large_hole_diameter : ℝ := 4
  let sphere_volume := (4/3) * π * (sphere_diameter/2)^3
  let small_hole_volume := π * (small_hole_diameter/2)^2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter/2)^2 * hole_depth
  sphere_volume - 2 * small_hole_volume - large_hole_volume = 2261.25 * π :=
by sorry

end NUMINAMATH_CALUDE_custom_bowling_ball_volume_l3236_323657


namespace NUMINAMATH_CALUDE_expression_evaluation_l3236_323655

theorem expression_evaluation :
  let f (x : ℚ) := ((x + 1) / (x - 1) - 1) * ((x + 1) / (x - 1) + 1)
  f (-1/2) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3236_323655


namespace NUMINAMATH_CALUDE_f_greater_than_one_exists_max_a_l3236_323622

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * a * x^2

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Statement 1
theorem f_greater_than_one (x : ℝ) (h : x > 0) : f 2 x > 1 := by
  sorry

-- Statement 2
theorem exists_max_a :
  ∃ (a : ℕ), (∀ (x : ℝ), x > 0 → f_deriv a x ≥ x^2 * Real.log x) ∧
  (∀ (b : ℕ), (∀ (x : ℝ), x > 0 → f_deriv b x ≥ x^2 * Real.log x) → b ≤ a) := by
  sorry

end

end NUMINAMATH_CALUDE_f_greater_than_one_exists_max_a_l3236_323622


namespace NUMINAMATH_CALUDE_parabola_sum_l3236_323603

/-- A parabola with equation y = ax^2 + bx + c, vertex (-3, 4), vertical axis of symmetry, 
    and passing through (4, -2) has a + b + c = 100/49 -/
theorem parabola_sum (a b c : ℚ) : 
  (∀ x y : ℚ, y = a * x^2 + b * x + c ↔ 
    (x = -3 ∧ y = 4) ∨ 
    (x = 4 ∧ y = -2) ∨ 
    (∃ k : ℚ, y - 4 = k * (x + 3)^2)) →
  a + b + c = 100/49 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l3236_323603


namespace NUMINAMATH_CALUDE_annual_turbans_is_one_l3236_323608

/-- Represents the salary structure and conditions of Gopi's servant --/
structure SalaryStructure where
  annual_cash : ℕ  -- Annual cash salary in Rs.
  months_worked : ℕ  -- Number of months the servant worked
  cash_received : ℕ  -- Cash received by the servant
  turbans_received : ℕ  -- Number of turbans received by the servant
  turban_price : ℕ  -- Price of one turban in Rs.

/-- Calculates the number of turbans given as part of the annual salary --/
def calculate_annual_turbans (s : SalaryStructure) : ℕ :=
  -- Implementation not provided, use 'sorry'
  sorry

/-- Theorem stating that the number of turbans given annually is 1 --/
theorem annual_turbans_is_one (s : SalaryStructure) 
  (h1 : s.annual_cash = 90)
  (h2 : s.months_worked = 9)
  (h3 : s.cash_received = 45)
  (h4 : s.turbans_received = 1)
  (h5 : s.turban_price = 90) : 
  calculate_annual_turbans s = 1 := by
  sorry

end NUMINAMATH_CALUDE_annual_turbans_is_one_l3236_323608


namespace NUMINAMATH_CALUDE_expected_heads_is_60_l3236_323636

/-- The number of coins --/
def num_coins : ℕ := 64

/-- The maximum number of tosses for each coin --/
def max_tosses : ℕ := 4

/-- The probability of getting heads on a single toss --/
def p_heads : ℚ := 1/2

/-- The probability of getting heads after up to four tosses --/
def p_heads_four_tosses : ℚ := 
  p_heads + (1 - p_heads) * p_heads + (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The expected number of coins showing heads after up to four tosses --/
def expected_heads : ℚ := num_coins * p_heads_four_tosses

theorem expected_heads_is_60 : expected_heads = 60 := by sorry

end NUMINAMATH_CALUDE_expected_heads_is_60_l3236_323636


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l3236_323677

theorem movie_ticket_cost 
  (total_spent : ℚ)
  (num_tickets : ℕ)
  (rental_cost : ℚ)
  (purchase_cost : ℚ)
  (h1 : total_spent = 36.78)
  (h2 : num_tickets = 2)
  (h3 : rental_cost = 1.59)
  (h4 : purchase_cost = 13.95) :
  (total_spent - rental_cost - purchase_cost) / num_tickets = 10.62 := by
sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l3236_323677


namespace NUMINAMATH_CALUDE_possible_x_coordinates_of_p_l3236_323637

/-- Line l: x - y + 2 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

/-- Circle C: (x-2)^2 + y^2 = 2 -/
def circle_c (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

/-- Point A is the intersection of line l and x-axis -/
def point_a : ℝ × ℝ := (-2, 0)

/-- Point P is on line l -/
def point_p (x y : ℝ) : Prop := line_l x y

/-- There exists exactly one point B on circle C such that AB ⊥ BP -/
axiom exists_unique_b (p : ℝ × ℝ) : 
  ∃! b : ℝ × ℝ, circle_c b.1 b.2 ∧ 
  ((b.1 - point_a.1) * (p.1 - b.1) + (b.2 - point_a.2) * (p.2 - b.2) = 0)

/-- The set of possible x-coordinates of point P -/
def possible_x_coords : Set ℝ := {1/3, 5}

theorem possible_x_coordinates_of_p :
  ∀ x y : ℝ, point_p x y → x ∈ possible_x_coords :=
sorry

end NUMINAMATH_CALUDE_possible_x_coordinates_of_p_l3236_323637


namespace NUMINAMATH_CALUDE_snow_leopard_arrangements_l3236_323607

/-- The number of ways to arrange n different objects --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of snow leopards --/
def total_leopards : ℕ := 8

/-- The number of leopards that can be freely arranged --/
def free_leopards : ℕ := total_leopards - 2

/-- The number of ways to arrange the shortest and tallest leopards --/
def end_arrangements : ℕ := 2

theorem snow_leopard_arrangements :
  end_arrangements * permutations free_leopards = 1440 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangements_l3236_323607


namespace NUMINAMATH_CALUDE_min_sum_squares_l3236_323666

def S : Finset ℤ := {-8, -6, -4, -1, 1, 3, 5, 12}

theorem min_sum_squares (p q r s t u v w : ℤ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  2 ≤ (p + q + r + s)^2 + (t + u + v + w)^2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3236_323666


namespace NUMINAMATH_CALUDE_unique_prime_for_equal_sets_l3236_323660

theorem unique_prime_for_equal_sets (p : Nat) (g : Nat) : 
  Nat.Prime p → 
  p % 2 = 1 → 
  (∀ a : Nat, 1 ≤ a → a < p → g^a % p ≠ 1) → 
  g^(p-1) % p = 1 → 
  (∀ k : Nat, 1 ≤ k → k ≤ (p-1)/2 → ∃ m : Nat, 1 ≤ m ∧ m ≤ (p-1)/2 ∧ (k^2 + 1) % p = g^m % p) → 
  (∀ m : Nat, 1 ≤ m → m ≤ (p-1)/2 → ∃ k : Nat, 1 ≤ k ∧ k ≤ (p-1)/2 ∧ g^m % p = (k^2 + 1) % p) → 
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_for_equal_sets_l3236_323660


namespace NUMINAMATH_CALUDE_expression_simplification_l3236_323669

theorem expression_simplification : 
  ∃ (a b c : ℕ+), 
    (2 * Real.sqrt 3 + 2 / Real.sqrt 3 + 3 * Real.sqrt 2 + 3 / Real.sqrt 2 = (a * Real.sqrt 3 + b * Real.sqrt 2) / c) ∧
    (∀ (a' b' c' : ℕ+), 
      (2 * Real.sqrt 3 + 2 / Real.sqrt 3 + 3 * Real.sqrt 2 + 3 / Real.sqrt 2 = (a' * Real.sqrt 3 + b' * Real.sqrt 2) / c') →
      c ≤ c') ∧
    (a + b + c = 45) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3236_323669


namespace NUMINAMATH_CALUDE_new_alcohol_concentration_l3236_323619

/-- Calculates the new alcohol concentration after adding water to an alcohol solution -/
theorem new_alcohol_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 3)
  (h2 : initial_concentration = 0.33)
  (h3 : added_water = 1)
  : (initial_volume * initial_concentration) / (initial_volume + added_water) = 0.2475 := by
  sorry

end NUMINAMATH_CALUDE_new_alcohol_concentration_l3236_323619


namespace NUMINAMATH_CALUDE_max_distance_complex_numbers_l3236_323672

theorem max_distance_complex_numbers (z : ℂ) (h : Complex.abs z = 3) :
  Complex.abs ((2 + 3*Complex.I) * z^2 - z^4) ≤ 9*Real.sqrt 13 + 81 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_complex_numbers_l3236_323672


namespace NUMINAMATH_CALUDE_always_even_expression_l3236_323645

theorem always_even_expression (x y : ℕ) : 
  x ∈ Finset.range 15 → 
  y ∈ Finset.range 15 → 
  x ≠ y → 
  Even (x * y - 2 * x - 2 * y) := by
  sorry

#check always_even_expression

end NUMINAMATH_CALUDE_always_even_expression_l3236_323645


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l3236_323638

theorem smallest_x_for_equation : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∃ (y : ℕ), y > 0 ∧ (0.8 : ℚ) = y / (196 + x)) ∧
  (∀ (x' : ℕ), x' > 0 → x' < x → 
    ¬∃ (y : ℕ), y > 0 ∧ (0.8 : ℚ) = y / (196 + x')) ∧
  x = 49 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l3236_323638


namespace NUMINAMATH_CALUDE_difference_of_squares_l3236_323659

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3236_323659


namespace NUMINAMATH_CALUDE_nina_widget_purchase_l3236_323670

/-- The problem of determining how many widgets Nina can purchase --/
theorem nina_widget_purchase (total_money : ℚ) (reduced_price_quantity : ℕ) (price_reduction : ℚ) : 
  total_money = 27.6 →
  reduced_price_quantity = 8 →
  price_reduction = 1.15 →
  (reduced_price_quantity : ℚ) * ((total_money / (reduced_price_quantity : ℚ)) - price_reduction) = total_money →
  (total_money / (total_money / (reduced_price_quantity : ℚ))).floor = 6 :=
by sorry

end NUMINAMATH_CALUDE_nina_widget_purchase_l3236_323670


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3236_323605

/-- Given two rectangles with equal area, where one rectangle measures 12 inches by 15 inches
    and the other has a width of 30 inches, the length of the second rectangle is 6 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_width : ℝ) 
  (h1 : carol_length = 12)
  (h2 : carol_width = 15)
  (h3 : jordan_width = 30)
  (h4 : carol_length * carol_width = jordan_width * (carol_length * carol_width / jordan_width)) :
  carol_length * carol_width / jordan_width = 6 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l3236_323605


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l3236_323642

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem chocolate_milk_probability : 
  binomial_probability 7 5 (1/2) = 21/128 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l3236_323642


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l3236_323674

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l3236_323674


namespace NUMINAMATH_CALUDE_angle_at_two_fifteen_l3236_323694

/-- Represents a clock with hour and minute hands -/
structure Clock where
  hourHand : ℝ  -- Position of hour hand (in hours)
  minuteHand : ℝ  -- Position of minute hand (in minutes)

/-- Calculates the angle between hour and minute hands at a given time -/
def angleBetweenHands (c : Clock) : ℝ :=
  let hourAngle := c.hourHand * 30 + c.minuteHand * 0.5
  let minuteAngle := c.minuteHand * 6
  abs (hourAngle - minuteAngle)

/-- Theorem stating that at 2:15, the angle between hour and minute hands is 22.5° -/
theorem angle_at_two_fifteen :
  let c : Clock := { hourHand := 2, minuteHand := 15 }
  angleBetweenHands c = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_angle_at_two_fifteen_l3236_323694


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3236_323617

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 3)^2 < 1 ↔ -4 < x ∧ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3236_323617


namespace NUMINAMATH_CALUDE_area_not_covered_by_square_l3236_323644

/-- Given a rectangle with dimensions 10 units by 8 units and an inscribed square
    with side length 5 units, the area of the region not covered by the square
    is 55 square units. -/
theorem area_not_covered_by_square (rectangle_length : ℝ) (rectangle_width : ℝ) 
    (square_side : ℝ) (h1 : rectangle_length = 10) (h2 : rectangle_width = 8) 
    (h3 : square_side = 5) : 
    rectangle_length * rectangle_width - square_side^2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_area_not_covered_by_square_l3236_323644


namespace NUMINAMATH_CALUDE_stating_snail_reaches_top_l3236_323675

/-- Represents the height of the tree in meters -/
def tree_height : ℕ := 10

/-- Represents the distance the snail climbs during the day in meters -/
def day_climb : ℕ := 4

/-- Represents the distance the snail slips at night in meters -/
def night_slip : ℕ := 3

/-- Calculates the net distance the snail moves in one day -/
def net_daily_progress : ℤ := day_climb - night_slip

/-- Represents the number of days it takes for the snail to reach the top -/
def days_to_reach_top : ℕ := 7

/-- 
Theorem stating that the snail reaches the top of the tree in 7 days
given the defined tree height, day climb, and night slip distances.
-/
theorem snail_reaches_top : 
  (days_to_reach_top - 1) * net_daily_progress + day_climb ≥ tree_height :=
sorry

end NUMINAMATH_CALUDE_stating_snail_reaches_top_l3236_323675


namespace NUMINAMATH_CALUDE_length_of_ac_l3236_323604

/-- Given 5 consecutive points on a straight line, prove that the length of ac is 11 -/
theorem length_of_ac (a b c d e : Real) : 
  (b - a) = 5 →
  (c - b) = 3 * (d - c) →
  (e - d) = 7 →
  (e - a) = 20 →
  (c - a) = 11 :=
by sorry

end NUMINAMATH_CALUDE_length_of_ac_l3236_323604


namespace NUMINAMATH_CALUDE_island_ocean_depth_l3236_323699

/-- Represents a cone-shaped island -/
structure ConeIsland where
  height : ℝ
  volumeAboveWater : ℝ

/-- Calculates the depth of the ocean at the base of a cone-shaped island -/
def oceanDepth (island : ConeIsland) : ℝ :=
  sorry

/-- Theorem stating the depth of the ocean for the given island -/
theorem island_ocean_depth :
  let island : ConeIsland := { height := 10000, volumeAboveWater := 1/10 }
  oceanDepth island = 350 :=
sorry

end NUMINAMATH_CALUDE_island_ocean_depth_l3236_323699


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_l3236_323668

/-- The opposite number of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- Theorem: The opposite number of -3 is 3 -/
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_l3236_323668


namespace NUMINAMATH_CALUDE_inequality_chain_l3236_323648

theorem inequality_chain (b a x : ℝ) (h1 : b > a) (h2 : a > x) (h3 : x > 0) :
  x^2 < x*a ∧ x*a < a^2 ∧ a^2 < x*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3236_323648


namespace NUMINAMATH_CALUDE_domain_of_f_squared_l3236_323664

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x - 2)
def domain_f_shifted : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem domain_of_f_squared (h : ∀ x ∈ domain_f_shifted, f (x - 2) = f (x - 2)) :
  {x : ℝ | ∃ y, f (y^2) = x} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_domain_of_f_squared_l3236_323664


namespace NUMINAMATH_CALUDE_vector_operation_l3236_323623

def a : Fin 2 → ℝ := ![2, 4]
def b : Fin 2 → ℝ := ![-1, 1]

theorem vector_operation : 
  (2 • a - b) = ![5, 7] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3236_323623


namespace NUMINAMATH_CALUDE_alternating_color_probability_alternating_color_probability_value_l3236_323654

/-- The probability of drawing 10 balls with alternating colors (starting and ending with the same color) from a box containing 5 white and 5 black balls. -/
theorem alternating_color_probability : ℚ :=
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let successful_sequences : ℕ := 2
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  successful_sequences / total_arrangements

/-- The probability of drawing 10 balls with alternating colors (starting and ending with the same color) from a box containing 5 white and 5 black balls is 1/126. -/
theorem alternating_color_probability_value : alternating_color_probability = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_alternating_color_probability_value_l3236_323654


namespace NUMINAMATH_CALUDE_rhombus_area_l3236_323635

/-- The area of a rhombus with side length √145 and diagonals differing by 10 units is 100 square units. -/
theorem rhombus_area (side_length : ℝ) (diagonal_difference : ℝ) (area : ℝ) :
  side_length = Real.sqrt 145 →
  diagonal_difference = 10 →
  area = 100 →
  ∃ (d1 d2 : ℝ), d1 > 0 ∧ d2 > 0 ∧ 
    d2 - d1 = diagonal_difference ∧
    d1 * d2 / 2 = area ∧
    d1^2 / 4 + d2^2 / 4 = side_length^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l3236_323635


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3236_323624

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Two vectors are parallel -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : parallel (a 2, 2) (a 3, 3)) :
  (a 2 + a 4) / (a 3 + a 5) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3236_323624


namespace NUMINAMATH_CALUDE_percentage_equality_l3236_323656

theorem percentage_equality (x y : ℝ) (h1 : 2.5 * x = 0.75 * y) (h2 : x = 20) : y = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3236_323656


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l3236_323621

theorem fraction_sum_difference : (3 / 50 + 2 / 25 - 5 / 1000 : ℚ) = 0.135 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l3236_323621


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l3236_323641

/-- The x-coordinate of a point on the parabola y^2 = 4x that is 4 units away from the focus -/
theorem parabola_point_x_coordinate (x y : ℝ) : 
  y^2 = 4*x →                            -- Point (x, y) is on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 4^2 →                -- Distance from (x, y) to focus (1, 0) is 4
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l3236_323641


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l3236_323620

theorem trig_expression_equals_negative_four :
  (Real.sqrt 3 * Real.sin (10 * π / 180) - Real.cos (10 * π / 180)) /
  (Real.cos (10 * π / 180) * Real.sin (10 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l3236_323620


namespace NUMINAMATH_CALUDE_expansion_equality_l3236_323661

theorem expansion_equality (x : ℝ) : (x + 6) * (x - 1) = x^2 + 5*x - 6 := by sorry

end NUMINAMATH_CALUDE_expansion_equality_l3236_323661


namespace NUMINAMATH_CALUDE_curve_through_center_l3236_323625

-- Define the square
structure Square where
  center : ℝ × ℝ

-- Define the curve
structure Curve where
  -- A function that takes a real number parameter and returns a point on the curve
  pointAt : ℝ → ℝ × ℝ

-- Define the property that the curve divides the square into two equal areas
def divides_equally (s : Square) (γ : Curve) : Prop :=
  -- This is a placeholder for the actual condition
  sorry

-- Define the property that a line segment passes through a point
def passes_through (a b c : ℝ × ℝ) : Prop :=
  -- This is a placeholder for the actual condition
  sorry

-- The main theorem
theorem curve_through_center (s : Square) (γ : Curve) 
  (h : divides_equally s γ) : 
  ∃ (a b : ℝ × ℝ), (∃ (t₁ t₂ : ℝ), γ.pointAt t₁ = a ∧ γ.pointAt t₂ = b) ∧ 
    passes_through a b s.center := by
  sorry

end NUMINAMATH_CALUDE_curve_through_center_l3236_323625


namespace NUMINAMATH_CALUDE_least_distinct_values_l3236_323691

theorem least_distinct_values (list : List ℕ+) : 
  list.length = 2023 →
  ∃! m : ℕ+, (list.count m = 11 ∧ ∀ n : ℕ+, n ≠ m → list.count n < 11) →
  (∀ k : ℕ+, k < 203 → ∃ x : ℕ+, list.count x > list.count k) →
  ∃ x : ℕ+, list.count x = 203 :=
by sorry

end NUMINAMATH_CALUDE_least_distinct_values_l3236_323691


namespace NUMINAMATH_CALUDE_min_value_of_f_l3236_323639

/-- The function we want to minimize -/
def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x - 8*y + 6

theorem min_value_of_f :
  (∀ x y : ℝ, f x y ≥ 0) ∧ f (-1) 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3236_323639


namespace NUMINAMATH_CALUDE_polygon_not_covered_by_homothetic_polygons_l3236_323671

/-- A polygon in a 2D space -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  -- Add more properties as needed

/-- Homothetic transformation of a polygon -/
def homothetic_transform (p : Polygon) (center : ℝ × ℝ) (k : ℝ) : Polygon :=
  sorry

/-- Predicate to check if a point is contained in a polygon -/
def point_in_polygon (point : ℝ × ℝ) (p : Polygon) : Prop :=
  sorry

theorem polygon_not_covered_by_homothetic_polygons 
  (M : Polygon) (k : ℝ) (h : 0 < k ∧ k < 1) :
  ∃ (point : ℝ × ℝ), 
    point_in_polygon point M ∧
    ∀ (center1 center2 : ℝ × ℝ),
      ¬(point_in_polygon point (homothetic_transform M center1 k) ∨
        point_in_polygon point (homothetic_transform M center2 k)) :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_not_covered_by_homothetic_polygons_l3236_323671


namespace NUMINAMATH_CALUDE_lucy_deposit_l3236_323640

def initial_balance : ℕ := 65
def withdrawal : ℕ := 4
def final_balance : ℕ := 76

theorem lucy_deposit :
  ∃ (deposit : ℕ), initial_balance + deposit - withdrawal = final_balance :=
by
  sorry

end NUMINAMATH_CALUDE_lucy_deposit_l3236_323640
