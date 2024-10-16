import Mathlib

namespace NUMINAMATH_CALUDE_product_sum_theorem_l57_5736

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156) 
  (h2 : a + b + c = 16) : 
  a * b + b * c + a * c = 50 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l57_5736


namespace NUMINAMATH_CALUDE_mushroom_problem_solution_l57_5758

/-- Represents a basket of mushrooms with two types: ryzhiki and gruzdi -/
structure MushroomBasket where
  total : ℕ
  ryzhiki : ℕ
  gruzdi : ℕ
  sum_eq_total : ryzhiki + gruzdi = total

/-- Predicate to check if the basket satisfies the ryzhiki condition -/
def has_ryzhik_in_12 (b : MushroomBasket) : Prop :=
  ∀ n : ℕ, n ≤ b.total - 12 → b.ryzhiki > n

/-- Predicate to check if the basket satisfies the gruzdi condition -/
def has_gruzd_in_20 (b : MushroomBasket) : Prop :=
  ∀ n : ℕ, n ≤ b.total - 20 → b.gruzdi > n

/-- Theorem stating the solution to the mushroom problem -/
theorem mushroom_problem_solution :
  ∀ b : MushroomBasket,
  b.total = 30 →
  has_ryzhik_in_12 b →
  has_gruzd_in_20 b →
  b.ryzhiki = 19 ∧ b.gruzdi = 11 := by
  sorry


end NUMINAMATH_CALUDE_mushroom_problem_solution_l57_5758


namespace NUMINAMATH_CALUDE_fraction_product_equals_twelve_l57_5737

theorem fraction_product_equals_twelve :
  (1/3) * (9/2) * (1/27) * 54 * (1/81) * 162 * (1/243) * 486 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_twelve_l57_5737


namespace NUMINAMATH_CALUDE_right_triangle_side_ratio_l57_5700

theorem right_triangle_side_ratio (a d : ℝ) (ha : a > 0) (hd : d > 0) :
  (a^2 + (a + d)^2 = (a + 2*d)^2) → (a = 3*d) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_ratio_l57_5700


namespace NUMINAMATH_CALUDE_exponent_division_l57_5799

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by sorry

end NUMINAMATH_CALUDE_exponent_division_l57_5799


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l57_5778

-- Define the type for quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to check if a point (x, y) is in a given quadrant
def in_quadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.First => x > 0 ∧ y > 0
  | Quadrant.Second => x < 0 ∧ y > 0
  | Quadrant.Third => x < 0 ∧ y < 0
  | Quadrant.Fourth => x > 0 ∧ y < 0

-- Define a function to check if a line passes through a quadrant
def line_passes_through (m b : ℝ) (q : Quadrant) : Prop :=
  ∃ (x y : ℝ), y = m * x + b ∧ in_quadrant x y q

-- State the theorem
theorem line_passes_through_quadrants
  (a b c p : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h1 : (a + b) / c = p)
  (h2 : (b + c) / a = p)
  (h3 : (c + a) / b = p) :
  line_passes_through p p Quadrant.Second ∧
  line_passes_through p p Quadrant.Third :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l57_5778


namespace NUMINAMATH_CALUDE_sum_of_numbers_l57_5702

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y = 16) (h4 : 1 / x = 3 * (1 / y)) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l57_5702


namespace NUMINAMATH_CALUDE_combined_work_theorem_l57_5718

/-- The number of days it takes for three workers to complete a task together,
    given their individual completion times. -/
def combinedWorkDays (raviDays prakashDays seemaDays : ℚ) : ℚ :=
  1 / (1 / raviDays + 1 / prakashDays + 1 / seemaDays)

/-- Theorem stating that if Ravi can do the work in 50 days, Prakash in 75 days,
    and Seema in 60 days, they will finish the work together in 20 days. -/
theorem combined_work_theorem :
  combinedWorkDays 50 75 60 = 20 := by sorry

end NUMINAMATH_CALUDE_combined_work_theorem_l57_5718


namespace NUMINAMATH_CALUDE_total_revenue_calculation_l57_5726

/-- Calculates the total revenue from selling various reading materials -/
theorem total_revenue_calculation (magazines newspapers books pamphlets : ℕ) 
  (magazine_price newspaper_price book_price pamphlet_price : ℚ) : 
  magazines = 425 → 
  newspapers = 275 → 
  books = 150 → 
  pamphlets = 75 → 
  magazine_price = 5/2 → 
  newspaper_price = 3/2 → 
  book_price = 5 → 
  pamphlet_price = 1/2 → 
  (magazines : ℚ) * magazine_price + 
  (newspapers : ℚ) * newspaper_price + 
  (books : ℚ) * book_price + 
  (pamphlets : ℚ) * pamphlet_price = 2262.5 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_calculation_l57_5726


namespace NUMINAMATH_CALUDE_smallest_modulus_w_l57_5776

theorem smallest_modulus_w (w : ℂ) (h : Complex.abs (w - 8) + Complex.abs (w - 3 * I) = 15) :
  ∃ (w_min : ℂ), Complex.abs w_min ≤ Complex.abs w ∧ Complex.abs w_min = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_modulus_w_l57_5776


namespace NUMINAMATH_CALUDE_square_pattern_properties_l57_5707

/-- Represents the number of squares in Figure n of the pattern --/
def num_squares (n : ℕ+) : ℕ := 3 + 2 * (n - 1)

/-- Represents the perimeter of Figure n of the pattern --/
def perimeter (n : ℕ+) : ℕ := 8 + 4 * (n - 1)

/-- Theorem stating the properties of the square pattern --/
theorem square_pattern_properties (n : ℕ+) :
  (num_squares n = 3 + 2 * (n - 1)) ∧ (perimeter n = 8 + 4 * (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_square_pattern_properties_l57_5707


namespace NUMINAMATH_CALUDE_calculation_proof_l57_5781

theorem calculation_proof : (30 / (7 + 2 - 6)) * 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l57_5781


namespace NUMINAMATH_CALUDE_equality_of_fractions_l57_5746

theorem equality_of_fractions (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + N = 183 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l57_5746


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_retailer_profit_is_65_percent_l57_5760

/-- Calculates the profit percentage for a retailer selling pens -/
theorem retailer_profit_percentage 
  (num_pens : ℕ) 
  (cost_price : ℝ) 
  (market_price : ℝ) 
  (discount_rate : ℝ) : ℝ :=
  let selling_price := num_pens * (market_price * (1 - discount_rate))
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

/-- Proves that the retailer's profit percentage is 65% under given conditions -/
theorem retailer_profit_is_65_percent : 
  retailer_profit_percentage 60 36 1 0.01 = 65 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_retailer_profit_is_65_percent_l57_5760


namespace NUMINAMATH_CALUDE_point_symmetry_l57_5786

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

theorem point_symmetry (M N P : Point) (hM : M = Point.mk (-4) 3)
    (hN : symmetricOrigin M N) (hP : symmetricYAxis N P) :
    P = Point.mk 4 3 := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l57_5786


namespace NUMINAMATH_CALUDE_base_conversion_problem_l57_5798

theorem base_conversion_problem (a b : ℕ) : 
  (a < 10 ∧ b < 10) → -- Ensuring a and b are single digits
  (6 * 7^2 + 5 * 7 + 6 = 300 + 10 * a + b) → -- 656₇ = 3ab₁₀
  (a * b) / 15 = 1 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l57_5798


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l57_5770

-- Define the triangle
structure IsoscelesTriangle where
  base_angle : Real
  inscribed_square_side : Real
  inscribed_circle_radius : Real

-- Define the conditions
def triangle_conditions (t : IsoscelesTriangle) : Prop :=
  t.inscribed_square_side / t.inscribed_circle_radius = 8 / 5

-- Theorem statement
theorem isosceles_triangle_angles (t : IsoscelesTriangle) 
  (h : triangle_conditions t) : 
  t.base_angle = 2 * Real.arctan (1 / 2) ∧ 
  π - 2 * t.base_angle = π - 4 * Real.arctan (1 / 2) := by
  sorry

#check isosceles_triangle_angles

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l57_5770


namespace NUMINAMATH_CALUDE_square_side_length_l57_5703

theorem square_side_length (rectangle_side1 rectangle_side2 : ℝ) 
  (h1 : rectangle_side1 = 9)
  (h2 : rectangle_side2 = 16) :
  ∃ (square_side : ℝ), 
    square_side * square_side = rectangle_side1 * rectangle_side2 ∧ 
    square_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l57_5703


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l57_5779

theorem largest_prime_divisor (crayons paper : ℕ) 
  (h1 : crayons = 385) (h2 : paper = 95) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ crayons ∧ p ∣ paper ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ crayons → q ∣ paper → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l57_5779


namespace NUMINAMATH_CALUDE_tan_identity_l57_5710

theorem tan_identity (α β γ n : Real) 
  (h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)) : 
  Real.tan (α + β + γ) = (n + 1) / (n - 1) * Real.tan (α - β + γ) := by
  sorry

end NUMINAMATH_CALUDE_tan_identity_l57_5710


namespace NUMINAMATH_CALUDE_expected_vote_percentage_a_l57_5742

/-- Percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 70

/-- Percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 100 - democrat_percentage

/-- Percentage of Democrats expected to vote for candidate A -/
def democrat_vote_a : ℝ := 80

/-- Percentage of Republicans expected to vote for candidate A -/
def republican_vote_a : ℝ := 30

/-- Theorem stating the percentage of registered voters expected to vote for candidate A -/
theorem expected_vote_percentage_a : 
  (democrat_percentage / 100 * democrat_vote_a + 
   republican_percentage / 100 * republican_vote_a) = 65 := by
  sorry

end NUMINAMATH_CALUDE_expected_vote_percentage_a_l57_5742


namespace NUMINAMATH_CALUDE_arrangements_not_head_tail_six_arrangements_not_adjacent_six_l57_5772

/-- The number of students in the row -/
def n : ℕ := 6

/-- The number of arrangements where one student doesn't stand at the head or tail -/
def arrangements_not_head_tail (n : ℕ) : ℕ := sorry

/-- The number of arrangements where three specific students are not adjacent -/
def arrangements_not_adjacent (n : ℕ) : ℕ := sorry

/-- Theorem for the first question -/
theorem arrangements_not_head_tail_six : 
  arrangements_not_head_tail n = 480 := by sorry

/-- Theorem for the second question -/
theorem arrangements_not_adjacent_six : 
  arrangements_not_adjacent n = 144 := by sorry

end NUMINAMATH_CALUDE_arrangements_not_head_tail_six_arrangements_not_adjacent_six_l57_5772


namespace NUMINAMATH_CALUDE_ellipse_min_reciprocal_sum_l57_5706

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_min_reciprocal_sum :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
  (1 / distance P left_focus + 1 / distance P right_focus ≥ 1) ∧
  (∃ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 ∧
    1 / distance Q left_focus + 1 / distance Q right_focus = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_min_reciprocal_sum_l57_5706


namespace NUMINAMATH_CALUDE_moe_has_least_money_l57_5709

-- Define the set of people
inductive Person : Type
  | Bo | Coe | Flo | Jo | Moe | Zoe

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom flo_more_than_jo_and_bo : money Person.Flo > money Person.Jo ∧ money Person.Flo > money Person.Bo
axiom bo_and_coe_more_than_moe : money Person.Bo > money Person.Moe ∧ money Person.Coe > money Person.Moe
axiom jo_between_bo_and_moe : money Person.Bo > money Person.Jo ∧ money Person.Jo > money Person.Moe
axiom zoe_between_jo_and_moe : money Person.Jo > money Person.Zoe ∧ money Person.Zoe > money Person.Moe

-- Theorem to prove
theorem moe_has_least_money : ∀ (p : Person), p ≠ Person.Moe → money Person.Moe < money p := by
  sorry


end NUMINAMATH_CALUDE_moe_has_least_money_l57_5709


namespace NUMINAMATH_CALUDE_square_rotation_octagon_l57_5787

/-- Represents a regular polygon with n sides -/
structure RegularPolygon where
  sides : ℕ
  mk_sides_pos : sides > 0

/-- Represents a square -/
structure Square

/-- Represents the position of an object on a square -/
inductive Position
  | Top
  | Right
  | Bottom
  | Left

/-- Calculates the inner angle of a regular polygon -/
def inner_angle (p : RegularPolygon) : ℚ :=
  (p.sides - 2 : ℚ) * 180 / p.sides

/-- Calculates the rotation per movement when a square rolls around a regular polygon -/
def rotation_per_movement (p : RegularPolygon) : ℚ :=
  360 - (inner_angle p + 90)

/-- Theorem: After a full rotation around an octagon, an object on a square returns to its original position -/
theorem square_rotation_octagon (s : Square) (initial_pos : Position) :
  let octagon : RegularPolygon := ⟨8, by norm_num⟩
  let total_rotation : ℚ := 8 * rotation_per_movement octagon
  total_rotation % 360 = 0 → initial_pos = Position.Bottom → initial_pos = Position.Bottom :=
by
  sorry


end NUMINAMATH_CALUDE_square_rotation_octagon_l57_5787


namespace NUMINAMATH_CALUDE_article_count_l57_5794

theorem article_count (cost_price selling_price : ℝ) (gain_percentage : ℝ) : 
  gain_percentage = 42.857142857142854 →
  50 * cost_price = 35 * selling_price →
  selling_price = cost_price * (1 + gain_percentage / 100) →
  35 = 50 * (100 / (100 + gain_percentage)) :=
by sorry

end NUMINAMATH_CALUDE_article_count_l57_5794


namespace NUMINAMATH_CALUDE_original_quantities_l57_5769

/-- The original planned quantities of products A and B -/
def original_plan (x y : ℕ) : Prop :=
  ∃ (a b : ℝ), 
    -- Original plan: spend 1500 yuan
    a * x + b * y = 1500 ∧
    -- New scenario 1
    (a + 1.5) * (x - 10) + (b + 1) * y = 1529 ∧
    -- New scenario 2
    (a + 1) * (x - 5) + (b + 1) * y = 1563.5 ∧
    -- Constraint
    205 < 2 * x + y ∧ 2 * x + y < 210

theorem original_quantities : 
  ∃ (x y : ℕ), original_plan x y ∧ x = 76 ∧ y = 55 := by
  sorry

end NUMINAMATH_CALUDE_original_quantities_l57_5769


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l57_5763

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 24 → 
  Even heart → 
  (∀ h c : ℕ, h * c = 24 → Even h → heart + club ≥ h + c) →
  heart + club = 14 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l57_5763


namespace NUMINAMATH_CALUDE_simplify_expression_l57_5728

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (20 * x^2) * (5 * x) * (1 / (2 * x)^2) * (2 * x)^2 = 100 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l57_5728


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l57_5724

theorem laptop_sticker_price :
  ∀ (sticker_price : ℝ),
    (0.8 * sticker_price - 120 = 0.7 * sticker_price - 18) →
    sticker_price = 1020 := by
  sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l57_5724


namespace NUMINAMATH_CALUDE_wall_width_l57_5774

theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  l = 7 * h →
  volume = w * h * l →
  volume = 6804 →
  w = 3 := by
sorry

end NUMINAMATH_CALUDE_wall_width_l57_5774


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l57_5740

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + m - 1 = 0 ∧ y^2 - 4*y + m - 1 = 0) → m < 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l57_5740


namespace NUMINAMATH_CALUDE_range_of_b_over_a_l57_5795

def quadratic_equation (a b x : ℝ) : ℝ := x^2 + (a+1)*x + a + b + 1

theorem range_of_b_over_a (a b : ℝ) (x₁ x₂ : ℝ) :
  (∃ x, quadratic_equation a b x = 0) →
  (x₁ ≠ x₂) →
  (quadratic_equation a b x₁ = 0) →
  (quadratic_equation a b x₂ = 0) →
  (0 < x₁ ∧ x₁ < 1) →
  (x₂ > 1) →
  (-2 < b/a ∧ b/a < -1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_over_a_l57_5795


namespace NUMINAMATH_CALUDE_repeating_decimal_three_six_equals_eleven_thirtieths_l57_5717

def repeating_decimal (a b : ℕ) : ℚ :=
  (a : ℚ) / 10 + (b : ℚ) / (9 * 10)

theorem repeating_decimal_three_six_equals_eleven_thirtieths :
  repeating_decimal 3 6 = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_three_six_equals_eleven_thirtieths_l57_5717


namespace NUMINAMATH_CALUDE_triangle_with_100_degree_angle_is_obtuse_l57_5788

/-- A triangle is obtuse if it has an interior angle greater than 90 degrees. -/
def IsObtuse (a b c : ℝ) : Prop := 
  (a + b + c = 180) ∧ (max a (max b c) > 90)

/-- If a triangle has an interior angle of 100 degrees, then it is obtuse. -/
theorem triangle_with_100_degree_angle_is_obtuse (a b c : ℝ) : 
  (a + b + c = 180) → (max a (max b c) = 100) → IsObtuse a b c := by
  sorry

#check triangle_with_100_degree_angle_is_obtuse

end NUMINAMATH_CALUDE_triangle_with_100_degree_angle_is_obtuse_l57_5788


namespace NUMINAMATH_CALUDE_round_trip_time_ratio_l57_5762

/-- Proves that for a round trip with given average speeds, the ratio of return to outbound journey times is 3:2 -/
theorem round_trip_time_ratio 
  (distance : ℝ) 
  (speed_to_destination : ℝ) 
  (average_speed_round_trip : ℝ) 
  (h1 : speed_to_destination = 54) 
  (h2 : average_speed_round_trip = 36) 
  (h3 : distance > 0) 
  (h4 : speed_to_destination > 0) 
  (h5 : average_speed_round_trip > 0) : 
  (distance / average_speed_round_trip - distance / speed_to_destination) / (distance / speed_to_destination) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_time_ratio_l57_5762


namespace NUMINAMATH_CALUDE_bag_of_balls_l57_5725

theorem bag_of_balls (num_black : ℕ) (prob_black : ℚ) (total : ℕ) : 
  num_black = 4 → prob_black = 1/3 → total = num_black / prob_black → total = 12 := by
sorry

end NUMINAMATH_CALUDE_bag_of_balls_l57_5725


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l57_5729

theorem mystery_book_shelves (books_per_shelf : ℕ) (picture_book_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 4 →
  picture_book_shelves = 3 →
  total_books = 32 →
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 5 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l57_5729


namespace NUMINAMATH_CALUDE_not_parabola_l57_5723

theorem not_parabola (α x y : ℝ) : 
  ∃ (a b c : ℝ), ∀ (x y : ℝ), x^2 * Real.sin α + y^2 * Real.cos α = 1 → y ≠ a*x^2 + b*x + c :=
sorry

end NUMINAMATH_CALUDE_not_parabola_l57_5723


namespace NUMINAMATH_CALUDE_smallest_common_rose_purchase_l57_5753

theorem smallest_common_rose_purchase : Nat.lcm 9 19 = 171 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_rose_purchase_l57_5753


namespace NUMINAMATH_CALUDE_player_two_wins_l57_5704

/-- Number of contacts in the microcircuit -/
def num_contacts : ℕ := 2000

/-- Total number of wires initially -/
def total_wires : ℕ := num_contacts * (num_contacts - 1) / 2

/-- Represents a player in the game -/
inductive Player
| One
| Two

/-- Represents the state of the game -/
structure GameState where
  remaining_wires : ℕ
  current_player : Player

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (wires_cut : ℕ) : Prop :=
  match state.current_player with
  | Player.One => wires_cut = 1
  | Player.Two => wires_cut = 2 ∨ wires_cut = 3

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.remaining_wires = 0

/-- Theorem stating that Player Two has a winning strategy -/
theorem player_two_wins : 
  ∃ (strategy : GameState → ℕ), 
    ∀ (game : GameState), 
      game.remaining_wires > 0 → 
      game.current_player = Player.Two → 
      valid_move game (strategy game) ∧ 
      (∀ (opponent_move : ℕ), 
        valid_move (GameState.mk (game.remaining_wires - strategy game) Player.One) opponent_move → 
        is_winning_state (GameState.mk (game.remaining_wires - strategy game - opponent_move) Player.Two)) :=
sorry

end NUMINAMATH_CALUDE_player_two_wins_l57_5704


namespace NUMINAMATH_CALUDE_statement_a_correct_statement_b_correct_statement_c_incorrect_statement_d_correct_l57_5739

-- Statement A
theorem statement_a_correct (x y a : ℝ) : x^2 = y^2 → -3*a*x^2 = -3*a*y^2 := by sorry

-- Statement B
theorem statement_b_correct (x y a : ℝ) (h : a ≠ 0) : x / a = y / a → x = y := by sorry

-- Statement C (incorrect)
theorem statement_c_incorrect : ∃ a b c : ℝ, a * c = b * c ∧ a ≠ b := by sorry

-- Statement D
theorem statement_d_correct (a b : ℝ) : a = b → a^2 = b^2 := by sorry

end NUMINAMATH_CALUDE_statement_a_correct_statement_b_correct_statement_c_incorrect_statement_d_correct_l57_5739


namespace NUMINAMATH_CALUDE_hours_to_minutes_l57_5791

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the number of hours Ava watched television
def hours_watched : ℕ := 4

-- Theorem to prove
theorem hours_to_minutes :
  hours_watched * minutes_per_hour = 240 := by
  sorry

end NUMINAMATH_CALUDE_hours_to_minutes_l57_5791


namespace NUMINAMATH_CALUDE_printer_cost_l57_5756

/-- The cost of a single printer given the total cost of keyboards and printers, 
    the number of each item, and the cost of a single keyboard. -/
theorem printer_cost 
  (total_cost : ℕ) 
  (num_keyboards num_printers : ℕ) 
  (keyboard_cost : ℕ) 
  (h1 : total_cost = 2050)
  (h2 : num_keyboards = 15)
  (h3 : num_printers = 25)
  (h4 : keyboard_cost = 20) :
  (total_cost - num_keyboards * keyboard_cost) / num_printers = 70 :=
by sorry

end NUMINAMATH_CALUDE_printer_cost_l57_5756


namespace NUMINAMATH_CALUDE_sin_minus_cos_equals_one_l57_5743

theorem sin_minus_cos_equals_one (x : Real) : 
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = 1 ↔ x = Real.pi / 2 ∨ x = Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_equals_one_l57_5743


namespace NUMINAMATH_CALUDE_derivative_at_pi_sixth_l57_5711

theorem derivative_at_pi_sixth (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = Real.cos x - Real.sin x) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' (π/6) = -(1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_sixth_l57_5711


namespace NUMINAMATH_CALUDE_triangle_side_expression_l57_5735

theorem triangle_side_expression (m : ℝ) : 
  (2 : ℝ) > 0 ∧ 5 > 0 ∧ m > 0 ∧ 
  2 + 5 > m ∧ 2 + m > 5 ∧ 5 + m > 2 →
  Real.sqrt ((m - 3)^2) + Real.sqrt ((m - 7)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_expression_l57_5735


namespace NUMINAMATH_CALUDE_fraction_decomposition_l57_5721

theorem fraction_decomposition (n : ℕ) (h : n ≥ 3) :
  (2 : ℚ) / (2 * n - 1) = 1 / n + 1 / (n * (2 * n - 1)) := by
  sorry

#check fraction_decomposition

end NUMINAMATH_CALUDE_fraction_decomposition_l57_5721


namespace NUMINAMATH_CALUDE_line_plane_relationships_l57_5730

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relationships between lines and planes
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry
def parallel_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry
def perpendicular_line_line (l1 : Line) (l2 : Line) : Prop := sorry
def parallel_line_line (l1 : Line) (l2 : Line) : Prop := sorry

-- Define the theorem
theorem line_plane_relationships 
  (m n : Line) (α β : Plane) 
  (hm : m ≠ n) (hα : α ≠ β) : 
  (perpendicular_line_plane m α ∧ 
   perpendicular_line_plane n β ∧ 
   perpendicular_plane_plane α β → 
   perpendicular_line_line m n) ∧
  (¬ (parallel_line_plane m α ∧ 
      perpendicular_line_plane n β ∧ 
      perpendicular_plane_plane α β → 
      parallel_line_line m n)) ∧
  (perpendicular_line_plane m α ∧ 
   parallel_line_plane n β ∧ 
   parallel_plane_plane α β → 
   perpendicular_line_line m n) ∧
  (perpendicular_line_plane m α ∧ 
   perpendicular_line_plane n β ∧ 
   parallel_plane_plane α β → 
   parallel_line_line m n) := by
  sorry


end NUMINAMATH_CALUDE_line_plane_relationships_l57_5730


namespace NUMINAMATH_CALUDE_continuous_additive_function_is_linear_l57_5749

-- Define the property of the function
def SatisfiesAdditiveProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

-- State the theorem
theorem continuous_additive_function_is_linear
  (f : ℝ → ℝ)
  (hf_cont : Continuous f)
  (hf_add : SatisfiesAdditiveProperty f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_CALUDE_continuous_additive_function_is_linear_l57_5749


namespace NUMINAMATH_CALUDE_equation_real_root_implies_a_range_l57_5766

theorem equation_real_root_implies_a_range (a : ℝ) :
  (∃ x : ℝ, 2^(2*x) + 2^x * a + a + 1 = 0) →
  a ∈ Set.Iic (2 - 2 * Real.sqrt 2) ∪ Set.Ici (2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_real_root_implies_a_range_l57_5766


namespace NUMINAMATH_CALUDE_distance_on_number_line_l57_5748

theorem distance_on_number_line : 
  let point_a : ℝ := 3
  let point_b : ℝ := -2
  |point_a - point_b| = 5 := by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_l57_5748


namespace NUMINAMATH_CALUDE_alternative_plan_savings_l57_5755

/-- Proves that the alternative phone plan is $1 cheaper than the current plan -/
theorem alternative_plan_savings :
  ∀ (current_plan_cost : ℚ)
    (texts_sent : ℕ)
    (call_minutes : ℕ)
    (text_package_size : ℕ)
    (call_package_size : ℕ)
    (text_package_cost : ℚ)
    (call_package_cost : ℚ),
  current_plan_cost = 12 →
  texts_sent = 60 →
  call_minutes = 60 →
  text_package_size = 30 →
  call_package_size = 20 →
  text_package_cost = 1 →
  call_package_cost = 3 →
  current_plan_cost - 
    ((texts_sent / text_package_size : ℚ) * text_package_cost +
     (call_minutes / call_package_size : ℚ) * call_package_cost) = 1 :=
by
  sorry

#check alternative_plan_savings

end NUMINAMATH_CALUDE_alternative_plan_savings_l57_5755


namespace NUMINAMATH_CALUDE_f_composition_l57_5712

def f (x : ℝ) : ℝ := 2 * x - 1

theorem f_composition (x : ℝ) : f (x - 1) = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_l57_5712


namespace NUMINAMATH_CALUDE_sum_of_bases_equal_1193_l57_5720

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 14 to base 10 -/
def base14_to_base10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 14 -/
def C : ℕ := 12

theorem sum_of_bases_equal_1193 :
  base8_to_base10 356 + base14_to_base10 (4 * 14^2 + C * 14 + 3) = 1193 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equal_1193_l57_5720


namespace NUMINAMATH_CALUDE_next_door_neighbor_subscriptions_l57_5792

/-- Represents the number of subscriptions sold to the next-door neighbor -/
def next_door_subscriptions : ℕ := sorry

/-- Represents the total number of subscriptions sold -/
def total_subscriptions : ℕ := sorry

/-- The amount earned per subscription -/
def amount_per_subscription : ℕ := 5

/-- The total amount earned -/
def total_amount_earned : ℕ := 55

/-- Subscriptions sold to parents -/
def parent_subscriptions : ℕ := 4

/-- Subscriptions sold to grandfather -/
def grandfather_subscriptions : ℕ := 1

theorem next_door_neighbor_subscriptions :
  (next_door_subscriptions * amount_per_subscription +
   2 * next_door_subscriptions * amount_per_subscription +
   parent_subscriptions * amount_per_subscription +
   grandfather_subscriptions * amount_per_subscription = total_amount_earned) →
  (total_subscriptions = total_amount_earned / amount_per_subscription) →
  (next_door_subscriptions = 2) := by
  sorry

end NUMINAMATH_CALUDE_next_door_neighbor_subscriptions_l57_5792


namespace NUMINAMATH_CALUDE_largest_expression_l57_5708

theorem largest_expression :
  let a := 3 + 1 + 2 + 8
  let b := 3 * 1 + 2 + 8
  let c := 3 + 1 * 2 + 8
  let d := 3 + 1 + 2 * 8
  let e := 3 * 1 * 2 * 8
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l57_5708


namespace NUMINAMATH_CALUDE_students_in_both_subjects_range_l57_5705

def total_students : ℕ := 3000

def history_min : ℕ := 2100
def history_max : ℕ := 2250

def psychology_min : ℕ := 1200
def psychology_max : ℕ := 1500

theorem students_in_both_subjects_range :
  ∃ (min_both max_both : ℕ),
    (∀ (h p both : ℕ),
      history_min ≤ h ∧ h ≤ history_max →
      psychology_min ≤ p ∧ p ≤ psychology_max →
      h + p - both = total_students →
      min_both ≤ both ∧ both ≤ max_both) ∧
    max_both - min_both = 450 :=
sorry

end NUMINAMATH_CALUDE_students_in_both_subjects_range_l57_5705


namespace NUMINAMATH_CALUDE_equation_solution_l57_5741

theorem equation_solution (x : ℝ) : (x + 6) / (x - 3) = 4 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l57_5741


namespace NUMINAMATH_CALUDE_segment_length_ratio_l57_5782

/-- Given two line segments with points placed at equal intervals, 
    prove that the longer segment is 101 times the length of the shorter segment. -/
theorem segment_length_ratio 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_points_a : ∃ (d : ℝ), a = 99 * d ∧ d > 0) 
  (h_points_b : ∃ (d : ℝ), b = 9999 * d ∧ d > 0) 
  (h_same_interval : ∀ (d1 d2 : ℝ), (a = 99 * d1 ∧ d1 > 0) → (b = 9999 * d2 ∧ d2 > 0) → d1 = d2) :
  b = 101 * a := by
  sorry

end NUMINAMATH_CALUDE_segment_length_ratio_l57_5782


namespace NUMINAMATH_CALUDE_flour_needed_for_loaves_l57_5757

/-- The number of cups of flour needed for one loaf of bread -/
def flour_per_loaf : ℝ := 2.5

/-- The number of loaves of bread to be baked -/
def number_of_loaves : ℕ := 2

/-- Theorem: The total number of cups of flour needed for baking the desired number of loaves is 5 -/
theorem flour_needed_for_loaves : flour_per_loaf * (number_of_loaves : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_for_loaves_l57_5757


namespace NUMINAMATH_CALUDE_gus_egg_consumption_l57_5751

/-- The number of eggs Gus ate for breakfast -/
def breakfast_eggs : ℕ := 2

/-- The number of eggs Gus ate for lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs Gus ate for dinner -/
def dinner_eggs : ℕ := 1

/-- The total number of eggs Gus ate -/
def total_eggs : ℕ := breakfast_eggs + lunch_eggs + dinner_eggs

theorem gus_egg_consumption : total_eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_gus_egg_consumption_l57_5751


namespace NUMINAMATH_CALUDE_sum_of_even_integers_between_0_and_18_l57_5780

theorem sum_of_even_integers_between_0_and_18 : 
  (Finset.filter (fun n => n % 2 = 0) (Finset.range 18)).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_between_0_and_18_l57_5780


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l57_5790

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 3) (h2 : x * y = 1) : x^2 + y^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l57_5790


namespace NUMINAMATH_CALUDE_min_distance_parabola_circle_l57_5714

theorem min_distance_parabola_circle : 
  let parabola := {P : ℝ × ℝ | P.2^2 = P.1}
  let circle := {Q : ℝ × ℝ | (Q.1 - 3)^2 + Q.2^2 = 1}
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ parabola ∧ Q ∈ circle ∧
    ∀ (P' : ℝ × ℝ) (Q' : ℝ × ℝ), P' ∈ parabola → Q' ∈ circle →
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = (Real.sqrt 11 - 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_circle_l57_5714


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l57_5767

/-- The lateral surface area of a cylinder with given circumference and height -/
def lateral_surface_area (circumference : ℝ) (height : ℝ) : ℝ :=
  circumference * height

/-- Theorem: The lateral surface area of a cylinder with circumference 5cm and height 2cm is 10 cm² -/
theorem cylinder_lateral_area :
  lateral_surface_area 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l57_5767


namespace NUMINAMATH_CALUDE_theater_seats_l57_5716

/-- Represents a theater with ticket sales information -/
structure Theater where
  adultTicketPrice : ℕ
  childTicketPrice : ℕ
  totalRevenue : ℕ
  childTicketsSold : ℕ

/-- Calculates the total number of seats in the theater -/
def totalSeats (t : Theater) : ℕ :=
  t.childTicketsSold + ((t.totalRevenue - t.childTicketPrice * t.childTicketsSold) / t.adultTicketPrice)

/-- Theorem stating that the theater has 80 seats -/
theorem theater_seats (t : Theater) 
  (h1 : t.adultTicketPrice = 12)
  (h2 : t.childTicketPrice = 5)
  (h3 : t.totalRevenue = 519)
  (h4 : t.childTicketsSold = 63) :
  totalSeats t = 80 := by
  sorry

#eval totalSeats ⟨12, 5, 519, 63⟩

end NUMINAMATH_CALUDE_theater_seats_l57_5716


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l57_5784

-- Problem 1
theorem problem_1 : Real.sqrt 12 + (-1/3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2*a / (a^2 - 4)) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l57_5784


namespace NUMINAMATH_CALUDE_expression_evaluation_l57_5783

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l57_5783


namespace NUMINAMATH_CALUDE_quadratic_solution_with_nested_root_l57_5701

theorem quadratic_solution_with_nested_root (a b : ℤ) :
  (∃ x : ℝ, x^2 + a*x + b = 0 ∧ x = Real.sqrt (2010 + 2 * Real.sqrt 2009)) →
  a = 0 ∧ b = -2008 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_with_nested_root_l57_5701


namespace NUMINAMATH_CALUDE_intersection_value_l57_5796

def A : Set ℕ := {1, 3, 5}
def B (m : ℕ) : Set ℕ := {1, m}

theorem intersection_value (m : ℕ) : A ∩ B m = {1, m} → m = 3 ∨ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l57_5796


namespace NUMINAMATH_CALUDE_simplify_expression_l57_5773

theorem simplify_expression (a : ℝ) : (2 : ℝ) * (2 * a) * (4 * a^2) * (3 * a^3) * (6 * a^4) = 288 * a^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l57_5773


namespace NUMINAMATH_CALUDE_total_erasers_l57_5797

/-- Given an initial number of erasers and a number of erasers added, 
    the total number of erasers is equal to the sum of the initial number and the added number. -/
theorem total_erasers (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_erasers_l57_5797


namespace NUMINAMATH_CALUDE_x_range_l57_5750

theorem x_range (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l57_5750


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l57_5793

theorem solution_set_of_inequality (x : ℝ) : 
  (x-1)/(x^2-x-6) ≥ 0 ↔ x ∈ Set.Ioc (-2) 1 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l57_5793


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l57_5754

theorem cubic_equation_roots (p q : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 - Complex.I : ℂ) ^ 3 + p * (2 - Complex.I : ℂ) ^ 2 + q * (2 - Complex.I : ℂ) - 6 = 0 →
  p = -26/5 ∧ q = 49/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l57_5754


namespace NUMINAMATH_CALUDE_apple_boxes_l57_5785

theorem apple_boxes (class_5A class_5B class_5C : ℕ) 
  (h1 : class_5A = 560)
  (h2 : class_5B = 595)
  (h3 : class_5C = 735) :
  let box_weight := Nat.gcd class_5A (Nat.gcd class_5B class_5C)
  (class_5A / box_weight, class_5B / box_weight, class_5C / box_weight) = (16, 17, 21) := by
  sorry

#check apple_boxes

end NUMINAMATH_CALUDE_apple_boxes_l57_5785


namespace NUMINAMATH_CALUDE_max_profit_at_70_optimal_selling_price_l57_5713

def purchase_price : ℕ := 40
def initial_selling_price : ℕ := 50
def initial_sales_volume : ℕ := 500
def price_increment : ℕ := 1
def sales_volume_decrement : ℕ := 10

def profit (x : ℕ) : ℤ :=
  (initial_sales_volume - sales_volume_decrement * x) * (initial_selling_price + x) -
  (initial_sales_volume - sales_volume_decrement * x) * purchase_price

theorem max_profit_at_70 :
  ∀ x : ℕ, x ≤ 50 → profit x ≤ profit 20 := by sorry

theorem optimal_selling_price :
  ∃ x : ℕ, x ≤ 50 ∧ ∀ y : ℕ, y ≤ 50 → profit y ≤ profit x :=
by
  use 20
  sorry

#eval initial_selling_price + 20

end NUMINAMATH_CALUDE_max_profit_at_70_optimal_selling_price_l57_5713


namespace NUMINAMATH_CALUDE_city_distance_ratio_l57_5764

/-- Prove that the ratio of distances between cities is 2:1 --/
theorem city_distance_ratio :
  ∀ (AB BC CD AD : ℝ),
  AB = 100 →
  BC = AB + 50 →
  AD = 550 →
  AD = AB + BC + CD →
  ∃ (k : ℝ), CD = k * BC →
  CD / BC = 2 := by
sorry

end NUMINAMATH_CALUDE_city_distance_ratio_l57_5764


namespace NUMINAMATH_CALUDE_q_polynomial_form_l57_5731

/-- Given a function q(x) satisfying the equation
    q(x) + (2x^6 + 5x^4 + 10x^2) = (9x^4 + 30x^3 + 50x^2 + 4),
    prove that q(x) = -2x^6 + 4x^4 + 30x^3 + 40x^2 + 4 -/
theorem q_polynomial_form (q : ℝ → ℝ) 
    (h : ∀ x, q x + (2 * x^6 + 5 * x^4 + 10 * x^2) = 9 * x^4 + 30 * x^3 + 50 * x^2 + 4) :
  ∀ x, q x = -2 * x^6 + 4 * x^4 + 30 * x^3 + 40 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l57_5731


namespace NUMINAMATH_CALUDE_maintenance_check_time_l57_5765

theorem maintenance_check_time (initial_time : ℝ) : 
  (initial_time + (1/3) * initial_time = 60) → initial_time = 45 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_time_l57_5765


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l57_5732

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 30 * x + 10 = 0) : 
  ∃ x, b * x^2 + 30 * x + 10 = 0 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l57_5732


namespace NUMINAMATH_CALUDE_employed_female_parttime_ratio_is_60_percent_l57_5747

/-- Represents the population statistics of Town P -/
structure TownP where
  total_population : ℝ
  employed_percentage : ℝ
  employed_female_percentage : ℝ
  employed_female_parttime_percentage : ℝ
  employed_male_percentage : ℝ

/-- Calculates the percentage of employed females who are part-time workers in Town P -/
def employed_female_parttime_ratio (town : TownP) : ℝ :=
  town.employed_female_parttime_percentage

/-- Theorem stating that 60% of employed females in Town P are part-time workers -/
theorem employed_female_parttime_ratio_is_60_percent (town : TownP) 
  (h1 : town.employed_percentage = 0.6)
  (h2 : town.employed_female_percentage = 0.4)
  (h3 : town.employed_female_parttime_percentage = 0.6)
  (h4 : town.employed_male_percentage = 0.48) :
  employed_female_parttime_ratio town = 0.6 := by
  sorry

#check employed_female_parttime_ratio_is_60_percent

end NUMINAMATH_CALUDE_employed_female_parttime_ratio_is_60_percent_l57_5747


namespace NUMINAMATH_CALUDE_three_digit_number_difference_l57_5745

theorem three_digit_number_difference (X Y : ℕ) : 
  X > Y → 
  X + Y = 999 → 
  X ≥ 100 → 
  X ≤ 999 → 
  Y ≥ 100 → 
  Y ≤ 999 → 
  1000 * X + Y = 6 * (1000 * Y + X) → 
  X - Y = 715 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_difference_l57_5745


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l57_5775

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 6*x + y^2 + 2*y - 11 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 10*x + y^2 - 8*y + 25 = 0}
  (shortest_distance : ℝ) →
  shortest_distance = Real.sqrt 89 - Real.sqrt 21 - 4 ∧
  ∀ (p1 : ℝ × ℝ) (p2 : ℝ × ℝ), 
    p1 ∈ circle1 → p2 ∈ circle2 → 
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) ≥ shortest_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_between_circles_l57_5775


namespace NUMINAMATH_CALUDE_smallest_odd_minimizer_l57_5722

/-- The number of positive integer divisors of n, including 1 and n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The function g(n) = d(n) / n^(1/4) -/
noncomputable def g (n : ℕ) : ℝ := (d n : ℝ) / n^(1/4 : ℝ)

/-- n is an odd positive integer -/
def isOddPositive (n : ℕ) : Prop := n > 0 ∧ n % 2 = 1

/-- 9 is the smallest odd positive integer N such that g(N) < g(n) for all odd positive integers n ≠ N -/
theorem smallest_odd_minimizer :
  isOddPositive 9 ∧
  (∀ n : ℕ, isOddPositive n → n ≠ 9 → g 9 < g n) ∧
  (∀ N : ℕ, isOddPositive N → N < 9 → ∃ n : ℕ, isOddPositive n ∧ n ≠ N ∧ g N ≥ g n) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_minimizer_l57_5722


namespace NUMINAMATH_CALUDE_traffic_light_probability_l57_5761

theorem traffic_light_probability (p_A p_B p_C : ℚ) 
  (h_A : p_A = 1/3) (h_B : p_B = 1/2) (h_C : p_C = 2/3) : 
  (1 - p_A) * p_B * p_C + p_A * (1 - p_B) * p_C + p_A * p_B * (1 - p_C) = 7/18 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l57_5761


namespace NUMINAMATH_CALUDE_linear_function_properties_l57_5759

def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_properties :
  ∃ (k b : ℝ), 
    k ≠ 0 ∧
    linear_function k b 1 = 2 ∧
    linear_function k b (-1) = 4 ∧
    linear_function (-1) 3 = linear_function k b ∧
    linear_function (-1) 3 2 ≠ 3 ∧
    linear_function (-1) 3 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l57_5759


namespace NUMINAMATH_CALUDE_odd_number_product_difference_l57_5777

theorem odd_number_product_difference (x : ℤ) : 
  Odd x → x * (x + 2) - x * (x - 2) = 44 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_product_difference_l57_5777


namespace NUMINAMATH_CALUDE_initial_cd_count_l57_5738

theorem initial_cd_count (initial : ℕ) : 
  (2 / 3 : ℚ) * initial + 8 = 22 → initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_cd_count_l57_5738


namespace NUMINAMATH_CALUDE_sum_of_possible_radii_l57_5771

/-- Given a circle with center C(r,r) that is tangent to the positive x-axis,
    positive y-axis, and externally tangent to a circle centered at (4,0) with radius 1,
    the sum of all possible radii of circle C is 10. -/
theorem sum_of_possible_radii : ∀ r : ℝ,
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 1)^2) →
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧
    ((r₁ - 4)^2 + r₁^2 = (r₁ + 1)^2) ∧
    ((r₂ - 4)^2 + r₂^2 = (r₂ + 1)^2) ∧
    r₁ + r₂ = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_radii_l57_5771


namespace NUMINAMATH_CALUDE_divisible_by_3_4_5_count_l57_5727

theorem divisible_by_3_4_5_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 50 ∧ (3 ∣ n ∨ 4 ∣ n ∨ 5 ∣ n)) ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 50 ∧ (3 ∣ n ∨ 4 ∣ n ∨ 5 ∣ n) → n ∈ S) ∧ 
  Finset.card S = 29 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_3_4_5_count_l57_5727


namespace NUMINAMATH_CALUDE_projectile_max_height_l57_5734

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 10

/-- Theorem stating that the maximum height of the projectile is 30 -/
theorem projectile_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 30 :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l57_5734


namespace NUMINAMATH_CALUDE_inequality_proof_l57_5744

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l57_5744


namespace NUMINAMATH_CALUDE_max_value_f_on_interval_l57_5789

def f (x : ℝ) := x^3 - 3*x^2 + 2

theorem max_value_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 1 ∧ 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x ≤ f c) ∧
  f c = 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_f_on_interval_l57_5789


namespace NUMINAMATH_CALUDE_lily_initial_money_l57_5768

def celery_cost : ℝ := 5
def cereal_original_cost : ℝ := 12
def cereal_discount : ℝ := 0.5
def bread_cost : ℝ := 8
def milk_original_cost : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_cost : ℝ := 1
def potato_quantity : ℕ := 6
def coffee_budget : ℝ := 26

def total_cost : ℝ := 
  celery_cost + 
  cereal_original_cost * (1 - cereal_discount) + 
  bread_cost + 
  milk_original_cost * (1 - milk_discount) + 
  potato_cost * potato_quantity +
  coffee_budget

theorem lily_initial_money : total_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_lily_initial_money_l57_5768


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l57_5752

def total_students : ℕ := 5
def male_students : ℕ := 3
def female_students : ℕ := 2
def selected_students : ℕ := 3

theorem probability_at_least_one_female :
  let total_combinations := Nat.choose total_students selected_students
  let all_male_combinations := Nat.choose male_students selected_students
  (1 : ℚ) - (all_male_combinations : ℚ) / (total_combinations : ℚ) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l57_5752


namespace NUMINAMATH_CALUDE_fifth_grade_students_l57_5715

theorem fifth_grade_students (total_boys : ℕ) (soccer_players : ℕ) (boys_soccer_percentage : ℚ) (girls_not_soccer : ℕ) :
  total_boys = 296 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  girls_not_soccer = 89 →
  ∃ (total_students : ℕ), total_students = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_grade_students_l57_5715


namespace NUMINAMATH_CALUDE_circle_radius_l57_5733

theorem circle_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + k = 0 ↔ (x - 4)^2 + (y + 5)^2 = 5^2) → 
  k = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l57_5733


namespace NUMINAMATH_CALUDE_notebook_cost_l57_5719

theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (total_cost : notebook_cost + pencil_cost = 2.40)
  (cost_difference : notebook_cost = pencil_cost + 2) : 
  notebook_cost = 2.20 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l57_5719
