import Mathlib

namespace imaginary_difference_condition_l2289_228946

def is_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem imaginary_difference_condition (z₁ z₂ : ℂ) :
  (is_imaginary (z₁ - z₂) → (is_imaginary z₁ ∨ is_imaginary z₂)) ∧
  ∃ z₁ z₂ : ℂ, (is_imaginary z₁ ∨ is_imaginary z₂) ∧ ¬is_imaginary (z₁ - z₂) :=
sorry

end imaginary_difference_condition_l2289_228946


namespace middle_integer_is_six_l2289_228927

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧
  b = a + 2 ∧ c = b + 2 ∧
  a + b + c = (a * b * c) / 8

theorem middle_integer_is_six :
  ∀ a b c : ℕ, is_valid_triple a b c → b = 6 :=
by sorry

end middle_integer_is_six_l2289_228927


namespace music_shop_total_cost_l2289_228973

/-- Calculates the total cost of CDs purchased from a music shop --/
theorem music_shop_total_cost 
  (life_journey_price : ℝ) 
  (life_journey_discount : ℝ) 
  (day_life_price : ℝ) 
  (rescind_price : ℝ) 
  (life_journey_quantity : ℕ) 
  (day_life_quantity : ℕ) 
  (rescind_quantity : ℕ) : 
  life_journey_price = 100 →
  life_journey_discount = 0.2 →
  day_life_price = 50 →
  rescind_price = 85 →
  life_journey_quantity = 3 →
  day_life_quantity = 4 →
  rescind_quantity = 2 →
  (life_journey_quantity * (life_journey_price * (1 - life_journey_discount))) +
  ((day_life_quantity / 2) * day_life_price) +
  (rescind_quantity * rescind_price) = 510 := by
sorry

end music_shop_total_cost_l2289_228973


namespace vector_dot_product_properties_l2289_228981

/-- Given two vectors in ℝ², prove dot product properties --/
theorem vector_dot_product_properties (a b : ℝ × ℝ) 
    (h1 : a = (1, 2)) 
    (h2 : b = (2, -3)) : 
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 9) ∧ 
  ((a.1 + b.1) * (a.1 - (1/9) * b.1) + (a.2 + b.2) * (a.2 - (1/9) * b.2) = 0) := by
  sorry

end vector_dot_product_properties_l2289_228981


namespace neds_weekly_revenue_l2289_228939

/-- Calculates the weekly revenue for Ned's left-handed mouse store -/
def calculate_weekly_revenue (normal_mouse_price : ℝ) (price_increase_percentage : ℝ) 
  (daily_sales : ℕ) (open_days_per_week : ℕ) : ℝ :=
  let left_handed_mouse_price := normal_mouse_price * (1 + price_increase_percentage)
  let daily_revenue := left_handed_mouse_price * daily_sales
  daily_revenue * open_days_per_week

/-- Theorem stating that Ned's weekly revenue is $15600 -/
theorem neds_weekly_revenue : 
  calculate_weekly_revenue 120 0.3 25 4 = 15600 := by
  sorry

#eval calculate_weekly_revenue 120 0.3 25 4

end neds_weekly_revenue_l2289_228939


namespace rectangle_diagonal_triangle_area_l2289_228905

/-- The area of a triangle formed by the diagonal and one side of a rectangle with length 40 units and width 24 units is 480 square units. -/
theorem rectangle_diagonal_triangle_area : 
  let rectangle_length : ℝ := 40
  let rectangle_width : ℝ := 24
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let triangle_area : ℝ := rectangle_area / 2
  triangle_area = 480 := by sorry

end rectangle_diagonal_triangle_area_l2289_228905


namespace b_4_lt_b_7_l2289_228915

def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1 + 1 / α 1
  | n+1 => 1 + 1 / (b n α + 1 / α (n+1))

theorem b_4_lt_b_7 (α : ℕ → ℕ) : b 4 α < b 7 α := by
  sorry

end b_4_lt_b_7_l2289_228915


namespace bus_ride_cost_l2289_228974

theorem bus_ride_cost (train_cost bus_cost : ℝ) : 
  train_cost = bus_cost + 6.85 →
  (train_cost * 0.85 + (bus_cost + 1.25)) = 10.50 →
  bus_cost = 1.85 := by
  sorry

end bus_ride_cost_l2289_228974


namespace f_f_zero_equals_3pi_squared_minus_4_l2289_228951

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

-- Theorem statement
theorem f_f_zero_equals_3pi_squared_minus_4 :
  f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end f_f_zero_equals_3pi_squared_minus_4_l2289_228951


namespace arthur_walk_distance_l2289_228956

/-- The distance Arthur walked in miles -/
def distance_walked (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem: Arthur walks 4.5 miles -/
theorem arthur_walk_distance :
  distance_walked 8 10 (1/4) = 4.5 := by
  sorry

end arthur_walk_distance_l2289_228956


namespace third_term_expansion_l2289_228958

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of the third term in (3b+2a)^6
def third_term_coefficient : ℕ := binomial 6 2 * 3^4 * 2^2

-- Theorem statement
theorem third_term_expansion :
  third_term_coefficient = 4860 ∧ binomial 6 2 = 15 := by sorry

end third_term_expansion_l2289_228958


namespace line_parabola_intersection_l2289_228920

/-- A line with equation y = kx + 2 and a parabola with equation y² = 8x have exactly one point in common if and only if k = 1 or k = 0 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.2^2 = 8 * p.1) ↔ (k = 1 ∨ k = 0) :=
sorry

end line_parabola_intersection_l2289_228920


namespace total_spent_is_40_l2289_228992

def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def num_ingredients : ℕ := 5
def apron_cost : ℕ := recipe_book_cost + 1

def total_cost : ℕ := recipe_book_cost + baking_dish_cost + (ingredient_cost * num_ingredients) + apron_cost

theorem total_spent_is_40 : total_cost = 40 := by
  sorry

end total_spent_is_40_l2289_228992


namespace victors_specific_earnings_l2289_228953

/-- Victor's earnings over two days given his hourly wage and hours worked each day -/
def victors_earnings (hourly_wage : ℕ) (hours_monday : ℕ) (hours_tuesday : ℕ) : ℕ :=
  hourly_wage * (hours_monday + hours_tuesday)

/-- Theorem: Victor's earnings over two days given specific conditions -/
theorem victors_specific_earnings :
  victors_earnings 6 5 5 = 60 := by
  sorry

end victors_specific_earnings_l2289_228953


namespace altitudes_sum_eq_nine_inradius_implies_equilateral_l2289_228952

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from a vertex to the opposite side of a triangle -/
def altitude (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- The radius of the inscribed circle of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- 
If the sum of the altitudes of a triangle is equal to nine times 
the radius of its inscribed circle, then the triangle is equilateral 
-/
theorem altitudes_sum_eq_nine_inradius_implies_equilateral (t : Triangle) :
  altitude t t.A + altitude t t.B + altitude t t.C = 9 * inradius t →
  is_equilateral t := by
  sorry

end altitudes_sum_eq_nine_inradius_implies_equilateral_l2289_228952


namespace attention_index_properties_l2289_228985

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 10 then 100 * a^(x/10) - 60
  else if 10 < x ∧ x ≤ 20 then 340
  else if 20 < x ∧ x ≤ 40 then 640 - 15*x
  else 0

theorem attention_index_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 5 = 140) :
  a = 4 ∧ 
  f a 5 > f a 35 ∧ 
  (Set.Icc 5 (100/3) : Set ℝ) = {x | 0 ≤ x ∧ x ≤ 40 ∧ f a x ≥ 140} :=
by sorry

end attention_index_properties_l2289_228985


namespace valid_paintings_count_l2289_228993

/-- Represents a 3x3 grid of squares that can be painted green or red -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Bool :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Checks if a grid painting is valid (no green square adjacent to a red square) -/
def valid_painting (g : Grid) : Bool :=
  ∀ p1 p2 : Fin 3 × Fin 3, adjacent p1 p2 → (g p1.1 p1.2 = g p2.1 p2.2)

/-- Counts the number of valid grid paintings -/
def count_valid_paintings : Nat :=
  (List.filter valid_painting (List.map (λf : Fin 9 → Bool => λi j => f (3 * i + j)) 
    (List.map (λn : Fin 512 => λi => n.val.testBit i) (List.range 512)))).length

/-- The main theorem stating that the number of valid paintings is 10 -/
theorem valid_paintings_count : count_valid_paintings = 10 := by
  sorry

end valid_paintings_count_l2289_228993


namespace unique_triple_l2289_228910

theorem unique_triple : ∃! (x y z : ℕ+), 
  x ≤ y ∧ y ≤ z ∧ 
  x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧
  x = 2 ∧ y = 251 ∧ z = 252 := by
  sorry

end unique_triple_l2289_228910


namespace complex_solutions_count_l2289_228957

theorem complex_solutions_count : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, (z^3 - 8) / (z^2 - 3*z + 2) = 0) ∧ 
  (∀ z : ℂ, (z^3 - 8) / (z^2 - 3*z + 2) = 0 → z ∈ S) ∧ 
  Finset.card S = 2 := by
  sorry

end complex_solutions_count_l2289_228957


namespace smallest_triangle_leg_l2289_228994

/-- Represents a 45-45-90 triangle -/
structure Triangle45 where
  hypotenuse : ℝ
  leg : ℝ
  hyp_leg_relation : leg = hypotenuse / Real.sqrt 2

/-- A sequence of four 45-45-90 triangles where the hypotenuse of one is the leg of the next -/
def TriangleSequence (t1 t2 t3 t4 : Triangle45) : Prop :=
  t1.leg = t2.hypotenuse ∧ t2.leg = t3.hypotenuse ∧ t3.leg = t4.hypotenuse

theorem smallest_triangle_leg 
  (t1 t2 t3 t4 : Triangle45) 
  (seq : TriangleSequence t1 t2 t3 t4) 
  (largest_hyp : t1.hypotenuse = 16) : 
  t4.leg = 4 := by
  sorry

end smallest_triangle_leg_l2289_228994


namespace closest_integer_to_sqrt35_l2289_228990

theorem closest_integer_to_sqrt35 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 35| ≤ |m - Real.sqrt 35| ∧ n = 6 :=
sorry

end closest_integer_to_sqrt35_l2289_228990


namespace triangle_rectangle_equal_area_l2289_228912

theorem triangle_rectangle_equal_area (s h : ℝ) (s_pos : 0 < s) :
  (1 / 2) * s * h = 2 * s^2 → h = 4 * s :=
by sorry

end triangle_rectangle_equal_area_l2289_228912


namespace interest_rate_proof_l2289_228986

/-- Given a principal sum P, if the simple interest on P for 4 years is one-fifth of P,
    then the rate of interest per annum is 25%. -/
theorem interest_rate_proof (P : ℝ) (P_pos : P > 0) : 
  (P * 25 * 4) / 100 = P / 5 → 25 = 100 * (P / 5) / (P * 4) := by
sorry

end interest_rate_proof_l2289_228986


namespace sqrt_three_div_sqrt_one_third_eq_three_l2289_228989

theorem sqrt_three_div_sqrt_one_third_eq_three : 
  Real.sqrt 3 / Real.sqrt (1/3) = 3 := by
  sorry

end sqrt_three_div_sqrt_one_third_eq_three_l2289_228989


namespace line_through_point_parallel_to_line_l2289_228922

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem line_through_point_parallel_to_line 
  (M : Point) 
  (A B : Point) 
  (h_M : M.x = 2 ∧ M.y = -3)
  (h_A : A.x = 1 ∧ A.y = 2)
  (h_B : B.x = -1 ∧ B.y = -5) :
  ∃ (l : Line), 
    l.a = 7 ∧ l.b = -2 ∧ l.c = -20 ∧ 
    M.liesOn l ∧
    l.isParallelTo (Line.mk (B.y - A.y) (A.x - B.x) (B.x * A.y - A.x * B.y)) :=
sorry

end line_through_point_parallel_to_line_l2289_228922


namespace richard_twice_scott_age_l2289_228923

/-- The number of years until Richard is twice as old as Scott -/
def years_until_double : ℕ := 8

/-- David's current age -/
def david_age : ℕ := 14

/-- Richard's current age -/
def richard_age : ℕ := david_age + 6

/-- Scott's current age -/
def scott_age : ℕ := david_age - 8

theorem richard_twice_scott_age : 
  richard_age + years_until_double = 2 * (scott_age + years_until_double) :=
by sorry

end richard_twice_scott_age_l2289_228923


namespace trajectory_of_point_P_l2289_228907

/-- Given a circle centered at F(4,0) with radius 2, and points A(-4,0) and B on the circle,
    the point P is defined as the intersection of the perpendicular bisector of AB and line BF.
    This theorem states that the trajectory of P as B moves along the circle
    is a hyperbola with equation x^2 - y^2/15 = 1 (x ≠ 0). -/
theorem trajectory_of_point_P (A B P F : ℝ × ℝ) :
  A = (-4, 0) →
  F = (4, 0) →
  (B.1 - 4)^2 + B.2^2 = 4 →
  (∀ M : ℝ × ℝ, (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - B.1)^2 + (M.2 - B.2)^2 →
                 (M.1 - P.1) * (B.2 - F.2) = (M.2 - P.2) * (B.1 - F.1)) →
  (P.1 - F.1) * (B.2 - F.2) = (P.2 - F.2) * (B.1 - F.1) →
  P.1 ≠ 0 →
  P.1^2 - P.2^2 / 15 = 1 :=
by sorry

end trajectory_of_point_P_l2289_228907


namespace problem_statement_l2289_228906

open Real

theorem problem_statement :
  (∃ x : ℝ, x - 2 > log x) ∧
  ¬(∀ x : ℝ, exp x > 1) ∧
  ((∃ x : ℝ, x - 2 > log x) ∧ ¬(∀ x : ℝ, exp x > 1)) := by
  sorry

end problem_statement_l2289_228906


namespace unique_sum_of_squares_and_product_l2289_228950

theorem unique_sum_of_squares_and_product (a b : ℕ+) : 
  a ≤ b → 
  a.val^2 + b.val^2 + 8 * a.val * b.val = 2010 → 
  a.val + b.val = 42 :=
by sorry

end unique_sum_of_squares_and_product_l2289_228950


namespace inscribed_circle_tangent_sum_l2289_228984

/-- A point on the inscribed circle of a square -/
structure InscribedCirclePoint (α β : ℝ) where
  -- P is on the inscribed circle of square ABCD
  on_inscribed_circle : True
  -- Angle APC = α
  angle_apc : True
  -- Angle BPD = β
  angle_bpd : True

/-- The sum of squared tangents of angles α and β is 8 -/
theorem inscribed_circle_tangent_sum (α β : ℝ) (p : InscribedCirclePoint α β) : 
  Real.tan α ^ 2 + Real.tan β ^ 2 = 8 := by
  sorry

end inscribed_circle_tangent_sum_l2289_228984


namespace unique_root_of_equation_l2289_228930

open Real

theorem unique_root_of_equation :
  ∃! x : ℝ, x > 0 ∧ 1 - x - x * log x = 0 :=
by
  -- Define the function
  let f : ℝ → ℝ := λ x ↦ 1 - x - x * log x

  -- Assume f is decreasing on (0, +∞)
  have h_decreasing : ∀ x y, 0 < x → 0 < y → x < y → f y < f x := sorry

  -- Prove there exists exactly one root
  sorry

end unique_root_of_equation_l2289_228930


namespace sum_of_distances_constant_l2289_228921

/-- An equilateral triangle with side length a -/
structure EquilateralTriangle where
  a : ℝ
  a_pos : a > 0

/-- A point on one side of the equilateral triangle -/
structure PointOnSide (triangle : EquilateralTriangle) where
  x : ℝ
  y : ℝ

/-- The sum of perpendicular distances from a point on one side to the other two sides -/
def sumOfDistances (triangle : EquilateralTriangle) (point : PointOnSide triangle) : ℝ := sorry

/-- Theorem: The sum of distances from any point on one side of an equilateral triangle
    to the other two sides is constant and equal to (a√3)/2 -/
theorem sum_of_distances_constant (triangle : EquilateralTriangle) 
  (point : PointOnSide triangle) : 
  sumOfDistances triangle point = (triangle.a * Real.sqrt 3) / 2 := by sorry

end sum_of_distances_constant_l2289_228921


namespace mitchell_gum_chewing_l2289_228967

theorem mitchell_gum_chewing (packets : ℕ) (pieces_per_packet : ℕ) (unchewed_pieces : ℕ) :
  packets = 8 →
  pieces_per_packet = 7 →
  unchewed_pieces = 2 →
  packets * pieces_per_packet - unchewed_pieces = 54 :=
by sorry

end mitchell_gum_chewing_l2289_228967


namespace consecutive_odd_numbers_l2289_228965

/-- Given five consecutive odd numbers, prove that if the sum of the first and third is 146, then the fifth number is 79 -/
theorem consecutive_odd_numbers (a b c d e : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →               -- b is the next odd number after a
  c = a + 4 →               -- c is the next odd number after b
  d = a + 6 →               -- d is the next odd number after c
  e = a + 8 →               -- e is the next odd number after d
  a + c = 146 →             -- sum of a and c is 146
  e = 79 := by              -- prove that e equals 79
sorry


end consecutive_odd_numbers_l2289_228965


namespace headcount_analysis_l2289_228935

/-- Student headcount data for spring terms -/
structure HeadcountData where
  y02_03 : ℕ
  y03_04 : ℕ
  y04_05 : ℕ
  y05_06 : ℕ

/-- Calculate average headcount -/
def average_headcount (data : HeadcountData) : ℚ :=
  (data.y02_03 + data.y03_04 + data.y04_05 + data.y05_06) / 4

/-- Calculate percentage change -/
def percentage_change (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Theorem stating the average headcount and percentage change -/
theorem headcount_analysis (data : HeadcountData)
  (h1 : data.y02_03 = 10000)
  (h2 : data.y03_04 = 11000)
  (h3 : data.y04_05 = 9500)
  (h4 : data.y05_06 = 10500) :
  average_headcount data = 10125 ∧ percentage_change data.y02_03 data.y05_06 = 5 := by
  sorry

#eval average_headcount ⟨10000, 11000, 9500, 10500⟩
#eval percentage_change 10000 10500

end headcount_analysis_l2289_228935


namespace prime_triplet_l2289_228945

theorem prime_triplet (p : ℕ) : 
  Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) → p = 3 :=
by sorry

end prime_triplet_l2289_228945


namespace product_increased_by_amount_l2289_228968

theorem product_increased_by_amount (x y : ℝ) (h1 : x = 3) (h2 : 5 * x + y = 19) : y = 4 := by
  sorry

end product_increased_by_amount_l2289_228968


namespace mathematics_players_count_l2289_228970

def total_players : ℕ := 15
def physics_players : ℕ := 9
def both_subjects : ℕ := 3

theorem mathematics_players_count :
  ∃ (math_players : ℕ),
    math_players = total_players - physics_players + both_subjects ∧
    math_players = 9 :=
by sorry

end mathematics_players_count_l2289_228970


namespace largest_n_with_odd_residues_l2289_228959

theorem largest_n_with_odd_residues : ∃ (n : ℕ), n = 505 ∧ n > 10 ∧
  (∀ (k : ℕ), 2 ≤ k^2 ∧ k^2 ≤ n / 2 → n % (k^2) % 2 = 1) ∧
  (∀ (m : ℕ), m > n → ∃ (j : ℕ), 2 ≤ j^2 ∧ j^2 ≤ m / 2 ∧ m % (j^2) % 2 = 0) :=
sorry

end largest_n_with_odd_residues_l2289_228959


namespace upgrade_ways_count_l2289_228914

/-- Represents the number of levels in the game -/
def totalLevels : ℕ := 16

/-- Represents the level at which the special ability can first be upgraded -/
def firstSpecialLevel : ℕ := 6

/-- Represents the level at which the special ability can be upgraded for the second time -/
def secondSpecialLevel : ℕ := 11

/-- Represents the number of times the special ability must be upgraded -/
def specialUpgrades : ℕ := 2

/-- Represents the number of choices for upgrading regular abilities at each level -/
def regularChoices : ℕ := 3

/-- The function that calculates the number of ways to upgrade abilities -/
def upgradeWays : ℕ := 5 * (regularChoices ^ totalLevels)

/-- Theorem stating that the number of ways to upgrade abilities is 5 · 3^16 -/
theorem upgrade_ways_count : upgradeWays = 5 * (3 ^ 16) := by
  sorry

end upgrade_ways_count_l2289_228914


namespace geometric_sequence_product_l2289_228947

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a → a 5 * a 14 = 5 → a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end geometric_sequence_product_l2289_228947


namespace emily_walks_farther_l2289_228918

/-- The distance Emily walks farther than Troy over five days -/
def distance_difference (troy_distance emily_distance : ℕ) : ℕ :=
  ((emily_distance - troy_distance) * 2) * 5

/-- Theorem stating the difference in distance walked by Emily and Troy over five days -/
theorem emily_walks_farther :
  distance_difference 75 98 = 230 := by
  sorry

end emily_walks_farther_l2289_228918


namespace complex_fraction_simplification_l2289_228909

theorem complex_fraction_simplification :
  (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end complex_fraction_simplification_l2289_228909


namespace nancy_coffee_consumption_l2289_228963

/-- Represents the daily coffee consumption and costs for Nancy --/
structure CoffeeConsumption where
  double_espresso_cost : ℝ
  iced_coffee_cost : ℝ
  total_spent : ℝ
  days : ℕ

/-- Calculates the number of coffees Nancy buys each day --/
def coffees_per_day (c : CoffeeConsumption) : ℕ :=
  2

/-- Theorem stating that Nancy buys 2 coffees per day given the conditions --/
theorem nancy_coffee_consumption (c : CoffeeConsumption) 
  (h1 : c.double_espresso_cost = 3)
  (h2 : c.iced_coffee_cost = 2.5)
  (h3 : c.total_spent = 110)
  (h4 : c.days = 20) :
  coffees_per_day c = 2 := by
  sorry

#check nancy_coffee_consumption

end nancy_coffee_consumption_l2289_228963


namespace tangent_through_origin_l2289_228996

/-- The curve y = x^α + 1 has a tangent line at (1, 2) that passes through the origin if and only if α = 2 -/
theorem tangent_through_origin (α : ℝ) : 
  (∃ (m : ℝ), (∀ x : ℝ, x^α + 1 = m * (x - 1) + 2) ∧ m * (-1) + 2 = 0) ↔ α = 2 := by
  sorry

end tangent_through_origin_l2289_228996


namespace cloth_sale_loss_per_metre_l2289_228943

/-- Given the following conditions for a cloth sale:
  * 500 metres of cloth sold
  * Total selling price of Rs. 15000
  * Cost price of Rs. 40 per metre
  Prove that the loss per metre of cloth sold is Rs. 10. -/
theorem cloth_sale_loss_per_metre 
  (total_metres : ℕ) 
  (total_selling_price : ℕ) 
  (cost_price_per_metre : ℕ) 
  (h1 : total_metres = 500)
  (h2 : total_selling_price = 15000)
  (h3 : cost_price_per_metre = 40) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 10 := by
  sorry

end cloth_sale_loss_per_metre_l2289_228943


namespace restaurant_glasses_count_l2289_228960

/-- Represents the number of glasses in a restaurant with two box sizes --/
def total_glasses (small_box_count : ℕ) (large_box_count : ℕ) : ℕ :=
  12 * small_box_count + 16 * large_box_count

/-- Represents the average number of glasses per box --/
def average_glasses_per_box (small_box_count : ℕ) (large_box_count : ℕ) : ℚ :=
  (total_glasses small_box_count large_box_count : ℚ) / (small_box_count + large_box_count : ℚ)

theorem restaurant_glasses_count :
  ∃ (small_box_count large_box_count : ℕ),
    large_box_count = small_box_count + 16 ∧
    average_glasses_per_box small_box_count large_box_count = 15 ∧
    total_glasses small_box_count large_box_count = 480 :=
sorry

end restaurant_glasses_count_l2289_228960


namespace zoo_animals_l2289_228988

/-- The number of ostriches in the zoo -/
def num_ostriches : ℕ := 15

/-- The number of sika deer in the zoo -/
def num_deer : ℕ := 23

/-- The number of legs an ostrich has -/
def ostrich_legs : ℕ := 2

/-- The number of legs a sika deer has -/
def deer_legs : ℕ := 4

/-- The total number of legs of all animals -/
def total_legs : ℕ := 122

/-- The total number of legs if the numbers of ostriches and deer were swapped -/
def swapped_legs : ℕ := 106

theorem zoo_animals :
  num_ostriches * ostrich_legs + num_deer * deer_legs = total_legs ∧
  num_deer * ostrich_legs + num_ostriches * deer_legs = swapped_legs :=
by sorry

end zoo_animals_l2289_228988


namespace sum_of_xyz_l2289_228948

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end sum_of_xyz_l2289_228948


namespace arithmetic_sequence_sum_l2289_228913

theorem arithmetic_sequence_sum (a₁ d : ℝ) (h₁ : d ≠ 0) :
  let a : ℕ → ℝ := λ n => a₁ + (n - 1) * d
  let S : ℕ → ℝ := λ n => n * a₁ + n * (n - 1) / 2 * d
  (a 4)^2 = (a 3) * (a 7) ∧ S 8 = 32 → S 10 = 60 := by
  sorry

end arithmetic_sequence_sum_l2289_228913


namespace log_exists_iff_power_of_base_no_log_for_numbers_between_zero_and_one_l2289_228911

-- Define the base for logarithms
variable (a : ℝ)

-- Define the conditions for the base
variable (ha : a > 0 ∧ a ≠ 1)

-- Theorem 1: With only integer exponents, logarithms exist only for powers of the base
theorem log_exists_iff_power_of_base (b : ℝ) :
  (∃ n : ℤ, b = a^n) ↔ ∃ x : ℝ, a^x = b :=
sorry

-- Theorem 2: With only positive exponents, logarithms don't exist for numbers between 0 and 1
theorem no_log_for_numbers_between_zero_and_one (x : ℝ) (hx : 0 < x ∧ x < 1) :
  ¬∃ y : ℝ, y > 0 ∧ a^y = x :=
sorry

end log_exists_iff_power_of_base_no_log_for_numbers_between_zero_and_one_l2289_228911


namespace online_price_calculation_l2289_228980

/-- Calculates the price a buyer observes online for a product sold by a distributor through an online store, given various costs and desired profit margin. -/
theorem online_price_calculation 
  (producer_price : ℝ) 
  (shipping_cost : ℝ) 
  (commission_rate : ℝ) 
  (tax_rate : ℝ) 
  (profit_margin : ℝ) 
  (h1 : producer_price = 19) 
  (h2 : shipping_cost = 5) 
  (h3 : commission_rate = 0.2) 
  (h4 : tax_rate = 0.1) 
  (h5 : profit_margin = 0.2) : 
  ∃ (online_price : ℝ), online_price = 39.6 ∧ 
  online_price * (1 - commission_rate) = 
    (producer_price + shipping_cost) * (1 + profit_margin) * (1 + tax_rate) := by
  sorry

end online_price_calculation_l2289_228980


namespace hyperbola_minimum_value_l2289_228934

theorem hyperbola_minimum_value (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) :
  let eccentricity := (a^2 + b^2).sqrt / a
  eccentricity = 2 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b^2 + 1) / (Real.sqrt 3 * a) ≥ 4 * Real.sqrt 3 / 3 :=
by sorry

end hyperbola_minimum_value_l2289_228934


namespace fraction_equality_l2289_228979

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 4) :
  (4 * a + 3 * b) / (4 * a - 3 * b) = 4 := by
  sorry

end fraction_equality_l2289_228979


namespace unique_solution_l2289_228926

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 6

def is_valid_row (a b c d e : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ is_valid_digit e ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def sum_constraint (a b c d e : ℕ) : Prop :=
  100 * a + 10 * b + c + 10 * c + d + e = 696

theorem unique_solution (a b c d e : ℕ) :
  is_valid_row a b c d e ∧ sum_constraint a b c d e →
  a = 6 ∧ b = 2 ∧ c = 3 ∧ d = 6 ∧ e = 3 :=
sorry

end unique_solution_l2289_228926


namespace inscribed_squares_ratio_l2289_228919

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (is_right : a^2 + b^2 = c^2)
  (side_lengths : a = 5 ∧ b = 12 ∧ c = 13)

/-- Square inscribed in a right triangle with one vertex at the right angle -/
def square_at_right_angle (t : RightTriangle) (x : ℝ) : Prop :=
  0 < x ∧ x ≤ min t.a t.b

/-- Square inscribed in a right triangle with one side on the hypotenuse -/
def square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  0 < y ∧ y ≤ t.c

theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ)
  (h1 : square_at_right_angle t1 x)
  (h2 : square_on_hypotenuse t2 y) :
  x / y = 1800 / 2863 := by
  sorry

#check inscribed_squares_ratio

end inscribed_squares_ratio_l2289_228919


namespace infinitely_many_divisible_by_sum_of_digits_l2289_228971

/-- Function to create a number with n ones -/
def ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Function to calculate the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Theorem: For all natural numbers n, the number formed by 3^n ones is divisible by the sum of its digits -/
theorem infinitely_many_divisible_by_sum_of_digits (n : ℕ) :
  ∃ (k : ℕ), k > 0 ∧ (ones (3^n) % sumOfDigits (ones (3^n)) = 0) :=
sorry

end infinitely_many_divisible_by_sum_of_digits_l2289_228971


namespace absent_workers_l2289_228931

/-- Given a group of workers and their work schedule, calculate the number of absent workers. -/
theorem absent_workers 
  (total_workers : ℕ) 
  (original_days : ℕ) 
  (actual_days : ℕ) 
  (h1 : total_workers = 15)
  (h2 : original_days = 40)
  (h3 : actual_days = 60) :
  ∃ (absent : ℕ), 
    absent = 5 ∧ 
    (total_workers - absent) * actual_days = total_workers * original_days :=
by sorry


end absent_workers_l2289_228931


namespace existence_of_parallel_plane_l2289_228944

/-- Two lines in space are non-intersecting (skew) -/
def NonIntersecting (a b : Line3) : Prop := sorry

/-- A line is parallel to a plane -/
def ParallelToPlane (l : Line3) (p : Plane3) : Prop := sorry

theorem existence_of_parallel_plane (a b : Line3) (h : NonIntersecting a b) :
  ∃ α : Plane3, ParallelToPlane a α ∧ ParallelToPlane b α := by sorry

end existence_of_parallel_plane_l2289_228944


namespace shortest_distance_parabola_line_l2289_228991

/-- The shortest distance between a point on the parabola y = x^2 - 4x + 11 
    and a point on the line y = 2x - 6 is 8/√5. -/
theorem shortest_distance_parabola_line : 
  let parabola := {P : ℝ × ℝ | P.2 = P.1^2 - 4*P.1 + 11}
  let line := {Q : ℝ × ℝ | Q.2 = 2*Q.1 - 6}
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ parabola ∧ Q ∈ line ∧
    ∀ (P' : ℝ × ℝ) (Q' : ℝ × ℝ), P' ∈ parabola → Q' ∈ line →
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ 8 / Real.sqrt 5 :=
by
  sorry

end shortest_distance_parabola_line_l2289_228991


namespace probability_of_trio_l2289_228938

-- Define the original deck
def original_deck : ℕ := 52

-- Define the number of cards for each number
def cards_per_number : ℕ := 4

-- Define the number of different numbers in the deck
def different_numbers : ℕ := 13

-- Define the number of cards removed
def cards_removed : ℕ := 3

-- Define the remaining deck size
def remaining_deck : ℕ := original_deck - cards_removed

-- Define the number of ways to choose 3 cards from the remaining deck
def total_ways : ℕ := Nat.choose remaining_deck 3

-- Define the number of ways to choose a trio of the same number
def trio_ways : ℕ := (different_numbers - 2) * Nat.choose cards_per_number 3 + 1

-- Theorem statement
theorem probability_of_trio : 
  (trio_ways : ℚ) / total_ways = 45 / 18424 := by sorry

end probability_of_trio_l2289_228938


namespace chef_butter_remaining_l2289_228928

/-- Represents the recipe and chef's actions for making brownies. -/
structure BrownieRecipe where
  /-- The amount of butter (in ounces) required per cup of baking mix. -/
  butter_per_cup : ℝ
  /-- The amount of baking mix (in cups) the chef planned to use. -/
  planned_baking_mix : ℝ
  /-- The amount of coconut oil (in ounces) the chef used. -/
  coconut_oil_used : ℝ

/-- Calculates the amount of butter remaining after substituting with coconut oil. -/
def butter_remaining (recipe : BrownieRecipe) : ℝ :=
  recipe.butter_per_cup * recipe.planned_baking_mix - recipe.coconut_oil_used

/-- Theorem stating that the chef had 4 ounces of butter remaining. -/
theorem chef_butter_remaining (recipe : BrownieRecipe)
    (h1 : recipe.butter_per_cup = 2)
    (h2 : recipe.planned_baking_mix = 6)
    (h3 : recipe.coconut_oil_used = 8) :
    butter_remaining recipe = 4 := by
  sorry

#eval butter_remaining { butter_per_cup := 2, planned_baking_mix := 6, coconut_oil_used := 8 }

end chef_butter_remaining_l2289_228928


namespace circle_symmetry_implies_slope_l2289_228902

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 6*y + 14 = 0

-- Define the line equation
def line_equation (a x y : ℝ) : Prop :=
  a*x + 4*y - 6 = 0

-- Define symmetry of circle about line
def circle_symmetrical_about_line (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), circle_equation x₀ y₀ ∧ line_equation a x₀ y₀

-- Theorem statement
theorem circle_symmetry_implies_slope :
  ∀ a : ℝ, circle_symmetrical_about_line a → (a = 6) :=
sorry

end circle_symmetry_implies_slope_l2289_228902


namespace symmetry_about_x_equals_one_l2289_228933

/-- Given a function f(x) = x, if its graph is symmetric about the line x = 1,
    then the corresponding function g(x) is equal to 3x - 2. -/
theorem symmetry_about_x_equals_one (f g : ℝ → ℝ) :
  (∀ x, f x = x) →
  (∀ x, f (2 - x) = g x) →
  (∀ x, g x = 3*x - 2) := by
sorry

end symmetry_about_x_equals_one_l2289_228933


namespace largest_y_coordinate_degenerate_hyperbola_l2289_228908

theorem largest_y_coordinate_degenerate_hyperbola : 
  ∀ (x y : ℝ), x^2 / 49 - (y - 3)^2 / 25 = 0 → y ≤ 3 :=
by sorry

end largest_y_coordinate_degenerate_hyperbola_l2289_228908


namespace unique_solution_x_zero_l2289_228941

theorem unique_solution_x_zero (x y : ℝ) : 
  y = 2 * x → (3 * y^2 + y + 4 = 2 * (6 * x^2 + y + 2)) ↔ x = 0 := by
  sorry

end unique_solution_x_zero_l2289_228941


namespace binary_multiplication_theorem_l2289_228975

/-- Converts a list of binary digits to a natural number -/
def binaryToNat (digits : List Bool) : Nat :=
  digits.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, true, true]        -- 110₂
  let expected := [false, true, true, true, true, false, true]  -- 1011110₂
  binaryToNat a * binaryToNat b = binaryToNat expected := by
  sorry

end binary_multiplication_theorem_l2289_228975


namespace owen_sleep_time_l2289_228978

/-- Owen's daily schedule and sleep time calculation -/
theorem owen_sleep_time :
  let hours_in_day : ℝ := 24
  let work_hours : ℝ := 6
  let commute_hours : ℝ := 2
  let exercise_hours : ℝ := 3
  let cooking_hours : ℝ := 1
  let leisure_hours : ℝ := 3
  let grooming_hours : ℝ := 1.5
  let total_activity_hours := work_hours + commute_hours + exercise_hours + 
                              cooking_hours + leisure_hours + grooming_hours
  let sleep_hours := hours_in_day - total_activity_hours
  sleep_hours = 7.5 := by sorry

end owen_sleep_time_l2289_228978


namespace red_faces_cube_l2289_228972

theorem red_faces_cube (n : ℕ) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/4 ↔ n = 4 := by sorry

end red_faces_cube_l2289_228972


namespace sum_of_x_coordinates_is_seven_l2289_228940

-- Define the system of equations
def equation1 (x : ℝ) : ℝ := |x^2 - 8*x + 15|
def equation2 (x : ℝ) : ℝ := 8 - x

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | equation1 x = equation2 x}

-- State the theorem
theorem sum_of_x_coordinates_is_seven :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solution_set ∧ x₂ ∈ solution_set ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 7 :=
sorry

end sum_of_x_coordinates_is_seven_l2289_228940


namespace min_value_sum_reciprocals_l2289_228976

theorem min_value_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 5) :
  (1 / x + 4 / y + 9 / z) ≥ 36 / 5 :=
by sorry

end min_value_sum_reciprocals_l2289_228976


namespace even_function_implies_m_equals_two_l2289_228903

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem even_function_implies_m_equals_two (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = 2 := by
  sorry

end even_function_implies_m_equals_two_l2289_228903


namespace wall_bricks_count_l2289_228983

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 192

/-- Represents Beth's individual rate in bricks per hour -/
def beth_rate : ℚ := total_bricks / 8

/-- Represents Ben's individual rate in bricks per hour -/
def ben_rate : ℚ := total_bricks / 12

/-- Represents the reduction in combined output due to chatting, in bricks per hour -/
def chat_reduction : ℕ := 8

/-- Represents the time taken to complete the wall when working together, in hours -/
def time_together : ℕ := 6

theorem wall_bricks_count :
  (beth_rate + ben_rate - chat_reduction) * time_together = total_bricks := by
  sorry

#check wall_bricks_count

end wall_bricks_count_l2289_228983


namespace power_calculation_l2289_228917

theorem power_calculation (a : ℝ) : (-a)^2 * (-a^5)^4 / a^12 * (-2 * a^4) = -2 * a^14 := by
  sorry

end power_calculation_l2289_228917


namespace sphere_equation_implies_zero_difference_l2289_228964

theorem sphere_equation_implies_zero_difference (x y z : ℝ) :
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 →
  (x - y - z)^2002 = 0 := by
  sorry

end sphere_equation_implies_zero_difference_l2289_228964


namespace specialArrangements_eq_480_l2289_228916

/-- The number of ways to arrange six distinct objects in a row,
    where two specific objects must be on the same side of a third specific object -/
def specialArrangements : ℕ :=
  let totalPositions := 6
  let fixedObjects := 3  -- A, B, and C
  let remainingObjects := 3  -- D, E, and F
  let positionsForC := totalPositions
  let waysToArrangeAB := 2  -- A and B can be swapped
  let waysToChooseSide := 2  -- A and B can be on either side of C
  let remainingArrangements := Nat.factorial remainingObjects

  positionsForC * waysToArrangeAB * waysToChooseSide * remainingArrangements

theorem specialArrangements_eq_480 : specialArrangements = 480 := by
  sorry

end specialArrangements_eq_480_l2289_228916


namespace gcd_pow_minus_one_l2289_228954

theorem gcd_pow_minus_one (a b : ℕ) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := by
  sorry

end gcd_pow_minus_one_l2289_228954


namespace sphere_volume_circumscribing_rectangular_solid_l2289_228966

theorem sphere_volume_circumscribing_rectangular_solid :
  let length : ℝ := 2
  let width : ℝ := 1
  let height : ℝ := 2
  let space_diagonal : ℝ := Real.sqrt (length^2 + width^2 + height^2)
  let sphere_radius : ℝ := space_diagonal / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = (9 / 2) * Real.pi :=
by sorry

end sphere_volume_circumscribing_rectangular_solid_l2289_228966


namespace abs_sum_inequality_l2289_228901

theorem abs_sum_inequality (x : ℝ) : 
  |x - 3| + |x + 4| < 10 ↔ x ∈ Set.Ioo (-5.5) 4.5 := by sorry

end abs_sum_inequality_l2289_228901


namespace always_returns_to_present_max_stations_visited_l2289_228997

/-- Represents the time machine's movement on a circular track of 2009 stations. -/
def TimeMachine :=
  { s : ℕ // s ≤ 2009 }

/-- Moves the time machine to the next station according to the rules. -/
def nextStation (s : TimeMachine) : TimeMachine :=
  sorry

/-- Checks if a number is a power of 2. -/
def isPowerOfTwo (n : ℕ) : Bool :=
  sorry

/-- Returns the sequence of stations visited by the time machine starting from a given station. -/
def stationSequence (start : TimeMachine) : List TimeMachine :=
  sorry

/-- Theorem stating that the time machine always returns to station 1. -/
theorem always_returns_to_present (start : TimeMachine) :
  1 ∈ (stationSequence start).map (fun s => s.val) := by
  sorry

/-- Theorem stating the maximum number of stations the time machine can stop at. -/
theorem max_stations_visited :
  ∃ (start : TimeMachine), (stationSequence start).length = 812 ∧
  ∀ (s : TimeMachine), (stationSequence s).length ≤ 812 := by
  sorry

end always_returns_to_present_max_stations_visited_l2289_228997


namespace diamond_two_five_l2289_228962

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a + 3 * b ^ 2 + b

-- Theorem statement
theorem diamond_two_five : diamond 2 5 = 82 := by
  sorry

end diamond_two_five_l2289_228962


namespace bread_cost_l2289_228987

/-- Prove that the cost of the bread is $1.25 given the conditions --/
theorem bread_cost (total_cost change_nickels : ℚ) 
  (h1 : total_cost = 205/100)  -- Total cost is $2.05
  (h2 : change_nickels = 8 * 5/100)  -- 8 nickels in change
  (h3 : ∃ (change_quarter change_dime : ℚ), 
    change_quarter = 25/100 ∧ 
    change_dime = 10/100 ∧ 
    700/100 - total_cost = change_quarter + change_dime + change_nickels + 420/100) 
  : ∃ (bread_cost cheese_cost : ℚ), 
    bread_cost = 125/100 ∧ 
    cheese_cost = 80/100 ∧ 
    bread_cost + cheese_cost = total_cost := by
  sorry

end bread_cost_l2289_228987


namespace alyssa_book_count_l2289_228904

/-- The number of books Alyssa has -/
def alyssas_books : ℕ := 36

/-- The number of books Nancy has -/
def nancys_books : ℕ := 252

theorem alyssa_book_count :
  (nancys_books = 7 * alyssas_books) → alyssas_books = 36 := by
  sorry

end alyssa_book_count_l2289_228904


namespace greatest_integer_no_real_roots_l2289_228929

theorem greatest_integer_no_real_roots (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 15 ≠ 0) ↔ c ≤ 7 :=
by sorry

end greatest_integer_no_real_roots_l2289_228929


namespace product_and_sum_of_consecutive_integers_l2289_228995

theorem product_and_sum_of_consecutive_integers : 
  ∃ (a b c d e : ℤ), 
    (b = a + 1) ∧ 
    (d = c + 1) ∧ 
    (e = d + 1) ∧ 
    (a > 0) ∧ 
    (a * b = 198) ∧ 
    (c * d * e = 198) ∧ 
    (a + b + c + d + e = 39) := by
  sorry

end product_and_sum_of_consecutive_integers_l2289_228995


namespace coes_speed_l2289_228936

theorem coes_speed (teena_speed : ℝ) (initial_distance : ℝ) (time : ℝ) (final_distance : ℝ) :
  teena_speed = 55 →
  initial_distance = 7.5 →
  time = 1.5 →
  final_distance = 15 →
  ∃ coe_speed : ℝ,
    coe_speed = 50 ∧
    teena_speed * time - coe_speed * time = final_distance + initial_distance :=
by
  sorry

end coes_speed_l2289_228936


namespace largest_inscribed_triangle_area_l2289_228937

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_triangle_area := (1/2) * diameter * r
  max_triangle_area = 64 := by sorry

end largest_inscribed_triangle_area_l2289_228937


namespace vector_equality_l2289_228925

theorem vector_equality (m : ℝ) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, 2)
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = (a.1^2 + a.2^2) + (b.1^2 + b.2^2) → m = -2 :=
by sorry

end vector_equality_l2289_228925


namespace right_triangle_hypotenuse_l2289_228900

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 15 ∧ b = 36 ∧ a^2 + b^2 = h^2 → h = 39 := by
  sorry

end right_triangle_hypotenuse_l2289_228900


namespace divisibility_by_three_l2289_228998

theorem divisibility_by_three (u v : ℤ) (h : (9 : ℤ) ∣ (u^2 + u*v + v^2)) : (3 : ℤ) ∣ u ∧ (3 : ℤ) ∣ v := by
  sorry

end divisibility_by_three_l2289_228998


namespace least_number_with_remainder_forty_is_least_forty_has_remainder_four_least_number_is_forty_l2289_228949

theorem least_number_with_remainder (n : ℕ) : n ≥ 40 → n % 6 = 4 → ∃ k : ℕ, n = 6 * k + 4 :=
sorry

theorem forty_is_least : ∀ n : ℕ, n < 40 → n % 6 ≠ 4 :=
sorry

theorem forty_has_remainder_four : 40 % 6 = 4 :=
sorry

theorem least_number_is_forty : 
  (∃ n : ℕ, n % 6 = 4) ∧ 
  (∀ n : ℕ, n % 6 = 4 → n ≥ 40) ∧
  (40 % 6 = 4) :=
sorry

end least_number_with_remainder_forty_is_least_forty_has_remainder_four_least_number_is_forty_l2289_228949


namespace sum_smallest_largest_consecutive_odds_l2289_228982

/-- Given an even number of consecutive odd integers with arithmetic mean y + 1,
    the sum of the smallest and largest integers is 2y. -/
theorem sum_smallest_largest_consecutive_odds (y : ℝ) (n : ℕ) (h : n > 0) :
  let a := y - 2 * n + 2
  let sequence := fun i => a + 2 * i
  let mean := (sequence 0 + sequence (2 * n - 1)) / 2
  (mean = y + 1) → (sequence 0 + sequence (2 * n - 1) = 2 * y) :=
by sorry

end sum_smallest_largest_consecutive_odds_l2289_228982


namespace symmetric_complex_division_l2289_228977

/-- Two complex numbers are symmetric about y = x if their real and imaginary parts are swapped -/
def symmetric_about_y_eq_x (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

theorem symmetric_complex_division (z₁ z₂ : ℂ) : 
  symmetric_about_y_eq_x z₁ z₂ → z₁ = 1 + 2*I → z₁ / z₂ = 4/5 + 3/5*I := by
  sorry

end symmetric_complex_division_l2289_228977


namespace sum_positive_implies_both_positive_is_false_l2289_228969

theorem sum_positive_implies_both_positive_is_false : 
  ¬(∀ a b : ℝ, a + b > 0 → a > 0 ∧ b > 0) := by
  sorry

end sum_positive_implies_both_positive_is_false_l2289_228969


namespace prime_characterization_l2289_228955

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, 2 ≤ k → k < n → ¬(k ∣ n)

theorem prime_characterization (n : ℕ) :
  Nat.Prime n ↔ is_prime n := by
  sorry

end prime_characterization_l2289_228955


namespace remainder_2345_times_1976_mod_300_l2289_228961

theorem remainder_2345_times_1976_mod_300 : (2345 * 1976) % 300 = 220 := by
  sorry

end remainder_2345_times_1976_mod_300_l2289_228961


namespace unique_linear_m_l2289_228942

/-- A function to represent the linearity of an equation -/
def is_linear (m : ℝ) : Prop :=
  (abs m = 1) ∧ (m + 1 ≠ 0)

/-- Theorem stating that m = 1 is the only value satisfying the linearity condition -/
theorem unique_linear_m : ∃! m : ℝ, is_linear m :=
  sorry

end unique_linear_m_l2289_228942


namespace mans_speed_in_still_water_l2289_228924

/-- The speed of a man rowing a boat in still water, given downstream conditions. -/
theorem mans_speed_in_still_water (current_speed : ℝ) (distance : ℝ) (time : ℝ) :
  current_speed = 8 →
  distance = 40 →
  time = 4.499640028797696 →
  ∃ (speed_still_water : ℝ), 
    abs (speed_still_water - ((distance / time) - (current_speed * 1000 / 3600))) < 0.001 :=
by sorry

end mans_speed_in_still_water_l2289_228924


namespace foreign_trade_income_equation_l2289_228999

/-- The foreign trade income equation over two years with a constant growth rate -/
theorem foreign_trade_income_equation
  (m : ℝ) -- foreign trade income in 2001 (billion yuan)
  (x : ℝ) -- annual growth rate
  (n : ℝ) -- foreign trade income in 2003 (billion yuan)
  : m * (1 + x)^2 = n :=
by sorry

end foreign_trade_income_equation_l2289_228999


namespace commercial_viewers_l2289_228932

/-- Calculates the number of commercial viewers given revenue data -/
theorem commercial_viewers (revenue_per_view : ℚ) (revenue_per_sub : ℚ) 
  (num_subs : ℕ) (total_revenue : ℚ) : 
  revenue_per_view > 0 → 
  (total_revenue - revenue_per_sub * num_subs) / revenue_per_view = 100 → 
  ∃ (num_viewers : ℕ), num_viewers = 100 :=
by
  sorry

#check commercial_viewers (1/2) 1 27 77

end commercial_viewers_l2289_228932
