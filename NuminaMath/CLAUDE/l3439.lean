import Mathlib

namespace emily_dresses_l3439_343973

theorem emily_dresses (melissa : ℕ) (debora : ℕ) (emily : ℕ) : 
  debora = melissa + 12 →
  melissa = emily / 2 →
  melissa + debora + emily = 44 →
  emily = 16 := by sorry

end emily_dresses_l3439_343973


namespace distance_from_two_equals_three_l3439_343968

theorem distance_from_two_equals_three (x : ℝ) : 
  |x - 2| = 3 ↔ x = 5 ∨ x = -1 := by sorry

end distance_from_two_equals_three_l3439_343968


namespace complex_power_sum_l3439_343975

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^100 + 1/z^100 = -2 * Real.cos (40 * π / 180) := by
  sorry

end complex_power_sum_l3439_343975


namespace two_digit_number_property_l3439_343950

theorem two_digit_number_property (n : ℕ) : 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 + n % 10 = 3) →
  (n / 2 : ℚ) - (n / 4 : ℚ) = 5.25 := by
  sorry

end two_digit_number_property_l3439_343950


namespace chord_dot_product_l3439_343988

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a chord passing through the focus
def chord_through_focus (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = (1 - t, -2*t) ∧ B = (1 + t, 2*t)

-- Theorem statement
theorem chord_dot_product (A B : ℝ × ℝ) :
  parabola A.1 A.2 → parabola B.1 B.2 → chord_through_focus A B →
  A.1 * B.1 + A.2 * B.2 = -3 :=
by sorry

end chord_dot_product_l3439_343988


namespace biscuit_dimensions_l3439_343938

theorem biscuit_dimensions (sheet_side : ℝ) (num_biscuits : ℕ) (biscuit_side : ℝ) : 
  sheet_side = 12 →
  num_biscuits = 16 →
  (sheet_side * sheet_side) = (biscuit_side * biscuit_side * num_biscuits) →
  biscuit_side = 3 := by
sorry

end biscuit_dimensions_l3439_343938


namespace infinitely_many_special_even_numbers_l3439_343917

theorem infinitely_many_special_even_numbers :
  ∃ (n : ℕ → ℕ), 
    (∀ k, Even (n k)) ∧ 
    (∀ k, n k < n (k + 1)) ∧
    (∀ k, (n k) ∣ (2^(n k) + 2)) ∧
    (∀ k, (n k - 1) ∣ (2^(n k) + 1)) :=
sorry

end infinitely_many_special_even_numbers_l3439_343917


namespace power_76_mod_7_l3439_343916

theorem power_76_mod_7 (n : ℕ) (h : Odd n) : 76^n % 7 = 6 := by
  sorry

end power_76_mod_7_l3439_343916


namespace find_other_number_l3439_343971

theorem find_other_number (a b : ℕ) (ha : a = 36) 
  (hhcf : Nat.gcd a b = 20) (hlcm : Nat.lcm a b = 396) : b = 220 := by
  sorry

end find_other_number_l3439_343971


namespace train_length_l3439_343983

/-- The length of a train given its crossing time, bridge length, and speed -/
theorem train_length (crossing_time : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) :
  crossing_time = 26.997840172786177 →
  bridge_length = 170 →
  train_speed_kmph = 36 →
  ∃ (train_length : ℝ), abs (train_length - 99.978) < 0.001 := by
  sorry

end train_length_l3439_343983


namespace parallel_vectors_k_value_l3439_343921

/-- Given vectors a, b, and c in ℝ², prove that if (a + k * c) is parallel to (2 * b - a), then k = -16/13 -/
theorem parallel_vectors_k_value (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (3, 2)) 
    (hb : b = (-1, 2)) 
    (hc : c = (4, 1)) 
    (h_parallel : ∃ (t : ℝ), t • (a.1 + k * c.1, a.2 + k * c.2) = (2 * b.1 - a.1, 2 * b.2 - a.2)) :
  k = -16/13 := by
  sorry

end parallel_vectors_k_value_l3439_343921


namespace regular_polygon_sides_l3439_343996

theorem regular_polygon_sides (n : ℕ) (h : n > 0) : 
  (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end regular_polygon_sides_l3439_343996


namespace buttons_needed_for_shirts_l3439_343997

theorem buttons_needed_for_shirts 
  (shirts_per_kid : ℕ)
  (num_kids : ℕ)
  (buttons_per_shirt : ℕ)
  (h1 : shirts_per_kid = 3)
  (h2 : num_kids = 3)
  (h3 : buttons_per_shirt = 7) :
  shirts_per_kid * num_kids * buttons_per_shirt = 63 :=
by sorry

end buttons_needed_for_shirts_l3439_343997


namespace rectangle_dimensions_l3439_343993

/-- Proves that a rectangle with perimeter 34 and length 5 more than width has width 6 and length 11 -/
theorem rectangle_dimensions :
  ∀ (w l : ℕ), 
    (2 * w + 2 * l = 34) →  -- Perimeter is 34
    (l = w + 5) →           -- Length is 5 more than width
    (w = 6 ∧ l = 11) :=     -- Width is 6 and length is 11
by
  sorry

#check rectangle_dimensions

end rectangle_dimensions_l3439_343993


namespace unique_two_digit_reverse_pair_l3439_343926

theorem unique_two_digit_reverse_pair (z : ℕ) (h : z ≥ 3) :
  ∃! (A B : ℕ),
    (A < z^2 ∧ A ≥ z) ∧
    (B < z^2 ∧ B ≥ z) ∧
    (∃ (p q : ℕ), A = p * z + q ∧ B = q * z + p) ∧
    (∀ x : ℝ, (x^2 - A*x + B = 0) → (∃! r : ℝ, x = r)) ∧
    A = (z - 1)^2 ∧
    B = 2*(z - 1) :=
by
  sorry

end unique_two_digit_reverse_pair_l3439_343926


namespace absolute_value_equation_solution_difference_l3439_343976

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (|2 * x₁ - 3| = 15) ∧ 
  (|2 * x₂ - 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 15) := by
  sorry

end absolute_value_equation_solution_difference_l3439_343976


namespace triangle_is_equilateral_l3439_343991

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a^3 + t.b^3 + t.c^3) / (t.a + t.b + t.c) = t.c^2

def condition2 (t : Triangle) : Prop :=
  Real.sin t.α * Real.sin t.β = (Real.sin t.γ)^2

-- Define the theorem
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) 
  (h2 : condition2 t) : 
  t.a = t.b ∧ t.b = t.c := by
  sorry


end triangle_is_equilateral_l3439_343991


namespace arithmetic_sequence_problem_l3439_343992

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_sum : a 2 + a 3 + a 4 = 15)
  (h_geom : is_geometric_sequence (λ n => 
    match n with
    | 1 => a 1 + 2
    | 2 => a 3 + 4
    | 3 => a 6 + 16
    | _ => 0
  )) :
  a 10 = 19 := by
sorry

end arithmetic_sequence_problem_l3439_343992


namespace wall_length_is_800_l3439_343969

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.width * w.height

theorem wall_length_is_800 (brick : BrickDimensions) (wall : WallDimensions) 
    (h1 : brick.length = 25)
    (h2 : brick.width = 11.25)
    (h3 : brick.height = 6)
    (h4 : wall.width = 600)
    (h5 : wall.height = 22.5)
    (h6 : brickVolume brick * 6400 = wallVolume wall) :
    wall.length = 800 := by
  sorry

#check wall_length_is_800

end wall_length_is_800_l3439_343969


namespace exactly_two_triples_l3439_343900

/-- Least common multiple of two positive integers -/
def lcm (r s : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given conditions -/
def count_triples : ℕ := sorry

/-- Theorem stating that there are exactly 2 ordered triples satisfying the conditions -/
theorem exactly_two_triples : 
  count_triples = 2 ∧ 
  ∀ a b c : ℕ+, 
    (lcm a b = 1250 ∧ lcm b c = 2500 ∧ lcm c a = 2500) → 
    (a, b, c) ∈ {x | count_triples > 0} :=
sorry

end exactly_two_triples_l3439_343900


namespace parabola_properties_l3439_343999

-- Define the function f(x) = -x^2
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem parabola_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f y < f x) :=
by sorry

end parabola_properties_l3439_343999


namespace triangle_side_length_l3439_343915

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 2 →
  b = Real.sqrt 3 - 1 →
  C = π / 6 →
  c^2 = 5 - Real.sqrt 3 := by
sorry

end triangle_side_length_l3439_343915


namespace pet_food_discount_l3439_343914

theorem pet_food_discount (msrp : ℝ) (regular_discount : ℝ) (final_price : ℝ) (additional_discount : ℝ) : 
  msrp = 40 →
  regular_discount = 0.3 →
  final_price = 22.4 →
  additional_discount = (msrp * (1 - regular_discount) - final_price) / (msrp * (1 - regular_discount)) →
  additional_discount = 0.2 := by
sorry

end pet_food_discount_l3439_343914


namespace total_balloons_l3439_343980

def fred_balloons : ℕ := 5
def sam_balloons : ℕ := 6
def mary_balloons : ℕ := 7

theorem total_balloons : fred_balloons + sam_balloons + mary_balloons = 18 := by
  sorry

end total_balloons_l3439_343980


namespace carol_blocks_l3439_343932

/-- Given that Carol starts with 42 blocks and loses 25 blocks, 
    prove that she ends with 17 blocks. -/
theorem carol_blocks : 
  let initial_blocks : ℕ := 42
  let lost_blocks : ℕ := 25
  initial_blocks - lost_blocks = 17 := by
  sorry

end carol_blocks_l3439_343932


namespace no_integer_points_between_l3439_343994

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- The line passing through points A(2, 3) and B(50, 500) -/
def line (p : IntPoint) : Prop :=
  (p.y - 3) * 48 = 497 * (p.x - 2)

/-- A point is strictly between A and B if its x-coordinate is between 2 and 50 exclusively -/
def strictly_between (p : IntPoint) : Prop :=
  2 < p.x ∧ p.x < 50

theorem no_integer_points_between : 
  ¬ ∃ p : IntPoint, line p ∧ strictly_between p :=
sorry

end no_integer_points_between_l3439_343994


namespace triangle_properties_l3439_343989

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove properties about angle A and area. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b^2 + c^2 - a^2 + b*c = 0 →
  Real.sin C = Real.sqrt 2 / 2 →
  a = Real.sqrt 3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  A = 2 * Real.pi / 3 ∧
  (1/2 * a * c * Real.sin B) = (3 - Real.sqrt 3) / 4 :=
by sorry

end triangle_properties_l3439_343989


namespace age_of_twentieth_student_l3439_343957

theorem age_of_twentieth_student (total_students : Nat) (total_avg_age : Nat)
  (group1_count : Nat) (group1_avg_age : Nat)
  (group2_count : Nat) (group2_avg_age : Nat)
  (group3_count : Nat) (group3_avg_age : Nat) :
  total_students = 20 →
  total_avg_age = 18 →
  group1_count = 6 →
  group1_avg_age = 16 →
  group2_count = 8 →
  group2_avg_age = 17 →
  group3_count = 5 →
  group3_avg_age = 21 →
  (total_students * total_avg_age) - 
  (group1_count * group1_avg_age + group2_count * group2_avg_age + group3_count * group3_avg_age) = 23 := by
  sorry

end age_of_twentieth_student_l3439_343957


namespace quadratic_function_ratio_l3439_343931

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  max_at_neg_two : ∀ x : ℝ, a * x^2 + b * x + c ≤ a^2
  max_value : a * (-2)^2 + b * (-2) + c = a^2
  passes_through_point : a * (-1)^2 + b * (-1) + c = 6

/-- Theorem stating that (a + c) / b = 1/2 for the given quadratic function -/
theorem quadratic_function_ratio (f : QuadraticFunction) : (f.a + f.c) / f.b = 1/2 := by
  sorry

end quadratic_function_ratio_l3439_343931


namespace election_total_votes_l3439_343963

/-- Represents the total number of votes in the election -/
def total_votes : ℕ := sorry

/-- Represents the vote percentage for Candidate A -/
def candidate_a_percentage : ℚ := 30 / 100

/-- Represents the vote percentage for Candidate B -/
def candidate_b_percentage : ℚ := 25 / 100

/-- Represents the vote difference between Candidate A and Candidate B -/
def vote_difference_a_b : ℕ := 1800

theorem election_total_votes : 
  (candidate_a_percentage - candidate_b_percentage) * total_votes = vote_difference_a_b ∧ 
  total_votes = 36000 := by sorry

end election_total_votes_l3439_343963


namespace legs_product_ge_parallel_sides_product_l3439_343970

/-- A trapezoid with perpendicular diagonals -/
structure PerpDiagonalTrapezoid where
  -- Parallel sides
  a : ℝ
  c : ℝ
  -- Legs
  b : ℝ
  d : ℝ
  -- All sides are positive
  a_pos : 0 < a
  c_pos : 0 < c
  b_pos : 0 < b
  d_pos : 0 < d
  -- Diagonals are perpendicular (using the property from the solution)
  perp_diag : b^2 + d^2 = a^2 + c^2

/-- 
  The product of the legs is at least as large as 
  the product of the parallel sides in a trapezoid 
  with perpendicular diagonals
-/
theorem legs_product_ge_parallel_sides_product (t : PerpDiagonalTrapezoid) : 
  t.b * t.d ≥ t.a * t.c := by
  sorry

end legs_product_ge_parallel_sides_product_l3439_343970


namespace abs_f_decreasing_on_4_6_l3439_343952

-- Define the properties of the function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem abs_f_decreasing_on_4_6 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_sym : is_symmetric_about f 2)
  (h_inc : is_increasing_on f (-2) 0)
  (h_nonneg : f (-2) ≥ 0) :
  ∀ x₁ x₂, 4 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 6 → |f x₁| > |f x₂| :=
sorry

end abs_f_decreasing_on_4_6_l3439_343952


namespace paperback_count_l3439_343918

theorem paperback_count (total_books hardbacks selections : ℕ) : 
  total_books = 6 → 
  hardbacks = 4 → 
  selections = 14 →
  (∃ paperbacks : ℕ, 
    paperbacks + hardbacks = total_books ∧
    paperbacks = 2 ↔ 
    (Nat.choose paperbacks 1 * Nat.choose hardbacks 3 +
     Nat.choose paperbacks 2 * Nat.choose hardbacks 2 = selections)) :=
by sorry

end paperback_count_l3439_343918


namespace remainder_of_sum_of_powers_l3439_343979

theorem remainder_of_sum_of_powers (n : ℕ) : (9^24 + 12^37) % 23 = 4 := by
  sorry

end remainder_of_sum_of_powers_l3439_343979


namespace triangle_constructibility_l3439_343904

/-- Given two sides of a triangle and the median to the third side,
    this theorem proves the condition for the triangle's constructibility. -/
theorem triangle_constructibility 
  (a b s : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hs : s > 0) :
  ((a - b) / 2 < s ∧ s < (a + b) / 2) ↔ 
  ∃ (c : ℝ), c > 0 ∧ 
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
    s^2 = (2 * (a^2 + b^2) - c^2) / 4 :=
by sorry


end triangle_constructibility_l3439_343904


namespace weight_of_b_l3439_343905

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) : 
  b = 31 := by
  sorry

end weight_of_b_l3439_343905


namespace cubic_function_increasing_l3439_343911

theorem cubic_function_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁^3 < x₂^3 := by
  sorry

end cubic_function_increasing_l3439_343911


namespace tangent_relations_l3439_343902

theorem tangent_relations (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan α = -4/3 ∧
  Real.tan (α + π/4) = -1/7 ∧
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7/6 := by
  sorry

end tangent_relations_l3439_343902


namespace least_b_with_conditions_l3439_343923

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The least positive integer with a given number of factors -/
def least_with_factors (k : ℕ) : ℕ+ := sorry

theorem least_b_with_conditions (a b : ℕ+) 
  (ha : num_factors a = 4)
  (hb : num_factors b = 2 * (num_factors a))
  (hdiv : b.val % a.val = 0) :
  b ≥ 60 ∧ ∃ (a₀ b₀ : ℕ+), 
    num_factors a₀ = 4 ∧ 
    num_factors b₀ = 2 * (num_factors a₀) ∧ 
    b₀.val % a₀.val = 0 ∧ 
    b₀ = 60 := by sorry

end least_b_with_conditions_l3439_343923


namespace square_root_problem_l3439_343909

theorem square_root_problem (c d : ℕ) (h : 241 * c + 214 = d^2) : d = 334 := by
  sorry

end square_root_problem_l3439_343909


namespace proposition_truth_values_l3439_343981

open Real

theorem proposition_truth_values :
  ∃ (p q : Prop),
  (∀ x, 0 < x → x < π / 2 → (p ↔ sin x > x)) ∧
  (∀ x, 0 < x → x < π / 2 → (q ↔ tan x > x)) ∧
  (¬(p ∧ q)) ∧
  (p ∨ q) ∧
  (¬(p ∨ ¬q)) ∧
  ((¬p) ∨ q) := by
  sorry

end proposition_truth_values_l3439_343981


namespace second_machine_copies_per_minute_l3439_343934

/-- 
Given two copy machines working at constant rates, where the first machine makes 35 copies per minute,
and together they make 3300 copies in 30 minutes, prove that the second machine makes 75 copies per minute.
-/
theorem second_machine_copies_per_minute 
  (rate1 : ℕ) 
  (rate2 : ℕ) 
  (total_time : ℕ) 
  (total_copies : ℕ) 
  (h1 : rate1 = 35)
  (h2 : total_time = 30)
  (h3 : total_copies = 3300)
  (h4 : rate1 * total_time + rate2 * total_time = total_copies) : 
  rate2 = 75 := by
  sorry

end second_machine_copies_per_minute_l3439_343934


namespace triangle_radius_ratio_l3439_343951

/-- Given a triangle with area S, circumradius R, and inradius r, 
    such that S^2 = 2R^2 + 8Rr + 3r^2, prove that R/r = 2 or R/r ≥ √2 + 1 -/
theorem triangle_radius_ratio (S R r : ℝ) (h : S^2 = 2*R^2 + 8*R*r + 3*r^2) :
  R / r = 2 ∨ R / r ≥ Real.sqrt 2 + 1 := by
  sorry

end triangle_radius_ratio_l3439_343951


namespace z_greater_than_w_by_50_percent_l3439_343998

theorem z_greater_than_w_by_50_percent 
  (w x y z : ℝ) 
  (hw : w = 0.6 * x) 
  (hx : x = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w = 0.5 := by
sorry

end z_greater_than_w_by_50_percent_l3439_343998


namespace triangle_properties_l3439_343959

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (∀ a' b' : ℝ, a' + b' + c ≤ 6) ∧
  (2 * Real.sin (2 * A) + Real.sin (2 * B + C) = Real.sin C →
   1/2 * a * b * Real.sin C = 2 * Real.sqrt 6 / 3) :=
by sorry

end triangle_properties_l3439_343959


namespace tshirt_sale_revenue_l3439_343954

/-- Calculates the money made per minute during a t-shirt sale -/
def money_per_minute (total_shirts : ℕ) (sale_duration : ℕ) (black_price white_price : ℚ) : ℚ :=
  let black_shirts := total_shirts / 2
  let white_shirts := total_shirts / 2
  let total_revenue := (black_shirts : ℚ) * black_price + (white_shirts : ℚ) * white_price
  total_revenue / (sale_duration : ℚ)

/-- Proves that the money made per minute during the specific t-shirt sale is $220 -/
theorem tshirt_sale_revenue : money_per_minute 200 25 30 25 = 220 := by
  sorry

end tshirt_sale_revenue_l3439_343954


namespace square_position_2010_l3439_343995

/-- Represents the positions of the square's vertices -/
inductive SquarePosition
| ABCD
| CABD
| DACB
| BCAD
| ADCB
| CBDA
| BADC
| CDAB

/-- Applies the transformation sequence to a given position -/
def applyTransformation (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.CABD
  | SquarePosition.CABD => SquarePosition.DACB
  | SquarePosition.DACB => SquarePosition.BCAD
  | SquarePosition.BCAD => SquarePosition.ADCB
  | SquarePosition.ADCB => SquarePosition.CBDA
  | SquarePosition.CBDA => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.ABCD

/-- Returns the position after n transformations -/
def nthPosition (n : Nat) : SquarePosition :=
  match n % 8 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.CABD
  | 2 => SquarePosition.DACB
  | 3 => SquarePosition.BCAD
  | 4 => SquarePosition.ADCB
  | 5 => SquarePosition.CBDA
  | 6 => SquarePosition.BADC
  | 7 => SquarePosition.CDAB
  | _ => SquarePosition.ABCD  -- This case should never occur due to % 8

theorem square_position_2010 :
  nthPosition 2010 = SquarePosition.CABD := by
  sorry

end square_position_2010_l3439_343995


namespace not_p_sufficient_not_necessary_for_q_l3439_343947

-- Define the propositions p and q
def p (x : ℝ) : Prop := -1 < x ∧ x < 3
def q (x : ℝ) : Prop := x > 5

-- Define the relationship between ¬p and q
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x, ¬(p x) → q x) ∧ ¬(∀ x, q x → ¬(p x)) :=
sorry

end not_p_sufficient_not_necessary_for_q_l3439_343947


namespace cube_sum_divisibility_l3439_343965

theorem cube_sum_divisibility (a b c : ℤ) 
  (h1 : 6 ∣ (a^2 + b^2 + c^2)) 
  (h2 : 3 ∣ (a*b + b*c + c*a)) : 
  6 ∣ (a^3 + b^3 + c^3) :=
by sorry

end cube_sum_divisibility_l3439_343965


namespace arithmetic_sequence_general_term_l3439_343936

/-- Given an arithmetic sequence {aₙ} where the sum of the first n terms
    is Sₙ = 3n² + 2n, prove that the general term aₙ = 6n - 1 for all
    positive integers n. -/
theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h_sum : ∀ n : ℕ, S n = 3 * n^2 + 2 * n)  -- Given condition
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence property
  : ∀ n : ℕ, n > 0 → a n = 6 * n - 1 :=
by sorry

end arithmetic_sequence_general_term_l3439_343936


namespace opposite_number_l3439_343929

theorem opposite_number (a : ℤ) : (∀ b : ℤ, a + b = 0 → b = -2022) → a = 2022 := by
  sorry

end opposite_number_l3439_343929


namespace mary_dog_walking_earnings_l3439_343961

/-- 
Given:
- Mary earns $20 washing cars each month
- Mary earns D dollars walking dogs each month
- Mary saves half of her total earnings each month
- It takes Mary 5 months to save $150

Prove that D = $40
-/
theorem mary_dog_walking_earnings (D : ℝ) : 
  (5 : ℝ) * ((20 + D) / 2) = 150 → D = 40 := by sorry

end mary_dog_walking_earnings_l3439_343961


namespace equation_to_general_form_l3439_343945

theorem equation_to_general_form :
  ∀ x : ℝ, (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by
  sorry

end equation_to_general_form_l3439_343945


namespace correct_proof_by_contradiction_components_l3439_343958

/-- Represents the components used in a proof by contradiction --/
inductive ProofByContradictionComponent
  | assumption
  | originalConditions
  | axiomTheoremsDefinitions
  | originalConclusion

/-- Defines the set of components used in a proof by contradiction --/
def proofByContradictionComponents : Set ProofByContradictionComponent :=
  {ProofByContradictionComponent.assumption,
   ProofByContradictionComponent.originalConditions,
   ProofByContradictionComponent.axiomTheoremsDefinitions}

/-- Theorem stating the correct components used in a proof by contradiction --/
theorem correct_proof_by_contradiction_components :
  proofByContradictionComponents =
    {ProofByContradictionComponent.assumption,
     ProofByContradictionComponent.originalConditions,
     ProofByContradictionComponent.axiomTheoremsDefinitions} :=
by
  sorry


end correct_proof_by_contradiction_components_l3439_343958


namespace total_lemons_picked_l3439_343910

theorem total_lemons_picked (sally_lemons mary_lemons : ℕ) 
  (h1 : sally_lemons = 7)
  (h2 : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 := by
sorry

end total_lemons_picked_l3439_343910


namespace fifi_closet_total_hangers_l3439_343985

/-- The number of colored hangers in Fifi's closet -/
def total_hangers (pink green blue yellow : ℕ) : ℕ := pink + green + blue + yellow

/-- The conditions of Fifi's closet hangers -/
def fifi_closet_conditions (pink green blue yellow : ℕ) : Prop :=
  pink = 7 ∧ green = 4 ∧ blue = green - 1 ∧ yellow = blue - 1

/-- Theorem: The total number of colored hangers in Fifi's closet is 16 -/
theorem fifi_closet_total_hangers :
  ∃ (pink green blue yellow : ℕ),
    fifi_closet_conditions pink green blue yellow ∧
    total_hangers pink green blue yellow = 16 := by
  sorry

end fifi_closet_total_hangers_l3439_343985


namespace half_angle_in_second_quadrant_l3439_343966

-- Define the property of being in the fourth quadrant
def is_fourth_quadrant (θ : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + 3 * Real.pi / 2 ≤ θ ∧ θ ≤ 2 * k * Real.pi + 2 * Real.pi

-- Define the property of being in the second quadrant
def is_second_quadrant (θ : Real) : Prop :=
  ∃ k : Int, k * Real.pi + Real.pi / 2 ≤ θ ∧ θ ≤ k * Real.pi + Real.pi

-- State the theorem
theorem half_angle_in_second_quadrant (θ : Real) 
  (h1 : is_fourth_quadrant θ) 
  (h2 : |Real.cos (θ/2)| = -Real.cos (θ/2)) : 
  is_second_quadrant (θ/2) :=
sorry

end half_angle_in_second_quadrant_l3439_343966


namespace middle_trapezoid_radius_l3439_343944

/-- Given a trapezoid divided into three similar trapezoids by lines parallel to the bases,
    each with an inscribed circle, this theorem proves that the radius of the middle circle
    is the geometric mean of the radii of the other two circles. -/
theorem middle_trapezoid_radius (R r x : ℝ) 
  (h_positive : R > 0 ∧ r > 0 ∧ x > 0) 
  (h_similar : r / x = x / R) : 
  x = Real.sqrt (r * R) := by
sorry

end middle_trapezoid_radius_l3439_343944


namespace range_of_a_l3439_343986

theorem range_of_a (a : ℝ) 
  (h : ∀ x ∈ Set.Icc 3 4, x^2 - 3 > a*x - a) : 
  a < 3 := by sorry

end range_of_a_l3439_343986


namespace probability_third_smallest_is_five_l3439_343949

def set_size : ℕ := 15
def selection_size : ℕ := 8
def target_number : ℕ := 5
def target_position : ℕ := 3

theorem probability_third_smallest_is_five :
  let total_combinations := Nat.choose set_size selection_size
  let favorable_combinations := 
    (Nat.choose (set_size - target_number) (selection_size - target_position)) *
    (Nat.choose (target_number - 1) (target_position - 1))
  (favorable_combinations : ℚ) / total_combinations = 4 / 21 := by
  sorry

end probability_third_smallest_is_five_l3439_343949


namespace complement_A_intersect_B_A_union_B_intersect_C_nonempty_iff_l3439_343907

open Set

-- Define the sets A, B, and C
def A : Set ℝ := Ioc 2 3
def B : Set ℝ := Ioo 1 3
def C (m : ℝ) : Set ℝ := Ici m

-- Statement for part (1)
theorem complement_A_intersect_B : (Aᶜ ∩ B) = Ico 1 2 := by sorry

-- Statement for part (2)
theorem A_union_B_intersect_C_nonempty_iff (m : ℝ) :
  ((A ∪ B) ∩ C m).Nonempty ↔ m ≤ 3 := by sorry

end complement_A_intersect_B_A_union_B_intersect_C_nonempty_iff_l3439_343907


namespace prime_factor_puzzle_l3439_343978

theorem prime_factor_puzzle (a b c d w x y z : ℕ) : 
  w.Prime → x.Prime → y.Prime → z.Prime →
  w < x → x < y → y < z →
  (w^a) * (x^b) * (y^c) * (z^d) = 660 →
  (a + b) - (c + d) = 1 →
  b = 1 := by sorry

end prime_factor_puzzle_l3439_343978


namespace khali_snow_volume_l3439_343901

/-- The volume of snow on Khali's driveway -/
def snow_volume (length width height : ℚ) : ℚ := length * width * height

theorem khali_snow_volume :
  snow_volume 30 4 (3/4) = 90 := by
  sorry

end khali_snow_volume_l3439_343901


namespace percentage_increase_l3439_343990

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 600 → final = 660 → (final - initial) / initial * 100 = 10 := by
  sorry

end percentage_increase_l3439_343990


namespace modular_inverse_17_mod_1021_l3439_343953

theorem modular_inverse_17_mod_1021 (p : Nat) (prime_p : Nat.Prime p) (h : p = 1021) :
  (17 * 961) % p = 1 :=
by sorry

end modular_inverse_17_mod_1021_l3439_343953


namespace sin_cos_identity_l3439_343919

theorem sin_cos_identity : Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (200 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_identity_l3439_343919


namespace peaches_at_stand_l3439_343972

/-- The total number of peaches at the stand after picking more is equal to the sum of the initial number of peaches and the number of peaches picked. -/
theorem peaches_at_stand (initial_peaches picked_peaches : ℕ) :
  initial_peaches + picked_peaches = initial_peaches + picked_peaches :=
by sorry

end peaches_at_stand_l3439_343972


namespace sum_difference_problem_l3439_343941

theorem sum_difference_problem (x y : ℤ) : x + y = 45 → x = 25 → x - y = 5 := by
  sorry

end sum_difference_problem_l3439_343941


namespace factor_polynomial_l3439_343942

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = -27 * x^5 * (5 * x^4 - 2) := by
  sorry

end factor_polynomial_l3439_343942


namespace segment_length_l3439_343943

/-- Given a line segment AB with points P, Q, and R, prove that AB has length 567 -/
theorem segment_length (A B P Q R : Real) : 
  (P - A) / (B - P) = 3 / 4 →  -- P divides AB in ratio 3:4
  (Q - A) / (B - Q) = 4 / 5 →  -- Q divides AB in ratio 4:5
  (R - P) / (Q - R) = 1 / 2 →  -- R divides PQ in ratio 1:2
  R - P = 3 →                  -- Length of PR is 3 units
  B - A = 567 := by            -- Length of AB is 567 units
  sorry


end segment_length_l3439_343943


namespace cloth_selling_price_l3439_343964

/-- Calculates the total selling price of cloth given the quantity, cost price, and loss per metre -/
def total_selling_price (quantity : ℕ) (cost_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  quantity * (cost_price - loss_per_metre)

/-- Theorem stating that the total selling price of 200 metres of cloth
    with a cost price of Rs. 72 per metre and a loss of Rs. 12 per metre
    is Rs. 12,000 -/
theorem cloth_selling_price :
  total_selling_price 200 72 12 = 12000 := by
  sorry

end cloth_selling_price_l3439_343964


namespace adjacent_empty_seats_l3439_343984

theorem adjacent_empty_seats (n : ℕ) (k : ℕ) : n = 6 → k = 3 →
  (number_of_arrangements : ℕ) →
  (number_of_arrangements = 
    -- Case 1: Two adjacent empty seats at the ends
    (2 * (Nat.choose 3 1) * (Nat.choose 3 2)) +
    -- Case 2: Two adjacent empty seats not at the ends
    (3 * (Nat.choose 3 2) * (Nat.choose 2 1))) →
  number_of_arrangements = 72 := by sorry

end adjacent_empty_seats_l3439_343984


namespace lauryn_company_men_count_l3439_343956

theorem lauryn_company_men_count :
  ∀ (men women : ℕ),
    men + women = 180 →
    men = women - 20 →
    men = 80 := by
  sorry

end lauryn_company_men_count_l3439_343956


namespace pizza_order_theorem_l3439_343935

/-- Represents the cost calculation for a pizza order with special pricing --/
def pizza_order_cost (small_price medium_price large_price topping_price : ℚ)
  (triple_cheese_count triple_cheese_toppings : ℕ)
  (meat_lovers_count meat_lovers_toppings : ℕ)
  (veggie_delight_count veggie_delight_toppings : ℕ) : ℚ :=
  let triple_cheese_cost := (triple_cheese_count / 2) * large_price + 
                            triple_cheese_count * triple_cheese_toppings * topping_price
  let meat_lovers_cost := ((meat_lovers_count + 1) / 3) * 2 * medium_price + 
                          meat_lovers_count * meat_lovers_toppings * topping_price
  let veggie_delight_cost := ((veggie_delight_count + 1) / 3) * 2 * small_price + 
                             veggie_delight_count * veggie_delight_toppings * topping_price
  triple_cheese_cost + meat_lovers_cost + veggie_delight_cost

/-- Theorem stating that the given pizza order costs $169 --/
theorem pizza_order_theorem : 
  pizza_order_cost 5 8 10 (5/2) 6 2 4 3 10 1 = 169 := by
  sorry

end pizza_order_theorem_l3439_343935


namespace solution_set_quadratic_inequality_l3439_343937

theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 + 5*x > 6} = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end solution_set_quadratic_inequality_l3439_343937


namespace sum_of_15th_set_l3439_343924

/-- The first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 1 + (n - 1) * n / 2

/-- The last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set of the sequence -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem stating that the sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by sorry

end sum_of_15th_set_l3439_343924


namespace sequence_properties_l3439_343987

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_property (n : ℕ) : S n + n = 2 * sequence_a n

def sequence_b (n : ℕ) : ℝ := n * sequence_a n + n

def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, n > 0 → sequence_a n = 2^n - 1) ∧
  (∀ n : ℕ, n ≥ 11 → (T n - 2) / n > 2018) ∧
  (∀ n : ℕ, n < 11 → (T n - 2) / n ≤ 2018) :=
by sorry

end sequence_properties_l3439_343987


namespace two_transformations_preserve_pattern_l3439_343920

/-- Represents the pattern of squares on the infinite line -/
structure SquarePattern where
  s : ℝ  -- side length of each square
  ℓ : Line2  -- the infinite line

/-- Enumeration of the four transformations -/
inductive Transformation
  | rotation180 : Point → Transformation
  | translation4s : Transformation
  | reflectionAcrossL : Transformation
  | reflectionPerpendicular : Point → Transformation

/-- Predicate to check if a transformation maps the pattern onto itself -/
def mapsOntoItself (t : Transformation) (p : SquarePattern) : Prop :=
  sorry

theorem two_transformations_preserve_pattern (p : SquarePattern) :
  ∃! (ts : Finset Transformation), ts.card = 2 ∧
    ∀ t ∈ ts, mapsOntoItself t p ∧
    ∀ t, mapsOntoItself t p → t ∈ ts :=
  sorry

end two_transformations_preserve_pattern_l3439_343920


namespace ant_count_approximation_l3439_343948

/-- Represents the dimensions and ant densities of a park -/
structure ParkInfo where
  width : ℝ
  length : ℝ
  mainDensity : ℝ
  squareSide : ℝ
  squareDensity : ℝ

/-- Calculates the total number of ants in the park -/
def totalAnts (park : ParkInfo) : ℝ :=
  let totalArea := park.width * park.length
  let squareArea := park.squareSide * park.squareSide
  let mainArea := totalArea - squareArea
  let mainAnts := mainArea * 144 * park.mainDensity  -- Convert to square inches
  let squareAnts := squareArea * park.squareDensity
  mainAnts + squareAnts

/-- The park information as given in the problem -/
def givenPark : ParkInfo := {
  width := 250
  length := 350
  mainDensity := 4
  squareSide := 50
  squareDensity := 6
}

/-- Theorem stating that the total number of ants is approximately 50 million -/
theorem ant_count_approximation :
  abs (totalAnts givenPark - 50000000) ≤ 1000000 := by
  sorry

end ant_count_approximation_l3439_343948


namespace fir_trees_not_adjacent_probability_l3439_343913

def num_pine : ℕ := 5
def num_cedar : ℕ := 6
def num_fir : ℕ := 7
def total_trees : ℕ := num_pine + num_cedar + num_fir

def valid_arrangements : ℕ := Nat.choose (num_pine + num_cedar + 1) num_fir
def total_arrangements : ℕ := Nat.choose total_trees num_fir

theorem fir_trees_not_adjacent_probability :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 40 := by sorry

end fir_trees_not_adjacent_probability_l3439_343913


namespace solution_set_when_a_eq_2_range_of_a_l3439_343925

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (a x : ℝ) : ℝ := 2 * |x - a|

-- Question 1
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f x - g 2 x ≤ x - 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

-- Question 2
theorem range_of_a :
  {a : ℝ | ∀ m > 1, ∃ x₀ : ℝ, f x₀ + g a x₀ ≤ (m^2 + m + 4) / (m - 1)} =
  {a : ℝ | -2 - 2 * Real.sqrt 6 ≤ a ∧ a ≤ 2 * Real.sqrt 6 + 4} := by sorry

end solution_set_when_a_eq_2_range_of_a_l3439_343925


namespace grass_field_width_l3439_343903

theorem grass_field_width (length width path_width cost_per_sqm total_cost : ℝ) :
  length = 95 →
  path_width = 2.5 →
  cost_per_sqm = 2 →
  total_cost = 1550 →
  (length + 2 * path_width) * (width + 2 * path_width) - length * width = total_cost / cost_per_sqm →
  width = 55 := by
sorry

end grass_field_width_l3439_343903


namespace problem_solution_l3439_343946

theorem problem_solution (a b x y m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : x * y = 1) 
  (h3 : |m| = 2) : 
  m^2 + (a + b) / 2 + (-x * y)^2023 = 3 := by
sorry

end problem_solution_l3439_343946


namespace min_value_theorem_l3439_343908

-- Define a positive term geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  is_positive_geometric_sequence a →
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  a 6 = a 5 + 2 * a 4 →
  (∀ k l : ℕ, 1 / k + 4 / l ≥ 3 / 2) ∧
  (∃ k l : ℕ, 1 / k + 4 / l = 3 / 2) :=
by sorry

end min_value_theorem_l3439_343908


namespace factors_of_1320_l3439_343912

/-- The number of distinct positive factors of a natural number n -/
def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- 1320 is our number of interest -/
def our_number : ℕ := 1320

/-- Theorem stating that 1320 has 32 distinct positive factors -/
theorem factors_of_1320 : num_factors our_number = 32 := by
  sorry

end factors_of_1320_l3439_343912


namespace ice_cream_volume_l3439_343922

/-- The volume of ice cream in a cone with a spherical top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let sphere_volume := (4 / 3) * π * r^3
  h = 12 ∧ r = 3 → cone_volume + sphere_volume = 72 * π := by sorry

end ice_cream_volume_l3439_343922


namespace ratio_hcf_to_lcm_l3439_343906

/-- Given three positive integers a, b, and c in the ratio 3:4:5 with HCF 40, their LCM is 2400 -/
theorem ratio_hcf_to_lcm (a b c : ℕ+) : 
  (a : ℚ) / 3 = (b : ℚ) / 4 ∧ (b : ℚ) / 4 = (c : ℚ) / 5 → 
  Nat.gcd a.val (Nat.gcd b.val c.val) = 40 →
  Nat.lcm a.val (Nat.lcm b.val c.val) = 2400 := by
sorry

end ratio_hcf_to_lcm_l3439_343906


namespace initial_players_l3439_343928

theorem initial_players (initial_players : ℕ) : 
  (∀ (players : ℕ), 
    (players = initial_players + 2) →
    (7 * players = 63)) →
  initial_players = 7 :=
by
  sorry

end initial_players_l3439_343928


namespace p_sufficient_not_necessary_l3439_343962

-- Define the propositions
def p (x : ℝ) : Prop := -2 < x ∧ x < 0
def q (x : ℝ) : Prop := |x| < 2

-- Theorem statement
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end p_sufficient_not_necessary_l3439_343962


namespace supermarket_spending_l3439_343974

theorem supermarket_spending (total : ℝ) (category1 : ℝ) (category2 : ℝ) (category3 : ℝ) (category4 : ℝ) :
  total = 120 →
  category1 = (1 / 2) * total →
  category2 = (1 / 10) * total →
  category3 = 8 →
  category1 + category2 + category3 + category4 = total →
  category4 / total = 1 / 3 := by
  sorry

end supermarket_spending_l3439_343974


namespace parking_lot_cars_remaining_l3439_343977

theorem parking_lot_cars_remaining (initial_cars : ℕ) 
  (first_group_left : ℕ) (second_group_left : ℕ) : 
  initial_cars = 24 → first_group_left = 8 → second_group_left = 6 →
  initial_cars - first_group_left - second_group_left = 10 := by
  sorry

end parking_lot_cars_remaining_l3439_343977


namespace a_profit_share_l3439_343967

-- Define the total investment and profit
def total_investment : ℕ := 90000
def total_profit : ℕ := 8640

-- Define the relationships between investments
def investment_relations (a b c : ℕ) : Prop :=
  a = b + 6000 ∧ b + 3000 = c ∧ a + b + c = total_investment

-- Define the profit sharing ratio
def profit_ratio (a b c : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ a / k = 11 ∧ b / k = 9 ∧ c / k = 10

-- Theorem statement
theorem a_profit_share (a b c : ℕ) :
  investment_relations a b c →
  profit_ratio a b c →
  (11 : ℚ) / 30 * total_profit = 3168 :=
by sorry

end a_profit_share_l3439_343967


namespace ruth_total_score_l3439_343960

-- Define the given conditions
def dean_total_points : ℕ := 252
def dean_games : ℕ := 28
def games_difference : ℕ := 10
def average_difference : ℚ := 1/2

-- Define Ruth's games
def ruth_games : ℕ := dean_games - games_difference

-- Define Dean's average
def dean_average : ℚ := dean_total_points / dean_games

-- Define Ruth's average
def ruth_average : ℚ := dean_average + average_difference

-- Theorem to prove
theorem ruth_total_score : ℕ := by
  -- The proof goes here
  sorry

end ruth_total_score_l3439_343960


namespace fraction_simplification_l3439_343927

theorem fraction_simplification (x y : ℚ) (hx : x = 5) (hy : y = 8) :
  (1 / x - 1 / y) / (1 / x) = 3 / 8 := by
  sorry

end fraction_simplification_l3439_343927


namespace base_five_representation_of_156_l3439_343940

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBaseFive (n / 5)

/-- Checks if a list of digits represents a valid base 5 number --/
def isValidBaseFive (digits : List ℕ) : Prop :=
  digits.all (· < 5)

theorem base_five_representation_of_156 :
  let base5Repr := toBaseFive 156
  isValidBaseFive base5Repr ∧ base5Repr = [1, 1, 1, 1] := by
  sorry

#eval toBaseFive 156  -- Should output [1, 1, 1, 1]

end base_five_representation_of_156_l3439_343940


namespace remainder_theorem_l3439_343930

theorem remainder_theorem (n : ℤ) (h : n % 7 = 3) : (5 * n - 11) % 7 = 4 := by
  sorry

end remainder_theorem_l3439_343930


namespace library_meeting_problem_l3439_343933

theorem library_meeting_problem (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_prime : ∀ (p : ℕ), Prime p → ¬(p^2 ∣ z))
  (h_n : (x : ℝ) - y * Real.sqrt z = 120 - 60 * Real.sqrt 2)
  (h_prob : (14400 - (120 - (x - y * Real.sqrt z))^2) / 14400 = 1/2) :
  x + y + z = 182 := by
sorry

end library_meeting_problem_l3439_343933


namespace supermarket_spending_l3439_343939

theorem supermarket_spending (total : ℚ) :
  (1 / 4 : ℚ) * total +
  (1 / 3 : ℚ) * total +
  (1 / 6 : ℚ) * total +
  6 = total →
  total = 24 := by
sorry

end supermarket_spending_l3439_343939


namespace stick_ratio_proof_l3439_343982

/-- Prove that the ratio of the uncovered portion of Pat's stick to Sarah's stick is 1/2 -/
theorem stick_ratio_proof (pat_stick : ℕ) (pat_covered : ℕ) (jane_stick : ℕ) (sarah_stick : ℕ) : 
  pat_stick = 30 →
  pat_covered = 7 →
  jane_stick = 22 →
  sarah_stick = jane_stick + 24 →
  (pat_stick - pat_covered : ℚ) / sarah_stick = 1 / 2 := by
  sorry


end stick_ratio_proof_l3439_343982


namespace ab_nonpositive_l3439_343955

theorem ab_nonpositive (a b : ℝ) : (∀ x, (2*a + b)*x - 1 ≠ 0) → a*b ≤ 0 :=
sorry

end ab_nonpositive_l3439_343955
