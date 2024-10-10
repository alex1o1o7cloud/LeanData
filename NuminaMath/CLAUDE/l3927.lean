import Mathlib

namespace circle_intersection_and_reflection_l3927_392715

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 1

-- Define point A
def A : ℝ × ℝ := (4, 0)

-- Define the reflecting line
def reflecting_line (x y : ℝ) : Prop := x - y - 3 = 0

theorem circle_intersection_and_reflection :
  -- Part I: Equation of line l
  (∃ (k : ℝ), (∀ x y : ℝ, y = k * (x - 4) → 
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ C₁ x₁ (k * (x₁ - 4)) ∧ C₁ x₂ (k * (x₂ - 4)) ∧ 
    (x₁ - x₂)^2 + (k * (x₁ - 4) - k * (x₂ - 4))^2 = 12)) ↔
    (k = 0 ∨ 7 * x + 24 * y - 28 = 0)) ∧
  -- Part II: Range of slope of reflected line
  (∀ k : ℝ, (∃ x y : ℝ, C₂ x y ∧ k * x - y - 4 * k - 6 = 0) ↔ 
    (k ≤ -2 * Real.sqrt 30 ∨ k ≥ 2 * Real.sqrt 30)) :=
sorry

end circle_intersection_and_reflection_l3927_392715


namespace race_distance_l3927_392798

/-- Represents a race between two participants A and B -/
structure Race where
  distance : ℝ
  timeA : ℝ
  timeB : ℝ
  speedA : ℝ
  speedB : ℝ

/-- The conditions of the race -/
def raceConditions (r : Race) : Prop :=
  r.timeA = 18 ∧
  r.timeB = r.timeA + 7 ∧
  r.distance = r.speedA * r.timeA ∧
  r.distance = r.speedB * r.timeB ∧
  r.distance - r.speedB * r.timeA = 56

theorem race_distance (r : Race) (h : raceConditions r) : r.distance = 200 := by
  sorry

#check race_distance

end race_distance_l3927_392798


namespace square_fold_visible_area_l3927_392763

theorem square_fold_visible_area (side_length : ℝ) (ao_length : ℝ) : 
  side_length = 1 → ao_length = 1/3 → 
  (visible_area : ℝ) = side_length * ao_length :=
by sorry

end square_fold_visible_area_l3927_392763


namespace remainder_divisibility_l3927_392728

theorem remainder_divisibility (y : ℤ) : 
  ∃ k : ℤ, y = 288 * k + 45 → ∃ m : ℤ, y = 24 * m + 21 := by
  sorry

end remainder_divisibility_l3927_392728


namespace hannah_movie_remaining_time_l3927_392722

/-- Calculates the remaining movie time given the total duration and watched duration. -/
def remaining_movie_time (total_duration watched_duration : ℕ) : ℕ :=
  total_duration - watched_duration

/-- Proves that for a 3-hour movie watched for 2 hours and 24 minutes, 36 minutes remain. -/
theorem hannah_movie_remaining_time :
  let total_duration : ℕ := 3 * 60  -- 3 hours in minutes
  let watched_duration : ℕ := 2 * 60 + 24  -- 2 hours and 24 minutes
  remaining_movie_time total_duration watched_duration = 36 := by
  sorry

#eval remaining_movie_time (3 * 60) (2 * 60 + 24)

end hannah_movie_remaining_time_l3927_392722


namespace cost_of_three_l3927_392799

/-- Represents the prices of fruits and vegetables -/
structure Prices where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ
  eggplant : ℝ

/-- The total cost of all items is $30 -/
def total_cost (p : Prices) : Prop :=
  p.apples + p.bananas + p.cantaloupe + p.dates + p.eggplant = 30

/-- The carton of dates costs twice as much as the sack of apples -/
def dates_cost (p : Prices) : Prop :=
  p.dates = 2 * p.apples

/-- The price of cantaloupe equals price of apples minus price of bananas -/
def cantaloupe_cost (p : Prices) : Prop :=
  p.cantaloupe = p.apples - p.bananas

/-- The price of eggplant is the sum of apples and bananas prices -/
def eggplant_cost (p : Prices) : Prop :=
  p.eggplant = p.apples + p.bananas

/-- The main theorem: Given the conditions, the cost of bananas, cantaloupe, and eggplant is $12 -/
theorem cost_of_three (p : Prices) 
  (h1 : total_cost p) 
  (h2 : dates_cost p) 
  (h3 : cantaloupe_cost p) 
  (h4 : eggplant_cost p) : 
  p.bananas + p.cantaloupe + p.eggplant = 12 := by
  sorry

end cost_of_three_l3927_392799


namespace jimmy_has_more_sheets_l3927_392737

/-- Given the initial number of sheets and additional sheets received,
    calculate the difference between Jimmy's and Tommy's final sheet counts. -/
def sheet_difference (jimmy_initial : ℕ) (tommy_more_initial : ℕ) 
  (jimmy_additional1 : ℕ) (jimmy_additional2 : ℕ)
  (tommy_additional1 : ℕ) (tommy_additional2 : ℕ) : ℕ :=
  let tommy_initial := jimmy_initial + tommy_more_initial
  let jimmy_final := jimmy_initial + jimmy_additional1 + jimmy_additional2
  let tommy_final := tommy_initial + tommy_additional1 + tommy_additional2
  jimmy_final - tommy_final

/-- Theorem stating that Jimmy will have 58 more sheets than Tommy
    after receiving additional sheets. -/
theorem jimmy_has_more_sheets :
  sheet_difference 58 25 85 47 30 19 = 58 := by
  sorry

end jimmy_has_more_sheets_l3927_392737


namespace bottle_recycling_result_l3927_392746

/-- Calculates the number of new bottles created through recycling -/
def recycleBottles (initialBottles : ℕ) : ℕ :=
  let firstRound := initialBottles / 5
  let secondRound := firstRound / 5
  let thirdRound := secondRound / 5
  firstRound + secondRound + thirdRound

/-- Represents the recycling process with initial conditions -/
def bottleRecyclingProcess (initialBottles : ℕ) : Prop :=
  recycleBottles initialBottles = 179

/-- Theorem stating the result of the bottle recycling process -/
theorem bottle_recycling_result :
  bottleRecyclingProcess 729 := by sorry

end bottle_recycling_result_l3927_392746


namespace equation_solution_l3927_392751

theorem equation_solution :
  ∃ x : ℚ, (2 / 7) * (1 / 4) * x = 8 ∧ x = 112 :=
by
  sorry

end equation_solution_l3927_392751


namespace quadratic_polynomial_with_complex_root_l3927_392748

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (z : ℂ), z = (2 : ℝ) + (2 : ℝ) * I → a * z^2 + b * z + c = 0) ∧
    (a * X^2 + b * X + c = 3 * X^2 - 12 * X + 24) :=
sorry

end quadratic_polynomial_with_complex_root_l3927_392748


namespace complex_fraction_equality_l3927_392761

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (1 + 2*i) / ((1 - i)^2) = 1 - (1/2)*i := by
  sorry

end complex_fraction_equality_l3927_392761


namespace x_value_proof_l3927_392727

theorem x_value_proof (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end x_value_proof_l3927_392727


namespace ab_equals_op_l3927_392739

noncomputable section

/-- Line l with parametric equations x = -1/2 * t, y = a + (√3/2) * t -/
def line_l (a t : ℝ) : ℝ × ℝ := (-1/2 * t, a + (Real.sqrt 3 / 2) * t)

/-- Curve C with rectangular equation x² + y² - 4x = 0 -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- Length of AB, where A and B are intersection points of line l and curve C -/
def length_AB (a : ℝ) : ℝ := Real.sqrt (4 + 4 * Real.sqrt 3 * a - a^2)

/-- Theorem stating that |AB| = 2 if and only if a = 0 or a = 4√3 -/
theorem ab_equals_op (a : ℝ) : length_AB a = 2 ↔ a = 0 ∨ a = 4 * Real.sqrt 3 := by
  sorry

end

end ab_equals_op_l3927_392739


namespace flight_duration_sum_main_flight_theorem_l3927_392777

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a flight with departure and arrival times -/
structure Flight where
  departure : Time
  arrival : Time

def Flight.duration (f : Flight) : ℕ × ℕ :=
  sorry

theorem flight_duration_sum (f : Flight) (time_zone_diff : ℕ) (daylight_saving : ℕ) :
  let (h, m) := f.duration
  h + m = 32 :=
by
  sorry

/-- The main theorem proving the flight duration sum -/
theorem main_flight_theorem : ∃ (f : Flight) (time_zone_diff daylight_saving : ℕ),
  f.departure = ⟨7, 15, sorry⟩ ∧
  f.arrival = ⟨17, 40, sorry⟩ ∧
  time_zone_diff = 3 ∧
  daylight_saving = 1 ∧
  (let (h, m) := f.duration
   0 < m ∧ m < 60 ∧ h + m = 32) :=
by
  sorry

end flight_duration_sum_main_flight_theorem_l3927_392777


namespace farm_sheep_count_l3927_392784

/-- Represents the farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  racehorses : ℕ
  draft_horses : ℕ

/-- The ratio of sheep to total horses is 7:8 -/
def sheep_horse_ratio (f : Farm) : Prop :=
  7 * (f.racehorses + f.draft_horses) = 8 * f.sheep

/-- Total horse food consumption per day -/
def total_horse_food (f : Farm) : ℕ :=
  250 * f.racehorses + 300 * f.draft_horses

/-- There is 1/3 more racehorses than draft horses -/
def racehorse_draft_ratio (f : Farm) : Prop :=
  f.racehorses = f.draft_horses + (f.draft_horses / 3)

/-- The farm satisfies all given conditions -/
def valid_farm (f : Farm) : Prop :=
  sheep_horse_ratio f ∧
  total_horse_food f = 21000 ∧
  racehorse_draft_ratio f

theorem farm_sheep_count :
  ∃ f : Farm, valid_farm f ∧ f.sheep = 67 :=
sorry

end farm_sheep_count_l3927_392784


namespace sine_tangent_comparison_l3927_392732

open Real

theorem sine_tangent_comparison (α : ℝ) (h : 0 < α ∧ α < π / 2) : 
  sin α < tan α ∧ (deriv sin) α < (deriv tan) α := by sorry

end sine_tangent_comparison_l3927_392732


namespace cosine_value_in_triangle_l3927_392754

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem cosine_value_in_triangle (t : Triangle) 
  (hm : Vector2D := ⟨Real.sqrt 3 * t.b - t.c, Real.cos t.C⟩)
  (hn : Vector2D := ⟨t.a, Real.cos t.A⟩)
  (h_parallel : parallel hm hn) :
  Real.cos t.A = Real.sqrt 3 / 3 :=
sorry

end cosine_value_in_triangle_l3927_392754


namespace overlapping_triangle_is_equilateral_l3927_392797

/-- Represents a right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- Represents the overlapping triangle formed by two identical right-angled triangles -/
structure OverlappingTriangle where
  original : RightTriangle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- 
Given two identical right-angled triangles arranged such that the right angle vertex 
of one triangle lies on the side of the other, the resulting overlapping triangle is equilateral.
-/
theorem overlapping_triangle_is_equilateral (t : RightTriangle) 
  (ot : OverlappingTriangle) (h : ot.original = t) : 
  ot.side1 = ot.side2 ∧ ot.side2 = ot.side3 := by
  sorry


end overlapping_triangle_is_equilateral_l3927_392797


namespace largest_three_digit_multiple_of_8_with_digit_sum_16_l3927_392759

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_16 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → digit_sum n = 16 → n ≤ 880 :=
by sorry

end largest_three_digit_multiple_of_8_with_digit_sum_16_l3927_392759


namespace min_a_value_l3927_392782

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * Real.log x - 3 * x^2 - 11 * x

-- Define the inequality condition
def inequality_condition (a : ℝ) : Prop :=
  ∀ x > 0, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x + 1

-- State the theorem
theorem min_a_value :
  ∃ a : ℤ, (inequality_condition a ∧ ∀ b : ℤ, b < a → ¬inequality_condition b) :=
sorry

end

end min_a_value_l3927_392782


namespace no_square_root_among_options_l3927_392710

theorem no_square_root_among_options : ∃ (x : ℝ), x ^ 2 = 0 ∧
                                       ∃ (x : ℝ), x ^ 2 = (-2)^2 ∧
                                       ∃ (x : ℝ), x ^ 2 = |9| ∧
                                       ¬∃ (x : ℝ), x ^ 2 = -|(-5)| := by
  sorry

#check no_square_root_among_options

end no_square_root_among_options_l3927_392710


namespace range_of_a_l3927_392775

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp (x - 1) + 2 * x - log a * x / log (sqrt 2)

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  (∃ x y, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) →
  a > 2^(3/2) ∧ a < 2^((exp 1 + 4)/4) :=
sorry

end range_of_a_l3927_392775


namespace triangle_perimeter_l3927_392766

/-- Theorem: For a triangle with sides in the ratio 5:6:7 and the longest side measuring 280 cm, the perimeter is 720 cm. -/
theorem triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  a / b = 5 / 6 →          -- Ratio of first two sides
  b / c = 6 / 7 →          -- Ratio of second two sides
  c = 280 →                -- Length of longest side
  a + b + c = 720 :=       -- Perimeter
by sorry

end triangle_perimeter_l3927_392766


namespace binary_110101_is_53_l3927_392714

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110101_is_53 :
  binary_to_decimal [true, false, true, false, true, true] = 53 := by
  sorry

end binary_110101_is_53_l3927_392714


namespace fermat_little_theorem_extension_l3927_392707

theorem fermat_little_theorem_extension (p : ℕ) (a b : ℤ) 
  (hp : Nat.Prime p) (hab : a ≡ b [ZMOD p]) : 
  a^p ≡ b^p [ZMOD p^2] := by
  sorry

end fermat_little_theorem_extension_l3927_392707


namespace regular_polygon_sides_l3927_392743

theorem regular_polygon_sides (interior_angle : ℝ) (n : ℕ) : 
  interior_angle = 120 → (n : ℝ) * (180 - interior_angle) = 360 → n = 6 := by
  sorry

end regular_polygon_sides_l3927_392743


namespace cube_root_floor_equality_l3927_392735

theorem cube_root_floor_equality (n : ℕ) :
  ⌊(n : ℝ)^(1/3) + (n + 1 : ℝ)^(1/3)⌋ = ⌊(8*n + 3 : ℝ)^(1/3)⌋ :=
sorry

end cube_root_floor_equality_l3927_392735


namespace temperature_conversion_l3927_392767

theorem temperature_conversion (t k a : ℝ) : 
  t = 5/9 * (k - 32) + a * k → t = 20 → a = 3 → k = 10.625 := by
  sorry

end temperature_conversion_l3927_392767


namespace find_r_value_l3927_392785

theorem find_r_value (x y k : ℝ) (h : y^2 + 4*y + 4 + Real.sqrt (x + y + k) = 0) :
  let r := |x * y|
  r = 2 := by
  sorry

end find_r_value_l3927_392785


namespace inequality_solution_l3927_392781

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  {x | (a + 2) * x - 4 ≤ 2 * (x - 1)}

theorem inequality_solution (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 2 → solution_set a = {x | 1 < x ∧ x ≤ 2/a}) ∧
  (a = 2 → solution_set a = ∅) ∧
  (a > 2 → solution_set a = {x | 2/a ≤ x ∧ x < 1}) :=
by sorry

end inequality_solution_l3927_392781


namespace negation_of_existence_negation_of_quadratic_inequality_l3927_392712

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ ∀ x > 0, ¬ P x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x > 0, x^2 - 2*x + 1 > 0) ↔ (∀ x > 0, x^2 - 2*x + 1 ≤ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l3927_392712


namespace jacob_ladder_price_l3927_392765

/-- The price per rung for Jacob's ladders -/
def price_per_rung : ℚ := 2

/-- The number of ladders with 50 rungs -/
def ladders_50 : ℕ := 10

/-- The number of ladders with 60 rungs -/
def ladders_60 : ℕ := 20

/-- The number of rungs per ladder in the first group -/
def rungs_per_ladder_50 : ℕ := 50

/-- The number of rungs per ladder in the second group -/
def rungs_per_ladder_60 : ℕ := 60

/-- The total cost for all ladders -/
def total_cost : ℚ := 3400

theorem jacob_ladder_price : 
  price_per_rung * (ladders_50 * rungs_per_ladder_50 + ladders_60 * rungs_per_ladder_60) = total_cost := by
  sorry

end jacob_ladder_price_l3927_392765


namespace jelly_bean_count_l3927_392723

/-- The number of red jelly beans in one bag -/
def red_in_bag : ℕ := 24

/-- The number of white jelly beans in one bag -/
def white_in_bag : ℕ := 18

/-- The number of bags needed to fill the fishbowl -/
def bags_to_fill : ℕ := 3

/-- The total number of red and white jelly beans in the fishbowl -/
def total_red_white : ℕ := (red_in_bag + white_in_bag) * bags_to_fill

theorem jelly_bean_count : total_red_white = 126 := by
  sorry

end jelly_bean_count_l3927_392723


namespace subtraction_equality_l3927_392796

theorem subtraction_equality : 8888888888888 - 4444444444444 = 4444444444444 := by
  sorry

end subtraction_equality_l3927_392796


namespace max_value_of_symmetric_f_l3927_392772

/-- A function f(x) that is symmetric about the line x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The symmetry condition for f(x) about x = -2 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x, f a b x = f a b (-4 - x)

/-- The maximum value of f(x) is 16 when it's symmetric about x = -2 -/
theorem max_value_of_symmetric_f (a b : ℝ) (h : is_symmetric a b) :
  ∃ x₀, ∀ x, f a b x ≤ f a b x₀ ∧ f a b x₀ = 16 :=
sorry

end max_value_of_symmetric_f_l3927_392772


namespace z_purely_imaginary_z_squared_over_z_plus_5_plus_2i_l3927_392747

def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2 : ℝ) + (m^2 - 3 * m + 2 : ℝ) * Complex.I

theorem z_purely_imaginary : z (-1/2) = Complex.I * ((-1/4)^2 - 3 * (-1/4) + 2) := by sorry

theorem z_squared_over_z_plus_5_plus_2i :
  z 0 ^ 2 / (z 0 + 5 + 2 * Complex.I) = -32/25 - 24/25 * Complex.I := by sorry

end z_purely_imaginary_z_squared_over_z_plus_5_plus_2i_l3927_392747


namespace unique_triple_l3927_392709

/-- Least common multiple of two positive integers -/
def lcm (x y : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given LCM conditions -/
def count_triples : ℕ := sorry

theorem unique_triple : count_triples = 1 := by sorry

end unique_triple_l3927_392709


namespace gcd_12n_plus_5_7n_plus_3_l3927_392730

theorem gcd_12n_plus_5_7n_plus_3 (n : ℕ+) : Nat.gcd (12 * n + 5) (7 * n + 3) = 1 := by
  sorry

end gcd_12n_plus_5_7n_plus_3_l3927_392730


namespace focus_of_ellipse_l3927_392791

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 5 = 1

/-- Definition of a focus of an ellipse -/
def is_focus (a b c : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = c^2 ∧ a^2 = b^2 + c^2 ∧ a > b ∧ b > 0

/-- Theorem: (0, 1) is a focus of the given ellipse -/
theorem focus_of_ellipse :
  ∃ (a b c : ℝ), a^2 = 5 ∧ b^2 = 4 ∧ 
  (∀ (x y : ℝ), ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  is_focus a b c 0 1 :=
sorry

end focus_of_ellipse_l3927_392791


namespace hill_climbing_speed_l3927_392771

/-- Proves that given a round trip with specified conditions, 
    the average speed for the upward journey is 1.125 km/h -/
theorem hill_climbing_speed 
  (up_time : ℝ) 
  (down_time : ℝ) 
  (avg_speed : ℝ) 
  (h1 : up_time = 4) 
  (h2 : down_time = 2) 
  (h3 : avg_speed = 1.5) : 
  (avg_speed * (up_time + down_time)) / (2 * up_time) = 1.125 := by
  sorry

#check hill_climbing_speed

end hill_climbing_speed_l3927_392771


namespace total_children_l3927_392733

theorem total_children (happy : ℕ) (sad : ℕ) (neutral : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neutral_boys : ℕ)
  (h1 : happy = 30)
  (h2 : sad = 10)
  (h3 : neutral = 20)
  (h4 : boys = 16)
  (h5 : girls = 44)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 4) :
  boys + girls = 60 := by
  sorry

end total_children_l3927_392733


namespace mod_equivalence_problem_l3927_392700

theorem mod_equivalence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 27483 % 17 = n := by
  sorry

end mod_equivalence_problem_l3927_392700


namespace sundae_price_l3927_392738

/-- Given the following conditions:
  * The caterer ordered 125 ice-cream bars
  * The caterer ordered 125 sundaes
  * The total price was $225.00
  * The price of each ice-cream bar was $0.60
Prove that the price of each sundae was $1.20 -/
theorem sundae_price 
  (num_ice_cream : ℕ) 
  (num_sundae : ℕ) 
  (total_price : ℚ) 
  (ice_cream_price : ℚ) 
  (h1 : num_ice_cream = 125)
  (h2 : num_sundae = 125)
  (h3 : total_price = 225)
  (h4 : ice_cream_price = 6/10) : 
  (total_price - num_ice_cream * ice_cream_price) / num_sundae = 12/10 := by
  sorry


end sundae_price_l3927_392738


namespace simplify_expression_1_simplify_expression_2_l3927_392726

-- Part 1
theorem simplify_expression_1 (a b : ℝ) : 2*a - (-3*b - 3*(3*a - b)) = 11*a := by sorry

-- Part 2
theorem simplify_expression_2 (a b : ℝ) : 12*a*b^2 - (7*a^2*b - (a*b^2 - 3*a^2*b)) = 13*a*b^2 - 10*a^2*b := by sorry

end simplify_expression_1_simplify_expression_2_l3927_392726


namespace equation_solutions_l3927_392768

theorem equation_solutions :
  let f : ℝ → ℝ → ℝ := λ x y => y^4 + 4*y^2*x - 11*y^2 + 4*x*y - 8*y + 8*x^2 - 40*x + 52
  ∀ x y : ℝ, f x y = 0 ↔ (x = 1 ∧ y = 2) ∨ (x = 5/2 ∧ y = -1) :=
by sorry

end equation_solutions_l3927_392768


namespace fifty_billion_scientific_notation_l3927_392720

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem fifty_billion_scientific_notation :
  toScientificNotation 50000000000 = ScientificNotation.mk 5 10 sorry :=
sorry

end fifty_billion_scientific_notation_l3927_392720


namespace max_value_z_l3927_392745

/-- The maximum value of z = 2x + y given the specified constraints -/
theorem max_value_z (x y : ℝ) (h1 : y ≤ 2 * x) (h2 : x - 2 * y - 4 ≤ 0) (h3 : y ≤ 4 - x) :
  (∀ x' y' : ℝ, y' ≤ 2 * x' → x' - 2 * y' - 4 ≤ 0 → y' ≤ 4 - x' → 2 * x' + y' ≤ 2 * x + y) ∧
  2 * x + y = 8 :=
by sorry

end max_value_z_l3927_392745


namespace book_arrangement_and_selection_l3927_392757

/-- Given 3 math books, 4 physics books, and 2 chemistry books, prove:
    1. The number of arrangements keeping books of the same subject together
    2. The number of ways to select exactly 2 math books, 2 physics books, and 1 chemistry book
    3. The number of ways to select 5 books with at least 1 math book -/
theorem book_arrangement_and_selection 
  (math_books : ℕ) (physics_books : ℕ) (chemistry_books : ℕ) 
  (h_math : math_books = 3) 
  (h_physics : physics_books = 4) 
  (h_chemistry : chemistry_books = 2) :
  (-- 1. Number of arrangements
   (Nat.factorial math_books) * (Nat.factorial physics_books) * 
   (Nat.factorial chemistry_books) * (Nat.factorial 3) = 1728) ∧ 
  (-- 2. Number of ways to select 2 math, 2 physics, 1 chemistry
   (Nat.choose math_books 2) * (Nat.choose physics_books 2) * 
   (Nat.choose chemistry_books 1) = 36) ∧
  (-- 3. Number of ways to select 5 books with at least 1 math
   (Nat.choose (math_books + physics_books + chemistry_books) 5) - 
   (Nat.choose (physics_books + chemistry_books) 5) = 120) := by
  sorry

end book_arrangement_and_selection_l3927_392757


namespace distinct_primes_dividing_sequence_l3927_392752

theorem distinct_primes_dividing_sequence (n M : ℕ) (h : M > n^(n-1)) :
  ∃ (p : Fin n → ℕ), (∀ i : Fin n, Nat.Prime (p i)) ∧ 
  (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
  (∀ i : Fin n, (p i) ∣ (M + i.val + 1)) :=
sorry

end distinct_primes_dividing_sequence_l3927_392752


namespace algebraic_expression_equality_l3927_392788

theorem algebraic_expression_equality (x y : ℝ) (h : 2*x - 3*y = 1) : 
  6*y - 4*x + 8 = 6 := by
sorry

end algebraic_expression_equality_l3927_392788


namespace factor_expression_l3927_392718

theorem factor_expression (x y : ℝ) : 
  5 * x * (x + 1) + 7 * (x + 1) - 2 * y * (x + 1) = (x + 1) * (5 * x + 7 - 2 * y) := by
  sorry

end factor_expression_l3927_392718


namespace unique_four_digit_number_l3927_392729

theorem unique_four_digit_number : ∃! n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧
  n % 131 = 112 ∧
  n % 132 = 98 :=
by
  -- The proof goes here
  sorry

end unique_four_digit_number_l3927_392729


namespace derivative_periodicity_l3927_392724

theorem derivative_periodicity (f : ℝ → ℝ) (T : ℝ) (h_diff : Differentiable ℝ f) (h_periodic : ∀ x, f (x + T) = f x) (h_pos : T > 0) :
  ∀ x, deriv f (x + T) = deriv f x :=
by sorry

end derivative_periodicity_l3927_392724


namespace complex_quadrant_l3927_392705

theorem complex_quadrant (z : ℂ) (h : z * (1 + Complex.I) = -2 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_quadrant_l3927_392705


namespace sunshine_orchard_pumpkins_l3927_392789

/-- The number of pumpkins at Moonglow Orchard -/
def moonglow_pumpkins : ℕ := 14

/-- The number of pumpkins at Sunshine Orchard -/
def sunshine_pumpkins : ℕ := 3 * moonglow_pumpkins + 12

theorem sunshine_orchard_pumpkins : sunshine_pumpkins = 54 := by
  sorry

end sunshine_orchard_pumpkins_l3927_392789


namespace solution_approximation_l3927_392793

/-- A linear function f. -/
noncomputable def f (x : ℝ) : ℝ := x

/-- The equation to be solved. -/
def equation (x : ℝ) : Prop :=
  f (x * 0.004) / 0.03 = 9.237333333333334

/-- The theorem stating that the solution to the equation is approximately 69.3. -/
theorem solution_approximation :
  ∃ x : ℝ, equation x ∧ abs (x - 69.3) < 0.001 :=
sorry

end solution_approximation_l3927_392793


namespace canned_food_bins_l3927_392773

theorem canned_food_bins (soup : ℝ) (vegetables : ℝ) (pasta : ℝ)
  (h1 : soup = 0.125)
  (h2 : vegetables = 0.125)
  (h3 : pasta = 0.5) :
  soup + vegetables + pasta = 0.75 := by
sorry

end canned_food_bins_l3927_392773


namespace sons_ages_l3927_392734

def father_age : ℕ := 33
def youngest_son_age : ℕ := 2
def years_until_sum_equal : ℕ := 12

def is_valid_ages (middle_son_age oldest_son_age : ℕ) : Prop :=
  (father_age + years_until_sum_equal = 
   (youngest_son_age + years_until_sum_equal) + 
   (middle_son_age + years_until_sum_equal) + 
   (oldest_son_age + years_until_sum_equal)) ∧
  (middle_son_age > youngest_son_age) ∧
  (oldest_son_age > middle_son_age)

theorem sons_ages : 
  ∃ (middle_son_age oldest_son_age : ℕ),
    is_valid_ages middle_son_age oldest_son_age ∧
    middle_son_age = 3 ∧ oldest_son_age = 4 :=
by
  sorry

end sons_ages_l3927_392734


namespace exactly_two_lines_l3927_392719

/-- Two lines in 3D space -/
structure Line3D where
  -- We'll represent a line by a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ := sorry

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- A point in 3D space -/
def Point3D := ℝ × ℝ × ℝ

/-- Count lines through a point forming a specific angle with two given lines -/
def count_lines_with_angle (a b : Line3D) (P : Point3D) (θ : ℝ) : ℕ := sorry

theorem exactly_two_lines 
  (a b : Line3D) (P : Point3D) 
  (h_skew : are_skew a b) 
  (h_angle : angle_between_lines a b = 40 * π / 180) :
  count_lines_with_angle a b P (30 * π / 180) = 2 := by
  sorry

end exactly_two_lines_l3927_392719


namespace weight_of_b_l3927_392776

/-- Given the average weights of three people (a, b, c) and two pairs (a, b) and (b, c),
    prove that the weight of b is 31 kg. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)  -- average weight of a, b, and c is 45 kg
  (h2 : (a + b) / 2 = 40)      -- average weight of a and b is 40 kg
  (h3 : (b + c) / 2 = 43) :    -- average weight of b and c is 43 kg
  b = 31 := by
  sorry

end weight_of_b_l3927_392776


namespace count_satisfying_integers_l3927_392749

def is_geometric_mean_integer (n : ℕ+) : Prop :=
  ∃ k : ℕ+, n = 2015 * k^2

def is_harmonic_mean_integer (n : ℕ+) : Prop :=
  ∃ m : ℕ+, 2 * 2015 * n = m * (2015 + n)

def satisfies_conditions (n : ℕ+) : Prop :=
  is_geometric_mean_integer n ∧ is_harmonic_mean_integer n

theorem count_satisfying_integers :
  (∃! (s : Finset ℕ+), s.card = 5 ∧ ∀ n, n ∈ s ↔ satisfies_conditions n) ∧
  2015 = 5 * 13 * 31 := by
  sorry

end count_satisfying_integers_l3927_392749


namespace one_point_45_deg_equals_1_deg_27_min_l3927_392725

/-- Conversion of degrees to minutes -/
def deg_to_min (d : ℝ) : ℝ := d * 60

/-- Theorem stating that 1.45° is equal to 1°27′ -/
theorem one_point_45_deg_equals_1_deg_27_min :
  ∃ (deg min : ℕ), deg = 1 ∧ min = 27 ∧ 1.45 = deg + (min : ℝ) / 60 :=
by
  sorry

end one_point_45_deg_equals_1_deg_27_min_l3927_392725


namespace equation_solutions_l3927_392756

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧
    x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 5/3 ∧
    3*(x₁-1)^2 = 2*(x₁-1) ∧ 3*(x₂-1)^2 = 2*(x₂-1)) :=
by sorry

end equation_solutions_l3927_392756


namespace quadratic_divisibility_l3927_392786

theorem quadratic_divisibility (p : ℕ) (a b c : ℕ) (h_prime : Nat.Prime p) 
  (h_a : 0 < a ∧ a ≤ p) (h_b : 0 < b ∧ b ≤ p) (h_c : 0 < c ∧ c ≤ p)
  (h_div : ∀ (x : ℕ), x > 0 → (p ∣ (a * x^2 + b * x + c))) :
  a + b + c = 3 * p := by
sorry

end quadratic_divisibility_l3927_392786


namespace negation_of_universal_proposition_l3927_392736

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 1) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 1) := by sorry

end negation_of_universal_proposition_l3927_392736


namespace quadratic_inequality_solution_implies_product_l3927_392713

theorem quadratic_inequality_solution_implies_product (a b : ℝ) :
  (∀ x : ℝ, ax^2 + bx + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a * b = 6 := by
  sorry

end quadratic_inequality_solution_implies_product_l3927_392713


namespace cube_surface_area_l3927_392758

theorem cube_surface_area (volume : ℝ) (surface_area : ℝ) : 
  volume = 343 → surface_area = 294 → 
  (∃ (side : ℝ), volume = side^3 ∧ surface_area = 6 * side^2) := by
  sorry

end cube_surface_area_l3927_392758


namespace pirate_rick_sand_ratio_l3927_392716

/-- Pirate Rick's treasure digging problem -/
theorem pirate_rick_sand_ratio :
  let initial_sand : ℝ := 8
  let initial_time : ℝ := 4
  let tsunami_sand : ℝ := 2
  let final_time : ℝ := 3
  let digging_rate : ℝ := initial_sand / initial_time
  let final_sand : ℝ := final_time * digging_rate
  let storm_sand : ℝ := initial_sand + tsunami_sand - final_sand
  storm_sand / initial_sand = 1 / 2 := by
sorry

end pirate_rick_sand_ratio_l3927_392716


namespace sara_letters_total_l3927_392783

/-- The number of letters Sara sent in January -/
def january_letters : ℕ := 6

/-- The number of letters Sara sent in February -/
def february_letters : ℕ := 9

/-- The number of letters Sara sent in March -/
def march_letters : ℕ := 3 * january_letters

/-- The total number of letters Sara sent -/
def total_letters : ℕ := january_letters + february_letters + march_letters

theorem sara_letters_total : total_letters = 33 := by
  sorry

end sara_letters_total_l3927_392783


namespace potato_count_l3927_392774

/-- Given the initial number of potatoes and the number of new potatoes left after rabbits ate some,
    prove that the total number of potatoes is equal to the sum of the initial number and the number of new potatoes left. -/
theorem potato_count (initial : ℕ) (new_left : ℕ) : 
  initial + new_left = initial + new_left :=
by sorry

end potato_count_l3927_392774


namespace simplify_complex_expression_l3927_392742

theorem simplify_complex_expression (a : ℝ) (h : a > 0) :
  Real.sqrt ((2 * a) / ((1 + a) * (1 + a) ^ (1/3))) *
  ((4 + 8 / a + 4 / a^2) / Real.sqrt 2) ^ (1/3) =
  (2 * a^(5/6)) / a := by
  sorry

end simplify_complex_expression_l3927_392742


namespace chimney_bricks_chimney_bricks_proof_l3927_392779

theorem chimney_bricks : ℕ → Prop :=
  fun n =>
    let brenda_rate := n / 12
    let brandon_rate := n / 15
    let combined_rate := n / 12 + n / 15 - 15
    6 * combined_rate = n →
    n = 900

-- The proof is omitted
theorem chimney_bricks_proof : chimney_bricks 900 := by sorry

end chimney_bricks_chimney_bricks_proof_l3927_392779


namespace jean_speed_is_45_over_46_l3927_392753

/-- Represents the hiking scenario with Chantal and Jean --/
structure HikingScenario where
  speed_first_third : ℝ
  speed_uphill : ℝ
  break_time : ℝ
  speed_downhill : ℝ
  meeting_point : ℝ

/-- Calculates Jean's average speed given a hiking scenario --/
def jeanAverageSpeed (scenario : HikingScenario) : ℝ :=
  sorry

/-- Theorem stating that Jean's average speed is 45/46 miles per hour --/
theorem jean_speed_is_45_over_46 (scenario : HikingScenario) :
  scenario.speed_first_third = 5 ∧
  scenario.speed_uphill = 3 ∧
  scenario.break_time = 1/6 ∧
  scenario.speed_downhill = 4 ∧
  scenario.meeting_point = 3/2 →
  jeanAverageSpeed scenario = 45/46 :=
sorry

end jean_speed_is_45_over_46_l3927_392753


namespace cylinder_height_in_hemisphere_l3927_392795

/-- The height of a right circular cylinder inscribed in a hemisphere --/
theorem cylinder_height_in_hemisphere (r_cylinder r_hemisphere : ℝ) 
  (h_cylinder : r_cylinder = 3)
  (h_hemisphere : r_hemisphere = 7)
  (h_inscribed : r_cylinder ≤ r_hemisphere) :
  Real.sqrt (r_hemisphere^2 - r_cylinder^2) = 2 * Real.sqrt 10 :=
by sorry

end cylinder_height_in_hemisphere_l3927_392795


namespace prob_not_sold_is_one_fifth_expected_profit_four_batches_l3927_392770

-- Define the probability of not passing each round
def prob_fail_first : ℚ := 1 / 9
def prob_fail_second : ℚ := 1 / 10

-- Define the profit/loss values
def profit_if_sold : ℤ := 400
def loss_if_not_sold : ℤ := 800

-- Define the number of batches
def num_batches : ℕ := 4

-- Define the probability of a batch being sold
def prob_sold : ℚ := (1 - prob_fail_first) * (1 - prob_fail_second)

-- Define the probability of a batch not being sold
def prob_not_sold : ℚ := 1 - prob_sold

-- Define the expected profit for a single batch
def expected_profit_single : ℚ := prob_sold * profit_if_sold - prob_not_sold * loss_if_not_sold

-- Theorem: Probability of a batch not being sold is 1/5
theorem prob_not_sold_is_one_fifth : prob_not_sold = 1 / 5 := by sorry

-- Theorem: Expected profit from 4 batches is 640 yuan
theorem expected_profit_four_batches : num_batches * expected_profit_single = 640 := by sorry

end prob_not_sold_is_one_fifth_expected_profit_four_batches_l3927_392770


namespace sum_of_ages_l3927_392750

/-- 
Given that Tom is 15 years old now and in 3 years he will be twice Tim's age,
prove that the sum of their current ages is 21 years.
-/
theorem sum_of_ages : 
  ∀ (tim_age : ℕ), 
  (15 + 3 = 2 * (tim_age + 3)) →
  (15 + tim_age = 21) := by
sorry

end sum_of_ages_l3927_392750


namespace expression_equality_l3927_392790

theorem expression_equality : 10 * 0.2 * 5 * 0.1 + 5 = 6 := by
  sorry

end expression_equality_l3927_392790


namespace triangle_inequality_on_side_l3927_392706

/-- Given a triangle ABC and a point O on side AB (not coinciding with A or B),
    prove that OC · AB < OA · BC + OB · AC. -/
theorem triangle_inequality_on_side (A B C O : EuclideanSpace ℝ (Fin 2)) :
  O ≠ A →
  O ≠ B →
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ O = A + t • (B - A) →
  ‖C - O‖ * ‖B - A‖ < ‖O - A‖ * ‖C - B‖ + ‖O - B‖ * ‖C - A‖ := by
  sorry

end triangle_inequality_on_side_l3927_392706


namespace period_1989_points_count_l3927_392778

-- Define the unit circle
def UnitCircle : Set ℂ := {z : ℂ | Complex.abs z = 1}

-- Define the function f
def f (m : ℕ) (z : ℂ) : ℂ := z ^ m

-- Define the set of period n points
def PeriodPoints (m : ℕ) (n : ℕ) : Set ℂ :=
  {z ∈ UnitCircle | (f m)^[n] z = z ∧ ∀ k < n, (f m)^[k] z ≠ z}

-- Theorem statement
theorem period_1989_points_count (m : ℕ) (h : m > 1) :
  (PeriodPoints m 1989).ncard = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 := by
  sorry

end period_1989_points_count_l3927_392778


namespace digit_move_correction_l3927_392711

theorem digit_move_correction : ∃ (a b c : ℕ), 
  (a = 101 ∧ b = 102 ∧ c = 1) ∧ 
  (a - b ≠ c) ∧
  (a - 10^2 = c) := by
  sorry

end digit_move_correction_l3927_392711


namespace vector_BA_complex_l3927_392787

/-- Given two complex numbers representing vectors OA and OB, 
    prove that the complex number representing vector BA is their difference. -/
theorem vector_BA_complex (OA OB : ℂ) (h1 : OA = 2 - 3*I) (h2 : OB = -3 + 2*I) :
  OA - OB = 5 - 5*I := by
  sorry

end vector_BA_complex_l3927_392787


namespace percentage_relation_l3927_392764

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.06 * x) (h2 : b = 0.3 * x) :
  a = 0.2 * b := by
  sorry

end percentage_relation_l3927_392764


namespace units_digit_of_sum_l3927_392792

-- Define the function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_sum : unitsDigit ((56 ^ 78) + (87 ^ 65)) = 3 := by
  sorry

end units_digit_of_sum_l3927_392792


namespace equation_solution_l3927_392701

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by sorry

end equation_solution_l3927_392701


namespace banana_arrangements_count_l3927_392769

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of occurrences of 'A' in "BANANA" -/
def count_A : ℕ := 3

/-- The number of occurrences of 'N' in "BANANA" -/
def count_N : ℕ := 2

/-- The number of occurrences of 'B' in "BANANA" -/
def count_B : ℕ := 1

/-- Theorem stating that the number of distinct arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangements_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial count_A) * (Nat.factorial count_N)) :=
by sorry

end banana_arrangements_count_l3927_392769


namespace sin_960_degrees_l3927_392721

theorem sin_960_degrees : Real.sin (960 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_960_degrees_l3927_392721


namespace cube_roots_of_specific_numbers_l3927_392741

theorem cube_roots_of_specific_numbers :
  (∃ x : ℕ, x^3 = 59319) ∧ (∃ y : ℕ, y^3 = 195112) :=
by
  have h1 : (10 : ℕ)^3 = 1000 := by norm_num
  have h2 : (100 : ℕ)^3 = 1000000 := by norm_num
  sorry

end cube_roots_of_specific_numbers_l3927_392741


namespace cistern_fill_time_l3927_392731

/-- If a cistern can be emptied by a tap in 10 hours, and when both this tap and another tap
    are opened simultaneously the cistern gets filled in 20/3 hours, then the time it takes
    for the other tap alone to fill the cistern is 4 hours. -/
theorem cistern_fill_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) :
  empty_rate = 10 →
  combined_fill_time = 20 / 3 →
  1 / fill_time - 1 / empty_rate = 1 / combined_fill_time →
  fill_time = 4 := by
  sorry

end cistern_fill_time_l3927_392731


namespace area_of_curve_l3927_392703

theorem area_of_curve (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 16 ∧ 
   A = Real.pi * (Real.sqrt ((x - 2)^2 + (y + 3)^2))^2 ∧
   x^2 + y^2 - 4*x + 6*y - 3 = 0) := by
sorry

end area_of_curve_l3927_392703


namespace expression_simplification_l3927_392717

theorem expression_simplification (w : ℝ) : 2*w + 4*w + 6*w + 8*w + 10*w + 12 = 30*w + 12 := by
  sorry

end expression_simplification_l3927_392717


namespace parabola_intersection_theorem_parabola_equation_l3927_392708

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- Check if two line segments are perpendicular -/
def perpendicular (a b c d : Point) : Prop :=
  (b.x - a.x) * (d.x - c.x) + (b.y - a.y) * (d.y - c.y) = 0

/-- Theorem: If a parabola y² = 2px intersects the line x = 2 at points D and E,
    and OD ⊥ OE where O is the origin, then p = 1 -/
theorem parabola_intersection_theorem (C : Parabola) (D E : Point) :
  D.x = 2 ∧ E.x = 2 ∧                        -- D and E are on the line x = 2
  D.y^2 = 2 * C.p * D.x ∧ E.y^2 = 2 * C.p * E.x ∧  -- D and E are on the parabola
  perpendicular origin D origin E            -- OD ⊥ OE
  → C.p = 1 := by sorry

/-- Corollary: Under the conditions of the theorem, the parabola's equation is y² = 2x -/
theorem parabola_equation (C : Parabola) (D E : Point) :
  D.x = 2 ∧ E.x = 2 ∧
  D.y^2 = 2 * C.p * D.x ∧ E.y^2 = 2 * C.p * E.x ∧
  perpendicular origin D origin E
  → ∀ x y : ℝ, y^2 = 2 * x ↔ y^2 = 2 * C.p * x := by sorry

end parabola_intersection_theorem_parabola_equation_l3927_392708


namespace cosine_sum_simplification_l3927_392760

theorem cosine_sum_simplification :
  Real.cos (2 * Real.pi / 15) + Real.cos (4 * Real.pi / 15) + 
  Real.cos (10 * Real.pi / 15) + Real.cos (14 * Real.pi / 15) = 
  (Real.sqrt 17 - 1) / 4 :=
by sorry

end cosine_sum_simplification_l3927_392760


namespace cos_sin_relation_l3927_392740

theorem cos_sin_relation (α : ℝ) (h : Real.cos (α - π/5) = 5/13) :
  Real.sin (α - 7*π/10) = -5/13 := by sorry

end cos_sin_relation_l3927_392740


namespace second_grade_sample_l3927_392762

/-- Represents the number of students to be sampled from a stratum in stratified sampling -/
def stratifiedSample (totalSample : ℕ) (stratumWeight : ℕ) (totalWeight : ℕ) : ℕ :=
  (stratumWeight * totalSample) / totalWeight

/-- Theorem: In a school with grades in 3:3:4 ratio, stratified sampling of 50 students
    results in 15 students from the second grade -/
theorem second_grade_sample :
  let totalSample : ℕ := 50
  let firstGradeWeight : ℕ := 3
  let secondGradeWeight : ℕ := 3
  let thirdGradeWeight : ℕ := 4
  let totalWeight : ℕ := firstGradeWeight + secondGradeWeight + thirdGradeWeight
  stratifiedSample totalSample secondGradeWeight totalWeight = 15 := by
  sorry

#eval stratifiedSample 50 3 10  -- Expected output: 15

end second_grade_sample_l3927_392762


namespace modular_inverse_of_3_mod_29_l3927_392780

theorem modular_inverse_of_3_mod_29 : ∃ x : ℕ, x < 29 ∧ (3 * x) % 29 = 1 :=
by
  -- Proof goes here
  sorry

end modular_inverse_of_3_mod_29_l3927_392780


namespace dihedral_angle_range_l3927_392744

/-- The dihedral angle between two adjacent faces in a regular n-sided prism -/
def dihedral_angle (n : ℕ) (θ : ℝ) : Prop :=
  n ≥ 3 ∧ ((n - 2 : ℝ) / n * Real.pi < θ ∧ θ < Real.pi)

/-- Theorem: The dihedral angle in a regular n-sided prism is within the specified range -/
theorem dihedral_angle_range (n : ℕ) :
  ∃ θ : ℝ, dihedral_angle n θ :=
sorry

end dihedral_angle_range_l3927_392744


namespace car_distance_theorem_l3927_392755

/-- Calculates the total distance traveled by a car over a given number of hours,
    where the car's speed increases by a fixed amount each hour. -/
def total_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc h => acc + (initial_speed + h * speed_increase)) 0

/-- Theorem stating that a car traveling 40 km in the first hour and increasing speed by 2 km/h
    every hour will travel 600 km in 12 hours. -/
theorem car_distance_theorem : total_distance 40 2 12 = 600 := by
  sorry

end car_distance_theorem_l3927_392755


namespace flute_cost_calculation_l3927_392794

/-- The cost of Jason's purchases at the music store -/
def total_spent : ℝ := 158.35

/-- The cost of the music tool -/
def music_tool_cost : ℝ := 8.89

/-- The cost of the song book -/
def song_book_cost : ℝ := 7

/-- The cost of the flute -/
def flute_cost : ℝ := total_spent - (music_tool_cost + song_book_cost)

theorem flute_cost_calculation : flute_cost = 142.46 := by
  sorry

end flute_cost_calculation_l3927_392794


namespace course_selection_problem_l3927_392702

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of ways two people can choose 2 courses each from 4 courses -/
def totalWays : ℕ :=
  choose 4 2 * choose 4 2

/-- The number of ways two people can choose 2 courses each from 4 courses with at least one course in common -/
def waysWithCommon : ℕ :=
  totalWays - choose 4 2

theorem course_selection_problem :
  (totalWays = 36) ∧
  (waysWithCommon / totalWays = 5 / 6) := by sorry

end course_selection_problem_l3927_392702


namespace partial_fraction_decomposition_l3927_392704

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 7/3 ∧ B = 5/3 ∧
  ∀ (x : ℝ), x ≠ 6 ∧ x ≠ -3 →
    (4*x - 3) / (x^2 - 3*x - 18) = A / (x - 6) + B / (x + 3) := by
  sorry

end partial_fraction_decomposition_l3927_392704
