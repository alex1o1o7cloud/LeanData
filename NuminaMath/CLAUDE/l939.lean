import Mathlib

namespace decagon_diagonals_l939_93956

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l939_93956


namespace number_of_combinations_is_736_l939_93906

/-- Represents the number of different ways to occupy planets given the specified conditions --/
def number_of_combinations : ℕ :=
  let earth_like_planets : ℕ := 7
  let mars_like_planets : ℕ := 6
  let earth_like_units : ℕ := 3
  let mars_like_units : ℕ := 1
  let total_units : ℕ := 15

  -- The actual calculation of combinations
  0 -- placeholder, replace with actual calculation

/-- Theorem stating that the number of combinations is 736 --/
theorem number_of_combinations_is_736 : number_of_combinations = 736 := by
  sorry


end number_of_combinations_is_736_l939_93906


namespace max_sum_of_solutions_l939_93983

def is_solution (x y : ℤ) : Prop := x^2 + y^2 = 100

theorem max_sum_of_solutions :
  ∃ (a b : ℤ), is_solution a b ∧ 
  (∀ (x y : ℤ), is_solution x y → x + y ≤ a + b) ∧
  a + b = 14 := by sorry

end max_sum_of_solutions_l939_93983


namespace remainder_theorem_l939_93968

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) (hv : v < y) (hxdiv : x = u * y + v) :
  (x + 3 * u * y + 4) % y = (v + 4) % y :=
sorry

end remainder_theorem_l939_93968


namespace geometric_sequence_sum_l939_93978

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 20) →
  (a 3 + a 4 = 40) →
  (a 5 + a 6 = 80) :=
by
  sorry

end geometric_sequence_sum_l939_93978


namespace cut_pyramid_volume_ratio_l939_93923

/-- Represents a pyramid cut by a plane parallel to its base -/
structure CutPyramid where
  lateralAreaRatio : ℚ  -- Ratio of lateral surface areas (small pyramid : frustum)
  volumeRatio : ℚ       -- Ratio of volumes (small pyramid : frustum)

/-- Theorem: If the lateral area ratio is 9:16, then the volume ratio is 27:98 -/
theorem cut_pyramid_volume_ratio (p : CutPyramid) 
  (h : p.lateralAreaRatio = 9 / 16) : p.volumeRatio = 27 / 98 := by
  sorry

end cut_pyramid_volume_ratio_l939_93923


namespace circumradius_inequality_circumradius_equality_condition_l939_93916

/-- Triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_R : 0 < R
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem about the relationship between side lengths and circumradius -/
theorem circumradius_inequality (t : Triangle) :
  t.R ≥ (t.a^2 + t.b^2) / (2 * Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2)) :=
sorry

/-- Condition for equality in the circumradius inequality -/
theorem circumradius_equality_condition (t : Triangle) :
  t.R = (t.a^2 + t.b^2) / (2 * Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2)) ↔
  t.a = t.b ∨ t.a^2 + t.b^2 = t.c^2 :=
sorry

end circumradius_inequality_circumradius_equality_condition_l939_93916


namespace show_attendance_l939_93944

theorem show_attendance (adult_price children_price total_receipts : ℚ)
  (h1 : adult_price = 5.5)
  (h2 : children_price = 2.5)
  (h3 : total_receipts = 1026) :
  ∃ (adults children : ℕ),
    adults = 2 * children ∧
    adult_price * adults + children_price * children = total_receipts ∧
    adults = 152 :=
by sorry

end show_attendance_l939_93944


namespace valid_arrangements_l939_93927

/-- The number of ways to arrange students in a classroom. -/
def arrange_students : ℕ :=
  let num_students : ℕ := 30
  let num_rows : ℕ := 5
  let num_cols : ℕ := 6
  let num_boys : ℕ := 15
  let num_girls : ℕ := 15
  2 * (Nat.factorial num_boys) * (Nat.factorial num_girls)

/-- Theorem stating the number of valid arrangements of students. -/
theorem valid_arrangements (num_students num_rows num_cols num_boys num_girls : ℕ) 
  (h1 : num_students = 30)
  (h2 : num_rows = 5)
  (h3 : num_cols = 6)
  (h4 : num_boys = 15)
  (h5 : num_girls = 15)
  (h6 : num_students = num_boys + num_girls)
  (h7 : num_students = num_rows * num_cols) :
  arrange_students = 2 * (Nat.factorial num_boys) * (Nat.factorial num_girls) :=
by
  sorry

#eval arrange_students

end valid_arrangements_l939_93927


namespace root_equation_problem_l939_93922

theorem root_equation_problem (a b : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ((x + a) * (x + b) * (x + 10) = 0 ∧ x + 2 ≠ 0) ∧
    ((y + a) * (y + b) * (y + 10) = 0 ∧ y + 2 ≠ 0) ∧
    ((z + a) * (z + b) * (z + 10) = 0 ∧ z + 2 ≠ 0)) →
  (∃! w : ℝ, (w + 2*a) * (w + 4) * (w + 8) = 0 ∧ 
    (w + b) * (w + 10) ≠ 0) →
  100 * a + b = 208 :=
by sorry

end root_equation_problem_l939_93922


namespace student_selection_methods_l939_93912

theorem student_selection_methods (first_year second_year third_year : ℕ) 
  (h1 : first_year = 3) 
  (h2 : second_year = 5) 
  (h3 : third_year = 4) : 
  first_year + second_year + third_year = 12 := by
  sorry

end student_selection_methods_l939_93912


namespace solve_equation_l939_93991

/-- Given the equation fp - w = 20000, where f = 10 and w = 10 + 250i, prove that p = 2001 + 25i -/
theorem solve_equation (f w p : ℂ) : 
  f = 10 → w = 10 + 250 * I → f * p - w = 20000 → p = 2001 + 25 * I := by
  sorry

end solve_equation_l939_93991


namespace midpoint_coordinate_sum_l939_93995

/-- Given that (10, -6) is the midpoint of a line segment with one endpoint at (12, 4),
    prove that the sum of coordinates of the other endpoint is -8. -/
theorem midpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (10 : ℝ) = (x + 12) / 2 →
  (-6 : ℝ) = (y + 4) / 2 →
  x + y = -8 := by
sorry

end midpoint_coordinate_sum_l939_93995


namespace markup_is_100_percent_l939_93928

/-- Calculates the markup percentage given wholesale price, initial price, and price increase. -/
def markup_percentage (wholesale_price initial_price price_increase : ℚ) : ℚ :=
  let new_price := initial_price + price_increase
  (new_price - wholesale_price) / wholesale_price * 100

/-- Proves that the markup percentage is 100% given the specified conditions. -/
theorem markup_is_100_percent (wholesale_price initial_price price_increase : ℚ) 
  (h1 : wholesale_price = 20)
  (h2 : initial_price = 34)
  (h3 : price_increase = 6) :
  markup_percentage wholesale_price initial_price price_increase = 100 := by
  sorry

#eval markup_percentage 20 34 6

end markup_is_100_percent_l939_93928


namespace quadratic_even_iff_m_eq_neg_two_l939_93966

/-- A quadratic function f(x) = mx^2 + (m+2)mx + 2 is even if and only if m = -2 -/
theorem quadratic_even_iff_m_eq_neg_two (m : ℝ) :
  (∀ x : ℝ, m * x^2 + (m + 2) * m * x + 2 = m * (-x)^2 + (m + 2) * m * (-x) + 2) ↔ m = -2 :=
by sorry

end quadratic_even_iff_m_eq_neg_two_l939_93966


namespace fraction_equals_zero_l939_93979

theorem fraction_equals_zero (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := by
  sorry

end fraction_equals_zero_l939_93979


namespace sum_inequality_l939_93903

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) :
  a + c > b + d := by
  sorry

end sum_inequality_l939_93903


namespace quadratic_equation_root_value_l939_93988

theorem quadratic_equation_root_value (a b : ℝ) : 
  (∀ x, a * x^2 + b * x = 6) → -- The quadratic equation
  (a * 2^2 + b * 2 = 6) →     -- x = 2 is a root
  4 * a + 2 * b = 6 :=        -- The value of 4a + 2b
by sorry

end quadratic_equation_root_value_l939_93988


namespace largest_among_decimals_l939_93981

theorem largest_among_decimals :
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end largest_among_decimals_l939_93981


namespace function_and_extrema_l939_93976

noncomputable def f (a b c x : ℝ) : ℝ := a * x - b / x + c

theorem function_and_extrema :
  ∀ a b c : ℝ,
  (f a b c 1 = 0) →
  (∀ x : ℝ, x ≠ 0 → HasDerivAt (f a b c) (-x + 3) 2) →
  (∀ x : ℝ, x ≠ 0 → f a b c x = -3 * x - 8 / x + 11) ∧
  (∃ x : ℝ, f a b c x = 11 + 4 * Real.sqrt 6 ∧ IsLocalMin (f a b c) x) ∧
  (∃ x : ℝ, f a b c x = 11 - 4 * Real.sqrt 6 ∧ IsLocalMax (f a b c) x) :=
by sorry

end function_and_extrema_l939_93976


namespace stating_min_handshakes_in_gathering_l939_93940

/-- Represents a gathering of people and their handshakes. -/
structure Gathering where
  people : ℕ
  min_handshakes_per_person : ℕ
  non_handshaking_group : ℕ
  total_handshakes : ℕ

/-- The specific gathering described in the problem. -/
def problem_gathering : Gathering where
  people := 25
  min_handshakes_per_person := 2
  non_handshaking_group := 3
  total_handshakes := 28

/-- 
Theorem stating that the minimum number of handshakes in the given gathering is 28.
-/
theorem min_handshakes_in_gathering (g : Gathering) 
  (h1 : g.people = 25)
  (h2 : g.min_handshakes_per_person = 2)
  (h3 : g.non_handshaking_group = 3) :
  g.total_handshakes = 28 := by
  sorry

#check min_handshakes_in_gathering

end stating_min_handshakes_in_gathering_l939_93940


namespace all_positive_integers_l939_93997

def is_valid_set (A : Set ℕ) : Prop :=
  1 ∈ A ∧
  ∃ k : ℕ, k ≠ 1 ∧ k ∈ A ∧
  ∀ m n : ℕ, m ∈ A → n ∈ A → m ≠ n →
    ((m + 1) / (Nat.gcd (m + 1) (n + 1))) ∈ A

theorem all_positive_integers (A : Set ℕ) :
  is_valid_set A → A = {n : ℕ | n > 0} :=
by sorry

end all_positive_integers_l939_93997


namespace trapezium_area_from_equilateral_triangles_l939_93919

theorem trapezium_area_from_equilateral_triangles 
  (triangle_area : ℝ) 
  (h : ℝ) -- height of small triangle
  (b : ℝ) -- base of small triangle
  (h_pos : h > 0)
  (b_pos : b > 0)
  (area_eq : (1/2) * b * h = triangle_area)
  (triangle_area_val : triangle_area = 4) :
  let trapezium_area := (1/2) * (4*h + 5*h) * (5/2*b)
  trapezium_area = 90 := by
sorry

end trapezium_area_from_equilateral_triangles_l939_93919


namespace slower_plane_speed_l939_93911

/-- Given two planes flying in opposite directions for 3 hours, where one plane's speed is twice 
    the other's, and they end up 2700 miles apart, prove that the slower plane's speed is 300 
    miles per hour. -/
theorem slower_plane_speed (slower_speed faster_speed : ℝ) 
    (h1 : faster_speed = 2 * slower_speed)
    (h2 : 3 * slower_speed + 3 * faster_speed = 2700) : 
  slower_speed = 300 := by
  sorry

end slower_plane_speed_l939_93911


namespace men_joined_correct_l939_93998

/-- The number of men who joined the camp -/
def men_joined : ℕ := 30

/-- The initial number of men in the camp -/
def initial_men : ℕ := 10

/-- The initial number of days the food would last -/
def initial_days : ℕ := 20

/-- The number of days the food lasts after more men join -/
def final_days : ℕ := 5

/-- The total amount of food in man-days -/
def total_food : ℕ := initial_men * initial_days

theorem men_joined_correct :
  (initial_men + men_joined) * final_days = total_food :=
by sorry

end men_joined_correct_l939_93998


namespace number_thought_of_l939_93994

theorem number_thought_of (x : ℝ) : (x / 5 + 10 = 21) → x = 55 := by
  sorry

end number_thought_of_l939_93994


namespace equation_one_solutions_l939_93959

theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 - 5 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
sorry

end equation_one_solutions_l939_93959


namespace coach_cost_l939_93960

/-- Proves that the cost of the coach before discount is $2500 given the problem conditions -/
theorem coach_cost (sectional_cost other_cost total_paid : ℝ) 
  (h1 : sectional_cost = 3500)
  (h2 : other_cost = 2000)
  (h3 : total_paid = 7200)
  (discount : ℝ) (h4 : discount = 0.1)
  : ∃ (coach_cost : ℝ), 
    coach_cost = 2500 ∧ 
    (1 - discount) * (coach_cost + sectional_cost + other_cost) = total_paid :=
by sorry

end coach_cost_l939_93960


namespace gcd_8251_6105_l939_93936

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by sorry

end gcd_8251_6105_l939_93936


namespace point_in_fourth_quadrant_l939_93985

theorem point_in_fourth_quadrant :
  ∀ x : ℝ, (x^2 + 2 > 0) ∧ (-3 < 0) := by
  sorry

end point_in_fourth_quadrant_l939_93985


namespace b_cubed_is_zero_l939_93937

theorem b_cubed_is_zero (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end b_cubed_is_zero_l939_93937


namespace unique_solution_aabb_equation_l939_93961

theorem unique_solution_aabb_equation :
  ∃! (a b n : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1000 * a + 100 * a + 10 * b + b = n^4 - 6 * n^3 :=
by
  -- The proof goes here
  sorry

end unique_solution_aabb_equation_l939_93961


namespace sum_of_integers_5_to_20_l939_93901

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_of_integers_5_to_20 :
  sum_of_integers 5 20 = 200 :=
by sorry

end sum_of_integers_5_to_20_l939_93901


namespace triangle_perimeter_range_l939_93990

/-- Given a triangle with two sides of lengths that are roots of x^2 - 5x + 6 = 0,
    the perimeter l of the triangle satisfies 6 < l < 10 -/
theorem triangle_perimeter_range : ∀ a b c : ℝ,
  (a^2 - 5*a + 6 = 0) →
  (b^2 - 5*b + 6 = 0) →
  (a ≠ b) →
  (a + b > c) →
  (b + c > a) →
  (c + a > b) →
  let l := a + b + c
  6 < l ∧ l < 10 := by
sorry

end triangle_perimeter_range_l939_93990


namespace complex_number_in_third_quadrant_l939_93984

theorem complex_number_in_third_quadrant : 
  let i : ℂ := Complex.I
  let z : ℂ := i + 2 * i^2 + 3 * i^3
  (z.re < 0) ∧ (z.im < 0) := by sorry

end complex_number_in_third_quadrant_l939_93984


namespace lewis_harvest_earnings_l939_93952

/-- Calculates the total earnings during harvest season --/
def harvest_earnings (regular_weekly : ℕ) (overtime_weekly : ℕ) (weeks : ℕ) : ℕ :=
  (regular_weekly + overtime_weekly) * weeks

/-- Theorem stating Lewis's total earnings during harvest season --/
theorem lewis_harvest_earnings :
  harvest_earnings 28 939 1091 = 1055497 := by
  sorry

end lewis_harvest_earnings_l939_93952


namespace cost_increase_percentage_l939_93935

/-- Represents the initial ratio of costs for raw material, labor, and overheads -/
def initial_ratio : Fin 3 → ℚ
  | 0 => 4
  | 1 => 3
  | 2 => 2

/-- Represents the percentage changes in costs for raw material, labor, and overheads -/
def cost_changes : Fin 3 → ℚ
  | 0 => 110 / 100  -- 10% increase
  | 1 => 108 / 100  -- 8% increase
  | 2 => 95 / 100   -- 5% decrease

/-- Theorem stating that the overall percentage increase in cost is 6% -/
theorem cost_increase_percentage : 
  let initial_total := (Finset.sum Finset.univ initial_ratio)
  let new_total := (Finset.sum Finset.univ (λ i => initial_ratio i * cost_changes i))
  (new_total - initial_total) / initial_total * 100 = 6 := by
  sorry

end cost_increase_percentage_l939_93935


namespace farm_horses_and_cows_l939_93965

theorem farm_horses_and_cows (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 3 * initial_cows →
  (initial_horses - 15) * 3 = 5 * (initial_cows + 15) →
  initial_horses - 15 - (initial_cows + 15) = 30 := by
  sorry

end farm_horses_and_cows_l939_93965


namespace volleyball_club_boys_count_l939_93982

theorem volleyball_club_boys_count :
  ∀ (total_members boys girls present : ℕ),
  total_members = 30 →
  present = 18 →
  boys + girls = total_members →
  present = boys + girls / 3 →
  boys = 12 :=
by
  sorry

end volleyball_club_boys_count_l939_93982


namespace bus_seating_capacity_l939_93908

theorem bus_seating_capacity 
  (total_students : ℕ) 
  (bus_capacity : ℕ) 
  (h1 : 4 * bus_capacity + 30 = total_students) 
  (h2 : 5 * bus_capacity = total_students + 10) : 
  bus_capacity = 40 := by
sorry

end bus_seating_capacity_l939_93908


namespace millet_exceeds_half_on_wednesday_l939_93962

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  millet : Real
  otherSeeds : Real

/-- Calculates the next day's feeder state based on the current state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := 0.7 * state.millet + 0.2,
    otherSeeds := 0.1 * state.otherSeeds + 0.3 }

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1, millet := 0.2, otherSeeds := 0.3 }

/-- Theorem stating that on Wednesday, the proportion of millet exceeds half of the total seeds -/
theorem millet_exceeds_half_on_wednesday :
  let wednesdayState := nextDay (nextDay initialState)
  wednesdayState.millet > (wednesdayState.millet + wednesdayState.otherSeeds) / 2 :=
by sorry


end millet_exceeds_half_on_wednesday_l939_93962


namespace S_is_circle_l939_93975

-- Define the set of complex numbers satisfying the condition
def S : Set ℂ := {z : ℂ | Complex.abs (z - Complex.I) = Complex.abs (3 + 4 * Complex.I)}

-- Theorem stating that S is a circle
theorem S_is_circle : ∃ (c : ℂ) (r : ℝ), S = {z : ℂ | Complex.abs (z - c) = r} :=
sorry

end S_is_circle_l939_93975


namespace butterflies_in_garden_l939_93909

/-- The number of butterflies left in the garden after some fly away -/
def butterflies_left (initial : ℕ) : ℕ :=
  initial - initial / 3

/-- Theorem stating that for 9 initial butterflies, 6 are left after one-third fly away -/
theorem butterflies_in_garden : butterflies_left 9 = 6 := by
  sorry

end butterflies_in_garden_l939_93909


namespace cube_sum_problem_l939_93917

theorem cube_sum_problem (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (sum_prod_eq : x * y + y * z + z * x = -6)
  (prod_eq : x * y * z = -6) :
  x^3 + y^3 + z^3 = 25 := by sorry

end cube_sum_problem_l939_93917


namespace red_subset_existence_l939_93910

theorem red_subset_existence (n k m : ℕ) (X : Finset ℕ) 
  (red_subsets : Finset (Finset ℕ)) :
  n > 0 → k > 0 → k < n →
  Finset.card X = n →
  (∀ A ∈ red_subsets, Finset.card A = k ∧ A ⊆ X) →
  Finset.card red_subsets = m →
  m > ((k - 1) * (n - k) + k) / (k^2 : ℚ) * (Nat.choose n (k - 1)) →
  ∃ Y : Finset ℕ, Y ⊆ X ∧ Finset.card Y = k + 1 ∧
    ∀ Z : Finset ℕ, Z ⊆ Y → Finset.card Z = k → Z ∈ red_subsets :=
by sorry


end red_subset_existence_l939_93910


namespace pasture_rent_problem_l939_93996

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ

/-- Calculates the total ox-months for a given rent share -/
def oxMonths (share : RentShare) : ℕ := share.oxen * share.months

/-- The problem statement -/
theorem pasture_rent_problem (a b c : RentShare) (c_rent : ℚ) 
  (h1 : a.oxen = 10 ∧ a.months = 7)
  (h2 : b.oxen = 12 ∧ b.months = 5)
  (h3 : c.oxen = 15 ∧ c.months = 3)
  (h4 : c_rent = 53.99999999999999)
  : ∃ (total_rent : ℚ), total_rent = 210 := by
  sorry

#check pasture_rent_problem

end pasture_rent_problem_l939_93996


namespace cube_volume_from_space_diagonal_l939_93953

theorem cube_volume_from_space_diagonal :
  ∀ s : ℝ,
  s > 0 →
  s * Real.sqrt 3 = 10 * Real.sqrt 3 →
  s^3 = 1000 :=
by
  sorry

end cube_volume_from_space_diagonal_l939_93953


namespace gas_pressure_volume_relationship_l939_93932

/-- Given inverse proportionality of pressure and volume at constant temperature,
    calculate the new pressure when the volume changes. -/
theorem gas_pressure_volume_relationship
  (initial_volume initial_pressure new_volume : ℝ)
  (h_positive : initial_volume > 0 ∧ initial_pressure > 0 ∧ new_volume > 0)
  (h_inverse_prop : ∀ (v p : ℝ), v > 0 → p > 0 → v * p = initial_volume * initial_pressure) :
  let new_pressure := (initial_volume * initial_pressure) / new_volume
  new_pressure = 2 ∧ initial_volume = 2.28 ∧ initial_pressure = 5 ∧ new_volume = 5.7 := by
sorry

end gas_pressure_volume_relationship_l939_93932


namespace sum_difference_equals_eight_ninths_l939_93945

open BigOperators

-- Define the harmonic series
def harmonic_sum (n : ℕ) : ℚ := ∑ y in Finset.range n, (1 : ℚ) / (y + 1)

-- State the theorem
theorem sum_difference_equals_eight_ninths :
  (∑ y in Finset.range 8, (1 : ℚ) / (y + 1)) - (∑ y in Finset.range 8, (1 : ℚ) / (y + 2)) = 8 / 9 := by
  sorry

end sum_difference_equals_eight_ninths_l939_93945


namespace cricket_average_theorem_l939_93971

/-- Represents a cricket player's batting statistics -/
structure CricketStats where
  matches_played : ℕ
  total_runs : ℕ
  
/-- Calculates the batting average -/
def batting_average (stats : CricketStats) : ℚ :=
  stats.total_runs / stats.matches_played

/-- Theorem: If a player has played 5 matches and scoring 69 runs in the next match
    would bring their batting average to 54, then their current batting average is 51 -/
theorem cricket_average_theorem (stats : CricketStats) 
    (h1 : stats.matches_played = 5)
    (h2 : batting_average ⟨stats.matches_played + 1, stats.total_runs + 69⟩ = 54) :
  batting_average stats = 51 := by
  sorry

end cricket_average_theorem_l939_93971


namespace largest_solution_and_ratio_l939_93918

theorem largest_solution_and_ratio : ∃ (a b c d : ℤ),
  let x : ℝ := (a + b * Real.sqrt c) / d
  ∀ y : ℝ, (6 * y / 5 - 2 = 4 / y) → y ≤ x ∧
  x = (5 + Real.sqrt 145) / 6 ∧
  a * c * d / b = 4350 :=
by sorry

end largest_solution_and_ratio_l939_93918


namespace wheat_distribution_theorem_l939_93941

def wheat_distribution (x y z : ℕ) : Prop :=
  x + y + z = 100 ∧ 3 * x + 2 * y + (1/2) * z = 100

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(20,0,80), (17,5,78), (14,10,76), (11,15,74), (8,20,72), (5,25,70), (2,30,68)}

theorem wheat_distribution_theorem :
  {p : ℕ × ℕ × ℕ | wheat_distribution p.1 p.2.1 p.2.2} = solution_set :=
sorry

end wheat_distribution_theorem_l939_93941


namespace expenditure_for_specific_hall_l939_93938

/-- Calculates the total expenditure for covering a rectangular floor with a mat. -/
def total_expenditure (length width cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Proves that the total expenditure for covering a specific rectangular floor is 3000. -/
theorem expenditure_for_specific_hall : 
  total_expenditure 20 15 10 = 3000 := by
  sorry

end expenditure_for_specific_hall_l939_93938


namespace store_a_more_cost_effective_l939_93905

/-- Represents the cost of purchasing tennis equipment from two different stores -/
def tennis_purchase_cost (x : ℝ) : Prop :=
  x > 40 ∧ 
  (25 * x + 3000 < 22.5 * x + 3600) = (x > 120)

/-- Theorem stating that Store A is more cost-effective when x = 100 -/
theorem store_a_more_cost_effective : tennis_purchase_cost 100 := by
  sorry

end store_a_more_cost_effective_l939_93905


namespace union_of_sets_l939_93926

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {0, 1, a}
def B (a : ℝ) : Set ℝ := {0, 3, 3*a}

-- Theorem statement
theorem union_of_sets (a : ℝ) (h : A a ∩ B a = {0, 3}) : 
  A a ∪ B a = {0, 1, 3, 9} := by
sorry

end union_of_sets_l939_93926


namespace fraction_sum_equality_l939_93946

theorem fraction_sum_equality : (18 : ℚ) / 45 - 3 / 8 + 1 / 9 = 49 / 360 := by sorry

end fraction_sum_equality_l939_93946


namespace expected_unpoked_babies_l939_93948

/-- The number of babies in the circle -/
def num_babies : ℕ := 2006

/-- The probability of a baby poking either of its adjacent neighbors -/
def poke_prob : ℚ := 1/2

/-- The probability of a baby being unpoked -/
def unpoked_prob : ℚ := (1 - poke_prob) * (1 - poke_prob)

/-- The expected number of unpoked babies -/
def expected_unpoked : ℚ := num_babies * unpoked_prob

theorem expected_unpoked_babies :
  expected_unpoked = 1003/2 := by sorry

end expected_unpoked_babies_l939_93948


namespace square_of_97_l939_93958

theorem square_of_97 : 97 * 97 = 9409 := by
  sorry

end square_of_97_l939_93958


namespace inscribed_rectangle_circle_circumference_l939_93969

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle_circumference : ℝ),
    rectangle_width = 5 →
    rectangle_height = 12 →
    circle_circumference = π * Real.sqrt (rectangle_width^2 + rectangle_height^2) →
    circle_circumference = 13 * π :=
by
  sorry

end inscribed_rectangle_circle_circumference_l939_93969


namespace divisibility_implies_equality_l939_93973

theorem divisibility_implies_equality (a b : ℕ) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) →
  a = b := by
  sorry

end divisibility_implies_equality_l939_93973


namespace distance_to_focus_l939_93955

/-- A point on a parabola with a specific distance property -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_to_line : |x + 3| = 5

/-- The theorem stating the distance from the point to the focus -/
theorem distance_to_focus (P : ParabolaPoint) :
  Real.sqrt ((P.x - 1)^2 + P.y^2) = 3 := by
  sorry

end distance_to_focus_l939_93955


namespace modulus_of_z_l939_93920

-- Define the complex number z
def z : ℂ := 2 + Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_l939_93920


namespace min_value_quadratic_l939_93963

/-- The function f(x) = 3x^2 - 15x + 7 attains its minimum value when x = 5/2. -/
theorem min_value_quadratic (x : ℝ) :
  ∀ y : ℝ, 3 * x^2 - 15 * x + 7 ≤ 3 * y^2 - 15 * y + 7 ↔ x = 5/2 := by
  sorry

end min_value_quadratic_l939_93963


namespace parabola_intersection_length_l939_93999

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the line passing through the focus
def line_through_focus (p : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p = (focus.1 + t, focus.2 + t)

-- Theorem statement
theorem parabola_intersection_length 
  (A B : PointOnParabola) 
  (h_line_A : line_through_focus (A.x, A.y))
  (h_line_B : line_through_focus (B.x, B.y))
  (h_sum : A.x + B.x = 6) :
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 4 := by
  sorry

end parabola_intersection_length_l939_93999


namespace difference_smallest_three_largest_two_l939_93970

def smallest_three_digit_number : ℕ := 100
def largest_two_digit_number : ℕ := 99

theorem difference_smallest_three_largest_two : 
  smallest_three_digit_number - largest_two_digit_number = 1 := by
  sorry

end difference_smallest_three_largest_two_l939_93970


namespace product_of_sums_and_differences_l939_93977

theorem product_of_sums_and_differences : (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * (2 * Real.sqrt 5 - 5 * Real.sqrt 2) = -30 := by
  sorry

end product_of_sums_and_differences_l939_93977


namespace independence_test_relationship_l939_93964

-- Define the random variable K²
def K_squared : ℝ → ℝ := sorry

-- Define the probability of judging variables as related
def prob_related : ℝ → ℝ := sorry

-- Define the test of independence
def test_of_independence : (ℝ → ℝ) → (ℝ → ℝ) → Prop := sorry

-- Theorem statement
theorem independence_test_relationship :
  ∀ (x y : ℝ), x > y →
  test_of_independence K_squared prob_related →
  prob_related (K_squared x) < prob_related (K_squared y) :=
sorry

end independence_test_relationship_l939_93964


namespace remainder_of_product_product_remainder_l939_93939

theorem remainder_of_product (a b c : ℕ) : (a * b * c) % 12 = ((a % 12) * (b % 12) * (c % 12)) % 12 := by sorry

theorem product_remainder : (1625 * 1627 * 1629) % 12 = 3 := by
  have h1 : 1625 % 12 = 5 := by sorry
  have h2 : 1627 % 12 = 7 := by sorry
  have h3 : 1629 % 12 = 9 := by sorry
  have h4 : (5 * 7 * 9) % 12 = 3 := by sorry
  exact calc
    (1625 * 1627 * 1629) % 12 = ((1625 % 12) * (1627 % 12) * (1629 % 12)) % 12 := by apply remainder_of_product
    _ = (5 * 7 * 9) % 12 := by rw [h1, h2, h3]
    _ = 3 := by exact h4

end remainder_of_product_product_remainder_l939_93939


namespace least_addition_for_divisibility_problem_solution_l939_93933

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 10 ∧ (1056 + x) % 26 = 0 ∧ ∀ (y : ℕ), y < x → (1056 + y) % 26 ≠ 0 :=
sorry

end least_addition_for_divisibility_problem_solution_l939_93933


namespace area_depends_on_arc_length_l939_93904

-- Define the unit circle
def unitCircle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a point on the unit circle with positive coordinates
def PointOnCircle (p : ℝ × ℝ) : Prop :=
  p ∈ unitCircle ∧ p.1 > 0 ∧ p.2 > 0

-- Define the projection points
def X₁ (x : ℝ × ℝ) : ℝ × ℝ := (x.1, 0)
def X₂ (x : ℝ × ℝ) : ℝ × ℝ := (0, x.2)

-- Define the area of region XYY₁X₁
def areaXYY₁X₁ (x y : ℝ × ℝ) : ℝ := sorry

-- Define the area of region XYY₂X₂
def areaXYY₂X₂ (x y : ℝ × ℝ) : ℝ := sorry

-- Define the angle subtended by arc XY at the center
def arcAngle (x y : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem area_depends_on_arc_length (x y : ℝ × ℝ) 
  (hx : PointOnCircle x) (hy : PointOnCircle y) :
  areaXYY₁X₁ x y + areaXYY₂X₂ x y = arcAngle x y := by
  sorry

end area_depends_on_arc_length_l939_93904


namespace parabola_equation_l939_93942

/-- A parabola with focus on the line 3x - 4y - 12 = 0 has standard equation x^2 = 6y and directrix y = 3 -/
theorem parabola_equation (x y : ℝ) :
  (∃ (a b : ℝ), 3*a - 4*b - 12 = 0 ∧ (x - a)^2 + (y - b)^2 = (y - 3)^2) →
  x^2 = 6*y ∧ y = 3 := by sorry

end parabola_equation_l939_93942


namespace polygon_sides_l939_93934

theorem polygon_sides (S : ℕ) (h : S = 2160) : ∃ n : ℕ, n = 14 ∧ S = 180 * (n - 2) := by
  sorry

end polygon_sides_l939_93934


namespace matrix_multiplication_l939_93993

theorem matrix_multiplication (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec ![3, -2] = ![4, 1])
  (h2 : N.mulVec ![-4, 6] = ![-2, 0]) :
  N.mulVec ![7, 0] = ![14, 4.2] := by
  sorry

end matrix_multiplication_l939_93993


namespace quadratic_roots_relation_l939_93902

theorem quadratic_roots_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ s₁ s₂ : ℝ, s₁ + s₂ = -c ∧ s₁ * s₂ = a ∧
   3 * s₁ + 3 * s₂ = -a ∧ 9 * s₁ * s₂ = b) →
  b / c = 27 := by
sorry

end quadratic_roots_relation_l939_93902


namespace circle_equation_for_given_points_l939_93907

/-- Given two points A and B, this function returns the standard equation of the circle
    with AB as its diameter in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def circleEquationFromDiameter (A B : ℝ × ℝ) : ℝ × ℝ × ℝ := by sorry

/-- Theorem stating that for points A(1, -4) and B(-5, 4),
    the standard equation of the circle with AB as its diameter is (x + 2)^2 + y^2 = 25 -/
theorem circle_equation_for_given_points :
  let A : ℝ × ℝ := (1, -4)
  let B : ℝ × ℝ := (-5, 4)
  let (h, k, r) := circleEquationFromDiameter A B
  h = -2 ∧ k = 0 ∧ r^2 = 25 := by sorry

end circle_equation_for_given_points_l939_93907


namespace ellens_age_l939_93921

/-- Proves Ellen's age given Martha's age and the relationship between their ages -/
theorem ellens_age (martha_age : ℕ) (h : martha_age = 32) :
  ∃ (ellen_age : ℕ), martha_age = 2 * (ellen_age + 6) ∧ ellen_age = 10 := by
  sorry

end ellens_age_l939_93921


namespace prime_sum_inequality_l939_93980

theorem prime_sum_inequality (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r →
  (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r ≥ 1 →
  ({p, q, r} : Set ℕ) = {2, 3, 5} :=
sorry

end prime_sum_inequality_l939_93980


namespace parabola_translation_right_l939_93913

/-- Translates a parabola to the right by a given amount -/
def translate_parabola_right (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ :=
  λ x => f (x - h)

/-- The original parabola function -/
def original_parabola : ℝ → ℝ :=
  λ x => -x^2

theorem parabola_translation_right :
  translate_parabola_right original_parabola 1 = λ x => -(x - 1)^2 := by
  sorry

end parabola_translation_right_l939_93913


namespace charlie_same_color_probability_l939_93972

def total_marbles : ℕ := 10
def red_marbles : ℕ := 3
def green_marbles : ℕ := 3
def blue_marbles : ℕ := 4

def alice_draw : ℕ := 3
def bob_draw : ℕ := 3
def charlie_draw : ℕ := 4

theorem charlie_same_color_probability :
  let total_outcomes := (total_marbles.choose alice_draw) * ((total_marbles - alice_draw).choose bob_draw) * ((total_marbles - alice_draw - bob_draw).choose charlie_draw)
  let favorable_outcomes := 
    2 * (red_marbles.min green_marbles).choose 3 * (total_marbles - red_marbles - green_marbles).choose 1 +
    (blue_marbles.choose 3) * (total_marbles - blue_marbles).choose 1 +
    blue_marbles.choose 4
  (favorable_outcomes : ℚ) / total_outcomes = 13 / 1400 := by
  sorry

end charlie_same_color_probability_l939_93972


namespace square_difference_of_sum_and_diff_l939_93992

theorem square_difference_of_sum_and_diff (x y : ℕ) 
  (sum_eq : x + y = 70) 
  (diff_eq : x - y = 20) 
  (pos_x : x > 0) 
  (pos_y : y > 0) : 
  x^2 - y^2 = 1400 := by
sorry

end square_difference_of_sum_and_diff_l939_93992


namespace complement_of_P_l939_93950

def U := Set ℝ
def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_of_P : (Set.univ \ P) = {x | x < -1 ∨ x > 1} := by
  sorry

end complement_of_P_l939_93950


namespace min_value_theorem_l939_93986

theorem min_value_theorem (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + 2*n = 1) :
  (1 / (2*m)) + (1 / n) ≥ 9/2 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ m₀ + 2*n₀ = 1 ∧ (1 / (2*m₀)) + (1 / n₀) = 9/2 :=
sorry

end min_value_theorem_l939_93986


namespace valid_k_characterization_l939_93974

/-- A function f: ℤ → ℤ is nonlinear if there exist x, y ∈ ℤ such that 
    f(x + y) ≠ f(x) + f(y) or f(ax) ≠ af(x) for some a ∈ ℤ -/
def Nonlinear (f : ℤ → ℤ) : Prop :=
  ∃ x y : ℤ, f (x + y) ≠ f x + f y ∨ ∃ a : ℤ, f (a * x) ≠ a * f x

/-- The set of non-negative integer values of k for which there exists a nonlinear function
    f: ℤ → ℤ satisfying the given equation for all integers a, b, c with a + b + c = 0 -/
def ValidK : Set ℕ :=
  {k : ℕ | ∃ f : ℤ → ℤ, Nonlinear f ∧
    ∀ a b c : ℤ, a + b + c = 0 →
      f a + f b + f c = (f (a - b) + f (b - c) + f (c - a)) / k}

theorem valid_k_characterization : ValidK = {0, 1, 3, 9} := by
  sorry

end valid_k_characterization_l939_93974


namespace percentage_sum_l939_93924

theorem percentage_sum : 
  (20 / 100 * 30) + (15 / 100 * 50) + (25 / 100 * 120) + (-10 / 100 * 45) = 39 := by
  sorry

end percentage_sum_l939_93924


namespace sqrt_difference_equals_negative_six_sqrt_two_l939_93947

theorem sqrt_difference_equals_negative_six_sqrt_two :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) - Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = -6 * Real.sqrt 2 := by
  sorry

end sqrt_difference_equals_negative_six_sqrt_two_l939_93947


namespace eleventh_term_of_arithmetic_sequence_l939_93914

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 11th term of an arithmetic sequence is the average of its 5th and 17th terms. -/
theorem eleventh_term_of_arithmetic_sequence
  (a : ℕ → ℚ) (h : is_arithmetic_sequence a)
  (h5 : a 5 = 3/8) (h17 : a 17 = 7/12) :
  a 11 = 23/48 := by
sorry

end eleventh_term_of_arithmetic_sequence_l939_93914


namespace tillys_star_ratio_l939_93900

/-- Proves that given the conditions of Tilly's star counting, the ratio of stars to the west to stars to the east is 6:1 -/
theorem tillys_star_ratio :
  ∀ (stars_east stars_west : ℕ),
    stars_east = 120 →
    (∃ k : ℕ, stars_west = k * stars_east) →
    stars_east + stars_west = 840 →
    stars_west / stars_east = 6 := by
  sorry

end tillys_star_ratio_l939_93900


namespace linear_function_proof_l939_93930

/-- A linear function of the form y = kx - 3 -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - 3

/-- The k value for which the linear function passes through (1, 7) -/
def k : ℝ := 10

theorem linear_function_proof :
  (linear_function k 1 = 7) ∧
  (linear_function k 2 ≠ 15) := by
  sorry

#check linear_function_proof

end linear_function_proof_l939_93930


namespace probability_of_red_ball_l939_93949

/-- The probability of drawing a red ball from a bag containing 2 yellow balls and 3 red balls -/
theorem probability_of_red_ball (yellow_balls red_balls : ℕ) 
  (h1 : yellow_balls = 2)
  (h2 : red_balls = 3) :
  (red_balls : ℚ) / ((yellow_balls + red_balls) : ℚ) = 3 / 5 := by
  sorry

#check probability_of_red_ball

end probability_of_red_ball_l939_93949


namespace impossible_table_l939_93925

/-- Represents a 7x7 table of natural numbers -/
def Table := Fin 7 → Fin 7 → ℕ

/-- Checks if the sum of numbers in a 2x2 square starting at (i, j) is odd -/
def is_2x2_sum_odd (t : Table) (i j : Fin 7) : Prop :=
  Odd (t i j + t i (j+1) + t (i+1) j + t (i+1) (j+1))

/-- Checks if the sum of numbers in a 3x3 square starting at (i, j) is odd -/
def is_3x3_sum_odd (t : Table) (i j : Fin 7) : Prop :=
  Odd (t i j + t i (j+1) + t i (j+2) +
       t (i+1) j + t (i+1) (j+1) + t (i+1) (j+2) +
       t (i+2) j + t (i+2) (j+1) + t (i+2) (j+2))

/-- The main theorem stating that it's impossible to construct a table satisfying the conditions -/
theorem impossible_table : ¬ ∃ (t : Table), 
  (∀ (i j : Fin 7), i < 6 ∧ j < 6 → is_2x2_sum_odd t i j) ∧ 
  (∀ (i j : Fin 7), i < 5 ∧ j < 5 → is_3x3_sum_odd t i j) :=
sorry

end impossible_table_l939_93925


namespace fraction_equality_l939_93943

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 25)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 1 / 3 := by sorry

end fraction_equality_l939_93943


namespace integral_reciprocal_plus_one_l939_93967

theorem integral_reciprocal_plus_one : ∫ x in (0:ℝ)..1, 1 / (1 + x) = Real.log 2 := by
  sorry

end integral_reciprocal_plus_one_l939_93967


namespace hyperbola_range_l939_93929

theorem hyperbola_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (m - 2) + y^2 / (m + 3) = 1) ↔ -3 < m ∧ m < 2 :=
sorry

end hyperbola_range_l939_93929


namespace basketball_players_count_l939_93954

/-- The number of boys playing basketball in a group with given conditions -/
def boys_playing_basketball (total : ℕ) (football : ℕ) (neither : ℕ) (both : ℕ) : ℕ :=
  total - neither

theorem basketball_players_count :
  boys_playing_basketball 22 15 3 18 = 19 :=
by sorry

end basketball_players_count_l939_93954


namespace pure_imaginary_z_implies_a_plus_2i_modulus_l939_93989

theorem pure_imaginary_z_implies_a_plus_2i_modulus (a : ℝ) : 
  let z : ℂ := (a + 3 * Complex.I) / (1 - 2 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (a + 2 * Complex.I) = 2 * Real.sqrt 10 :=
by sorry

end pure_imaginary_z_implies_a_plus_2i_modulus_l939_93989


namespace equality_theorem_l939_93951

theorem equality_theorem (a b c d e f : ℝ) 
  (h1 : a + b + c = d + e + f)
  (h2 : a^2 + b^2 + c^2 = d^2 + e^2 + f^2)
  (h3 : a^3 + b^3 + c^3 ≠ d^3 + e^3 + f^3) :
  (∀ k : ℝ, 
    (a + b + c + (d+k) + (e+k) + (f+k) = d + e + f + (a+k) + (b+k) + (c+k)) ∧
    (a^2 + b^2 + c^2 + (d+k)^2 + (e+k)^2 + (f+k)^2 = d^2 + e^2 + f^2 + (a+k)^2 + (b+k)^2 + (c+k)^2) ∧
    (a^3 + b^3 + c^3 + (d+k)^3 + (e+k)^3 + (f+k)^3 = d^3 + e^3 + f^3 + (a+k)^3 + (b+k)^3 + (c+k)^3)) ∧
  (∀ k : ℝ, k ≠ 0 → 
    a^4 + b^4 + c^4 + (d+k)^4 + (e+k)^4 + (f+k)^4 ≠ d^4 + e^4 + f^4 + (a+k)^4 + (b+k)^4 + (c+k)^4) :=
by sorry

end equality_theorem_l939_93951


namespace imaginary_part_of_i_times_one_minus_3i_l939_93915

theorem imaginary_part_of_i_times_one_minus_3i (i : ℂ) :
  i * i = -1 →
  (i * (1 - 3 * i)).im = 1 := by
sorry

end imaginary_part_of_i_times_one_minus_3i_l939_93915


namespace f_extremum_range_l939_93987

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - a * x^2 + x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * a * x + 1

-- Define the condition for exactly one extremum point in (-1, 0)
def has_one_extremum (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo (-1) 0 ∧ f' a x = 0

-- State the theorem
theorem f_extremum_range :
  ∀ a : ℝ, has_one_extremum a ↔ a ∈ Set.Ioi (-1/5) ∪ {-1} :=
sorry

end f_extremum_range_l939_93987


namespace machine_working_time_l939_93957

/-- The number of shirts made by the machine -/
def total_shirts : ℕ := 12

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 2

/-- The time the machine was working in minutes -/
def working_time : ℕ := total_shirts / shirts_per_minute

theorem machine_working_time : working_time = 6 := by sorry

end machine_working_time_l939_93957


namespace polynomial_simplification_l939_93931

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 2 * x^2 + 6 * x - 15) =
  x^3 + 2 * x^2 + 3 * x + 10 := by
  sorry

end polynomial_simplification_l939_93931
