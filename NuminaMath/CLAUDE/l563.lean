import Mathlib

namespace cooking_cleaning_arrangements_l563_56339

theorem cooking_cleaning_arrangements (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  Nat.choose n k = 10 := by
  sorry

end cooking_cleaning_arrangements_l563_56339


namespace intersection_property_l563_56388

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 13*x - 8

-- Define the line
def line (k m x : ℝ) : ℝ := k*x + m

-- Theorem statement
theorem intersection_property (k m : ℝ) 
  (hA : ∃ xA, f xA = line k m xA)  -- A exists
  (hB : ∃ xB, f xB = line k m xB)  -- B exists
  (hC : ∃ xC, f xC = line k m xC)  -- C exists
  (h_distinct : ∀ x y, x ≠ y → (f x = line k m x ∧ f y = line k m y) → 
                 ∃ z, f z = line k m z ∧ z ≠ x ∧ z ≠ y)  -- A, B, C are distinct
  (h_midpoint : ∃ xA xB xC, f xA = line k m xA ∧ f xB = line k m xB ∧ f xC = line k m xC ∧
                 xB = (xA + xC) / 2)  -- B is the midpoint of AC
  : 2*k + m = 2 := by
  sorry

end intersection_property_l563_56388


namespace inverse_variation_cube_l563_56308

/-- Given that x and y are positive real numbers, x^3 and y vary inversely,
    and y = 8 when x = 2, prove that x = 1 when y = 64. -/
theorem inverse_variation_cube (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ k : ℝ, ∀ x y, x^3 * y = k)
  (h_initial : 2^3 * 8 = (Classical.choose h_inverse))
  (h_y : y = 64) : x = 1 := by
  sorry

end inverse_variation_cube_l563_56308


namespace quadratic_properties_l563_56369

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a

-- Theorem statement
theorem quadratic_properties (a : ℝ) :
  (∀ x y, x < 1 ∧ y < 1 ∧ x < y → f a x > f a y) ∧
  (∃ x, f a x = 0 → a ≤ 4) ∧
  (¬(a = 3 → ∀ x, f a x > 0 ↔ 1 < x ∧ x < 3)) ∧
  (∀ b, f a 2013 = b → f a (-2009) = b) :=
sorry

end quadratic_properties_l563_56369


namespace zephyr_in_top_three_l563_56356

-- Define the propositions
variable (X : Prop) -- Xenon is in the top three
variable (Y : Prop) -- Yenofa is in the top three
variable (Z : Prop) -- Zephyr is in the top three

-- Define the conditions
axiom condition1 : Z → X
axiom condition2 : (X ∨ Y) → ¬Z
axiom condition3 : ¬((X ∨ Y) → ¬Z)

-- Theorem to prove
theorem zephyr_in_top_three : Z ∧ ¬X ∧ ¬Y := by
  sorry

end zephyr_in_top_three_l563_56356


namespace arithmetic_square_root_of_81_l563_56385

theorem arithmetic_square_root_of_81 : Real.sqrt 81 = 9 := by
  sorry

end arithmetic_square_root_of_81_l563_56385


namespace subcommittee_count_l563_56396

def total_members : ℕ := 12
def officers : ℕ := 5
def subcommittee_size : ℕ := 5

def subcommittees_with_at_least_two_officers : ℕ :=
  Nat.choose total_members subcommittee_size -
  (Nat.choose (total_members - officers) subcommittee_size +
   Nat.choose officers 1 * Nat.choose (total_members - officers) (subcommittee_size - 1))

theorem subcommittee_count :
  subcommittees_with_at_least_two_officers = 596 := by
  sorry

end subcommittee_count_l563_56396


namespace libby_igloo_top_bricks_l563_56324

/-- Represents the structure of an igloo --/
structure Igloo where
  total_rows : ℕ
  bottom_rows : ℕ
  top_rows : ℕ
  bottom_bricks_per_row : ℕ
  total_bricks : ℕ

/-- Calculates the number of bricks in each row of the top half of the igloo --/
def top_bricks_per_row (i : Igloo) : ℕ :=
  (i.total_bricks - i.bottom_rows * i.bottom_bricks_per_row) / i.top_rows

/-- Theorem stating the number of bricks in each row of the top half of Libby's igloo --/
theorem libby_igloo_top_bricks :
  let i : Igloo := {
    total_rows := 10,
    bottom_rows := 5,
    top_rows := 5,
    bottom_bricks_per_row := 12,
    total_bricks := 100
  }
  top_bricks_per_row i = 8 := by
  sorry

end libby_igloo_top_bricks_l563_56324


namespace tea_mixture_price_l563_56304

theorem tea_mixture_price (price1 price2 : ℝ) (ratio : ℝ) (mixture_price : ℝ) : 
  price1 = 64 →
  price2 = 74 →
  ratio = 1 →
  mixture_price = (price1 + price2) / (2 * ratio) →
  mixture_price = 69 := by
sorry

end tea_mixture_price_l563_56304


namespace circle_center_transformation_l563_56316

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Translates a point vertically by a given amount -/
def translate_vertical (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

/-- The initial center of circle U -/
def initial_center : ℝ × ℝ := (3, -4)

/-- The transformation applied to the center of circle U -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_vertical (reflect_y (reflect_x p)) (-10)

theorem circle_center_transformation :
  transform initial_center = (-3, -6) := by sorry

end circle_center_transformation_l563_56316


namespace ratio_x_to_y_l563_56325

theorem ratio_x_to_y (x y : ℚ) (h : (15 * x - 4 * y) / (18 * x - 3 * y) = 2 / 3) :
  x / y = 2 / 3 := by sorry

end ratio_x_to_y_l563_56325


namespace unbounded_expression_l563_56350

theorem unbounded_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∀ M : ℝ, ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ (x*y + 1)^2 + (x - y)^2 > M :=
sorry

end unbounded_expression_l563_56350


namespace may_cookie_cost_l563_56335

/-- The total amount spent on cookies in May -/
def total_cookie_cost (weekday_count : ℕ) (weekend_count : ℕ) 
  (weekday_cookie_count : ℕ) (weekend_cookie_count : ℕ)
  (weekday_cookie1_price : ℕ) (weekday_cookie2_price : ℕ)
  (weekend_cookie1_price : ℕ) (weekend_cookie2_price : ℕ) : ℕ :=
  (weekday_count * (2 * weekday_cookie1_price + 2 * weekday_cookie2_price)) +
  (weekend_count * (3 * weekend_cookie1_price + 2 * weekend_cookie2_price))

/-- Theorem stating the total amount spent on cookies in May -/
theorem may_cookie_cost : 
  total_cookie_cost 22 9 4 5 15 18 12 20 = 2136 := by
  sorry

end may_cookie_cost_l563_56335


namespace exists_special_number_l563_56399

def is_twelve_digit (n : ℕ) : Prop := 10^11 ≤ n ∧ n < 10^12

def is_not_perfect_square (n : ℕ) : Prop := ∃ (d : ℕ), n % 10 = d ∧ (d = 2 ∨ d = 3 ∨ d = 7 ∨ d = 8)

def is_ambiguous_cube (n : ℕ) : Prop := ∀ (d : ℕ), d < 10 → ∃ (k : ℕ), k^3 % 10 = d

theorem exists_special_number : ∃ (n : ℕ), 
  is_twelve_digit n ∧ 
  is_not_perfect_square n ∧ 
  is_ambiguous_cube n := by
  sorry

end exists_special_number_l563_56399


namespace union_of_sets_l563_56348

theorem union_of_sets : 
  let M : Set ℕ := {0, 3}
  let N : Set ℕ := {1, 2, 3}
  M ∪ N = {0, 1, 2, 3} := by
sorry

end union_of_sets_l563_56348


namespace mismatching_socks_l563_56358

theorem mismatching_socks (total_socks : ℕ) (matching_pairs : ℕ) 
  (h1 : total_socks = 25) (h2 : matching_pairs = 4) :
  total_socks - 2 * matching_pairs = 17 := by
  sorry

end mismatching_socks_l563_56358


namespace book_cost_is_15_l563_56336

def total_books : ℕ := 10
def num_magazines : ℕ := 10
def magazine_cost : ℚ := 2
def total_spent : ℚ := 170

theorem book_cost_is_15 :
  ∃ (book_cost : ℚ),
    book_cost * total_books + magazine_cost * num_magazines = total_spent ∧
    book_cost = 15 :=
by sorry

end book_cost_is_15_l563_56336


namespace river_crossing_possible_l563_56397

/-- Represents the state of the river crossing -/
structure RiverState where
  left_soldiers : Nat
  left_robbers : Nat
  right_soldiers : Nat
  right_robbers : Nat

/-- Represents a boat trip -/
inductive BoatTrip
  | SoldierSoldier
  | SoldierRobber
  | RobberRobber
  | Soldier
  | Robber

/-- Checks if a state is safe (soldiers not outnumbered by robbers) -/
def is_safe_state (state : RiverState) : Prop :=
  (state.left_soldiers ≥ state.left_robbers || state.left_soldiers = 0) &&
  (state.right_soldiers ≥ state.right_robbers || state.right_soldiers = 0)

/-- Applies a boat trip to a state -/
def apply_trip (state : RiverState) (trip : BoatTrip) (direction : Bool) : RiverState :=
  sorry

/-- Checks if the final state is reached -/
def is_final_state (state : RiverState) : Prop :=
  state.left_soldiers = 0 && state.left_robbers = 0 &&
  state.right_soldiers = 3 && state.right_robbers = 3

/-- Theorem: There exists a sequence of boat trips that safely transports everyone across -/
theorem river_crossing_possible : ∃ (trips : List (BoatTrip × Bool)),
  let final_state := trips.foldl (λ s (trip, dir) => apply_trip s trip dir)
    (RiverState.mk 3 3 0 0)
  is_final_state final_state ∧
  ∀ (intermediate_state : RiverState),
    intermediate_state ∈ trips.scanl (λ s (trip, dir) => apply_trip s trip dir)
      (RiverState.mk 3 3 0 0) →
    is_safe_state intermediate_state :=
  sorry

end river_crossing_possible_l563_56397


namespace james_monthly_earnings_l563_56371

/-- Calculates the monthly earnings of a Twitch streamer --/
def monthly_earnings (initial_subscribers : ℕ) (gifted_subscribers : ℕ) (earnings_per_subscriber : ℕ) : ℕ :=
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber

theorem james_monthly_earnings :
  monthly_earnings 150 50 9 = 1800 := by
  sorry

end james_monthly_earnings_l563_56371


namespace isosceles_triangle_other_side_l563_56379

structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2
  perimeter : side1 + side2 + base = 15

def has_side_6 (t : IsoscelesTriangle) : Prop :=
  t.side1 = 6 ∨ t.side2 = 6 ∨ t.base = 6

theorem isosceles_triangle_other_side (t : IsoscelesTriangle) 
  (h : has_side_6 t) : 
  (t.side1 = 3 ∧ t.side2 = 3) ∨ (t.side1 = 4.5 ∧ t.side2 = 4.5) ∨ t.base = 3 := by
  sorry

end isosceles_triangle_other_side_l563_56379


namespace connie_marbles_l563_56362

/-- The number of marbles Connie has after giving some away -/
def remaining_marbles (initial : ℝ) (given_away : ℝ) : ℝ :=
  initial - given_away

/-- Theorem stating that Connie has 3.2 marbles after giving some away -/
theorem connie_marbles : remaining_marbles 73.5 70.3 = 3.2 := by
  sorry

end connie_marbles_l563_56362


namespace fourth_grade_students_l563_56351

/-- Calculates the final number of students in fourth grade -/
def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that the final number of students is 43 -/
theorem fourth_grade_students : final_student_count 4 3 42 = 43 := by
  sorry

end fourth_grade_students_l563_56351


namespace light_reflection_theorem_l563_56334

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a line segment -/
def isOnSegment (P : Point) (A : Point) (B : Point) : Prop := sorry

/-- Checks if a quadrilateral is cyclic -/
def isCyclic (q : Quadrilateral) : Prop := sorry

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Represents the light ray path -/
structure LightPath (q : Quadrilateral) where
  P : Point
  Q : Point
  R : Point
  S : Point
  pOnAB : isOnSegment P q.A q.B
  qOnBC : isOnSegment Q q.B q.C
  rOnCD : isOnSegment R q.C q.D
  sOnDA : isOnSegment S q.D q.A

theorem light_reflection_theorem (q : Quadrilateral) :
  (∀ (path : LightPath q), isCyclic q) ∧
  (∃ (c : ℝ), ∀ (path : LightPath q), perimeter ⟨path.P, path.Q, path.R, path.S⟩ = c) :=
sorry

end light_reflection_theorem_l563_56334


namespace length_breadth_difference_l563_56320

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Theorem stating that for a rectangular plot with given conditions, 
    the length is 12 meters more than the breadth. -/
theorem length_breadth_difference (plot : RectangularPlot) 
  (h1 : plot.length = 56)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : plot.total_fencing_cost = (2 * (plot.length + plot.breadth)) * plot.fencing_cost_per_meter) :
  plot.length - plot.breadth = 12 := by
  sorry

#check length_breadth_difference

end length_breadth_difference_l563_56320


namespace complement_of_A_in_U_l563_56370

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U :
  (U \ A) = {2, 4, 5} := by sorry

end complement_of_A_in_U_l563_56370


namespace triangle_tangent_product_l563_56340

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_tangent_product (t : Triangle) 
  (h : t.a + t.c = 2 * t.b) : 
  Real.tan (t.A / 2) * Real.tan (t.C / 2) = 1 / 3 := by
  sorry

end triangle_tangent_product_l563_56340


namespace equilateral_triangle_division_l563_56315

theorem equilateral_triangle_division (side_length : ℕ) (h : side_length = 1536) :
  ∃ (n : ℕ), side_length^2 = 3 * n := by
  sorry

end equilateral_triangle_division_l563_56315


namespace complex_square_i_positive_l563_56322

theorem complex_square_i_positive (a : ℝ) : 
  (((a + Complex.I) ^ 2) * Complex.I).re > 0 → a = -1 := by
  sorry

end complex_square_i_positive_l563_56322


namespace koschei_coins_count_l563_56376

theorem koschei_coins_count :
  ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 10 = 7 ∧ n % 12 = 9 ∧ n = 357 := by
  sorry

end koschei_coins_count_l563_56376


namespace car_distance_calculation_l563_56398

/-- Proves that a car traveling at 260 km/h for 2 2/5 hours covers a distance of 624 km -/
theorem car_distance_calculation (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 260 → time = 2 + 2/5 → distance = speed * time → distance = 624 := by
  sorry

end car_distance_calculation_l563_56398


namespace max_sundays_in_84_days_l563_56380

theorem max_sundays_in_84_days : ℕ :=
  let days_in_period : ℕ := 84
  let days_in_week : ℕ := 7
  let sundays_per_week : ℕ := 1

  have h1 : days_in_period % days_in_week = 0 := by sorry
  have h2 : days_in_period / days_in_week * sundays_per_week = 12 := by sorry

  12

/- Proof
sorry
-/

end max_sundays_in_84_days_l563_56380


namespace right_triangle_sine_l563_56394

theorem right_triangle_sine (X Y Z : ℝ) : 
  -- XYZ is a right triangle with Y as the right angle
  (X + Y + Z = π) →
  (Y = π / 2) →
  -- sin X = 8/17
  (Real.sin X = 8 / 17) →
  -- Conclusion: sin Z = 15/17
  (Real.sin Z = 15 / 17) := by
sorry

end right_triangle_sine_l563_56394


namespace xyz_value_l563_56361

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 10)
  (eq5 : x + y + z = 6) :
  x * y * z = 15 := by
  sorry

end xyz_value_l563_56361


namespace solution_count_l563_56344

/-- The number of different integer solutions (x, y) for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := sorry

theorem solution_count :
  (num_solutions 1 = 4) →
  (num_solutions 2 = 8) →
  (num_solutions 20 = 80) := by sorry

end solution_count_l563_56344


namespace consecutive_triples_divisible_by_1001_l563_56386

def is_valid_triple (a b c : ℕ) : Prop :=
  a < 101 ∧ b < 101 ∧ c < 101 ∧
  b = a + 1 ∧ c = b + 1 ∧
  (a * b * c) % 1001 = 0

theorem consecutive_triples_divisible_by_1001 :
  ∀ a b c : ℕ,
    is_valid_triple a b c ↔ (a = 76 ∧ b = 77 ∧ c = 78) ∨ (a = 77 ∧ b = 78 ∧ c = 79) :=
by sorry

end consecutive_triples_divisible_by_1001_l563_56386


namespace unique_solution_for_equation_l563_56373

theorem unique_solution_for_equation (m n : ℕ+) : 
  (m : ℤ)^(n : ℕ) - (n : ℤ)^(m : ℕ) = 3 ↔ m = 4 ∧ n = 1 := by
  sorry

end unique_solution_for_equation_l563_56373


namespace intersection_of_specific_sets_l563_56360

theorem intersection_of_specific_sets :
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5}
  A ∩ B = {3} := by
sorry

end intersection_of_specific_sets_l563_56360


namespace lines_properties_l563_56363

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - 3 * y + 1 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := x - b * y + 2 = 0

-- Define parallelism for two lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem lines_properties (a b : ℝ) :
  (parallel (l₁ a) (l₂ b) → a * b = 3) ∧
  (b < 0 → ¬∃ (x y : ℝ), l₂ b x y ∧ first_quadrant x y) :=
sorry

end lines_properties_l563_56363


namespace product_equals_sum_solutions_l563_56338

theorem product_equals_sum_solutions (x y : ℤ) :
  x * y = x + y ↔ (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) := by
  sorry

end product_equals_sum_solutions_l563_56338


namespace f_of_5_l563_56359

def f (x : ℝ) : ℝ := x^2 - x

theorem f_of_5 : f 5 = 20 := by sorry

end f_of_5_l563_56359


namespace circle_equation_and_max_ratio_l563_56346

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 2

-- Define the given line equations
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*y + 3 = 0

theorem circle_equation_and_max_ratio :
  (∃ (x₀ y₀ : ℝ), line1 x₀ y₀ ∧ y₀ = 0 ∧
    ∀ (x y : ℝ), circle_C x y ↔ (x - x₀)^2 + (y - y₀)^2 = 2 ∧
    ∃ (x₁ y₁ : ℝ), line2 x₁ y₁ ∧ (x₁ - x₀)^2 + (y₁ - y₀)^2 = 2) ∧
  (∀ (x y : ℝ), circle2 x y → y / x ≤ Real.sqrt 3 / 3) ∧
  (∃ (x y : ℝ), circle2 x y ∧ y / x = Real.sqrt 3 / 3) :=
sorry

end circle_equation_and_max_ratio_l563_56346


namespace circle_center_point_distance_l563_56353

/-- The distance between the center of a circle and a point --/
theorem circle_center_point_distance (x y : ℝ) : 
  (x^2 + y^2 = 6*x - 2*y - 15) → 
  Real.sqrt ((3 - (-2))^2 + ((-1) - 5)^2) = Real.sqrt 61 := by
  sorry

end circle_center_point_distance_l563_56353


namespace balloon_blowup_ratio_l563_56323

theorem balloon_blowup_ratio (total : ℕ) (intact : ℕ) : 
  total = 200 → 
  intact = 80 → 
  (total - intact) / (total / 5) = 2 := by
sorry

end balloon_blowup_ratio_l563_56323


namespace same_terminal_side_angle_l563_56393

theorem same_terminal_side_angle : ∃ k : ℤ, k * 360 - 70 = 290 := by
  sorry

end same_terminal_side_angle_l563_56393


namespace line_tangent_to_parabola_l563_56310

theorem line_tangent_to_parabola :
  ∃! d : ℝ, ∀ x y : ℝ,
    (y = 3 * x + d) ∧ (y^2 = 12 * x) →
    (∃! t : ℝ, y = 3 * t + d ∧ y^2 = 12 * t) →
    d = 1 := by
  sorry

end line_tangent_to_parabola_l563_56310


namespace lindas_nickels_l563_56377

/-- The number of nickels Linda initially has -/
def initial_nickels : ℕ := 5

/-- The total number of coins Linda has after receiving additional coins -/
def total_coins : ℕ := 35

/-- The number of initial dimes -/
def initial_dimes : ℕ := 2

/-- The number of initial quarters -/
def initial_quarters : ℕ := 6

/-- The number of additional dimes given by her mother -/
def additional_dimes : ℕ := 2

/-- The number of additional quarters given by her mother -/
def additional_quarters : ℕ := 10

theorem lindas_nickels :
  initial_dimes + initial_quarters + initial_nickels +
  additional_dimes + additional_quarters + 2 * initial_nickels = total_coins :=
by sorry

end lindas_nickels_l563_56377


namespace jason_pokemon_cards_l563_56318

theorem jason_pokemon_cards (initial_cards given_away_cards : ℕ) :
  initial_cards = 9 →
  given_away_cards = 4 →
  initial_cards - given_away_cards = 5 :=
by sorry

end jason_pokemon_cards_l563_56318


namespace fourth_child_receives_24_l563_56311

/-- Represents the distribution of sweets among a mother and her children -/
structure SweetDistribution where
  total : ℕ
  mother_fraction : ℚ
  num_children : ℕ
  eldest_youngest_ratio : ℕ
  second_third_diff : ℕ
  third_fourth_diff : ℕ
  youngest_second_ratio : ℚ

/-- Calculates the number of sweets the fourth child receives -/
def fourth_child_sweets (d : SweetDistribution) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, the fourth child receives 24 sweets -/
theorem fourth_child_receives_24 (d : SweetDistribution) 
  (h1 : d.total = 120)
  (h2 : d.mother_fraction = 1/4)
  (h3 : d.num_children = 5)
  (h4 : d.eldest_youngest_ratio = 2)
  (h5 : d.second_third_diff = 6)
  (h6 : d.third_fourth_diff = 8)
  (h7 : d.youngest_second_ratio = 4/5) : 
  fourth_child_sweets d = 24 :=
sorry

end fourth_child_receives_24_l563_56311


namespace number_equality_l563_56300

theorem number_equality (y : ℚ) : 
  (30 / 100 : ℚ) * y = (25 / 100 : ℚ) * 40 → y = 100 / 3 := by
  sorry

end number_equality_l563_56300


namespace share_ratio_l563_56384

theorem share_ratio (total c b a : ℕ) (h1 : total = 406) (h2 : total = a + b + c) 
  (h3 : b = c / 2) (h4 : c = 232) : a / b = 1 / 2 := by
  sorry

end share_ratio_l563_56384


namespace power_product_cube_l563_56312

theorem power_product_cube (a b : ℝ) : (a * b^2)^3 = a^3 * b^6 := by
  sorry

end power_product_cube_l563_56312


namespace shipping_cost_shipping_cost_cents_shipping_cost_proof_l563_56329

/-- Shipping cost calculation for a book -/
theorem shipping_cost (G : ℝ) : ℝ :=
  8 * ⌈G / 100⌉

/-- The shipping cost in cents for a book weighing G grams -/
theorem shipping_cost_cents (G : ℝ) : ℝ :=
  shipping_cost G

/-- Proof that the shipping cost in cents is equal to 8 * ⌈G / 100⌉ -/
theorem shipping_cost_proof (G : ℝ) : shipping_cost_cents G = 8 * ⌈G / 100⌉ := by
  sorry

end shipping_cost_shipping_cost_cents_shipping_cost_proof_l563_56329


namespace systematic_sampling_result_l563_56331

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (last_sampled : ℕ) : List ℕ :=
  sorry

/-- Theorem for systematic sampling results -/
theorem systematic_sampling_result 
  (total : ℕ) 
  (sample_size : ℕ) 
  (last_sampled : ℕ) 
  (h1 : total = 8000) 
  (h2 : sample_size = 50) 
  (h3 : last_sampled = 7894) :
  let segment_size := total / sample_size
  let last_segment_start := total - segment_size
  let samples := systematic_sample total sample_size last_sampled
  (last_segment_start = 7840 ∧ 
   samples.take 5 = [54, 214, 374, 534, 694]) :=
sorry

end systematic_sampling_result_l563_56331


namespace jenny_cans_collected_l563_56389

/-- Represents the number of cans Jenny collects -/
def num_cans : ℕ := 20

/-- Represents the number of bottles Jenny collects -/
def num_bottles : ℕ := (100 - 2 * num_cans) / 6

/-- The weight of a bottle in ounces -/
def bottle_weight : ℕ := 6

/-- The weight of a can in ounces -/
def can_weight : ℕ := 2

/-- The payment for a bottle in cents -/
def bottle_payment : ℕ := 10

/-- The payment for a can in cents -/
def can_payment : ℕ := 3

/-- The total weight Jenny can carry in ounces -/
def total_weight : ℕ := 100

/-- The total payment Jenny receives in cents -/
def total_payment : ℕ := 160

theorem jenny_cans_collected :
  (num_bottles * bottle_weight + num_cans * can_weight = total_weight) ∧
  (num_bottles * bottle_payment + num_cans * can_payment = total_payment) :=
sorry

end jenny_cans_collected_l563_56389


namespace balanced_scale_l563_56319

/-- The weight of a children's book in kilograms. -/
def book_weight : ℝ := 1.1

/-- The weight of a doll in kilograms. -/
def doll_weight : ℝ := 0.3

/-- The weight of a toy car in kilograms. -/
def toy_car_weight : ℝ := 0.5

/-- The number of dolls on the scale. -/
def num_dolls : ℕ := 2

/-- The number of toy cars on the scale. -/
def num_toy_cars : ℕ := 1

theorem balanced_scale : 
  book_weight = num_dolls * doll_weight + num_toy_cars * toy_car_weight :=
by sorry

end balanced_scale_l563_56319


namespace greyEyedBlackHairedCount_l563_56302

/-- Represents the characteristics of students in a class. -/
structure ClassCharacteristics where
  total : ℕ
  greenEyedRedHaired : ℕ
  blackHaired : ℕ
  greyEyed : ℕ

/-- Calculates the number of grey-eyed black-haired students given the class characteristics. -/
def greyEyedBlackHaired (c : ClassCharacteristics) : ℕ :=
  c.greyEyed - (c.total - c.blackHaired - c.greenEyedRedHaired)

/-- Theorem stating that for the given class characteristics, 
    the number of grey-eyed black-haired students is 20. -/
theorem greyEyedBlackHairedCount : 
  let c : ClassCharacteristics := {
    total := 60,
    greenEyedRedHaired := 20,
    blackHaired := 35,
    greyEyed := 25
  }
  greyEyedBlackHaired c = 20 := by
  sorry


end greyEyedBlackHairedCount_l563_56302


namespace problem_solution_l563_56366

theorem problem_solution (m n : ℕ+) 
  (h1 : m.val + 5 < n.val)
  (h2 : (m.val + (m.val + 3) + (m.val + 5) + n.val + (n.val + 1) + (2 * n.val - 1)) / 6 = n.val)
  (h3 : (m.val + 5 + n.val) / 2 = n.val) : 
  m.val + n.val = 11 := by
  sorry

end problem_solution_l563_56366


namespace greatest_integer_gcd_18_is_9_l563_56382

theorem greatest_integer_gcd_18_is_9 :
  ∃ n : ℕ, n < 200 ∧ n.gcd 18 = 9 ∧ ∀ m : ℕ, m < 200 ∧ m.gcd 18 = 9 → m ≤ n :=
by sorry

end greatest_integer_gcd_18_is_9_l563_56382


namespace min_distance_point_to_circle_l563_56378

/-- The minimum distance between the point (3,4) and any point on the circle x^2 + y^2 = 1 is 4 -/
theorem min_distance_point_to_circle : ∃ (d : ℝ),
  d = 4 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 1 → 
  d ≤ Real.sqrt ((x - 3)^2 + (y - 4)^2) :=
by sorry

end min_distance_point_to_circle_l563_56378


namespace change_is_three_l563_56303

/-- Calculates the change received after a restaurant visit -/
def calculate_change (lee_amount : ℕ) (friend_amount : ℕ) (wings_cost : ℕ) (salad_cost : ℕ) (soda_cost : ℕ) (soda_quantity : ℕ) (tax : ℕ) : ℕ :=
  let total_amount := lee_amount + friend_amount
  let food_cost := wings_cost + salad_cost + soda_cost * soda_quantity
  let total_cost := food_cost + tax
  total_amount - total_cost

/-- Proves that the change received is $3 given the specific conditions -/
theorem change_is_three :
  calculate_change 10 8 6 4 1 2 3 = 3 := by
  sorry

end change_is_three_l563_56303


namespace reciprocal_sum_of_roots_l563_56368

theorem reciprocal_sum_of_roots (α β : ℝ) : 
  (∃ x y : ℝ, 7 * x^2 - 6 * x + 8 = 0 ∧ 
               7 * y^2 - 6 * y + 8 = 0 ∧ 
               x ≠ y ∧ 
               α = 1 / x ∧ 
               β = 1 / y) → 
  α + β = 3/4 := by
sorry


end reciprocal_sum_of_roots_l563_56368


namespace silver_knights_enchanted_fraction_l563_56395

structure Kingdom where
  total_knights : ℕ
  silver_knights : ℕ
  gold_knights : ℕ
  enchanted_knights : ℕ
  enchanted_silver : ℕ
  enchanted_gold : ℕ

def is_valid_kingdom (k : Kingdom) : Prop :=
  k.silver_knights + k.gold_knights = k.total_knights ∧
  k.silver_knights = (3 * k.total_knights) / 8 ∧
  k.enchanted_knights = k.total_knights / 8 ∧
  k.enchanted_silver + k.enchanted_gold = k.enchanted_knights ∧
  3 * k.enchanted_gold * k.silver_knights = k.enchanted_silver * k.gold_knights

theorem silver_knights_enchanted_fraction (k : Kingdom) 
  (h : is_valid_kingdom k) : 
  (k.enchanted_silver : ℚ) / k.silver_knights = 1 / 14 := by
  sorry

end silver_knights_enchanted_fraction_l563_56395


namespace solution_value_l563_56333

theorem solution_value (m : ℝ) : 
  (∃ x y : ℝ, m * x + 2 * y = 6 ∧ x = 1 ∧ y = 2) → m = 2 := by
  sorry

end solution_value_l563_56333


namespace five_topping_pizzas_l563_56345

theorem five_topping_pizzas (n : ℕ) (k : ℕ) : n = 8 ∧ k = 5 → Nat.choose n k = 56 := by
  sorry

end five_topping_pizzas_l563_56345


namespace factorization_proof_l563_56314

theorem factorization_proof (a : ℝ) : 3 * a^2 - 27 = 3 * (a + 3) * (a - 3) := by
  sorry

end factorization_proof_l563_56314


namespace no_matching_product_and_sum_l563_56392

theorem no_matching_product_and_sum : 
  ¬ ∃ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 15 ∧ 
  a * b = (List.range 16).sum - a - b :=
by sorry

end no_matching_product_and_sum_l563_56392


namespace sqrt_multiplication_equality_l563_56342

theorem sqrt_multiplication_equality : 3 * Real.sqrt 2 * Real.sqrt 6 = 6 * Real.sqrt 3 := by
  sorry

end sqrt_multiplication_equality_l563_56342


namespace cube_root_8000_l563_56332

theorem cube_root_8000 : ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3 : ℝ) = 8000^(1/3 : ℝ) ∧ b = 1 := by
  sorry

end cube_root_8000_l563_56332


namespace danys_farm_bushels_l563_56357

/-- Represents the farm animals and their food consumption -/
structure Farm where
  cows : Nat
  sheep : Nat
  chickens : Nat
  cow_sheep_consumption : Nat
  chicken_consumption : Nat

/-- Calculates the total bushels needed for a day -/
def total_bushels (farm : Farm) : Nat :=
  (farm.cows + farm.sheep) * farm.cow_sheep_consumption + 
  farm.chickens * farm.chicken_consumption

/-- Dany's farm -/
def danys_farm : Farm := {
  cows := 4,
  sheep := 3,
  chickens := 7,
  cow_sheep_consumption := 2,
  chicken_consumption := 3
}

/-- Theorem stating that Dany needs 35 bushels for a day -/
theorem danys_farm_bushels : total_bushels danys_farm = 35 := by
  sorry

end danys_farm_bushels_l563_56357


namespace parallel_vectors_x_value_l563_56305

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, 1)
  parallel a b → x = 4 := by
sorry

end parallel_vectors_x_value_l563_56305


namespace polynomial_identity_proof_l563_56391

theorem polynomial_identity_proof :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x : ℝ, (x^2 - x + 1)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                              a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  (a + a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂)^2 - (a₁ + a₃ + a₅ + a₇ + a₉ + a₁₁)^2 = 729 := by
sorry

end polynomial_identity_proof_l563_56391


namespace negation_existential_l563_56317

theorem negation_existential (f : ℝ → Prop) :
  (¬ ∃ x₀ > -1, x₀^2 + x₀ - 2018 > 0) ↔ (∀ x > -1, x^2 + x - 2018 ≤ 0) :=
sorry

end negation_existential_l563_56317


namespace diplomats_conference_l563_56367

/-- The number of diplomats who attended the conference -/
def D : ℕ := 120

/-- The number of diplomats who spoke Japanese -/
def J : ℕ := 20

/-- The number of diplomats who did not speak Russian -/
def not_R : ℕ := 32

/-- The percentage of diplomats who spoke neither Japanese nor Russian -/
def neither_percent : ℚ := 20 / 100

/-- The percentage of diplomats who spoke both Japanese and Russian -/
def both_percent : ℚ := 10 / 100

theorem diplomats_conference :
  D = 120 ∧
  J = 20 ∧
  not_R = 32 ∧
  neither_percent = 20 / 100 ∧
  both_percent = 10 / 100 ∧
  (D : ℚ) * neither_percent = (D - (J + (D - not_R) - (D : ℚ) * both_percent) : ℚ) :=
by sorry

end diplomats_conference_l563_56367


namespace sin_theta_plus_pi_fourth_l563_56355

theorem sin_theta_plus_pi_fourth (θ : Real) 
  (h1 : θ > π/2 ∧ θ < π) 
  (h2 : Real.tan (θ - π/4) = -4/3) : 
  Real.sin (θ + π/4) = -3/5 := by
  sorry

end sin_theta_plus_pi_fourth_l563_56355


namespace birthday_cake_problem_l563_56347

/-- Represents a cube cake with icing -/
structure CakeCube where
  size : Nat
  has_icing : Bool

/-- Counts the number of small cubes with icing on exactly two sides -/
def count_two_sided_icing (cake : CakeCube) : Nat :=
  sorry

/-- The main theorem about the birthday cake problem -/
theorem birthday_cake_problem (cake : CakeCube) :
  cake.size = 5 ∧ cake.has_icing = true → count_two_sided_icing cake = 96 := by
  sorry

end birthday_cake_problem_l563_56347


namespace fraction_sum_l563_56390

theorem fraction_sum : (3 : ℚ) / 4 + 9 / 12 = 3 / 2 := by
  sorry

end fraction_sum_l563_56390


namespace bicycle_cost_price_l563_56383

theorem bicycle_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (final_price : ℝ) 
  (h1 : profit_A_to_B = 0.35)
  (h2 : profit_B_to_C = 0.45)
  (h3 : final_price = 225) :
  final_price / ((1 + profit_A_to_B) * (1 + profit_B_to_C)) = 
    final_price / (1.35 * 1.45) := by
  sorry

end bicycle_cost_price_l563_56383


namespace beta_function_integral_l563_56374

theorem beta_function_integral (p q : ℕ) :
  ∫ x in (0:ℝ)..1, x^p * (1-x)^q = (p.factorial * q.factorial) / (p+q+1).factorial :=
sorry

end beta_function_integral_l563_56374


namespace derivative_product_at_4_and_neg1_l563_56372

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 / Real.sqrt x else 1 + x^2

theorem derivative_product_at_4_and_neg1 :
  (deriv f 4) * (deriv f (-1)) = -1/8 := by sorry

end derivative_product_at_4_and_neg1_l563_56372


namespace tenth_minus_ninth_square_tiles_l563_56326

-- Define the sequence of square side lengths
def squareSideLength (n : ℕ) : ℕ := n

-- Define the number of tiles in the nth square
def tilesInSquare (n : ℕ) : ℕ := (squareSideLength n) ^ 2

-- Theorem statement
theorem tenth_minus_ninth_square_tiles : 
  tilesInSquare 10 - tilesInSquare 9 = 19 := by sorry

end tenth_minus_ninth_square_tiles_l563_56326


namespace range_of_a_l563_56387

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then 2^x + a else x + a^2

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → 
  (a ≤ -1 ∨ a ≥ 2) :=
sorry

end range_of_a_l563_56387


namespace power_mod_seven_l563_56364

theorem power_mod_seven : 76^77 % 7 = 6 := by
  sorry

end power_mod_seven_l563_56364


namespace binary_110011_eq_51_l563_56343

/-- Converts a list of binary digits to a decimal number -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110011 -/
def binary_110011 : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that 110011₂ is equal to 51 in decimal -/
theorem binary_110011_eq_51 : binary_to_decimal binary_110011 = 51 := by
  sorry

end binary_110011_eq_51_l563_56343


namespace irrational_among_given_numbers_l563_56341

theorem irrational_among_given_numbers : 
  (¬ (∃ (a b : ℤ), (22 : ℚ) / 7 = a / b)) ∧ 
  (¬ (∃ (a b : ℤ), (0.303003 : ℚ) = a / b)) ∧ 
  (¬ (∃ (a b : ℤ), Real.sqrt 27 = a / b)) ∧ 
  (¬ (∃ (a b : ℤ), ((-64 : ℝ) ^ (1/3 : ℝ)) = a / b)) ↔ 
  (∃ (a b : ℤ), (22 : ℚ) / 7 = a / b) ∧ 
  (∃ (a b : ℤ), (0.303003 : ℚ) = a / b) ∧ 
  (¬ (∃ (a b : ℤ), Real.sqrt 27 = a / b)) ∧ 
  (∃ (a b : ℤ), ((-64 : ℝ) ^ (1/3 : ℝ)) = a / b) :=
sorry

end irrational_among_given_numbers_l563_56341


namespace magic_square_vector_sum_l563_56337

/-- Represents a magic square of size n × n -/
structure MagicSquare (n : ℕ) where
  grid : Matrix (Fin n) (Fin n) ℕ
  elements : ∀ i j, grid i j ∈ Finset.range (n^2 + 1) \ {0}
  row_sum : ∀ i, (Finset.univ.sum fun j => grid i j) = n * (n^2 + 1) / 2
  col_sum : ∀ j, (Finset.univ.sum fun i => grid i j) = n * (n^2 + 1) / 2
  diag_sum : (Finset.univ.sum fun i => grid i i) = n * (n^2 + 1) / 2

/-- Vector connecting centers of two cells -/
def cellVector (n : ℕ) (i j k l : Fin n) : ℝ × ℝ :=
  (↑k - ↑i, ↑l - ↑j)

/-- The theorem to be proved -/
theorem magic_square_vector_sum (n : ℕ) (ms : MagicSquare n) :
  (Finset.univ.sum fun i =>
    (Finset.univ.sum fun j =>
      (Finset.univ.sum fun k =>
        (Finset.univ.sum fun l =>
          if ms.grid i j > ms.grid k l
          then cellVector n i j k l
          else (0, 0))))) = (0, 0) :=
sorry

end magic_square_vector_sum_l563_56337


namespace jiaki_calculation_final_result_l563_56375

-- Define A as a function of x
def A (x : ℤ) : ℤ := 3 * x^2 - x + 1

-- Define B as a function of x
def B (x : ℤ) : ℤ := -x^2 - 2*x - 3

-- State the theorem
theorem jiaki_calculation (x : ℤ) :
  A x - B x = 2 * x^2 - 3*x - 2 ∧
  (x = -1 → A x - B x = 3) :=
by sorry

-- Define the largest negative integer
def largest_negative_integer : ℤ := -1

-- State the final result
theorem final_result :
  A largest_negative_integer - B largest_negative_integer = 3 :=
by sorry

end jiaki_calculation_final_result_l563_56375


namespace max_min_product_l563_56354

theorem max_min_product (a b : ℕ+) (h : a + b = 100) :
  (∀ x y : ℕ+, x + y = 100 → x * y ≤ a * b) → a * b = 2500 ∧
  (∀ x y : ℕ+, x + y = 100 → a * b ≤ x * y) → a * b = 99 :=
by sorry

end max_min_product_l563_56354


namespace haley_recycling_cans_l563_56309

theorem haley_recycling_cans (collected : ℕ) (in_bag : ℕ) 
  (h1 : collected = 9) (h2 : in_bag = 7) : 
  collected - in_bag = 2 := by
  sorry

end haley_recycling_cans_l563_56309


namespace divisibility_count_l563_56301

def count_numbers (n : ℕ) : ℕ :=
  (n / 6) - ((n / 12) + (n / 18) - (n / 36))

theorem divisibility_count : count_numbers 2018 = 112 := by
  sorry

end divisibility_count_l563_56301


namespace sin_cos_sum_special_angle_l563_56352

theorem sin_cos_sum_special_angle : 
  Real.sin (5 * π / 180) * Real.cos (55 * π / 180) + 
  Real.cos (5 * π / 180) * Real.sin (55 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_special_angle_l563_56352


namespace probability_negative_product_l563_56306

def dice_faces : Finset Int := {-3, -2, -1, 0, 1, 2}

def is_negative_product (x y : Int) : Bool :=
  x * y < 0

def count_negative_products : Nat :=
  (dice_faces.filter (λ x => x < 0)).card * (dice_faces.filter (λ x => x > 0)).card * 2

theorem probability_negative_product :
  (count_negative_products : ℚ) / (dice_faces.card * dice_faces.card) = 1/3 := by
  sorry

end probability_negative_product_l563_56306


namespace triangle_side_sum_range_l563_56365

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    if b^2 + c^2 - a^2 = bc, AB · BC > 0, and a = √3/2,
    then √3/2 < b + c < 3/2. -/
theorem triangle_side_sum_range (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- positive angles
  A + B + C = π →  -- angles sum to π
  b^2 + c^2 - a^2 = b * c →  -- given condition
  (b * c * Real.cos A) > 0 →  -- AB · BC > 0
  a = Real.sqrt 3 / 2 →  -- given condition
  Real.sqrt 3 / 2 < b + c ∧ b + c < 3 / 2 := by
  sorry

end triangle_side_sum_range_l563_56365


namespace special_function_unique_l563_56381

/-- A function f: ℤ × ℤ → ℝ satisfying specific conditions -/
def special_function (f : ℤ × ℤ → ℝ) : Prop :=
  (∀ x y z : ℤ, f (x, y) * f (y, z) * f (z, x) = 1) ∧
  (∀ x : ℤ, f (x + 1, x) = 2)

/-- Theorem stating that any function satisfying the special_function conditions 
    must be of the form f(x,y) = 2^(x-y) -/
theorem special_function_unique (f : ℤ × ℤ → ℝ) 
  (hf : special_function f) : 
  ∀ x y : ℤ, f (x, y) = 2^(x - y) := by
  sorry

end special_function_unique_l563_56381


namespace square_root_property_l563_56328

theorem square_root_property (x : ℝ) :
  Real.sqrt (x + 4) = 3 → (x + 4)^2 = 81 := by
  sorry

end square_root_property_l563_56328


namespace lele_can_afford_cars_with_change_l563_56330

def price_a : ℚ := 46.5
def price_b : ℚ := 54.5
def lele_money : ℚ := 120

theorem lele_can_afford_cars_with_change : 
  price_a + price_b ≤ lele_money ∧ lele_money - (price_a + price_b) = 19 :=
by sorry

end lele_can_afford_cars_with_change_l563_56330


namespace hexagonal_grid_toothpicks_l563_56313

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of toothpicks in each side of the hexagon -/
def toothpicks_per_side : ℕ := 6

/-- The total number of toothpicks used to build the hexagonal grid -/
def total_toothpicks : ℕ := hexagon_sides * toothpicks_per_side

/-- Theorem: The total number of toothpicks used to build the hexagonal grid is 36 -/
theorem hexagonal_grid_toothpicks :
  total_toothpicks = 36 := by
  sorry

end hexagonal_grid_toothpicks_l563_56313


namespace event_ratio_l563_56327

theorem event_ratio (total : ℕ) (children : ℕ) (adults : ℕ) : 
  total = 42 → children = 28 → adults = total - children → 
  (children : ℚ) / (adults : ℚ) = 2 / 1 := by
  sorry

end event_ratio_l563_56327


namespace airplane_seats_total_l563_56307

theorem airplane_seats_total (first_class : ℕ) (coach : ℕ) : 
  first_class = 77 → 
  coach = 4 * first_class + 2 → 
  first_class + coach = 387 := by
sorry

end airplane_seats_total_l563_56307


namespace fundraising_contribution_l563_56321

theorem fundraising_contribution
  (total_goal : ℕ)
  (num_participants : ℕ)
  (admin_fee : ℕ)
  (h1 : total_goal = 2400)
  (h2 : num_participants = 8)
  (h3 : admin_fee = 20) :
  (total_goal / num_participants) + admin_fee = 320 := by
sorry

end fundraising_contribution_l563_56321


namespace dans_candy_bars_l563_56349

theorem dans_candy_bars (total_spent : ℝ) (cost_per_bar : ℝ) (h1 : total_spent = 4) (h2 : cost_per_bar = 2) :
  total_spent / cost_per_bar = 2 := by
  sorry

end dans_candy_bars_l563_56349
