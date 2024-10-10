import Mathlib

namespace square_sum_equals_twice_square_a_l2457_245786

theorem square_sum_equals_twice_square_a 
  (x y a θ : ℝ) 
  (h1 : x * Real.cos θ - y * Real.sin θ = a) 
  (h2 : (x - a * Real.sin θ)^2 + (y - a * Real.cos θ)^2 = a^2) : 
  x^2 + y^2 = 2 * a^2 := by
sorry

end square_sum_equals_twice_square_a_l2457_245786


namespace last_three_digits_of_8_to_108_l2457_245743

theorem last_three_digits_of_8_to_108 : 8^108 ≡ 38 [ZMOD 1000] := by
  sorry

end last_three_digits_of_8_to_108_l2457_245743


namespace remainder_of_Q_mod_1000_l2457_245765

theorem remainder_of_Q_mod_1000 :
  (202^1 + 20^21 + 2^21) % 1000 = 354 := by
  sorry

end remainder_of_Q_mod_1000_l2457_245765


namespace max_value_function_l2457_245784

theorem max_value_function (a : ℝ) (h : a > 0) :
  ∃ (max : ℝ), ∀ (x : ℝ), x > 0 → a > 2*x → x*(a - 2*x) ≤ max ∧
  ∃ (x₀ : ℝ), x₀ > 0 ∧ a > 2*x₀ ∧ x₀*(a - 2*x₀) = max :=
by sorry

end max_value_function_l2457_245784


namespace quadratic_inequality_solution_set_l2457_245709

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x | a * x^2 + b * x + c > 0}) :
  Set.Ioo (-3 : ℝ) (1/2) = {x | c * x^2 + b * x + a < 0} := by
  sorry

end quadratic_inequality_solution_set_l2457_245709


namespace principal_booking_l2457_245759

/-- The number of rooms needed to accommodate a class on a field trip -/
def rooms_needed (total_students : ℕ) (students_per_room : ℕ) : ℕ :=
  (total_students + students_per_room - 1) / students_per_room

/-- Theorem: The principal needs to book 6 rooms for 30 students -/
theorem principal_booking : 
  let total_students : ℕ := 30
  let queen_bed_capacity : ℕ := 2
  let pullout_couch_capacity : ℕ := 1
  let room_capacity : ℕ := 2 * queen_bed_capacity + pullout_couch_capacity
  rooms_needed total_students room_capacity = 6 := by
sorry

end principal_booking_l2457_245759


namespace statement3_is_analogous_reasoning_l2457_245778

-- Define the concept of a geometric figure
structure GeometricFigure where
  name : String

-- Define the concept of a property for geometric figures
structure Property where
  description : String

-- Define the concept of reasoning
inductive Reasoning
| Analogous
| Inductive
| Deductive

-- Define the statement about equilateral triangles
def equilateralTriangleProperty : Property :=
  { description := "The sum of distances from a point inside to its sides is constant" }

-- Define the statement about regular tetrahedrons
def regularTetrahedronProperty : Property :=
  { description := "The sum of distances from a point inside to its faces is constant" }

-- Define the reasoning process in statement ③
def statement3 (equilateralTriangle regularTetrahedron : GeometricFigure)
               (equilateralProp tetrahedronProp : Property) : Prop :=
  (equilateralProp = equilateralTriangleProperty) →
  (tetrahedronProp = regularTetrahedronProperty) →
  ∃ (r : Reasoning), r = Reasoning.Analogous

-- Theorem statement
theorem statement3_is_analogous_reasoning 
  (equilateralTriangle regularTetrahedron : GeometricFigure)
  (equilateralProp tetrahedronProp : Property) :
  statement3 equilateralTriangle regularTetrahedron equilateralProp tetrahedronProp :=
by
  sorry

end statement3_is_analogous_reasoning_l2457_245778


namespace reflect_x_of_P_l2457_245761

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The theorem stating that reflecting the point P(-2, √5) across the x-axis 
    results in the point (-2, -√5) -/
theorem reflect_x_of_P : 
  let P : Point := { x := -2, y := Real.sqrt 5 }
  reflect_x P = { x := -2, y := -Real.sqrt 5 } := by
  sorry

end reflect_x_of_P_l2457_245761


namespace friends_money_distribution_l2457_245735

theorem friends_money_distribution (x : ℚ) :
  x > 0 →
  let total := 6*x + 5*x + 4*x + 7*x + 0
  let pete_received := x + x + x + x
  pete_received / total = 2 / 11 := by
sorry

end friends_money_distribution_l2457_245735


namespace square_root_product_plus_one_l2457_245729

theorem square_root_product_plus_one (a : ℕ) (n : ℕ) : 
  a = 2020 ∧ n = 4086461 → a * (a + 1) * (a + 2) * (a + 3) + 1 = n^2 := by
  sorry

end square_root_product_plus_one_l2457_245729


namespace sin_beta_value_l2457_245747

theorem sin_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.cos α = 1 / 7)
  (h4 : Real.cos (α + β) = -11 / 14) :
  Real.sin β = Real.sqrt 3 / 2 := by
sorry

end sin_beta_value_l2457_245747


namespace probability_of_three_ones_l2457_245757

def probability_of_sum_three (n : ℕ) (sides : ℕ) (target_sum : ℕ) : ℚ :=
  if n = 3 ∧ sides = 6 ∧ target_sum = 3 then 1 / 216 else 0

theorem probability_of_three_ones :
  probability_of_sum_three 3 6 3 = 1 / 216 := by
  sorry

end probability_of_three_ones_l2457_245757


namespace pop_albums_count_l2457_245704

def country_albums : ℕ := 2
def songs_per_album : ℕ := 6
def total_songs : ℕ := 30

theorem pop_albums_count : 
  ∃ (pop_albums : ℕ), 
    country_albums * songs_per_album + pop_albums * songs_per_album = total_songs ∧ 
    pop_albums = 3 := by
  sorry

end pop_albums_count_l2457_245704


namespace voucher_distribution_l2457_245746

-- Define the number of representatives and vouchers
def num_representatives : ℕ := 5
def num_vouchers : ℕ := 4

-- Define the distribution method
def distribution_method (n m : ℕ) : ℕ := Nat.choose n m

-- Theorem statement
theorem voucher_distribution :
  distribution_method num_representatives num_vouchers = 5 := by
  sorry

end voucher_distribution_l2457_245746


namespace quadratic_inequality_solution_set_l2457_245797

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end quadratic_inequality_solution_set_l2457_245797


namespace jimmy_sandwiches_l2457_245771

/-- The number of sandwiches Jimmy can make given the number of bread packs,
    slices per pack, and slices needed per sandwich. -/
def sandwiches_made (bread_packs : ℕ) (slices_per_pack : ℕ) (slices_per_sandwich : ℕ) : ℕ :=
  (bread_packs * slices_per_pack) / slices_per_sandwich

/-- Theorem stating that Jimmy made 8 sandwiches under the given conditions. -/
theorem jimmy_sandwiches :
  sandwiches_made 4 4 2 = 8 := by
  sorry

end jimmy_sandwiches_l2457_245771


namespace living_room_walls_count_l2457_245739

/-- The number of walls in Eric's living room -/
def living_room_walls : ℕ := 7

/-- The time Eric spent removing wallpaper from one wall in the dining room (in hours) -/
def time_per_wall : ℕ := 2

/-- The total time it will take Eric to remove wallpaper from the living room (in hours) -/
def total_time : ℕ := 14

/-- Theorem stating that the number of walls in Eric's living room is 7 -/
theorem living_room_walls_count :
  living_room_walls = total_time / time_per_wall :=
by sorry

end living_room_walls_count_l2457_245739


namespace fixed_points_subset_stable_points_exists_function_with_infinite_stable_points_stable_points_are_fixed_points_for_increasing_functions_l2457_245741

-- Define the concept of a fixed point
def IsFixedPoint (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Define the concept of a stable point
def IsStablePoint (f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x

-- Define the set of fixed points
def FixedPoints (f : ℝ → ℝ) : Set ℝ := {x | IsFixedPoint f x}

-- Define the set of stable points
def StablePoints (f : ℝ → ℝ) : Set ℝ := {x | IsStablePoint f x}

-- Statement 1: Fixed points are a subset of stable points
theorem fixed_points_subset_stable_points (f : ℝ → ℝ) :
  FixedPoints f ⊆ StablePoints f := by sorry

-- Statement 2: There exists a function with infinitely many stable points
theorem exists_function_with_infinite_stable_points :
  ∃ f : ℝ → ℝ, ¬(Finite (StablePoints f)) := by sorry

-- Statement 3: For monotonically increasing functions, stable points are fixed points
theorem stable_points_are_fixed_points_for_increasing_functions
  (f : ℝ → ℝ) (h : ∀ x y, x < y → f x < f y) :
  ∀ x, IsStablePoint f x → IsFixedPoint f x := by sorry

end fixed_points_subset_stable_points_exists_function_with_infinite_stable_points_stable_points_are_fixed_points_for_increasing_functions_l2457_245741


namespace fraction_sum_product_equality_l2457_245712

theorem fraction_sum_product_equality (x y : ℤ) :
  (19 : ℚ) / x + (96 : ℚ) / y = ((19 : ℚ) / x) * ((96 : ℚ) / y) →
  ∃ m : ℤ, x = 19 * m ∧ y = 96 - 96 * m :=
by sorry

end fraction_sum_product_equality_l2457_245712


namespace max_m_value_l2457_245742

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem max_m_value : 
  (∃ (m : ℝ), m > 0 ∧ 
    (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m → f (x + t) ≤ x) ∧ 
    (∀ (m' : ℝ), m' > m → 
      ¬(∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m' → f (x + t) ≤ x))) ∧
  (∀ (m : ℝ), 
    (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m → f (x + t) ≤ x) → 
    m ≤ 4) :=
sorry

end max_m_value_l2457_245742


namespace square_root_of_neg_five_squared_l2457_245715

theorem square_root_of_neg_five_squared : Real.sqrt ((-5)^2) = 5 ∨ Real.sqrt ((-5)^2) = -5 := by
  sorry

end square_root_of_neg_five_squared_l2457_245715


namespace solve_for_h_l2457_245718

/-- The y-intercept of the first equation -/
def y_intercept1 : ℝ := 2025

/-- The y-intercept of the second equation -/
def y_intercept2 : ℝ := 2026

/-- The first equation -/
def equation1 (h j x y : ℝ) : Prop := y = 4 * (x - h)^2 + j

/-- The second equation -/
def equation2 (h k x y : ℝ) : Prop := y = x^3 - 3 * (x - h)^2 + k

/-- Positive integer x-intercepts for the first equation -/
def positive_integer_roots1 (h j : ℝ) : Prop :=
  ∃ (x1 x2 : ℕ), x1 ≠ 0 ∧ x2 ≠ 0 ∧ equation1 h j x1 0 ∧ equation1 h j x2 0

/-- Positive integer x-intercepts for the second equation -/
def positive_integer_roots2 (h k : ℝ) : Prop :=
  ∃ (x1 x2 : ℕ), x1 ≠ 0 ∧ x2 ≠ 0 ∧ equation2 h k x1 0 ∧ equation2 h k x2 0

/-- The main theorem -/
theorem solve_for_h :
  ∃ (h j k : ℝ),
    equation1 h j 0 y_intercept1 ∧
    equation2 h k 0 y_intercept2 ∧
    positive_integer_roots1 h j ∧
    positive_integer_roots2 h k ∧
    h = 45 := by sorry

end solve_for_h_l2457_245718


namespace rogers_nickels_l2457_245773

theorem rogers_nickels :
  ∀ (N : ℕ),
  (42 + N + 15 : ℕ) - 66 = 27 →
  N = 36 :=
by
  sorry

end rogers_nickels_l2457_245773


namespace midpoint_sum_l2457_245708

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (3, 4) and (10, 20) is 18.5 -/
theorem midpoint_sum : 
  let x₁ : ℝ := 3
  let y₁ : ℝ := 4
  let x₂ : ℝ := 10
  let y₂ : ℝ := 20
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 18.5 := by
sorry

end midpoint_sum_l2457_245708


namespace power_product_cube_l2457_245790

theorem power_product_cube (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 := by
  sorry

end power_product_cube_l2457_245790


namespace square_root_of_four_l2457_245703

theorem square_root_of_four : 
  {x : ℝ | x ^ 2 = 4} = {2, -2} := by sorry

end square_root_of_four_l2457_245703


namespace geometric_series_ratio_l2457_245737

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0) :
  (a / (1 - r) = 81 * (a * r^4) / (1 - r)) → r = 1/3 := by
sorry

end geometric_series_ratio_l2457_245737


namespace inequality_relationship_l2457_245727

theorem inequality_relationship (x : ℝ) :
  ¬(((x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧
    ((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) :=
by sorry

end inequality_relationship_l2457_245727


namespace sum_of_altitudes_l2457_245792

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 10 * x + 8 * y = 80

-- Define the triangle formed by the line and coordinate axes
def triangle_vertices : Set (ℝ × ℝ) :=
  {(0, 0), (8, 0), (0, 10)}

-- State the theorem
theorem sum_of_altitudes :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ triangle_vertices → line_equation x y) ∧
    a + b + c = 18 + (80 * Real.sqrt 164) / 164 :=
sorry

end sum_of_altitudes_l2457_245792


namespace equation_solutions_l2457_245733

theorem equation_solutions :
  (∃ x : ℚ, 4 * (x + 3) = 25 ∧ x = 13 / 4) ∧
  (∃ x₁ x₂ : ℚ, 5 * x₁^2 - 3 * x₁ = x₁ + 1 ∧ x₁ = -1 / 5 ∧
               5 * x₂^2 - 3 * x₂ = x₂ + 1 ∧ x₂ = 1) ∧
  (∃ x₁ x₂ : ℚ, 2 * (x₁ - 2)^2 - (x₁ - 2) = 0 ∧ x₁ = 2 ∧
               2 * (x₂ - 2)^2 - (x₂ - 2) = 0 ∧ x₂ = 5 / 2) :=
by sorry

end equation_solutions_l2457_245733


namespace new_person_weight_l2457_245763

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3.5 →
  replaced_weight = 62 →
  initial_count * weight_increase + replaced_weight = 90 :=
by
  sorry

end new_person_weight_l2457_245763


namespace intersection_M_N_l2457_245758

-- Define the sets M and N
def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end intersection_M_N_l2457_245758


namespace distance_between_circumcenters_l2457_245749

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the side lengths of the triangle
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (dist t.A t.B, dist t.B t.C, dist t.C t.A)

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem distance_between_circumcenters (t : Triangle) :
  let H := orthocenter t
  let side_len := side_lengths t
  side_len.1 = 13 ∧ side_len.2.1 = 14 ∧ side_len.2.2 = 15 →
  dist (circumcenter ⟨t.A, H, t.B⟩) (circumcenter ⟨t.A, H, t.C⟩) = 14 :=
sorry

end distance_between_circumcenters_l2457_245749


namespace cyclic_sum_divisibility_l2457_245724

theorem cyclic_sum_divisibility (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
  sorry

end cyclic_sum_divisibility_l2457_245724


namespace ab_value_l2457_245716

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the main theorem
theorem ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ (w x y z : ℕ), w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    (w : ℝ) = (log10 a) ^ (1/3 : ℝ) ∧
    (x : ℝ) = (log10 b) ^ (1/3 : ℝ) ∧
    (y : ℝ) = log10 (a ^ (1/3 : ℝ)) ∧
    (z : ℝ) = log10 (b ^ (1/3 : ℝ)) ∧
    w + x + y + z = 12) →
  a * b = 10^9 :=
by sorry

end ab_value_l2457_245716


namespace parallel_vectors_magnitude_l2457_245728

/-- Given two parallel vectors a and b, prove that the magnitude of b is √13 -/
theorem parallel_vectors_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![(-4), 6]
  let b : Fin 2 → ℝ := ![2, x]
  (∃ (k : ℝ), ∀ i, b i = k * a i) →  -- Parallel vectors condition
  Real.sqrt ((b 0) ^ 2 + (b 1) ^ 2) = Real.sqrt 13 := by
  sorry

end parallel_vectors_magnitude_l2457_245728


namespace arctan_equation_solution_l2457_245726

theorem arctan_equation_solution :
  ∃ x : ℚ, 2 * Real.arctan (1/3) + 4 * Real.arctan (1/5) + Real.arctan (1/x) = π/4 ∧ x = -978/2029 := by
  sorry

end arctan_equation_solution_l2457_245726


namespace sum_remainder_l2457_245779

theorem sum_remainder (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end sum_remainder_l2457_245779


namespace mary_max_earnings_l2457_245767

/-- Calculates the maximum weekly earnings for a worker under specific pay conditions -/
def maxWeeklyEarnings (maxHours : ℕ) (regularRate : ℚ) (overtime1Multiplier : ℚ) (overtime2Multiplier : ℚ) : ℚ :=
  let regularHours := min maxHours 40
  let overtime1Hours := min (maxHours - regularHours) 20
  let overtime2Hours := maxHours - regularHours - overtime1Hours
  let regularPay := regularRate * regularHours
  let overtime1Pay := regularRate * overtime1Multiplier * overtime1Hours
  let overtime2Pay := regularRate * overtime2Multiplier * overtime2Hours
  regularPay + overtime1Pay + overtime2Pay

theorem mary_max_earnings :
  maxWeeklyEarnings 80 15 1.6 2 = 1680 := by
  sorry

#eval maxWeeklyEarnings 80 15 1.6 2

end mary_max_earnings_l2457_245767


namespace license_plate_count_l2457_245730

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- The total number of characters available (letters + digits) -/
def num_chars : ℕ := num_letters + num_digits

/-- The format of the license plate -/
inductive LicensePlateChar
| Letter
| Digit
| Any

/-- The structure of the license plate -/
def license_plate_format : List LicensePlateChar :=
  [LicensePlateChar.Letter, LicensePlateChar.Digit, LicensePlateChar.Any, LicensePlateChar.Digit]

/-- 
  The number of ways to create a 4-character license plate 
  where the format is a letter followed by a digit, then any character, and ending with a digit,
  ensuring that exactly two characters on the license plate are the same.
-/
theorem license_plate_count : 
  (num_letters * num_digits * num_chars) = 9360 := by
  sorry

end license_plate_count_l2457_245730


namespace trapezoid_area_l2457_245744

theorem trapezoid_area (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 4 →
  small_square_side = 1 →
  let large_square_area := large_square_side ^ 2
  let small_square_area := small_square_side ^ 2
  let total_trapezoid_area := large_square_area - small_square_area
  let num_trapezoids := 4
  (total_trapezoid_area / num_trapezoids : ℝ) = 15 / 4 := by
  sorry

end trapezoid_area_l2457_245744


namespace cryptarithmetic_puzzle_l2457_245714

theorem cryptarithmetic_puzzle (D E F G : ℕ) : 
  (∀ (X Y : ℕ), (X = D ∨ X = E ∨ X = F ∨ X = G) ∧ (Y = D ∨ Y = E ∨ Y = F ∨ Y = G) ∧ X ≠ Y → X ≠ Y) →
  F - E = D - 1 →
  D + E + F = 16 →
  F - E = D →
  G = F - E →
  G = 5 := by
sorry

end cryptarithmetic_puzzle_l2457_245714


namespace white_balls_count_l2457_245796

theorem white_balls_count (n : ℕ) : 
  n = 27 ∧ 
  (∃ (total : ℕ), 
    total = n + 3 ∧ 
    (3 : ℚ) / total = 1 / 10) := by
  sorry

end white_balls_count_l2457_245796


namespace geometric_sequence_properties_l2457_245706

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- Main theorem
theorem geometric_sequence_properties (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (is_geometric_sequence (fun n => |a n|)) ∧
  (is_geometric_sequence (fun n => a n * a (n + 1))) ∧
  (is_geometric_sequence (fun n => 1 / a n)) ∧
  ¬(∀ (a : ℕ → ℝ), is_geometric_sequence a → is_geometric_sequence (fun n => Real.log (a n ^ 2))) :=
by sorry

end geometric_sequence_properties_l2457_245706


namespace range_of_f_l2457_245750

def f (x : ℤ) : ℤ := x^2 - 2*x

def domain : Set ℤ := {x : ℤ | -2 ≤ x ∧ x ≤ 4}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3, 8} := by
  sorry

end range_of_f_l2457_245750


namespace bread_duration_l2457_245783

-- Define the parameters
def household_members : ℕ := 4
def breakfast_slices : ℕ := 3
def snack_slices : ℕ := 2
def slices_per_loaf : ℕ := 12
def number_of_loaves : ℕ := 5

-- Define the theorem
theorem bread_duration : 
  let total_slices := number_of_loaves * slices_per_loaf
  let daily_consumption := household_members * (breakfast_slices + snack_slices)
  total_slices / daily_consumption = 3 := by
  sorry


end bread_duration_l2457_245783


namespace sin_product_theorem_l2457_245769

theorem sin_product_theorem :
  Real.sin (10 * Real.pi / 180) *
  Real.sin (30 * Real.pi / 180) *
  Real.sin (50 * Real.pi / 180) *
  Real.sin (70 * Real.pi / 180) = 1 / 16 := by
  sorry

end sin_product_theorem_l2457_245769


namespace rationalized_factor_simplify_fraction_special_sqrt_l2457_245748

-- Part 1
theorem rationalized_factor (x : ℝ) : 
  (3 + Real.sqrt 11) * (3 - Real.sqrt 11) = -2 :=
sorry

-- Part 2
theorem simplify_fraction (b : ℝ) (h1 : b ≥ 0) (h2 : b ≠ 1) : 
  (1 - b) / (1 - Real.sqrt b) = 1 + Real.sqrt b :=
sorry

-- Part 3
theorem special_sqrt (a b : ℝ) 
  (ha : a = 1 / (Real.sqrt 3 - 2)) 
  (hb : b = 1 / (Real.sqrt 3 + 2)) : 
  Real.sqrt (a^2 + b^2 + 2) = 4 :=
sorry

end rationalized_factor_simplify_fraction_special_sqrt_l2457_245748


namespace perpendicular_lines_sum_l2457_245789

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- Definition of Line 1: ax + 4y - 2 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y - 2 = 0

/-- Definition of Line 2: 2x - 5y + b = 0 -/
def line2 (b : ℝ) (x y : ℝ) : Prop := 2 * x - 5 * y + b = 0

/-- The foot of the perpendicular (1, c) lies on both lines -/
def foot_on_lines (a b c : ℝ) : Prop := line1 a 1 c ∧ line2 b 1 c

theorem perpendicular_lines_sum (a b c : ℝ) : 
  perpendicular (-a/4) (2/5) → foot_on_lines a b c → a + b + c = -4 := by
  sorry

end perpendicular_lines_sum_l2457_245789


namespace button_to_magnet_ratio_l2457_245754

/-- Represents the number of earrings in a set -/
def earrings_per_set : ℕ := 2

/-- Represents the number of sets Rebecca wants to make -/
def sets : ℕ := 4

/-- Represents the total number of gemstones needed -/
def total_gemstones : ℕ := 24

/-- Represents the number of magnets used in each earring -/
def magnets_per_earring : ℕ := 2

/-- Represents the ratio of gemstones to buttons -/
def gemstone_to_button_ratio : ℕ := 3

/-- Theorem stating the ratio of buttons to magnets for each earring -/
theorem button_to_magnet_ratio :
  let total_earrings := sets * earrings_per_set
  let total_buttons := total_gemstones / gemstone_to_button_ratio
  let buttons_per_earring := total_buttons / total_earrings
  (buttons_per_earring : ℚ) / magnets_per_earring = 1 / 2 := by
  sorry

end button_to_magnet_ratio_l2457_245754


namespace inner_circle_radius_l2457_245722

theorem inner_circle_radius (s : ℝ) (h : s = 4) :
  let quarter_circle_radius := s / 2
  let square_diagonal := s * Real.sqrt 2
  let center_to_corner := square_diagonal / 2
  let r := (center_to_corner ^ 2 - quarter_circle_radius ^ 2).sqrt + quarter_circle_radius - center_to_corner
  r = 1 + Real.sqrt 3 := by sorry

end inner_circle_radius_l2457_245722


namespace exists_number_divisible_by_5_1000_without_zeros_l2457_245751

theorem exists_number_divisible_by_5_1000_without_zeros : 
  ∃ n : ℕ, (5^1000 ∣ n) ∧ (∀ d : ℕ, d < 10 → d ≠ 0 → ∃ k : ℕ, n / 10^k % 10 = d) :=
sorry

end exists_number_divisible_by_5_1000_without_zeros_l2457_245751


namespace sector_area_l2457_245705

/-- Given a circular sector where the arc length is 4 cm and the central angle is 2 radians,
    prove that the area of the sector is 4 cm². -/
theorem sector_area (s : ℝ) (θ : ℝ) (A : ℝ) : 
  s = 4 → θ = 2 → s = 2 * θ → A = (1/2) * (s/θ)^2 * θ → A = 4 := by
  sorry

end sector_area_l2457_245705


namespace simplify_expression_l2457_245787

theorem simplify_expression (x : ℝ) : 4*x + 9*x^2 + 8 - (5 - 4*x - 9*x^2) = 18*x^2 + 8*x + 3 := by
  sorry

end simplify_expression_l2457_245787


namespace max_sum_of_digits_24hour_watch_l2457_245791

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours ≤ 23
  minutes_valid : minutes ≤ 59
  seconds_valid : seconds ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of all digits in a Time24 -/
def totalSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes + sumOfDigits t.seconds

/-- The theorem to be proved -/
theorem max_sum_of_digits_24hour_watch :
  ∃ (t : Time24), ∀ (t' : Time24), totalSumOfDigits t' ≤ totalSumOfDigits t ∧ totalSumOfDigits t = 38 := by
  sorry

end max_sum_of_digits_24hour_watch_l2457_245791


namespace travelers_getting_off_subway_l2457_245799

/-- The number of stations ahead -/
def num_stations : ℕ := 10

/-- The number of travelers -/
def num_travelers : ℕ := 3

/-- The total number of ways travelers can get off at any station -/
def total_ways : ℕ := num_stations ^ num_travelers

/-- The number of ways all travelers can get off at the same station -/
def same_station_ways : ℕ := num_stations

/-- The number of ways travelers can get off without all disembarking at the same station -/
def different_station_ways : ℕ := total_ways - same_station_ways

theorem travelers_getting_off_subway :
  different_station_ways = 990 := by sorry

end travelers_getting_off_subway_l2457_245799


namespace solve_equation_l2457_245785

theorem solve_equation (x : ℚ) : 15 * x = 165 ↔ x = 11 := by
  sorry

end solve_equation_l2457_245785


namespace equation_graph_is_two_lines_l2457_245736

-- Define the set of points satisfying the original equation
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = 3 * p.1^2 + p.2^2}

-- Define the two lines
def L1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}
def L2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -p.1}

-- State the theorem
theorem equation_graph_is_two_lines : S = L1 ∪ L2 := by
  sorry

end equation_graph_is_two_lines_l2457_245736


namespace complex_addition_simplification_l2457_245713

theorem complex_addition_simplification :
  (4 : ℂ) + 3*I + (-7 : ℂ) + 5*I = -3 + 8*I :=
by sorry

end complex_addition_simplification_l2457_245713


namespace final_values_correct_l2457_245781

/-- Represents the state of variables a, b, and c -/
structure State where
  a : Int
  b : Int
  c : Int

/-- Applies the assignments a = b, b = c, c = a to a given state -/
def applyAssignments (s : State) : State :=
  { a := s.b, b := s.c, c := s.b }

/-- The theorem statement -/
theorem final_values_correct :
  let initialState : State := { a := 3, b := -5, c := 8 }
  let finalState := applyAssignments initialState
  finalState.a = -5 ∧ finalState.b = 8 ∧ finalState.c = -5 := by
  sorry


end final_values_correct_l2457_245781


namespace final_result_proof_l2457_245752

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 740) : 
  (chosen_number / 4 : ℚ) - 175 = 10 := by
  sorry

end final_result_proof_l2457_245752


namespace profit_percentage_is_fifty_percent_l2457_245777

/-- Calculates the profit percentage given the costs and selling price -/
def profit_percentage (purchase_price repair_cost transport_cost selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percentage is 50% given the specific costs and selling price -/
theorem profit_percentage_is_fifty_percent :
  profit_percentage 10000 5000 1000 24000 = 50 := by
  sorry

end profit_percentage_is_fifty_percent_l2457_245777


namespace min_value_of_f_l2457_245719

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

-- Theorem stating that the minimum value of f(x) is 0
theorem min_value_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≥ f x₀ ∧ f x₀ = 0 :=
by sorry

end min_value_of_f_l2457_245719


namespace power_set_intersection_nonempty_l2457_245775

theorem power_set_intersection_nonempty :
  ∃ (A B : Set α), (A ∩ B).Nonempty ∧ (𝒫 A ∩ 𝒫 B).Nonempty :=
sorry

end power_set_intersection_nonempty_l2457_245775


namespace line_m_equation_l2457_245774

-- Define the xy-plane
def xy_plane : Set (ℝ × ℝ) := Set.univ

-- Define lines ℓ and m
def line_ℓ : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 = 0}
def line_m : Set (ℝ × ℝ) := {p : ℝ × ℝ | 7 * p.1 - p.2 = 0}

-- Define points
def Q : ℝ × ℝ := (-3, 2)
def Q'' : ℝ × ℝ := (-4, -3)

-- Define the reflection operation (as a placeholder, actual implementation not provided)
def reflect (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem line_m_equation :
  line_ℓ ⊆ xy_plane ∧
  line_m ⊆ xy_plane ∧
  line_ℓ ≠ line_m ∧
  (0, 0) ∈ line_ℓ ∩ line_m ∧
  Q ∈ xy_plane ∧
  Q'' ∈ xy_plane ∧
  reflect (reflect Q line_ℓ) line_m = Q'' →
  line_m = {p : ℝ × ℝ | 7 * p.1 - p.2 = 0} :=
by sorry

end line_m_equation_l2457_245774


namespace complex_equation_solution_l2457_245721

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 / (1 + Complex.I) → z = -1 - Complex.I := by
  sorry

end complex_equation_solution_l2457_245721


namespace town_population_problem_l2457_245732

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 1500) * 85 / 100 : ℕ) = original_population - 45) → 
  original_population = 8800 := by
  sorry

end town_population_problem_l2457_245732


namespace compound_hydrogen_count_l2457_245700

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight oxygenWeight hydrogenWeight : ℕ) : ℕ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem compound_hydrogen_count :
  ∀ (c : Compound),
    c.carbon = 3 →
    c.oxygen = 1 →
    molecularWeight c 12 16 1 = 58 →
    c.hydrogen = 6 :=
by sorry

end compound_hydrogen_count_l2457_245700


namespace rest_area_location_l2457_245725

theorem rest_area_location (city_a city_b rest_area : ℝ) : 
  city_a = 50 →
  city_b = 230 →
  rest_area - city_a = (5/8) * (city_b - city_a) →
  rest_area = 162.5 := by
sorry

end rest_area_location_l2457_245725


namespace sqrt_equation_solution_l2457_245768

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
sorry

end sqrt_equation_solution_l2457_245768


namespace negation_of_universal_proposition_l2457_245760

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l2457_245760


namespace sum_of_first_12_terms_of_arithmetic_sequence_l2457_245770

/-- Given the sum of the first 4 terms and the sum of the first 8 terms of an arithmetic sequence,
    this theorem proves that the sum of the first 12 terms is 210. -/
theorem sum_of_first_12_terms_of_arithmetic_sequence 
  (S₄ S₈ : ℕ) (h₁ : S₄ = 30) (h₂ : S₈ = 100) : ∃ S₁₂ : ℕ, S₁₂ = 210 :=
by
  sorry


end sum_of_first_12_terms_of_arithmetic_sequence_l2457_245770


namespace collinear_probability_5x4_l2457_245707

/-- Represents a rectangular array of dots. -/
structure DotArray :=
  (rows : ℕ)
  (cols : ℕ)

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of collinear sets of 4 dots in a 5x4 array. -/
def collinearSets (arr : DotArray) : ℕ := arr.cols * choose arr.rows 4

/-- The total number of ways to choose 4 dots from the array. -/
def totalChoices (arr : DotArray) : ℕ := choose (arr.rows * arr.cols) 4

/-- The probability of choosing 4 collinear dots. -/
def collinearProbability (arr : DotArray) : ℚ :=
  collinearSets arr / totalChoices arr

/-- Theorem: The probability of choosing 4 collinear dots in a 5x4 array is 4/969. -/
theorem collinear_probability_5x4 :
  collinearProbability ⟨5, 4⟩ = 4 / 969 := by
  sorry

end collinear_probability_5x4_l2457_245707


namespace problem_solution_l2457_245723

noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x) - 2 * x

noncomputable def g (b : ℝ) (x : ℝ) := f (2 * x) - 4 * b * f x

theorem problem_solution :
  (∀ x : ℝ, (deriv f) x ≥ 0) ∧
  (∃ b_max : ℝ, b_max = 2 ∧ ∀ b : ℝ, (∀ x : ℝ, x > 0 → g b x > 0) → b ≤ b_max) ∧
  (0.693 < Real.log 2 ∧ Real.log 2 < 0.694) :=
by sorry

end problem_solution_l2457_245723


namespace reciprocal_inequality_l2457_245734

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end reciprocal_inequality_l2457_245734


namespace antimatter_prescription_fulfillment_l2457_245756

theorem antimatter_prescription_fulfillment :
  ∃ (x y z : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧
  (11 : ℝ) * x + 1.1 * y + 0.11 * z = 20.13 := by
  sorry

end antimatter_prescription_fulfillment_l2457_245756


namespace extreme_points_condition_one_zero_point_no_zero_points_l2457_245782

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (a * x) / (x + 1)

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x + a / ((x + 1) ^ 2)

-- Theorem for the number of extreme points
theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) ↔ a < -4 :=
sorry

-- Theorem for the number of zero points when a ≥ -4
theorem one_zero_point (a : ℝ) (h : a ≥ -4) :
  ∃! x : ℝ, f a x = 0 :=
sorry

-- Theorem for the number of zero points when a < -4
theorem no_zero_points (a : ℝ) (h : a < -4) :
  ¬∃ x : ℝ, f a x = 0 :=
sorry

end extreme_points_condition_one_zero_point_no_zero_points_l2457_245782


namespace unique_cube_ending_in_nine_l2457_245794

theorem unique_cube_ending_in_nine :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 1000 ≤ n^3 ∧ n^3 < 10000 ∧ n^3 % 10 = 9 ∧ n = 19 := by
  sorry

end unique_cube_ending_in_nine_l2457_245794


namespace parabola_intersections_and_point_position_l2457_245731

/-- Represents a parabola of the form y = x^2 + px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- A point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about parabola intersections and point position -/
theorem parabola_intersections_and_point_position 
  (parabola : Parabola) 
  (M : Point) 
  (h_below_x_axis : M.y < 0) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + parabola.p * x₁ + parabola.q = 0) ∧ 
    (x₂^2 + parabola.p * x₂ + parabola.q = 0) ∧ 
    (x₁ < x₂) ∧
    (x₁ < M.x) ∧ (M.x < x₂) := by
  sorry


end parabola_intersections_and_point_position_l2457_245731


namespace exponent_division_l2457_245753

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end exponent_division_l2457_245753


namespace success_rate_paradox_l2457_245755

structure Player :=
  (name : String)
  (attempts_season1 : ℕ)
  (successes_season1 : ℕ)
  (attempts_season2 : ℕ)
  (successes_season2 : ℕ)

def success_rate (attempts : ℕ) (successes : ℕ) : ℚ :=
  if attempts = 0 then 0 else (successes : ℚ) / (attempts : ℚ)

def combined_success_rate (p : Player) : ℚ :=
  success_rate (p.attempts_season1 + p.attempts_season2) (p.successes_season1 + p.successes_season2)

theorem success_rate_paradox (p1 p2 : Player) :
  (success_rate p1.attempts_season1 p1.successes_season1 > success_rate p2.attempts_season1 p2.successes_season1) ∧
  (success_rate p1.attempts_season2 p1.successes_season2 > success_rate p2.attempts_season2 p2.successes_season2) ∧
  (combined_success_rate p1 < combined_success_rate p2) :=
sorry

end success_rate_paradox_l2457_245755


namespace cars_per_row_in_section_G_l2457_245710

/-- The number of rows in Section G -/
def section_G_rows : ℕ := 15

/-- The number of rows in Section H -/
def section_H_rows : ℕ := 20

/-- The number of cars per row in Section H -/
def section_H_cars_per_row : ℕ := 9

/-- The number of cars Nate can walk past per minute -/
def cars_per_minute : ℕ := 11

/-- The number of minutes Nate spent searching -/
def search_time : ℕ := 30

/-- The number of cars per row in Section G -/
def section_G_cars_per_row : ℕ := 10

theorem cars_per_row_in_section_G :
  section_G_cars_per_row = 
    (cars_per_minute * search_time - section_H_rows * section_H_cars_per_row) / section_G_rows :=
by sorry

end cars_per_row_in_section_G_l2457_245710


namespace family_weight_problem_l2457_245717

theorem family_weight_problem (total_weight daughter_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : daughter_weight = 42) :
  ∃ (grandmother_weight child_weight : ℝ),
    grandmother_weight + daughter_weight + child_weight = total_weight ∧
    child_weight = (1 / 5) * grandmother_weight ∧
    daughter_weight + child_weight = 60 := by
  sorry

end family_weight_problem_l2457_245717


namespace tan_beta_value_l2457_245745

open Real

theorem tan_beta_value (α β : ℝ) 
  (h1 : tan (α + β) = 3) 
  (h2 : tan (α + π/4) = 2) : 
  tan β = 2 := by
sorry

end tan_beta_value_l2457_245745


namespace max_intersections_nested_polygons_l2457_245776

/-- Represents a convex polygon -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two nested convex polygons -/
structure NestedPolygons where
  inner : ConvexPolygon
  outer : ConvexPolygon
  nested : Bool
  no_shared_segments : Bool

/-- Calculates the maximum number of intersection points between two nested convex polygons -/
def max_intersections (np : NestedPolygons) : ℕ :=
  np.inner.sides * np.outer.sides

/-- Theorem stating the maximum number of intersections for the given configuration -/
theorem max_intersections_nested_polygons :
  ∀ (np : NestedPolygons),
    np.inner.sides = 5 →
    np.outer.sides = 8 →
    np.inner.convex = true →
    np.outer.convex = true →
    np.nested = true →
    np.no_shared_segments = true →
    max_intersections np = 40 :=
by sorry

end max_intersections_nested_polygons_l2457_245776


namespace sum_of_satisfying_numbers_is_34_l2457_245702

def satisfies_condition (n : ℕ) : Prop :=
  1.5 * (n : ℝ) - 5.5 > 4.5

def sum_of_satisfying_numbers : ℕ :=
  (Finset.range 4).sum (fun i => i + 7)

theorem sum_of_satisfying_numbers_is_34 :
  sum_of_satisfying_numbers = 34 ∧
  ∀ n, 7 ≤ n → n ≤ 10 → satisfies_condition n :=
sorry

end sum_of_satisfying_numbers_is_34_l2457_245702


namespace expression_simplification_l2457_245793

theorem expression_simplification :
  ((3 + 4 + 6 + 7) / 4) + ((2 * 6 + 10) / 4) = 10.5 := by
  sorry

end expression_simplification_l2457_245793


namespace chromatic_number_of_our_graph_l2457_245772

/-- Represents a vertex in the graph -/
inductive Vertex : Type
| x : Vertex
| y : Vertex
| z : Vertex
| w : Vertex
| v : Vertex
| u : Vertex

/-- The graph structure -/
def Graph : Type := Vertex → Vertex → Prop

/-- The degree of a vertex in the graph -/
def degree (G : Graph) (v : Vertex) : ℕ := sorry

/-- Predicate to check if three vertices form a triangle in the graph -/
def isTriangle (G : Graph) (a b c : Vertex) : Prop := sorry

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph) : ℕ := sorry

/-- Our specific graph -/
def ourGraph : Graph := sorry

theorem chromatic_number_of_our_graph :
  let G := ourGraph
  (degree G Vertex.x = 5) →
  (degree G Vertex.z = 4) →
  (degree G Vertex.y = 3) →
  isTriangle G Vertex.x Vertex.y Vertex.z →
  chromaticNumber G = 3 := by sorry

end chromatic_number_of_our_graph_l2457_245772


namespace sum_of_roots_quadratic_l2457_245780

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ - 4 = 0) → (x₂^2 - 3*x₂ - 4 = 0) → x₁ + x₂ = 3 := by
  sorry

end sum_of_roots_quadratic_l2457_245780


namespace complex_exponential_product_l2457_245764

theorem complex_exponential_product (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = -1/3 + 4/5 * Complex.I →
  (Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β)) *
  (Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β)) = 169/225 := by
  sorry

end complex_exponential_product_l2457_245764


namespace final_balance_is_214_12_l2457_245798

/-- Calculates the credit card balance after five months given the initial balance and monthly transactions. -/
def creditCardBalance (initialBalance : ℚ) 
  (month1Interest : ℚ)
  (month2Spent month2Payment month2Interest : ℚ)
  (month3Spent month3Payment month3Interest : ℚ)
  (month4Spent month4Payment month4Interest : ℚ)
  (month5Spent month5Payment month5Interest : ℚ) : ℚ :=
  let balance1 := initialBalance * (1 + month1Interest)
  let balance2 := (balance1 + month2Spent - month2Payment) * (1 + month2Interest)
  let balance3 := (balance2 + month3Spent - month3Payment) * (1 + month3Interest)
  let balance4 := (balance3 + month4Spent - month4Payment) * (1 + month4Interest)
  let balance5 := (balance4 + month5Spent - month5Payment) * (1 + month5Interest)
  balance5

/-- Theorem stating that the credit card balance after five months is $214.12 given the specific transactions. -/
theorem final_balance_is_214_12 : 
  creditCardBalance 50 0.2 20 15 0.18 30 5 0.22 25 20 0.15 40 10 0.2 = 214.12 := by
  sorry

end final_balance_is_214_12_l2457_245798


namespace inequality_and_equality_condition_l2457_245788

theorem inequality_and_equality_condition (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  ((1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b)) ↔ (a = b ∧ a < 1)) :=
by sorry

end inequality_and_equality_condition_l2457_245788


namespace smallest_number_of_sweets_l2457_245740

theorem smallest_number_of_sweets (x : ℕ) : 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 8 = 7 ∧ 
  x % 9 = 8 ∧ 
  (∀ y : ℕ, y > 0 → y % 6 = 5 → y % 8 = 7 → y % 9 = 8 → x ≤ y) → 
  x = 71 := by
sorry

end smallest_number_of_sweets_l2457_245740


namespace correct_participants_cars_needed_rental_plans_and_min_cost_l2457_245766

/-- Represents the number of teachers -/
def teachers : ℕ := 6

/-- Represents the number of students -/
def students : ℕ := 234

/-- Represents the total number of participants -/
def total_participants : ℕ := teachers + students

/-- Represents the capacity of bus A -/
def bus_A_capacity : ℕ := 45

/-- Represents the capacity of bus B -/
def bus_B_capacity : ℕ := 30

/-- Represents the rental cost of bus A -/
def bus_A_cost : ℕ := 400

/-- Represents the rental cost of bus B -/
def bus_B_cost : ℕ := 280

/-- Represents the total rental cost limit -/
def total_cost_limit : ℕ := 2300

/-- Theorem stating the correctness of the number of teachers and students -/
theorem correct_participants : teachers = 6 ∧ students = 234 ∧
  38 * teachers + 6 = students ∧ 40 * teachers - 6 = students := by sorry

/-- Theorem stating the number of cars needed -/
theorem cars_needed : ∃ (n : ℕ), n = 6 ∧ 
  n * bus_A_capacity ≥ total_participants ∧
  n ≥ teachers := by sorry

/-- Theorem stating the number of rental car plans and minimum cost -/
theorem rental_plans_and_min_cost : 
  ∃ (plans : ℕ) (min_cost : ℕ), plans = 2 ∧ min_cost = 2160 ∧
  ∀ (x : ℕ), 4 ≤ x ∧ x ≤ 5 →
    x * bus_A_capacity + (6 - x) * bus_B_capacity ≥ total_participants ∧
    x * bus_A_cost + (6 - x) * bus_B_cost ≤ total_cost_limit ∧
    (x = 4 → x * bus_A_cost + (6 - x) * bus_B_cost = min_cost) := by sorry

end correct_participants_cars_needed_rental_plans_and_min_cost_l2457_245766


namespace mikes_muffins_l2457_245711

/-- The number of muffins in a dozen -/
def dozen : ℕ := 12

/-- The number of boxes Mike needs to pack all his muffins -/
def boxes : ℕ := 8

/-- Mike's muffins theorem -/
theorem mikes_muffins : dozen * boxes = 96 := by
  sorry

end mikes_muffins_l2457_245711


namespace work_done_by_resistive_force_l2457_245701

def mass : Real := 0.01
def initial_velocity : Real := 400
def final_velocity : Real := 100

def kinetic_energy (m : Real) (v : Real) : Real :=
  0.5 * m * v^2

def work_done (m : Real) (v1 : Real) (v2 : Real) : Real :=
  kinetic_energy m v1 - kinetic_energy m v2

theorem work_done_by_resistive_force :
  work_done mass initial_velocity final_velocity = 750 := by
  sorry

end work_done_by_resistive_force_l2457_245701


namespace odd_expressions_l2457_245795

-- Define positive odd integers
def is_positive_odd (n : ℤ) : Prop := n > 0 ∧ ∃ k : ℤ, n = 2*k + 1

-- Theorem statement
theorem odd_expressions (p q : ℤ) 
  (hp : is_positive_odd p) (hq : is_positive_odd q) : 
  ∃ m n : ℤ, p * q + 2 = 2*m + 1 ∧ p^3 * q + q^2 = 2*n + 1 :=
sorry

end odd_expressions_l2457_245795


namespace friends_assignment_l2457_245720

/-- The number of ways to assign friends to rooms -/
def assignFriends (n : ℕ) (m : ℕ) (maxPerRoom : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of assignments for the given problem -/
theorem friends_assignment :
  assignFriends 7 7 3 = 15120 := by
  sorry

end friends_assignment_l2457_245720


namespace task_completion_proof_l2457_245762

def task_completion (x : ℝ) : Prop :=
  let a := x
  let b := x + 6
  let c := x + 9
  (3 / a + 4 / b = 9 / c) ∧ (a = 18) ∧ (b = 24) ∧ (c = 27)

theorem task_completion_proof : ∃ x : ℝ, task_completion x := by
  sorry

end task_completion_proof_l2457_245762


namespace ant_movement_probability_l2457_245738

structure Octahedron where
  middleVertices : Finset Nat
  topVertex : Nat
  bottomVertex : Nat

def moveToMiddle (o : Octahedron) (start : Nat) : Finset Nat :=
  o.middleVertices.filter (λ v => v ≠ start)

def moveFromMiddle (o : Octahedron) (middle : Nat) : Finset Nat :=
  insert o.bottomVertex (insert o.topVertex (o.middleVertices.filter (λ v => v ≠ middle)))

theorem ant_movement_probability (o : Octahedron) (start : Nat) :
  start ∈ o.middleVertices →
  (1 : ℚ) / 4 = (moveToMiddle o start).sum (λ a =>
    (1 : ℚ) / (moveToMiddle o start).card *
    (1 : ℚ) / (moveFromMiddle o a).card *
    if o.bottomVertex ∈ moveFromMiddle o a then 1 else 0) :=
sorry

end ant_movement_probability_l2457_245738
