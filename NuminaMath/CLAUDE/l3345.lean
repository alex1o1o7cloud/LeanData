import Mathlib

namespace mollys_bike_age_l3345_334563

/-- Molly's bike riding problem -/
theorem mollys_bike_age : 
  ∀ (miles_per_day : ℕ) (age_stopped : ℕ) (total_miles : ℕ) (days_per_year : ℕ),
  miles_per_day = 3 →
  age_stopped = 16 →
  total_miles = 3285 →
  days_per_year = 365 →
  age_stopped - (total_miles / miles_per_day / days_per_year) = 13 := by
sorry

end mollys_bike_age_l3345_334563


namespace circle_area_through_points_l3345_334508

/-- The area of a circle with center P(2, -1) passing through Q(-4, 6) is 85π -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (2, -1)
  let Q : ℝ × ℝ := (-4, 6)
  let r := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  π * r^2 = 85 * π := by sorry

end circle_area_through_points_l3345_334508


namespace cello_count_l3345_334536

/-- Given a music store with cellos and violas, prove the number of cellos. -/
theorem cello_count (violas : ℕ) (matching_pairs : ℕ) (probability : ℚ) (cellos : ℕ) : 
  violas = 600 →
  matching_pairs = 70 →
  probability = 70 / (cellos * 600) →
  probability = 0.00014583333333333335 →
  cellos = 800 := by
sorry

#eval (70 : ℚ) / (800 * 600)  -- To verify the probability

end cello_count_l3345_334536


namespace birthday_celebration_attendance_l3345_334539

theorem birthday_celebration_attendance 
  (total_guests : ℕ) 
  (women_percentage men_percentage : ℚ) 
  (men_left_fraction women_left_fraction : ℚ) 
  (children_left : ℕ) 
  (h1 : total_guests = 750)
  (h2 : women_percentage = 432 / 1000)
  (h3 : men_percentage = 314 / 1000)
  (h4 : men_left_fraction = 5 / 12)
  (h5 : women_left_fraction = 7 / 15)
  (h6 : children_left = 19) :
  ∃ (women_count men_count children_count : ℕ),
    women_count + men_count + children_count = total_guests ∧
    women_count = ⌊women_percentage * total_guests⌋ ∧
    men_count = ⌈men_percentage * total_guests⌉ ∧
    children_count = total_guests - women_count - men_count ∧
    total_guests - 
      (⌊men_left_fraction * men_count⌋ + 
       ⌊women_left_fraction * women_count⌋ + 
       children_left) = 482 := by
  sorry


end birthday_celebration_attendance_l3345_334539


namespace special_triangle_area_squared_l3345_334578

/-- An equilateral triangle with vertices on the hyperbola xy = 4 and centroid at a vertex of the hyperbola -/
structure SpecialTriangle where
  -- The hyperbola equation
  hyperbola : ℝ → ℝ → Prop
  hyperbola_def : hyperbola = fun x y ↦ x * y = 4

  -- The triangle is equilateral
  is_equilateral : Prop

  -- Vertices lie on the hyperbola
  vertices_on_hyperbola : Prop

  -- Centroid is at a vertex of the hyperbola
  centroid_on_hyperbola : Prop

/-- The square of the area of the special triangle is 3888 -/
theorem special_triangle_area_squared (t : SpecialTriangle) : 
  ∃ (area : ℝ), area^2 = 3888 := by sorry

end special_triangle_area_squared_l3345_334578


namespace geometric_progression_fourth_term_l3345_334505

theorem geometric_progression_fourth_term (a : ℝ) (r : ℝ) :
  (∃ (b c : ℝ), a = 3^(3/4) ∧ b = 3^(2/4) ∧ c = 3^(1/4) ∧ 
   b / a = c / b) → 
  c^2 / b = 1 :=
by sorry

end geometric_progression_fourth_term_l3345_334505


namespace previous_painting_price_l3345_334515

/-- Proves the price of a previous painting given the price of the most recent painting and the relationship between the two prices. -/
theorem previous_painting_price (recent_price : ℝ) (h1 : recent_price = 49000) 
  (h2 : recent_price = 3.5 * previous_price - 1000) : previous_price = 14285.71 := by
  sorry

end previous_painting_price_l3345_334515


namespace quadratic_root_sum_l3345_334524

theorem quadratic_root_sum (m n : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I * Real.sqrt 3)^2 + m * (1 - Complex.I * Real.sqrt 3) + n = 0 →
  m + n = 2 := by
sorry

end quadratic_root_sum_l3345_334524


namespace points_four_units_from_negative_one_l3345_334546

theorem points_four_units_from_negative_one : 
  ∀ x : ℝ, abs (x - (-1)) = 4 ↔ x = 3 ∨ x = -5 := by
  sorry

end points_four_units_from_negative_one_l3345_334546


namespace sum_of_coefficients_l3345_334586

-- Define the polynomial expression
def f (d : ℝ) : ℝ := -(5 - d) * (d + 2 * (5 - d))

-- Define the expanded form of the polynomial
def expanded_form (d : ℝ) : ℝ := -d^2 + 15*d - 50

-- Theorem statement
theorem sum_of_coefficients :
  (∀ d, f d = expanded_form d) →
  (-1 : ℝ) + 15 + (-50) = -36 := by sorry

end sum_of_coefficients_l3345_334586


namespace common_internal_tangent_length_bound_l3345_334572

/-- Two circles touching the sides of an angle but not each other -/
structure AngleTouchingCircles where
  R : ℝ
  r : ℝ
  h1 : R > r
  h2 : R > 0
  h3 : r > 0
  PQ : ℝ
  h4 : PQ > 0

/-- The length of the common internal tangent segment is greater than twice the geometric mean of the radii -/
theorem common_internal_tangent_length_bound (c : AngleTouchingCircles) : 
  c.PQ > 2 * Real.sqrt (c.R * c.r) := by
  sorry

end common_internal_tangent_length_bound_l3345_334572


namespace drought_periods_correct_max_water_storage_volume_l3345_334506

noncomputable def v (t : ℝ) : ℝ :=
  if 0 < t ∧ t ≤ 9 then
    (1 / 240) * (-(t^2) + 15*t - 51) * Real.exp t + 50
  else if 9 < t ∧ t ≤ 12 then
    4 * (t - 9) * (3*t - 41) + 50
  else
    0

def isDroughtPeriod (t : ℝ) : Prop := v t < 50

def monthToPeriod (m : ℕ) : Set ℝ := {t | m - 1 < t ∧ t ≤ m}

def droughtMonths : Set ℕ := {1, 2, 3, 4, 5, 10, 11, 12}

theorem drought_periods_correct (m : ℕ) (hm : m ∈ droughtMonths) :
  ∀ t ∈ monthToPeriod m, isDroughtPeriod t :=
sorry

theorem max_water_storage_volume :
  ∃ t ∈ Set.Icc (0 : ℝ) 12, v t = 150 ∧ ∀ s ∈ Set.Icc (0 : ℝ) 12, v s ≤ v t :=
sorry

axiom e_cubed_eq_20 : Real.exp 3 = 20

end drought_periods_correct_max_water_storage_volume_l3345_334506


namespace polynomial_remainder_theorem_l3345_334577

theorem polynomial_remainder_theorem (f : ℝ → ℝ) (a b c p q r l m n : ℝ) 
  (h_abc : a * b * c ≠ 0)
  (h_rem1 : ∀ x, ∃ k, f x = k * (x - a) * (x - b) + p * x + l)
  (h_rem2 : ∀ x, ∃ k, f x = k * (x - b) * (x - c) + q * x + m)
  (h_rem3 : ∀ x, ∃ k, f x = k * (x - c) * (x - a) + r * x + n) :
  l * (1/a - 1/b) + m * (1/b - 1/c) + n * (1/c - 1/a) = 0 := by
  sorry

end polynomial_remainder_theorem_l3345_334577


namespace triangle_isosceles_or_right_l3345_334590

theorem triangle_isosceles_or_right 
  (a b c : ℝ) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_lengths_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h : a^2 * c^2 - b^2 * c^2 = a^4 - b^4) : 
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end triangle_isosceles_or_right_l3345_334590


namespace billys_age_l3345_334540

theorem billys_age (B J A : ℕ) 
  (h1 : B = 3 * J)
  (h2 : J = A / 2)
  (h3 : B + J + A = 90) :
  B = 45 := by sorry

end billys_age_l3345_334540


namespace specific_polygon_triangulation_l3345_334526

/-- Represents a convex polygon with additional internal points -/
structure EnhancedPolygon where
  sides : ℕ
  internal_points : ℕ
  no_collinear_triples : Prop

/-- Represents the triangulation of an EnhancedPolygon -/
def triangulation (p : EnhancedPolygon) : ℕ := sorry

/-- The theorem stating the number of triangles in the specific polygon -/
theorem specific_polygon_triangulation :
  ∀ (p : EnhancedPolygon),
    p.sides = 1000 →
    p.internal_points = 500 →
    p.no_collinear_triples →
    triangulation p = 1998 := by sorry

end specific_polygon_triangulation_l3345_334526


namespace no_always_largest_l3345_334597

theorem no_always_largest (a b c d : ℝ) (h : a - 2 = b + 3 ∧ a - 2 = c * 2 ∧ a - 2 = d + 5) :
  ¬(∀ x y : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≤ y) :=
by sorry

end no_always_largest_l3345_334597


namespace quadratic_factorization_sum_l3345_334513

theorem quadratic_factorization_sum (p q r : ℤ) : 
  (∀ x, x^2 + 19*x + 88 = (x + p) * (x + q)) →
  (∀ x, x^2 - 23*x + 132 = (x - q) * (x - r)) →
  p + q + r = 31 := by
sorry

end quadratic_factorization_sum_l3345_334513


namespace range_of_a_l3345_334552

-- Define p as a predicate on m
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧ 
  ∃ (c : ℝ), c > 0 ∧ (∀ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 → y^2 ≤ c^2)

-- Define q as a predicate on m and a
def q (m a : ℝ) : Prop := m^2 - (2*a + 1)*m + a^2 + a < 0

-- State the theorem
theorem range_of_a : 
  (∀ m : ℝ, p m → ∃ a : ℝ, q m a) → 
  ∃ a : ℝ, 1/2 ≤ a ∧ a ≤ 1 ∧ 
    (∀ b : ℝ, (∀ m : ℝ, p m → q m b) → 1/2 ≤ b ∧ b ≤ 1) :=
by sorry

end range_of_a_l3345_334552


namespace budget_allocation_home_electronics_l3345_334565

theorem budget_allocation_home_electronics (total_degrees : ℝ) 
  (microphotonics_percent : ℝ) (food_additives_percent : ℝ) 
  (genetically_modified_microorganisms_percent : ℝ) (industrial_lubricants_percent : ℝ) 
  (basic_astrophysics_degrees : ℝ) :
  total_degrees = 360 ∧ 
  microphotonics_percent = 13 ∧ 
  food_additives_percent = 15 ∧ 
  genetically_modified_microorganisms_percent = 29 ∧ 
  industrial_lubricants_percent = 8 ∧ 
  basic_astrophysics_degrees = 39.6 →
  (100 - (microphotonics_percent + food_additives_percent + 
    genetically_modified_microorganisms_percent + industrial_lubricants_percent + 
    (basic_astrophysics_degrees / total_degrees * 100))) = 24 := by
  sorry

end budget_allocation_home_electronics_l3345_334565


namespace reciprocal_problem_l3345_334532

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 5) : 200 * (1 / x) = 320 := by
  sorry

end reciprocal_problem_l3345_334532


namespace inequality_solution_range_inequality_equal_solution_sets_l3345_334520

-- Define the inequality
def inequality (m x : ℝ) : Prop := m * x - 3 > 2 * x + m

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ := {x | x < (m + 3) / (m - 2)}

-- Define the alternate inequality
def alt_inequality (x : ℝ) : Prop := 2 * x - 1 > 3 - x

theorem inequality_solution_range (m : ℝ) :
  (∀ x, inequality m x ↔ x ∈ solution_set m) → m < 2 := by sorry

theorem inequality_equal_solution_sets (m : ℝ) :
  (∀ x, inequality m x ↔ alt_inequality x) → m = 17 := by sorry

end inequality_solution_range_inequality_equal_solution_sets_l3345_334520


namespace monotonicity_intervals_when_a_2_range_of_a_with_extreme_point_l3345_334516

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*x + 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3

-- Part I: Monotonicity intervals when a = 2
theorem monotonicity_intervals_when_a_2 :
  let a := 2
  ∀ x : ℝ, 
    (x ≤ 2 - Real.sqrt 3 ∨ x ≥ 2 + Real.sqrt 3 → f' a x > 0) ∧
    (2 - Real.sqrt 3 < x ∧ x < 2 + Real.sqrt 3 → f' a x < 0) :=
sorry

-- Part II: Range of a when f(x) has at least one extreme value point in (2,3)
theorem range_of_a_with_extreme_point :
  ∀ a : ℝ,
    (∃ x : ℝ, 2 < x ∧ x < 3 ∧ f' a x = 0) →
    (5/4 < a ∧ a < 5/3) :=
sorry

end monotonicity_intervals_when_a_2_range_of_a_with_extreme_point_l3345_334516


namespace count_divides_sum_product_l3345_334545

def divides_sum_product (n : ℕ+) : Prop :=
  (n.val * (n.val + 1) / 2) ∣ (10 * n.val)

theorem count_divides_sum_product :
  ∃ (S : Finset ℕ+), (∀ n, n ∈ S ↔ divides_sum_product n) ∧ S.card = 5 :=
sorry

end count_divides_sum_product_l3345_334545


namespace inequality_holds_l3345_334561

theorem inequality_holds (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : 0 < c) (h4 : c < 1) :
  b * (a ^ c) < a * (b ^ c) := by
  sorry

end inequality_holds_l3345_334561


namespace seating_arrangements_count_l3345_334562

/-- Represents a seating arrangement for 3 people on 5 chairs -/
structure SeatingArrangement where
  seats : Fin 5 → Option (Fin 3)
  all_seated : ∀ p : Fin 3, ∃ s : Fin 5, seats s = some p
  no_sharing : ∀ s : Fin 5, ∀ p q : Fin 3, seats s = some p → seats s = some q → p = q
  ab_adjacent : ∃ s : Fin 5, (seats s = some 0 ∧ seats (s + 1) = some 1) ∨ (seats s = some 1 ∧ seats (s + 1) = some 0)
  not_all_adjacent : ¬∃ s : Fin 5, (seats s).isSome ∧ (seats (s + 1)).isSome ∧ (seats (s + 2)).isSome

/-- The number of valid seating arrangements -/
def num_seating_arrangements : ℕ := sorry

/-- Theorem stating that there are exactly 12 valid seating arrangements -/
theorem seating_arrangements_count : num_seating_arrangements = 12 := by sorry

end seating_arrangements_count_l3345_334562


namespace shelter_dogs_l3345_334533

theorem shelter_dogs (D C R P : ℕ) : 
  D * 7 = C * 15 →  -- Initial ratio of dogs to cats
  R * 5 = P * 9 →   -- Initial ratio of rabbits to parrots
  D * 11 = (C + 8) * 15 →  -- New ratio of dogs to cats after adding 8 cats
  (R + 6) * 5 = P * 7 →    -- New ratio of rabbits to parrots after adding 6 rabbits
  D = 30 :=
by sorry

end shelter_dogs_l3345_334533


namespace shirt_price_reduction_l3345_334585

theorem shirt_price_reduction (original_price : ℝ) (h1 : original_price > 0) : 
  let sale_price := 0.70 * original_price
  let final_price := 0.63 * original_price
  ∃ markdown_percent : ℝ, 
    markdown_percent = 10 ∧ 
    final_price = sale_price * (1 - markdown_percent / 100) :=
by sorry

end shirt_price_reduction_l3345_334585


namespace range_of_m_l3345_334593

-- Define the ellipse C
def ellipse_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m + y^2 / (8 - m) = 1

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x - y + m = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

-- Define proposition p
def prop_p (m : ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ m = 4 + c ∧ 8 - m = 4 - c

-- Define proposition q
def prop_q (m : ℝ) : Prop :=
  abs m ≤ 3 * Real.sqrt 2

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ, (prop_p m ∧ ¬prop_q m) ∨ (¬prop_p m ∧ prop_q m) →
    (3 * Real.sqrt 2 < m ∧ m < 8) ∨ (-3 * Real.sqrt 2 ≤ m ∧ m ≤ 4) :=
sorry

end range_of_m_l3345_334593


namespace computer_purchase_cost_l3345_334530

theorem computer_purchase_cost (computer_cost : ℕ) (base_video_card_cost : ℕ) 
  (h1 : computer_cost = 1500)
  (h2 : base_video_card_cost = 300) : 
  computer_cost + 
  (computer_cost / 5) + 
  (2 * base_video_card_cost - base_video_card_cost) = 2100 := by
  sorry

#check computer_purchase_cost

end computer_purchase_cost_l3345_334530


namespace billy_points_billy_points_proof_l3345_334547

theorem billy_points : ℕ → Prop := fun b =>
  let friend_points : ℕ := 9
  let point_difference : ℕ := 2
  (b - friend_points = point_difference) → (b = 11)

-- The proof is omitted
theorem billy_points_proof : billy_points 11 := by sorry

end billy_points_billy_points_proof_l3345_334547


namespace vector_equation_solution_l3345_334521

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a x : V) :
  3 • (a + x) = x → x = -(3/2 : ℝ) • a := by
  sorry

end vector_equation_solution_l3345_334521


namespace triangle_area_inequality_l3345_334511

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (area : ℝ)
  (valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)

-- Define the theorem
theorem triangle_area_inequality (ABC : Triangle) :
  ∃ (A₁B₁C₁ : Triangle),
    A₁B₁C₁.a = Real.sqrt ABC.a ∧
    A₁B₁C₁.b = Real.sqrt ABC.b ∧
    A₁B₁C₁.c = Real.sqrt ABC.c ∧
    A₁B₁C₁.area ^ 2 ≥ (ABC.area * Real.sqrt 3) / 4 :=
by sorry

end triangle_area_inequality_l3345_334511


namespace heart_ratio_l3345_334519

/-- The heart operation defined as n ♥ m = n^2 * m^3 -/
def heart (n m : ℝ) : ℝ := n^2 * m^3

/-- Theorem stating that (3 ♥ 5) / (5 ♥ 3) = 5/3 -/
theorem heart_ratio : (heart 3 5) / (heart 5 3) = 5/3 := by
  sorry

end heart_ratio_l3345_334519


namespace basketball_problem_l3345_334537

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let descents := List.range (bounces + 1) |>.map (fun n => initialHeight * reboundFactor ^ n)
  let ascents := List.range bounces |>.map (fun n => initialHeight * reboundFactor ^ (n + 1))
  (descents.sum + ascents.sum)

/-- The basketball problem -/
theorem basketball_problem :
  totalDistance 150 (2/5) 5 = 347.952 := by
  sorry

end basketball_problem_l3345_334537


namespace cone_volume_from_half_sector_l3345_334504

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let base_radius := r / 2
  let cone_height := Real.sqrt (r^2 - base_radius^2)
  (1/3 : ℝ) * Real.pi * base_radius^2 * cone_height = 9 * Real.pi * Real.sqrt 3 :=
by sorry

end cone_volume_from_half_sector_l3345_334504


namespace proportional_division_l3345_334551

theorem proportional_division (total : ℚ) (a b c : ℚ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 →
  (a : ℚ) = 2 →
  (b : ℚ) = 1/2 →
  (c : ℚ) = 1/4 →
  ∃ (x : ℚ), a * x + b * x + c * x = total ∧ b * x = 208/11 := by
  sorry

end proportional_division_l3345_334551


namespace james_walking_distance_l3345_334527

def base7_to_base10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem james_walking_distance :
  base7_to_base10 3 6 5 2 = 1360 := by
  sorry

end james_walking_distance_l3345_334527


namespace total_flight_distance_l3345_334523

theorem total_flight_distance (beka_distance jackson_distance maria_distance : ℕ) 
  (h1 : beka_distance = 873)
  (h2 : jackson_distance = 563)
  (h3 : maria_distance = 786) :
  beka_distance + jackson_distance + maria_distance = 2222 := by
  sorry

end total_flight_distance_l3345_334523


namespace painting_choices_l3345_334501

/-- The number of traditional Chinese paintings -/
def traditional_paintings : ℕ := 5

/-- The number of oil paintings -/
def oil_paintings : ℕ := 2

/-- The number of watercolor paintings -/
def watercolor_paintings : ℕ := 7

/-- The number of ways to choose one painting from each category -/
def choose_one_each : ℕ := traditional_paintings * oil_paintings * watercolor_paintings

/-- The number of ways to choose two paintings of different types -/
def choose_two_different : ℕ := 
  traditional_paintings * oil_paintings + 
  traditional_paintings * watercolor_paintings + 
  oil_paintings * watercolor_paintings

theorem painting_choices :
  choose_one_each = 70 ∧ choose_two_different = 59 := by
  sorry

end painting_choices_l3345_334501


namespace isabella_total_items_l3345_334596

/-- Given that Alexis bought 3 times more pants and dresses than Isabella,
    and Alexis bought 21 pairs of pants and 18 dresses,
    prove that Isabella bought a total of 13 items (pants and dresses combined). -/
theorem isabella_total_items (alexis_pants : ℕ) (alexis_dresses : ℕ) 
    (h1 : alexis_pants = 21) 
    (h2 : alexis_dresses = 18) 
    (h3 : ∃ (isabella_pants isabella_dresses : ℕ), 
      alexis_pants = 3 * isabella_pants ∧ 
      alexis_dresses = 3 * isabella_dresses) : 
  ∃ (isabella_total : ℕ), isabella_total = 13 := by
  sorry

end isabella_total_items_l3345_334596


namespace division_powers_equality_l3345_334581

theorem division_powers_equality (a : ℝ) (h : a ≠ 0) :
  a^6 / ((1/2) * a^2) = 2 * a^4 := by sorry

end division_powers_equality_l3345_334581


namespace workers_gone_home_is_120_l3345_334535

/-- Represents the problem of workers leaving a factory for Chinese New Year --/
structure WorkerProblem where
  total_days : Nat
  weekend_days : Nat
  remaining_workers : Nat
  total_worker_days : Nat

/-- The specific instance of the worker problem --/
def factory_problem : WorkerProblem := {
  total_days := 15
  weekend_days := 4
  remaining_workers := 121
  total_worker_days := 2011
}

/-- Calculates the number of workers who have gone home --/
def workers_gone_home (p : WorkerProblem) : Nat :=
  sorry

/-- Theorem stating that 120 workers have gone home --/
theorem workers_gone_home_is_120 : 
  workers_gone_home factory_problem = 120 := by
  sorry

end workers_gone_home_is_120_l3345_334535


namespace clayton_shells_proof_l3345_334544

/-- The number of shells collected by Jillian -/
def jillian_shells : ℕ := 29

/-- The number of shells collected by Savannah -/
def savannah_shells : ℕ := 17

/-- The number of friends who received shells -/
def num_friends : ℕ := 2

/-- The number of shells each friend received -/
def shells_per_friend : ℕ := 27

/-- The number of shells Clayton collected -/
def clayton_shells : ℕ := 8

theorem clayton_shells_proof :
  clayton_shells = 
    num_friends * shells_per_friend - (jillian_shells + savannah_shells) :=
by sorry

end clayton_shells_proof_l3345_334544


namespace largest_perfect_square_product_l3345_334584

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A function that checks if a number is a one-digit positive integer -/
def is_one_digit_positive (n : ℕ) : Prop :=
  0 < n ∧ n ≤ 9

/-- The main theorem stating that 144 is the largest perfect square
    that can be written as the product of three different one-digit positive integers -/
theorem largest_perfect_square_product : 
  (∀ a b c : ℕ, 
    is_one_digit_positive a ∧ 
    is_one_digit_positive b ∧ 
    is_one_digit_positive c ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_perfect_square (a * b * c) →
    a * b * c ≤ 144) ∧
  (∃ a b c : ℕ,
    is_one_digit_positive a ∧
    is_one_digit_positive b ∧
    is_one_digit_positive c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a * b * c = 144 ∧
    is_perfect_square 144) :=
by sorry

end largest_perfect_square_product_l3345_334584


namespace smallest_gcd_bc_l3345_334512

theorem smallest_gcd_bc (a b c : ℕ+) (hab : Nat.gcd a b = 120) (hac : Nat.gcd a c = 360) :
  ∃ (b' c' : ℕ+), Nat.gcd a b' = 120 ∧ Nat.gcd a c' = 360 ∧ Nat.gcd b' c' = 120 ∧
  ∀ (b'' c'' : ℕ+), Nat.gcd a b'' = 120 → Nat.gcd a c'' = 360 → Nat.gcd b'' c'' ≥ 120 :=
by sorry

end smallest_gcd_bc_l3345_334512


namespace fixed_point_on_circle_l3345_334556

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the endpoints of the transverse axis
def A₁ : ℝ × ℝ := (2, 0)
def A₂ : ℝ × ℝ := (-2, 0)

-- Define a point P on the hyperbola
def P : ℝ × ℝ → Prop := λ p => 
  hyperbola p.1 p.2 ∧ p ≠ A₁ ∧ p ≠ A₂

-- Define the line x = 1
def line_x_1 (x y : ℝ) : Prop := x = 1

-- Define the intersection points M₁ and M₂
def M₁ (p : ℝ × ℝ) : ℝ × ℝ := sorry
def M₂ (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define a circle with diameter M₁M₂
def circle_M₁M₂ (p c : ℝ × ℝ) : Prop := sorry

-- The theorem to prove
theorem fixed_point_on_circle : 
  ∃ c : ℝ × ℝ, ∀ p : ℝ × ℝ, P p → circle_M₁M₂ p c := by sorry

end fixed_point_on_circle_l3345_334556


namespace inequality_and_equality_condition_l3345_334553

theorem inequality_and_equality_condition (a b c : ℝ) :
  (5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * b * c + 4 * a * c) ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * b * c + 4 * a * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by sorry

end inequality_and_equality_condition_l3345_334553


namespace systematic_sampling_theorem_l3345_334525

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) : ℕ → ℕ :=
  fun n => start + (n - 1) * (total / sample_size)

theorem systematic_sampling_theorem :
  let total := 200
  let sample_size := 40
  let group_size := total / sample_size
  let fifth_group_sample := 22
  systematic_sample total sample_size fifth_group_sample 8 = 37 := by
  sorry

end systematic_sampling_theorem_l3345_334525


namespace class_size_l3345_334543

theorem class_size (n : ℕ) (best_rank : ℕ) (worst_rank : ℕ) 
  (h1 : best_rank = 30) 
  (h2 : worst_rank = 25) 
  (h3 : n = (best_rank - 1) + (worst_rank - 1) + 1) : 
  n = 54 := by
  sorry

end class_size_l3345_334543


namespace student_line_count_l3345_334592

/-- The number of students in the line -/
def num_students : ℕ := 26

/-- The counting cycle -/
def cycle_length : ℕ := 4

/-- The last number called -/
def last_number : ℕ := 2

theorem student_line_count :
  num_students % cycle_length = last_number :=
by sorry

end student_line_count_l3345_334592


namespace receipts_change_l3345_334518

theorem receipts_change 
  (original_price : ℝ) 
  (original_sales : ℝ) 
  (price_reduction_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : price_reduction_rate = 0.3)
  (h2 : sales_increase_rate = 0.5) :
  let new_price := original_price * (1 - price_reduction_rate)
  let new_sales := original_sales * (1 + sales_increase_rate)
  let original_receipts := original_price * original_sales
  let new_receipts := new_price * new_sales
  (new_receipts - original_receipts) / original_receipts = 0.05 := by
sorry

end receipts_change_l3345_334518


namespace sum_of_numbers_l3345_334559

theorem sum_of_numbers (a b : ℝ) 
  (h1 : a^2 - b^2 = 6) 
  (h2 : (a-2)^2 - (b-2)^2 = 18) : 
  a + b = -2 := by
sorry

end sum_of_numbers_l3345_334559


namespace tan_double_angle_special_case_l3345_334503

theorem tan_double_angle_special_case (θ : ℝ) :
  2 * Real.cos (θ - π / 3) = 3 * Real.cos θ →
  Real.tan (2 * θ) = -4 * Real.sqrt 3 := by
  sorry

end tan_double_angle_special_case_l3345_334503


namespace annes_bottle_caps_l3345_334507

/-- Anne's initial number of bottle caps -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Anne finds -/
def found_caps : ℕ := 5

/-- Anne's final number of bottle caps -/
def final_caps : ℕ := 15

/-- Theorem stating that Anne's initial number of bottle caps plus the found caps equals her final number of caps -/
theorem annes_bottle_caps : initial_caps + found_caps = final_caps := by sorry

end annes_bottle_caps_l3345_334507


namespace fraction_simplification_l3345_334576

theorem fraction_simplification (a : ℝ) (h : a ≠ 2) :
  (a^2 / (a - 2)) - ((4*a - 4) / (a - 2)) = a - 2 := by
  sorry

end fraction_simplification_l3345_334576


namespace f_is_even_l3345_334560

def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_is_even_l3345_334560


namespace triangle_perimeter_l3345_334510

theorem triangle_perimeter (a b x : ℝ) : 
  a = 7 → 
  b = 11 → 
  x^2 - 25 = 2*(x - 5)^2 → 
  x > 0 →
  a + b > x →
  x + b > a →
  x + a > b →
  (a + b + x = 23 ∨ a + b + x = 33) :=
by sorry

end triangle_perimeter_l3345_334510


namespace team_selection_count_l3345_334580

/-- The number of ways to select a team of 8 members with an equal number of boys and girls
    from a group of 8 boys and 10 girls -/
def select_team (boys girls team_size : ℕ) : ℕ :=
  Nat.choose boys (team_size / 2) * Nat.choose girls (team_size / 2)

/-- Theorem stating the number of ways to select the team -/
theorem team_selection_count :
  select_team 8 10 8 = 14700 := by
  sorry

end team_selection_count_l3345_334580


namespace percentage_problem_l3345_334566

theorem percentage_problem (x : ℝ) : x * 2 = 0.8 → x * 100 = 40 := by
  sorry

end percentage_problem_l3345_334566


namespace division_problem_l3345_334500

/-- Given a division with quotient 20, divisor 66, and remainder 55, the dividend is 1375. -/
theorem division_problem :
  ∀ (dividend quotient divisor remainder : ℕ),
    quotient = 20 →
    divisor = 66 →
    remainder = 55 →
    dividend = divisor * quotient + remainder →
    dividend = 1375 := by
  sorry

end division_problem_l3345_334500


namespace max_value_theorem_l3345_334522

theorem max_value_theorem (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 8 + 6 * y * z ≤ Real.sqrt 2 := by
  sorry

end max_value_theorem_l3345_334522


namespace notebook_reorganization_theorem_l3345_334582

/-- Represents the notebook reorganization problem --/
structure NotebookProblem where
  initial_notebooks : ℕ
  pages_per_notebook : ℕ
  initial_drawings_per_page : ℕ
  new_drawings_per_page : ℕ
  full_notebooks_after_reorg : ℕ
  full_pages_in_last_notebook : ℕ

/-- Calculates the number of drawings on the last page after reorganization --/
def drawings_on_last_page (p : NotebookProblem) : ℕ :=
  let total_drawings := p.initial_notebooks * p.pages_per_notebook * p.initial_drawings_per_page
  let full_pages := (p.full_notebooks_after_reorg * p.pages_per_notebook) + p.full_pages_in_last_notebook
  total_drawings - (full_pages * p.new_drawings_per_page)

/-- Theorem stating that for the given problem, the number of drawings on the last page is 4 --/
theorem notebook_reorganization_theorem (p : NotebookProblem) 
  (h1 : p.initial_notebooks = 10)
  (h2 : p.pages_per_notebook = 50)
  (h3 : p.initial_drawings_per_page = 5)
  (h4 : p.new_drawings_per_page = 8)
  (h5 : p.full_notebooks_after_reorg = 6)
  (h6 : p.full_pages_in_last_notebook = 40) :
  drawings_on_last_page p = 4 := by
  sorry

end notebook_reorganization_theorem_l3345_334582


namespace triangle_max_value_l3345_334549

/-- In a triangle ABC, given the conditions, prove the maximum value of (1/2)b + a -/
theorem triangle_max_value (a b c : ℝ) (h1 : a^2 + b^2 = c^2 + a*b) (h2 : c = 1) :
  (∃ (x y : ℝ), x^2 + y^2 = 1^2 + x*y ∧ (1/2)*y + x ≤ (1/2)*b + a) ∧
  (∀ (x y : ℝ), x^2 + y^2 = 1^2 + x*y → (1/2)*y + x ≤ (1/2)*b + a) →
  (1/2)*b + a = Real.sqrt 21 / 3 :=
sorry

end triangle_max_value_l3345_334549


namespace trapezoid_area_l3345_334567

/-- The area of a trapezoid with height x, one base 4x, and the other base (4x - 2x) is 3x² -/
theorem trapezoid_area (x : ℝ) : 
  let height := x
  let base1 := 4 * x
  let base2 := 4 * x - 2 * x
  (base1 + base2) / 2 * height = 3 * x^2 :=
by sorry

end trapezoid_area_l3345_334567


namespace vacation_cost_balance_l3345_334550

/-- Proves that the difference between what Tom and Dorothy owe Sammy is -50 --/
theorem vacation_cost_balance (tom_paid dorothy_paid sammy_paid t d : ℚ) : 
  tom_paid = 140 →
  dorothy_paid = 90 →
  sammy_paid = 220 →
  (tom_paid + t) = (dorothy_paid + d) →
  (tom_paid + t) = (sammy_paid - t - d) →
  t - d = -50 := by
sorry

end vacation_cost_balance_l3345_334550


namespace min_reciprocal_sum_l3345_334558

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then 1 - Real.log x
  else if x > 1 then -1 + Real.log x
  else 0  -- This case is added to make the function total, but it's not used in our problem

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ab : f a = f b) :
  (∃ m : ℝ, m = 1 + 1 / Real.exp 2 ∧ ∀ x y : ℝ, 0 < x → 0 < y → f x = f y → 1 / x + 1 / y ≥ m) :=
sorry

end min_reciprocal_sum_l3345_334558


namespace prime_power_sum_l3345_334502

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 882 →
  2*w + 3*x + 5*y + 7*z = 22 := by
sorry

end prime_power_sum_l3345_334502


namespace inequality_properties_l3345_334542

theorem inequality_properties (x y : ℝ) (h : x > y) :
  (x - 3 > y - 3) ∧
  (x / 3 > y / 3) ∧
  (x + 3 > y + 3) ∧
  (1 - 3*x < 1 - 3*y) := by
  sorry

end inequality_properties_l3345_334542


namespace polynomial_factorization_l3345_334588

theorem polynomial_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x*y + x*z + y*z) := by sorry

end polynomial_factorization_l3345_334588


namespace triangle_shape_l3345_334571

theorem triangle_shape (A B C : ℝ) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) 
  (hcos : Real.cos A > Real.sin B) : 
  A + B + C = π ∧ C > π/2 :=
sorry

end triangle_shape_l3345_334571


namespace imaginary_power_sum_l3345_334589

theorem imaginary_power_sum : ∃ i : ℂ, i^2 = -1 ∧ i^50 + i^250 = -2 := by
  sorry

end imaginary_power_sum_l3345_334589


namespace symmetric_point_coordinates_l3345_334555

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0, 0, 0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Symmetric point with respect to the origin -/
def symmetricPoint (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, -p.z⟩

theorem symmetric_point_coordinates :
  let p : Point3D := ⟨1, -2, 1⟩
  let q : Point3D := symmetricPoint p
  q = ⟨-1, 2, -1⟩ := by sorry

end symmetric_point_coordinates_l3345_334555


namespace complex_computation_l3345_334534

theorem complex_computation :
  let A : ℂ := 3 + 2*I
  let B : ℂ := -1 - 2*I
  let C : ℂ := 5*I
  let D : ℂ := 3 + I
  2 * (A - B + C + D) = 8 + 20*I :=
by sorry

end complex_computation_l3345_334534


namespace parallel_vectors_k_l3345_334557

def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 1]
def c : Fin 2 → ℝ := ![-5, 1]

theorem parallel_vectors_k (k : ℝ) :
  (∀ i : Fin 2, (a i + k * b i) * c (1 - i) = (a (1 - i) + k * b (1 - i)) * c i) →
  k = 1/2 := by
sorry

end parallel_vectors_k_l3345_334557


namespace polynomial_identity_l3345_334568

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end polynomial_identity_l3345_334568


namespace floor_equation_solutions_range_l3345_334554

theorem floor_equation_solutions_range (a : ℝ) (n : ℕ) 
  (h1 : a > 1) 
  (h2 : n ≥ 2) 
  (h3 : ∃! (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, ⌊a * x⌋ = x) :
  1 + 1 / n ≤ a ∧ a < 1 + 1 / (n - 1) := by
  sorry

end floor_equation_solutions_range_l3345_334554


namespace base_conversion_proof_l3345_334517

/-- 
Given a positive integer n with the following properties:
1. Its base 9 representation is AB
2. Its base 7 representation is BA
3. A and B are single digits in their respective bases

This theorem proves that n = 31 in base 10.
-/
theorem base_conversion_proof (n : ℕ) (A B : ℕ) 
  (h1 : n = 9 * A + B)
  (h2 : n = 7 * B + A)
  (h3 : A < 9 ∧ B < 9)
  (h4 : A < 7 ∧ B < 7)
  (h5 : n > 0) :
  n = 31 := by
  sorry


end base_conversion_proof_l3345_334517


namespace fence_cost_calculation_l3345_334541

/-- The cost of building a fence around a rectangular plot -/
def fence_cost (length width price_length price_width : ℕ) : ℕ :=
  2 * (length * price_length + width * price_width)

/-- Theorem: The cost of the fence for the given dimensions and prices is 5408 -/
theorem fence_cost_calculation :
  fence_cost 17 21 59 81 = 5408 := by
  sorry

end fence_cost_calculation_l3345_334541


namespace range_of_m_l3345_334538

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + 2*y = x*y) 
  (h_ineq : ∀ m : ℝ, m^2 + 2*m < x + 2*y) : 
  m ∈ Set.Ioo (-2 : ℝ) 4 := by
sorry

end range_of_m_l3345_334538


namespace max_value_of_S_l3345_334574

def S (a b c d e f g h k : Int) : Int :=
  a*e*k - a*f*h + b*f*g - b*d*k + c*d*h - c*e*g

theorem max_value_of_S :
  ∃ (a b c d e f g h k : Int),
    (a = 1 ∨ a = -1) ∧
    (b = 1 ∨ b = -1) ∧
    (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧
    (e = 1 ∨ e = -1) ∧
    (f = 1 ∨ f = -1) ∧
    (g = 1 ∨ g = -1) ∧
    (h = 1 ∨ h = -1) ∧
    (k = 1 ∨ k = -1) ∧
    S a b c d e f g h k = 4 ∧
    ∀ (a' b' c' d' e' f' g' h' k' : Int),
      (a' = 1 ∨ a' = -1) →
      (b' = 1 ∨ b' = -1) →
      (c' = 1 ∨ c' = -1) →
      (d' = 1 ∨ d' = -1) →
      (e' = 1 ∨ e' = -1) →
      (f' = 1 ∨ f' = -1) →
      (g' = 1 ∨ g' = -1) →
      (h' = 1 ∨ h' = -1) →
      (k' = 1 ∨ k' = -1) →
      S a' b' c' d' e' f' g' h' k' ≤ 4 := by
  sorry

end max_value_of_S_l3345_334574


namespace area_of_region_l3345_334514

-- Define the region
def region (x y : ℝ) : Prop := 
  |x - 2*y^2| + x + 2*y^2 ≤ 8 - 4*y

-- Define symmetry about y-axis
def symmetric_about_y_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S ↔ (-x, y) ∈ S

-- Theorem statement
theorem area_of_region : 
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ region x y) ∧ 
    symmetric_about_y_axis S ∧
    MeasureTheory.volume S = 30 := by
  sorry

end area_of_region_l3345_334514


namespace student_language_partition_l3345_334573

/-- Represents a student and the languages they speak -/
structure Student where
  speaksEnglish : Bool
  speaksFrench : Bool
  speaksSpanish : Bool

/-- Represents a group of students -/
def StudentGroup := List Student

/-- Checks if a group satisfies the language requirements -/
def isValidGroup (group : StudentGroup) : Bool :=
  (group.filter (·.speaksEnglish)).length = 10 ∧
  (group.filter (·.speaksFrench)).length = 10 ∧
  (group.filter (·.speaksSpanish)).length = 10

/-- Main theorem -/
theorem student_language_partition 
  (students : List Student)
  (h_english : (students.filter (·.speaksEnglish)).length = 50)
  (h_french : (students.filter (·.speaksFrench)).length = 50)
  (h_spanish : (students.filter (·.speaksSpanish)).length = 50) :
  ∃ (partition : List StudentGroup), 
    partition.length = 5 ∧ 
    (∀ group ∈ partition, isValidGroup group) ∧
    (partition.join = students) :=
  sorry

end student_language_partition_l3345_334573


namespace meat_for_community_event_l3345_334528

/-- The amount of meat (in pounds) needed to make a given number of hamburgers. -/
def meat_needed (hamburgers : ℕ) : ℚ :=
  (5 : ℚ) * hamburgers / 10

/-- Theorem stating that 15 pounds of meat are needed for 30 hamburgers. -/
theorem meat_for_community_event : meat_needed 30 = 15 := by
  sorry

end meat_for_community_event_l3345_334528


namespace cone_height_l3345_334569

/-- Given a cone whose lateral surface development is a sector with radius 2 and central angle 180°,
    the height of the cone is √3. -/
theorem cone_height (r : ℝ) (l : ℝ) (h : ℝ) :
  r = 1 →  -- radius of the base (derived from the sector's arc length)
  l = 2 →  -- slant height (radius of the sector)
  h^2 + r^2 = l^2 →  -- Pythagorean theorem
  h = Real.sqrt 3 := by
sorry

end cone_height_l3345_334569


namespace trigonometric_expression_proof_l3345_334579

theorem trigonometric_expression_proof (sin30 cos30 sin60 cos60 : ℝ) 
  (h1 : sin30 = 1/2)
  (h2 : cos30 = Real.sqrt 3 / 2)
  (h3 : sin60 = Real.sqrt 3 / 2)
  (h4 : cos60 = 1/2) :
  (1 - 1/(sin30^2)) * (1 + 1/(cos60^2)) * (1 - 1/(cos30^2)) * (1 + 1/(sin60^2)) = 35/3 := by
  sorry

end trigonometric_expression_proof_l3345_334579


namespace complex_power_thousand_l3345_334548

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Main theorem: ((1 + i) / (1 - i)) ^ 1000 = 1 -/
theorem complex_power_thousand :
  ((1 + i) / (1 - i)) ^ 1000 = 1 :=
by
  sorry


end complex_power_thousand_l3345_334548


namespace circle_radius_from_area_circumference_ratio_l3345_334529

theorem circle_radius_from_area_circumference_ratio 
  (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) (h3 : P / Q = 10) : 
  ∃ r : ℝ, r > 0 ∧ P = π * r^2 ∧ Q = 2 * π * r ∧ r = 20 := by
  sorry

end circle_radius_from_area_circumference_ratio_l3345_334529


namespace polynomial_product_simplification_l3345_334509

theorem polynomial_product_simplification (x y : ℝ) :
  (3 * x^2 - 7 * y^3) * (9 * x^4 + 21 * x^2 * y^3 + 49 * y^6) = 27 * x^6 - 343 * y^9 := by
  sorry

end polynomial_product_simplification_l3345_334509


namespace bacon_percentage_is_twenty_l3345_334595

/-- Calculates the percentage of calories from bacon in a sandwich -/
def bacon_calorie_percentage (total_calories : ℕ) (bacon_strips : ℕ) (calories_per_strip : ℕ) : ℚ :=
  (bacon_strips * calories_per_strip : ℚ) / total_calories * 100

/-- Theorem stating that the percentage of calories from bacon in the given sandwich is 20% -/
theorem bacon_percentage_is_twenty :
  bacon_calorie_percentage 1250 2 125 = 20 := by
  sorry

end bacon_percentage_is_twenty_l3345_334595


namespace quadratic_roots_range_l3345_334599

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x - m = 0 ∧ y^2 - 2*y - m = 0) → m ≥ -1 :=
by
  sorry

end quadratic_roots_range_l3345_334599


namespace square_plus_double_perfect_square_l3345_334583

theorem square_plus_double_perfect_square (a : ℕ) : 
  ∃ (k : ℕ), a^2 + 2*a = k^2 ↔ a = 0 :=
sorry

end square_plus_double_perfect_square_l3345_334583


namespace shopping_mall_probabilities_l3345_334591

/-- Probability of purchasing product A -/
def prob_A : ℝ := 0.5

/-- Probability of purchasing product B -/
def prob_B : ℝ := 0.6

/-- Number of customers -/
def n : ℕ := 3

/-- Probability of purchasing at least one product -/
def p : ℝ := 0.8

theorem shopping_mall_probabilities :
  let prob_either := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B
  let prob_at_least_one := 1 - (1 - prob_A) * (1 - prob_B)
  let ξ := fun k => (n.choose k : ℝ) * p^k * (1 - p)^(n - k)
  (prob_either = 0.5) ∧
  (prob_at_least_one = 0.8) ∧
  (ξ 0 = 0.008) ∧
  (ξ 1 = 0.096) ∧
  (ξ 2 = 0.384) ∧
  (ξ 3 = 0.512) := by
  sorry

end shopping_mall_probabilities_l3345_334591


namespace work_completion_time_l3345_334594

theorem work_completion_time 
  (total_work : ℝ) 
  (a_rate : ℝ) 
  (ab_rate : ℝ) 
  (h1 : a_rate = total_work / 12) 
  (h2 : 10 * ab_rate + 9 * a_rate = total_work) :
  ab_rate = total_work / 40 := by
  sorry

end work_completion_time_l3345_334594


namespace recursive_sequence_solution_l3345_334587

/-- A sequence of real numbers satisfying the given recursion -/
def RecursiveSequence (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → b n = b (n - 1) * b (n + 1)

theorem recursive_sequence_solution 
  (b : ℕ → ℝ) 
  (h_recursive : RecursiveSequence b) 
  (h_b1 : b 1 = 2 + Real.sqrt 8) 
  (h_b1980 : b 1980 = 15 + Real.sqrt 8) : 
  b 2013 = -1/6 + 13 * Real.sqrt 8 / 6 := by
  sorry

end recursive_sequence_solution_l3345_334587


namespace jerry_recycling_time_l3345_334598

/-- The time it takes Jerry to throw away all the cans -/
def total_time (num_cans : ℕ) (cans_per_trip : ℕ) (drain_time : ℕ) (walk_time : ℕ) : ℕ :=
  let num_trips := (num_cans + cans_per_trip - 1) / cans_per_trip
  let round_trip_time := 2 * walk_time
  let time_per_trip := round_trip_time + drain_time
  num_trips * time_per_trip

/-- Theorem stating that under the given conditions, it takes Jerry 350 seconds to throw away all the cans -/
theorem jerry_recycling_time :
  total_time 28 4 30 10 = 350 := by
  sorry

end jerry_recycling_time_l3345_334598


namespace herring_fat_proof_l3345_334570

/-- The amount of fat in ounces for a herring -/
def herring_fat : ℝ := 40

/-- The amount of fat in ounces for an eel -/
def eel_fat : ℝ := 20

/-- The amount of fat in ounces for a pike -/
def pike_fat : ℝ := eel_fat + 10

/-- The number of each type of fish cooked -/
def fish_count : ℕ := 40

/-- The total amount of fat served in ounces -/
def total_fat : ℝ := 3600

theorem herring_fat_proof : 
  herring_fat * fish_count + eel_fat * fish_count + pike_fat * fish_count = total_fat :=
by sorry

end herring_fat_proof_l3345_334570


namespace add_7455_seconds_to_8_15_00_l3345_334564

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The starting time: 8:15:00 -/
def startTime : Time :=
  { hours := 8, minutes := 15, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 7455

/-- The expected final time: 10:19:15 -/
def expectedFinalTime : Time :=
  { hours := 10, minutes := 19, seconds := 15 }

theorem add_7455_seconds_to_8_15_00 :
  addSeconds startTime secondsToAdd = expectedFinalTime := by
  sorry

end add_7455_seconds_to_8_15_00_l3345_334564


namespace megan_bought_42_songs_l3345_334575

/-- The number of songs Megan bought given the initial number of albums,
    the number of albums removed, and the number of songs per album. -/
def total_songs (initial_albums : ℕ) (removed_albums : ℕ) (songs_per_album : ℕ) : ℕ :=
  (initial_albums - removed_albums) * songs_per_album

/-- Theorem stating that Megan bought 42 songs in total. -/
theorem megan_bought_42_songs :
  total_songs 8 2 7 = 42 := by
  sorry

end megan_bought_42_songs_l3345_334575


namespace max_value_ad_minus_bc_l3345_334531

theorem max_value_ad_minus_bc :
  ∀ a b c d : ℤ,
  a ∈ ({-1, 1, 2} : Set ℤ) →
  b ∈ ({-1, 1, 2} : Set ℤ) →
  c ∈ ({-1, 1, 2} : Set ℤ) →
  d ∈ ({-1, 1, 2} : Set ℤ) →
  (∀ x y z w : ℤ,
    x ∈ ({-1, 1, 2} : Set ℤ) →
    y ∈ ({-1, 1, 2} : Set ℤ) →
    z ∈ ({-1, 1, 2} : Set ℤ) →
    w ∈ ({-1, 1, 2} : Set ℤ) →
    x * w - y * z ≤ 6) ∧
  (∃ x y z w : ℤ,
    x ∈ ({-1, 1, 2} : Set ℤ) ∧
    y ∈ ({-1, 1, 2} : Set ℤ) ∧
    z ∈ ({-1, 1, 2} : Set ℤ) ∧
    w ∈ ({-1, 1, 2} : Set ℤ) ∧
    x * w - y * z = 6) :=
by sorry

end max_value_ad_minus_bc_l3345_334531
