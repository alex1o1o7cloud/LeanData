import Mathlib

namespace NUMINAMATH_CALUDE_greatest_measuring_length_l3576_357626

theorem greatest_measuring_length
  (length1 length2 length3 : ℕ)
  (h1 : length1 = 1234)
  (h2 : length2 = 898)
  (h3 : length3 > 0)
  (h4 : Nat.gcd length1 (Nat.gcd length2 length3) = 1) :
  ∀ (measuring_length : ℕ),
    (measuring_length > 0 ∧
     length1 % measuring_length = 0 ∧
     length2 % measuring_length = 0 ∧
     length3 % measuring_length = 0) →
    measuring_length = 1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_measuring_length_l3576_357626


namespace NUMINAMATH_CALUDE_circle_radius_when_area_circumference_ratio_is_ten_l3576_357656

/-- Given a circle with area M cm² and circumference N cm, if M/N = 10, then the radius is 20 cm -/
theorem circle_radius_when_area_circumference_ratio_is_ten
  (M N : ℝ) -- M is the area, N is the circumference
  (h1 : M > 0) -- area is positive
  (h2 : N > 0) -- circumference is positive
  (h3 : M = π * (N / (2 * π))^2) -- area formula
  (h4 : M / N = 10) -- given ratio
  : N / (2 * π) = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_when_area_circumference_ratio_is_ten_l3576_357656


namespace NUMINAMATH_CALUDE_square_rectangle_equal_area_l3576_357683

theorem square_rectangle_equal_area (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  a^2 = b * c → a = Real.sqrt (b * c) := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_equal_area_l3576_357683


namespace NUMINAMATH_CALUDE_n_ge_digit_product_eq_digit_product_iff_eq_four_l3576_357605

/-- Function that returns the product of digits of a positive integer -/
def digit_product (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that n is greater than or equal to the product of its digits -/
theorem n_ge_digit_product (n : ℕ+) : (n : ℕ) ≥ digit_product n :=
  sorry

/-- Theorem stating that n^2 - 17n + 56 equals the product of digits of n if and only if n = 4 -/
theorem eq_digit_product_iff_eq_four (n : ℕ+) : 
  (n : ℕ)^2 - 17*(n : ℕ) + 56 = digit_product n ↔ n = 4 :=
  sorry

end NUMINAMATH_CALUDE_n_ge_digit_product_eq_digit_product_iff_eq_four_l3576_357605


namespace NUMINAMATH_CALUDE_second_number_value_l3576_357602

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 550 →
  a = 2 * b →
  c = (1 / 3) * a →
  b = 150 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l3576_357602


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3576_357622

/-- Given vectors a, b, and c in ℝ², prove that (a-b) is perpendicular to c -/
theorem vector_perpendicular (a b c : ℝ × ℝ) 
  (ha : a = (0, 5)) 
  (hb : b = (4, -3)) 
  (hc : c = (-2, -1)) : 
  (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3576_357622


namespace NUMINAMATH_CALUDE_exists_monochromatic_isosceles_right_triangle_l3576_357603

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define an isosceles right triangle
def isIsoscelesRightTriangle (a b c : Point) : Prop := sorry

-- Theorem statement
theorem exists_monochromatic_isosceles_right_triangle :
  ∃ a b c : Point, isIsoscelesRightTriangle a b c ∧ 
    coloring a = coloring b ∧ coloring b = coloring c := by sorry

end NUMINAMATH_CALUDE_exists_monochromatic_isosceles_right_triangle_l3576_357603


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3576_357637

/-- Proves that a boat's speed in still water is 42 km/hr given specific conditions -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 3) 
  (h2 : downstream_distance = 33) 
  (h3 : downstream_time = 44 / 60) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 42 ∧ 
    downstream_distance = (still_water_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3576_357637


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2551_l3576_357667

theorem smallest_prime_factor_of_2551 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2551 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2551 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2551_l3576_357667


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l3576_357643

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + (y+1)^2 = 4

-- Define the centers of the circles
def center1 : ℝ × ℝ := (0, 1)
def center2 : ℝ × ℝ := (0, -1)

-- Define the condition for point P
def point_condition (x y : ℝ) : Prop :=
  x ≠ 0 → ((y - 1) / x) * ((y + 1) / x) = -1/2

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- State the theorem
theorem trajectory_and_intersection :
  -- Part 1: Trajectory equation
  (∀ x y : ℝ, point_condition x y → trajectory x y) ∧
  -- Part 2: Line x = 0 intersects at two points with equal distance from C₁
  (∃ C D : ℝ × ℝ,
    C.1 = 0 ∧ D.1 = 0 ∧
    C ≠ D ∧
    trajectory C.1 C.2 ∧
    trajectory D.1 D.2 ∧
    (C.1 - center1.1)^2 + (C.2 - center1.2)^2 =
    (D.1 - center1.1)^2 + (D.2 - center1.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l3576_357643


namespace NUMINAMATH_CALUDE_mans_wage_to_womans_wage_ratio_l3576_357616

/-- Prove that the ratio of a man's daily wage to a woman's daily wage is 4:1 -/
theorem mans_wage_to_womans_wage_ratio :
  ∀ (man_wage woman_wage : ℚ),
  (∃ k : ℚ, man_wage = k * woman_wage) →  -- Man's wage is some multiple of woman's wage
  (8 * 25 * man_wage = 14400) →           -- 8 men working for 25 days earn Rs. 14400
  (40 * 30 * woman_wage = 21600) →        -- 40 women working for 30 days earn Rs. 21600
  man_wage / woman_wage = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_mans_wage_to_womans_wage_ratio_l3576_357616


namespace NUMINAMATH_CALUDE_right_triangle_from_equation_l3576_357636

theorem right_triangle_from_equation (a b c : ℝ) 
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (h : (a - 6)^2 + Real.sqrt (b - 8) + |c - 10| = 0) : 
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_from_equation_l3576_357636


namespace NUMINAMATH_CALUDE_limit_f_limit_φ_l3576_357699

-- Function f(x)
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 2*x + 4

-- Function φ(t)
noncomputable def φ (t : ℝ) : ℝ := t * Real.sqrt (t^2 - 20) - Real.log (t + Real.sqrt (t^2 - 20)) / Real.log 10

-- Theorem for the limit of f(x) as x → -3
theorem limit_f : 
  Filter.Tendsto f (nhds (-3)) (nhds (-74)) :=
sorry

-- Theorem for the limit of φ(t) as t → 6
theorem limit_φ :
  Filter.Tendsto φ (nhds 6) (nhds 23) :=
sorry

end NUMINAMATH_CALUDE_limit_f_limit_φ_l3576_357699


namespace NUMINAMATH_CALUDE_proportional_relationship_and_point_value_l3576_357661

/-- Given that y is directly proportional to x-1 and y = 4 when x = 3,
    prove that the relationship between y and x is y = 2x - 2,
    and when the point (-1,m) lies on this graph, m = -4. -/
theorem proportional_relationship_and_point_value 
  (y : ℝ → ℝ) 
  (h1 : ∃ k : ℝ, ∀ x, y x = k * (x - 1)) 
  (h2 : y 3 = 4) :
  (∀ x, y x = 2*x - 2) ∧ 
  y (-1) = -4 := by
sorry

end NUMINAMATH_CALUDE_proportional_relationship_and_point_value_l3576_357661


namespace NUMINAMATH_CALUDE_all_values_equal_l3576_357668

-- Define the type for coordinates
def Coord := ℤ × ℤ

-- Define the type for the value assignment function
def ValueAssignment := Coord → ℕ

-- Define the property that each value is the average of its neighbors
def IsAverageOfNeighbors (f : ValueAssignment) : Prop :=
  ∀ (x y : ℤ), f (x, y) * 4 = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)

-- State the theorem
theorem all_values_equal (f : ValueAssignment) (h : IsAverageOfNeighbors f) :
  ∀ (x₁ y₁ x₂ y₂ : ℤ), f (x₁, y₁) = f (x₂, y₂) := by
  sorry

end NUMINAMATH_CALUDE_all_values_equal_l3576_357668


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3576_357609

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ := sorry

/-- Theorem: For a line with slope -3 and x-intercept (7,0), the y-intercept is (0,21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := (7, 0) }
  y_intercept l = (0, 21) := by sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3576_357609


namespace NUMINAMATH_CALUDE_value_of_x_l3576_357663

theorem value_of_x : ∃ x : ℝ, (3 * x + 15 = (1 / 3) * (6 * x + 45)) ∧ (x = 0) := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3576_357663


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3576_357601

theorem quadratic_root_range (k : ℝ) : 
  (∃ α β : ℝ, 
    (7 * α^2 - (k + 13) * α + k^2 - k - 2 = 0) ∧ 
    (7 * β^2 - (k + 13) * β + k^2 - k - 2 = 0) ∧ 
    (0 < α) ∧ (α < 1) ∧ (1 < β) ∧ (β < 2)) →
  ((3 < k ∧ k < 4) ∨ (-2 < k ∧ k < -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3576_357601


namespace NUMINAMATH_CALUDE_divisors_of_12m_squared_l3576_357634

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem divisors_of_12m_squared (m : ℕ) 
  (h_even : is_even m) 
  (h_divisors : count_divisors m = 7) : 
  count_divisors (12 * m^2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_12m_squared_l3576_357634


namespace NUMINAMATH_CALUDE_cube_root_three_irrational_l3576_357652

theorem cube_root_three_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ (3 : ℝ) ^ (1/3 : ℝ) = (p : ℝ) / (q : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_three_irrational_l3576_357652


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l3576_357689

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 2 * a * b) :
  ∀ x y, x > 0 → y > 0 → 3 * x + y = 2 * x * y → a + b ≤ x + y ∧ a + b = 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l3576_357689


namespace NUMINAMATH_CALUDE_john_and_alice_money_sum_l3576_357651

theorem john_and_alice_money_sum :
  let john_money : ℚ := 5 / 8
  let alice_money : ℚ := 7 / 20
  (john_money + alice_money : ℚ) = 39 / 40 := by
  sorry

end NUMINAMATH_CALUDE_john_and_alice_money_sum_l3576_357651


namespace NUMINAMATH_CALUDE_total_deduction_is_137_5_l3576_357621

/-- Represents David's hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Represents the local tax rate as a decimal -/
def local_tax_rate : ℝ := 0.025

/-- Represents the retirement fund contribution rate as a decimal -/
def retirement_rate : ℝ := 0.03

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

/-- Calculates the total deduction in cents -/
def total_deduction : ℝ :=
  dollars_to_cents (hourly_wage * local_tax_rate + hourly_wage * retirement_rate)

/-- Theorem stating that the total deduction is 137.5 cents -/
theorem total_deduction_is_137_5 : total_deduction = 137.5 := by
  sorry


end NUMINAMATH_CALUDE_total_deduction_is_137_5_l3576_357621


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l3576_357698

theorem sum_of_roots_equals_one :
  ∀ x₁ x₂ : ℝ,
  (x₁ + 3) * (x₁ - 4) = 18 →
  (x₂ + 3) * (x₂ - 4) = 18 →
  x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l3576_357698


namespace NUMINAMATH_CALUDE_trig_identity_l3576_357623

theorem trig_identity (α : ℝ) : 
  (Real.sin α)^2 + (Real.cos (30 * π / 180 - α))^2 - 
  (Real.sin α) * (Real.cos (30 * π / 180 - α)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3576_357623


namespace NUMINAMATH_CALUDE_parabola_min_distance_sum_l3576_357670

/-- The minimum distance sum from a point on a parabola to its focus and an external point -/
theorem parabola_min_distance_sum (F : ℝ × ℝ) (B : ℝ × ℝ) :
  let parabola := {P : ℝ × ℝ | P.2^2 = 4 * P.1}
  F = (1, 0) →
  B = (3, 4) →
  (∃ (min : ℝ), ∀ (P : ℝ × ℝ), P ∈ parabola →
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) +
    Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) ≥ min ∧
    min = 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_parabola_min_distance_sum_l3576_357670


namespace NUMINAMATH_CALUDE_lucys_fish_count_l3576_357655

theorem lucys_fish_count (initial_fish : ℕ) 
  (h1 : initial_fish + 68 = 280) : initial_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucys_fish_count_l3576_357655


namespace NUMINAMATH_CALUDE_unique_x_value_l3576_357666

/-- Binary operation ⋆ on pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := 
  fun (a, b) (c, d) ↦ (a + c, b - d)

/-- Theorem stating the unique value of x satisfying the equation -/
theorem unique_x_value : 
  ∃! x : ℤ, star (x, 4) (2, 1) = star (5, 2) (1, -3) := by sorry

end NUMINAMATH_CALUDE_unique_x_value_l3576_357666


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3576_357610

theorem line_segment_endpoint (y : ℝ) : y > 0 →
  Real.sqrt ((3 - (-5))^2 + (7 - y)^2) = 12 →
  y = 7 + 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3576_357610


namespace NUMINAMATH_CALUDE_tv_sets_in_shop_d_l3576_357644

theorem tv_sets_in_shop_d (total_shops : Nat) (avg_tv_sets : Nat)
  (shop_a shop_b shop_c shop_e : Nat) :
  total_shops = 5 →
  avg_tv_sets = 48 →
  shop_a = 20 →
  shop_b = 30 →
  shop_c = 60 →
  shop_e = 50 →
  ∃ shop_d : Nat, shop_d = 80 ∧
    avg_tv_sets * total_shops = shop_a + shop_b + shop_c + shop_d + shop_e :=
by sorry

end NUMINAMATH_CALUDE_tv_sets_in_shop_d_l3576_357644


namespace NUMINAMATH_CALUDE_open_box_volume_l3576_357678

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_length : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_length = 8) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 5120 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l3576_357678


namespace NUMINAMATH_CALUDE_guest_author_payment_l3576_357660

theorem guest_author_payment (B : ℕ) (h1 : B < 10) (h2 : B > 0) 
  (h3 : (200 + 10 * B) % 14 = 0) : B = 8 := by
  sorry

end NUMINAMATH_CALUDE_guest_author_payment_l3576_357660


namespace NUMINAMATH_CALUDE_christian_yard_charge_l3576_357620

/-- Proves that Christian charged $5 for mowing each yard --/
theorem christian_yard_charge :
  let perfume_cost : ℚ := 50
  let christian_savings : ℚ := 5
  let sue_savings : ℚ := 7
  let christian_yards : ℕ := 4
  let sue_dogs : ℕ := 6
  let sue_dog_charge : ℚ := 2
  let additional_needed : ℚ := 6
  
  let total_needed := perfume_cost - additional_needed
  let initial_savings := christian_savings + sue_savings
  let chores_earnings := total_needed - initial_savings
  let sue_earnings := sue_dogs * sue_dog_charge
  let christian_earnings := chores_earnings - sue_earnings
  let christian_yard_charge := christian_earnings / christian_yards

  christian_yard_charge = 5 := by sorry

end NUMINAMATH_CALUDE_christian_yard_charge_l3576_357620


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l3576_357654

/-- A polynomial is a perfect square trinomial if it can be expressed as (x + a)^2 for some real number a. -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, p x = (x + a)^2

/-- Given that x^2 + kx + 25 is a perfect square trinomial, prove that k = 10 or k = -10. -/
theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial (fun x => x^2 + k*x + 25) →
  k = 10 ∨ k = -10 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l3576_357654


namespace NUMINAMATH_CALUDE_filter_kit_price_l3576_357630

theorem filter_kit_price :
  let individual_prices : List ℝ := [12.45, 12.45, 14.05, 14.05, 11.50]
  let total_individual_price := individual_prices.sum
  let discount_percentage : ℝ := 11.03448275862069 / 100
  let kit_price := total_individual_price * (1 - discount_percentage)
  kit_price = 57.382758620689655 := by
sorry

end NUMINAMATH_CALUDE_filter_kit_price_l3576_357630


namespace NUMINAMATH_CALUDE_expression_evaluation_l3576_357649

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := -2
  let z : ℤ := 3
  3*x - 2*y - (2*x + 2*y - (2*x*y*z + x + 2*z) - 4*x + 2*z) - x*y*z = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3576_357649


namespace NUMINAMATH_CALUDE_sin_390_degrees_l3576_357645

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  have h1 : ∀ x, Real.sin (x + 2 * π) = Real.sin x := by sorry
  have h2 : Real.sin (π / 6) = 1 / 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l3576_357645


namespace NUMINAMATH_CALUDE_companion_point_on_hyperbola_companion_point_on_line_companion_point_line_equation_l3576_357640

/-- Definition of companion point -/
def is_companion_point (P Q : ℝ × ℝ) : Prop :=
  Q.1 = P.1 + 2 ∧ Q.2 = P.2 - 4

/-- Theorem 1: The companion point of P(2,-1) lies on y = -20/x -/
theorem companion_point_on_hyperbola :
  ∀ Q : ℝ × ℝ, is_companion_point (2, -1) Q → Q.2 = -20 / Q.1 :=
sorry

/-- Theorem 2: If P(a,b) lies on y = x+5 and (-1,-2) is its companion point, then P = (-3,2) -/
theorem companion_point_on_line :
  ∀ P : ℝ × ℝ, P.2 = P.1 + 5 → is_companion_point P (-1, -2) → P = (-3, 2) :=
sorry

/-- Theorem 3: If P(a,b) lies on y = 2x+3, then its companion point Q lies on y = 2x-5 -/
theorem companion_point_line_equation :
  ∀ P Q : ℝ × ℝ, P.2 = 2 * P.1 + 3 → is_companion_point P Q → Q.2 = 2 * Q.1 - 5 :=
sorry

end NUMINAMATH_CALUDE_companion_point_on_hyperbola_companion_point_on_line_companion_point_line_equation_l3576_357640


namespace NUMINAMATH_CALUDE_candy_distribution_l3576_357627

theorem candy_distribution (total_candies : ℕ) (sour_percentage : ℚ) (num_people : ℕ) :
  total_candies = 300 →
  sour_percentage = 40 / 100 →
  num_people = 3 →
  (total_candies - (sour_percentage * total_candies).floor) / num_people = 60 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3576_357627


namespace NUMINAMATH_CALUDE_brick_packing_theorem_l3576_357662

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular parallelepiped -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

theorem brick_packing_theorem (box : Dimensions) 
  (brick1 brick2 : Dimensions) 
  (h_box : box = ⟨10, 11, 14⟩) 
  (h_brick1 : brick1 = ⟨2, 5, 8⟩) 
  (h_brick2 : brick2 = ⟨2, 3, 7⟩) :
  ∃ (x y : ℕ), 
    x * volume brick1 + y * volume brick2 = volume box ∧ 
    x + y = 24 ∧ 
    ∀ (a b : ℕ), a * volume brick1 + b * volume brick2 = volume box → a + b ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_brick_packing_theorem_l3576_357662


namespace NUMINAMATH_CALUDE_unique_solution_xyz_squared_l3576_357693

theorem unique_solution_xyz_squared (x y z : ℕ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xyz_squared_l3576_357693


namespace NUMINAMATH_CALUDE_seating_arrangement_for_100_people_l3576_357686

/-- Represents a seating arrangement with rows of 9 or 10 people -/
structure SeatingArrangement where
  rows_of_10 : ℕ
  rows_of_9 : ℕ

/-- The total number of people in the seating arrangement -/
def total_people (s : SeatingArrangement) : ℕ :=
  10 * s.rows_of_10 + 9 * s.rows_of_9

/-- The theorem stating that for 100 people, there are 10 rows of 10 people -/
theorem seating_arrangement_for_100_people :
  ∃ (s : SeatingArrangement), total_people s = 100 ∧ s.rows_of_10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_for_100_people_l3576_357686


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3576_357669

theorem complex_fraction_equality : (1 / (1 + 1 / (2 + 1 / 3))) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3576_357669


namespace NUMINAMATH_CALUDE_february_warmer_than_january_l3576_357613

/-- The average temperature in January 2023 in Taiyuan City (in °C) -/
def jan_temp : ℝ := -12

/-- The average temperature in February 2023 in Taiyuan City (in °C) -/
def feb_temp : ℝ := -6

/-- The difference in average temperature between February and January 2023 in Taiyuan City -/
def temp_difference : ℝ := feb_temp - jan_temp

theorem february_warmer_than_january : temp_difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_february_warmer_than_january_l3576_357613


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l3576_357697

theorem triangle_inequality_expression (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l3576_357697


namespace NUMINAMATH_CALUDE_tax_free_amount_satisfies_equation_l3576_357653

/-- The tax-free amount for goods purchased in country B -/
def tax_free_amount : ℝ :=
  -- We define the tax-free amount, but don't provide its value
  -- as it needs to be proved
  sorry

/-- The total value of goods purchased -/
def total_value : ℝ := 1720

/-- The tax rate as a decimal -/
def tax_rate : ℝ := 0.11

/-- The amount of tax paid -/
def tax_paid : ℝ := 123.2

/-- Theorem stating that the tax-free amount satisfies the given equation -/
theorem tax_free_amount_satisfies_equation :
  tax_rate * (total_value - tax_free_amount) = tax_paid := by
  sorry

end NUMINAMATH_CALUDE_tax_free_amount_satisfies_equation_l3576_357653


namespace NUMINAMATH_CALUDE_complex_number_problem_l3576_357692

def z (b : ℝ) : ℂ := 3 + b * Complex.I

theorem complex_number_problem (b : ℝ) 
  (h : ∃ k : ℝ, (1 + 3 * Complex.I) * z b = k * Complex.I) :
  z b = 3 + Complex.I ∧ Complex.abs ((z b) / (2 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3576_357692


namespace NUMINAMATH_CALUDE_left_handed_to_non_throwers_ratio_l3576_357619

/- Define the football team -/
def total_players : ℕ := 70
def throwers : ℕ := 37
def right_handed : ℕ := 59

/- Theorem to prove the ratio -/
theorem left_handed_to_non_throwers_ratio :
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed := non_throwers - right_handed_non_throwers
  (left_handed : ℚ) / non_throwers = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_to_non_throwers_ratio_l3576_357619


namespace NUMINAMATH_CALUDE_line_plane_relationship_l3576_357625

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : perpendicular a b) 
  (h2 : parallel_line_plane a α) : 
  intersects b α ∨ contained_in b α ∨ parallel_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l3576_357625


namespace NUMINAMATH_CALUDE_birds_cannot_all_be_on_same_tree_l3576_357628

/-- Represents the state of birds on trees -/
structure BirdState where
  white : Nat -- Number of birds on white trees
  black : Nat -- Number of birds on black trees

/-- A move represents two birds switching to neighboring trees -/
def move (state : BirdState) : BirdState :=
  { white := state.white, black := state.black }

theorem birds_cannot_all_be_on_same_tree :
  ∀ (n : Nat), n > 0 →
  let initial_state : BirdState := { white := 3, black := 3 }
  let final_state := (move^[n]) initial_state
  (final_state.white ≠ 0 ∧ final_state.black ≠ 6) ∧
  (final_state.white ≠ 6 ∧ final_state.black ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_birds_cannot_all_be_on_same_tree_l3576_357628


namespace NUMINAMATH_CALUDE_factorial_ratio_l3576_357677

theorem factorial_ratio : Nat.factorial 10 / Nat.factorial 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3576_357677


namespace NUMINAMATH_CALUDE_power_difference_in_set_l3576_357684

theorem power_difference_in_set (m n : ℕ) :
  (3 ^ m - 2 ^ n ∈ ({-1, 5, 7} : Set ℤ)) ↔ 
  ((m, n) ∈ ({(0, 1), (1, 2), (2, 2), (2, 1)} : Set (ℕ × ℕ))) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_in_set_l3576_357684


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3576_357632

theorem inequality_system_solution_set : 
  {x : ℝ | 2 * x + 1 ≥ 3 ∧ 4 * x - 1 < 7} = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3576_357632


namespace NUMINAMATH_CALUDE_question_1_question_2_l3576_357658

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x - a < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Define the range of x for question 1
def range_x : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Define the range of a for question 2
def range_a : Set ℝ := {a | a > 3}

-- Theorem for question 1
theorem question_1 (a : ℝ) (h : a = 2) :
  {x : ℝ | p x a ∧ q x} = range_x := by sorry

-- Theorem for question 2
theorem question_2 :
  (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x) →
  a ∈ range_a := by sorry

end NUMINAMATH_CALUDE_question_1_question_2_l3576_357658


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l3576_357635

theorem sum_of_roots_squared_equation (x : ℝ) :
  (x - 3)^2 = 16 → ∃ y : ℝ, (y - 3)^2 = 16 ∧ x + y = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l3576_357635


namespace NUMINAMATH_CALUDE_proposition_is_false_l3576_357696

theorem proposition_is_false : 
  ¬(∀ x : ℝ, (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_is_false_l3576_357696


namespace NUMINAMATH_CALUDE_solution_set_characterization_l3576_357664

open Set

def solution_set (f : ℝ → ℝ) : Set ℝ := {x | x > 0 ∧ f x ≤ Real.log x}

theorem solution_set_characterization
  (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h1 : f 1 = 0)
  (h2 : ∀ x > 0, x * (deriv f x) > 1) :
  solution_set f = Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l3576_357664


namespace NUMINAMATH_CALUDE_bacteria_growth_30_min_l3576_357680

/-- Calculates the bacterial population after a given number of 5-minute intervals -/
def bacterial_population (initial_population : ℕ) (num_intervals : ℕ) : ℕ :=
  initial_population * (3 ^ num_intervals)

/-- Theorem stating the bacterial population after 30 minutes -/
theorem bacteria_growth_30_min (initial_population : ℕ) 
  (h1 : initial_population = 50) : 
  bacterial_population initial_population 6 = 36450 := by
  sorry

#eval bacterial_population 50 6

end NUMINAMATH_CALUDE_bacteria_growth_30_min_l3576_357680


namespace NUMINAMATH_CALUDE_right_triangles_count_l3576_357659

/-- Represents a geometric solid with front, top, and side views -/
structure GeometricSolid where
  front_view : Set (Point × Point)
  top_view : Set (Point × Point)
  side_view : Set (Point × Point)

/-- Counts the number of unique right-angled triangles in a geometric solid -/
def count_right_triangles (solid : GeometricSolid) : ℕ :=
  sorry

/-- Theorem stating that the number of right-angled triangles is 3 -/
theorem right_triangles_count (solid : GeometricSolid) :
  count_right_triangles solid = 3 :=
sorry

end NUMINAMATH_CALUDE_right_triangles_count_l3576_357659


namespace NUMINAMATH_CALUDE_sugar_price_reduction_l3576_357608

/-- Calculates the percentage reduction in sugar price given the original price and the amount that can be bought after reduction. -/
theorem sugar_price_reduction 
  (original_price : ℝ) 
  (budget : ℝ) 
  (extra_amount : ℝ) 
  (h1 : original_price = 8) 
  (h2 : budget = 120) 
  (h3 : extra_amount = 1) 
  (h4 : budget / original_price + extra_amount = budget / (budget / (budget / original_price + extra_amount))) : 
  (original_price - budget / (budget / original_price + extra_amount)) / original_price * 100 = 6.25 := by
sorry

end NUMINAMATH_CALUDE_sugar_price_reduction_l3576_357608


namespace NUMINAMATH_CALUDE_max_popsicles_for_8_dollars_l3576_357646

/-- Represents the different popsicle purchase options -/
inductive PopsicleOption
  | Single
  | Box3
  | Box5

/-- Returns the cost of a given popsicle option -/
def cost (option : PopsicleOption) : ℕ :=
  match option with
  | .Single => 1
  | .Box3 => 2
  | .Box5 => 3

/-- Returns the number of popsicles in a given option -/
def popsicles (option : PopsicleOption) : ℕ :=
  match option with
  | .Single => 1
  | .Box3 => 3
  | .Box5 => 5

/-- Represents a purchase of popsicles -/
structure Purchase where
  singles : ℕ
  box3s : ℕ
  box5s : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.singles * cost PopsicleOption.Single +
  p.box3s * cost PopsicleOption.Box3 +
  p.box5s * cost PopsicleOption.Box5

/-- Calculates the total number of popsicles in a purchase -/
def totalPopsicles (p : Purchase) : ℕ :=
  p.singles * popsicles PopsicleOption.Single +
  p.box3s * popsicles PopsicleOption.Box3 +
  p.box5s * popsicles PopsicleOption.Box5

/-- Theorem: The maximum number of popsicles that can be purchased with $8 is 13 -/
theorem max_popsicles_for_8_dollars :
  ∀ p : Purchase, totalCost p ≤ 8 → totalPopsicles p ≤ 13 ∧
  ∃ p' : Purchase, totalCost p' = 8 ∧ totalPopsicles p' = 13 :=
sorry

end NUMINAMATH_CALUDE_max_popsicles_for_8_dollars_l3576_357646


namespace NUMINAMATH_CALUDE_gift_wrapping_expenses_l3576_357641

def total_spent : ℝ := 700
def gift_cost : ℝ := 561

theorem gift_wrapping_expenses : total_spent - gift_cost = 139 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_expenses_l3576_357641


namespace NUMINAMATH_CALUDE_base13_addition_proof_l3576_357642

/-- Represents a digit in base 13 -/
inductive Base13Digit
  | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Represents a number in base 13 -/
def Base13Number := List Base13Digit

/-- Addition of two Base13Numbers -/
def add_base13 : Base13Number → Base13Number → Base13Number
  | _, _ => sorry  -- Implementation details omitted

/-- Conversion of a natural number to Base13Number -/
def nat_to_base13 : Nat → Base13Number
  | _ => sorry  -- Implementation details omitted

theorem base13_addition_proof :
  add_base13 (nat_to_base13 528) (nat_to_base13 274) =
  [Base13Digit.D7, Base13Digit.A, Base13Digit.C] :=
by sorry

end NUMINAMATH_CALUDE_base13_addition_proof_l3576_357642


namespace NUMINAMATH_CALUDE_irrational_sqrt_10_and_others_rational_l3576_357682

theorem irrational_sqrt_10_and_others_rational : 
  (Irrational (Real.sqrt 10)) ∧ 
  (¬ Irrational (1 / 7 : ℝ)) ∧ 
  (¬ Irrational (3.5 : ℝ)) ∧ 
  (¬ Irrational (-0.3030030003 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_irrational_sqrt_10_and_others_rational_l3576_357682


namespace NUMINAMATH_CALUDE_only_one_correct_proposition_l3576_357690

/-- A proposition about the relationship between lines and planes in 3D space -/
inductive GeometryProposition
  | InfinitePointsImpliesParallel
  | ParallelToPlaneImpliesParallelToLines
  | ParallelLineImpliesParallelToPlane
  | ParallelToPlaneImpliesNoIntersection

/-- Predicate to check if a geometry proposition is correct -/
def is_correct_proposition (p : GeometryProposition) : Prop :=
  match p with
  | GeometryProposition.InfinitePointsImpliesParallel => False
  | GeometryProposition.ParallelToPlaneImpliesParallelToLines => False
  | GeometryProposition.ParallelLineImpliesParallelToPlane => False
  | GeometryProposition.ParallelToPlaneImpliesNoIntersection => True

/-- Theorem stating that only one of the geometry propositions is correct -/
theorem only_one_correct_proposition :
  ∃! (p : GeometryProposition), is_correct_proposition p :=
sorry

end NUMINAMATH_CALUDE_only_one_correct_proposition_l3576_357690


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3576_357629

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  is_arithmetic_sequence a → a 2 = 5 → a 5 = 33 → a 3 + a 4 = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3576_357629


namespace NUMINAMATH_CALUDE_impossibility_of_arrangement_l3576_357671

theorem impossibility_of_arrangement : ¬ ∃ (a b : Fin 1986 → ℕ), 
  (∀ k : Fin 1986, b k - a k = k.val + 1) ∧
  (∀ i j : Fin 1986, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j)) ∧
  (∀ n : ℕ, n ∈ Set.range a ∪ Set.range b → n ≤ 2 * 1986) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_arrangement_l3576_357671


namespace NUMINAMATH_CALUDE_no_gar_is_tren_l3576_357611

-- Define the types
variable (U : Type) -- Universe set
variable (Gar Plin Tren : Set U) -- Subsets of U

-- Define the hypotheses
variable (h1 : Gar ⊆ Plin) -- All Gars are Plins
variable (h2 : Plin ∩ Tren = ∅) -- No Plins are Trens

-- State the theorem
theorem no_gar_is_tren : Gar ∩ Tren = ∅ := by
  sorry

end NUMINAMATH_CALUDE_no_gar_is_tren_l3576_357611


namespace NUMINAMATH_CALUDE_basketball_game_scores_l3576_357617

/-- Represents the scores of a team in a basketball game -/
structure TeamScores where
  q1 : ℕ
  q2 : ℕ
  q3 : ℕ
  q4 : ℕ

/-- Calculates the total score of a team -/
def totalScore (scores : TeamScores) : ℕ :=
  scores.q1 + scores.q2 + scores.q3 + scores.q4

/-- Calculates the score for the first half -/
def firstHalfScore (scores : TeamScores) : ℕ :=
  scores.q1 + scores.q2

/-- Calculates the score for the second half -/
def secondHalfScore (scores : TeamScores) : ℕ :=
  scores.q3 + scores.q4

/-- Checks if the scores form an increasing geometric sequence -/
def isGeometricSequence (scores : TeamScores) : Prop :=
  ∃ r : ℚ, r > 1 ∧ 
    scores.q2 = scores.q1 * r ∧
    scores.q3 = scores.q2 * r ∧
    scores.q4 = scores.q3 * r

/-- Checks if the scores form an increasing arithmetic sequence -/
def isArithmeticSequence (scores : TeamScores) : Prop :=
  ∃ d : ℕ, d > 0 ∧
    scores.q2 = scores.q1 + d ∧
    scores.q3 = scores.q2 + d ∧
    scores.q4 = scores.q3 + d

theorem basketball_game_scores 
  (eagles : TeamScores) (tigers : TeamScores) 
  (h1 : isGeometricSequence eagles)
  (h2 : isArithmeticSequence tigers)
  (h3 : firstHalfScore eagles = firstHalfScore tigers)
  (h4 : totalScore eagles = totalScore tigers + 2)
  (h5 : totalScore eagles ≤ 100)
  (h6 : totalScore tigers ≤ 100) :
  secondHalfScore eagles + secondHalfScore tigers = 116 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l3576_357617


namespace NUMINAMATH_CALUDE_school_garden_flowers_l3576_357679

theorem school_garden_flowers :
  let green_flowers : ℕ := 9
  let red_flowers : ℕ := 3 * green_flowers
  let yellow_flowers : ℕ := 12
  let total_flowers : ℕ := green_flowers + red_flowers + yellow_flowers + (green_flowers + red_flowers + yellow_flowers)
  total_flowers = 96 := by
  sorry

end NUMINAMATH_CALUDE_school_garden_flowers_l3576_357679


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3576_357624

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 < a*x + b ↔ 1 < x ∧ x < 3) → b^a = 81 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3576_357624


namespace NUMINAMATH_CALUDE_project_men_count_l3576_357672

/-- The number of men originally working on the project -/
def original_men : ℕ := 110

/-- The number of days it takes the original number of men to complete the work -/
def original_days : ℕ := 100

/-- The reduction in the number of men -/
def men_reduction : ℕ := 10

/-- The increase in days when the number of men is reduced -/
def days_increase : ℕ := 10

theorem project_men_count :
  (original_men * original_days = (original_men - men_reduction) * (original_days + days_increase)) →
  original_men = 110 := by
  sorry

end NUMINAMATH_CALUDE_project_men_count_l3576_357672


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3576_357688

/-- Represents a geometric sequence with first term a and common ratio q -/
structure GeometricSequence (α : Type*) [Field α] where
  a : α
  q : α

/-- Sum of first n terms of a geometric sequence -/
def sumGeometric {α : Type*} [Field α] (seq : GeometricSequence α) (n : ℕ) : α :=
  seq.a * (1 - seq.q ^ n) / (1 - seq.q)

theorem geometric_sequence_property {α : Type*} [Field α] (seq : GeometricSequence α) :
  (sumGeometric seq 3 + sumGeometric seq 6 = 2 * sumGeometric seq 9) →
  (seq.a * seq.q + seq.a * seq.q^4 = 4) →
  seq.a * seq.q^7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3576_357688


namespace NUMINAMATH_CALUDE_parallel_lines_in_plane_l3576_357600

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem parallel_lines_in_plane 
  (α β : Plane) (a b c : Line) :
  parallel a α →
  parallel b α →
  intersect β α c →
  contained_in a β →
  contained_in b β →
  parallel_lines a b :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_in_plane_l3576_357600


namespace NUMINAMATH_CALUDE_differential_equation_solution_l3576_357618

/-- The differential equation dy/dx + xy = x^2 has the general solution y(x) = x^3/4 + C/x -/
theorem differential_equation_solution (x : ℝ) (C : ℝ) :
  let y : ℝ → ℝ := λ x => x^3 / 4 + C / x
  let dy_dx : ℝ → ℝ := λ x => 3 * x^2 / 4 - C / x^2
  ∀ x ≠ 0, dy_dx x + x * y x = x^2 := by
sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l3576_357618


namespace NUMINAMATH_CALUDE_product_of_polynomials_l3576_357676

theorem product_of_polynomials (g h : ℚ) : 
  (∀ d : ℚ, (5*d^2 - 2*d + g) * (2*d^2 + h*d - 3) = 10*d^4 - 19*d^3 + g*d^2 + d - 6) →
  g + h = -1/2 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l3576_357676


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l3576_357638

theorem jelly_bean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_yellow + p_green = 0.5 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l3576_357638


namespace NUMINAMATH_CALUDE_ognev_phone_number_l3576_357685

/-- Represents a surname -/
structure Surname :=
  (name : String)

/-- Calculates the length of a surname -/
def surname_length (s : Surname) : Nat :=
  s.name.length

/-- Gets the position of a character in the alphabet (A=1, B=2, etc.) -/
def char_position (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat + 1
  else if 'a' ≤ c ∧ c ≤ 'z' then c.toNat - 'a'.toNat + 1
  else 0

/-- Calculates the phone number for a given surname -/
def phone_number (s : Surname) : Nat :=
  let len := surname_length s
  let first_pos := char_position s.name.front
  let last_pos := char_position s.name.back
  len * 1000 + first_pos * 100 + last_pos

/-- The theorem to be proved -/
theorem ognev_phone_number :
  phone_number { name := "Ognev" } = 5163 := by
  sorry

end NUMINAMATH_CALUDE_ognev_phone_number_l3576_357685


namespace NUMINAMATH_CALUDE_total_jogging_distance_l3576_357675

def monday_distance : ℕ := 2
def tuesday_distance : ℕ := 5
def wednesday_distance : ℕ := 9

theorem total_jogging_distance :
  monday_distance + tuesday_distance + wednesday_distance = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_jogging_distance_l3576_357675


namespace NUMINAMATH_CALUDE_candy_distribution_l3576_357694

theorem candy_distribution (total_candies : ℕ) (num_bags : ℕ) (candies_per_bag : ℕ) :
  total_candies = 15 →
  num_bags = 5 →
  total_candies = num_bags * candies_per_bag →
  candies_per_bag = 3 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3576_357694


namespace NUMINAMATH_CALUDE_no_solution_iff_m_zero_l3576_357607

theorem no_solution_iff_m_zero (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (2 - x) / (1 - x) ≠ (m + x) / (1 - x) + 1) ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_zero_l3576_357607


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l3576_357631

def point_A (x : ℝ) : ℝ × ℝ := (x - 4, 2 * x + 3)

theorem distance_to_y_axis (x : ℝ) : 
  (|x - 4| = 1) ↔ (x = 5 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l3576_357631


namespace NUMINAMATH_CALUDE_sixth_day_work_time_l3576_357614

def work_time (n : ℕ) : ℕ := 15 * 2^(n - 1)

theorem sixth_day_work_time :
  work_time 6 = 8 * 60 := by
  sorry

end NUMINAMATH_CALUDE_sixth_day_work_time_l3576_357614


namespace NUMINAMATH_CALUDE_jason_attended_twelve_games_l3576_357687

/-- The number of games Jason attended given his planned and missed games -/
def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (missed : ℕ) : ℕ :=
  (planned_this_month + planned_last_month) - missed

/-- Theorem stating that Jason attended 12 games given the problem conditions -/
theorem jason_attended_twelve_games :
  games_attended 11 17 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_jason_attended_twelve_games_l3576_357687


namespace NUMINAMATH_CALUDE_dance_ratio_l3576_357606

/-- Given the conditions of a dance, prove the ratio of boys to girls -/
theorem dance_ratio :
  ∀ (boys girls teachers : ℕ),
  girls = 60 →
  teachers = boys / 5 →
  boys + girls + teachers = 114 →
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ boys * b = girls * a ∧ a = 3 ∧ b = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_dance_ratio_l3576_357606


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l3576_357615

theorem neither_sufficient_nor_necessary (a b : ℝ) :
  (∃ x y : ℝ, x - y > 0 ∧ x^2 - y^2 ≤ 0) ∧
  (∃ x y : ℝ, x - y ≤ 0 ∧ x^2 - y^2 > 0) :=
sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l3576_357615


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3576_357657

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3576_357657


namespace NUMINAMATH_CALUDE_last_three_digits_are_427_l3576_357674

/-- A function that generates the nth digit in the list of increasing positive integers 
    starting with 2 and containing all numbers with a first digit of 2 -/
def nthDigitInList (n : ℕ) : ℕ := sorry

/-- The last three digits of the 2000-digit sequence -/
def lastThreeDigits : ℕ × ℕ × ℕ := (nthDigitInList 1998, nthDigitInList 1999, nthDigitInList 2000)

theorem last_three_digits_are_427 : lastThreeDigits = (4, 2, 7) := by sorry

end NUMINAMATH_CALUDE_last_three_digits_are_427_l3576_357674


namespace NUMINAMATH_CALUDE_associated_number_equality_l3576_357691

-- Define the associated number function
def associated_number (x : ℚ) : ℚ :=
  if x ≥ 0 then 2 * x - 1 else -2 * x + 1

-- State the theorem
theorem associated_number_equality (a b : ℚ) (ha : a > 0) (hb : b < 0) 
  (h_eq : associated_number a = associated_number b) : 
  (a + b)^2 - 2*a - 2*b = -1 := by sorry

end NUMINAMATH_CALUDE_associated_number_equality_l3576_357691


namespace NUMINAMATH_CALUDE_f_symmetric_about_pi_third_l3576_357650

/-- A function is symmetric about a point (a, 0) if f(a + x) = -f(a - x) for all x in the domain of f -/
def SymmetricAboutPoint (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = -f (a - x)

/-- The tangent function -/
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

/-- The given function f(x) = tan(x + π/6) -/
noncomputable def f (x : ℝ) : ℝ := tan (x + Real.pi / 6)

/-- Theorem stating that f(x) = tan(x + π/6) is symmetric about the point (π/3, 0) -/
theorem f_symmetric_about_pi_third : SymmetricAboutPoint f (Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_f_symmetric_about_pi_third_l3576_357650


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3576_357648

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x ≤ x + 1) ↔ (∀ x : ℝ, x > 0 → Real.log x > x + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3576_357648


namespace NUMINAMATH_CALUDE_rotten_eggs_calculation_l3576_357639

/-- The percentage of spoiled milk bottles -/
def spoiled_milk_percentage : ℝ := 0.20

/-- The percentage of flour canisters with weevils -/
def weevil_flour_percentage : ℝ := 0.25

/-- The probability of all three ingredients being good -/
def all_good_probability : ℝ := 0.24

/-- The percentage of rotten eggs -/
def rotten_eggs_percentage : ℝ := 0.60

theorem rotten_eggs_calculation :
  (1 - spoiled_milk_percentage) * (1 - rotten_eggs_percentage) * (1 - weevil_flour_percentage) = all_good_probability :=
by sorry

end NUMINAMATH_CALUDE_rotten_eggs_calculation_l3576_357639


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3576_357681

theorem decimal_to_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 0.425875 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.425875 → c ≤ a ∧ d ≤ b → 
  (a : ℕ) + (b : ℕ) = 11407 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3576_357681


namespace NUMINAMATH_CALUDE_shaded_area_problem_l3576_357695

theorem shaded_area_problem (diagonal : ℝ) (num_squares : ℕ) : 
  diagonal = 10 → num_squares = 25 → 
  (diagonal^2 / 2) = (num_squares : ℝ) * (diagonal^2 / (2 * num_squares : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_problem_l3576_357695


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l3576_357647

/-- The ellipse with semi-major axis 13 and semi-minor axis 12 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 169) + (p.2^2 / 144) = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focal_property (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  P ∈ Ellipse → F₁ ∈ Foci → F₂ ∈ Foci → distance P F₁ = 4 →
  distance P F₂ = 22 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l3576_357647


namespace NUMINAMATH_CALUDE_line_equation_slope_intercept_l3576_357665

/-- Given a line equation, prove its slope and y-intercept -/
theorem line_equation_slope_intercept :
  ∀ (x y : ℝ),
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - 8) = 0 →
  ∃ (m b : ℝ), m = 3/4 ∧ b = 13/2 ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_equation_slope_intercept_l3576_357665


namespace NUMINAMATH_CALUDE_seal_initial_money_l3576_357633

/-- Represents the amount of coins Seal has at each stage --/
def seal_money (initial : ℕ) : ℕ → ℕ
| 0 => initial  -- Initial amount
| 1 => 2 * initial - 20  -- After first crossing
| 2 => 2 * (2 * initial - 20) - 40  -- After second crossing
| 3 => 2 * (2 * (2 * initial - 20) - 40) - 60  -- After third crossing
| _ => 0  -- We only care about the first three crossings

/-- The theorem stating that Seal must have started with 25 coins --/
theorem seal_initial_money : 
  ∃ (initial : ℕ), 
    initial = 25 ∧ 
    seal_money initial 0 > 0 ∧
    seal_money initial 1 > 0 ∧
    seal_money initial 2 > 0 ∧
    seal_money initial 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_seal_initial_money_l3576_357633


namespace NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_progression_l3576_357612

theorem binomial_coeff_not_arithmetic_progression (n k : ℕ) (h1 : k ≤ n - 3) :
  ¬∃ (a d : ℤ), 
    (Nat.choose n k : ℤ) = a ∧
    (Nat.choose n (k + 1) : ℤ) = a + d ∧
    (Nat.choose n (k + 2) : ℤ) = a + 2*d ∧
    (Nat.choose n (k + 3) : ℤ) = a + 3*d :=
by sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_progression_l3576_357612


namespace NUMINAMATH_CALUDE_percentage_decrease_l3576_357604

/-- Proves that for an original number of 40, if the difference between its value 
    increased by 25% and its value decreased by x% is 22, then x = 30. -/
theorem percentage_decrease (x : ℝ) : 
  (40 + 0.25 * 40) - (40 - 0.01 * x * 40) = 22 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l3576_357604


namespace NUMINAMATH_CALUDE_robin_water_consumption_l3576_357673

def bottles_morning : ℕ := sorry
def bottles_afternoon : ℕ := sorry
def total_bottles : ℕ := 14

theorem robin_water_consumption :
  (bottles_morning = bottles_afternoon) →
  (bottles_morning + bottles_afternoon = total_bottles) →
  bottles_morning = 7 := by
  sorry

end NUMINAMATH_CALUDE_robin_water_consumption_l3576_357673
