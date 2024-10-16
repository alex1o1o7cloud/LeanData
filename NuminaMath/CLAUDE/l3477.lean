import Mathlib

namespace NUMINAMATH_CALUDE_base_7_321_equals_162_l3477_347781

def base_7_to_10 (a b c : ℕ) : ℕ := a * 7^2 + b * 7 + c

theorem base_7_321_equals_162 : base_7_to_10 3 2 1 = 162 := by
  sorry

end NUMINAMATH_CALUDE_base_7_321_equals_162_l3477_347781


namespace NUMINAMATH_CALUDE_fiftieth_parenthesis_sum_l3477_347731

def sequence_term (n : ℕ) : ℕ := 24 * ((n - 1) / 4) + 1

def parenthesis_sum (n : ℕ) : ℕ :=
  if n % 4 = 1 then sequence_term n
  else if n % 4 = 2 then sequence_term n + (sequence_term n + 2)
  else if n % 4 = 3 then sequence_term n + (sequence_term n + 2) + (sequence_term n + 4)
  else sequence_term n

theorem fiftieth_parenthesis_sum : parenthesis_sum 50 = 392 := by sorry

end NUMINAMATH_CALUDE_fiftieth_parenthesis_sum_l3477_347731


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3477_347737

/-- An arithmetic sequence with second term -5 and common difference 3 has first term -8 -/
theorem arithmetic_sequence_first_term (a : ℕ → ℤ) :
  (∀ n, a (n + 1) = a n + 3) →  -- arithmetic sequence condition
  a 2 = -5 →                    -- given second term
  a 1 = -8 :=                   -- conclusion: first term
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3477_347737


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l3477_347766

/-- Given a polynomial function g(x) = px^4 + qx^3 + rx^2 + sx + t 
    where g(-1) = 1, prove that 16p - 8q + 4r - 2s + t = 1 -/
theorem polynomial_value_at_negative_one 
  (p q r s t : ℝ) 
  (g : ℝ → ℝ)
  (h1 : ∀ x, g x = p * x^4 + q * x^3 + r * x^2 + s * x + t)
  (h2 : g (-1) = 1) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l3477_347766


namespace NUMINAMATH_CALUDE_cone_volume_l3477_347773

theorem cone_volume (d h : ℝ) (h1 : d = 16) (h2 : h = 12) :
  (1 / 3 : ℝ) * π * (d / 2) ^ 2 * h = 256 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3477_347773


namespace NUMINAMATH_CALUDE_system_solution_l3477_347703

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (y - 2 * Real.sqrt (x * y) - Real.sqrt (y / x) + 2 = 0) ∧
  (3 * x^2 * y^2 + y^4 = 84) →
  ((x = 1/3 ∧ y = 3) ∨ (x = (21/76)^(1/4) ∧ y = 2 * (84/19)^(1/4))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3477_347703


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l3477_347734

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l3477_347734


namespace NUMINAMATH_CALUDE_total_books_proof_l3477_347785

/-- The number of books taken by the librarian -/
def books_taken_by_librarian : ℕ := 7

/-- The number of books Jerry can fit on one shelf -/
def books_per_shelf : ℕ := 3

/-- The number of shelves Jerry needs -/
def shelves_needed : ℕ := 9

/-- The total number of books to put away -/
def total_books : ℕ := books_per_shelf * shelves_needed + books_taken_by_librarian

theorem total_books_proof : total_books = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_books_proof_l3477_347785


namespace NUMINAMATH_CALUDE_toenail_size_ratio_l3477_347708

/-- Represents the capacity of the jar in terms of regular toenails -/
def jar_capacity : ℕ := 100

/-- Represents the number of big toenails in the jar -/
def big_toenails : ℕ := 20

/-- Represents the number of regular toenails initially in the jar -/
def regular_toenails : ℕ := 40

/-- Represents the additional regular toenails that can fit in the jar -/
def additional_regular_toenails : ℕ := 20

/-- Represents the ratio of the size of a big toenail to a regular toenail -/
def big_to_regular_ratio : ℚ := 2

theorem toenail_size_ratio :
  (jar_capacity - regular_toenails - additional_regular_toenails) / big_toenails = big_to_regular_ratio :=
sorry

end NUMINAMATH_CALUDE_toenail_size_ratio_l3477_347708


namespace NUMINAMATH_CALUDE_sum_of_pyramid_edges_l3477_347706

/-- Represents a pyramid structure -/
structure Pyramid where
  vertices : ℕ

/-- The number of edges in a pyramid -/
def Pyramid.edges (p : Pyramid) : ℕ := 2 * p.vertices - 2

/-- Theorem: For three pyramids with a total of 40 vertices, the sum of their edges is 74 -/
theorem sum_of_pyramid_edges (a b c : Pyramid) 
  (h : a.vertices + b.vertices + c.vertices = 40) : 
  a.edges + b.edges + c.edges = 74 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_pyramid_edges_l3477_347706


namespace NUMINAMATH_CALUDE_louisa_second_day_travel_l3477_347761

/-- Proves that Louisa traveled 280 miles on the second day of her vacation --/
theorem louisa_second_day_travel :
  ∀ (first_day_distance second_day_distance : ℝ) 
    (speed : ℝ) 
    (time_difference : ℝ),
  first_day_distance = 160 →
  speed = 40 →
  time_difference = 3 →
  first_day_distance / speed + time_difference = second_day_distance / speed →
  second_day_distance = 280 := by
sorry

end NUMINAMATH_CALUDE_louisa_second_day_travel_l3477_347761


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3477_347758

/-- A line is tangent to a circle if and only if the distance from the center of the circle
    to the line is equal to the radius of the circle. -/
axiom tangent_line_distance_eq_radius {a b c : ℝ} {x₀ y₀ r : ℝ} :
  (∀ x y, (x - x₀)^2 + (y - y₀)^2 = r^2 → a*x + b*y + c = 0 → 
    (x - x₀)^2 + (y - y₀)^2 = r^2 ∧ a*x + b*y + c = 0) ↔ 
  |a*x₀ + b*y₀ + c| / Real.sqrt (a^2 + b^2) = r

/-- The theorem to be proved -/
theorem tangent_line_to_circle (m : ℝ) (h_pos : m > 0) 
  (h_tangent : ∀ x y, (x - 3)^2 + (y - 4)^2 = 4 → 3*x - 4*y - m = 0 → 
    (x - 3)^2 + (y - 4)^2 = 4 ∧ 3*x - 4*y - m = 0) : 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3477_347758


namespace NUMINAMATH_CALUDE_ellipse_equation_l3477_347732

/-- Given an ellipse with foci F₁(0,-4) and F₂(0,4), and the shortest distance from a point on the ellipse to F₁ is 2, the equation of the ellipse is x²/20 + y²/36 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let f₁ : ℝ × ℝ := (0, -4)
  let f₂ : ℝ × ℝ := (0, 4)
  let shortest_distance : ℝ := 2
  x^2 / 20 + y^2 / 36 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3477_347732


namespace NUMINAMATH_CALUDE_correct_pages_per_booklet_l3477_347707

/-- The number of booklets in Jack's short story section -/
def num_booklets : ℕ := 49

/-- The total number of pages in all booklets -/
def total_pages : ℕ := 441

/-- The number of pages per booklet -/
def pages_per_booklet : ℕ := total_pages / num_booklets

theorem correct_pages_per_booklet : pages_per_booklet = 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_pages_per_booklet_l3477_347707


namespace NUMINAMATH_CALUDE_largest_angle_is_70_l3477_347789

-- Define a right angle in degrees
def right_angle : ℝ := 90

-- Define the triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_eq_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

-- Define the specific conditions of our triangle
def special_triangle (t : Triangle) : Prop :=
  ∃ (x : ℝ), 
    t.angle1 = x ∧ 
    t.angle2 = x + 20 ∧
    t.angle1 + t.angle2 = (4/3) * right_angle

-- Theorem statement
theorem largest_angle_is_70 (t : Triangle) (h : special_triangle t) : 
  max t.angle1 (max t.angle2 t.angle3) = 70 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_is_70_l3477_347789


namespace NUMINAMATH_CALUDE_integral_inequality_l3477_347701

open Real MeasureTheory

theorem integral_inequality : 
  ∫ x in (1:ℝ)..2, (1 / x) < ∫ x in (1:ℝ)..2, x ∧ ∫ x in (1:ℝ)..2, x < ∫ x in (1:ℝ)..2, exp x := by
  sorry

end NUMINAMATH_CALUDE_integral_inequality_l3477_347701


namespace NUMINAMATH_CALUDE_seconds_in_minutes_l3477_347702

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes we are converting to seconds -/
def minutes : ℚ := 12.5

/-- Theorem: The number of seconds in 12.5 minutes is 750 -/
theorem seconds_in_minutes : (minutes * seconds_per_minute : ℚ) = 750 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_minutes_l3477_347702


namespace NUMINAMATH_CALUDE_simplify_expression_l3477_347760

theorem simplify_expression (x : ℝ) : 
  x * (3 * x^2 - 2) - 5 * (x^2 - 2*x + 7) = 3 * x^3 - 5 * x^2 + 8 * x - 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3477_347760


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_72_l3477_347771

theorem smallest_non_factor_product_of_72 (x y : ℕ+) : 
  x ≠ y →
  x ∣ 72 →
  y ∣ 72 →
  ¬(x * y ∣ 72) →
  (∀ a b : ℕ+, a ≠ b → a ∣ 72 → b ∣ 72 → ¬(a * b ∣ 72) → x * y ≤ a * b) →
  x * y = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_72_l3477_347771


namespace NUMINAMATH_CALUDE_triangle_determines_plane_l3477_347739

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Three points are non-collinear if they do not lie on the same line -/
def NonCollinear (p1 p2 p3 : Point3D) : Prop :=
  ¬∃ (t : ℝ), p3.x = p1.x + t * (p2.x - p1.x) ∧
               p3.y = p1.y + t * (p2.y - p1.y) ∧
               p3.z = p1.z + t * (p2.z - p1.z)

/-- A plane contains a point if the point satisfies the plane equation -/
def PlaneContainsPoint (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Theorem: Three non-collinear points uniquely determine a plane -/
theorem triangle_determines_plane (p1 p2 p3 : Point3D) 
  (h : NonCollinear p1 p2 p3) : 
  ∃! (plane : Plane), PlaneContainsPoint plane p1 ∧ 
                      PlaneContainsPoint plane p2 ∧ 
                      PlaneContainsPoint plane p3 :=
sorry

end NUMINAMATH_CALUDE_triangle_determines_plane_l3477_347739


namespace NUMINAMATH_CALUDE_area_of_triangle_formed_by_tangent_points_l3477_347750

/-- Represents a circle with a center point and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to the x-axis -/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  let (_, y) := c.center
  y = c.radius

/-- The main theorem -/
theorem area_of_triangle_formed_by_tangent_points : 
  ∀ (c1 c2 c3 : Circle),
  c1.radius = 1 ∧ c2.radius = 3 ∧ c3.radius = 5 →
  are_externally_tangent c1 c2 ∧ 
  are_externally_tangent c2 c3 ∧ 
  are_externally_tangent c1 c3 →
  is_tangent_to_x_axis c1 ∧ 
  is_tangent_to_x_axis c2 ∧ 
  is_tangent_to_x_axis c3 →
  let (x1, _) := c1.center
  let (x2, _) := c2.center
  let (x3, _) := c3.center
  (1/2) * (|x2 - x1| + |x3 - x2| + |x3 - x1|) * (c3.radius - c1.radius) = 6 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_formed_by_tangent_points_l3477_347750


namespace NUMINAMATH_CALUDE_trouser_sale_price_l3477_347727

theorem trouser_sale_price (original_price : ℝ) (discount_percentage : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_percentage = 70) : 
  original_price * (1 - discount_percentage / 100) = 30 := by
  sorry

end NUMINAMATH_CALUDE_trouser_sale_price_l3477_347727


namespace NUMINAMATH_CALUDE_banana_groups_count_l3477_347776

def total_bananas : ℕ := 290
def bananas_per_group : ℕ := 145

theorem banana_groups_count : total_bananas / bananas_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_banana_groups_count_l3477_347776


namespace NUMINAMATH_CALUDE_mall_sales_problem_l3477_347791

/-- Represents the cost price of the item in yuan -/
def cost_price : ℝ := 500

/-- Represents the markup percentage in the first month -/
def markup1 : ℝ := 0.2

/-- Represents the markup percentage in the second month -/
def markup2 : ℝ := 0.1

/-- Represents the profit in the first month in yuan -/
def profit1 : ℝ := 6000

/-- Represents the increase in profit in the second month in yuan -/
def profit_increase : ℝ := 2000

/-- Represents the increase in sales volume in the second month -/
def sales_increase : ℕ := 100

/-- Theorem stating the cost price and second month sales volume -/
theorem mall_sales_problem :
  (cost_price * markup1 * (profit1 / (cost_price * markup1)) +
   cost_price * markup2 * ((profit1 + profit_increase) / (cost_price * markup2)) -
   cost_price * markup1 * (profit1 / (cost_price * markup1))) / cost_price = sales_increase ∧
  (profit1 + profit_increase) / (cost_price * markup2) = 160 :=
by sorry

end NUMINAMATH_CALUDE_mall_sales_problem_l3477_347791


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_p_less_than_one_l3477_347711

theorem intersection_nonempty_iff_p_less_than_one (p : ℝ) :
  let M : Set ℝ := {x | x ≤ 1}
  let N : Set ℝ := {x | x > p}
  (M ∩ N).Nonempty ↔ p < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_p_less_than_one_l3477_347711


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l3477_347725

theorem complex_roots_on_circle : 
  ∀ z : ℂ, (z - 2)^4 = 16 * z^4 → Complex.abs (z - (2/3 : ℂ)) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l3477_347725


namespace NUMINAMATH_CALUDE_days_worked_by_a_l3477_347797

/-- Represents the number of days worked by person a -/
def days_a : ℕ := 16

/-- Represents the number of days worked by person b -/
def days_b : ℕ := 9

/-- Represents the number of days worked by person c -/
def days_c : ℕ := 4

/-- Represents the daily wage ratio of person a -/
def wage_ratio_a : ℚ := 3

/-- Represents the daily wage ratio of person b -/
def wage_ratio_b : ℚ := 4

/-- Represents the daily wage ratio of person c -/
def wage_ratio_c : ℚ := 5

/-- Represents the daily wage of person c -/
def wage_c : ℚ := 71.15384615384615

/-- Represents the total earnings of all three workers -/
def total_earnings : ℚ := 1480

/-- Theorem stating that given the conditions, the number of days worked by person a is 16 -/
theorem days_worked_by_a : 
  (days_a : ℚ) * (wage_ratio_a * wage_c / wage_ratio_c) + 
  (days_b : ℚ) * (wage_ratio_b * wage_c / wage_ratio_c) + 
  (days_c : ℚ) * wage_c = total_earnings :=
sorry

end NUMINAMATH_CALUDE_days_worked_by_a_l3477_347797


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1547_l3477_347763

theorem smallest_prime_factor_of_1547 :
  Nat.minFac 1547 = 7 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1547_l3477_347763


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l3477_347788

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equals_negative_two :
  (1 + i)^3 / (1 - i) = -2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l3477_347788


namespace NUMINAMATH_CALUDE_sum_binary_digits_345_l3477_347799

/-- Sum of binary digits of a natural number -/
def sum_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- The sum of the binary digits of 345 is 5 -/
theorem sum_binary_digits_345 : sum_binary_digits 345 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_binary_digits_345_l3477_347799


namespace NUMINAMATH_CALUDE_prob_king_or_queen_is_two_thirteenths_l3477_347716

-- Define the properties of a standard deck
structure StandardDeck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)
  (h_total : total_cards = 52)
  (h_ranks : num_ranks = 13)
  (h_suits : num_suits = 4)
  (h_kings : num_kings = 4)
  (h_queens : num_queens = 4)
  (h_cards_per_rank : total_cards = num_ranks * num_suits)

-- Define the probability function
def probability_king_or_queen (deck : StandardDeck) : ℚ :=
  (deck.num_kings + deck.num_queens : ℚ) / deck.total_cards

-- State the theorem
theorem prob_king_or_queen_is_two_thirteenths (deck : StandardDeck) :
  probability_king_or_queen deck = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_or_queen_is_two_thirteenths_l3477_347716


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3477_347712

/-- The volume of a cylinder minus two congruent cones -/
theorem cylinder_minus_cones_volume 
  (r : ℝ) 
  (h_cone : ℝ) 
  (h_cyl : ℝ) 
  (h_r : r = 10) 
  (h_cone_height : h_cone = 15) 
  (h_cyl_height : h_cyl = 30) : 
  π * r^2 * h_cyl - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3477_347712


namespace NUMINAMATH_CALUDE_remaining_area_of_19x11_rectangle_l3477_347790

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of the remaining rectangle after placing the largest possible squares inside a given rectangle -/
def remainingArea (rect : Rectangle) : ℝ :=
  sorry

/-- Theorem stating that the remaining area of a 19x11 rectangle after placing four squares is 6 -/
theorem remaining_area_of_19x11_rectangle : 
  remainingArea ⟨19, 11⟩ = 6 := by sorry

end NUMINAMATH_CALUDE_remaining_area_of_19x11_rectangle_l3477_347790


namespace NUMINAMATH_CALUDE_original_group_size_l3477_347744

/-- Given a group of men working on a task, this theorem proves that the original number of men is 42, based on the conditions provided. -/
theorem original_group_size (total_days : ℕ) (remaining_days : ℕ) (absent_men : ℕ) : 
  (total_days = 17) → (remaining_days = 21) → (absent_men = 8) →
  ∃ (original_size : ℕ), 
    (original_size > absent_men) ∧ 
    (1 : ℚ) / (total_days * original_size) = (1 : ℚ) / (remaining_days * (original_size - absent_men)) ∧
    original_size = 42 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l3477_347744


namespace NUMINAMATH_CALUDE_correct_algorithm_l3477_347733

theorem correct_algorithm : 
  ((-8) / (-4) = 8 / 4) ∧ 
  ((-5) + 9 ≠ -(9 - 5)) ∧ 
  (7 - (-10) ≠ 7 - 10) ∧ 
  ((-5) * 0 ≠ -5) := by
  sorry

end NUMINAMATH_CALUDE_correct_algorithm_l3477_347733


namespace NUMINAMATH_CALUDE_gcd_459_357_l3477_347782

def euclidean_gcd (a b : ℕ) : ℕ := sorry

def successive_subtraction_gcd (a b : ℕ) : ℕ := sorry

theorem gcd_459_357 : 
  euclidean_gcd 459 357 = 51 ∧ 
  successive_subtraction_gcd 459 357 = 51 := by sorry

end NUMINAMATH_CALUDE_gcd_459_357_l3477_347782


namespace NUMINAMATH_CALUDE_divisor_of_p_l3477_347792

theorem divisor_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 75)
  (h4 : 120 < Nat.gcd s p ∧ Nat.gcd s p < 180) :
  5 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_p_l3477_347792


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3477_347778

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3477_347778


namespace NUMINAMATH_CALUDE_election_vote_difference_l3477_347700

theorem election_vote_difference (total_votes : ℕ) (invalid_votes : ℕ) (losing_percentage : ℚ) :
  total_votes = 12600 →
  invalid_votes = 100 →
  losing_percentage = 30 / 100 →
  ∃ (valid_votes winning_votes losing_votes : ℕ),
    valid_votes = total_votes - invalid_votes ∧
    losing_votes = (losing_percentage * valid_votes).floor ∧
    winning_votes = valid_votes - losing_votes ∧
    winning_votes - losing_votes = 5000 := by
  sorry

#check election_vote_difference

end NUMINAMATH_CALUDE_election_vote_difference_l3477_347700


namespace NUMINAMATH_CALUDE_fraction_problem_l3477_347742

theorem fraction_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → 
  (40/100 : ℝ) * N = 384 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3477_347742


namespace NUMINAMATH_CALUDE_basketball_probabilities_l3477_347759

/-- Represents a basketball player's shooting accuracy -/
structure Player where
  accuracy : ℝ
  accuracy_nonneg : 0 ≤ accuracy
  accuracy_le_one : accuracy ≤ 1

/-- The probability of a player hitting at least one shot in two attempts -/
def prob_hit_at_least_one (player : Player) : ℝ :=
  1 - (1 - player.accuracy)^2

/-- The probability of two players making exactly three shots in four attempts -/
def prob_three_out_of_four (player_a player_b : Player) : ℝ :=
  2 * (player_a.accuracy * (1 - player_a.accuracy) * player_b.accuracy^2 +
       player_b.accuracy * (1 - player_b.accuracy) * player_a.accuracy^2)

theorem basketball_probabilities 
  (player_a : Player) 
  (player_b : Player) 
  (h_a : player_a.accuracy = 1/2) 
  (h_b : (1 - player_b.accuracy)^2 = 1/16) :
  prob_hit_at_least_one player_a = 3/4 ∧ 
  prob_three_out_of_four player_a player_b = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probabilities_l3477_347759


namespace NUMINAMATH_CALUDE_reservoir_fullness_after_storm_l3477_347757

theorem reservoir_fullness_after_storm 
  (original_content : ℝ) 
  (original_percentage : ℝ) 
  (storm_deposit : ℝ) 
  (h1 : original_content = 245)
  (h2 : original_percentage = 54.44444444444444)
  (h3 : storm_deposit = 115) :
  let total_capacity := original_content / (original_percentage / 100)
  let new_content := original_content + storm_deposit
  (new_content / total_capacity) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_reservoir_fullness_after_storm_l3477_347757


namespace NUMINAMATH_CALUDE_seven_twentyfour_twentyfive_pythagorean_triple_l3477_347754

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem seven_twentyfour_twentyfive_pythagorean_triple :
  is_pythagorean_triple 7 24 25 :=
by
  sorry

end NUMINAMATH_CALUDE_seven_twentyfour_twentyfive_pythagorean_triple_l3477_347754


namespace NUMINAMATH_CALUDE_complement_A_in_U_l3477_347729

-- Define the sets U and A
def U : Set ℝ := Set.Icc 0 1
def A : Set ℝ := Set.Ico 0 1

-- State the theorem
theorem complement_A_in_U : (U \ A) = {1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l3477_347729


namespace NUMINAMATH_CALUDE_min_vertical_distance_l3477_347769

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 - 4*x - 1

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x_min : ℝ), ∀ (x : ℝ), vertical_distance x_min ≤ vertical_distance x ∧ 
  vertical_distance x_min = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l3477_347769


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l3477_347746

/-- 
Given a convex pentagon with interior angles measuring y, 2y+2, 3y-3, 4y+4, and 5y-5 degrees,
where the sum of these angles is 540 degrees, prove that the largest angle measures 176 degrees
when rounded to the nearest integer.
-/
theorem pentagon_largest_angle : 
  ∀ y : ℝ, 
  y + (2*y+2) + (3*y-3) + (4*y+4) + (5*y-5) = 540 → 
  round (5*y - 5) = 176 := by
sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l3477_347746


namespace NUMINAMATH_CALUDE_james_shirts_count_l3477_347741

/-- The number of shirts James has -/
def num_shirts : ℕ := 10

/-- The number of pairs of pants James has -/
def num_pants : ℕ := 12

/-- The time it takes to fix a shirt (in hours) -/
def shirt_time : ℚ := 3/2

/-- The hourly rate charged by the tailor (in dollars) -/
def hourly_rate : ℕ := 30

/-- The total cost for fixing all shirts and pants (in dollars) -/
def total_cost : ℕ := 1530

theorem james_shirts_count :
  num_shirts = 10 ∧
  num_pants = 12 ∧
  shirt_time = 3/2 ∧
  hourly_rate = 30 ∧
  total_cost = 1530 →
  num_shirts * (shirt_time * hourly_rate) + num_pants * (2 * shirt_time * hourly_rate) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_james_shirts_count_l3477_347741


namespace NUMINAMATH_CALUDE_crayons_lost_l3477_347723

theorem crayons_lost (initial : ℕ) (given_away : ℕ) (final : ℕ) 
  (h1 : initial = 440)
  (h2 : given_away = 111)
  (h3 : final = 223) :
  initial - given_away - final = 106 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_l3477_347723


namespace NUMINAMATH_CALUDE_jims_initial_reading_speed_l3477_347774

/-- Represents Jim's reading habits and speeds -/
structure ReadingHabits where
  initial_speed : ℝ  -- Initial reading speed in pages per hour
  initial_hours : ℝ  -- Initial hours read per week
  new_speed : ℝ      -- New reading speed in pages per hour
  new_hours : ℝ      -- New hours read per week

/-- Theorem stating Jim's initial reading speed -/
theorem jims_initial_reading_speed 
  (h : ReadingHabits) 
  (initial_pages : h.initial_speed * h.initial_hours = 600) 
  (speed_increase : h.new_speed = 1.5 * h.initial_speed)
  (time_decrease : h.new_hours = h.initial_hours - 4)
  (new_pages : h.new_speed * h.new_hours = 660) : 
  h.initial_speed = 40 := by
  sorry


end NUMINAMATH_CALUDE_jims_initial_reading_speed_l3477_347774


namespace NUMINAMATH_CALUDE_ali_seashells_left_l3477_347709

/-- The number of seashells Ali has left after giving some away and selling half --/
def seashells_left (initial : ℕ) (given_to_friends : ℕ) (given_to_brothers : ℕ) : ℕ :=
  let remaining_after_giving := initial - given_to_friends - given_to_brothers
  remaining_after_giving - remaining_after_giving / 2

/-- Theorem stating that Ali has 55 seashells left --/
theorem ali_seashells_left : seashells_left 180 40 30 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ali_seashells_left_l3477_347709


namespace NUMINAMATH_CALUDE_vector_magnitude_l3477_347745

theorem vector_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (-1, x)
  (2 • a - b) • b = 0 → ‖a‖ = 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3477_347745


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l3477_347704

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define vector addition
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define vector from a point to another
def vec_from_to (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

-- Define the condition for point M
def M_condition (M A B : ℝ × ℝ) : Prop :=
  vec_from_to F1 M = vec_add (vec_add (vec_from_to F1 A) (vec_from_to F1 B)) (vec_from_to F1 O)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Theorem statement
theorem hyperbola_theorem (A B M : ℝ × ℝ) 
  (hA : hyperbola A.1 A.2) 
  (hB : hyperbola B.1 B.2) 
  (hM : M_condition M A B) :
  -- 1. The locus of M is (x-6)^2 - y^2 = 4
  ((M.1 - 6)^2 - M.2^2 = 4) ∧
  -- 2. There exists a fixed point C(1, 0) such that CA · CB is constant
  (∃ (C : ℝ × ℝ), C = (1, 0) ∧ 
    dot_product (vec_from_to C A) (vec_from_to C B) = -1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l3477_347704


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3477_347721

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3477_347721


namespace NUMINAMATH_CALUDE_last_number_is_one_l3477_347793

/-- A sequence of 1999 numbers with specific properties -/
def SpecialSequence : Type :=
  { a : Fin 1999 → ℤ // 
    a 0 = 1 ∧ 
    ∀ i : Fin 1997, a (i + 1) = a i + a (i + 2) }

/-- The last number in the SpecialSequence is 1 -/
theorem last_number_is_one (seq : SpecialSequence) : 
  seq.val (Fin.last 1998) = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_number_is_one_l3477_347793


namespace NUMINAMATH_CALUDE_video_game_enemies_l3477_347770

theorem video_game_enemies (points_per_enemy : ℕ) (undefeated_enemies : ℕ) (total_points : ℕ) : 
  points_per_enemy = 9 →
  undefeated_enemies = 3 →
  total_points = 72 →
  (total_points / points_per_enemy + undefeated_enemies : ℕ) = 11 := by
sorry

end NUMINAMATH_CALUDE_video_game_enemies_l3477_347770


namespace NUMINAMATH_CALUDE_equation_equivalence_l3477_347719

theorem equation_equivalence (x z : ℝ) 
  (h1 : 3 * x^2 + 4 * x + 6 * z + 2 = 0)
  (h2 : x - 2 * z + 1 = 0) :
  12 * z^2 + 2 * z + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3477_347719


namespace NUMINAMATH_CALUDE_polynomial_roots_l3477_347767

theorem polynomial_roots : 
  let p : ℝ → ℝ := fun x ↦ x^4 - 3*x^3 + 3*x^2 - x - 6
  ∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3477_347767


namespace NUMINAMATH_CALUDE_lucas_sequence_property_l3477_347714

def L : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => L (n + 1) + L n

theorem lucas_sequence_property (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) :
  p ∣ (L (2 * k) - 2) → p ∣ (L (2 * k + 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_lucas_sequence_property_l3477_347714


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3477_347713

theorem right_triangle_hypotenuse (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a = 1994 →         -- One cathetus is 1994
  c = 994010         -- Hypotenuse is 994010
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3477_347713


namespace NUMINAMATH_CALUDE_negative_reals_sup_and_max_l3477_347730

-- Define the set of negative real numbers
def NegativeReals : Set ℝ := {x | x < 0}

-- Theorem statement
theorem negative_reals_sup_and_max :
  (∃ s : ℝ, IsLUB NegativeReals s) ∧
  (¬∃ m : ℝ, m ∈ NegativeReals ∧ ∀ x ∈ NegativeReals, x ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_negative_reals_sup_and_max_l3477_347730


namespace NUMINAMATH_CALUDE_process_result_l3477_347794

def process (x : ℕ) : ℕ := 3 * (2 * x + 9)

theorem process_result : process 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_process_result_l3477_347794


namespace NUMINAMATH_CALUDE_food_budget_fraction_l3477_347747

theorem food_budget_fraction (grocery_fraction eating_out_fraction : ℚ) 
  (h1 : grocery_fraction = 0.6)
  (h2 : eating_out_fraction = 0.2) : 
  grocery_fraction + eating_out_fraction = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_food_budget_fraction_l3477_347747


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_1033_l3477_347779

theorem modular_inverse_11_mod_1033 : ∃ x : ℕ, x < 1033 ∧ (11 * x) % 1033 = 1 :=
by
  use 94
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_1033_l3477_347779


namespace NUMINAMATH_CALUDE_integral_cos_plus_exp_l3477_347710

theorem integral_cos_plus_exp (π : Real) : ∫ x in -π..0, (Real.cos x + Real.exp x) = 1 - 1 / Real.exp π := by
  sorry

end NUMINAMATH_CALUDE_integral_cos_plus_exp_l3477_347710


namespace NUMINAMATH_CALUDE_order_of_roots_l3477_347777

theorem order_of_roots : 5^(2/3) > 16^(1/3) ∧ 16^(1/3) > 2^(4/5) := by
  sorry

end NUMINAMATH_CALUDE_order_of_roots_l3477_347777


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3477_347717

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The theorem states that if the point P(x-1, x+1) is in the second quadrant and x is an integer, then x must be 0 -/
theorem point_in_second_quadrant (x : ℤ) : in_second_quadrant (x - 1 : ℝ) (x + 1 : ℝ) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3477_347717


namespace NUMINAMATH_CALUDE_sum_after_transformation_l3477_347751

theorem sum_after_transformation (S a b : ℝ) (h : a + b = S) :
  3 * (a + 4) + 3 * (b + 4) = 3 * S + 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_transformation_l3477_347751


namespace NUMINAMATH_CALUDE_tangent_slope_at_two_l3477_347756

/-- The function representing the curve y = x^2 + 3x -/
def f (x : ℝ) : ℝ := x^2 + 3*x

/-- The derivative of the function f -/
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_two :
  f' 2 = 7 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_two_l3477_347756


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3477_347753

theorem geometric_arithmetic_sequence (a b c : ℝ) (q : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Ensure positive terms
  a > b → b > c →  -- Decreasing sequence
  b = a * q →  -- Geometric progression
  c = b * q →  -- Geometric progression
  2 * (2020 * b / 7) = 577 * a + c / 7 →  -- Arithmetic progression condition
  q = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3477_347753


namespace NUMINAMATH_CALUDE_cone_base_radius_l3477_347755

/-- Given a sector paper with radius 24 cm and area 120π cm², 
    prove that the radius of the circular base of the cone formed by this sector is 5 cm -/
theorem cone_base_radius (sector_radius : ℝ) (sector_area : ℝ) (base_radius : ℝ) : 
  sector_radius = 24 →
  sector_area = 120 * Real.pi →
  sector_area = Real.pi * base_radius * sector_radius →
  base_radius = 5 := by
sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3477_347755


namespace NUMINAMATH_CALUDE_smallest_q_property_l3477_347775

theorem smallest_q_property : ∃ (q : ℕ), q > 0 ∧ q = 2015 ∧
  (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 →
    ∃ (n : ℤ), (m : ℚ) / 1007 * q < n ∧ n < (m + 1 : ℚ) / 1008 * q) ∧
  (∀ (q' : ℕ), 0 < q' ∧ q' < q →
    ¬(∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 →
      ∃ (n : ℤ), (m : ℚ) / 1007 * q' < n ∧ n < (m + 1 : ℚ) / 1008 * q')) :=
by sorry

end NUMINAMATH_CALUDE_smallest_q_property_l3477_347775


namespace NUMINAMATH_CALUDE_exists_spanish_couple_l3477_347798

-- Define the set S
def S : Set ℝ := {x | ∃ (a b : ℕ), x = (a - 1) / b}

-- Define the property of being strictly increasing
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the Spanish Couple property
def SpanishCouple (f g : ℝ → ℝ) : Prop :=
  (∀ x ∈ S, f x ∈ S) ∧
  (∀ x ∈ S, g x ∈ S) ∧
  StrictlyIncreasing f ∧
  StrictlyIncreasing g ∧
  ∀ x ∈ S, f (g (g x)) < g (f x)

-- Theorem statement
theorem exists_spanish_couple : ∃ f g, SpanishCouple f g := by
  sorry

end NUMINAMATH_CALUDE_exists_spanish_couple_l3477_347798


namespace NUMINAMATH_CALUDE_zach_needs_six_dollars_l3477_347749

/-- The amount of money Zach needs to earn to buy the bike -/
def money_needed (bike_cost allowance lawn_pay babysit_rate current_savings babysit_hours : ℕ) : ℕ :=
  let total_earnings := allowance + lawn_pay + babysit_rate * babysit_hours
  let total_savings := current_savings + total_earnings
  if total_savings ≥ bike_cost then 0
  else bike_cost - total_savings

theorem zach_needs_six_dollars :
  money_needed 100 5 10 7 65 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_zach_needs_six_dollars_l3477_347749


namespace NUMINAMATH_CALUDE_triangular_array_digit_sum_l3477_347738

/-- The number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem triangular_array_digit_sum :
  ∃ (n : ℕ), triangular_sum n = 2145 ∧ sum_of_digits n = 11 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_digit_sum_l3477_347738


namespace NUMINAMATH_CALUDE_circle_radius_proof_l3477_347735

theorem circle_radius_proof (A₁ A₂ : ℝ) (h1 : A₁ > 0) (h2 : A₂ > 0) : 
  (A₁ + A₂ = 16 * Real.pi) →
  (2 * A₁ = 16 * Real.pi - A₁) →
  (∃ (r : ℝ), r > 0 ∧ A₁ = Real.pi * r^2 ∧ r = 4 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l3477_347735


namespace NUMINAMATH_CALUDE_fraction_comparison_l3477_347726

theorem fraction_comparison (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m < n) :
  (m + 3 : ℚ) / (n + 3) > (m : ℚ) / n := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3477_347726


namespace NUMINAMATH_CALUDE_reinforcement_size_l3477_347787

/-- Calculates the size of a reinforcement given initial garrison size, initial provision duration,
    time passed before reinforcement, and remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                             (time_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions := initial_garrison * initial_duration
  let provisions_left := initial_garrison * (initial_duration - time_passed)
  (provisions_left / remaining_duration) - initial_garrison

theorem reinforcement_size :
  let initial_garrison := 2000
  let initial_duration := 62
  let time_passed := 15
  let remaining_duration := 20
  calculate_reinforcement initial_garrison initial_duration time_passed remaining_duration = 2700 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l3477_347787


namespace NUMINAMATH_CALUDE_trigonometric_ratio_equality_l3477_347784

theorem trigonometric_ratio_equality 
  (a b c α β : ℝ) 
  (eq1 : a * Real.cos α + b * Real.sin α = c)
  (eq2 : a * Real.cos β + b * Real.sin β = c) :
  ∃ (k : ℝ), k ≠ 0 ∧ 
    a = k * Real.cos ((α + β) / 2) ∧
    b = k * Real.sin ((α + β) / 2) ∧
    c = k * Real.cos ((α - β) / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_equality_l3477_347784


namespace NUMINAMATH_CALUDE_physical_examination_count_l3477_347764

theorem physical_examination_count (boys girls examined : ℕ) 
  (h1 : boys = 121)
  (h2 : girls = 83)
  (h3 : examined = 150) :
  boys + girls - examined = 54 := by
  sorry

end NUMINAMATH_CALUDE_physical_examination_count_l3477_347764


namespace NUMINAMATH_CALUDE_product_inequality_l3477_347715

theorem product_inequality (a b c d : ℝ) : a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a * c > b * d := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3477_347715


namespace NUMINAMATH_CALUDE_five_sundays_in_july_l3477_347765

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month -/
structure Month where
  days : ℕ
  first_day : DayOfWeek

/-- Given a day of the week, returns the next day -/
def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : ℕ :=
  sorry

/-- Theorem: If June has five Fridays and 30 days, July (with 31 days) must have five Sundays -/
theorem five_sundays_in_july 
  (june : Month) 
  (july : Month) 
  (h1 : june.days = 30)
  (h2 : july.days = 31)
  (h3 : count_day_occurrences june DayOfWeek.Friday = 5)
  (h4 : july.first_day = next_day june.first_day) :
  count_day_occurrences july DayOfWeek.Sunday = 5 :=
sorry

end NUMINAMATH_CALUDE_five_sundays_in_july_l3477_347765


namespace NUMINAMATH_CALUDE_fraction_conversions_l3477_347736

theorem fraction_conversions :
  (7 / 9 : ℚ) = 7 / 9 ∧
  (12 / 7 : ℚ) = 12 / 7 ∧
  (3 + 5 / 8 : ℚ) = 29 / 8 ∧
  (6 : ℚ) = 66 / 11 := by
sorry

end NUMINAMATH_CALUDE_fraction_conversions_l3477_347736


namespace NUMINAMATH_CALUDE_unique_k_solution_l3477_347752

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n + 5 else (n + 1) / 2

theorem unique_k_solution (k : ℤ) (h1 : k % 2 = 0) (h2 : g (g (g k)) = 61) : k = 236 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_solution_l3477_347752


namespace NUMINAMATH_CALUDE_milk_cartons_accepted_l3477_347780

/-- Proves that given 400 total cartons equally distributed among 4 customers,
    with each customer returning 60 damaged cartons, the total number of
    cartons accepted by all customers is 160. -/
theorem milk_cartons_accepted (total_cartons : ℕ) (num_customers : ℕ) (damaged_per_customer : ℕ)
    (h1 : total_cartons = 400)
    (h2 : num_customers = 4)
    (h3 : damaged_per_customer = 60) :
    (total_cartons / num_customers - damaged_per_customer) * num_customers = 160 :=
  by sorry

end NUMINAMATH_CALUDE_milk_cartons_accepted_l3477_347780


namespace NUMINAMATH_CALUDE_even_monotone_inequality_l3477_347724

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_monotone_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  f (-2) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_even_monotone_inequality_l3477_347724


namespace NUMINAMATH_CALUDE_solution_set_for_t_equals_one_range_of_t_l3477_347772

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - t| + |x + t|

-- Part 1
theorem solution_set_for_t_equals_one :
  {x : ℝ | f 1 x ≤ 8 - x^2} = Set.Icc (-2) 2 := by sorry

-- Part 2
theorem range_of_t (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 4) :
  (∀ x : ℝ, f t x = (4 * m^2 + n) / (m * n)) →
  t ∈ Set.Iic (-9/8) ∪ Set.Ici (9/8) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_t_equals_one_range_of_t_l3477_347772


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3477_347740

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∃ (x y : ℝ), ax - y + 2*a = 0 ∧ (2*a - 1)*x + a*y + a = 0) →
  (a*(2*a - 1) + (-1)*a = 0) →
  (a = 0 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3477_347740


namespace NUMINAMATH_CALUDE_abc_sum_in_base_7_l3477_347722

theorem abc_sum_in_base_7 : ∃ (A B C : Nat), 
  A < 7 ∧ B < 7 ∧ C < 7 ∧  -- digits less than 7
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧  -- non-zero digits
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧  -- distinct digits
  Nat.Prime A ∧            -- A is prime
  (A * 49 + B * 7 + C) + (B * 7 + C) = A * 49 + C * 7 + A ∧  -- ABC₇ + BC₇ = ACA₇
  A + B + C = 13           -- sum is 16₇ in base 7, which is 13 in base 10
  := by sorry

end NUMINAMATH_CALUDE_abc_sum_in_base_7_l3477_347722


namespace NUMINAMATH_CALUDE_select_students_result_l3477_347762

/-- The number of ways to select 4 students from two classes, with 2 students from each class, 
    such that exactly 1 female student is among them. -/
def select_students (class_a_male class_a_female class_b_male class_b_female : ℕ) : ℕ :=
  Nat.choose class_a_male 1 * Nat.choose class_a_female 1 * Nat.choose class_b_male 2 +
  Nat.choose class_a_male 2 * Nat.choose class_b_male 1 * Nat.choose class_b_female 1

/-- Theorem stating that the number of ways to select 4 students from two classes, 
    with 2 students from each class, such that exactly 1 female student is among them, 
    is equal to 345, given the specific class compositions. -/
theorem select_students_result : select_students 5 3 6 2 = 345 := by
  sorry

end NUMINAMATH_CALUDE_select_students_result_l3477_347762


namespace NUMINAMATH_CALUDE_dime_difference_is_90_l3477_347795

/-- Represents the number of coins of each type in the piggy bank -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  halfDollars : ℕ

/-- Checks if the given coin count satisfies the problem conditions -/
def isValidCoinCount (c : CoinCount) : Prop :=
  c.nickels + c.dimes + c.halfDollars = 120 ∧
  5 * c.nickels + 10 * c.dimes + 50 * c.halfDollars = 1050

/-- Theorem stating the difference between max and min number of dimes -/
theorem dime_difference_is_90 :
  ∃ (min_dimes max_dimes : ℕ),
    (∃ c : CoinCount, isValidCoinCount c ∧ c.dimes = min_dimes) ∧
    (∃ c : CoinCount, isValidCoinCount c ∧ c.dimes = max_dimes) ∧
    (∀ c : CoinCount, isValidCoinCount c → c.dimes ≥ min_dimes ∧ c.dimes ≤ max_dimes) ∧
    max_dimes - min_dimes = 90 :=
by sorry

end NUMINAMATH_CALUDE_dime_difference_is_90_l3477_347795


namespace NUMINAMATH_CALUDE_total_boxes_theorem_l3477_347786

/-- Calculates the total number of boxes sold over four days given specific sales conditions --/
def total_boxes_sold (thursday_boxes : ℕ) : ℕ :=
  let friday_boxes : ℕ := thursday_boxes + (thursday_boxes * 50) / 100
  let saturday_boxes : ℕ := friday_boxes + (friday_boxes * 80) / 100
  let sunday_boxes : ℕ := saturday_boxes - (saturday_boxes * 30) / 100
  thursday_boxes + friday_boxes + saturday_boxes + sunday_boxes

/-- Theorem stating that given the specific sales conditions, the total number of boxes sold is 425 --/
theorem total_boxes_theorem : total_boxes_sold 60 = 425 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_theorem_l3477_347786


namespace NUMINAMATH_CALUDE_one_is_last_digit_to_appear_l3477_347796

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 2
  | n + 2 => (modifiedFibonacci n + modifiedFibonacci (n + 1)) % 10

def digitAppearsInSequence (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ modifiedFibonacci k % 10 = d

def allDigitsAppear (n : ℕ) : Prop :=
  ∀ d, d < 10 → digitAppearsInSequence d n

def isLastDigitToAppear (d : ℕ) : Prop :=
  ∃ n, allDigitsAppear n ∧
    ¬(allDigitsAppear (n - 1)) ∧
    ¬(digitAppearsInSequence d (n - 1))

theorem one_is_last_digit_to_appear :
  isLastDigitToAppear 1 := by sorry

end NUMINAMATH_CALUDE_one_is_last_digit_to_appear_l3477_347796


namespace NUMINAMATH_CALUDE_power_of_two_representation_l3477_347743

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℕ), 2^n = 7*x^2 + y^2 ∧ Odd x ∧ Odd y :=
sorry

end NUMINAMATH_CALUDE_power_of_two_representation_l3477_347743


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3477_347728

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3477_347728


namespace NUMINAMATH_CALUDE_zeros_inequality_l3477_347718

open Real

theorem zeros_inequality (x₁ x₂ m : ℝ) (h₁ : x₁ < x₂) 
  (h₂ : exp (m * x₁) - log x₁ + (m - 1) * x₁ = 0) 
  (h₃ : exp (m * x₂) - log x₂ + (m - 1) * x₂ = 0) : 
  2 * log x₁ + log x₂ > exp 1 := by
  sorry

end NUMINAMATH_CALUDE_zeros_inequality_l3477_347718


namespace NUMINAMATH_CALUDE_M_equals_N_l3477_347748

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l3477_347748


namespace NUMINAMATH_CALUDE_sand_in_partial_bag_l3477_347768

theorem sand_in_partial_bag (total_sand : ℝ) (bag_capacity : ℝ) (h1 : total_sand = 1254.75) (h2 : bag_capacity = 73.5) :
  total_sand - (bag_capacity * ⌊total_sand / bag_capacity⌋) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_sand_in_partial_bag_l3477_347768


namespace NUMINAMATH_CALUDE_smallest_perimeter_l3477_347783

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  h1 : side1 = 7
  h2 : side2 = 10
  h3 : Even side3
  h4 : side1 + side2 > side3
  h5 : side1 + side3 > side2
  h6 : side2 + side3 > side1

/-- The perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ := t.side1 + t.side2 + t.side3

/-- Theorem: The smallest possible perimeter of the given triangle is 21 --/
theorem smallest_perimeter :
  ∃ (t : Triangle), ∀ (t' : Triangle), perimeter t ≤ perimeter t' ∧ perimeter t = 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l3477_347783


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l3477_347705

theorem tip_percentage_calculation (total_bill : ℝ) (sales_tax_rate : ℝ) (food_price : ℝ) : 
  total_bill = 211.20 ∧ 
  sales_tax_rate = 0.10 ∧ 
  food_price = 160 → 
  (total_bill - food_price * (1 + sales_tax_rate)) / (food_price * (1 + sales_tax_rate)) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l3477_347705


namespace NUMINAMATH_CALUDE_quadratic_touch_existence_l3477_347720

theorem quadratic_touch_existence (p q r : ℤ) : 
  (∃ x : ℝ, p * x^2 + q * x + r = 0 ∧ ∀ y : ℝ, p * y^2 + q * y + r ≥ 0) →
  ∃ a b : ℤ, 
    (∃ x : ℝ, p * x^2 + q * x + r = (b : ℝ)) ∧
    (∃ x : ℝ, x^2 + (a : ℝ) * x + (b : ℝ) = 0 ∧ ∀ y : ℝ, y^2 + (a : ℝ) * y + (b : ℝ) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_touch_existence_l3477_347720
