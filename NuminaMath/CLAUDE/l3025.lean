import Mathlib

namespace NUMINAMATH_CALUDE_nancy_seeds_l3025_302518

/-- Calculates the total number of seeds Nancy started with. -/
def total_seeds (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  big_garden_seeds + small_gardens * seeds_per_small_garden

/-- Proves that Nancy started with 52 seeds given the problem conditions. -/
theorem nancy_seeds :
  let big_garden_seeds : ℕ := 28
  let small_gardens : ℕ := 6
  let seeds_per_small_garden : ℕ := 4
  total_seeds big_garden_seeds small_gardens seeds_per_small_garden = 52 := by
  sorry

#eval total_seeds 28 6 4

end NUMINAMATH_CALUDE_nancy_seeds_l3025_302518


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3025_302513

theorem cube_volume_problem (a : ℝ) : 
  (a + 2) * (a + 2) * (a - 2) = a^3 - 16 → a^3 = 9 + 12 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3025_302513


namespace NUMINAMATH_CALUDE_cosine_largest_angle_triangle_l3025_302549

theorem cosine_largest_angle_triangle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  cos_C = -(1/4) := by
  sorry

end NUMINAMATH_CALUDE_cosine_largest_angle_triangle_l3025_302549


namespace NUMINAMATH_CALUDE_total_area_is_200_l3025_302599

/-- The total area of Yanni's paintings -/
def total_area : ℕ :=
  let painting1_count : ℕ := 3
  let painting1_width : ℕ := 5
  let painting1_height : ℕ := 5
  let painting2_width : ℕ := 10
  let painting2_height : ℕ := 8
  let painting3_width : ℕ := 9
  let painting3_height : ℕ := 5
  (painting1_count * painting1_width * painting1_height) +
  (painting2_width * painting2_height) +
  (painting3_width * painting3_height)

/-- Theorem stating that the total area of Yanni's paintings is 200 square feet -/
theorem total_area_is_200 : total_area = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_area_is_200_l3025_302599


namespace NUMINAMATH_CALUDE_max_value_of_f_l3025_302598

def f (x : ℝ) : ℝ := -x^2 + 6*x - 10

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3025_302598


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l3025_302555

/-- Given a parabola with equation y^2 = -4x, its focus has coordinates (-1, 0) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = -4*x}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (-1, 0) ∧ ∀ (p : ℝ × ℝ), p ∈ parabola → ‖p - f‖ = ‖p - (p.1, 0)‖ := by
  sorry


end NUMINAMATH_CALUDE_parabola_focus_coordinates_l3025_302555


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3025_302553

/-- Calculates the profit percentage for a retailer given wholesale price, retail price, and discount percentage. -/
def profit_percentage (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price / 100
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that given the specific conditions, the profit percentage is 20%. -/
theorem retailer_profit_percentage :
  let wholesale_price : ℚ := 81
  let retail_price : ℚ := 108
  let discount_percent : ℚ := 10
  profit_percentage wholesale_price retail_price discount_percent = 20 := by
sorry

#eval profit_percentage 81 108 10

end NUMINAMATH_CALUDE_retailer_profit_percentage_l3025_302553


namespace NUMINAMATH_CALUDE_pizza_eaters_fraction_l3025_302560

theorem pizza_eaters_fraction (total_people : ℕ) (total_pizza : ℕ) (pieces_per_person : ℕ) (remaining_pizza : ℕ)
  (h1 : total_people = 15)
  (h2 : total_pizza = 50)
  (h3 : pieces_per_person = 4)
  (h4 : remaining_pizza = 14) :
  (total_pizza - remaining_pizza) / (pieces_per_person * total_people) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_pizza_eaters_fraction_l3025_302560


namespace NUMINAMATH_CALUDE_sum_inequality_l3025_302574

theorem sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 3) : 
  1 / (x^5 - x^2 + 3) + 1 / (y^5 - y^2 + 3) + 1 / (z^5 - z^2 + 3) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3025_302574


namespace NUMINAMATH_CALUDE_janet_action_figures_l3025_302569

/-- The number of action figures Janet initially owns -/
def initial_figures : ℕ := 10

/-- The number of new action figures Janet buys -/
def new_figures : ℕ := 4

/-- The total number of action figures Janet has at the end -/
def total_figures : ℕ := 24

/-- The number of action figures Janet sold -/
def sold_figures : ℕ := 6

theorem janet_action_figures :
  ∃ (x : ℕ),
    x = sold_figures ∧
    initial_figures - x + new_figures +
    2 * (initial_figures - x + new_figures) = total_figures :=
by sorry

end NUMINAMATH_CALUDE_janet_action_figures_l3025_302569


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_l3025_302532

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    prove that the area of the sector formed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (r : ℝ) : 
  r * 2 = 4 → (1 / 2) * r^2 * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_length_l3025_302532


namespace NUMINAMATH_CALUDE_selection_theorem_l3025_302566

def num_boys : ℕ := 5
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls
def num_selected : ℕ := 4

/-- The number of ways to select 4 people from 5 boys and 4 girls, 
    ensuring at least one of boy A and girl B participates, 
    and both boys and girls are present -/
def selection_ways : ℕ := sorry

theorem selection_theorem : 
  selection_ways = (total_people.choose num_selected) - 
                   (num_boys.choose num_selected) - 
                   (num_girls.choose num_selected) := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l3025_302566


namespace NUMINAMATH_CALUDE_min_rectangles_cover_square_l3025_302587

/-- The smallest number of 3-by-4 non-overlapping rectangles needed to cover a square region -/
def min_rectangles : ℕ := 16

/-- The width of each rectangle -/
def rectangle_width : ℕ := 4

/-- The height of each rectangle -/
def rectangle_height : ℕ := 3

/-- The side length of the square region -/
def square_side : ℕ := 12

theorem min_rectangles_cover_square :
  (min_rectangles * rectangle_width * rectangle_height = square_side * square_side) ∧
  (square_side % rectangle_height = 0) ∧
  (∀ n : ℕ, n < min_rectangles →
    n * rectangle_width * rectangle_height < square_side * square_side) := by
  sorry

#check min_rectangles_cover_square

end NUMINAMATH_CALUDE_min_rectangles_cover_square_l3025_302587


namespace NUMINAMATH_CALUDE_proposition_relationship_l3025_302531

theorem proposition_relationship (a b : ℝ) : 
  ¬(((a + b ≠ 4) → (a ≠ 1 ∧ b ≠ 3)) ∧ ((a ≠ 1 ∧ b ≠ 3) → (a + b ≠ 4))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l3025_302531


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l3025_302579

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a6 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_prod : a 4 * a 10 = 16) : 
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l3025_302579


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3025_302544

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 + a 2 = 30) →
  (a 3 + a 4 = 120) →
  (a 5 + a 6 = 480) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3025_302544


namespace NUMINAMATH_CALUDE_sharon_supplies_theorem_l3025_302571

def angela_pots : ℕ → ℕ := λ p => p

def angela_plates : ℕ → ℕ := λ p => 3 * p + 6

def angela_cutlery : ℕ → ℕ := λ p => (3 * p + 6) / 2

def sharon_pots : ℕ → ℕ := λ p => p / 2

def sharon_plates : ℕ → ℕ := λ p => 3 * (3 * p + 6) - 20

def sharon_cutlery : ℕ → ℕ := λ p => 3 * p + 6

def sharon_total_supplies : ℕ → ℕ := λ p => 
  sharon_pots p + sharon_plates p + sharon_cutlery p

theorem sharon_supplies_theorem (p : ℕ) : 
  p = 20 → sharon_total_supplies p = 254 := by
  sorry

end NUMINAMATH_CALUDE_sharon_supplies_theorem_l3025_302571


namespace NUMINAMATH_CALUDE_plant_supplier_pots_cost_l3025_302508

/-- The cost of new pots for a plant supplier --/
theorem plant_supplier_pots_cost :
  let orchid_count : ℕ := 20
  let orchid_price : ℕ := 50
  let money_plant_count : ℕ := 15
  let money_plant_price : ℕ := 25
  let worker_count : ℕ := 2
  let worker_pay : ℕ := 40
  let remaining_money : ℕ := 1145
  let total_earnings := orchid_count * orchid_price + money_plant_count * money_plant_price
  let total_expenses := worker_count * worker_pay + remaining_money
  total_earnings - total_expenses = 150 :=
by sorry

end NUMINAMATH_CALUDE_plant_supplier_pots_cost_l3025_302508


namespace NUMINAMATH_CALUDE_root_sum_sixth_power_l3025_302541

theorem root_sum_sixth_power (r s : ℝ) : 
  r^2 - 2*r + Real.sqrt 2 = 0 → 
  s^2 - 2*s + Real.sqrt 2 = 0 → 
  r^6 + s^6 = 904 - 640 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_sixth_power_l3025_302541


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l3025_302528

theorem complex_modulus_equality (x : ℝ) :
  x > 0 → (Complex.abs (5 + x * Complex.I) = 13 ↔ x = 12) := by sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l3025_302528


namespace NUMINAMATH_CALUDE_last_digit_379_base_4_l3025_302551

def last_digit_base_4 (n : ℕ) : ℕ := n % 4

theorem last_digit_379_base_4 :
  last_digit_base_4 379 = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_379_base_4_l3025_302551


namespace NUMINAMATH_CALUDE_triangle_existence_implies_m_greater_than_six_l3025_302507

/-- The function f(x) = x^3 - 3x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- Theorem: If there exists a triangle with side lengths f(a), f(b), f(c) for a, b, c in [0, 2], then m > 6 -/
theorem triangle_existence_implies_m_greater_than_six (m : ℝ) : 
  (∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 
    f m a + f m b > f m c ∧ 
    f m b + f m c > f m a ∧ 
    f m c + f m a > f m b) → 
  m > 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_existence_implies_m_greater_than_six_l3025_302507


namespace NUMINAMATH_CALUDE_diagonals_in_polygon_l3025_302542

/-- The number of diagonals in a convex k-sided polygon. -/
def num_diagonals (k : ℕ) : ℕ := k * (k - 3) / 2

/-- Theorem stating that the number of diagonals in a convex k-sided polygon
    (where k > 3) is equal to k(k-3)/2. -/
theorem diagonals_in_polygon (k : ℕ) (h : k > 3) :
  num_diagonals k = k * (k - 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_diagonals_in_polygon_l3025_302542


namespace NUMINAMATH_CALUDE_divisibility_property_l3025_302556

theorem divisibility_property (y : ℕ) (h : y ≠ 0) :
  (y - 1) ∣ (y^(y^2) - 2*y^(y+1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l3025_302556


namespace NUMINAMATH_CALUDE_amy_soup_count_l3025_302522

/-- The number of cans of chicken soup Amy bought -/
def chicken_soup : ℕ := 6

/-- The number of cans of tomato soup Amy bought -/
def tomato_soup : ℕ := 3

/-- The number of cans of vegetable soup Amy bought -/
def vegetable_soup : ℕ := 4

/-- The number of cans of clam chowder Amy bought -/
def clam_chowder : ℕ := 2

/-- The number of cans of French onion soup Amy bought -/
def french_onion_soup : ℕ := 1

/-- The number of cans of minestrone soup Amy bought -/
def minestrone_soup : ℕ := 5

/-- The total number of cans of soup Amy bought -/
def total_soups : ℕ := chicken_soup + tomato_soup + vegetable_soup + clam_chowder + french_onion_soup + minestrone_soup

theorem amy_soup_count : total_soups = 21 := by
  sorry

end NUMINAMATH_CALUDE_amy_soup_count_l3025_302522


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_wire_isosceles_triangle_with_side_6_l3025_302547

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Represents an isosceles triangle -/
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

theorem isosceles_triangle_from_wire (wire_length : ℝ) 
  (h_wire : wire_length = 24) :
  ∃ (t : Triangle), IsoscelesTriangle t ∧ 
    t.a + t.b + t.c = wire_length ∧
    t.a = t.b ∧ t.a = 2 * t.c ∧
    t.a = 48 / 5 := by
  sorry

theorem isosceles_triangle_with_side_6 (wire_length : ℝ) 
  (h_wire : wire_length = 24) :
  ∃ (t : Triangle), IsoscelesTriangle t ∧ 
    t.a + t.b + t.c = wire_length ∧
    (t.a = 6 ∨ t.b = 6 ∨ t.c = 6) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_wire_isosceles_triangle_with_side_6_l3025_302547


namespace NUMINAMATH_CALUDE_max_value_xyz_l3025_302581

theorem max_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + 2*y + 3*z = 1) :
  x^3 * y^2 * z ≤ 2048 / 11^6 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 1 ∧ x₀^3 * y₀^2 * z₀ = 2048 / 11^6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_xyz_l3025_302581


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l3025_302554

/-- Triangle DEF with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Tetrahedron ODEF -/
structure Tetrahedron where
  O : Point3D
  D : Point3D
  E : Point3D
  F : Point3D

def origin : Point3D := ⟨0, 0, 0⟩

/-- Volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: Volume of tetrahedron ODEF is 110/3 -/
theorem volume_of_specific_tetrahedron (tri : Triangle) (t : Tetrahedron) :
  tri.a = 8 ∧ tri.b = 10 ∧ tri.c = 12 ∧
  t.O = origin ∧
  t.D.y = 0 ∧ t.D.z = 0 ∧
  t.E.x = 0 ∧ t.E.z = 0 ∧
  t.F.x = 0 ∧ t.F.y = 0 →
  tetrahedronVolume t = 110 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l3025_302554


namespace NUMINAMATH_CALUDE_bus_ride_net_change_l3025_302559

/-- Represents the number of children on a bus and the changes at each stop -/
structure BusRide where
  initial : Int
  first_stop_off : Int
  first_stop_on : Int
  second_stop_off : Int
  final : Int

/-- Calculates the difference between total children who got off and got on -/
def net_change (ride : BusRide) : Int :=
  ride.first_stop_off + ride.second_stop_off - 
  (ride.first_stop_on + (ride.final - (ride.initial - ride.first_stop_off + ride.first_stop_on - ride.second_stop_off)))

/-- Theorem stating the net change in children for the given bus ride -/
theorem bus_ride_net_change :
  let ride : BusRide := {
    initial := 36,
    first_stop_off := 45,
    first_stop_on := 25,
    second_stop_off := 68,
    final := 12
  }
  net_change ride = 24 := by sorry

end NUMINAMATH_CALUDE_bus_ride_net_change_l3025_302559


namespace NUMINAMATH_CALUDE_integral_cos_plus_exp_l3025_302521

theorem integral_cos_plus_exp : 
  ∫ x in -Real.pi..0, (Real.cos x + Real.exp x) = 1 - 1 / Real.exp Real.pi := by
  sorry

end NUMINAMATH_CALUDE_integral_cos_plus_exp_l3025_302521


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3025_302535

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 6*x + k = (x + a)^2) → k = 9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3025_302535


namespace NUMINAMATH_CALUDE_sequence_comparison_theorem_l3025_302512

theorem sequence_comparison_theorem (a b : ℕ → ℕ) :
  ∃ r s : ℕ, r ≠ s ∧ a r ≥ a s ∧ b r ≥ b s := by
  sorry

end NUMINAMATH_CALUDE_sequence_comparison_theorem_l3025_302512


namespace NUMINAMATH_CALUDE_drums_filled_per_day_l3025_302593

/-- Given the total number of drums filled and the number of days, 
    calculate the number of drums filled per day -/
def drums_per_day (total_drums : ℕ) (num_days : ℕ) : ℕ :=
  total_drums / num_days

/-- Theorem stating that given 6264 drums filled in 58 days, 
    the number of drums filled per day is 108 -/
theorem drums_filled_per_day : 
  drums_per_day 6264 58 = 108 := by
  sorry

#eval drums_per_day 6264 58

end NUMINAMATH_CALUDE_drums_filled_per_day_l3025_302593


namespace NUMINAMATH_CALUDE_gcf_72_108_l3025_302597

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_108_l3025_302597


namespace NUMINAMATH_CALUDE_triangle_properties_l3025_302573

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  c * Real.cos B = (2 * a - b) * Real.cos C →
  c = 2 →
  a + b + c = 2 * Real.sqrt 3 + 2 →
  -- Triangle validity conditions
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  -- Theorem statements
  C = Real.pi / 3 ∧
  (1/2) * a * b * Real.sin C = (2 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3025_302573


namespace NUMINAMATH_CALUDE_paul_remaining_money_l3025_302590

/-- The amount of money Paul had for shopping -/
def initial_money : ℕ := 15

/-- The cost of bread -/
def bread_cost : ℕ := 2

/-- The cost of butter -/
def butter_cost : ℕ := 3

/-- The cost of juice (twice the price of bread) -/
def juice_cost : ℕ := 2 * bread_cost

/-- The total cost of groceries -/
def total_cost : ℕ := bread_cost + butter_cost + juice_cost

/-- The remaining money after shopping -/
def remaining_money : ℕ := initial_money - total_cost

theorem paul_remaining_money :
  remaining_money = 6 :=
sorry

end NUMINAMATH_CALUDE_paul_remaining_money_l3025_302590


namespace NUMINAMATH_CALUDE_ticket_price_reduction_l3025_302578

theorem ticket_price_reduction (x : ℝ) (y : ℝ) (h1 : x > 0) : 
  (4/3 * x * (50 - y) = 5/4 * x * 50) → y = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_reduction_l3025_302578


namespace NUMINAMATH_CALUDE_ruths_school_schedule_l3025_302540

/-- Ruth's school schedule problem -/
theorem ruths_school_schedule 
  (days_per_week : ℕ) 
  (math_class_percentage : ℚ) 
  (math_class_hours_per_week : ℕ) 
  (h1 : days_per_week = 5)
  (h2 : math_class_percentage = 1/4)
  (h3 : math_class_hours_per_week = 10) :
  let total_school_hours_per_week := math_class_hours_per_week / math_class_percentage
  let school_hours_per_day := total_school_hours_per_week / days_per_week
  school_hours_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_ruths_school_schedule_l3025_302540


namespace NUMINAMATH_CALUDE_inf_a_plus_2b_is_3_l3025_302527

open Real

/-- Given 0 < a < b and |log a| = |log b|, the infimum of a + 2b is 3 -/
theorem inf_a_plus_2b_is_3 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |log a| = |log b|) :
  ∃ (inf : ℝ), inf = 3 ∧ ∀ x, (∃ (a' b' : ℝ), 0 < a' ∧ a' < b' ∧ |log a'| = |log b'| ∧ x = a' + 2*b') → inf ≤ x :=
sorry

end NUMINAMATH_CALUDE_inf_a_plus_2b_is_3_l3025_302527


namespace NUMINAMATH_CALUDE_golden_silk_button_optimal_price_reduction_l3025_302594

/-- Represents the problem of finding the optimal price reduction for Golden Silk Button --/
theorem golden_silk_button_optimal_price_reduction 
  (initial_cost : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (sales_increase_rate : ℝ) 
  (target_profit : ℝ) 
  (price_reduction : ℝ) : 
  initial_cost = 24 → 
  initial_price = 40 → 
  initial_sales = 20 → 
  sales_increase_rate = 2 → 
  target_profit = 330 → 
  price_reduction = 5 → 
  (initial_price - price_reduction - initial_cost) * (initial_sales + sales_increase_rate * price_reduction) = target_profit :=
by sorry

end NUMINAMATH_CALUDE_golden_silk_button_optimal_price_reduction_l3025_302594


namespace NUMINAMATH_CALUDE_max_value_theorem_l3025_302509

theorem max_value_theorem (a b : ℝ) (h : a^2 - b^2 = -1) :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x y : ℝ), x^2 - y^2 = -1 → (|x| + 1) / y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3025_302509


namespace NUMINAMATH_CALUDE_find_A_l3025_302543

theorem find_A : ∃ A : ℕ, A = 23 ∧ A / 8 = 2 ∧ A % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l3025_302543


namespace NUMINAMATH_CALUDE_cube_volume_from_painting_cost_l3025_302545

/-- Given a cube where the cost of painting its entire surface area is Rs. 343.98
    at a rate of 13 paise per sq. cm, the volume of the cube is 9261 cubic cm. -/
theorem cube_volume_from_painting_cost (cost : ℚ) (rate : ℚ) (volume : ℚ) : 
  cost = 343.98 →
  rate = 13 / 100 →
  volume = (((cost * 100) / rate / 6).sqrt ^ 3) →
  volume = 9261 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_painting_cost_l3025_302545


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3025_302552

theorem inequality_solution_set : 
  ∀ x : ℝ, abs (x - 4) + abs (3 - x) < 2 ↔ 2.5 < x ∧ x < 4.5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3025_302552


namespace NUMINAMATH_CALUDE_sum_odd_sequence_to_99_l3025_302564

/-- Sum of arithmetic sequence -/
def sum_arithmetic_sequence (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: Sum of sequence 1+3+5+...+99 -/
theorem sum_odd_sequence_to_99 :
  sum_arithmetic_sequence 1 99 2 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_sequence_to_99_l3025_302564


namespace NUMINAMATH_CALUDE_acute_triangle_side_constraint_acute_triangle_side_constraint_converse_l3025_302514

/-- A triangle with side lengths a, b, and c is acute if and only if a² + b² > c², where c is the longest side. -/
def is_acute_triangle (a b c : ℝ) : Prop :=
  c ≥ a ∧ c ≥ b ∧ a^2 + b^2 > c^2

/-- The theorem states that for an acute triangle with side lengths x²+4, 4x, and x²+6,
    where x is a positive real number, x must be greater than √(15)/3. -/
theorem acute_triangle_side_constraint (x : ℝ) :
  x > 0 →
  is_acute_triangle (x^2 + 4) (4*x) (x^2 + 6) →
  x > Real.sqrt 15 / 3 :=
by sorry

/-- The converse of the theorem: if x > √(15)/3, then the triangle with side lengths
    x²+4, 4x, and x²+6 is acute. -/
theorem acute_triangle_side_constraint_converse (x : ℝ) :
  x > Real.sqrt 15 / 3 →
  is_acute_triangle (x^2 + 4) (4*x) (x^2 + 6) :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_side_constraint_acute_triangle_side_constraint_converse_l3025_302514


namespace NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l3025_302567

/-- S(n) is defined as n minus the largest perfect square not exceeding n -/
def S (n : ℕ) : ℕ :=
  n - (Nat.sqrt n) ^ 2

/-- The sequence a_k is defined recursively -/
def a (A : ℕ) : ℕ → ℕ
  | 0 => A
  | k + 1 => a A k + S (a A k)

/-- A positive integer A makes the sequence eventually constant
    if and only if A is a perfect square -/
theorem sequence_constant_iff_perfect_square (A : ℕ) (h : A > 0) :
  (∃ N : ℕ, ∀ k ≥ N, a A k = a A N) ↔ ∃ m : ℕ, A = m^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l3025_302567


namespace NUMINAMATH_CALUDE_simplify_polynomial_expression_l3025_302523

theorem simplify_polynomial_expression (x : ℝ) :
  (3 * x - 4) * (x + 9) + (x + 6) * (3 * x + 2) = 6 * x^2 + 43 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_expression_l3025_302523


namespace NUMINAMATH_CALUDE_ajay_dal_transaction_gain_l3025_302570

/-- Represents the transaction of buying and selling dal -/
structure DalTransaction where
  quantity1 : ℝ
  price1 : ℝ
  quantity2 : ℝ
  price2 : ℝ
  selling_price : ℝ

/-- Calculate the total gain from a dal transaction -/
def calculate_gain (t : DalTransaction) : ℝ :=
  let total_quantity := t.quantity1 + t.quantity2
  let total_cost := t.quantity1 * t.price1 + t.quantity2 * t.price2
  let total_revenue := total_quantity * t.selling_price
  total_revenue - total_cost

/-- Theorem stating that Ajay's total gain in the dal transaction is 27.50 rs -/
theorem ajay_dal_transaction_gain :
  let t : DalTransaction := {
    quantity1 := 15,
    price1 := 14.50,
    quantity2 := 10,
    price2 := 13,
    selling_price := 15
  }
  calculate_gain t = 27.50 := by
  sorry

end NUMINAMATH_CALUDE_ajay_dal_transaction_gain_l3025_302570


namespace NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_five_l3025_302534

theorem smallest_addition_for_multiple_of_five :
  ∀ n : ℕ, n > 0 ∧ (725 + n) % 5 = 0 → n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_five_l3025_302534


namespace NUMINAMATH_CALUDE_min_value_theorem_l3025_302517

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4*a + 3*b - 1 = 0) :
  ∃ (min : ℝ), min = 3 + 2*Real.sqrt 2 ∧ 
  ∀ (x : ℝ), x = 1/(2*a + b) + 1/(a + b) → x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3025_302517


namespace NUMINAMATH_CALUDE_largest_term_binomial_expansion_l3025_302572

theorem largest_term_binomial_expansion (k : ℕ) :
  k ≠ 64 →
  Nat.choose 100 64 * (Real.sqrt 3) ^ 64 > Nat.choose 100 k * (Real.sqrt 3) ^ k :=
sorry

end NUMINAMATH_CALUDE_largest_term_binomial_expansion_l3025_302572


namespace NUMINAMATH_CALUDE_directrix_of_parabola_l3025_302510

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- State the theorem
theorem directrix_of_parabola :
  ∀ x y : ℝ, parabola x y → (∃ p : ℝ, x = -3 ∧ p = y) :=
by sorry

end NUMINAMATH_CALUDE_directrix_of_parabola_l3025_302510


namespace NUMINAMATH_CALUDE_mms_given_to_sister_correct_l3025_302525

/-- The number of m&m's Cheryl gave to her sister -/
def mms_given_to_sister (initial : ℕ) (eaten_lunch : ℕ) (eaten_dinner : ℕ) : ℕ :=
  initial - (eaten_lunch + eaten_dinner)

/-- Theorem stating that the number of m&m's given to sister is correct -/
theorem mms_given_to_sister_correct (initial : ℕ) (eaten_lunch : ℕ) (eaten_dinner : ℕ) 
  (h1 : initial ≥ eaten_lunch + eaten_dinner) :
  mms_given_to_sister initial eaten_lunch eaten_dinner = initial - (initial - (eaten_lunch + eaten_dinner)) :=
by
  sorry

#eval mms_given_to_sister 25 7 5

end NUMINAMATH_CALUDE_mms_given_to_sister_correct_l3025_302525


namespace NUMINAMATH_CALUDE_hilton_marbles_l3025_302592

/-- Calculates the final number of marbles Hilton has -/
def final_marbles (initial : ℝ) (found : ℝ) (lost : ℝ) (compensation_rate : ℝ) : ℝ :=
  initial + found - lost + compensation_rate * lost

/-- Proves that Hilton ends up with 44.5 marbles given the initial conditions -/
theorem hilton_marbles :
  final_marbles 30 8.5 12 1.5 = 44.5 := by
  sorry

end NUMINAMATH_CALUDE_hilton_marbles_l3025_302592


namespace NUMINAMATH_CALUDE_no_prime_covering_triples_l3025_302591

/-- A polynomial is prime-covering if for every prime p, there exists an integer n for which p divides P(n) -/
def IsPrimeCovering (P : ℤ → ℤ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ∃ n : ℤ, (p : ℤ) ∣ P n

/-- The polynomial P(x) = (x^2 - a)(x^2 - b)(x^2 - c) -/
def P (a b c : ℤ) (x : ℤ) : ℤ :=
  (x^2 - a) * (x^2 - b) * (x^2 - c)

theorem no_prime_covering_triples :
  ¬ ∃ a b c : ℤ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 25 ∧ IsPrimeCovering (P a b c) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_covering_triples_l3025_302591


namespace NUMINAMATH_CALUDE_expression_evaluation_l3025_302580

theorem expression_evaluation : (100 - (1000 - 300)) - (1000 - (300 - 100)) = -1400 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3025_302580


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt6_l3025_302584

theorem sqrt_sum_equals_2sqrt6 : 
  Real.sqrt (9 - 6 * Real.sqrt 2) + Real.sqrt (9 + 6 * Real.sqrt 2) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt6_l3025_302584


namespace NUMINAMATH_CALUDE_minimum_other_sales_l3025_302565

/-- Represents the sales distribution of a stationery store -/
structure SalesDistribution where
  pens : ℝ
  pencils : ℝ
  other : ℝ

/-- The sales distribution meets the store's goals -/
def MeetsGoals (s : SalesDistribution) : Prop :=
  s.pens = 40 ∧
  s.pencils = 28 ∧
  s.other ≥ 20 ∧
  s.pens + s.pencils + s.other = 100

theorem minimum_other_sales (s : SalesDistribution) (h : MeetsGoals s) :
  s.other = 32 ∧ s.pens + s.pencils + s.other = 100 := by
  sorry

#check minimum_other_sales

end NUMINAMATH_CALUDE_minimum_other_sales_l3025_302565


namespace NUMINAMATH_CALUDE_no_solution_x6_2y2_plus_2_l3025_302500

theorem no_solution_x6_2y2_plus_2 : ∀ (x y : ℤ), x^6 ≠ 2*y^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_x6_2y2_plus_2_l3025_302500


namespace NUMINAMATH_CALUDE_truck_speed_l3025_302537

/-- Proves that a truck traveling 600 meters in 40 seconds has a speed of 54 kilometers per hour -/
theorem truck_speed : ∀ (distance : ℝ) (time : ℝ) (speed_ms : ℝ) (speed_kmh : ℝ),
  distance = 600 →
  time = 40 →
  speed_ms = distance / time →
  speed_kmh = speed_ms * 3.6 →
  speed_kmh = 54 := by
  sorry

#check truck_speed

end NUMINAMATH_CALUDE_truck_speed_l3025_302537


namespace NUMINAMATH_CALUDE_arithmetic_equations_l3025_302538

theorem arithmetic_equations : 
  (12 * 12 / (12 + 12) = 6) ∧ ((12 * 12 + 12) / 12 = 13) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equations_l3025_302538


namespace NUMINAMATH_CALUDE_discount_percentage_retailer_discount_approx_25_percent_l3025_302502

/-- Calculates the discount percentage given markup and profit percentages -/
theorem discount_percentage (markup : ℝ) (actual_profit : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let actual_selling_price := 1 + actual_profit
  let discount := marked_price - actual_selling_price
  (discount / marked_price) * 100

/-- Proves that the discount percentage is approximately 25% given the specified markup and profit -/
theorem retailer_discount_approx_25_percent :
  ∀ (ε : ℝ), ε > 0 →
  abs (discount_percentage 0.60 0.20000000000000018 - 25) < ε :=
sorry

end NUMINAMATH_CALUDE_discount_percentage_retailer_discount_approx_25_percent_l3025_302502


namespace NUMINAMATH_CALUDE_sequence_properties_l3025_302511

def sequence_a (n : ℕ) : ℝ := 1 - 2^n

def sum_S (n : ℕ) : ℝ := n + 2 - 2^(n+1)

theorem sequence_properties :
  ∀ (n : ℕ), n ≥ 1 → 
  (∃ (a : ℕ → ℝ) (S : ℕ → ℝ), 
    (∀ k, k ≥ 1 → S k = 2 * a k + k) ∧ 
    (∃ r : ℝ, ∀ k, k ≥ 1 → a (k+1) - 1 = r * (a k - 1)) ∧
    (∀ k, k ≥ 1 → a k = sequence_a k) ∧
    (∀ k, k ≥ 1 → S k = sum_S k)) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3025_302511


namespace NUMINAMATH_CALUDE_approx_small_number_to_large_place_l3025_302520

/-- Given a real number less than 10000, the highest meaningful place value
    for approximation is the hundreds place when attempting to approximate
    to the ten thousand place. -/
theorem approx_small_number_to_large_place (x : ℝ) : 
  x < 10000 → 
  ∃ (approx : ℝ), 
    (approx = 100 * ⌊x / 100⌋) ∧ 
    (∀ (y : ℝ), y = 1000 * ⌊x / 1000⌋ ∨ y = 10000 * ⌊x / 10000⌋ → |x - approx| ≤ |x - y|) :=
by sorry

end NUMINAMATH_CALUDE_approx_small_number_to_large_place_l3025_302520


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3025_302524

/-- A quadratic function with a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The condition that f has only one zero -/
def has_one_zero (a : ℝ) : Prop := ∃! x, f a x = 0

/-- The statement to be proved -/
theorem not_sufficient_nor_necessary :
  (∃ a, a ≤ -2 ∧ ¬(has_one_zero a)) ∧ 
  (∃ a, a > -2 ∧ has_one_zero a) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3025_302524


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l3025_302530

theorem polynomial_sum_theorem (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ), (10 * d - 3 + 16 * d^2) + (4 * d + 7) = a * d + b + c * d^2 ∧ a + b + c = 34 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l3025_302530


namespace NUMINAMATH_CALUDE_power_five_fifteen_div_power_twentyfive_six_l3025_302596

theorem power_five_fifteen_div_power_twentyfive_six :
  5^15 / 25^6 = 125 := by
sorry

end NUMINAMATH_CALUDE_power_five_fifteen_div_power_twentyfive_six_l3025_302596


namespace NUMINAMATH_CALUDE_find_number_l3025_302588

theorem find_number : ∃ x : ℤ, x - 27 = 49 ∧ x = 76 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3025_302588


namespace NUMINAMATH_CALUDE_jessicas_purchases_total_cost_l3025_302562

/-- The total cost of Jessica's purchases is $21.95, given that she spent $10.22 on a cat toy and $11.73 on a cage. -/
theorem jessicas_purchases_total_cost : 
  let cat_toy_cost : ℚ := 10.22
  let cage_cost : ℚ := 11.73
  cat_toy_cost + cage_cost = 21.95 := by sorry

end NUMINAMATH_CALUDE_jessicas_purchases_total_cost_l3025_302562


namespace NUMINAMATH_CALUDE_family_theater_cost_l3025_302503

/-- Represents the cost of a theater ticket --/
structure TicketCost where
  full : ℝ
  senior : ℝ
  student : ℝ

/-- Calculates the total cost of tickets for a family group --/
def totalCost (t : TicketCost) : ℝ :=
  3 * t.senior + 3 * t.full + 3 * t.student

/-- Theorem: Given the specified discounts and senior ticket cost, 
    the total cost for all family members is $90 --/
theorem family_theater_cost : 
  ∀ (t : TicketCost), 
    t.senior = 10 ∧ 
    t.senior = 0.8 * t.full ∧ 
    t.student = 0.6 * t.full → 
    totalCost t = 90 := by
  sorry


end NUMINAMATH_CALUDE_family_theater_cost_l3025_302503


namespace NUMINAMATH_CALUDE_min_triangle_area_l3025_302505

theorem min_triangle_area (p q : ℤ) : ∃ (min_area : ℚ), 
  min_area = 1 ∧ 
  ∀ (area : ℚ), area = (1 : ℚ) / 2 * |10 * p - 24 * q| → area ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l3025_302505


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3025_302558

theorem opposite_of_negative_2023 :
  (∀ x : ℤ, x + (-2023) = 0 → x = 2023) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3025_302558


namespace NUMINAMATH_CALUDE_burger_non_filler_percentage_l3025_302515

/-- Given a burger with total weight and filler weight, calculate the percentage that is not filler -/
theorem burger_non_filler_percentage 
  (total_weight : ℝ) 
  (filler_weight : ℝ) 
  (h1 : total_weight = 120)
  (h2 : filler_weight = 30) : 
  (total_weight - filler_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_burger_non_filler_percentage_l3025_302515


namespace NUMINAMATH_CALUDE_stamp_collection_total_l3025_302529

theorem stamp_collection_total (foreign : ℕ) (old : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : foreign = 90)
  (h2 : old = 60)
  (h3 : both = 20)
  (h4 : neither = 70) :
  foreign + old - both + neither = 200 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_total_l3025_302529


namespace NUMINAMATH_CALUDE_point_on_circle_range_l3025_302582

/-- Given two points A(a,0) and B(-a,0) where a > 0, and a circle with center (2√3, 2) and radius 3,
    if there exists a point P on the circle such that ∠APB = 90°, then 1 ≤ a ≤ 7. -/
theorem point_on_circle_range (a : ℝ) (h_a_pos : a > 0) :
  (∃ P : ℝ × ℝ, (P.1 - 2 * Real.sqrt 3)^2 + (P.2 - 2)^2 = 9 ∧ 
   (P.1 - a)^2 + P.2^2 + (P.1 + a)^2 + P.2^2 = ((P.1 - a)^2 + P.2^2) + ((P.1 + a)^2 + P.2^2)) →
  1 ≤ a ∧ a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_point_on_circle_range_l3025_302582


namespace NUMINAMATH_CALUDE_fraction_sum_l3025_302501

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3025_302501


namespace NUMINAMATH_CALUDE_average_length_of_strings_l3025_302563

def string1_length : ℝ := 2
def string2_length : ℝ := 6
def num_strings : ℕ := 2

theorem average_length_of_strings :
  (string1_length + string2_length) / num_strings = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_length_of_strings_l3025_302563


namespace NUMINAMATH_CALUDE_factorial_inequality_l3025_302585

theorem factorial_inequality (n p : ℕ) (h : 2 * p ≤ n) :
  (n - p).factorial / p.factorial ≤ ((n + 1) / 2 : ℚ) ^ (n - 2 * p) ∧
  ((n - p).factorial / p.factorial = ((n + 1) / 2 : ℚ) ^ (n - 2 * p) ↔ n = 2 * p ∨ n = 2 * p + 1) :=
by sorry

end NUMINAMATH_CALUDE_factorial_inequality_l3025_302585


namespace NUMINAMATH_CALUDE_bus_problem_l3025_302504

theorem bus_problem (initial : ℕ) (got_off : ℕ) (final : ℕ) :
  initial = 36 →
  got_off = 68 →
  final = 12 →
  got_off - (initial - got_off + final) = 24 :=
by sorry

end NUMINAMATH_CALUDE_bus_problem_l3025_302504


namespace NUMINAMATH_CALUDE_less_than_minus_l3025_302577

theorem less_than_minus (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a < a - b := by
  sorry

end NUMINAMATH_CALUDE_less_than_minus_l3025_302577


namespace NUMINAMATH_CALUDE_fraction_comparison_l3025_302539

theorem fraction_comparison : 
  (9 : ℚ) / 21 = (3 : ℚ) / 7 ∧ 
  (12 : ℚ) / 28 = (3 : ℚ) / 7 ∧ 
  (30 : ℚ) / 70 = (3 : ℚ) / 7 ∧ 
  (13 : ℚ) / 28 ≠ (3 : ℚ) / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3025_302539


namespace NUMINAMATH_CALUDE_right_triangle_with_special_point_l3025_302561

theorem right_triangle_with_special_point (A B C P : ℝ × ℝ) 
  (h_right : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0)
  (h_AP : Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 2)
  (h_BP : Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 1)
  (h_CP : Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = Real.sqrt 5)
  (h_inside : ∃ (t u : ℝ), t > 0 ∧ u > 0 ∧ t + u < 1 ∧ 
    P.1 = t * B.1 + u * C.1 + (1 - t - u) * A.1 ∧
    P.2 = t * B.2 + u * C.2 + (1 - t - u) * A.2) :
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 5 := by
  sorry

#check right_triangle_with_special_point

end NUMINAMATH_CALUDE_right_triangle_with_special_point_l3025_302561


namespace NUMINAMATH_CALUDE_chicken_cost_per_person_l3025_302533

def total_cost : ℝ := 16
def beef_price_per_pound : ℝ := 4
def beef_pounds : ℝ := 3
def oil_price : ℝ := 1
def number_of_people : ℕ := 3

theorem chicken_cost_per_person : 
  (total_cost - (beef_price_per_pound * beef_pounds + oil_price)) / number_of_people = 1 := by
  sorry

end NUMINAMATH_CALUDE_chicken_cost_per_person_l3025_302533


namespace NUMINAMATH_CALUDE_total_commute_time_l3025_302557

def first_bus_duration : ℕ := 40
def first_wait_duration : ℕ := 10
def second_bus_duration : ℕ := 50
def second_wait_duration : ℕ := 15
def third_bus_duration : ℕ := 95

theorem total_commute_time :
  first_bus_duration + first_wait_duration + second_bus_duration +
  second_wait_duration + third_bus_duration = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_commute_time_l3025_302557


namespace NUMINAMATH_CALUDE_three_digit_twice_divisible_by_1001_l3025_302526

theorem three_digit_twice_divisible_by_1001 (a : ℕ) : 
  100 ≤ a ∧ a < 1000 → ∃ k : ℕ, 1000 * a + a = 1001 * k := by
sorry

end NUMINAMATH_CALUDE_three_digit_twice_divisible_by_1001_l3025_302526


namespace NUMINAMATH_CALUDE_max_area_inscribed_triangle_l3025_302536

/-- The maximum area of a right-angled isosceles triangle inscribed in a 12x15 rectangle -/
theorem max_area_inscribed_triangle (a b : ℝ) (ha : a = 12) (hb : b = 15) :
  let max_area := Real.sqrt (min a b ^ 2 / 2)
  ∃ (x y : ℝ), x ≤ a ∧ y ≤ b ∧ x = y ∧ x * y / 2 = max_area ^ 2 ∧ max_area ^ 2 = 72 := by
  sorry


end NUMINAMATH_CALUDE_max_area_inscribed_triangle_l3025_302536


namespace NUMINAMATH_CALUDE_max_value_real_complex_l3025_302519

theorem max_value_real_complex (α β : ℝ) :
  (∃ (M : ℝ), ∀ (x y : ℝ), abs x ≤ 1 → abs y ≤ 1 →
    abs (α * x + β * y) + abs (α * x - β * y) ≤ M ∧
    M = 2 * Real.sqrt 2 * Real.sqrt (α^2 + β^2)) ∧
  (∃ (N : ℝ), ∀ (x y : ℂ), Complex.abs x ≤ 1 → Complex.abs y ≤ 1 →
    Complex.abs (α * x + β * y) + Complex.abs (α * x - β * y) ≤ N ∧
    N = 2 * abs α + 2 * abs β) :=
by sorry

end NUMINAMATH_CALUDE_max_value_real_complex_l3025_302519


namespace NUMINAMATH_CALUDE_problem_solution_l3025_302550

theorem problem_solution (x : ℝ) : (400 * 7000 : ℝ) = 28000 * (100 ^ x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3025_302550


namespace NUMINAMATH_CALUDE_cube_volume_puzzle_l3025_302583

theorem cube_volume_puzzle (a : ℝ) : 
  a > 0 → 
  (a + 2) * (a - 2) * a = a^3 - 8 → 
  a^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_puzzle_l3025_302583


namespace NUMINAMATH_CALUDE_divisible_by_six_l3025_302516

theorem divisible_by_six (n : ℕ) : 
  6 ∣ (n + 20) * (n + 201) * (n + 2020) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l3025_302516


namespace NUMINAMATH_CALUDE_interior_angle_regular_octagon_l3025_302576

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ) (sum_interior_angles : ℝ) (interior_angle : ℝ),
  n = 8 →
  sum_interior_angles = (n - 2) * 180 →
  interior_angle = sum_interior_angles / n →
  interior_angle = 135 := by
sorry

end NUMINAMATH_CALUDE_interior_angle_regular_octagon_l3025_302576


namespace NUMINAMATH_CALUDE_students_in_both_teams_l3025_302575

theorem students_in_both_teams (total : ℕ) (baseball : ℕ) (hockey : ℕ) 
  (h1 : total = 36) 
  (h2 : baseball = 25) 
  (h3 : hockey = 19) : 
  baseball + hockey - total = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_teams_l3025_302575


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3025_302546

theorem quadratic_root_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -c ∧ s₁ * s₂ = a) ∧
               (3*s₁ + 3*s₂ = -a ∧ 9*s₁*s₂ = b)) →
  b / c = 27 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3025_302546


namespace NUMINAMATH_CALUDE_john_balloons_l3025_302586

/-- The number of balloons John bought -/
def num_balloons : ℕ := sorry

/-- The volume of air each balloon holds in liters -/
def air_per_balloon : ℕ := 10

/-- The volume of gas in each tank in liters -/
def gas_per_tank : ℕ := 500

/-- The number of tanks John needs to fill all balloons -/
def num_tanks : ℕ := 20

theorem john_balloons :
  num_balloons = 1000 :=
by sorry

end NUMINAMATH_CALUDE_john_balloons_l3025_302586


namespace NUMINAMATH_CALUDE_spoiled_apple_probability_l3025_302595

/-- The probability of selecting a spoiled apple from a basket -/
def prob_spoiled_apple (total : ℕ) (spoiled : ℕ) (selected : ℕ) : ℚ :=
  (selected : ℚ) / total

/-- The number of ways to choose k items from n items -/
def combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem spoiled_apple_probability :
  let total := 7
  let spoiled := 1
  let selected := 2
  prob_spoiled_apple total spoiled selected = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_spoiled_apple_probability_l3025_302595


namespace NUMINAMATH_CALUDE_special_sum_equals_250_l3025_302548

/-- The sum of two arithmetic sequences with 5 terms each, where the first sequence starts at 3 and increases by 10, and the second sequence starts at 7 and increases by 10 -/
def special_sum : ℕ := (3+13+23+33+43)+(7+17+27+37+47)

/-- Theorem stating that the special sum equals 250 -/
theorem special_sum_equals_250 : special_sum = 250 := by
  sorry

end NUMINAMATH_CALUDE_special_sum_equals_250_l3025_302548


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3025_302568

universe u

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3025_302568


namespace NUMINAMATH_CALUDE_probability_same_heads_l3025_302589

/-- Represents the outcome of tossing two coins -/
inductive CoinToss
| HH -- Two heads
| HT -- Head then tail
| TH -- Tail then head
| TT -- Two tails

/-- The sample space of all possible outcomes when two people each toss two coins -/
def sampleSpace : List (CoinToss × CoinToss) :=
  [(CoinToss.HH, CoinToss.HH), (CoinToss.HH, CoinToss.HT), (CoinToss.HH, CoinToss.TH), (CoinToss.HH, CoinToss.TT),
   (CoinToss.HT, CoinToss.HH), (CoinToss.HT, CoinToss.HT), (CoinToss.HT, CoinToss.TH), (CoinToss.HT, CoinToss.TT),
   (CoinToss.TH, CoinToss.HH), (CoinToss.TH, CoinToss.HT), (CoinToss.TH, CoinToss.TH), (CoinToss.TH, CoinToss.TT),
   (CoinToss.TT, CoinToss.HH), (CoinToss.TT, CoinToss.HT), (CoinToss.TT, CoinToss.TH), (CoinToss.TT, CoinToss.TT)]

/-- Counts the number of heads in a single coin toss -/
def countHeads : CoinToss → Nat
  | CoinToss.HH => 2
  | CoinToss.HT => 1
  | CoinToss.TH => 1
  | CoinToss.TT => 0

/-- Checks if two coin tosses have the same number of heads -/
def sameHeads : CoinToss × CoinToss → Bool
  | (t1, t2) => countHeads t1 = countHeads t2

/-- The probability of getting the same number of heads -/
theorem probability_same_heads :
  (sampleSpace.filter sameHeads).length / sampleSpace.length = 3 / 8 := by
  sorry


end NUMINAMATH_CALUDE_probability_same_heads_l3025_302589


namespace NUMINAMATH_CALUDE_locus_of_tangent_points_theorem_l3025_302506

/-- The locus of points for which an ellipse or hyperbola with center at the origin is tangent -/
def locus_of_tangent_points (x y a b c : ℝ) : Prop :=
  (a^2 * y^2 + b^2 * x^2 = x^2 * y^2 ∧ b^2 = a^2 - c^2) ∨
  (a^2 * y^2 - b^2 * x^2 = x^2 * y^2 ∧ b^2 = c^2 - a^2)

/-- Theorem stating the locus of points for ellipses and hyperbolas with center at the origin -/
theorem locus_of_tangent_points_theorem (x y a b c : ℝ) :
  locus_of_tangent_points x y a b c :=
sorry

end NUMINAMATH_CALUDE_locus_of_tangent_points_theorem_l3025_302506
