import Mathlib

namespace NUMINAMATH_CALUDE_percentage_of_720_is_356_4_l3431_343174

theorem percentage_of_720_is_356_4 : 
  let whole : ℝ := 720
  let part : ℝ := 356.4
  let percentage : ℝ := (part / whole) * 100
  percentage = 49.5 := by sorry

end NUMINAMATH_CALUDE_percentage_of_720_is_356_4_l3431_343174


namespace NUMINAMATH_CALUDE_two_thirds_of_number_is_36_l3431_343182

theorem two_thirds_of_number_is_36 (x : ℚ) : (2 : ℚ) / 3 * x = 36 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_number_is_36_l3431_343182


namespace NUMINAMATH_CALUDE_polyhedron_volume_from_parallelepiped_l3431_343173

/-- Given a parallelepiped with volume V, the volume of the polyhedron formed by
    connecting the centers of its faces is 1/6 * V -/
theorem polyhedron_volume_from_parallelepiped (V : ℝ) (V_pos : V > 0) :
  ∃ (polyhedron_volume : ℝ),
    polyhedron_volume = (1 / 6 : ℝ) * V ∧
    polyhedron_volume > 0 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_from_parallelepiped_l3431_343173


namespace NUMINAMATH_CALUDE_adam_books_theorem_l3431_343181

def initial_books : ℕ := 67
def sold_fraction : ℚ := 2/3
def reinvestment_fraction : ℚ := 3/4
def new_book_price : ℕ := 3

def books_after_transactions : ℕ := 56

theorem adam_books_theorem :
  let sold_books := (initial_books * sold_fraction).floor
  let money_earned := sold_books * new_book_price
  let money_for_new_books := (money_earned : ℚ) * reinvestment_fraction
  let new_books := (money_for_new_books / new_book_price).floor
  initial_books - sold_books + new_books = books_after_transactions := by
  sorry

end NUMINAMATH_CALUDE_adam_books_theorem_l3431_343181


namespace NUMINAMATH_CALUDE_symmetry_axis_l3431_343125

/-- Given two lines l₁ and l₂ in a 2D plane, this function returns true if they are symmetric about a third line l. -/
def are_symmetric (l₁ l₂ l : ℝ → ℝ → Prop) : Prop := sorry

/-- The line with equation y = -x -/
def line_l₁ (x y : ℝ) : Prop := y = -x

/-- The line with equation x + y - 2 = 0 -/
def line_l₂ (x y : ℝ) : Prop := x + y - 2 = 0

/-- The proposed axis of symmetry -/
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

theorem symmetry_axis :
  are_symmetric line_l₁ line_l₂ line_l :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_l3431_343125


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3431_343194

theorem expansion_coefficient (a : ℝ) : 
  (∃ k : ℝ, k = 21 ∧ k = a^2 * 15 - 6 * a) ↔ (a = -1 ∨ a = 7/5) := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3431_343194


namespace NUMINAMATH_CALUDE_inequality_proof_l3431_343145

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 ∧
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3431_343145


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3431_343187

theorem geometric_sequence_problem (a b c d e : ℕ) : 
  (2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100) →
  Nat.gcd a e = 1 →
  (∃ (r : ℚ), r > 1 ∧ b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4) →
  c = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3431_343187


namespace NUMINAMATH_CALUDE_integral_sin_cos_power_l3431_343133

theorem integral_sin_cos_power : ∫ x in (-π/2)..0, (2^8 * Real.sin x^4 * Real.cos x^4) = 3*π := by sorry

end NUMINAMATH_CALUDE_integral_sin_cos_power_l3431_343133


namespace NUMINAMATH_CALUDE_weight_of_CaI2_l3431_343107

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of calcium atoms in CaI2 -/
def num_Ca_atoms : ℕ := 1

/-- The number of iodine atoms in CaI2 -/
def num_I_atoms : ℕ := 2

/-- The number of moles of CaI2 -/
def num_moles : ℝ := 3

/-- The molecular weight of CaI2 in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca * num_Ca_atoms + atomic_weight_I * num_I_atoms

/-- The total weight of CaI2 in grams -/
def weight_CaI2 : ℝ := molecular_weight_CaI2 * num_moles

theorem weight_of_CaI2 : weight_CaI2 = 881.64 := by sorry

end NUMINAMATH_CALUDE_weight_of_CaI2_l3431_343107


namespace NUMINAMATH_CALUDE_smallest_base_for_101_l3431_343197

/-- A number n can be expressed in base b using only two digits if b ≤ n < b^2 -/
def expressibleInTwoDigits (n : ℕ) (b : ℕ) : Prop :=
  b ≤ n ∧ n < b^2

/-- The smallest whole number b such that 101 can be expressed in base b using only two digits -/
def smallestBase : ℕ := 10

theorem smallest_base_for_101 :
  (∀ b : ℕ, b < smallestBase → ¬expressibleInTwoDigits 101 b) ∧
  expressibleInTwoDigits 101 smallestBase := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_101_l3431_343197


namespace NUMINAMATH_CALUDE_fence_repair_boards_count_l3431_343113

/-- Represents the number of boards nailed with a specific number of nails -/
structure BoardCount where
  count : ℕ
  nails_per_board : ℕ

/-- Represents a person's nailing work -/
structure NailingWork where
  first_type : BoardCount
  second_type : BoardCount

/-- Calculates the total number of nails used -/
def total_nails (work : NailingWork) : ℕ :=
  work.first_type.count * work.first_type.nails_per_board +
  work.second_type.count * work.second_type.nails_per_board

/-- Calculates the total number of boards nailed -/
def total_boards (work : NailingWork) : ℕ :=
  work.first_type.count + work.second_type.count

theorem fence_repair_boards_count :
  ∀ (petrov vasechkin : NailingWork),
    petrov.first_type.nails_per_board = 2 →
    petrov.second_type.nails_per_board = 3 →
    vasechkin.first_type.nails_per_board = 3 →
    vasechkin.second_type.nails_per_board = 5 →
    total_nails petrov = 87 →
    total_nails vasechkin = 94 →
    total_boards petrov = total_boards vasechkin →
    total_boards petrov = 30 :=
by sorry

end NUMINAMATH_CALUDE_fence_repair_boards_count_l3431_343113


namespace NUMINAMATH_CALUDE_car_distance_proof_l3431_343109

theorem car_distance_proof (west_speed : ℝ) (east_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  west_speed = 20 →
  east_speed = 60 →
  time = 5 →
  final_distance = 500 →
  ∃ (initial_north_south_distance : ℝ),
    initial_north_south_distance = 300 ∧
    initial_north_south_distance ^ 2 + (west_speed * time + east_speed * time) ^ 2 = final_distance ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_car_distance_proof_l3431_343109


namespace NUMINAMATH_CALUDE_polynomial_factorization_and_range_l3431_343193

-- Define the polynomial and factored form
def P (x : ℝ) := x^3 - 2*x^2 - x + 2
def Q (a b c x : ℝ) := (x + a) * (x + b) * (x + c)

-- State the theorem
theorem polynomial_factorization_and_range :
  ∃ (a b c : ℝ),
    (∀ x, P x = Q a b c x) ∧
    (a > b) ∧ (b > c) ∧
    (a = 1) ∧ (b = -1) ∧ (c = -2) ∧
    (∀ x ∈ Set.Icc 0 3, a*x^2 + 2*b*x + c ∈ Set.Icc (-3) 1) ∧
    (∃ x₁ ∈ Set.Icc 0 3, a*x₁^2 + 2*b*x₁ + c = -3) ∧
    (∃ x₂ ∈ Set.Icc 0 3, a*x₂^2 + 2*b*x₂ + c = 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_and_range_l3431_343193


namespace NUMINAMATH_CALUDE_remainder_x15_plus_1_div_x_plus_1_l3431_343179

theorem remainder_x15_plus_1_div_x_plus_1 (x : ℝ) : (x^15 + 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_x15_plus_1_div_x_plus_1_l3431_343179


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l3431_343132

def is_in_third_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, 180 * (2 * n + 1) < α ∧ α < 180 * (2 * n + 1) + 90

theorem angle_in_third_quadrant (k : ℤ) (α : ℝ) 
  (h : (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60) : 
  is_in_third_quadrant α :=
sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l3431_343132


namespace NUMINAMATH_CALUDE_bus_capacity_proof_l3431_343195

theorem bus_capacity_proof (C : ℕ) : 
  (3 : ℚ) / 5 * C + 32 = C → C = 80 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_proof_l3431_343195


namespace NUMINAMATH_CALUDE_minimum_value_theorem_equality_condition_l3431_343122

theorem minimum_value_theorem (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 2) : ∃ x, x > 2 ∧ x + 1 / (x - 2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_equality_condition_l3431_343122


namespace NUMINAMATH_CALUDE_first_book_price_l3431_343188

/-- Given 41 books arranged in increasing price order with a $3 difference between adjacent books,
    if the sum of the prices of the first and last books is $246,
    then the price of the first book is $63. -/
theorem first_book_price (n : ℕ) (price_diff : ℝ) (total_sum : ℝ) :
  n = 41 →
  price_diff = 3 →
  total_sum = 246 →
  ∃ (first_price : ℝ),
    first_price + (first_price + price_diff * (n - 1)) = total_sum ∧
    first_price = 63 := by
  sorry

end NUMINAMATH_CALUDE_first_book_price_l3431_343188


namespace NUMINAMATH_CALUDE_circle_radius_implies_c_l3431_343171

/-- Given a circle with equation x^2 + 6x + y^2 - 4y + c = 0 and radius 6, prove that c = -23 -/
theorem circle_radius_implies_c (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 6*x + y^2 - 4*y + c = 0 → (x+3)^2 + (y-2)^2 = 36) → 
  c = -23 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_implies_c_l3431_343171


namespace NUMINAMATH_CALUDE_daily_wage_of_c_l3431_343184

theorem daily_wage_of_c (a b c : ℕ) (total_earning : ℚ) : 
  a = 6 ∧ b = 9 ∧ c = 4 → 
  ∃ (x : ℚ), 
    (3 * x * a + 4 * x * b + 5 * x * c = total_earning) ∧
    (total_earning = 1480) →
    5 * x = 100 := by
  sorry

end NUMINAMATH_CALUDE_daily_wage_of_c_l3431_343184


namespace NUMINAMATH_CALUDE_min_value_product_l3431_343143

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 2) :
  (x + y) * (y + 3 * z) * (2 * x * z + 1) ≥ 16 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l3431_343143


namespace NUMINAMATH_CALUDE_S_6_value_l3431_343196

theorem S_6_value (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 := by
  sorry

end NUMINAMATH_CALUDE_S_6_value_l3431_343196


namespace NUMINAMATH_CALUDE_children_with_vip_seats_l3431_343142

/-- Proves the number of children with VIP seats in a concert hall -/
theorem children_with_vip_seats
  (total_attendees : ℕ)
  (children_percentage : ℚ)
  (vip_children_percentage : ℚ)
  (h1 : total_attendees = 400)
  (h2 : children_percentage = 75 / 100)
  (h3 : vip_children_percentage = 20 / 100) :
  ⌊(total_attendees : ℚ) * children_percentage * vip_children_percentage⌋ = 60 := by
  sorry

#check children_with_vip_seats

end NUMINAMATH_CALUDE_children_with_vip_seats_l3431_343142


namespace NUMINAMATH_CALUDE_largest_d_for_g_range_contains_two_l3431_343185

/-- The function g(x) defined as x^2 - 6x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + d

/-- The theorem stating that the largest value of d such that 2 is in the range of g(x) is 11 -/
theorem largest_d_for_g_range_contains_two :
  ∃ (d_max : ℝ), d_max = 11 ∧
  (∀ d : ℝ, (∃ x : ℝ, g d x = 2) → d ≤ d_max) ∧
  (∃ x : ℝ, g d_max x = 2) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_g_range_contains_two_l3431_343185


namespace NUMINAMATH_CALUDE_adams_purchase_cost_l3431_343120

/-- The cost of Adam's purchases of nuts and dried fruits -/
theorem adams_purchase_cost :
  let nuts_quantity : ℝ := 3
  let dried_fruits_quantity : ℝ := 2.5
  let nuts_price_per_kg : ℝ := 12
  let dried_fruits_price_per_kg : ℝ := 8
  let total_cost : ℝ := nuts_quantity * nuts_price_per_kg + dried_fruits_quantity * dried_fruits_price_per_kg
  total_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_adams_purchase_cost_l3431_343120


namespace NUMINAMATH_CALUDE_smallest_rectangle_area_l3431_343114

theorem smallest_rectangle_area (r : ℝ) (h : r = 5) :
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ w * l = 200 ∧
  ∀ (w' l' : ℝ), w' > 0 → l' > 0 → w' * l' ≥ 200 →
  (∀ (x y : ℝ), x^2 + y^2 ≤ r^2 → 0 ≤ x ∧ x ≤ w' ∧ 0 ≤ y ∧ y ≤ l') :=
by sorry

end NUMINAMATH_CALUDE_smallest_rectangle_area_l3431_343114


namespace NUMINAMATH_CALUDE_abs_z_equals_five_l3431_343128

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := sorry

-- State the theorem
theorem abs_z_equals_five :
  z * i^2018 = 3 + 4*i → Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_abs_z_equals_five_l3431_343128


namespace NUMINAMATH_CALUDE_delicate_triangle_existence_and_property_l3431_343117

/-- Definition of a delicate triangle -/
def is_delicate_triangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧ (1 / a : ℚ) = (1 / b : ℚ) + (1 / c : ℚ)

theorem delicate_triangle_existence_and_property :
  (∃ a b c : ℕ, is_delicate_triangle a b c) ∧
  (∀ a b c : ℕ, is_delicate_triangle a b c → ∃ n : ℕ, a^2 + b^2 + c^2 = n^2) :=
by sorry

end NUMINAMATH_CALUDE_delicate_triangle_existence_and_property_l3431_343117


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l3431_343103

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (9 * x) = 2 * Real.sin (6 * x) * Real.cos (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l3431_343103


namespace NUMINAMATH_CALUDE_correct_transformation_l3431_343191

theorem correct_transformation (y : ℝ) : 
  (|y + 1| / 2 = |y| / 3 - |3*y - 1| / 6 - y) ↔ 
  (3*y + 3 = 2*y - 3*y + 1 - 6*y) := by
sorry

end NUMINAMATH_CALUDE_correct_transformation_l3431_343191


namespace NUMINAMATH_CALUDE_tire_cost_calculation_l3431_343106

theorem tire_cost_calculation (total_cost : ℕ) (num_tires : ℕ) (h1 : total_cost = 240) (h2 : num_tires = 4) :
  total_cost / num_tires = 60 := by
  sorry

end NUMINAMATH_CALUDE_tire_cost_calculation_l3431_343106


namespace NUMINAMATH_CALUDE_post_office_distance_l3431_343111

/-- Proves that the distance of a round trip journey is 10 km given specific conditions -/
theorem post_office_distance (outward_speed return_speed total_time : ℝ) 
  (h1 : outward_speed = 12.5)
  (h2 : return_speed = 2)
  (h3 : total_time = 5.8) : 
  (total_time * outward_speed * return_speed) / (outward_speed + return_speed) = 10 := by
  sorry

end NUMINAMATH_CALUDE_post_office_distance_l3431_343111


namespace NUMINAMATH_CALUDE_phd_total_time_l3431_343127

def phd_timeline (acclimation_time : ℝ) (basics_time : ℝ) (research_ratio : ℝ) (dissertation_ratio : ℝ) : ℝ :=
  let research_time := basics_time * (1 + research_ratio)
  let dissertation_time := acclimation_time * dissertation_ratio
  acclimation_time + basics_time + research_time + dissertation_time

theorem phd_total_time :
  phd_timeline 1 2 0.75 0.5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_phd_total_time_l3431_343127


namespace NUMINAMATH_CALUDE_pizza_problem_l3431_343154

/-- Calculates the number of pizza slices left per person given the initial number of slices and the number of slices eaten. -/
def slices_left_per_person (small_pizza_slices large_pizza_slices eaten_per_person : ℕ) : ℕ :=
  let total_slices := small_pizza_slices + large_pizza_slices
  let total_eaten := 2 * eaten_per_person
  let slices_left := total_slices - total_eaten
  slices_left / 2

/-- Theorem stating that given the specific conditions of the problem, the number of slices left per person is 2. -/
theorem pizza_problem : slices_left_per_person 8 14 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l3431_343154


namespace NUMINAMATH_CALUDE_pool_volume_is_60_gallons_l3431_343124

/-- The volume of water in Lydia's pool when full -/
def pool_volume (inflow_rate outflow_rate fill_time : ℝ) : ℝ :=
  (inflow_rate - outflow_rate) * fill_time

/-- Theorem stating that the pool volume is 60 gallons -/
theorem pool_volume_is_60_gallons :
  pool_volume 1.6 0.1 40 = 60 := by
  sorry

end NUMINAMATH_CALUDE_pool_volume_is_60_gallons_l3431_343124


namespace NUMINAMATH_CALUDE_inequality_proof_l3431_343169

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3431_343169


namespace NUMINAMATH_CALUDE_min_breaks_for_40_tiles_l3431_343198

/-- Represents a chocolate bar -/
structure ChocolateBar where
  tiles : ℕ

/-- Represents the breaking process -/
def breakChocolate (initial : ChocolateBar) (breaks : ℕ) : ℕ :=
  initial.tiles + breaks

/-- Theorem: The minimum number of breaks required for a 40-tile chocolate bar is 39 -/
theorem min_breaks_for_40_tiles (bar : ChocolateBar) (h : bar.tiles = 40) :
  ∃ (breaks : ℕ), breakChocolate bar breaks = 40 ∧ 
  ∀ (n : ℕ), breakChocolate bar n = 40 → breaks ≤ n :=
by sorry

end NUMINAMATH_CALUDE_min_breaks_for_40_tiles_l3431_343198


namespace NUMINAMATH_CALUDE_geometric_sequence_special_property_l3431_343166

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_special_property 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arithmetic : 2 * ((1/2) * a 3) = a 1 + 2 * a 2) :
  q = 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_property_l3431_343166


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l3431_343137

/-- A bag containing balls of two colors -/
structure Bag where
  black : ℕ
  white : ℕ

/-- The event of drawing balls from the bag -/
structure Draw where
  total : ℕ
  black : ℕ
  white : ℕ

/-- Definition of the specific bag in the problem -/
def problem_bag : Bag := { black := 2, white := 2 }

/-- Definition of drawing two balls -/
def two_ball_draw (b : Bag) : Set Draw := 
  {d | d.total = 2 ∧ d.black + d.white = d.total ∧ d.black ≤ b.black ∧ d.white ≤ b.white}

/-- Event: At least one black ball is drawn -/
def at_least_one_black (d : Draw) : Prop := d.black ≥ 1

/-- Event: All drawn balls are white -/
def all_white (d : Draw) : Prop := d.white = d.total

/-- Theorem: The events are mutually exclusive and complementary -/
theorem events_mutually_exclusive_and_complementary :
  let draw_set := two_ball_draw problem_bag
  ∀ d ∈ draw_set, (at_least_one_black d ↔ ¬ all_white d) ∧ 
                  (at_least_one_black d ∨ all_white d) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l3431_343137


namespace NUMINAMATH_CALUDE_contiguous_substring_divisible_by_2011_l3431_343126

def isContiguousSubstring (s t : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), t = (s / 10^k) % 10^m

theorem contiguous_substring_divisible_by_2011 :
  ∃ (N : ℕ), ∀ (a : ℕ), a > N →
    ∃ (s : ℕ), isContiguousSubstring a s ∧ s % 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_contiguous_substring_divisible_by_2011_l3431_343126


namespace NUMINAMATH_CALUDE_equal_prod_of_divisors_implies_equal_numbers_l3431_343172

/-- The sum of positive divisors of a natural number -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The product of positive divisors of a natural number -/
def prod_of_divisors (n : ℕ) : ℕ := n ^ ((sum_of_divisors n).div 2)

/-- The number of positive divisors of a natural number -/
def num_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the product of all positive divisors of two natural numbers are equal, 
    then the two numbers are equal -/
theorem equal_prod_of_divisors_implies_equal_numbers (n m : ℕ) : 
  prod_of_divisors n = prod_of_divisors m → n = m := by sorry

end NUMINAMATH_CALUDE_equal_prod_of_divisors_implies_equal_numbers_l3431_343172


namespace NUMINAMATH_CALUDE_solve_seashells_problem_l3431_343112

def seashells_problem (monday_shells : ℕ) (total_money : ℚ) : Prop :=
  let tuesday_shells : ℕ := monday_shells / 2
  let total_shells : ℕ := monday_shells + tuesday_shells
  let money_per_shell : ℚ := total_money / total_shells
  monday_shells = 30 ∧ total_money = 54 → money_per_shell = 1.20

theorem solve_seashells_problem :
  seashells_problem 30 54 := by sorry

end NUMINAMATH_CALUDE_solve_seashells_problem_l3431_343112


namespace NUMINAMATH_CALUDE_school_camp_buses_l3431_343140

theorem school_camp_buses (B : ℕ) (S : ℕ) : 
  B ≤ 18 ∧                           -- No more than 18 buses
  S = 22 * B + 3 ∧                   -- Initial distribution with 3 left out
  ∃ (n : ℕ), n ≤ 36 ∧                -- Each bus can hold up to 36 people
  S = n * (B - 1) ∧                  -- Even distribution after one bus leaves
  n = (22 * B + 3) / (B - 1) →       -- Relationship between n, B, and S
  S = 355 :=
by sorry

end NUMINAMATH_CALUDE_school_camp_buses_l3431_343140


namespace NUMINAMATH_CALUDE_inverse_undefined_at_one_l3431_343108

/-- Given a function g(x) = (x - 5) / (x - 6), prove that its inverse g⁻¹(x) is undefined when x = 1 -/
theorem inverse_undefined_at_one (g : ℝ → ℝ) (h : ∀ x, g x = (x - 5) / (x - 6)) :
  ¬∃ y, g y = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_undefined_at_one_l3431_343108


namespace NUMINAMATH_CALUDE_expression_evaluation_l3431_343158

theorem expression_evaluation (x y z : ℤ) (hx : x = -2) (hy : y = -4) (hz : z = 3) :
  (5 * (x - y)^2 - x * z^2) / (z - y) = 38 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3431_343158


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_1000_l3431_343190

theorem negation_of_existence (P : ℕ → Prop) :
  (¬∃ n, P n) ↔ (∀ n, ¬P n) := by sorry

theorem negation_of_greater_than_1000 :
  (¬∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_1000_l3431_343190


namespace NUMINAMATH_CALUDE_conductor_loop_properties_l3431_343144

/-- Parameters for the conductor and loop setup -/
structure ConductorLoopSetup where
  k : Real  -- Current rate of change (A/s)
  r : Real  -- Side length of square loop (m)
  R : Real  -- Resistance of loop (Ω)
  l : Real  -- Distance from straight conductor to loop (m)

/-- Calculate the induced voltage in the loop -/
noncomputable def inducedVoltage (setup : ConductorLoopSetup) : Real :=
  (setup.k * setup.r * Real.log (1 + setup.r / setup.l)) / (2 * Real.pi)

/-- Calculate the time when magnetic induction at the center is zero -/
noncomputable def zeroInductionTime (setup : ConductorLoopSetup) : Real :=
  (4 * Real.sqrt 2 * (setup.l + setup.r / 2) * (inducedVoltage setup / setup.R)) / (setup.k * setup.r)

/-- Theorem stating the properties of the conductor-loop system -/
theorem conductor_loop_properties (setup : ConductorLoopSetup) 
  (h_k : setup.k = 1000)
  (h_r : setup.r = 0.2)
  (h_R : setup.R = 0.01)
  (h_l : setup.l = 0.05) :
  abs (inducedVoltage setup - 6.44e-5) < 1e-7 ∧ 
  abs (zeroInductionTime setup - 2.73e-4) < 1e-6 := by
  sorry


end NUMINAMATH_CALUDE_conductor_loop_properties_l3431_343144


namespace NUMINAMATH_CALUDE_min_value_of_exp_minus_x_l3431_343186

theorem min_value_of_exp_minus_x :
  ∃ (x : ℝ), ∀ (y : ℝ), Real.exp y - y ≥ Real.exp x - x ∧ Real.exp x - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exp_minus_x_l3431_343186


namespace NUMINAMATH_CALUDE_range_of_a_l3431_343156

def p (a : ℝ) : Prop :=
  ∀ m ∈ Set.Icc (-1) 1, a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a*x + 2 < 0

theorem range_of_a :
  ∃ S : Set ℝ, S = Set.Icc (-2 * Real.sqrt 2) (-1) ∪ Set.Ioo (2 * Real.sqrt 2) 6 ∧
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ S :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3431_343156


namespace NUMINAMATH_CALUDE_unique_square_pattern_l3431_343101

def fits_pattern (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∀ d₁ d₂ d₃, n = 100 * d₁ + 10 * d₂ + d₃ →
    d₁ * d₁ < 10 ∧
    d₁ * d₂ < 10 ∧
    d₁ * d₃ < 10 ∧
    d₂ * d₂ < 10 ∧
    d₂ * d₃ < 10 ∧
    d₃ * d₃ < 10

theorem unique_square_pattern :
  ∃! n : ℕ, fits_pattern n ∧ n = 233 :=
sorry

end NUMINAMATH_CALUDE_unique_square_pattern_l3431_343101


namespace NUMINAMATH_CALUDE_g_of_3_eq_38_div_5_l3431_343192

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 1 / (f.invFun x) + 7

theorem g_of_3_eq_38_div_5 : g 3 = 38 / 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_38_div_5_l3431_343192


namespace NUMINAMATH_CALUDE_smallest_b_in_geometric_series_l3431_343148

theorem smallest_b_in_geometric_series (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- a, b, c are positive
  (∃ r : ℝ, a = b * r ∧ c = b / r) →  -- a, b, c form a geometric series
  a * b * c = 216 →  -- product condition
  b ≥ 6 ∧ (∀ b' : ℝ, 
    (∃ a' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    (∃ r' : ℝ, a' = b' * r' ∧ c' = b' / r') ∧ 
    a' * b' * c' = 216) → 
    b' ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_geometric_series_l3431_343148


namespace NUMINAMATH_CALUDE_problem_solution_l3431_343105

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (x / 4) * Real.cos (x / 4) + Real.cos (x / 4) ^ 2

theorem problem_solution :
  (∀ x : ℝ, f x = 1 → Real.cos (2 * Real.pi / 3 - x) = -1 / 2) ∧
  (∀ A B C a b c : ℝ,
    0 < A ∧ A < Real.pi / 2 ∧
    0 < B ∧ B < Real.pi / 2 ∧
    0 < C ∧ C < Real.pi / 2 ∧
    A + B + C = Real.pi ∧
    a * Real.cos C + c / 2 = b →
    (1 + Real.sqrt 3) / 2 < f (2 * B) ∧ f (2 * B) < 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3431_343105


namespace NUMINAMATH_CALUDE_calculate_total_profit_l3431_343138

/-- Given the investments of three partners and the profit share of one partner,
    calculate the total profit of the business. -/
theorem calculate_total_profit
  (a_investment b_investment c_investment : ℕ)
  (c_profit_share : ℕ)
  (h1 : a_investment = 5000)
  (h2 : b_investment = 15000)
  (h3 : c_investment = 30000)
  (h4 : c_profit_share = 3000) :
  (a_investment + b_investment + c_investment) * c_profit_share
  / c_investment = 5000 :=
sorry

end NUMINAMATH_CALUDE_calculate_total_profit_l3431_343138


namespace NUMINAMATH_CALUDE_sqrt_three_times_sqrt_six_equals_three_sqrt_two_l3431_343104

theorem sqrt_three_times_sqrt_six_equals_three_sqrt_two :
  Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_sqrt_six_equals_three_sqrt_two_l3431_343104


namespace NUMINAMATH_CALUDE_max_gcd_of_consecutive_cubic_sequence_l3431_343134

theorem max_gcd_of_consecutive_cubic_sequence :
  let b : ℕ → ℕ := fun n => 150 + n^3
  let d : ℕ → ℕ := fun n => Nat.gcd (b n) (b (n + 1))
  ∀ n : ℕ, n ≥ 1 → d n ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_consecutive_cubic_sequence_l3431_343134


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3431_343199

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a3_eq_6 : a 3 = 6
  S3_eq_12 : S 3 = 12

/-- The theorem stating properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 2 * n) ∧
  (∀ n, seq.S n = n * (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3431_343199


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l3431_343149

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2*a*x^2 + 3*(a+2)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) →
  (a > 2 ∨ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l3431_343149


namespace NUMINAMATH_CALUDE_inequality_system_sum_l3431_343183

theorem inequality_system_sum (a b : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x + 2*a > 4 ∧ 2*x < b)) →
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_sum_l3431_343183


namespace NUMINAMATH_CALUDE_jerrys_average_score_l3431_343163

theorem jerrys_average_score (A : ℝ) : 
  (∀ (new_average : ℝ), new_average = A + 2 → 
    3 * A + 102 = 4 * new_average) → 
  A = 94 := by
sorry

end NUMINAMATH_CALUDE_jerrys_average_score_l3431_343163


namespace NUMINAMATH_CALUDE_solution_set_implies_values_l3431_343110

/-- Given that the solution set of ax^2 + bx + a^2 - 1 ≤ 0 is [-1, +∞), prove a = 0 and b = -1 -/
theorem solution_set_implies_values (a b : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → a * x^2 + b * x + a^2 - 1 ≤ 0) → 
  a = 0 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_values_l3431_343110


namespace NUMINAMATH_CALUDE_decreasing_geometric_sequence_properties_l3431_343168

/-- A decreasing geometric sequence -/
def DecreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ |a (n + 1)| < |a n|

theorem decreasing_geometric_sequence_properties
  (a : ℕ → ℝ) (h : DecreasingGeometricSequence a) :
  (∀ n : ℕ, a n > 0 → ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ m : ℕ, a (m + 1) = q * a m) ∧
  (∀ n : ℕ, a n < 0 → ∃ q : ℝ, q > 1 ∧ ∀ m : ℕ, a (m + 1) = q * a m) :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_geometric_sequence_properties_l3431_343168


namespace NUMINAMATH_CALUDE_average_goat_price_l3431_343177

/-- Given the number of goats and hens, their total cost, and the average cost of a hen,
    calculate the average cost of a goat. -/
theorem average_goat_price
  (num_goats : ℕ)
  (num_hens : ℕ)
  (total_cost : ℕ)
  (avg_hen_price : ℕ)
  (h1 : num_goats = 5)
  (h2 : num_hens = 10)
  (h3 : total_cost = 2500)
  (h4 : avg_hen_price = 50) :
  (total_cost - num_hens * avg_hen_price) / num_goats = 400 := by
  sorry

#check average_goat_price

end NUMINAMATH_CALUDE_average_goat_price_l3431_343177


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l3431_343155

theorem complex_fraction_calculation : 
  ∃ ε > 0, |((9/20 : ℚ) - 11/30 + 13/42 - 15/56 + 17/72) * 120 - (1/3) / (1/4) - 42| < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l3431_343155


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3431_343141

theorem quadratic_equation_roots (p : ℝ) : 
  (∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ 
   (∀ x : ℂ, x^2 - p*x + 1 = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
   x₁.im ≠ 0 ∧ x₂.im ≠ 0 ∧
   Complex.abs (x₁ - x₂) = 1) →
  p = Real.sqrt 3 ∨ p = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3431_343141


namespace NUMINAMATH_CALUDE_dexter_cards_total_dexter_cards_count_l3431_343118

theorem dexter_cards_total (basketball_boxes : ℕ) (basketball_cards_per_box : ℕ) 
  (football_cards_per_box : ℕ) (box_difference : ℕ) : ℕ :=
  let football_boxes := basketball_boxes - box_difference
  let total_basketball_cards := basketball_boxes * basketball_cards_per_box
  let total_football_cards := football_boxes * football_cards_per_box
  total_basketball_cards + total_football_cards

-- Main theorem
theorem dexter_cards_count : 
  dexter_cards_total 12 20 25 5 = 415 := by
  sorry

end NUMINAMATH_CALUDE_dexter_cards_total_dexter_cards_count_l3431_343118


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3431_343115

theorem solution_set_inequality (x : ℝ) : 
  (1 / x > 1 / (x - 1)) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3431_343115


namespace NUMINAMATH_CALUDE_sequence_limit_implies_first_term_l3431_343146

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = n

def limit_property (a : ℕ → ℝ) : Prop :=
  Filter.Tendsto (λ n => a n / a (n + 1)) Filter.atTop (nhds 1)

theorem sequence_limit_implies_first_term (a : ℕ → ℝ) 
    (h1 : sequence_property a) 
    (h2 : limit_property a) : 
    a 1 = Real.sqrt (2 / Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sequence_limit_implies_first_term_l3431_343146


namespace NUMINAMATH_CALUDE_thirty_people_three_groups_l3431_343102

/-- The number of ways to divide n people into k groups of m people each -/
def group_divisions (n m k : ℕ) : ℕ :=
  if n = m * k then
    Nat.factorial n / (Nat.factorial m ^ k)
  else
    0

/-- Theorem: The number of ways to divide 30 people into 3 groups of 10 each
    is equal to 30! / (10!)³ -/
theorem thirty_people_three_groups :
  group_divisions 30 10 3 = Nat.factorial 30 / (Nat.factorial 10 ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_thirty_people_three_groups_l3431_343102


namespace NUMINAMATH_CALUDE_bus_journey_speed_l3431_343157

/-- Given a bus journey with specific conditions, prove the average speed for the remaining distance -/
theorem bus_journey_speed (total_distance : ℝ) (total_time : ℝ) (partial_distance : ℝ) (partial_speed : ℝ)
  (h1 : total_distance = 250)
  (h2 : total_time = 6)
  (h3 : partial_distance = 220)
  (h4 : partial_speed = 40)
  (h5 : partial_distance / partial_speed + (total_distance - partial_distance) / (total_time - partial_distance / partial_speed) = total_time) :
  (total_distance - partial_distance) / (total_time - partial_distance / partial_speed) = 60 := by
  sorry

#check bus_journey_speed

end NUMINAMATH_CALUDE_bus_journey_speed_l3431_343157


namespace NUMINAMATH_CALUDE_problem_statement_l3431_343139

theorem problem_statement : ¬(
  (∀ (p q : Prop), (p → ¬p) ↔ (q → ¬p)) ∧
  ((∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≥ 1) ∧ 
   (∃ x : ℝ, x^2 + x + 1 < 0)) ∧
  (¬∀ (a b m : ℝ), a * m^2 < b * m^2 → a < b) ∧
  (∀ (a b : ℝ), (a + b) / 2 ≥ Real.sqrt (a * b) → (a > 0 ∧ b > 0))
) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3431_343139


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_l3431_343129

theorem quadratic_roots_distance (t : ℝ) (x₁ x₂ : ℂ) :
  x₁^2 + t*x₁ + 2 = 0 →
  x₂^2 + t*x₂ + 2 = 0 →
  x₁ ≠ x₂ →
  Complex.abs (x₁ - x₂) = 2 * Real.sqrt 2 →
  t = -4 ∨ t = 0 ∨ t = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distance_l3431_343129


namespace NUMINAMATH_CALUDE_right_triangle_area_l3431_343161

/-- The area of a right triangle with hypotenuse 10√2 and one 45° angle is 50 square inches. -/
theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 10 * Real.sqrt 2 →
  angle = 45 * π / 180 →
  (1 / 2) * (h / Real.sqrt 2) * (h / Real.sqrt 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3431_343161


namespace NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l3431_343152

/-- A positive integer greater than 1 is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ∃ k, 1 < k ∧ k < n ∧ k ∣ n

/-- The largest integer that always divides n^4 - n for all composite n is 6 -/
theorem largest_divisor_n4_minus_n (n : ℕ) (h : IsComposite n) :
  (∀ m : ℕ, m > 6 → ¬(m ∣ (n^4 - n))) ∧ (6 ∣ (n^4 - n)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l3431_343152


namespace NUMINAMATH_CALUDE_seed_germination_probability_l3431_343176

/-- The probability of success in a single trial -/
def p : ℝ := 0.9

/-- The probability of failure in a single trial -/
def q : ℝ := 1 - p

/-- The number of trials -/
def n : ℕ := 4

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Probability of exactly k successes in n trials -/
def P (k : ℕ) : ℝ := (binomial n k : ℝ) * p^k * q^(n - k)

theorem seed_germination_probability :
  (P 3 = 0.2916) ∧ (P 3 + P 4 = 0.9477) := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_probability_l3431_343176


namespace NUMINAMATH_CALUDE_largest_valid_number_l3431_343160

def is_valid_number (n : ℕ) : Prop :=
  ∃ (r : ℕ) (i : ℕ), 
    i > 0 ∧ 
    i < (Nat.digits 10 n).length ∧ 
    n % 10 ≠ 0 ∧
    r > 1 ∧
    r * (n / 10^(i + 1) * 10^i + n % 10^i) = n

theorem largest_valid_number : 
  is_valid_number 180625 ∧ 
  ∀ m : ℕ, m > 180625 → ¬(is_valid_number m) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3431_343160


namespace NUMINAMATH_CALUDE_fruits_picked_and_ratio_l3431_343165

/-- Represents the number of fruits picked by a person -/
structure FruitsPicked where
  pears : ℕ
  apples : ℕ

/-- Represents the orchard -/
structure Orchard where
  pear_trees : ℕ
  apple_trees : ℕ

def keith_picked : FruitsPicked := { pears := 6, apples := 4 }
def jason_picked : FruitsPicked := { pears := 9, apples := 8 }
def joan_picked : FruitsPicked := { pears := 4, apples := 12 }

def orchard : Orchard := { pear_trees := 4, apple_trees := 3 }

def total_fruits (keith jason joan : FruitsPicked) : ℕ :=
  keith.pears + keith.apples + jason.pears + jason.apples + joan.pears + joan.apples

def total_apples (keith jason joan : FruitsPicked) : ℕ :=
  keith.apples + jason.apples + joan.apples

def total_pears (keith jason joan : FruitsPicked) : ℕ :=
  keith.pears + jason.pears + joan.pears

theorem fruits_picked_and_ratio 
  (keith jason joan : FruitsPicked) 
  (o : Orchard) 
  (h_keith : keith = keith_picked)
  (h_jason : jason = jason_picked)
  (h_joan : joan = joan_picked)
  (h_orchard : o = orchard) :
  total_fruits keith jason joan = 43 ∧ 
  total_apples keith jason joan = 24 ∧
  total_pears keith jason joan = 19 := by
  sorry

end NUMINAMATH_CALUDE_fruits_picked_and_ratio_l3431_343165


namespace NUMINAMATH_CALUDE_eva_marks_total_l3431_343121

/-- Represents Eva's marks in a single semester -/
structure SemesterMarks where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total marks for a semester -/
def totalMarks (s : SemesterMarks) : ℕ :=
  s.maths + s.arts + s.science

/-- Represents Eva's marks for the entire year -/
structure YearMarks where
  first : SemesterMarks
  second : SemesterMarks

/-- Calculates the total marks for the year -/
def yearTotal (y : YearMarks) : ℕ :=
  totalMarks y.first + totalMarks y.second

theorem eva_marks_total (eva : YearMarks) 
  (h1 : eva.first.maths = eva.second.maths + 10)
  (h2 : eva.first.arts = eva.second.arts - 15)
  (h3 : eva.first.science = eva.second.science - eva.second.science / 3)
  (h4 : eva.second.maths = 80)
  (h5 : eva.second.arts = 90)
  (h6 : eva.second.science = 90) :
  yearTotal eva = 485 := by
  sorry

end NUMINAMATH_CALUDE_eva_marks_total_l3431_343121


namespace NUMINAMATH_CALUDE_median_equations_l3431_343151

/-- Triangle ABC with given coordinates -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Equation of a line in general form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given triangle ABC -/
def givenTriangle : Triangle :=
  { A := (1, -4)
  , B := (6, 6)
  , C := (-2, 0) }

/-- Theorem stating the equations of the two medians -/
theorem median_equations (t : Triangle) 
  (h : t = givenTriangle) : 
  ∃ (l1 l2 : LineEquation),
    (l1.a = 6 ∧ l1.b = -8 ∧ l1.c = -13) ∧
    (l2.a = 7 ∧ l2.b = -1 ∧ l2.c = -11) :=
  sorry

end NUMINAMATH_CALUDE_median_equations_l3431_343151


namespace NUMINAMATH_CALUDE_no_common_points_l3431_343147

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (1 + 2*t, -2 + 4*t)

-- Define the line L: 2x - y = 0
def line_L (x y : ℝ) : Prop := 2*x - y = 0

-- Theorem statement
theorem no_common_points :
  ∀ (t : ℝ), ¬(line_L (curve_C t).1 (curve_C t).2) := by
  sorry

end NUMINAMATH_CALUDE_no_common_points_l3431_343147


namespace NUMINAMATH_CALUDE_sweater_cost_l3431_343131

def original_savings : ℚ := 80

def makeup_fraction : ℚ := 3/4

theorem sweater_cost :
  let makeup_cost : ℚ := makeup_fraction * original_savings
  let sweater_cost : ℚ := original_savings - makeup_cost
  sweater_cost = 20 := by sorry

end NUMINAMATH_CALUDE_sweater_cost_l3431_343131


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3431_343162

theorem quadratic_unique_solution (a b c k : ℝ) (h1 : a = 8) (h2 : b = 36) 
  (h3 : ∀ x, a * x^2 + b * x + k = 0 ↔ x = -2.25) : k = 40.5 := by
  sorry

#check quadratic_unique_solution

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3431_343162


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3431_343150

/-- If x^2 + 80x + c is equal to the square of a binomial, then c = 1600 -/
theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 80*x + c = (x + a)^2) → c = 1600 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3431_343150


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3431_343123

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 * (x + 1)^2 = a*x^8 + a₁*x^7 + a₂*x^6 + a₃*x^5 + a₄*x^4 + a₅*x^3 + a₆*x^2 + a₇*x + a₈) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3431_343123


namespace NUMINAMATH_CALUDE_sanhat_integers_l3431_343136

theorem sanhat_integers (x y : ℤ) (h1 : 3 * x + 2 * y = 160) (h2 : x = 36 ∨ y = 36) :
  (x = 36 ∧ y = 26) ∨ (y = 36 ∧ x = 26) :=
sorry

end NUMINAMATH_CALUDE_sanhat_integers_l3431_343136


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3431_343135

-- Problem 1
theorem problem_1 : 
  (2 + Real.sqrt 3) ^ 0 + 3 * Real.tan (30 * π / 180) - |Real.sqrt 3 - 2| + (1/2)⁻¹ = 1 + 2 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) (ha : a^2 - 4*a + 3 = 0) (hne : a*(a+3)*(a-3) ≠ 0) : 
  (a^2 - 9) / (a^2 - 3*a) / ((a^2 + 9) / a + 6) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3431_343135


namespace NUMINAMATH_CALUDE_box_filling_theorem_l3431_343159

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  sorry

/-- The theorem stating that for a box with dimensions 36x45x18 inches, 
    the smallest number of identical cubes that can fill it is 40 -/
theorem box_filling_theorem : 
  let box : BoxDimensions := { length := 36, width := 45, depth := 18 }
  smallestNumberOfCubes box = 40 := by
  sorry

end NUMINAMATH_CALUDE_box_filling_theorem_l3431_343159


namespace NUMINAMATH_CALUDE_cos_3pi_plus_alpha_minus_sin_pi_plus_alpha_is_zero_l3431_343116

theorem cos_3pi_plus_alpha_minus_sin_pi_plus_alpha_is_zero (α k : ℝ) : 
  (∃ x y : ℝ, x = Real.tan α ∧ y = (Real.tan α)⁻¹ ∧ 
   x^2 - k*x + k^2 - 3 = 0 ∧ y^2 - k*y + k^2 - 3 = 0) →
  3*Real.pi < α ∧ α < (7/2)*Real.pi →
  Real.cos (3*Real.pi + α) - Real.sin (Real.pi + α) = 0 := by
sorry

end NUMINAMATH_CALUDE_cos_3pi_plus_alpha_minus_sin_pi_plus_alpha_is_zero_l3431_343116


namespace NUMINAMATH_CALUDE_number_of_sailors_l3431_343180

theorem number_of_sailors (W : ℝ) (n : ℕ) : 
  (W + 64 - 56) / n = W / n + 1 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_sailors_l3431_343180


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l3431_343130

theorem complex_sum_of_powers : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^12 + z₂^12 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l3431_343130


namespace NUMINAMATH_CALUDE_league_score_range_l3431_343178

/-- Represents a sports league -/
structure League where
  numTeams : ℕ
  pointsForWin : ℕ
  pointsForDraw : ℕ

/-- Calculate the total number of games in a double round-robin tournament -/
def totalGames (league : League) : ℕ :=
  league.numTeams * (league.numTeams - 1)

/-- Calculate the minimum possible total score for the league -/
def minTotalScore (league : League) : ℕ :=
  (totalGames league) * (2 * league.pointsForDraw)

/-- Calculate the maximum possible total score for the league -/
def maxTotalScore (league : League) : ℕ :=
  (totalGames league) * league.pointsForWin

/-- Theorem stating that the total score for a 15-team league with 3 points for a win
    and 1 point for a draw is between 420 and 630, inclusive -/
theorem league_score_range :
  let league := League.mk 15 3 1
  420 ≤ minTotalScore league ∧ maxTotalScore league ≤ 630 := by
  sorry

#eval minTotalScore (League.mk 15 3 1)
#eval maxTotalScore (League.mk 15 3 1)

end NUMINAMATH_CALUDE_league_score_range_l3431_343178


namespace NUMINAMATH_CALUDE_diagonals_bisect_in_rhombus_rectangle_square_l3431_343100

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the property of diagonals bisecting each other
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let d1_mid := ((q.vertices 0 + q.vertices 2) : ℝ × ℝ) / 2
  let d2_mid := ((q.vertices 1 + q.vertices 3) : ℝ × ℝ) / 2
  d1_mid = d2_mid

-- Define rhombus, rectangle, and square as specific types of quadrilaterals
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- State the theorem
theorem diagonals_bisect_in_rhombus_rectangle_square (q : Quadrilateral) :
  (is_rhombus q ∨ is_rectangle q ∨ is_square q) → diagonals_bisect q :=
by sorry

end NUMINAMATH_CALUDE_diagonals_bisect_in_rhombus_rectangle_square_l3431_343100


namespace NUMINAMATH_CALUDE_john_bought_490_packs_l3431_343119

/-- The number of packs John buys for each student -/
def packsPerStudent : ℕ := 4

/-- The number of extra packs John purchases for supplies -/
def extraPacks : ℕ := 10

/-- The number of students in each class -/
def studentsPerClass : List ℕ := [24, 18, 30, 20, 28]

/-- The total number of packs John bought -/
def totalPacks : ℕ := 
  (studentsPerClass.map (· * packsPerStudent)).sum + extraPacks

theorem john_bought_490_packs : totalPacks = 490 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_490_packs_l3431_343119


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3431_343189

/-- Given a parabola y = 2x² - bx + 3 with axis of symmetry x = 1, prove that b = 4 -/
theorem parabola_axis_of_symmetry (b : ℝ) : 
  (∀ x y : ℝ, y = 2*x^2 - b*x + 3) → 
  (1 = -b / (2*2)) → 
  b = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3431_343189


namespace NUMINAMATH_CALUDE_prob_less_than_8_ring_l3431_343153

def prob_10_ring : ℝ := 0.3
def prob_9_ring : ℝ := 0.3
def prob_8_ring : ℝ := 0.2

theorem prob_less_than_8_ring :
  1 - (prob_10_ring + prob_9_ring + prob_8_ring) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_8_ring_l3431_343153


namespace NUMINAMATH_CALUDE_combined_cost_apples_strawberries_l3431_343167

def total_cost : ℕ := 82
def banana_cost : ℕ := 12
def bread_cost : ℕ := 9
def milk_cost : ℕ := 7
def apple_cost : ℕ := 15
def orange_cost : ℕ := 13
def strawberry_cost : ℕ := 26

theorem combined_cost_apples_strawberries :
  apple_cost + strawberry_cost = 41 :=
by sorry

end NUMINAMATH_CALUDE_combined_cost_apples_strawberries_l3431_343167


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3431_343164

/-- The area of a triangle with vertices at (0, 0), (2, 2), and (4, 0) is 4 -/
theorem triangle_area : ℝ → Prop :=
  fun a =>
    let X : ℝ × ℝ := (0, 0)
    let Y : ℝ × ℝ := (2, 2)
    let Z : ℝ × ℝ := (4, 0)
    let base : ℝ := 4
    let height : ℝ := 2
    a = (1 / 2) * base * height ∧ a = 4

/-- The proof of the theorem -/
theorem triangle_area_proof : triangle_area 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3431_343164


namespace NUMINAMATH_CALUDE_matrix_addition_problem_l3431_343170

theorem matrix_addition_problem : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 0, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 2; 7, -10]
  A + B = !![-2, -1; 7, -5] := by
sorry

end NUMINAMATH_CALUDE_matrix_addition_problem_l3431_343170


namespace NUMINAMATH_CALUDE_system_solution_l3431_343175

theorem system_solution (a : ℕ+) 
  (h_system : ∃ (x y : ℝ), a * x + y = -4 ∧ 2 * x + y = -2 ∧ x < 0 ∧ y > 0) :
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3431_343175
