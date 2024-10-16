import Mathlib

namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4059_405910

theorem fraction_to_decimal : (7 : ℚ) / 16 = (4375 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4059_405910


namespace NUMINAMATH_CALUDE_defective_bulb_probability_l4059_405946

/-- The probability of randomly picking a defective bulb from a box with a given pass rate -/
theorem defective_bulb_probability (pass_rate : ℝ) (h : pass_rate = 0.875) :
  1 - pass_rate = 0.125 := by
  sorry

#check defective_bulb_probability

end NUMINAMATH_CALUDE_defective_bulb_probability_l4059_405946


namespace NUMINAMATH_CALUDE_special_square_area_l4059_405981

/-- A square with special points and segments -/
structure SpecialSquare where
  -- The side length of the square
  side : ℝ
  -- The distance BS
  bs : ℝ
  -- The distance PS
  ps : ℝ
  -- Assumption that BS = 8
  bs_eq : bs = 8
  -- Assumption that PS = 9
  ps_eq : ps = 9
  -- Assumption that BP and DQ intersect perpendicularly
  perpendicular : True

/-- The area of a SpecialSquare is 136 -/
theorem special_square_area (sq : SpecialSquare) : sq.side ^ 2 = 136 := by
  sorry

#check special_square_area

end NUMINAMATH_CALUDE_special_square_area_l4059_405981


namespace NUMINAMATH_CALUDE_ham_and_cake_probability_l4059_405937

/-- The probability of packing a ham sandwich and cake on the same day -/
def prob_ham_and_cake (total_days : ℕ) (ham_days : ℕ) (cake_days : ℕ) : ℚ :=
  (ham_days : ℚ) / total_days * (cake_days : ℚ) / total_days

theorem ham_and_cake_probability :
  let total_days : ℕ := 5
  let ham_days : ℕ := 3
  let cake_days : ℕ := 1
  prob_ham_and_cake total_days ham_days cake_days = 12 / 100 := by
sorry

end NUMINAMATH_CALUDE_ham_and_cake_probability_l4059_405937


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l4059_405902

theorem absolute_value_inequality (a b : ℝ) (ha : a ≠ 0) :
  ∃ (m : ℝ), m = 2 ∧ (∀ (a b : ℝ), a ≠ 0 → |a + b| + |a - b| ≥ m * |a|) ∧
  ∀ (m' : ℝ), (∀ (a b : ℝ), a ≠ 0 → |a + b| + |a - b| ≥ m' * |a|) → m' ≤ m :=
sorry

theorem solution_set (m : ℝ) (hm : m = 2) :
  {x : ℝ | |x - 1| + |x - 2| ≤ m} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 5/2} :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l4059_405902


namespace NUMINAMATH_CALUDE_periodic_functions_exist_l4059_405984

-- Define a type for periodic functions
def PeriodicFunction (p : ℝ) := { f : ℝ → ℝ // ∀ x, f (x + p) = f x }

-- Define a predicate for the smallest positive period
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) :=
  (∀ x, f (x + p) = f x) ∧ (∀ q, 0 < q → q < p → ∃ x, f (x + q) ≠ f x)

-- Main theorem
theorem periodic_functions_exist (p₁ p₂ : ℝ) (hp₁ : 0 < p₁) (hp₂ : 0 < p₂) :
  ∃ (f₁ f₂ : ℝ → ℝ),
    SmallestPositivePeriod f₁ p₁ ∧
    SmallestPositivePeriod f₂ p₂ ∧
    ∃ (p : ℝ), ∀ x, (f₁ - f₂) (x + p) = (f₁ - f₂) x :=
by
  sorry


end NUMINAMATH_CALUDE_periodic_functions_exist_l4059_405984


namespace NUMINAMATH_CALUDE_tan_sum_of_roots_l4059_405987

theorem tan_sum_of_roots (α β : ℝ) : 
  (∃ x y : ℝ, x^2 - 3*x + 2 = 0 ∧ y^2 - 3*y + 2 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  Real.tan (α + β) = -3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_of_roots_l4059_405987


namespace NUMINAMATH_CALUDE_moment_of_inertia_unit_mass_moment_of_inertia_arbitrary_mass_l4059_405929

/-- The moment of inertia of a system of points -/
noncomputable def moment_of_inertia {n : ℕ} (a : Fin n → Fin n → ℝ) (m : Fin n → ℝ) : ℝ :=
  let total_mass := (Finset.univ.sum m)
  (1 / total_mass) * (Finset.sum (Finset.univ.filter (λ i => i.val < n)) 
    (λ i => Finset.sum (Finset.univ.filter (λ j => i.val < j.val)) 
      (λ j => m i * m j * (a i j)^2)))

/-- Theorem: Moment of inertia for unit masses -/
theorem moment_of_inertia_unit_mass {n : ℕ} (a : Fin n → Fin n → ℝ) :
  moment_of_inertia a (λ _ => 1) = 
  (1 / n) * (Finset.sum (Finset.univ.filter (λ i => i.val < n)) 
    (λ i => Finset.sum (Finset.univ.filter (λ j => i.val < j.val)) 
      (λ j => (a i j)^2))) :=
sorry

/-- Theorem: Moment of inertia for arbitrary masses -/
theorem moment_of_inertia_arbitrary_mass {n : ℕ} (a : Fin n → Fin n → ℝ) (m : Fin n → ℝ) :
  moment_of_inertia a m = 
  (1 / (Finset.univ.sum m)) * (Finset.sum (Finset.univ.filter (λ i => i.val < n)) 
    (λ i => Finset.sum (Finset.univ.filter (λ j => i.val < j.val)) 
      (λ j => m i * m j * (a i j)^2))) :=
sorry

end NUMINAMATH_CALUDE_moment_of_inertia_unit_mass_moment_of_inertia_arbitrary_mass_l4059_405929


namespace NUMINAMATH_CALUDE_triangle_angle_relationship_l4059_405921

structure Triangle where
  A : Real
  median : Real → Real → Real
  angle_bisector : Real → Real → Real
  altitude : Real → Real → Real

def angle_between (f g : Real → Real → Real) : Real := sorry

theorem triangle_angle_relationship (t : Triangle) :
  let α := angle_between t.median t.angle_bisector
  let β := angle_between t.angle_bisector t.altitude
  (t.A < 90 → α < β) ∧
  (t.A > 90 → α > β) ∧
  (t.A = 90 → α = β) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relationship_l4059_405921


namespace NUMINAMATH_CALUDE_basket_capacity_l4059_405942

/-- The number of placards taken by each person -/
def placards_per_person : ℕ := 2

/-- The number of people who entered the stadium -/
def people_entered : ℕ := 2317

/-- The total number of placards taken -/
def total_placards : ℕ := people_entered * placards_per_person

theorem basket_capacity : total_placards = 4634 := by
  sorry

end NUMINAMATH_CALUDE_basket_capacity_l4059_405942


namespace NUMINAMATH_CALUDE_rectangle_y_value_l4059_405938

/-- A rectangle with vertices at (0, 0), (0, 5), (y, 5), and (y, 0) has an area of 35 square units. -/
def rectangle_area (y : ℝ) : Prop :=
  y > 0 ∧ y * 5 = 35

/-- The value of y for which the rectangle has an area of 35 square units is 7. -/
theorem rectangle_y_value : ∃ y : ℝ, rectangle_area y ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l4059_405938


namespace NUMINAMATH_CALUDE_barbed_wire_rate_problem_solution_l4059_405978

/-- Calculates the rate of drawing barbed wire per meter given the conditions of the problem. -/
theorem barbed_wire_rate (field_area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) : ℝ :=
  let side_length := Real.sqrt field_area
  let perimeter := 4 * side_length
  let wire_length := perimeter - (↑num_gates * gate_width)
  let rate_per_meter := total_cost / wire_length
  by
    -- Proof goes here
    sorry

/-- The rate of drawing barbed wire per meter for the given problem is 4.5 Rs/m. -/
theorem problem_solution : 
  barbed_wire_rate 3136 1 2 999 = 4.5 := by sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_problem_solution_l4059_405978


namespace NUMINAMATH_CALUDE_total_coins_proof_l4059_405907

theorem total_coins_proof (jayden_coins jasmine_coins : ℕ) 
  (h1 : jayden_coins = 300)
  (h2 : jasmine_coins = 335)
  (h3 : ∃ jason_coins : ℕ, jason_coins = jayden_coins + 60 ∧ jason_coins = jasmine_coins + 25) :
  ∃ total_coins : ℕ, total_coins = jayden_coins + jasmine_coins + (jayden_coins + 60) ∧ total_coins = 995 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_proof_l4059_405907


namespace NUMINAMATH_CALUDE_bug_flower_problem_l4059_405995

theorem bug_flower_problem (total_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) :
  total_bugs = 3 →
  total_flowers = 6 →
  total_flowers = total_bugs * flowers_per_bug →
  flowers_per_bug = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_bug_flower_problem_l4059_405995


namespace NUMINAMATH_CALUDE_handshake_count_l4059_405999

theorem handshake_count (n : ℕ) (h : n = 8) :
  let pairs := n / 2
  let handshakes_per_person := n - 2
  (n * handshakes_per_person) / 2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_handshake_count_l4059_405999


namespace NUMINAMATH_CALUDE_sqrt_180_simplification_l4059_405925

theorem sqrt_180_simplification : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_simplification_l4059_405925


namespace NUMINAMATH_CALUDE_function_passes_through_point_l4059_405927

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f := fun x => a^(x - 1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l4059_405927


namespace NUMINAMATH_CALUDE_periodic_even_function_value_l4059_405905

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_even_function_value 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h_even : is_even f)
  (h_periodic : has_period f 6)
  (h_interval : ∀ x ∈ Set.Icc (-3) 3, f x = (x + 1) * (x - a)) :
  f (-6) = -1 :=
sorry

end NUMINAMATH_CALUDE_periodic_even_function_value_l4059_405905


namespace NUMINAMATH_CALUDE_two_thousand_seventeenth_number_l4059_405928

def is_divisible_by_2_or_3 (n : ℕ) : Prop := 2 ∣ n ∨ 3 ∣ n

def sequence_2_or_3 : ℕ → ℕ := sorry

theorem two_thousand_seventeenth_number :
  sequence_2_or_3 2017 = 3026 := by sorry

end NUMINAMATH_CALUDE_two_thousand_seventeenth_number_l4059_405928


namespace NUMINAMATH_CALUDE_snake_eggs_problem_l4059_405975

theorem snake_eggs_problem (num_snakes : ℕ) (regular_price : ℕ) (total_revenue : ℕ) :
  num_snakes = 3 →
  regular_price = 250 →
  total_revenue = 2250 →
  ∃ (eggs_per_snake : ℕ),
    eggs_per_snake * num_snakes = (total_revenue - 4 * regular_price) / regular_price + 1 ∧
    eggs_per_snake = 2 :=
by sorry

end NUMINAMATH_CALUDE_snake_eggs_problem_l4059_405975


namespace NUMINAMATH_CALUDE_fixed_point_on_curve_circle_center_on_line_l4059_405998

-- Define the curve C
def C (a x y : ℝ) : Prop := x^2 + y^2 - 4*a*x + 2*a*y - 20 + 20*a = 0

-- Theorem 1: The point (4, -2) always lies on C for any value of a
theorem fixed_point_on_curve (a : ℝ) : C a 4 (-2) := by sorry

-- Theorem 2: When a ≠ 2, C is a circle and its center lies on the line x + 2y = 0
theorem circle_center_on_line (a : ℝ) (h : a ≠ 2) :
  ∃ (x y : ℝ), C a x y ∧ (∀ (x' y' : ℝ), C a x' y' → (x' - x)^2 + (y' - y)^2 = (x - x')^2 + (y - y')^2) ∧ x + 2*y = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_curve_circle_center_on_line_l4059_405998


namespace NUMINAMATH_CALUDE_brick_height_is_6cm_l4059_405957

/-- Proves that the height of each brick is 6 cm given the wall dimensions,
    known brick dimensions, and the number of bricks needed. -/
theorem brick_height_is_6cm
  (wall_length : ℝ) (wall_height : ℝ) (wall_width : ℝ)
  (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) :
  wall_length = 800 →
  wall_height = 600 →
  wall_width = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 6400 →
  ∃ (brick_height : ℝ),
    wall_length * wall_height * wall_width =
    (↑num_bricks : ℝ) * brick_length * brick_width * brick_height ∧
    brick_height = 6 :=
by sorry

end NUMINAMATH_CALUDE_brick_height_is_6cm_l4059_405957


namespace NUMINAMATH_CALUDE_f_properties_imply_a_equals_four_l4059_405992

/-- A function f(x) = x^2 - ax that is decreasing on (-∞, 2] and increasing on (2, +∞) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 - a*x

/-- The property of f being decreasing on (-∞, 2] -/
def decreasing_on_left (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 2 → f a x > f a y

/-- The property of f being increasing on (2, +∞) -/
def increasing_on_right (a : ℝ) : Prop :=
  ∀ x y, 2 < x → x < y → f a x < f a y

/-- Theorem stating that if f(x) = x^2 - ax is decreasing on (-∞, 2] and increasing on (2, +∞), then a = 4 -/
theorem f_properties_imply_a_equals_four :
  ∀ a : ℝ, decreasing_on_left a → increasing_on_right a → a = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_imply_a_equals_four_l4059_405992


namespace NUMINAMATH_CALUDE_existence_of_special_subset_l4059_405933

theorem existence_of_special_subset : 
  ∃ X : Set ℕ+, 
    ∀ n : ℕ+, ∃! (pair : ℕ+ × ℕ+), 
      pair.1 ∈ X ∧ pair.2 ∈ X ∧ n = pair.1 - pair.2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_subset_l4059_405933


namespace NUMINAMATH_CALUDE_longest_path_old_town_l4059_405969

structure OldTown where
  intersections : Nat
  start_color : Bool
  end_color : Bool

def longest_path (town : OldTown) : Nat :=
  sorry

theorem longest_path_old_town (town : OldTown) :
  town.intersections = 36 →
  town.start_color = town.end_color →
  longest_path town = 34 := by
  sorry

end NUMINAMATH_CALUDE_longest_path_old_town_l4059_405969


namespace NUMINAMATH_CALUDE_equal_money_days_l4059_405954

/-- The daily interest rate when leaving money with mother -/
def mother_rate : ℕ := 300

/-- The daily interest rate when leaving money with father -/
def father_rate : ℕ := 500

/-- The initial amount Kyu-won gave to her mother -/
def kyu_won_initial : ℕ := 8000

/-- The initial amount Seok-gi left with his father -/
def seok_gi_initial : ℕ := 5000

/-- The number of days needed for Kyu-won and Seok-gi to have the same amount of money -/
def days_needed : ℕ := 15

theorem equal_money_days :
  kyu_won_initial + mother_rate * days_needed = seok_gi_initial + father_rate * days_needed :=
by sorry

end NUMINAMATH_CALUDE_equal_money_days_l4059_405954


namespace NUMINAMATH_CALUDE_two_divisors_of_ten_billion_sum_to_157_l4059_405961

theorem two_divisors_of_ten_billion_sum_to_157 :
  ∃ (a b : ℕ), 
    a ≠ b ∧
    a > 0 ∧
    b > 0 ∧
    (10^10 % a = 0) ∧
    (10^10 % b = 0) ∧
    a + b = 157 ∧
    a = 32 ∧
    b = 125 := by
  sorry

end NUMINAMATH_CALUDE_two_divisors_of_ten_billion_sum_to_157_l4059_405961


namespace NUMINAMATH_CALUDE_square_of_one_plus_i_l4059_405908

theorem square_of_one_plus_i (i : ℂ) (h : i^2 = -1) : (1 + i)^2 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_square_of_one_plus_i_l4059_405908


namespace NUMINAMATH_CALUDE_middle_number_is_twelve_l4059_405994

/-- Given three distinct integers x, y, z satisfying the given conditions,
    prove that the middle number y equals 12. -/
theorem middle_number_is_twelve (x y z : ℤ)
  (h_distinct : x < y ∧ y < z)
  (h_sum1 : x + y = 21)
  (h_sum2 : x + z = 25)
  (h_sum3 : y + z = 28) :
  y = 12 := by sorry

end NUMINAMATH_CALUDE_middle_number_is_twelve_l4059_405994


namespace NUMINAMATH_CALUDE_squared_gt_not_sufficient_nor_necessary_for_cubed_gt_l4059_405980

theorem squared_gt_not_sufficient_nor_necessary_for_cubed_gt (a b : ℝ) :
  ¬(∀ a b : ℝ, a^2 > b^2 → a^3 > b^3) ∧ ¬(∀ a b : ℝ, a^3 > b^3 → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_squared_gt_not_sufficient_nor_necessary_for_cubed_gt_l4059_405980


namespace NUMINAMATH_CALUDE_student_distribution_l4059_405973

theorem student_distribution (total : ℕ) (a b : ℕ) : 
  total = 81 →
  a + b = total →
  a = b - 9 →
  a = 36 ∧ b = 45 := by
sorry

end NUMINAMATH_CALUDE_student_distribution_l4059_405973


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_extrema_l4059_405958

theorem sum_of_reciprocal_extrema (x y : ℝ) : 
  (4 * x^2 - 5 * x * y + 4 * y^2 = 5) → 
  let S := x^2 + y^2
  ∃ (S_max S_min : ℝ), 
    (∀ (x' y' : ℝ), (4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5) → x'^2 + y'^2 ≤ S_max) ∧
    (∀ (x' y' : ℝ), (4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5) → S_min ≤ x'^2 + y'^2) ∧
    (1 / S_max + 1 / S_min = 8 / 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_extrema_l4059_405958


namespace NUMINAMATH_CALUDE_food_bank_donations_boudin_del_monte_multiple_of_seven_l4059_405951

/-- Represents the total number of food items donated by five companies to a local food bank. -/
def total_donations (foster_farms : ℕ) : ℕ :=
  let american_summits := 2 * foster_farms
  let hormel := 3 * foster_farms
  let boudin_butchers := hormel / 3
  let del_monte := american_summits - 30
  foster_farms + american_summits + hormel + boudin_butchers + del_monte

/-- Theorem stating the total number of food items donated by the five companies. -/
theorem food_bank_donations : 
  total_donations 45 = 375 ∧ 
  (total_donations 45 - (45 + (2 * 45 - 30))) % 7 = 0 := by
  sorry

/-- Verification that the combined donations from Boudin Butchers and Del Monte Foods is a multiple of 7. -/
theorem boudin_del_monte_multiple_of_seven (foster_farms : ℕ) : 
  ((3 * foster_farms) / 3 + (2 * foster_farms - 30)) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_food_bank_donations_boudin_del_monte_multiple_of_seven_l4059_405951


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_k_eq_five_l4059_405979

/-- The equation 3(5+kx) = 15x + 15 has infinitely many solutions for x if and only if k = 5 -/
theorem infinite_solutions_iff_k_eq_five (k : ℝ) : 
  (∀ x : ℝ, 3 * (5 + k * x) = 15 * x + 15) ↔ k = 5 := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_k_eq_five_l4059_405979


namespace NUMINAMATH_CALUDE_inscribed_rhombus_square_area_l4059_405952

/-- Represents a rhombus inscribed in a square -/
structure InscribedRhombus where
  -- Square side length
  a : ℝ
  -- Distances from square vertices to rhombus vertices
  pb : ℝ
  bq : ℝ
  pr : ℝ
  qs : ℝ
  -- Conditions
  pb_positive : pb > 0
  bq_positive : bq > 0
  pr_positive : pr > 0
  qs_positive : qs > 0
  pb_plus_bq : pb + bq = a
  pr_plus_qs : pr + qs = a

/-- The area of the square given the inscribed rhombus properties -/
def square_area (r : InscribedRhombus) : ℝ :=
  r.a ^ 2

/-- Theorem: The area of the square with the given inscribed rhombus is 40000/58 -/
theorem inscribed_rhombus_square_area :
  ∀ r : InscribedRhombus,
  r.pb = 10 ∧ r.bq = 25 ∧ r.pr = 20 ∧ r.qs = 40 →
  square_area r = 40000 / 58 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_square_area_l4059_405952


namespace NUMINAMATH_CALUDE_intersection_midpoint_sum_l4059_405935

/-- Given a line y = ax + b that intersects y = x^2 at two distinct points,
    if the midpoint of these points is (5, 101), then a + b = -41 -/
theorem intersection_midpoint_sum (a b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    a * x₁ + b = x₁^2 ∧ 
    a * x₂ + b = x₂^2 ∧ 
    (x₁ + x₂) / 2 = 5 ∧ 
    (x₁^2 + x₂^2) / 2 = 101) →
  a + b = -41 := by
sorry

end NUMINAMATH_CALUDE_intersection_midpoint_sum_l4059_405935


namespace NUMINAMATH_CALUDE_expression_factorization_l4059_405917

theorem expression_factorization (a : ℝ) :
  (8 * a^4 + 92 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 5 * a^2 + 2) = 
  a^2 * (10 * a^2 + 89 * a - 10) - 1 := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l4059_405917


namespace NUMINAMATH_CALUDE_power_sum_of_i_l4059_405991

-- Define the imaginary unit i
axiom i : ℂ
axiom i_squared : i^2 = -1

-- State the theorem
theorem power_sum_of_i : i^2023 + i^303 = -2*i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l4059_405991


namespace NUMINAMATH_CALUDE_square_roots_problem_l4059_405965

theorem square_roots_problem (x : ℝ) :
  (∃ (a : ℝ), a > 0 ∧ (x + 1)^2 = a ∧ (x - 5)^2 = a) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l4059_405965


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4059_405989

theorem trigonometric_identity (x : ℝ) (h : Real.sin (x + π / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4059_405989


namespace NUMINAMATH_CALUDE_lillians_candies_l4059_405996

theorem lillians_candies (initial_candies final_candies : ℕ) 
  (h1 : initial_candies = 88)
  (h2 : final_candies = 93) :
  final_candies - initial_candies = 5 := by
  sorry

end NUMINAMATH_CALUDE_lillians_candies_l4059_405996


namespace NUMINAMATH_CALUDE_wreath_distribution_l4059_405988

/-- The number of wreaths each Greek initially had -/
def wreaths_per_greek (m : ℕ) : ℕ := 4 * m

/-- The number of Greeks -/
def num_greeks : ℕ := 3

/-- The number of Muses -/
def num_muses : ℕ := 9

/-- The total number of people (Greeks and Muses) -/
def total_people : ℕ := num_greeks + num_muses

theorem wreath_distribution (m : ℕ) (h : m > 0) :
  ∃ (initial_wreaths : ℕ),
    initial_wreaths = wreaths_per_greek m ∧
    (initial_wreaths * num_greeks) % total_people = 0 ∧
    ∀ (final_wreaths : ℕ),
      final_wreaths * total_people = initial_wreaths * num_greeks →
      final_wreaths = m :=
by sorry

end NUMINAMATH_CALUDE_wreath_distribution_l4059_405988


namespace NUMINAMATH_CALUDE_decimal_437_equals_fraction_l4059_405914

/-- The decimal representation of 0.4̄37 as a rational number -/
def decimal_437 : ℚ := 437/990 - 4/990

/-- The fraction 43693/99900 -/
def fraction_43693_99900 : ℚ := 43693/99900

theorem decimal_437_equals_fraction : 
  decimal_437 = fraction_43693_99900 ∧ 
  (∀ n d : ℕ, n ≠ 0 ∧ d ≠ 0 → fraction_43693_99900 = n / d → n = 43693 ∧ d = 99900) := by
  sorry

#check decimal_437_equals_fraction

end NUMINAMATH_CALUDE_decimal_437_equals_fraction_l4059_405914


namespace NUMINAMATH_CALUDE_count_divisible_by_eight_l4059_405930

theorem count_divisible_by_eight : ∃ n : ℕ, n = (Finset.filter (fun x => x % 8 = 0) (Finset.Icc 200 400)).card ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_eight_l4059_405930


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l4059_405971

theorem simplify_fraction_product : 
  (256 : ℚ) / 20 * (10 : ℚ) / 160 * ((16 : ℚ) / 6)^2 = (256 : ℚ) / 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l4059_405971


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l4059_405915

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- State the theorem
theorem f_monotone_decreasing (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is even
  ∀ x : ℝ, x > 0 → ∀ y : ℝ, y > x → f m y < f m x :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l4059_405915


namespace NUMINAMATH_CALUDE_winnie_balloon_distribution_l4059_405967

/-- The number of balloons Winnie keeps for herself when distributing balloons to friends -/
def balloons_kept (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem winnie_balloon_distribution :
  let total_balloons : ℕ := 20 + 40 + 70 + 90
  let num_friends : ℕ := 9
  balloons_kept total_balloons num_friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloon_distribution_l4059_405967


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l4059_405939

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 100)
  (h2 : final_price = 81)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) :
  x = 0.1 := by sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l4059_405939


namespace NUMINAMATH_CALUDE_sine_identity_l4059_405976

theorem sine_identity (α : Real) (h : α = π / 7) :
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_sine_identity_l4059_405976


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4059_405986

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a set of four points
def FourPoints := Fin 4 → Point3D

-- Define collinearity for three points
def collinear (p q r : Point3D) : Prop := sorry

-- Define coplanarity for four points
def coplanar (points : FourPoints) : Prop := sorry

-- No three points are collinear
def no_three_collinear (points : FourPoints) : Prop :=
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬collinear (points i) (points j) (points k)

theorem sufficient_but_not_necessary :
  (∀ (points : FourPoints), no_three_collinear points → ¬coplanar points) ∧
  (∃ (points : FourPoints), ¬coplanar points ∧ ¬no_three_collinear points) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4059_405986


namespace NUMINAMATH_CALUDE_smallest_n_with_unique_k_l4059_405960

theorem smallest_n_with_unique_k : ∃ (k : ℤ),
  (7 : ℚ) / 16 < (63 : ℚ) / (63 + k) ∧ (63 : ℚ) / (63 + k) < 9 / 20 ∧
  (∀ (k' : ℤ), k' ≠ k →
    ((7 : ℚ) / 16 ≥ (63 : ℚ) / (63 + k') ∨ (63 : ℚ) / (63 + k') ≥ 9 / 20)) ∧
  (∀ (n : ℕ), 0 < n → n < 63 →
    ¬(∃! (k : ℤ), (7 : ℚ) / 16 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 9 / 20)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_unique_k_l4059_405960


namespace NUMINAMATH_CALUDE_inequality_reversal_l4059_405941

theorem inequality_reversal (a b c : ℝ) (h1 : a < b) (h2 : c < 0) : ¬(a * c < b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l4059_405941


namespace NUMINAMATH_CALUDE_joes_lift_l4059_405972

theorem joes_lift (total : ℕ) (diff : ℕ) (first_lift : ℕ) (second_lift : ℕ) 
  (h1 : total = 600)
  (h2 : first_lift + second_lift = total)
  (h3 : 2 * first_lift = second_lift + diff)
  (h4 : diff = 300) : 
  first_lift = 300 := by
sorry

end NUMINAMATH_CALUDE_joes_lift_l4059_405972


namespace NUMINAMATH_CALUDE_expansion_coefficient_l4059_405964

/-- The coefficient of x^(3/2) in the expansion of (√x - a/√x)^5 -/
def coefficient_x_3_2 (a : ℝ) : ℝ := 
  (5 : ℝ) * (-a)

theorem expansion_coefficient (a : ℝ) : 
  coefficient_x_3_2 a = 30 → a = -6 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l4059_405964


namespace NUMINAMATH_CALUDE_root_in_interval_l4059_405983

-- Define the function f(x) = x^2 + 12x - 15
def f (x : ℝ) : ℝ := x^2 + 12*x - 15

-- State the theorem
theorem root_in_interval :
  (f 1.1 < 0) → (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l4059_405983


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_5_l4059_405966

theorem tan_alpha_2_implies_expression_5 (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_5_l4059_405966


namespace NUMINAMATH_CALUDE_division_remainder_problem_l4059_405968

theorem division_remainder_problem (a b : ℕ) 
  (h1 : a - b = 1390)
  (h2 : a = 1650)
  (h3 : a / b = 6) :
  a % b = 90 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l4059_405968


namespace NUMINAMATH_CALUDE_clay_molding_minimum_operations_l4059_405904

/-- Represents a clay molding operation -/
structure ClayOperation where
  groups : List (List Nat)
  deriving Repr

/-- The result of applying a clay molding operation -/
def applyOperation (pieces : List Nat) (op : ClayOperation) : List Nat :=
  sorry

/-- Checks if all elements in a list are distinct -/
def allDistinct (l : List Nat) : Prop :=
  sorry

/-- The main theorem stating that 2 operations are sufficient and minimal -/
theorem clay_molding_minimum_operations :
  ∃ (op1 op2 : ClayOperation),
    let initial_pieces := List.replicate 111 1
    let after_op1 := applyOperation initial_pieces op1
    let final_pieces := applyOperation after_op1 op2
    (final_pieces.length = 11) ∧
    (allDistinct final_pieces) ∧
    (∀ (op1' op2' : ClayOperation),
      let after_op1' := applyOperation initial_pieces op1'
      let final_pieces' := applyOperation after_op1' op2'
      (final_pieces'.length = 11 ∧ allDistinct final_pieces') →
      ¬∃ (single_op : ClayOperation),
        let result := applyOperation initial_pieces single_op
        (result.length = 11 ∧ allDistinct result)) :=
  sorry

end NUMINAMATH_CALUDE_clay_molding_minimum_operations_l4059_405904


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l4059_405956

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  2 * x + 1 / (x + 1) ≥ 2 * Real.sqrt 2 - 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > -1) :
  2 * x + 1 / (x + 1) = 2 * Real.sqrt 2 - 2 ↔ x = Real.sqrt 2 / 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l4059_405956


namespace NUMINAMATH_CALUDE_N_is_composite_l4059_405913

/-- The number formed by 2n ones -/
def N (n : ℕ) : ℕ := (10^(2*n) - 1) / 9

/-- Theorem: For all natural numbers n ≥ 1, N(n) is composite -/
theorem N_is_composite (n : ℕ) (h : n ≥ 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N n = a * b := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l4059_405913


namespace NUMINAMATH_CALUDE_a_squared_coefficient_zero_l4059_405970

theorem a_squared_coefficient_zero (p : ℚ) : 
  (∀ a : ℚ, (a^2 - p*a + 6) * (2*a - 1) = (-p*a + 6) * (2*a - 1)) → p = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_a_squared_coefficient_zero_l4059_405970


namespace NUMINAMATH_CALUDE_line_vector_proof_l4059_405922

def line_vector (t : ℝ) : ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 1 = (2, 5) ∧ line_vector 4 = (8, -7)) →
  line_vector (-3) = (-6, 21) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_proof_l4059_405922


namespace NUMINAMATH_CALUDE_base8_cube_c_is_zero_l4059_405911

/-- Represents a number in base 8 of the form 4c3 --/
def base8Number (c : ℕ) : ℕ := 4 * 8^2 + c * 8 + 3

/-- Checks if a number is a perfect cube --/
def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem base8_cube_c_is_zero :
  ∃ c : ℕ, isPerfectCube (base8Number c) → c = 0 :=
sorry

end NUMINAMATH_CALUDE_base8_cube_c_is_zero_l4059_405911


namespace NUMINAMATH_CALUDE_lap_time_calculation_l4059_405990

/-- Represents the field and boy's running conditions -/
structure FieldConditions where
  side_length : ℝ
  normal_speed : ℝ
  sandy_length : ℝ
  sandy_speed_reduction : ℝ
  hurdle_count_low : ℕ
  hurdle_count_high : ℕ
  hurdle_time_low : ℝ
  hurdle_time_high : ℝ
  corner_slowdown : ℝ

/-- Calculates the total time to complete one lap around the field -/
def total_lap_time (conditions : FieldConditions) : ℝ :=
  sorry

/-- Theorem stating the total time to complete one lap -/
theorem lap_time_calculation (conditions : FieldConditions) 
  (h1 : conditions.side_length = 50)
  (h2 : conditions.normal_speed = 9 * 1000 / 3600)
  (h3 : conditions.sandy_length = 20)
  (h4 : conditions.sandy_speed_reduction = 0.25)
  (h5 : conditions.hurdle_count_low = 2)
  (h6 : conditions.hurdle_count_high = 2)
  (h7 : conditions.hurdle_time_low = 2)
  (h8 : conditions.hurdle_time_high = 3)
  (h9 : conditions.corner_slowdown = 2) :
  total_lap_time conditions = 138.68 := by
  sorry

end NUMINAMATH_CALUDE_lap_time_calculation_l4059_405990


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_square_property_l4059_405936

theorem quadratic_inequality_and_square_property : 
  (¬∃ x : ℝ, x^2 - x + 2 < 0) ∧ (∀ x ∈ Set.Icc 1 2, x^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_square_property_l4059_405936


namespace NUMINAMATH_CALUDE_airport_distance_proof_l4059_405906

/-- The distance from Victor's home to the airport -/
def airport_distance : ℝ := 150

/-- Victor's initial speed -/
def initial_speed : ℝ := 60

/-- Victor's increased speed -/
def increased_speed : ℝ := 80

/-- Time Victor drives at initial speed -/
def initial_drive_time : ℝ := 0.5

/-- Time difference if Victor continued at initial speed -/
def late_time : ℝ := 0.25

/-- Time difference after increasing speed -/
def early_time : ℝ := 0.25

theorem airport_distance_proof :
  ∃ (planned_time : ℝ),
    -- Distance covered at initial speed
    initial_speed * initial_drive_time +
    -- Remaining distance if continued at initial speed
    initial_speed * (planned_time + late_time) =
    -- Distance covered at initial speed
    initial_speed * initial_drive_time +
    -- Remaining distance at increased speed
    increased_speed * (planned_time - early_time) ∧
    -- Total distance equals airport_distance
    airport_distance = initial_speed * initial_drive_time +
                       increased_speed * (planned_time - early_time) := by
  sorry

end NUMINAMATH_CALUDE_airport_distance_proof_l4059_405906


namespace NUMINAMATH_CALUDE_find_divisor_l4059_405940

theorem find_divisor (divisor : ℕ) : divisor = 2 := by
  have h1 : 2 = (433126 : ℕ) - 433124 := by sorry
  have h2 : (433126 : ℕ) % divisor = 0 := by sorry
  have h3 : ∀ n : ℕ, n < 2 → (433124 + n) % divisor ≠ 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_find_divisor_l4059_405940


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l4059_405903

theorem sum_of_reciprocals_squared (a b : ℝ) (h : a > 0) (k : b > 0) (hab : a * b = 1) :
  1 / (1 + a^2) + 1 / (1 + b^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l4059_405903


namespace NUMINAMATH_CALUDE_aluminum_foil_thickness_thickness_satisfies_density_l4059_405932

/-- The thickness of a rectangular piece of aluminum foil -/
noncomputable def thickness (d m l w : ℝ) : ℝ := m / (d * l * w)

/-- The volume of a rectangular piece of aluminum foil -/
noncomputable def volume (l w t : ℝ) : ℝ := l * w * t

/-- Theorem: The thickness of a rectangular piece of aluminum foil
    is equal to its mass divided by the product of its density, length, and width -/
theorem aluminum_foil_thickness (d m l w : ℝ) (hd : d > 0) (hl : l > 0) (hw : w > 0) :
  thickness d m l w = m / (d * l * w) :=
by sorry

/-- Theorem: The thickness formula satisfies the density definition -/
theorem thickness_satisfies_density (d m l w : ℝ) (hd : d > 0) (hl : l > 0) (hw : w > 0) :
  d = m / volume l w (thickness d m l w) :=
by sorry

end NUMINAMATH_CALUDE_aluminum_foil_thickness_thickness_satisfies_density_l4059_405932


namespace NUMINAMATH_CALUDE_odd_root_symmetry_l4059_405909

theorem odd_root_symmetry (x : ℝ) (n : ℕ) : 
  (x ^ (1 / (2 * n + 1 : ℝ))) = -((-x) ^ (1 / (2 * n + 1 : ℝ))) := by
  sorry

end NUMINAMATH_CALUDE_odd_root_symmetry_l4059_405909


namespace NUMINAMATH_CALUDE_picture_area_l4059_405923

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h_frame_area : (2 * x + 5) * (y + 4) = 60) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l4059_405923


namespace NUMINAMATH_CALUDE_least_common_solution_l4059_405963

theorem least_common_solution : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 8 = 7 ∧ 
  x % 7 = 6 ∧
  (∀ y : ℕ, y > 0 ∧ y % 6 = 5 ∧ y % 8 = 7 ∧ y % 7 = 6 → x ≤ y) ∧
  x = 167 := by
sorry

end NUMINAMATH_CALUDE_least_common_solution_l4059_405963


namespace NUMINAMATH_CALUDE_range_of_function_l4059_405962

theorem range_of_function (x : ℝ) (h : x^2 ≥ 1) :
  x^2 + Real.sqrt (x^2 - 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l4059_405962


namespace NUMINAMATH_CALUDE_sqrt_two_minus_x_real_range_l4059_405985

theorem sqrt_two_minus_x_real_range :
  {x : ℝ | ∃ y : ℝ, y ^ 2 = 2 - x} = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_x_real_range_l4059_405985


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l4059_405997

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 3 ∧ (2 - x) / (x - 3) + 1 / (3 - x) = 1 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l4059_405997


namespace NUMINAMATH_CALUDE_musician_earnings_per_song_l4059_405945

/-- Represents the earnings of a musician over a period of time --/
structure MusicianEarnings where
  songs_per_month : ℕ
  total_earnings : ℕ
  years : ℕ

/-- Calculates the earnings per song for a musician --/
def earnings_per_song (m : MusicianEarnings) : ℚ :=
  m.total_earnings / (m.songs_per_month * 12 * m.years)

/-- Theorem: A musician releasing 3 songs per month and earning $216,000 in 3 years makes $2,000 per song --/
theorem musician_earnings_per_song :
  let m : MusicianEarnings := { songs_per_month := 3, total_earnings := 216000, years := 3 }
  earnings_per_song m = 2000 := by
  sorry


end NUMINAMATH_CALUDE_musician_earnings_per_song_l4059_405945


namespace NUMINAMATH_CALUDE_tan_half_product_squared_l4059_405900

theorem tan_half_product_squared (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) : 
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 26 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_squared_l4059_405900


namespace NUMINAMATH_CALUDE_unique_quadrilateral_from_centers_l4059_405993

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Checks if a circle can be inscribed in a quadrilateral -/
def hasInscribedCircle (q : Quadrilateral) : Prop := sorry

/-- Checks if a circle can be circumscribed around a quadrilateral -/
def hasCircumscribedCircle (q : Quadrilateral) : Prop := sorry

/-- Gets the center of the inscribed circle of a quadrilateral -/
def getInscribedCenter (q : Quadrilateral) : Point2D := sorry

/-- Gets the center of the circumscribed circle of a quadrilateral -/
def getCircumscribedCenter (q : Quadrilateral) : Point2D := sorry

/-- Gets the intersection point of lines connecting midpoints of opposite sides -/
def getMidpointIntersection (q : Quadrilateral) : Point2D := sorry

/-- Theorem: A unique quadrilateral can be determined from its inscribed circle center,
    circumscribed circle center, and the intersection of midpoint lines -/
theorem unique_quadrilateral_from_centers
  (I O M : Point2D) :
  ∃! q : Quadrilateral,
    hasInscribedCircle q ∧
    hasCircumscribedCircle q ∧
    getInscribedCenter q = I ∧
    getCircumscribedCenter q = O ∧
    getMidpointIntersection q = M :=
  sorry

end NUMINAMATH_CALUDE_unique_quadrilateral_from_centers_l4059_405993


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l4059_405920

theorem roots_sum_and_product (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l4059_405920


namespace NUMINAMATH_CALUDE_rational_pair_sum_reciprocal_natural_l4059_405953

theorem rational_pair_sum_reciprocal_natural (x y : ℚ) :
  (∃ (m n : ℕ), x + 1 / y = m ∧ y + 1 / x = n) →
  ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_rational_pair_sum_reciprocal_natural_l4059_405953


namespace NUMINAMATH_CALUDE_angle_trisector_theorem_l4059_405926

/-- 
Given a triangle ABC with angle γ = ∠ACB, if the trisectors of γ divide 
the opposite side AB into segments d, e, f, then cos²(γ/3) = ((d+e)(e+f))/(4df)
-/
theorem angle_trisector_theorem (d e f : ℝ) (γ : ℝ) 
  (h1 : d > 0) (h2 : e > 0) (h3 : f > 0) (h4 : γ > 0) (h5 : γ < π) :
  (Real.cos (γ / 3))^2 = ((d + e) * (e + f)) / (4 * d * f) :=
sorry

end NUMINAMATH_CALUDE_angle_trisector_theorem_l4059_405926


namespace NUMINAMATH_CALUDE_andre_flowers_l4059_405919

/-- The number of flowers Andre gave to Rosa -/
def flowers_given : ℕ := 90 - 67

/-- Rosa's initial number of flowers -/
def initial_flowers : ℕ := 67

/-- Rosa's final number of flowers -/
def final_flowers : ℕ := 90

theorem andre_flowers : flowers_given = final_flowers - initial_flowers := by
  sorry

end NUMINAMATH_CALUDE_andre_flowers_l4059_405919


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4059_405944

/-- Given a geometric sequence {a_n} with positive terms, where a_1, (1/2)a_3, 2a_2 form an arithmetic sequence,
    prove that (a_8 + a_9) / (a_6 + a_7) = 3 + 2√2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : ∃ d : ℝ, a 1 + d = (1/2) * a 3 ∧ (1/2) * a 3 + d = 2 * a 2) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4059_405944


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l4059_405950

open Real

theorem inequality_implies_upper_bound (a : ℝ) : 
  (∀ x > 0, 2 * x * log x ≥ -x^2 + a*x - 3) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l4059_405950


namespace NUMINAMATH_CALUDE_valentines_given_away_l4059_405959

/-- Given Mrs. Franklin's initial and remaining Valentines, calculate how many she gave away. -/
theorem valentines_given_away
  (initial : ℝ)
  (remaining : ℝ)
  (h_initial : initial = 58.5)
  (h_remaining : remaining = 16.25) :
  initial - remaining = 42.25 := by
  sorry

end NUMINAMATH_CALUDE_valentines_given_away_l4059_405959


namespace NUMINAMATH_CALUDE_percent_relation_l4059_405949

/-- Given that c is 25% of a and 10% of b, prove that b is 250% of a. -/
theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.25 * a) 
  (h2 : c = 0.10 * b) : 
  b = 2.5 * a := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l4059_405949


namespace NUMINAMATH_CALUDE_baseton_transaction_baseton_base_equation_baseton_base_value_l4059_405947

/-- The base of the number system in Baseton -/
def r : ℕ := sorry

/-- The cost of the laptop in base r -/
def laptop_cost : ℕ := 534

/-- The amount paid in base r -/
def amount_paid : ℕ := 1000

/-- The change received in base r -/
def change_received : ℕ := 366

/-- Conversion from base r to base 10 -/
def to_base_10 (n : ℕ) : ℕ := 
  (n / 100) * r^2 + ((n / 10) % 10) * r + (n % 10)

theorem baseton_transaction :
  to_base_10 laptop_cost + to_base_10 change_received = to_base_10 amount_paid :=
by sorry

theorem baseton_base_equation :
  r^3 - 8*r^2 - 9*r - 10 = 0 :=
by sorry

theorem baseton_base_value : r = 10 :=
by sorry

end NUMINAMATH_CALUDE_baseton_transaction_baseton_base_equation_baseton_base_value_l4059_405947


namespace NUMINAMATH_CALUDE_ellipse_properties_l4059_405955

/-- Properties of an ellipse with equation x²/4 + y²/2 = 1 -/
theorem ellipse_properties :
  let a := 2  -- semi-major axis
  let b := Real.sqrt 2  -- semi-minor axis
  let c := Real.sqrt (a^2 - b^2)  -- focal distance / 2
  let e := c / a  -- eccentricity
  (∀ x y, x^2/4 + y^2/2 = 1 →
    (2*a = 4 ∧  -- length of major axis
     2*c = 2*Real.sqrt 2 ∧  -- focal distance
     e = Real.sqrt 2 / 2))  -- eccentricity
  := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4059_405955


namespace NUMINAMATH_CALUDE_class_representatives_count_l4059_405931

/-- The number of ways to select and arrange class representatives -/
def class_representatives (num_boys num_girls num_subjects : ℕ) : ℕ :=
  (num_boys.choose 1) * (num_girls.choose 2) * (num_subjects.factorial)

/-- Theorem: The number of ways to select 2 girls from 3 girls, 1 boy from 3 boys,
    and arrange them as representatives for 3 subjects is 54 -/
theorem class_representatives_count :
  class_representatives 3 3 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_class_representatives_count_l4059_405931


namespace NUMINAMATH_CALUDE_complex_value_calculation_l4059_405901

theorem complex_value_calculation : Complex.I - (1 / Complex.I) = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_value_calculation_l4059_405901


namespace NUMINAMATH_CALUDE_geometry_statements_l4059_405974

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)

variable (m n : Line)
variable (α β : Plane)

-- Assume m and n are distinct, α and β are different
variable (h_distinct_lines : m ≠ n)
variable (h_different_planes : α ≠ β)

theorem geometry_statements :
  (parallel_line_plane m α ∧ perpendicular_line_plane n β ∧ parallel_lines m n → perpendicular_planes α β) ∧
  (perpendicular_line_plane m α ∧ parallel_lines m n → perpendicular_line_plane n α) ∧
  ¬(perpendicular_lines m n ∧ line_in_plane n α ∧ line_in_plane m β → perpendicular_planes α β) ∧
  (parallel_line_plane m β ∧ line_in_plane m α ∧ plane_intersection α β = n → parallel_lines m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_statements_l4059_405974


namespace NUMINAMATH_CALUDE_odd_function_when_c_zero_increasing_when_b_zero_central_symmetry_more_than_two_roots_possible_l4059_405948

-- Define the function f
def f (b c x : ℝ) : ℝ := x * |x| + b * x + c

-- Statement 1
theorem odd_function_when_c_zero (b : ℝ) :
  (∀ x : ℝ, f b 0 (-x) = -(f b 0 x)) := by sorry

-- Statement 2
theorem increasing_when_b_zero (c : ℝ) :
  Monotone (f 0 c) := by sorry

-- Statement 3
theorem central_symmetry (b c : ℝ) :
  ∀ x : ℝ, f b c (-x) + f b c x = 2 * c := by sorry

-- Statement 4 (negation of the original statement)
theorem more_than_two_roots_possible :
  ∃ b c : ℝ, ∃ x₁ x₂ x₃ : ℝ, (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
  (f b c x₁ = 0 ∧ f b c x₂ = 0 ∧ f b c x₃ = 0) := by sorry

end NUMINAMATH_CALUDE_odd_function_when_c_zero_increasing_when_b_zero_central_symmetry_more_than_two_roots_possible_l4059_405948


namespace NUMINAMATH_CALUDE_intersection_A_B_l4059_405924

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - 2*x ≤ 0}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4059_405924


namespace NUMINAMATH_CALUDE_parabola_directrix_l4059_405943

/-- The directrix of the parabola y = -1/4 * x^2 is y = 1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = -1/4 * x^2 → (∃ (d : ℝ), d = 1 ∧ 
    ∀ (p : ℝ × ℝ), p.2 = -1/4 * p.1^2 → 
      ∃ (f : ℝ), (p.1 - 0)^2 + (p.2 - f)^2 = (p.2 - d)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4059_405943


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_three_l4059_405912

theorem cubic_fraction_equals_three (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^3 + b^3 + c^3) / (a * b * c * (a * b + a * c + b * c)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_three_l4059_405912


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l4059_405918

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*k*x + 3*k^2 + 2*k = 0) ↔ -1 ≤ k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l4059_405918


namespace NUMINAMATH_CALUDE_jessica_exam_progress_l4059_405916

/-- Represents the exam parameters and Jessica's progress -/
structure ExamProgress where
  total_time : ℕ  -- Total time for the exam in minutes
  total_questions : ℕ  -- Total number of questions in the exam
  time_used : ℕ  -- Time used so far in minutes
  time_remaining : ℕ  -- Time remaining when exam is finished

/-- Represents that it's impossible to determine the exact number of questions answered -/
def cannot_determine_questions_answered (ep : ExamProgress) : Prop :=
  ∀ (questions_answered : ℕ), 
    questions_answered ≤ ep.total_questions → 
    ∃ (other_answered : ℕ), 
      other_answered ≠ questions_answered ∧ 
      other_answered ≤ ep.total_questions

/-- Theorem stating that given the exam conditions, it's impossible to determine
    the exact number of questions Jessica has answered so far -/
theorem jessica_exam_progress : 
  let ep : ExamProgress := {
    total_time := 60,
    total_questions := 80,
    time_used := 12,
    time_remaining := 0
  }
  cannot_determine_questions_answered ep :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_exam_progress_l4059_405916


namespace NUMINAMATH_CALUDE_expression_value_l4059_405934

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (sum_zero : x + y + z = 0) (sum_prod_nonzero : x*y + x*z + y*z ≠ 0) :
  (x^5 + y^5 + z^5) / (x*y*z * (x*y + x*z + y*z)) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4059_405934


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l4059_405982

/-- Represents the selling price of the product -/
def selling_price : ℝ → ℝ := id

/-- Represents the purchase cost of the product -/
def purchase_cost : ℝ := 40

/-- Represents the number of units sold at a given price -/
def units_sold (x : ℝ) : ℝ := 500 - 20 * (x - 50)

/-- Represents the profit at a given selling price -/
def profit (x : ℝ) : ℝ := (x - purchase_cost) * (units_sold x)

/-- Theorem stating that the profit-maximizing selling price is 57.5 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), ∀ (y : ℝ), profit x ≥ profit y ∧ x = 57.5 := by sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l4059_405982


namespace NUMINAMATH_CALUDE_greatest_square_with_nine_factors_l4059_405977

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem greatest_square_with_nine_factors :
  ∃ n : ℕ, n = 196 ∧
    n < 200 ∧
    is_perfect_square n ∧
    count_factors n = 9 ∧
    ∀ m : ℕ, m < 200 → is_perfect_square m → count_factors m = 9 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_square_with_nine_factors_l4059_405977
