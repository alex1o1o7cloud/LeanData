import Mathlib

namespace NUMINAMATH_CALUDE_cost_of_cherries_l3135_313595

/-- Given Sally's purchase of peaches and cherries, prove the cost of cherries. -/
theorem cost_of_cherries
  (peaches_after_coupon : ℝ)
  (coupon_value : ℝ)
  (total_cost : ℝ)
  (h1 : peaches_after_coupon = 12.32)
  (h2 : coupon_value = 3)
  (h3 : total_cost = 23.86) :
  total_cost - (peaches_after_coupon + coupon_value) = 8.54 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_cherries_l3135_313595


namespace NUMINAMATH_CALUDE_santa_candy_problem_l3135_313576

theorem santa_candy_problem (total : ℕ) (chocolate : ℕ) (gummy : ℕ) :
  total = 2023 →
  chocolate + gummy = total →
  chocolate = (75 * gummy) / 100 →
  chocolate = 867 := by
sorry

end NUMINAMATH_CALUDE_santa_candy_problem_l3135_313576


namespace NUMINAMATH_CALUDE_perfect_square_sum_l3135_313560

theorem perfect_square_sum : ∃ k : ℕ, 2^8 + 2^11 + 2^12 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l3135_313560


namespace NUMINAMATH_CALUDE_cylinder_volume_l3135_313590

theorem cylinder_volume (r : ℝ) (h : ℝ) : 
  r > 0 → h > 0 → h = 2 * r → 2 * π * r * h = π → π * r^2 * h = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3135_313590


namespace NUMINAMATH_CALUDE_license_plate_count_l3135_313513

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in the license plate -/
def num_plate_digits : ℕ := 5

/-- The number of letters in the license plate -/
def num_plate_letters : ℕ := 2

/-- The number of possible positions for the letter block (start or end) -/
def num_letter_positions : ℕ := 2

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  num_digits ^ num_plate_digits * 
  num_letters ^ num_plate_letters * 
  num_letter_positions

theorem license_plate_count : total_license_plates = 2704000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3135_313513


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_1360_l3135_313553

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x + 4
def g (x : ℝ) : ℝ := x^2 + 5 * x + 3

-- State the theorem
theorem f_g_f_3_equals_1360 : f (g (f 3)) = 1360 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_1360_l3135_313553


namespace NUMINAMATH_CALUDE_second_discarded_number_l3135_313592

theorem second_discarded_number 
  (n₁ : ℕ) (a₁ : ℚ) (n₂ : ℕ) (a₂ : ℚ) (x₁ : ℚ) :
  n₁ = 50 →
  a₁ = 38 →
  n₂ = 48 →
  a₂ = 37.5 →
  x₁ = 45 →
  ∃ x₂ : ℚ, x₂ = 55 ∧ n₁ * a₁ = n₂ * a₂ + x₁ + x₂ :=
by sorry

end NUMINAMATH_CALUDE_second_discarded_number_l3135_313592


namespace NUMINAMATH_CALUDE_friday_attendance_l3135_313535

/-- Calculates the percentage of students present on a given day -/
def students_present (initial_absenteeism : ℝ) (daily_increase : ℝ) (day : ℕ) : ℝ :=
  100 - (initial_absenteeism + daily_increase * day)

/-- Proves that the percentage of students present on Friday is 78% -/
theorem friday_attendance 
  (initial_absenteeism : ℝ) 
  (daily_increase : ℝ) 
  (h1 : initial_absenteeism = 14) 
  (h2 : daily_increase = 2) : 
  students_present initial_absenteeism daily_increase 4 = 78 := by
  sorry

#eval students_present 14 2 4

end NUMINAMATH_CALUDE_friday_attendance_l3135_313535


namespace NUMINAMATH_CALUDE_inequality_proof_l3135_313534

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3135_313534


namespace NUMINAMATH_CALUDE_banana_bread_loaves_l3135_313547

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The total number of bananas used for both days -/
def total_bananas : ℕ := 36

/-- The number of loaves made on Monday -/
def monday_loaves : ℕ := 3

theorem banana_bread_loaves :
  monday_loaves * bananas_per_loaf + 2 * monday_loaves * bananas_per_loaf = total_bananas :=
sorry

end NUMINAMATH_CALUDE_banana_bread_loaves_l3135_313547


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3135_313544

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ + 3 = 0 → x₂^2 - 4*x₂ + 3 = 0 → x₁ + x₂ - 2*x₁*x₂ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3135_313544


namespace NUMINAMATH_CALUDE_sum_a_c_l3135_313579

theorem sum_a_c (a b c d : ℝ) 
  (h1 : a * b + a * c + b * c + b * d + c * d + a * d = 40)
  (h2 : b^2 + d^2 = 29) : 
  a + c = 42 / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_a_c_l3135_313579


namespace NUMINAMATH_CALUDE_price_change_effect_l3135_313570

theorem price_change_effect (a : ℝ) (h : a > 0) : a * 1.02 * 0.98 < a := by
  sorry

end NUMINAMATH_CALUDE_price_change_effect_l3135_313570


namespace NUMINAMATH_CALUDE_batsman_average_l3135_313583

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : ℕ
  lastScore : ℕ
  averageIncrease : ℕ
  neverNotOut : Prop

/-- Calculates the average score after the latest innings -/
def averageAfterLatestInnings (b : Batsman) : ℕ :=
  sorry

/-- Theorem: Given the conditions, the batsman's average after the 12th innings is 37 runs -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.lastScore = 70)
  (h3 : b.averageIncrease = 3)
  (h4 : b.neverNotOut) :
  averageAfterLatestInnings b = 37 :=
sorry

end NUMINAMATH_CALUDE_batsman_average_l3135_313583


namespace NUMINAMATH_CALUDE_alpha_set_property_l3135_313505

theorem alpha_set_property (r s : ℕ) (hr : r > s) (hgcd : Nat.gcd r s = 1) :
  let α : ℚ := r / s
  let N_α : Set ℕ := {m | ∃ n : ℕ, m = ⌊n * α⌋}
  ∀ m ∈ N_α, ¬(r ∣ (m + 1)) := by
  sorry

end NUMINAMATH_CALUDE_alpha_set_property_l3135_313505


namespace NUMINAMATH_CALUDE_translation_coordinates_l3135_313509

/-- Given a point A(-1, 2) in the Cartesian coordinate system,
    translated 4 units to the right and 2 units down to obtain point A₁,
    the coordinates of A₁ are (3, 0). -/
theorem translation_coordinates :
  let A : ℝ × ℝ := (-1, 2)
  let right_translation : ℝ := 4
  let down_translation : ℝ := 2
  let A₁ : ℝ × ℝ := (A.1 + right_translation, A.2 - down_translation)
  A₁ = (3, 0) := by
sorry

end NUMINAMATH_CALUDE_translation_coordinates_l3135_313509


namespace NUMINAMATH_CALUDE_reservoir_capacity_difference_l3135_313533

/-- Proves that the difference between total capacity and normal level is 25 million gallons --/
theorem reservoir_capacity_difference (current_amount : ℝ) (normal_level : ℝ) (total_capacity : ℝ)
  (h1 : current_amount = 30)
  (h2 : current_amount = 2 * normal_level)
  (h3 : current_amount = 0.75 * total_capacity) :
  total_capacity - normal_level = 25 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_difference_l3135_313533


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3135_313539

theorem cubic_root_sum (α β γ : ℂ) : 
  α^3 - α - 1 = 0 → β^3 - β - 1 = 0 → γ^3 - γ - 1 = 0 →
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3135_313539


namespace NUMINAMATH_CALUDE_present_giving_property_l3135_313548

/-- Represents a child in the class -/
structure Child where
  id : Nat

/-- Represents a triple of children -/
structure Triple where
  a : Child
  b : Child
  c : Child

/-- The main theorem to be proved -/
theorem present_giving_property (n : Nat) (h : Odd n) :
  ∃ (children : Finset Child) (S : Finset Triple),
    (children.card = 3 * n) ∧
    (∀ (x y : Child), x ∈ children → y ∈ children → x ≠ y →
      ∃! (t : Triple), t ∈ S ∧ (t.a = x ∧ t.b = y ∨ t.a = x ∧ t.c = y ∨ t.b = x ∧ t.c = y)) ∧
    (∀ (t : Triple), t ∈ S →
      ∃ (t' : Triple), t' ∈ S ∧ t'.a = t.a ∧ t'.b = t.c ∧ t'.c = t.b) := by
  sorry

end NUMINAMATH_CALUDE_present_giving_property_l3135_313548


namespace NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_intersection_l3135_313580

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem union_necessary_not_sufficient_for_intersection :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧
  (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) :=
sorry

end NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_intersection_l3135_313580


namespace NUMINAMATH_CALUDE_ellipse_equation_from_properties_l3135_313517

/-- An ellipse with specified properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  min_distance_to_focus : ℝ
  eccentricity : ℝ

/-- The equation of an ellipse given its properties -/
def ellipse_equation (E : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 8 + y^2 / 4 = 1

/-- Theorem stating that an ellipse with the given properties has the specified equation -/
theorem ellipse_equation_from_properties (E : Ellipse) 
  (h1 : E.center = (0, 0))
  (h2 : E.foci_on_x_axis = true)
  (h3 : E.min_distance_to_focus = 2 * Real.sqrt 2 - 2)
  (h4 : E.eccentricity = Real.sqrt 2 / 2) :
  ellipse_equation E = fun x y => x^2 / 8 + y^2 / 4 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_properties_l3135_313517


namespace NUMINAMATH_CALUDE_function_behavior_l3135_313512

theorem function_behavior (f : ℝ → ℝ) (h : ∀ x : ℝ, f x < f (x + 1)) :
  (∃ a b : ℝ, a < b ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∃ g : ℝ → ℝ, (∀ x : ℝ, g x < g (x + 1)) ∧
    ∀ a b : ℝ, a < b → ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ g x ≥ g y) :=
by sorry

end NUMINAMATH_CALUDE_function_behavior_l3135_313512


namespace NUMINAMATH_CALUDE_division_problem_l3135_313581

theorem division_problem (x y : ℕ+) (h1 : x = 10 * y + 3) (h2 : 2 * x = 21 * y + 1) : 
  11 * y - x = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3135_313581


namespace NUMINAMATH_CALUDE_square_difference_l3135_313554

theorem square_difference (a b : ℝ) : a^2 - 2*a*b + b^2 = (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3135_313554


namespace NUMINAMATH_CALUDE_star_wars_earnings_value_l3135_313585

/-- The cost to make The Lion King in millions of dollars -/
def lion_king_cost : ℝ := 10

/-- The box office earnings of The Lion King in millions of dollars -/
def lion_king_earnings : ℝ := 200

/-- The cost to make Star Wars in millions of dollars -/
def star_wars_cost : ℝ := 25

/-- The profit of The Lion King in millions of dollars -/
def lion_king_profit : ℝ := lion_king_earnings - lion_king_cost

/-- The profit of Star Wars in millions of dollars -/
def star_wars_profit : ℝ := 2 * lion_king_profit

/-- The earnings of Star Wars in millions of dollars -/
def star_wars_earnings : ℝ := star_wars_cost + star_wars_profit

theorem star_wars_earnings_value : star_wars_earnings = 405 := by
  sorry

#eval star_wars_earnings

end NUMINAMATH_CALUDE_star_wars_earnings_value_l3135_313585


namespace NUMINAMATH_CALUDE_orange_pill_cost_l3135_313557

/-- Represents the cost of pills for Alice's treatment --/
structure PillCost where
  orange : ℝ
  blue : ℝ
  duration : ℕ
  daily_intake : ℕ
  total_cost : ℝ

/-- The cost of pills satisfies the given conditions --/
def is_valid_cost (cost : PillCost) : Prop :=
  cost.orange = cost.blue + 2 ∧
  cost.duration = 21 ∧
  cost.daily_intake = 1 ∧
  cost.total_cost = 735 ∧
  cost.duration * cost.daily_intake * (cost.orange + cost.blue) = cost.total_cost

/-- The theorem stating that the cost of one orange pill is $18.5 --/
theorem orange_pill_cost (cost : PillCost) (h : is_valid_cost cost) : cost.orange = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_orange_pill_cost_l3135_313557


namespace NUMINAMATH_CALUDE_probability_red_before_green_l3135_313575

def num_red : ℕ := 4
def num_green : ℕ := 3
def num_blue : ℕ := 1

def total_chips : ℕ := num_red + num_green + num_blue

theorem probability_red_before_green :
  let favorable_arrangements := (total_chips - 1).choose num_green
  let total_arrangements := total_chips.choose num_green * total_chips.choose num_blue
  (favorable_arrangements * total_chips) / total_arrangements = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_red_before_green_l3135_313575


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3135_313578

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 →
  E = 4 * F + 30 →
  D + E + F = 180 →
  F = 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3135_313578


namespace NUMINAMATH_CALUDE_smallest_prime_with_42_divisors_l3135_313532

-- Define a function to count the number of divisors
def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- Define the function F(p) = p^3 + 2p^2 + p
def F (p : ℕ) : ℕ := p^3 + 2*p^2 + p

-- Main theorem
theorem smallest_prime_with_42_divisors :
  ∃ (p : ℕ), Nat.Prime p ∧ 
             count_divisors (F p) = 42 ∧ 
             (∀ q < p, Nat.Prime q → count_divisors (F q) ≠ 42) ∧
             p = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_with_42_divisors_l3135_313532


namespace NUMINAMATH_CALUDE_logical_equivalences_l3135_313558

theorem logical_equivalences (p q : Prop) : 
  ((p ∧ q) ↔ ¬(¬p ∨ ¬q)) ∧
  ((p ∨ q) ↔ ¬(¬p ∧ ¬q)) ∧
  ((p → q) ↔ (¬q → ¬p)) ∧
  ((p ↔ q) ↔ ((p → q) ∧ (q → p))) :=
by sorry

end NUMINAMATH_CALUDE_logical_equivalences_l3135_313558


namespace NUMINAMATH_CALUDE_walking_speed_l3135_313515

/-- Given a constant walking speed, prove that traveling 30 km in 6 hours results in a speed of 5 kmph -/
theorem walking_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 30) 
    (h2 : time = 6) 
    (h3 : speed = distance / time) : speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_l3135_313515


namespace NUMINAMATH_CALUDE_cube_side_area_l3135_313541

/-- Given a cube with volume 125 cubic decimeters, 
    prove that the surface area of one side is 2500 square centimeters. -/
theorem cube_side_area (volume : ℝ) (side_length : ℝ) : 
  volume = 125 →
  side_length^3 = volume →
  (side_length * 10)^2 = 2500 := by
sorry

end NUMINAMATH_CALUDE_cube_side_area_l3135_313541


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3135_313506

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3135_313506


namespace NUMINAMATH_CALUDE_max_angle_APB_l3135_313574

-- Define the circles C and M
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1
def circle_M (x y θ : ℝ) : Prop := (x - 3 - 3 * Real.cos θ)^2 + (y - 3 * Real.sin θ)^2 = 1

-- Define a point P on circle M
def point_on_M (P : ℝ × ℝ) (θ : ℝ) : Prop := circle_M P.1 P.2 θ

-- Define points A and B on circle C
def points_on_C (A B : ℝ × ℝ) : Prop := circle_C A.1 A.2 ∧ circle_C B.1 B.2

-- Define the line PAB touching circle C
def line_touches_C (P A B : ℝ × ℝ) : Prop := 
  ∃ θ : ℝ, point_on_M P θ ∧ points_on_C A B

-- Theorem stating the maximum value of angle APB
theorem max_angle_APB : 
  ∀ P A B : ℝ × ℝ, line_touches_C P A B → 
  ∃ angle : ℝ, angle ≤ π / 3 ∧ 
  (∀ P' A' B' : ℝ × ℝ, line_touches_C P' A' B' → 
   ∃ angle' : ℝ, angle' ≤ angle) :=
sorry

end NUMINAMATH_CALUDE_max_angle_APB_l3135_313574


namespace NUMINAMATH_CALUDE_danicas_car_arrangement_l3135_313519

/-- The number of cars Danica currently has -/
def current_cars : ℕ := 29

/-- The number of cars required in each row -/
def cars_per_row : ℕ := 8

/-- The function to calculate the number of additional cars needed -/
def additional_cars_needed (current : ℕ) (per_row : ℕ) : ℕ :=
  (per_row - (current % per_row)) % per_row

theorem danicas_car_arrangement :
  additional_cars_needed current_cars cars_per_row = 3 := by
  sorry

end NUMINAMATH_CALUDE_danicas_car_arrangement_l3135_313519


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l3135_313521

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 - 2*a - 1 = 0) → (b^2 - 2*b - 1 = 0) → a^2 + a + 3*b = 7 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l3135_313521


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3135_313503

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3135_313503


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3135_313516

theorem min_value_of_expression (a b : ℝ) (ha : a > 1) (hab : a * b = 2 * a + b) :
  (∀ x y : ℝ, x > 1 ∧ x * y = 2 * x + y → (a + 1) * (b + 2) ≤ (x + 1) * (y + 2)) ∧
  (a + 1) * (b + 2) = 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3135_313516


namespace NUMINAMATH_CALUDE_range_of_a_l3135_313563

-- Define propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*m*a + 12*a^2 < 0 ∧ a > 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m-1) + y^2 / (2-m) = 1 ∧ 
  2 - m > 0 ∧ m - 1 > 0 ∧ 2 - m > m - 1

-- Define the relationship between p and q
def relationship (a : ℝ) : Prop := 
  (∀ m : ℝ, ¬(p m a) → ¬(q m)) ∧ 
  ¬(∀ m : ℝ, ¬(q m) → ¬(p m a))

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, (∃ m : ℝ, p m a ∨ q m) ∧ relationship a ↔ 1/3 ≤ a ∧ a ≤ 3/8 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3135_313563


namespace NUMINAMATH_CALUDE_annes_walking_time_l3135_313508

/-- Anne's walking problem -/
theorem annes_walking_time (speed : ℝ) (distance : ℝ) (h1 : speed = 2) (h2 : distance = 6) :
  distance / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_annes_walking_time_l3135_313508


namespace NUMINAMATH_CALUDE_sam_gave_thirteen_cards_l3135_313514

/-- The number of baseball cards Sam gave to Mike -/
def cards_from_sam (initial_cards final_cards : ℕ) : ℕ :=
  final_cards - initial_cards

theorem sam_gave_thirteen_cards :
  let initial_cards : ℕ := 87
  let final_cards : ℕ := 100
  cards_from_sam initial_cards final_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_sam_gave_thirteen_cards_l3135_313514


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3135_313531

theorem absolute_value_inequality (x : ℝ) : |x| ≠ 3 → x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3135_313531


namespace NUMINAMATH_CALUDE_factor_values_l3135_313599

def polynomial (x : ℝ) : ℝ := 8 * x^2 + 18 * x - 5

theorem factor_values (t : ℝ) : 
  (∀ x, polynomial x = 0 → x = t) ↔ t = 1/4 ∨ t = -5 := by
  sorry

end NUMINAMATH_CALUDE_factor_values_l3135_313599


namespace NUMINAMATH_CALUDE_circus_tent_sections_l3135_313568

theorem circus_tent_sections (section_capacity : ℕ) (total_capacity : ℕ) : 
  section_capacity = 246 → total_capacity = 984 → total_capacity / section_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_sections_l3135_313568


namespace NUMINAMATH_CALUDE_percentage_difference_l3135_313572

theorem percentage_difference (x y : ℝ) (h : y = x * (1 + 0.8181818181818181)) :
  x = y * 0.55 :=
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3135_313572


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3135_313507

theorem line_tangent_to_circle (m : ℝ) : 
  (∃ (x y : ℝ), 3*x + 4*y + m = 0 ∧ (x - 1)^2 + (y + 2)^2 = 4 ∧
  ∀ (x' y' : ℝ), 3*x' + 4*y' + m = 0 → (x' - 1)^2 + (y' + 2)^2 ≥ 4) →
  m = 15 ∨ m = -5 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3135_313507


namespace NUMINAMATH_CALUDE_range_of_a_l3135_313530

/-- Given sets A and B, prove that if A ∪ B = A, then 0 < a ≤ 9/5 -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := { x | 0 < x ∧ x ≤ 3 }
  let B : Set ℝ := { x | x^2 - 2*a*x + a ≤ 0 }
  (A ∪ B = A) → (0 < a ∧ a ≤ 9/5) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3135_313530


namespace NUMINAMATH_CALUDE_addition_subtraction_equality_l3135_313550

theorem addition_subtraction_equality : 147 + 31 - 19 + 21 = 180 := by sorry

end NUMINAMATH_CALUDE_addition_subtraction_equality_l3135_313550


namespace NUMINAMATH_CALUDE_robins_haircut_l3135_313577

/-- Given Robin's initial hair length and current hair length, 
    prove the amount of hair cut is the difference between these lengths. -/
theorem robins_haircut (initial_length current_length : ℕ) 
  (h1 : initial_length = 17)
  (h2 : current_length = 13) :
  initial_length - current_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_robins_haircut_l3135_313577


namespace NUMINAMATH_CALUDE_marble_ratio_l3135_313525

def dans_marbles : ℕ := 5
def marys_marbles : ℕ := 10

theorem marble_ratio : 
  (marys_marbles : ℚ) / (dans_marbles : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l3135_313525


namespace NUMINAMATH_CALUDE_circle_center_distance_to_line_l3135_313571

theorem circle_center_distance_to_line : ∃ (center : ℝ × ℝ),
  (∀ (x y : ℝ), x^2 + 2*x + y^2 = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = 1) ∧
  |center.1 - 3| = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_distance_to_line_l3135_313571


namespace NUMINAMATH_CALUDE_paper_I_maximum_mark_l3135_313565

/-- The maximum mark for Paper I -/
def maximum_mark : ℝ := 150

/-- The passing percentage for Paper I -/
def passing_percentage : ℝ := 0.40

/-- The marks secured by the candidate -/
def secured_marks : ℝ := 40

/-- The marks by which the candidate failed -/
def failing_margin : ℝ := 20

/-- Theorem stating that the maximum mark for Paper I is 150 -/
theorem paper_I_maximum_mark :
  (passing_percentage * maximum_mark = secured_marks + failing_margin) ∧
  (maximum_mark = 150) := by
  sorry

end NUMINAMATH_CALUDE_paper_I_maximum_mark_l3135_313565


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3135_313526

/-- The usual time to catch the bus, given that walking at 3/5 of the usual speed results in missing the bus by 5 minutes -/
theorem usual_time_to_catch_bus : ∃ (T : ℝ), T > 0 ∧ (5/3 * T = T + 5) ∧ T = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3135_313526


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3135_313538

theorem polynomial_factorization (a b : ℝ) :
  2 * a^3 - 3 * a^2 * b - 3 * a * b^2 + 2 * b^3 = (a + b) * (a - 2*b) * (2*a - b) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3135_313538


namespace NUMINAMATH_CALUDE_car_race_distance_l3135_313597

theorem car_race_distance (v_A v_B : ℝ) (d : ℝ) :
  v_A > 0 ∧ v_B > 0 ∧ d > 0 →
  (v_A / v_B = (2 * v_A) / (2 * v_B)) →
  (d / v_A = (d/2) / (2 * v_A)) →
  15 = 15 := by sorry

end NUMINAMATH_CALUDE_car_race_distance_l3135_313597


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l3135_313524

/-- Given two cubes of the same material, if the second cube has sides twice as long
    as the first cube, and the first cube weighs 4 pounds, then the second cube weighs 32 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (weight_first : ℝ) (volume_first : ℝ) (weight_second : ℝ) :
  s > 0 →
  weight_first = 4 →
  volume_first = s^3 →
  weight_first / volume_first = weight_second / ((2*s)^3) →
  weight_second = 32 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_doubled_cube_l3135_313524


namespace NUMINAMATH_CALUDE_total_value_of_coins_l3135_313528

/-- Represents the value of a coin in paise -/
inductive CoinType
| OneRupee
| FiftyPaise
| TwentyFivePaise

/-- The number of coins of each type in the bag -/
def coinsPerType : ℕ := 120

/-- Converts a coin type to its value in paise -/
def coinValueInPaise (c : CoinType) : ℕ :=
  match c with
  | CoinType.OneRupee => 100
  | CoinType.FiftyPaise => 50
  | CoinType.TwentyFivePaise => 25

/-- Calculates the total value of all coins of a given type in rupees -/
def totalValueOfCoinType (c : CoinType) : ℚ :=
  (coinsPerType * coinValueInPaise c : ℚ) / 100

/-- Theorem: The total value of all coins in the bag is 210 rupees -/
theorem total_value_of_coins :
  totalValueOfCoinType CoinType.OneRupee +
  totalValueOfCoinType CoinType.FiftyPaise +
  totalValueOfCoinType CoinType.TwentyFivePaise = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_value_of_coins_l3135_313528


namespace NUMINAMATH_CALUDE_circle_area_through_triangle_points_l3135_313564

/-- Given a right triangle PQR with legs PQ = 6 and PR = 8, the area of the circle 
    passing through points Q, R, and the midpoint M of hypotenuse QR is 25π. -/
theorem circle_area_through_triangle_points (P Q R M : ℝ × ℝ) : 
  -- Triangle PQR is a right triangle
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 →
  -- PQ = 6
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 36 →
  -- PR = 8
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 64 →
  -- M is the midpoint of QR
  M = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  -- The area of the circle passing through Q, R, and M
  π * ((Q.1 - M.1)^2 + (Q.2 - M.2)^2) = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_through_triangle_points_l3135_313564


namespace NUMINAMATH_CALUDE_fraction_equality_l3135_313569

theorem fraction_equality : 
  let f (x : ℕ) := x^4 + 324
  (∀ x, f x = (x^2 - 6*x + 18) * (x^2 + 6*x + 18)) →
  (f 64 * f 52 * f 40 * f 28 * f 16) / (f 58 * f 46 * f 34 * f 22 * f 10) = 137 / 1513 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3135_313569


namespace NUMINAMATH_CALUDE_lowry_bonsai_sales_l3135_313555

/-- The number of small bonsai sold by Lowry -/
def small_bonsai_sold : ℕ := 3

/-- The cost of a small bonsai in dollars -/
def small_bonsai_cost : ℕ := 30

/-- The cost of a big bonsai in dollars -/
def big_bonsai_cost : ℕ := 20

/-- The number of big bonsai sold -/
def big_bonsai_sold : ℕ := 5

/-- The total earnings in dollars -/
def total_earnings : ℕ := 190

theorem lowry_bonsai_sales :
  small_bonsai_sold * small_bonsai_cost + big_bonsai_sold * big_bonsai_cost = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_lowry_bonsai_sales_l3135_313555


namespace NUMINAMATH_CALUDE_bacteria_growth_l3135_313502

theorem bacteria_growth (initial_count : ℕ) : 
  (initial_count * (4 ^ 15) = 4194304) → initial_count = 1 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l3135_313502


namespace NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l3135_313593

theorem equation_holds_iff_specific_values :
  ∀ (a b p q : ℝ),
  (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
  ((b = (1/2) * (2^20 - 1)^(1/20) ∧ a = -(2^20 - 1)^(1/20)) ∨
   (b = -(1/2) * (2^20 - 1)^(1/20) ∧ a = (2^20 - 1)^(1/20))) ∧
  p = -1 ∧ q = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l3135_313593


namespace NUMINAMATH_CALUDE_work_completion_time_l3135_313537

/-- Given workers a, b, and c with their work rates, prove the time taken when all work together -/
theorem work_completion_time
  (total_work : ℝ)
  (time_ab : ℝ)
  (time_a : ℝ)
  (time_c : ℝ)
  (h1 : time_ab = 9)
  (h2 : time_a = 18)
  (h3 : time_c = 24)
  : (total_work / (total_work / time_ab + total_work / time_a + total_work / time_c)) = 72 / 11 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3135_313537


namespace NUMINAMATH_CALUDE_intersection_M_N_l3135_313549

def M : Set ℝ := {x | 0 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3135_313549


namespace NUMINAMATH_CALUDE_f_monotone_range_l3135_313596

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem f_monotone_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_range_l3135_313596


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3135_313542

/-- A line that bisects a circle passes through its center -/
axiom line_bisects_circle_passes_through_center 
  (a b c d : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = c^2 → y = d*x + (b - d*a)) → 
  b = d*a + c^2/(2*d)

/-- The equation of a circle -/
def is_on_circle (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

/-- The equation of a line -/
def is_on_line (x y m c : ℝ) : Prop :=
  y = m*x + c

theorem line_tangent_to_circle 
  (a b r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, is_on_circle x y 1 2 2 ↔ is_on_circle x y a b r) →
  (∀ x y : ℝ, is_on_line x y 1 1 → is_on_circle x y a b r) →
  ∀ y : ℝ, is_on_circle 3 y a b r ↔ y = 2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3135_313542


namespace NUMINAMATH_CALUDE_abs_minus_self_nonnegative_l3135_313529

theorem abs_minus_self_nonnegative (a : ℚ) : |a| - a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_self_nonnegative_l3135_313529


namespace NUMINAMATH_CALUDE_distinct_sums_count_l3135_313587

/-- Represents the number of coins of each denomination -/
structure CoinSet :=
  (one_yuan : ℕ)
  (half_yuan : ℕ)

/-- Represents a selection of coins -/
structure CoinSelection :=
  (one_yuan : ℕ)
  (half_yuan : ℕ)

/-- Calculates the sum of face values for a given coin selection -/
def sumFaceValues (selection : CoinSelection) : ℚ :=
  selection.one_yuan + selection.half_yuan / 2

/-- Generates all possible coin selections given a coin set and total number of coins to select -/
def possibleSelections (coins : CoinSet) (total : ℕ) : List CoinSelection :=
  sorry

/-- Calculates the number of distinct sums from all possible selections -/
def distinctSums (coins : CoinSet) (total : ℕ) : ℕ :=
  (possibleSelections coins total).map sumFaceValues |> List.eraseDups |> List.length

/-- The main theorem stating that there are exactly 7 distinct sums when selecting 6 coins from 5 one-yuan and 6 half-yuan coins -/
theorem distinct_sums_count :
  distinctSums (CoinSet.mk 5 6) 6 = 7 := by sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l3135_313587


namespace NUMINAMATH_CALUDE_n_value_for_specific_x_and_y_l3135_313561

theorem n_value_for_specific_x_and_y :
  let x : ℕ := 3
  let y : ℕ := 1
  let n : ℤ := x - 3 * y^(x - y) + 1
  n = 1 := by sorry

end NUMINAMATH_CALUDE_n_value_for_specific_x_and_y_l3135_313561


namespace NUMINAMATH_CALUDE_luke_spent_eleven_l3135_313501

/-- The amount of money Luke spent, given his initial amount, 
    the amount he received, and his current amount. -/
def money_spent (initial amount_received current : ℕ) : ℕ :=
  initial + amount_received - current

/-- Theorem stating that Luke spent $11 -/
theorem luke_spent_eleven : 
  money_spent 48 21 58 = 11 := by sorry

end NUMINAMATH_CALUDE_luke_spent_eleven_l3135_313501


namespace NUMINAMATH_CALUDE_apple_pie_problem_l3135_313591

/-- The number of apples needed per pie -/
def apples_per_pie (total_pies : ℕ) (apples_from_garden : ℕ) (apples_to_buy : ℕ) : ℕ :=
  (apples_from_garden + apples_to_buy) / total_pies

theorem apple_pie_problem :
  apples_per_pie 10 50 30 = 8 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_problem_l3135_313591


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l3135_313510

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = 370 →
    12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
    balcony_tickets - orchestra_tickets = 190 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l3135_313510


namespace NUMINAMATH_CALUDE_equation_root_implies_m_values_l3135_313520

theorem equation_root_implies_m_values (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*m*x + m^2 - 1 = 0) ∧ 
  (3^2 + 2*m*3 + m^2 - 1 = 0) →
  m = -2 ∨ m = -4 := by
sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_values_l3135_313520


namespace NUMINAMATH_CALUDE_average_weight_equation_indeterminate_section_b_size_l3135_313540

theorem average_weight_equation (x : ℕ) : (36 * 30) + (x * 30) = (36 + x) * 30 := by
  sorry

theorem indeterminate_section_b_size : 
  ∀ (x : ℕ), (36 * 30) + (x * 30) = (36 + x) * 30 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_equation_indeterminate_section_b_size_l3135_313540


namespace NUMINAMATH_CALUDE_four_numbers_product_sum_prime_l3135_313567

theorem four_numbers_product_sum_prime :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Nat.Prime (a * b + c * d) ∧
    Nat.Prime (a * c + b * d) ∧
    Nat.Prime (a * d + b * c) := by
  sorry

end NUMINAMATH_CALUDE_four_numbers_product_sum_prime_l3135_313567


namespace NUMINAMATH_CALUDE_proposition_p_is_false_l3135_313545

theorem proposition_p_is_false : ¬(∀ x : ℝ, 2 * x^2 + 2 * x + (1/2 : ℝ) < 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_is_false_l3135_313545


namespace NUMINAMATH_CALUDE_fox_bridge_crossing_fox_initial_money_unique_l3135_313594

/-- The function that doubles the money and subtracts the toll -/
def f (x : ℝ) : ℝ := 2 * x - 40

/-- Theorem stating that applying f three times to 35 results in 0 -/
theorem fox_bridge_crossing :
  f (f (f 35)) = 0 := by sorry

/-- Theorem proving that 35 is the only initial value that results in 0 after three crossings -/
theorem fox_initial_money_unique (x : ℝ) :
  f (f (f x)) = 0 → x = 35 := by sorry

end NUMINAMATH_CALUDE_fox_bridge_crossing_fox_initial_money_unique_l3135_313594


namespace NUMINAMATH_CALUDE_tetrahedron_division_l3135_313589

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- A plane passing through one edge and the midpoint of the opposite edge of a tetrahedron -/
structure DividingPlane where
  tetrahedron : RegularTetrahedron

/-- The parts into which a tetrahedron is divided by the planes -/
structure TetrahedronPart where
  tetrahedron : RegularTetrahedron
  planes : Finset DividingPlane

/-- The theorem stating the division of a regular tetrahedron by six specific planes -/
theorem tetrahedron_division (t : RegularTetrahedron) 
  (h_volume : t.volume = 1) 
  (planes : Finset DividingPlane) 
  (h_planes : planes.card = 6) 
  (h_plane_position : ∀ p ∈ planes, p.tetrahedron = t) : 
  ∃ (parts : Finset TetrahedronPart), 
    (parts.card = 24) ∧ 
    (∀ part ∈ parts, part.tetrahedron = t ∧ part.planes = planes) ∧
    (∀ part ∈ parts, ∃ v : ℝ, v = 1 / 24) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_division_l3135_313589


namespace NUMINAMATH_CALUDE_amelia_remaining_money_l3135_313522

-- Define the given amounts and percentages
def initial_amount : ℝ := 60
def first_course_cost : ℝ := 15
def second_course_additional_cost : ℝ := 5
def dessert_percentage : ℝ := 0.25
def drink_percentage : ℝ := 0.20

-- Define the theorem
theorem amelia_remaining_money :
  let second_course_cost := first_course_cost + second_course_additional_cost
  let dessert_cost := dessert_percentage * second_course_cost
  let first_three_courses_cost := first_course_cost + second_course_cost + dessert_cost
  let drink_cost := drink_percentage * first_three_courses_cost
  let total_cost := first_three_courses_cost + drink_cost
  initial_amount - total_cost = 12 := by sorry

end NUMINAMATH_CALUDE_amelia_remaining_money_l3135_313522


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_l3135_313566

theorem smallest_n_for_probability (n : ℕ) : n ≥ 11 ↔ (n - 4 : ℝ)^3 / (n - 2 : ℝ)^3 > 1/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_l3135_313566


namespace NUMINAMATH_CALUDE_max_decreasing_votes_is_five_l3135_313504

/-- A movie rating system with integer ratings from 0 to 10 -/
structure MovieRating where
  ratings : List ℕ
  valid_ratings : ∀ r ∈ ratings, r ≤ 10

/-- Calculate the current rating as the average of all ratings -/
def current_rating (mr : MovieRating) : ℚ :=
  (mr.ratings.sum : ℚ) / mr.ratings.length

/-- The maximum number of consecutive votes that can decrease the rating by 1 each time -/
def max_consecutive_decreasing_votes (mr : MovieRating) : ℕ :=
  sorry

/-- Theorem: The maximum number of consecutive votes that can decrease 
    an integer rating by 1 each time is 5 -/
theorem max_decreasing_votes_is_five (mr : MovieRating) 
  (h : ∃ n : ℕ, current_rating mr = n) :
  max_consecutive_decreasing_votes mr = 5 :=
sorry

end NUMINAMATH_CALUDE_max_decreasing_votes_is_five_l3135_313504


namespace NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_and_time_l3135_313551

/-- Given two partners with investments and profits, prove their investment ratio -/
theorem investment_ratio_from_profit_ratio_and_time (p q : ℝ) (h1 : p > 0) (h2 : q > 0) :
  (p * 20) / (q * 40) = 7 / 10 → p / q = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_from_profit_ratio_and_time_l3135_313551


namespace NUMINAMATH_CALUDE_player_B_more_consistent_l3135_313511

def player_A_scores : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def player_B_scores : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance (scores : List ℕ) : ℚ :=
  let m := mean scores
  (scores.map (fun x => ((x : ℚ) - m) ^ 2)).sum / scores.length

theorem player_B_more_consistent :
  mean player_A_scores = mean player_B_scores ∧
  variance player_B_scores < variance player_A_scores := by
  sorry

#eval mean player_A_scores
#eval mean player_B_scores
#eval variance player_A_scores
#eval variance player_B_scores

end NUMINAMATH_CALUDE_player_B_more_consistent_l3135_313511


namespace NUMINAMATH_CALUDE_floor_sqrt_150_l3135_313573

theorem floor_sqrt_150 : ⌊Real.sqrt 150⌋ = 12 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_150_l3135_313573


namespace NUMINAMATH_CALUDE_area_triangle_ABC_l3135_313518

/-- Regular octagon with side length 3 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Triangle ABC in the regular octagon -/
def triangle_ABC (octagon : RegularOctagon) : Set (ℝ × ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: Area of triangle ABC in a regular octagon with side length 3 -/
theorem area_triangle_ABC (octagon : RegularOctagon) :
  area (triangle_ABC octagon) = 9 * (2 + Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_l3135_313518


namespace NUMINAMATH_CALUDE_chord_length_in_isosceles_trapezoid_l3135_313562

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : Bool
  /-- The circle is inscribed in the trapezoid -/
  isInscribed : Bool

/-- The theorem stating the length of the chord connecting the tangent points -/
theorem chord_length_in_isosceles_trapezoid 
  (t : IsoscelesTrapezoidWithInscribedCircle) 
  (h1 : t.r = 3)
  (h2 : t.area = 48)
  (h3 : t.isIsosceles = true)
  (h4 : t.isInscribed = true) :
  ∃ (chord_length : ℝ), chord_length = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_in_isosceles_trapezoid_l3135_313562


namespace NUMINAMATH_CALUDE_lobster_theorem_l3135_313527

/-- The total pounds of lobster in three harbors -/
def total_lobster (hooper_bay other1 other2 : ℕ) : ℕ := hooper_bay + other1 + other2

/-- Theorem stating the total pounds of lobster in the three harbors -/
theorem lobster_theorem (hooper_bay other1 other2 : ℕ) 
  (h1 : hooper_bay = 2 * (other1 + other2)) 
  (h2 : other1 = 80) 
  (h3 : other2 = 80) : 
  total_lobster hooper_bay other1 other2 = 480 := by
  sorry

#check lobster_theorem

end NUMINAMATH_CALUDE_lobster_theorem_l3135_313527


namespace NUMINAMATH_CALUDE_clock_rings_eight_times_l3135_313552

/-- A clock that rings every 3 hours, starting at 1 A.M. -/
structure Clock :=
  (ring_interval : ℕ := 3)
  (first_ring : ℕ := 1)

/-- The number of times the clock rings in a 24-hour period -/
def rings_per_day (c : Clock) : ℕ :=
  ((24 - c.first_ring) / c.ring_interval) + 1

theorem clock_rings_eight_times (c : Clock) : rings_per_day c = 8 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_eight_times_l3135_313552


namespace NUMINAMATH_CALUDE_circle_point_range_l3135_313500

/-- Given a point A(0,-3) and a circle C: (x-a)^2 + (y-a+2)^2 = 1,
    if there exists a point M on C such that MA = 2MO,
    then 0 ≤ a ≤ 3 -/
theorem circle_point_range (a : ℝ) :
  (∃ x y : ℝ, (x - a)^2 + (y - a + 2)^2 = 1 ∧
              (x^2 + (y + 3)^2) = 4 * (x^2 + y^2)) →
  0 ≤ a ∧ a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_circle_point_range_l3135_313500


namespace NUMINAMATH_CALUDE_missing_shirts_count_l3135_313584

theorem missing_shirts_count (trousers : ℕ) (total_bill : ℕ) (shirt_cost : ℕ) (trouser_cost : ℕ) (claimed_shirts : ℕ) : 
  trousers = 10 →
  total_bill = 140 →
  shirt_cost = 5 →
  trouser_cost = 9 →
  claimed_shirts = 2 →
  (total_bill - trousers * trouser_cost) / shirt_cost - claimed_shirts = 8 := by
sorry

end NUMINAMATH_CALUDE_missing_shirts_count_l3135_313584


namespace NUMINAMATH_CALUDE_garage_sale_games_l3135_313543

/-- The number of games Luke bought from a friend -/
def games_from_friend : ℕ := 2

/-- The number of games that didn't work -/
def broken_games : ℕ := 2

/-- The number of good games Luke ended up with -/
def good_games : ℕ := 2

/-- The number of games Luke bought at the garage sale -/
def games_from_garage_sale : ℕ := 2

theorem garage_sale_games :
  games_from_friend + games_from_garage_sale - broken_games = good_games :=
by sorry

end NUMINAMATH_CALUDE_garage_sale_games_l3135_313543


namespace NUMINAMATH_CALUDE_arithmetic_square_root_when_negative_root_is_five_l3135_313586

theorem arithmetic_square_root_when_negative_root_is_five (x : ℝ) : 
  ((-5 : ℝ)^2 = x) → Real.sqrt x = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_when_negative_root_is_five_l3135_313586


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l3135_313588

theorem min_value_abs_sum (x : ℝ) (h : 0 ≤ x ∧ x ≤ 10) :
  |x - 4| + |x + 2| + |x - 5| + |3*x - 1| + |2*x + 6| ≥ 17.333 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l3135_313588


namespace NUMINAMATH_CALUDE_mr_slinkums_order_l3135_313598

theorem mr_slinkums_order (on_shelves_percent : ℚ) (in_storage : ℕ) : 
  on_shelves_percent = 1/5 ∧ in_storage = 120 → 
  (1 - on_shelves_percent) * 150 = in_storage :=
by
  sorry

end NUMINAMATH_CALUDE_mr_slinkums_order_l3135_313598


namespace NUMINAMATH_CALUDE_translation_result_l3135_313556

/-- A line in the xy-plane is represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount. -/
def translateLine (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

/-- The original line y = 2x - 1 -/
def originalLine : Line :=
  { slope := 2, intercept := -1 }

/-- The amount of upward translation -/
def translationAmount : ℝ := 2

/-- The resulting line after translation -/
def resultingLine : Line := translateLine originalLine translationAmount

theorem translation_result :
  resultingLine.slope = 2 ∧ resultingLine.intercept = 1 := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l3135_313556


namespace NUMINAMATH_CALUDE_equation_solution_l3135_313559

theorem equation_solution : ∃ x : ℝ, 
  (x^2 - 7*x + 12) / (x^2 - 9*x + 20) = (x^2 - 4*x - 21) / (x^2 - 5*x - 24) ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3135_313559


namespace NUMINAMATH_CALUDE_sqrt_floor_equality_l3135_313546

theorem sqrt_floor_equality (n : ℕ) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_floor_equality_l3135_313546


namespace NUMINAMATH_CALUDE_green_marbles_taken_l3135_313523

theorem green_marbles_taken (initial_green : ℝ) (remaining_green : ℝ) 
  (h1 : initial_green = 32.0)
  (h2 : remaining_green = 9.0) :
  initial_green - remaining_green = 23.0 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_taken_l3135_313523


namespace NUMINAMATH_CALUDE_sequence_problem_l3135_313536

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d ∧ d ≠ 0

-- Define a geometric sequence
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, b (n + 1) = r * b n

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  (2 * a 3 - (a 7)^2 + 2 * a 11 = 0) →
  geometric_sequence b →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3135_313536


namespace NUMINAMATH_CALUDE_book_purchase_problem_l3135_313582

theorem book_purchase_problem (total_A total_B only_B both : ℕ) 
  (h1 : total_A = 2 * total_B)
  (h2 : both = 500)
  (h3 : both = 2 * only_B) :
  total_A - both = 1000 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l3135_313582
