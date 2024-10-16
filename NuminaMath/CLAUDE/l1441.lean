import Mathlib

namespace NUMINAMATH_CALUDE_remainder_zero_l1441_144101

/-- A polynomial of degree 5 with real coefficients -/
structure Poly5 (D E F G H : ℝ) where
  q : ℝ → ℝ
  eq : ∀ x, q x = D * x^5 + E * x^4 + F * x^3 + G * x^2 + H * x + 2

/-- The remainder theorem for polynomials -/
axiom remainder_theorem {p : ℝ → ℝ} {a r : ℝ} :
  (∀ x, ∃ q, p x = (x - a) * q + r) ↔ p a = r

/-- Main theorem: If the remainder of q(x) divided by (x - 4) is 15,
    then the remainder of q(x) divided by (x + 4) is 0 -/
theorem remainder_zero {D E F G H : ℝ} (p : Poly5 D E F G H) :
  p.q 4 = 15 → p.q (-4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_zero_l1441_144101


namespace NUMINAMATH_CALUDE_zeros_bound_l1441_144194

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

theorem zeros_bound (a : ℝ) :
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → (f a x ≠ 0 ∨ f a y ≠ 0 ∨ f a z ≠ 0)) →
  a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_bound_l1441_144194


namespace NUMINAMATH_CALUDE_andrew_payment_l1441_144113

/-- The total amount Andrew paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_weight : ℕ) (grape_rate : ℕ) (mango_weight : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_weight * grape_rate + mango_weight * mango_rate

/-- Theorem stating that Andrew paid 908 to the shopkeeper -/
theorem andrew_payment : total_amount 7 68 9 48 = 908 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l1441_144113


namespace NUMINAMATH_CALUDE_inconsistent_statistics_l1441_144129

theorem inconsistent_statistics (x_bar m S_squared : ℝ) 
  (h1 : x_bar = 0)
  (h2 : m = 4)
  (h3 : S_squared = 15.917)
  : ¬ (|x_bar - m| ≤ Real.sqrt S_squared) := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_statistics_l1441_144129


namespace NUMINAMATH_CALUDE_blaine_fish_count_l1441_144125

theorem blaine_fish_count :
  ∀ (blaine_fish keith_fish : ℕ),
    blaine_fish > 0 →
    keith_fish = 2 * blaine_fish →
    blaine_fish + keith_fish = 15 →
    blaine_fish = 5 := by
  sorry

end NUMINAMATH_CALUDE_blaine_fish_count_l1441_144125


namespace NUMINAMATH_CALUDE_fraction_of_nuts_eaten_l1441_144152

def initial_nuts : ℕ := 30
def remaining_nuts : ℕ := 5

theorem fraction_of_nuts_eaten :
  (initial_nuts - remaining_nuts) / initial_nuts = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_nuts_eaten_l1441_144152


namespace NUMINAMATH_CALUDE_runner_speed_l1441_144170

/-- Given a runner who runs 5 days a week, 1.5 hours each day, and covers 60 miles in a week,
    prove that their running speed is 8 mph. -/
theorem runner_speed (days_per_week : ℕ) (hours_per_day : ℝ) (miles_per_week : ℝ) :
  days_per_week = 5 →
  hours_per_day = 1.5 →
  miles_per_week = 60 →
  miles_per_week / (days_per_week * hours_per_day) = 8 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_l1441_144170


namespace NUMINAMATH_CALUDE_original_number_proof_l1441_144192

theorem original_number_proof (x : ℝ) (h : x * 1.25 = 250) : x = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1441_144192


namespace NUMINAMATH_CALUDE_picnic_attendance_theorem_l1441_144127

/-- Represents the percentage of employees who are men -/
def male_percentage : ℝ := 0.5

/-- Represents the percentage of women who attended the picnic -/
def women_attendance_percentage : ℝ := 0.4

/-- Represents the percentage of all employees who attended the picnic -/
def total_attendance_percentage : ℝ := 0.3

/-- Represents the percentage of men who attended the picnic -/
def male_attendance_percentage : ℝ := 0.2

theorem picnic_attendance_theorem :
  male_attendance_percentage * male_percentage + 
  women_attendance_percentage * (1 - male_percentage) = 
  total_attendance_percentage := by
  sorry

#check picnic_attendance_theorem

end NUMINAMATH_CALUDE_picnic_attendance_theorem_l1441_144127


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1441_144164

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (3 * x - 1 ≥ x + 1) ∧ (x + 4 > 4 * x - 2)

-- Define the solution set
def solution_set : Set ℝ :=
  {x : ℝ | 1 ≤ x ∧ x < 2}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1441_144164


namespace NUMINAMATH_CALUDE_car_speed_problem_l1441_144148

/-- Proves that for a journey of 225 km, if a car arrives 45 minutes late when traveling at 50 kmph, then its on-time average speed is 60 kmph. -/
theorem car_speed_problem (journey_length : ℝ) (late_speed : ℝ) (delay : ℝ) :
  journey_length = 225 →
  late_speed = 50 →
  delay = 3/4 →
  ∃ (on_time_speed : ℝ),
    (journey_length / on_time_speed) + delay = (journey_length / late_speed) ∧
    on_time_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1441_144148


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1441_144118

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1441_144118


namespace NUMINAMATH_CALUDE_combination_equality_l1441_144195

theorem combination_equality (x : ℕ+) : (Nat.choose 9 x.val = Nat.choose 9 (2 * x.val)) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l1441_144195


namespace NUMINAMATH_CALUDE_meal_cost_is_25_l1441_144198

/-- The cost of Hilary's meal at Delicious Delhi restaurant -/
def meal_cost : ℝ :=
  let samosa_price : ℝ := 2
  let pakora_price : ℝ := 3
  let lassi_price : ℝ := 2
  let samosa_quantity : ℕ := 3
  let pakora_quantity : ℕ := 4
  let tip_percentage : ℝ := 0.25
  let subtotal : ℝ := samosa_price * samosa_quantity + pakora_price * pakora_quantity + lassi_price
  let tip : ℝ := subtotal * tip_percentage
  subtotal + tip

theorem meal_cost_is_25 : meal_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_is_25_l1441_144198


namespace NUMINAMATH_CALUDE_father_chips_amount_l1441_144199

theorem father_chips_amount (son_chips brother_chips total_chips : ℕ) 
  (h1 : son_chips = 350)
  (h2 : brother_chips = 182)
  (h3 : total_chips = 800) :
  total_chips - (son_chips + brother_chips) = 268 := by
  sorry

end NUMINAMATH_CALUDE_father_chips_amount_l1441_144199


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1441_144132

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → (x + 2) / (x - 1) > 0) ∧
  (∃ x : ℝ, x ≤ 1 ∧ (x + 2) / (x - 1) > 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1441_144132


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1441_144145

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- The condition "a = -1" -/
def condition (a : ℝ) : Prop := a = -1

/-- The line ax + y - 1 = 0 is parallel to x + ay + 5 = 0 -/
def lines_are_parallel (a : ℝ) : Prop := parallel_lines (-a) (1/a)

/-- "a = -1" is a sufficient but not necessary condition for the lines to be parallel -/
theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, condition a → lines_are_parallel a) ∧
  (∃ a, lines_are_parallel a ∧ ¬condition a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1441_144145


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1441_144135

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water : ∃ (x : ℝ),
  (x + 3) * (24 / 60) = 7.2 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1441_144135


namespace NUMINAMATH_CALUDE_rhombus_triangle_inscribed_circle_ratio_l1441_144151

/-- Given a rhombus ABCD with acute angle α and a triangle ABC formed by two sides of the rhombus
    and its longer diagonal, this theorem states that the ratio of the radius of the circle
    inscribed in the rhombus to the radius of the circle inscribed in the triangle ABC
    is equal to 1 + cos(α/2). -/
theorem rhombus_triangle_inscribed_circle_ratio (α : Real) 
  (h1 : 0 < α ∧ α < π / 2) : 
  ∃ (r1 r2 : Real), r1 > 0 ∧ r2 > 0 ∧
    (r1 / r2 = 1 + Real.cos (α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_triangle_inscribed_circle_ratio_l1441_144151


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l1441_144146

theorem smallest_positive_angle (α : Real) : 
  let P : Real × Real := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  (∃ t : Real, t > 0 ∧ P.1 = t * Real.sin α ∧ P.2 = t * Real.cos α) →
  (∀ β : Real, β > 0 ∧ (∃ s : Real, s > 0 ∧ P.1 = s * Real.sin β ∧ P.2 = s * Real.cos β) → α ≤ β) →
  α = 11 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l1441_144146


namespace NUMINAMATH_CALUDE_jaysons_mom_age_at_birth_l1441_144111

theorem jaysons_mom_age_at_birth (jayson_age : ℕ) (dad_age : ℕ) (mom_age : ℕ) : 
  jayson_age = 10 →
  dad_age = 4 * jayson_age →
  mom_age = dad_age - 2 →
  mom_age - jayson_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_jaysons_mom_age_at_birth_l1441_144111


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1441_144139

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 486 → volume = (surface_area / 6) ^ (3/2) → volume = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1441_144139


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1441_144179

theorem sufficient_but_not_necessary :
  (∃ x : ℝ, (|x - 1| < 4 ∧ ¬(x * (x - 5) < 0))) ∧
  (∀ x : ℝ, (x * (x - 5) < 0) → |x - 1| < 4) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1441_144179


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l1441_144124

/-- The volume of a right circular cone formed by rolling up a five-sixth sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 5 / 6
  let base_radius : ℝ := sector_fraction * r
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let volume : ℝ := (1 / 3) * Real.pi * base_radius^2 * height
  volume = (25 / 3) * Real.pi * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l1441_144124


namespace NUMINAMATH_CALUDE_sqrt_6_bounds_l1441_144147

theorem sqrt_6_bounds : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_bounds_l1441_144147


namespace NUMINAMATH_CALUDE_angle_X_measure_l1441_144122

-- Define the angles in the configuration
def angle_Y : ℝ := 130
def angle_60 : ℝ := 60
def right_angle : ℝ := 90

-- Theorem statement
theorem angle_X_measure :
  ∀ (angle_X : ℝ),
  -- Conditions
  (angle_Y + (180 - angle_Y) = 180) →  -- Y and Z form a linear pair
  (angle_X + angle_60 + right_angle = 180) →  -- Sum of angles in the smaller triangle
  -- Conclusion
  angle_X = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_X_measure_l1441_144122


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1441_144138

/-- The perimeter of a rectangle with length 0.54 meters and width 0.08 meters shorter than the length is 2 meters. -/
theorem rectangle_perimeter : 
  let length : ℝ := 0.54
  let width_difference : ℝ := 0.08
  let width : ℝ := length - width_difference
  let perimeter : ℝ := 2 * (length + width)
  perimeter = 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1441_144138


namespace NUMINAMATH_CALUDE_luncheon_table_capacity_l1441_144128

/-- Given a luncheon where 24 people were invited, 10 didn't show up, and 2 tables were needed,
    prove that each table could hold 7 people. -/
theorem luncheon_table_capacity :
  ∀ (invited : ℕ) (no_show : ℕ) (tables : ℕ) (capacity : ℕ),
    invited = 24 →
    no_show = 10 →
    tables = 2 →
    capacity = (invited - no_show) / tables →
    capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_table_capacity_l1441_144128


namespace NUMINAMATH_CALUDE_triangle_problem_l1441_144171

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  c = 1 →
  b * Real.sin A = a * Real.sin C →
  0 < A →
  A < Real.pi →
  -- Conclusions
  b = 1 ∧
  (∀ x y z : Real, x > 0 → y > 0 → z > 0 → x * Real.sin y ≤ 1/2 * z * Real.sin x) →
  (∃ x y : Real, x > 0 → y > 0 → 1/2 * c * b * Real.sin x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1441_144171


namespace NUMINAMATH_CALUDE_polynomial_equivalence_l1441_144189

theorem polynomial_equivalence (x : ℝ) (y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 4*x^2 + x + 1 = x^2 * (y^2 + y - 6) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equivalence_l1441_144189


namespace NUMINAMATH_CALUDE_A_equals_B_l1441_144173

-- Define sets A and B
def A : Set ℕ := {3, 2}
def B : Set ℕ := {2, 3}

-- Theorem stating that A and B are equal
theorem A_equals_B : A = B := by
  sorry

end NUMINAMATH_CALUDE_A_equals_B_l1441_144173


namespace NUMINAMATH_CALUDE_max_eggs_per_basket_l1441_144193

def purple_eggs : ℕ := 30
def yellow_eggs : ℕ := 45
def min_eggs_per_basket : ℕ := 5

theorem max_eggs_per_basket : 
  ∃ (n : ℕ), n ≥ min_eggs_per_basket ∧ 
  purple_eggs % n = 0 ∧ 
  yellow_eggs % n = 0 ∧
  ∀ (m : ℕ), m > n → 
    (m ≥ min_eggs_per_basket ∧ 
     purple_eggs % m = 0 ∧ 
     yellow_eggs % m = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_per_basket_l1441_144193


namespace NUMINAMATH_CALUDE_exponent_rules_l1441_144149

theorem exponent_rules (a b : ℝ) : 
  (a^3 * a^3 = a^6) ∧ 
  ¬((a*b)^3 = a*b^3) ∧ 
  ¬((a^3)^3 = a^6) ∧ 
  ¬(a^8 / a^4 = a^2) := by
  sorry

end NUMINAMATH_CALUDE_exponent_rules_l1441_144149


namespace NUMINAMATH_CALUDE_is_perfect_square_l1441_144150

/-- Given a = 2992² + 2992² × 2993² + 2993², prove that a is a perfect square -/
theorem is_perfect_square (a : ℕ) (h : a = 2992^2 + 2992^2 * 2993^2 + 2993^2) :
  ∃ n : ℕ, a = n^2 := by sorry

end NUMINAMATH_CALUDE_is_perfect_square_l1441_144150


namespace NUMINAMATH_CALUDE_length_width_ratio_l1441_144191

/-- Represents a rectangle with given width and area -/
structure Rectangle where
  width : ℝ
  area : ℝ

/-- Theorem: For a rectangle with width 4 and area 48, the ratio of length to width is 3:1 -/
theorem length_width_ratio (rect : Rectangle) 
    (h_width : rect.width = 4)
    (h_area : rect.area = 48) :
    rect.area / rect.width / rect.width = 3 := by
  sorry

#check length_width_ratio

end NUMINAMATH_CALUDE_length_width_ratio_l1441_144191


namespace NUMINAMATH_CALUDE_cans_recycled_l1441_144190

/-- Proves the number of cans recycled given the bottle and can deposits, number of bottles, and total money earned -/
theorem cans_recycled 
  (bottle_deposit : ℚ) 
  (can_deposit : ℚ) 
  (bottles_recycled : ℕ) 
  (total_money : ℚ) 
  (h1 : bottle_deposit = 10 / 100)
  (h2 : can_deposit = 5 / 100)
  (h3 : bottles_recycled = 80)
  (h4 : total_money = 15) :
  (total_money - (bottle_deposit * bottles_recycled)) / can_deposit = 140 := by
sorry

end NUMINAMATH_CALUDE_cans_recycled_l1441_144190


namespace NUMINAMATH_CALUDE_x_value_when_y_is_half_l1441_144136

theorem x_value_when_y_is_half :
  ∀ x y : ℝ, y = 1 / (4 * x + 2) → y = 1 / 2 → x = 0 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_half_l1441_144136


namespace NUMINAMATH_CALUDE_three_percent_difference_l1441_144143

theorem three_percent_difference (x y : ℝ) : 
  3 = 0.15 * x → 3 = 0.25 * y → x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_three_percent_difference_l1441_144143


namespace NUMINAMATH_CALUDE_fourth_ball_black_prob_l1441_144165

/-- A box containing colored balls. -/
structure Box where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball from a box. -/
def prob_black_ball (box : Box) : ℚ :=
  box.black_balls / (box.red_balls + box.black_balls)

/-- The theorem stating that the probability of the fourth ball being black
    is equal to the probability of selecting a black ball from the box. -/
theorem fourth_ball_black_prob (box : Box) (h1 : box.red_balls = 2) (h2 : box.black_balls = 5) :
  prob_black_ball box = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_ball_black_prob_l1441_144165


namespace NUMINAMATH_CALUDE_square_root_divided_by_13_equals_4_l1441_144162

theorem square_root_divided_by_13_equals_4 :
  ∃ x : ℝ, (Real.sqrt x) / 13 = 4 ∧ x = 2704 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_13_equals_4_l1441_144162


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1441_144187

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x - k = 0 ∧ x = 1) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1441_144187


namespace NUMINAMATH_CALUDE_skew_lines_equivalent_l1441_144174

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line

-- Define a type for planes in 3D space
structure Plane3D where
  -- Add necessary fields to represent a plane

-- Define what it means for two lines to be parallel
def parallel (a b : Line3D) : Prop :=
  sorry

-- Define what it means for a line to be a subset of a plane
def line_subset_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Define what it means for two lines to intersect
def intersect (a b : Line3D) : Prop :=
  sorry

-- Define skew lines according to the first definition
def skew_def1 (a b : Line3D) : Prop :=
  ¬(intersect a b) ∧ ¬(parallel a b)

-- Define skew lines according to the second definition
def skew_def2 (a b : Line3D) : Prop :=
  ¬∃ (p : Plane3D), line_subset_plane a p ∧ line_subset_plane b p

-- Theorem stating the equivalence of the two definitions
theorem skew_lines_equivalent (a b : Line3D) :
  skew_def1 a b ↔ skew_def2 a b :=
sorry

end NUMINAMATH_CALUDE_skew_lines_equivalent_l1441_144174


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l1441_144115

theorem factorial_ratio_equals_sixty_sevenths : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l1441_144115


namespace NUMINAMATH_CALUDE_grade_students_ratio_l1441_144116

theorem grade_students_ratio (sixth_grade seventh_grade : ℕ) : 
  (sixth_grade : ℚ) / seventh_grade = 3 / 4 →
  seventh_grade - sixth_grade = 13 →
  sixth_grade = 39 ∧ seventh_grade = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_grade_students_ratio_l1441_144116


namespace NUMINAMATH_CALUDE_sin_squared_sum_l1441_144121

theorem sin_squared_sum (α β : ℝ) : 
  Real.sin (α + β) ^ 2 = Real.cos α ^ 2 + Real.cos β ^ 2 - 2 * Real.cos α * Real.cos β * Real.cos (α + β) := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_l1441_144121


namespace NUMINAMATH_CALUDE_decimal_fraction_equality_l1441_144100

theorem decimal_fraction_equality (b : ℕ+) : 
  (4 * b + 19 : ℚ) / (6 * b + 11) = 19 / 25 → b = 19 := by
  sorry

end NUMINAMATH_CALUDE_decimal_fraction_equality_l1441_144100


namespace NUMINAMATH_CALUDE_cost_price_is_95_l1441_144186

/-- Represents the cost price for one metre of cloth -/
def cost_price_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  total_selling_price / total_metres + loss_per_metre

/-- Theorem stating that the cost price for one metre of cloth is 95 -/
theorem cost_price_is_95 :
  cost_price_per_metre 200 18000 5 = 95 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_95_l1441_144186


namespace NUMINAMATH_CALUDE_min_distance_sum_l1441_144102

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4*y

def focus : ℝ × ℝ := (0, 1)

def point_A : ℝ × ℝ := (2, 3)

theorem min_distance_sum (P : ℝ × ℝ) :
  parabola P.1 P.2 →
  Real.sqrt ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2) +
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1441_144102


namespace NUMINAMATH_CALUDE_election_majority_proof_l1441_144117

theorem election_majority_proof :
  ∀ (total_votes : ℕ) (winning_percentage : ℚ),
    total_votes = 470 →
    winning_percentage = 70 / 100 →
    ∃ (winning_votes losing_votes : ℕ),
      winning_votes = (winning_percentage * total_votes).floor ∧
      losing_votes = total_votes - winning_votes ∧
      winning_votes - losing_votes = 188 :=
by
  sorry

end NUMINAMATH_CALUDE_election_majority_proof_l1441_144117


namespace NUMINAMATH_CALUDE_spinner_probability_l1441_144160

theorem spinner_probability : 
  ∀ (p_A p_B p_C p_D p_E : ℚ),
  p_A = 2/7 →
  p_B = 3/14 →
  p_C = p_E →
  p_D = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/8 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l1441_144160


namespace NUMINAMATH_CALUDE_largest_p_value_l1441_144158

theorem largest_p_value (m n p : ℕ) : 
  m ≤ n → n ≤ p → 
  2 * m * n * p = (m + 2) * (n + 2) * (p + 2) → 
  p ≤ 130 :=
by sorry

end NUMINAMATH_CALUDE_largest_p_value_l1441_144158


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1441_144167

theorem quadratic_two_distinct_roots :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := -1
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1441_144167


namespace NUMINAMATH_CALUDE_rental_cost_equality_l1441_144180

/-- The daily rate charged by Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate charged by Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The daily rate charged by City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The per-mile rate charged by City Rentals in dollars -/
def city_mile_rate : ℝ := 0.21

/-- The mileage at which the cost is the same for both rental companies -/
def equal_cost_mileage : ℝ := 150

theorem rental_cost_equality :
  safety_daily_rate + safety_mile_rate * equal_cost_mileage =
  city_daily_rate + city_mile_rate * equal_cost_mileage :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_equality_l1441_144180


namespace NUMINAMATH_CALUDE_fraction_equality_l1441_144104

theorem fraction_equality : 48 / (7 - 3/8 + 4/9) = 3456 / 509 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1441_144104


namespace NUMINAMATH_CALUDE_two_hour_charge_l1441_144169

/-- Represents the pricing scheme of a psychologist's therapy sessions. -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  hourDifference : firstHourCharge = additionalHourCharge + 35

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

theorem two_hour_charge (pricing : TherapyPricing) 
  (h : totalCharge pricing 5 = 350) : totalCharge pricing 2 = 161 := by
  sorry

#check two_hour_charge

end NUMINAMATH_CALUDE_two_hour_charge_l1441_144169


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1441_144185

/-- The equation (x+4)(x+1) = m + 2x has exactly one real solution if and only if m = 7/4 -/
theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (x + 4) * (x + 1) = m + 2 * x) ↔ m = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1441_144185


namespace NUMINAMATH_CALUDE_kevin_max_sum_l1441_144155

def kevin_process (S : Finset ℕ) : Finset ℕ :=
  sorry

theorem kevin_max_sum :
  let initial_set : Finset ℕ := Finset.range 15
  let final_set := kevin_process initial_set
  Finset.sum final_set id = 360864 :=
sorry

end NUMINAMATH_CALUDE_kevin_max_sum_l1441_144155


namespace NUMINAMATH_CALUDE_sum_of_legs_is_462_l1441_144141

/-- A right triangle with two inscribed squares -/
structure RightTriangleWithSquares where
  -- The right triangle
  AC : ℝ
  CB : ℝ
  -- The two inscribed squares
  S1 : ℝ
  S2 : ℝ
  -- Conditions
  right_triangle : AC^2 + CB^2 = (AC + CB)^2 / 2
  area_S1 : S1^2 = 441
  area_S2 : S2^2 = 440

/-- The sum of the legs of the right triangle is 462 -/
theorem sum_of_legs_is_462 (t : RightTriangleWithSquares) : t.AC + t.CB = 462 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_legs_is_462_l1441_144141


namespace NUMINAMATH_CALUDE_sixth_grade_boys_l1441_144140

theorem sixth_grade_boys (total_students : ℕ) (boys : ℕ) : 
  total_students = 152 →
  boys * 10 = (total_students - boys - 5) * 11 →
  boys = 77 := by
sorry

end NUMINAMATH_CALUDE_sixth_grade_boys_l1441_144140


namespace NUMINAMATH_CALUDE_basketball_volume_after_drilling_l1441_144126

/-- The volume of a basketball after drilling holes for handles -/
theorem basketball_volume_after_drilling (d : ℝ) (r1 r2 h : ℝ) :
  d = 50 ∧ r1 = 2 ∧ r2 = 1.5 ∧ h = 10 →
  (4/3 * π * (d/2)^3) - (2 * π * r1^2 * h + 2 * π * r2^2 * h) = (62250/3) * π :=
by sorry

end NUMINAMATH_CALUDE_basketball_volume_after_drilling_l1441_144126


namespace NUMINAMATH_CALUDE_first_girl_productivity_higher_l1441_144183

/-- Represents a knitter with their working time and break time -/
structure Knitter where
  workTime : ℕ
  breakTime : ℕ

/-- Calculates the total cycle time for a knitter -/
def cycleTime (k : Knitter) : ℕ :=
  k.workTime + k.breakTime

/-- Calculates the number of complete cycles in a given time -/
def completeCycles (k : Knitter) (totalTime : ℕ) : ℕ :=
  totalTime / cycleTime k

/-- Calculates the total working time within a given time period -/
def totalWorkTime (k : Knitter) (totalTime : ℕ) : ℕ :=
  completeCycles k totalTime * k.workTime

/-- Theorem: The first girl's productivity is 5% higher than the second girl's -/
theorem first_girl_productivity_higher (girl1 girl2 : Knitter)
    (h1 : girl1.workTime = 5)
    (h2 : girl2.workTime = 7)
    (h3 : girl1.breakTime = 1)
    (h4 : girl2.breakTime = 1)
    (h5 : ∃ t : ℕ, totalWorkTime girl1 t = totalWorkTime girl2 t ∧ t > 0) :
    (21 : ℚ) / 20 = girl2.workTime / girl1.workTime := by
  sorry

end NUMINAMATH_CALUDE_first_girl_productivity_higher_l1441_144183


namespace NUMINAMATH_CALUDE_ball_selection_count_l1441_144144

def num_colors : ℕ := 4
def balls_per_color : ℕ := 6
def balls_to_select : ℕ := 3

def valid_number_combinations : List (List ℕ) :=
  [[1, 3, 5], [1, 3, 6], [1, 4, 6], [2, 4, 6]]

theorem ball_selection_count :
  (num_colors.choose balls_to_select) *
  (valid_number_combinations.length) *
  (balls_to_select.factorial) = 96 := by
sorry

end NUMINAMATH_CALUDE_ball_selection_count_l1441_144144


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1441_144175

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ (p : ℚ) / q < 5 / 8 →
  q ≥ 13 ∧ (q = 13 → p = 8) :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1441_144175


namespace NUMINAMATH_CALUDE_kishore_savings_percentage_l1441_144197

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 700
def savings : ℕ := 1800

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous
def total_salary : ℕ := total_expenses + savings

theorem kishore_savings_percentage :
  (savings : ℚ) / (total_salary : ℚ) * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_kishore_savings_percentage_l1441_144197


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_quadrilateral_property_l1441_144119

structure Triangle where
  angles : Fin 3 → ℝ
  sum_eq_pi : angles 0 + angles 1 + angles 2 = π

structure Quadrilateral where
  angles : Fin 4 → ℝ
  sum_eq_2pi : angles 0 + angles 1 + angles 2 + angles 3 = 2 * π

def has_sum_angle_property (t : Triangle) (q : Quadrilateral) : Prop :=
  ∀ (i j : Fin 3), i ≠ j → ∃ (k : Fin 4), q.angles k = t.angles i + t.angles j

def is_isosceles (t : Triangle) : Prop :=
  ∃ (i j : Fin 3), i ≠ j ∧ t.angles i = t.angles j

theorem triangle_isosceles_from_quadrilateral_property
  (t : Triangle) (q : Quadrilateral) (h : has_sum_angle_property t q) :
  is_isosceles t :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_from_quadrilateral_property_l1441_144119


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l1441_144114

theorem jelly_bean_problem (b c : ℕ) : 
  b = 2 * c →                 -- Initial condition
  b - 5 = 4 * (c - 5) →       -- Condition after eating jelly beans
  b = 30 :=                   -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l1441_144114


namespace NUMINAMATH_CALUDE_sum_in_interval_l1441_144131

theorem sum_in_interval : 
  let sum := 4 + 3/8 + 5 + 3/4 + 7 + 2/25
  17 < sum ∧ sum < 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_interval_l1441_144131


namespace NUMINAMATH_CALUDE_fruit_purchase_theorem_l1441_144112

/-- Calculates the total cost of a fruit purchase with a quantity-based discount --/
def fruitPurchaseCost (lemonPrice papayaPrice mangoPrice : ℕ) 
                      (lemonQty papayaQty mangoQty : ℕ) 
                      (discountPerFruits : ℕ) 
                      (discountAmount : ℕ) : ℕ :=
  let totalFruits := lemonQty + papayaQty + mangoQty
  let totalCost := lemonPrice * lemonQty + papayaPrice * papayaQty + mangoPrice * mangoQty
  let discountCount := totalFruits / discountPerFruits
  let totalDiscount := discountCount * discountAmount
  totalCost - totalDiscount

theorem fruit_purchase_theorem : 
  fruitPurchaseCost 2 1 4 6 4 2 4 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_theorem_l1441_144112


namespace NUMINAMATH_CALUDE_set_operations_l1441_144130

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5,6,7,8}

-- Define set A
def A : Finset Nat := {5,6,7,8}

-- Define set B
def B : Finset Nat := {2,4,6,8}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {6,8}) ∧
  (U \ A = {1,2,3,4}) ∧
  (U \ B = {1,3,5,7}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1441_144130


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_one_l1441_144176

theorem at_least_one_not_less_than_one (x : ℝ) :
  let a := |Real.cos x - Real.sin x|
  let b := |Real.cos x + Real.sin x|
  max a b ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_one_l1441_144176


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1441_144159

theorem sum_of_x_and_y (x y : ℝ) 
  (eq1 : x^2 + y^2 = 16*x - 10*y + 14) 
  (eq2 : x - y = 6) : 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1441_144159


namespace NUMINAMATH_CALUDE_min_value_of_squared_sum_l1441_144161

theorem min_value_of_squared_sum (a b c t : ℝ) 
  (sum_condition : a + b + c = t) 
  (squared_sum_condition : a^2 + b^2 + c^2 = 1) : 
  2 * (a^2 + b^2 + c^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_squared_sum_l1441_144161


namespace NUMINAMATH_CALUDE_calculate_expression_l1441_144163

theorem calculate_expression : -1^4 + 16 / (-2)^3 * |(-3) - 1| = -9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1441_144163


namespace NUMINAMATH_CALUDE_card_ratio_l1441_144184

/-- Proves the ratio of cards eaten by the dog to the total cards before the incident -/
theorem card_ratio (new_cards : ℕ) (remaining_cards : ℕ) : 
  new_cards = 4 → remaining_cards = 34 → 
  (new_cards + remaining_cards - remaining_cards) / (new_cards + remaining_cards) = 2 / 19 := by
  sorry

end NUMINAMATH_CALUDE_card_ratio_l1441_144184


namespace NUMINAMATH_CALUDE_prove_annes_cleaning_time_l1441_144178

/-- Represents the time it takes Anne to clean the house alone -/
def annes_cleaning_time : ℝ := 12

/-- Represents Bruce's cleaning rate (houses per hour) -/
noncomputable def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate (houses per hour) -/
noncomputable def anne_rate : ℝ := sorry

theorem prove_annes_cleaning_time :
  -- Bruce and Anne can clean the house in 4 hours together
  (bruce_rate + anne_rate) * 4 = 1 →
  -- If Anne's speed is doubled, they can clean the house in 3 hours
  (bruce_rate + 2 * anne_rate) * 3 = 1 →
  -- Then Anne's individual cleaning time is 12 hours
  annes_cleaning_time = 1 / anne_rate :=
by sorry

end NUMINAMATH_CALUDE_prove_annes_cleaning_time_l1441_144178


namespace NUMINAMATH_CALUDE_train_length_l1441_144110

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 16 → speed_kmh * (5/18) * time_s = 240 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1441_144110


namespace NUMINAMATH_CALUDE_number_categorization_l1441_144133

def given_numbers : List ℚ := [-18, -3/5, 0, 2023, -22/7, -0.142857, 95/100]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_fraction (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def positive_set : Set ℚ := {x | is_positive x}
def negative_set : Set ℚ := {x | is_negative x}
def integer_set : Set ℚ := {x | is_integer x}
def fraction_set : Set ℚ := {x | is_fraction x}

theorem number_categorization :
  (positive_set ∩ given_numbers.toFinset = {2023, 95/100}) ∧
  (negative_set ∩ given_numbers.toFinset = {-18, -3/5, -22/7, -0.142857}) ∧
  (integer_set ∩ given_numbers.toFinset = {-18, 0, 2023}) ∧
  (fraction_set ∩ given_numbers.toFinset = {-3/5, -22/7, -0.142857, 95/100}) :=
by sorry

end NUMINAMATH_CALUDE_number_categorization_l1441_144133


namespace NUMINAMATH_CALUDE_max_area_difference_l1441_144153

/-- A rectangle with integer dimensions and perimeter 160 cm -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_constraint : length + width = 80

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem stating the maximum difference between areas of two rectangles -/
theorem max_area_difference :
  ∃ (r1 r2 : Rectangle), ∀ (s1 s2 : Rectangle),
    area r1 - area r2 ≥ area s1 - area s2 ∧
    area r1 - area r2 = 1521 := by
  sorry


end NUMINAMATH_CALUDE_max_area_difference_l1441_144153


namespace NUMINAMATH_CALUDE_correct_purchase_ways_l1441_144134

def num_cookie_types : ℕ := 7
def num_cupcake_types : ℕ := 4
def total_items : ℕ := 4

def purchase_ways : ℕ := sorry

theorem correct_purchase_ways : purchase_ways = 4054 := by sorry

end NUMINAMATH_CALUDE_correct_purchase_ways_l1441_144134


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1441_144108

/-- Given a geometric sequence {a_n} with S_n being the sum of its first n terms,
    if a_6 = 8a_3, then S_6 / S_3 = 9 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h_sum : ∀ n, S n = a 1 * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)) 
    (h_ratio : a 6 = 8 * a 3) : 
  S 6 / S 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1441_144108


namespace NUMINAMATH_CALUDE_associate_prof_charts_l1441_144157

theorem associate_prof_charts (
  associate_profs : ℕ) 
  (assistant_profs : ℕ) 
  (charts_per_associate : ℕ) 
  (h1 : associate_profs + assistant_profs = 7)
  (h2 : 2 * associate_profs + assistant_profs = 10)
  (h3 : charts_per_associate * associate_profs + 2 * assistant_profs = 11)
  : charts_per_associate = 1 := by
  sorry

end NUMINAMATH_CALUDE_associate_prof_charts_l1441_144157


namespace NUMINAMATH_CALUDE_sum_squared_expression_lower_bound_l1441_144106

theorem sum_squared_expression_lower_bound 
  (x y z : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (h_sum : x + y + z = x * y * z) : 
  ((x^2 - 1) / x)^2 + ((y^2 - 1) / y)^2 + ((z^2 - 1) / z)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_expression_lower_bound_l1441_144106


namespace NUMINAMATH_CALUDE_equation_solution_l1441_144120

theorem equation_solution : ∃ n : ℝ, 0.03 * n + 0.05 * (30 + n) + 2 = 8.5 ∧ n = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1441_144120


namespace NUMINAMATH_CALUDE_lattice_points_on_segment_l1441_144109

/-- The number of lattice points on a line segment --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem stating the number of lattice points on the given line segment --/
theorem lattice_points_on_segment : latticePointCount 5 13 35 97 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_segment_l1441_144109


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l1441_144103

/-- Represents a school with classes and a student congress. -/
structure School where
  num_classes : ℕ
  students_per_class : ℕ
  students_sent_per_class : ℕ

/-- Calculates the sample size for the Student Congress. -/
def sample_size (s : School) : ℕ :=
  s.num_classes * s.students_sent_per_class

/-- Theorem: The sample size for the given school is 120. -/
theorem student_congress_sample_size :
  let s : School := {
    num_classes := 40,
    students_per_class := 50,
    students_sent_per_class := 3
  }
  sample_size s = 120 := by
  sorry


end NUMINAMATH_CALUDE_student_congress_sample_size_l1441_144103


namespace NUMINAMATH_CALUDE_complete_square_sum_l1441_144172

theorem complete_square_sum (x : ℝ) : ∃ (d e f : ℤ), 
  d > 0 ∧ 
  (25 * x^2 + 30 * x - 72 = 0 ↔ (d * x + e)^2 = f) ∧ 
  d + e + f = 89 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1441_144172


namespace NUMINAMATH_CALUDE_mean_squares_sum_l1441_144105

theorem mean_squares_sum (x y z : ℝ) : 
  (x + y + z) / 3 = 10 →
  (x * y * z) ^ (1/3 : ℝ) = 6 →
  3 / (1/x + 1/y + 1/z) = 4 →
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_mean_squares_sum_l1441_144105


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1441_144154

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.a 3 = 5)
    (h2 : seq.S 6 = 42) :
  seq.S 9 = 117 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1441_144154


namespace NUMINAMATH_CALUDE_measure_10_liters_l1441_144123

/-- Represents the state of water in two containers -/
structure WaterState :=
  (container1 : ℕ)  -- Amount of water in container 1 (11-liter container)
  (container2 : ℕ)  -- Amount of water in container 2 (9-liter container)

/-- Defines the possible operations on the water containers -/
inductive WaterOperation
  | Fill1      -- Fill container 1
  | Fill2      -- Fill container 2
  | Empty1     -- Empty container 1
  | Empty2     -- Empty container 2
  | Pour1to2   -- Pour from container 1 to container 2
  | Pour2to1   -- Pour from container 2 to container 1

/-- Applies a single operation to a water state -/
def applyOperation (state : WaterState) (op : WaterOperation) : WaterState :=
  match op with
  | WaterOperation.Fill1    => { container1 := 11, container2 := state.container2 }
  | WaterOperation.Fill2    => { container1 := state.container1, container2 := 9 }
  | WaterOperation.Empty1   => { container1 := 0,  container2 := state.container2 }
  | WaterOperation.Empty2   => { container1 := state.container1, container2 := 0 }
  | WaterOperation.Pour1to2 => 
      let amount := min state.container1 (9 - state.container2)
      { container1 := state.container1 - amount, container2 := state.container2 + amount }
  | WaterOperation.Pour2to1 => 
      let amount := min state.container2 (11 - state.container1)
      { container1 := state.container1 + amount, container2 := state.container2 - amount }

/-- Theorem: It is possible to measure out exactly 10 liters of water -/
theorem measure_10_liters : ∃ (ops : List WaterOperation), 
  (ops.foldl applyOperation { container1 := 0, container2 := 0 }).container1 = 10 ∨
  (ops.foldl applyOperation { container1 := 0, container2 := 0 }).container2 = 10 :=
sorry

end NUMINAMATH_CALUDE_measure_10_liters_l1441_144123


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1441_144156

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ (r s : ℤ), x^2 + b*x + 1764 = (x + r) * (x + s)) ∧ 
  (∀ (b' : ℕ), b' < b → ¬∃ (r s : ℤ), x^2 + b'*x + 1764 = (x + r) * (x + s)) → 
  b = 84 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l1441_144156


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l1441_144107

/-- Represents the ages of John and Mary -/
structure Ages where
  john : ℕ
  mary : ℕ

/-- The conditions from the problem -/
def age_conditions (a : Ages) : Prop :=
  (a.john - 3 = 2 * (a.mary - 3)) ∧ 
  (a.john - 7 = 3 * (a.mary - 7))

/-- The future condition we're looking for -/
def future_ratio (a : Ages) (years : ℕ) : Prop :=
  3 * (a.mary + years) = 2 * (a.john + years)

/-- The main theorem -/
theorem age_ratio_theorem (a : Ages) :
  age_conditions a → ∃ y : ℕ, y = 5 ∧ future_ratio a y := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l1441_144107


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1441_144181

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the quadratic equation -/
def are_roots (a : ℕ → ℝ) : Prop :=
  3 * (a 1)^2 + 7 * (a 1) - 9 = 0 ∧ 3 * (a 10)^2 + 7 * (a 10) - 9 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → are_roots a → a 4 * a 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1441_144181


namespace NUMINAMATH_CALUDE_max_remaining_pairs_l1441_144166

def original_total_pairs : ℕ := 20
def original_high_heeled_pairs : ℕ := 4
def original_flat_pairs : ℕ := 16
def lost_high_heeled_shoes : ℕ := 5
def lost_flat_shoes : ℕ := 11

def shoes_per_pair : ℕ := 2

theorem max_remaining_pairs : 
  let original_high_heeled_shoes := original_high_heeled_pairs * shoes_per_pair
  let original_flat_shoes := original_flat_pairs * shoes_per_pair
  let remaining_high_heeled_shoes := original_high_heeled_shoes - lost_high_heeled_shoes
  let remaining_flat_shoes := original_flat_shoes - lost_flat_shoes
  let remaining_high_heeled_pairs := remaining_high_heeled_shoes / shoes_per_pair
  let remaining_flat_pairs := remaining_flat_shoes / shoes_per_pair
  remaining_high_heeled_pairs + remaining_flat_pairs = 11 :=
by sorry

end NUMINAMATH_CALUDE_max_remaining_pairs_l1441_144166


namespace NUMINAMATH_CALUDE_proportion_solution_l1441_144168

theorem proportion_solution (x : ℝ) : (0.60 : ℝ) / x = (6 : ℝ) / 2 → x = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1441_144168


namespace NUMINAMATH_CALUDE_dot_path_length_on_rolling_cube_l1441_144177

/-- The path length traced by a dot on a rolling cube. -/
theorem dot_path_length_on_rolling_cube : 
  ∀ (cube_edge_length : ℝ) (dot_distance_from_edge : ℝ),
    cube_edge_length = 2 →
    dot_distance_from_edge = 1 →
    ∃ (path_length : ℝ),
      path_length = 2 * Real.pi * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_dot_path_length_on_rolling_cube_l1441_144177


namespace NUMINAMATH_CALUDE_melissa_bought_four_packs_l1441_144137

/-- The number of packs of tennis balls Melissa bought -/
def num_packs : ℕ := sorry

/-- The total cost of all packs in dollars -/
def total_cost : ℕ := 24

/-- The number of balls in each pack -/
def balls_per_pack : ℕ := 3

/-- The cost of each ball in dollars -/
def cost_per_ball : ℕ := 2

/-- Theorem stating that Melissa bought 4 packs of tennis balls -/
theorem melissa_bought_four_packs : num_packs = 4 := by sorry

end NUMINAMATH_CALUDE_melissa_bought_four_packs_l1441_144137


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l1441_144142

/-- Calculates the total shaded area of a carpet design with given ratios and square counts. -/
theorem carpet_shaded_area (carpet_side : ℝ) (ratio_12_S : ℝ) (ratio_S_T : ℝ) (ratio_T_U : ℝ)
  (count_S : ℕ) (count_T : ℕ) (count_U : ℕ) :
  carpet_side = 12 →
  ratio_12_S = 4 →
  ratio_S_T = 2 →
  ratio_T_U = 2 →
  count_S = 1 →
  count_T = 4 →
  count_U = 8 →
  let S := carpet_side / ratio_12_S
  let T := S / ratio_S_T
  let U := T / ratio_T_U
  count_S * S^2 + count_T * T^2 + count_U * U^2 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l1441_144142


namespace NUMINAMATH_CALUDE_stationery_profit_theorem_l1441_144188

/-- Profit function for a stationery item --/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 600 * x - 8000

/-- Daily sales volume function --/
def sales_volume (x : ℝ) : ℝ := -10 * x + 400

/-- Purchase price of the stationery item --/
def purchase_price : ℝ := 20

/-- Theorem stating the properties of the profit function and its maximum --/
theorem stationery_profit_theorem :
  (∀ x, profit_function x = (x - purchase_price) * sales_volume x) ∧
  (∃ x_max, ∀ x, profit_function x ≤ profit_function x_max ∧ x_max = 30) ∧
  (∃ x_constrained, 
    sales_volume x_constrained ≥ 120 ∧
    (∀ x, sales_volume x ≥ 120 → profit_function x ≤ profit_function x_constrained) ∧
    x_constrained = 28 ∧
    profit_function x_constrained = 960) :=
by sorry

end NUMINAMATH_CALUDE_stationery_profit_theorem_l1441_144188


namespace NUMINAMATH_CALUDE_consecutive_arithmetic_geometric_equality_l1441_144182

theorem consecutive_arithmetic_geometric_equality (a b c : ℝ) : 
  (∃ r : ℝ, b - a = r ∧ c - b = r) →  -- arithmetic progression condition
  (∃ q : ℝ, b / a = q ∧ c / b = q) →  -- geometric progression condition
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_consecutive_arithmetic_geometric_equality_l1441_144182


namespace NUMINAMATH_CALUDE_power_sum_and_division_l1441_144196

theorem power_sum_and_division : 2^10 + 3^6 / 3^2 = 1105 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_division_l1441_144196
