import Mathlib

namespace NUMINAMATH_CALUDE_school_trip_photos_l1996_199611

theorem school_trip_photos (claire lisa robert : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 28) :
  claire = 14 := by
sorry

end NUMINAMATH_CALUDE_school_trip_photos_l1996_199611


namespace NUMINAMATH_CALUDE_complex_number_properties_l1996_199691

theorem complex_number_properties (z : ℂ) (h : (Complex.I - 1) * z = 2 * Complex.I) : 
  (Complex.abs z = Real.sqrt 2) ∧ (z^2 - 2*z + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1996_199691


namespace NUMINAMATH_CALUDE_marble_ratio_l1996_199623

theorem marble_ratio (total : ℕ) (white : ℕ) (removed : ℕ) (remaining : ℕ)
  (h1 : total = 50)
  (h2 : white = 20)
  (h3 : removed = 2 * (white - (total - white - (total - removed - white))))
  (h4 : remaining = 40)
  (h5 : total = remaining + removed) :
  (total - removed - white) = (total - white - (total - removed - white)) :=
by sorry

end NUMINAMATH_CALUDE_marble_ratio_l1996_199623


namespace NUMINAMATH_CALUDE_min_value_of_4x2_plus_y2_l1996_199654

theorem min_value_of_4x2_plus_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 6) :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 6 → 4 * x^2 + y^2 ≤ 4 * a^2 + b^2 ∧ 4 * x^2 + y^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_4x2_plus_y2_l1996_199654


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1996_199645

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x | 1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1996_199645


namespace NUMINAMATH_CALUDE_expression_value_l1996_199698

theorem expression_value (x : ℝ) (h : x^2 + 3*x = 3) : -3*x^2 - 9*x - 2 = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1996_199698


namespace NUMINAMATH_CALUDE_only_set_D_forms_triangle_l1996_199687

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of line segments given in the problem -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 8), (5, 6, 11), (3, 1, 1), (3, 4, 6)]

/-- Theorem stating that only the set (3, 4, 6) can form a triangle -/
theorem only_set_D_forms_triangle :
  ∃! (a b c : ℝ), (a, b, c) ∈ segment_sets ∧ can_form_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_only_set_D_forms_triangle_l1996_199687


namespace NUMINAMATH_CALUDE_dog_age_is_12_l1996_199628

def cat_age : ℕ := 8

def rabbit_age (cat_age : ℕ) : ℕ := cat_age / 2

def dog_age (rabbit_age : ℕ) : ℕ := 3 * rabbit_age

theorem dog_age_is_12 : dog_age (rabbit_age cat_age) = 12 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_is_12_l1996_199628


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1996_199664

-- Define the types for plane and line
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (α : Plane) (a b : Line) (ha : a ≠ b) :
  perpendicular a α → parallel a b → perpendicular b α :=
by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1996_199664


namespace NUMINAMATH_CALUDE_expression_simplification_l1996_199675

theorem expression_simplification : 
  ((3 + 4 + 5 + 6) / 3) + ((3 * 4 + 9) / 4) = 45 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1996_199675


namespace NUMINAMATH_CALUDE_sqrt_special_sum_l1996_199604

def digits_to_num (d : ℕ) (n : ℕ) : ℕ := (10^n - 1) / (10 - 1) * d

theorem sqrt_special_sum (n : ℕ) (h : n > 0) :
  Real.sqrt (digits_to_num 4 (2*n) + digits_to_num 1 (n+1) - digits_to_num 6 n) = 
  digits_to_num 6 (n-1) + 7 :=
sorry

end NUMINAMATH_CALUDE_sqrt_special_sum_l1996_199604


namespace NUMINAMATH_CALUDE_pushup_difference_l1996_199658

-- Define the number of push-ups for each person
def zachary_pushups : ℕ := 51
def john_pushups : ℕ := 69

-- Define David's push-ups in terms of Zachary's
def david_pushups : ℕ := zachary_pushups + 22

-- Theorem to prove
theorem pushup_difference : david_pushups - john_pushups = 4 := by
  sorry

end NUMINAMATH_CALUDE_pushup_difference_l1996_199658


namespace NUMINAMATH_CALUDE_chord_ratio_is_sqrt6_to_2_l1996_199693

-- Define the points and circles
structure PointOnLine where
  position : ℝ

structure Circle where
  center : ℝ
  radius : ℝ

-- Define the problem setup
def setup (A B C D : PointOnLine) (circle_AB circle_BC circle_CD : Circle) :=
  -- Points are on a line and equally spaced
  B.position - A.position = C.position - B.position ∧
  C.position - B.position = D.position - C.position ∧
  -- Circles have diameters AB, BC, and CD
  circle_AB.radius = (B.position - A.position) / 2 ∧
  circle_BC.radius = (C.position - B.position) / 2 ∧
  circle_CD.radius = (D.position - C.position) / 2 ∧
  circle_AB.center = (A.position + B.position) / 2 ∧
  circle_BC.center = (B.position + C.position) / 2 ∧
  circle_CD.center = (C.position + D.position) / 2

-- Define the tangent line and chords
def tangent_and_chords (A : PointOnLine) (circle_CD : Circle) (chord_AB chord_BC : ℝ) :=
  ∃ (l : ℝ → ℝ), 
    -- l is tangent to circle_CD at point A
    (l A.position - circle_CD.center)^2 = circle_CD.radius^2 ∧
    -- chord_AB and chord_BC are the lengths of the chords cut by l on circles with diameters AB and BC
    chord_AB > 0 ∧ chord_BC > 0

-- The main theorem
theorem chord_ratio_is_sqrt6_to_2 
  (A B C D : PointOnLine) 
  (circle_AB circle_BC circle_CD : Circle) 
  (chord_AB chord_BC : ℝ) :
  setup A B C D circle_AB circle_BC circle_CD →
  tangent_and_chords A circle_CD chord_AB chord_BC →
  chord_AB / chord_BC = Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_chord_ratio_is_sqrt6_to_2_l1996_199693


namespace NUMINAMATH_CALUDE_expand_product_l1996_199668

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 6) = 6 * x^2 + 26 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1996_199668


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1996_199615

theorem power_fraction_simplification :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1996_199615


namespace NUMINAMATH_CALUDE_beth_cookie_price_l1996_199600

/-- Represents a cookie batch with a count and price per cookie -/
structure CookieBatch where
  count : ℕ
  price : ℚ

/-- Calculates the total earnings from a cookie batch -/
def totalEarnings (batch : CookieBatch) : ℚ :=
  batch.count * batch.price

theorem beth_cookie_price (alan_batch beth_batch : CookieBatch) : 
  alan_batch.count = 15 → 
  alan_batch.price = 1/2 → 
  beth_batch.count = 18 → 
  totalEarnings alan_batch = totalEarnings beth_batch → 
  beth_batch.price = 21/50 := by
sorry

#eval (21 : ℚ) / 50

end NUMINAMATH_CALUDE_beth_cookie_price_l1996_199600


namespace NUMINAMATH_CALUDE_max_page_number_l1996_199624

/-- The number of '2's available -/
def available_twos : ℕ := 34

/-- The number of '2's used in numbers from 1 to 99 -/
def twos_in_1_to_99 : ℕ := 19

/-- The number of '2's used in numbers from 100 to 199 -/
def twos_in_100_to_199 : ℕ := 10

/-- The highest page number that can be reached with the available '2's -/
def highest_page_number : ℕ := 199

theorem max_page_number :
  available_twos = twos_in_1_to_99 + twos_in_100_to_199 + 5 ∧
  highest_page_number = 199 :=
sorry

end NUMINAMATH_CALUDE_max_page_number_l1996_199624


namespace NUMINAMATH_CALUDE_courtyard_width_l1996_199674

theorem courtyard_width (length : ℝ) (width : ℝ) (stone_length : ℝ) (stone_width : ℝ) 
  (total_stones : ℕ) (h1 : length = 30) (h2 : stone_length = 2) (h3 : stone_width = 1) 
  (h4 : total_stones = 240) (h5 : length * width = stone_length * stone_width * total_stones) : 
  width = 16 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_l1996_199674


namespace NUMINAMATH_CALUDE_inequality_conditions_l1996_199607

theorem inequality_conditions (x y z : ℝ) 
  (h1 : y - x < 1.5 * Real.sqrt (x^2))
  (h2 : z = 2 * (y + x)) :
  (x ≥ 0 → z < 7 * x) ∧ (x < 0 → z < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_conditions_l1996_199607


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1996_199673

theorem solution_set_equivalence (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1996_199673


namespace NUMINAMATH_CALUDE_cake_division_l1996_199619

theorem cake_division (pooh_initial piglet_initial : ℚ) : 
  pooh_initial + piglet_initial = 1 →
  piglet_initial + (1/3) * pooh_initial = 3 * piglet_initial →
  pooh_initial = 6/7 ∧ piglet_initial = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_cake_division_l1996_199619


namespace NUMINAMATH_CALUDE_cos_a3_value_l1996_199610

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem cos_a3_value (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 + a 5 = Real.pi →
  Real.cos (a 3) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_a3_value_l1996_199610


namespace NUMINAMATH_CALUDE_probability_three_heads_five_tosses_l1996_199680

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def probability_k_heads (n k : ℕ) : ℚ :=
  (n.choose k : ℚ) * (1/2)^k * (1/2)^(n-k)

/-- The probability of getting exactly 3 heads in 5 tosses of a fair coin is 5/16 -/
theorem probability_three_heads_five_tosses :
  probability_k_heads 5 3 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_five_tosses_l1996_199680


namespace NUMINAMATH_CALUDE_replaced_person_weight_l1996_199659

/-- Given a group of 8 people, if replacing one person with a new person weighing 89 kg
    increases the average weight by 3 kg, then the replaced person's weight was 65 kg. -/
theorem replaced_person_weight (initial_count : Nat) (new_person_weight : ℝ) (average_increase : ℝ) :
  initial_count = 8 →
  new_person_weight = 89 →
  average_increase = 3 →
  new_person_weight - (initial_count : ℝ) * average_increase = 65 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l1996_199659


namespace NUMINAMATH_CALUDE_taxi_charge_proof_l1996_199620

/-- The charge for each additional 1/5 mile in a taxi ride -/
def additional_fifth_mile_charge : ℚ := 0.40

/-- The initial charge for the first 1/5 mile -/
def initial_charge : ℚ := 3.00

/-- The total charge for an 8-mile ride -/
def total_charge_8_miles : ℚ := 18.60

/-- The length of the ride in miles -/
def ride_length : ℚ := 8

theorem taxi_charge_proof :
  initial_charge + (ride_length * 5 - 1) * additional_fifth_mile_charge = total_charge_8_miles :=
by sorry

end NUMINAMATH_CALUDE_taxi_charge_proof_l1996_199620


namespace NUMINAMATH_CALUDE_compare_fractions_l1996_199633

theorem compare_fractions (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / (a - c) > e / (b - d) := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l1996_199633


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1996_199639

theorem smallest_constant_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) +
  Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) +
  Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) ≥ 2 ∧
  ∀ m : ℝ, m < 2 → ∃ a' b' c' d' e' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ e' > 0 ∧
    Real.sqrt (a' / (b' + c' + d' + e')) +
    Real.sqrt (b' / (a' + c' + d' + e')) +
    Real.sqrt (c' / (a' + b' + d' + e')) +
    Real.sqrt (d' / (a' + b' + c' + e')) +
    Real.sqrt (e' / (a' + b' + c' + d')) < m :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1996_199639


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1996_199694

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27 →
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1996_199694


namespace NUMINAMATH_CALUDE_quadratic_equation_prime_roots_l1996_199640

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem quadratic_equation_prime_roots :
  ∃! k : ℕ, ∃ p q : ℕ,
    is_prime p ∧ is_prime q ∧
    p + q = 58 ∧
    p * q = k ∧
    k = 265 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_prime_roots_l1996_199640


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l1996_199665

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the upstream speed given the rowing speeds in still water and downstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the specific conditions, the upstream speed is 15 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 25) 
  (h2 : s.downstream = 35) : 
  upstreamSpeed s = 15 := by
  sorry

#eval upstreamSpeed { stillWater := 25, downstream := 35 }

end NUMINAMATH_CALUDE_upstream_speed_calculation_l1996_199665


namespace NUMINAMATH_CALUDE_trapezoid_to_square_l1996_199651

/-- A trapezoid composed of a square and a triangle -/
structure Trapezoid where
  square_area : ℝ
  triangle_area : ℝ

/-- The given trapezoid -/
def given_trapezoid : Trapezoid where
  square_area := 4
  triangle_area := 1

/-- The total area of the trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ := t.square_area + t.triangle_area

/-- A function to check if a trapezoid can be rearranged into a square -/
def can_form_square (t : Trapezoid) : Prop :=
  ∃ (side : ℝ), side^2 = trapezoid_area t ∧
  ∃ (a b : ℝ), a^2 + b^2 = side^2 ∧ a * b = t.triangle_area

/-- Theorem: The given trapezoid can be cut and rearranged to form a square -/
theorem trapezoid_to_square :
  can_form_square given_trapezoid := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_to_square_l1996_199651


namespace NUMINAMATH_CALUDE_sector_arc_length_l1996_199627

theorem sector_arc_length (s r p : ℝ) : 
  s = 4 → r = 2 → s = (1/2) * r * p → p = 4 := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l1996_199627


namespace NUMINAMATH_CALUDE_price_after_nine_years_l1996_199697

def initial_price : ℝ := 640
def decrease_rate : ℝ := 0.25
def years : ℕ := 9
def price_after_n_years (n : ℕ) : ℝ := initial_price * (1 - decrease_rate) ^ (n / 3)

theorem price_after_nine_years :
  price_after_n_years years = 270 := by
  sorry

end NUMINAMATH_CALUDE_price_after_nine_years_l1996_199697


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l1996_199634

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 160) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 975 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l1996_199634


namespace NUMINAMATH_CALUDE_afternoon_pear_sales_l1996_199652

/-- Given a salesman who sold pears in the morning and afternoon, this theorem proves
    that if he sold twice as much in the afternoon as in the morning, and the total
    amount sold was 480 kilograms, then he sold 320 kilograms in the afternoon. -/
theorem afternoon_pear_sales (morning_sales afternoon_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  morning_sales + afternoon_sales = 480 →
  afternoon_sales = 320 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_pear_sales_l1996_199652


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1996_199608

/-- Definition of the sum of a geometric sequence -/
def geometric_sum (a : ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  a * (1 - q^n) / (1 - q)

/-- Theorem: Given S_4 = 1 and S_8 = 17, the first term a_1 is either 1/15 or -1/5 -/
theorem geometric_sequence_first_term
  (a : ℚ) (q : ℚ)
  (h1 : geometric_sum a q 4 = 1)
  (h2 : geometric_sum a q 8 = 17) :
  a = 1/15 ∨ a = -1/5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1996_199608


namespace NUMINAMATH_CALUDE_solution_difference_l1996_199647

theorem solution_difference (r s : ℝ) : 
  ((r - 5) * (r + 5) = 26 * r - 130) →
  ((s - 5) * (s + 5) = 26 * s - 130) →
  r ≠ s →
  r > s →
  r - s = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1996_199647


namespace NUMINAMATH_CALUDE_forest_farms_theorem_l1996_199644

-- Define a farm as a pair of natural numbers (total years, high-quality years)
def Farm := ℕ × ℕ

-- Function to calculate probability of selecting two high-quality years
def prob_two_high_quality (f : Farm) : ℚ :=
  let (total, high) := f
  (high.choose 2 : ℚ) / (total.choose 2 : ℚ)

-- Function to calculate probability of selecting a high-quality year
def prob_high_quality (f : Farm) : ℚ :=
  let (total, high) := f
  high / total

-- Distribution type for discrete random variable
def Distribution := List (ℕ × ℚ)

-- Function to calculate the distribution of high-quality projects
def distribution_high_quality (f1 f2 f3 : Farm) : Distribution :=
  sorry  -- Placeholder for the actual calculation

-- Main theorem
theorem forest_farms_theorem (farm_b farm_c : Farm) :
  -- Part 1
  prob_two_high_quality (7, 4) = 2/7 ∧
  -- Part 2
  distribution_high_quality (6, 3) (7, 4) (10, 5) = 
    [(0, 3/28), (1, 5/14), (2, 11/28), (3, 1/7)] ∧
  -- Part 3
  ∃ (avg_b avg_c : ℚ), 
    prob_high_quality farm_b = 4/7 ∧ 
    prob_high_quality farm_c = 1/2 ∧ 
    avg_b ≠ avg_c :=
by sorry

end NUMINAMATH_CALUDE_forest_farms_theorem_l1996_199644


namespace NUMINAMATH_CALUDE_logical_equivalence_l1996_199661

theorem logical_equivalence (P Q R S : Prop) :
  ((P ∨ ¬R) → (¬Q ∧ S)) ↔ ((Q ∨ ¬S) → (¬P ∧ R)) :=
by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l1996_199661


namespace NUMINAMATH_CALUDE_smaller_cuboid_width_l1996_199625

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

theorem smaller_cuboid_width 
  (large : Cuboid)
  (small_length : ℝ)
  (small_height : ℝ)
  (num_small : ℕ)
  (h1 : large.length = 18)
  (h2 : large.width = 15)
  (h3 : large.height = 2)
  (h4 : small_length = 5)
  (h5 : small_height = 3)
  (h6 : num_small = 18)
  (h7 : volume large = num_small * volume { length := small_length, width := 2, height := small_height }) :
  ∃ (small : Cuboid), small.length = small_length ∧ small.height = small_height ∧ small.width = 2 :=
sorry

end NUMINAMATH_CALUDE_smaller_cuboid_width_l1996_199625


namespace NUMINAMATH_CALUDE_average_weight_of_class_l1996_199614

theorem average_weight_of_class (n1 : ℕ) (n2 : ℕ) (w1 : ℚ) (w2 : ℚ) :
  n1 = 24 →
  n2 = 8 →
  w1 = 50.25 →
  w2 = 45.15 →
  (n1 * w1 + n2 * w2) / (n1 + n2 : ℚ) = 48.975 := by sorry

end NUMINAMATH_CALUDE_average_weight_of_class_l1996_199614


namespace NUMINAMATH_CALUDE_arun_age_is_sixty_l1996_199622

/-- Given the ages of Arun, Gokul, and Madan, prove that Arun's age is 60 years. -/
theorem arun_age_is_sixty (arun_age gokul_age madan_age : ℕ) : 
  ((arun_age - 6) / 18 = gokul_age) →
  (gokul_age = madan_age - 2) →
  (madan_age = 5) →
  arun_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_arun_age_is_sixty_l1996_199622


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l1996_199613

def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | |x| < 2}

theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l1996_199613


namespace NUMINAMATH_CALUDE_log_lt_x_div_one_minus_x_l1996_199662

theorem log_lt_x_div_one_minus_x (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  Real.log (1 + x) < x / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_log_lt_x_div_one_minus_x_l1996_199662


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1996_199605

theorem isosceles_triangle_perimeter : ∀ x : ℝ,
  x^2 - 8*x + 15 = 0 →
  x > 0 →
  x < 4 →
  2 + 2 + x = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1996_199605


namespace NUMINAMATH_CALUDE_bisected_parallelogram_perimeter_l1996_199666

/-- Represents a parallelogram with a bisected angle -/
structure BisectedParallelogram where
  -- The length of one segment created by the angle bisector
  segment1 : ℝ
  -- The length of the other segment created by the angle bisector
  segment2 : ℝ
  -- Assumption that the segments are 7 and 14 (in either order)
  h_segments : (segment1 = 7 ∧ segment2 = 14) ∨ (segment1 = 14 ∧ segment2 = 7)

/-- The perimeter of the parallelogram is either 56 or 70 -/
theorem bisected_parallelogram_perimeter (p : BisectedParallelogram) :
  let perimeter := 2 * (p.segment1 + p.segment2)
  perimeter = 56 ∨ perimeter = 70 := by
  sorry


end NUMINAMATH_CALUDE_bisected_parallelogram_perimeter_l1996_199666


namespace NUMINAMATH_CALUDE_absolute_value_plus_exponent_l1996_199688

theorem absolute_value_plus_exponent : |(-2 : ℝ)| + (π - 3)^(0 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_plus_exponent_l1996_199688


namespace NUMINAMATH_CALUDE_junyoung_remaining_pencils_l1996_199603

/-- Calculates the number of remaining pencils after giving some away -/
def remaining_pencils (initial_dozens : ℕ) (given_dozens : ℕ) (given_individual : ℕ) : ℕ :=
  initial_dozens * 12 - (given_dozens * 12 + given_individual)

/-- Theorem stating that given the initial conditions, 75 pencils remain -/
theorem junyoung_remaining_pencils :
  remaining_pencils 11 4 9 = 75 := by
  sorry

end NUMINAMATH_CALUDE_junyoung_remaining_pencils_l1996_199603


namespace NUMINAMATH_CALUDE_new_room_area_l1996_199699

/-- Given a bedroom and bathroom area, calculate the area of a new room that is twice as large as their combined area -/
theorem new_room_area (bedroom_area bathroom_area : ℕ) : 
  bedroom_area = 309 → bathroom_area = 150 → 
  2 * (bedroom_area + bathroom_area) = 918 := by
  sorry

end NUMINAMATH_CALUDE_new_room_area_l1996_199699


namespace NUMINAMATH_CALUDE_count_lattice_points_on_hyperbola_l1996_199655

def lattice_points_on_hyperbola : ℕ :=
  let a := 3000
  2 * (((2 + 1) * (2 + 1) * (6 + 1)) : ℕ)

theorem count_lattice_points_on_hyperbola :
  lattice_points_on_hyperbola = 126 := by sorry

end NUMINAMATH_CALUDE_count_lattice_points_on_hyperbola_l1996_199655


namespace NUMINAMATH_CALUDE_tourist_survival_l1996_199689

theorem tourist_survival (initial : ℕ) (eaten : ℕ) (poison_fraction : ℚ) (recovery_fraction : ℚ) : 
  initial = 30 →
  eaten = 2 →
  poison_fraction = 1/2 →
  recovery_fraction = 1/7 →
  (initial - eaten - (initial - eaten) * poison_fraction + 
   (initial - eaten) * poison_fraction * recovery_fraction : ℚ) = 16 := by
sorry

end NUMINAMATH_CALUDE_tourist_survival_l1996_199689


namespace NUMINAMATH_CALUDE_intersection_A_B_l1996_199670

def set_A : Set ℝ := {x | x^2 - 2*x < 0}
def set_B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_A_B : set_A ∩ set_B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1996_199670


namespace NUMINAMATH_CALUDE_probability_one_white_one_black_l1996_199653

/-- The probability of drawing one white ball and one black ball from a bag -/
theorem probability_one_white_one_black (white_balls black_balls : ℕ) :
  white_balls = 6 →
  black_balls = 5 →
  (white_balls.choose 1 * black_balls.choose 1 : ℚ) / (white_balls + black_balls).choose 2 = 6/11 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_white_one_black_l1996_199653


namespace NUMINAMATH_CALUDE_tan_squared_sum_l1996_199671

theorem tan_squared_sum (x y : ℝ) 
  (h : 2 * Real.sin x * Real.sin y + 3 * Real.cos y + 6 * Real.cos x * Real.sin y = 7) : 
  Real.tan x ^ 2 + 2 * Real.tan y ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_sum_l1996_199671


namespace NUMINAMATH_CALUDE_triangle_properties_l1996_199626

/-- Triangle with vertices A(4, 0), B(8, 10), and C(0, 6) -/
structure Triangle where
  A : Prod ℝ ℝ := (4, 0)
  B : Prod ℝ ℝ := (8, 10)
  C : Prod ℝ ℝ := (0, 6)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.altitudeFromAtoBC (t : Triangle) : LineEquation :=
  { a := 2, b := -3, c := 14 }

def Triangle.lineParallelToBCThroughA (t : Triangle) : LineEquation :=
  { a := 1, b := -2, c := -4 }

def Triangle.altitudeFromBtoAC (t : Triangle) : LineEquation :=
  { a := 2, b := 1, c := -8 }

theorem triangle_properties (t : Triangle) : 
  (t.altitudeFromAtoBC = { a := 2, b := -3, c := 14 }) ∧ 
  (t.lineParallelToBCThroughA = { a := 1, b := -2, c := -4 }) ∧
  (t.altitudeFromBtoAC = { a := 2, b := 1, c := -8 }) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1996_199626


namespace NUMINAMATH_CALUDE_grass_carp_probability_l1996_199667

/-- Represents a fish pond with grass carp, carp, and crucian carp -/
structure FishPond where
  grass_carp : ℕ
  carp : ℕ
  crucian_carp : ℕ

/-- The probability of catching a specific type of fish in the pond -/
def catch_probability (pond : FishPond) (fish_count : ℕ) : ℚ :=
  fish_count / (pond.grass_carp + pond.carp + pond.crucian_carp)

/-- The main theorem about the probability of catching a grass carp -/
theorem grass_carp_probability (pond : FishPond) :
  pond.grass_carp = 1000 →
  pond.carp = 500 →
  catch_probability pond pond.crucian_carp = 1/4 →
  catch_probability pond pond.grass_carp = 1/2 := by
  sorry

#check grass_carp_probability

end NUMINAMATH_CALUDE_grass_carp_probability_l1996_199667


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1996_199650

theorem quadratic_inequality_solution (x : ℤ) :
  1 ≤ x ∧ x ≤ 10 → (x^2 < 3*x ↔ x = 1 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1996_199650


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l1996_199692

/-- Represents the number of bottle caps Danny found at the park -/
def new_bottle_caps : ℕ := 50

/-- Represents the number of old bottle caps Danny threw away -/
def thrown_away_caps : ℕ := 6

/-- Represents the current number of bottle caps in Danny's collection -/
def current_collection : ℕ := 60

/-- Represents the difference between found and thrown away caps -/
def difference_found_thrown : ℕ := 44

theorem danny_bottle_caps :
  new_bottle_caps = thrown_away_caps + difference_found_thrown ∧
  current_collection = (new_bottle_caps + thrown_away_caps) - thrown_away_caps :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l1996_199692


namespace NUMINAMATH_CALUDE_city_population_change_l1996_199676

theorem city_population_change (n : ℕ) : 
  (0.85 * (n + 1500) : ℚ).floor = n - 50 → n = 8833 := by
  sorry

end NUMINAMATH_CALUDE_city_population_change_l1996_199676


namespace NUMINAMATH_CALUDE_greaterThanOne_is_random_event_l1996_199618

-- Define the type for outcomes of rolling a die
def DieOutcome := Fin 6

-- Define the event "greater than 1"
def greaterThanOne (outcome : DieOutcome) : Prop := outcome.val > 1

-- Define what it means for an event to be random
def isRandomEvent (event : DieOutcome → Prop) : Prop :=
  ∃ (o1 o2 : DieOutcome), event o1 ∧ ¬event o2

-- Theorem stating that "greater than 1" is a random event
theorem greaterThanOne_is_random_event : isRandomEvent greaterThanOne := by
  sorry


end NUMINAMATH_CALUDE_greaterThanOne_is_random_event_l1996_199618


namespace NUMINAMATH_CALUDE_symmetry_line_is_common_chord_l1996_199642

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 8
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the line of symmetry
def is_line_of_symmetry (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), circle1 x y ↔ ∃ (x' y' : ℝ), circle2 x' y' ∧ l x y ∧ l x' y'

-- Define the common chord
def is_common_chord (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), l x y → (circle1 x y ∧ circle2 x y)

-- Theorem statement
theorem symmetry_line_is_common_chord :
  ∀ (l : ℝ → ℝ → Prop), is_line_of_symmetry l → is_common_chord l :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_is_common_chord_l1996_199642


namespace NUMINAMATH_CALUDE_age_ratio_sydney_sherry_l1996_199649

/-- Given the ages of Randolph, Sydney, and Sherry, prove the ratio of Sydney's age to Sherry's age -/
theorem age_ratio_sydney_sherry :
  ∀ (randolph sydney sherry : ℕ),
    randolph = sydney + 5 →
    randolph = 55 →
    sherry = 25 →
    sydney / sherry = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sydney_sherry_l1996_199649


namespace NUMINAMATH_CALUDE_calculate_expression_solve_equation_l1996_199686

-- Problem 1
theorem calculate_expression : 2 * (-3)^2 - 4 * (-3) - 15 = 15 := by sorry

-- Problem 2
theorem solve_equation :
  ∀ x : ℚ, (4 - 2*x) / 3 - x = 1 → x = 1/5 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_equation_l1996_199686


namespace NUMINAMATH_CALUDE_expression_necessarily_negative_l1996_199631

theorem expression_necessarily_negative (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_necessarily_negative_l1996_199631


namespace NUMINAMATH_CALUDE_go_stones_problem_l1996_199629

theorem go_stones_problem (x : ℕ) (h1 : (x / 7 + 40) * 5 = 555) (h2 : x ≥ 55) : x - 55 = 442 := by
  sorry

end NUMINAMATH_CALUDE_go_stones_problem_l1996_199629


namespace NUMINAMATH_CALUDE_josie_remaining_money_l1996_199648

/-- Calculates the remaining money after Josie's grocery shopping --/
def remaining_money (initial_amount : ℚ) 
  (milk_price : ℚ) (milk_discount : ℚ) 
  (bread_price : ℚ) 
  (detergent_price : ℚ) (detergent_coupon : ℚ) 
  (banana_price_per_pound : ℚ) (banana_pounds : ℚ) : ℚ :=
  let milk_cost := milk_price * (1 - milk_discount)
  let detergent_cost := detergent_price - detergent_coupon
  let banana_cost := banana_price_per_pound * banana_pounds
  let total_cost := milk_cost + bread_price + detergent_cost + banana_cost
  initial_amount - total_cost

/-- Theorem stating that Josie has $4.00 left after shopping --/
theorem josie_remaining_money :
  remaining_money 20 4 (1/2) 3.5 10.25 1.25 0.75 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_josie_remaining_money_l1996_199648


namespace NUMINAMATH_CALUDE_cone_max_volume_surface_ratio_l1996_199669

/-- For a cone with slant height 2, the ratio of its volume to its lateral surface area
    is maximized when the radius of its base is √2. -/
theorem cone_max_volume_surface_ratio (r : ℝ) (h : ℝ) : 
  let l : ℝ := 2
  let S := 2 * Real.pi * r
  let V := (1/3) * Real.pi * r^2 * Real.sqrt (l^2 - r^2)
  (∀ r' : ℝ, 0 < r' → V / S ≤ ((1/3) * Real.pi * r'^2 * Real.sqrt (l^2 - r'^2)) / (2 * Real.pi * r')) →
  r = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_cone_max_volume_surface_ratio_l1996_199669


namespace NUMINAMATH_CALUDE_hemisphere_with_disk_surface_area_l1996_199612

/-- Given a hemisphere with base area 144π and an attached circular disk of radius 5,
    the total exposed surface area is 313π. -/
theorem hemisphere_with_disk_surface_area :
  ∀ (r : ℝ) (disk_radius : ℝ),
    r > 0 →
    disk_radius > 0 →
    π * r^2 = 144 * π →
    disk_radius = 5 →
    2 * π * r^2 + π * disk_radius^2 = 313 * π :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_with_disk_surface_area_l1996_199612


namespace NUMINAMATH_CALUDE_abs_sum_complex_roots_l1996_199679

/-- Given complex numbers a, b, and c satisfying certain conditions,
    prove that |a + b + c| is either 0 or 1. -/
theorem abs_sum_complex_roots (a b c : ℂ) 
    (h1 : Complex.abs a = 1)
    (h2 : Complex.abs b = 1)
    (h3 : Complex.abs c = 1)
    (h4 : a^2 * b + b^2 * c + c^2 * a = 0) :
    Complex.abs (a + b + c) = 0 ∨ Complex.abs (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_complex_roots_l1996_199679


namespace NUMINAMATH_CALUDE_tan_x0_equals_3_l1996_199678

/-- Given a function f(x) = sin x - cos x, prove that if f''(x₀) = 2f(x₀), then tan x₀ = 3 -/
theorem tan_x0_equals_3 (x₀ : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin x - Real.cos x
  (deriv (deriv f)) x₀ = 2 * f x₀ → Real.tan x₀ = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x0_equals_3_l1996_199678


namespace NUMINAMATH_CALUDE_necessary_condition_for_inequality_l1996_199602

theorem necessary_condition_for_inequality (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_inequality_l1996_199602


namespace NUMINAMATH_CALUDE_f_lower_bound_solution_set_range_l1996_199681

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: f(x) ≥ 1 for all x
theorem f_lower_bound : ∀ x : ℝ, f x ≥ 1 := by sorry

-- Define the set of x values that satisfy the equation
def solution_set : Set ℝ := {x | ∃ a : ℝ, f x = (a^2 + 2) / Real.sqrt (a^2 + 1)}

-- Theorem 2: The solution set is equal to (-∞, 1/2] ∪ [5/2, +∞)
theorem solution_set_range : solution_set = Set.Iic (1/2) ∪ Set.Ici (5/2) := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_solution_set_range_l1996_199681


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l1996_199641

/-- Represents the state of the switch -/
inductive SwitchState
| Left
| Right

/-- Represents a trainee's action in the room -/
inductive TraineeAction
| FlipSwitch
| DoNothing
| Declare

/-- Represents the result of the challenge -/
inductive ChallengeResult
| Success
| Failure

/-- The strategy function type for a trainee -/
def TraineeStrategy := Nat → SwitchState → TraineeAction

/-- The type representing the challenge setup -/
structure Challenge where
  numTrainees : Nat
  initialState : SwitchState

/-- The function to simulate the challenge -/
noncomputable def simulateChallenge (c : Challenge) (strategies : List TraineeStrategy) : ChallengeResult :=
  sorry

/-- The main theorem to prove -/
theorem exists_winning_strategy :
  ∃ (strategies : List TraineeStrategy),
    strategies.length = 42 ∧
    ∀ (c : Challenge),
      c.numTrainees = 42 →
      simulateChallenge c strategies = ChallengeResult.Success :=
sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_l1996_199641


namespace NUMINAMATH_CALUDE_stock_price_change_l1996_199630

theorem stock_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  let day1 := initial_price * (1 - 0.25)
  let day2 := day1 * (1 + 0.40)
  let day3 := day2 * (1 - 0.10)
  (day3 - initial_price) / initial_price = -0.055 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l1996_199630


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l1996_199617

/-- The number of vertices in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def r : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n r

theorem pentadecagon_triangles : num_triangles = 455 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l1996_199617


namespace NUMINAMATH_CALUDE_find_certain_number_l1996_199621

theorem find_certain_number (G N : ℕ) (h1 : G = 88) (h2 : N % G = 31) (h3 : 4521 % G = 33) : N = 4519 := by
  sorry

end NUMINAMATH_CALUDE_find_certain_number_l1996_199621


namespace NUMINAMATH_CALUDE_smallest_positive_solution_floor_equation_l1996_199682

theorem smallest_positive_solution_floor_equation :
  ∃ x : ℝ, x > 0 ∧
    (∀ y : ℝ, y > 0 → ⌊y^2⌋ - ⌊y⌋^2 = 25 → x ≤ y) ∧
    ⌊x^2⌋ - ⌊x⌋^2 = 25 ∧
    x = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_floor_equation_l1996_199682


namespace NUMINAMATH_CALUDE_substance_volume_l1996_199683

/-- Given a substance where 1 gram occupies 10 cubic centimeters, 
    prove that 100 kg of this substance occupies 1 cubic meter. -/
theorem substance_volume (substance : Type) 
  (volume : substance → ℝ) 
  (mass : substance → ℝ) 
  (s : substance) 
  (h1 : volume s = mass s * 10 * 1000000⁻¹) 
  (h2 : mass s = 100) : 
  volume s = 1 := by
sorry

end NUMINAMATH_CALUDE_substance_volume_l1996_199683


namespace NUMINAMATH_CALUDE_board_symbols_l1996_199632

theorem board_symbols (total : Nat) (plus minus : Nat → Prop) : 
  total = 23 →
  (∀ n : Nat, n ≤ total → plus n ∨ minus n) →
  (∀ s : Finset Nat, s.card = 10 → ∃ i ∈ s, plus i) →
  (∀ s : Finset Nat, s.card = 15 → ∃ i ∈ s, minus i) →
  (∃! p m : Nat, p + m = total ∧ plus = λ i => i < p ∧ minus = λ i => p ≤ i ∧ i < total ∧ p = 14 ∧ m = 9) :=
by sorry

end NUMINAMATH_CALUDE_board_symbols_l1996_199632


namespace NUMINAMATH_CALUDE_factorization_equality_l1996_199684

theorem factorization_equality (x y : ℝ) : 2 * x^3 - 8 * x^2 * y + 8 * x * y^2 = 2 * x * (x - 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1996_199684


namespace NUMINAMATH_CALUDE_point_on_curve_l1996_199690

theorem point_on_curve : 
  let x : ℝ := Real.sqrt 2
  let y : ℝ := Real.sqrt 2
  x^2 + y^2 - 3*x*y + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_point_on_curve_l1996_199690


namespace NUMINAMATH_CALUDE_gcd_7524_16083_l1996_199695

theorem gcd_7524_16083 : Nat.gcd 7524 16083 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7524_16083_l1996_199695


namespace NUMINAMATH_CALUDE_range_of_a_l1996_199656

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1996_199656


namespace NUMINAMATH_CALUDE_min_sum_a1_a2_l1996_199609

/-- The sequence (aᵢ) is defined by aₙ₊₂ = (aₙ + 3007) / (1 + aₙ₊₁) for n ≥ 1, where all aᵢ are positive integers. -/
def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, a (n + 2) = (a n + 3007) / (1 + a (n + 1))

/-- The minimum possible value of a₁ + a₂ is 114. -/
theorem min_sum_a1_a2 :
  ∀ a : ℕ → ℕ, is_valid_sequence a → a 1 + a 2 ≥ 114 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a1_a2_l1996_199609


namespace NUMINAMATH_CALUDE_problem_statement_l1996_199637

theorem problem_statement (a b : ℝ) 
  (h1 : a - b = 1) 
  (h2 : a^2 - b^2 = -1) : 
  a^4 - b^4 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1996_199637


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1996_199638

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1996_199638


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1996_199660

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x ≤ 1 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1996_199660


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1996_199696

theorem polar_to_rectangular_conversion (r θ : ℝ) (h1 : r = 5) (h2 : θ = 5 * π / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1996_199696


namespace NUMINAMATH_CALUDE_max_value_problem_l1996_199636

theorem max_value_problem (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (∃ (x y z : ℝ), 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 ∧ 3 * x + 4 * y + 5 * z > 3 * a + 4 * b + 5 * c) →
  3 * a + 4 * b + 5 * c ≤ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l1996_199636


namespace NUMINAMATH_CALUDE_arrange_five_from_ten_eq_30240_l1996_199606

/-- The number of ways to arrange 5 distinct numbers from a set of 10 numbers -/
def arrange_five_from_ten : ℕ := 10 * 9 * 8 * 7 * 6

/-- Theorem stating that arranging 5 distinct numbers from a set of 10 numbers results in 30240 possibilities -/
theorem arrange_five_from_ten_eq_30240 : arrange_five_from_ten = 30240 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_from_ten_eq_30240_l1996_199606


namespace NUMINAMATH_CALUDE_louisa_first_day_travel_l1996_199657

/-- Represents Louisa's travel details -/
structure LouisaTravel where
  first_day_miles : ℝ
  second_day_miles : ℝ
  average_speed : ℝ
  time_difference : ℝ

/-- Theorem stating that Louisa traveled 200 miles on the first day -/
theorem louisa_first_day_travel (t : LouisaTravel) 
  (h1 : t.second_day_miles = 350)
  (h2 : t.average_speed = 50)
  (h3 : t.time_difference = 3)
  (h4 : t.second_day_miles / t.average_speed = t.first_day_miles / t.average_speed + t.time_difference) :
  t.first_day_miles = 200 := by
  sorry

#check louisa_first_day_travel

end NUMINAMATH_CALUDE_louisa_first_day_travel_l1996_199657


namespace NUMINAMATH_CALUDE_sixth_degree_polynomial_identity_l1996_199672

theorem sixth_degree_polynomial_identity (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
     (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) : 
  b₁^2 + b₂^2 + b₃^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sixth_degree_polynomial_identity_l1996_199672


namespace NUMINAMATH_CALUDE_square_dissection_theorem_l1996_199646

/-- A dissection of a square is a list of polygons that can be rearranged to form the original square. -/
def Dissection (n : ℕ) := List (List (ℕ × ℕ))

/-- A function that checks if a list of polygons can be arranged to form a square of side length n. -/
def CanFormSquare (pieces : List (List (ℕ × ℕ))) (n : ℕ) : Prop := sorry

/-- A function that checks if two lists of polygons are equivalent up to translation and rotation. -/
def AreEquivalent (pieces1 pieces2 : List (List (ℕ × ℕ))) : Prop := sorry

theorem square_dissection_theorem :
  ∃ (d : Dissection 7),
    d.length ≤ 5 ∧
    ∃ (s1 s2 s3 : List (List (ℕ × ℕ))),
      CanFormSquare s1 6 ∧
      CanFormSquare s2 3 ∧
      CanFormSquare s3 2 ∧
      AreEquivalent (s1 ++ s2 ++ s3) d :=
sorry

end NUMINAMATH_CALUDE_square_dissection_theorem_l1996_199646


namespace NUMINAMATH_CALUDE_sqrt_31_minus_2_range_l1996_199663

theorem sqrt_31_minus_2_range : 
  (∃ x : ℝ, x = Real.sqrt 31 ∧ 5 < x ∧ x < 6) → 
  3 < Real.sqrt 31 - 2 ∧ Real.sqrt 31 - 2 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_31_minus_2_range_l1996_199663


namespace NUMINAMATH_CALUDE_pet_store_feet_count_l1996_199601

/-- A pet store sells dogs and parakeets. -/
structure PetStore :=
  (dogs : ℕ)
  (parakeets : ℕ)

/-- Calculate the total number of feet in the pet store. -/
def total_feet (store : PetStore) : ℕ :=
  4 * store.dogs + 2 * store.parakeets

/-- Theorem: Given 15 total heads and 9 dogs, the total number of feet is 48. -/
theorem pet_store_feet_count :
  ∀ (store : PetStore),
  store.dogs + store.parakeets = 15 →
  store.dogs = 9 →
  total_feet store = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_pet_store_feet_count_l1996_199601


namespace NUMINAMATH_CALUDE_iron_rod_weight_l1996_199635

/-- The weight of an iron rod given its length, cross-sectional area, and specific gravity -/
theorem iron_rod_weight 
  (length : Real) 
  (cross_sectional_area : Real) 
  (specific_gravity : Real) 
  (h1 : length = 1) -- 1 m
  (h2 : cross_sectional_area = 188) -- 188 cm²
  (h3 : specific_gravity = 7.8) -- 7.8 kp/dm³
  : Real :=
  let weight := 0.78 * cross_sectional_area
  have weight_eq : weight = 146.64 := by sorry
  weight

#check iron_rod_weight

end NUMINAMATH_CALUDE_iron_rod_weight_l1996_199635


namespace NUMINAMATH_CALUDE_fourth_root_equation_l1996_199685

theorem fourth_root_equation (P : ℝ) : (P^3)^(1/4) = 81 * 81^(1/16) → P = 27 * 3^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l1996_199685


namespace NUMINAMATH_CALUDE_complex_distance_range_l1996_199616

theorem complex_distance_range (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ Complex.abs (1 + Complex.I * Real.sqrt 3 + z) = x :=
sorry

end NUMINAMATH_CALUDE_complex_distance_range_l1996_199616


namespace NUMINAMATH_CALUDE_max_profit_at_seventh_grade_l1996_199677

/-- Represents the profit function for a product with different quality grades. -/
def profit_function (x : ℕ) : ℝ :=
  let profit_per_unit := 6 + 2 * (x - 1)
  let units_produced := 60 - 4 * (x - 1)
  profit_per_unit * units_produced

/-- Represents the maximum grade available. -/
def max_grade : ℕ := 10

/-- Theorem stating that the 7th grade maximizes profit and the maximum profit is 648 yuan. -/
theorem max_profit_at_seventh_grade :
  (∃ (x : ℕ), x ≤ max_grade ∧ ∀ (y : ℕ), y ≤ max_grade → profit_function x ≥ profit_function y) ∧
  (∃ (x : ℕ), x ≤ max_grade ∧ profit_function x = 648) ∧
  profit_function 7 = 648 := by
  sorry

#eval profit_function 7  -- Should output 648

end NUMINAMATH_CALUDE_max_profit_at_seventh_grade_l1996_199677


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1996_199643

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 9375) (h4 : y / x = 15) : x + y = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1996_199643
