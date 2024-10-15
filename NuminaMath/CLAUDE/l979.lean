import Mathlib

namespace NUMINAMATH_CALUDE_intersection_complement_equality_l979_97912

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {1, 2}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l979_97912


namespace NUMINAMATH_CALUDE_c_work_time_l979_97984

-- Define the work rates for each worker
def work_rate_a : ℚ := 1 / 36
def work_rate_b : ℚ := 1 / 18

-- Define the combined work rate
def combined_work_rate : ℚ := 1 / 4

-- Define the relationship between c and d's work rates
def d_work_rate (c : ℚ) : ℚ := c / 2

-- Theorem statement
theorem c_work_time :
  ∃ (c : ℚ), 
    work_rate_a + work_rate_b + c + d_work_rate c = combined_work_rate ∧
    c = 1 / 9 :=
by sorry

end NUMINAMATH_CALUDE_c_work_time_l979_97984


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l979_97989

theorem complex_purely_imaginary (a : ℝ) : 
  (Complex.I * (2 * a + 1) : ℂ) = (2 + Complex.I) * (1 + a * Complex.I) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l979_97989


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l979_97987

/-- The polynomial x^3 - 8x^2 + 17x - 14 -/
def polynomial (x : ℝ) : ℝ := x^3 - 8*x^2 + 17*x - 14

/-- The sum of the kth powers of the roots -/
def s (k : ℕ) : ℝ := sorry

/-- The relation between consecutive s_k values -/
def relation (a b c : ℝ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → s (k+1) = a * s k + b * s (k-1) + c * s (k-2)

theorem sum_of_coefficients :
  ∃ (a b c : ℝ),
    s 0 = 3 ∧ s 1 = 8 ∧ s 2 = 17 ∧
    relation a b c ∧
    a + b + c = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l979_97987


namespace NUMINAMATH_CALUDE_younger_son_age_in_30_years_l979_97925

/-- Given an elder son's age and the age difference between two sons, 
    calculate the younger son's age after a certain number of years. -/
def younger_son_future_age (elder_son_age : ℕ) (age_difference : ℕ) (years_from_now : ℕ) : ℕ :=
  (elder_son_age - age_difference) + years_from_now

theorem younger_son_age_in_30_years :
  younger_son_future_age 40 10 30 = 60 := by
  sorry

end NUMINAMATH_CALUDE_younger_son_age_in_30_years_l979_97925


namespace NUMINAMATH_CALUDE_cannon_hit_probability_l979_97934

theorem cannon_hit_probability (P1 P2 P3 : ℝ) 
  (h1 : P2 = 0.2)
  (h2 : P3 = 0.3)
  (h3 : (1 - P1) * (1 - P2) * (1 - P3) = 0.28) :
  P1 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_cannon_hit_probability_l979_97934


namespace NUMINAMATH_CALUDE_range_of_a_l979_97997

theorem range_of_a (p q : Prop) (h1 : ∀ x : ℝ, x > 0 → x + 1/x > a → a < 2)
  (h2 : (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) → (a ≤ -1 ∨ a ≥ 1))
  (h3 : q) (h4 : ¬p) : a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l979_97997


namespace NUMINAMATH_CALUDE_trigonometric_identities_l979_97933

theorem trigonometric_identities :
  (((Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) ^ 2) = 1/2) ∧
  ((Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3)) = -1) ∧
  ((Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - Real.cos (66 * π / 180) * Real.cos (54 * π / 180)) = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l979_97933


namespace NUMINAMATH_CALUDE_first_week_gain_l979_97938

/-- Proves that the percentage gain in the first week was 25% --/
theorem first_week_gain (initial_investment : ℝ) (final_value : ℝ) : 
  initial_investment = 400 →
  final_value = 750 →
  ∃ (x : ℝ), 
    (initial_investment + x / 100 * initial_investment) * 1.5 = final_value ∧
    x = 25 := by
  sorry

#check first_week_gain

end NUMINAMATH_CALUDE_first_week_gain_l979_97938


namespace NUMINAMATH_CALUDE_newer_train_theorem_l979_97945

/-- Calculates the distance traveled by a newer train given the distance of an older train and the percentage increase in distance. -/
def newer_train_distance (old_distance : ℝ) (percent_increase : ℝ) : ℝ :=
  old_distance * (1 + percent_increase)

/-- Theorem stating that a newer train traveling 30% farther than an older train that goes 180 miles will travel 234 miles. -/
theorem newer_train_theorem :
  newer_train_distance 180 0.3 = 234 := by
  sorry

#eval newer_train_distance 180 0.3

end NUMINAMATH_CALUDE_newer_train_theorem_l979_97945


namespace NUMINAMATH_CALUDE_blue_beads_count_l979_97979

theorem blue_beads_count (total_beads blue_beads yellow_beads : ℕ) : 
  yellow_beads = 16 →
  total_beads = blue_beads + yellow_beads →
  total_beads % 3 = 0 →
  (total_beads / 3 - 10) * 2 = 6 →
  blue_beads = 23 := by
sorry

end NUMINAMATH_CALUDE_blue_beads_count_l979_97979


namespace NUMINAMATH_CALUDE_total_cost_theorem_l979_97954

/-- Calculates the total cost of items with tax --/
def total_cost_with_tax (prices : List ℝ) (tax_rate : ℝ) : ℝ :=
  let subtotal := prices.sum
  let tax_amount := subtotal * tax_rate
  subtotal + tax_amount

/-- Theorem: The total cost of three items with given prices and 5% tax is $15.75 --/
theorem total_cost_theorem :
  let prices := [4.20, 7.60, 3.20]
  let tax_rate := 0.05
  total_cost_with_tax prices tax_rate = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l979_97954


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l979_97906

/-- Given a quadratic inequality ax² + bx + 1 > 0 with solution set {x | -1 < x < 1/3},
    prove that ab = 6 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + b * x + 1 > 0) ↔ (-1 < x ∧ x < 1/3)) → 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l979_97906


namespace NUMINAMATH_CALUDE_candle_recycling_l979_97977

def original_candle_weight : ℝ := 20
def wax_percentage : ℝ := 0.1
def num_candles : ℕ := 5
def new_candle_weight : ℝ := 5

theorem candle_recycling :
  (↑num_candles * original_candle_weight * wax_percentage) / new_candle_weight = 3 := by
  sorry

end NUMINAMATH_CALUDE_candle_recycling_l979_97977


namespace NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l979_97950

theorem second_smallest_hot_dog_packs : ∃ (n : ℕ), n > 0 ∧
  (12 * n) % 8 = 6 ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → (12 * m) % 8 ≠ 6) ∧
  (∃ (k : ℕ), k > 0 ∧ k < n ∧ (12 * k) % 8 = 6) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l979_97950


namespace NUMINAMATH_CALUDE_custom_op_example_l979_97908

-- Define the custom operation ※
def custom_op (a b : ℚ) : ℚ := 4 * b - a

-- Theorem statement
theorem custom_op_example : custom_op (custom_op (-1) 3) 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l979_97908


namespace NUMINAMATH_CALUDE_five_kg_to_g_eight_thousand_g_to_kg_l979_97930

-- Define the conversion factor
def kg_to_g : ℝ := 1000

-- Theorem for converting 5 kg to grams
theorem five_kg_to_g : 5 * kg_to_g = 5000 := by sorry

-- Theorem for converting 8000 g to kg
theorem eight_thousand_g_to_kg : 8000 / kg_to_g = 8 := by sorry

end NUMINAMATH_CALUDE_five_kg_to_g_eight_thousand_g_to_kg_l979_97930


namespace NUMINAMATH_CALUDE_expedition_investigation_days_l979_97935

theorem expedition_investigation_days 
  (upstream_speed : ℕ) 
  (downstream_speed : ℕ) 
  (total_days : ℕ) 
  (final_day_distance : ℕ) 
  (h1 : upstream_speed = 17)
  (h2 : downstream_speed = 25)
  (h3 : total_days = 60)
  (h4 : final_day_distance = 24) :
  ∃ (upstream_days downstream_days investigation_days : ℕ),
    upstream_days + downstream_days + investigation_days = total_days ∧
    upstream_speed * upstream_days - downstream_speed * downstream_days = final_day_distance - downstream_speed ∧
    investigation_days = 23 := by
  sorry

#check expedition_investigation_days

end NUMINAMATH_CALUDE_expedition_investigation_days_l979_97935


namespace NUMINAMATH_CALUDE_theo_donut_holes_l979_97947

/-- Represents a worker coating donut holes -/
structure Worker where
  name : String
  radius : ℕ

/-- Calculates the surface area of a spherical donut hole -/
def surfaceArea (r : ℕ) : ℕ := 4 * r * r

/-- Calculates the number of donut holes coated by a worker when all workers finish simultaneously -/
def donutHolesCoated (workers : List Worker) (w : Worker) : ℕ :=
  let surfaces := workers.map (λ worker => surfaceArea worker.radius)
  let lcm := surfaces.foldl Nat.lcm 1
  lcm / (surfaceArea w.radius)

/-- The main theorem stating the number of donut holes Theo will coat -/
theorem theo_donut_holes (workers : List Worker) :
  workers = [
    ⟨"Niraek", 5⟩,
    ⟨"Theo", 7⟩,
    ⟨"Akshaj", 9⟩,
    ⟨"Mira", 11⟩
  ] →
  donutHolesCoated workers (Worker.mk "Theo" 7) = 1036830 := by
  sorry

end NUMINAMATH_CALUDE_theo_donut_holes_l979_97947


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l979_97963

theorem complex_fraction_evaluation :
  (3/2 : ℚ) * (8/3 * (15/8 - 5/6)) / ((7/8 + 11/6) / (13/4)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l979_97963


namespace NUMINAMATH_CALUDE_correct_product_l979_97976

theorem correct_product (x : ℝ) (h : 21 * x = 27 * x - 48) : 27 * x = 27 * x := by
  sorry

end NUMINAMATH_CALUDE_correct_product_l979_97976


namespace NUMINAMATH_CALUDE_first_question_percentage_l979_97996

theorem first_question_percentage (second : ℝ) (neither : ℝ) (both : ℝ)
  (h1 : second = 50)
  (h2 : neither = 20)
  (h3 : both = 33)
  : ∃ first : ℝ, first = 63 ∧ first + second - both + neither = 100 :=
by sorry

end NUMINAMATH_CALUDE_first_question_percentage_l979_97996


namespace NUMINAMATH_CALUDE_number_of_white_balls_l979_97999

/-- Given a bag with red and white balls, prove the number of white balls when probability of drawing red is known -/
theorem number_of_white_balls (n : ℕ) : 
  (8 : ℚ) / (8 + n) = (2 : ℚ) / 5 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_white_balls_l979_97999


namespace NUMINAMATH_CALUDE_card_distribution_theorem_l979_97957

/-- Represents the number of cards -/
def num_cards : ℕ := 6

/-- Represents the number of envelopes -/
def num_envelopes : ℕ := 3

/-- Represents the number of cards per envelope -/
def cards_per_envelope : ℕ := 2

/-- Calculates the number of ways to distribute cards into envelopes -/
def distribute_cards : ℕ := sorry

theorem card_distribution_theorem : 
  distribute_cards = 18 := by sorry

end NUMINAMATH_CALUDE_card_distribution_theorem_l979_97957


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l979_97964

theorem polynomial_evaluation (x : ℝ) (h : x = 1 + Real.sqrt 2) : 
  x^4 - 4*x^3 + 4*x^2 + 4 = 5 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l979_97964


namespace NUMINAMATH_CALUDE_sum_of_tangent_slopes_l979_97902

/-- The parabola P defined by y = x^2 -/
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (10, 36)

/-- The quadratic equation whose roots are the slopes of tangent lines -/
def tangent_slope_equation (m : ℝ) : Prop := m^2 - 40*m + 144 = 0

/-- The theorem stating that the sum of roots of the tangent slope equation is 40 -/
theorem sum_of_tangent_slopes :
  ∃ (r s : ℝ), (∀ m : ℝ, tangent_slope_equation m ↔ m = r ∨ m = s) ∧ r + s = 40 := by sorry

end NUMINAMATH_CALUDE_sum_of_tangent_slopes_l979_97902


namespace NUMINAMATH_CALUDE_det_transformation_l979_97992

/-- Given a 2x2 matrix with determinant 7, prove that a specific transformation of this matrix also has determinant 7. -/
theorem det_transformation (p q r s : ℝ) : 
  Matrix.det !![p, q; r, s] = 7 → 
  Matrix.det !![p + 2*r, q + 2*s; r, s] = 7 := by
  sorry

end NUMINAMATH_CALUDE_det_transformation_l979_97992


namespace NUMINAMATH_CALUDE_sin_2000_in_terms_of_tan_160_l979_97982

theorem sin_2000_in_terms_of_tan_160 (a : ℝ) (h : Real.tan (160 * π / 180) = a) :
  Real.sin (2000 * π / 180) = -a / Real.sqrt (1 + a^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_2000_in_terms_of_tan_160_l979_97982


namespace NUMINAMATH_CALUDE_lassie_bones_problem_l979_97966

/-- The number of bones Lassie started with before Saturday -/
def initial_bones : ℕ := 50

/-- The number of bones Lassie has after eating on Saturday -/
def bones_after_saturday : ℕ := initial_bones / 2

/-- The number of bones Lassie receives on Sunday -/
def bones_received_sunday : ℕ := 10

/-- The total number of bones Lassie has after Sunday -/
def total_bones_after_sunday : ℕ := 35

theorem lassie_bones_problem :
  bones_after_saturday + bones_received_sunday = total_bones_after_sunday :=
by sorry

end NUMINAMATH_CALUDE_lassie_bones_problem_l979_97966


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l979_97914

theorem sqrt_sum_equality (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k > 0)
  (h : 2 * a * b * c + k * (a^2 + b^2 + c^2) = k^3) :
  Real.sqrt ((k - a) * (k - b) / ((k + a) * (k + b))) +
  Real.sqrt ((k - b) * (k - c) / ((k + b) * (k + c))) +
  Real.sqrt ((k - c) * (k - a) / ((k + c) * (k + a))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l979_97914


namespace NUMINAMATH_CALUDE_scientific_notation_274_million_l979_97993

theorem scientific_notation_274_million :
  274000000 = 2.74 * (10 : ℝ)^8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_274_million_l979_97993


namespace NUMINAMATH_CALUDE_card_game_result_l979_97918

/-- Represents the number of cards in each pile -/
structure CardPiles :=
  (left : ℕ)
  (middle : ℕ)
  (right : ℕ)

/-- The card game operations -/
def card_game_operations (initial : CardPiles) : CardPiles :=
  let step1 := initial
  let step2 := CardPiles.mk (step1.left - 2) (step1.middle + 2) step1.right
  let step3 := CardPiles.mk step2.left (step2.middle + 1) (step2.right - 1)
  CardPiles.mk step3.left.succ (step3.middle - step3.left) step3.right

theorem card_game_result (initial : CardPiles) 
  (h1 : initial.left = initial.middle)
  (h2 : initial.middle = initial.right)
  (h3 : initial.left ≥ 2) :
  (card_game_operations initial).middle = 5 :=
sorry

end NUMINAMATH_CALUDE_card_game_result_l979_97918


namespace NUMINAMATH_CALUDE_crayon_production_in_four_hours_l979_97904

/-- Represents a crayon factory with given specifications -/
structure CrayonFactory where
  colors : Nat
  crayonsPerColorPerBox : Nat
  boxesPerHour : Nat

/-- Calculates the total number of crayons produced in a given number of hours -/
def totalCrayonsProduced (factory : CrayonFactory) (hours : Nat) : Nat :=
  factory.colors * factory.crayonsPerColorPerBox * factory.boxesPerHour * hours

/-- Theorem stating that a factory with given specifications produces 160 crayons in 4 hours -/
theorem crayon_production_in_four_hours :
  ∀ (factory : CrayonFactory),
    factory.colors = 4 →
    factory.crayonsPerColorPerBox = 2 →
    factory.boxesPerHour = 5 →
    totalCrayonsProduced factory 4 = 160 :=
by sorry

end NUMINAMATH_CALUDE_crayon_production_in_four_hours_l979_97904


namespace NUMINAMATH_CALUDE_ln_gt_one_sufficient_not_necessary_for_x_gt_one_l979_97949

theorem ln_gt_one_sufficient_not_necessary_for_x_gt_one :
  (∃ x : ℝ, x > 1 ∧ ¬(Real.log x > 1)) ∧
  (∀ x : ℝ, Real.log x > 1 → x > 1) :=
sorry

end NUMINAMATH_CALUDE_ln_gt_one_sufficient_not_necessary_for_x_gt_one_l979_97949


namespace NUMINAMATH_CALUDE_five_bikes_in_driveway_l979_97951

/-- Calculates the number of bikes in the driveway given the total number of wheels and other vehicles --/
def number_of_bikes (total_wheels car_count tricycle_count trash_can_count roller_skate_wheels : ℕ) : ℕ :=
  let car_wheels := 4 * car_count
  let tricycle_wheels := 3 * tricycle_count
  let remaining_wheels := total_wheels - (car_wheels + tricycle_wheels + roller_skate_wheels)
  let bike_and_trash_can_wheels := remaining_wheels - (2 * trash_can_count)
  bike_and_trash_can_wheels / 2

/-- Theorem stating that there are 5 bikes in the driveway --/
theorem five_bikes_in_driveway :
  number_of_bikes 25 2 1 1 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_bikes_in_driveway_l979_97951


namespace NUMINAMATH_CALUDE_quadratic_transformation_l979_97974

theorem quadratic_transformation (y m n : ℝ) : 
  (2 * y^2 - 2 = 4 * y) → 
  ((y - m)^2 = n) → 
  (m - n)^2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l979_97974


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l979_97998

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (α : Plane) (a b : Line) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l979_97998


namespace NUMINAMATH_CALUDE_legos_won_l979_97910

def initial_legos : ℕ := 2080
def final_legos : ℕ := 2097

theorem legos_won : final_legos - initial_legos = 17 := by
  sorry

end NUMINAMATH_CALUDE_legos_won_l979_97910


namespace NUMINAMATH_CALUDE_sin_2phi_value_l979_97972

theorem sin_2phi_value (φ : ℝ) (h : 7/13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120/169 := by
  sorry

end NUMINAMATH_CALUDE_sin_2phi_value_l979_97972


namespace NUMINAMATH_CALUDE_car_speed_proof_l979_97913

/-- Proves that a car traveling at speed v km/h takes 15 seconds longer to travel 1 kilometer
    than it would at 48 km/h if and only if v = 40 km/h. -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v) * 3600 = (1 / 48) * 3600 + 15 ↔ v = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l979_97913


namespace NUMINAMATH_CALUDE_prob_one_or_two_sunny_days_l979_97926

-- Define the probability of rain
def rain_prob : ℚ := 3/5

-- Define the number of days
def num_days : ℕ := 5

-- Function to calculate the probability of exactly k sunny days
def prob_k_sunny_days (k : ℕ) : ℚ :=
  (num_days.choose k) * (1 - rain_prob)^k * rain_prob^(num_days - k)

-- Theorem statement
theorem prob_one_or_two_sunny_days :
  prob_k_sunny_days 1 + prob_k_sunny_days 2 = 378/625 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_or_two_sunny_days_l979_97926


namespace NUMINAMATH_CALUDE_cube_root_always_real_l979_97921

theorem cube_root_always_real : 
  ∀ x : ℝ, ∃ y : ℝ, y^3 = -(x + 3)^3 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_always_real_l979_97921


namespace NUMINAMATH_CALUDE_line_through_quadrants_line_through_fixed_point_point_slope_form_line_equation_l979_97903

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- 1. Line passing through first, second, and fourth quadrants
theorem line_through_quadrants (k b : ℝ) :
  (∀ x y, y = k * x + b → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) →
  k < 0 ∧ b > 0 :=
sorry

-- 2. Line passing through fixed point
theorem line_through_fixed_point (a : ℝ) :
  ∀ x y, y = a * x - 3 * a + 2 → (x = 3 → y = 2) :=
sorry

-- 3. Point-slope form equation
theorem point_slope_form (p : Point) (m : ℝ) :
  p.x = 2 ∧ p.y = -1 ∧ m = -Real.sqrt 3 →
  ∀ x y, y + 1 = m * (x - 2) ↔ y = m * (x - p.x) + p.y :=
sorry

-- 4. Line equation with given slope and intercept
theorem line_equation (l : Line) :
  l.slope = -2 ∧ l.intercept = 3 →
  ∀ x y, y = l.slope * x + l.intercept ↔ y = -2 * x + 3 :=
sorry

end NUMINAMATH_CALUDE_line_through_quadrants_line_through_fixed_point_point_slope_form_line_equation_l979_97903


namespace NUMINAMATH_CALUDE_tree_planting_change_l979_97991

/-- Represents the road with tree planting configuration -/
structure RoadConfig where
  length : ℕ
  initial_spacing : ℕ
  new_spacing : ℕ

/-- Calculates the number of trees for a given spacing -/
def trees_count (config : RoadConfig) (spacing : ℕ) : ℕ :=
  config.length / spacing + 1

/-- Calculates the change in number of holes -/
def hole_change (config : RoadConfig) : ℤ :=
  (trees_count config config.new_spacing : ℤ) - (trees_count config config.initial_spacing : ℤ)

theorem tree_planting_change (config : RoadConfig) 
  (h_length : config.length = 240)
  (h_initial : config.initial_spacing = 8)
  (h_new : config.new_spacing = 6) :
  hole_change config = 10 ∧ max (-(hole_change config)) 0 = 0 :=
sorry

end NUMINAMATH_CALUDE_tree_planting_change_l979_97991


namespace NUMINAMATH_CALUDE_increasing_function_equivalence_l979_97953

/-- A function f is increasing on ℝ -/
def IncreasingOnReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_equivalence (f : ℝ → ℝ) (h : IncreasingOnReals f) :
  ∀ a b : ℝ, (a + b ≥ 0 ↔ f a + f b ≥ f (-a) + f (-b)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_equivalence_l979_97953


namespace NUMINAMATH_CALUDE_min_period_sin_2x_plus_pi_third_l979_97988

/-- The minimum positive period of y = sin(2x + π/3) is π -/
theorem min_period_sin_2x_plus_pi_third (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 3)
  ∃ T : ℝ, T > 0 ∧ (∀ t : ℝ, f (x + T) = f x) ∧
    (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (x + S) = f x) → T ≤ S) ∧
    T = π :=
by sorry

end NUMINAMATH_CALUDE_min_period_sin_2x_plus_pi_third_l979_97988


namespace NUMINAMATH_CALUDE_paving_cost_l979_97922

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) : 
  length = 6.5 → 
  width = 2.75 → 
  rate = 600 → 
  length * width * rate = 10725 := by
sorry

end NUMINAMATH_CALUDE_paving_cost_l979_97922


namespace NUMINAMATH_CALUDE_water_remaining_after_four_replacements_l979_97931

/-- Represents the fraction of original water remaining after a number of replacements -/
def water_remaining (initial_water : ℚ) (tank_capacity : ℚ) (replacement_volume : ℚ) (n : ℕ) : ℚ :=
  (1 - replacement_volume / tank_capacity) ^ n * initial_water / tank_capacity

/-- Theorem stating the fraction of original water remaining after 4 replacements -/
theorem water_remaining_after_four_replacements : 
  water_remaining 10 20 5 4 = 81 / 256 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_after_four_replacements_l979_97931


namespace NUMINAMATH_CALUDE_smallest_difference_in_triangle_l979_97971

theorem smallest_difference_in_triangle (a b c : ℕ) : 
  a + b + c = 2023 →
  a < b →
  b ≤ c →
  (∀ x y z : ℕ, x + y + z = 2023 → x < y → y ≤ z → b - a ≤ y - x) →
  b - a = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_difference_in_triangle_l979_97971


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l979_97940

theorem quadratic_root_relation (b c : ℝ) : 
  (∃ p q : ℝ, 2 * p^2 - 4 * p - 6 = 0 ∧ 2 * q^2 - 4 * q - 6 = 0 ∧
   ∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = p - 3 ∨ x = q - 3)) →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l979_97940


namespace NUMINAMATH_CALUDE_rectangle_to_square_width_third_l979_97994

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- Theorem: Given a 9x27 rectangle that can be cut into two congruent hexagons
    which can be repositioned to form a square, one third of the rectangle's width is 9 -/
theorem rectangle_to_square_width_third (rect : Rectangle) (sq : Square) :
  rect.width = 27 ∧ 
  rect.height = 9 ∧ 
  sq.side ^ 2 = rect.width * rect.height →
  rect.width / 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_width_third_l979_97994


namespace NUMINAMATH_CALUDE_franks_money_duration_l979_97978

/-- The duration (in weeks) that Frank's money will last given his earnings and weekly spending. -/
def money_duration (lawn_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Frank's money will last for 9 weeks given his earnings and spending. -/
theorem franks_money_duration :
  money_duration 5 58 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_franks_money_duration_l979_97978


namespace NUMINAMATH_CALUDE_gcd_228_1995_base_conversion_l979_97932

-- Problem 1: GCD of 228 and 1995
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by sorry

-- Problem 2: Base conversion
theorem base_conversion :
  (1 * 3^4 + 1 * 3^3 + 1 * 3^2 + 0 * 3^1 + 2 * 3^0) = (3 * 6^2 + 1 * 6^1 + 5 * 6^0) := by sorry

end NUMINAMATH_CALUDE_gcd_228_1995_base_conversion_l979_97932


namespace NUMINAMATH_CALUDE_kaleb_first_half_score_l979_97911

/-- Calculates the first half score in a trivia game given the total score and second half score. -/
def first_half_score (total_score second_half_score : ℕ) : ℕ :=
  total_score - second_half_score

/-- Proves that Kaleb's first half score is 43 points given his total score of 66 and second half score of 23. -/
theorem kaleb_first_half_score :
  first_half_score 66 23 = 43 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_first_half_score_l979_97911


namespace NUMINAMATH_CALUDE_pool_filling_time_l979_97980

/-- Calculates the time in hours required to fill a pool given its capacity and the rate of water flow. -/
theorem pool_filling_time 
  (pool_capacity : ℚ)  -- Pool capacity in gallons
  (num_hoses : ℕ)      -- Number of hoses
  (flow_rate : ℚ)      -- Flow rate per hose in gallons per minute
  (h : pool_capacity = 36000 ∧ num_hoses = 6 ∧ flow_rate = 3) :
  (pool_capacity / (↑num_hoses * flow_rate * 60)) = 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l979_97980


namespace NUMINAMATH_CALUDE_tank_capacity_l979_97909

theorem tank_capacity (C : ℚ) : 
  (C > 0) →  -- The capacity is positive
  ((117 / 200) * C = 4680) →  -- Final volume equation
  (C = 8000) := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l979_97909


namespace NUMINAMATH_CALUDE_prob_not_same_cafeteria_prob_not_same_cafeteria_is_three_fourths_l979_97968

/-- The probability that three students do not dine in the same cafeteria when randomly choosing between two cafeterias -/
theorem prob_not_same_cafeteria : ℚ :=
  let num_cafeterias : ℕ := 2
  let num_students : ℕ := 3
  let total_choices : ℕ := num_cafeterias ^ num_students
  let same_cafeteria_choices : ℕ := num_cafeterias
  let diff_cafeteria_choices : ℕ := total_choices - same_cafeteria_choices
  (diff_cafeteria_choices : ℚ) / total_choices

theorem prob_not_same_cafeteria_is_three_fourths :
  prob_not_same_cafeteria = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_not_same_cafeteria_prob_not_same_cafeteria_is_three_fourths_l979_97968


namespace NUMINAMATH_CALUDE_inequality_solution_set_l979_97975

/-- The solution set of the inequality x^2 - ax + a - 1 ≤ 0 for real a -/
def solution_set (a : ℝ) : Set ℝ :=
  if a < 2 then Set.Icc (a - 1) 1
  else if a = 2 then {1}
  else Set.Icc 1 (a - 1)

/-- Theorem stating the solution set of the inequality x^2 - ax + a - 1 ≤ 0 -/
theorem inequality_solution_set (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ x^2 - a*x + a - 1 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l979_97975


namespace NUMINAMATH_CALUDE_algebraic_equation_proof_l979_97919

theorem algebraic_equation_proof (a b c : ℝ) 
  (h1 : a^2 + b*c = 14) 
  (h2 : b^2 - 2*b*c = -6) : 
  3*a^2 + 4*b^2 - 5*b*c = 18 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equation_proof_l979_97919


namespace NUMINAMATH_CALUDE_rental_cost_is_165_l979_97981

/-- Calculates the total cost of renting a car given the daily rate, per-mile rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℝ) (mile_rate : ℝ) (days : ℕ) (miles : ℕ) : ℝ :=
  daily_rate * (days : ℝ) + mile_rate * (miles : ℝ)

/-- Theorem stating that under the given conditions, the total rental cost is $165. -/
theorem rental_cost_is_165 :
  let daily_rate : ℝ := 30
  let mile_rate : ℝ := 0.15
  let days : ℕ := 3
  let miles : ℕ := 500
  total_rental_cost daily_rate mile_rate days miles = 165 := by
sorry


end NUMINAMATH_CALUDE_rental_cost_is_165_l979_97981


namespace NUMINAMATH_CALUDE_curve_is_line_l979_97915

/-- The curve defined by the polar equation θ = 5π/6 is a line -/
theorem curve_is_line : ∀ (r : ℝ) (θ : ℝ), 
  θ = (5 * Real.pi) / 6 → 
  ∃ (a b : ℝ), ∀ (x y : ℝ), x = r * Real.cos θ ∧ y = r * Real.sin θ → 
  a * x + b * y = 0 :=
sorry

end NUMINAMATH_CALUDE_curve_is_line_l979_97915


namespace NUMINAMATH_CALUDE_simplify_fraction_l979_97929

theorem simplify_fraction (a b c : ℕ) (h : b = a * c) :
  (a : ℚ) / b * c = 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l979_97929


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l979_97924

theorem orange_juice_percentage
  (total_volume : ℝ)
  (watermelon_percentage : ℝ)
  (grape_volume : ℝ)
  (h1 : total_volume = 300)
  (h2 : watermelon_percentage = 40)
  (h3 : grape_volume = 105) :
  (total_volume - watermelon_percentage / 100 * total_volume - grape_volume) / total_volume * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_l979_97924


namespace NUMINAMATH_CALUDE_power_function_through_point_l979_97986

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = x^a) →  -- f is a power function with exponent a
  f 2 = 16 →              -- f passes through the point (2, 16)
  a = 4 := by             -- prove that a = 4
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l979_97986


namespace NUMINAMATH_CALUDE_distance_after_five_hours_l979_97942

/-- The distance between two people walking in opposite directions -/
def distance_between (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 + speed2) * time

/-- Theorem: The distance between two people walking in opposite directions for 5 hours
    with speeds 5 km/hr and 10 km/hr is 75 km -/
theorem distance_after_five_hours :
  distance_between 5 10 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_five_hours_l979_97942


namespace NUMINAMATH_CALUDE_inequality_proof_l979_97983

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l979_97983


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l979_97958

-- Define the line equation
def line (x y : ℝ) : Prop := 4 * x + 7 * y + 49 = 0

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define what it means for a line to be tangent to a parabola
def is_tangent (l : ℝ → ℝ → Prop) (p : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₀ y₀ : ℝ), l x₀ y₀ ∧ p x₀ y₀ ∧
    ∀ (x y : ℝ), l x y ∧ p x y → (x, y) = (x₀, y₀)

-- Theorem statement
theorem line_tangent_to_parabola :
  is_tangent line parabola :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l979_97958


namespace NUMINAMATH_CALUDE_bumper_car_line_l979_97937

/-- The number of people initially in line for bumper cars -/
def initial_people : ℕ := sorry

/-- The number of people in line after 2 leave and 2 join -/
def final_people : ℕ := 10

/-- The condition that if 2 people leave and 2 join, there are 10 people in line -/
axiom condition : initial_people = final_people

theorem bumper_car_line : initial_people = 10 := by sorry

end NUMINAMATH_CALUDE_bumper_car_line_l979_97937


namespace NUMINAMATH_CALUDE_inverse_mod_103_l979_97961

theorem inverse_mod_103 (h : (7⁻¹ : ZMod 103) = 55) : (49⁻¹ : ZMod 103) = 38 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_103_l979_97961


namespace NUMINAMATH_CALUDE_carol_weight_l979_97917

/-- Given that Alice and Carol have a combined weight of 280 pounds,
    and the difference between Carol's and Alice's weights is one-third of Carol's weight,
    prove that Carol weighs 168 pounds. -/
theorem carol_weight (alice_weight carol_weight : ℝ) 
    (h1 : alice_weight + carol_weight = 280)
    (h2 : carol_weight - alice_weight = carol_weight / 3) :
    carol_weight = 168 := by
  sorry

end NUMINAMATH_CALUDE_carol_weight_l979_97917


namespace NUMINAMATH_CALUDE_percentage_difference_l979_97941

-- Define the variables
variable (x y z : ℝ)

-- State the theorem
theorem percentage_difference (h1 : z = 400) (h2 : y = 1.2 * z) (h3 : x + y + z = 1480) :
  (x - y) / y = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l979_97941


namespace NUMINAMATH_CALUDE_fine_amount_correct_l979_97928

/-- Calculates the fine amount given the quantity sold, price per ounce, and amount left after the fine -/
def calculate_fine (quantity_sold : ℕ) (price_per_ounce : ℕ) (amount_left : ℕ) : ℕ :=
  quantity_sold * price_per_ounce - amount_left

/-- Proves that the fine amount is correct given the problem conditions -/
theorem fine_amount_correct : calculate_fine 8 9 22 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fine_amount_correct_l979_97928


namespace NUMINAMATH_CALUDE_sin_90_degrees_l979_97927

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l979_97927


namespace NUMINAMATH_CALUDE_polynomial_negative_roots_l979_97969

theorem polynomial_negative_roots (q : ℝ) (hq : q > 1/2) :
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^4 + q*x₁^3 + 3*x₁^2 + q*x₁ + 9 = 0 ∧
  x₂^4 + q*x₂^3 + 3*x₂^2 + q*x₂ + 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_negative_roots_l979_97969


namespace NUMINAMATH_CALUDE_milk_set_cost_l979_97955

/-- The cost of a set of 2 packs of 500 mL milk -/
def set_cost : ℝ := 2.50

/-- The cost of an individual pack of 500 mL milk -/
def individual_cost : ℝ := 1.30

/-- The total savings when buying ten sets of 2 packs -/
def total_savings : ℝ := 1

theorem milk_set_cost :
  set_cost = 2 * individual_cost - total_savings / 10 :=
by sorry

end NUMINAMATH_CALUDE_milk_set_cost_l979_97955


namespace NUMINAMATH_CALUDE_f_not_mapping_l979_97939

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

-- Define the function f
def f (x : ℝ) : ℝ := x

-- Theorem stating that f is not a mapping from A to B
theorem f_not_mapping : ¬(∀ x ∈ A, f x ∈ B) :=
sorry

end NUMINAMATH_CALUDE_f_not_mapping_l979_97939


namespace NUMINAMATH_CALUDE_jumps_before_cleaning_l979_97962

-- Define the pool characteristics
def pool_capacity : ℝ := 1200  -- in liters
def splash_out_volume : ℝ := 0.2  -- in liters (200 ml = 0.2 L)
def cleaning_threshold : ℝ := 0.8  -- 80% capacity

-- Define the number of jumps
def number_of_jumps : ℕ := 1200

-- Theorem statement
theorem jumps_before_cleaning :
  ⌊(pool_capacity - pool_capacity * cleaning_threshold) / splash_out_volume⌋ = number_of_jumps := by
  sorry

end NUMINAMATH_CALUDE_jumps_before_cleaning_l979_97962


namespace NUMINAMATH_CALUDE_expand_and_simplify_l979_97944

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l979_97944


namespace NUMINAMATH_CALUDE_largest_integer_x_l979_97965

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def fraction (x : ℤ) : ℚ := (x^2 + 3*x + 8) / (x - 2)

theorem largest_integer_x : 
  (∀ x : ℤ, x > 1 → ¬ is_integer (fraction x)) ∧ 
  is_integer (fraction 1) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_x_l979_97965


namespace NUMINAMATH_CALUDE_production_value_decrease_l979_97995

theorem production_value_decrease (a : ℝ) :
  let increase_percent := a
  let decrease_percent := |a / (100 + a)|
  increase_percent > -100 →
  decrease_percent = |1 - 1 / (1 + a / 100)| :=
by sorry

end NUMINAMATH_CALUDE_production_value_decrease_l979_97995


namespace NUMINAMATH_CALUDE_power_45_equals_a_squared_b_l979_97905

theorem power_45_equals_a_squared_b (x a b : ℝ) (h1 : 3^x = a) (h2 : 5^x = b) : 45^x = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_power_45_equals_a_squared_b_l979_97905


namespace NUMINAMATH_CALUDE_book_has_two_chapters_l979_97985

/-- A book with chapters and pages -/
structure Book where
  total_pages : ℕ
  first_chapter_pages : ℕ
  second_chapter_pages : ℕ

/-- The number of chapters in a book -/
def num_chapters (b : Book) : ℕ :=
  if b.first_chapter_pages + b.second_chapter_pages = b.total_pages then 2 else 0

theorem book_has_two_chapters (b : Book) 
  (h1 : b.total_pages = 81) 
  (h2 : b.first_chapter_pages = 13) 
  (h3 : b.second_chapter_pages = 68) : 
  num_chapters b = 2 := by
  sorry

end NUMINAMATH_CALUDE_book_has_two_chapters_l979_97985


namespace NUMINAMATH_CALUDE_factor_tree_value_l979_97936

theorem factor_tree_value (F G H J X : ℕ) : 
  H = 2 * 5 →
  J = 3 * 7 →
  F = 7 * H →
  G = 11 * J →
  X = F * G →
  X = 16170 :=
by
  sorry

end NUMINAMATH_CALUDE_factor_tree_value_l979_97936


namespace NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l979_97970

-- Problem 1
theorem simplify_expression (a b : ℝ) :
  4 * a^2 + 3 * b^2 + 2 * a * b - 3 * a^2 - 3 * b * a - a^2 = a^2 - a * b + 3 * b^2 := by
  sorry

-- Problem 2
theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  3 * x - 4 * x^2 + 7 - 3 * x + 2 * x^2 + 1 = -10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l979_97970


namespace NUMINAMATH_CALUDE_quarter_percentage_approx_l979_97952

def dimes : ℕ := 60
def quarters : ℕ := 30
def nickels : ℕ := 40

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def total_value : ℕ := dimes * dime_value + quarters * quarter_value + nickels * nickel_value
def quarter_value_total : ℕ := quarters * quarter_value

theorem quarter_percentage_approx (ε : ℝ) (h : ε > 0) :
  ∃ (p : ℝ), abs (p - 48.4) < ε ∧ p = (quarter_value_total : ℝ) / total_value * 100 :=
sorry

end NUMINAMATH_CALUDE_quarter_percentage_approx_l979_97952


namespace NUMINAMATH_CALUDE_watermelon_customers_l979_97960

theorem watermelon_customers (total : ℕ) (one_melon : ℕ) (three_melons : ℕ) :
  total = 46 →
  one_melon = 17 →
  three_melons = 3 →
  ∃ (two_melons : ℕ),
    two_melons * 2 + one_melon * 1 + three_melons * 3 = total ∧
    two_melons = 10 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_customers_l979_97960


namespace NUMINAMATH_CALUDE_walking_problem_solution_l979_97948

def walking_problem (total_distance : ℝ) (speed_R : ℝ) (speed_S_initial : ℝ) (speed_S_second : ℝ) : Prop :=
  ∃ (k : ℕ) (x : ℝ),
    -- The total distance is 76 miles
    total_distance = 76 ∧
    -- Speed of person at R is 4.5 mph
    speed_R = 4.5 ∧
    -- Initial speed of person at S is 3.25 mph
    speed_S_initial = 3.25 ∧
    -- Second hour speed of person at S is 3.75 mph
    speed_S_second = 3.75 ∧
    -- They meet after k hours (k is a natural number)
    k > 0 ∧
    -- Distance traveled by person from R
    speed_R * k + x = total_distance / 2 ∧
    -- Distance traveled by person from S (arithmetic sequence sum)
    k * (speed_S_initial + (speed_S_second - speed_S_initial) * (k - 1) / 2) - x = total_distance / 2 ∧
    -- x is the difference in distances, and it equals 4
    x = 4

theorem walking_problem_solution :
  walking_problem 76 4.5 3.25 3.75 :=
sorry

end NUMINAMATH_CALUDE_walking_problem_solution_l979_97948


namespace NUMINAMATH_CALUDE_complement_of_union_eq_nonpositive_l979_97946

-- Define the sets U, P, and Q
def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x * (x - 2) < 0}

-- State the theorem
theorem complement_of_union_eq_nonpositive :
  (U \ (P ∪ Q)) = {x : ℝ | x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_eq_nonpositive_l979_97946


namespace NUMINAMATH_CALUDE_car_dealership_shipment_l979_97907

theorem car_dealership_shipment 
  (initial_cars : ℕ) 
  (initial_silver_percent : ℚ)
  (new_shipment_nonsilver_percent : ℚ)
  (final_silver_percent : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percent = 1/5)
  (h3 : new_shipment_nonsilver_percent = 7/20)
  (h4 : final_silver_percent = 3/10)
  : ∃ (new_shipment : ℕ), 
    (initial_silver_percent * initial_cars + (1 - new_shipment_nonsilver_percent) * new_shipment) / 
    (initial_cars + new_shipment) = final_silver_percent ∧ 
    new_shipment = 11 :=
sorry

end NUMINAMATH_CALUDE_car_dealership_shipment_l979_97907


namespace NUMINAMATH_CALUDE_f_is_quadratic_l979_97967

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we're checking -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l979_97967


namespace NUMINAMATH_CALUDE_bankers_discount_calculation_l979_97956

/-- Banker's discount calculation -/
theorem bankers_discount_calculation (face_value : ℝ) (interest_rate : ℝ) (true_discount : ℝ)
  (h1 : face_value = 74500)
  (h2 : interest_rate = 0.15)
  (h3 : true_discount = 11175) :
  face_value * interest_rate = true_discount :=
by sorry

end NUMINAMATH_CALUDE_bankers_discount_calculation_l979_97956


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l979_97920

/-- Given a right triangle with legs a and b, and hypotenuse c, where a:b = 2:5,
    if a perpendicular from the right angle to the hypotenuse divides it into
    segments r (adjacent to a) and s (adjacent to b), then r/s = 4/25. -/
theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a ^ 2 + b ^ 2 = c ^ 2 →  -- Pythagorean theorem
  a / b = 2 / 5 →  -- given ratio of legs
  r * s = a * b →  -- geometric mean theorem
  r + s = c →  -- sum of segments equals hypotenuse
  r / s = 4 / 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l979_97920


namespace NUMINAMATH_CALUDE_simplify_expression_l979_97973

theorem simplify_expression : 
  Real.sqrt 2 * 2^(1/2 : ℝ) + 18 / 3 * 3 - 8^(3/2 : ℝ) = 20 - 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l979_97973


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l979_97990

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and an asymptote 2x - √3y = 0,
    prove that its eccentricity is √21/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∀ x y : ℝ, 2 * x - Real.sqrt 3 * y = 0 → 
    (x^2 / a^2 - y^2 / b^2 = 1 ↔ x = 0 ∧ y = 0)) : 
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 21 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l979_97990


namespace NUMINAMATH_CALUDE_weight_puzzle_l979_97900

theorem weight_puzzle (w₁ w₂ w₃ w₄ : ℕ) 
  (h1 : w₁ + w₂ = 1700 ∨ w₁ + w₃ = 1700 ∨ w₁ + w₄ = 1700 ∨ w₂ + w₃ = 1700 ∨ w₂ + w₄ = 1700 ∨ w₃ + w₄ = 1700)
  (h2 : w₁ + w₂ = 1870 ∨ w₁ + w₃ = 1870 ∨ w₁ + w₄ = 1870 ∨ w₂ + w₃ = 1870 ∨ w₂ + w₄ = 1870 ∨ w₃ + w₄ = 1870)
  (h3 : w₁ + w₂ = 2110 ∨ w₁ + w₃ = 2110 ∨ w₁ + w₄ = 2110 ∨ w₂ + w₃ = 2110 ∨ w₂ + w₄ = 2110 ∨ w₃ + w₄ = 2110)
  (h4 : w₁ + w₂ = 2330 ∨ w₁ + w₃ = 2330 ∨ w₁ + w₄ = 2330 ∨ w₂ + w₃ = 2330 ∨ w₂ + w₄ = 2330 ∨ w₃ + w₄ = 2330)
  (h5 : w₁ + w₂ = 2500 ∨ w₁ + w₃ = 2500 ∨ w₁ + w₄ = 2500 ∨ w₂ + w₃ = 2500 ∨ w₂ + w₄ = 2500 ∨ w₃ + w₄ = 2500)
  (h_distinct : w₁ ≠ w₂ ∧ w₁ ≠ w₃ ∧ w₁ ≠ w₄ ∧ w₂ ≠ w₃ ∧ w₂ ≠ w₄ ∧ w₃ ≠ w₄) :
  w₁ + w₂ = 2090 ∨ w₁ + w₃ = 2090 ∨ w₁ + w₄ = 2090 ∨ w₂ + w₃ = 2090 ∨ w₂ + w₄ = 2090 ∨ w₃ + w₄ = 2090 :=
by sorry

end NUMINAMATH_CALUDE_weight_puzzle_l979_97900


namespace NUMINAMATH_CALUDE_flight_cost_X_to_Y_l979_97923

/-- Represents a city in the travel problem -/
inductive City : Type
| X : City
| Y : City
| Z : City

/-- The distance between two cities in kilometers -/
def distance : City → City → ℝ
| City.X, City.Y => 4800
| City.X, City.Z => 4000
| _, _ => 0  -- We don't need other distances for this problem

/-- The cost per kilometer for bus travel -/
def busCostPerKm : ℝ := 0.15

/-- The cost per kilometer for air travel -/
def airCostPerKm : ℝ := 0.12

/-- The booking fee for air travel -/
def airBookingFee : ℝ := 150

/-- The cost of flying between two cities -/
def flightCost (c1 c2 : City) : ℝ :=
  airBookingFee + airCostPerKm * distance c1 c2

/-- The main theorem: The cost of flying from X to Y is $726 -/
theorem flight_cost_X_to_Y : flightCost City.X City.Y = 726 := by
  sorry

end NUMINAMATH_CALUDE_flight_cost_X_to_Y_l979_97923


namespace NUMINAMATH_CALUDE_additional_deductible_calculation_l979_97943

/-- Calculates the additional deductible amount for an average family --/
def additional_deductible_amount (
  current_deductible : ℝ)
  (plan_a_increase : ℝ)
  (plan_b_increase : ℝ)
  (plan_c_increase : ℝ)
  (plan_a_percentage : ℝ)
  (plan_b_percentage : ℝ)
  (plan_c_percentage : ℝ)
  (inflation_rate : ℝ) : ℝ :=
  let plan_a_additional := current_deductible * plan_a_increase
  let plan_b_additional := current_deductible * plan_b_increase
  let plan_c_additional := current_deductible * plan_c_increase
  let weighted_additional := plan_a_additional * plan_a_percentage +
                             plan_b_additional * plan_b_percentage +
                             plan_c_additional * plan_c_percentage
  weighted_additional * (1 + inflation_rate)

/-- Theorem stating the additional deductible amount for an average family --/
theorem additional_deductible_calculation :
  additional_deductible_amount 3000 (2/3) (1/2) (3/5) 0.4 0.3 0.3 0.03 = 1843.70 := by
  sorry

end NUMINAMATH_CALUDE_additional_deductible_calculation_l979_97943


namespace NUMINAMATH_CALUDE_age_difference_l979_97959

theorem age_difference :
  ∀ (a b : ℕ),
  (0 < a ∧ a < 10) →
  (0 < b ∧ b < 10) →
  (10 * a + b + 5 = 2 * (10 * b + a + 5)) →
  (10 * a + b) - (10 * b + a) = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l979_97959


namespace NUMINAMATH_CALUDE_value_of_x_l979_97901

theorem value_of_x (x y z a b c : ℝ) 
  (ha : x * y / (x + y) = a)
  (hb : x * z / (x + z) = b)
  (hc : y * z / (y + z) = c)
  (ha_nonzero : a ≠ 0)
  (hb_nonzero : b ≠ 0)
  (hc_nonzero : c ≠ 0) :
  x = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end NUMINAMATH_CALUDE_value_of_x_l979_97901


namespace NUMINAMATH_CALUDE_total_payment_proof_l979_97916

def apple_quantity : ℕ := 15
def apple_price : ℕ := 85
def mango_quantity : ℕ := 12
def mango_price : ℕ := 60
def grape_quantity : ℕ := 10
def grape_price : ℕ := 75
def strawberry_quantity : ℕ := 6
def strawberry_price : ℕ := 150

def total_cost : ℕ := 
  apple_quantity * apple_price + 
  mango_quantity * mango_price + 
  grape_quantity * grape_price + 
  strawberry_quantity * strawberry_price

theorem total_payment_proof : total_cost = 3645 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_proof_l979_97916
