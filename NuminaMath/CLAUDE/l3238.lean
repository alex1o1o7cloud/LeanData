import Mathlib

namespace NUMINAMATH_CALUDE_disaster_relief_team_selection_l3238_323851

def internal_medicine_doctors : ℕ := 5
def surgeons : ℕ := 6
def total_doctors : ℕ := internal_medicine_doctors + surgeons
def team_size : ℕ := 4

theorem disaster_relief_team_selection :
  (Nat.choose total_doctors team_size) -
  (Nat.choose internal_medicine_doctors team_size) -
  (Nat.choose surgeons team_size) = 310 := by
  sorry

end NUMINAMATH_CALUDE_disaster_relief_team_selection_l3238_323851


namespace NUMINAMATH_CALUDE_f_g_f_2_equals_120_l3238_323889

def f (x : ℝ) : ℝ := 3 * x + 3

def g (x : ℝ) : ℝ := 4 * x + 3

theorem f_g_f_2_equals_120 : f (g (f 2)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_2_equals_120_l3238_323889


namespace NUMINAMATH_CALUDE_no_solution_equation_l3238_323808

theorem no_solution_equation : 
  ¬∃ (x : ℝ), x - 9 / (x - 4) = 4 - 9 / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l3238_323808


namespace NUMINAMATH_CALUDE_trackball_mice_count_l3238_323856

theorem trackball_mice_count (total : ℕ) (wireless : ℕ) (optical : ℕ) (trackball : ℕ) : 
  total = 80 →
  wireless = total / 2 →
  optical = total / 4 →
  trackball = total - (wireless + optical) →
  trackball = 20 := by
sorry

end NUMINAMATH_CALUDE_trackball_mice_count_l3238_323856


namespace NUMINAMATH_CALUDE_max_clock_digit_sum_l3238_323876

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10 + digit_sum (n / 10))

def clock_digit_sum (h m : ℕ) : ℕ := digit_sum h + digit_sum m

theorem max_clock_digit_sum : 
  ∀ h m, is_valid_hour h → is_valid_minute m → 
  clock_digit_sum h m ≤ 28 ∧ 
  ∃ h' m', is_valid_hour h' ∧ is_valid_minute m' ∧ clock_digit_sum h' m' = 28 := by
  sorry

end NUMINAMATH_CALUDE_max_clock_digit_sum_l3238_323876


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3238_323818

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_arith_mean : (a + b) / 2 = 5 / 2) (h_geom_mean : Real.sqrt (a * b) = Real.sqrt 6) :
  let c := Real.sqrt (a^2 - b^2)
  (c / a) = Real.sqrt 13 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3238_323818


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3238_323859

theorem rectangle_area_increase (l w : ℝ) (h_l : l > 0) (h_w : w > 0) :
  let new_area := (1.15 * l) * (1.25 * w)
  let orig_area := l * w
  (new_area - orig_area) / orig_area = 0.4375 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3238_323859


namespace NUMINAMATH_CALUDE_tv_price_increase_l3238_323873

theorem tv_price_increase (x : ℝ) : 
  (((1 + x / 100) * 0.8 - 1) * 100 = 28) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l3238_323873


namespace NUMINAMATH_CALUDE_difference_of_squares_value_l3238_323878

theorem difference_of_squares_value (x y : ℤ) (hx : x = -5) (hy : y = -10) :
  (y - x) * (y + x) = 75 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_value_l3238_323878


namespace NUMINAMATH_CALUDE_jeremy_song_count_l3238_323848

/-- The number of songs Jeremy listened to yesterday -/
def songs_yesterday : ℕ := 9

/-- The difference in songs between today and yesterday -/
def song_difference : ℕ := 5

/-- The number of songs Jeremy listened to today -/
def songs_today : ℕ := songs_yesterday + song_difference

/-- The total number of songs Jeremy listened to in two days -/
def total_songs : ℕ := songs_yesterday + songs_today

theorem jeremy_song_count : total_songs = 23 := by sorry

end NUMINAMATH_CALUDE_jeremy_song_count_l3238_323848


namespace NUMINAMATH_CALUDE_age_difference_proof_l3238_323835

def zion_age : ℕ := 8

def dad_age : ℕ := 4 * zion_age + 3

def age_difference_after_10_years : ℕ :=
  (dad_age + 10) - (zion_age + 10)

theorem age_difference_proof :
  age_difference_after_10_years = 27 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3238_323835


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l3238_323894

theorem fair_coin_probability_difference : 
  let n : ℕ := 5
  let p : ℚ := 1/2
  let prob_4_heads := (n.choose 4) * p^4 * (1-p)
  let prob_5_heads := p^n
  abs (prob_4_heads - prob_5_heads) = 9/32 := by
sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l3238_323894


namespace NUMINAMATH_CALUDE_wall_length_calculation_l3238_323890

theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) : 
  mirror_side = 21 →
  wall_width = 28 →
  (mirror_side ^ 2) * 2 = wall_width * (31.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l3238_323890


namespace NUMINAMATH_CALUDE_sector_radius_l3238_323885

theorem sector_radius (α : Real) (S : Real) (r : Real) : 
  α = 3/4 * Real.pi → 
  S = 3/2 * Real.pi → 
  S = 1/2 * r^2 * α → 
  r = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l3238_323885


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_40_l3238_323832

/-- The maximum area of a rectangle with a perimeter of 40 units is 100 square units. -/
theorem max_area_rectangle_with_perimeter_40 :
  ∃ (length width : ℝ),
    length > 0 ∧ 
    width > 0 ∧
    2 * (length + width) = 40 ∧
    length * width = 100 ∧
    ∀ (l w : ℝ), l > 0 → w > 0 → 2 * (l + w) = 40 → l * w ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_40_l3238_323832


namespace NUMINAMATH_CALUDE_distance_AB_is_5360_l3238_323892

/-- Represents a person in the problem -/
inductive Person
| A
| B
| C

/-- Represents a point on the path -/
structure Point where
  x : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  A : Point
  B : Point
  C : Point
  D : Point
  initialSpeed : Person → ℝ
  returnSpeed : Person → ℝ
  distanceTraveled : Person → Point → ℝ

/-- The main theorem to be proved -/
theorem distance_AB_is_5360 (setup : ProblemSetup) : 
  setup.B.x - setup.A.x = 5360 :=
sorry

end NUMINAMATH_CALUDE_distance_AB_is_5360_l3238_323892


namespace NUMINAMATH_CALUDE_jones_wardrobe_l3238_323805

/-- The ratio of shirts to pants in Mr. Jones' wardrobe -/
def shirt_to_pants_ratio : ℕ := 6

/-- The number of pants Mr. Jones owns -/
def number_of_pants : ℕ := 40

/-- The total number of pieces of clothes Mr. Jones owns -/
def total_clothes : ℕ := shirt_to_pants_ratio * number_of_pants + number_of_pants

theorem jones_wardrobe : total_clothes = 280 := by
  sorry

end NUMINAMATH_CALUDE_jones_wardrobe_l3238_323805


namespace NUMINAMATH_CALUDE_parabola_equation_l3238_323858

/-- Given a parabola and a line intersecting it, prove the equation of the parabola. -/
theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y = Real.sqrt 3 * x + (A.2 - Real.sqrt 3 * A.1)) →  -- Line equation
  (∀ x y, x^2 = 2 * p * y) →  -- Parabola equation
  A.1^2 = 2 * p * A.2 →  -- Point A satisfies parabola equation
  B.1^2 = 2 * p * B.2 →  -- Point B satisfies parabola equation
  A.2 = Real.sqrt 3 * A.1 + (A.2 - Real.sqrt 3 * A.1) →  -- Point A satisfies line equation
  B.2 = Real.sqrt 3 * B.1 + (A.2 - Real.sqrt 3 * A.1) →  -- Point B satisfies line equation
  A.1 + B.1 = 3 →  -- Sum of x-coordinates
  (∀ x y, x^2 = Real.sqrt 3 * y) :=  -- Conclusion: equation of the parabola
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3238_323858


namespace NUMINAMATH_CALUDE_polynomial_coefficient_l3238_323884

theorem polynomial_coefficient (a : Fin 11 → ℝ) :
  (∀ x : ℝ, x^2 + x^10 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + 
    a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + 
    a 6 * (x + 1)^6 + a 7 * (x + 1)^7 + a 8 * (x + 1)^8 + 
    a 9 * (x + 1)^9 + a 10 * (x + 1)^10) →
  a 9 = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_l3238_323884


namespace NUMINAMATH_CALUDE_m_range_theorem_l3238_323837

def is_ellipse_with_y_foci (m : ℝ) : Prop :=
  0 < m ∧ m < 3

def hyperbola_eccentricity_in_range (m : ℝ) : Prop :=
  m > 0 ∧ Real.sqrt 1.5 < (1 + m/5).sqrt ∧ (1 + m/5).sqrt < Real.sqrt 2

def p (m : ℝ) : Prop := is_ellipse_with_y_foci m
def q (m : ℝ) : Prop := hyperbola_eccentricity_in_range m

theorem m_range_theorem (m : ℝ) :
  (0 < m ∧ m < 9) →
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  ((0 < m ∧ m ≤ 5/2) ∨ (3 ≤ m ∧ m < 5)) :=
by
  sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3238_323837


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3238_323815

-- Define the vectors i and j
def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

-- Define vectors a and b
def a : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Define the dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Theorem statement
theorem perpendicular_vectors_k_value :
  ∃ k : ℝ, perpendicular a (b k) ∧ k = 6 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3238_323815


namespace NUMINAMATH_CALUDE_base5_to_decimal_conversion_l3238_323898

/-- Converts a base-5 digit to its decimal (base-10) value -/
def base5ToDecimal (digit : Nat) : Nat :=
  digit

/-- Converts a base-5 number to its decimal (base-10) equivalent -/
def convertBase5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + (base5ToDecimal d) * (5 ^ i)) 0

/-- The base-5 representation of the number -/
def base5Number : List Nat := [2, 1, 4, 3, 2]

theorem base5_to_decimal_conversion :
  convertBase5ToDecimal base5Number = 1732 := by
  sorry

end NUMINAMATH_CALUDE_base5_to_decimal_conversion_l3238_323898


namespace NUMINAMATH_CALUDE_advanced_tablet_price_relationship_l3238_323855

/-- The price of a smartphone in dollars. -/
def smartphone_price : ℕ := 300

/-- The price difference between a personal computer and a smartphone in dollars. -/
def pc_price_difference : ℕ := 500

/-- The total cost of buying one of each product (smartphone, personal computer, and advanced tablet) in dollars. -/
def total_cost : ℕ := 2200

/-- The price of a personal computer in dollars. -/
def pc_price : ℕ := smartphone_price + pc_price_difference

/-- The price of an advanced tablet in dollars. -/
def advanced_tablet_price : ℕ := total_cost - (smartphone_price + pc_price)

theorem advanced_tablet_price_relationship :
  advanced_tablet_price = smartphone_price + pc_price - 400 := by
  sorry

end NUMINAMATH_CALUDE_advanced_tablet_price_relationship_l3238_323855


namespace NUMINAMATH_CALUDE_canoe_production_sum_l3238_323886

theorem canoe_production_sum : 
  let a : ℕ := 8  -- first term
  let r : ℕ := 3  -- common ratio
  let n : ℕ := 8  -- number of terms
  let sum := a * (r^n - 1) / (r - 1)
  sum = 26240 := by sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l3238_323886


namespace NUMINAMATH_CALUDE_set_four_subsets_implies_a_not_zero_or_two_l3238_323896

theorem set_four_subsets_implies_a_not_zero_or_two (a : ℝ) : 
  (Finset.powerset {a, a^2 - a}).card = 4 → a ≠ 0 ∧ a ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_set_four_subsets_implies_a_not_zero_or_two_l3238_323896


namespace NUMINAMATH_CALUDE_total_staff_weekdays_and_weekends_l3238_323827

def weekday_chefs : ℕ := 16
def weekday_waiters : ℕ := 16
def weekday_busboys : ℕ := 10
def weekday_hostesses : ℕ := 5

def weekend_additional_chefs : ℕ := 5
def weekend_additional_hostesses : ℕ := 2

def chef_leave_percentage : ℚ := 25 / 100
def waiter_leave_percentage : ℚ := 20 / 100
def busboy_leave_percentage : ℚ := 30 / 100
def hostess_leave_percentage : ℚ := 15 / 100

theorem total_staff_weekdays_and_weekends :
  let weekday_chefs_left := weekday_chefs - Int.floor (chef_leave_percentage * weekday_chefs)
  let weekday_waiters_left := weekday_waiters - Int.floor (waiter_leave_percentage * weekday_waiters)
  let weekday_busboys_left := weekday_busboys - Int.floor (busboy_leave_percentage * weekday_busboys)
  let weekday_hostesses_left := weekday_hostesses - Int.floor (hostess_leave_percentage * weekday_hostesses)
  
  let weekday_total := weekday_chefs_left + weekday_waiters_left + weekday_busboys_left + weekday_hostesses_left
  
  let weekend_chefs := weekday_chefs + weekend_additional_chefs
  let weekend_waiters := weekday_waiters_left
  let weekend_busboys := weekday_busboys_left
  let weekend_hostesses := weekday_hostesses + weekend_additional_hostesses
  
  let weekend_total := weekend_chefs + weekend_waiters + weekend_busboys + weekend_hostesses
  
  weekday_total + weekend_total = 84 := by
    sorry

end NUMINAMATH_CALUDE_total_staff_weekdays_and_weekends_l3238_323827


namespace NUMINAMATH_CALUDE_delegation_selection_l3238_323804

theorem delegation_selection (n k : ℕ) (h1 : n = 12) (h2 : k = 3) : 
  Nat.choose n k = 220 := by
  sorry

end NUMINAMATH_CALUDE_delegation_selection_l3238_323804


namespace NUMINAMATH_CALUDE_direct_proportion_implies_m_zero_l3238_323812

/-- A function f is a direct proportion function if there exists a constant k such that f x = k * x for all x -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function y = -2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := -2 * x + m

theorem direct_proportion_implies_m_zero (m : ℝ) :
  is_direct_proportion (f m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_implies_m_zero_l3238_323812


namespace NUMINAMATH_CALUDE_fractional_parts_sum_not_one_l3238_323830

theorem fractional_parts_sum_not_one (x : ℚ) : 
  ¬(x - ⌊x⌋ + x^2 - ⌊x^2⌋ = 1) := by sorry

end NUMINAMATH_CALUDE_fractional_parts_sum_not_one_l3238_323830


namespace NUMINAMATH_CALUDE_goals_scored_l3238_323800

def bruce_goals : ℕ := 4

def michael_goals : ℕ := 3 * bruce_goals

def total_goals : ℕ := bruce_goals + michael_goals

theorem goals_scored : total_goals = 16 := by
  sorry

end NUMINAMATH_CALUDE_goals_scored_l3238_323800


namespace NUMINAMATH_CALUDE_rational_function_zeros_l3238_323869

theorem rational_function_zeros (x : ℝ) : 
  (x^2 - 5*x + 6) / (3*x - 1) = 0 ↔ x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_zeros_l3238_323869


namespace NUMINAMATH_CALUDE_joint_equation_solver_l3238_323807

/-- Given two equations and two solutions, prove the value of a specific expression --/
theorem joint_equation_solver (a b : ℤ) :
  (a * (-3) + 5 * (-1) = 15) →
  (4 * (-3) - b * (-1) = -2) →
  (a * 5 + 5 * 4 = 15) →
  (4 * 5 - b * 4 = -2) →
  a^2018 + (-1/10 * b : ℚ)^2019 = 0 := by
  sorry

end NUMINAMATH_CALUDE_joint_equation_solver_l3238_323807


namespace NUMINAMATH_CALUDE_same_color_probability_l3238_323825

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 9

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selection_size : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
theorem same_color_probability : 
  (num_pairs : ℚ) / (total_shoes.choose selection_size) = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3238_323825


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l3238_323824

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 2*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 4*x - 2

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) → y = 5*x - 5 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l3238_323824


namespace NUMINAMATH_CALUDE_be_length_l3238_323866

structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

def is_right_angle (p q r : ℝ × ℝ) : Prop := sorry

def on_line (p q r : ℝ × ℝ) : Prop := sorry

def perpendicular (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem be_length 
  (ABCD : Quadrilateral)
  (E F : ℝ × ℝ)
  (h1 : is_right_angle ABCD.A ABCD.B ABCD.C)
  (h2 : is_right_angle ABCD.B ABCD.C ABCD.D)
  (h3 : on_line ABCD.A E ABCD.C)
  (h4 : on_line ABCD.A F ABCD.C)
  (h5 : perpendicular (ABCD.D, F) (ABCD.A, ABCD.C))
  (h6 : perpendicular (ABCD.B, E) (ABCD.A, ABCD.C))
  (h7 : distance ABCD.A F = 4)
  (h8 : distance ABCD.D F = 6)
  (h9 : distance ABCD.C F = 8)
  : distance ABCD.B E = 16/3 := sorry

end NUMINAMATH_CALUDE_be_length_l3238_323866


namespace NUMINAMATH_CALUDE_continuity_at_one_l3238_323862

def f (x : ℝ) := -5 * x^2 - 7

theorem continuity_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_one_l3238_323862


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3238_323833

theorem initial_money_calculation (X : ℝ) : 
  X - (X / 2 + 50) = 25 → X = 150 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3238_323833


namespace NUMINAMATH_CALUDE_one_intersection_iff_tangent_l3238_323803

-- Define a line
def Line : Type := sorry

-- Define a conic curve
def ConicCurve : Type := sorry

-- Define the property of having only one intersection point
def hasOneIntersectionPoint (l : Line) (c : ConicCurve) : Prop := sorry

-- Define the property of being tangent
def isTangent (l : Line) (c : ConicCurve) : Prop := sorry

-- Theorem stating that having one intersection point is both sufficient and necessary for being tangent
theorem one_intersection_iff_tangent (l : Line) (c : ConicCurve) : 
  hasOneIntersectionPoint l c ↔ isTangent l c := by sorry

end NUMINAMATH_CALUDE_one_intersection_iff_tangent_l3238_323803


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3238_323828

theorem fraction_inequality_solution_set (x : ℝ) :
  (x + 1) / (x + 2) < 0 ↔ -2 < x ∧ x < -1 :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3238_323828


namespace NUMINAMATH_CALUDE_rationalize_result_l3238_323891

def rationalize_denominator (a b c : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ := sorry

theorem rationalize_result :
  let (A, B, C, D, E) := rationalize_denominator 5 7 13
  A = -4 ∧ B = 7 ∧ C = 3 ∧ D = 13 ∧ E = 1 ∧ B < D ∧
  A * Real.sqrt B + C * Real.sqrt D = 5 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) * E :=
by sorry

end NUMINAMATH_CALUDE_rationalize_result_l3238_323891


namespace NUMINAMATH_CALUDE_dawn_monthly_payments_l3238_323801

/-- Dawn's annual salary in dollars -/
def annual_salary : ℕ := 48000

/-- Dawn's monthly savings rate as a fraction -/
def savings_rate : ℚ := 1/10

/-- Dawn's monthly savings in dollars -/
def monthly_savings : ℕ := 400

/-- The number of months in a year -/
def months_in_year : ℕ := 12

theorem dawn_monthly_payments :
  (annual_salary / months_in_year : ℚ) * savings_rate = monthly_savings :=
sorry

end NUMINAMATH_CALUDE_dawn_monthly_payments_l3238_323801


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3238_323877

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  (a 3 / a 1) ^ 2 = a 4 / a 1

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) (h_geom : geometric_subseq a) : 
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3238_323877


namespace NUMINAMATH_CALUDE_class_fund_solution_l3238_323823

/-- Represents the number of bills in a class fund -/
structure ClassFund where
  bills_10 : ℕ
  bills_20 : ℕ

/-- Calculates the total amount in the fund -/
def total_amount (fund : ClassFund) : ℕ :=
  10 * fund.bills_10 + 20 * fund.bills_20

theorem class_fund_solution :
  ∃ (fund : ClassFund),
    total_amount fund = 120 ∧
    fund.bills_10 = 2 * fund.bills_20 ∧
    fund.bills_20 = 3 := by
  sorry

end NUMINAMATH_CALUDE_class_fund_solution_l3238_323823


namespace NUMINAMATH_CALUDE_some_number_equation_l3238_323870

theorem some_number_equation (n : ℤ) (y : ℤ) : 
  (n * (1 + y) + 17 = n * (-1 + y) - 21) → n = -19 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l3238_323870


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3238_323872

/-- Given a function f: ℝ → ℝ whose graph is tangent to the line 2x+y-1=0 at the point (1,f(1)),
    prove that f(1) + f'(1) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, 2*x + f x - 1 = 0 ↔ x = 1) : 
    f 1 + deriv f 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3238_323872


namespace NUMINAMATH_CALUDE_sqrt_144_squared_times_2_l3238_323883

theorem sqrt_144_squared_times_2 : 2 * (Real.sqrt 144)^2 = 288 := by sorry

end NUMINAMATH_CALUDE_sqrt_144_squared_times_2_l3238_323883


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l3238_323846

theorem dave_guitar_strings 
  (strings_per_night : ℕ) 
  (shows_per_week : ℕ) 
  (total_weeks : ℕ) 
  (h1 : strings_per_night = 2) 
  (h2 : shows_per_week = 6) 
  (h3 : total_weeks = 12) : 
  strings_per_night * shows_per_week * total_weeks = 144 := by
sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l3238_323846


namespace NUMINAMATH_CALUDE_tan_two_alpha_plus_pi_l3238_323847

-- Define the angle α
def α : Real := sorry

-- Define the conditions
axiom vertex_at_origin : True
axiom initial_side_on_x_axis : True
axiom terminal_side_on_line : ∀ (x y : Real), y = Real.sqrt 3 * x → (∃ t : Real, t > 0 ∧ x = t * Real.cos α ∧ y = t * Real.sin α)

-- State the theorem
theorem tan_two_alpha_plus_pi : Real.tan (2 * α + Real.pi) = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_two_alpha_plus_pi_l3238_323847


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3238_323853

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : Real.cos (2 * α) + Real.cos α ^ 2 = 0) : 
  Real.tan (α + Real.pi / 4) = -3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3238_323853


namespace NUMINAMATH_CALUDE_fraction_equality_l3238_323879

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hyx : y - x^2 ≠ 0) :
  (x^2 - 1/y) / (y - x^2) = (x^2 * y - 1) / (y^2 - x^2 * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3238_323879


namespace NUMINAMATH_CALUDE_even_function_implies_a_squared_one_l3238_323880

def f (x a : ℝ) : ℝ := x^2 + (a^2 - 1)*x + 6

theorem even_function_implies_a_squared_one (a : ℝ) :
  (∀ x, f x a = f (-x) a) → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_squared_one_l3238_323880


namespace NUMINAMATH_CALUDE_friendship_divisibility_criterion_l3238_323882

/-- Represents a friendship relation between students -/
def FriendshipRelation (n : ℕ) := Fin n → Fin n → Prop

/-- The friendship relation is symmetric -/
def symmetric {n : ℕ} (r : FriendshipRelation n) :=
  ∀ i j, r i j ↔ r j i

/-- The friendship relation is irreflexive -/
def irreflexive {n : ℕ} (r : FriendshipRelation n) :=
  ∀ i, ¬(r i i)

/-- Theorem: For any finite set of students with a friendship relation,
    there exists a positive integer N and an assignment of integers to students
    such that two students are friends if and only if N divides the product of their assigned integers -/
theorem friendship_divisibility_criterion
  {n : ℕ} (r : FriendshipRelation n) (h_sym : symmetric r) (h_irr : irreflexive r) :
  ∃ (N : ℕ) (N_pos : 0 < N) (a : Fin n → ℤ),
    ∀ i j, r i j ↔ (N : ℤ) ∣ (a i * a j) :=
sorry

end NUMINAMATH_CALUDE_friendship_divisibility_criterion_l3238_323882


namespace NUMINAMATH_CALUDE_sugar_water_sweetness_l3238_323865

theorem sugar_water_sweetness (a b m : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : m > 0) :
  (a + m) / (b + m) > a / b :=
by sorry

end NUMINAMATH_CALUDE_sugar_water_sweetness_l3238_323865


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_1000_l3238_323845

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sum_of_digits_up_to (n : ℕ) : ℕ := 
  (Finset.range n).sum (λ i => sum_of_digits (i + 1))

/-- Theorem: The sum of digits of all numbers from 1 to 1000 is 14446 -/
theorem sum_of_digits_1_to_1000 : sum_of_digits_up_to 1000 = 14446 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_1000_l3238_323845


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3238_323893

/-- Given that i is the imaginary unit, prove that (1+2i)/(1+i) = (3+i)/2 -/
theorem complex_fraction_equality : (1 + 2 * Complex.I) / (1 + Complex.I) = (3 + Complex.I) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3238_323893


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3238_323829

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 - x) ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3238_323829


namespace NUMINAMATH_CALUDE_exponent_equation_l3238_323897

theorem exponent_equation (a b : ℤ) : 3^a * 9^b = (1:ℚ)/3 → a + 2*b = -1 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_l3238_323897


namespace NUMINAMATH_CALUDE_midpoint_sum_invariant_l3238_323836

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Create a new polygon by connecting midpoints of sides -/
def midpointPolygon (P : Polygon) : Polygon :=
  sorry

/-- Sum of x-coordinates of vertices -/
def sumXCoordinates (P : Polygon) : ℝ :=
  sorry

theorem midpoint_sum_invariant (Q1 : Polygon) 
  (h1 : Q1.vertices.length = 45)
  (h2 : sumXCoordinates Q1 = 135) : 
  let Q2 := midpointPolygon Q1
  let Q3 := midpointPolygon Q2
  sumXCoordinates Q3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_invariant_l3238_323836


namespace NUMINAMATH_CALUDE_triangle_inequality_l3238_323871

theorem triangle_inequality (A B C : ℝ) (x y z : ℝ) (n : ℕ) 
  (h_triangle : A + B + C = π) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0) : 
  x^n * Real.cos (A/2) + y^n * Real.cos (B/2) + z^n * Real.cos (C/2) ≥ 
  (y*z)^(n/2) * Real.sin A + (z*x)^(n/2) * Real.sin B + (x*y)^(n/2) * Real.sin C := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3238_323871


namespace NUMINAMATH_CALUDE_annika_hiking_distance_l3238_323817

/-- Annika's hiking problem -/
theorem annika_hiking_distance 
  (rate : ℝ) -- Hiking rate in minutes per kilometer
  (initial_distance : ℝ) -- Initial distance hiked east in kilometers
  (total_time : ℝ) -- Total time available in minutes
  (h_rate : rate = 10) -- Hiking rate is 10 minutes per kilometer
  (h_initial : initial_distance = 2.5) -- Initial distance is 2.5 kilometers
  (h_time : total_time = 45) -- Total available time is 45 minutes
  : ∃ (total_east : ℝ), total_east = 3.5 ∧ 
    2 * (total_east - initial_distance) * rate + initial_distance * rate = total_time :=
by sorry

end NUMINAMATH_CALUDE_annika_hiking_distance_l3238_323817


namespace NUMINAMATH_CALUDE_regular_hexagon_side_length_l3238_323821

/-- A regular hexagon with opposite sides 18 inches apart has side length 12√3 inches -/
theorem regular_hexagon_side_length (h : RegularHexagon) 
  (opposite_sides_distance : ℝ) (side_length : ℝ) : 
  opposite_sides_distance = 18 → side_length = 12 * Real.sqrt 3 := by
  sorry

#check regular_hexagon_side_length

end NUMINAMATH_CALUDE_regular_hexagon_side_length_l3238_323821


namespace NUMINAMATH_CALUDE_distribute_seven_books_four_friends_l3238_323822

/-- The number of ways to distribute n identical books among k friends, 
    where each friend must have at least one book -/
def distribute_books (n k : ℕ) : ℕ := sorry

/-- Theorem: Distributing 7 books among 4 friends results in 34 ways -/
theorem distribute_seven_books_four_friends : 
  distribute_books 7 4 = 34 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_books_four_friends_l3238_323822


namespace NUMINAMATH_CALUDE_name_calculation_result_l3238_323814

/-- Represents the alphabetical position of a letter (A=1, B=2, ..., Z=26) -/
def alphabeticalPosition (c : Char) : Nat :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26
  | _ => 0

theorem name_calculation_result :
  let elida := "ELIDA"
  let adrianna := "ADRIANNA"
  let belinda := "BELINDA"

  let elida_sum := (elida.data.map alphabeticalPosition).sum
  let adrianna_sum := (adrianna.data.map alphabeticalPosition).sum
  let belinda_sum := (belinda.data.map alphabeticalPosition).sum

  let total_sum := elida_sum + adrianna_sum + belinda_sum
  let average := total_sum / 3

  elida.length = 5 →
  adrianna.length = 2 * elida.length - 2 →
  (average * 3 : ℕ) - elida_sum = 109 := by
  sorry

#check name_calculation_result

end NUMINAMATH_CALUDE_name_calculation_result_l3238_323814


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3238_323868

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) ↔
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3238_323868


namespace NUMINAMATH_CALUDE_square_side_equations_l3238_323867

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- Represents a square in 2D space --/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ
  parallel_line : Line

/-- Check if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are perpendicular --/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem --/
theorem square_side_equations (s : Square)
  (h1 : s.center = (-3, -4))
  (h2 : s.side_length = 2 * Real.sqrt 5)
  (h3 : s.parallel_line = ⟨2, 1, 3, Or.inl (by norm_num)⟩) :
  ∃ (l1 l2 l3 l4 : Line),
    (l1 = ⟨2, 1, 15, Or.inl (by norm_num)⟩) ∧
    (l2 = ⟨2, 1, 5, Or.inl (by norm_num)⟩) ∧
    (l3 = ⟨1, -2, 0, Or.inr (by norm_num)⟩) ∧
    (l4 = ⟨1, -2, -10, Or.inr (by norm_num)⟩) ∧
    are_parallel l1 s.parallel_line ∧
    are_parallel l2 s.parallel_line ∧
    are_perpendicular l1 l3 ∧
    are_perpendicular l1 l4 ∧
    are_perpendicular l2 l3 ∧
    are_perpendicular l2 l4 :=
  sorry

end NUMINAMATH_CALUDE_square_side_equations_l3238_323867


namespace NUMINAMATH_CALUDE_fish_ratio_l3238_323863

theorem fish_ratio (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) (total : ℕ) :
  bass = 32 →
  trout = bass / 4 →
  total = 104 →
  total = bass + trout + blue_gill →
  blue_gill / bass = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_ratio_l3238_323863


namespace NUMINAMATH_CALUDE_two_true_propositions_l3238_323813

theorem two_true_propositions (a b c : ℝ) : 
  (∃! n : ℕ, n = 2 ∧ 
    n = (if (a > b → a*c^2 > b*c^2) then 1 else 0) +
        (if (a*c^2 > b*c^2 → a > b) then 1 else 0) +
        (if (a ≤ b → a*c^2 ≤ b*c^2) then 1 else 0) +
        (if (a*c^2 ≤ b*c^2 → a ≤ b) then 1 else 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l3238_323813


namespace NUMINAMATH_CALUDE_curvilinearTrapezoidAreaStepsCorrect_l3238_323857

/-- The steps required to calculate the area of a curvilinear trapezoid. -/
inductive CurvilinearTrapezoidAreaStep
  | division
  | approximation
  | summation
  | takingLimit

/-- The list of steps to calculate the area of a curvilinear trapezoid. -/
def curvilinearTrapezoidAreaSteps : List CurvilinearTrapezoidAreaStep :=
  [CurvilinearTrapezoidAreaStep.division,
   CurvilinearTrapezoidAreaStep.approximation,
   CurvilinearTrapezoidAreaStep.summation,
   CurvilinearTrapezoidAreaStep.takingLimit]

/-- Theorem stating that the steps to calculate the area of a curvilinear trapezoid
    are division, approximation, summation, and taking the limit. -/
theorem curvilinearTrapezoidAreaStepsCorrect :
  curvilinearTrapezoidAreaSteps =
    [CurvilinearTrapezoidAreaStep.division,
     CurvilinearTrapezoidAreaStep.approximation,
     CurvilinearTrapezoidAreaStep.summation,
     CurvilinearTrapezoidAreaStep.takingLimit] := by
  sorry

end NUMINAMATH_CALUDE_curvilinearTrapezoidAreaStepsCorrect_l3238_323857


namespace NUMINAMATH_CALUDE_garden_area_l3238_323874

theorem garden_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  area = width * length →
  area = 243 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l3238_323874


namespace NUMINAMATH_CALUDE_meaningful_range_l3238_323895

def is_meaningful (x : ℝ) : Prop :=
  x - 1 ≥ 0 ∧ x ≠ 3

theorem meaningful_range : 
  ∀ x : ℝ, is_meaningful x ↔ x ≥ 1 ∧ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_l3238_323895


namespace NUMINAMATH_CALUDE_revenue_growth_equation_l3238_323839

/-- Represents the average monthly growth rate of revenue -/
def x : ℝ := sorry

/-- Represents the revenue in January in thousands of dollars -/
def january_revenue : ℝ := 36

/-- Represents the revenue in March in thousands of dollars -/
def march_revenue : ℝ := 48

/-- Theorem stating that the equation representing the revenue growth is 36(1+x)^2 = 48 -/
theorem revenue_growth_equation : 
  january_revenue * (1 + x)^2 = march_revenue := by sorry

end NUMINAMATH_CALUDE_revenue_growth_equation_l3238_323839


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3238_323840

theorem sqrt_equation_solution (x : ℚ) :
  Real.sqrt (2 - 5 * x) = 8 → x = -62 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3238_323840


namespace NUMINAMATH_CALUDE_choir_members_count_l3238_323841

theorem choir_members_count : ∃! n : ℕ, 
  200 ≤ n ∧ n ≤ 300 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 ∧ 
  n = 226 := by sorry

end NUMINAMATH_CALUDE_choir_members_count_l3238_323841


namespace NUMINAMATH_CALUDE_proposition_1_proposition_3_l3238_323887

-- Proposition ①
theorem proposition_1 : ∀ a b : ℝ, (a + b ≠ 5) → (a ≠ 2 ∨ b ≠ 3) := by sorry

-- Proposition ③
theorem proposition_3 : 
  (∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ∧ 
  (∀ ε > 0, ∃ x : ℝ, x > 0 ∧ x + 1/x < 2 + ε) := by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_3_l3238_323887


namespace NUMINAMATH_CALUDE_playground_children_count_l3238_323861

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 27) 
  (h2 : girls = 35) : 
  boys + girls = 62 := by
sorry

end NUMINAMATH_CALUDE_playground_children_count_l3238_323861


namespace NUMINAMATH_CALUDE_system_solution_unique_l3238_323811

theorem system_solution_unique :
  ∃! (x y z : ℝ), 5 * x + 3 * y = 65 ∧ 2 * y - z = 11 ∧ 3 * x + 4 * z = 57 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3238_323811


namespace NUMINAMATH_CALUDE_original_number_proof_l3238_323854

theorem original_number_proof : ∃ (n : ℕ), n + 859560 ≡ 0 [MOD 456] ∧ n = 696 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3238_323854


namespace NUMINAMATH_CALUDE_lemon_price_increase_l3238_323810

/-- Proves that the increase in lemon price is $4 given the conditions of Erick's fruit sale --/
theorem lemon_price_increase :
  ∀ (x : ℝ),
    (80 * (8 + x) + 140 * (7 + x / 2) = 2220) →
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_lemon_price_increase_l3238_323810


namespace NUMINAMATH_CALUDE_apple_distribution_l3238_323820

theorem apple_distribution (x : ℕ) (total_apples : ℕ) : 
  (total_apples = 3 * x + 8) →
  (total_apples > 5 * (x - 1) ∧ total_apples < 5 * x) →
  ((x = 5 ∧ total_apples = 23) ∨ (x = 6 ∧ total_apples = 26)) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l3238_323820


namespace NUMINAMATH_CALUDE_magnified_diameter_calculation_l3238_323881

/-- Given a circular piece of tissue with an actual diameter and a magnification factor,
    calculate the diameter of the magnified image. -/
theorem magnified_diameter_calculation
  (actual_diameter : ℝ)
  (magnification_factor : ℝ)
  (h1 : actual_diameter = 0.0002)
  (h2 : magnification_factor = 1000) :
  actual_diameter * magnification_factor = 0.2 := by
sorry

end NUMINAMATH_CALUDE_magnified_diameter_calculation_l3238_323881


namespace NUMINAMATH_CALUDE_solve_equation_l3238_323819

theorem solve_equation (x : ℝ) (y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) 
  (h2 : y = 3*x + 1) (h3 : x ≠ 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3238_323819


namespace NUMINAMATH_CALUDE_bike_travel_time_l3238_323844

-- Constants
def highway_length : Real := 5280  -- in feet
def highway_width : Real := 50     -- in feet
def bike_speed : Real := 6         -- in miles per hour

-- Theorem
theorem bike_travel_time :
  let semicircle_radius : Real := highway_width / 2
  let num_semicircles : Real := highway_length / highway_width
  let total_distance : Real := num_semicircles * (π * semicircle_radius)
  let total_distance_miles : Real := total_distance / 5280
  let time_taken : Real := total_distance_miles / bike_speed
  time_taken = π / 12 := by sorry

end NUMINAMATH_CALUDE_bike_travel_time_l3238_323844


namespace NUMINAMATH_CALUDE_mouse_testes_most_appropriate_l3238_323802

-- Define the possible experimental materials
inductive ExperimentalMaterial
| AscarisEggs
| ChickenLiver
| MouseTestes
| OnionEpidermis

-- Define the cell division processes
inductive CellDivisionProcess
| Mitosis
| Meiosis
| NoDivision

-- Define the property of continuous cell formation
def hasContinuousCellFormation : ExperimentalMaterial → Prop
| ExperimentalMaterial.MouseTestes => True
| _ => False

-- Define the cell division process for each material
def cellDivisionProcess : ExperimentalMaterial → CellDivisionProcess
| ExperimentalMaterial.AscarisEggs => CellDivisionProcess.Mitosis
| ExperimentalMaterial.ChickenLiver => CellDivisionProcess.Mitosis
| ExperimentalMaterial.MouseTestes => CellDivisionProcess.Meiosis
| ExperimentalMaterial.OnionEpidermis => CellDivisionProcess.NoDivision

-- Define the property of being appropriate for observing meiosis
def isAppropriateForMeiosis (material : ExperimentalMaterial) : Prop :=
  cellDivisionProcess material = CellDivisionProcess.Meiosis ∧ hasContinuousCellFormation material

-- Theorem statement
theorem mouse_testes_most_appropriate :
  ∀ material : ExperimentalMaterial,
    isAppropriateForMeiosis material → material = ExperimentalMaterial.MouseTestes :=
by
  sorry

end NUMINAMATH_CALUDE_mouse_testes_most_appropriate_l3238_323802


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l3238_323838

theorem quadratic_function_k_value (a b c k : ℤ) (f : ℝ → ℝ) :
  (∀ x, f x = a * x^2 + b * x + c) →
  f 1 = 0 →
  50 < f 7 ∧ f 7 < 60 →
  70 < f 8 ∧ f 8 < 80 →
  5000 * k < f 100 ∧ f 100 < 5000 * (k + 1) →
  k = 3 := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_k_value_l3238_323838


namespace NUMINAMATH_CALUDE_unique_right_triangle_perimeter_area_ratio_l3238_323831

theorem unique_right_triangle_perimeter_area_ratio :
  ∃! (a b : ℝ), a > 0 ∧ b > 0 ∧
  (a + b + Real.sqrt (a^2 + b^2)) / ((1/2) * a * b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_right_triangle_perimeter_area_ratio_l3238_323831


namespace NUMINAMATH_CALUDE_B_power_48_l3238_323816

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 0; 0, 0, 2; 0, -2, 0]

theorem B_power_48 : 
  B^48 = !![0, 0, 0; 0, 16^12, 0; 0, 0, 16^12] := by sorry

end NUMINAMATH_CALUDE_B_power_48_l3238_323816


namespace NUMINAMATH_CALUDE_prime_or_composite_a4_3a2_9_l3238_323809

theorem prime_or_composite_a4_3a2_9 (a : ℕ) :
  (a = 1 ∨ a = 2 → Nat.Prime (a^4 - 3*a^2 + 9)) ∧
  (a > 2 → ¬Nat.Prime (a^4 - 3*a^2 + 9)) :=
by sorry

end NUMINAMATH_CALUDE_prime_or_composite_a4_3a2_9_l3238_323809


namespace NUMINAMATH_CALUDE_lcm_of_5_6_10_12_l3238_323875

theorem lcm_of_5_6_10_12 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 12)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_10_12_l3238_323875


namespace NUMINAMATH_CALUDE_missing_items_count_l3238_323806

def initial_tshirts : ℕ := 9

def initial_sweaters (t : ℕ) : ℕ := 2 * t

def final_sweaters : ℕ := 3

def final_tshirts (t : ℕ) : ℕ := 3 * t

def missing_items (init_t init_s final_t final_s : ℕ) : ℕ :=
  if final_t > init_t
  then init_s - final_s
  else (init_t - final_t) + (init_s - final_s)

theorem missing_items_count :
  missing_items initial_tshirts (initial_sweaters initial_tshirts) 
                (final_tshirts initial_tshirts) final_sweaters = 15 := by
  sorry

end NUMINAMATH_CALUDE_missing_items_count_l3238_323806


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3238_323899

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) (perimeter : x + y = 24) :
  x * y ≤ 144 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 24 ∧ a * b = 144 := by
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3238_323899


namespace NUMINAMATH_CALUDE_expansion_terms_count_l3238_323850

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^12 -/
def dissimilarTerms : ℕ :=
  Nat.choose 15 3

/-- The number of ways to distribute 12 indistinguishable objects into 4 distinguishable boxes -/
def distributionWays : ℕ :=
  Nat.choose (12 + 4 - 1) (4 - 1)

theorem expansion_terms_count :
  dissimilarTerms = distributionWays ∧ dissimilarTerms = 455 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l3238_323850


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3238_323849

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 0 → x ≠ -1 → x ≠ 1 →
  (((1 / x - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1))) = 1 / (x * (x - 1))) ∧
  (((1 / 2 - 1 / 3) / ((2^2 - 1) / (2^2 + 2*2 + 1))) = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3238_323849


namespace NUMINAMATH_CALUDE_rosie_pies_from_36_apples_l3238_323864

/-- Given that Rosie can make three pies out of twelve apples, 
    this function calculates how many pies she can make from a given number of apples. -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples * 3) / 12

theorem rosie_pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

#eval pies_from_apples 36

end NUMINAMATH_CALUDE_rosie_pies_from_36_apples_l3238_323864


namespace NUMINAMATH_CALUDE_simplify_fraction_l3238_323888

theorem simplify_fraction : (111 : ℚ) / 9999 * 33 = 11 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3238_323888


namespace NUMINAMATH_CALUDE_total_ways_eq_17922_l3238_323852

/-- Number of cookie flavors --/
def num_cookie_flavors : ℕ := 7

/-- Number of milk types --/
def num_milk_types : ℕ := 4

/-- Total number of products to purchase --/
def total_products : ℕ := 5

/-- Maximum number of same flavor Alpha can order --/
def alpha_max_same_flavor : ℕ := 2

/-- Function to calculate the number of ways Alpha can choose items --/
def alpha_choices (n : ℕ) : ℕ := sorry

/-- Function to calculate the number of ways Beta can choose cookies --/
def beta_choices (n : ℕ) : ℕ := sorry

/-- The total number of ways Alpha and Beta can purchase 5 products --/
def total_ways : ℕ := sorry

/-- Theorem stating the total number of ways is 17922 --/
theorem total_ways_eq_17922 : total_ways = 17922 := by sorry

end NUMINAMATH_CALUDE_total_ways_eq_17922_l3238_323852


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l3238_323842

theorem min_perimeter_rectangle (area : Real) (perimeter : Real) : 
  area = 64 → perimeter ≥ 32 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l3238_323842


namespace NUMINAMATH_CALUDE_seed_flower_probability_l3238_323826

theorem seed_flower_probability : ∀ (total_seeds small_seeds large_seeds : ℕ)
  (p_small_to_small p_large_to_large : ℝ),
  total_seeds = small_seeds + large_seeds →
  0 ≤ p_small_to_small ∧ p_small_to_small ≤ 1 →
  0 ≤ p_large_to_large ∧ p_large_to_large ≤ 1 →
  total_seeds = 10 →
  small_seeds = 6 →
  large_seeds = 4 →
  p_small_to_small = 0.9 →
  p_large_to_large = 0.8 →
  (small_seeds : ℝ) / (total_seeds : ℝ) * p_small_to_small +
  (large_seeds : ℝ) / (total_seeds : ℝ) * (1 - p_large_to_large) = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_seed_flower_probability_l3238_323826


namespace NUMINAMATH_CALUDE_chord_length_squared_l3238_323843

theorem chord_length_squared (r₁ r₂ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 7) (h₃ : R = 10)
  (h₄ : r₁ > 0) (h₅ : r₂ > 0) (h₆ : R > 0) (h₇ : r₁ + r₂ < R) :
  let d := R - r₂
  ∃ x, x^2 = d^2 + (R - r₁)^2 ∧ 4 * x^2 = 364 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_squared_l3238_323843


namespace NUMINAMATH_CALUDE_cloth_sold_l3238_323834

/-- Proves the number of meters of cloth sold by a shopkeeper -/
theorem cloth_sold (total_price : ℝ) (loss_per_meter : ℝ) (cost_price : ℝ) :
  total_price = 18000 ∧ loss_per_meter = 5 ∧ cost_price = 50 →
  (total_price / (cost_price - loss_per_meter) : ℝ) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sold_l3238_323834


namespace NUMINAMATH_CALUDE_negation_equivalence_l3238_323860

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3238_323860
