import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_factorization_l2008_200872

theorem quadratic_factorization (x : ℝ) : 4 * x^2 - 8 * x + 4 = 4 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2008_200872


namespace NUMINAMATH_CALUDE_janes_farm_chickens_l2008_200861

/-- Represents the farm scenario with chickens and egg production --/
structure Farm where
  chickens : ℕ
  eggs_per_chicken_per_week : ℕ
  price_per_dozen : ℕ
  weeks : ℕ
  total_revenue : ℕ

/-- Calculates the total number of eggs produced by the farm in the given period --/
def total_eggs (f : Farm) : ℕ :=
  f.chickens * f.eggs_per_chicken_per_week * f.weeks

/-- Calculates the revenue generated from selling all eggs --/
def revenue (f : Farm) : ℕ :=
  (total_eggs f / 12) * f.price_per_dozen

/-- Theorem stating that Jane's farm has 10 chickens given the conditions --/
theorem janes_farm_chickens :
  ∃ (f : Farm),
    f.eggs_per_chicken_per_week = 6 ∧
    f.price_per_dozen = 2 ∧
    f.weeks = 2 ∧
    f.total_revenue = 20 ∧
    f.chickens = 10 :=
  sorry

end NUMINAMATH_CALUDE_janes_farm_chickens_l2008_200861


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2008_200844

/-- The repeating decimal 4.363636... -/
def repeating_decimal : ℚ := 4 + 36 / 99

/-- The fraction 144/33 -/
def fraction : ℚ := 144 / 33

/-- Theorem stating that the repeating decimal 4.363636... is equal to the fraction 144/33 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2008_200844


namespace NUMINAMATH_CALUDE_bike_ride_speed_l2008_200887

theorem bike_ride_speed (x : ℝ) : 
  (210 / x = 210 / (x - 5) - 1) → x = 35 := by
sorry

end NUMINAMATH_CALUDE_bike_ride_speed_l2008_200887


namespace NUMINAMATH_CALUDE_projectile_meeting_distance_l2008_200840

theorem projectile_meeting_distance
  (speed1 : ℝ)
  (speed2 : ℝ)
  (meeting_time_minutes : ℝ)
  (h1 : speed1 = 444)
  (h2 : speed2 = 555)
  (h3 : meeting_time_minutes = 120) :
  speed1 * (meeting_time_minutes / 60) + speed2 * (meeting_time_minutes / 60) = 1998 :=
by
  sorry

end NUMINAMATH_CALUDE_projectile_meeting_distance_l2008_200840


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2008_200890

theorem unique_solution_quadratic_system :
  ∃! x : ℚ, (10 * x^2 + 9 * x - 2 = 0) ∧ (30 * x^2 + 59 * x - 6 = 0) ∧ (x = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2008_200890


namespace NUMINAMATH_CALUDE_right_triangle_angle_b_l2008_200816

/-- Given a right triangle ABC with ∠A = 70°, prove that ∠B = 20° -/
theorem right_triangle_angle_b (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  C = 90 →           -- One angle is 90° (right angle)
  A = 70 →           -- Given ∠A = 70°
  B = 20 :=          -- To prove: ∠B = 20°
by sorry

end NUMINAMATH_CALUDE_right_triangle_angle_b_l2008_200816


namespace NUMINAMATH_CALUDE_cards_distribution_l2008_200869

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) 
  (h1 : total_cards = 60) (h2 : total_people = 9) : 
  (total_people - (total_cards % total_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2008_200869


namespace NUMINAMATH_CALUDE_quadratic_and_system_solution_l2008_200876

theorem quadratic_and_system_solution :
  (∃ x₁ x₂ : ℚ, (4 * (x₁ - 1)^2 - 25 = 0 ∧ x₁ = 7/2) ∧
                (4 * (x₂ - 1)^2 - 25 = 0 ∧ x₂ = -3/2)) ∧
  (∃ x y : ℚ, (2*x - y = 4 ∧ 3*x + 2*y = 1) ∧ x = 9/7 ∧ y = -10/7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_and_system_solution_l2008_200876


namespace NUMINAMATH_CALUDE_work_left_for_given_days_l2008_200807

/-- The fraction of work left after two workers collaborate for a given time --/
def work_left (a_days b_days collab_days : ℚ) : ℚ :=
  1 - collab_days * (1 / a_days + 1 / b_days)

/-- Theorem: If A can complete the work in 15 days and B in 20 days,
    then after working together for 3 days, the fraction of work left is 13/20 --/
theorem work_left_for_given_days :
  work_left 15 20 3 = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_work_left_for_given_days_l2008_200807


namespace NUMINAMATH_CALUDE_circle_tangent_triangle_area_l2008_200838

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop := sorry

/-- Checks if two circles are externally tangent -/
def externallyTangent (c1 c2 : Circle) : Prop := sorry

/-- Checks if a line segment is tangent to a circle -/
def isTangent (p1 p2 : Point) (c : Circle) : Prop := sorry

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

theorem circle_tangent_triangle_area 
  (ω₁ ω₂ ω₃ : Circle)
  (P₁ P₂ P₃ : Point)
  (h_radius : ω₁.radius = 24 ∧ ω₂.radius = 24 ∧ ω₃.radius = 24)
  (h_tangent : externallyTangent ω₁ ω₂ ∧ externallyTangent ω₂ ω₃ ∧ externallyTangent ω₃ ω₁)
  (h_on_circle : onCircle P₁ ω₁ ∧ onCircle P₂ ω₂ ∧ onCircle P₃ ω₃)
  (h_equidistant : distance P₁ P₂ = distance P₂ P₃ ∧ distance P₂ P₃ = distance P₃ P₁)
  (h_tangent_sides : isTangent P₁ P₂ ω₂ ∧ isTangent P₂ P₃ ω₃ ∧ isTangent P₃ P₁ ω₁)
  : ∃ (a b : ℕ), triangleArea P₁ P₂ P₃ = Real.sqrt a + Real.sqrt b ∧ a + b = 288 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_triangle_area_l2008_200838


namespace NUMINAMATH_CALUDE_no_integer_solution_l2008_200830

theorem no_integer_solution : ¬∃ (x y : ℤ), x * (x + 1) = 13 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2008_200830


namespace NUMINAMATH_CALUDE_combined_savings_equal_individual_savings_l2008_200825

/-- Represents the number of windows in a bundle -/
def bundle_size : ℕ := 7

/-- Represents the number of windows paid for in a bundle -/
def paid_windows_per_bundle : ℕ := 5

/-- Represents the cost of a single window -/
def window_cost : ℕ := 100

/-- Calculates the number of bundles needed for a given number of windows -/
def bundles_needed (windows : ℕ) : ℕ :=
  (windows + bundle_size - 1) / bundle_size

/-- Calculates the cost of windows with the promotion -/
def promotional_cost (windows : ℕ) : ℕ :=
  bundles_needed windows * paid_windows_per_bundle * window_cost

/-- Calculates the savings for a given number of windows -/
def savings (windows : ℕ) : ℕ :=
  windows * window_cost - promotional_cost windows

/-- Dave's required number of windows -/
def dave_windows : ℕ := 12

/-- Doug's required number of windows -/
def doug_windows : ℕ := 10

theorem combined_savings_equal_individual_savings :
  savings (dave_windows + doug_windows) = savings dave_windows + savings doug_windows :=
by sorry

end NUMINAMATH_CALUDE_combined_savings_equal_individual_savings_l2008_200825


namespace NUMINAMATH_CALUDE_fraction_multiplication_division_main_proof_l2008_200817

theorem fraction_multiplication_division (a b c d e f : ℚ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → e ≠ 0 → f ≠ 0 →
  (a / b * c / d) / (e / f) = (a * c * f) / (b * d * e) :=
by sorry

theorem main_proof : (3 / 4 * 5 / 6) / (7 / 8) = 5 / 7 :=
by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_division_main_proof_l2008_200817


namespace NUMINAMATH_CALUDE_extreme_value_implies_sum_l2008_200870

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- Theorem: If f(x) has an extreme value of 10 at x = 1, then a + b = -7 -/
theorem extreme_value_implies_sum (a b : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = 10 →
  a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_sum_l2008_200870


namespace NUMINAMATH_CALUDE_princess_daphne_necklaces_l2008_200858

def total_cost : ℕ := 240000
def necklace_cost : ℕ := 40000

def number_of_necklaces : ℕ := 3

theorem princess_daphne_necklaces :
  ∃ (n : ℕ), n * necklace_cost + 3 * necklace_cost = total_cost ∧ n = number_of_necklaces :=
by sorry

end NUMINAMATH_CALUDE_princess_daphne_necklaces_l2008_200858


namespace NUMINAMATH_CALUDE_math_competition_problem_l2008_200891

theorem math_competition_problem (a b c : ℕ) 
  (h1 : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h3 : (1/a + 1/b + 1/c - 1/a*1/b - 1/a*1/c - 1/b*1/c + 1/a*1/b*1/c : ℚ) = 7/15) :
  ((1 - 1/a) * (1 - 1/b) * (1 - 1/c) : ℚ) = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_problem_l2008_200891


namespace NUMINAMATH_CALUDE_solution_set_not_negative_interval_l2008_200871

theorem solution_set_not_negative_interval (a b : ℝ) :
  {x : ℝ | a * x > b} ≠ Set.Iio (-b/a) :=
sorry

end NUMINAMATH_CALUDE_solution_set_not_negative_interval_l2008_200871


namespace NUMINAMATH_CALUDE_lost_money_proof_l2008_200827

def money_lost (initial_amount spent_amount remaining_amount : ℕ) : ℕ :=
  (initial_amount - spent_amount) - remaining_amount

theorem lost_money_proof (initial_amount spent_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  money_lost initial_amount spent_amount remaining_amount = 6 := by
  sorry

#eval money_lost 11 2 3

end NUMINAMATH_CALUDE_lost_money_proof_l2008_200827


namespace NUMINAMATH_CALUDE_rectangle_perimeter_16_l2008_200806

def rectangle_perimeter (length width : ℚ) : ℚ := 2 * (length + width)

theorem rectangle_perimeter_16 :
  let length : ℚ := 5
  let width : ℚ := 30 / 10
  rectangle_perimeter length width = 16 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_16_l2008_200806


namespace NUMINAMATH_CALUDE_seq_of_nat_countable_l2008_200897

/-- The set of all sequences of n natural numbers -/
def SeqOfNat (n : ℕ) : Set (Fin n → ℕ) := Set.univ

/-- A set is countable if there exists an injection from the set to ℕ -/
def IsCountable (α : Type*) : Prop := ∃ f : α → ℕ, Function.Injective f

/-- For any natural number n, the set of all sequences of n natural numbers is countable -/
theorem seq_of_nat_countable (n : ℕ) : IsCountable (SeqOfNat n) := by sorry

end NUMINAMATH_CALUDE_seq_of_nat_countable_l2008_200897


namespace NUMINAMATH_CALUDE_lines_perpendicular_l2008_200829

-- Define the slopes of the lines
def slope_l1 : ℚ := -2
def slope_l2 : ℚ := 1/2

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem lines_perpendicular : perpendicular slope_l1 slope_l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l2008_200829


namespace NUMINAMATH_CALUDE_triangle_similarity_l2008_200862

theorem triangle_similarity (DC CB : ℝ) (AD AB ED : ℝ) (FC : ℝ) : 
  DC = 9 → 
  CB = 6 → 
  AB = (1/3) * AD → 
  ED = (2/3) * AD → 
  FC = 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_similarity_l2008_200862


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_k_l2008_200811

theorem quadratic_roots_imply_k (k : ℝ) : 
  (∃ x : ℂ, 5 * x^2 + 7 * x + k = 0 ∧ 
   (x = Complex.mk (-7/10) (Real.sqrt 171 / 10) ∨ 
    x = Complex.mk (-7/10) (-Real.sqrt 171 / 10))) → 
  k = 11 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_k_l2008_200811


namespace NUMINAMATH_CALUDE_corner_to_triangle_ratio_is_one_l2008_200892

/-- Represents a square board with four equally spaced lines passing through its center -/
structure Board :=
  (side_length : ℝ)
  (is_square : side_length > 0)

/-- Represents the area of a triangular section in the board -/
def triangular_area (b : Board) : ℝ := sorry

/-- Represents the area of a corner region in the board -/
def corner_area (b : Board) : ℝ := sorry

/-- Theorem stating that the ratio of corner area to triangular area is 1 for a board with side length 2 -/
theorem corner_to_triangle_ratio_is_one :
  ∀ (b : Board), b.side_length = 2 → corner_area b / triangular_area b = 1 := by sorry

end NUMINAMATH_CALUDE_corner_to_triangle_ratio_is_one_l2008_200892


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2008_200882

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (((2 : ℂ) - a * Complex.I) / ((1 : ℂ) + Complex.I)).re = 0 →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2008_200882


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l2008_200880

/-- The correlation coefficient between two variables -/
def correlation_coefficient (X Y : Type*) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := sorry

/-- The strength of correlation between two variables -/
def correlation_strength (X Y : Type*) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := sorry

theorem correlation_coefficient_properties (X Y : Type*) [NormedAddCommGroup X] [NormedAddCommGroup Y] :
  let r := correlation_coefficient X Y
  ∃ (strength : ℝ → ℝ),
    (∀ x, |x| ≤ 1 → strength x ≥ 0) ∧
    (∀ x y, |x| ≤ 1 → |y| ≤ 1 → |x| < |y| → strength x < strength y) ∧
    (∀ x, |x| ≤ 1 → strength x = correlation_strength X Y) ∧
    |r| ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l2008_200880


namespace NUMINAMATH_CALUDE_loan_amount_proof_l2008_200833

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℝ
  rate : ℝ
  time : ℝ
  interest : ℝ

/-- The loan satisfies the given conditions -/
def loan_conditions (loan : SimpleLoan) : Prop :=
  loan.rate = 0.06 ∧
  loan.time = loan.rate ∧
  loan.interest = 432 ∧
  loan.interest = loan.principal * loan.rate * loan.time

theorem loan_amount_proof (loan : SimpleLoan) 
  (h : loan_conditions loan) : loan.principal = 1200 := by
  sorry

#check loan_amount_proof

end NUMINAMATH_CALUDE_loan_amount_proof_l2008_200833


namespace NUMINAMATH_CALUDE_min_diff_integers_avg_l2008_200889

theorem min_diff_integers_avg (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- Five different positive integers
  (a + b + c + d + e) / 5 = 5 ∧    -- Average is 5
  ∀ x y z w v : ℕ,                 -- For any other set of 5 different positive integers
    x < y ∧ y < z ∧ z < w ∧ w < v ∧
    (x + y + z + w + v) / 5 = 5 →
    (e - a) ≤ (v - x) →            -- with minimum difference
  (b + c + d) / 3 = 5 :=           -- Average of middle three is 5
by sorry

end NUMINAMATH_CALUDE_min_diff_integers_avg_l2008_200889


namespace NUMINAMATH_CALUDE_sequence_value_l2008_200831

theorem sequence_value (a : ℕ → ℚ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = n / (n + 1)) :
  a 8 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_value_l2008_200831


namespace NUMINAMATH_CALUDE_problem_solution_l2008_200805

theorem problem_solution :
  ∀ m n : ℕ+,
  (m : ℝ)^2 - (n : ℝ) = 32 →
  (∃ x : ℝ, x = (m + n^(1/2))^(1/5) + (m - n^(1/2))^(1/5) ∧ x^5 - 10*x^3 + 20*x - 40 = 0) →
  (m : ℕ) + n = 388 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2008_200805


namespace NUMINAMATH_CALUDE_max_min_A_values_l2008_200859

open Complex Real

theorem max_min_A_values (z : ℂ) (h : abs (z - I) ≤ 1) :
  let A := (z.re : ℝ) * ((abs (z - I))^2 - 1)
  ∃ (max_A min_A : ℝ), 
    (∀ z', abs (z' - I) ≤ 1 → (z'.re : ℝ) * ((abs (z' - I))^2 - 1) ≤ max_A) ∧
    (∀ z', abs (z' - I) ≤ 1 → (z'.re : ℝ) * ((abs (z' - I))^2 - 1) ≥ min_A) ∧
    max_A = 2 * Real.sqrt 3 / 9 ∧
    min_A = -2 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_min_A_values_l2008_200859


namespace NUMINAMATH_CALUDE_ellipse_sum_range_l2008_200815

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 144 + y^2 / 25 = 1

-- Theorem statement
theorem ellipse_sum_range :
  ∀ x y : ℝ, is_on_ellipse x y →
  ∃ (a b : ℝ), a = -13 ∧ b = 13 ∧
  a ≤ x + y ∧ x + y ≤ b ∧
  (∃ (x₁ y₁ : ℝ), is_on_ellipse x₁ y₁ ∧ x₁ + y₁ = a) ∧
  (∃ (x₂ y₂ : ℝ), is_on_ellipse x₂ y₂ ∧ x₂ + y₂ = b) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_sum_range_l2008_200815


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2008_200875

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that the given conditions result in the correct principal -/
theorem simple_interest_problem :
  let interest : ℚ := 4016.25
  let rate : ℚ := 3
  let time : ℕ := 5
  calculate_principal interest rate time = 26775 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2008_200875


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2008_200868

-- Define a type for lines in space
structure Line3D where
  -- You might want to add more specific properties here
  -- but for this problem, we just need to distinguish between lines

-- Define parallelism for lines in space
def parallel (l1 l2 : Line3D) : Prop :=
  -- The actual definition of parallelism would go here
  sorry

-- The theorem statement
theorem parallel_transitivity (l m n : Line3D) : 
  parallel l m → parallel l n → parallel m n := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l2008_200868


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2008_200888

theorem sum_of_reciprocals (x y : ℕ+) 
  (sum_eq : x + y = 45)
  (hcf_eq : Nat.gcd x y = 3)
  (lcm_eq : Nat.lcm x y = 100) :
  (1 : ℚ) / x + (1 : ℚ) / y = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2008_200888


namespace NUMINAMATH_CALUDE_wire_length_l2008_200873

/-- Given two vertical poles on a flat surface, where:
    - The distance between the pole bottoms is 8 feet
    - The height of the first pole is 10 feet
    - The height of the second pole is 4 feet
    This theorem proves that the length of a wire stretched from the top of the taller pole
    to the top of the shorter pole is 10 feet. -/
theorem wire_length (pole1_height pole2_height pole_distance : ℝ) 
  (h1 : pole1_height = 10)
  (h2 : pole2_height = 4)
  (h3 : pole_distance = 8) :
  Real.sqrt ((pole1_height - pole2_height)^2 + pole_distance^2) = 10 := by
  sorry

#check wire_length

end NUMINAMATH_CALUDE_wire_length_l2008_200873


namespace NUMINAMATH_CALUDE_ellipse_equation_l2008_200856

/-- An ellipse with center at the origin, a focus on a coordinate axis,
    eccentricity √3/2, and passing through (2,0) -/
structure Ellipse where
  -- The focus is either on the x-axis or y-axis
  focus_on_axis : Bool
  -- The equation of the ellipse in the form x²/a² + y²/b² = 1
  a : ℝ
  b : ℝ
  -- Conditions
  center_origin : a > 0 ∧ b > 0
  passes_through_2_0 : (2 : ℝ)^2 / a^2 + 0^2 / b^2 = 1
  eccentricity : Real.sqrt (1 - b^2 / a^2) = Real.sqrt 3 / 2

/-- The equation of the ellipse is either x²/4 + y² = 1 or x²/4 + y²/16 = 1 -/
theorem ellipse_equation (e : Ellipse) :
  (e.a^2 = 4 ∧ e.b^2 = 1) ∨ (e.a^2 = 16 ∧ e.b^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2008_200856


namespace NUMINAMATH_CALUDE_subtracted_value_l2008_200812

theorem subtracted_value (chosen_number : ℕ) (final_answer : ℕ) : 
  chosen_number = 848 → final_answer = 6 → 
  ∃ x : ℚ, (chosen_number / 8 : ℚ) - x = final_answer ∧ x = 100 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_l2008_200812


namespace NUMINAMATH_CALUDE_count_perfect_squares_l2008_200802

theorem count_perfect_squares (n : ℕ) : 
  (Finset.filter (fun k => 16 * k * k < 5000) (Finset.range n)).card = 17 :=
sorry

end NUMINAMATH_CALUDE_count_perfect_squares_l2008_200802


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l2008_200846

theorem complex_sum_theorem (p r s u v x y : ℝ) : 
  let q : ℝ := 4
  let sum_real : ℝ := p + r + u + x
  let sum_imag : ℝ := q + s + v + y
  u = -p - r - x →
  sum_real = 0 →
  sum_imag = 7 →
  s + v + y = 3 := by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l2008_200846


namespace NUMINAMATH_CALUDE_elvins_internet_charge_l2008_200842

/-- Proves that the fixed monthly charge for internet service is $6 given the conditions of Elvin's telephone bills. -/
theorem elvins_internet_charge (january_bill february_bill : ℕ) 
  (h1 : january_bill = 48)
  (h2 : february_bill = 90)
  (fixed_charge : ℕ) (january_calls february_calls : ℕ)
  (h3 : february_calls = 2 * january_calls)
  (h4 : january_bill = fixed_charge + january_calls)
  (h5 : february_bill = fixed_charge + february_calls) :
  fixed_charge = 6 := by
  sorry

end NUMINAMATH_CALUDE_elvins_internet_charge_l2008_200842


namespace NUMINAMATH_CALUDE_min_toothpicks_theorem_l2008_200857

/-- Represents a triangular grid made of toothpicks -/
structure ToothpickGrid where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (grid : ToothpickGrid) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_theorem (grid : ToothpickGrid) 
  (h1 : grid.total_toothpicks = 40)
  (h2 : grid.upward_triangles = 10)
  (h3 : grid.downward_triangles = 15) : 
  min_toothpicks_to_remove grid = 10 := by sorry

end NUMINAMATH_CALUDE_min_toothpicks_theorem_l2008_200857


namespace NUMINAMATH_CALUDE_f_max_value_l2008_200837

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt (2 * x^3 + 7 * x^2 + 6 * x)) / (x^2 + 4 * x + 3)

theorem f_max_value :
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f x ≤ 1/2) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f x = 1/2) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l2008_200837


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l2008_200893

/-- 
Given an arithmetic sequence where:
- a₁ is the first term
- a₂₀ is the 20th term
- a₃₀ is the 30th term
This theorem states that if a₁ = 3 and a₂₀ = 41, then a₃₀ = 61.
-/
theorem arithmetic_sequence_30th_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1))
  (h_first : a 1 = 3)
  (h_twentieth : a 20 = 41) : 
  a 30 = 61 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l2008_200893


namespace NUMINAMATH_CALUDE_first_quartile_of_list_l2008_200828

def number_list : List ℝ := [42, 24, 30, 22, 26, 27, 33, 35]

def median (l : List ℝ) : ℝ := sorry

def first_quartile (l : List ℝ) : ℝ :=
  let m := median l
  median (l.filter (λ x => x < m))

theorem first_quartile_of_list :
  first_quartile number_list = 25 := by sorry

end NUMINAMATH_CALUDE_first_quartile_of_list_l2008_200828


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l2008_200801

theorem square_root_fraction_equality : 
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = 17 / Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l2008_200801


namespace NUMINAMATH_CALUDE_largest_B_term_l2008_200898

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The term B_k in the binomial expansion -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.3 ^ k)

/-- The theorem stating that B_k is largest when k = 125 -/
theorem largest_B_term : ∀ k : ℕ, k ≤ 500 → B k ≤ B 125 := by sorry

end NUMINAMATH_CALUDE_largest_B_term_l2008_200898


namespace NUMINAMATH_CALUDE_product_digit_count_l2008_200813

def x : ℕ := 3659893456789325678
def y : ℕ := 342973489379256

theorem product_digit_count :
  (String.length (toString (x * y))) = 34 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_count_l2008_200813


namespace NUMINAMATH_CALUDE_training_hours_per_day_l2008_200877

/-- 
Given a person who trains for a constant number of hours per day over a period of time,
this theorem proves that if the total training period is 42 days and the total training time
is 210 hours, then the person trains for 5 hours every day.
-/
theorem training_hours_per_day 
  (total_days : ℕ) 
  (total_hours : ℕ) 
  (hours_per_day : ℕ) 
  (h1 : total_days = 42) 
  (h2 : total_hours = 210) 
  (h3 : total_hours = total_days * hours_per_day) : 
  hours_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_training_hours_per_day_l2008_200877


namespace NUMINAMATH_CALUDE_triangle_angle_range_l2008_200809

theorem triangle_angle_range (A B C : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle conditions
  2 * Real.sin A + Real.sin B = Real.sqrt 3 * Real.sin C →  -- Given equation
  π / 6 ≤ A ∧ A ≤ π / 2 := by  -- Conclusion to prove
sorry

end NUMINAMATH_CALUDE_triangle_angle_range_l2008_200809


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_fourth_power_l2008_200834

theorem nearest_integer_to_3_plus_sqrt5_fourth_power :
  ∃ n : ℤ, n = 752 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_fourth_power_l2008_200834


namespace NUMINAMATH_CALUDE_dispatch_riders_travel_time_l2008_200854

/-- Represents the travel scenario of two dispatch riders -/
structure DispatchRiders where
  a : ℝ  -- Speed increase of the first rider in km/h
  x : ℝ  -- Initial speed of the first rider in km/h
  y : ℝ  -- Speed of the second rider in km/h
  z : ℝ  -- Actual travel time of the first rider in hours

/-- The conditions of the dispatch riders' travel -/
def travel_conditions (d : DispatchRiders) : Prop :=
  d.a > 0 ∧ d.a < 30 ∧
  d.x > 0 ∧ d.y > 0 ∧ d.z > 0 ∧
  180 / d.x - 180 / d.y = 6 ∧
  d.z * (d.x + d.a) = 180 ∧
  (d.z - 3) * d.y = 180

/-- The theorem stating the travel times of both riders -/
theorem dispatch_riders_travel_time (d : DispatchRiders) 
  (h : travel_conditions d) : 
  d.z = (-3 * d.a + 3 * Real.sqrt (d.a^2 + 240 * d.a)) / (2 * d.a) ∧
  d.z - 3 = (-9 * d.a + 3 * Real.sqrt (d.a^2 + 240 * d.a)) / (2 * d.a) := by
  sorry

end NUMINAMATH_CALUDE_dispatch_riders_travel_time_l2008_200854


namespace NUMINAMATH_CALUDE_paula_karl_age_sum_l2008_200841

theorem paula_karl_age_sum : ∀ (P K : ℕ),
  (P - 5 = 3 * (K - 5)) →
  (P + 6 = 2 * (K + 6)) →
  P + K = 54 := by
  sorry

end NUMINAMATH_CALUDE_paula_karl_age_sum_l2008_200841


namespace NUMINAMATH_CALUDE_expand_expression_l2008_200881

theorem expand_expression (x : ℝ) : (15 * x + 17 + 3) * (3 * x) = 45 * x^2 + 60 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2008_200881


namespace NUMINAMATH_CALUDE_solve_for_y_l2008_200818

theorem solve_for_y (x y : ℝ) (h1 : x + 2 * y = 20) (h2 : x = 10) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2008_200818


namespace NUMINAMATH_CALUDE_remaining_potatoes_l2008_200819

/-- Given an initial number of potatoes and a number of eaten potatoes,
    prove that the remaining number of potatoes is equal to their difference. -/
theorem remaining_potatoes (initial : ℕ) (eaten : ℕ) :
  initial ≥ eaten → initial - eaten = initial - eaten :=
by sorry

end NUMINAMATH_CALUDE_remaining_potatoes_l2008_200819


namespace NUMINAMATH_CALUDE_factorization_sum_l2008_200879

theorem factorization_sum (a b : ℤ) : 
  (∀ x : ℝ, 24 * x^2 - 50 * x - 84 = (6 * x + a) * (4 * x + b)) → 
  a + 2 * b = -17 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l2008_200879


namespace NUMINAMATH_CALUDE_coin_count_l2008_200894

theorem coin_count (total_value : ℕ) (nickel_value dime_value quarter_value : ℕ) :
  total_value = 360 →
  nickel_value = 5 →
  dime_value = 10 →
  quarter_value = 25 →
  ∃ (x : ℕ), x * (nickel_value + dime_value + quarter_value) = total_value ∧
              3 * x = 27 :=
by
  sorry

#check coin_count

end NUMINAMATH_CALUDE_coin_count_l2008_200894


namespace NUMINAMATH_CALUDE_intersection_probability_l2008_200899

/-- A regular decagon is a 10-sided polygon with all sides equal and all angles equal. -/
def RegularDecagon : Type := Unit

/-- The number of vertices in a regular decagon. -/
def num_vertices : ℕ := 10

/-- The number of diagonals in a regular decagon. -/
def num_diagonals (d : RegularDecagon) : ℕ := 35

/-- The number of ways to choose 2 diagonals from a regular decagon. -/
def num_diagonal_pairs (d : RegularDecagon) : ℕ := 595

/-- The number of sets of 4 points that determine intersecting diagonals. -/
def num_intersecting_sets (d : RegularDecagon) : ℕ := 210

/-- The probability that two randomly chosen diagonals of a regular decagon intersect inside the decagon. -/
theorem intersection_probability (d : RegularDecagon) : 
  (num_intersecting_sets d : ℚ) / (num_diagonal_pairs d) = 210 / 595 := by sorry

end NUMINAMATH_CALUDE_intersection_probability_l2008_200899


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2008_200810

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  isArithmeticSequence a →
  a 1 + a 2017 = 10 →
  a 2 + a 1009 + a 2016 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2008_200810


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2008_200820

theorem floor_equation_solution :
  {x : ℚ | ⌊(8*x + 19)/7⌋ = (16*(x + 1))/11} =
  {1 + 1/16, 1 + 3/4, 2 + 7/16, 3 + 1/8, 3 + 13/16} := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2008_200820


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2008_200800

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The common ratio of a geometric sequence can be found by dividing
    the second term by the first term -/
def common_ratio (a₁ a₂ : ℚ) : ℚ := a₂ / a₁

theorem seventh_term_of_geometric_sequence (a₁ a₂ : ℚ) 
  (h₁ : a₁ = 3)
  (h₂ : a₂ = -3/2) :
  geometric_sequence a₁ (common_ratio a₁ a₂) 7 = 3/64 := by
  sorry


end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2008_200800


namespace NUMINAMATH_CALUDE_money_division_theorem_l2008_200853

/-- Represents the shares of P, Q, and R in the money division problem -/
structure Shares where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The problem of dividing money between P, Q, and R -/
def MoneyDivisionProblem (s : Shares) : Prop :=
  ∃ (x : ℝ),
    s.p = 5 * x ∧
    s.q = 11 * x ∧
    s.r = 19 * x ∧
    s.q - s.p = 12100

theorem money_division_theorem (s : Shares) 
  (h : MoneyDivisionProblem s) : s.r - s.q = 16133.36 := by
  sorry


end NUMINAMATH_CALUDE_money_division_theorem_l2008_200853


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l2008_200839

theorem inequality_solution_implies_m_range 
  (m : ℝ) 
  (h : ∀ x : ℝ, (m * x - 1) * (x - 2) > 0 ↔ (1 / m < x ∧ x < 2)) : 
  m < 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_range_l2008_200839


namespace NUMINAMATH_CALUDE_max_pieces_theorem_l2008_200843

/-- Represents the size of the cake in inches -/
def cake_size : ℕ := 100

/-- Represents the size of each piece in inches -/
def piece_size : ℕ := 4

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Theorem stating the maximum number of pieces that can be cut from the cake -/
theorem max_pieces_theorem :
  (is_even cake_size) →
  (is_even piece_size) →
  (cake_size % piece_size = 0) →
  (cake_size / piece_size) * (cake_size / piece_size) = 625 := by
  sorry

#check max_pieces_theorem

end NUMINAMATH_CALUDE_max_pieces_theorem_l2008_200843


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l2008_200823

/-- Represents a 3x3 matrix of integers -/
def Matrix3x3 := Fin 3 → Fin 3 → ℤ

/-- Checks if two integers are coprime -/
def are_coprime (a b : ℤ) : Prop := Nat.gcd a.natAbs b.natAbs = 1

/-- Checks if the matrix satisfies the adjacency condition -/
def satisfies_adjacency_condition (m : Matrix3x3) : Prop :=
  ∀ i j i' j', (i = i' ∧ j.succ = j') ∨ (i = i' ∧ j = j'.succ) ∨
                (i.succ = i' ∧ j = j') ∨ (i = i'.succ ∧ j = j') ∨
                (i.succ = i' ∧ j.succ = j') ∨ (i.succ = i' ∧ j = j'.succ) ∨
                (i = i'.succ ∧ j.succ = j') ∨ (i = i'.succ ∧ j = j'.succ) →
                are_coprime (m i j) (m i' j')

/-- Checks if the matrix contains nine consecutive integers -/
def contains_consecutive_integers (m : Matrix3x3) : Prop :=
  ∃ start : ℤ, ∀ i j, ∃ k : ℕ, k < 9 ∧ m i j = start + k

/-- The main theorem stating the existence of a valid arrangement -/
theorem valid_arrangement_exists : ∃ m : Matrix3x3, 
  satisfies_adjacency_condition m ∧ contains_consecutive_integers m := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l2008_200823


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2008_200824

theorem cos_alpha_value (α : Real) : 
  (∃ (x y : Real), x = 2 * Real.sin (π / 6) ∧ y = -2 * Real.cos (π / 6) ∧ 
   x = 2 * Real.sin α ∧ y = -2 * Real.cos α) → 
  Real.cos α = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2008_200824


namespace NUMINAMATH_CALUDE_ellipse_equation_l2008_200850

/-- Definition of an ellipse with given focal distance and major axis length -/
structure Ellipse :=
  (focal_distance : ℝ)
  (major_axis_length : ℝ)

/-- Standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), 
    a = e.major_axis_length / 2 ∧
    b^2 = a^2 - (e.focal_distance / 2)^2 ∧
    x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the given ellipse -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.focal_distance = 8)
  (h2 : e.major_axis_length = 10) :
  standard_equation e x y ↔ x^2 / 25 + y^2 / 9 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2008_200850


namespace NUMINAMATH_CALUDE_polynomial_factors_imply_relation_l2008_200803

theorem polynomial_factors_imply_relation (h k : ℝ) : 
  (∃ a : ℝ, 2 * x^3 - h * x + k = (x + 2) * (x - 1) * a) → 
  2 * h - 3 * k = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factors_imply_relation_l2008_200803


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2008_200885

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 5 * Real.cos (2 * α) = Real.sqrt 2 * Real.sin (π / 4 - α)) : 
  Real.tan α = -4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2008_200885


namespace NUMINAMATH_CALUDE_min_value_inequality_l2008_200847

theorem min_value_inequality (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2008_200847


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l2008_200835

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (boat_speed stream_speed : ℝ) : ℝ := boat_speed - stream_speed

theorem boat_upstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : boat_speed = 8)
  (h2 : boat_speed + stream_speed = 11) :
  boat_distance boat_speed stream_speed = 5 := by
  sorry

#check boat_upstream_distance

end NUMINAMATH_CALUDE_boat_upstream_distance_l2008_200835


namespace NUMINAMATH_CALUDE_line_through_P_and_origin_equation_line_l_equation_l2008_200865

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Define the line passing through P and the origin
def line_through_P_and_origin (x y : ℝ) : Prop := x + y = 0

-- Define the line l passing through P and perpendicular to l₃
def line_l (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Theorem 1: The line passing through P and the origin has the equation x + y = 0
theorem line_through_P_and_origin_equation :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y → line_through_P_and_origin x y :=
by sorry

-- Theorem 2: The line l passing through P and perpendicular to l₃ has the equation x - 2y + 6 = 0
theorem line_l_equation :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y ∧ l₃ x y → line_l x y :=
by sorry

end NUMINAMATH_CALUDE_line_through_P_and_origin_equation_line_l_equation_l2008_200865


namespace NUMINAMATH_CALUDE_multiply_658217_by_99999_l2008_200886

theorem multiply_658217_by_99999 : 658217 * 99999 = 65821034183 := by
  sorry

end NUMINAMATH_CALUDE_multiply_658217_by_99999_l2008_200886


namespace NUMINAMATH_CALUDE_nine_to_ten_div_eightyone_to_four_equals_eightyone_l2008_200851

theorem nine_to_ten_div_eightyone_to_four_equals_eightyone :
  9^10 / 81^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_nine_to_ten_div_eightyone_to_four_equals_eightyone_l2008_200851


namespace NUMINAMATH_CALUDE_curve_is_part_of_ellipse_l2008_200874

-- Define the curve
def curve (x y : ℝ) : Prop := x = Real.sqrt (1 - 4 * y^2)

-- Define an ellipse
def is_ellipse (x y : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem curve_is_part_of_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), curve x y → x ≥ 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_part_of_ellipse_l2008_200874


namespace NUMINAMATH_CALUDE_max_height_triangle_DEF_l2008_200863

/-- Triangle DEF with side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  FD : ℝ

/-- The maximum possible height of a table constructed from a triangle -/
def max_table_height (t : Triangle) : ℝ := sorry

/-- The given triangle DEF -/
def triangle_DEF : Triangle :=
  { DE := 25,
    EF := 28,
    FD := 33 }

theorem max_height_triangle_DEF :
  max_table_height triangle_DEF = 60 * Real.sqrt 129 / 61 := by sorry

end NUMINAMATH_CALUDE_max_height_triangle_DEF_l2008_200863


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2008_200884

theorem smallest_integer_with_remainders : ∃ n : ℕ, n > 0 ∧
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 6 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2008_200884


namespace NUMINAMATH_CALUDE_ricky_age_solution_l2008_200826

def ricky_age_problem (rickys_age : ℕ) (fathers_age : ℕ) : Prop :=
  fathers_age = 45 ∧
  rickys_age + 5 = (1 / 5 : ℚ) * (fathers_age + 5 : ℚ) + 5

theorem ricky_age_solution :
  ∃ (rickys_age : ℕ), ricky_age_problem rickys_age 45 ∧ rickys_age = 10 :=
sorry

end NUMINAMATH_CALUDE_ricky_age_solution_l2008_200826


namespace NUMINAMATH_CALUDE_quadrilateral_is_trapezoid_or_parallelogram_l2008_200896

/-- A quadrilateral with angles A, B, C, and D, where the products of cosines of opposite angles are equal. -/
structure Quadrilateral where
  A : Real
  B : Real
  C : Real
  D : Real
  angle_sum : A + B + C + D = 2 * Real.pi
  cosine_product : Real.cos A * Real.cos C = Real.cos B * Real.cos D

/-- A quadrilateral is either a trapezoid or a parallelogram if the products of cosines of opposite angles are equal. -/
theorem quadrilateral_is_trapezoid_or_parallelogram (q : Quadrilateral) :
  (∃ (x y : Real), x + y = Real.pi ∧ (q.A = x ∧ q.C = x) ∨ (q.B = y ∧ q.D = y)) ∨
  (q.A = q.C ∧ q.B = q.D) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_is_trapezoid_or_parallelogram_l2008_200896


namespace NUMINAMATH_CALUDE_log_3_81_sqrt_81_equals_6_l2008_200867

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_3_81_sqrt_81_equals_6 :
  log 3 (81 * Real.sqrt 81) = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_3_81_sqrt_81_equals_6_l2008_200867


namespace NUMINAMATH_CALUDE_final_water_fraction_l2008_200849

def container_size : ℚ := 25

def initial_water : ℚ := 25

def replacement_volume : ℚ := 5

def third_replacement_water : ℚ := 2

def calculate_final_water_fraction (initial_water : ℚ) (container_size : ℚ) 
  (replacement_volume : ℚ) (third_replacement_water : ℚ) : ℚ :=
  sorry

theorem final_water_fraction :
  calculate_final_water_fraction initial_water container_size replacement_volume third_replacement_water
  = 14.8 / 25 :=
sorry

end NUMINAMATH_CALUDE_final_water_fraction_l2008_200849


namespace NUMINAMATH_CALUDE_sum_of_valid_a_l2008_200852

theorem sum_of_valid_a : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, (∃! (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 
    5 * x₁ ≥ 3 * (x₁ + 2) ∧ x₁ - (x₁ + 3) / 2 ≤ a / 16 ∧
    5 * x₂ ≥ 3 * (x₂ + 2) ∧ x₂ - (x₂ + 3) / 2 ≤ a / 16) ∧
   (∃ y : ℤ, y < 0 ∧ 5 + a * y = 2 * y - 7)) ∧
  (S.sum id = 22) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_l2008_200852


namespace NUMINAMATH_CALUDE_rectangle_lcm_gcd_product_l2008_200866

theorem rectangle_lcm_gcd_product : 
  let a : ℕ := 24
  let b : ℕ := 36
  Nat.lcm a b * Nat.gcd a b = 864 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_lcm_gcd_product_l2008_200866


namespace NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l2008_200864

/-- The equation of a circle with center (0, b) that is tangent to the line y = 2x + 1 at point (1, 3) -/
theorem circle_equation_tangent_to_line (b : ℝ) :
  (∀ x y : ℝ, y = 2 * x + 1 → (x - 1)^2 + (y - 3)^2 ≠ 0) →
  (1 : ℝ)^2 + (3 - b)^2 = (0 - 1)^2 + ((2 * 0 + 1) - b)^2 →
  (∀ x y : ℝ, (x : ℝ)^2 + (y - 7/2)^2 = 5/4 ↔ (x - 0)^2 + (y - b)^2 = (1 - 0)^2 + (3 - b)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l2008_200864


namespace NUMINAMATH_CALUDE_dan_remaining_money_l2008_200895

/-- Given an initial amount and a spent amount, calculate the remaining amount --/
def remaining_amount (initial : ℚ) (spent : ℚ) : ℚ :=
  initial - spent

/-- Proof that Dan has $1 left --/
theorem dan_remaining_money :
  let initial_amount : ℚ := 4
  let spent_amount : ℚ := 3
  remaining_amount initial_amount spent_amount = 1 := by
  sorry

end NUMINAMATH_CALUDE_dan_remaining_money_l2008_200895


namespace NUMINAMATH_CALUDE_correct_dimes_calculation_l2008_200848

/-- Represents the number of dimes each sibling has -/
structure Dimes where
  barry : ℕ
  dan : ℕ
  emily : ℕ
  frank : ℕ

/-- Calculates the correct number of dimes for each sibling based on the given conditions -/
def calculate_dimes : Dimes :=
  let barry_dimes := 1000 / 10  -- $10.00 worth of dimes
  let dan_initial := barry_dimes / 2
  let dan_final := dan_initial + 2
  let emily_dimes := 2 * dan_initial
  let frank_dimes := emily_dimes - 7
  { barry := barry_dimes
  , dan := dan_final
  , emily := emily_dimes
  , frank := frank_dimes }

/-- Theorem stating that the calculated dimes match the expected values -/
theorem correct_dimes_calculation : 
  let dimes := calculate_dimes
  dimes.barry = 100 ∧ 
  dimes.dan = 52 ∧ 
  dimes.emily = 100 ∧ 
  dimes.frank = 93 := by
  sorry

end NUMINAMATH_CALUDE_correct_dimes_calculation_l2008_200848


namespace NUMINAMATH_CALUDE_sandra_son_age_l2008_200860

/-- Sandra's current age -/
def sandra_age : ℕ := 36

/-- The ratio of Sandra's age to her son's age 3 years ago -/
def age_ratio : ℕ := 3

/-- Sandra's son's current age -/
def son_age : ℕ := 14

theorem sandra_son_age : 
  sandra_age - 3 = age_ratio * (son_age - 3) :=
sorry

end NUMINAMATH_CALUDE_sandra_son_age_l2008_200860


namespace NUMINAMATH_CALUDE_waiter_earnings_l2008_200883

theorem waiter_earnings (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : 
  total_customers = 7 → 
  non_tipping_customers = 5 → 
  tip_amount = 3 → 
  (total_customers - non_tipping_customers) * tip_amount = 6 := by
sorry

end NUMINAMATH_CALUDE_waiter_earnings_l2008_200883


namespace NUMINAMATH_CALUDE_excellent_students_problem_l2008_200804

theorem excellent_students_problem (B₁ B₂ B₃ : Finset ℕ) 
  (h_total : (B₁ ∪ B₂ ∪ B₃).card = 100)
  (h_math : B₁.card = 70)
  (h_phys : B₂.card = 65)
  (h_chem : B₃.card = 75)
  (h_math_phys : (B₁ ∩ B₂).card = 40)
  (h_math_chem : (B₁ ∩ B₃).card = 45)
  (h_all : (B₁ ∩ B₂ ∩ B₃).card = 25) :
  ((B₂ ∩ B₃) \ B₁).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_excellent_students_problem_l2008_200804


namespace NUMINAMATH_CALUDE_expression_evaluation_l2008_200836

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -2
  ((2 * x + y) * (2 * x - y) - (2 * x - 3 * y)^2) / (-2 * y) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2008_200836


namespace NUMINAMATH_CALUDE_value_of_a_l2008_200814

theorem value_of_a (a b c : ℤ) (h1 : a + b = 10) (h2 : b + c = 8) (h3 : c = 4) : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2008_200814


namespace NUMINAMATH_CALUDE_trajectory_and_range_l2008_200845

-- Define the circle D
def circle_D (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 32

-- Define point P
def P : ℝ × ℝ := (-6, 3)

-- Define the trajectory of M
def trajectory_M (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 8

-- Define the range of t
def t_range (t : ℝ) : Prop :=
  t ∈ Set.Icc (-Real.sqrt 5 - 1) (-Real.sqrt 5 + 1) ∪
      Set.Icc (Real.sqrt 5 - 1) (Real.sqrt 5 + 1)

theorem trajectory_and_range :
  (∀ x y : ℝ, ∃ x_H y_H : ℝ,
    circle_D x_H y_H ∧
    x = (x_H + P.1) / 2 ∧
    y = (y_H + P.2) / 2 →
    trajectory_M x y) ∧
  (∀ k t : ℝ,
    (∃ x_B y_B x_C y_C : ℝ,
      trajectory_M x_B y_B ∧
      trajectory_M x_C y_C ∧
      y_B = k * x_B ∧
      y_C = k * x_C ∧
      (x_B - 0) * (x_C - 0) + (y_B - t) * (y_C - t) = 0) →
    t_range t) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_range_l2008_200845


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2008_200808

/-- A geometric sequence is a sequence where the ratio of any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that for a geometric sequence satisfying certain conditions, 
    the sum of specific terms equals 3. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h1 : a 1 + a 3 = 8) 
    (h2 : a 5 + a 7 = 4) : 
  a 9 + a 11 + a 13 + a 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2008_200808


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_l2008_200832

/-- An unfair coin with given probabilities and payoffs -/
structure UnfairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  payoff_heads : ℝ
  payoff_tails : ℝ
  prob_sum : prob_heads + prob_tails = 1

/-- The expected worth of a coin flip -/
def expected_worth (c : UnfairCoin) : ℝ :=
  c.prob_heads * c.payoff_heads + c.prob_tails * c.payoff_tails

/-- Theorem stating the expected worth of the specific unfair coin -/
theorem expected_worth_unfair_coin :
  ∃ c : UnfairCoin, c.prob_heads = 3/4 ∧ c.prob_tails = 1/4 ∧
  c.payoff_heads = 3 ∧ c.payoff_tails = -8 ∧ expected_worth c = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_l2008_200832


namespace NUMINAMATH_CALUDE_negation_of_implication_l2008_200821

theorem negation_of_implication (x : ℝ) : 
  (¬(x^2 = 1 → x = 1)) ↔ (x^2 ≠ 1 → x ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2008_200821


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l2008_200855

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) ∧
  (a.val = 18 ∧ b.val = 1) := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l2008_200855


namespace NUMINAMATH_CALUDE_james_payment_is_correct_l2008_200822

def james_total_payment (steak_price dessert_price drink_price : ℚ)
  (steak_discount : ℚ) (friend_steak_price friend_dessert_price friend_drink_price : ℚ)
  (friend_steak_discount : ℚ) (meal_tax_rate drink_tax_rate : ℚ)
  (james_tip_rate : ℚ) : ℚ :=
  let james_meal := steak_price * (1 - steak_discount)
  let friend_meal := friend_steak_price * (1 - friend_steak_discount)
  let james_total := james_meal + dessert_price + drink_price
  let friend_total := friend_meal + friend_dessert_price + friend_drink_price
  let james_tax := james_meal * meal_tax_rate + dessert_price * meal_tax_rate + drink_price * drink_tax_rate
  let friend_tax := friend_meal * meal_tax_rate + friend_dessert_price * meal_tax_rate + friend_drink_price * drink_tax_rate
  let total_bill := james_total + friend_total + james_tax + friend_tax
  let james_share := total_bill / 2
  let james_tip := james_share * james_tip_rate
  james_share + james_tip

theorem james_payment_is_correct :
  james_total_payment 16 5 3 0.1 14 4 2 0.05 0.08 0.05 0.2 = 265/10 := by sorry

end NUMINAMATH_CALUDE_james_payment_is_correct_l2008_200822


namespace NUMINAMATH_CALUDE_circle_equation_radius_l2008_200878

theorem circle_equation_radius (k : ℝ) :
  (∃ (h : ℝ) (v : ℝ),
    ∀ (x y : ℝ),
      x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x - h)^2 + (y - v)^2 = 10^2) ↔
  k = 35 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l2008_200878
