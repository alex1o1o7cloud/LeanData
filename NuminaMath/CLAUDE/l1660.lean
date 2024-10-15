import Mathlib

namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1660_166019

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B : Set ℝ := {x | x^3 = x}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1660_166019


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_sufficient_not_necessary_l1660_166052

/-- A hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 --/
structure Hyperbola (a b : ℝ) : Type :=
  (hap : a > 0)
  (hbp : b > 0)

/-- The asymptotes of a hyperbola --/
def asymptotes (h : Hyperbola a b) (x : ℝ) : Set ℝ :=
  {y | y = (b / a) * x ∨ y = -(b / a) * x}

/-- The theorem stating that the hyperbola equation is a sufficient but not necessary condition for its asymptotes --/
theorem hyperbola_asymptotes_sufficient_not_necessary (a b : ℝ) :
  (∃ (h : Hyperbola a b), ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → y ∈ asymptotes h x) ∧
  (∃ a' b' : ℝ, ∃ (h : Hyperbola a' b'), ∀ x y : ℝ, y ∈ asymptotes h x ∧ (x^2 / a'^2) - (y^2 / b'^2) ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_sufficient_not_necessary_l1660_166052


namespace NUMINAMATH_CALUDE_sign_language_size_l1660_166016

theorem sign_language_size :
  ∀ n : ℕ,
  (n ≥ 2) →
  (n^2 - (n-2)^2 = 888) →
  n = 223 := by
sorry

end NUMINAMATH_CALUDE_sign_language_size_l1660_166016


namespace NUMINAMATH_CALUDE_expression_simplification_l1660_166057

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 5) * 4 + (5 - 2 / 4) * (8 * p - 12) = 4 * p - 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1660_166057


namespace NUMINAMATH_CALUDE_f_inequality_l1660_166026

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition for f
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) > 0

-- State the theorem
theorem f_inequality (h : strictly_increasing f) : f (-2) < f 1 ∧ f 1 < f 3 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1660_166026


namespace NUMINAMATH_CALUDE_more_stable_performance_l1660_166018

/-- Given two students A and B with their respective variances, 
    proves that the student with lower variance has more stable performance -/
theorem more_stable_performance (S_A_squared S_B_squared : ℝ) 
  (h1 : S_A_squared = 0.3)
  (h2 : S_B_squared = 0.1) : 
  S_B_squared < S_A_squared := by sorry


end NUMINAMATH_CALUDE_more_stable_performance_l1660_166018


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1660_166098

theorem other_root_of_quadratic (p : ℝ) : 
  (∃ x : ℝ, 7 * x^2 + p * x = 9) ∧ 
  (7 * (-3)^2 + p * (-3) = 9) → 
  7 * (3/7)^2 + p * (3/7) = 9 :=
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1660_166098


namespace NUMINAMATH_CALUDE_spade_calculation_l1660_166021

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 3 (spade 4 5) = -72 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l1660_166021


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1660_166095

theorem arithmetic_expression_evaluation : 8 / 2 - (3 - 5 + 7) + 3 * 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1660_166095


namespace NUMINAMATH_CALUDE_smallest_positive_angle_exists_l1660_166046

theorem smallest_positive_angle_exists : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 360 ∧ 
  (∀ φ : ℝ, φ > 0 ∧ φ < 360 ∧ 
    Real.cos (φ * Real.pi / 180) = 
      Real.sin (45 * Real.pi / 180) + 
      Real.cos (30 * Real.pi / 180) - 
      Real.sin (18 * Real.pi / 180) - 
      Real.cos (12 * Real.pi / 180) → 
    θ ≤ φ) ∧
  Real.cos (θ * Real.pi / 180) = 
    Real.sin (45 * Real.pi / 180) + 
    Real.cos (30 * Real.pi / 180) - 
    Real.sin (18 * Real.pi / 180) - 
    Real.cos (12 * Real.pi / 180) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_exists_l1660_166046


namespace NUMINAMATH_CALUDE_root_square_transformation_l1660_166044

/-- The original polynomial f(x) -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

/-- The resulting polynomial g(x) -/
def g (x : ℝ) : ℝ := x^3 - 4*x^2 - 7*x + 16

theorem root_square_transformation (r : ℝ) : 
  f r = 0 → ∃ s, g s = 0 ∧ s = r^2 := by sorry

end NUMINAMATH_CALUDE_root_square_transformation_l1660_166044


namespace NUMINAMATH_CALUDE_periodic_function_value_l1660_166014

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_value :
  ∀ (f : ℝ → ℝ),
  is_periodic f 4 →
  (∀ x ∈ Set.Icc 0 4, f x = x) →
  f 7.6 = 3.6 := by
sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1660_166014


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1660_166091

theorem sum_of_coefficients (a b c : ℤ) : 
  (∀ x, x^2 + 19*x + 88 = (x + a) * (x + b)) →
  (∀ x, x^2 - 21*x + 108 = (x - b) * (x - c)) →
  a + b + c = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1660_166091


namespace NUMINAMATH_CALUDE_max_a_value_l1660_166005

theorem max_a_value (a : ℝ) : (∀ x : ℝ, x * a ≤ Real.exp (x - 1) + x^2 + 1) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l1660_166005


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l1660_166053

theorem quadratic_form_k_value :
  ∀ (a h k : ℝ),
  (∀ x, x^2 - 7*x + 1 = a*(x - h)^2 + k) →
  k = -45/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l1660_166053


namespace NUMINAMATH_CALUDE_inequality_proof_l1660_166093

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1660_166093


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1660_166032

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3/4 < 0) ↔ -3 < k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1660_166032


namespace NUMINAMATH_CALUDE_representation_of_2008_l1660_166088

theorem representation_of_2008 : ∃ (a b c : ℕ), 
  2008 = a + 40 * b + 40 * c ∧ 
  (1 : ℚ) / a + (b : ℚ) / 40 + (c : ℚ) / 40 = 1 := by
  sorry

end NUMINAMATH_CALUDE_representation_of_2008_l1660_166088


namespace NUMINAMATH_CALUDE_coffee_preference_expectation_l1660_166080

theorem coffee_preference_expectation (total_sample : ℕ) 
  (coffee_ratio : ℚ) (h1 : coffee_ratio = 3 / 7) (h2 : total_sample = 350) : 
  ℕ := by
  sorry

#check coffee_preference_expectation

end NUMINAMATH_CALUDE_coffee_preference_expectation_l1660_166080


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1660_166067

-- Problem 1
theorem problem_1 (x : ℝ) (h : x = -1) :
  (4 - x) * (2 * x + 1) + 3 * x * (x - 3) = 7 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 1) (hy : y = 1/2) :
  ((x + 2*y)^2 - (3*x + y)*(3*x - y) - 5*y^2) / (-1/2 * x) = 12 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1660_166067


namespace NUMINAMATH_CALUDE_sum_properties_l1660_166086

theorem sum_properties (c d : ℤ) 
  (hc : ∃ m : ℤ, c = 6 * m) 
  (hd : ∃ n : ℤ, d = 9 * n) : 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ ¬(Even (x + y))) ∧ 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ ¬(∃ k : ℤ, x + y = 6 * k)) ∧ 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ ¬(∃ k : ℤ, x + y = 9 * k)) ∧ 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ (∃ k : ℤ, x + y = 9 * k)) :=
by sorry

end NUMINAMATH_CALUDE_sum_properties_l1660_166086


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_binomial_coefficient_extremes_l1660_166082

theorem binomial_coefficient_divisibility (p k : ℕ) (hp : Nat.Prime p) (hk : 0 < k ∧ k < p) :
  ∃ m : ℕ, (Nat.choose p k) = m * p :=
sorry

theorem binomial_coefficient_extremes (p : ℕ) (hp : Nat.Prime p) :
  Nat.choose p 0 = 1 ∧ Nat.choose p p = 1 :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_binomial_coefficient_extremes_l1660_166082


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l1660_166062

/-- Represents the percentage of employees who are men -/
def percentMen : ℝ := 35

/-- Represents the percentage of employees who are women -/
def percentWomen : ℝ := 100 - percentMen

/-- Represents the percentage of all employees who attended the picnic -/
def percentAttended : ℝ := 33

/-- Represents the percentage of women who attended the picnic -/
def percentWomenAttended : ℝ := 40

/-- Represents the percentage of men who attended the picnic -/
def percentMenAttended : ℝ := 20

theorem company_picnic_attendance :
  percentMenAttended * (percentMen / 100) + percentWomenAttended * (percentWomen / 100) = percentAttended / 100 :=
by sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l1660_166062


namespace NUMINAMATH_CALUDE_comparison_of_products_l1660_166094

theorem comparison_of_products (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_products_l1660_166094


namespace NUMINAMATH_CALUDE_hundredth_odd_followed_by_hundredth_even_l1660_166038

/-- The nth odd positive integer -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

/-- The nth even positive integer -/
def nth_even (n : ℕ) : ℕ := 2 * n

theorem hundredth_odd_followed_by_hundredth_even :
  nth_odd 100 = 199 ∧ nth_even 100 = nth_odd 100 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_odd_followed_by_hundredth_even_l1660_166038


namespace NUMINAMATH_CALUDE_survey_total_is_260_l1660_166023

/-- Represents the survey results of households using different brands of soap -/
structure SoapSurvey where
  neither : Nat
  onlyA : Nat
  onlyB : Nat
  both : Nat

/-- Calculates the total number of households surveyed -/
def totalHouseholds (survey : SoapSurvey) : Nat :=
  survey.neither + survey.onlyA + survey.onlyB + survey.both

/-- Theorem stating the total number of households surveyed is 260 -/
theorem survey_total_is_260 : ∃ (survey : SoapSurvey),
  survey.neither = 80 ∧
  survey.onlyA = 60 ∧
  survey.onlyB = 3 * survey.both ∧
  survey.both = 30 ∧
  totalHouseholds survey = 260 := by
  sorry

end NUMINAMATH_CALUDE_survey_total_is_260_l1660_166023


namespace NUMINAMATH_CALUDE_orange_selling_loss_l1660_166092

def total_money : ℚ := 75
def ratio_sum : ℕ := 4 + 5 + 6
def cara_ratio : ℕ := 4
def janet_ratio : ℕ := 5
def selling_percentage : ℚ := 80 / 100

theorem orange_selling_loss :
  let cara_money := (cara_ratio : ℚ) / ratio_sum * total_money
  let janet_money := (janet_ratio : ℚ) / ratio_sum * total_money
  let combined_money := cara_money + janet_money
  let selling_price := selling_percentage * combined_money
  combined_money - selling_price = 9 := by sorry

end NUMINAMATH_CALUDE_orange_selling_loss_l1660_166092


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1660_166096

/-- Theorem: Eccentricity of a hyperbola with specific properties --/
theorem hyperbola_eccentricity (a b c : ℝ) (h : c^2 = a^2 + b^2) : 
  let f1 : ℝ × ℝ := (-c, 0)
  let f2 : ℝ × ℝ := (c, 0)
  let A : ℝ × ℝ := (c, b^2 / a)
  let B : ℝ × ℝ := (c, -b^2 / a)
  let G : ℝ × ℝ := (c / 3, 0)
  ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 →
    (G.1 - A.1) * (f1.1 - B.1) + (G.2 - A.2) * (f1.2 - B.2) = 0 →
    c / a = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1660_166096


namespace NUMINAMATH_CALUDE_parabola_theorem_l1660_166074

/-- Parabola C defined by x²=4y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Point P on parabola C -/
def point_P : ℝ × ℝ := (2, 1)

/-- Focus F of parabola C -/
def focus_F : ℝ × ℝ := (0, 1)

/-- Point H where the axis of the parabola intersects the y-axis -/
def point_H : ℝ × ℝ := (0, -1)

/-- Line l passing through focus F and intersecting parabola C at points A and B -/
def line_l (x y : ℝ) : Prop := ∃ (k b : ℝ), y = k*x + b ∧ (0 = k*0 + b - 1)

/-- Points A and B on parabola C -/
def points_AB (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

/-- AB is perpendicular to HB -/
def AB_perp_HB (A B : ℝ × ℝ) : Prop :=
  (A.2 - B.2) * (B.1 - point_H.1) = -(A.1 - B.1) * (B.2 - point_H.2)

/-- Main theorem: |AF| - |BF| = 4 -/
theorem parabola_theorem (A B : ℝ × ℝ) :
  points_AB A B → AB_perp_HB A B →
  Real.sqrt ((A.1 - focus_F.1)^2 + (A.2 - focus_F.2)^2) -
  Real.sqrt ((B.1 - focus_F.1)^2 + (B.2 - focus_F.2)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_theorem_l1660_166074


namespace NUMINAMATH_CALUDE_book_club_single_people_count_l1660_166076

/-- Represents a book club with members and book selection turns. -/
structure BookClub where
  total_turns : ℕ  -- Total number of turns per year
  couple_count : ℕ  -- Number of couples in the club
  ron_turns : ℕ  -- Number of turns Ron gets per year

/-- Calculates the number of single people in the book club. -/
def single_people_count (club : BookClub) : ℕ :=
  club.total_turns - (club.couple_count + 1)

/-- Theorem stating that the number of single people in the given book club is 9. -/
theorem book_club_single_people_count :
  ∃ (club : BookClub),
    club.total_turns = 52 / 4 ∧
    club.couple_count = 3 ∧
    club.ron_turns = 4 ∧
    single_people_count club = 9 := by
  sorry

end NUMINAMATH_CALUDE_book_club_single_people_count_l1660_166076


namespace NUMINAMATH_CALUDE_duck_ratio_l1660_166043

theorem duck_ratio (total_birds : ℕ) (chicken_feed_cost : ℚ) (total_chicken_feed_cost : ℚ) :
  total_birds = 15 →
  chicken_feed_cost = 2 →
  total_chicken_feed_cost = 20 →
  (total_birds - (total_chicken_feed_cost / chicken_feed_cost)) / total_birds = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_duck_ratio_l1660_166043


namespace NUMINAMATH_CALUDE_same_side_line_range_l1660_166083

theorem same_side_line_range (a : ℝ) : 
  (∀ x y : ℝ, (x = 3 ∧ y = -1) ∨ (x = -1 ∧ y = 2) → 
    (a * x + 2 * y - 1) * (a * 3 + 2 * (-1) - 1) > 0) ↔ 
  a ∈ Set.Ioo 1 3 :=
sorry

end NUMINAMATH_CALUDE_same_side_line_range_l1660_166083


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l1660_166011

theorem complex_equation_real_solution (a : ℝ) : 
  (((a : ℂ) / (1 + Complex.I)) + ((1 + Complex.I) / 2)).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l1660_166011


namespace NUMINAMATH_CALUDE_quadratic_polynomial_half_coefficient_integer_values_l1660_166013

theorem quadratic_polynomial_half_coefficient_integer_values :
  ∃ (b c : ℚ), ∀ (x : ℤ), ∃ (y : ℤ), ((1/2 : ℚ) * x^2 + b * x + c : ℚ) = y := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_half_coefficient_integer_values_l1660_166013


namespace NUMINAMATH_CALUDE_line_perp_para_implies_plane_perp_l1660_166089

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (planePara : Plane → Plane → Prop)
variable (planePerpDir : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_para_implies_plane_perp
  (m n : Line) (α β : Plane)
  (h1 : perp m α)
  (h2 : para m β) :
  planePerpDir α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_para_implies_plane_perp_l1660_166089


namespace NUMINAMATH_CALUDE_dan_picked_nine_limes_l1660_166028

/-- The number of limes Dan gave to Sara -/
def limes_given_to_Sara : ℕ := 4

/-- The number of limes Dan has left -/
def limes_left_with_Dan : ℕ := 5

/-- The total number of limes Dan picked initially -/
def total_limes : ℕ := limes_given_to_Sara + limes_left_with_Dan

theorem dan_picked_nine_limes : total_limes = 9 := by
  sorry

end NUMINAMATH_CALUDE_dan_picked_nine_limes_l1660_166028


namespace NUMINAMATH_CALUDE_change_in_expression_l1660_166006

theorem change_in_expression (x : ℝ) (b : ℕ+) : 
  let f : ℝ → ℝ := λ t => t^2 - 5*t + 6
  (f (x + b) - f x = 2*b*x + b^2 - 5*b) ∧ 
  (f (x - b) - f x = -2*b*x + b^2 + 5*b) := by
  sorry

end NUMINAMATH_CALUDE_change_in_expression_l1660_166006


namespace NUMINAMATH_CALUDE_x_cubed_plus_y_cubed_le_two_l1660_166041

theorem x_cubed_plus_y_cubed_le_two (x y : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_ineq : x^2 + y^3 ≥ x^3 + y^4) : 
  x^3 + y^3 ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_x_cubed_plus_y_cubed_le_two_l1660_166041


namespace NUMINAMATH_CALUDE_unreserved_seat_cost_l1660_166007

theorem unreserved_seat_cost (total_revenue : ℚ) (reserved_seat_cost : ℚ) 
  (reserved_tickets : ℕ) (unreserved_tickets : ℕ) :
  let unreserved_seat_cost := (total_revenue - reserved_seat_cost * reserved_tickets) / unreserved_tickets
  total_revenue = 26170 ∧ 
  reserved_seat_cost = 25 ∧ 
  reserved_tickets = 246 ∧ 
  unreserved_tickets = 246 → 
  unreserved_seat_cost = 81.3 := by
sorry

end NUMINAMATH_CALUDE_unreserved_seat_cost_l1660_166007


namespace NUMINAMATH_CALUDE_factorization_of_a_squared_minus_2a_l1660_166072

theorem factorization_of_a_squared_minus_2a (a : ℝ) : a^2 - 2*a = a*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_a_squared_minus_2a_l1660_166072


namespace NUMINAMATH_CALUDE_man_walked_40_minutes_l1660_166002

/-- Represents the scenario of a man meeting his wife at the train station and going home. -/
structure TrainScenario where
  T : ℕ  -- usual arrival time at the station
  X : ℕ  -- usual driving time from station to home

/-- Calculates the time spent walking in the given scenario. -/
def time_walking (s : TrainScenario) : ℕ :=
  s.X - 40

/-- Theorem stating that the man spent 40 minutes walking. -/
theorem man_walked_40_minutes (s : TrainScenario) :
  time_walking s = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_man_walked_40_minutes_l1660_166002


namespace NUMINAMATH_CALUDE_lcm_5_7_10_14_l1660_166075

theorem lcm_5_7_10_14 : Nat.lcm 5 (Nat.lcm 7 (Nat.lcm 10 14)) = 70 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_7_10_14_l1660_166075


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1660_166029

theorem min_value_sum_squares (x y z k : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = k) (hk : k ≥ -1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = k → a^2 + b^2 + c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1660_166029


namespace NUMINAMATH_CALUDE_bacon_suggestion_count_l1660_166033

/-- The number of students who suggested bacon, given the total number of students
    and the number of students who suggested mashed potatoes. -/
def students_suggested_bacon (total : ℕ) (mashed_potatoes : ℕ) : ℕ :=
  total - mashed_potatoes

/-- Theorem stating that the number of students who suggested bacon is 125,
    given the total number of students and those who suggested mashed potatoes. -/
theorem bacon_suggestion_count :
  students_suggested_bacon 310 185 = 125 := by
  sorry

end NUMINAMATH_CALUDE_bacon_suggestion_count_l1660_166033


namespace NUMINAMATH_CALUDE_quartic_equation_real_roots_l1660_166024

theorem quartic_equation_real_roots :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (∀ x : ℝ, 3 * x^4 + x^3 - 6 * x^2 + x + 3 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_quartic_equation_real_roots_l1660_166024


namespace NUMINAMATH_CALUDE_same_number_on_four_dice_l1660_166001

theorem same_number_on_four_dice (n : ℕ) (h : n = 8) :
  (1 : ℚ) / (n ^ 3) = 1 / 512 :=
by sorry

end NUMINAMATH_CALUDE_same_number_on_four_dice_l1660_166001


namespace NUMINAMATH_CALUDE_square_area_with_diagonal_l1660_166050

/-- The area of a square with sides of length 12 meters is 144 square meters, 
    given that the diagonal of the square satisfies the Pythagorean theorem. -/
theorem square_area_with_diagonal (x : ℝ) : 
  (x^2 = 2 * 12^2) →  -- Pythagorean theorem for the diagonal
  (12 * 12 : ℝ) = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_diagonal_l1660_166050


namespace NUMINAMATH_CALUDE_repeating_decimal_417_equals_fraction_sum_of_numerator_and_denominator_l1660_166066

/-- Represents a repeating decimal with a 3-digit repetend -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_417_equals_fraction :
  RepeatingDecimal 4 1 7 = 46 / 111 := by sorry

theorem sum_of_numerator_and_denominator :
  46 + 111 = 157 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_417_equals_fraction_sum_of_numerator_and_denominator_l1660_166066


namespace NUMINAMATH_CALUDE_january_salary_l1660_166022

/-- Represents the monthly salary structure --/
structure MonthlySalary where
  january : ℝ
  february : ℝ
  march : ℝ
  april : ℝ
  may : ℝ

/-- Theorem stating the salary for January given the conditions --/
theorem january_salary (s : MonthlySalary) 
  (h1 : (s.january + s.february + s.march + s.april) / 4 = 8000)
  (h2 : (s.february + s.march + s.april + s.may) / 4 = 8450)
  (h3 : s.may = 6500) :
  s.january = 4700 := by
sorry

end NUMINAMATH_CALUDE_january_salary_l1660_166022


namespace NUMINAMATH_CALUDE_larger_number_problem_l1660_166058

theorem larger_number_problem (x y : ℝ) (h1 : x - y = 3) (h2 : x + y = 47) :
  max x y = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1660_166058


namespace NUMINAMATH_CALUDE_kangaroo_arrangement_count_l1660_166047

/-- The number of kangaroos -/
def n : ℕ := 8

/-- The number of ways to arrange the tallest and shortest kangaroos at the ends -/
def end_arrangements : ℕ := 2

/-- The number of remaining kangaroos to be arranged -/
def remaining_kangaroos : ℕ := n - 2

/-- The total number of ways to arrange the kangaroos -/
def total_arrangements : ℕ := end_arrangements * (Nat.factorial remaining_kangaroos)

theorem kangaroo_arrangement_count :
  total_arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_arrangement_count_l1660_166047


namespace NUMINAMATH_CALUDE_coffee_expense_l1660_166079

theorem coffee_expense (items_per_day : ℕ) (cost_per_item : ℕ) (days : ℕ) :
  items_per_day = 2 →
  cost_per_item = 2 →
  days = 30 →
  items_per_day * cost_per_item * days = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_expense_l1660_166079


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_d_equals_five_l1660_166008

theorem infinite_solutions_imply_d_equals_five (d : ℝ) :
  (∀ (S : Set ℝ), S.Infinite → (∀ x ∈ S, 3 * (5 + d * x) = 15 * x + 15)) →
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_d_equals_five_l1660_166008


namespace NUMINAMATH_CALUDE_cupcake_price_l1660_166090

/-- Proves that the price of each cupcake is $2 given the problem conditions --/
theorem cupcake_price (cookies_sold : ℕ) (cookie_price : ℚ) (cupcakes_sold : ℕ) 
  (spoons_bought : ℕ) (spoon_price : ℚ) (money_left : ℚ)
  (h1 : cookies_sold = 40)
  (h2 : cookie_price = 4/5)
  (h3 : cupcakes_sold = 30)
  (h4 : spoons_bought = 2)
  (h5 : spoon_price = 13/2)
  (h6 : money_left = 79) :
  let total_earned := cookies_sold * cookie_price + cupcakes_sold * (2 : ℚ)
  let total_spent := spoons_bought * spoon_price + money_left
  total_earned = total_spent := by sorry

end NUMINAMATH_CALUDE_cupcake_price_l1660_166090


namespace NUMINAMATH_CALUDE_product_and_remainder_l1660_166040

theorem product_and_remainder (a b c d : ℤ) : 
  d = a * b * c → 
  1 < a → a < b → b < c → 
  233 % d = 79 → 
  a + c = 13 := by
sorry

end NUMINAMATH_CALUDE_product_and_remainder_l1660_166040


namespace NUMINAMATH_CALUDE_initial_puppies_count_l1660_166037

/-- The number of puppies Sandy's dog had initially -/
def initial_puppies : ℕ := sorry

/-- The number of puppies Sandy gave away -/
def puppies_given_away : ℕ := 4

/-- The number of puppies Sandy has now -/
def puppies_left : ℕ := 4

/-- Theorem stating that the initial number of puppies is 8 -/
theorem initial_puppies_count : initial_puppies = 8 :=
by
  sorry

#check initial_puppies_count

end NUMINAMATH_CALUDE_initial_puppies_count_l1660_166037


namespace NUMINAMATH_CALUDE_local_maximum_at_e_l1660_166071

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem local_maximum_at_e :
  ∃ δ > 0, ∀ x ∈ Set.Ioo (Real.exp 1 - δ) (Real.exp 1 + δ),
    x ≠ Real.exp 1 → f x < f (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_local_maximum_at_e_l1660_166071


namespace NUMINAMATH_CALUDE_hidden_primes_average_l1660_166060

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def card_sum (visible hidden : ℕ) : ℕ := visible + hidden

theorem hidden_primes_average (h1 h2 h3 : ℕ) :
  is_prime h1 →
  is_prime h2 →
  is_prime h3 →
  card_sum 44 h1 = card_sum 59 h2 →
  card_sum 44 h1 = card_sum 38 h3 →
  (h1 + h2 + h3) / 3 = 14 :=
by sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l1660_166060


namespace NUMINAMATH_CALUDE_expression_evaluation_l1660_166042

theorem expression_evaluation : (π - 2) ^ 0 - 2 * Real.sqrt 3 * 2⁻¹ - Real.sqrt 16 + |1 - Real.sqrt 3| = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1660_166042


namespace NUMINAMATH_CALUDE_no_two_obtuse_angles_l1660_166031

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  sum_180 : angle1 + angle2 + angle3 = 180
  positive : angle1 > 0 ∧ angle2 > 0 ∧ angle3 > 0

-- Define what an obtuse angle is
def isObtuse (angle : Real) : Prop := angle > 90

-- Theorem: A triangle cannot have two obtuse angles
theorem no_two_obtuse_angles (t : Triangle) : 
  ¬(isObtuse t.angle1 ∧ isObtuse t.angle2) ∧
  ¬(isObtuse t.angle1 ∧ isObtuse t.angle3) ∧
  ¬(isObtuse t.angle2 ∧ isObtuse t.angle3) := by
  sorry


end NUMINAMATH_CALUDE_no_two_obtuse_angles_l1660_166031


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l1660_166077

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l1660_166077


namespace NUMINAMATH_CALUDE_paper_used_calculation_l1660_166055

-- Define the variables
def total_paper : ℕ := 900
def remaining_paper : ℕ := 744

-- Define the theorem
theorem paper_used_calculation : total_paper - remaining_paper = 156 := by
  sorry

end NUMINAMATH_CALUDE_paper_used_calculation_l1660_166055


namespace NUMINAMATH_CALUDE_expression_evaluation_l1660_166025

theorem expression_evaluation (a b : ℚ) (h1 : a = 1) (h2 : b = 1/2) :
  a * (a - 2*b) + (a + b) * (a - b) + (a - b)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1660_166025


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1660_166015

theorem unique_solution_for_equation : ∃! (x y : ℕ), 1983 = 1982 * x - 1981 * y ∧ x = 11888 ∧ y = 11893 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1660_166015


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1660_166078

theorem least_addition_for_divisibility :
  ∃ (x : ℕ), x = 8 ∧ 
  (∀ (y : ℕ), y < x → ¬(37 ∣ (157639 + y))) ∧
  (37 ∣ (157639 + x)) := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1660_166078


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1660_166064

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1660_166064


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1660_166000

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1660_166000


namespace NUMINAMATH_CALUDE_tangent_lines_with_slope_one_to_cubic_l1660_166004

/-- The number of tangent lines with slope 1 to the curve y = x³ -/
theorem tangent_lines_with_slope_one_to_cubic (x : ℝ) :
  (∃ m : ℝ, 3 * m^2 = 1) ∧ (∀ m₁ m₂ : ℝ, 3 * m₁^2 = 1 ∧ 3 * m₂^2 = 1 → m₁ = m₂ ∨ m₁ = -m₂) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_with_slope_one_to_cubic_l1660_166004


namespace NUMINAMATH_CALUDE_probability_of_convex_quadrilateral_l1660_166068

/-- The number of points on the circle -/
def n : ℕ := 6

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords between n points -/
def total_chords : ℕ := n.choose 2

/-- The probability of four randomly selected chords forming a convex quadrilateral -/
def probability : ℚ := (n.choose k : ℚ) / (total_chords.choose k : ℚ)

/-- Theorem stating the probability is 1/91 -/
theorem probability_of_convex_quadrilateral : probability = 1 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_convex_quadrilateral_l1660_166068


namespace NUMINAMATH_CALUDE_largest_divisible_by_digits_correct_l1660_166035

/-- A function that returns true if n is divisible by all of its distinct, non-zero digits -/
def divisible_by_digits (n : ℕ) : Bool :=
  let digits := n.digits 10
  digits.all (λ d => d ≠ 0 ∧ n % d = 0)

/-- The largest three-digit number divisible by all its distinct, non-zero digits -/
def largest_divisible_by_digits : ℕ := 936

theorem largest_divisible_by_digits_correct :
  (∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ divisible_by_digits n → n ≤ largest_divisible_by_digits) ∧
  divisible_by_digits largest_divisible_by_digits :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_by_digits_correct_l1660_166035


namespace NUMINAMATH_CALUDE_unique_solution_equation_l1660_166056

theorem unique_solution_equation : ∃! x : ℝ, 3 * x - 8 - 2 = x := by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l1660_166056


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l1660_166097

/-- A Pythagorean triple is a set of three positive integers (a, b, c) where a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean_triple : isPythagoreanTriple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l1660_166097


namespace NUMINAMATH_CALUDE_quadrilateral_is_right_angled_trapezoid_l1660_166003

/-- A quadrilateral in 2D space --/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Vector from point P to point Q --/
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Dot product of two 2D vectors --/
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Definition of a right-angled trapezoid --/
def is_right_angled_trapezoid (q : Quadrilateral) : Prop :=
  ∃ (k : ℝ), k ≠ 1 ∧ vec q.A q.B = k • (vec q.D q.C) ∧
  dot (vec q.A q.D) (vec q.A q.B) = 0

/-- The main theorem --/
theorem quadrilateral_is_right_angled_trapezoid (q : Quadrilateral) 
  (h1 : vec q.A q.B = 2 • (vec q.D q.C))
  (h2 : dot (vec q.C q.D - vec q.C q.A) (vec q.A q.B) = 0) :
  is_right_angled_trapezoid q := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_is_right_angled_trapezoid_l1660_166003


namespace NUMINAMATH_CALUDE_johns_weight_change_l1660_166087

theorem johns_weight_change (initial_weight : ℝ) (loss_percentage : ℝ) (weight_gain : ℝ) : 
  initial_weight = 220 →
  loss_percentage = 10 →
  weight_gain = 2 →
  initial_weight * (1 - loss_percentage / 100) + weight_gain = 200 := by
  sorry

end NUMINAMATH_CALUDE_johns_weight_change_l1660_166087


namespace NUMINAMATH_CALUDE_vegan_soy_free_menu_fraction_l1660_166084

theorem vegan_soy_free_menu_fraction 
  (total_menu : ℕ) 
  (vegan_fraction : Rat) 
  (soy_containing_vegan_fraction : Rat) 
  (h1 : vegan_fraction = 1 / 10) 
  (h2 : soy_containing_vegan_fraction = 2 / 3) : 
  (1 - soy_containing_vegan_fraction) * vegan_fraction = 1 / 30 := by
sorry

end NUMINAMATH_CALUDE_vegan_soy_free_menu_fraction_l1660_166084


namespace NUMINAMATH_CALUDE_base3_21021_equals_196_l1660_166045

def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_21021_equals_196 :
  base3_to_base10 [2, 1, 0, 2, 1] = 196 := by
  sorry

end NUMINAMATH_CALUDE_base3_21021_equals_196_l1660_166045


namespace NUMINAMATH_CALUDE_intersection_volume_l1660_166081

-- Define the two cubes
def cube1 (x y z : ℝ) : Prop := max (|x|) (max |y| |z|) ≤ 1
def cube2 (x y z : ℝ) : Prop := max (|x-1|) (max |y-1| |z-1|) ≤ 1

-- Define the intersection of the two cubes
def intersection (x y z : ℝ) : Prop := cube1 x y z ∧ cube2 x y z

-- Define the volume of a region
noncomputable def volume (region : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem intersection_volume : volume intersection = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_volume_l1660_166081


namespace NUMINAMATH_CALUDE_max_value_is_b_l1660_166051

theorem max_value_is_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b = max (max (max (1/2) b) (2*a*b)) (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_is_b_l1660_166051


namespace NUMINAMATH_CALUDE_sarah_shopping_theorem_l1660_166070

/-- The amount of money Sarah started with -/
def initial_amount : ℕ := sorry

/-- The cost of one toy car -/
def toy_car_cost : ℕ := 11

/-- The number of toy cars Sarah bought -/
def num_toy_cars : ℕ := 2

/-- The cost of the scarf -/
def scarf_cost : ℕ := 10

/-- The cost of the beanie -/
def beanie_cost : ℕ := 14

/-- The amount of money Sarah has remaining after all purchases -/
def remaining_money : ℕ := 7

/-- Theorem stating that the initial amount is equal to the sum of all purchases plus the remaining money -/
theorem sarah_shopping_theorem : 
  initial_amount = toy_car_cost * num_toy_cars + scarf_cost + beanie_cost + remaining_money :=
by sorry

end NUMINAMATH_CALUDE_sarah_shopping_theorem_l1660_166070


namespace NUMINAMATH_CALUDE_volume_of_rotated_region_l1660_166099

/-- The volume of a solid formed by rotating a region consisting of a 6x1 rectangle
    and a 4x3 rectangle about the x-axis -/
theorem volume_of_rotated_region : ℝ := by
  -- Define the dimensions of the rectangles
  let height1 : ℝ := 6
  let width1 : ℝ := 1
  let height2 : ℝ := 3
  let width2 : ℝ := 4

  -- Define the volumes of the two cylinders
  let volume1 : ℝ := Real.pi * height1^2 * width1
  let volume2 : ℝ := Real.pi * height2^2 * width2

  -- Define the total volume
  let total_volume : ℝ := volume1 + volume2

  -- Prove that the total volume equals 72π
  have : total_volume = 72 * Real.pi := by sorry

  -- Return the result
  exact 72 * Real.pi

end NUMINAMATH_CALUDE_volume_of_rotated_region_l1660_166099


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l1660_166069

theorem min_sum_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = x * y → a + b ≤ x + y ∧ a + b = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l1660_166069


namespace NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l1660_166034

/-- 
Given a convex pentagon with interior angles measuring x+1, 2x, 3x, 4x, and 5x-1 degrees,
where x is a positive real number and the sum of these angles is 540 degrees,
prove that the measure of the largest angle is 179 degrees.
-/
theorem largest_angle_convex_pentagon (x : ℝ) 
  (h_positive : x > 0)
  (h_sum : (x + 1) + 2*x + 3*x + 4*x + (5*x - 1) = 540) :
  max (x + 1) (max (2*x) (max (3*x) (max (4*x) (5*x - 1)))) = 179 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l1660_166034


namespace NUMINAMATH_CALUDE_integer_fraction_conditions_l1660_166020

theorem integer_fraction_conditions (p a b : ℕ) : 
  Prime p → 
  a > 0 → 
  b > 0 → 
  (∃ k : ℤ, (4 * a + p : ℤ) / b + (4 * b + p : ℤ) / a = k) → 
  (∃ m : ℤ, (a^2 : ℤ) / b + (b^2 : ℤ) / a = m) → 
  a = b ∨ a = p * b :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_conditions_l1660_166020


namespace NUMINAMATH_CALUDE_total_girls_count_l1660_166036

theorem total_girls_count (van1_students van2_students van3_students van4_students van5_students : Nat)
                          (van1_boys van2_boys van3_boys van4_boys van5_boys : Nat)
                          (h1 : van1_students = 24) (h2 : van2_students = 30) (h3 : van3_students = 20)
                          (h4 : van4_students = 36) (h5 : van5_students = 29)
                          (h6 : van1_boys = 12) (h7 : van2_boys = 16) (h8 : van3_boys = 10)
                          (h9 : van4_boys = 18) (h10 : van5_boys = 8) :
  (van1_students - van1_boys) + (van2_students - van2_boys) + (van3_students - van3_boys) +
  (van4_students - van4_boys) + (van5_students - van5_boys) = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_girls_count_l1660_166036


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l1660_166027

/-- Proves that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is as calculated. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 55)
  (h2 : breadth = 45)
  (h3 : length = breadth + 10)
  (h4 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l1660_166027


namespace NUMINAMATH_CALUDE_dollar_squared_diff_zero_l1660_166039

/-- Custom operation definition -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem statement -/
theorem dollar_squared_diff_zero (x y : ℝ) : dollar ((x - y)^2) ((y - x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_squared_diff_zero_l1660_166039


namespace NUMINAMATH_CALUDE_shaded_area_is_16pi_l1660_166061

/-- Represents the pattern of semicircles as described in the problem -/
structure SemicirclePattern where
  diameter : ℝ
  length : ℝ

/-- Calculates the area of the shaded region in the semicircle pattern -/
def shaded_area (pattern : SemicirclePattern) : ℝ :=
  sorry

/-- Theorem stating that the shaded area of the given pattern is 16π square inches -/
theorem shaded_area_is_16pi (pattern : SemicirclePattern) 
  (h1 : pattern.diameter = 4)
  (h2 : pattern.length = 18) : 
  shaded_area pattern = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_16pi_l1660_166061


namespace NUMINAMATH_CALUDE_boxes_delivered_to_orphanage_l1660_166065

def total_lemon_cupcakes : ℕ := 53
def total_chocolate_cupcakes : ℕ := 76
def lemon_cupcakes_left_at_home : ℕ := 7
def chocolate_cupcakes_left_at_home : ℕ := 8
def cupcakes_per_box : ℕ := 5

def lemon_cupcakes_delivered : ℕ := total_lemon_cupcakes - lemon_cupcakes_left_at_home
def chocolate_cupcakes_delivered : ℕ := total_chocolate_cupcakes - chocolate_cupcakes_left_at_home

def total_cupcakes_delivered : ℕ := lemon_cupcakes_delivered + chocolate_cupcakes_delivered

theorem boxes_delivered_to_orphanage :
  (total_cupcakes_delivered / cupcakes_per_box : ℕ) +
  (if total_cupcakes_delivered % cupcakes_per_box > 0 then 1 else 0) = 23 :=
by sorry

end NUMINAMATH_CALUDE_boxes_delivered_to_orphanage_l1660_166065


namespace NUMINAMATH_CALUDE_salary_increase_after_five_years_l1660_166063

theorem salary_increase_after_five_years (annual_raise : ℝ) (num_years : ℕ) : 
  annual_raise = 0.12 → num_years = 5 → (1 + annual_raise) ^ num_years > 1.76 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_five_years_l1660_166063


namespace NUMINAMATH_CALUDE_smallest_m_for_divisibility_l1660_166073

theorem smallest_m_for_divisibility : 
  ∃ (n : ℕ), 
    n % 2 = 1 ∧ 
    (55^n + 436 * 32^n) % 2001 = 0 ∧ 
    ∀ (m : ℕ), m < 436 → 
      ∀ (k : ℕ), k % 2 = 1 → (55^k + m * 32^k) % 2001 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_divisibility_l1660_166073


namespace NUMINAMATH_CALUDE_point_coordinates_l1660_166030

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian coordinate system -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (M : Point) 
  (h1 : fourth_quadrant M)
  (h2 : distance_to_x_axis M = 3)
  (h3 : distance_to_y_axis M = 4) :
  M.x = 4 ∧ M.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1660_166030


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1660_166054

theorem polynomial_expansion (t : ℝ) : 
  (2 * t^3 - 3 * t + 2) * (-3 * t^2 + 3 * t - 5) = 
  -6 * t^5 + 6 * t^4 - t^3 + 3 * t^2 + 21 * t - 10 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1660_166054


namespace NUMINAMATH_CALUDE_simplify_expressions_l1660_166009

variable (a b : ℝ)

theorem simplify_expressions :
  (4 * a^3 - a^2 + 1 - a^2 - 2 * a^3 = 2 * a^3 - 2 * a^2 + 1) ∧
  (2 * a - 3 * (5 * a - b) + 7 * (a + 2 * b) = -6 * a + 17 * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1660_166009


namespace NUMINAMATH_CALUDE_factorization_problem1_l1660_166085

theorem factorization_problem1 (a b : ℝ) :
  -3 * a^3 + 12 * a^2 * b - 12 * a * b^2 = -3 * a * (a - 2*b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_l1660_166085


namespace NUMINAMATH_CALUDE_circle_partition_exists_l1660_166010

/-- Represents a person with their country and position -/
structure Person where
  country : Fin 25
  position : Fin 100

/-- Defines the arrangement of people in a circle -/
def arrangement : Fin 100 → Person :=
  sorry

/-- Checks if two people are adjacent in the circle -/
def are_adjacent (p1 p2 : Person) : Prop :=
  sorry

/-- Represents a partition of people into 4 groups -/
def Partition := Fin 100 → Fin 4

/-- Checks if a partition is valid according to the problem conditions -/
def is_valid_partition (p : Partition) : Prop :=
  ∀ i j : Fin 100,
    i ≠ j →
    (arrangement i).country = (arrangement j).country ∨ are_adjacent (arrangement i) (arrangement j) →
    p i ≠ p j

theorem circle_partition_exists :
  ∃ p : Partition, is_valid_partition p :=
sorry

end NUMINAMATH_CALUDE_circle_partition_exists_l1660_166010


namespace NUMINAMATH_CALUDE_storage_wheels_count_l1660_166048

def total_wheels (bicycles tricycles unicycles four_wheelers : ℕ) : ℕ :=
  bicycles * 2 + tricycles * 3 + unicycles * 1 + four_wheelers * 4

theorem storage_wheels_count : total_wheels 16 7 10 5 = 83 := by
  sorry

end NUMINAMATH_CALUDE_storage_wheels_count_l1660_166048


namespace NUMINAMATH_CALUDE_card_distribution_events_l1660_166017

-- Define the set of colors
inductive Color
| Red
| Yellow
| Blue
| White

-- Define the set of people
inductive Person
| A
| B
| C
| D

-- Define the distribution of cards
def Distribution := Person → Color

-- Define the event "A receives the red card"
def A_red (d : Distribution) : Prop := d Person.A = Color.Red

-- Define the event "D receives the red card"
def D_red (d : Distribution) : Prop := d Person.D = Color.Red

-- State the theorem
theorem card_distribution_events :
  -- Each person receives one card
  (∀ p : Person, ∃! c : Color, ∀ d : Distribution, d p = c) →
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(A_red d ∧ D_red d)) ∧
  -- The events are not complementary
  ¬(∀ d : Distribution, A_red d ↔ ¬(D_red d)) :=
by sorry

end NUMINAMATH_CALUDE_card_distribution_events_l1660_166017


namespace NUMINAMATH_CALUDE_smallest_period_of_special_function_l1660_166049

/-- A function satisfying the given condition -/
def is_special_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The smallest positive period of a function -/
def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_period f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬ is_period f q

/-- The main theorem -/
theorem smallest_period_of_special_function (f : ℝ → ℝ) (h : is_special_function f) :
  is_smallest_positive_period f 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_period_of_special_function_l1660_166049


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l1660_166059

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √2, b = 2, and sin B + cos B = √2, then the measure of angle A is π/6. -/
theorem angle_measure_in_triangle (A B C : ℝ) (a b c : ℝ) :
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l1660_166059


namespace NUMINAMATH_CALUDE_systematic_sample_valid_l1660_166012

/-- Checks if a list of integers forms a valid systematic sample -/
def is_valid_systematic_sample (population_size : ℕ) (sample_size : ℕ) (sample : List ℕ) : Prop :=
  let interval := population_size / sample_size
  sample.length = sample_size ∧
  ∀ i j, i < j → j < sample.length →
    sample[j]! - sample[i]! = (j - i) * interval

theorem systematic_sample_valid :
  let population_size := 50
  let sample_size := 5
  let sample := [3, 13, 23, 33, 43]
  is_valid_systematic_sample population_size sample_size sample := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_valid_l1660_166012
