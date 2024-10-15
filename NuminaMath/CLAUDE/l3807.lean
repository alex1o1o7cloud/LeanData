import Mathlib

namespace NUMINAMATH_CALUDE_total_weight_CaI2_is_1469_4_l3807_380795

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of moles of calcium iodide -/
def moles_CaI2 : ℝ := 5

/-- The molecular weight of calcium iodide (CaI2) in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_I

/-- The total weight of calcium iodide in grams -/
def total_weight_CaI2 : ℝ := moles_CaI2 * molecular_weight_CaI2

theorem total_weight_CaI2_is_1469_4 :
  total_weight_CaI2 = 1469.4 := by sorry

end NUMINAMATH_CALUDE_total_weight_CaI2_is_1469_4_l3807_380795


namespace NUMINAMATH_CALUDE_f_properties_l3807_380775

-- Define the function f(x) = -x^3 + 12x
def f (x : ℝ) : ℝ := -x^3 + 12*x

-- Define the interval [-3, 1]
def interval : Set ℝ := Set.Icc (-3) 1

theorem f_properties :
  -- f(x) is decreasing on [-3, -2] and increasing on [-2, 1]
  (∀ x ∈ Set.Icc (-3) (-2), ∀ y ∈ Set.Icc (-3) (-2), x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioo (-2) 1, ∀ y ∈ Set.Ioo (-2) 1, x < y → f x < f y) ∧
  -- The maximum value is 11
  (∃ x ∈ interval, f x = 11 ∧ ∀ y ∈ interval, f y ≤ 11) ∧
  -- The minimum value is -16
  (∃ x ∈ interval, f x = -16 ∧ ∀ y ∈ interval, f y ≥ -16) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l3807_380775


namespace NUMINAMATH_CALUDE_motel_rent_is_400_l3807_380760

/-- The total rent charged by a motel on a Saturday night. -/
def totalRent (r50 r60 : ℕ) : ℝ := 50 * r50 + 60 * r60

/-- The rent after changing 10 rooms from $60 to $50. -/
def newRent (r50 r60 : ℕ) : ℝ := 50 * (r50 + 10) + 60 * (r60 - 10)

/-- The theorem stating that the total rent is $400. -/
theorem motel_rent_is_400 (r50 r60 : ℕ) : 
  (∃ (r50 r60 : ℕ), totalRent r50 r60 = 400 ∧ 
    newRent r50 r60 = 0.75 * totalRent r50 r60) := by
  sorry

end NUMINAMATH_CALUDE_motel_rent_is_400_l3807_380760


namespace NUMINAMATH_CALUDE_zero_subset_X_l3807_380703

-- Define the set X
def X : Set ℝ := {x | x > -4}

-- State the theorem
theorem zero_subset_X : {0} ⊆ X := by sorry

end NUMINAMATH_CALUDE_zero_subset_X_l3807_380703


namespace NUMINAMATH_CALUDE_problem_statement_l3807_380772

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : 
  (a + b)^2021 + a^2022 = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3807_380772


namespace NUMINAMATH_CALUDE_fifteenth_entry_is_22_l3807_380741

/-- r_7(n) represents the remainder when n is divided by 7 -/
def r_7 (n : ℕ) : ℕ := n % 7

/-- The list of nonnegative integers n that satisfy r_7(3n) ≤ 3 -/
def satisfying_list : List ℕ :=
  (List.range (100 : ℕ)).filter (fun n => r_7 (3 * n) ≤ 3)

theorem fifteenth_entry_is_22 : satisfying_list[14] = 22 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_entry_is_22_l3807_380741


namespace NUMINAMATH_CALUDE_total_insects_eaten_l3807_380777

theorem total_insects_eaten (num_geckos : ℕ) (insects_per_gecko : ℕ) (num_lizards : ℕ) : 
  num_geckos = 5 → insects_per_gecko = 6 → num_lizards = 3 → 
  num_geckos * insects_per_gecko + num_lizards * (2 * insects_per_gecko) = 66 := by
  sorry

#check total_insects_eaten

end NUMINAMATH_CALUDE_total_insects_eaten_l3807_380777


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l3807_380757

/-- The complex number z = (2-i)/(1+i) is located in the fourth quadrant of the complex plane. -/
theorem complex_in_fourth_quadrant : ∃ (x y : ℝ), 
  (x > 0 ∧ y < 0) ∧ 
  (Complex.I : ℂ) * ((2 : ℂ) - Complex.I) = ((1 : ℂ) + Complex.I) * (x + y * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l3807_380757


namespace NUMINAMATH_CALUDE_inverse_proportion_translation_l3807_380785

/-- Given a non-zero constant k and a function f(x) = k/(x+1) - 2,
    if f(-3) = 1, then k = -6 -/
theorem inverse_proportion_translation (k : ℝ) (hk : k ≠ 0) :
  let f : ℝ → ℝ := λ x => k / (x + 1) - 2
  f (-3) = 1 → k = -6 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_translation_l3807_380785


namespace NUMINAMATH_CALUDE_certain_number_divisibility_l3807_380782

theorem certain_number_divisibility : ∃ (k : ℕ), k = 65 ∧ 
  (∀ (n : ℕ), n < 6 → ¬(k ∣ 11 * n - 1)) ∧ 
  (k ∣ 11 * 6 - 1) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_divisibility_l3807_380782


namespace NUMINAMATH_CALUDE_jogging_time_proportional_to_distance_l3807_380733

/-- Given a constant jogging speed, prove that if it takes 30 minutes to jog 4 miles,
    then it will take 15 minutes to jog 2 miles. -/
theorem jogging_time_proportional_to_distance
  (speed : ℝ) -- Constant jogging speed
  (h1 : speed > 0) -- Assumption that speed is positive
  (h2 : 4 / speed = 30) -- It takes 30 minutes to jog 4 miles
  : 2 / speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_jogging_time_proportional_to_distance_l3807_380733


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3807_380796

def P : Set ℝ := {x : ℝ | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x : ℝ | a * x + 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, Q a ⊆ P → a = 0 ∨ a = -1/2 ∨ a = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3807_380796


namespace NUMINAMATH_CALUDE_video_watching_time_l3807_380723

theorem video_watching_time (video_length : ℕ) (num_videos : ℕ) : 
  video_length = 100 → num_videos = 6 → 
  (num_videos * video_length / 2 + num_videos * video_length) = 900 := by
  sorry

end NUMINAMATH_CALUDE_video_watching_time_l3807_380723


namespace NUMINAMATH_CALUDE_sine_transformation_l3807_380742

theorem sine_transformation (ω A a φ : Real) 
  (h_ω : ω > 0) (h_A : A > 0) (h_a : a > 0) (h_φ : 0 < φ ∧ φ < π) :
  (∀ x, A * Real.sin (ω * x - φ) + a = 3 * Real.sin (2 * x - π / 6) + 1) →
  A + a + ω + φ = 16 / 3 + 11 * π / 12 := by
sorry

end NUMINAMATH_CALUDE_sine_transformation_l3807_380742


namespace NUMINAMATH_CALUDE_largest_negative_integer_congruence_l3807_380715

theorem largest_negative_integer_congruence :
  ∃ (x : ℤ), x = -14 ∧ 
  (∀ (y : ℤ), y < 0 → 26 * y + 8 ≡ 4 [ZMOD 18] → y ≤ x) ∧
  (26 * x + 8 ≡ 4 [ZMOD 18]) := by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_congruence_l3807_380715


namespace NUMINAMATH_CALUDE_sqrt_18_div_sqrt_8_l3807_380740

theorem sqrt_18_div_sqrt_8 : Real.sqrt 18 / Real.sqrt 8 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_sqrt_8_l3807_380740


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_universal_negation_of_implication_l3807_380788

-- 1. Negation of existence
theorem negation_of_existence :
  (¬ ∃ x : ℤ, x^2 - 2*x - 3 = 0) ↔ (∀ x : ℤ, x^2 - 2*x - 3 ≠ 0) :=
by sorry

-- 2. Negation of universal quantification
theorem negation_of_universal :
  (¬ ∀ x : ℝ, x^2 + 3 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 3 < 2*x) :=
by sorry

-- 3. Negation of implication
theorem negation_of_implication :
  (¬ (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2)) ↔
  (∃ x y : ℝ, (x ≤ 1 ∨ y ≤ 1) ∧ x + y ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_universal_negation_of_implication_l3807_380788


namespace NUMINAMATH_CALUDE_handshake_count_l3807_380722

/-- Represents a company at the convention -/
inductive Company
| A | B | C | D | E

/-- The number of companies at the convention -/
def num_companies : Nat := 5

/-- The number of representatives per company -/
def reps_per_company : Nat := 4

/-- The total number of attendees at the convention -/
def total_attendees : Nat := num_companies * reps_per_company

/-- Determines if two companies are the same -/
def same_company (c1 c2 : Company) : Bool :=
  match c1, c2 with
  | Company.A, Company.A => true
  | Company.B, Company.B => true
  | Company.C, Company.C => true
  | Company.D, Company.D => true
  | Company.E, Company.E => true
  | _, _ => false

/-- Determines if a company is Company A -/
def is_company_a (c : Company) : Bool :=
  match c with
  | Company.A => true
  | _ => false

/-- Calculates the number of handshakes for a given company -/
def handshakes_for_company (c : Company) : Nat :=
  if is_company_a c then
    reps_per_company * (total_attendees - reps_per_company)
  else
    reps_per_company * (total_attendees - 2 * reps_per_company)

/-- The total number of handshakes at the convention -/
def total_handshakes : Nat :=
  (handshakes_for_company Company.A +
   handshakes_for_company Company.B +
   handshakes_for_company Company.C +
   handshakes_for_company Company.D +
   handshakes_for_company Company.E) / 2

/-- The main theorem stating that the total number of handshakes is 128 -/
theorem handshake_count : total_handshakes = 128 := by
  sorry


end NUMINAMATH_CALUDE_handshake_count_l3807_380722


namespace NUMINAMATH_CALUDE_parallel_line_and_plane_existence_l3807_380780

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define a plane in 3D space
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

-- Define parallelism between lines
def parallel_lines (l1 l2 : Line3D) : Prop := sorry

-- Define parallelism between a line and a plane
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

-- Define a point not on a line
def point_not_on_line (p : ℝ × ℝ × ℝ) (l : Line3D) : Prop := sorry

theorem parallel_line_and_plane_existence 
  (l : Line3D) (p : ℝ × ℝ × ℝ) (h : point_not_on_line p l) : 
  (∃! l' : Line3D, parallel_lines l l' ∧ l'.point = p) ∧ 
  (∃ f : ℕ → Plane3D, (∀ n : ℕ, parallel_line_plane l (f n) ∧ (f n).point = p) ∧ 
                      (∀ n m : ℕ, n ≠ m → f n ≠ f m)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_and_plane_existence_l3807_380780


namespace NUMINAMATH_CALUDE_grade_distribution_l3807_380793

theorem grade_distribution (total_students : ℕ) 
  (fraction_A : ℚ) (fraction_C : ℚ) (num_D : ℕ) :
  total_students = 800 →
  fraction_A = 1/5 →
  fraction_C = 1/2 →
  num_D = 40 →
  (total_students : ℚ) * (1 - fraction_A - fraction_C) - num_D = (1/4 : ℚ) * total_students :=
by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l3807_380793


namespace NUMINAMATH_CALUDE_consecutive_odd_product_divisibility_l3807_380792

theorem consecutive_odd_product_divisibility :
  ∀ (a b c : ℕ), 
    (a > 0) → 
    (b > 0) → 
    (c > 0) → 
    (Odd a) → 
    (b = a + 2) → 
    (c = b + 2) → 
    (∃ (k : ℕ), a * b * c = 3 * k) ∧ 
    (∀ (m : ℕ), m > 3 → ¬(∀ (x y z : ℕ), 
      (x > 0) → 
      (y > 0) → 
      (z > 0) → 
      (Odd x) → 
      (y = x + 2) → 
      (z = y + 2) → 
      (∃ (n : ℕ), x * y * z = m * n))) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_divisibility_l3807_380792


namespace NUMINAMATH_CALUDE_stationery_store_problem_l3807_380753

/-- Represents the weekly sales volume as a function of the selling price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 80

/-- Represents the weekly profit as a function of the selling price -/
def profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x)

theorem stationery_store_problem 
  (h_price_range : ∀ x, 20 ≤ x ∧ x ≤ 28 → x ∈ Set.Icc 20 28)
  (h_sales_22 : sales_volume 22 = 36)
  (h_sales_24 : sales_volume 24 = 32) :
  (∀ x, sales_volume x = -2 * x + 80) ∧
  (∃ x ∈ Set.Icc 20 28, profit x = 150 ∧ x = 25) ∧
  (∀ x ∈ Set.Icc 20 28, profit x ≤ profit 28 ∧ profit 28 = 192) := by
  sorry

#check stationery_store_problem

end NUMINAMATH_CALUDE_stationery_store_problem_l3807_380753


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3807_380709

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) :
  IsGeometricSequence a → a 3 = 18 → a 4 = 24 → a 5 = 32 := by
  sorry


end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3807_380709


namespace NUMINAMATH_CALUDE_equal_elements_from_inequalities_l3807_380737

theorem equal_elements_from_inequalities (a : Fin 100 → ℝ)
  (h : ∀ i : Fin 100, a i - 3 * a (i + 1) + 2 * a (i + 2) ≥ 0) :
  ∀ i j : Fin 100, a i = a j :=
sorry

end NUMINAMATH_CALUDE_equal_elements_from_inequalities_l3807_380737


namespace NUMINAMATH_CALUDE_unique_solution_l3807_380799

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_solution : ∃! x : ℕ, x > 0 ∧ digit_product x = x^2 - 10*x - 22 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3807_380799


namespace NUMINAMATH_CALUDE_speed_of_X_is_60_l3807_380704

-- Define the speed of person Y
def speed_Y : ℝ := 60

-- Define the time difference between X and Y's start
def time_difference : ℝ := 3

-- Define the distance ahead
def distance_ahead : ℝ := 30

-- Define the time difference between Y catching up to X and X catching up to Y
def catch_up_time_difference : ℝ := 3

-- Define the speed of person X
def speed_X : ℝ := 60

-- Theorem statement
theorem speed_of_X_is_60 :
  ∀ (t₁ t₂ : ℝ),
  t₂ - t₁ = catch_up_time_difference →
  speed_X * (time_difference + t₁) = speed_Y * t₁ + distance_ahead →
  speed_X * (time_difference + t₂) + distance_ahead = speed_Y * t₂ →
  speed_X = speed_Y :=
by sorry

end NUMINAMATH_CALUDE_speed_of_X_is_60_l3807_380704


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3807_380738

theorem smallest_n_for_inequality : ∃ (n : ℕ+),
  (∀ (m : ℕ), 0 < m → m < 2001 → 
    ∃ (k : ℤ), (m : ℚ) / 2001 < (k : ℚ) / (n : ℚ) ∧ (k : ℚ) / (n : ℚ) < ((m + 1) : ℚ) / 2002) ∧
  (∀ (n' : ℕ+), 
    (∀ (m : ℕ), 0 < m → m < 2001 → 
      ∃ (k : ℤ), (m : ℚ) / 2001 < (k : ℚ) / (n' : ℚ) ∧ (k : ℚ) / (n' : ℚ) < ((m + 1) : ℚ) / 2002) →
    n ≤ n') ∧
  n = 4003 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3807_380738


namespace NUMINAMATH_CALUDE_triangle_area_l3807_380791

/-- The area of a triangle formed by the points (0,0), (1,1), and (2,1) is 1/2. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (2, 1)
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  area = 1/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3807_380791


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l3807_380750

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^13 + i^18 + i^23 + i^28 + i^33 = i := by sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l3807_380750


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l3807_380731

theorem compare_negative_fractions : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l3807_380731


namespace NUMINAMATH_CALUDE_inequality_solution_l3807_380763

noncomputable def f (x : ℝ) : ℝ := x^4 + Real.exp (abs x)

theorem inequality_solution :
  let S := {t : ℝ | 2 * f (Real.log t) - f (Real.log (1 / t)) ≤ f 2}
  S = {t : ℝ | Real.exp (-2) ≤ t ∧ t ≤ Real.exp 2} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3807_380763


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3807_380752

theorem chess_tournament_games (n : ℕ) 
  (total_players : ℕ) (total_games : ℕ) 
  (h1 : total_players = 17) 
  (h2 : total_games = 272) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3807_380752


namespace NUMINAMATH_CALUDE_two_digit_three_digit_percentage_equality_l3807_380711

theorem two_digit_three_digit_percentage_equality :
  ∃! (A B : ℕ),
    (A ≥ 10 ∧ A ≤ 99) ∧
    (B ≥ 100 ∧ B ≤ 999) ∧
    (A * (1 + B / 100 : ℚ) = B * (1 - A / 100 : ℚ)) ∧
    A = 40 ∧
    B = 200 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_three_digit_percentage_equality_l3807_380711


namespace NUMINAMATH_CALUDE_largest_value_when_x_is_9_l3807_380756

theorem largest_value_when_x_is_9 :
  let x : ℝ := 9
  (x / 2 > Real.sqrt x) ∧
  (x / 2 > x - 5) ∧
  (x / 2 > 40 / x) ∧
  (x / 2 > x^2 / 20) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_when_x_is_9_l3807_380756


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l3807_380718

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l3807_380718


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l3807_380784

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 2, 3, 4}
def Q : Set Nat := {3, 4, 5}

theorem intersection_P_complement_Q : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l3807_380784


namespace NUMINAMATH_CALUDE_positive_integer_division_problem_l3807_380781

theorem positive_integer_division_problem (a b : ℕ) : 
  a > 1 → b > 1 → (∃k : ℕ, b + 1 = k * a) → (∃l : ℕ, a^3 - 1 = l * b) →
  ((b = a - 1) ∨ (∃p : ℕ, p = 1 ∨ p = 2 ∧ a = a^p ∧ b = a^3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_division_problem_l3807_380781


namespace NUMINAMATH_CALUDE_gain_percent_when_selling_price_twice_cost_price_l3807_380789

/-- If the selling price of an item is twice its cost price, then the gain percent is 100% -/
theorem gain_percent_when_selling_price_twice_cost_price 
  (cost : ℝ) (selling : ℝ) (h : selling = 2 * cost) : 
  (selling - cost) / cost * 100 = 100 :=
sorry

end NUMINAMATH_CALUDE_gain_percent_when_selling_price_twice_cost_price_l3807_380789


namespace NUMINAMATH_CALUDE_base_8_5214_equals_2700_l3807_380778

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ (digits.length - 1 - i))) 0

theorem base_8_5214_equals_2700 :
  base_8_to_10 [5, 2, 1, 4] = 2700 := by
  sorry

end NUMINAMATH_CALUDE_base_8_5214_equals_2700_l3807_380778


namespace NUMINAMATH_CALUDE_pet_shop_dogs_count_l3807_380726

/-- Represents the number of animals of each type in the pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ

/-- The ratio of dogs to cats to bunnies -/
def ratio : Fin 3 → ℕ
| 0 => 3  -- dogs
| 1 => 7  -- cats
| 2 => 12 -- bunnies

theorem pet_shop_dogs_count (shop : PetShop) :
  (ratio 0 : ℚ) / shop.dogs = (ratio 1 : ℚ) / shop.cats ∧
  (ratio 0 : ℚ) / shop.dogs = (ratio 2 : ℚ) / shop.bunnies ∧
  shop.dogs + shop.bunnies = 375 →
  shop.dogs = 75 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_count_l3807_380726


namespace NUMINAMATH_CALUDE_outfit_combinations_l3807_380707

/-- Calculates the number of outfits given the number of clothing items --/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (jackets : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (jackets + 1)

/-- Theorem stating the number of outfits given specific quantities of clothing items --/
theorem outfit_combinations :
  number_of_outfits 8 5 4 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3807_380707


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3807_380798

theorem inequality_and_equality_condition (a b c : ℝ) (α : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^α + b^α + c^α) ≥ 
    a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ∧
  (a * b * c * (a^α + b^α + c^α) = 
    a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔
   a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3807_380798


namespace NUMINAMATH_CALUDE_walking_speed_equation_l3807_380783

theorem walking_speed_equation (x : ℝ) 
  (h1 : x > 0) -- Xiao Wang's speed is positive
  (h2 : x + 1 > 0) -- Xiao Zhang's speed is positive
  : 
  (15 / x - 15 / (x + 1) = 1 / 2) ↔ 
  (15 / x = 15 / (x + 1) + 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_equation_l3807_380783


namespace NUMINAMATH_CALUDE_five_digit_number_divisible_by_9_l3807_380797

theorem five_digit_number_divisible_by_9 (a b c d e : ℕ) : 
  (10000 * a + 1000 * b + 100 * c + 10 * d + e) % 9 = 0 →
  (100 * a + 10 * c + e) - (100 * b + 10 * d + a) = 760 →
  10000 ≤ (10000 * a + 1000 * b + 100 * c + 10 * d + e) →
  (10000 * a + 1000 * b + 100 * c + 10 * d + e) < 100000 →
  10000 * a + 1000 * b + 100 * c + 10 * d + e = 81828 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_divisible_by_9_l3807_380797


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3807_380710

open Set
open Function
open Real

noncomputable def f (x : ℝ) : ℝ := x * (x^2 - Real.cos (x/3) + 2)

theorem solution_set_of_inequality :
  let S := {x : ℝ | x ∈ Ioo (-3) 3 ∧ f (1 + x) + f 2 < f (1 - x)}
  S = Ioo (-2) (-1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3807_380710


namespace NUMINAMATH_CALUDE_price_per_apple_l3807_380769

/-- Calculate the price per apple given the orchard layout, apple production, and total revenue -/
theorem price_per_apple (rows : ℕ) (columns : ℕ) (apples_per_tree : ℕ) (total_revenue : ℚ) : 
  rows = 3 → columns = 4 → apples_per_tree = 5 → total_revenue = 30 →
  total_revenue / (rows * columns * apples_per_tree) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_price_per_apple_l3807_380769


namespace NUMINAMATH_CALUDE_inequality_preservation_l3807_380773

theorem inequality_preservation (a b : ℝ) (h : a > b) : a / 3 > b / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3807_380773


namespace NUMINAMATH_CALUDE_employee_pay_l3807_380719

theorem employee_pay (total : ℝ) (x y : ℝ) (h1 : total = 770) (h2 : x = 1.2 * y) (h3 : x + y = total) : y = 350 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l3807_380719


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3807_380708

/-- Represents a hyperbola equation with parameter m -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / a - y^2 / b = 1 ↔ x^2 / (m - 10) - y^2 / (m - 8) = 1

/-- m > 10 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem hyperbola_condition (m : ℝ) :
  (m > 10 → is_hyperbola m) ∧ (∃ m₀ : ℝ, m₀ ≤ 10 ∧ is_hyperbola m₀) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3807_380708


namespace NUMINAMATH_CALUDE_fifth_root_of_x_times_fourth_root_l3807_380790

theorem fifth_root_of_x_times_fourth_root (x : ℝ) (hx : x > 0) :
  (x * x^(1/4))^(1/5) = x^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_x_times_fourth_root_l3807_380790


namespace NUMINAMATH_CALUDE_mohamed_donated_three_bags_l3807_380749

/-- The number of bags Leila donated -/
def leila_bags : ℕ := 2

/-- The number of toys in each of Leila's bags -/
def leila_toys_per_bag : ℕ := 25

/-- The number of toys in each of Mohamed's bags -/
def mohamed_toys_per_bag : ℕ := 19

/-- The difference between Mohamed's and Leila's toy donations -/
def toy_difference : ℕ := 7

/-- Calculates the total number of toys Leila donated -/
def leila_total_toys : ℕ := leila_bags * leila_toys_per_bag

/-- Calculates the total number of toys Mohamed donated -/
def mohamed_total_toys : ℕ := leila_total_toys + toy_difference

/-- The number of bags Mohamed donated -/
def mohamed_bags : ℕ := mohamed_total_toys / mohamed_toys_per_bag

theorem mohamed_donated_three_bags : mohamed_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_mohamed_donated_three_bags_l3807_380749


namespace NUMINAMATH_CALUDE_inequality_not_always_holds_l3807_380735

theorem inequality_not_always_holds (a b : ℝ) (h : a < b) :
  ¬ ∀ m : ℝ, a * m^2 < b * m^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_holds_l3807_380735


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_10_15_30_l3807_380716

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time -/
def initialTime : Time :=
  { hours := 10, minutes := 15, seconds := 30 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 9999

/-- The expected final time -/
def expectedFinalTime : Time :=
  { hours := 13, minutes := 2, seconds := 9 }

theorem add_9999_seconds_to_10_15_30 :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_10_15_30_l3807_380716


namespace NUMINAMATH_CALUDE_cytoplasm_distribution_in_cell_division_l3807_380727

/-- Represents a cell in a diploid organism -/
structure DiploidCell where
  cytoplasm : Set ℝ
  deriving Inhabited

/-- Represents the process of cell division -/
def cell_division (parent : DiploidCell) : DiploidCell × DiploidCell :=
  sorry

/-- The distribution of cytoplasm during cell division is random -/
def is_random_distribution (division : DiploidCell → DiploidCell × DiploidCell) : Prop :=
  sorry

/-- The distribution of cytoplasm during cell division is unequal -/
def is_unequal_distribution (division : DiploidCell → DiploidCell × DiploidCell) : Prop :=
  sorry

/-- Theorem: In diploid organism cells, the distribution of cytoplasm during cell division is random and unequal -/
theorem cytoplasm_distribution_in_cell_division :
  is_random_distribution cell_division ∧ is_unequal_distribution cell_division :=
sorry

end NUMINAMATH_CALUDE_cytoplasm_distribution_in_cell_division_l3807_380727


namespace NUMINAMATH_CALUDE_flowers_per_bouquet_l3807_380734

theorem flowers_per_bouquet 
  (initial_flowers : ℕ) 
  (wilted_flowers : ℕ) 
  (num_bouquets : ℕ) 
  (h1 : initial_flowers = 66) 
  (h2 : wilted_flowers = 10) 
  (h3 : num_bouquets = 7) : 
  (initial_flowers - wilted_flowers) / num_bouquets = 8 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_bouquet_l3807_380734


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3807_380720

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b < c)
  (h4 : a^2 + b^2 = c^2) :
  (1/a) + (1/b) + (1/c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3807_380720


namespace NUMINAMATH_CALUDE_charles_and_jen_whistles_l3807_380717

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference in whistles between Sean and Charles -/
def sean_charles_diff : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - sean_charles_diff

/-- The difference in whistles between Jen and Charles -/
def jen_charles_diff : ℕ := 15

/-- The number of whistles Jen has -/
def jen_whistles : ℕ := charles_whistles + jen_charles_diff

/-- The total number of whistles Charles and Jen have -/
def total_whistles : ℕ := charles_whistles + jen_whistles

theorem charles_and_jen_whistles : total_whistles = 41 := by
  sorry

end NUMINAMATH_CALUDE_charles_and_jen_whistles_l3807_380717


namespace NUMINAMATH_CALUDE_fraction_equality_l3807_380748

theorem fraction_equality (x y : ℚ) (hx : x = 3/5) (hy : y = 7/9) :
  (5*x + 9*y) / (45*x*y) = 10/21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3807_380748


namespace NUMINAMATH_CALUDE_triangle_equilateral_l3807_380725

/-- Given a triangle ABC with angle C = 60° and c² = ab, prove that ABC is equilateral -/
theorem triangle_equilateral (a b c : ℝ) (angleC : ℝ) :
  angleC = π / 3 →  -- 60° in radians
  c^2 = a * b →
  a > 0 → b > 0 → c > 0 →
  a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l3807_380725


namespace NUMINAMATH_CALUDE_carlys_running_schedule_l3807_380762

/-- Carly's running schedule over four weeks -/
theorem carlys_running_schedule (x : ℝ) : 
  (∃ week2 week3 : ℝ,
    week2 = 2*x + 3 ∧ 
    week3 = (9/7) * week2 ∧ 
    week3 - 5 = 4) → 
  x = 2 := by sorry

end NUMINAMATH_CALUDE_carlys_running_schedule_l3807_380762


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3807_380765

theorem complex_equation_solution (z : ℂ) : z * (1 + 2*I) = 3 + I → z = 1 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3807_380765


namespace NUMINAMATH_CALUDE_sin_cos_tan_l3807_380705

theorem sin_cos_tan (α : Real) (h : Real.tan α = 4) : 
  Real.sin α * Real.cos α = 4 / 17 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_tan_l3807_380705


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3807_380758

theorem arithmetic_calculations :
  ((-20) + (-14) - (-18) - 13 = -29) ∧
  (-24 * ((-1/2) + (3/4) - (1/3)) = 2) ∧
  (-49 * (24/25) * 10 = -499.6) ∧
  (-(3^2) + (((-1/3) * (-3)) - ((8/5) / (2^2))) = -(8 + 2/5)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3807_380758


namespace NUMINAMATH_CALUDE_workshop_workers_count_workshop_workers_count_proof_l3807_380702

/-- Given a workshop with workers, prove that the total number of workers is 28 -/
theorem workshop_workers_count : ℕ :=
  let total_average : ℚ := 8000
  let technician_count : ℕ := 7
  let technician_average : ℚ := 14000
  let non_technician_average : ℚ := 6000
  28

/-- Proof of the theorem -/
theorem workshop_workers_count_proof :
  let total_average : ℚ := 8000
  let technician_count : ℕ := 7
  let technician_average : ℚ := 14000
  let non_technician_average : ℚ := 6000
  workshop_workers_count = 28 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_workshop_workers_count_proof_l3807_380702


namespace NUMINAMATH_CALUDE_decagon_perimeter_l3807_380770

/-- A decagon is a polygon with 10 sides -/
def Decagon := Nat

/-- The number of sides in a decagon -/
def decagon_sides : Nat := 10

/-- The length of each side of our specific decagon -/
def side_length : ℝ := 3

/-- The perimeter of a polygon is the sum of the lengths of all its sides -/
def perimeter (n : Nat) (s : ℝ) : ℝ := n * s

/-- Theorem: The perimeter of a decagon with sides of length 3 units is 30 units -/
theorem decagon_perimeter : perimeter decagon_sides side_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_decagon_perimeter_l3807_380770


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l3807_380701

/-- A parallelogram with sides measuring 10, 12, 5y-2, and 3x+6 units consecutively has x + y = 22/5 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  3 * x + 6 = 12 → 5 * y - 2 = 10 → x + y = 22 / 5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l3807_380701


namespace NUMINAMATH_CALUDE_paint_cans_calculation_l3807_380732

theorem paint_cans_calculation (initial_cans : ℚ) : 
  (initial_cans / 2 - (initial_cans / 6 + 5) = 5) → initial_cans = 30 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_calculation_l3807_380732


namespace NUMINAMATH_CALUDE_celeste_opod_probability_l3807_380764

/-- Represents the duration of a song in seconds -/
def SongDuration := ℕ

/-- Represents the o-Pod with its songs -/
structure OPod where
  songs : List SongDuration
  favorite_song : SongDuration

/-- Creates an o-Pod with 10 songs, where each song is 30 seconds longer than the previous one -/
def create_opod (favorite_duration : ℕ) : OPod :=
  { songs := List.range 10 |>.map (fun i => 30 * (i + 1)),
    favorite_song := favorite_duration }

/-- Calculates the probability of not hearing the entire favorite song within a given time -/
noncomputable def prob_not_hear_favorite (opod : OPod) (total_time : ℕ) : ℚ :=
  sorry

theorem celeste_opod_probability :
  let celeste_opod := create_opod 210
  prob_not_hear_favorite celeste_opod 270 = 79 / 90 := by
  sorry

end NUMINAMATH_CALUDE_celeste_opod_probability_l3807_380764


namespace NUMINAMATH_CALUDE_choose_six_three_equals_twenty_l3807_380779

theorem choose_six_three_equals_twenty : Nat.choose 6 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_choose_six_three_equals_twenty_l3807_380779


namespace NUMINAMATH_CALUDE_union_determines_x_l3807_380729

theorem union_determines_x (A B : Set ℕ) (x : ℕ) :
  A = {1, 2, x} →
  B = {2, 4, 5} →
  A ∪ B = {1, 2, 3, 4, 5} →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_union_determines_x_l3807_380729


namespace NUMINAMATH_CALUDE_product_remainder_l3807_380743

/-- The number of times 23 is repeated in the product -/
def n : ℕ := 23

/-- The divisor -/
def m : ℕ := 32

/-- Function to calculate the remainder of the product of n 23's when divided by m -/
def f (n m : ℕ) : ℕ := (23^n) % m

theorem product_remainder : f n m = 19 := by sorry

end NUMINAMATH_CALUDE_product_remainder_l3807_380743


namespace NUMINAMATH_CALUDE_exactly_three_special_triangles_l3807_380721

/-- A right-angled triangle with integer sides where the area is twice the perimeter -/
structure SpecialTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  right_angled : a.val^2 + b.val^2 = c.val^2
  area_perimeter : (a.val * b.val : ℕ) = 4 * (a.val + b.val + c.val)

/-- There are exactly three special triangles -/
theorem exactly_three_special_triangles : 
  ∃! (list : List SpecialTriangle), list.length = 3 ∧ 
  (∀ t : SpecialTriangle, t ∈ list) ∧
  (∀ t ∈ list, t ∈ [⟨9, 40, 41, sorry, sorry⟩, ⟨10, 24, 26, sorry, sorry⟩, ⟨12, 16, 20, sorry, sorry⟩]) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_special_triangles_l3807_380721


namespace NUMINAMATH_CALUDE_camel_cost_is_5200_l3807_380794

-- Define the cost of each animal
def camel_cost : ℝ := 5200
def elephant_cost : ℝ := 13000
def ox_cost : ℝ := 8666.67
def horse_cost : ℝ := 2166.67

-- Define the relationships between animal costs
axiom camel_horse_ratio : 10 * camel_cost = 24 * horse_cost
axiom horse_ox_ratio : ∃ x : ℕ, x * horse_cost = 4 * ox_cost
axiom ox_elephant_ratio : 6 * ox_cost = 4 * elephant_cost
axiom elephant_total_cost : 10 * elephant_cost = 130000

-- Theorem to prove
theorem camel_cost_is_5200 : camel_cost = 5200 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_is_5200_l3807_380794


namespace NUMINAMATH_CALUDE_no_rational_cos_sqrt2_l3807_380759

theorem no_rational_cos_sqrt2 : ¬∃ (x : ℝ), (∃ (a b : ℚ), (Real.cos x + Real.sqrt 2 = a) ∧ (Real.cos (2 * x) + Real.sqrt 2 = b)) := by
  sorry

end NUMINAMATH_CALUDE_no_rational_cos_sqrt2_l3807_380759


namespace NUMINAMATH_CALUDE_power_of_power_l3807_380767

theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3807_380767


namespace NUMINAMATH_CALUDE_plane_perpendicular_condition_l3807_380739

/-- The normal vector of a plane -/
structure NormalVector where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Dot product of two normal vectors -/
def dot_product (v1 v2 : NormalVector) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

/-- Two planes are perpendicular if their normal vectors are orthogonal -/
def perpendicular (v1 v2 : NormalVector) : Prop :=
  dot_product v1 v2 = 0

theorem plane_perpendicular_condition (k : ℝ) :
  let α : NormalVector := ⟨3, 1, -2⟩
  let β : NormalVector := ⟨-1, 1, k⟩
  perpendicular α β → k = -1 :=
by sorry

end NUMINAMATH_CALUDE_plane_perpendicular_condition_l3807_380739


namespace NUMINAMATH_CALUDE_total_watching_time_l3807_380786

/-- Calculates the total watching time for two people watching multiple videos at different speeds -/
theorem total_watching_time
  (video_length : ℝ)
  (num_videos : ℕ)
  (speed_ratio_1 : ℝ)
  (speed_ratio_2 : ℝ)
  (h1 : video_length = 100)
  (h2 : num_videos = 6)
  (h3 : speed_ratio_1 = 2)
  (h4 : speed_ratio_2 = 1) :
  (num_videos * video_length / speed_ratio_1) + (num_videos * video_length / speed_ratio_2) = 900 := by
  sorry

#check total_watching_time

end NUMINAMATH_CALUDE_total_watching_time_l3807_380786


namespace NUMINAMATH_CALUDE_extremum_implies_a_eq_neg_two_l3807_380736

/-- The function f(x) = a ln x + x^2 has an extremum at x = 1 -/
def has_extremum_at_one (a : ℝ) : Prop :=
  let f := fun x : ℝ => a * Real.log x + x^2
  ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1

/-- If f(x) = a ln x + x^2 has an extremum at x = 1, then a = -2 -/
theorem extremum_implies_a_eq_neg_two (a : ℝ) :
  has_extremum_at_one a → a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_a_eq_neg_two_l3807_380736


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3807_380755

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := ∃ y : ℝ, x = y^2

-- Define what it means for a quadratic radical to be simplest
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ 
  ∀ y : ℝ, (∃ z : ℝ, y = z^2 ∧ x = y * z) → y = 1

-- State the theorem
theorem simplest_quadratic_radical :
  SimplestQuadraticRadical 6 ∧
  ¬SimplestQuadraticRadical 12 ∧
  ¬SimplestQuadraticRadical 0.3 ∧
  ¬SimplestQuadraticRadical (1/2) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3807_380755


namespace NUMINAMATH_CALUDE_gcd_power_sum_l3807_380761

theorem gcd_power_sum (n : ℕ) (h : n > 32) :
  Nat.gcd (n^5 + 5^3) (n + 5) = if n % 5 = 0 then 5 else 1 := by sorry

end NUMINAMATH_CALUDE_gcd_power_sum_l3807_380761


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3807_380712

theorem other_root_of_quadratic (m : ℝ) : 
  ((-4 : ℝ)^2 + m * (-4) - 20 = 0) → 
  ((5 : ℝ)^2 + m * 5 - 20 = 0) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3807_380712


namespace NUMINAMATH_CALUDE_kosher_meals_count_l3807_380766

/-- Calculates the number of kosher meals given the total number of clients,
    number of vegan meals, number of both vegan and kosher meals,
    and number of meals that are neither vegan nor kosher. -/
def kosher_meals (total : ℕ) (vegan : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  total - neither - (vegan - both)

/-- Proves that the number of clients needing kosher meals is 8,
    given the specific conditions from the problem. -/
theorem kosher_meals_count :
  kosher_meals 30 7 3 18 = 8 := by
  sorry

end NUMINAMATH_CALUDE_kosher_meals_count_l3807_380766


namespace NUMINAMATH_CALUDE_constant_function_l3807_380706

theorem constant_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2011 * x) = 2011) :
  ∀ x : ℝ, f (3 * x) = 2011 := by
sorry

end NUMINAMATH_CALUDE_constant_function_l3807_380706


namespace NUMINAMATH_CALUDE_major_axis_length_for_specific_cylinder_l3807_380730

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def majorAxisLength (cylinderRadius : ℝ) (majorMinorRatio : ℝ) : ℝ :=
  2 * cylinderRadius * majorMinorRatio

/-- Theorem: The major axis length for the given conditions --/
theorem major_axis_length_for_specific_cylinder :
  majorAxisLength 3 1.2 = 7.2 :=
by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_for_specific_cylinder_l3807_380730


namespace NUMINAMATH_CALUDE_weight_meets_standard_l3807_380744

/-- The nominal weight of the strawberry box in kilograms -/
def nominal_weight : ℝ := 5

/-- The allowed deviation from the nominal weight in kilograms -/
def allowed_deviation : ℝ := 0.03

/-- The actual weight of the strawberry box in kilograms -/
def actual_weight : ℝ := 4.98

/-- Theorem stating that the actual weight meets the standard -/
theorem weight_meets_standard : 
  nominal_weight - allowed_deviation ≤ actual_weight ∧ 
  actual_weight ≤ nominal_weight + allowed_deviation := by
  sorry

end NUMINAMATH_CALUDE_weight_meets_standard_l3807_380744


namespace NUMINAMATH_CALUDE_nested_bracket_value_l3807_380745

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- State the theorem
theorem nested_bracket_value :
  bracket (bracket 80 40 120) (bracket 4 2 6) (bracket 50 25 75) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_value_l3807_380745


namespace NUMINAMATH_CALUDE_secretary_work_time_l3807_380768

theorem secretary_work_time (t1 t2 t3 : ℕ) : 
  t1 + t2 + t3 = 110 ∧ 
  t3 = 55 ∧ 
  2 * t2 = 3 * t1 ∧ 
  5 * t1 = 3 * t3 :=
by sorry

end NUMINAMATH_CALUDE_secretary_work_time_l3807_380768


namespace NUMINAMATH_CALUDE_sin_square_sum_range_l3807_380713

open Real

theorem sin_square_sum_range (α β : ℝ) (h : 3 * (sin α)^2 - 2 * sin α + 2 * (sin β)^2 = 0) :
  ∃ (x : ℝ), x = (sin α)^2 + (sin β)^2 ∧ 0 ≤ x ∧ x ≤ 4/9 ∧
  ∀ (y : ℝ), y = (sin α)^2 + (sin β)^2 → 0 ≤ y ∧ y ≤ 4/9 :=
by sorry

end NUMINAMATH_CALUDE_sin_square_sum_range_l3807_380713


namespace NUMINAMATH_CALUDE_tan_difference_pi_12_pi_6_l3807_380700

theorem tan_difference_pi_12_pi_6 : 
  Real.tan (π / 12) - Real.tan (π / 6) = 7 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_pi_12_pi_6_l3807_380700


namespace NUMINAMATH_CALUDE_apple_distribution_l3807_380787

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribute_apples (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- Theorem stating that there are 253 ways to distribute 30 apples among 3 people, with each person receiving at least 3 apples -/
theorem apple_distribution : distribute_apples 30 3 3 = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3807_380787


namespace NUMINAMATH_CALUDE_shopping_equation_system_l3807_380728

theorem shopping_equation_system (x y : ℤ) : 
  (∀ (coins_per_person excess : ℤ), coins_per_person * x - y = excess → 
    ((coins_per_person = 8 ∧ excess = 3) ∨ (coins_per_person = 7 ∧ excess = -4))) → 
  (8 * x - y = 3 ∧ y - 7 * x = 4) := by
sorry

end NUMINAMATH_CALUDE_shopping_equation_system_l3807_380728


namespace NUMINAMATH_CALUDE_final_position_is_correct_l3807_380771

/-- Movement pattern A: 1 unit up, 2 units right -/
def pattern_a : ℤ × ℤ := (2, 1)

/-- Movement pattern B: 3 units left, 2 units down -/
def pattern_b : ℤ × ℤ := (-3, -2)

/-- Calculate the position after n movements -/
def position_after_n_movements (n : ℕ) : ℤ × ℤ :=
  let a_count := (n + 1) / 2
  let b_count := n / 2
  (a_count * pattern_a.1 + b_count * pattern_b.1,
   a_count * pattern_a.2 + b_count * pattern_b.2)

/-- The final position after 15 movements -/
def final_position : ℤ × ℤ := position_after_n_movements 15

theorem final_position_is_correct : final_position = (-5, -6) := by
  sorry

end NUMINAMATH_CALUDE_final_position_is_correct_l3807_380771


namespace NUMINAMATH_CALUDE_min_value_z_l3807_380747

theorem min_value_z (x y : ℝ) : 3 * x^2 + 5 * y^2 + 8 * x - 10 * y + 4 * x * y + 35 ≥ 251 / 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l3807_380747


namespace NUMINAMATH_CALUDE_max_no_draw_participants_16_550_l3807_380774

/-- Represents a tic-tac-toe tournament -/
structure Tournament where
  participants : ℕ
  total_points : ℕ
  win_points : ℕ
  draw_points : ℕ
  loss_points : ℕ

/-- Calculates the total number of games in a tournament -/
def total_games (t : Tournament) : ℕ :=
  t.participants * (t.participants - 1) / 2

/-- Calculates the maximum number of participants who could have played without a draw -/
def max_no_draw_participants (t : Tournament) : ℕ :=
  sorry

/-- Theorem stating the maximum number of participants without a draw in the given tournament -/
theorem max_no_draw_participants_16_550 :
  let t : Tournament := {
    participants := 16,
    total_points := 550,
    win_points := 5,
    draw_points := 2,
    loss_points := 0
  }
  max_no_draw_participants t = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_no_draw_participants_16_550_l3807_380774


namespace NUMINAMATH_CALUDE_locus_of_vertex_A_l3807_380746

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the median CM
def Median (C M : ℝ × ℝ) : Prop := True

-- Define the constant length of CM
def ConstantLength (CM : ℝ) : Prop := True

-- Define the midpoint of BC
def Midpoint (K B C : ℝ × ℝ) : Prop := 
  K.1 = (B.1 + C.1) / 2 ∧ K.2 = (B.2 + C.2) / 2

-- Define a circle
def Circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Theorem statement
theorem locus_of_vertex_A (A B C : ℝ × ℝ) (K : ℝ × ℝ) (CM : ℝ) :
  Triangle A B C →
  Midpoint K B C →
  ConstantLength CM →
  ∀ M, Median C M →
  ∃ center radius, Circle center radius A ∧ 
    center = K ∧ 
    radius = 2 * CM ∧
    ¬(A.1 = B.1 ∧ A.2 = B.2) ∧ 
    ¬(A.1 = C.1 ∧ A.2 = C.2) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_vertex_A_l3807_380746


namespace NUMINAMATH_CALUDE_line_reflection_l3807_380724

-- Define the slope of the original line
def k : ℝ := sorry

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := x + y = 1

-- Define the original line
def original_line (x y : ℝ) : Prop := y = k * x

-- Define the resulting line after reflection
def reflected_line (x y : ℝ) : Prop := y = (1 / k) * x + (k - 1) / k

-- State the theorem
theorem line_reflection (h1 : k ≠ 0) (h2 : k ≠ -1) :
  ∀ x y : ℝ, reflected_line x y ↔ 
  ∃ x' y' : ℝ, original_line x' y' ∧ 
  reflection_line ((x + x') / 2) ((y + y') / 2) :=
sorry

end NUMINAMATH_CALUDE_line_reflection_l3807_380724


namespace NUMINAMATH_CALUDE_cow_selling_price_l3807_380754

/-- Calculates the selling price of a cow given the initial cost, daily food cost,
    vaccination and deworming cost, number of days, and profit made. -/
theorem cow_selling_price
  (initial_cost : ℕ)
  (daily_food_cost : ℕ)
  (vaccination_cost : ℕ)
  (num_days : ℕ)
  (profit : ℕ)
  (h1 : initial_cost = 600)
  (h2 : daily_food_cost = 20)
  (h3 : vaccination_cost = 500)
  (h4 : num_days = 40)
  (h5 : profit = 600) :
  initial_cost + num_days * daily_food_cost + vaccination_cost + profit = 2500 :=
by
  sorry

end NUMINAMATH_CALUDE_cow_selling_price_l3807_380754


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sums_l3807_380714

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sums (a : ℕ → ℝ) :
  isArithmeticSequence a →
  isArithmeticSequence (λ n : ℕ => a (3*n + 1) + a (3*n + 2) + a (3*n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sums_l3807_380714


namespace NUMINAMATH_CALUDE_brother_age_relation_l3807_380751

theorem brother_age_relation : 
  let current_age_older : ℕ := 15
  let current_age_younger : ℕ := 5
  let years_passed : ℕ := 5
  (current_age_older + years_passed) = 2 * (current_age_younger + years_passed) :=
by sorry

end NUMINAMATH_CALUDE_brother_age_relation_l3807_380751


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3807_380776

theorem complex_number_quadrant : 
  let z : ℂ := (3 + Complex.I) * (1 - Complex.I)
  (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3807_380776
